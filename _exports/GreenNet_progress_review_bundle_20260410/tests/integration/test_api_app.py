from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api_app
from greennet.persistence import get_run_repository


pytestmark = pytest.mark.integration


@pytest.fixture
def api_client(generated_run: Path, repo_root: Path, tmp_path, monkeypatch) -> TestClient:
    results_dir = generated_run.parent
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    locked_dir = tmp_path / "artifacts" / "locked"
    locked_dir.mkdir(parents=True)

    monkeypatch.setattr(api_app, "REPO_ROOT", repo_root)
    monkeypatch.setattr(api_app, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(api_app, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(api_app, "LOCKED_ARTIFACTS_DIR", locked_dir)

    api_app._read_locked_eval_rows.cache_clear()
    api_app._read_all_per_step_rows.cache_clear()
    api_app._build_topology_bundle.cache_clear()
    api_app._build_step_payload.cache_clear()
    api_app._latest_final_evaluation_artifact.cache_clear()

    return TestClient(api_app.app)


def test_runs_and_health_endpoints_return_expected_shapes(api_client: TestClient, generated_run: Path) -> None:
    health = api_client.get("/api/health")
    runs = api_client.get("/api/runs", params={"base": "results"})
    aggregate = api_client.get("/api/aggregate", params={"base": "results", "group_by": "policy,scenario"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    assert runs.status_code == 200
    runs_payload = runs.json()
    assert runs_payload["total"] == 1
    assert runs_payload["items"][0]["run_id"] == generated_run.name
    assert runs_payload["items"][0]["has"]["per_step"] is True
    assert runs_payload["items"][0]["has"]["summary"] is True
    assert runs_payload["items"][0]["qos_acceptance_status"] is not None
    assert runs_payload["items"][0]["stability_status"] is not None
    assert runs_payload["items"][0]["highlights"]["avg_delay_ms_mean"] is not None
    assert runs_payload["items"][0]["highlights"]["qos_violation_rate_mean"] is not None
    assert "transition_rate_mean" in runs_payload["items"][0]["highlights"]

    assert aggregate.status_code == 200
    aggregate_payload = aggregate.json()
    assert len(aggregate_payload) == 1
    assert aggregate_payload[0]["group"]["policy"] == "all_on"
    assert aggregate_payload[0]["group"]["scenario"] == "normal"
    assert "avg_delay_ms_mean_mean" in aggregate_payload[0]
    assert "qos_violation_rate_mean_mean" in aggregate_payload[0]
    assert "transition_rate_mean_mean" in aggregate_payload[0]


def test_run_detail_endpoints_return_expected_shapes(api_client: TestClient, generated_run: Path) -> None:
    run_id = generated_run.name

    env_response = api_client.get(f"/api/runs/{run_id}/env", params={"base": "results"})
    summary_response = api_client.get(f"/api/runs/{run_id}/summary", params={"base": "results"})
    files_response = api_client.get(f"/api/runs/{run_id}/files", params={"base": "results"})
    per_step_response = api_client.get(f"/api/runs/{run_id}/per_step", params={"base": "results", "limit": 2})
    topology_response = api_client.get(f"/api/runs/{run_id}/topology", params={"base": "results"})
    timeline_response = api_client.get(f"/api/runs/{run_id}/timeline", params={"base": "results"})
    links_response = api_client.get(f"/api/runs/{run_id}/links", params={"base": "results", "step": 1})
    packet_events_response = api_client.get(
        f"/api/runs/{run_id}/packet_events",
        params={"base": "results", "step": 1},
    )

    assert env_response.status_code == 200
    assert env_response.json()["max_steps"] == 4

    assert summary_response.status_code == 200
    overall_summary = summary_response.json()["overall"]
    assert "qos_acceptance_status" in overall_summary
    assert "qos_thresholds" in overall_summary
    assert "qos_violation_rate_mean" in overall_summary
    assert "stability_status" in overall_summary
    assert "transition_rate_mean" in overall_summary

    assert files_response.status_code == 200
    assert "per_step.csv" in files_response.json()["files"]

    assert per_step_response.status_code == 200
    per_step_payload = per_step_response.json()
    assert len(per_step_payload) == 2
    assert {"run_id", "policy", "scenario", "step"} <= set(per_step_payload[0])

    assert topology_response.status_code == 200
    topology_payload = topology_response.json()
    assert topology_payload["run_id"] == run_id
    assert topology_payload["nodes"]
    assert topology_payload["edges"]

    assert timeline_response.status_code == 200
    timeline_payload = timeline_response.json()
    assert timeline_payload
    assert {"t", "metrics", "links_on", "packet_events"} <= set(timeline_payload[0])

    assert links_response.status_code == 200
    assert links_response.json()["step"] == 1
    assert isinstance(links_response.json()["links_on"], dict)

    assert packet_events_response.status_code == 200
    packet_events_payload = packet_events_response.json()
    assert packet_events_payload["step"] == 1
    assert isinstance(packet_events_payload["events"], list)


def test_missing_run_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/runs/does-not-exist/summary", params={"base": "results"})

    assert response.status_code == 404


def test_final_evaluation_endpoint_returns_latest_valid_artifact(
    api_client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    newer_dir = tmp_path / "artifacts" / "final_pipeline" / "demo" / "summary" / "final_evaluation"
    newer_dir.mkdir(parents=True)
    older_dir = tmp_path / "experiments" / "matrix_v1" / "final_evaluation"
    older_dir.mkdir(parents=True)

    valid_payload = {
        "generated_at_utc": "2026-03-22T00:00:00+00:00",
        "summary_rows": [
            {
                "scope_type": "overall",
                "scope": "ALL",
                "scenario": "ALL",
                "policy": "ppo",
                "policy_class": "ai_policy",
                "hypothesis_status": "not_achieved",
            }
        ],
    }
    older_payload = {
        "generated_at_utc": "2026-03-21T00:00:00+00:00",
        "summary_rows": [
            {
                "scope_type": "overall",
                "scope": "ALL",
                "scenario": "ALL",
                "policy": "heuristic",
                "policy_class": "heuristic_baseline",
                "hypothesis_status": "not_achieved",
            }
        ],
    }

    newer_summary = newer_dir / api_app.FINAL_EVALUATION_SUMMARY_FILENAME
    newer_summary.write_text(json.dumps(valid_payload), encoding="utf-8")
    (newer_dir / api_app.FINAL_EVALUATION_REPORT_FILENAME).write_text("# report", encoding="utf-8")

    older_summary = older_dir / api_app.FINAL_EVALUATION_SUMMARY_FILENAME
    older_summary.write_text(json.dumps(older_payload), encoding="utf-8")

    monkeypatch.setattr(api_app, "REPO_ROOT", tmp_path)
    api_app._latest_final_evaluation_artifact.cache_clear()

    response = api_client.get("/api/final_evaluation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["generated_at_utc"] == "2026-03-22T00:00:00+00:00"
    assert payload["artifact"]["summary_path"].endswith("artifacts/final_pipeline/demo/summary/final_evaluation/final_evaluation_summary.json")
    assert payload["artifact"]["report_path"].endswith("artifacts/final_pipeline/demo/summary/final_evaluation/final_evaluation_report.md")


def test_final_evaluation_endpoint_prefers_database_payload(
    api_client: TestClient,
    repo_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "greennet.sqlite3"
    monkeypatch.setenv("GREENNET_DB_PATH", str(db_path))
    repository = get_run_repository(db_path)
    repository.upsert_final_evaluation(
        output_dir=str(repo_root / "artifacts" / "final_pipeline" / "official_acceptance_v1"),
        payload={
            "generated_at_utc": "2026-04-10T08:00:00+00:00",
            "source": {
                "matrix_id": "official_acceptance_v1",
                "matrix_name": "GreenNet Official Acceptance Matrix v1",
                "matrix_manifest": str(
                    repo_root / "configs" / "acceptance_matrices" / "official_acceptance_v1.json"
                ),
                "matrix_tag": "official_acceptance_v1",
            },
            "summary_rows": [
                {
                    "scope_type": "overall",
                    "scope": "ALL",
                    "policy": "ppo",
                    "policy_class": "ai_policy",
                    "hypothesis_status": "not_achieved",
                }
            ],
        },
        summary_path=str(
            repo_root
            / "artifacts"
            / "final_pipeline"
            / "official_acceptance_v1"
            / "summary"
            / "final_evaluation"
            / api_app.FINAL_EVALUATION_SUMMARY_FILENAME
        ),
        report_path=str(
            repo_root
            / "artifacts"
            / "final_pipeline"
            / "official_acceptance_v1"
            / "summary"
            / "final_evaluation"
            / api_app.FINAL_EVALUATION_REPORT_FILENAME
        ),
        source_summary_csv=str(
            repo_root
            / "artifacts"
            / "final_pipeline"
            / "official_acceptance_v1"
            / "summary"
            / "results_summary_official_acceptance_v1.csv"
        ),
    )

    api_app._latest_final_evaluation_artifact.cache_clear()

    response = api_client.get("/api/final_evaluation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["generated_at_utc"] == "2026-04-10T08:00:00+00:00"
    assert payload["source"]["matrix_id"] == "official_acceptance_v1"
    assert payload["artifact"]["output_dir"].endswith("artifacts/final_pipeline/official_acceptance_v1")
    assert payload["artifact"]["summary_path"].endswith(
        "artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json"
    )
