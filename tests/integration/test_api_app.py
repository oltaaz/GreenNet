from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api_app


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

    assert aggregate.status_code == 200
    aggregate_payload = aggregate.json()
    assert len(aggregate_payload) == 1
    assert aggregate_payload[0]["group"]["policy"] == "noop"
    assert aggregate_payload[0]["group"]["scenario"] == "normal"


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
    assert "overall" in summary_response.json()

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
                "policy_class": "ai",
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
                "policy_class": "baseline",
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
