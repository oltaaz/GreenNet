from __future__ import annotations

import csv
import json
import subprocess
import sys

import pytest


pytestmark = pytest.mark.integration


def _write_manifest(path, *, ai_policies: list[str] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matrix_id": "test_acceptance_matrix",
                "matrix_name": "Test Acceptance Matrix",
                "tag": "test_acceptance_matrix",
                "description": "Small manifest-driven integration test",
                "policies": ["all_on", "heuristic"],
                "baseline_policies": ["all_on", "heuristic"],
                "ai_policies": ai_policies or [],
                "primary_baseline_policy": "all_on",
                "routing_baseline": "min_hop_single_path",
                "routing_link_cost_model": "unit",
                "seeds": [0],
                "episodes": 1,
                "steps": 4,
                "deterministic": True,
                "cases": [
                    {
                        "id": "small_normal",
                        "label": "Small normal",
                        "scenario": "normal",
                        "topology_name": "small",
                    },
                    {
                        "id": "small_replay",
                        "label": "Small replay",
                        "scenario": "custom",
                        "topology_name": "small",
                        "traffic_name": "regional_ring_commuter_matrices",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_run_matrix_supports_acceptance_manifest(tmp_path, repo_root) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(manifest_path)

    results_dir = tmp_path / "results"
    summary_path = results_dir / "results_summary_test_acceptance_matrix.csv"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "run_matrix.py"),
            "--matrix-manifest",
            str(manifest_path),
            "--out-dir",
            str(results_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "run_matrix.py with acceptance manifest failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert {row["matrix_case_id"] for row in rows} == {"small_normal", "small_replay"}
    assert {row["matrix_id"] for row in rows} == {"test_acceptance_matrix"}
    assert {row["topology_name"] for row in rows} == {"small"}
    assert {row["traffic_name"] for row in rows} == {"", "regional_ring_commuter_matrices"}


def test_final_pipeline_uses_acceptance_manifest_as_authoritative_filter(tmp_path, repo_root) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(manifest_path)

    results_dir = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "run_matrix.py"),
            "--matrix-manifest",
            str(manifest_path),
            "--out-dir",
            str(results_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )

    summary_csv = results_dir / "results_summary_test_acceptance_matrix.csv"
    output_dir = tmp_path / "artifacts"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "greennet.evaluation.final_pipeline",
            "--matrix-manifest",
            str(manifest_path),
            "--skip-eval",
            "--summary-csv",
            str(summary_csv),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "final_pipeline with acceptance manifest failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    pipeline_config = json.loads((output_dir / "metadata" / "pipeline_config.json").read_text(encoding="utf-8"))
    final_payload = json.loads(
        (output_dir / "summary" / "final_evaluation" / "final_evaluation_summary.json").read_text(encoding="utf-8")
    )

    assert pipeline_config["acceptance_matrix"]["matrix_id"] == "test_acceptance_matrix"
    assert pipeline_config["acceptance_matrix"]["matrix_case_count"] == 2
    assert final_payload["source"]["matrix_id"] == "test_acceptance_matrix"
    assert set(final_payload["source"]["matrix_case_ids"]) == {"small_normal", "small_replay"}
