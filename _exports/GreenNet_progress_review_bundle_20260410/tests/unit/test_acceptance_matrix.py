from __future__ import annotations

import json

from greennet.evaluation.acceptance_matrix import load_acceptance_matrix


def test_load_acceptance_matrix_supports_case_based_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matrix_id": "test_matrix",
                "matrix_name": "Test Matrix",
                "tag": "test_matrix",
                "description": "test",
                "policies": ["all_on", "heuristic"],
                "baseline_policies": ["all_on", "heuristic"],
                "ai_policies": [],
                "primary_baseline_policy": "all_on",
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

    matrix = load_acceptance_matrix(manifest_path)

    assert matrix.matrix_id == "test_matrix"
    assert matrix.policies == ("all_on", "heuristic")
    assert matrix.primary_baseline_policy == "all_on"
    assert matrix.cases[0].case_id == "small_normal"
    assert matrix.cases[1].traffic_name == "regional_ring_commuter_matrices"


def test_load_acceptance_matrix_rejects_replay_case_without_custom_scenario(tmp_path) -> None:
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matrix_id": "bad_matrix",
                "matrix_name": "Bad Matrix",
                "tag": "bad_matrix",
                "description": "test",
                "policies": ["all_on"],
                "baseline_policies": ["all_on"],
                "ai_policies": [],
                "primary_baseline_policy": "all_on",
                "seeds": [0],
                "episodes": 1,
                "steps": 4,
                "deterministic": True,
                "cases": [
                    {
                        "id": "bad_replay",
                        "label": "Bad replay",
                        "scenario": "normal",
                        "topology_name": "small",
                        "traffic_name": "regional_ring_commuter_matrices",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_acceptance_matrix(manifest_path)
    except ValueError as exc:
        assert "must set scenario='custom'" in str(exc)
    else:
        raise AssertionError("Expected load_acceptance_matrix() to reject replay case without custom scenario")
