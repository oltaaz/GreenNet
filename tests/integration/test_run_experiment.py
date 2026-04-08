from __future__ import annotations

import csv
import json

import pytest


pytestmark = pytest.mark.integration


def test_run_experiment_writes_expected_artifacts(tmp_path, run_experiment_cli) -> None:
    run_dir, completed = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="normal",
        seed=17,
        steps=4,
        episodes=2,
    )

    files = {path.name for path in run_dir.iterdir()}
    assert {"env_config.json", "per_step.csv", "run_meta.json", "summary.json"} <= files
    assert "[run_experiment] results saved to" in completed.stdout

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))

    with (run_dir / "per_step.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert summary["overall"]["episodes"] == 2
    assert len(summary["episodes"]) == 2
    assert run_meta["policy"] == "noop"
    assert run_meta["controller_policy"] == "all_on"
    assert run_meta["controller_policy_class"] == "traditional_baseline"
    assert run_meta["routing_baseline"] == "min_hop_single_path"
    assert run_meta["routing_link_cost_model"] == "unit"
    assert run_meta["scenario"] == "normal"
    assert run_meta["seed"] == 17
    assert run_meta["episodes"] == 2
    assert env_config["max_steps"] == 4
    assert env_config["routing_baseline"] == "min_hop_single_path"
    assert rows[0]["policy"] == "noop"
    assert rows[0]["scenario"] == "normal"


def test_run_experiment_accepts_explicit_routing_baseline(tmp_path, run_experiment_cli) -> None:
    run_dir, _completed = run_experiment_cli(
        tmp_path,
        policy="heuristic",
        scenario="normal",
        seed=19,
        steps=3,
        episodes=1,
        extra_args=["--routing-baseline", "ospf_ecmp", "--routing-link-cost-model", "unit"],
    )

    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))

    assert run_meta["policy"] == "heuristic"
    assert run_meta["controller_policy"] == "utilization_threshold"
    assert run_meta["routing_baseline"] == "ospf_ecmp"
    assert run_meta["routing_path_split"] == "ecmp"
    assert env_config["routing_baseline"] == "ospf_ecmp"
    assert env_config["routing_link_cost_model"] == "unit"


def test_run_experiment_supports_custom_topology_and_traffic_files(tmp_path, run_experiment_cli) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    topology_path = inputs_dir / "custom_topology.json"
    topology_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "directed": False,
                "nodes": [0, 1, 2, 3],
                "edges": [
                    {"source": 0, "target": 1, "capacity": 9.0},
                    {"source": 1, "target": 2, "capacity": 9.0},
                    {"source": 2, "target": 3, "capacity": 9.0},
                    {"source": 3, "target": 0, "capacity": 9.0}
                ]
            }
        ),
        encoding="utf-8",
    )

    traffic_path = inputs_dir / "custom_traffic.json"
    traffic_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "node_count": 4,
                "repeat": True,
                "matrices": [
                    [
                        [0, 3, 0, 0],
                        [0, 0, 2, 0],
                        [0, 0, 0, 4],
                        [1, 0, 0, 0]
                    ],
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [2, 0, 0, 0]
                    ]
                ]
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "custom_inputs.json"
    config_path.write_text(
        json.dumps(
            {
                "env": {
                    "topology_path": "inputs/custom_topology.json",
                    "traffic_path": "inputs/custom_traffic.json"
                }
            }
        ),
        encoding="utf-8",
    )

    run_dir, _ = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="custom",
        seed=19,
        steps=4,
        episodes=1,
        extra_args=["--config", str(config_path)],
    )

    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

    assert env_config["node_count"] == 4
    assert env_config["topology_path"] == str(topology_path.resolve())
    assert env_config["traffic_path"] == str(traffic_path.resolve())
    assert run_meta["scenario"] == "custom"
    assert run_meta["topology_path"] == str(topology_path.resolve())
    assert run_meta["traffic_path"] == str(traffic_path.resolve())
