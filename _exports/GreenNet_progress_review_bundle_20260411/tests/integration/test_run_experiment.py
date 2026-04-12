from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from greennet.persistence import get_run_repository


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
    assert run_meta["policy"] == "all_on"
    assert run_meta["policy_requested"] == "noop"
    assert run_meta["policy_class"] == "traditional_baseline"
    assert run_meta["controller_policy"] == "all_on"
    assert run_meta["controller_policy_class"] == "traditional_baseline"
    assert run_meta["routing_baseline"] == "min_hop_single_path"
    assert run_meta["routing_link_cost_model"] == "unit"
    assert run_meta["scenario"] == "normal"
    assert run_meta["seed"] == 17
    assert run_meta["episodes"] == 2
    assert run_meta["qos_policy_name"] == "official_qos_v1"
    assert run_meta["qos_policy_signature"]
    assert env_config["max_steps"] == 4
    assert env_config["initial_off_edges"] == 0
    assert env_config["disable_off_actions"] is True
    assert env_config["max_total_toggles_per_episode"] == 0
    assert env_config["routing_baseline"] == "min_hop_single_path"
    assert summary["overall"]["qos_policy_name"] == "official_qos_v1"
    assert "qos_acceptance_status" in summary["overall"]
    assert summary["overall"]["qos_thresholds"]["normalized_drop_ratio_threshold"] == pytest.approx(
        env_config["qos_target_norm_drop"]
    )
    assert rows[0]["policy"] == "all_on"
    assert rows[0]["scenario"] == "normal"


def test_run_experiment_normalizes_learned_policy_eval_start_to_all_on(tmp_path, run_experiment_cli) -> None:
    model_path = Path(
        "/Users/oltazagraxha/Desktop/GreenNet/artifacts/models/official_acceptance_v1/small/ppo_greennet.zip"
    )
    if not model_path.exists():
        pytest.skip("official small PPO checkpoint missing")

    run_dir, completed = run_experiment_cli(
        tmp_path,
        policy="ppo",
        scenario="normal",
        seed=13,
        steps=3,
        episodes=1,
        extra_args=["--topology-name", "small", "--model", str(model_path)],
    )

    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert env_config["initial_off_edges"] == 0
    assert env_config["disable_off_actions"] is False
    assert env_config["cost_estimator_enabled"] is False
    assert env_config["decision_interval_steps"] == 1
    assert env_config["off_calm_steps_required"] == 5
    assert env_config["util_block_threshold"] == pytest.approx(0.85)
    assert run_meta["policy"] == "ppo"


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
    assert run_meta["policy_class"] == "heuristic_baseline"
    assert run_meta["controller_policy_class"] == "energy_aware_heuristic"


def test_run_experiment_supports_packaged_named_topologies(tmp_path, run_experiment_cli) -> None:
    run_dir, _completed = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="normal",
        seed=23,
        steps=3,
        episodes=1,
        extra_args=["--topology-name", "large"],
    )

    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))

    assert run_meta["topology_name"] == "large"
    assert run_meta["topology_seed"] == 0
    assert env_config["topology_name"] == "large"
    assert env_config["node_count"] == 12
    assert env_config["directed"] is False


def test_run_experiment_supports_named_traffic_profiles_with_matching_topology(tmp_path, run_experiment_cli) -> None:
    run_dir, _completed = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="custom",
        seed=29,
        steps=4,
        episodes=1,
        extra_args=["--topology-name", "small", "--traffic-name", "regional_ring_commuter_matrices"],
    )

    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))

    assert run_meta["topology_name"] == "small"
    assert run_meta["traffic_name"] == "regional_ring_commuter_matrices"
    assert run_meta["traffic_mode"] == "replay"
    assert env_config["node_count"] == 6
    assert env_config["traffic_name"] == "regional_ring_commuter_matrices"


def test_run_experiment_supports_extended_traffic_scenarios(tmp_path, run_experiment_cli) -> None:
    run_dir, _completed = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="flash_crowd",
        seed=31,
        steps=4,
        episodes=1,
        extra_args=["--topology-name", "large"],
    )

    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))

    assert run_meta["traffic_scenario"] == "flash_crowd"
    assert run_meta["traffic_model"] == "stochastic"
    assert env_config["traffic_scenario"] == "flash_crowd"
    assert env_config["node_count"] == 12


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


def test_aggregate_results_preserves_traffic_identity(tmp_path, repo_root, run_experiment_cli) -> None:
    results_dir = tmp_path / "results"
    run_dir, _ = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="custom",
        seed=37,
        steps=4,
        episodes=1,
        extra_args=["--topology-name", "large", "--traffic-name", "backbone_large_flash_crowd_bursts"],
    )

    output_path = tmp_path / "results_summary.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "aggregate_results.py"),
            "--out-dir",
            str(results_dir),
            "--output",
            str(output_path),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "aggregate_results.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["traffic_name"] == "backbone_large_flash_crowd_bursts"
    assert rows[0]["qos_policy_name"] == "official_qos_v1"
    assert rows[0]["qos_acceptance_status"] != ""


def test_run_experiment_accepts_top_level_qos_config_overrides(tmp_path, run_experiment_cli) -> None:
    config_path = tmp_path / "qos_config.json"
    config_path.write_text(
        json.dumps(
            {
                "qos_target_norm_drop": 0.05,
                "qos_min_volume": 100.0,
                "qos_avg_delay_guard_multiplier": 3.0,
                "qos_avg_delay_guard_margin_ms": 11.0,
            }
        ),
        encoding="utf-8",
    )

    run_dir, _ = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="normal",
        seed=41,
        steps=4,
        episodes=1,
        extra_args=["--config", str(config_path)],
    )

    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

    assert env_config["qos_target_norm_drop"] == pytest.approx(0.05)
    assert env_config["qos_min_volume"] == pytest.approx(100.0)
    assert env_config["qos_avg_delay_guard_multiplier"] == pytest.approx(3.0)
    assert env_config["qos_avg_delay_guard_margin_ms"] == pytest.approx(11.0)
    assert run_meta["qos_target_norm_drop"] == pytest.approx(0.05)
    assert run_meta["qos_avg_delay_guard_multiplier"] == pytest.approx(3.0)


def test_run_experiment_persists_stability_metadata_and_aggregate_fields(
    tmp_path,
    repo_root,
    run_experiment_cli,
) -> None:
    config_path = tmp_path / "stability_config.json"
    config_path.write_text(
        json.dumps(
            {
                "stability_reversal_window_steps": 12,
                "stability_reversal_penalty": 0.2,
                "stability_min_steps_for_assessment": 1,
                "stability_max_transition_rate": 0.5,
                "stability_max_flap_rate": 0.5,
                "stability_max_flap_count": 4,
            }
        ),
        encoding="utf-8",
    )

    run_dir, _ = run_experiment_cli(
        tmp_path,
        policy="heuristic",
        scenario="normal",
        seed=43,
        steps=4,
        episodes=1,
        extra_args=["--config", str(config_path)],
    )

    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    overall = summary["overall"]

    assert run_meta["stability_policy_name"] == "official_stability_v1"
    assert run_meta["stability_policy_signature"]
    assert run_meta["stability_reversal_window_steps"] == 12
    assert run_meta["stability_reversal_penalty"] == pytest.approx(0.2)
    assert overall["stability_status"] != ""
    assert "transition_rate_mean" in overall
    assert "flap_rate_mean" in overall

    output_path = tmp_path / "results_summary_stability.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "aggregate_results.py"),
            "--out-dir",
            str(tmp_path / "results"),
            "--output",
            str(output_path),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "aggregate_results.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["stability_policy_name"] == "official_stability_v1"
    assert rows[0]["stability_policy_signature"] != ""
    assert rows[0]["transition_rate_mean"] != ""
    assert rows[0]["stability_status"] != ""


def test_run_experiment_persists_energy_model_config_and_aggregate_power_metrics(
    tmp_path,
    repo_root,
    run_experiment_cli,
) -> None:
    config_path = tmp_path / "energy_model.json"
    config_path.write_text(
        json.dumps(
            {
                "env": {
                    "power_network_fixed_watts": 10.0,
                    "power_device_active_watts": 5.0,
                    "power_device_sleep_watts": 1.0,
                    "power_device_dynamic_watts": 0.0,
                    "power_link_active_watts": 3.0,
                    "power_link_sleep_watts": 0.5,
                    "power_link_dynamic_watts": 0.0,
                    "power_utilization_sensitive": False,
                    "power_transition_on_joules": 3600.0,
                    "power_transition_off_joules": 7200.0,
                    "carbon_base_intensity_g_per_kwh": 123.0,
                    "carbon_amplitude_g_per_kwh": 0.0,
                    "carbon_period_seconds": 86400.0,
                }
            }
        ),
        encoding="utf-8",
    )

    run_dir, _ = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="normal",
        seed=41,
        steps=4,
        episodes=1,
        extra_args=["--config", str(config_path)],
    )

    env_config = json.loads((run_dir / "env_config.json").read_text(encoding="utf-8"))
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    overall = summary["overall"]

    assert env_config["power_network_fixed_watts"] == pytest.approx(10.0)
    assert env_config["power_device_dynamic_watts"] == pytest.approx(0.0)
    assert env_config["power_link_dynamic_watts"] == pytest.approx(0.0)
    assert env_config["power_utilization_sensitive"] is False
    assert env_config["power_transition_on_joules"] == pytest.approx(3600.0)
    assert env_config["power_transition_off_joules"] == pytest.approx(7200.0)
    assert env_config["carbon_base_intensity_g_per_kwh"] == pytest.approx(123.0)
    assert env_config["carbon_amplitude_g_per_kwh"] == pytest.approx(0.0)
    assert run_meta["energy_model_name"] == "active_sleep_linear"
    assert run_meta["carbon_model_name"] == "diurnal_sinusoid"
    assert run_meta["power_utilization_sensitive"] is False
    assert run_meta["power_transition_on_joules"] == pytest.approx(3600.0)
    assert run_meta["power_transition_off_joules"] == pytest.approx(7200.0)

    assert float(overall["power_variable_watts_mean"]) == pytest.approx(0.0)
    assert float(overall["power_total_watts_mean"]) == pytest.approx(float(overall["power_fixed_watts_mean"]))
    assert float(overall["carbon_g_total_mean"]) == pytest.approx(float(overall["energy_kwh_total_mean"]) * 123.0)

    output_path = tmp_path / "results_summary_energy.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "aggregate_results.py"),
            "--out-dir",
            str(tmp_path / "results"),
            "--output",
            str(output_path),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "aggregate_results.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["results_dir"] == str(run_dir)
    assert float(rows[0]["power_total_watts_mean"]) == pytest.approx(float(overall["power_total_watts_mean"]))
    assert float(rows[0]["power_fixed_watts_mean"]) == pytest.approx(float(overall["power_fixed_watts_mean"]))
    assert float(rows[0]["power_variable_watts_mean"]) == pytest.approx(float(overall["power_variable_watts_mean"]))
    assert float(rows[0]["energy_kwh_total_mean"]) == pytest.approx(float(overall["energy_kwh_total_mean"]))
    assert float(rows[0]["carbon_g_total_mean"]) == pytest.approx(float(overall["carbon_g_total_mean"]))
    assert rows[0]["energy_model_name"] == "active_sleep_linear"
    assert rows[0]["carbon_model_name"] == "diurnal_sinusoid"
    assert rows[0]["power_utilization_sensitive"].lower() == "false"
    assert float(rows[0]["power_transition_on_joules"]) == pytest.approx(3600.0)
    assert float(rows[0]["power_transition_off_joules"]) == pytest.approx(7200.0)


def test_run_experiment_persists_full_official_story_to_sqlite_and_export_summary(
    tmp_path,
    repo_root: Path,
    run_experiment_cli,
    monkeypatch,
) -> None:
    db_path = tmp_path / "artifacts" / "db" / "greennet.sqlite3"
    monkeypatch.setenv("GREENNET_DB_PATH", str(db_path))

    run_dir, _ = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="custom",
        seed=47,
        steps=4,
        episodes=1,
        extra_args=[
            "--topology-name",
            "large",
            "--traffic-name",
            "backbone_large_flash_crowd_bursts",
            "--matrix-id",
            "official_acceptance_v1",
            "--matrix-name",
            "GreenNet Official Acceptance Matrix v1",
            "--matrix-manifest",
            str(repo_root / "configs" / "acceptance_matrices" / "official_acceptance_v1.json"),
            "--matrix-case-id",
            "large_flash_replay",
            "--matrix-case-label",
            "Large topology / replay flash bursts",
        ],
    )

    repo = get_run_repository(db_path)
    snapshots = repo.list_run_snapshots(base="results")
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["matrix_id"] == "official_acceptance_v1"
    assert snapshot["matrix_case_id"] == "large_flash_replay"
    assert snapshot["topology_name"] == "large"
    assert snapshot["traffic_name"] == "backbone_large_flash_crowd_bursts"
    assert snapshot["qos_acceptance_status"] is not None
    assert snapshot["stability_status"] is not None

    exported_path = tmp_path / "exported_summary.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "greennet.persistence",
            "export-summary",
            "--db-path",
            str(db_path),
            "--base",
            "results",
            "--output",
            str(exported_path),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "greennet.persistence export-summary failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    with exported_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["run_id"] == json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))["run_id"]
    assert rows[0]["matrix_id"] == "official_acceptance_v1"
    assert rows[0]["matrix_case_id"] == "large_flash_replay"
    assert rows[0]["energy_model_name"] == "active_sleep_linear"
    assert rows[0]["qos_policy_name"] == "official_qos_v1"
    assert rows[0]["stability_policy_name"] == "official_stability_v1"
