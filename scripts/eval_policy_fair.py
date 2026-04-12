#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = REPO_ROOT / "run_experiment.py"


@dataclass
class PolicySpec:
    name: str
    model_path: Path | None = None


def _parse_seed_list(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if text:
            values.append(int(text))
    return values


def _build_base_config(args: argparse.Namespace) -> dict[str, Any]:
    env = {
        "traffic_model": "stochastic",
        "traffic_scenario": None if args.scenario == "normal" else args.scenario,
        "traffic_scenario_version": 2,
        "traffic_scenario_intensity": 1.0,
        "traffic_scenario_duration": 1.0,
        "traffic_scenario_frequency": 1.0,
        "topology_name": args.topology_name,
        "topology_seed": args.topology_seed,
        "topology_randomize": False,
        "initial_off_edges": args.initial_off_edges,
        "decision_interval_steps": args.decision_interval_steps,
        "toggle_cooldown_steps": args.toggle_cooldown_steps,
        "global_toggle_cooldown_steps": args.global_toggle_cooldown_steps,
        "off_calm_steps_required": args.off_calm_steps_required,
        "adaptive_toggle_budgets": False,
        "max_off_toggles_per_episode": args.max_off_toggles_per_episode,
        "max_total_toggles_per_episode": args.max_total_toggles_per_episode,
        "max_emergency_on_toggles_per_episode": args.max_emergency_on_toggles_per_episode,
        "disable_off_actions": False,
        "energy_weight": args.energy_weight,
        "drop_penalty_lambda": args.drop_penalty_lambda,
        "normalize_drop": True,
        "qos_target_norm_drop": args.qos_target_norm_drop,
        "qos_violation_penalty_scale": args.qos_violation_penalty_scale,
        "qos_min_volume": args.qos_min_volume,
        "qos_guard_margin": args.qos_guard_margin,
        "qos_guard_margin_off": args.qos_guard_margin_off,
        "qos_guard_margin_on": args.qos_guard_margin_on,
        "qos_guard_penalty_scale": args.qos_guard_penalty_scale,
        "toggle_penalty": args.toggle_penalty,
        "toggle_apply_penalty": args.toggle_apply_penalty,
        "toggle_on_penalty_scale": args.toggle_on_penalty_scale,
        "toggle_off_penalty_scale": args.toggle_off_penalty_scale,
        "off_toggle_penalty_scale": args.off_toggle_penalty_scale,
        "toggle_attempt_penalty": args.toggle_attempt_penalty,
        "blocked_action_penalty": args.blocked_action_penalty,
        "revert_penalty_scale": args.revert_penalty_scale,
        "util_block_threshold": args.util_block_threshold,
        "max_util_off_allow_threshold": args.max_util_off_allow_threshold,
        "util_unblock_threshold": args.util_unblock_threshold,
        "cost_estimator_enabled": args.cost_estimator_enabled,
        "flows_per_step": args.flows_per_step,
        "base_capacity": args.base_capacity,
        "power_network_fixed_watts": 20.0,
        "power_device_active_watts": 12.0,
        "power_device_sleep_watts": 2.0,
        "power_device_dynamic_watts": 3.0,
        "power_link_active_watts": 6.0,
        "power_link_sleep_watts": 0.2,
        "power_link_dynamic_watts": 2.0,
        "carbon_base_intensity_g_per_kwh": 400.0,
        "carbon_amplitude_g_per_kwh": 25.0,
        "carbon_period_seconds": 86400.0,
        "traffic_avg_bursts_per_step": 3.0,
        "traffic_p_elephant": 0.08,
        "traffic_elephant_size_range": [6, 20],
        "traffic_duration_range": [1, 4],
        "traffic_spike_prob": 0.02,
        "traffic_spike_multiplier_range": [1.5, 4.0],
        "traffic_spike_duration_range": [2, 8],
        "traffic_hotspots": [[0, 5, 3.0], [2, 7, 2.0]],
        "enable_forecasting": True,
        "forecast_model": "adaptive_ema",
        "forecast_horizon_steps": 5,
        "forecast_adaptive_alphas": [0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
        "forecast_adaptive_error_alpha": 0.02,
        "forecast_adaptive_temperature": 0.25,
    }
    return {"env": env}


def _run_once(
    *,
    python_bin: str,
    config_path: Path,
    out_dir: Path,
    policy: PolicySpec,
    scenario: str,
    seed: int,
    steps: int,
    tag: str,
) -> dict[str, Any]:
    cmd = [
        python_bin,
        str(RUN_EXPERIMENT),
        "--config",
        str(config_path),
        "--policy",
        policy.name,
        "--scenario",
        scenario,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--episodes",
        "1",
        "--out-dir",
        str(out_dir),
        "--tag",
        tag,
    ]
    if policy.model_path is not None:
        cmd.extend(["--model", str(policy.model_path)])

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "run_experiment failed").strip()
        raise RuntimeError(detail[-3000:])

    marker = "[run_experiment] results saved to "
    run_dir: Path | None = None
    for line in (completed.stdout or "").splitlines():
        if line.startswith(marker):
            run_dir = Path(line[len(marker) :].strip())
            break
    if run_dir is None:
        candidates = [path for path in out_dir.iterdir() if path.is_dir()]
        if not candidates:
            raise RuntimeError("Run completed but no output directory was found")
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    overall = summary.get("overall", {})
    return {
        "policy": policy.name,
        "model_path": None if policy.model_path is None else str(policy.model_path),
        "scenario": scenario,
        "seed": seed,
        "steps": steps,
        "run_dir": str(run_dir),
        "reward_total_mean": float(overall.get("reward_total_mean", 0.0)),
        "energy_kwh_total_mean": float(overall.get("energy_kwh_total_mean", 0.0)),
        "dropped_total_mean": float(overall.get("dropped_total_mean", 0.0)),
        "avg_delay_ms_mean": float(overall.get("avg_delay_ms_mean", 0.0)),
        "active_ratio_mean": float(overall.get("active_ratio_mean", 0.0)),
        "toggles_applied_mean": float(overall.get("toggles_applied_mean", 0.0)),
        "delivered_total_mean": float(overall.get("delivered_total_mean", 0.0)),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def _summarize(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = policy_rows[0]
    return {
        "policy": first["policy"],
        "model_path": first["model_path"],
        "runs": len(policy_rows),
        "reward_total_mean": _mean(policy_rows, "reward_total_mean"),
        "energy_kwh_total_mean": _mean(policy_rows, "energy_kwh_total_mean"),
        "dropped_total_mean": _mean(policy_rows, "dropped_total_mean"),
        "avg_delay_ms_mean": _mean(policy_rows, "avg_delay_ms_mean"),
        "active_ratio_mean": _mean(policy_rows, "active_ratio_mean"),
        "toggles_applied_mean": _mean(policy_rows, "toggles_applied_mean"),
        "delivered_total_mean": _mean(policy_rows, "delivered_total_mean"),
    }


def _dominates(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        candidate["reward_total_mean"] >= baseline["reward_total_mean"]
        and candidate["energy_kwh_total_mean"] <= baseline["energy_kwh_total_mean"]
        and candidate["dropped_total_mean"] <= baseline["dropped_total_mean"]
        and candidate["avg_delay_ms_mean"] <= baseline["avg_delay_ms_mean"]
        and candidate["active_ratio_mean"] <= baseline["active_ratio_mean"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair same-condition benchmark for all_on, heuristic, and PPO checkpoints.")
    parser.add_argument("--scenario", default="normal", choices=["normal", "burst", "hotspot"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--topology-name", default="medium")
    parser.add_argument("--topology-seed", type=int, default=0)
    parser.add_argument("--initial-off-edges", type=int, default=0)
    parser.add_argument("--decision-interval-steps", type=int, default=1)
    parser.add_argument("--toggle-cooldown-steps", type=int, default=2)
    parser.add_argument("--global-toggle-cooldown-steps", type=int, default=1)
    parser.add_argument("--off-calm-steps-required", type=int, default=0)
    parser.add_argument("--max-off-toggles-per-episode", type=int, default=48)
    parser.add_argument("--max-total-toggles-per-episode", type=int, default=96)
    parser.add_argument("--max-emergency-on-toggles-per-episode", type=int, default=96)
    parser.add_argument("--energy-weight", type=float, default=2600.0)
    parser.add_argument("--drop-penalty-lambda", type=float, default=12.0)
    parser.add_argument("--qos-target-norm-drop", type=float, default=0.072)
    parser.add_argument("--qos-violation-penalty-scale", type=float, default=180.0)
    parser.add_argument("--qos-min-volume", type=float, default=2500.0)
    parser.add_argument("--qos-guard-margin", type=float, default=0.002)
    parser.add_argument("--qos-guard-margin-off", type=float, default=0.003)
    parser.add_argument("--qos-guard-margin-on", type=float, default=0.001)
    parser.add_argument("--qos-guard-penalty-scale", type=float, default=2.5)
    parser.add_argument("--toggle-penalty", type=float, default=0.0005)
    parser.add_argument("--toggle-apply-penalty", type=float, default=0.001)
    parser.add_argument("--toggle-on-penalty-scale", type=float, default=0.2)
    parser.add_argument("--toggle-off-penalty-scale", type=float, default=0.05)
    parser.add_argument("--off-toggle-penalty-scale", type=float, default=0.05)
    parser.add_argument("--toggle-attempt-penalty", type=float, default=0.0)
    parser.add_argument("--blocked-action-penalty", type=float, default=0.0)
    parser.add_argument("--revert-penalty-scale", type=float, default=0.25)
    parser.add_argument("--util-block-threshold", type=float, default=0.9)
    parser.add_argument("--max-util-off-allow-threshold", type=float, default=0.88)
    parser.add_argument("--util-unblock-threshold", type=float, default=0.82)
    parser.add_argument("--flows-per-step", type=int, default=6)
    parser.add_argument("--base-capacity", type=float, default=15.0)
    parser.add_argument("--cost-estimator-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate", action="append", required=True, help="Path to a PPO checkpoint to benchmark. Repeat for multiple candidates.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "fair_eval")
    args = parser.parse_args()

    seeds = _parse_seed_list(args.seeds)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "runs"
    work_dir.mkdir(parents=True, exist_ok=True)

    config_payload = _build_base_config(args)
    config_path = output_dir / "fair_eval_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    python_bin = str((REPO_ROOT / ".venv" / "bin" / "python")) if (REPO_ROOT / ".venv" / "bin" / "python").exists() else sys.executable
    policies = [PolicySpec("all_on"), PolicySpec("heuristic")]
    policies.extend(PolicySpec("ppo", Path(path).expanduser().resolve()) for path in args.candidate)

    rows: list[dict[str, Any]] = []
    for policy in policies:
        tag = f"fair-{policy.name}"
        if policy.model_path is not None:
            tag = f"{tag}-{policy.model_path.parent.name}"
        for seed in seeds:
            rows.append(
                _run_once(
                    python_bin=python_bin,
                    config_path=config_path,
                    out_dir=work_dir,
                    policy=policy,
                    scenario=args.scenario,
                    seed=seed,
                    steps=args.steps,
                    tag=tag,
                )
            )

    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), row["model_path"])
        grouped.setdefault(key, []).append(row)

    summaries = [_summarize(policy_rows) for policy_rows in grouped.values()]
    summaries.sort(key=lambda row: (row["policy"], str(row["model_path"] or "")))

    baseline_all_on = next(row for row in summaries if row["policy"] == "all_on")
    baseline_heuristic = next(row for row in summaries if row["policy"] == "heuristic")

    rankings: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["policy"] != "ppo":
            continue
        rankings.append(
            {
                **summary,
                "beats_all_on": _dominates(summary, baseline_all_on),
                "beats_heuristic": _dominates(summary, baseline_heuristic),
                "delta_vs_all_on_energy": summary["energy_kwh_total_mean"] - baseline_all_on["energy_kwh_total_mean"],
                "delta_vs_all_on_dropped": summary["dropped_total_mean"] - baseline_all_on["dropped_total_mean"],
                "delta_vs_all_on_delay": summary["avg_delay_ms_mean"] - baseline_all_on["avg_delay_ms_mean"],
                "delta_vs_all_on_active_ratio": summary["active_ratio_mean"] - baseline_all_on["active_ratio_mean"],
                "delta_vs_heuristic_energy": summary["energy_kwh_total_mean"] - baseline_heuristic["energy_kwh_total_mean"],
                "delta_vs_heuristic_dropped": summary["dropped_total_mean"] - baseline_heuristic["dropped_total_mean"],
                "delta_vs_heuristic_delay": summary["avg_delay_ms_mean"] - baseline_heuristic["avg_delay_ms_mean"],
                "delta_vs_heuristic_active_ratio": summary["active_ratio_mean"] - baseline_heuristic["active_ratio_mean"],
            }
        )

    rankings.sort(
        key=lambda row: (
            not row["beats_all_on"],
            not row["beats_heuristic"],
            row["delta_vs_heuristic_active_ratio"],
            row["delta_vs_heuristic_energy"],
            row["delta_vs_heuristic_dropped"],
            -row["reward_total_mean"],
        )
    )

    csv_path = output_dir / "fair_eval_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "fair_eval_summary.json"
    payload = {
        "scenario": args.scenario,
        "seeds": seeds,
        "steps": args.steps,
        "config_path": str(config_path),
        "summaries": summaries,
        "rankings": rankings,
        "recommended_checkpoint": rankings[0]["model_path"] if rankings else None,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Fair evaluation summary")
    print(f"scenario={args.scenario} seeds={seeds} steps={args.steps} topology={args.topology_name}:{args.topology_seed}")
    print()
    for summary in summaries:
        model_label = summary["model_path"] or "-"
        print(
            f"{summary['policy']:<9} "
            f"energy={summary['energy_kwh_total_mean']:.6f} "
            f"dropped={summary['dropped_total_mean']:.3f} "
            f"delay={summary['avg_delay_ms_mean']:.3f} "
            f"active={summary['active_ratio_mean']:.4f} "
            f"reward={summary['reward_total_mean']:.3f} "
            f"model={model_label}"
        )
    print()
    if rankings:
        best = rankings[0]
        print("Best PPO candidate")
        print(
            f"model={best['model_path']} beats_all_on={best['beats_all_on']} "
            f"beats_heuristic={best['beats_heuristic']} "
            f"delta_active_vs_heuristic={best['delta_vs_heuristic_active_ratio']:+.6f}"
        )
    print(f"per-run CSV: {csv_path}")
    print(f"summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
