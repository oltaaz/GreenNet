#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from greennet.evaluation.acceptance_matrix import load_acceptance_matrix
from greennet.evaluation.official_ppo import canonical_official_ppo_model_path, normalize_official_topology_name


RUN_DIR_RE = re.compile(r"\[run_experiment\] results saved to (.+)")

SUMMARY_COLUMNS = [
    "matrix_id",
    "matrix_name",
    "matrix_manifest",
    "matrix_case_id",
    "matrix_case_label",
    "run_id",
    "policy",
    "policy_class",
    "controller_policy",
    "controller_policy_class",
    "scenario",
    "seed",
    "episodes",
    "max_steps",
    "deterministic",
    "topology_seed",
    "topology_name",
    "topology_path",
    "traffic_seed",
    "traffic_mode",
    "traffic_model",
    "traffic_name",
    "traffic_path",
    "traffic_scenario",
    "traffic_scenario_version",
    "traffic_scenario_intensity",
    "traffic_scenario_duration",
    "traffic_scenario_frequency",
    "qos_policy_name",
    "qos_policy_signature",
    "qos_target_norm_drop",
    "qos_min_volume",
    "qos_avg_delay_guard_multiplier",
    "qos_avg_delay_guard_margin_ms",
    "stability_policy_name",
    "stability_policy_signature",
    "stability_reversal_window_steps",
    "stability_reversal_penalty",
    "stability_max_transition_rate",
    "stability_max_flap_rate",
    "stability_max_flap_count",
    "energy_model_name",
    "energy_model_signature",
    "power_utilization_sensitive",
    "power_transition_on_joules",
    "power_transition_off_joules",
    "carbon_model_name",
    "routing_baseline",
    "routing_link_cost_model",
    "reward_total_mean",
    "delivered_total_mean",
    "dropped_total_mean",
    "energy_kwh_total_mean",
    "energy_steady_kwh_total_mean",
    "energy_transition_kwh_total_mean",
    "carbon_g_total_mean",
    "power_total_watts_mean",
    "power_fixed_watts_mean",
    "power_variable_watts_mean",
    "power_transition_watts_mean",
    "active_devices_mean",
    "active_links_mean",
    "avg_utilization_mean",
    "active_ratio_mean",
    "delivery_loss_rate_mean",
    "avg_delay_ms_mean",
    "avg_path_latency_ms_mean",
    "qos_violation_rate_mean",
    "qos_violation_count_mean",
    "qos_acceptance_status",
    "qos_acceptance_missing",
    "transition_count_total_mean",
    "transition_on_count_total_mean",
    "transition_off_count_total_mean",
    "transition_rate_mean",
    "flap_event_count_total_mean",
    "flap_rate_mean",
    "stability_status",
    "stability_missing",
    "results_dir",
    "status",
    "error",
]


def _parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_seed_list(value: str) -> List[int]:
    return [int(part) for part in _parse_csv_list(value)]


def _parse_bool_arg(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def _list_run_dirs(out_dir: Path) -> List[Path]:
    if not out_dir.exists():
        return []
    return [path for path in out_dir.iterdir() if path.is_dir()]


def _extract_run_dir(stdout: str, stderr: str) -> Optional[Path]:
    match = None
    for line in (stdout + "\n" + stderr).splitlines():
        cand = RUN_DIR_RE.search(line)
        if cand:
            match = cand
    if match:
        return Path(match.group(1).strip())
    return None


def _detect_new_run_dir(before: Sequence[Path], after: Sequence[Path]) -> Optional[Path]:
    before_set = {path.resolve() for path in before}
    new_dirs = [path for path in after if path.resolve() not in before_set]
    if len(new_dirs) == 1:
        return new_dirs[0]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _find_latest_model(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    candidates: List[Path] = []
    candidates.extend(sorted(runs_dir.glob("*/ppo_greennet.zip")))
    candidates.extend(sorted(runs_dir.glob("*/ppo_greennet")))
    candidates.extend(sorted(runs_dir.glob("ppo_greennet.zip")))
    candidates.extend(sorted(runs_dir.glob("ppo_greennet")))
    return candidates[-1] if candidates else None


def _infer_topology_seed_from_model(model_path: Path) -> Optional[int]:
    cfg = _load_json(model_path.parent / "env_config.json")
    if not cfg:
        return None
    topo = cfg.get("topology_seed")
    if isinstance(topo, (int, float, str)):
        try:
            return int(topo)
        except Exception:
            return None
    seeds = cfg.get("topology_seeds")
    if isinstance(seeds, list) and seeds:
        try:
            return int(seeds[0])
        except Exception:
            return None
    return None


def _resolve_case_ppo_model(
    *,
    explicit_model_path: Path | None,
    fallback_latest_model: Path | None,
    topology_name: str | None,
) -> Path | None:
    if explicit_model_path is not None:
        return explicit_model_path
    normalized_topology = normalize_official_topology_name(topology_name)
    if normalized_topology is not None:
        canonical_path = canonical_official_ppo_model_path(normalized_topology)
        if canonical_path.exists():
            return canonical_path
    return fallback_latest_model


def _stderr_snip(stderr: str, limit: int = 300) -> str:
    s = (stderr or "").strip().replace("\n", " | ")
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


def _row_from_meta(
    run_meta: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    results_dir: Optional[Path],
    *,
    status: str,
    error: str,
    fallback_policy: str,
    fallback_scenario: str,
    fallback_seed: int,
    episodes: int,
    max_steps: int,
    deterministic: bool,
) -> Dict[str, Any]:
    meta = run_meta or {}
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}

    return {
        "matrix_id": meta.get("matrix_id", ""),
        "matrix_name": meta.get("matrix_name", ""),
        "matrix_manifest": meta.get("matrix_manifest", ""),
        "matrix_case_id": meta.get("matrix_case_id", ""),
        "matrix_case_label": meta.get("matrix_case_label", ""),
        "run_id": meta.get("run_id", ""),
        "policy": meta.get("policy", fallback_policy),
        "policy_class": meta.get("policy_class", ""),
        "controller_policy": meta.get("controller_policy", ""),
        "controller_policy_class": meta.get("controller_policy_class", ""),
        "scenario": meta.get("scenario", fallback_scenario),
        "seed": meta.get("seed", fallback_seed),
        "episodes": meta.get("episodes", episodes),
        "max_steps": meta.get("max_steps", max_steps),
        "deterministic": meta.get("deterministic", deterministic),
        "topology_seed": meta.get("topology_seed", ""),
        "topology_name": meta.get("topology_name", ""),
        "topology_path": meta.get("topology_path", ""),
        "traffic_seed": meta.get("traffic_seed", ""),
        "traffic_mode": meta.get("traffic_mode", ""),
        "traffic_model": meta.get("traffic_model", ""),
        "traffic_name": meta.get("traffic_name", ""),
        "traffic_path": meta.get("traffic_path", ""),
        "traffic_scenario": meta.get("traffic_scenario", ""),
        "traffic_scenario_version": meta.get("traffic_scenario_version", ""),
        "traffic_scenario_intensity": meta.get("traffic_scenario_intensity", ""),
        "traffic_scenario_duration": meta.get("traffic_scenario_duration", ""),
        "traffic_scenario_frequency": meta.get("traffic_scenario_frequency", ""),
        "qos_policy_name": meta.get("qos_policy_name", ""),
        "qos_policy_signature": meta.get("qos_policy_signature", ""),
        "qos_target_norm_drop": meta.get("qos_target_norm_drop", ""),
        "qos_min_volume": meta.get("qos_min_volume", ""),
        "qos_avg_delay_guard_multiplier": meta.get("qos_avg_delay_guard_multiplier", ""),
        "qos_avg_delay_guard_margin_ms": meta.get("qos_avg_delay_guard_margin_ms", ""),
        "stability_policy_name": meta.get("stability_policy_name", ""),
        "stability_policy_signature": meta.get("stability_policy_signature", ""),
        "stability_reversal_window_steps": meta.get("stability_reversal_window_steps", ""),
        "stability_reversal_penalty": meta.get("stability_reversal_penalty", ""),
        "stability_max_transition_rate": meta.get("stability_max_transition_rate", ""),
        "stability_max_flap_rate": meta.get("stability_max_flap_rate", ""),
        "stability_max_flap_count": meta.get("stability_max_flap_count", ""),
        "energy_model_name": meta.get("energy_model_name", ""),
        "energy_model_signature": meta.get("energy_model_signature", ""),
        "power_utilization_sensitive": meta.get("power_utilization_sensitive", ""),
        "power_transition_on_joules": meta.get("power_transition_on_joules", ""),
        "power_transition_off_joules": meta.get("power_transition_off_joules", ""),
        "carbon_model_name": meta.get("carbon_model_name", ""),
        "routing_baseline": meta.get("routing_baseline", ""),
        "routing_link_cost_model": meta.get("routing_link_cost_model", ""),
        "reward_total_mean": overall.get("reward_total_mean", ""),
        "delivered_total_mean": overall.get("delivered_total_mean", ""),
        "dropped_total_mean": overall.get("dropped_total_mean", ""),
        "energy_kwh_total_mean": overall.get("energy_kwh_total_mean", ""),
        "energy_steady_kwh_total_mean": overall.get("energy_steady_kwh_total_mean", ""),
        "energy_transition_kwh_total_mean": overall.get("energy_transition_kwh_total_mean", ""),
        "carbon_g_total_mean": overall.get("carbon_g_total_mean", ""),
        "power_total_watts_mean": overall.get("power_total_watts_mean", ""),
        "power_fixed_watts_mean": overall.get("power_fixed_watts_mean", ""),
        "power_variable_watts_mean": overall.get("power_variable_watts_mean", ""),
        "power_transition_watts_mean": overall.get("power_transition_watts_mean", ""),
        "active_devices_mean": overall.get("active_devices_mean", ""),
        "active_links_mean": overall.get("active_links_mean", ""),
        "avg_utilization_mean": overall.get("avg_utilization_mean", ""),
        "active_ratio_mean": overall.get("active_ratio_mean", ""),
        "delivery_loss_rate_mean": overall.get("delivery_loss_rate_mean", ""),
        "avg_delay_ms_mean": overall.get("avg_delay_ms_mean", ""),
        "avg_path_latency_ms_mean": overall.get("avg_path_latency_ms_mean", ""),
        "qos_violation_rate_mean": overall.get("qos_violation_rate_mean", ""),
        "qos_violation_count_mean": overall.get("qos_violation_count_mean", ""),
        "qos_acceptance_status": overall.get("qos_acceptance_status", ""),
        "qos_acceptance_missing": overall.get("qos_acceptance_missing", ""),
        "transition_count_total_mean": overall.get("transition_count_total_mean", ""),
        "transition_on_count_total_mean": overall.get("transition_on_count_total_mean", ""),
        "transition_off_count_total_mean": overall.get("transition_off_count_total_mean", ""),
        "transition_rate_mean": overall.get("transition_rate_mean", ""),
        "flap_event_count_total_mean": overall.get("flap_event_count_total_mean", ""),
        "flap_rate_mean": overall.get("flap_rate_mean", ""),
        "stability_status": overall.get("stability_status", ""),
        "stability_missing": overall.get("stability_missing", ""),
        "results_dir": str(results_dir) if results_dir is not None else "",
        "status": status,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matrix of GreenNet experiments.")
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        default=None,
        help="Optional acceptance-matrix JSON manifest. When provided, it becomes the canonical source of policies, seeds, cases, and evaluation identity.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--scenarios", type=str, default="normal,burst,hotspot")
    parser.add_argument("--policies", type=str, default="all_on,heuristic,ppo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--ppo-model", type=Path, default=None, help="Optional PPO model path to use for all PPO runs.")
    parser.add_argument(
        "--topology-seed",
        type=int,
        default=None,
        help="Force topology seed for ALL runs. If omitted and PPO is included, inferred from PPO model env_config.json.",
    )
    parser.add_argument("--topology-name", type=str, default=None, help="Use a packaged named topology for all runs.")
    parser.add_argument("--topology-path", type=Path, default=None, help="Use a custom topology JSON file for all runs.")
    parser.add_argument("--traffic-seed", type=int, default=None, help="Force traffic seed base for all runs.")
    parser.add_argument("--traffic-model", type=str, default=None, help="Override traffic model for all runs.")
    parser.add_argument("--traffic-name", type=str, default=None, help="Use a packaged named traffic replay profile for all runs.")
    parser.add_argument("--traffic-path", type=Path, default=None, help="Use a custom traffic replay JSON file for all runs.")
    parser.add_argument("--traffic-scenario", type=str, default=None, help="Override traffic scenario for all runs.")
    parser.add_argument("--traffic-scenario-version", type=int, default=None)
    parser.add_argument("--traffic-scenario-intensity", type=float, default=None)
    parser.add_argument("--traffic-scenario-duration", type=float, default=None)
    parser.add_argument("--traffic-scenario-frequency", type=float, default=None)
    parser.add_argument("--stability-reversal-window-steps", type=int, default=None)
    parser.add_argument("--stability-reversal-penalty", type=float, default=None)
    parser.add_argument("--stability-min-steps-for-assessment", type=int, default=None)
    parser.add_argument("--stability-max-transition-rate", type=float, default=None)
    parser.add_argument("--stability-max-flap-rate", type=float, default=None)
    parser.add_argument("--stability-max-flap-count", type=int, default=None)
    parser.add_argument("--power-utilization-sensitive", type=_parse_bool_arg, default=None)
    parser.add_argument("--power-transition-on-joules", type=float, default=None)
    parser.add_argument("--power-transition-off-joules", type=float, default=None)
    parser.add_argument("--routing-baseline", type=str, default=None)
    parser.add_argument("--routing-link-cost-model", type=str, default=None)
    args = parser.parse_args()

    matrix_manifest = load_acceptance_matrix(args.matrix_manifest) if args.matrix_manifest is not None else None

    seeds = list(matrix_manifest.seeds) if matrix_manifest is not None else _parse_seed_list(args.seeds)
    scenarios = (
        sorted({case.scenario for case in matrix_manifest.cases})
        if matrix_manifest is not None
        else _parse_csv_list(args.scenarios)
    )
    policies = list(matrix_manifest.policies) if matrix_manifest is not None else _parse_csv_list(args.policies)
    episodes = int(matrix_manifest.episodes) if matrix_manifest is not None else int(args.episodes)
    steps = int(matrix_manifest.steps) if matrix_manifest is not None else int(args.steps)
    deterministic = bool(matrix_manifest.deterministic) if matrix_manifest is not None else bool(args.deterministic)
    routing_baseline = (
        matrix_manifest.routing_baseline if matrix_manifest is not None else args.routing_baseline
    )
    routing_link_cost_model = (
        matrix_manifest.routing_link_cost_model if matrix_manifest is not None else args.routing_link_cost_model
    )
    tag = matrix_manifest.tag if matrix_manifest is not None else args.tag
    if matrix_manifest is not None:
        matrix_cases = list(matrix_manifest.cases)
    else:
        matrix_cases = scenarios

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_name = "results_summary.csv"
    if tag:
        summary_name = f"results_summary_{tag}.csv"
    summary_path = args.out_dir / summary_name

    global_topology_name = str(args.topology_name).strip() if args.topology_name else None
    global_topology_path = args.topology_path.expanduser().resolve() if args.topology_path is not None else None
    global_traffic_name = str(args.traffic_name).strip() if args.traffic_name else None
    global_traffic_path = args.traffic_path.expanduser().resolve() if args.traffic_path is not None else None

    # Determine a single fixed topology seed for fairness + PPO action-space consistency.
    fixed_topology_seed: Optional[int] = args.topology_seed
    ppo_model_path = args.ppo_model
    if ppo_model_path is not None and not ppo_model_path.exists():
        raise SystemExit(f"--ppo-model does not exist: {ppo_model_path}")
    if fixed_topology_seed is None and global_topology_name is None and global_topology_path is None and "ppo" in policies:
        mp = ppo_model_path or _find_latest_model(args.runs_dir)
        if mp is None:
            # PPO requested but no model found -> we will mark PPO runs as skipped.
            fixed_topology_seed = None
        else:
            inferred = _infer_topology_seed_from_model(mp)
            fixed_topology_seed = inferred if inferred is not None else 0
    if fixed_topology_seed is None and global_topology_name is None and global_topology_path is None:
        # Default fallback if PPO not requested or inference not possible
        fixed_topology_seed = 0

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()

        fallback_model_path = ppo_model_path or _find_latest_model(args.runs_dir)

        for case in matrix_cases:
            if matrix_manifest is not None:
                case_id = case.case_id
                case_label = case.label
                explicit_topology_name = case.topology_name
                explicit_topology_path = Path(case.topology_path) if case.topology_path else global_topology_path
                explicit_traffic_name = case.traffic_name
                explicit_traffic_path = Path(case.traffic_path) if case.traffic_path else global_traffic_path
                case_topology_seed = case.topology_seed if case.topology_seed is not None else fixed_topology_seed
                case_traffic_model = case.traffic_model if case.traffic_model is not None else args.traffic_model
                case_traffic_scenario = case.traffic_scenario if case.traffic_scenario is not None else args.traffic_scenario
                case_traffic_scenario_version = (
                    case.traffic_scenario_version if case.traffic_scenario_version is not None else args.traffic_scenario_version
                )
                case_traffic_scenario_intensity = (
                    case.traffic_scenario_intensity if case.traffic_scenario_intensity is not None else args.traffic_scenario_intensity
                )
                case_traffic_scenario_duration = (
                    case.traffic_scenario_duration if case.traffic_scenario_duration is not None else args.traffic_scenario_duration
                )
                case_traffic_scenario_frequency = (
                    case.traffic_scenario_frequency if case.traffic_scenario_frequency is not None else args.traffic_scenario_frequency
                )
                case_scenario = case.scenario
            else:
                case_id = ""
                case_label = ""
                explicit_topology_name = global_topology_name
                explicit_topology_path = global_topology_path
                explicit_traffic_name = global_traffic_name
                explicit_traffic_path = global_traffic_path
                case_topology_seed = fixed_topology_seed
                case_traffic_model = args.traffic_model
                case_traffic_scenario = args.traffic_scenario
                case_traffic_scenario_version = args.traffic_scenario_version
                case_traffic_scenario_intensity = args.traffic_scenario_intensity
                case_traffic_scenario_duration = args.traffic_scenario_duration
                case_traffic_scenario_frequency = args.traffic_scenario_frequency
                case_scenario = str(case)

            for seed in seeds:
                for policy in policies:
                    case_model_path = _resolve_case_ppo_model(
                        explicit_model_path=ppo_model_path,
                        fallback_latest_model=fallback_model_path,
                        topology_name=explicit_topology_name,
                    )
                    if policy == "ppo" and case_model_path is None:
                        row = _row_from_meta(
                            {
                                "matrix_id": matrix_manifest.matrix_id if matrix_manifest is not None else "",
                                "matrix_name": matrix_manifest.matrix_name if matrix_manifest is not None else "",
                                "matrix_manifest": matrix_manifest.manifest_path if matrix_manifest is not None else "",
                                "matrix_case_id": case_id,
                                "matrix_case_label": case_label,
                            },
                            None,
                            None,
                            status="skipped",
                            error="ppo model not found in runs_dir",
                            fallback_policy=policy,
                            fallback_scenario=case_scenario,
                            fallback_seed=seed,
                            episodes=episodes,
                            max_steps=steps,
                            deterministic=bool(deterministic),
                        )
                        writer.writerow(row)
                        print(
                            f"[matrix] case={case_id or '<default>'} seed={seed} scenario={case_scenario} policy={policy} -> SKIP (no model)"
                        )
                        continue

                    before = _list_run_dirs(args.out_dir)
                    cmd = [
                        sys.executable,
                        "run_experiment.py",
                        "--policy",
                        policy,
                        "--scenario",
                        case_scenario,
                        "--seed",
                        str(seed),
                        "--episodes",
                        str(episodes),
                        "--steps",
                        str(steps),
                        "--out-dir",
                        str(args.out_dir),
                        "--runs-dir",
                        str(args.runs_dir),
                    ]
                    if matrix_manifest is not None:
                        cmd.extend(
                            [
                                "--matrix-id",
                                matrix_manifest.matrix_id,
                                "--matrix-name",
                                matrix_manifest.matrix_name,
                                "--matrix-manifest",
                                str(matrix_manifest.manifest_path),
                                "--matrix-case-id",
                                case_id,
                                "--matrix-case-label",
                                case_label,
                            ]
                        )
                    if case_topology_seed is not None:
                        cmd.extend(["--topology-seed", str(case_topology_seed)])
                    if explicit_topology_name is not None:
                        cmd.extend(["--topology-name", explicit_topology_name])
                    if explicit_topology_path is not None:
                        cmd.extend(["--topology-path", str(explicit_topology_path)])
                    if args.traffic_seed is not None:
                        cmd.extend(["--traffic-seed", str(args.traffic_seed)])
                    if case_traffic_model is not None:
                        cmd.extend(["--traffic-model", str(case_traffic_model)])
                    if explicit_traffic_name is not None:
                        cmd.extend(["--traffic-name", explicit_traffic_name])
                    if explicit_traffic_path is not None:
                        cmd.extend(["--traffic-path", str(explicit_traffic_path)])
                    if case_traffic_scenario is not None:
                        cmd.extend(["--traffic-scenario", str(case_traffic_scenario)])
                    if case_traffic_scenario_version is not None:
                        cmd.extend(["--traffic-scenario-version", str(case_traffic_scenario_version)])
                    if case_traffic_scenario_intensity is not None:
                        cmd.extend(["--traffic-scenario-intensity", str(case_traffic_scenario_intensity)])
                    if case_traffic_scenario_duration is not None:
                        cmd.extend(["--traffic-scenario-duration", str(case_traffic_scenario_duration)])
                    if case_traffic_scenario_frequency is not None:
                        cmd.extend(["--traffic-scenario-frequency", str(case_traffic_scenario_frequency)])
                    if args.stability_reversal_window_steps is not None:
                        cmd.extend(["--stability-reversal-window-steps", str(args.stability_reversal_window_steps)])
                    if args.stability_reversal_penalty is not None:
                        cmd.extend(["--stability-reversal-penalty", str(args.stability_reversal_penalty)])
                    if args.stability_min_steps_for_assessment is not None:
                        cmd.extend(
                            ["--stability-min-steps-for-assessment", str(args.stability_min_steps_for_assessment)]
                        )
                    if args.stability_max_transition_rate is not None:
                        cmd.extend(["--stability-max-transition-rate", str(args.stability_max_transition_rate)])
                    if args.stability_max_flap_rate is not None:
                        cmd.extend(["--stability-max-flap-rate", str(args.stability_max_flap_rate)])
                    if args.stability_max_flap_count is not None:
                        cmd.extend(["--stability-max-flap-count", str(args.stability_max_flap_count)])
                    if args.power_utilization_sensitive is not None:
                        cmd.extend(["--power-utilization-sensitive", str(args.power_utilization_sensitive).lower()])
                    if args.power_transition_on_joules is not None:
                        cmd.extend(["--power-transition-on-joules", str(args.power_transition_on_joules)])
                    if args.power_transition_off_joules is not None:
                        cmd.extend(["--power-transition-off-joules", str(args.power_transition_off_joules)])
                    if tag:
                        cmd.extend(["--tag", str(tag)])
                    if policy == "ppo" and case_model_path is not None:
                        cmd.extend(["--model", str(case_model_path)])
                    if routing_baseline:
                        cmd.extend(["--routing-baseline", str(routing_baseline)])
                    if routing_link_cost_model:
                        cmd.extend(["--routing-link-cost-model", str(routing_link_cost_model)])
                    if not deterministic:
                        cmd.append("--stochastic")

                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    after = _list_run_dirs(args.out_dir)

                    run_dir = _extract_run_dir(result.stdout, result.stderr) or _detect_new_run_dir(before, after)

                    status = "ok" if result.returncode == 0 else "failed"
                    error = ""
                    if result.returncode != 0:
                        error = f"exit {result.returncode}"
                        sn = _stderr_snip(result.stderr)
                        if sn:
                            error = f"{error}: {sn}"

                    run_meta = None
                    summary = None
                    if run_dir is not None and run_dir.exists():
                        run_meta = _load_json(run_dir / "run_meta.json")
                        summary = _load_json(run_dir / "summary.json")
                    elif status == "ok":
                        status = "failed"
                        error = "run_dir not found"

                    row = _row_from_meta(
                        run_meta,
                        summary,
                        run_dir,
                        status=status,
                        error=error,
                        fallback_policy=policy,
                        fallback_scenario=case_scenario,
                        fallback_seed=seed,
                        episodes=episodes,
                        max_steps=steps,
                        deterministic=bool(deterministic),
                    )
                    writer.writerow(row)

                    if status == "ok":
                        print(
                            f"[matrix] case={case_id or '<default>'} seed={seed} scenario={case_scenario} policy={policy} -> OK ({run_dir})"
                        )
                    else:
                        print(
                            f"[matrix] case={case_id or '<default>'} seed={seed} scenario={case_scenario} policy={policy} -> FAIL ({error})"
                        )

    if tag and summary_path.name != "results_summary.csv":
        generic_path = args.out_dir / "results_summary.csv"
        try:
            shutil.copyfile(summary_path, generic_path)
        except Exception:
            pass
    print(f"[matrix] results_summary.csv saved to {summary_path}")


if __name__ == "__main__":
    main()
