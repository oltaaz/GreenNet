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


RUN_DIR_RE = re.compile(r"\[run_experiment\] results saved to (.+)")

SUMMARY_COLUMNS = [
    "run_id",
    "policy",
    "scenario",
    "seed",
    "episodes",
    "max_steps",
    "deterministic",
    "reward_total_mean",
    "delivered_total_mean",
    "dropped_total_mean",
    "energy_kwh_total_mean",
    "carbon_g_total_mean",
    "avg_utilization_mean",
    "active_ratio_mean",
    "avg_delay_ms_mean",
    "results_dir",
    "status",
    "error",
]


def _parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_seed_list(value: str) -> List[int]:
    return [int(part) for part in _parse_csv_list(value)]


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
        "run_id": meta.get("run_id", ""),
        "policy": meta.get("policy", fallback_policy),
        "scenario": meta.get("scenario", fallback_scenario),
        "seed": meta.get("seed", fallback_seed),
        "episodes": meta.get("episodes", episodes),
        "max_steps": meta.get("max_steps", max_steps),
        "deterministic": meta.get("deterministic", deterministic),
        "reward_total_mean": overall.get("reward_total_mean", ""),
        "delivered_total_mean": overall.get("delivered_total_mean", ""),
        "dropped_total_mean": overall.get("dropped_total_mean", ""),
        "energy_kwh_total_mean": overall.get("energy_kwh_total_mean", ""),
        "carbon_g_total_mean": overall.get("carbon_g_total_mean", ""),
        "avg_utilization_mean": overall.get("avg_utilization_mean", ""),
        "active_ratio_mean": overall.get("active_ratio_mean", ""),
        "avg_delay_ms_mean": overall.get("avg_delay_ms_mean", ""),
        "results_dir": str(results_dir) if results_dir is not None else "",
        "status": status,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matrix of GreenNet experiments.")
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
    args = parser.parse_args()

    seeds = _parse_seed_list(args.seeds)
    scenarios = _parse_csv_list(args.scenarios)
    policies = _parse_csv_list(args.policies)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_name = "results_summary.csv"
    if args.tag:
        summary_name = f"results_summary_{args.tag}.csv"
    summary_path = args.out_dir / summary_name

    # Determine a single fixed topology seed for fairness + PPO action-space consistency.
    fixed_topology_seed: Optional[int] = args.topology_seed
    ppo_model_path = args.ppo_model
    if ppo_model_path is not None and not ppo_model_path.exists():
        raise SystemExit(f"--ppo-model does not exist: {ppo_model_path}")
    if fixed_topology_seed is None and "ppo" in policies:
        mp = ppo_model_path or _find_latest_model(args.runs_dir)
        if mp is None:
            # PPO requested but no model found -> we will mark PPO runs as skipped.
            fixed_topology_seed = None
        else:
            inferred = _infer_topology_seed_from_model(mp)
            fixed_topology_seed = inferred if inferred is not None else 0
    if fixed_topology_seed is None:
        # Default fallback if PPO not requested or inference not possible
        fixed_topology_seed = 0

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()

        model_path = ppo_model_path or _find_latest_model(args.runs_dir)

        for seed in seeds:
            for scenario in scenarios:
                for policy in policies:
                    if policy == "ppo" and model_path is None:
                        row = _row_from_meta(
                            None,
                            None,
                            None,
                            status="skipped",
                            error="ppo model not found in runs_dir",
                            fallback_policy=policy,
                            fallback_scenario=scenario,
                            fallback_seed=seed,
                            episodes=args.episodes,
                            max_steps=args.steps,
                            deterministic=bool(args.deterministic),
                        )
                        writer.writerow(row)
                        print(f"[matrix] seed={seed} scenario={scenario} policy={policy} -> SKIP (no model)")
                        continue

                    before = _list_run_dirs(args.out_dir)
                    cmd = [
                        sys.executable,
                        "run_experiment.py",
                        "--policy",
                        policy,
                        "--scenario",
                        scenario,
                        "--seed",
                        str(seed),
                        "--topology-seed",
                        str(fixed_topology_seed),
                        "--episodes",
                        str(args.episodes),
                        "--steps",
                        str(args.steps),
                        "--out-dir",
                        str(args.out_dir),
                        "--runs-dir",
                        str(args.runs_dir),
                    ]
                    if args.tag:
                        cmd.extend(["--tag", str(args.tag)])
                    if policy == "ppo" and model_path is not None:
                        cmd.extend(["--model", str(model_path)])
                    if not args.deterministic:
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
                        fallback_scenario=scenario,
                        fallback_seed=seed,
                        episodes=args.episodes,
                        max_steps=args.steps,
                        deterministic=bool(args.deterministic),
                    )
                    writer.writerow(row)

                    if status == "ok":
                        print(f"[matrix] seed={seed} scenario={scenario} policy={policy} -> OK ({run_dir})")
                    else:
                        print(f"[matrix] seed={seed} scenario={scenario} policy={policy} -> FAIL ({error})")

    if args.tag and summary_path.name != "results_summary.csv":
        generic_path = args.out_dir / "results_summary.csv"
        try:
            shutil.copyfile(summary_path, generic_path)
        except Exception:
            pass
    print(f"[matrix] results_summary.csv saved to {summary_path}")


if __name__ == "__main__":
    main()
