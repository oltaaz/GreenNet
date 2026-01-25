#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


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


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _row_from_meta(
    run_meta: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    results_dir: Path,
) -> Dict[str, Any]:
    meta = run_meta or {}
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}

    return {
        "run_id": meta.get("run_id", ""),
        "policy": meta.get("policy", ""),
        "scenario": meta.get("scenario", ""),
        "seed": meta.get("seed", ""),
        "episodes": meta.get("episodes", ""),
        "max_steps": meta.get("max_steps", ""),
        "deterministic": meta.get("deterministic", ""),
        "reward_total_mean": overall.get("reward_total_mean", ""),
        "delivered_total_mean": overall.get("delivered_total_mean", ""),
        "dropped_total_mean": overall.get("dropped_total_mean", ""),
        "energy_kwh_total_mean": overall.get("energy_kwh_total_mean", ""),
        "carbon_g_total_mean": overall.get("carbon_g_total_mean", ""),
        "avg_utilization_mean": overall.get("avg_utilization_mean", ""),
        "active_ratio_mean": overall.get("active_ratio_mean", ""),
        "avg_delay_ms_mean": overall.get("avg_delay_ms_mean", ""),
        "results_dir": str(results_dir),
        "status": "ok" if run_meta and summary else "partial",
        "error": "" if run_meta and summary else "missing run_meta.json or summary.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate existing run results into results_summary.csv.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.exists():
        raise SystemExit(f"out-dir does not exist: {out_dir}")

    output_path = args.output or (out_dir / "results_summary.csv")

    run_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
    if args.tag:
        token = f"__tag-{args.tag}"
        run_dirs = [path for path in run_dirs if token in path.name]
    run_dirs.sort(key=lambda p: p.name)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()

        for run_dir in run_dirs:
            run_meta = _load_json(run_dir / "run_meta.json")
            summary = _load_json(run_dir / "summary.json")
            row = _row_from_meta(run_meta, summary, run_dir)
            writer.writerow(row)

    print(f"[aggregate] results_summary.csv saved to {output_path}")


if __name__ == "__main__":
    main()
