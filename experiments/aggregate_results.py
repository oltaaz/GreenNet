#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


SUMMARY_COLUMNS = [
    "run_id",
    "policy",
    "scenario",
    "seed",
    "tag",
    "created_at_utc",
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

TAG_RE = re.compile(r"__tag-(.+)$")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _infer_tag_from_dir_name(run_dir: Path) -> Optional[str]:
    match = TAG_RE.search(run_dir.name)
    if not match:
        return None
    tag = match.group(1).strip()
    return tag or None


def _pick_created_at(meta: Dict[str, Any], run_dir: Path) -> str:
    created = meta.get("created_at_utc") or meta.get("timestamp_utc")
    if created:
        return str(created)
    return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()


def _sort_epoch(created_at_utc: str, run_dir: Path) -> float:
    try:
        # Handle trailing Z as UTC for fromisoformat compatibility.
        normalized = str(created_at_utc).replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return float(run_dir.stat().st_mtime)


def _row_from_meta(
    run_meta: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    results_dir: Path,
) -> Dict[str, Any]:
    meta = run_meta or {}
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
    tag = meta.get("tag")
    if tag in ("", None):
        tag = _infer_tag_from_dir_name(results_dir)
    created_at_utc = _pick_created_at(meta, results_dir)

    return {
        "run_id": meta.get("run_id", ""),
        "policy": meta.get("policy", ""),
        "scenario": meta.get("scenario", ""),
        "seed": meta.get("seed", ""),
        "tag": tag,
        "created_at_utc": created_at_utc,
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

    selected_rows: Dict[tuple[str, str, str, str], tuple[float, Dict[str, Any]]] = {}
    for run_dir in sorted(run_dirs, key=lambda p: p.name):
        run_meta = _load_json(run_dir / "run_meta.json")
        summary = _load_json(run_dir / "summary.json")
        row = _row_from_meta(run_meta, summary, run_dir)

        key = (
            str(row.get("tag") if row.get("tag") is not None else ""),
            str(row.get("scenario", "")),
            str(row.get("policy", "")),
            str(row.get("seed", "")),
        )
        row_created = str(row.get("created_at_utc", ""))
        row_epoch = _sort_epoch(row_created, run_dir)

        existing = selected_rows.get(key)
        if existing is None or row_epoch >= existing[0]:
            selected_rows[key] = (row_epoch, row)

    rows = [item[1] for item in selected_rows.values()]
    rows.sort(key=lambda r: (str(r.get("tag", "")), str(r.get("scenario", "")), str(r.get("policy", "")), str(r.get("seed", ""))))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[aggregate] results_summary.csv saved to {output_path}")


if __name__ == "__main__":
    main()
