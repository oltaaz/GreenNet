#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from greennet.persistence import get_run_repository

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
    "tag",
    "created_at_utc",
    "episodes",
    "max_steps",
    "deterministic",
    "topology_seed",
    "topology_name",
    "topology_path",
    "traffic_seed",
    "traffic_seed_base",
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

TAG_RE = re.compile(r"__tag-(.+)$")
REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "matrix_id": meta.get("matrix_id", ""),
        "matrix_name": meta.get("matrix_name", ""),
        "matrix_manifest": meta.get("matrix_manifest", ""),
        "matrix_case_id": meta.get("matrix_case_id", ""),
        "matrix_case_label": meta.get("matrix_case_label", ""),
        "run_id": meta.get("run_id", ""),
        "policy": meta.get("policy", ""),
        "policy_class": meta.get("policy_class", ""),
        "controller_policy": meta.get("controller_policy", ""),
        "controller_policy_class": meta.get("controller_policy_class", ""),
        "scenario": meta.get("scenario", ""),
        "seed": meta.get("seed", ""),
        "tag": tag,
        "created_at_utc": created_at_utc,
        "episodes": meta.get("episodes", ""),
        "max_steps": meta.get("max_steps", ""),
        "deterministic": meta.get("deterministic", ""),
        "topology_seed": meta.get("topology_seed", ""),
        "topology_name": meta.get("topology_name", ""),
        "topology_path": meta.get("topology_path", ""),
        "traffic_seed": meta.get("traffic_seed", ""),
        "traffic_seed_base": meta.get("traffic_seed_base", ""),
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

    selected_rows: Dict[tuple[str, ...], tuple[float, Dict[str, Any]]] = {}
    db_rows: list[Dict[str, Any]] = []
    use_db = False
    resolved_out_dir = out_dir.resolve()
    if os.getenv("GREENNET_DB_PATH"):
        use_db = True
    elif resolved_out_dir in {(REPO_ROOT / "results").resolve(), (REPO_ROOT / "runs").resolve()}:
        use_db = True
    if use_db:
        try:
            repo = get_run_repository()
            if resolved_out_dir == (REPO_ROOT / "results").resolve():
                base = "results"
            elif resolved_out_dir == (REPO_ROOT / "runs").resolve():
                base = "runs"
            else:
                base = "both"
            db_rows = repo.list_summary_rows(base=base, tag=args.tag)
        except Exception:
            db_rows = []

    if db_rows:
        input_rows = db_rows
    else:
        run_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
        if args.tag:
            token = f"__tag-{args.tag}"
            run_dirs = [path for path in run_dirs if token in path.name]
        input_rows = []
        for run_dir in sorted(run_dirs, key=lambda p: p.name):
            run_meta = _load_json(run_dir / "run_meta.json")
            summary = _load_json(run_dir / "summary.json")
            input_rows.append(_row_from_meta(run_meta, summary, run_dir))

    for row in input_rows:
        
        key = (
            str(row.get("tag") if row.get("tag") is not None else ""),
            str(row.get("matrix_id", "")),
            str(row.get("matrix_case_id", "")),
            str(row.get("scenario", "")),
            str(row.get("policy", "")),
            str(row.get("seed", "")),
            str(row.get("topology_seed", "")),
            str(row.get("topology_name", "")),
            str(row.get("topology_path", "")),
            str(row.get("traffic_model", "")),
            str(row.get("traffic_mode", "")),
            str(row.get("traffic_name", "")),
            str(row.get("traffic_path", "")),
            str(row.get("traffic_scenario", "")),
            str(row.get("traffic_scenario_version", "")),
            str(row.get("traffic_scenario_intensity", "")),
            str(row.get("traffic_scenario_duration", "")),
            str(row.get("traffic_scenario_frequency", "")),
            str(row.get("qos_policy_signature", "")),
            str(row.get("stability_policy_signature", "")),
            str(row.get("energy_model_signature", "")),
        )
        row_created = str(row.get("created_at_utc", ""))
        run_dir_path = out_dir / Path(str(row.get("results_dir", ""))).name if str(row.get("results_dir", "")).strip() else out_dir
        row_epoch = _sort_epoch(row_created, run_dir_path)

        existing = selected_rows.get(key)
        if existing is None or row_epoch >= existing[0]:
            selected_rows[key] = (row_epoch, row)

    rows = [item[1] for item in selected_rows.values()]
    rows.sort(
        key=lambda r: (
            str(r.get("tag", "")),
            str(r.get("matrix_id", "")),
            str(r.get("matrix_case_id", "")),
            str(r.get("scenario", "")),
            str(r.get("policy", "")),
            str(r.get("topology_name", "")),
            str(r.get("topology_path", "")),
            str(r.get("topology_seed", "")),
            str(r.get("traffic_model", "")),
            str(r.get("traffic_mode", "")),
            str(r.get("traffic_name", "")),
            str(r.get("traffic_path", "")),
            str(r.get("traffic_scenario", "")),
            str(r.get("traffic_scenario_version", "")),
            str(r.get("qos_policy_signature", "")),
            str(r.get("energy_model_signature", "")),
            str(r.get("seed", "")),
        )
    )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[aggregate] results_summary.csv saved to {output_path}")


if __name__ == "__main__":
    main()
