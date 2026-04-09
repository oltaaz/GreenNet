from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence


TAG_RE = re.compile(r"__tag-(.+)$")
DEFAULT_BASELINE_POLICIES = ("heuristic", "utilization_threshold", "baseline", "all_on", "noop")
DEFAULT_AI_POLICIES = ("ppo",)


@dataclass(frozen=True)
class MetricSpec:
    field: str
    label: str
    higher_is_better: bool
    part_of_qos_gate: bool = False


@dataclass(frozen=True)
class HypothesisThresholds:
    energy_target_pct: float = 15.0
    max_qos_violation_rate_increase_abs: float = 0.02
    max_delivered_loss_pct: float = 2.0
    max_dropped_increase_pct: float = 5.0
    max_delay_increase_pct: float = 10.0
    max_path_latency_increase_pct: float = 10.0


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("energy_kwh", "Energy (kWh)", higher_is_better=False),
    MetricSpec("delivered_traffic", "Delivered traffic", higher_is_better=True, part_of_qos_gate=True),
    MetricSpec("dropped_traffic", "Dropped traffic", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("avg_delay_ms", "Average delay (ms)", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("avg_path_latency_ms", "Average path latency (ms)", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("qos_violation_rate", "QoS violation rate", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("qos_violation_count", "QoS violation count", higher_is_better=False),
    MetricSpec("carbon_g", "Carbon emissions (g)", higher_is_better=False),
)
METRIC_BY_FIELD = {metric.field: metric for metric in METRIC_SPECS}


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    num = _to_float(value)
    if num is None:
        return None
    return int(num)


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _mean(values: Sequence[float]) -> float | None:
    return float(fmean(values)) if values else None


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _clean_csv_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _infer_tag_from_dir_name(run_dir: Path) -> str | None:
    match = TAG_RE.search(run_dir.name)
    if not match:
        return None
    tag = match.group(1).strip()
    return tag or None


def _pick_created_at(meta: Mapping[str, Any], run_dir: Path) -> str:
    created = meta.get("created_at_utc") or meta.get("timestamp_utc")
    if created:
        return str(created)
    return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()


def _sort_epoch(created_at_utc: str, run_dir: Path) -> float:
    try:
        normalized = str(created_at_utc).replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return float(run_dir.stat().st_mtime)


def _resolve_run_dir(path_text: str, *, repo_root: Path, context_dir: Path | None = None) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw

    candidates: List[Path] = []
    if context_dir is not None:
        candidates.append((context_dir / raw).resolve())
    candidates.append((repo_root / raw).resolve())
    candidates.append(raw.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _matches_filter(value: str | None, allowed: set[str]) -> bool:
    if not allowed:
        return True
    if value is None:
        return False
    return str(value) in allowed


def _common_text(rows: Sequence[Mapping[str, Any]], field: str) -> str | None:
    values = {
        str(row.get(field)).strip()
        for row in rows
        if row.get(field) not in ("", None)
    }
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _policy_class(policy: str, *, baseline_policies: set[str], ai_policies: set[str]) -> str:
    if policy in ai_policies:
        return "ai"
    if policy in baseline_policies:
        return "baseline"
    return "other"


def _choose_primary_baseline(
    available_policies: Iterable[str],
    *,
    requested: str | None,
    baseline_priority: Sequence[str],
) -> str:
    available = {str(policy) for policy in available_policies}
    if requested:
        if requested not in available:
            raise SystemExit(
                f"Requested baseline policy '{requested}' is not present in the selected runs. "
                f"Available policies: {sorted(available)}"
            )
        return requested

    for policy in baseline_priority:
        if policy in available:
            return policy

    raise SystemExit(
        "Unable to infer a baseline policy from the selected runs. "
        f"Available policies: {sorted(available)}. "
        "Use --primary-baseline-policy to choose one explicitly."
    )


def _build_run_row_from_summary_csv_row(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    context_dir: Path,
    baseline_policies: set[str],
    ai_policies: set[str],
) -> Dict[str, Any]:
    policy = str(row.get("policy") or "")
    episodes = _to_int(row.get("episodes"))
    max_steps = _to_int(row.get("max_steps"))
    steps_total = None
    if episodes is not None and max_steps is not None:
        steps_total = int(episodes * max_steps)

    results_dir_text = str(row.get("results_dir") or "").strip()
    resolved_results_dir = (
        _resolve_run_dir(results_dir_text, repo_root=repo_root, context_dir=context_dir)
        if results_dir_text
        else None
    )

    return {
        "run_id": str(row.get("run_id") or results_dir_text or ""),
        "policy": policy,
        "policy_class": _policy_class(policy, baseline_policies=baseline_policies, ai_policies=ai_policies),
        "controller_policy": str(row.get("controller_policy") or policy),
        "controller_policy_class": str(
            row.get("controller_policy_class")
            or ("ai_enhanced" if policy in ai_policies else "traditional_baseline" if policy in baseline_policies else "other")
        ),
        "scenario": str(row.get("scenario") or ""),
        "seed": _to_int(row.get("seed")),
        "tag": str(row.get("tag") or ""),
        "deterministic": _to_bool(row.get("deterministic")),
        "episodes": episodes,
        "max_steps": max_steps,
        "results_dir": str(resolved_results_dir or results_dir_text or ""),
        "created_at_utc": str(row.get("created_at_utc") or ""),
        "routing_baseline": str(row.get("routing_baseline") or "min_hop_single_path"),
        "routing_link_cost_model": str(row.get("routing_link_cost_model") or "unit"),
        "routing_forwarding_model": str(row.get("routing_forwarding_model") or "single_shortest_path"),
        "routing_path_split": str(row.get("routing_path_split") or "single_path"),
        "steps_total": steps_total,
        "energy_kwh": _to_float(row.get("energy_kwh_total_mean") or row.get("energy_kwh_mean")),
        "delivered_traffic": _to_float(row.get("delivered_total_mean") or row.get("delivered_traffic_mean")),
        "dropped_traffic": _to_float(row.get("dropped_total_mean") or row.get("dropped_traffic_mean")),
        "avg_delay_ms": _to_float(row.get("avg_delay_ms_mean")),
        "avg_path_latency_ms": _to_float(row.get("avg_path_latency_ms_mean")),
        "qos_violation_rate": _to_float(row.get("qos_violation_rate_mean") or row.get("qos_violation_rate")),
        "qos_violation_count": _to_float(row.get("qos_violation_count_mean") or row.get("qos_violation_count")),
        "carbon_g": _to_float(row.get("carbon_g_total_mean") or row.get("carbon_g_mean")),
    }


def _load_run_rows_from_summary_csv(
    summary_csv: Path,
    *,
    repo_root: Path,
    tag_filter: str | None,
    scenario_filter: set[str],
    policy_filter: set[str],
    deterministic_filter: bool | None,
    baseline_policies: set[str],
    ai_policies: set[str],
) -> List[Dict[str, Any]]:
    run_rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status and status != "ok":
                continue
            if tag_filter and "tag" in row and str(row.get("tag") or "") != tag_filter:
                continue
            if not _matches_filter(str(row.get("scenario") or None), scenario_filter):
                continue
            if not _matches_filter(str(row.get("policy") or None), policy_filter):
                continue
            row_det = _to_bool(row.get("deterministic"))
            if deterministic_filter is not None and row_det is not None and row_det != deterministic_filter:
                continue

            results_dir = str(row.get("results_dir") or "").strip()
            resolved = (
                _resolve_run_dir(results_dir, repo_root=repo_root, context_dir=summary_csv.parent)
                if results_dir
                else None
            )
            dedupe_key = str(resolved or row.get("run_id") or "")
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            if resolved is not None and resolved.exists():
                run_row = _extract_run_metrics(resolved, baseline_policies=baseline_policies, ai_policies=ai_policies)
            else:
                run_row = _build_run_row_from_summary_csv_row(
                    row,
                    repo_root=repo_root,
                    context_dir=summary_csv.parent,
                    baseline_policies=baseline_policies,
                    ai_policies=ai_policies,
                )
            run_rows.append(run_row)
    return run_rows


def _scan_authoritative_run_dirs(
    results_dir: Path,
    *,
    tag_filter: str | None,
    scenario_filter: set[str],
    policy_filter: set[str],
    deterministic_filter: bool | None,
) -> List[Path]:
    selected_rows: Dict[tuple[str, str, str, str, str], tuple[float, Path]] = {}

    for run_dir in sorted(results_dir.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir():
            continue
        meta = _load_json(run_dir / "run_meta.json")
        if not meta:
            continue
        if not (run_dir / "per_step.csv").exists() and not (run_dir / "summary.json").exists():
            continue

        tag = meta.get("tag")
        if tag in ("", None):
            tag = _infer_tag_from_dir_name(run_dir)
        policy = str(meta.get("policy") or "")
        scenario = str(meta.get("scenario") or "")
        seed = str(meta.get("seed") or "")
        deterministic = _to_bool(meta.get("deterministic"))

        if tag_filter and str(tag or "") != tag_filter:
            continue
        if not _matches_filter(scenario or None, scenario_filter):
            continue
        if not _matches_filter(policy or None, policy_filter):
            continue
        if deterministic_filter is not None and deterministic is not None and deterministic != deterministic_filter:
            continue

        created_at_utc = _pick_created_at(meta, run_dir)
        epoch = _sort_epoch(created_at_utc, run_dir)
        key = (
            str(tag or ""),
            scenario,
            policy,
            seed,
            "" if deterministic is None else str(deterministic),
        )
        existing = selected_rows.get(key)
        if existing is None or epoch >= existing[0]:
            selected_rows[key] = (epoch, run_dir)

    rows = [item[1] for item in selected_rows.values()]
    rows.sort(key=lambda path: path.name)
    return rows


def _parse_per_step_rollup(per_step_path: Path) -> Dict[str, Any]:
    if not per_step_path.exists():
        return {}

    episode_totals: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "delivered_traffic": 0.0,
            "dropped_traffic": 0.0,
            "energy_kwh": 0.0,
            "carbon_g": 0.0,
            "avg_delay_sum": 0.0,
            "avg_delay_count": 0,
            "avg_path_latency_sum": 0.0,
            "avg_path_latency_count": 0,
            "qos_violation_count": 0,
            "qos_present": False,
            "steps": 0,
        }
    )

    with per_step_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            episode = _to_int(row.get("episode"))
            episode_id = episode if episode is not None else 0
            stats = episode_totals[episode_id]
            stats["steps"] += 1

            for field in ("delivered_traffic", "dropped_traffic", "energy_kwh", "carbon_g"):
                delta_key = {
                    "delivered_traffic": "delta_delivered",
                    "dropped_traffic": "delta_dropped",
                    "energy_kwh": "delta_energy_kwh",
                    "carbon_g": "delta_carbon_g",
                }[field]
                value = _to_float(row.get(delta_key))
                if value is not None:
                    stats[field] += value

            delay = _to_float(row.get("avg_delay_ms"))
            if delay is not None:
                stats["avg_delay_sum"] += delay
                stats["avg_delay_count"] += 1

            path_latency = _to_float(row.get("avg_path_latency_ms"))
            if path_latency is not None:
                stats["avg_path_latency_sum"] += path_latency
                stats["avg_path_latency_count"] += 1

            qos_violation = _to_bool(row.get("qos_violation"))
            if qos_violation is not None:
                stats["qos_present"] = True
                stats["qos_violation_count"] += int(qos_violation)

    if not episode_totals:
        return {}

    per_episode = list(episode_totals.values())
    delay_means = [
        float(stats["avg_delay_sum"]) / float(stats["avg_delay_count"])
        for stats in per_episode
        if int(stats["avg_delay_count"]) > 0
    ]
    path_latency_means = [
        float(stats["avg_path_latency_sum"]) / float(stats["avg_path_latency_count"])
        for stats in per_episode
        if int(stats["avg_path_latency_count"]) > 0
    ]

    total_qos_count = 0
    total_qos_steps = 0
    for stats in per_episode:
        if bool(stats["qos_present"]):
            total_qos_count += int(stats["qos_violation_count"])
            total_qos_steps += int(stats["steps"])

    return {
        "delivered_traffic": _mean([float(stats["delivered_traffic"]) for stats in per_episode]),
        "dropped_traffic": _mean([float(stats["dropped_traffic"]) for stats in per_episode]),
        "energy_kwh": _mean([float(stats["energy_kwh"]) for stats in per_episode]),
        "carbon_g": _mean([float(stats["carbon_g"]) for stats in per_episode]),
        "avg_delay_ms": _mean(delay_means),
        "avg_path_latency_ms": _mean(path_latency_means),
        # We use the weighted rate across all selected steps to avoid over-weighting shorter episodes.
        "qos_violation_rate": (float(total_qos_count) / float(total_qos_steps)) if total_qos_steps > 0 else None,
        "qos_violation_count": float(total_qos_count) if total_qos_steps > 0 else None,
        "steps_total": int(sum(int(stats["steps"]) for stats in per_episode)),
        "episode_count": int(len(per_episode)),
    }


def _extract_run_metrics(
    run_dir: Path,
    *,
    baseline_policies: set[str],
    ai_policies: set[str],
) -> Dict[str, Any]:
    meta = _load_json(run_dir / "run_meta.json") or {}
    summary = _load_json(run_dir / "summary.json") or {}
    overall = summary.get("overall", {}) if isinstance(summary.get("overall"), dict) else {}
    per_step = _parse_per_step_rollup(run_dir / "per_step.csv")

    tag = meta.get("tag")
    if tag in ("", None):
        tag = _infer_tag_from_dir_name(run_dir)

    policy = str(meta.get("policy") or "")
    scenario = str(meta.get("scenario") or "")
    episodes = _to_int(meta.get("episodes"))
    if episodes is None:
        episodes = _to_int(per_step.get("episode_count"))

    row = {
        "run_id": str(meta.get("run_id") or run_dir.name),
        "policy": policy,
        "policy_class": _policy_class(policy, baseline_policies=baseline_policies, ai_policies=ai_policies),
        "controller_policy": str(meta.get("controller_policy") or policy),
        "controller_policy_class": str(
            meta.get("controller_policy_class")
            or ("ai_enhanced" if policy in ai_policies else "traditional_baseline" if policy in baseline_policies else "other")
        ),
        "scenario": scenario,
        "seed": _to_int(meta.get("seed")),
        "tag": tag,
        "deterministic": _to_bool(meta.get("deterministic")),
        "episodes": episodes,
        "max_steps": _to_int(meta.get("max_steps")),
        "results_dir": str(run_dir),
        "created_at_utc": _pick_created_at(meta, run_dir),
        "routing_baseline": str(meta.get("routing_baseline") or "min_hop_single_path"),
        "routing_link_cost_model": str(meta.get("routing_link_cost_model") or "unit"),
        "routing_forwarding_model": str(meta.get("routing_forwarding_model") or "single_shortest_path"),
        "routing_path_split": str(meta.get("routing_path_split") or "single_path"),
        "steps_total": _to_int(per_step.get("steps_total")),
        "energy_kwh": _to_float(overall.get("energy_kwh_total_mean")),
        "delivered_traffic": _to_float(overall.get("delivered_total_mean")),
        "dropped_traffic": _to_float(overall.get("dropped_total_mean")),
        "avg_delay_ms": _to_float(overall.get("avg_delay_ms_mean")),
        "avg_path_latency_ms": _to_float(overall.get("avg_path_latency_ms_mean")),
        "qos_violation_rate": _to_float(per_step.get("qos_violation_rate")),
        "qos_violation_count": _to_float(per_step.get("qos_violation_count")),
        "carbon_g": _to_float(overall.get("carbon_g_total_mean")),
    }

    for field in ("energy_kwh", "delivered_traffic", "dropped_traffic", "avg_delay_ms", "avg_path_latency_ms", "carbon_g"):
        if row[field] is None:
            row[field] = _to_float(per_step.get(field))

    return row


def _aggregate_group(
    scope_type: str,
    scope_value: str,
    policy: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") is not None})
    out: Dict[str, Any] = {
        "scope_type": scope_type,
        "scope": scope_value,
        "scenario": scope_value if scope_type == "scenario" else "ALL",
        "policy": policy,
        "policy_class": str(rows[0].get("policy_class") or "other"),
        "controller_policy": _common_text(rows, "controller_policy") or policy,
        "controller_policy_class": _common_text(rows, "controller_policy_class") or "other",
        "routing_baseline": _common_text(rows, "routing_baseline") or "min_hop_single_path",
        "routing_link_cost_model": _common_text(rows, "routing_link_cost_model") or "unit",
        "routing_forwarding_model": _common_text(rows, "routing_forwarding_model") or "single_shortest_path",
        "routing_path_split": _common_text(rows, "routing_path_split") or "single_path",
        "run_count": int(len(rows)),
        "seed_count": int(len(seeds)),
        "episodes_total": int(sum(int(row.get("episodes") or 0) for row in rows)),
        "steps_total": int(sum(int(row.get("steps_total") or 0) for row in rows)),
        "seed_list": ",".join(str(seed) for seed in seeds),
    }

    for metric in METRIC_SPECS:
        values = [float(row[metric.field]) for row in rows if row.get(metric.field) is not None]
        out[f"{metric.field}_mean"] = _mean(values)
        out[f"{metric.field}_std"] = _std(values)
        out[f"{metric.field}_count"] = int(len(values))

    qos_counts = [float(row["qos_violation_count"]) for row in rows if row.get("qos_violation_count") is not None]
    out["qos_violation_count_total"] = float(sum(qos_counts)) if qos_counts else None
    return out


def _pct_change(baseline_value: float | None, candidate_value: float | None) -> float | None:
    if baseline_value is None or candidate_value is None:
        return None
    if abs(float(baseline_value)) <= 1e-12:
        return None
    return float((float(candidate_value) - float(baseline_value)) / float(baseline_value) * 100.0)


def _pct_reduction(baseline_value: float | None, candidate_value: float | None) -> float | None:
    if baseline_value is None or candidate_value is None:
        return None
    if abs(float(baseline_value)) <= 1e-12:
        return None
    return float((float(baseline_value) - float(candidate_value)) / float(baseline_value) * 100.0)


def _attach_baseline_comparison(
    row: Dict[str, Any],
    baseline_row: Mapping[str, Any] | None,
    *,
    primary_baseline_policy: str,
    thresholds: HypothesisThresholds,
) -> None:
    row["comparison_baseline_policy"] = primary_baseline_policy
    row["comparison_available"] = bool(baseline_row is not None)

    if baseline_row is None:
        row["qos_acceptability_status"] = "insufficient_data"
        row["hypothesis_status"] = "insufficient_data"
        row["qos_acceptability_missing"] = "missing baseline row"
        row["is_primary_baseline"] = False
        return

    row["is_primary_baseline"] = row["policy"] == primary_baseline_policy

    delivered = row.get("delivered_traffic_mean")
    baseline_delivered = baseline_row.get("delivered_traffic_mean")
    dropped = row.get("dropped_traffic_mean")
    baseline_dropped = baseline_row.get("dropped_traffic_mean")
    delay = row.get("avg_delay_ms_mean")
    baseline_delay = baseline_row.get("avg_delay_ms_mean")
    path_latency = row.get("avg_path_latency_ms_mean")
    baseline_path_latency = baseline_row.get("avg_path_latency_ms_mean")
    qos_rate = row.get("qos_violation_rate_mean")
    baseline_qos_rate = baseline_row.get("qos_violation_rate_mean")
    energy = row.get("energy_kwh_mean")
    baseline_energy = baseline_row.get("energy_kwh_mean")
    carbon = row.get("carbon_g_mean")
    baseline_carbon = baseline_row.get("carbon_g_mean")
    qos_count = row.get("qos_violation_count_mean")
    baseline_qos_count = baseline_row.get("qos_violation_count_mean")

    row["delivered_traffic_delta_vs_baseline"] = (
        None if delivered is None or baseline_delivered is None else float(delivered) - float(baseline_delivered)
    )
    row["delivered_traffic_change_pct_vs_baseline"] = _pct_change(baseline_delivered, delivered)
    row["dropped_traffic_delta_vs_baseline"] = (
        None if dropped is None or baseline_dropped is None else float(dropped) - float(baseline_dropped)
    )
    row["dropped_traffic_change_pct_vs_baseline"] = _pct_change(baseline_dropped, dropped)
    row["avg_delay_ms_delta_vs_baseline"] = (
        None if delay is None or baseline_delay is None else float(delay) - float(baseline_delay)
    )
    row["avg_delay_ms_change_pct_vs_baseline"] = _pct_change(baseline_delay, delay)
    row["avg_path_latency_ms_delta_vs_baseline"] = (
        None if path_latency is None or baseline_path_latency is None else float(path_latency) - float(baseline_path_latency)
    )
    row["avg_path_latency_ms_change_pct_vs_baseline"] = _pct_change(baseline_path_latency, path_latency)
    row["qos_violation_rate_delta_vs_baseline"] = (
        None if qos_rate is None or baseline_qos_rate is None else float(qos_rate) - float(baseline_qos_rate)
    )
    row["qos_violation_count_delta_vs_baseline"] = (
        None if qos_count is None or baseline_qos_count is None else float(qos_count) - float(baseline_qos_count)
    )
    row["energy_kwh_delta_vs_baseline"] = (
        None if energy is None or baseline_energy is None else float(energy) - float(baseline_energy)
    )
    row["energy_reduction_pct_vs_baseline"] = _pct_reduction(baseline_energy, energy)
    row["carbon_g_delta_vs_baseline"] = (
        None if carbon is None or baseline_carbon is None else float(carbon) - float(baseline_carbon)
    )
    row["carbon_reduction_pct_vs_baseline"] = _pct_reduction(baseline_carbon, carbon)

    # These thresholds are reporting heuristics, not simulator-native acceptance checks.
    # The defaults are intentionally exposed on the CLI because "acceptable QoS degradation"
    # is a thesis/reporting choice rather than a single hard-coded product rule.
    gate_checks: Dict[str, bool] = {}
    missing: List[str] = []

    delivered_change_pct = row.get("delivered_traffic_change_pct_vs_baseline")
    if delivered_change_pct is None:
        missing.append("delivered_traffic")
    else:
        gate_checks["delivered_traffic"] = float(delivered_change_pct) >= -float(thresholds.max_delivered_loss_pct)

    dropped_change_pct = row.get("dropped_traffic_change_pct_vs_baseline")
    if dropped_change_pct is None:
        missing.append("dropped_traffic")
    else:
        gate_checks["dropped_traffic"] = float(dropped_change_pct) <= float(thresholds.max_dropped_increase_pct)

    delay_change_pct = row.get("avg_delay_ms_change_pct_vs_baseline")
    if delay_change_pct is None:
        missing.append("avg_delay_ms")
    else:
        gate_checks["avg_delay_ms"] = float(delay_change_pct) <= float(thresholds.max_delay_increase_pct)

    path_latency_change_pct = row.get("avg_path_latency_ms_change_pct_vs_baseline")
    if path_latency_change_pct is None:
        missing.append("avg_path_latency_ms")
    else:
        gate_checks["avg_path_latency_ms"] = float(path_latency_change_pct) <= float(thresholds.max_path_latency_increase_pct)

    qos_rate_delta = row.get("qos_violation_rate_delta_vs_baseline")
    if qos_rate_delta is None:
        missing.append("qos_violation_rate")
    else:
        gate_checks["qos_violation_rate"] = float(qos_rate_delta) <= float(thresholds.max_qos_violation_rate_increase_abs)

    row["qos_acceptability_missing"] = ",".join(missing)
    if missing:
        row["qos_acceptability_status"] = "insufficient_data"
    else:
        row["qos_acceptability_status"] = "acceptable" if all(gate_checks.values()) else "not_acceptable"

    energy_reduction_pct = row.get("energy_reduction_pct_vs_baseline")
    energy_ok = (
        energy_reduction_pct is not None
        and float(energy_reduction_pct) >= float(thresholds.energy_target_pct)
    )
    if row["qos_acceptability_status"] == "insufficient_data" or energy_reduction_pct is None:
        row["hypothesis_status"] = "insufficient_data"
    elif energy_ok and row["qos_acceptability_status"] == "acceptable":
        row["hypothesis_status"] = "achieved"
    else:
        row["hypothesis_status"] = "not_achieved"


def _rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, float, str]:
    status = str(row.get("hypothesis_status") or "")
    qos_status = str(row.get("qos_acceptability_status") or "")
    if status == "achieved":
        status_rank = 2.0
    elif qos_status == "acceptable":
        status_rank = 1.0
    elif qos_status == "not_acceptable":
        status_rank = 0.0
    else:
        status_rank = -1.0

    def _num(key: str, *, invert_if_positive: bool = False) -> float:
        value = _to_float(row.get(key))
        if value is None:
            return -1e18
        if invert_if_positive:
            return -max(value, 0.0)
        return value

    return (
        status_rank,
        _num("energy_reduction_pct_vs_baseline"),
        _num("delivered_traffic_change_pct_vs_baseline"),
        _num("qos_violation_rate_delta_vs_baseline", invert_if_positive=True),
        _num("dropped_traffic_change_pct_vs_baseline", invert_if_positive=True),
        _num("avg_delay_ms_change_pct_vs_baseline", invert_if_positive=True),
        _num("avg_path_latency_ms_change_pct_vs_baseline", invert_if_positive=True),
        str(row.get("policy") or ""),
    )


def _mark_best_policies(rows: List[Dict[str, Any]], *, ai_policies: set[str]) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope_type"]), str(row["scope"]))].append(row)
        row["is_best_policy_for_scope"] = False
        row["is_best_ai_policy_for_scope"] = False

    best_overall: Dict[str, Any] | None = None
    best_ai_overall: Dict[str, Any] | None = None

    for key, bucket in grouped.items():
        best_row = max(bucket, key=_rank_key)
        best_row["is_best_policy_for_scope"] = True
        if key == ("overall", "ALL"):
            best_overall = best_row

        ai_bucket = [row for row in bucket if str(row.get("policy")) in ai_policies]
        if ai_bucket:
            best_ai_row = max(ai_bucket, key=_rank_key)
            best_ai_row["is_best_ai_policy_for_scope"] = True
            if key == ("overall", "ALL"):
                best_ai_overall = best_ai_row

    return best_overall, best_ai_overall


def _build_summary_rows(
    run_rows: Sequence[Mapping[str, Any]],
    *,
    primary_baseline_policy: str,
    thresholds: HypothesisThresholds,
    ai_policies: set[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        scenario = str(row.get("scenario") or "")
        policy = str(row.get("policy") or "")
        grouped[("scenario", scenario, policy)].append(row)
        grouped[("overall", "ALL", policy)].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (scope_type, scope_value, policy), rows in sorted(grouped.items()):
        summary_rows.append(_aggregate_group(scope_type, scope_value, policy, rows))

    baseline_lookup = {
        (row["scope_type"], row["scope"]): row
        for row in summary_rows
        if row["policy"] == primary_baseline_policy
    }
    for row in summary_rows:
        baseline_row = baseline_lookup.get((row["scope_type"], row["scope"]))
        _attach_baseline_comparison(
            row,
            baseline_row,
            primary_baseline_policy=primary_baseline_policy,
            thresholds=thresholds,
        )

    _mark_best_policies(summary_rows, ai_policies=ai_policies)
    return summary_rows


def _fmt_number(value: Any, digits: int = 3) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{num:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{num:.{digits}f}%"


def _headline_for_row(row: Mapping[str, Any] | None, *, primary_baseline_policy: str) -> str:
    if row is None:
        return "No overall policy row was available."

    policy = str(row.get("policy") or "<unknown>")
    policy_class = str(row.get("policy_class") or "other")
    energy_reduction = _fmt_pct(row.get("energy_reduction_pct_vs_baseline"))
    delivered_change = _fmt_pct(row.get("delivered_traffic_change_pct_vs_baseline"))
    qos_rate_delta = row.get("qos_violation_rate_delta_vs_baseline")
    qos_phrase = "n/a"
    if qos_rate_delta is not None:
        qos_phrase = f"{float(qos_rate_delta):+.4f}"

    return (
        f"{policy} ({policy_class}) vs {primary_baseline_policy}: "
        f"energy {energy_reduction}, delivered {delivered_change}, "
        f"QoS violation rate delta {qos_phrase}, hypothesis={row.get('hypothesis_status', 'n/a')}"
    )


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | "
        "Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
    )
    lines = [header]
    for row in rows:
        policy_label = str(row["policy"])
        if row.get("is_best_policy_for_scope"):
            policy_label += " [best]"
        elif row.get("is_best_ai_policy_for_scope"):
            policy_label += " [best-ai]"

        lines.append(
            "| {policy} | {klass} | {runs} | {energy} | {energy_red} | {delivered} | {dropped} | "
            "{delay} | {path} | {qos_rate} | {qos_count} | {carbon} | {qos_status} | {hypothesis} |".format(
                policy=policy_label,
                klass=row.get("policy_class", "other"),
                runs=int(row.get("run_count") or 0),
                energy=_fmt_number(row.get("energy_kwh_mean"), 6),
                energy_red=_fmt_pct(row.get("energy_reduction_pct_vs_baseline")),
                delivered=_fmt_number(row.get("delivered_traffic_mean"), 3),
                dropped=_fmt_number(row.get("dropped_traffic_mean"), 3),
                delay=_fmt_number(row.get("avg_delay_ms_mean"), 3),
                path=_fmt_number(row.get("avg_path_latency_ms_mean"), 3),
                qos_rate=_fmt_number(row.get("qos_violation_rate_mean"), 4),
                qos_count=_fmt_number(row.get("qos_violation_count_mean"), 2),
                carbon=_fmt_number(row.get("carbon_g_mean"), 6),
                qos_status=row.get("qos_acceptability_status", "n/a"),
                hypothesis=row.get("hypothesis_status", "n/a"),
            )
        )
    return "\n".join(lines)


def _write_markdown_report(
    report_path: Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    generated_at_utc: str,
    primary_baseline_policy: str,
    baseline_policies: Sequence[str],
    ai_policies: Sequence[str],
    routing_baselines: Sequence[str],
    routing_link_cost_models: Sequence[str],
    thresholds: HypothesisThresholds,
    best_overall: Mapping[str, Any] | None,
    best_ai_overall: Mapping[str, Any] | None,
    selected_run_count: int,
    source_description: str,
) -> None:
    overall_rows = [row for row in summary_rows if row["scope_type"] == "overall"]
    scenario_names = sorted({str(row["scope"]) for row in summary_rows if row["scope_type"] == "scenario"})
    routing_consistency = "consistent"
    if len(routing_baselines) > 1 or len(routing_link_cost_models) > 1:
        routing_consistency = "mixed"

    parts: List[str] = [
        "# GreenNet Final Evaluation",
        "",
        "## Headline",
        f"- Generated at: `{generated_at_utc}`",
        f"- Source selection: `{source_description}`",
        f"- Selected runs: `{selected_run_count}`",
        f"- Primary baseline policy: `{primary_baseline_policy}`",
        f"- Baseline policies in scope: `{', '.join(baseline_policies) if baseline_policies else 'n/a'}`",
        f"- AI policies in scope: `{', '.join(ai_policies) if ai_policies else 'n/a'}`",
        f"- Routing baselines in scope: `{', '.join(routing_baselines) if routing_baselines else 'n/a'}`",
        f"- Routing link-cost models in scope: `{', '.join(routing_link_cost_models) if routing_link_cost_models else 'n/a'}`",
        f"- Routing comparison consistency: `{routing_consistency}`",
        f"- Best overall policy: `{best_overall.get('policy') if best_overall else 'n/a'}`",
        f"- Best AI policy: `{best_ai_overall.get('policy') if best_ai_overall else 'n/a'}`",
        f"- Overall best-policy summary: {_headline_for_row(best_overall, primary_baseline_policy=primary_baseline_policy)}",
        f"- Overall best-AI summary: {_headline_for_row(best_ai_overall, primary_baseline_policy=primary_baseline_policy)}",
        "",
        "## Hypothesis Gate",
        (
            "- Target: `energy reduction >= {energy:.1f}%` with acceptable QoS defined as "
            "`delivered loss <= {delivered:.1f}%`, `dropped increase <= {dropped:.1f}%`, "
            "`delay increase <= {delay:.1f}%`, `path latency increase <= {path:.1f}%`, "
            "`QoS violation rate increase <= {qos:.4f}`."
        ).format(
            energy=thresholds.energy_target_pct,
            delivered=thresholds.max_delivered_loss_pct,
            dropped=thresholds.max_dropped_increase_pct,
            delay=thresholds.max_delay_increase_pct,
            path=thresholds.max_path_latency_increase_pct,
            qos=thresholds.max_qos_violation_rate_increase_abs,
        ),
        "",
        "## Overall Comparison",
        _markdown_table(overall_rows),
        "",
    ]

    for scenario in scenario_names:
        scenario_rows = [row for row in summary_rows if row["scope_type"] == "scenario" and row["scope"] == scenario]
        parts.extend(
            [
                f"## Scenario: {scenario}",
                _markdown_table(scenario_rows),
                "",
            ]
        )

    report_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


CSV_COLUMNS = [
    "scope_type",
    "scope",
    "scenario",
    "policy",
    "policy_class",
    "controller_policy",
    "controller_policy_class",
    "routing_baseline",
    "routing_link_cost_model",
    "routing_forwarding_model",
    "routing_path_split",
    "run_count",
    "seed_count",
    "episodes_total",
    "steps_total",
    "seed_list",
    "comparison_baseline_policy",
    "comparison_available",
    "is_primary_baseline",
    "is_best_policy_for_scope",
    "is_best_ai_policy_for_scope",
    "qos_acceptability_status",
    "qos_acceptability_missing",
    "hypothesis_status",
    "energy_kwh_mean",
    "energy_kwh_std",
    "energy_kwh_count",
    "energy_kwh_delta_vs_baseline",
    "energy_reduction_pct_vs_baseline",
    "delivered_traffic_mean",
    "delivered_traffic_std",
    "delivered_traffic_count",
    "delivered_traffic_delta_vs_baseline",
    "delivered_traffic_change_pct_vs_baseline",
    "dropped_traffic_mean",
    "dropped_traffic_std",
    "dropped_traffic_count",
    "dropped_traffic_delta_vs_baseline",
    "dropped_traffic_change_pct_vs_baseline",
    "avg_delay_ms_mean",
    "avg_delay_ms_std",
    "avg_delay_ms_count",
    "avg_delay_ms_delta_vs_baseline",
    "avg_delay_ms_change_pct_vs_baseline",
    "avg_path_latency_ms_mean",
    "avg_path_latency_ms_std",
    "avg_path_latency_ms_count",
    "avg_path_latency_ms_delta_vs_baseline",
    "avg_path_latency_ms_change_pct_vs_baseline",
    "qos_violation_rate_mean",
    "qos_violation_rate_std",
    "qos_violation_rate_count",
    "qos_violation_rate_delta_vs_baseline",
    "qos_violation_count_mean",
    "qos_violation_count_std",
    "qos_violation_count_count",
    "qos_violation_count_total",
    "qos_violation_count_delta_vs_baseline",
    "carbon_g_mean",
    "carbon_g_std",
    "carbon_g_count",
    "carbon_g_delta_vs_baseline",
    "carbon_reduction_pct_vs_baseline",
]


def _write_csv_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def _build_json_payload(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    run_rows: Sequence[Mapping[str, Any]],
    generated_at_utc: str,
    source_description: str,
    source_mode: str,
    primary_baseline_policy: str,
    baseline_policies: Sequence[str],
    ai_policies: Sequence[str],
    routing_baselines: Sequence[str],
    routing_link_cost_models: Sequence[str],
    thresholds: HypothesisThresholds,
    best_overall: Mapping[str, Any] | None,
    best_ai_overall: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "generated_at_utc": generated_at_utc,
        "source": {
            "mode": source_mode,
            "description": source_description,
            "selected_run_count": len(run_rows),
            "selected_policies": sorted({str(row.get("policy") or "") for row in run_rows}),
            "selected_scenarios": sorted({str(row.get("scenario") or "") for row in run_rows}),
        },
        "classification": {
            "primary_baseline_policy": primary_baseline_policy,
            "baseline_policies": list(baseline_policies),
            "ai_policies": list(ai_policies),
            "routing_baselines": list(routing_baselines),
            "routing_link_cost_models": list(routing_link_cost_models),
            "routing_comparison_consistency": (
                "mixed" if len(routing_baselines) > 1 or len(routing_link_cost_models) > 1 else "consistent"
            ),
        },
        "hypothesis_thresholds": {
            "energy_target_pct": thresholds.energy_target_pct,
            "max_qos_violation_rate_increase_abs": thresholds.max_qos_violation_rate_increase_abs,
            "max_delivered_loss_pct": thresholds.max_delivered_loss_pct,
            "max_dropped_increase_pct": thresholds.max_dropped_increase_pct,
            "max_delay_increase_pct": thresholds.max_delay_increase_pct,
            "max_path_latency_increase_pct": thresholds.max_path_latency_increase_pct,
        },
        "best_policy": best_overall,
        "best_ai_policy": best_ai_overall,
        "overall_hypothesis_status": None if best_ai_overall is None else best_ai_overall.get("hypothesis_status"),
        "summary_rows": list(summary_rows),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a thesis/reporting-ready baseline-vs-AI evaluation summary from existing GreenNet result artifacts."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional authoritative results_summary*.csv to select runs.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag filter when scanning results or reading a shared summary CSV.")
    parser.add_argument("--scenario", type=str, default=None, help="Optional comma-separated scenario filter.")
    parser.add_argument("--policy", type=str, default=None, help="Optional comma-separated policy filter.")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help="Only include deterministic runs.",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Only include stochastic runs.",
    )
    parser.add_argument(
        "--baseline-policies",
        type=str,
        default=",".join(DEFAULT_BASELINE_POLICIES),
        help="Comma-separated baseline policies used for classification.",
    )
    parser.add_argument(
        "--ai-policies",
        type=str,
        default=",".join(DEFAULT_AI_POLICIES),
        help="Comma-separated AI-enhanced policies used for classification.",
    )
    parser.add_argument(
        "--primary-baseline-policy",
        type=str,
        default=None,
        help="Baseline policy used for authoritative deltas and hypothesis checks. Defaults to heuristic/baseline/all_on/noop if present.",
    )
    parser.add_argument("--energy-target-pct", type=float, default=15.0)
    parser.add_argument("--max-qos-violation-rate-increase", type=float, default=0.02)
    parser.add_argument("--max-delivered-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-dropped-increase-pct", type=float, default=5.0)
    parser.add_argument("--max-delay-increase-pct", type=float, default=10.0)
    parser.add_argument("--max-path-latency-increase-pct", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_evaluation/latest"),
        help="Output folder for CSV/JSON/Markdown exports.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    scenario_filter = set(_clean_csv_list(args.scenario))
    policy_filter = set(_clean_csv_list(args.policy))
    baseline_policies = set(_clean_csv_list(args.baseline_policies))
    ai_policies = set(_clean_csv_list(args.ai_policies))

    if args.summary_csv is not None:
        summary_csv = _resolve_run_dir(str(args.summary_csv), repo_root=repo_root)
        if not summary_csv.exists():
            raise SystemExit(f"--summary-csv does not exist: {summary_csv}")
        run_rows = _load_run_rows_from_summary_csv(
            summary_csv,
            repo_root=repo_root,
            tag_filter=args.tag,
            scenario_filter=scenario_filter,
            policy_filter=policy_filter,
            deterministic_filter=args.deterministic,
            baseline_policies=baseline_policies,
            ai_policies=ai_policies,
        )
        source_mode = "summary_csv"
        source_description = _display_path(summary_csv, repo_root=repo_root)
    else:
        results_dir = _resolve_run_dir(str(args.results_dir), repo_root=repo_root)
        if not results_dir.exists():
            raise SystemExit(f"--results-dir does not exist: {results_dir}")
        selected_run_dirs = _scan_authoritative_run_dirs(
            results_dir,
            tag_filter=args.tag,
            scenario_filter=scenario_filter,
            policy_filter=policy_filter,
            deterministic_filter=args.deterministic,
        )
        run_rows = [
            _extract_run_metrics(run_dir, baseline_policies=baseline_policies, ai_policies=ai_policies)
            for run_dir in selected_run_dirs
        ]
        source_mode = "scan"
        source_description = _display_path(results_dir, repo_root=repo_root)

    if not run_rows:
        raise SystemExit("No runs matched the requested filters.")
    run_rows = [row for row in run_rows if row.get("policy") and row.get("scenario")]
    if not run_rows:
        raise SystemExit("No valid run artifacts were found after loading run metadata.")

    primary_baseline_policy = _choose_primary_baseline(
        (row["policy"] for row in run_rows),
        requested=args.primary_baseline_policy,
        baseline_priority=DEFAULT_BASELINE_POLICIES,
    )

    thresholds = HypothesisThresholds(
        energy_target_pct=float(args.energy_target_pct),
        max_qos_violation_rate_increase_abs=float(args.max_qos_violation_rate_increase),
        max_delivered_loss_pct=float(args.max_delivered_loss_pct),
        max_dropped_increase_pct=float(args.max_dropped_increase_pct),
        max_delay_increase_pct=float(args.max_delay_increase_pct),
        max_path_latency_increase_pct=float(args.max_path_latency_increase_pct),
    )
    summary_rows = _build_summary_rows(
        run_rows,
        primary_baseline_policy=primary_baseline_policy,
        thresholds=thresholds,
        ai_policies=ai_policies,
    )

    best_overall = next(
        (
            row
            for row in summary_rows
            if row.get("scope_type") == "overall" and row.get("scope") == "ALL" and row.get("is_best_policy_for_scope")
        ),
        None,
    )
    best_ai_overall = next(
        (
            row
            for row in summary_rows
            if row.get("scope_type") == "overall" and row.get("scope") == "ALL" and row.get("is_best_ai_policy_for_scope")
        ),
        None,
    )

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    output_dir = _resolve_run_dir(str(args.output_dir), repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "final_evaluation_summary.csv"
    json_path = output_dir / "final_evaluation_summary.json"
    report_path = output_dir / "final_evaluation_report.md"

    _write_csv_summary(csv_path, summary_rows)
    baseline_policies_in_scope = sorted({row["policy"] for row in run_rows if row["policy"] in baseline_policies})
    ai_policies_in_scope = sorted({row["policy"] for row in run_rows if row["policy"] in ai_policies})
    routing_baselines_in_scope = sorted({str(row["routing_baseline"]) for row in run_rows if row.get("routing_baseline")})
    routing_link_cost_models_in_scope = sorted(
        {str(row["routing_link_cost_model"]) for row in run_rows if row.get("routing_link_cost_model")}
    )

    json_payload = _build_json_payload(
        summary_rows=summary_rows,
        run_rows=run_rows,
        generated_at_utc=generated_at_utc,
        source_description=source_description,
        source_mode=source_mode,
        primary_baseline_policy=primary_baseline_policy,
        baseline_policies=baseline_policies_in_scope,
        ai_policies=ai_policies_in_scope,
        routing_baselines=routing_baselines_in_scope,
        routing_link_cost_models=routing_link_cost_models_in_scope,
        thresholds=thresholds,
        best_overall=best_overall,
        best_ai_overall=best_ai_overall,
    )
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    _write_markdown_report(
        report_path,
        summary_rows=summary_rows,
        generated_at_utc=generated_at_utc,
        primary_baseline_policy=primary_baseline_policy,
        baseline_policies=baseline_policies_in_scope,
        ai_policies=ai_policies_in_scope,
        routing_baselines=routing_baselines_in_scope,
        routing_link_cost_models=routing_link_cost_models_in_scope,
        thresholds=thresholds,
        best_overall=best_overall,
        best_ai_overall=best_ai_overall,
        selected_run_count=len(run_rows),
        source_description=source_description,
    )

    print(f"[final_evaluation] CSV: {csv_path}")
    print(f"[final_evaluation] JSON: {json_path}")
    print(f"[final_evaluation] MD: {report_path}")
    if best_overall is not None:
        print(f"[final_evaluation] best overall: {best_overall.get('policy')} ({best_overall.get('hypothesis_status')})")
    if best_ai_overall is not None:
        print(f"[final_evaluation] best ai: {best_ai_overall.get('policy')} ({best_ai_overall.get('hypothesis_status')})")


if __name__ == "__main__":
    main()
