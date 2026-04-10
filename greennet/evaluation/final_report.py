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

from greennet.policy_taxonomy import (
    DEFAULT_AI_POLICIES,
    DEFAULT_HEURISTIC_BASELINE_POLICIES,
    DEFAULT_NON_AI_BASELINE_POLICIES,
    DEFAULT_TRADITIONAL_BASELINE_POLICIES,
    canonical_controller_policy_name,
    canonical_experiment_policy_name,
    controller_policy_class,
    experiment_policy_class,
    is_ai_policy,
    is_heuristic_baseline_policy,
    is_traditional_baseline_policy,
)
from greennet.qos import (
    QoSAcceptanceThresholds,
    acceptance_thresholds_metadata,
    evaluate_qos_against_baseline,
)
from greennet.stability import evaluate_run_stability, stability_policy_from_config

TAG_RE = re.compile(r"__tag-(.+)$")
DEFAULT_BASELINE_POLICIES = DEFAULT_NON_AI_BASELINE_POLICIES
DEFAULT_BASELINE_PRIORITY = (
    *DEFAULT_TRADITIONAL_BASELINE_POLICIES,
    *DEFAULT_HEURISTIC_BASELINE_POLICIES,
)
DEFAULT_QOS_ACCEPTANCE_THRESHOLDS = QoSAcceptanceThresholds()


@dataclass(frozen=True)
class MetricSpec:
    field: str
    label: str
    higher_is_better: bool
    part_of_qos_gate: bool = False


@dataclass(frozen=True)
class HypothesisThresholds:
    energy_target_pct: float = 15.0
    max_qos_violation_rate_increase_abs: float = DEFAULT_QOS_ACCEPTANCE_THRESHOLDS.max_qos_violation_rate_increase_abs
    max_delivered_loss_pct: float = DEFAULT_QOS_ACCEPTANCE_THRESHOLDS.max_delivered_loss_pct
    max_dropped_increase_pct: float = DEFAULT_QOS_ACCEPTANCE_THRESHOLDS.max_dropped_increase_pct
    max_delay_increase_pct: float = DEFAULT_QOS_ACCEPTANCE_THRESHOLDS.max_delay_increase_pct
    max_path_latency_increase_pct: float = DEFAULT_QOS_ACCEPTANCE_THRESHOLDS.max_path_latency_increase_pct

    def qos_thresholds(self) -> QoSAcceptanceThresholds:
        return QoSAcceptanceThresholds(
            max_delivered_loss_pct=float(self.max_delivered_loss_pct),
            max_dropped_increase_pct=float(self.max_dropped_increase_pct),
            max_delay_increase_pct=float(self.max_delay_increase_pct),
            max_path_latency_increase_pct=float(self.max_path_latency_increase_pct),
            max_qos_violation_rate_increase_abs=float(self.max_qos_violation_rate_increase_abs),
        )


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("energy_kwh", "Energy (kWh)", higher_is_better=False),
    MetricSpec("delivered_traffic", "Delivered traffic", higher_is_better=True, part_of_qos_gate=True),
    MetricSpec("dropped_traffic", "Dropped traffic", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("delivery_loss_rate", "Delivery-loss rate", higher_is_better=False),
    MetricSpec("avg_delay_ms", "Average delay (ms)", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("avg_path_latency_ms", "Average path latency (ms)", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("qos_violation_rate", "QoS violation rate", higher_is_better=False, part_of_qos_gate=True),
    MetricSpec("qos_violation_count", "QoS violation count", higher_is_better=False),
    MetricSpec("transition_count_total", "Transitions", higher_is_better=False),
    MetricSpec("transition_rate", "Transition rate", higher_is_better=False),
    MetricSpec("flap_event_count_total", "Flap events", higher_is_better=False),
    MetricSpec("flap_rate", "Flap rate", higher_is_better=False),
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
    canonical = canonical_experiment_policy_name(policy)
    if canonical in ai_policies or is_ai_policy(canonical):
        return "ai_policy"
    if is_heuristic_baseline_policy(canonical):
        return "heuristic_baseline"
    if canonical in baseline_policies or is_traditional_baseline_policy(canonical):
        return "traditional_baseline" if is_traditional_baseline_policy(canonical) else "non_ai_baseline"
    return experiment_policy_class(canonical)


def _controller_class_for_row(
    policy: str,
    *,
    explicit_value: Any,
    ai_policies: set[str],
) -> str:
    if explicit_value not in ("", None):
        return str(explicit_value)
    if policy in ai_policies or is_ai_policy(policy):
        return "ai_policy"
    return controller_policy_class(policy)


def _baseline_priority_for(baseline_policies: set[str]) -> tuple[str, ...]:
    ordered = [policy for policy in DEFAULT_BASELINE_PRIORITY if policy in baseline_policies]
    extras = sorted(policy for policy in baseline_policies if policy not in ordered)
    return tuple(ordered + extras)


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


def _choose_reference_policy(
    available_policies: Iterable[str],
    *,
    requested: str | None,
    priority: Sequence[str],
) -> str | None:
    available = {str(policy) for policy in available_policies}
    if requested:
        requested_canonical = canonical_experiment_policy_name(requested)
        if requested_canonical not in available:
            return None
        return requested_canonical
    for policy in priority:
        canonical = canonical_experiment_policy_name(policy)
        if canonical in available:
            return canonical
    return None


def _build_run_row_from_summary_csv_row(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    context_dir: Path,
    baseline_policies: set[str],
    ai_policies: set[str],
) -> Dict[str, Any]:
    policy = canonical_experiment_policy_name(row.get("policy"))
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
        "policy_class": str(row.get("policy_class") or _policy_class(policy, baseline_policies=baseline_policies, ai_policies=ai_policies)),
        "controller_policy": canonical_controller_policy_name(row.get("controller_policy") or policy),
        "controller_policy_class": _controller_class_for_row(
            policy,
            explicit_value=row.get("controller_policy_class"),
            ai_policies=ai_policies,
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
        "delivery_loss_rate": _to_float(row.get("delivery_loss_rate_mean") or row.get("delivery_loss_rate")),
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
            "transition_count_total": 0.0,
            "flap_event_count_total": 0.0,
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

            transition_count = _to_float(row.get("transition_count"))
            if transition_count is None:
                transition_count = float(int(_to_bool(row.get("toggle_applied")) is True))
            stats["transition_count_total"] += float(transition_count)

            flap_events = _to_float(row.get("flap_event_count"))
            stats["flap_event_count_total"] += float(0.0 if flap_events is None else flap_events)

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
        "delivery_loss_rate": _mean(
            [
                float(stats["dropped_traffic"]) / max(float(stats["delivered_traffic"]) + float(stats["dropped_traffic"]), 1e-9)
                for stats in per_episode
            ]
        ),
        "energy_kwh": _mean([float(stats["energy_kwh"]) for stats in per_episode]),
        "carbon_g": _mean([float(stats["carbon_g"]) for stats in per_episode]),
        "avg_delay_ms": _mean(delay_means),
        "avg_path_latency_ms": _mean(path_latency_means),
        # We use the weighted rate across all selected steps to avoid over-weighting shorter episodes.
        "qos_violation_rate": (float(total_qos_count) / float(total_qos_steps)) if total_qos_steps > 0 else None,
        "qos_violation_count": float(total_qos_count) if total_qos_steps > 0 else None,
        "transition_count_total": _mean([float(stats["transition_count_total"]) for stats in per_episode]),
        "transition_rate": _mean(
            [
                float(stats["transition_count_total"]) / max(float(stats["steps"]), 1.0)
                for stats in per_episode
            ]
        ),
        "flap_event_count_total": _mean([float(stats["flap_event_count_total"]) for stats in per_episode]),
        "flap_rate": _mean(
            [
                float(stats["flap_event_count_total"]) / max(float(stats["transition_count_total"]), 1.0)
                for stats in per_episode
            ]
        ),
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

    policy = canonical_experiment_policy_name(meta.get("policy"))
    scenario = str(meta.get("scenario") or "")
    episodes = _to_int(meta.get("episodes"))
    if episodes is None:
        episodes = _to_int(per_step.get("episode_count"))

    row = {
        "matrix_id": str(meta.get("matrix_id") or ""),
        "matrix_name": str(meta.get("matrix_name") or ""),
        "matrix_manifest": str(meta.get("matrix_manifest") or ""),
        "matrix_case_id": str(meta.get("matrix_case_id") or ""),
        "matrix_case_label": str(meta.get("matrix_case_label") or ""),
        "run_id": str(meta.get("run_id") or run_dir.name),
        "policy": policy,
        "policy_class": str(meta.get("policy_class") or _policy_class(policy, baseline_policies=baseline_policies, ai_policies=ai_policies)),
        "controller_policy": canonical_controller_policy_name(meta.get("controller_policy") or policy),
        "controller_policy_class": _controller_class_for_row(
            policy,
            explicit_value=meta.get("controller_policy_class"),
            ai_policies=ai_policies,
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
        "stability_policy_name": str(meta.get("stability_policy_name") or ""),
        "stability_policy_signature": str(meta.get("stability_policy_signature") or ""),
        "stability_thresholds": meta.get("stability_thresholds"),
        "stability_status": str(
            overall.get("stability_status")
            or meta.get("stability_status")
            or ""
        ),
        "stability_missing": str(
            overall.get("stability_missing")
            or meta.get("stability_missing")
            or ""
        ),
        "steps_total": _to_int(per_step.get("steps_total")),
        "energy_kwh": _to_float(overall.get("energy_kwh_total_mean")),
        "delivered_traffic": _to_float(overall.get("delivered_total_mean")),
        "dropped_traffic": _to_float(overall.get("dropped_total_mean")),
        "delivery_loss_rate": _to_float(overall.get("delivery_loss_rate_mean")),
        "avg_delay_ms": _to_float(overall.get("avg_delay_ms_mean")),
        "avg_path_latency_ms": _to_float(overall.get("avg_path_latency_ms_mean")),
        "qos_violation_rate": _to_float(per_step.get("qos_violation_rate")),
        "qos_violation_count": _to_float(per_step.get("qos_violation_count")),
        "transition_count_total": _to_float(overall.get("transition_count_total_mean")),
        "transition_rate": _to_float(overall.get("transition_rate_mean")),
        "flap_event_count_total": _to_float(overall.get("flap_event_count_total_mean")),
        "flap_rate": _to_float(overall.get("flap_rate_mean")),
        "carbon_g": _to_float(overall.get("carbon_g_total_mean")),
    }

    for field in (
        "energy_kwh",
        "delivered_traffic",
        "dropped_traffic",
        "delivery_loss_rate",
        "avg_delay_ms",
        "avg_path_latency_ms",
        "carbon_g",
        "transition_count_total",
        "transition_rate",
        "flap_event_count_total",
        "flap_rate",
    ):
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
        "matrix_id": _common_text(rows, "matrix_id") or "",
        "matrix_name": _common_text(rows, "matrix_name") or "",
        "matrix_manifest": _common_text(rows, "matrix_manifest") or "",
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
        "stability_policy_name": _common_text(rows, "stability_policy_name") or "",
        "stability_policy_signature": _common_text(rows, "stability_policy_signature") or "",
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

    stability_gate = evaluate_run_stability(
        steps=(float(out["steps_total"]) / float(max(out["run_count"], 1))) if out.get("steps_total") is not None else None,
        transition_count_total=_to_float(out.get("transition_count_total_mean")),
        flap_event_count_total=_to_float(out.get("flap_event_count_total_mean")),
        policy=stability_policy_from_config(rows[0].get("stability_thresholds", {})),
    )
    out["stability_status"] = stability_gate.get("stability_status")
    out["stability_missing"] = stability_gate.get("stability_missing")
    out["transition_rate_mean"] = stability_gate.get("transition_rate")
    out["flap_rate_mean"] = stability_gate.get("flap_rate")
    out["stability_thresholds"] = stability_gate.get("stability_thresholds")
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


def _attach_reference_metrics(
    row: Dict[str, Any],
    reference_row: Mapping[str, Any] | None,
    *,
    reference_policy: str | None,
    suffix: str,
) -> bool:
    row[f"comparison_{suffix}_policy"] = reference_policy
    row[f"comparison_{suffix}_available"] = bool(reference_row is not None)

    if reference_row is None:
        return False

    metric_pairs = (
        ("delivered_traffic", "change_pct"),
        ("dropped_traffic", "change_pct"),
        ("avg_delay_ms", "change_pct"),
        ("avg_path_latency_ms", "change_pct"),
        ("qos_violation_rate", "delta_only"),
        ("qos_violation_count", "delta_only"),
        ("energy_kwh", "reduction_pct"),
        ("carbon_g", "reduction_pct"),
    )

    for field, mode in metric_pairs:
        current_value = row.get(f"{field}_mean")
        reference_value = reference_row.get(f"{field}_mean")
        row[f"{field}_delta_vs_{suffix}"] = (
            None if current_value is None or reference_value is None else float(current_value) - float(reference_value)
        )
        if mode == "change_pct":
            row[f"{field}_change_pct_vs_{suffix}"] = _pct_change(reference_value, current_value)
        elif mode == "reduction_pct":
            target_field = "energy_reduction_pct" if field == "energy_kwh" else "carbon_reduction_pct"
            row[f"{target_field}_vs_{suffix}"] = _pct_reduction(reference_value, current_value)

    return True


def _attach_baseline_comparison(
    row: Dict[str, Any],
    baseline_row: Mapping[str, Any] | None,
    *,
    primary_baseline_policy: str,
    heuristic_baseline_policy: str | None,
    heuristic_baseline_row: Mapping[str, Any] | None,
    thresholds: HypothesisThresholds,
) -> None:
    row["official_traditional_baseline_policy"] = primary_baseline_policy
    row["strongest_heuristic_baseline_policy"] = heuristic_baseline_policy
    row["comparison_official_baseline_policy"] = primary_baseline_policy
    row["comparison_heuristic_baseline_policy"] = heuristic_baseline_policy

    comparison_available = _attach_reference_metrics(
        row,
        baseline_row,
        reference_policy=primary_baseline_policy,
        suffix="baseline",
    )
    row["comparison_available"] = comparison_available
    row["comparison_baseline_policy"] = primary_baseline_policy
    row["comparison_official_baseline_available"] = comparison_available
    row["comparison_official_baseline_policy"] = primary_baseline_policy

    _attach_reference_metrics(
        row,
        heuristic_baseline_row,
        reference_policy=heuristic_baseline_policy,
        suffix="heuristic_baseline",
    )

    if baseline_row is None:
        row["qos_acceptability_status"] = "insufficient_data"
        row["hypothesis_status"] = "insufficient_data"
        row["qos_acceptability_missing"] = "missing official baseline row"
        row["is_primary_baseline"] = False
        row["is_official_traditional_baseline"] = False
        row["is_heuristic_baseline"] = bool(heuristic_baseline_policy and row["policy"] == heuristic_baseline_policy)
        return

    row["is_primary_baseline"] = row["policy"] == primary_baseline_policy
    row["is_official_traditional_baseline"] = row["policy"] == primary_baseline_policy
    row["is_heuristic_baseline"] = bool(heuristic_baseline_policy and row["policy"] == heuristic_baseline_policy)

    qos_gate = evaluate_qos_against_baseline(
        delivered_change_pct=_to_float(row.get("delivered_traffic_change_pct_vs_baseline")),
        dropped_change_pct=_to_float(row.get("dropped_traffic_change_pct_vs_baseline")),
        avg_delay_change_pct=_to_float(row.get("avg_delay_ms_change_pct_vs_baseline")),
        avg_path_latency_change_pct=_to_float(row.get("avg_path_latency_ms_change_pct_vs_baseline")),
        qos_violation_rate_delta=_to_float(row.get("qos_violation_rate_delta_vs_baseline")),
        thresholds=thresholds.qos_thresholds(),
    )
    row.update(qos_gate)

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

    stability_status = str(row.get("stability_status") or "")
    if row["hypothesis_status"] == "insufficient_data" or stability_status in {"", "insufficient_data"}:
        row["stability_qualified_hypothesis_status"] = "insufficient_data"
    elif row["hypothesis_status"] == "achieved" and stability_status == "stable":
        row["stability_qualified_hypothesis_status"] = "achieved"
    else:
        row["stability_qualified_hypothesis_status"] = "not_achieved"


def _rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, float, str]:
    status = str(row.get("stability_qualified_hypothesis_status") or row.get("hypothesis_status") or "")
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
    heuristic_baseline_policy: str | None,
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
    heuristic_lookup = (
        {
            (row["scope_type"], row["scope"]): row
            for row in summary_rows
            if heuristic_baseline_policy is not None and row["policy"] == heuristic_baseline_policy
        }
        if heuristic_baseline_policy is not None
        else {}
    )
    for row in summary_rows:
        baseline_row = baseline_lookup.get((row["scope_type"], row["scope"]))
        heuristic_row = heuristic_lookup.get((row["scope_type"], row["scope"]))
        _attach_baseline_comparison(
            row,
            baseline_row,
            primary_baseline_policy=primary_baseline_policy,
            heuristic_baseline_policy=heuristic_baseline_policy,
            heuristic_baseline_row=heuristic_row,
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
        f"QoS violation rate delta {qos_phrase}, stability={row.get('stability_status', 'n/a')}, "
        f"operational={row.get('stability_qualified_hypothesis_status', row.get('hypothesis_status', 'n/a'))}"
    )


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | "
        "Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Transitions | Flap rate | Carbon (g) | "
        "QoS status | Stability | Operational | Hypothesis |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"
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
            "{delay} | {path} | {qos_rate} | {qos_count} | {transitions} | {flap_rate} | {carbon} | "
            "{qos_status} | {stability} | {operational} | {hypothesis} |".format(
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
                transitions=_fmt_number(row.get("transition_count_total_mean"), 2),
                flap_rate=_fmt_pct(
                    None if row.get("flap_rate_mean") is None else float(row.get("flap_rate_mean")) * 100.0
                ),
                carbon=_fmt_number(row.get("carbon_g_mean"), 6),
                qos_status=row.get("qos_acceptability_status", "n/a"),
                stability=row.get("stability_status", "n/a"),
                operational=row.get("stability_qualified_hypothesis_status", "n/a"),
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
    heuristic_baseline_policy: str | None,
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
        f"- Official traditional baseline policy: `{primary_baseline_policy}`",
        f"- Strongest heuristic baseline policy: `{heuristic_baseline_policy or 'n/a'}`",
        f"- Non-AI policies in scope: `{', '.join(baseline_policies) if baseline_policies else 'n/a'}`",
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
        "- Stability gate: use the exported `stability_status` from the centralized stability policy. "
        "Operational success requires the energy/QoS hypothesis to be achieved while stability remains `stable`.",
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
    "matrix_id",
    "matrix_name",
    "matrix_manifest",
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
    "stability_policy_name",
    "stability_policy_signature",
    "run_count",
    "seed_count",
    "episodes_total",
    "steps_total",
    "seed_list",
    "official_traditional_baseline_policy",
    "strongest_heuristic_baseline_policy",
    "comparison_baseline_policy",
    "comparison_available",
    "comparison_official_baseline_policy",
    "comparison_official_baseline_available",
    "comparison_heuristic_baseline_policy",
    "comparison_heuristic_baseline_available",
    "is_primary_baseline",
    "is_official_traditional_baseline",
    "is_heuristic_baseline",
    "is_best_policy_for_scope",
    "is_best_ai_policy_for_scope",
    "qos_acceptability_status",
    "qos_acceptability_missing",
    "stability_status",
    "stability_missing",
    "stability_qualified_hypothesis_status",
    "hypothesis_status",
    "energy_kwh_mean",
    "energy_kwh_std",
    "energy_kwh_count",
    "energy_kwh_delta_vs_baseline",
    "energy_reduction_pct_vs_baseline",
    "energy_kwh_delta_vs_heuristic_baseline",
    "energy_reduction_pct_vs_heuristic_baseline",
        "delivered_traffic_mean",
        "delivered_traffic_std",
        "delivered_traffic_count",
        "delivered_traffic_delta_vs_baseline",
        "delivered_traffic_change_pct_vs_baseline",
        "delivered_traffic_delta_vs_heuristic_baseline",
        "delivered_traffic_change_pct_vs_heuristic_baseline",
        "delivery_loss_rate_mean",
        "delivery_loss_rate_std",
        "delivery_loss_rate_count",
        "dropped_traffic_mean",
        "dropped_traffic_std",
        "dropped_traffic_count",
    "dropped_traffic_delta_vs_baseline",
    "dropped_traffic_change_pct_vs_baseline",
    "dropped_traffic_delta_vs_heuristic_baseline",
    "dropped_traffic_change_pct_vs_heuristic_baseline",
    "avg_delay_ms_mean",
    "avg_delay_ms_std",
    "avg_delay_ms_count",
    "avg_delay_ms_delta_vs_baseline",
    "avg_delay_ms_change_pct_vs_baseline",
    "avg_delay_ms_delta_vs_heuristic_baseline",
    "avg_delay_ms_change_pct_vs_heuristic_baseline",
    "avg_path_latency_ms_mean",
    "avg_path_latency_ms_std",
    "avg_path_latency_ms_count",
    "avg_path_latency_ms_delta_vs_baseline",
    "avg_path_latency_ms_change_pct_vs_baseline",
    "avg_path_latency_ms_delta_vs_heuristic_baseline",
    "avg_path_latency_ms_change_pct_vs_heuristic_baseline",
    "qos_violation_rate_mean",
    "qos_violation_rate_std",
    "qos_violation_rate_count",
    "qos_violation_rate_delta_vs_baseline",
    "qos_violation_rate_delta_vs_heuristic_baseline",
    "qos_violation_count_mean",
    "qos_violation_count_std",
    "qos_violation_count_count",
    "qos_violation_count_total",
    "qos_violation_count_delta_vs_baseline",
    "qos_violation_count_delta_vs_heuristic_baseline",
    "transition_count_total_mean",
    "transition_count_total_std",
    "transition_count_total_count",
    "transition_rate_mean",
    "transition_rate_std",
    "transition_rate_count",
    "flap_event_count_total_mean",
    "flap_event_count_total_std",
    "flap_event_count_total_count",
    "flap_rate_mean",
    "flap_rate_std",
    "flap_rate_count",
    "carbon_g_mean",
    "carbon_g_std",
    "carbon_g_count",
    "carbon_g_delta_vs_baseline",
    "carbon_reduction_pct_vs_baseline",
    "carbon_g_delta_vs_heuristic_baseline",
    "carbon_reduction_pct_vs_heuristic_baseline",
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
    heuristic_baseline_policy: str | None,
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
            "matrix_id": _common_text(run_rows, "matrix_id"),
            "matrix_name": _common_text(run_rows, "matrix_name"),
            "matrix_manifest": _common_text(run_rows, "matrix_manifest"),
            "matrix_case_ids": sorted({str(row.get("matrix_case_id") or "") for row in run_rows if str(row.get("matrix_case_id") or "")}),
        },
        "classification": {
            "primary_baseline_policy": primary_baseline_policy,
            "official_traditional_baseline_policy": primary_baseline_policy,
            "strongest_heuristic_baseline_policy": heuristic_baseline_policy,
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
        "qos_thresholds": acceptance_thresholds_metadata(thresholds.qos_thresholds()),
        "stability_thresholds": (
            None if best_ai_overall is None else best_ai_overall.get("stability_thresholds")
        ),
        "best_policy": best_overall,
        "best_ai_policy": best_ai_overall,
        "overall_hypothesis_status": None if best_ai_overall is None else best_ai_overall.get("hypothesis_status"),
        "overall_stability_status": None if best_ai_overall is None else best_ai_overall.get("stability_status"),
        "overall_operational_status": (
            None if best_ai_overall is None else best_ai_overall.get("stability_qualified_hypothesis_status")
        ),
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
        help="Comma-separated non-AI policies used for classification.",
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
        help="Official traditional baseline policy used for authoritative deltas and hypothesis checks. Defaults to all_on/noop if present.",
    )
    parser.add_argument(
        "--heuristic-baseline-policy",
        type=str,
        default=None,
        help="Optional strongest handcrafted heuristic baseline used for secondary comparisons. Defaults to heuristic/baseline if present.",
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

    baseline_priority = _baseline_priority_for(baseline_policies)
    primary_baseline_policy = _choose_primary_baseline(
        (row["policy"] for row in run_rows),
        requested=args.primary_baseline_policy,
        baseline_priority=baseline_priority,
    )
    heuristic_baseline_policy = _choose_reference_policy(
        (row["policy"] for row in run_rows),
        requested=args.heuristic_baseline_policy,
        priority=DEFAULT_HEURISTIC_BASELINE_POLICIES,
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
        heuristic_baseline_policy=heuristic_baseline_policy,
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
        heuristic_baseline_policy=heuristic_baseline_policy,
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
        heuristic_baseline_policy=heuristic_baseline_policy,
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
