from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

OFFICIAL_QOS_POLICY_NAME = "official_qos_v1"
OFFICIAL_QOS_POLICY_VERSION = 1
OFFICIAL_QOS_ACCEPTANCE_POLICY_NAME = "official_qos_acceptance_v1"
OFFICIAL_QOS_ACCEPTANCE_POLICY_VERSION = 1


@dataclass(frozen=True)
class QoSRuntimeThresholds:
    normalized_drop_ratio_threshold: float = 0.072
    min_volume: float = 500.0
    avg_delay_guard_multiplier: float = 4.0
    avg_delay_guard_margin_ms: float = 15.0
    recovery_delay_multiplier: float = 6.0
    recovery_delay_guard_margin_ms: float = 20.0
    p95_delay_threshold_ms: float | None = None


@dataclass(frozen=True)
class QoSAcceptanceThresholds:
    max_delivered_loss_pct: float = 2.0
    max_dropped_increase_pct: float = 5.0
    max_delay_increase_pct: float = 10.0
    max_path_latency_increase_pct: float = 10.0
    max_qos_violation_rate_increase_abs: float = 0.02


def _read_value(source: Any, key: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _signature_for_payload(name: str, version: int, payload: Mapping[str, Any]) -> str:
    stable_payload = json.dumps(
        {
            "policy_name": name,
            "policy_version": version,
            **payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:12]


def runtime_thresholds_from_config(config: Any) -> QoSRuntimeThresholds:
    defaults = QoSRuntimeThresholds()
    return QoSRuntimeThresholds(
        normalized_drop_ratio_threshold=float(
            _read_value(
                config,
                "qos_target_norm_drop",
                _read_value(config, "normalized_drop_ratio_threshold", defaults.normalized_drop_ratio_threshold),
            )
        ),
        min_volume=float(_read_value(config, "qos_min_volume", _read_value(config, "min_volume", defaults.min_volume))),
        avg_delay_guard_multiplier=float(
            _read_value(
                config,
                "qos_avg_delay_guard_multiplier",
                _read_value(config, "avg_delay_guard_multiplier", defaults.avg_delay_guard_multiplier),
            )
        ),
        avg_delay_guard_margin_ms=float(
            _read_value(
                config,
                "qos_avg_delay_guard_margin_ms",
                _read_value(config, "avg_delay_guard_margin_ms", defaults.avg_delay_guard_margin_ms),
            )
        ),
        recovery_delay_multiplier=float(
            _read_value(
                config,
                "qos_recovery_delay_multiplier",
                _read_value(config, "recovery_delay_multiplier", defaults.recovery_delay_multiplier),
            )
        ),
        recovery_delay_guard_margin_ms=float(
            _read_value(
                config,
                "qos_recovery_delay_guard_margin_ms",
                _read_value(
                    config,
                    "recovery_delay_guard_margin_ms",
                    defaults.recovery_delay_guard_margin_ms,
                ),
            )
        ),
        p95_delay_threshold_ms=_to_float(
            _read_value(
                config,
                "qos_p95_delay_threshold_ms",
                _read_value(config, "p95_delay_threshold_ms", defaults.p95_delay_threshold_ms),
            )
        ),
    )


def runtime_thresholds_metadata(thresholds: QoSRuntimeThresholds) -> dict[str, Any]:
    payload = asdict(thresholds)
    payload.update(
        {
            "policy_name": OFFICIAL_QOS_POLICY_NAME,
            "policy_version": OFFICIAL_QOS_POLICY_VERSION,
            "policy_signature": _signature_for_payload(
                OFFICIAL_QOS_POLICY_NAME,
                OFFICIAL_QOS_POLICY_VERSION,
                payload,
            ),
            "p95_delay_supported": thresholds.p95_delay_threshold_ms is not None,
        }
    )
    return payload


def acceptance_thresholds_metadata(thresholds: QoSAcceptanceThresholds) -> dict[str, Any]:
    payload = asdict(thresholds)
    payload.update(
        {
            "policy_name": OFFICIAL_QOS_ACCEPTANCE_POLICY_NAME,
            "policy_version": OFFICIAL_QOS_ACCEPTANCE_POLICY_VERSION,
            "policy_signature": _signature_for_payload(
                OFFICIAL_QOS_ACCEPTANCE_POLICY_NAME,
                OFFICIAL_QOS_ACCEPTANCE_POLICY_VERSION,
                payload,
            ),
        }
    )
    return payload


def delay_threshold_ms(
    avg_path_latency_ms: float | None,
    thresholds: QoSRuntimeThresholds,
    *,
    recovery: bool = False,
) -> float | None:
    if avg_path_latency_ms is None:
        return None
    path_latency = float(avg_path_latency_ms)
    multiplier = (
        float(thresholds.recovery_delay_multiplier)
        if recovery
        else float(thresholds.avg_delay_guard_multiplier)
    )
    margin_ms = (
        float(thresholds.recovery_delay_guard_margin_ms)
        if recovery
        else float(thresholds.avg_delay_guard_margin_ms)
    )
    return float(max(path_latency * multiplier, path_latency + margin_ms))


def delivery_loss_rate(delivered_total: float | None, dropped_total: float | None) -> float | None:
    if delivered_total is None or dropped_total is None:
        return None
    delivered = float(delivered_total)
    dropped = float(dropped_total)
    total = delivered + dropped
    if total <= 0.0:
        return 0.0
    return float(dropped / total)


def evaluate_run_qos(
    *,
    delivered_total: float | None,
    dropped_total: float | None,
    avg_delay_ms: float | None,
    avg_path_latency_ms: float | None,
    qos_violation_rate: float | None,
    qos_violation_count: float | None,
    thresholds: QoSRuntimeThresholds,
) -> dict[str, Any]:
    missing: list[str] = []
    checks: dict[str, bool] = {}

    delivered = None if delivered_total is None else float(delivered_total)
    dropped = None if dropped_total is None else float(dropped_total)
    volume = None if delivered is None or dropped is None else float(delivered + dropped)
    loss_rate = delivery_loss_rate(delivered, dropped)

    if volume is None:
        missing.append("delivery_loss_rate")
    elif volume < float(thresholds.min_volume):
        missing.append("min_volume")
    elif loss_rate is None:
        missing.append("delivery_loss_rate")
    else:
        checks["delivery_loss_rate"] = (
            float(loss_rate) <= float(thresholds.normalized_drop_ratio_threshold)
        )

    avg_delay = None if avg_delay_ms is None else float(avg_delay_ms)
    avg_path_latency = None if avg_path_latency_ms is None else float(avg_path_latency_ms)
    avg_delay_limit = delay_threshold_ms(avg_path_latency, thresholds, recovery=False)
    if avg_delay is None:
        missing.append("avg_delay_ms")
    elif avg_delay_limit is None:
        missing.append("avg_path_latency_ms")
    else:
        checks["avg_delay_ms"] = float(avg_delay) <= float(avg_delay_limit)

    status = "insufficient_data" if missing else ("acceptable" if all(checks.values()) else "not_acceptable")

    return {
        "qos_acceptance_status": status,
        "qos_acceptance_missing": ",".join(missing),
        "delivery_loss_rate": loss_rate,
        "delivery_success_rate": None if loss_rate is None else float(1.0 - loss_rate),
        "qos_delay_threshold_ms": avg_delay_limit,
        "qos_violation_rate": None if qos_violation_rate is None else float(qos_violation_rate),
        "qos_violation_count": None if qos_violation_count is None else float(qos_violation_count),
        "qos_checks": dict(checks),
        "qos_thresholds": runtime_thresholds_metadata(thresholds),
    }


def evaluate_qos_against_baseline(
    *,
    delivered_change_pct: float | None,
    dropped_change_pct: float | None,
    avg_delay_change_pct: float | None,
    avg_path_latency_change_pct: float | None,
    qos_violation_rate_delta: float | None,
    thresholds: QoSAcceptanceThresholds,
) -> dict[str, Any]:
    missing: list[str] = []
    checks: dict[str, bool] = {}

    if delivered_change_pct is None:
        missing.append("delivered_traffic")
    else:
        checks["delivered_traffic"] = float(delivered_change_pct) >= -float(thresholds.max_delivered_loss_pct)

    if dropped_change_pct is None:
        missing.append("dropped_traffic")
    else:
        checks["dropped_traffic"] = float(dropped_change_pct) <= float(thresholds.max_dropped_increase_pct)

    if avg_delay_change_pct is None:
        missing.append("avg_delay_ms")
    else:
        checks["avg_delay_ms"] = float(avg_delay_change_pct) <= float(thresholds.max_delay_increase_pct)

    if avg_path_latency_change_pct is None:
        missing.append("avg_path_latency_ms")
    else:
        checks["avg_path_latency_ms"] = (
            float(avg_path_latency_change_pct) <= float(thresholds.max_path_latency_increase_pct)
        )

    if qos_violation_rate_delta is None:
        missing.append("qos_violation_rate")
    else:
        checks["qos_violation_rate"] = (
            float(qos_violation_rate_delta) <= float(thresholds.max_qos_violation_rate_increase_abs)
        )

    status = "insufficient_data" if missing else ("acceptable" if all(checks.values()) else "not_acceptable")
    missing_text = ",".join(missing)
    return {
        "qos_acceptability_status": status,
        "qos_acceptability_missing": missing_text,
        "qos_acceptance_status": status,
        "qos_acceptance_missing": missing_text,
        "qos_gate_checks": dict(checks),
        "qos_thresholds": acceptance_thresholds_metadata(thresholds),
    }
