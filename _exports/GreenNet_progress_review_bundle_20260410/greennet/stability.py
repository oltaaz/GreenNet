from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

OFFICIAL_STABILITY_POLICY_NAME = "official_stability_v1"
OFFICIAL_STABILITY_POLICY_VERSION = 1


@dataclass(frozen=True)
class StabilityPolicy:
    decision_interval_steps: int = 10
    toggle_cooldown_steps: int = 10
    global_toggle_cooldown_steps: int = 5
    off_calm_steps_required: int = 20
    max_off_toggles_per_episode: int = 1
    max_total_toggles_per_episode: int = 4
    max_emergency_on_toggles_per_episode: int = 8
    emergency_on_bypasses_cooldown: bool = True
    reversal_window_steps: int = 20
    reversal_penalty: float = 0.05
    min_steps_for_assessment: int = 50
    max_transition_rate: float = 0.02
    max_flap_rate: float = 0.25
    max_flap_count: int = 2


def _read_value(source: Any, key: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


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


def stability_policy_from_config(config: Any) -> StabilityPolicy:
    defaults = StabilityPolicy()
    return StabilityPolicy(
        decision_interval_steps=int(_read_value(config, "decision_interval_steps", defaults.decision_interval_steps)),
        toggle_cooldown_steps=int(_read_value(config, "toggle_cooldown_steps", defaults.toggle_cooldown_steps)),
        global_toggle_cooldown_steps=int(
            _read_value(config, "global_toggle_cooldown_steps", defaults.global_toggle_cooldown_steps)
        ),
        off_calm_steps_required=int(_read_value(config, "off_calm_steps_required", defaults.off_calm_steps_required)),
        max_off_toggles_per_episode=int(
            _read_value(config, "max_off_toggles_per_episode", defaults.max_off_toggles_per_episode)
        ),
        max_total_toggles_per_episode=int(
            _read_value(config, "max_total_toggles_per_episode", defaults.max_total_toggles_per_episode)
        ),
        max_emergency_on_toggles_per_episode=int(
            _read_value(
                config,
                "max_emergency_on_toggles_per_episode",
                defaults.max_emergency_on_toggles_per_episode,
            )
        ),
        emergency_on_bypasses_cooldown=bool(
            _read_value(config, "emergency_on_bypasses_cooldown", defaults.emergency_on_bypasses_cooldown)
        ),
        reversal_window_steps=int(
            _read_value(config, "stability_reversal_window_steps", defaults.reversal_window_steps)
        ),
        reversal_penalty=float(_read_value(config, "stability_reversal_penalty", defaults.reversal_penalty)),
        min_steps_for_assessment=int(
            _read_value(config, "stability_min_steps_for_assessment", defaults.min_steps_for_assessment)
        ),
        max_transition_rate=float(
            _read_value(config, "stability_max_transition_rate", defaults.max_transition_rate)
        ),
        max_flap_rate=float(_read_value(config, "stability_max_flap_rate", defaults.max_flap_rate)),
        max_flap_count=int(_read_value(config, "stability_max_flap_count", defaults.max_flap_count)),
    )


def stability_policy_metadata(policy: StabilityPolicy) -> dict[str, Any]:
    payload = asdict(policy)
    payload.update(
        {
            "policy_name": OFFICIAL_STABILITY_POLICY_NAME,
            "policy_version": OFFICIAL_STABILITY_POLICY_VERSION,
            "policy_signature": _signature_for_payload(
                OFFICIAL_STABILITY_POLICY_NAME,
                OFFICIAL_STABILITY_POLICY_VERSION,
                payload,
            ),
        }
    )
    return payload


def transition_rate(transition_count_total: float | None, steps: float | None) -> float | None:
    if transition_count_total is None or steps is None:
        return None
    step_count = float(steps)
    if step_count <= 0.0:
        return None
    return float(float(transition_count_total) / step_count)


def flap_rate(flap_event_count_total: float | None, transition_count_total: float | None) -> float | None:
    if flap_event_count_total is None or transition_count_total is None:
        return None
    transitions = float(transition_count_total)
    if transitions <= 0.0:
        return 0.0
    return float(float(flap_event_count_total) / transitions)


def blocked_rate(blocked_count: float | None, attempted_count: float | None) -> float | None:
    if blocked_count is None or attempted_count is None:
        return None
    attempts = float(attempted_count)
    if attempts <= 0.0:
        return 0.0
    return float(float(blocked_count) / attempts)


def evaluate_run_stability(
    *,
    steps: float | None,
    transition_count_total: float | None,
    flap_event_count_total: float | None,
    blocked_by_cooldown_count: float | None = None,
    toggles_attempted_count: float | None = None,
    policy: StabilityPolicy,
) -> dict[str, Any]:
    missing: list[str] = []

    step_count = None if steps is None else float(steps)
    transitions = None if transition_count_total is None else float(transition_count_total)
    flaps = None if flap_event_count_total is None else float(flap_event_count_total)

    if step_count is None:
        missing.append("steps")
    elif step_count < float(policy.min_steps_for_assessment):
        missing.append("min_steps_for_assessment")

    if transitions is None:
        missing.append("transition_count_total")
    if flaps is None:
        missing.append("flap_event_count_total")

    measured_transition_rate = transition_rate(transitions, step_count)
    measured_flap_rate = flap_rate(flaps, transitions)
    measured_cooldown_block_rate = blocked_rate(blocked_by_cooldown_count, toggles_attempted_count)

    checks: dict[str, bool] = {}
    if not missing and measured_transition_rate is not None:
        checks["transition_rate"] = measured_transition_rate <= float(policy.max_transition_rate)
    if not missing and measured_flap_rate is not None:
        checks["flap_rate"] = measured_flap_rate <= float(policy.max_flap_rate)
    if not missing and flaps is not None:
        checks["flap_count"] = flaps <= float(policy.max_flap_count)

    status = "insufficient_data" if missing else ("stable" if all(checks.values()) else "unstable")

    return {
        "stability_status": status,
        "stability_missing": ",".join(missing),
        "transition_count_total": None if transitions is None else float(transitions),
        "flap_event_count_total": None if flaps is None else float(flaps),
        "transition_rate": measured_transition_rate,
        "flap_rate": measured_flap_rate,
        "blocked_by_cooldown_rate": measured_cooldown_block_rate,
        "stability_flapping_detected": bool((flaps or 0.0) > 0.0),
        "stability_checks": dict(checks),
        "stability_thresholds": stability_policy_metadata(policy),
    }
