from __future__ import annotations

from greennet.stability import StabilityPolicy, evaluate_run_stability


def test_evaluate_run_stability_reports_stable_for_low_transition_and_flap_rates() -> None:
    result = evaluate_run_stability(
        steps=300.0,
        transition_count_total=4.0,
        flap_event_count_total=0.0,
        blocked_by_cooldown_count=1.0,
        toggles_attempted_count=5.0,
        policy=StabilityPolicy(),
    )

    assert result["stability_status"] == "stable"
    assert result["transition_rate"] == 4.0 / 300.0
    assert result["flap_rate"] == 0.0


def test_evaluate_run_stability_reports_unstable_for_excess_flapping() -> None:
    result = evaluate_run_stability(
        steps=300.0,
        transition_count_total=6.0,
        flap_event_count_total=3.0,
        policy=StabilityPolicy(max_flap_rate=0.25, max_flap_count=2),
    )

    assert result["stability_status"] == "unstable"
    assert result["stability_checks"]["flap_rate"] is False
    assert result["stability_checks"]["flap_count"] is False


def test_evaluate_run_stability_requires_minimum_steps() -> None:
    result = evaluate_run_stability(
        steps=10.0,
        transition_count_total=1.0,
        flap_event_count_total=0.0,
        policy=StabilityPolicy(min_steps_for_assessment=50),
    )

    assert result["stability_status"] == "insufficient_data"
    assert "min_steps_for_assessment" in result["stability_missing"]
