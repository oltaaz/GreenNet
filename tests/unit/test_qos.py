from __future__ import annotations

import pytest

from greennet.qos import (
    QoSAcceptanceThresholds,
    delay_threshold_ms,
    evaluate_qos_against_baseline,
    evaluate_run_qos,
    runtime_thresholds_from_config,
    runtime_thresholds_metadata,
)


def test_runtime_thresholds_from_metadata_round_trips() -> None:
    thresholds = runtime_thresholds_from_config(
        {
            "qos_target_norm_drop": 0.05,
            "qos_min_volume": 200.0,
            "qos_avg_delay_guard_multiplier": 3.0,
            "qos_avg_delay_guard_margin_ms": 12.0,
            "qos_recovery_delay_multiplier": 5.0,
            "qos_recovery_delay_guard_margin_ms": 18.0,
        }
    )

    metadata = runtime_thresholds_metadata(thresholds)
    reconstructed = runtime_thresholds_from_config(metadata)

    assert reconstructed == thresholds
    assert metadata["policy_name"] == "official_qos_v1"
    assert metadata["policy_signature"]


def test_evaluate_run_qos_checks_loss_and_delay() -> None:
    thresholds = runtime_thresholds_from_config(
        {
            "qos_target_norm_drop": 0.10,
            "qos_min_volume": 0.0,
            "qos_avg_delay_guard_multiplier": 2.0,
            "qos_avg_delay_guard_margin_ms": 5.0,
        }
    )

    accepted = evaluate_run_qos(
        delivered_total=90.0,
        dropped_total=5.0,
        avg_delay_ms=18.0,
        avg_path_latency_ms=10.0,
        qos_violation_rate=0.0,
        qos_violation_count=0.0,
        thresholds=thresholds,
    )
    rejected = evaluate_run_qos(
        delivered_total=90.0,
        dropped_total=20.0,
        avg_delay_ms=28.0,
        avg_path_latency_ms=10.0,
        qos_violation_rate=0.5,
        qos_violation_count=10.0,
        thresholds=thresholds,
    )

    assert delay_threshold_ms(10.0, thresholds) == pytest.approx(20.0)
    assert accepted["qos_acceptance_status"] == "acceptable"
    assert accepted["delivery_loss_rate"] == pytest.approx(5.0 / 95.0)
    assert accepted["qos_delay_threshold_ms"] == pytest.approx(20.0)
    assert rejected["qos_acceptance_status"] == "not_acceptable"


def test_evaluate_qos_against_baseline_uses_shared_thresholds() -> None:
    thresholds = QoSAcceptanceThresholds(
        max_delivered_loss_pct=2.0,
        max_dropped_increase_pct=5.0,
        max_delay_increase_pct=10.0,
        max_path_latency_increase_pct=10.0,
        max_qos_violation_rate_increase_abs=0.02,
    )

    accepted = evaluate_qos_against_baseline(
        delivered_change_pct=-1.0,
        dropped_change_pct=4.0,
        avg_delay_change_pct=8.0,
        avg_path_latency_change_pct=6.0,
        qos_violation_rate_delta=0.01,
        thresholds=thresholds,
    )
    rejected = evaluate_qos_against_baseline(
        delivered_change_pct=-3.0,
        dropped_change_pct=4.0,
        avg_delay_change_pct=8.0,
        avg_path_latency_change_pct=6.0,
        qos_violation_rate_delta=0.01,
        thresholds=thresholds,
    )

    assert accepted["qos_acceptability_status"] == "acceptable"
    assert accepted["qos_acceptance_status"] == "acceptable"
    assert rejected["qos_acceptability_status"] == "not_acceptable"
