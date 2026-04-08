from __future__ import annotations

import numpy as np
import pytest

from greennet.env import EnvConfig, GreenNetEnv
from greennet.forecasting import (
    AdaptiveEmaDemandForecaster,
    DemandForecastConfig,
    EmaDemandForecaster,
    HoltDemandForecaster,
    build_demand_forecaster,
)
from greennet.simulator import Flow, StepMetrics


def _mean_abs_error(forecaster, series: list[float]) -> float:
    forecaster.reset(initial=0.0)
    horizon = max(1, int(forecaster.cfg.horizon_steps))
    preds = []
    actuals = []
    for idx, value in enumerate(series):
        forecaster.update(float(value))
        target_idx = idx + horizon
        if target_idx >= len(series):
            continue
        preds.append(float(forecaster.predict()))
        actuals.append(float(series[target_idx]))
    return float(np.mean(np.abs(np.asarray(preds) - np.asarray(actuals))))


def test_build_demand_forecaster_supports_ema_and_holt() -> None:
    ema = build_demand_forecaster(DemandForecastConfig(model="ema"))
    holt = build_demand_forecaster(DemandForecastConfig(model="holt"))
    adaptive = build_demand_forecaster(DemandForecastConfig(model="adaptive_ema"))

    assert isinstance(ema, EmaDemandForecaster)
    assert isinstance(holt, HoltDemandForecaster)
    assert isinstance(adaptive, AdaptiveEmaDemandForecaster)

    with pytest.raises(ValueError):
        build_demand_forecaster(DemandForecastConfig(model="unknown"))


def test_adaptive_ema_beats_single_ema_on_regime_shift_sequence() -> None:
    series = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        20.0,
        25.0,
        30.0,
        35.0,
        18.0,
        12.0,
        8.0,
        6.0,
        5.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
    ]

    ema = EmaDemandForecaster(DemandForecastConfig(model="ema", alpha=0.6, horizon_steps=3))
    adaptive = AdaptiveEmaDemandForecaster(
        DemandForecastConfig(
            model="adaptive_ema",
            adaptive_expert_alphas=(0.1, 0.2, 0.4, 0.6, 0.8, 0.95),
            adaptive_error_alpha=0.02,
            adaptive_temperature=0.25,
            horizon_steps=3,
        )
    )

    ema_mae = _mean_abs_error(ema, series)
    adaptive_mae = _mean_abs_error(adaptive, series)

    assert adaptive_mae < ema_mae


def test_env_can_use_adaptive_forecaster_without_breaking_observation_shape() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=4,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=2,
            decision_interval_steps=1,
            flows_per_step=0,
            enable_forecasting=True,
            forecast_model="adaptive_ema",
            forecast_adaptive_alphas=(0.1, 0.2, 0.4, 0.6, 0.8, 0.95),
            forecast_adaptive_error_alpha=0.02,
            forecast_adaptive_temperature=0.25,
            forecast_horizon_steps=3,
        )
    )
    try:
        env.reset(seed=11)
        assert isinstance(env._demand_forecaster, AdaptiveEmaDemandForecaster)

        flow_batches = [
            (Flow(0, 1, 2.0),),
            (Flow(0, 1, 4.0),),
        ]

        def _next_flows():
            return flow_batches.pop(0) if flow_batches else tuple()

        env._generate_flows = _next_flows  # type: ignore[method-assign]
        env.simulator.step = lambda flows: StepMetrics(  # type: ignore[method-assign]
            delivered=2.0,
            dropped=0.0,
            avg_utilization=0.1,
            avg_delay=0.0,
            avg_delay_ms=1.0,
            avg_path_latency_ms=1.0,
            energy_kwh=0.1,
            carbon_g=0.2,
        )

        obs1, _reward1, _terminated1, _truncated1, _info1 = env.step(0)
        obs2, _reward2, _terminated2, _truncated2, _info2 = env.step(0)

        assert obs1["demand_forecast"].shape == (1,)
        assert obs2["demand_forecast"].shape == (1,)
        assert float(obs2["demand_forecast"][0]) >= float(obs1["demand_forecast"][0])
    finally:
        env.close()
