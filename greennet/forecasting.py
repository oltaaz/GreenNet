"""Lightweight traffic/demand forecasting utilities.

The environment needs cheap online forecasting rather than an offline training
stack. We keep the baseline EMA forecaster, add a damped Holt trend model, and
provide an adaptive EMA ensemble that can outperform a single fixed smoother on
mixed traffic regimes without introducing a training step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _clip_open_unit(value: float, default: float) -> float:
    value = float(value)
    if 0.0 < value <= 1.0:
        return value
    return float(default)


def _normalize_model_name(name: str) -> str:
    return str(name or "ema").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class DemandForecastConfig:
    """Config for the online demand forecaster."""

    # Forecaster choice. "ema" preserves existing behavior.
    model: str = "ema"

    # Exponential moving average smoothing factor in (0, 1].
    # Higher alpha => reacts faster to spikes; lower alpha => smoother.
    alpha: float = 0.3

    # Trend smoothing factor for Holt-style updates.
    beta: float = 0.2

    # Damps multi-step trend extrapolation so forecasts stay practical on bursty
    # traffic. 1.0 disables damping.
    trend_damping: float = 0.9

    # Adaptive EMA ensemble configuration.
    adaptive_expert_alphas: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 0.95)
    adaptive_error_alpha: float = 0.02
    adaptive_temperature: float = 0.25

    # How many steps ahead we forecast.
    horizon_steps: int = 1


class DemandForecaster(Protocol):
    """Small runtime interface shared by all demand forecasters."""

    def reset(self, initial: float = 0.0) -> None: ...

    @property
    def last_observation(self) -> float: ...

    def update(self, x: float) -> None: ...

    def predict(self) -> float: ...


class EmaDemandForecaster:
    """Online exponential moving-average forecaster for scalar demand."""

    def __init__(self, cfg: DemandForecastConfig | None = None) -> None:
        self.cfg = cfg or DemandForecastConfig()
        self._ema: float = 0.0
        self._last_obs: float = 0.0

    def reset(self, initial: float = 0.0) -> None:
        self._ema = float(initial)
        self._last_obs = float(initial)

    @property
    def last_observation(self) -> float:
        return float(self._last_obs)

    def update(self, x: float) -> None:
        x = float(x)
        a = _clip_open_unit(self.cfg.alpha, default=0.3)
        self._ema = a * x + (1.0 - a) * self._ema
        self._last_obs = x

    def predict(self) -> float:
        # For EMA, the best "next step" forecast is the current EMA.
        return max(0.0, float(self._ema))


class HoltDemandForecaster:
    """Online damped Holt linear-trend forecaster for scalar demand.

    Compared with EMA, this keeps an explicit trend term and uses the configured
    forecast horizon when producing a look-ahead estimate.
    """

    def __init__(self, cfg: DemandForecastConfig | None = None) -> None:
        self.cfg = cfg or DemandForecastConfig(model="holt")
        self._level: float = 0.0
        self._trend: float = 0.0
        self._last_obs: float = 0.0
        self._seen: int = 0

    def reset(self, initial: float = 0.0) -> None:
        value = float(initial)
        self._level = value
        self._trend = 0.0
        self._last_obs = value
        self._seen = 0

    @property
    def last_observation(self) -> float:
        return float(self._last_obs)

    def update(self, x: float) -> None:
        x = float(x)
        alpha = _clip_open_unit(self.cfg.alpha, default=0.35)
        beta = _clip_open_unit(self.cfg.beta, default=0.2)
        phi = _clip_open_unit(self.cfg.trend_damping, default=0.9)

        if self._seen == 0:
            self._level = x
            self._trend = 0.0
        else:
            prev_level = float(self._level)
            prev_trend = float(self._trend)
            next_level = alpha * x + (1.0 - alpha) * (prev_level + phi * prev_trend)
            next_trend = beta * (next_level - prev_level) + (1.0 - beta) * phi * prev_trend
            self._level = float(next_level)
            self._trend = float(next_trend)

        self._last_obs = x
        self._seen += 1

    def predict(self) -> float:
        horizon = max(1, int(self.cfg.horizon_steps))
        phi = _clip_open_unit(self.cfg.trend_damping, default=0.9)
        if self._seen == 0:
            return max(0.0, float(self._level))
        if abs(phi - 1.0) < 1e-6:
            trend_term = float(horizon) * float(self._trend)
        else:
            trend_term = float(self._trend) * (phi * (1.0 - (phi ** horizon)) / (1.0 - phi))
        return max(0.0, float(self._level) + trend_term)


class AdaptiveEmaDemandForecaster:
    """Adaptive ensemble over multiple EMA experts.

    Each expert uses a different smoothing factor. The ensemble tracks horizon-
    aligned absolute errors online and blends experts with lower recent error
    more heavily. This keeps the model lightweight while adapting across calm
    periods and sharp traffic bursts.
    """

    def __init__(self, cfg: DemandForecastConfig | None = None) -> None:
        self.cfg = cfg or DemandForecastConfig(model="adaptive_ema")
        self._experts: list[EmaDemandForecaster] = []
        self._expert_errors = np.ones(1, dtype=np.float64)
        self._pending_predictions: list[deque[float]] = []
        self._last_obs: float = 0.0
        self.reset(initial=0.0)

    def _expert_alphas(self) -> tuple[float, ...]:
        raw = tuple(float(value) for value in self.cfg.adaptive_expert_alphas)
        valid = tuple(_clip_open_unit(value, default=0.3) for value in raw if value > 0.0)
        if valid:
            return valid
        return (0.1, 0.2, 0.4, 0.6, 0.8, 0.95)

    def reset(self, initial: float = 0.0) -> None:
        initial_value = float(initial)
        horizon = max(1, int(self.cfg.horizon_steps))
        self._experts = [
            EmaDemandForecaster(
                DemandForecastConfig(
                    model="ema",
                    alpha=float(alpha),
                    horizon_steps=horizon,
                )
            )
            for alpha in self._expert_alphas()
        ]
        for expert in self._experts:
            expert.reset(initial=initial_value)
        self._expert_errors = np.ones(max(1, len(self._experts)), dtype=np.float64)
        self._pending_predictions = [deque() for _ in self._experts]
        self._last_obs = initial_value

    @property
    def last_observation(self) -> float:
        return float(self._last_obs)

    def update(self, x: float) -> None:
        x = float(x)
        horizon = max(1, int(self.cfg.horizon_steps))
        err_alpha = _clip_open_unit(self.cfg.adaptive_error_alpha, default=0.02)
        for idx, expert in enumerate(self._experts):
            pending = self._pending_predictions[idx]
            if len(pending) >= horizon:
                resolved_prediction = float(pending.popleft())
                err = abs(resolved_prediction - x)
                self._expert_errors[idx] = err_alpha * err + (1.0 - err_alpha) * self._expert_errors[idx]
            expert.update(x)
            pending.append(float(expert.predict()))
        self._last_obs = x

    def predict(self) -> float:
        if not self._experts:
            return 0.0
        temperature = float(self.cfg.adaptive_temperature)
        if temperature <= 0.0:
            temperature = 0.25
        mean_error = max(float(np.mean(self._expert_errors)), 1e-9)
        scaled = (-temperature * self._expert_errors / mean_error).astype(np.float64, copy=False)
        scaled -= float(np.max(scaled))
        weights = np.exp(scaled)
        weights /= max(float(np.sum(weights)), 1e-9)
        expert_predictions = np.asarray([expert.predict() for expert in self._experts], dtype=np.float64)
        return max(0.0, float(np.dot(weights, expert_predictions)))


def build_demand_forecaster(cfg: DemandForecastConfig | None = None) -> DemandForecaster:
    """Instantiate a configured demand forecaster."""

    cfg = cfg or DemandForecastConfig()
    model = _normalize_model_name(cfg.model)
    if model in {"ema", "baseline"}:
        return EmaDemandForecaster(cfg)
    if model in {"holt", "damped_holt", "holt_linear", "trend"}:
        return HoltDemandForecaster(cfg)
    if model in {"adaptive_ema", "adaptive", "ema_ensemble", "adaptive_ensemble"}:
        return AdaptiveEmaDemandForecaster(cfg)
    raise ValueError(
        f"Unsupported forecast model '{cfg.model}'. Expected one of: ema, adaptive_ema, holt"
    )


__all__ = [
    "DemandForecastConfig",
    "DemandForecaster",
    "AdaptiveEmaDemandForecaster",
    "EmaDemandForecaster",
    "HoltDemandForecaster",
    "build_demand_forecaster",
]
