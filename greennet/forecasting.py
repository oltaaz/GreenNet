"""
Lightweight traffic/demand forecasting utilities.

This project doesn't need a heavyweight forecasting stack to get the benefit of
"look-ahead" signals. A cheap online forecaster (EMA) is often enough to help
the agent anticipate demand spikes and avoid risky energy-saving actions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DemandForecastConfig:
    """Config for the online demand forecaster."""

    # Exponential moving average smoothing factor in (0, 1].
    # Higher alpha => reacts faster to spikes; lower alpha => smoother.
    alpha: float = 0.3

    # How many steps ahead we forecast. For EMA, horizon>1 is a flat forecast,
    # but keeping this parameter makes it easy to upgrade later.
    horizon_steps: int = 1


class EmaDemandForecaster:
    """Online exponential moving-average forecaster for scalar demand.

    Usage:
      f = EmaDemandForecaster(DemandForecastConfig(alpha=0.3))
      f.reset(initial=0.0)
      f.update(observed_demand)
      pred = f.predict()
    """

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
        a = float(self.cfg.alpha)
        # Clamp alpha defensively to keep things stable.
        if not (0.0 < a <= 1.0):
            a = 0.3
        self._ema = a * x + (1.0 - a) * self._ema
        self._last_obs = x

    def predict(self) -> float:
        # For EMA, the best "next step" forecast is the current EMA.
        return float(self._ema)