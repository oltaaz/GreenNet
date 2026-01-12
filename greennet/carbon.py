"""Carbon intensity model (time-varying)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarbonModel:
    """Simple sinusoidal carbon intensity profile."""

    base_intensity: float = 400.0
    amplitude: float = 50.0
    period: float = 24.0

    def intensity_at(self, time: float) -> float:
        if self.period <= 0:
            return self.base_intensity
        phase = (time % self.period) / self.period
        return self.base_intensity + self.amplitude * (2.0 * phase - 1.0)
