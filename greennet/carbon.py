"""Carbon intensity model used to convert energy into emissions."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class CarbonModel:
    """Smooth diurnal carbon intensity profile.

    The simulator time argument is interpreted in the same units as `period`.
    GreenNet currently advances in seconds, so the default period represents a
    24-hour day.
    """

    base_intensity: float = 400.0
    amplitude: float = 25.0
    period: float = 24.0 * 3600.0

    def intensity_at(self, time: float) -> float:
        if self.period <= 0:
            return max(0.0, float(self.base_intensity))
        phase = 2.0 * math.pi * ((time % self.period) / self.period)
        return max(0.0, float(self.base_intensity + self.amplitude * math.sin(phase)))
