"""Carbon intensity model used to convert energy into emissions."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict


CARBON_MODEL_NAME = "diurnal_sinusoid"


def _require_non_negative(value: Any, *, field_name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative number") from exc
    if x != x or x in (float("inf"), float("-inf")) or x < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(x)


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

    @classmethod
    def from_env_config(cls, env_config: Any) -> "CarbonModel":
        return cls(
            base_intensity=float(getattr(env_config, "carbon_base_intensity_g_per_kwh", 400.0)),
            amplitude=float(getattr(env_config, "carbon_amplitude_g_per_kwh", 25.0)),
            period=float(getattr(env_config, "carbon_period_seconds", 24.0 * 3600.0)),
        )

    def validate(self) -> "CarbonModel":
        _require_non_negative(self.base_intensity, field_name="carbon_base_intensity_g_per_kwh")
        _require_non_negative(self.amplitude, field_name="carbon_amplitude_g_per_kwh")
        _require_non_negative(self.period, field_name="carbon_period_seconds")
        return self

    def metadata(self) -> Dict[str, Any]:
        return {
            "carbon_model_name": CARBON_MODEL_NAME,
            "carbon_model_signature": self.signature(),
        }

    def signature(self) -> str:
        return (
            f"{CARBON_MODEL_NAME}"
            f"|base={float(self.base_intensity):.6g}"
            f"|amplitude={float(self.amplitude):.6g}"
            f"|period_s={float(self.period):.6g}"
        )

    def intensity_at(self, time: float) -> float:
        if self.period <= 0:
            return max(0.0, float(self.base_intensity))
        phase = 2.0 * math.pi * ((time % self.period) / self.period)
        return max(0.0, float(self.base_intensity + self.amplitude * math.sin(phase)))
