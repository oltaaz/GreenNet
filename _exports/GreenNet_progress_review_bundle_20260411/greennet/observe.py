"""Observation builders with optional noise."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ObservationConfig:
    """Config for observation generation."""

    noise_std: float = 0.0


def build_observation(state: Dict[str, Any], config: ObservationConfig) -> Dict[str, Any]:
    """Return a shallow copy of state with optional numeric noise."""
    observation = dict(state)
    if config.noise_std <= 0:
        return observation

    noisy = {}
    for key, value in observation.items():
        if isinstance(value, (int, float)):
            noisy[key] = value + config.noise_std
        else:
            noisy[key] = value
    return noisy
