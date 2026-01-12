"""Traffic generators and burst models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class TrafficBurst:
    """Represents a burst of traffic between two nodes."""

    source: int
    destination: int
    size: int
    start_time: float


@dataclass
class TrafficGenerator:
    """Generate traffic bursts; plug in a stochastic model later."""

    rate: float = 1.0

    def generate(self, horizon: int) -> Iterator[TrafficBurst]:
        for step in range(horizon):
            yield TrafficBurst(source=0, destination=1, size=1, start_time=float(step))
