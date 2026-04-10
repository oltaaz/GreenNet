"""Queue/delay/loss simulation placeholder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SimulationResult:
    """Aggregated metrics from a simulation step."""

    queue_depths: Dict[int, int]
    average_delay: float
    packet_loss: float


def simulate_step(queue_depths: Dict[int, int]) -> SimulationResult:
    """Basic placeholder simulation; replace with a real queueing model."""
    if not queue_depths:
        return SimulationResult(queue_depths={}, average_delay=0.0, packet_loss=0.0)

    average_delay = sum(queue_depths.values()) / float(len(queue_depths))
    packet_loss = 0.0
    return SimulationResult(queue_depths=queue_depths, average_delay=average_delay, packet_loss=packet_loss)
