"""Metrics logging helpers for dashboards."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricLog:
    """Collect metrics for later visualization."""

    series: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.series.setdefault(name, []).append(value)

    def latest(self) -> Dict[str, float]:
        return {name: values[-1] for name, values in self.series.items() if values}
