"""Power controller and power model placeholders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PowerModel:
    """Simple power model stub; extend with hardware parameters."""

    base_watts: float = 50.0
    per_link_watts: float = 5.0

    def estimate(self, active_links: int) -> float:
        return self.base_watts + self.per_link_watts * active_links


@dataclass
class PowerController:
    """Track power usage for each node or device."""

    model: PowerModel

    def snapshot(self, active_links_by_node: Dict[int, int]) -> Dict[int, float]:
        return {
            node_id: self.model.estimate(active_links)
            for node_id, active_links in active_links_by_node.items()
        }
