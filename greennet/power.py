"""Lightweight network power model used by the simulator.

The model stays intentionally simple for thesis/demo use:

- Devices (nodes) draw a fixed active or sleep power level.
- Active devices and links add a small linear utilization-dependent term.
- Links and devices that are not active still retain a small sleep draw.

This keeps the energy signal easy to explain while separating fixed power from
traffic-sensitive power in a defensible way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


def _clip_unit(value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if x != x or x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass(frozen=True)
class PowerSnapshot:
    """Per-step network power breakdown in watts."""

    total_watts: float
    fixed_watts: float
    variable_watts: float
    device_watts: float
    link_watts: float
    active_devices: int
    inactive_devices: int
    active_links: int
    inactive_links: int


@dataclass
class PowerModel:
    """Simple but explicit power model for links and devices.

    Assumptions:
    - A device is considered active when it has at least one active incident link.
    - Utilization is the per-link carried-load ratio already computed by the simulator.
    - Dynamic power is linear in utilization to avoid overfitting a small demo model.
    - In the current GreenNet environment, most energy variation comes from links
      because connectivity-preserving actions usually keep devices reachable.
    """

    network_fixed_watts: float = 20.0
    device_active_watts: float = 12.0
    device_sleep_watts: float = 2.0
    device_dynamic_watts: float = 3.0
    link_active_watts: float = 6.0
    link_sleep_watts: float = 0.2
    link_dynamic_watts: float = 2.0

    def estimate(self, active_links: int, mean_utilization: float = 0.0) -> float:
        """Backward-compatible single-device estimate in watts."""
        return self.estimate_device_watts(active_links=active_links, mean_utilization=mean_utilization)

    def estimate_device_watts(self, active_links: int, mean_utilization: float = 0.0) -> float:
        util = _clip_unit(mean_utilization)
        if int(active_links) > 0:
            return float(self.device_active_watts + self.device_dynamic_watts * util)
        return float(self.device_sleep_watts)

    def estimate_link_watts(self, *, active: bool, utilization: float = 0.0) -> float:
        if bool(active):
            util = _clip_unit(utilization)
            return float(self.link_active_watts + self.link_dynamic_watts * util)
        return float(self.link_sleep_watts)

    def estimate_network(self, graph: "nx.Graph") -> PowerSnapshot:
        """Estimate total network power from the graph's active/utilization attributes."""
        fixed_watts = float(self.network_fixed_watts)
        variable_watts = 0.0
        device_watts = 0.0
        link_watts = 0.0
        active_devices = 0
        inactive_devices = 0
        active_links = 0
        inactive_links = 0

        for _u, _v, data in graph.edges(data=True):
            is_active = bool(data.get("active", True))
            util = _clip_unit(data.get("utilization", 0.0)) if is_active else 0.0
            if is_active:
                active_links += 1
                fixed_watts += float(self.link_active_watts)
                variable_watts += float(self.link_dynamic_watts) * util
            else:
                inactive_links += 1
                fixed_watts += float(self.link_sleep_watts)
            link_watts += self.estimate_link_watts(active=is_active, utilization=util)

        for node_id in graph.nodes():
            active_incident_utils = [
                _clip_unit(data.get("utilization", 0.0))
                for data in self._incident_edge_data(graph, node_id)
                if bool(data.get("active", True))
            ]
            if active_incident_utils:
                active_devices += 1
                mean_util = float(sum(active_incident_utils) / len(active_incident_utils))
                fixed_watts += float(self.device_active_watts)
                variable_watts += float(self.device_dynamic_watts) * mean_util
                device_watts += self.estimate_device_watts(
                    active_links=len(active_incident_utils),
                    mean_utilization=mean_util,
                )
            else:
                inactive_devices += 1
                fixed_watts += float(self.device_sleep_watts)
                device_watts += self.estimate_device_watts(active_links=0, mean_utilization=0.0)

        total_watts = float(fixed_watts + variable_watts)
        return PowerSnapshot(
            total_watts=total_watts,
            fixed_watts=float(fixed_watts),
            variable_watts=float(variable_watts),
            device_watts=float(device_watts),
            link_watts=float(link_watts),
            active_devices=int(active_devices),
            inactive_devices=int(inactive_devices),
            active_links=int(active_links),
            inactive_links=int(inactive_links),
        )

    def _incident_edge_data(self, graph: "nx.Graph", node_id: int) -> Iterator[Dict]:
        if graph.is_directed():
            seen: set[tuple[int, int]] = set()
            for u, v, data in graph.in_edges(node_id, data=True):
                key = (int(u), int(v))
                if key in seen:
                    continue
                seen.add(key)
                yield data
            for u, v, data in graph.out_edges(node_id, data=True):
                key = (int(u), int(v))
                if key in seen:
                    continue
                seen.add(key)
                yield data
            return

        for _u, _v, data in graph.edges(node_id, data=True):
            yield data


@dataclass
class PowerController:
    """Track per-device power using the same simple active/sleep assumptions."""

    model: PowerModel

    def snapshot(
        self,
        active_links_by_node: Dict[int, int],
        utilization_by_node: Dict[int, float] | None = None,
    ) -> Dict[int, float]:
        utilization_by_node = utilization_by_node or {}
        return {
            node_id: self.model.estimate(
                active_links=active_links,
                mean_utilization=float(utilization_by_node.get(node_id, 0.0)),
            )
            for node_id, active_links in active_links_by_node.items()
        }
