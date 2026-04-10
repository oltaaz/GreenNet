"""Lightweight network power model used by the simulator.

The model stays intentionally simple for thesis/demo use:

- Devices (nodes) draw a fixed active or sleep power level.
- Active devices and links add a small linear utilization-dependent term.
- Links and devices that are not active still retain a small sleep draw.
- Link or device state transitions can optionally add one-off wake/sleep energy.

This keeps the energy signal easy to explain while separating fixed power from
traffic-sensitive power in a defensible way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


POWER_MODEL_NAME = "active_sleep_linear"


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


def _require_non_negative(value: Any, *, field_name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative number") from exc
    if x != x or x in (float("inf"), float("-inf")) or x < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(x)


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
    - Transition costs are optional and modeled as one-off energy in joules per toggle.
    """

    network_fixed_watts: float = 20.0
    device_active_watts: float = 12.0
    device_sleep_watts: float = 2.0
    device_dynamic_watts: float = 3.0
    link_active_watts: float = 6.0
    link_sleep_watts: float = 0.2
    link_dynamic_watts: float = 2.0
    utilization_sensitive: bool = True
    transition_on_joules: float = 0.0
    transition_off_joules: float = 0.0

    @classmethod
    def from_env_config(cls, env_config: Any) -> "PowerModel":
        return cls(
            network_fixed_watts=float(getattr(env_config, "power_network_fixed_watts", 20.0)),
            device_active_watts=float(getattr(env_config, "power_device_active_watts", 12.0)),
            device_sleep_watts=float(getattr(env_config, "power_device_sleep_watts", 2.0)),
            device_dynamic_watts=float(getattr(env_config, "power_device_dynamic_watts", 3.0)),
            link_active_watts=float(getattr(env_config, "power_link_active_watts", 6.0)),
            link_sleep_watts=float(getattr(env_config, "power_link_sleep_watts", 0.2)),
            link_dynamic_watts=float(getattr(env_config, "power_link_dynamic_watts", 2.0)),
            utilization_sensitive=bool(getattr(env_config, "power_utilization_sensitive", True)),
            transition_on_joules=float(getattr(env_config, "power_transition_on_joules", 0.0)),
            transition_off_joules=float(getattr(env_config, "power_transition_off_joules", 0.0)),
        )

    def validate(self) -> "PowerModel":
        for field_name in (
            "network_fixed_watts",
            "device_active_watts",
            "device_sleep_watts",
            "device_dynamic_watts",
            "link_active_watts",
            "link_sleep_watts",
            "link_dynamic_watts",
            "transition_on_joules",
            "transition_off_joules",
        ):
            _require_non_negative(getattr(self, field_name), field_name=field_name)
        return self

    def metadata(self) -> Dict[str, Any]:
        return {
            "energy_model_name": POWER_MODEL_NAME,
            "power_utilization_sensitive": bool(self.utilization_sensitive),
            "power_transition_on_joules": float(self.transition_on_joules),
            "power_transition_off_joules": float(self.transition_off_joules),
            "power_model_signature": self.signature(),
        }

    def signature(self) -> str:
        util_flag = "1" if bool(self.utilization_sensitive) else "0"
        return (
            f"{POWER_MODEL_NAME}"
            f"|net={float(self.network_fixed_watts):.6g}"
            f"|dev={float(self.device_active_watts):.6g},{float(self.device_sleep_watts):.6g},{float(self.device_dynamic_watts):.6g}"
            f"|link={float(self.link_active_watts):.6g},{float(self.link_sleep_watts):.6g},{float(self.link_dynamic_watts):.6g}"
            f"|util={util_flag}"
            f"|transition_j={float(self.transition_on_joules):.6g},{float(self.transition_off_joules):.6g}"
        )

    def transition_energy_kwh(
        self,
        *,
        toggled_on_count: int = 0,
        toggled_off_count: int = 0,
    ) -> float:
        on_count = max(0, int(toggled_on_count))
        off_count = max(0, int(toggled_off_count))
        total_joules = (
            float(self.transition_on_joules) * float(on_count)
            + float(self.transition_off_joules) * float(off_count)
        )
        return max(0.0, total_joules / 3_600_000.0)

    def transition_watts_equivalent(
        self,
        *,
        toggled_on_count: int = 0,
        toggled_off_count: int = 0,
        dt_seconds: float = 1.0,
    ) -> float:
        if float(dt_seconds) <= 0.0:
            return 0.0
        energy_kwh = self.transition_energy_kwh(
            toggled_on_count=toggled_on_count,
            toggled_off_count=toggled_off_count,
        )
        return float(energy_kwh * 1000.0 * 3600.0 / float(dt_seconds))

    def _dynamic_multiplier(self) -> float:
        return 1.0 if bool(self.utilization_sensitive) else 0.0

    def estimate(self, active_links: int, mean_utilization: float = 0.0) -> float:
        """Backward-compatible single-device estimate in watts."""
        return self.estimate_device_watts(active_links=active_links, mean_utilization=mean_utilization)

    def estimate_device_watts(self, active_links: int, mean_utilization: float = 0.0) -> float:
        util = _clip_unit(mean_utilization)
        if int(active_links) > 0:
            return float(self.device_active_watts + (self.device_dynamic_watts * self._dynamic_multiplier() * util))
        return float(self.device_sleep_watts)

    def estimate_link_watts(self, *, active: bool, utilization: float = 0.0) -> float:
        if bool(active):
            util = _clip_unit(utilization)
            return float(self.link_active_watts + (self.link_dynamic_watts * self._dynamic_multiplier() * util))
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
                variable_watts += float(self.link_dynamic_watts) * self._dynamic_multiplier() * util
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
                variable_watts += float(self.device_dynamic_watts) * self._dynamic_multiplier() * mean_util
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
