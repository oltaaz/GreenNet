"""Core network simulator with simple capacity and utilization tracking."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

# Type-checking-only import so annotations like "nx.Graph" don't depend on a runtime variable.
if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

try:
    import networkx as nx_runtime
except ImportError as exc:  # pragma: no cover - optional dependency
    nx_runtime = None
    _nx_import_error = exc
else:
    _nx_import_error = None

from greennet.routing import RouteSplit, ShortestPathPolicy
from greennet.power import PowerSnapshot
from greennet.topology import TopologyConfig, build_random_topology

EdgeKey = Tuple[int, int]


def _safe_float(value, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if x == x and x not in (float("inf"), float("-inf")) else default


@dataclass(frozen=True)
class Flow:
    """Traffic demand between two nodes."""

    source: int
    destination: int
    demand: float


@dataclass
class StepMetrics:
    """Aggregated metrics from a simulation step."""

    delivered: float
    dropped: float
    avg_utilization: float
    avg_delay: float
    avg_delay_ms: float = 0.0
    avg_path_latency_ms: float = 0.0
    energy_kwh: float = 0.0
    carbon_g: float = 0.0
    power_total_watts: float = 0.0
    power_fixed_watts: float = 0.0
    power_variable_watts: float = 0.0
    power_device_watts: float = 0.0
    power_link_watts: float = 0.0
    active_devices: int = 0
    inactive_devices: int = 0
    active_links: int = 0
    inactive_links: int = 0


class Simulator:
    """Simple simulator core with per-edge capacity constraints."""

    def __init__(
        self,
        graph: "nx.Graph",
        routing_policy: Callable[["nx.Graph", int, int], object],
        *,
        dt_seconds: float = 1.0,
        default_capacity: float = 10.0,
        default_latency_ms: float = 10.0,
        congestion_alpha: float = 1.0,
        congestion_eps: float = 1e-6,
        congestion_mult_cap: float = 1000.0,
        power_model_watts: Optional[Callable[["nx.Graph"], float | PowerSnapshot]] = None,
        carbon_intensity_g_per_kwh: Optional[Callable[[float], float]] = None,
    ) -> None:
        if nx_runtime is None:
            raise ImportError("networkx is required for Simulator") from _nx_import_error
        self.graph = graph
        self.routing_policy = routing_policy
        self.dt_seconds = float(dt_seconds)
        self.default_capacity = float(default_capacity)
        self.default_latency_ms = float(default_latency_ms)
        self.congestion_alpha = float(congestion_alpha)
        self.congestion_eps = float(congestion_eps)
        self.congestion_mult_cap = float(congestion_mult_cap)
        self.power_model_watts = power_model_watts
        self.carbon_intensity_g_per_kwh = carbon_intensity_g_per_kwh
        self.t = 0.0
        self.active: Dict[EdgeKey, bool] = {}
        self.capacity: Dict[EdgeKey, float] = {}
        self.utilization: Dict[EdgeKey, float] = {}
        self.reset()

    def reset(self) -> None:
        """Reset time and per-edge state."""
        self.t = 0.0
        self.active.clear()
        self.capacity.clear()
        self.utilization.clear()
        for u, v, data in self.graph.edges(data=True):
            key = self._edge_key(u, v)
            self.active[key] = bool(data.get("active", True))
            self.capacity[key] = _safe_float(data.get("capacity", self.default_capacity), self.default_capacity)
            self.utilization[key] = 0.0
            # Keep graph attributes in sync (useful for routing policies and dashboards).
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v]["active"] = self.active[key]
                self.graph.edges[u, v]["capacity"] = self.capacity[key]
                self.graph.edges[u, v]["utilization"] = 0.0
                # Ensure latency is present and numeric (safe for routing/delay calculations).
                self.graph.edges[u, v]["latency_ms"] = _safe_float(
                    self.graph.edges[u, v].get("latency_ms", self.default_latency_ms),
                    self.default_latency_ms,
                )

    def step(self, flows: Sequence[Flow | Tuple[int, int, float]]) -> StepMetrics:
        """Advance the simulation by one step."""
        normalized_flows = [self._normalize_flow(flow) for flow in flows]
        routing_graph = self._active_routing_graph()
        desired_by_edge: Dict[EdgeKey, float] = {}
        path_allocations: List[Tuple[List[EdgeKey], float, float]] = []  # (edges, desired, base_latency_ms)
        dropped_unrouted = 0.0

        for flow in normalized_flows:
            paths, weights = self._resolve_paths(flow.source, flow.destination, routing_graph)
            if not paths:
                dropped_unrouted += flow.demand
                continue
            for path, weight in zip(paths, weights):
                desired = flow.demand * weight
                edge_keys = self._path_edges(path)
                if not edge_keys:
                    continue
                base_latency_ms = self._path_latency_ms(routing_graph, path)
                path_allocations.append((edge_keys, desired, base_latency_ms))
                for edge_key in edge_keys:
                    desired_by_edge[edge_key] = desired_by_edge.get(edge_key, 0.0) + desired

        edge_scale: Dict[EdgeKey, float] = {}
        for edge_key, desired in desired_by_edge.items():
            if not self.active.get(edge_key, True):
                edge_scale[edge_key] = 0.0
                continue
            capacity = self.capacity.get(edge_key, 0.0)
            if capacity <= 0.0:
                edge_scale[edge_key] = 0.0
                continue
            if desired <= 0.0:
                edge_scale[edge_key] = 1.0
                continue
            edge_scale[edge_key] = min(1.0, capacity / desired)

        delivered = 0.0
        dropped = dropped_unrouted
        actual_by_edge: Dict[EdgeKey, float] = {}

        for edge_keys, desired, _ in path_allocations:
            scale = min(edge_scale.get(edge_key, 0.0) for edge_key in edge_keys)
            path_delivered = desired * scale
            delivered += path_delivered
            dropped += max(0.0, desired - path_delivered)
            for edge_key in edge_keys:
                actual_by_edge[edge_key] = actual_by_edge.get(edge_key, 0.0) + path_delivered

        utilizations: List[float] = []
        for edge_key, capacity in self.capacity.items():
            is_active = self.active.get(edge_key, True) and capacity > 0.0
            if not is_active:
                util = 0.0
            else:
                usage = actual_by_edge.get(edge_key, 0.0)
                util = min(1.0, usage / capacity)
                utilizations.append(util)

            self.utilization[edge_key] = util
            u, v = edge_key
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v]["utilization"] = util
                self.graph.edges[u, v]["active"] = bool(self.active.get(edge_key, True))
                self.graph.edges[u, v]["capacity"] = float(self.capacity.get(edge_key, capacity))

        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.0

        # Traffic-weighted average delay (ms): base path latency + congestion penalty.
        total_delivered = max(delivered, 0.0)
        delay_weighted_sum_ms = 0.0
        base_latency_weighted_sum_ms = 0.0

        if total_delivered > 0.0:
            util_by_edge = dict(self.utilization)

            def edge_delay_multiplier(util: float) -> float:
                # multiplier = 1 + alpha * util / (1 - util + eps)
                u = min(max(util, 0.0), 0.999999)
                mult = 1.0 + self.congestion_alpha * (u / (1.0 - u + self.congestion_eps))
                return min(mult, self.congestion_mult_cap)

            for edge_keys, desired, base_latency_ms in path_allocations:
                scale = min(edge_scale.get(edge_key, 0.0) for edge_key in edge_keys)
                path_delivered = desired * scale
                if path_delivered <= 0.0:
                    continue

                # Additive per-edge delay: sum(latency_ms(edge) * multiplier(util(edge))).
                path_delay_ms = 0.0
                for edge_key in edge_keys:
                    a, b = edge_key
                    latency = self.default_latency_ms
                    if routing_graph.has_edge(a, b):
                        latency = _safe_float(
                            routing_graph.edges[a, b].get("latency_ms", self.default_latency_ms),
                            self.default_latency_ms,
                        )
                    util = util_by_edge.get(edge_key, 0.0)
                    path_delay_ms += latency * edge_delay_multiplier(util)
                delay_weighted_sum_ms += path_delay_ms * path_delivered
                base_latency_weighted_sum_ms += base_latency_ms * path_delivered

        avg_delay_ms = (delay_weighted_sum_ms / total_delivered) if total_delivered > 0.0 else 0.0
        avg_path_latency_ms = (base_latency_weighted_sum_ms / total_delivered) if total_delivered > 0.0 else 0.0

        # Keep original field for backward compatibility (seconds).
        avg_delay = avg_delay_ms / 1000.0

        energy_kwh = 0.0
        carbon_g = 0.0
        power_total_watts = 0.0
        power_fixed_watts = 0.0
        power_variable_watts = 0.0
        power_device_watts = 0.0
        power_link_watts = 0.0
        active_devices = 0
        inactive_devices = 0
        active_links = 0
        inactive_links = 0
        if self.power_model_watts is not None:
            power_result = self.power_model_watts(self.graph)
            if isinstance(power_result, PowerSnapshot):
                power_total_watts = _safe_float(power_result.total_watts, 0.0)
                power_fixed_watts = _safe_float(power_result.fixed_watts, 0.0)
                power_variable_watts = _safe_float(power_result.variable_watts, 0.0)
                power_device_watts = _safe_float(power_result.device_watts, 0.0)
                power_link_watts = _safe_float(power_result.link_watts, 0.0)
                active_devices = max(0, int(power_result.active_devices))
                inactive_devices = max(0, int(power_result.inactive_devices))
                active_links = max(0, int(power_result.active_links))
                inactive_links = max(0, int(power_result.inactive_links))
            else:
                power_total_watts = _safe_float(power_result, 0.0)
            energy_kwh = max(0.0, power_total_watts) * (self.dt_seconds / 3600.0) / 1000.0
            if self.carbon_intensity_g_per_kwh is not None:
                intensity = _safe_float(self.carbon_intensity_g_per_kwh(self.t), 0.0)
                carbon_g = max(0.0, energy_kwh * intensity)

        self.t += self.dt_seconds
        return StepMetrics(
            delivered=delivered,
            dropped=dropped,
            avg_utilization=avg_utilization,
            avg_delay=avg_delay,
            avg_delay_ms=avg_delay_ms,
            avg_path_latency_ms=avg_path_latency_ms,
            energy_kwh=energy_kwh,
            carbon_g=carbon_g,
            power_total_watts=power_total_watts,
            power_fixed_watts=power_fixed_watts,
            power_variable_watts=power_variable_watts,
            power_device_watts=power_device_watts,
            power_link_watts=power_link_watts,
            active_devices=active_devices,
            inactive_devices=inactive_devices,
            active_links=active_links,
            inactive_links=inactive_links,
        )

    def _active_routing_graph(self) -> "nx.Graph":
        """Return a graph containing only active edges with positive capacity.

        Routing policies should use this graph so that paths never traverse sleeping
        or zero-capacity links.
        """
        if self.graph.is_directed():
            H = nx_runtime.DiGraph()
        else:
            H = nx_runtime.Graph()
        H.add_nodes_from(self.graph.nodes(data=True))
        for u, v, data in self.graph.edges(data=True):
            key = self._edge_key(u, v)

            # Respect current active state (fall back to graph attribute if not tracked yet).
            if not self.active.get(key, bool(data.get("active", True))):
                continue

            # Use tracked capacity when available; otherwise fall back to graph attribute/default.
            cap = _safe_float(self.capacity.get(key, data.get("capacity", self.default_capacity)), self.default_capacity)
            if cap <= 0.0:
                continue

            # Copy edge attributes so weights/latency/etc. remain available for routing.
            new_data = dict(data)
            new_data["active"] = True
            new_data["capacity"] = cap

            # Ensure commonly-used numeric attributes are numeric (prevents NetworkX weight errors).
            if "weight" in new_data:
                new_data["weight"] = _safe_float(new_data.get("weight"), 1.0)
            if "latency_ms" in new_data:
                new_data["latency_ms"] = _safe_float(new_data.get("latency_ms"), self.default_latency_ms)

            H.add_edge(u, v, **new_data)
        return H

    def _path_latency_ms(self, graph: "nx.Graph", path: Sequence[int]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if graph.has_edge(u, v):
                data = graph.edges[u, v]
                total += _safe_float(data.get("latency_ms", self.default_latency_ms), self.default_latency_ms)
            else:
                total += self.default_latency_ms
        return total

    def _edge_key(self, u: int, v: int) -> EdgeKey:
        if self.graph.is_directed():
            return (u, v)
        return (u, v) if u <= v else (v, u)

    def _path_edges(self, path: Sequence[int]) -> List[EdgeKey]:
        edge_keys: List[EdgeKey] = []
        for idx in range(len(path) - 1):
            edge_keys.append(self._edge_key(path[idx], path[idx + 1]))
        return edge_keys

    def _normalize_flow(self, flow: Flow | Tuple[int, int, float]) -> Flow:
        if isinstance(flow, Flow):
            return flow
        source, destination, demand = flow
        return Flow(source=source, destination=destination, demand=float(demand))

    def _resolve_paths(
        self, source: int, destination: int, routing_graph: "nx.Graph"
    ) -> Tuple[List[List[int]], List[float]]:
        result = self.routing_policy(routing_graph, source, destination)
        if result is None:
            return [], []
        if isinstance(result, RouteSplit):
            paths = [list(path) for path in result.paths]
            weights = list(result.weights)
        elif isinstance(result, tuple) and len(result) == 2:
            paths = [list(path) for path in result[0]]
            weights = list(result[1])
        else:
            # Assume iterable of paths; if not, treat as no route.
            try:
                paths = [list(path) for path in result]
            except TypeError:
                return [], []
            weights = []

        if not paths:
            return [], []

        if not weights:
            weights = [1.0 / len(paths) for _ in paths]

        # Clamp bad weights and align length to paths.
        weights = [max(0.0, _safe_float(w, 0.0)) for w in weights]
        if len(weights) != len(paths):
            weights = [1.0 / len(paths) for _ in paths]

        total = sum(weights)
        if total > 0.0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(paths) for _ in paths]
        return paths, weights


def _smoke_run(steps: int, seed: int, output_path: Path | None) -> None:
    if nx_runtime is None:
        raise ImportError("networkx is required for simulator smoke run") from _nx_import_error

    rng = random.Random(seed)
    graph = build_random_topology(TopologyConfig(node_count=12, edge_prob=0.25, directed=False))
    for u, v in graph.edges():
        graph.edges[u, v]["weight"] = 1.0
        graph.edges[u, v]["capacity"] = 10.0
        graph.edges[u, v]["latency_ms"] = 5.0 + rng.random() * 5.0

    simulator = Simulator(graph, routing_policy=ShortestPathPolicy())
    metrics_log: List[Dict[str, float]] = []

    for _ in range(steps):
        flows: List[Flow] = []
        for _ in range(rng.randint(2, 5)):
            source = rng.randrange(0, graph.number_of_nodes())
            destination = rng.randrange(0, graph.number_of_nodes())
            while destination == source:
                destination = rng.randrange(0, graph.number_of_nodes())
            demand = 1.0 + rng.random() * 4.0
            flows.append(Flow(source=source, destination=destination, demand=demand))

        metrics = simulator.step(flows)
        row = {
            "delivered": metrics.delivered,
            "dropped": metrics.dropped,
            "avg_delay_ms": metrics.avg_delay_ms,
            "avg_utilization": metrics.avg_utilization,
        }
        metrics_log.append(row)
        print(
            f"delivered={row['delivered']:.2f} "
            f"dropped={row['dropped']:.2f} "
            f"avg_delay_ms={row['avg_delay_ms']:.2f} "
            f"avg_utilization={row['avg_utilization']:.3f}"
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_log, handle, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GreenNet simulator smoke run.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = args.output
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = Path("runs") / f"smoke_metrics_{timestamp}.json"
    _smoke_run(steps=args.steps, seed=args.seed, output_path=output)
