"""Core network simulator with simple capacity and utilization tracking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - optional dependency
    nx = None
    _nx_import_error = exc
else:
    _nx_import_error = None

from greennet.routing import RouteSplit

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
        power_model_watts: Optional[Callable[["nx.Graph"], float]] = None,
        carbon_intensity_g_per_kwh: Optional[Callable[[float], float]] = None,
    ) -> None:
        if nx is None:
            raise ImportError("networkx is required for Simulator") from _nx_import_error
        self.graph = graph
        self.routing_policy = routing_policy
        self.dt_seconds = float(dt_seconds)
        self.default_capacity = float(default_capacity)
        self.default_latency_ms = float(default_latency_ms)
        self.congestion_alpha = float(congestion_alpha)
        self.congestion_eps = float(congestion_eps)
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
                return 1.0 + self.congestion_alpha * (u / (1.0 - u + self.congestion_eps))

            for edge_keys, desired, base_latency_ms in path_allocations:
                scale = min(edge_scale.get(edge_key, 0.0) for edge_key in edge_keys)
                path_delivered = desired * scale
                if path_delivered <= 0.0:
                    continue

                congestion_multiplier = 1.0
                for edge_key in edge_keys:
                    congestion_multiplier *= edge_delay_multiplier(util_by_edge.get(edge_key, 0.0))

                path_delay_ms = base_latency_ms * congestion_multiplier
                delay_weighted_sum_ms += path_delay_ms * path_delivered
                base_latency_weighted_sum_ms += base_latency_ms * path_delivered

        avg_delay_ms = (delay_weighted_sum_ms / total_delivered) if total_delivered > 0.0 else 0.0
        avg_path_latency_ms = (base_latency_weighted_sum_ms / total_delivered) if total_delivered > 0.0 else 0.0

        # Keep original field for backward compatibility (seconds).
        avg_delay = avg_delay_ms / 1000.0

        energy_kwh = 0.0
        carbon_g = 0.0
        if self.power_model_watts is not None:
            watts = _safe_float(self.power_model_watts(self.graph), 0.0)
            energy_kwh = max(0.0, watts) * (self.dt_seconds / 3600.0) / 1000.0
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
        )

    def _active_routing_graph(self) -> "nx.Graph":
        """Return a graph containing only active edges with positive capacity.

        Routing policies should use this graph so that paths never traverse sleeping
        or zero-capacity links.
        """
        if self.graph.is_directed():
            H = nx.DiGraph()
        else:
            H = nx.Graph()
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
