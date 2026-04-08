"""Routing utilities for static shortest-path, ECMP, and softmin baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - optional dependency
    nx = None
    _nx_import_error = exc
else:
    _nx_import_error = None


@dataclass
class RouteSplit:
    """Represents how traffic is split across candidate paths."""

    paths: Sequence[Sequence[int]]
    weights: Sequence[float]


DEFAULT_ROUTING_COST_ATTR = "routing_cost"
DEFAULT_WEIGHT_ATTR = "weight"

ROUTING_BASELINE_ALIASES = {
    "min_hop_single_path": "min_hop_single_path",
    "min_hop": "min_hop_single_path",
    "shortest": "min_hop_single_path",
    "shortest_path": "min_hop_single_path",
    "single_shortest_path": "min_hop_single_path",
    "spf_single": "min_hop_single_path",
    "ospf_ecmp": "ospf_ecmp",
    "ospf": "ospf_ecmp",
    "ospf_like": "ospf_ecmp",
    "link_state_ecmp": "ospf_ecmp",
    "ecmp": "ospf_ecmp",
    "k_shortest_softmin": "k_shortest_softmin",
    "softmin": "k_shortest_softmin",
    "ksp_softmin": "k_shortest_softmin",
}

ROUTING_COST_MODEL_ALIASES = {
    "unit": "unit",
    "hop": "unit",
    "hop_count": "unit",
    "uniform": "unit",
    "latency": "latency",
    "latency_ms": "latency",
    "delay": "latency",
    "inverse_capacity": "inverse_capacity",
    "inv_capacity": "inverse_capacity",
    "capacity_inverse": "inverse_capacity",
}


def _safe_float(value, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _normalize_route_split(paths: Sequence[Sequence[int]], weights: Sequence[float]) -> RouteSplit:
    if len(paths) != len(weights):
        raise ValueError(f"RouteSplit mismatch: {len(paths)} paths vs {len(weights)} weights")

    if not paths:
        return RouteSplit(paths=[], weights=[])

    cleaned: List[float] = []
    for w in weights:
        wf = _safe_float(w, 0.0)
        cleaned.append(wf if wf > 0.0 else 0.0)

    s = sum(cleaned)
    if s <= 0.0:
        # fallback: uniform split
        uni = 1.0 / len(paths)
        return RouteSplit(paths=list(paths), weights=[uni] * len(paths))

    return RouteSplit(paths=list(paths), weights=[w / s for w in cleaned])


def canonicalize_routing_baseline_name(name: str | None) -> str:
    key = str(name or "min_hop_single_path").strip().lower()
    canonical = ROUTING_BASELINE_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            "Unknown routing baseline: "
            f"{name!r}. Expected one of {sorted(set(ROUTING_BASELINE_ALIASES.values()))} "
            f"or aliases {sorted(ROUTING_BASELINE_ALIASES)}."
        )
    return canonical


def canonicalize_routing_link_cost_model(name: str | None) -> str:
    key = str(name or "unit").strip().lower()
    canonical = ROUTING_COST_MODEL_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            "Unknown routing link-cost model: "
            f"{name!r}. Expected one of {sorted(set(ROUTING_COST_MODEL_ALIASES.values()))} "
            f"or aliases {sorted(ROUTING_COST_MODEL_ALIASES)}."
        )
    return canonical


def static_link_cost(
    edge_data: Dict[str, object],
    *,
    model: str = "unit",
    default_capacity: float = 10.0,
    default_latency_ms: float = 10.0,
    reference_bandwidth: float = 100.0,
) -> float:
    """Return a static additive link cost for a traditional routing baseline.

    This stays intentionally simple. We model forwarding-time route selection,
    not OSPF control-plane dynamics such as hellos, LSAs, or timers.
    """
    cost_model = canonicalize_routing_link_cost_model(model)
    if cost_model == "unit":
        return 1.0
    if cost_model == "latency":
        latency_ms = _safe_float(edge_data.get("latency_ms", default_latency_ms), default_latency_ms)
        return max(1.0, latency_ms)

    capacity = _safe_float(edge_data.get("capacity", default_capacity), default_capacity)
    if capacity <= 0.0:
        return max(1.0, _safe_float(reference_bandwidth, 100.0))
    return max(1.0, float(math.ceil(_safe_float(reference_bandwidth, 100.0) / capacity)))


def annotate_routing_costs(
    graph: "nx.Graph",
    *,
    model: str = "unit",
    cost_attr: str = DEFAULT_ROUTING_COST_ATTR,
    mirror_weight_attr: str | None = DEFAULT_WEIGHT_ATTR,
    default_capacity: float = 10.0,
    default_latency_ms: float = 10.0,
    reference_bandwidth: float = 100.0,
) -> None:
    """Populate a static routing-cost attribute on every edge.

    By default we also mirror the cost onto `weight` for backward compatibility
    with older code paths and tests that still reference that attribute directly.
    """
    if nx is None:
        raise ImportError("networkx is required for routing") from _nx_import_error

    for _u, _v, data in graph.edges(data=True):
        cost = static_link_cost(
            data,
            model=model,
            default_capacity=default_capacity,
            default_latency_ms=default_latency_ms,
            reference_bandwidth=reference_bandwidth,
        )
        data[cost_attr] = float(cost)
        if mirror_weight_attr:
            data[mirror_weight_attr] = float(cost)


def _as_simple_weighted_graph(graph: "nx.Graph", weight: Optional[str] = "weight") -> "nx.Graph":
    """Return a simple Graph/DiGraph view with a single edge per (u, v).

    If `graph` is a MultiGraph/MultiDiGraph, we collapse parallel edges by keeping the
    minimum weight edge (OSPF-like behavior for shortest paths).
    """
    if nx is None:
        raise ImportError("networkx is required for routing") from _nx_import_error

    is_multi = getattr(graph, "is_multigraph", lambda: False)()
    if not is_multi:
        return graph

    H = nx.DiGraph() if graph.is_directed() else nx.Graph()
    H.add_nodes_from(graph.nodes(data=True))

    for u, v, data in graph.edges(data=True):
        w = _safe_float(data.get(weight, 1.0) if weight else 1.0, 1.0)
        new_data = dict(data)
        if weight:
            new_data[weight] = w  # ensure stored weight is numeric

        if H.has_edge(u, v):
            prev = _safe_float(H.edges[u, v].get(weight, 1.0) if weight else 1.0, 1.0)
            if w < prev:
                H.edges[u, v].update(new_data)
                if weight:
                    H.edges[u, v][weight] = w  # enforce numeric weight after update
        else:
            H.add_edge(u, v, **new_data)

    return H


def path_cost(
    graph: "nx.Graph",
    path: Sequence[int],
    weight: Optional[str] = "weight",
    default_weight: float = 1.0,
) -> float:
    """Compute the sum of edge weights along `path`.

    Works for Graph/DiGraph and MultiGraph (parallel edges are treated by taking
    the minimum edge weight for each hop).
    """
    if nx is None:
        raise ImportError("networkx is required for routing") from _nx_import_error
    if len(path) < 2:
        return 0.0

    G = _as_simple_weighted_graph(graph, weight=weight)

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.edges[u, v]
        total += _safe_float(data.get(weight, default_weight) if weight else default_weight, default_weight)
    return total


def k_shortest_paths(
    graph: "nx.Graph",
    source: int,
    target: int,
    k: int,
    weight: Optional[str] = "weight",
) -> List[List[int]]:
    """Return up to k shortest simple paths from source to target.

    If `weight` is provided, paths are ordered by total weight (OSPF-like).
    For MultiGraph, parallel edges are collapsed by keeping the minimum-weight edge.
    """
    if nx is None:
        raise ImportError("networkx is required for routing") from _nx_import_error

    G = _as_simple_weighted_graph(graph, weight=weight)

    paths: List[List[int]] = []
    try:
        # networkx.shortest_simple_paths yields paths ordered by (weighted) length
        for path in nx.shortest_simple_paths(G, source, target, weight=weight):
            paths.append(list(path))
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return paths


def equal_cost_shortest_paths(
    graph: "nx.Graph",
    source: int,
    target: int,
    *,
    weight: Optional[str] = "weight",
    max_paths: int | None = None,
) -> List[List[int]]:
    """Return equal-cost shortest paths for ECMP-style routing."""
    if nx is None:
        raise ImportError("networkx is required for routing") from _nx_import_error

    G = _as_simple_weighted_graph(graph, weight=weight)

    try:
        paths = [list(path) for path in nx.all_shortest_paths(G, source, target, weight=weight)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    paths.sort(key=lambda path: tuple(path))
    if max_paths is not None and max_paths > 0:
        return paths[: int(max_paths)]
    return paths


def softmin_split(costs: Iterable[float], temperature: float = 1.0) -> List[float]:
    """Compute softmin weights from path costs.

    Lower cost -> higher weight. Uses a numerically-stable softmax on -cost/temperature.
    """
    costs_list = list(costs)
    if not costs_list:
        return []
    finite_costs = [cost for cost in costs_list if math.isfinite(cost)]
    if not finite_costs:
        return [1.0 / len(costs_list) for _ in costs_list]
    max_finite = max(finite_costs)
    cleaned = [cost if math.isfinite(cost) else max_finite + 1.0 for cost in costs_list]

    if temperature <= 0:
        best = min(range(len(cleaned)), key=cleaned.__getitem__)
        return [1.0 if idx == best else 0.0 for idx in range(len(costs_list))]

    scaled = [-(float(cost) / float(temperature)) for cost in cleaned]
    max_scaled = max(scaled)
    exp_scores = [math.exp(score - max_scaled) for score in scaled]
    total = sum(exp_scores)
    if total <= 0.0:
        # Fallback to uniform split if something pathological happens
        return [1.0 / len(costs_list) for _ in costs_list]
    return [score / total for score in exp_scores]


def split_for_paths_by_weight(
    graph: "nx.Graph",
    paths: Sequence[Sequence[int]],
    weight: Optional[str] = "weight",
    temperature: float = 1.0,
) -> RouteSplit:
    """Compute RouteSplit for candidate paths using their weighted costs."""
    costs = [path_cost(graph, path, weight=weight) for path in paths]
    return split_for_paths(paths, costs, temperature=temperature)


def split_for_paths(paths: Sequence[Sequence[int]], costs: Iterable[float],
                    temperature: float = 1.0) -> RouteSplit:
    weights = softmin_split(costs, temperature=temperature)
    return _normalize_route_split(paths, weights)


class SinglePathShortestPolicy:
    """Route using a single weighted shortest path."""

    def __init__(self, *, weight: Optional[str] = "weight") -> None:
        self.weight = weight

    def __call__(self, graph: "nx.Graph", source: int, target: int) -> RouteSplit:
        paths = k_shortest_paths(graph, source, target, k=1, weight=self.weight)
        if not paths:
            return RouteSplit(paths=[], weights=[])
        return _normalize_route_split(paths, [1.0])


class ShortestPathPolicy(SinglePathShortestPolicy):
    """Backward-compatible alias for single-path shortest routing."""


class ECMPShortestPathPolicy:
    """Route using equal-cost shortest paths with a uniform split."""

    def __init__(self, *, max_paths: int = 8, weight: Optional[str] = "weight") -> None:
        self.max_paths = max(1, int(max_paths))
        self.weight = weight

    def __call__(self, graph: "nx.Graph", source: int, target: int) -> RouteSplit:
        paths = equal_cost_shortest_paths(
            graph,
            source,
            target,
            weight=self.weight,
            max_paths=self.max_paths,
        )
        if not paths:
            return RouteSplit(paths=[], weights=[])
        return _normalize_route_split(paths, [1.0] * len(paths))


class OSPFLikePolicy(ECMPShortestPathPolicy):
    """Static link-state SPF with ECMP-style equal-cost splitting.

    This models forwarding behavior after convergence. It does not simulate the
    OSPF protocol itself.
    """


class KShortestSoftminPolicy:
    """Route using k-shortest paths with a softmin split."""

    def __init__(self, k: int = 3, *, temperature: float = 1.0,
                 weight: Optional[str] = "weight") -> None:
        self.k = k
        self.temperature = temperature
        self.weight = weight

    def __call__(self, graph: "nx.Graph", source: int, target: int) -> RouteSplit:
        paths = k_shortest_paths(graph, source, target, k=self.k, weight=self.weight)
        if not paths:
            return RouteSplit(paths=[], weights=[])
        split = split_for_paths_by_weight(graph, paths, weight=self.weight, temperature=self.temperature)
        return _normalize_route_split(split.paths, split.weights)


def build_routing_policy(
    routing_baseline: str,
    *,
    metric_attr: str = DEFAULT_ROUTING_COST_ATTR,
    ecmp_max_paths: int = 8,
    softmin_k: int = 3,
    softmin_temperature: float = 1.0,
) -> Tuple[Callable[["nx.Graph", int, int], RouteSplit], Dict[str, object]]:
    """Build a routing policy and return stable metadata about it."""
    canonical = canonicalize_routing_baseline_name(routing_baseline)

    if canonical == "min_hop_single_path":
        return SinglePathShortestPolicy(weight=metric_attr), {
            "routing_baseline": canonical,
            "routing_forwarding_model": "single_shortest_path",
            "routing_path_split": "single_path",
            "routing_metric_attr": metric_attr,
        }

    if canonical == "ospf_ecmp":
        return OSPFLikePolicy(max_paths=ecmp_max_paths, weight=metric_attr), {
            "routing_baseline": canonical,
            "routing_forwarding_model": "static_link_state_spf",
            "routing_path_split": "ecmp",
            "routing_metric_attr": metric_attr,
            "routing_ecmp_max_paths": int(max(1, ecmp_max_paths)),
        }

    if canonical == "k_shortest_softmin":
        return KShortestSoftminPolicy(
            k=softmin_k,
            temperature=softmin_temperature,
            weight=metric_attr,
        ), {
            "routing_baseline": canonical,
            "routing_forwarding_model": "k_shortest_softmin",
            "routing_path_split": "softmin",
            "routing_metric_attr": metric_attr,
            "routing_k_paths": int(max(1, softmin_k)),
            "routing_softmin_temperature": float(softmin_temperature),
        }

    raise ValueError(f"Unhandled routing baseline: {routing_baseline!r}")


def _self_test() -> None:
    if nx is None:
        print("networkx not installed; skipping routing self-test")
        return

    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)

    # shortest path should be 0-1-3 after lexical tie-breaking
    sp = ShortestPathPolicy(weight="weight")(G, 0, 3)
    assert sp.paths and list(sp.paths[0]) == [0, 1, 3]
    assert abs(sum(sp.weights) - 1.0) < 1e-9

    ecmp = OSPFLikePolicy(weight="weight", max_paths=4)(G, 0, 3)
    assert len(ecmp.paths) == 2
    assert abs(sum(ecmp.weights) - 1.0) < 1e-9

    # softmin split should sum to 1
    pol = KShortestSoftminPolicy(k=2, temperature=1.0, weight="weight")(G, 0, 3)
    assert pol.paths and len(pol.weights) == len(pol.paths)
    assert all(w >= 0.0 for w in pol.weights)
    assert abs(sum(pol.weights) - 1.0) < 1e-9

    print("routing.py self-test OK")


if __name__ == "__main__":
    _self_test()
