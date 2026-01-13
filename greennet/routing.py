"""Routing utilities: k-shortest paths and softmin splitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional
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


class ShortestPathPolicy:
    """Route using a single shortest path (returned as a RouteSplit)."""

    def __init__(self, *, weight: Optional[str] = "weight") -> None:
        self.weight = weight

    def __call__(self, graph: "nx.Graph", source: int, target: int) -> RouteSplit:
        paths = k_shortest_paths(graph, source, target, k=1, weight=self.weight)
        if not paths:
            return RouteSplit(paths=[], weights=[])
        return _normalize_route_split(paths, [1.0])


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
    

def _self_test() -> None:
    if nx is None:
        print("networkx not installed; skipping routing self-test")
        return

    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(0, 2, weight=10.0)
    G.add_edge(2, 3, weight=10.0)

    # shortest path should be 0-1-3
    sp = ShortestPathPolicy(weight="weight")(G, 0, 3)
    assert sp.paths and list(sp.paths[0]) == [0, 1, 3]
    assert abs(sum(sp.weights) - 1.0) < 1e-9

    # softmin split should sum to 1
    pol = KShortestSoftminPolicy(k=2, temperature=1.0, weight="weight")(G, 0, 3)
    assert pol.paths and len(pol.weights) == len(pol.paths)
    assert all(w >= 0.0 for w in pol.weights)
    assert abs(sum(pol.weights) - 1.0) < 1e-9

    print("routing.py self-test OK")


if __name__ == "__main__":
    _self_test()