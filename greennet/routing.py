"""Routing utilities: k-shortest paths and softmin splitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

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


def k_shortest_paths(graph: "nx.Graph", source: int, target: int, k: int
                     ) -> List[List[int]]:
    if nx is None:
        raise ImportError("networkx is required for routing") from _nx_import_error

    paths: List[List[int]] = []
    for path in nx.shortest_simple_paths(graph, source, target):
        paths.append(list(path))
        if len(paths) >= k:
            break
    return paths


def softmin_split(costs: Iterable[float], temperature: float = 1.0) -> List[float]:
    """Compute softmin weights from path costs."""
    costs_list = list(costs)
    if not costs_list:
        return []
    if temperature <= 0:
        best = min(range(len(costs_list)), key=costs_list.__getitem__)
        return [1.0 if idx == best else 0.0 for idx in range(len(costs_list))]

    scaled = [-(cost / temperature) for cost in costs_list]
    max_scaled = max(scaled)
    exp_scores = [pow(2.718281828459045, score - max_scaled) for score in scaled]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def split_for_paths(paths: Sequence[Sequence[int]], costs: Iterable[float],
                    temperature: float = 1.0) -> RouteSplit:
    weights = softmin_split(costs, temperature=temperature)
    return RouteSplit(paths=paths, weights=weights)
