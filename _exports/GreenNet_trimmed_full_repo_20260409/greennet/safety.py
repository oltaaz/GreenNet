"""Connectivity repair helpers."""
from __future__ import annotations

from typing import Iterable, Tuple

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - optional dependency
    nx = None
    _nx_import_error = exc
else:
    _nx_import_error = None


def ensure_connectivity(graph: "nx.Graph") -> Iterable[Tuple[int, int]]:
    """Add edges to make the graph connected; returns added edges."""
    if nx is None:
        raise ImportError("networkx is required for safety checks") from _nx_import_error

    added_edges = []
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        return added_edges

    for left, right in zip(components, components[1:]):
        u = next(iter(left))
        v = next(iter(right))
        graph.add_edge(u, v)
        added_edges.append((u, v))

    return added_edges
