"""Build and load NetworkX graph topologies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - optional dependency
    nx = None
    _nx_import_error = exc
else:
    _nx_import_error = None


@dataclass
class TopologyConfig:
    """Configuration for generating a topology graph."""

    node_count: int = 10
    edge_prob: float = 0.2
    directed: bool = False


def build_random_topology(config: TopologyConfig) -> "nx.Graph":
    """Create a random topology based on an Erdos-Renyi model."""
    if nx is None:
        raise ImportError("networkx is required to build a topology") from _nx_import_error

    graph_cls = nx.DiGraph if config.directed else nx.Graph
    graph = nx.erdos_renyi_graph(config.node_count, config.edge_prob, create_using=graph_cls)
    return graph


def load_topology_from_edges(edges: Iterable[tuple[int, int]], *, directed: bool = False
                              ) -> "nx.Graph":
    """Load a topology graph from an edge list."""
    if nx is None:
        raise ImportError("networkx is required to load a topology") from _nx_import_error

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    return graph
