"""Build and load NetworkX graph topologies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - optional dependency
    nx = None
    _nx_import_error = exc
else:
    _nx_import_error = None

# networkx is an optional dependency. We avoid referencing the runtime `nx` variable in
# type annotations because Pylance flags it as an invalid type form.
GraphT = Any


@dataclass
class TopologyConfig:
    """Configuration for generating a topology graph."""

    node_count: int = 10
    edge_prob: float = 0.2
    directed: bool = False
    seed: int | None = None


def build_random_topology(config: TopologyConfig) -> GraphT:
    """Create a random topology based on an Erdos-Renyi model."""
    if nx is None:
        raise ImportError("networkx is required to build a topology") from _nx_import_error

    graph = nx.erdos_renyi_graph(
        n=config.node_count,
        p=config.edge_prob,
        seed=config.seed,
        directed=bool(config.directed),
    )

    # Force plain Graph/DiGraph (keeps things predictable)
    graph = nx.DiGraph(graph) if bool(config.directed) else nx.Graph(graph)
    return graph


def load_topology_from_edges(
    edges: Iterable[tuple[int, int]], *, directed: bool = False
) -> GraphT:
    """Load a topology graph from an edge list."""
    if nx is None:
        raise ImportError("networkx is required to load a topology") from _nx_import_error

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    return graph
