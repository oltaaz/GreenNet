from __future__ import annotations

import networkx as nx
import pytest

from greennet.topology import (
    TopologyConfig,
    TopologyValidationError,
    build_random_topology,
    build_topology,
    list_named_topologies,
    load_topology_from_dict,
    load_topology_from_edges,
)


def test_build_random_topology_is_deterministic_for_a_seed() -> None:
    config = TopologyConfig(node_count=6, edge_prob=0.4, directed=False, seed=13)

    graph_a = build_random_topology(config)
    graph_b = build_random_topology(config)

    assert isinstance(graph_a, nx.Graph)
    assert graph_a.number_of_nodes() == 6
    assert sorted(graph_a.edges()) == sorted(graph_b.edges())


def test_build_random_topology_returns_directed_graph_when_requested() -> None:
    graph = build_random_topology(TopologyConfig(node_count=5, edge_prob=0.5, directed=True, seed=3))

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 5


def test_load_topology_from_edges_preserves_edges_and_graph_type() -> None:
    undirected = load_topology_from_edges([(0, 1), (1, 2), (2, 0)])
    directed = load_topology_from_edges([(0, 1), (1, 0)], directed=True)

    assert isinstance(undirected, nx.Graph)
    assert sorted(undirected.edges()) == [(0, 1), (0, 2), (1, 2)]

    assert isinstance(directed, nx.DiGraph)
    assert sorted(directed.edges()) == [(0, 1), (1, 0)]


def test_build_topology_loads_packaged_named_topology() -> None:
    graph = build_topology(TopologyConfig(topology_name="metro_hub"))

    assert "metro_hub" in list_named_topologies()
    assert graph.number_of_nodes() == 8
    assert graph.edges[1, 4]["capacity"] == pytest.approx(22.0)
    assert graph.edges[0, 6]["latency_ms"] == pytest.approx(8.0)


def test_load_topology_from_dict_rejects_non_contiguous_node_ids() -> None:
    payload = {
        "format_version": 1,
        "directed": False,
        "nodes": [0, 2],
        "edges": [{"source": 0, "target": 2}],
    }

    with pytest.raises(TopologyValidationError, match="contiguous integer node IDs"):
        load_topology_from_dict(payload)
