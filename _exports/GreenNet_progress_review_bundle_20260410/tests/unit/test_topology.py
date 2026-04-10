from __future__ import annotations

import json

import networkx as nx
import pytest

from greennet.topology import (
    TopologyConfig,
    TopologyValidationError,
    build_random_topology,
    build_topology,
    list_official_topology_classes,
    list_named_topologies,
    load_named_topology,
    load_topology_from_dict,
    load_topology_from_edges,
    load_topology_from_file,
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
    graph = build_topology(TopologyConfig(topology_name="medium"))

    assert "metro_hub" in list_named_topologies()
    assert "small" in list_named_topologies()
    assert "medium" in list_named_topologies()
    assert "large" in list_named_topologies()
    assert list_official_topology_classes() == ["small", "medium", "large"]
    assert graph.number_of_nodes() == 8
    assert graph.edges[1, 4]["capacity"] == pytest.approx(22.0)
    assert graph.edges[0, 6]["latency_ms"] == pytest.approx(8.0)
    assert graph.graph["topology_name"] == "medium"
    assert graph.graph["topology_class"] == "medium"


def test_load_named_topology_supports_legacy_aliases() -> None:
    medium_graph = load_named_topology("medium")
    legacy_graph = load_named_topology("metro_hub")

    assert sorted(medium_graph.edges(data="capacity")) == sorted(legacy_graph.edges(data="capacity"))
    assert legacy_graph.graph["topology_requested_name"] == "metro_hub"
    assert legacy_graph.graph["topology_name"] == "medium"


def test_build_topology_loads_large_named_topology() -> None:
    graph = build_topology(TopologyConfig(topology_name="large"))

    assert graph.number_of_nodes() == 12
    assert graph.number_of_edges() == 16
    assert graph.edges[0, 4]["capacity"] == pytest.approx(28.0)


def test_build_topology_prefers_file_path_over_named_topology(tmp_path) -> None:
    topology_path = tmp_path / "override_topology.json"
    topology_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "directed": False,
                "nodes": [0, 1, 2, 3],
                "edges": [
                    {"source": 0, "target": 1},
                    {"source": 1, "target": 2},
                    {"source": 2, "target": 3},
                ],
            }
        ),
        encoding="utf-8",
    )

    graph = build_topology(TopologyConfig(topology_name="large", topology_path=str(topology_path)))

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3


def test_load_named_topology_rejects_unknown_name() -> None:
    with pytest.raises(TopologyValidationError, match="Official topology classes: small, medium, large"):
        load_named_topology("does_not_exist")


def test_load_topology_from_dict_rejects_non_contiguous_node_ids() -> None:
    payload = {
        "format_version": 1,
        "directed": False,
        "nodes": [0, 2],
        "edges": [{"source": 0, "target": 2}],
    }

    with pytest.raises(TopologyValidationError, match="contiguous integer node IDs"):
        load_topology_from_dict(payload)


def test_load_topology_from_file_rejects_disconnected_graph(tmp_path) -> None:
    topology_path = tmp_path / "disconnected.json"
    topology_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "directed": False,
                "nodes": [0, 1, 2, 3],
                "edges": [
                    {"source": 0, "target": 1},
                    {"source": 2, "target": 3},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TopologyValidationError, match="must be connected"):
        load_topology_from_file(topology_path)


def test_load_topology_from_file_rejects_duplicate_edges(tmp_path) -> None:
    topology_path = tmp_path / "duplicate_edges.json"
    topology_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "directed": False,
                "nodes": [0, 1, 2],
                "edges": [
                    {"source": 0, "target": 1},
                    {"source": 1, "target": 0},
                    {"source": 1, "target": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TopologyValidationError, match="duplicate edge"):
        load_topology_from_file(topology_path)
