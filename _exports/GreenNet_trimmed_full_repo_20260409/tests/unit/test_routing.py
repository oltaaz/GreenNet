from __future__ import annotations

import networkx as nx
import pytest

from greennet.routing import (
    KShortestSoftminPolicy,
    OSPFLikePolicy,
    ShortestPathPolicy,
    annotate_routing_costs,
    build_routing_policy,
    equal_cost_shortest_paths,
    k_shortest_paths,
    path_cost,
    softmin_split,
    static_link_cost,
)


def test_path_cost_uses_lowest_parallel_edge_weight() -> None:
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, weight=9.0)
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(1, 2, weight=3.0)

    assert path_cost(graph, [0, 1, 2]) == pytest.approx(5.0)


def test_k_shortest_paths_orders_by_total_weight() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 3, weight=1.0)
    graph.add_edge(0, 2, weight=2.0)
    graph.add_edge(2, 3, weight=2.0)
    graph.add_edge(0, 4, weight=4.0)
    graph.add_edge(4, 3, weight=4.0)

    paths = k_shortest_paths(graph, 0, 3, k=3, weight="weight")

    assert paths == [[0, 1, 3], [0, 2, 3], [0, 4, 3]]


def test_softmin_split_picks_best_path_at_zero_temperature() -> None:
    weights = softmin_split([5.0, 1.0, 3.0], temperature=0.0)

    assert weights == [0.0, 1.0, 0.0]


def test_shortest_path_policy_and_softmin_policy_return_normalized_splits() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 3, weight=1.0)
    graph.add_edge(0, 2, weight=4.0)
    graph.add_edge(2, 3, weight=4.0)

    shortest = ShortestPathPolicy(weight="weight")(graph, 0, 3)
    softmin = KShortestSoftminPolicy(k=2, temperature=1.0, weight="weight")(graph, 0, 3)

    assert list(shortest.paths[0]) == [0, 1, 3]
    assert shortest.weights == [1.0]

    assert len(softmin.paths) == 2
    assert sum(softmin.weights) == pytest.approx(1.0)
    assert softmin.weights[0] > softmin.weights[1]


def test_equal_cost_shortest_paths_and_ospf_like_policy_use_ecmp() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, routing_cost=1.0)
    graph.add_edge(1, 3, routing_cost=1.0)
    graph.add_edge(0, 2, routing_cost=1.0)
    graph.add_edge(2, 3, routing_cost=1.0)

    paths = equal_cost_shortest_paths(graph, 0, 3, weight="routing_cost", max_paths=8)
    split = OSPFLikePolicy(weight="routing_cost", max_paths=8)(graph, 0, 3)

    assert paths == [[0, 1, 3], [0, 2, 3]]
    assert split.paths == paths
    assert split.weights == pytest.approx([0.5, 0.5])


def test_static_link_cost_inverse_capacity_and_annotation_are_deterministic() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, capacity=10.0, latency_ms=7.0)
    graph.add_edge(1, 2, capacity=25.0, latency_ms=3.0)

    annotate_routing_costs(
        graph,
        model="inverse_capacity",
        cost_attr="routing_cost",
        mirror_weight_attr="weight",
        reference_bandwidth=100.0,
    )

    assert static_link_cost({"capacity": 10.0}, model="inverse_capacity", reference_bandwidth=100.0) == pytest.approx(10.0)
    assert graph.edges[0, 1]["routing_cost"] == pytest.approx(10.0)
    assert graph.edges[0, 1]["weight"] == pytest.approx(10.0)
    assert graph.edges[1, 2]["routing_cost"] == pytest.approx(4.0)


def test_build_routing_policy_normalizes_aliases() -> None:
    _policy, meta = build_routing_policy("ospf", metric_attr="routing_cost", ecmp_max_paths=4)

    assert meta["routing_baseline"] == "ospf_ecmp"
    assert meta["routing_path_split"] == "ecmp"
    assert meta["routing_ecmp_max_paths"] == 4
