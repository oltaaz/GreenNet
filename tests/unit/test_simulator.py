from __future__ import annotations

import networkx as nx
import pytest

from greennet.routing import ShortestPathPolicy
from greennet.simulator import Flow, Simulator


def test_simulator_respects_capacity_and_reports_drop() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, capacity=5.0, latency_ms=10.0, weight=1.0, active=True)
    graph.add_edge(1, 2, capacity=5.0, latency_ms=10.0, weight=1.0, active=True)

    simulator = Simulator(
        graph,
        routing_policy=ShortestPathPolicy(weight="weight"),
        default_capacity=5.0,
        default_latency_ms=10.0,
        congestion_alpha=0.0,
    )

    metrics = simulator.step([Flow(source=0, destination=2, demand=8.0)])

    assert metrics.delivered == pytest.approx(5.0)
    assert metrics.dropped == pytest.approx(3.0)
    assert metrics.avg_utilization == pytest.approx(1.0)
    assert metrics.avg_delay_ms == pytest.approx(20.0)
