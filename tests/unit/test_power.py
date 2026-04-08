from __future__ import annotations

import networkx as nx
import pytest

from greennet.power import PowerModel
from greennet.routing import ShortestPathPolicy
from greennet.simulator import Flow, Simulator


def test_power_model_separates_fixed_and_variable_power() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, active=True, utilization=0.5)
    graph.add_edge(1, 2, active=False, utilization=0.0)

    model = PowerModel(
        network_fixed_watts=10.0,
        device_active_watts=5.0,
        device_sleep_watts=1.0,
        device_dynamic_watts=2.0,
        link_active_watts=3.0,
        link_sleep_watts=0.5,
        link_dynamic_watts=4.0,
    )

    snapshot = model.estimate_network(graph)

    assert snapshot.active_devices == 2
    assert snapshot.inactive_devices == 1
    assert snapshot.active_links == 1
    assert snapshot.inactive_links == 1
    assert snapshot.fixed_watts == pytest.approx(24.5)
    assert snapshot.variable_watts == pytest.approx(4.0)
    assert snapshot.device_watts == pytest.approx(13.0)
    assert snapshot.link_watts == pytest.approx(5.5)
    assert snapshot.total_watts == pytest.approx(28.5)


def test_simulator_reports_power_breakdown_and_energy() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, capacity=10.0, latency_ms=5.0, weight=1.0, active=True)
    graph.add_edge(1, 2, capacity=10.0, latency_ms=5.0, weight=1.0, active=False)

    model = PowerModel(
        network_fixed_watts=10.0,
        device_active_watts=5.0,
        device_sleep_watts=1.0,
        device_dynamic_watts=2.0,
        link_active_watts=3.0,
        link_sleep_watts=0.5,
        link_dynamic_watts=4.0,
    )

    simulator = Simulator(
        graph,
        routing_policy=ShortestPathPolicy(weight="weight"),
        dt_seconds=3600.0,
        default_capacity=10.0,
        default_latency_ms=5.0,
        congestion_alpha=0.0,
        power_model_watts=model.estimate_network,
        carbon_intensity_g_per_kwh=lambda _t: 400.0,
    )

    metrics = simulator.step([Flow(source=0, destination=1, demand=5.0)])

    assert metrics.delivered == pytest.approx(5.0)
    assert metrics.dropped == pytest.approx(0.0)
    assert metrics.avg_utilization == pytest.approx(0.5)
    assert metrics.power_total_watts == pytest.approx(28.5)
    assert metrics.power_fixed_watts == pytest.approx(24.5)
    assert metrics.power_variable_watts == pytest.approx(4.0)
    assert metrics.power_device_watts == pytest.approx(13.0)
    assert metrics.power_link_watts == pytest.approx(5.5)
    assert metrics.active_devices == 2
    assert metrics.inactive_devices == 1
    assert metrics.active_links == 1
    assert metrics.inactive_links == 1
    assert metrics.energy_kwh == pytest.approx(0.0285)
    assert metrics.carbon_g == pytest.approx(11.4)
