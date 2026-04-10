from __future__ import annotations

import networkx as nx
import pytest

from greennet.env import EnvConfig, GreenNetEnv
from greennet.power import PowerModel, PowerSnapshot
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


def test_power_model_can_disable_utilization_sensitive_dynamic_power() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, active=True, utilization=0.8)
    graph.add_edge(1, 2, active=True, utilization=0.3)

    dynamic_model = PowerModel(
        network_fixed_watts=10.0,
        device_active_watts=5.0,
        device_sleep_watts=1.0,
        device_dynamic_watts=2.0,
        link_active_watts=3.0,
        link_sleep_watts=0.5,
        link_dynamic_watts=4.0,
    )
    fixed_only_model = PowerModel(
        network_fixed_watts=10.0,
        device_active_watts=5.0,
        device_sleep_watts=1.0,
        device_dynamic_watts=2.0,
        link_active_watts=3.0,
        link_sleep_watts=0.5,
        link_dynamic_watts=4.0,
        utilization_sensitive=False,
    )

    dynamic_snapshot = dynamic_model.estimate_network(graph)
    fixed_only_snapshot = fixed_only_model.estimate_network(graph)

    assert dynamic_snapshot.variable_watts > 0.0
    assert fixed_only_snapshot.variable_watts == pytest.approx(0.0)
    assert fixed_only_snapshot.total_watts == pytest.approx(fixed_only_snapshot.fixed_watts)
    assert fixed_only_snapshot.total_watts < dynamic_snapshot.total_watts


def test_simulator_uses_total_power_for_energy_and_carbon_when_snapshot_includes_transition_costs() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, capacity=10.0, latency_ms=5.0, weight=1.0, active=True)

    transition_power_watts = 6.5
    snapshot = PowerSnapshot(
        total_watts=35.0,
        fixed_watts=24.5,
        variable_watts=4.0,
        device_watts=13.0,
        link_watts=5.5,
        active_devices=2,
        inactive_devices=0,
        active_links=1,
        inactive_links=0,
    )

    simulator = Simulator(
        graph,
        routing_policy=ShortestPathPolicy(weight="weight"),
        dt_seconds=3600.0,
        default_capacity=10.0,
        default_latency_ms=5.0,
        congestion_alpha=0.0,
        power_model_watts=lambda _graph: snapshot,
        carbon_intensity_g_per_kwh=lambda _t: 400.0,
    )

    metrics = simulator.step([Flow(source=0, destination=1, demand=5.0)])

    assert metrics.power_total_watts == pytest.approx(snapshot.total_watts)
    assert metrics.power_fixed_watts == pytest.approx(snapshot.fixed_watts)
    assert metrics.power_variable_watts == pytest.approx(snapshot.variable_watts)
    assert metrics.energy_kwh == pytest.approx(snapshot.total_watts / 1000.0)
    assert metrics.carbon_g == pytest.approx((snapshot.total_watts / 1000.0) * 400.0)
    assert metrics.energy_kwh == pytest.approx((snapshot.fixed_watts + snapshot.variable_watts + transition_power_watts) / 1000.0)


def test_env_applies_transition_energy_to_total_energy_and_carbon() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=3,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=2,
            decision_interval_steps=1,
            flows_per_step=0,
            enable_forecasting=False,
            initial_off_edges=0,
            off_start_guard_decision_steps=0,
            off_calm_steps_required=0,
            max_util_off_allow_threshold=1.1,
            power_network_fixed_watts=0.0,
            power_device_active_watts=0.0,
            power_device_sleep_watts=0.0,
            power_device_dynamic_watts=0.0,
            power_link_active_watts=0.0,
            power_link_sleep_watts=0.0,
            power_link_dynamic_watts=0.0,
            power_transition_on_joules=3600.0,
            power_transition_off_joules=7200.0,
            carbon_base_intensity_g_per_kwh=100.0,
            carbon_amplitude_g_per_kwh=0.0,
        )
    )
    try:
        env.reset(seed=7)
        env._last_max_util = 1.0

        _obs, _reward, _terminated, _truncated, info = env.step(1)
        metrics = info["metrics"]

        assert info["toggle_applied"] is True
        assert metrics.transition_on_count == 0
        assert metrics.transition_off_count == 1
        assert metrics.energy_steady_kwh == pytest.approx(0.0)
        assert metrics.energy_transition_kwh == pytest.approx(7200.0 / 3_600_000.0)
        assert metrics.energy_kwh == pytest.approx(metrics.energy_transition_kwh)
        assert metrics.power_transition_watts_equiv == pytest.approx(7200.0)
        assert metrics.carbon_intensity_g_per_kwh == pytest.approx(100.0)
        assert metrics.carbon_g == pytest.approx(metrics.energy_kwh * 100.0)
        assert info["delta_energy_kwh"] == pytest.approx(metrics.energy_kwh)
        assert info["delta_carbon_g"] == pytest.approx(metrics.carbon_g)
    finally:
        env.close()
