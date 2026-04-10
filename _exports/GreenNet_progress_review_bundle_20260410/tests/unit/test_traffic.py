from __future__ import annotations

import pytest

from greennet.traffic import (
    ReplayTrafficGenerator,
    StochasticTrafficConfig,
    StochasticTrafficGenerator,
    TrafficValidationError,
    apply_traffic_scenario,
    load_named_traffic_profile,
    load_traffic_profile_from_dict,
)


def test_apply_traffic_scenario_normalizes_aliases_and_scales_fields() -> None:
    base = StochasticTrafficConfig(
        node_count=8,
        avg_bursts_per_step=1.0,
        duration_range=(2, 10),
        spike_duration_range=(4, 12),
    )

    config = apply_traffic_scenario(
        base,
        "failure",
        intensity=2.0,
        duration=0.5,
        frequency=0.5,
    )

    assert config.avg_bursts_per_step == pytest.approx(6.0)
    assert config.spike_prob == pytest.approx(0.075)
    assert config.duration_range == (1, 5)
    assert config.spike_duration_range == (2, 10)


def test_apply_traffic_scenario_diurnal_is_distinct_from_normal() -> None:
    normal = apply_traffic_scenario(StochasticTrafficConfig(node_count=8), "normal")
    diurnal = apply_traffic_scenario(StochasticTrafficConfig(node_count=8), "diurnal")

    assert normal.diurnal_profile is None
    assert diurnal.diurnal_profile is not None
    assert max(diurnal.diurnal_profile) == pytest.approx(1.0)


def test_apply_traffic_scenario_hotspot_is_safe_for_small_topologies() -> None:
    config = apply_traffic_scenario(StochasticTrafficConfig(node_count=6), "hotspot")

    assert config.hotspots
    assert all(0 <= src < 6 and 0 <= dst < 6 and src != dst for src, dst, _weight in config.hotspots)


def test_apply_traffic_scenario_supports_flash_crowd_and_multi_peak() -> None:
    flash = apply_traffic_scenario(StochasticTrafficConfig(node_count=12), "flash crowd")
    multi_peak = apply_traffic_scenario(StochasticTrafficConfig(node_count=12), "multi_peak")

    assert flash.spike_prob > 0.05
    assert flash.hotspots
    assert multi_peak.diurnal_profile is not None
    assert len(multi_peak.diurnal_profile) >= 8


def test_apply_traffic_scenario_rejects_unknown_scenarios() -> None:
    with pytest.raises(ValueError, match="Unknown traffic scenario"):
        apply_traffic_scenario(StochasticTrafficConfig(node_count=4), "not-a-scenario")


@pytest.mark.regression
def test_stochastic_generator_produces_stable_bursts_for_fixed_seed() -> None:
    config = StochasticTrafficConfig(
        node_count=4,
        avg_bursts_per_step=1.5,
        hotspots=((0, 1, 2.0),),
        p_elephant=0.0,
        mice_size_range=(2, 3),
        elephant_size_range=(10, 11),
        duration_range=(1, 2),
        spike_prob=0.0,
    )

    bursts = [
        (burst.source, burst.destination, burst.size, burst.start_time, burst.duration)
        for burst in StochasticTrafficGenerator(config, seed=17).generate(5)
    ]

    assert bursts == [
        (2, 0, 2, 0.0, 2),
        (3, 1, 3, 0.0, 2),
        (1, 0, 2, 1.0, 2),
        (0, 1, 2, 2.0, 2),
        (2, 0, 2, 2.0, 2),
        (0, 1, 2, 3.0, 2),
        (0, 1, 2, 3.0, 2),
        (0, 1, 3, 4.0, 2),
    ]


def test_named_replay_profile_repeats_on_its_declared_cycle() -> None:
    replay = load_named_traffic_profile("commuter_bursts", node_count=8)
    generator = ReplayTrafficGenerator(replay)

    starts = [
        burst.start_time
        for burst in generator.generate(10)
        if (burst.source, burst.destination, burst.size, burst.duration) == (6, 1, 7.0, 2)
    ]

    assert starts == [0.0, 8.0]


def test_named_replay_profiles_cover_small_and_large_topologies() -> None:
    small = load_named_traffic_profile("regional_ring_commuter_matrices", node_count=6)
    large = load_named_traffic_profile("backbone_large_flash_crowd_bursts", node_count=12)

    assert small.node_count == 6
    assert small.cycle_length == 4
    assert large.node_count == 12
    assert large.cycle_length == 10


def test_load_traffic_profile_rejects_node_count_mismatch() -> None:
    payload = {
        "format_version": 1,
        "node_count": 4,
        "bursts": [{"source": 0, "destination": 1, "size": 2.0, "start_time": 0, "duration": 1}],
    }

    with pytest.raises(TrafficValidationError, match="active topology uses node_count=5"):
        load_traffic_profile_from_dict(payload, node_count=5)


def test_load_traffic_profile_rejects_positive_diagonal_matrix() -> None:
    payload = {
        "format_version": 1,
        "node_count": 3,
        "matrices": [
            [
                [1, 0, 0],
                [0, 0, 2],
                [0, 0, 0],
            ]
        ],
    }

    with pytest.raises(TrafficValidationError, match="zero diagonal"):
        load_traffic_profile_from_dict(payload, node_count=3)


def test_named_traffic_profile_reports_node_count_incompatibility() -> None:
    with pytest.raises(TrafficValidationError, match="not compatible with node_count=6"):
        load_named_traffic_profile("commuter_matrices", node_count=6)
