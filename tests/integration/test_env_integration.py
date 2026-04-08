from __future__ import annotations

import pytest

from greennet.env import EnvConfig, GreenNetEnv


pytestmark = pytest.mark.integration


def test_environment_stepping_is_deterministic_for_fixed_seed() -> None:
    config = EnvConfig(
        node_count=5,
        edge_prob=0.8,
        topology_seed=2,
        max_steps=4,
        decision_interval_steps=1,
        traffic_model="stochastic",
        traffic_scenario="burst",
        initial_off_edges=1,
        initial_off_seed=77,
        off_calm_steps_required=0,
        off_start_guard_decision_steps=0,
    )

    env_a = GreenNetEnv(config)
    env_b = GreenNetEnv(config)
    try:
        obs_a, _ = env_a.reset(seed=21)
        obs_b, _ = env_b.reset(seed=21)

        assert obs_a["edge_active"].tolist() == pytest.approx(obs_b["edge_active"].tolist())

        seq_a = []
        seq_b = []
        for _ in range(config.max_steps):
            _obs_a, reward_a, term_a, trunc_a, info_a = env_a.step(0)
            _obs_b, reward_b, term_b, trunc_b, info_b = env_b.step(0)
            seq_a.append(
                (
                    reward_a,
                    info_a["delta_delivered"],
                    info_a["delta_dropped"],
                    info_a["norm_drop"],
                    info_a["metrics"].avg_utilization,
                )
            )
            seq_b.append(
                (
                    reward_b,
                    info_b["delta_delivered"],
                    info_b["delta_dropped"],
                    info_b["norm_drop"],
                    info_b["metrics"].avg_utilization,
                )
            )
            assert (term_a, trunc_a) == (term_b, trunc_b)

        assert seq_a == pytest.approx(seq_b)
        assert trunc_a is True
        assert term_a is False
    finally:
        env_a.close()
        env_b.close()
