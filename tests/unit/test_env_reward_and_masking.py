from __future__ import annotations

import pytest

from greennet.env import EnvConfig, GreenNetEnv
from greennet.simulator import StepMetrics


def test_env_step_computes_reward_components_from_metrics() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=4,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=2,
            decision_interval_steps=1,
            flows_per_step=0,
            enable_forecasting=False,
            energy_weight=2.0,
            drop_penalty_lambda=3.0,
            qos_target_norm_drop=0.1,
            qos_min_volume=0.0,
            qos_violation_penalty_scale=5.0,
            toggle_penalty=0.0,
            toggle_apply_penalty=0.0,
            toggle_on_penalty_scale=0.0,
            toggle_off_penalty_scale=0.0,
            normalize_drop=False,
        )
    )
    try:
        env.reset(seed=11)
        env._generate_flows = lambda: ()  # type: ignore[method-assign]
        env.simulator.step = lambda flows: StepMetrics(  # type: ignore[method-assign]
            delivered=3.0,
            dropped=1.0,
            avg_utilization=0.25,
            avg_delay=0.01,
            avg_delay_ms=10.0,
            avg_path_latency_ms=8.0,
            energy_kwh=0.5,
            carbon_g=1.0,
        )

        _obs, reward, _terminated, _truncated, info = env.step(0)

        assert info["reward_energy"] == pytest.approx(-1.0)
        assert info["reward_drop"] == pytest.approx(-3.0)
        assert info["qos_excess"] == pytest.approx(0.05)
        assert info["reward_qos"] == pytest.approx(-0.25)
        assert info["reward_toggle"] == pytest.approx(0.0)
        assert reward == pytest.approx(-4.25)
    finally:
        env.close()


def test_action_mask_only_allows_noop_outside_decision_steps() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=4,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=5,
            decision_interval_steps=3,
            initial_off_edges=1,
            initial_off_seed=123,
            off_calm_steps_required=0,
            off_start_guard_decision_steps=0,
        )
    )
    try:
        env.reset(seed=7)

        mask = env.get_action_mask()

        assert mask.tolist() == [True, False, False, False, False, False, False]
    finally:
        env.close()


def test_action_mask_disables_off_toggles_when_off_actions_are_disabled() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=4,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=5,
            decision_interval_steps=1,
            initial_off_edges=1,
            initial_off_seed=123,
            disable_off_actions=True,
            off_calm_steps_required=0,
            off_start_guard_decision_steps=0,
        )
    )
    try:
        env.reset(seed=7)

        mask = env.get_action_mask()
        off_actions = [action for action in range(1, env.action_space.n) if env._is_off_toggle_action(action)]
        on_actions = [action for action in range(1, env.action_space.n) if not env._is_off_toggle_action(action)]

        assert off_actions
        assert on_actions
        assert all(not mask[action] for action in off_actions)
        assert any(mask[action] for action in on_actions)
    finally:
        env.close()


def test_action_mask_blocks_off_toggles_when_qos_guard_is_active() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=4,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=5,
            decision_interval_steps=1,
            initial_off_edges=1,
            initial_off_seed=123,
            off_calm_steps_required=0,
            off_start_guard_decision_steps=0,
        )
    )
    try:
        env.reset(seed=7)
        env._last_norm_drop_step = env.config.qos_target_norm_drop - (env.config.qos_guard_margin_off / 2.0)

        mask = env.get_action_mask()
        off_actions = [action for action in range(1, env.action_space.n) if env._is_off_toggle_action(action)]

        assert off_actions
        assert all(not mask[action] for action in off_actions)
        assert env._last_mask_reason_counts["qos_guard"] > 0
    finally:
        env.close()


def test_action_mask_allows_off_toggles_after_warmup_when_starting_all_on() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=4,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=5,
            decision_interval_steps=1,
            initial_off_edges=0,
            off_calm_steps_required=0,
            off_start_guard_decision_steps=0,
            flows_per_step=0,
            enable_forecasting=False,
        )
    )
    try:
        env.reset(seed=7)
        env._last_qos_viol_step = False
        env._last_norm_drop_step = 0.0
        env._last_max_util = 0.0
        env._last_demand_now_norm = 0.0
        env._last_demand_forecast_norm = 0.0

        mask = env.get_action_mask()
        off_actions = [action for action in range(1, env.action_space.n) if env._is_off_toggle_action(action)]

        assert off_actions
        assert any(mask[action] for action in off_actions)
        assert env._last_mask_reason_counts["off_all_on_calm"] == 0
    finally:
        env.close()


def test_env_tracks_flap_event_and_reversal_penalty_on_emergency_recovery() -> None:
    env = GreenNetEnv(
        EnvConfig(
            node_count=4,
            edge_prob=1.0,
            topology_seed=0,
            max_steps=5,
            decision_interval_steps=1,
            initial_off_edges=1,
            initial_off_seed=123,
            off_calm_steps_required=0,
            off_start_guard_decision_steps=0,
            stability_reversal_window_steps=20,
            stability_reversal_penalty=0.125,
        )
    )
    try:
        env.reset(seed=7)

        off_actions = [action for action in range(1, env.action_space.n) if env._is_off_toggle_action(action)]
        assert off_actions

        off_action = off_actions[0]
        _obs, _reward, _terminated, _truncated, first_info = env.step(off_action)
        assert first_info["toggle_applied"] is True
        assert first_info["toggled_off"] is True
        assert first_info["flap_event_count"] == 0

        env._last_qos_viol_step = True
        _obs, _reward, _terminated, _truncated, second_info = env.step(off_action)
        assert second_info["toggle_applied"] is True
        assert second_info["toggled_on"] is True
        assert second_info["flap_event"] is True
        assert second_info["flap_event_count"] == 1
        assert second_info["stability_reversal_penalty"] == pytest.approx(0.125)
        assert env._episode_flap_event_count == 1
    finally:
        env.close()
