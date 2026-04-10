from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from stable_baselines3.common.monitor import Monitor

from greennet.env import EnvConfig, GreenNetEnv

try:
    from sb3_contrib import MaskablePPO  # type: ignore
    from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
    MASKABLE_AVAILABLE = True
except Exception:  # pragma: no cover
    MaskablePPO = None  # type: ignore
    get_action_masks = None  # type: ignore
    ActionMasker = None  # type: ignore
    MASKABLE_AVAILABLE = False

DEBUG_MASKS = False

EVAL_TOLS: Dict[str, float] = {
    "reward": 0.5,
    "dropped": 1.0,
    "energy": 0.02,
}
TRACK_TOLS: Dict[str, Dict[str, float]] = {
    "NORMAL_STABILITY": {"reward": EVAL_TOLS["reward"], "dropped": EVAL_TOLS["dropped"], "energy": EVAL_TOLS["energy"]},
    "NORMAL_CAPABILITY": {"reward": EVAL_TOLS["reward"], "dropped": EVAL_TOLS["dropped"], "energy": 0.03},
    "CUSTOM": {"reward": EVAL_TOLS["reward"], "dropped": EVAL_TOLS["dropped"], "energy": EVAL_TOLS["energy"]},
}


def custom_gate(
    delta_reward: float,
    delta_energy: float,
    delta_dropped: float,
    *,
    reward_min: float = -0.5,
    dropped_max: float = +1.0,
    energy_max: float = 0.0,
) -> tuple[bool, str]:
    """CUSTOM acceptance gate used by eval summary and sweep ranking."""
    ok_reward = float(delta_reward) >= float(reward_min)
    ok_dropped = float(delta_dropped) <= float(dropped_max)
    ok_energy = float(delta_energy) <= float(energy_max)
    ok_all = bool(ok_reward and ok_dropped and ok_energy)
    if ok_all:
        return True, "pass"

    failures: list[str] = []
    if not ok_reward:
        failures.append(f"reward({float(delta_reward):+.3f}<{float(reward_min):+.3f})")
    if not ok_dropped:
        failures.append(f"dropped({float(delta_dropped):+.3f}>+{float(dropped_max):.3f})")
    if not ok_energy:
        failures.append(f"energy({float(delta_energy):+.6f}>+{float(energy_max):.6f})")
    return False, "fail: " + ", ".join(failures)

def print_model_artifact_info(model_path: Path) -> None:
    """Print absolute path + lightweight file metadata for eval sanity checks."""
    resolved = Path(model_path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        print(f"[model] file not found: {resolved}")
        return

    size_mb = float(stat.st_size) / (1024.0 * 1024.0)
    mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    print(
        f"[model] path={resolved} size_bytes={int(stat.st_size)} "
        f"size_mb={size_mb:.3f} mtime_utc={mtime_utc}"
    )


def parse_seed_list(seed_text: str) -> list[int]:
    """Parse comma-separated integers like '0,1,2,3,4'."""
    parts = [p.strip() for p in seed_text.split(",") if p.strip()]
    return [int(p) for p in parts]

def eval_policy(
    model: Any | None,
    env_config: EnvConfig,
    episodes: int,
    seed: int,
    label: str,
    deterministic: bool,
    debug_energy: bool = False,
    policy_mode: str = "model",
    eval_max_on_edges: int | None = None,
) -> Dict[str, float]:
    """Evaluate a policy and print aggregate stats."""
    base_env = GreenNetEnv(config=replace(env_config, debug_logs=False))
    base_env = Monitor(base_env)
    if MASKABLE_AVAILABLE and ActionMasker is not None:
        base_env = ActionMasker(base_env, lambda e: e.unwrapped.get_action_mask())
    env = base_env
    if DEBUG_MASKS and MASKABLE_AVAILABLE:
        assert hasattr(env, "action_masks"), f"Masking broken in eval: env has no action_masks(), type={type(env)}"
        inner_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if getattr(inner_env, "debug_logs", False) and not getattr(inner_env, "_train_mask_debug_printed", False):
            print("[mask-debug] eval env:", type(env), "has action_masks:", hasattr(env, "action_masks"))
            setattr(inner_env, "_train_mask_debug_printed", True)

    ep_rewards: list[float] = []
    ep_energy: list[float] = []
    ep_dropped: list[float] = []
    ep_delivered: list[float] = []
    ep_applied: list[int] = []
    ep_reverted: list[int] = []
    ep_norm_drop: list[float] = []
    ep_reward_energy: list[float] = []
    ep_reward_drop: list[float] = []
    ep_reward_qos: list[float] = []
    ep_reward_toggle: list[float] = []
    ep_reward_other: list[float] = []
    ep_reward_toggle_on: list[float] = []
    ep_reward_toggle_off: list[float] = []
    ep_reward_toggle_harm: list[float] = []
    ep_steps_total: list[int] = []
    ep_toggle_total: list[int] = []
    ep_toggle_rate: list[float] = []
    ep_qos_violation_rate: list[float] = []
    ep_qos_violation_steps: list[int] = []
    ep_qos_allow_on_steps: list[int] = []
    ep_norm_drop_step_mean: list[float] = []
    ep_toggle_attempted: list[int] = []
    ep_toggle_allowed: list[int] = []
    ep_blocked_budget: list[int] = []
    ep_blocked_util: list[int] = []
    ep_blocked_cooldown: list[int] = []
    ep_blocked_global_off_stress: list[int] = []
    ep_toggle_applied_count: list[int] = []
    ep_emergency_on_applied_count: list[int] = []
    ep_dbg_cd_block_mask_val_last: list[float] = []
    ep_dbg_cd_block_mask_samples: list[float] = []
    ep_dbg_cd_block_mask_mismatches: list[float] = []
    ep_dbg_block_edge_cd: list[float] = []
    ep_dbg_block_global_cd: list[float] = []
    ep_dbg_block_budget: list[float] = []
    ep_dbg_block_util: list[float] = []
    ep_dbg_block_off_stress: list[float] = []
    ep_dbg_block_qos_off: list[float] = []
    ep_dbg_block_qos_on: list[float] = []
    ep_toggle_attempted_rate: list[float] = []
    ep_toggle_allowed_rate: list[float] = []
    ep_blocked_budget_rate: list[float] = []
    ep_blocked_util_rate: list[float] = []
    ep_blocked_cooldown_rate: list[float] = []
    ep_blocked_global_off_stress_rate: list[float] = []
    ep_toggle_applied_rate: list[float] = []
    ep_toggled_on: list[int] = []
    ep_toggled_off: list[int] = []
    ep_blocked_off_budget: list[int] = []
    ep_off_toggles_used: list[int] = []
    ep_blocked_total_budget: list[int] = []
    ep_total_toggles_used: list[int] = []
    ep_valid_actions_per_step: list[float] = []
    ep_decision_steps_count: list[int] = []
    ep_valid_actions_on_decision_steps: list[float] = []
    ep_mask_calls: list[float] = []
    ep_valid_toggle_actions_per_step: list[float] = []
    ep_valid_on_actions_per_step: list[float] = []
    ep_valid_off_actions_per_step: list[float] = []
    ep_valid_noop_actions_per_step: list[float] = []
    ep_valid_toggle_actions_on_decision_steps: list[float] = []
    ep_valid_on_actions_on_decision_steps: list[float] = []
    ep_valid_off_actions_on_decision_steps: list[float] = []
    ep_valid_noop_actions_on_decision_steps: list[float] = []
    ep_mean_demand_per_step: list[float] = []
    ep_delta_dropped_after_toggle_mean: list[float] = []
    ep_delta_qos_after_toggle_mean: list[float] = []
    ep_on_edges_count_mean: list[float] = []
    ep_off_edges_count_mean: list[float] = []
    ep_max_util_mean: list[float] = []
    ep_toggle_budget_remaining_mean: list[float] = []
    ep_toggle_budget_remaining_available: list[bool] = []
    ep_valid_on_first20_decision_steps_mean: list[float] = []
    ep_fraction_decision_steps_any_on_valid: list[float] = []
    ep_noop_chosen_on_decision_steps_mean: list[float] = []
    ep_controller_blocked_on_budget: list[int] = []
    ep_pending_toggle_harm_mean: list[float] = []
    ep_toggle_harm_max_dd_mean: list[float] = []
    ep_toggle_harm_max_dq_mean: list[float] = []
    ep_edge_universe_size: list[float] = []
    ep_initial_off_requested: list[float] = []
    ep_initial_off_applied: list[float] = []
    ep_off_edges_first20_decision_steps_mean: list[float] = []
    ep_off_edges_all_decision_steps_mean: list[float] = []
    ep_decision_step_when_off_zero: list[float] = []
    on_reason_totals: dict[str, int] = {}
    max_on_edges_budget = (
        max(0, int(eval_max_on_edges))
        if eval_max_on_edges is not None
        else None
    )

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_r = 0.0
        total_energy = 0.0
        total_dropped = 0.0
        total_delivered = 0.0
        applied = 0
        reverted = 0
        attempted = 0
        allowed = 0
        blocked_budget = 0
        blocked_util = 0
        blocked_cooldown = 0
        blocked_global_off_stress = 0
        applied_count = 0
        emergency_on_applied_count = 0
        toggled_on_count = 0
        toggled_off_count = 0
        blocked_off_budget = 0
        off_toggles_used_last = 0
        blocked_total_budget = 0
        total_toggles_used_last = 0
        steps = 0
        decision_steps = 0
        qos_violation_steps = 0
        qos_allow_on_steps = 0
        sum_reward_energy = 0.0
        sum_reward_drop = 0.0
        sum_reward_qos = 0.0
        sum_reward_toggle = 0.0
        sum_reward_other = 0.0
        sum_reward_toggle_on = 0.0
        sum_reward_toggle_off = 0.0
        sum_reward_toggle_harm = 0.0
        delta_drop_after_toggle_vals: list[float] = []
        delta_qos_after_toggle_vals: list[float] = []
        norm_drop_step_vals: list[float] = []
        valid_counts: list[int] = []
        decision_valid_counts: list[int] = []
        valid_toggle_counts: list[int] = []
        valid_on_counts: list[int] = []
        valid_off_counts: list[int] = []
        valid_noop_counts: list[int] = []
        decision_valid_toggle_counts: list[int] = []
        decision_valid_on_counts: list[int] = []
        decision_valid_off_counts: list[int] = []
        decision_valid_noop_counts: list[int] = []
        demand_step_vals: list[float] = []
        on_edges_vals: list[int] = []
        off_edges_vals: list[int] = []
        max_util_vals: list[float] = []
        budget_remaining_vals: list[float] = []
        edge_universe_size_last = 0
        initial_off_requested_last = 0
        initial_off_applied_last = 0
        decision_off_edges_counts: list[int] = []
        decision_step_when_off_zero: int | None = None
        noop_chosen_on_decision_steps = 0
        mask_calls_last = 0
        dbg_cd_block_mask_val_last = -999
        dbg_cd_block_mask_samples_last = 0
        dbg_cd_block_mask_mismatches_last = 0
        dbg_block_edge_cd_last = 0
        dbg_block_global_cd_last = 0
        dbg_block_budget_last = 0
        dbg_block_util_last = 0
        dbg_block_off_stress_last = 0
        dbg_block_qos_off_last = 0
        dbg_block_qos_on_last = 0
        controller_blocked_on_budget = 0
        pending_toggle_harm_vals: list[float] = []
        toggle_harm_max_dd_vals: list[float] = []
        toggle_harm_max_dq_vals: list[float] = []
        debug_decision_reason_prints = 0

        while not (terminated or truncated):
            action_masks: np.ndarray | None = None
            if MASKABLE_AVAILABLE and get_action_masks is not None:
                try:
                    raw_masks = get_action_masks(env)
                    if raw_masks is not None:
                        action_masks = np.asarray(raw_masks, dtype=bool).reshape(-1)
                except Exception:
                    action_masks = None
            if action_masks is None:
                try:
                    action_masks = np.asarray(env.unwrapped.get_action_mask(), dtype=bool).reshape(-1)
                except Exception:
                    action_masks = None

            if action_masks is not None:
                valid_counts.append(int(action_masks.sum()))
                inner_env = env.unwrapped if hasattr(env, "unwrapped") else None
                valid_toggle = int(action_masks[1:].sum()) if action_masks.size > 1 else 0
                valid_on = 0
                valid_off = 0
                if inner_env is not None and hasattr(inner_env, "edge_list") and hasattr(inner_env, "simulator"):
                    sim = getattr(inner_env, "simulator", None)
                    edge_list = getattr(inner_env, "edge_list", [])
                    for a in np.where(action_masks)[0]:
                        if int(a) <= 0:
                            continue
                        idx = int(a) - 1
                        if idx < 0 or idx >= len(edge_list) or sim is None:
                            continue
                        edge = edge_list[idx]
                        key = inner_env._edge_key(edge[0], edge[1])  # type: ignore[attr-defined]
                        current_state = bool(sim.active.get(key, True))
                        if current_state:
                            valid_off += 1
                        else:
                            valid_on += 1
                valid_toggle_counts.append(valid_toggle)
                valid_on_counts.append(valid_on)
                valid_off_counts.append(valid_off)
                valid_noop_counts.append(int(action_masks[0]) if action_masks.size > 0 else 0)

            if policy_mode == "noop" or model is None:
                action = 0
            elif policy_mode == "random_masked":
                if action_masks is not None and action_masks.size > 0:
                    allowed_actions = np.where(action_masks)[0]
                    non_noop = allowed_actions[allowed_actions != 0]
                    if non_noop.size > 0:
                        action = int(np.random.choice(non_noop))
                    elif allowed_actions.size > 0:
                        action = int(np.random.choice(allowed_actions))
                    else:
                        action = 0
                else:
                    action = int(env.action_space.sample())
            else:
                if (
                    MASKABLE_AVAILABLE
                    and get_action_masks is not None
                    and MaskablePPO is not None
                    and isinstance(model, MaskablePPO)
                ):
                    assert action_masks is not None
                    action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                try:
                    action = int(action)
                except Exception:
                    action = int(action[0])

            if max_on_edges_budget is not None:
                action_int = int(action)
                if action_int > 0:
                    inner_env = env.unwrapped if hasattr(env, "unwrapped") else None
                    if inner_env is not None and hasattr(inner_env, "edge_list") and hasattr(inner_env, "simulator"):
                        sim = getattr(inner_env, "simulator", None)
                        edge_list = getattr(inner_env, "edge_list", [])
                        idx = action_int - 1
                        if sim is not None and 0 <= idx < len(edge_list):
                            active = getattr(sim, "active", None)
                            if isinstance(active, dict):
                                edge = edge_list[idx]
                                key = inner_env._edge_key(edge[0], edge[1])  # type: ignore[attr-defined]
                                current_state = bool(active.get(key, True))
                                on_edges_now: int | None = None
                                edge_universe = getattr(inner_env, "_edge_universe", None)
                                if isinstance(edge_universe, (list, tuple)):
                                    try:
                                        on_edges_now = int(
                                            sum(1 for e in edge_universe if bool(active.get(e, False)))
                                        )
                                    except Exception:
                                        on_edges_now = None
                                if on_edges_now is None:
                                    on_edges_now = int(sum(1 for is_on in active.values() if bool(is_on)))
                                if (not current_state) and on_edges_now >= max_on_edges_budget:
                                    action = 0
                                    controller_blocked_on_budget += 1

            obs, r, terminated, truncated, info = env.step(action)
            steps += 1
            if info.get("qos_violation", False):
                qos_violation_steps += 1
            if info.get("qos_allow_on", False):
                qos_allow_on_steps += 1
            if debug_energy and steps in (1, 10, 100, 1000):
                print(
                    "dbg steps",
                    steps,
                    "delta_e",
                    info.get("delta_energy_kwh"),
                    "metrics_e",
                    getattr(info.get("metrics"), "energy_kwh", None),
                    "info_energy",
                    info.get("energy_kwh"),
                )
            total_r += float(r)
            sum_reward_energy += float(info.get("reward_energy", 0.0))
            sum_reward_drop += float(info.get("reward_drop", 0.0))
            sum_reward_qos += float(info.get("reward_qos", 0.0))
            sum_reward_toggle += float(info.get("reward_toggle", 0.0))
            reward_other_step = info.get(
                "r_other",
                float(info.get("total_reward", r))
                - float(info.get("reward_energy", 0.0))
                - float(info.get("reward_drop", 0.0))
                - float(info.get("reward_qos", 0.0))
                - float(info.get("reward_toggle", 0.0)),
            )
            sum_reward_other += float(reward_other_step)
            sum_reward_toggle_on += float(info.get("reward_toggle_on", 0.0))
            sum_reward_toggle_off += float(info.get("reward_toggle_off", 0.0))
            sum_reward_toggle_harm += float(info.get("reward_toggle_harm", 0.0))
            delta_drop_after_toggle_vals.append(float(info.get("delta_dropped_after_toggle", 0.0)))
            delta_qos_after_toggle_vals.append(float(info.get("delta_qos_after_toggle", 0.0)))
            norm_drop_step_vals.append(float(info.get("norm_drop_step", 0.0)))
            pending_toggle_harm_vals.append(float(info.get("pending_toggle_harm", 0)))
            toggle_harm_max_dd_vals.append(float(info.get("toggle_harm_max_dd", 0.0)))
            toggle_harm_max_dq_vals.append(float(info.get("toggle_harm_max_dq", 0.0)))

            metrics = info.get("metrics", None)

            if "delta_energy_kwh" in info:
                total_energy += float(info["delta_energy_kwh"])
            elif "energy_kwh" in info:
                total_energy += float(info["energy_kwh"])
            elif metrics is not None:
                total_energy += float(getattr(metrics, "energy_kwh", 0.0))

            if "delta_dropped" in info:
                total_dropped += float(info["delta_dropped"])
            elif metrics is not None:
                total_dropped += float(getattr(metrics, "dropped", 0.0))

            if "delta_delivered" in info:
                total_delivered += float(info["delta_delivered"])
            elif metrics is not None:
                total_delivered += float(getattr(metrics, "delivered", 0.0))

            applied += int(bool(info.get("toggle_applied")))
            reverted += int(bool(info.get("toggle_reverted")))
            attempted += int(info.get("toggles_attempted_count", 0))
            allowed += int(info.get("allowed_toggle_count", 0))
            blocked_budget += int(info.get("blocked_by_budget_count", 0))
            blocked_util += int(info.get("blocked_by_util_count", 0))
            blocked_cooldown += int(info.get("blocked_by_cooldown_count", 0))
            blocked_global_off_stress += int(info.get("blocked_by_global_off_stress_count", 0))
            applied_count += int(info.get("toggles_applied_count", 0))
            emergency_on_applied_count += int(info.get("emergency_on_applied_count", 0))
            toggled_on_count += int(bool(info.get("toggled_on", False)))
            toggled_off_count += int(bool(info.get("toggled_off", False)))
            blocked_off_budget += int(bool(info.get("toggle_blocked_off_budget", False)))
            off_toggles_used_last = int(info.get("off_toggles_used", off_toggles_used_last))
            blocked_total_budget += int(bool(info.get("toggle_blocked_total_budget", False)))
            total_toggles_used_last = int(info.get("total_toggles_used", total_toggles_used_last))
            flows = info.get("flows", ())
            demand_step_vals.append(float(sum(getattr(f, "demand", 0.0) for f in flows)) if flows else 0.0)
            on_edges_vals.append(int(info.get("on_edges_count", 0)))
            off_edges_vals.append(int(info.get("off_edges_count", 0)))
            max_util_vals.append(float(info.get("max_util", 0.0)))
            budget_rem = info.get("toggle_budget_remaining", None)
            if budget_rem is not None:
                budget_remaining_vals.append(float(budget_rem))
            mask_calls_last = int(info.get("mask_calls", mask_calls_last))
            dbg_cd_block_mask_val_last = int(
                info.get(
                    "dbg_cd_block_mask_val_last",
                    info.get("dbg_cd_block_mask_val", dbg_cd_block_mask_val_last),
                )
            )
            dbg_cd_block_mask_samples_last = int(
                info.get("dbg_cd_block_mask_samples", dbg_cd_block_mask_samples_last)
            )
            dbg_cd_block_mask_mismatches_last = int(
                info.get(
                    "dbg_cd_block_mask_mismatches",
                    info.get("cd_block_mask_mismatch_count", dbg_cd_block_mask_mismatches_last),
                )
            )
            dbg_block_edge_cd_last = int(info.get("dbg_block_edge_cd", dbg_block_edge_cd_last))
            dbg_block_global_cd_last = int(info.get("dbg_block_global_cd", dbg_block_global_cd_last))
            dbg_block_budget_last = int(info.get("dbg_block_budget", dbg_block_budget_last))
            dbg_block_util_last = int(info.get("dbg_block_util", dbg_block_util_last))
            dbg_block_off_stress_last = int(
                info.get("dbg_block_off_stress", dbg_block_off_stress_last)
            )
            dbg_block_qos_off_last = int(info.get("dbg_block_qos_off", dbg_block_qos_off_last))
            dbg_block_qos_on_last = int(info.get("dbg_block_qos_on", dbg_block_qos_on_last))
            edge_universe_size_last = int(info.get("edge_universe_size", edge_universe_size_last))
            initial_off_requested_last = int(info.get("initial_off_requested", initial_off_requested_last))
            initial_off_applied_last = int(info.get("initial_off_applied", initial_off_applied_last))
            if bool(info.get("is_decision_step", False)):
                decision_steps += 1
                noop_chosen_on_decision_steps += int(bool(info.get("noop_chosen", False)))
                off_now = int(info.get("off_edges_count", 0))
                decision_off_edges_counts.append(off_now)
                if decision_step_when_off_zero is None and off_now <= 0:
                    decision_step_when_off_zero = int(decision_steps)
                reason_counts = info.get("mask_reason_counts", {})
                if isinstance(reason_counts, dict):
                    for key, value in reason_counts.items():
                        if str(key).startswith("on_disabled_"):
                            on_reason_totals[str(key)] = on_reason_totals.get(str(key), 0) + int(value)
                if action_masks is not None:
                    decision_valid_counts.append(int(action_masks.sum()))
                    if valid_toggle_counts:
                        decision_valid_toggle_counts.append(valid_toggle_counts[-1])
                        decision_valid_on_counts.append(valid_on_counts[-1] if valid_on_counts else 0)
                        decision_valid_off_counts.append(valid_off_counts[-1] if valid_off_counts else 0)
                        decision_valid_noop_counts.append(valid_noop_counts[-1] if valid_noop_counts else 0)
                if label == "trained" and policy_mode == "model" and ep == 0 and debug_decision_reason_prints < 5:
                    print(
                        f"[mask reasons ep1 step={steps}] "
                        f"{info.get('mask_reason_counts', {})}"
                    )
                    debug_decision_reason_prints += 1

        if debug_energy:
            print("dbg episode steps total:", steps, "total_energy:", total_energy)

        denom = max(total_delivered + total_dropped, 1e-9)
        ep_norm_drop.append(float(total_dropped / denom))

        ep_rewards.append(total_r)
        ep_energy.append(total_energy)
        ep_dropped.append(total_dropped)
        ep_delivered.append(total_delivered)
        ep_applied.append(applied)
        ep_reverted.append(reverted)
        ep_reward_energy.append(sum_reward_energy)
        ep_reward_drop.append(sum_reward_drop)
        ep_reward_qos.append(sum_reward_qos)
        ep_reward_toggle.append(sum_reward_toggle)
        ep_reward_other.append(sum_reward_other)
        ep_reward_toggle_on.append(sum_reward_toggle_on)
        ep_reward_toggle_off.append(sum_reward_toggle_off)
        ep_reward_toggle_harm.append(sum_reward_toggle_harm)
        ep_steps_total.append(int(steps))
        toggle_total = applied_count
        ep_toggle_total.append(toggle_total)
        if steps > 0:
            ep_toggle_rate.append(toggle_total / float(steps))
        else:
            ep_toggle_rate.append(0.0)
        ep_toggle_attempted.append(attempted)
        ep_toggle_allowed.append(allowed)
        ep_blocked_budget.append(blocked_budget)
        ep_blocked_util.append(blocked_util)
        ep_blocked_cooldown.append(blocked_cooldown)
        ep_blocked_global_off_stress.append(blocked_global_off_stress)
        ep_toggle_applied_count.append(applied_count)
        ep_emergency_on_applied_count.append(emergency_on_applied_count)
        ep_toggled_on.append(toggled_on_count)
        ep_toggled_off.append(toggled_off_count)
        ep_blocked_off_budget.append(blocked_off_budget)
        ep_off_toggles_used.append(off_toggles_used_last)
        ep_blocked_total_budget.append(blocked_total_budget)
        ep_total_toggles_used.append(total_toggles_used_last)
        ep_controller_blocked_on_budget.append(controller_blocked_on_budget)
        ep_valid_actions_per_step.append(float(np.mean(valid_counts)) if valid_counts else 0.0)
        ep_decision_steps_count.append(int(decision_steps))
        ep_valid_actions_on_decision_steps.append(
            float(np.mean(decision_valid_counts)) if decision_valid_counts else 0.0
        )
        ep_mask_calls.append(float(mask_calls_last))
        ep_dbg_cd_block_mask_val_last.append(float(dbg_cd_block_mask_val_last))
        ep_dbg_cd_block_mask_samples.append(float(dbg_cd_block_mask_samples_last))
        ep_dbg_cd_block_mask_mismatches.append(float(dbg_cd_block_mask_mismatches_last))
        ep_dbg_block_edge_cd.append(float(dbg_block_edge_cd_last))
        ep_dbg_block_global_cd.append(float(dbg_block_global_cd_last))
        ep_dbg_block_budget.append(float(dbg_block_budget_last))
        ep_dbg_block_util.append(float(dbg_block_util_last))
        ep_dbg_block_off_stress.append(float(dbg_block_off_stress_last))
        ep_dbg_block_qos_off.append(float(dbg_block_qos_off_last))
        ep_dbg_block_qos_on.append(float(dbg_block_qos_on_last))
        ep_valid_toggle_actions_per_step.append(float(np.mean(valid_toggle_counts)) if valid_toggle_counts else 0.0)
        ep_valid_on_actions_per_step.append(float(np.mean(valid_on_counts)) if valid_on_counts else 0.0)
        ep_valid_off_actions_per_step.append(float(np.mean(valid_off_counts)) if valid_off_counts else 0.0)
        ep_valid_noop_actions_per_step.append(float(np.mean(valid_noop_counts)) if valid_noop_counts else 0.0)
        ep_valid_toggle_actions_on_decision_steps.append(
            float(np.mean(decision_valid_toggle_counts)) if decision_valid_toggle_counts else 0.0
        )
        ep_valid_on_actions_on_decision_steps.append(
            float(np.mean(decision_valid_on_counts)) if decision_valid_on_counts else 0.0
        )
        ep_valid_off_actions_on_decision_steps.append(
            float(np.mean(decision_valid_off_counts)) if decision_valid_off_counts else 0.0
        )
        ep_valid_noop_actions_on_decision_steps.append(
            float(np.mean(decision_valid_noop_counts)) if decision_valid_noop_counts else 0.0
        )
        ep_mean_demand_per_step.append(float(np.mean(demand_step_vals)) if demand_step_vals else 0.0)
        ep_delta_dropped_after_toggle_mean.append(
            float(np.mean(delta_drop_after_toggle_vals)) if delta_drop_after_toggle_vals else 0.0
        )
        ep_delta_qos_after_toggle_mean.append(
            float(np.mean(delta_qos_after_toggle_vals)) if delta_qos_after_toggle_vals else 0.0
        )
        ep_norm_drop_step_mean.append(
            float(np.mean(norm_drop_step_vals)) if norm_drop_step_vals else 0.0
        )
        ep_on_edges_count_mean.append(float(np.mean(on_edges_vals)) if on_edges_vals else 0.0)
        ep_off_edges_count_mean.append(float(np.mean(off_edges_vals)) if off_edges_vals else 0.0)
        ep_max_util_mean.append(float(np.mean(max_util_vals)) if max_util_vals else 0.0)
        if budget_remaining_vals:
            ep_toggle_budget_remaining_mean.append(float(np.mean(budget_remaining_vals)))
            ep_toggle_budget_remaining_available.append(True)
        else:
            ep_toggle_budget_remaining_mean.append(float("nan"))
            ep_toggle_budget_remaining_available.append(False)
        first20 = decision_valid_on_counts[:20]
        ep_valid_on_first20_decision_steps_mean.append(float(np.mean(first20)) if first20 else 0.0)
        if decision_steps > 0:
            ep_noop_chosen_on_decision_steps_mean.append(float(noop_chosen_on_decision_steps) / float(decision_steps))
        else:
            ep_noop_chosen_on_decision_steps_mean.append(0.0)
        ep_pending_toggle_harm_mean.append(
            float(np.mean(pending_toggle_harm_vals)) if pending_toggle_harm_vals else 0.0
        )
        ep_toggle_harm_max_dd_mean.append(
            float(np.mean(toggle_harm_max_dd_vals)) if toggle_harm_max_dd_vals else 0.0
        )
        ep_toggle_harm_max_dq_mean.append(
            float(np.mean(toggle_harm_max_dq_vals)) if toggle_harm_max_dq_vals else 0.0
        )
        first20_off = decision_off_edges_counts[:20]
        ep_off_edges_first20_decision_steps_mean.append(float(np.mean(first20_off)) if first20_off else 0.0)
        ep_off_edges_all_decision_steps_mean.append(float(np.mean(decision_off_edges_counts)) if decision_off_edges_counts else 0.0)
        ep_decision_step_when_off_zero.append(float(decision_step_when_off_zero) if decision_step_when_off_zero is not None else -1.0)
        ep_edge_universe_size.append(float(edge_universe_size_last))
        ep_initial_off_requested.append(float(initial_off_requested_last))
        ep_initial_off_applied.append(float(initial_off_applied_last))
        if decision_valid_on_counts:
            any_on = sum(1 for v in decision_valid_on_counts if v > 0)
            ep_fraction_decision_steps_any_on_valid.append(float(any_on) / float(len(decision_valid_on_counts)))
        else:
            ep_fraction_decision_steps_any_on_valid.append(0.0)
        if steps > 0:
            ep_toggle_attempted_rate.append(attempted / float(steps))
            ep_toggle_allowed_rate.append(allowed / float(steps))
            ep_blocked_budget_rate.append(blocked_budget / float(steps))
            ep_blocked_util_rate.append(blocked_util / float(steps))
            ep_blocked_cooldown_rate.append(blocked_cooldown / float(steps))
            ep_blocked_global_off_stress_rate.append(blocked_global_off_stress / float(steps))
            ep_toggle_applied_rate.append(applied_count / float(steps))
        else:
            ep_toggle_attempted_rate.append(0.0)
            ep_toggle_allowed_rate.append(0.0)
            ep_blocked_budget_rate.append(0.0)
            ep_blocked_util_rate.append(0.0)
            ep_blocked_cooldown_rate.append(0.0)
            ep_blocked_global_off_stress_rate.append(0.0)
            ep_toggle_applied_rate.append(0.0)
        if steps > 0:
            ep_qos_violation_rate.append(qos_violation_steps / float(steps))
        else:
            ep_qos_violation_rate.append(0.0)
        ep_qos_violation_steps.append(int(qos_violation_steps))
        ep_qos_allow_on_steps.append(int(qos_allow_on_steps))

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        arr = np.asarray(xs, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    r_m, r_s = _mean_std(ep_rewards)
    e_m, e_s = _mean_std(ep_energy)
    d_m, d_s = _mean_std(ep_dropped)
    dl_m, dl_s = _mean_std(ep_delivered)
    nd_m, nd_s = _mean_std(ep_norm_drop)
    re_m, re_s = _mean_std(ep_reward_energy)
    rd_m, rd_s = _mean_std(ep_reward_drop)
    rq_m, rq_s = _mean_std(ep_reward_qos)
    rt_m, rt_s = _mean_std(ep_reward_toggle)
    ro_m, ro_s = _mean_std(ep_reward_other)
    rto_m, rto_s = _mean_std(ep_reward_toggle_on)
    rtf_m, rtf_s = _mean_std(ep_reward_toggle_off)
    rth_m, rth_s = _mean_std(ep_reward_toggle_harm)
    tt_m, tt_s = _mean_std([float(v) for v in ep_toggle_total])
    tr_m, tr_s = _mean_std([float(v) for v in ep_toggle_rate])
    qv_m, qv_s = _mean_std(ep_qos_violation_rate)
    qvs_m, qvs_s = _mean_std([float(v) for v in ep_qos_violation_steps])
    qas_m, qas_s = _mean_std([float(v) for v in ep_qos_allow_on_steps])
    nds_m, nds_s = _mean_std(ep_norm_drop_step_mean)
    qpv_vals = [
        float(qos_sum) / float(max(1, int(qvs)))
        for qos_sum, qvs in zip(ep_reward_qos, ep_qos_violation_steps)
    ]
    qpv_m, qpv_s = _mean_std(qpv_vals if qpv_vals else [0.0])
    cdmv_m, cdmv_s = _mean_std(ep_dbg_cd_block_mask_val_last)
    cdsamp_m, cdsamp_s = _mean_std(ep_dbg_cd_block_mask_samples)
    cdmis_m, cdmis_s = _mean_std(ep_dbg_cd_block_mask_mismatches)
    dbg_edge_cd_m, dbg_edge_cd_s = _mean_std(ep_dbg_block_edge_cd)
    dbg_global_cd_m, dbg_global_cd_s = _mean_std(ep_dbg_block_global_cd)
    dbg_budget_m, dbg_budget_s = _mean_std(ep_dbg_block_budget)
    dbg_util_m, dbg_util_s = _mean_std(ep_dbg_block_util)
    dbg_off_stress_m, dbg_off_stress_s = _mean_std(ep_dbg_block_off_stress)
    dbg_qos_off_m, dbg_qos_off_s = _mean_std(ep_dbg_block_qos_off)
    dbg_qos_on_m, dbg_qos_on_s = _mean_std(ep_dbg_block_qos_on)
    ta_m, ta_s = _mean_std([float(v) for v in ep_toggle_attempted])
    al_m, al_s = _mean_std([float(v) for v in ep_toggle_allowed])
    bb_m, bb_s = _mean_std([float(v) for v in ep_blocked_budget])
    bu_m, bu_s = _mean_std([float(v) for v in ep_blocked_util])
    bc_m, bc_s = _mean_std([float(v) for v in ep_blocked_cooldown])
    bos_m, bos_s = _mean_std([float(v) for v in ep_blocked_global_off_stress])
    ap_m, ap_s = _mean_std([float(v) for v in ep_toggle_applied_count])
    eo_m, eo_s = _mean_std([float(v) for v in ep_emergency_on_applied_count])
    ta_r_m, ta_r_s = _mean_std([float(v) for v in ep_toggle_attempted_rate])
    al_r_m, al_r_s = _mean_std([float(v) for v in ep_toggle_allowed_rate])
    bb_r_m, bb_r_s = _mean_std([float(v) for v in ep_blocked_budget_rate])
    bu_r_m, bu_r_s = _mean_std([float(v) for v in ep_blocked_util_rate])
    bc_r_m, bc_r_s = _mean_std([float(v) for v in ep_blocked_cooldown_rate])
    bos_r_m, bos_r_s = _mean_std([float(v) for v in ep_blocked_global_off_stress_rate])
    ap_r_m, ap_r_s = _mean_std([float(v) for v in ep_toggle_applied_rate])

    mode = "det" if deterministic else "stoch"
    print(f"\n=== Evaluation: {label} ({mode}, {episodes} episodes) ===")
    print(
        f"traffic cfg:    model={env_config.traffic_model} seed={env_config.traffic_seed} "
        f"avg_bursts={env_config.traffic_avg_bursts_per_step} p_elephant={env_config.traffic_p_elephant} "
        f"scenario={env_config.traffic_scenario} "
        f"disable_off_actions={getattr(env_config, 'disable_off_actions', False)} "
        f"initial_off_edges={getattr(env_config, 'initial_off_edges', 0)}"
    )
    print(
        f"edge universe   size_mean={float(np.mean(ep_edge_universe_size)):.2f} "
        f"initial_off_requested_mean={float(np.mean(ep_initial_off_requested)):.2f} "
        f"initial_off_applied_mean={float(np.mean(ep_initial_off_applied)):.2f}"
    )
    print(f"episode_reward: mean={r_m:.3f} std={r_s:.3f}")
    print(f"energy_kwh:     mean={e_m:.6f} std={e_s:.6f}")
    print(f"dropped:        mean={d_m:.3f} std={d_s:.3f}")
    print(f"delivered:      mean={dl_m:.3f} std={dl_s:.3f}")
    print(f"norm_drop:      mean={nd_m:.5f} std={nd_s:.5f}")
    print(f"reward_energy:  mean={re_m:.3f} std={re_s:.3f}")
    print(f"reward_drop:    mean={rd_m:.3f} std={rd_s:.3f}")
    print(f"reward_qos:     mean={rq_m:.3f} std={rq_s:.3f}")
    print(f"reward_toggle:  mean={rt_m:.3f} std={rt_s:.3f}")
    print(f"reward_other:   mean={ro_m:.3f} std={ro_s:.3f}")
    mean_steps = float(np.mean(ep_steps_total)) if ep_steps_total else 1.0
    denom_steps = max(mean_steps, 1.0)
    print(
        "reward parts    "
        f"total={r_m:.3f} energy={re_m:.3f} drop={rd_m:.3f} qos={rq_m:.3f} "
        f"toggle={rt_m:.3f} other={ro_m:.3f} | per_step "
        f"total={r_m/denom_steps:.6f} energy={re_m/denom_steps:.6f} "
        f"drop={rd_m/denom_steps:.6f} qos={rq_m/denom_steps:.6f} "
        f"toggle={rt_m/denom_steps:.6f} other={ro_m/denom_steps:.6f}"
    )
    print(f"reward_toggle_on:  mean={rto_m:.3f} std={rto_s:.3f}")
    print(f"reward_toggle_off: mean={rtf_m:.3f} std={rtf_s:.3f}")
    print(f"reward_toggle_harm: mean={rth_m:.3f} std={rth_s:.3f}")
    print(
        f"delta_dropped_after_toggle mean={float(np.mean(ep_delta_dropped_after_toggle_mean)):.4f}"
    )
    print(
        f"delta_qos_after_toggle mean={float(np.mean(ep_delta_qos_after_toggle_mean)):.6f}"
    )
    print(
        f"pending_toggle_harm mean={float(np.mean(ep_pending_toggle_harm_mean)):.3f}"
    )
    print(
        f"toggle_harm_max_dd mean={float(np.mean(ep_toggle_harm_max_dd_mean)):.6f}"
    )
    print(
        f"toggle_harm_max_dq mean={float(np.mean(ep_toggle_harm_max_dq_mean)):.6f}"
    )
    print(f"qos_violation:  mean={qv_m:.4f} std={qv_s:.4f}")
    print(
        "qos debug       "
        f"qos_viol_steps_mean={qvs_m:.2f} "
        f"qos_allow_on_steps_mean={qas_m:.2f} "
        f"norm_drop_step_mean={nds_m:.6f} "
        f"qos_sum_mean={rq_m:.3f} "
        f"qos_penalty_per_viol_step_mean={qpv_m:.6f}"
    )
    print(
        "cd mask debug   "
        f"dbg_cd_block_mask_val_last_mean={cdmv_m:.2f} "
        f"dbg_cd_block_mask_samples_mean={cdsamp_m:.2f} "
        f"cd_block_mask_mismatch_count_mean={cdmis_m:.2f}"
    )
    print(
        f"toggles applied mean={float(np.mean(ep_applied)):.2f} "
        f"reverted mean={float(np.mean(ep_reverted)):.2f}"
    )
    print(f"toggles total   mean={tt_m:.2f} std={tt_s:.2f} rate={tr_m:.4f}")
    print(
        f"toggles dir     toggled_on mean={float(np.mean(ep_toggled_on)):.2f} "
        f"toggled_off mean={float(np.mean(ep_toggled_off)):.2f}"
    )
    print(
        f"off budget      used mean={float(np.mean(ep_off_toggles_used)):.2f} "
        f"blocked mean={float(np.mean(ep_blocked_off_budget)):.2f}"
    )
    print(
        f"total budget    used mean={float(np.mean(ep_total_toggles_used)):.2f} "
        f"blocked mean={float(np.mean(ep_blocked_total_budget)):.2f}"
    )
    if max_on_edges_budget is not None:
        print(
            f"on-edge budget  max_on_edges={max_on_edges_budget} "
            f"blocked_on_actions_mean={float(np.mean(ep_controller_blocked_on_budget)):.2f}"
        )
    print(
        "toggle gates   "
        f"attempted={ta_m:.2f} ({ta_r_m:.4f}/step) "
        f"allowed={al_m:.2f} ({al_r_m:.4f}/step) "
        f"blocked_budget={bb_m:.2f} ({bb_r_m:.4f}/step) "
        f"blocked_util={bu_m:.2f} ({bu_r_m:.4f}/step) "
        f"blocked_cd={bc_m:.2f} ({bc_r_m:.4f}/step) "
        f"blocked_off_stress={bos_m:.2f} ({bos_r_m:.4f}/step) "
        f"emergency_on_applied={eo_m:.2f} "
        f"max_util_mean={float(np.mean(ep_max_util_mean)):.4f} "
        f"applied={ap_m:.2f} ({ap_r_m:.4f}/step)"
    )
    print(
        "gate counters  "
        f"edge_cd={dbg_edge_cd_m:.2f} global_cd={dbg_global_cd_m:.2f} "
        f"budget={dbg_budget_m:.2f} util={dbg_util_m:.2f} "
        f"off_stress={dbg_off_stress_m:.2f} qos_off={dbg_qos_off_m:.2f} qos_on={dbg_qos_on_m:.2f}"
    )
    print(
        f"mask stats      valid/step mean={float(np.mean(ep_valid_actions_per_step)):.2f} "
        f"decision_steps mean={float(np.mean(ep_decision_steps_count)):.2f} "
        f"valid@decision mean={float(np.mean(ep_valid_actions_on_decision_steps)):.2f} "
        f"mask_calls_mean={float(np.mean(ep_mask_calls)):.2f}"
    )
    print(
        f"mask split      toggles/step={float(np.mean(ep_valid_toggle_actions_per_step)):.2f} "
        f"on/step={float(np.mean(ep_valid_on_actions_per_step)):.2f} "
        f"off/step={float(np.mean(ep_valid_off_actions_per_step)):.2f} "
        f"noop/step={float(np.mean(ep_valid_noop_actions_per_step)):.2f}"
    )
    print(
        f"mask split@dec  toggles={float(np.mean(ep_valid_toggle_actions_on_decision_steps)):.2f} "
        f"on={float(np.mean(ep_valid_on_actions_on_decision_steps)):.2f} "
        f"off={float(np.mean(ep_valid_off_actions_on_decision_steps)):.2f} "
        f"noop={float(np.mean(ep_valid_noop_actions_on_decision_steps)):.2f}"
    )
    print(
        f"valid_on_actions@decision mean={float(np.mean(ep_valid_on_actions_on_decision_steps)):.2f}"
    )
    on_top = sorted(on_reason_totals.items(), key=lambda kv: kv[1], reverse=True)[:5]
    on_top_ex_missing = sorted(
        [(k, v) for (k, v) in on_reason_totals.items() if k != "on_disabled_by_missing_edge"],
        key=lambda kv: kv[1],
        reverse=True,
    )[:5]
    print(f"on blockers    top={on_top}")
    print(f"on blockers    top_ex_missing={on_top_ex_missing}")
    print(
        f"on availability early20 mean={float(np.mean(ep_valid_on_first20_decision_steps_mean)):.2f} "
        f"frac_decision_with_any_on={float(np.mean(ep_fraction_decision_steps_any_on_valid)):.3f} "
        f"noop_chosen@decision={float(np.mean(ep_noop_chosen_on_decision_steps_mean)):.3f}"
    )
    budget_vals = [v for v, ok in zip(ep_toggle_budget_remaining_mean, ep_toggle_budget_remaining_available) if ok]
    budget_text = f"{float(np.mean(budget_vals)):.2f}" if budget_vals else "N/A"
    print(
        f"edge state      on_edges_mean={float(np.mean(ep_on_edges_count_mean)):.2f} "
        f"off_edges_mean={float(np.mean(ep_off_edges_count_mean)):.2f} "
        f"toggle_budget_remaining_mean={budget_text}"
    )
    valid_saturation = [v for v in ep_decision_step_when_off_zero if v > 0]
    sat_mean = float(np.mean(valid_saturation)) if valid_saturation else -1.0
    print(
        f"off saturation  off_edges_first20_dec_mean={float(np.mean(ep_off_edges_first20_decision_steps_mean)):.2f} "
        f"off_edges_all_dec_mean={float(np.mean(ep_off_edges_all_decision_steps_mean)):.2f} "
        f"decision_step_off_zero_mean={sat_mean:.2f}"
    )
    print(f"demand stats:   mean_demand_per_step={float(np.mean(ep_mean_demand_per_step)):.3f}")

    stats = {
        "reward_mean": r_m,
        "reward_std": r_s,
        "energy_mean": e_m,
        "energy_std": e_s,
        "dropped_mean": d_m,
        "dropped_std": d_s,
        "delivered_mean": dl_m,
        "delivered_std": dl_s,
        "norm_drop_mean": nd_m,
        "norm_drop_std": nd_s,
        "toggles_applied_mean": float(np.mean(ep_applied)),
        "toggles_reverted_mean": float(np.mean(ep_reverted)),
        "toggles_total_mean": tt_m,
        "toggles_rate_mean": tr_m,
        "toggles_rate_std": tr_s,
        "qos_violation_rate_mean": qv_m,
        "qos_violation_rate_std": qv_s,
        "qos_violation_steps_mean": qvs_m,
        "qos_violation_steps_std": qvs_s,
        "qos_allow_on_steps_mean": qas_m,
        "qos_allow_on_steps_std": qas_s,
        "norm_drop_step_mean": nds_m,
        "norm_drop_step_std": nds_s,
        "qos_penalty_per_viol_step_mean": qpv_m,
        "qos_penalty_per_viol_step_std": qpv_s,
        "dbg_cd_block_mask_val_last_mean": cdmv_m,
        "dbg_cd_block_mask_val_last_std": cdmv_s,
        "dbg_cd_block_mask_samples_mean": cdsamp_m,
        "dbg_cd_block_mask_samples_std": cdsamp_s,
        "cd_block_mask_mismatch_count_mean": cdmis_m,
        "cd_block_mask_mismatch_count_std": cdmis_s,
        "dbg_block_edge_cd_mean": dbg_edge_cd_m,
        "dbg_block_edge_cd_std": dbg_edge_cd_s,
        "dbg_block_global_cd_mean": dbg_global_cd_m,
        "dbg_block_global_cd_std": dbg_global_cd_s,
        "dbg_block_budget_mean": dbg_budget_m,
        "dbg_block_budget_std": dbg_budget_s,
        "dbg_block_util_mean": dbg_util_m,
        "dbg_block_util_std": dbg_util_s,
        "dbg_block_off_stress_mean": dbg_off_stress_m,
        "dbg_block_off_stress_std": dbg_off_stress_s,
        "dbg_block_qos_off_mean": dbg_qos_off_m,
        "dbg_block_qos_off_std": dbg_qos_off_s,
        "dbg_block_qos_on_mean": dbg_qos_on_m,
        "dbg_block_qos_on_std": dbg_qos_on_s,
        "reward_other_mean": ro_m,
        "reward_other_std": ro_s,
        "mean_episode_steps": float(np.mean(ep_steps_total)) if ep_steps_total else 0.0,
        "reward_toggle_on_mean": rto_m,
        "reward_toggle_on_std": rto_s,
        "reward_toggle_off_mean": rtf_m,
        "reward_toggle_off_std": rtf_s,
        "reward_toggle_harm_mean": rth_m,
        "reward_toggle_harm_std": rth_s,
        "delta_dropped_after_toggle_mean": float(np.mean(ep_delta_dropped_after_toggle_mean)),
        "delta_qos_after_toggle_mean": float(np.mean(ep_delta_qos_after_toggle_mean)),
        "pending_toggle_harm_mean": float(np.mean(ep_pending_toggle_harm_mean)),
        "toggle_harm_max_dd_mean": float(np.mean(ep_toggle_harm_max_dd_mean)),
        "toggle_harm_max_dq_mean": float(np.mean(ep_toggle_harm_max_dq_mean)),
        "toggles_attempted_count_mean": ta_m,
        "toggles_attempted_count_std": ta_s,
        "allowed_toggle_count_mean": al_m,
        "allowed_toggle_count_std": al_s,
        "blocked_by_budget_count_mean": bb_m,
        "blocked_by_budget_count_std": bb_s,
        "blocked_by_util_count_mean": bu_m,
        "blocked_by_util_count_std": bu_s,
        "blocked_by_cooldown_count_mean": bc_m,
        "blocked_by_cooldown_count_std": bc_s,
        "blocked_by_global_off_stress_count_mean": bos_m,
        "blocked_by_global_off_stress_count_std": bos_s,
        "toggles_applied_count_mean": ap_m,
        "toggles_applied_count_std": ap_s,
        "emergency_on_applied_count_mean": eo_m,
        "emergency_on_applied_count_std": eo_s,
        "toggles_attempted_rate_mean": ta_r_m,
        "toggles_attempted_rate_std": ta_r_s,
        "allowed_toggle_rate_mean": al_r_m,
        "allowed_toggle_rate_std": al_r_s,
        "blocked_by_budget_rate_mean": bb_r_m,
        "blocked_by_budget_rate_std": bb_r_s,
        "blocked_by_util_rate_mean": bu_r_m,
        "blocked_by_util_rate_std": bu_r_s,
        "blocked_by_cooldown_rate_mean": bc_r_m,
        "blocked_by_cooldown_rate_std": bc_r_s,
        "blocked_by_global_off_stress_rate_mean": bos_r_m,
        "blocked_by_global_off_stress_rate_std": bos_r_s,
        "toggles_applied_rate_mean": ap_r_m,
        "toggles_applied_rate_std": ap_r_s,
        "toggled_on_mean": float(np.mean(ep_toggled_on)),
        "toggled_off_mean": float(np.mean(ep_toggled_off)),
        "off_toggles_used_mean": float(np.mean(ep_off_toggles_used)),
        "toggle_blocked_off_budget_mean": float(np.mean(ep_blocked_off_budget)),
        "total_toggles_used_mean": float(np.mean(ep_total_toggles_used)),
        "toggle_blocked_total_budget_mean": float(np.mean(ep_blocked_total_budget)),
        "controller_blocked_on_budget_mean": float(np.mean(ep_controller_blocked_on_budget)),
        "mean_valid_actions_per_step": float(np.mean(ep_valid_actions_per_step)),
        "mean_decision_steps_count": float(np.mean(ep_decision_steps_count)),
        "mean_valid_actions_on_decision_steps": float(np.mean(ep_valid_actions_on_decision_steps)),
        "mask_calls_mean": float(np.mean(ep_mask_calls)),
        "mean_valid_toggle_actions_per_step": float(np.mean(ep_valid_toggle_actions_per_step)),
        "mean_valid_on_actions_per_step": float(np.mean(ep_valid_on_actions_per_step)),
        "mean_valid_off_actions_per_step": float(np.mean(ep_valid_off_actions_per_step)),
        "mean_valid_noop_actions_per_step": float(np.mean(ep_valid_noop_actions_per_step)),
        "mean_valid_toggle_actions_on_decision_steps": float(np.mean(ep_valid_toggle_actions_on_decision_steps)),
        "mean_valid_on_actions_on_decision_steps": float(np.mean(ep_valid_on_actions_on_decision_steps)),
        "mean_valid_off_actions_on_decision_steps": float(np.mean(ep_valid_off_actions_on_decision_steps)),
        "mean_valid_noop_actions_on_decision_steps": float(np.mean(ep_valid_noop_actions_on_decision_steps)),
        "mean_demand_per_step": float(np.mean(ep_mean_demand_per_step)),
        "on_blockers_top": dict(on_top),
        "on_blockers_top_ex_missing": dict(on_top_ex_missing),
        "mean_on_edges_count": float(np.mean(ep_on_edges_count_mean)),
        "mean_off_edges_count": float(np.mean(ep_off_edges_count_mean)),
        "mean_max_util": float(np.mean(ep_max_util_mean)),
        "mean_toggle_budget_remaining": (float(np.mean(budget_vals)) if budget_vals else float("nan")),
        "mean_valid_on_actions_on_first20_decision_steps": float(np.mean(ep_valid_on_first20_decision_steps_mean)),
        "fraction_decision_steps_with_any_on_valid": float(np.mean(ep_fraction_decision_steps_any_on_valid)),
        "noop_chosen_on_decision_steps_mean": float(np.mean(ep_noop_chosen_on_decision_steps_mean)),
        "edge_universe_size_mean": float(np.mean(ep_edge_universe_size)),
        "initial_off_requested_mean": float(np.mean(ep_initial_off_requested)),
        "initial_off_applied_mean": float(np.mean(ep_initial_off_applied)),
        "mean_off_edges_over_first20_decision_steps": float(np.mean(ep_off_edges_first20_decision_steps_mean)),
        "mean_off_edges_over_all_decision_steps": float(np.mean(ep_off_edges_all_decision_steps_mean)),
        "decision_step_index_when_off_edges_hits_zero_mean": sat_mean,
    }

    env.close()
    return stats
