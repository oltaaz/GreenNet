from __future__ import annotations

import argparse
import json
import random
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from greennet.env import EnvConfig, GreenNetEnv
from greennet.utils.config import load_env_config_from_run, save_env_config, save_train_config, load_train_config_from_run


from stable_baselines3 import PPO

# Optional: action masking support (recommended).
try:
    from sb3_contrib import MaskablePPO  # type: ignore
    from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
    MASKABLE_AVAILABLE = True
except Exception:  # pragma: no cover
    MaskablePPO = None  # type: ignore
    get_action_masks = None  # type: ignore
    ActionMasker = None  # type: ignore

DEBUG_MASKS = False


class ProgressBarCallback(BaseCallback):
    """Print simple percentage progress with an ASCII bar."""

    def __init__(self, total_timesteps: int, every_steps: int = 2000, bar_len: int = 30, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.every_steps = max(1, int(every_steps))
        self.bar_len = max(5, int(bar_len))
        self._last_step_print = -1

    def _render_line(self, step: int) -> str:
        pct = min(100.0, max(0.0, 100.0 * float(step) / float(self.total_timesteps)))
        filled = int((pct / 100.0) * self.bar_len)
        filled = max(0, min(self.bar_len, filled))
        bar = "=" * filled + "-" * (self.bar_len - filled)
        return f"Progress: {pct:5.1f}% [{bar}] {step}/{self.total_timesteps}"

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step >= self.total_timesteps:
            if self._last_step_print < self.total_timesteps:
                self._last_step_print = self.total_timesteps
                print(self._render_line(self.total_timesteps), end="\r", flush=True)
            return True
        if step - self._last_step_print >= self.every_steps:
            self._last_step_print = step
            print(self._render_line(step), end="\r", flush=True)
        return True

    def _on_training_end(self) -> None:
        print(self._render_line(self.total_timesteps), flush=True)


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "total_timesteps": 20_000,
    "ppo": {
        "policy": "MultiInputPolicy",
        "verbose": 0,
        "device": "auto",
        "n_steps": 2048,
        "batch_size": 256,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "ent_coef": 0.001,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
    },
}

_ENV_TUPLE_FIELDS = {
    "topology_seeds",
    "traffic_mice_size_range",
    "traffic_elephant_size_range",
    "traffic_duration_range",
    "traffic_spike_multiplier_range",
    "traffic_spike_duration_range",
}


def _coerce_env_value(key: str, value: Any) -> Any:
    if key == "traffic_hotspots" and isinstance(value, list):
        return tuple(tuple(item) for item in value)
    if key == "topology_seeds" and isinstance(value, list):
        return tuple(int(v) for v in value)
    if key in _ENV_TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)
    return value


def _extract_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    env_overrides: Dict[str, Any] = {}
    for key in ("env", "env_config", "env_kwargs"):
        block = config.get(key)
        if isinstance(block, dict):
            env_overrides.update(block)

    # Allow top-level seed keys to flow into env config.
    for key in ("topology_seed", "topology_seeds", "topology_randomize", "traffic_seed"):
        if key in config and key not in env_overrides:
            env_overrides[key] = config[key]

    if "traffic_seed_base" in config and "traffic_seed" not in env_overrides:
        env_overrides["traffic_seed"] = config["traffic_seed_base"]

    if is_dataclass(EnvConfig):
        allowed = {f.name for f in fields(EnvConfig)}
        env_overrides = {
            k: _coerce_env_value(k, v)
            for k, v in env_overrides.items()
            if k in allowed
        }
    else:
        env_overrides = {k: _coerce_env_value(k, v) for k, v in env_overrides.items()}

    return env_overrides


def _resolve_seed(config: Dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except Exception:
        return int(fallback)


def default_train_env_config() -> EnvConfig:
    return EnvConfig(
        traffic_model="stochastic",
        # Make energy matter enough to beat NOOP.
        energy_weight=25.0,

        # Drops still matter, but don't drown everything.
        drop_penalty_lambda=40.0,
        normalize_drop=True,

        # QoS penalty: sane + gated.
        qos_target_norm_drop=0.0720,
        qos_violation_penalty_scale=80.0,
        qos_min_volume=500.0,
        qos_guard_margin=0.004,
        qos_guard_penalty_scale=3.0,

        # Let it explore without fear (tune if action thrashing).
        toggle_penalty=0.02,
        off_toggle_penalty_scale=1.5,
        toggle_apply_penalty=0.02,
        toggle_on_penalty_scale=0.2,
        toggle_off_penalty_scale=5.0,
        toggle_attempt_penalty=0.0,
        noop_bonus=0.0,
        debug_logs=False,
        blocked_action_penalty=0.0,

        # Keep action responsiveness but avoid flapping.
        toggle_cooldown_steps=10,
        global_toggle_cooldown_steps=5,
        decision_interval_steps=10,
        max_off_toggles_per_episode=0,
        max_total_toggles_per_episode=10,
        disable_off_actions=False,
        initial_off_edges=10,
        initial_off_seed=123,
        util_block_threshold=0.85,

        traffic_hotspots=((0, 5, 3.0), (2, 7, 2.0)),
        traffic_avg_bursts_per_step=2.0,
        traffic_p_elephant=0.05,
        traffic_elephant_size_range=(6, 20),
        traffic_duration_range=(1, 3),
        traffic_spike_prob=0.0,
        traffic_spike_multiplier_range=(1.0, 1.0),
        traffic_spike_duration_range=(1, 1),
        topology_randomize=True,
        topology_seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        topology_seed=0,
        flows_per_step=6,
        base_capacity=15.0,



        # configs for forecasting
        enable_forecasting=True,
        forecast_alpha=0.6,
        forecast_horizon_steps=3,
        demand_norm_scale=0.0,  # scale factor to keep demand values in a reasonable range for the forecaster

    )


def build_train_env_config(config: Dict[str, Any]) -> EnvConfig:
    env_config = default_train_env_config()

    for key, value in _extract_env_overrides(config).items():
        setattr(env_config, key, value)

    return env_config


def load_config(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_env(seed: int, env_config: EnvConfig):
    def _init():
        # Use a fresh copy of the env config for each vectorized environment.
        env = GreenNetEnv(config=replace(env_config, debug_logs=False))
        env = Monitor(env)
        if MASKABLE_AVAILABLE and ActionMasker is not None:
            env = ActionMasker(env, lambda e: e.unwrapped.get_action_mask())
        if DEBUG_MASKS and MASKABLE_AVAILABLE:
            assert hasattr(env, "action_masks"), f"Masking broken: returned env has no action_masks(), type={type(env)}"
            inner_env = env.unwrapped if hasattr(env, "unwrapped") else env
            if getattr(inner_env, "debug_logs", False) and not getattr(inner_env, "_train_mask_debug_printed", False):
                print("[mask-debug] make_env returned:", type(env), "has action_masks:", hasattr(env, "action_masks"))
                setattr(inner_env, "_train_mask_debug_printed", True)
        env.reset(seed=seed)
        return env

    return _init


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def save_requirements_copy(run_dir: Path) -> None:
    requirements_src = Path("ml-env") / "requirements.txt"
    if requirements_src.exists():
        requirements_dst = run_dir / "requirements.txt"
        requirements_dst.write_text(requirements_src.read_text(encoding="utf-8"), encoding="utf-8")


def find_latest_model() -> Path:
    """Find the most recent runs/*/ppo_greennet.zip model."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        raise FileNotFoundError("runs/ directory not found. Train first or pass --model.")

    candidates = sorted(runs_dir.glob("*/ppo_greennet.zip"))
    if not candidates:
        candidates = sorted(runs_dir.glob("*/ppo_greennet"))
    if not candidates:
        raise FileNotFoundError("No saved model found under runs/*/ppo_greennet(.zip).")

    return candidates[-1]


def parse_seed_list(seed_text: str) -> list[int]:
    """Parse comma-separated integers like '0,1,2,3,4'."""
    parts = [p.strip() for p in seed_text.split(",") if p.strip()]
    return [int(p) for p in parts]


def run_robustness_eval(
    model: Any,
    base_env_config: EnvConfig,
    episodes: int,
    seed: int,
    topology_seeds: list[int],
    out_csv: Path,
    out_png: Path,
) -> None:
    """Evaluate NOOP vs trained(det) across multiple topology seeds.

    IMPORTANT: We keep the action space fixed (from the trained model's base topology) by setting
    topology_randomize=True and evaluating each requested topology via topology_seeds=(s,).
    """
    rows: list[dict[str, Any]] = []

    for s in topology_seeds:
        env_cfg = replace(
            base_env_config,
            topology_randomize=True,
            topology_seeds=(int(s),),
        )

        noop = eval_policy(
            None,
            env_cfg,
            episodes=int(episodes),
            seed=seed,
            label=f"always_noop_seed{s}",
            deterministic=True,
        )
        det = eval_policy(
            model,
            env_cfg,
            episodes=int(episodes),
            seed=seed,
            label=f"trained_seed{s}",
            deterministic=True,
        )

        for policy_name, stats in [("noop", noop), ("trained_det", det)]:
            rows.append({
                "topology_seed": int(s),
                "policy": policy_name,
                **stats,
            })

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = sorted(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in header) + "\n")

    print(f"[robustness] wrote CSV: {out_csv}")

    # Plot: energy vs norm_drop per topology seed
    if plt is None:
        print("[robustness] matplotlib not available; skipping plot.")
        return

    # Build series
    def _series(policy: str):
        xs, ys, labels = [], [], []
        for r in rows:
            if r.get("policy") != policy:
                continue
            xs.append(float(r.get("norm_drop_mean", 0.0)))
            ys.append(float(r.get("energy_mean", 0.0)))
            labels.append(str(r.get("topology_seed")))
        return xs, ys, labels

    x_noop, y_noop, lab_noop = _series("noop")
    x_det, y_det, lab_det = _series("trained_det")

    plt.figure()
    plt.scatter(x_noop, y_noop, label="NOOP (all-on)")
    plt.scatter(x_det, y_det, label="Trained (det)")

    for x, y, t in zip(x_noop, y_noop, lab_noop):
        plt.annotate(t, (x, y))
    for x, y, t in zip(x_det, y_det, lab_det):
        plt.annotate(t, (x, y))

    plt.xlabel("Normalized drop (mean)")
    plt.ylabel("Energy (kWh per episode, mean)")
    plt.title("Robustness across topology seeds")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    print(f"[robustness] wrote plot: {out_png}")


def eval_policy(
    model: Any | None,
    env_config: EnvConfig,
    episodes: int,
    seed: int,
    label: str,
    deterministic: bool,
    debug_energy: bool = False,
    policy_mode: str = "model",
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
    ep_reward_toggle_on: list[float] = []
    ep_reward_toggle_off: list[float] = []
    ep_toggle_total: list[int] = []
    ep_toggle_rate: list[float] = []
    ep_qos_violation_rate: list[float] = []
    ep_toggle_attempted: list[int] = []
    ep_toggle_allowed: list[int] = []
    ep_blocked_util: list[int] = []
    ep_blocked_cooldown: list[int] = []
    ep_toggle_applied_count: list[int] = []
    ep_toggle_attempted_rate: list[float] = []
    ep_toggle_allowed_rate: list[float] = []
    ep_blocked_util_rate: list[float] = []
    ep_blocked_cooldown_rate: list[float] = []
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
    ep_valid_toggle_actions_per_step: list[float] = []
    ep_valid_on_actions_per_step: list[float] = []
    ep_valid_off_actions_per_step: list[float] = []
    ep_valid_noop_actions_per_step: list[float] = []
    ep_valid_toggle_actions_on_decision_steps: list[float] = []
    ep_valid_on_actions_on_decision_steps: list[float] = []
    ep_valid_off_actions_on_decision_steps: list[float] = []
    ep_valid_noop_actions_on_decision_steps: list[float] = []
    ep_mean_demand_per_step: list[float] = []
    ep_on_edges_count_mean: list[float] = []
    ep_off_edges_count_mean: list[float] = []
    ep_toggle_budget_remaining_mean: list[float] = []
    ep_valid_on_first20_decision_steps_mean: list[float] = []
    ep_fraction_decision_steps_any_on_valid: list[float] = []
    ep_noop_chosen_on_decision_steps_mean: list[float] = []
    ep_edge_universe_size: list[float] = []
    ep_initial_off_requested: list[float] = []
    ep_initial_off_applied: list[float] = []
    ep_off_edges_first20_decision_steps_mean: list[float] = []
    ep_off_edges_all_decision_steps_mean: list[float] = []
    ep_decision_step_when_off_zero: list[float] = []
    on_reason_totals: dict[str, int] = {}

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
        blocked_util = 0
        blocked_cooldown = 0
        applied_count = 0
        toggled_on_count = 0
        toggled_off_count = 0
        blocked_off_budget = 0
        off_toggles_used_last = 0
        blocked_total_budget = 0
        total_toggles_used_last = 0
        steps = 0
        decision_steps = 0
        qos_violation_steps = 0
        sum_reward_energy = 0.0
        sum_reward_drop = 0.0
        sum_reward_qos = 0.0
        sum_reward_toggle = 0.0
        sum_reward_toggle_on = 0.0
        sum_reward_toggle_off = 0.0
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
        budget_remaining_vals: list[float] = []
        edge_universe_size_last = 0
        initial_off_requested_last = 0
        initial_off_applied_last = 0
        decision_off_edges_counts: list[int] = []
        decision_step_when_off_zero: int | None = None
        noop_chosen_on_decision_steps = 0
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

            obs, r, terminated, truncated, info = env.step(action)
            steps += 1
            if info.get("qos_violation", False):
                qos_violation_steps += 1
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
            sum_reward_toggle_on += float(info.get("reward_toggle_on", 0.0))
            sum_reward_toggle_off += float(info.get("reward_toggle_off", 0.0))

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
            blocked_util += int(info.get("blocked_by_util_count", 0))
            blocked_cooldown += int(info.get("blocked_by_cooldown_count", 0))
            applied_count += int(info.get("toggles_applied_count", 0))
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
            budget_rem = info.get("toggle_budget_remaining", None)
            if budget_rem is not None:
                budget_remaining_vals.append(float(budget_rem))
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
        ep_reward_toggle_on.append(sum_reward_toggle_on)
        ep_reward_toggle_off.append(sum_reward_toggle_off)
        toggle_total = applied_count
        ep_toggle_total.append(toggle_total)
        if steps > 0:
            ep_toggle_rate.append(toggle_total / float(steps))
        else:
            ep_toggle_rate.append(0.0)
        ep_toggle_attempted.append(attempted)
        ep_toggle_allowed.append(allowed)
        ep_blocked_util.append(blocked_util)
        ep_blocked_cooldown.append(blocked_cooldown)
        ep_toggle_applied_count.append(applied_count)
        ep_toggled_on.append(toggled_on_count)
        ep_toggled_off.append(toggled_off_count)
        ep_blocked_off_budget.append(blocked_off_budget)
        ep_off_toggles_used.append(off_toggles_used_last)
        ep_blocked_total_budget.append(blocked_total_budget)
        ep_total_toggles_used.append(total_toggles_used_last)
        ep_valid_actions_per_step.append(float(np.mean(valid_counts)) if valid_counts else 0.0)
        ep_decision_steps_count.append(int(decision_steps))
        ep_valid_actions_on_decision_steps.append(
            float(np.mean(decision_valid_counts)) if decision_valid_counts else 0.0
        )
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
        ep_on_edges_count_mean.append(float(np.mean(on_edges_vals)) if on_edges_vals else 0.0)
        ep_off_edges_count_mean.append(float(np.mean(off_edges_vals)) if off_edges_vals else 0.0)
        ep_toggle_budget_remaining_mean.append(float(np.mean(budget_remaining_vals)) if budget_remaining_vals else -1.0)
        first20 = decision_valid_on_counts[:20]
        ep_valid_on_first20_decision_steps_mean.append(float(np.mean(first20)) if first20 else 0.0)
        if decision_steps > 0:
            ep_noop_chosen_on_decision_steps_mean.append(float(noop_chosen_on_decision_steps) / float(decision_steps))
        else:
            ep_noop_chosen_on_decision_steps_mean.append(0.0)
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
            ep_blocked_util_rate.append(blocked_util / float(steps))
            ep_blocked_cooldown_rate.append(blocked_cooldown / float(steps))
            ep_toggle_applied_rate.append(applied_count / float(steps))
        else:
            ep_toggle_attempted_rate.append(0.0)
            ep_toggle_allowed_rate.append(0.0)
            ep_blocked_util_rate.append(0.0)
            ep_blocked_cooldown_rate.append(0.0)
            ep_toggle_applied_rate.append(0.0)
        if steps > 0:
            ep_qos_violation_rate.append(qos_violation_steps / float(steps))
        else:
            ep_qos_violation_rate.append(0.0)

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
    rto_m, rto_s = _mean_std(ep_reward_toggle_on)
    rtf_m, rtf_s = _mean_std(ep_reward_toggle_off)
    tt_m, tt_s = _mean_std([float(v) for v in ep_toggle_total])
    tr_m, tr_s = _mean_std([float(v) for v in ep_toggle_rate])
    qv_m, qv_s = _mean_std(ep_qos_violation_rate)
    ta_m, ta_s = _mean_std([float(v) for v in ep_toggle_attempted])
    al_m, al_s = _mean_std([float(v) for v in ep_toggle_allowed])
    bu_m, bu_s = _mean_std([float(v) for v in ep_blocked_util])
    bc_m, bc_s = _mean_std([float(v) for v in ep_blocked_cooldown])
    ap_m, ap_s = _mean_std([float(v) for v in ep_toggle_applied_count])
    ta_r_m, ta_r_s = _mean_std([float(v) for v in ep_toggle_attempted_rate])
    al_r_m, al_r_s = _mean_std([float(v) for v in ep_toggle_allowed_rate])
    bu_r_m, bu_r_s = _mean_std([float(v) for v in ep_blocked_util_rate])
    bc_r_m, bc_r_s = _mean_std([float(v) for v in ep_blocked_cooldown_rate])
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
    print(f"reward_toggle_on:  mean={rto_m:.3f} std={rto_s:.3f}")
    print(f"reward_toggle_off: mean={rtf_m:.3f} std={rtf_s:.3f}")
    print(f"qos_violation:  mean={qv_m:.4f} std={qv_s:.4f}")
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
    print(
        "toggle gates   "
        f"attempted={ta_m:.2f} ({ta_r_m:.4f}/step) "
        f"allowed={al_m:.2f} ({al_r_m:.4f}/step) "
        f"blocked_util={bu_m:.2f} ({bu_r_m:.4f}/step) "
        f"blocked_cd={bc_m:.2f} ({bc_r_m:.4f}/step) "
        f"applied={ap_m:.2f} ({ap_r_m:.4f}/step)"
    )
    print(
        f"mask stats      valid/step mean={float(np.mean(ep_valid_actions_per_step)):.2f} "
        f"decision_steps mean={float(np.mean(ep_decision_steps_count)):.2f} "
        f"valid@decision mean={float(np.mean(ep_valid_actions_on_decision_steps)):.2f}"
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
    print(
        f"edge state      on_edges_mean={float(np.mean(ep_on_edges_count_mean)):.2f} "
        f"off_edges_mean={float(np.mean(ep_off_edges_count_mean)):.2f} "
        f"toggle_budget_remaining_mean={float(np.mean(ep_toggle_budget_remaining_mean)):.2f}"
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
        "reward_toggle_on_mean": rto_m,
        "reward_toggle_on_std": rto_s,
        "reward_toggle_off_mean": rtf_m,
        "reward_toggle_off_std": rtf_s,
        "toggles_attempted_count_mean": ta_m,
        "toggles_attempted_count_std": ta_s,
        "allowed_toggle_count_mean": al_m,
        "allowed_toggle_count_std": al_s,
        "blocked_by_util_count_mean": bu_m,
        "blocked_by_util_count_std": bu_s,
        "blocked_by_cooldown_count_mean": bc_m,
        "blocked_by_cooldown_count_std": bc_s,
        "toggles_applied_count_mean": ap_m,
        "toggles_applied_count_std": ap_s,
        "toggles_attempted_rate_mean": ta_r_m,
        "toggles_attempted_rate_std": ta_r_s,
        "allowed_toggle_rate_mean": al_r_m,
        "allowed_toggle_rate_std": al_r_s,
        "blocked_by_util_rate_mean": bu_r_m,
        "blocked_by_util_rate_std": bu_r_s,
        "blocked_by_cooldown_rate_mean": bc_r_m,
        "blocked_by_cooldown_rate_std": bc_r_s,
        "toggles_applied_rate_mean": ap_r_m,
        "toggles_applied_rate_std": ap_r_s,
        "toggled_on_mean": float(np.mean(ep_toggled_on)),
        "toggled_off_mean": float(np.mean(ep_toggled_off)),
        "off_toggles_used_mean": float(np.mean(ep_off_toggles_used)),
        "toggle_blocked_off_budget_mean": float(np.mean(ep_blocked_off_budget)),
        "total_toggles_used_mean": float(np.mean(ep_total_toggles_used)),
        "toggle_blocked_total_budget_mean": float(np.mean(ep_blocked_total_budget)),
        "mean_valid_actions_per_step": float(np.mean(ep_valid_actions_per_step)),
        "mean_decision_steps_count": float(np.mean(ep_decision_steps_count)),
        "mean_valid_actions_on_decision_steps": float(np.mean(ep_valid_actions_on_decision_steps)),
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
        "mean_toggle_budget_remaining": float(np.mean(ep_toggle_budget_remaining_mean)),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on GreenNet.")
    parser.add_argument("--config", type=Path, help="Path to JSON config to load.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training.")
    parser.add_argument("--eval-noop", action="store_true", help="Run evaluation of the always-noop policy and exit.")
    parser.add_argument("--model", type=Path, help="Path to saved PPO model zip (default: latest under runs/).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument(
        "--debug-eval-energy",
        action="store_true",
        help="Print evaluation energy accumulation diagnostics.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run multi-seed robustness eval (NOOP vs trained(det)) and write CSV+PNG under the model run dir.",
    )
    parser.add_argument(
        "--sanity-eval",
        action="store_true",
        help="Quick sanity eval: print mean drops/energy/toggles/QoS violations for trained vs NOOP.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps for training (ignored for --eval / --eval-noop).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show training progress percentage bar.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress every N timesteps.",
    )
    parser.add_argument(
        "--progress-bar-len",
        type=int,
        default=30,
        help="Width of the ASCII progress bar.",
    )
    parser.add_argument(
        "--train-drop-lambda",
        type=float,
        default=30.0,
        help="Drop penalty lambda used for training environment reward shaping.",
    )
    parser.add_argument(
        "--eval-drop-lambda",
        type=float,
        default=25.0,
        help="Drop penalty lambda applied during --eval to compare policies under the real objective.",
    )
    parser.add_argument(
        "--eval-initial-off-edges",
        type=int,
        default=None,
        help="Override initial_off_edges during --eval for all evaluated policies.",
    )
    parser.add_argument(
        "--eval-disable-off-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override disable_off_actions during --eval for all evaluated policies.",
    )
    parser.add_argument(
        "--eval-max-total-toggles",
        type=int,
        default=None,
        help="Override max_total_toggles_per_episode during --eval for all evaluated policies.",
    )
    parser.add_argument(
        "--eval-traffic-seed",
        type=int,
        default=None,
        help="Override traffic_seed during --eval for all evaluated policies.",
    )
    parser.add_argument(
        "--eval-toggle-on-penalty-scale",
        type=float,
        default=None,
        help="Override toggle_on_penalty_scale during --eval for all evaluated policies.",
    )
    parser.add_argument(
        "--eval-toggle-off-penalty-scale",
        type=float,
        default=None,
        help="Override toggle_off_penalty_scale during --eval for all evaluated policies.",
    )
    parser.add_argument(
        "--eval-energy-weight",
        type=float,
        default=None,
        help="Override energy_weight during --eval for all evaluated policies.",
    )
    parser.add_argument(
        "--eval-normal-no-toggles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convenience: during --eval, set max_total_toggles_per_episode=0 when initial_off_edges==0.",
    )
    parser.add_argument(
        "--topology-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated topology seeds for --robustness (e.g., '0,1,2,3,4').",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Allow overriding training timesteps from CLI.
    if args.timesteps is not None:
        config["total_timesteps"] = int(args.timesteps)

    # Ensure defaults exist if a custom config omits them.
    config.setdefault("ppo", {})
    config["ppo"].setdefault("ent_coef", DEFAULT_CONFIG["ppo"]["ent_coef"])
    config["ppo"].setdefault("n_steps", DEFAULT_CONFIG["ppo"]["n_steps"])

    train_seed = _resolve_seed(config, "train_seed", int(config.get("seed", DEFAULT_CONFIG["seed"])))
    eval_seed = _resolve_seed(config, "eval_seed", train_seed)
    # Keep backwards-compatible key for older configs.
    config.setdefault("seed", train_seed)
    config.setdefault("train_seed", train_seed)
    config.setdefault("eval_seed", eval_seed)

    active_seed = eval_seed if (args.eval or args.eval_noop or args.robustness) else train_seed
    set_seeds(active_seed)

    # ---- Device selection (Apple Silicon GPU via MPS) ----
    # SB3 defaults to "auto" which primarily checks CUDA; it will NOT automatically pick MPS.
    preferred_device = "mps" if torch.backends.mps.is_available() else "cpu"
    config.setdefault("ppo", {})
    try:
        config["ppo"]["verbose"] = min(1, int(config["ppo"].get("verbose", DEFAULT_CONFIG["ppo"]["verbose"])))
    except Exception:
        config["ppo"]["verbose"] = DEFAULT_CONFIG["ppo"]["verbose"]
    # If the user didn't override device in the JSON config, pick MPS when available.
    if config["ppo"].get("device", "auto") in ("auto", None, ""):
        config["ppo"]["device"] = preferred_device

    print(
        f"[device] torch.mps_available={torch.backends.mps.is_available()} "
        f"torch.cuda_available={torch.cuda.is_available()} "
        f"sb3_device={config['ppo'].get('device')}"
    )

    if args.robustness:
        model_path = args.model or find_latest_model()
        print(f"[robustness] using model: {model_path}")

        model_run_dir = Path(model_path).parent
        env_config = load_env_config_from_run(model_run_dir, verbose=True)
        # Load onto the same device policy would use.
        load_device = config.get("ppo", {}).get("device", "cpu")

        if MASKABLE_AVAILABLE and MaskablePPO is not None:
            model = MaskablePPO.load(str(model_path), device=load_device)
        else:
            model = PPO.load(str(model_path), device=load_device)

        print(f"[device] loaded {type(model).__name__} on: {model.policy.device}")

        seeds = parse_seed_list(args.topology_seeds)
        out_csv = model_run_dir / "robustness_eval.csv"
        out_png = model_run_dir / "robustness_energy_vs_drop.png"

        run_robustness_eval(
            model=model,
            base_env_config=env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            topology_seeds=seeds,
            out_csv=out_csv,
            out_png=out_png,
        )
        return

    if args.sanity_eval:
        model_path = args.model or find_latest_model()
        print(f"[sanity] using model: {model_path}")

        model_run_dir = Path(model_path).parent
        env_config = load_env_config_from_run(model_run_dir, verbose=True)
        load_device = config.get("ppo", {}).get("device", "cpu")

        if MASKABLE_AVAILABLE and MaskablePPO is not None:
            model = MaskablePPO.load(str(model_path), device=load_device)
        else:
            model = PPO.load(str(model_path), device=load_device)

        trained = eval_policy(
            model,
            env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            label="trained",
            deterministic=True,
            debug_energy=args.debug_eval_energy,
        )
        noop = eval_policy(
            None,
            env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            label="always_noop",
            deterministic=True,
            debug_energy=args.debug_eval_energy,
        )

        def _print_sanity(label: str, stats: Dict[str, float]) -> None:
            print(
                f"[sanity] {label}: drops_mean={stats['dropped_mean']:.3f} "
                f"energy_mean={stats['energy_mean']:.6f} "
                f"toggles_mean={stats['toggles_total_mean']:.2f} "
                f"toggles_rate={stats.get('toggles_rate_mean', 0.0):.4f} "
                f"qos_violation_mean={stats['qos_violation_rate_mean']:.4f}"
            )
            print(
                f"[sanity] {label}: attempted={stats.get('toggles_attempted_count_mean', 0.0):.2f} "
                f"({stats.get('toggles_attempted_rate_mean', 0.0):.4f}/step) "
                f"allowed={stats.get('allowed_toggle_count_mean', 0.0):.2f} "
                f"({stats.get('allowed_toggle_rate_mean', 0.0):.4f}/step) "
                f"blocked_util={stats.get('blocked_by_util_count_mean', 0.0):.2f} "
                f"({stats.get('blocked_by_util_rate_mean', 0.0):.4f}/step) "
                f"blocked_cd={stats.get('blocked_by_cooldown_count_mean', 0.0):.2f} "
                f"({stats.get('blocked_by_cooldown_rate_mean', 0.0):.4f}/step) "
                f"applied={stats.get('toggles_applied_count_mean', 0.0):.2f} "
                f"({stats.get('toggles_applied_rate_mean', 0.0):.4f}/step)"
            )

        _print_sanity("trained", trained)
        _print_sanity("noop", noop)

        d_energy = trained["energy_mean"] - noop["energy_mean"]
        d_drop = trained["dropped_mean"] - noop["dropped_mean"]
        print(f"[sanity] Δenergy={d_energy:+.6f} Δdrops={d_drop:+.3f}")
        return

    if args.eval_noop:
        if args.model:
            model_run_dir = Path(args.model).parent
            env_config = load_env_config_from_run(model_run_dir, verbose=True)
        else:
            env_config = EnvConfig()
            print("[eval_noop] using EnvConfig defaults (no model provided).")
        eval_policy(
            None,
            env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            label="always_noop",
            deterministic=True,
            debug_energy=args.debug_eval_energy,
        )
        return

    if args.eval:
        model_path = args.model or find_latest_model()
        print(f"Evaluating model: {model_path}")

        model_run_dir = Path(model_path).parent
        env_config = load_env_config_from_run(model_run_dir, verbose=True)
        env_config = replace(env_config, drop_penalty_lambda=float(args.eval_drop_lambda), debug_logs=False)
        if args.eval_initial_off_edges is not None:
            env_config = replace(env_config, initial_off_edges=int(args.eval_initial_off_edges))
        if bool(args.eval_disable_off_actions):
            env_config = replace(env_config, disable_off_actions=True)
        if args.eval_max_total_toggles is not None:
            env_config = replace(env_config, max_total_toggles_per_episode=int(args.eval_max_total_toggles))
        if args.eval_traffic_seed is not None:
            env_config = replace(env_config, traffic_seed=int(args.eval_traffic_seed))
        if args.eval_toggle_on_penalty_scale is not None:
            env_config = replace(env_config, toggle_on_penalty_scale=float(args.eval_toggle_on_penalty_scale))
        if args.eval_toggle_off_penalty_scale is not None:
            env_config = replace(env_config, toggle_off_penalty_scale=float(args.eval_toggle_off_penalty_scale))
        if args.eval_energy_weight is not None:
            env_config = replace(env_config, energy_weight=float(args.eval_energy_weight))
        if bool(args.eval_normal_no_toggles) and int(getattr(env_config, "initial_off_edges", 0)) == 0:
            env_config = replace(env_config, max_total_toggles_per_episode=0)
        print(
            "[eval] using env_config keys: "
            f"{sorted(env_config.__dict__.keys())}"
        )
        print(
            f"EVAL MODE: disable_off_actions={bool(getattr(env_config, 'disable_off_actions', False))} "
            f"initial_off_edges={int(getattr(env_config, 'initial_off_edges', 0))} "
            f"max_total_toggles={int(getattr(env_config, 'max_total_toggles_per_episode', 0))} "
            f"toggle_on_penalty_scale={float(getattr(env_config, 'toggle_on_penalty_scale', 0.0)):.6f} "
            f"toggle_off_penalty_scale={float(getattr(env_config, 'toggle_off_penalty_scale', 0.0)):.6f} "
            f"energy_weight={float(getattr(env_config, 'energy_weight', 0.0)):.6f}"
        )
        train_cfg = load_train_config_from_run(model_run_dir, verbose=False)
        if train_cfg:
            print(f"[eval] loaded train_config keys: {sorted(train_cfg.keys())}")
        else:
            print("[eval] no train_config found; using defaults for PPO keys.")

        load_device = config.get("ppo", {}).get("device", "cpu")
        if MASKABLE_AVAILABLE and MaskablePPO is not None:
            model = MaskablePPO.load(str(model_path), device=load_device)
        else:
            model = PPO.load(str(model_path), device=load_device)
        print(f"[device] loaded {type(model).__name__} on: {model.policy.device}")

        trained_det = eval_policy(
            model,
            env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            label="trained",
            deterministic=True,
            debug_energy=args.debug_eval_energy,
            policy_mode="model",
        )
        trained_stoch = eval_policy(
            model,
            env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            label="trained",
            deterministic=False,
            debug_energy=args.debug_eval_energy,
            policy_mode="model",
        )
        random_masked_stoch = eval_policy(
            model,
            env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            label="random_masked",
            deterministic=False,
            debug_energy=args.debug_eval_energy,
            policy_mode="random_masked",
        )
        noop_det = eval_policy(
            None,
            env_config,
            episodes=int(args.episodes),
            seed=eval_seed,
            label="always_noop",
            deterministic=True,
            debug_energy=args.debug_eval_energy,
            policy_mode="noop",
        )

        d_reward = trained_det["reward_mean"] - noop_det["reward_mean"]
        d_energy = trained_det["energy_mean"] - noop_det["energy_mean"]
        d_dropped = trained_det["dropped_mean"] - noop_det["dropped_mean"]
        better = "YES" if d_reward > 0 else "NO"
        print(
            f"\n[summary] trained(det) vs noop: better={better} "
            f"Δreward={d_reward:+.3f} Δenergy={d_energy:+.6f} Δdropped={d_dropped:+.3f}"
        )

        d_reward_s = trained_stoch["reward_mean"] - noop_det["reward_mean"]
        d_energy_s = trained_stoch["energy_mean"] - noop_det["energy_mean"]
        d_dropped_s = trained_stoch["dropped_mean"] - noop_det["dropped_mean"]
        better_s = "YES" if d_reward_s > 0 else "NO"
        print(
            f"[summary] trained(stoch) vs noop: better={better_s} "
            f"Δreward={d_reward_s:+.3f} Δenergy={d_energy_s:+.6f} Δdropped={d_dropped_s:+.3f}"
        )
        d_reward_r = random_masked_stoch["reward_mean"] - noop_det["reward_mean"]
        d_energy_r = random_masked_stoch["energy_mean"] - noop_det["energy_mean"]
        d_dropped_r = random_masked_stoch["dropped_mean"] - noop_det["dropped_mean"]
        better_r = "YES" if d_reward_r > 0 else "NO"
        print(
            f"[summary] random_masked(stoch) vs noop: better={better_r} "
            f"Δreward={d_reward_r:+.3f} Δenergy={d_energy_r:+.6f} Δdropped={d_dropped_r:+.3f}"
        )
        if bool(args.eval_disable_off_actions):
            cond_initial = float(trained_stoch.get("initial_off_applied_mean", 0.0)) > 0.0
            cond_on_valid = float(trained_stoch.get("mean_valid_on_actions_on_decision_steps", 0.0)) > 0.0
            cond_off_zero = abs(float(trained_stoch.get("toggled_off_mean", 0.0))) <= 1e-9
            passed = cond_initial and cond_on_valid and cond_off_zero
            failed = []
            if not cond_initial:
                failed.append("initial_off_applied_mean<=0")
            if not cond_on_valid:
                failed.append("valid_on_actions@decision_mean<=0")
            if not cond_off_zero:
                failed.append("toggled_off_mean!=0")
            if passed:
                print("[demo-check] PASS")
            else:
                print(f"[demo-check] FAIL: {', '.join(failed)}")

        return

    # --- Training run setup (only when not evaluating) ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    save_train_config(run_dir, config)
    save_requirements_copy(run_dir)

    env_config = build_train_env_config(config)
    env_config = replace(env_config, drop_penalty_lambda=float(args.train_drop_lambda), debug_logs=False)

    save_env_config(run_dir, env_config)
    print(f"[train] saved env_config keys: {sorted(env_config.__dict__.keys())}")
    print(f"[train] saved train_config keys: {sorted(config.keys())}")

    env = DummyVecEnv([make_env(train_seed, env_config)])

    if MASKABLE_AVAILABLE and MaskablePPO is not None:
        model = MaskablePPO(env=env, **config["ppo"])
    else:
        model = PPO(env=env, **config["ppo"])

    print(f"[device] training {type(model).__name__} on: {model.policy.device}")

    class ActionEffectivenessCallback(BaseCallback):
        def __init__(self, log_every: int = 5000):
            super().__init__()
            self.log_every = log_every
            self.applied = 0
            self.reverted = 0
            self.noop = 0
            self.invalid = 0
            self.blocked = 0
            self.blocked_global = 0
            self.blocked_high_util = 0
            self.blocked_cooldown = 0

        def _on_training_start(self) -> None:
            self.applied = 0
            self.reverted = 0
            self.noop = 0
            self.invalid = 0
            self.blocked = 0
            self.blocked_global = 0
            self.blocked_high_util = 0
            self.blocked_cooldown = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                applied = bool(info.get("toggle_applied"))
                reverted = bool(info.get("toggle_reverted"))
                nooped = bool(info.get("action_is_noop"))
                invalid = bool(info.get("action_is_invalid"))
                b_any = bool(info.get("toggle_blocked_any"))
                b_global = bool(info.get("toggle_blocked_global_cooldown"))
                b_util = bool(info.get("toggle_blocked_high_util"))
                b_cd = bool(info.get("toggle_blocked_cooldown"))

                self.applied += int(applied)
                self.reverted += int(reverted)
                self.noop += int(nooped)
                self.invalid += int(invalid)
                self.blocked += int(b_any)
                self.blocked_global += int(b_global)
                self.blocked_high_util += int(b_util)
                self.blocked_cooldown += int(b_cd)


            total = self.applied + self.reverted + self.noop + self.invalid + self.blocked
            if total > 0 and self.num_timesteps % self.log_every == 0:
                applied_frac = self.applied / total
                reverted_frac = self.reverted / total
                noop_frac = self.noop / total
                invalid_frac = self.invalid / total
                blocked_frac = self.blocked / total
                bg = (self.blocked_global / self.blocked) if self.blocked > 0 else 0.0
                bu = (self.blocked_high_util / self.blocked) if self.blocked > 0 else 0.0
                bc = (self.blocked_cooldown / self.blocked) if self.blocked > 0 else 0.0
                if int(config.get("ppo", {}).get("verbose", 0)) > 0:
                    print(
                        f"[action_stats] steps={self.num_timesteps} "
                        f"applied={applied_frac:.3f} reverted={reverted_frac:.3f} "
                        f"noop={noop_frac:.3f} invalid={invalid_frac:.3f} blocked={blocked_frac:.3f} "
                        f"(blocked: global={bg:.2f} util={bu:.2f} cooldown={bc:.2f})"
                    )
                self.logger.record("action_stats/applied_frac", applied_frac)
                self.logger.record("action_stats/reverted_frac", reverted_frac)
                self.logger.record("action_stats/noop_frac", noop_frac)
                self.logger.record("action_stats/invalid_frac", invalid_frac)
                self.logger.record("action_stats/blocked_frac", blocked_frac)
                self.logger.record("action_stats/blocked_global_share", bg)
                self.logger.record("action_stats/blocked_util_share", bu)
                self.logger.record("action_stats/blocked_cooldown_share", bc)
            return True

    total_timesteps = int(config["total_timesteps"])
    callbacks: list[BaseCallback] = [ActionEffectivenessCallback(log_every=5000)]
    if bool(args.progress):
        callbacks.append(
            ProgressBarCallback(
                total_timesteps=total_timesteps,
                every_steps=int(args.progress_every),
                bar_len=int(args.progress_bar_len),
            )
        )
    callback: BaseCallback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    model.save(str(run_dir / "ppo_greennet"))
    print(f"[algo] {type(model).__name__}")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
