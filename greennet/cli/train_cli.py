from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

try:
    from sb3_contrib import MaskablePPO  # type: ignore
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
    MASKABLE_AVAILABLE = True
except Exception:  # pragma: no cover
    MaskablePPO = None  # type: ignore
    ActionMasker = None  # type: ignore
    MASKABLE_AVAILABLE = False

from greennet.env import EnvConfig, GreenNetEnv
from greennet.utils.config import (
    load_env_config_from_run,
    resolve_env_paths_in_config,
    save_env_config,
    save_train_config,
    load_train_config_from_run,
)

from greennet.rl.eval import (
    EVAL_TOLS,
    TRACK_TOLS,
    custom_gate,
    eval_policy,
    parse_seed_list,
    print_model_artifact_info,
)
from greennet.rl.robustness import run_robustness_eval
from greennet.rl.sweep import run_sweep

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
        "ent_coef": 0.0,
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
    for key in (
        "topology_seed",
        "topology_seeds",
        "topology_randomize",
        "topology_name",
        "topology_path",
        "traffic_model",
        "traffic_seed",
        "traffic_name",
        "traffic_path",
        "traffic_scenario",
        "traffic_scenario_version",
        "traffic_scenario_intensity",
        "traffic_scenario_duration",
        "traffic_scenario_frequency",
        "qos_target_norm_drop",
        "qos_min_volume",
        "qos_violation_penalty_scale",
        "qos_guard_margin",
        "qos_guard_margin_off",
        "qos_guard_margin_on",
        "qos_guard_penalty_scale",
        "qos_avg_delay_guard_multiplier",
        "qos_avg_delay_guard_margin_ms",
        "qos_recovery_delay_multiplier",
        "qos_recovery_delay_guard_margin_ms",
        "qos_p95_delay_threshold_ms",
        "stability_reversal_window_steps",
        "stability_reversal_penalty",
        "stability_min_steps_for_assessment",
        "stability_max_transition_rate",
        "stability_max_flap_rate",
        "stability_max_flap_count",
        "power_network_fixed_watts",
        "power_device_active_watts",
        "power_device_sleep_watts",
        "power_device_dynamic_watts",
        "power_link_active_watts",
        "power_link_sleep_watts",
        "power_link_dynamic_watts",
        "power_utilization_sensitive",
        "power_transition_on_joules",
        "power_transition_off_joules",
        "carbon_base_intensity_g_per_kwh",
        "carbon_amplitude_g_per_kwh",
        "carbon_period_seconds",
    ):
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
        energy_weight=240.0,

        # Drops still matter, but don't drown everything.
        drop_penalty_lambda=50.0,
        normalize_drop=True,

        # QoS penalty: sane + gated.
        qos_target_norm_drop=0.0720,
        qos_violation_penalty_scale=120.0,
        qos_min_volume=500.0,
        qos_guard_margin=0.002,
        qos_guard_margin_off=0.006,
        qos_guard_margin_on=0.002,
        qos_guard_penalty_scale=4.5,

        # Let it explore without fear (tune if action thrashing).
        toggle_penalty=0.001,
        off_toggle_penalty_scale=0.20,
        toggle_apply_penalty=0.002,
        toggle_on_penalty_scale=0.30,
        qos_toggle_discount_on=0.25,
        calm_toggle_multiplier_off=2.0,
        toggle_off_penalty_scale=0.20,
        toggle_attempt_penalty=0.001,
        noop_bonus=0.0,
        debug_logs=False,
        blocked_action_penalty=0.0,

        # Keep action responsiveness but avoid flapping.
        toggle_cooldown_steps=8,
        global_toggle_cooldown_steps=3,
        decision_interval_steps=10,
        max_off_toggles_per_episode=2,
        max_total_toggles_per_episode=4,
        max_emergency_on_toggles_per_episode=8,
        disable_off_actions=False,
        initial_off_edges=3,
        initial_off_seed=123,
        off_start_guard_decision_steps=3,  # block OFF actions for first N decision steps when starting all-on
        util_block_threshold=0.80,
        max_util_off_allow_threshold=0.80,
        util_unblock_threshold=0.95,

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

        # Forecasting config
        enable_forecasting=True,
        forecast_model="ema",
        forecast_alpha=0.6,
        forecast_beta=0.2,
        forecast_trend_damping=0.9,
        forecast_adaptive_alphas=(0.1, 0.2, 0.4, 0.6, 0.8, 0.95),
        forecast_adaptive_error_alpha=0.02,
        forecast_adaptive_temperature=0.25,
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
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return resolve_env_paths_in_config(config, base_dir=config_path.parent)


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

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on GreenNet.")
    parser.add_argument("--config", type=Path, help="Path to JSON config to load.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training.")
    parser.add_argument("--eval-noop", action="store_true", help="Run evaluation of the always-noop policy and exit.")
    parser.add_argument("--sweep", action="store_true", help="Run reward-shaping Pareto sweep and exit.")
    parser.add_argument("--model", type=Path, help="Path to saved PPO model zip (default: latest under runs/).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--sweep_timesteps", type=int, default=150000, help="Timesteps per sweep candidate.")
    parser.add_argument("--sweep_episodes", type=int, default=30, help="Evaluation episodes per sweep candidate.")
    parser.add_argument("--sweep-samples", type=int, default=10, help="Number of sweep candidates to run (random sample).")
    parser.add_argument(
        "--debug-eval-energy",
        action="store_true",
        help="Print evaluation energy running-total diagnostics.",
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
        default=60.0,
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
        "--eval-traffic-scenario",
        type=str,
        default=None,
        help="Override traffic_scenario during --eval for all evaluated policies (e.g. normal, burst, hotspot, none).",
    )
    parser.add_argument(
        "--eval-topology-seeds",
        type=str,
        default=None,
        help="Comma-separated topology seeds used ONLY during --eval, e.g. 10,11,12.",
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
        "--eval-max-on-edges",
        type=int,
        default=None,
        help="Inference-time controller: block ON actions when active on-edges already reach this budget during --eval.",
    )
    parser.add_argument(
        "--eval-normal-no-toggles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convenience: during --eval, set max_total_toggles_per_episode=0 when initial_off_edges==0.",
    )
    parser.add_argument(
        "--eval-two-track",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run canonical two-track eval: NORMAL_STABILITY and NORMAL_CAPABILITY.",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=None,
        help="Optional CSV path to append two-track eval rows.",
    )
    parser.add_argument(
        "--topology-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated topology seeds for --robustness (e.g., '0,1,2,3,4').",
    )
    args = parser.parse_args()

    eval_topology_seeds: tuple[int, ...] | None = None
    if args.eval_topology_seeds:
        parsed_eval_topology_seeds = parse_seed_list(args.eval_topology_seeds)
        if parsed_eval_topology_seeds:
            eval_topology_seeds = tuple(int(s) for s in parsed_eval_topology_seeds)
    eval_traffic_scenario_override = args.eval_traffic_scenario is not None
    eval_traffic_scenario: str | None = None
    if eval_traffic_scenario_override:
        scenario_text = str(args.eval_traffic_scenario).strip()
        if scenario_text.lower() in {"", "none", "null", "off"}:
            eval_traffic_scenario = None
        else:
            eval_traffic_scenario = scenario_text

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

    active_seed = eval_seed if (args.eval or args.eval_two_track or args.eval_noop or args.robustness) else train_seed
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

    if args.sweep:
        run_sweep(
            config=config,
            train_seed=train_seed,
            eval_seed=eval_seed,
            sweep_timesteps=int(args.sweep_timesteps),
            sweep_episodes=int(args.sweep_episodes),
            train_drop_lambda=float(args.train_drop_lambda),
            eval_drop_lambda=float(args.eval_drop_lambda),
            sweep_samples=int(args.sweep_samples),
            progress_every=int(args.progress_every),
        )
        return

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

    if args.eval_two_track:
        model_path = args.model or find_latest_model()
        print(f"Evaluating model (two-track): {model_path}")

        model_run_dir = Path(model_path).parent
        base_env_config = load_env_config_from_run(model_run_dir, verbose=True)
        base_env_config = replace(base_env_config, drop_penalty_lambda=float(args.eval_drop_lambda), debug_logs=False)
        if eval_topology_seeds is not None:
            base_env_config = replace(
                base_env_config,
                topology_randomize=True,
                topology_seeds=eval_topology_seeds,
            )
        if args.eval_traffic_seed is not None:
            base_env_config = replace(base_env_config, traffic_seed=int(args.eval_traffic_seed))
        if eval_traffic_scenario_override:
            base_env_config = replace(base_env_config, traffic_scenario=eval_traffic_scenario)
        if args.eval_toggle_on_penalty_scale is not None:
            base_env_config = replace(base_env_config, toggle_on_penalty_scale=float(args.eval_toggle_on_penalty_scale))
        if args.eval_toggle_off_penalty_scale is not None:
            base_env_config = replace(base_env_config, toggle_off_penalty_scale=float(args.eval_toggle_off_penalty_scale))
        if args.eval_energy_weight is not None:
            base_env_config = replace(base_env_config, energy_weight=float(args.eval_energy_weight))

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
        def _track_tols(track_name: str) -> Dict[str, float]:
            return dict(TRACK_TOLS.get(track_name, EVAL_TOLS))

        def _run_track(track_name: str, cfg: EnvConfig) -> Dict[str, float]:
            tols = _track_tols(track_name)
            print(
                f"\nEVAL TRACK: {track_name} "
                f"(initial_off_edges={int(getattr(cfg, 'initial_off_edges', 0))}, "
                f"max_total_toggles={int(getattr(cfg, 'max_total_toggles_per_episode', 0))})"
            )
            print(
                f"SUMMARY TOLERANCES ({track_name}): "
                f"reward>=-{float(tols['reward']):.3f} "
                f"dropped<=+{float(tols['dropped']):.3f} "
                f"|energy|<={float(tols['energy']):.6f}"
            )
            trained_stoch = eval_policy(
                model,
                cfg,
                episodes=int(args.episodes),
                seed=eval_seed,
                label="trained",
                deterministic=False,
                debug_energy=args.debug_eval_energy,
                policy_mode="model",
            )
            noop_det = eval_policy(
                None,
                cfg,
                episodes=int(args.episodes),
                seed=eval_seed,
                label="always_noop",
                deterministic=True,
                debug_energy=args.debug_eval_energy,
                policy_mode="noop",
            )
            return {
                "delta_reward": float(trained_stoch["reward_mean"] - noop_det["reward_mean"]),
                "delta_energy": float(trained_stoch["energy_mean"] - noop_det["energy_mean"]),
                "delta_dropped": float(trained_stoch["dropped_mean"] - noop_det["dropped_mean"]),
                "toggles_applied_mean": float(trained_stoch.get("toggles_applied_mean", 0.0)),
                "toggled_on_mean": float(trained_stoch.get("toggled_on_mean", 0.0)),
                "toggled_off_mean": float(trained_stoch.get("toggled_off_mean", 0.0)),
            }

        stability_cfg = replace(
            base_env_config,
            initial_off_edges=0,
        )
        capability_cfg = replace(
            base_env_config,
            initial_off_edges=3,
            max_total_toggles_per_episode=3,
            disable_off_actions=True,
        )
        stability = _run_track("NORMAL_STABILITY", stability_cfg)
        capability = _run_track("NORMAL_CAPABILITY", capability_cfg)

        print(
            f"\n[two-track] seed={int(getattr(base_env_config, 'traffic_seed', -1))} eps={int(args.episodes)} | "
            f"STABILITY: Δreward={stability['delta_reward']:+.3f}, Δenergy={stability['delta_energy']:+.6f}, Δdropped={stability['delta_dropped']:+.3f} | "
            f"CAPABILITY: Δreward={capability['delta_reward']:+.3f}, Δenergy={capability['delta_energy']:+.6f}, Δdropped={capability['delta_dropped']:+.3f}, off_disabled=1"
        )

        if args.eval_out is not None:
            out_path = Path(args.eval_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not out_path.exists()
            with out_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        [
                            "timestamp_utc",
                            "model_path",
                            "traffic_seed",
                            "episodes",
                            "stability_delta_reward",
                            "stability_delta_energy",
                            "stability_delta_dropped",
                            "capability_delta_reward",
                            "capability_delta_energy",
                            "capability_delta_dropped",
                            "capability_toggles_applied_mean",
                            "capability_toggled_on_mean",
                            "capability_toggled_off_mean",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        str(model_path),
                        int(getattr(base_env_config, "traffic_seed", -1)),
                        int(args.episodes),
                        f"{stability['delta_reward']:.6f}",
                        f"{stability['delta_energy']:.6f}",
                        f"{stability['delta_dropped']:.6f}",
                        f"{capability['delta_reward']:.6f}",
                        f"{capability['delta_energy']:.6f}",
                        f"{capability['delta_dropped']:.6f}",
                        f"{capability['toggles_applied_mean']:.6f}",
                        f"{capability['toggled_on_mean']:.6f}",
                        f"{capability['toggled_off_mean']:.6f}",
                    ]
                )
            print(f"[two-track] appended CSV row: {out_path}")
        return

    if args.eval:
        model_path = args.model or find_latest_model()
        print(f"Evaluating model: {model_path}")
        print_model_artifact_info(Path(model_path))

        model_run_dir = Path(model_path).parent
        env_config = load_env_config_from_run(model_run_dir, verbose=True)
        env_config = replace(env_config, drop_penalty_lambda=float(args.eval_drop_lambda), debug_logs=False)
        if eval_topology_seeds is not None:
            env_config = replace(
                env_config,
                topology_randomize=True,
                topology_seeds=eval_topology_seeds,
            )
        if args.eval_initial_off_edges is not None:
            env_config = replace(env_config, initial_off_edges=int(args.eval_initial_off_edges))
        if bool(args.eval_disable_off_actions):
            env_config = replace(env_config, disable_off_actions=True)
        if args.eval_max_total_toggles is not None:
            env_config = replace(env_config, max_total_toggles_per_episode=int(args.eval_max_total_toggles))
        if args.eval_traffic_seed is not None:
            env_config = replace(env_config, traffic_seed=int(args.eval_traffic_seed))
        if eval_traffic_scenario_override:
            env_config = replace(env_config, traffic_scenario=eval_traffic_scenario)
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
            f"traffic_scenario={getattr(env_config, 'traffic_scenario', None)} "
            f"toggle_on_penalty_scale={float(getattr(env_config, 'toggle_on_penalty_scale', 0.0)):.6f} "
            f"toggle_off_penalty_scale={float(getattr(env_config, 'toggle_off_penalty_scale', 0.0)):.6f} "
            f"energy_weight={float(getattr(env_config, 'energy_weight', 0.0)):.6f} "
            f"eval_max_on_edges={args.eval_max_on_edges}"
        )
        eval_initial_off_edges = int(getattr(env_config, "initial_off_edges", 0))
        eval_max_total_toggles = int(getattr(env_config, "max_total_toggles_per_episode", 0))
        if eval_initial_off_edges == 0:
            eval_track = "NORMAL_STABILITY"
        elif eval_initial_off_edges == 3 and eval_max_total_toggles == 3:
            eval_track = "NORMAL_CAPABILITY"
        else:
            eval_track = "CUSTOM"
        eval_track_tols = TRACK_TOLS.get(eval_track, EVAL_TOLS)
        print(
            f"SUMMARY TOLERANCES ({eval_track}): "
            f"reward>=-{float(eval_track_tols['reward']):.3f} "
            f"dropped<=+{float(eval_track_tols['dropped']):.3f} "
            f"|energy|<={float(eval_track_tols['energy']):.6f}"
        )
        print(
            f"EVAL TRACK: {eval_track} "
            f"(initial_off_edges={eval_initial_off_edges}, max_total_toggles={eval_max_total_toggles})"
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
            eval_max_on_edges=args.eval_max_on_edges,
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
            eval_max_on_edges=args.eval_max_on_edges,
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
            eval_max_on_edges=args.eval_max_on_edges,
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
            eval_max_on_edges=args.eval_max_on_edges,
        )

        def _better_with_tolerance(
            delta_reward: float,
            delta_energy: float,
            delta_dropped: float,
            track_name: str,
        ) -> tuple[str, str]:
            if str(track_name).upper() == "CUSTOM":
                if int(eval_initial_off_edges) > 0:
                    dropped_cap = 1.0
                    energy_cap = 0.15 if float(delta_dropped) <= -1.0 else 0.02
                    ok_dropped = float(delta_dropped) <= dropped_cap
                    ok_energy = float(delta_energy) <= energy_cap
                    ok_custom_offstart = bool(ok_dropped and ok_energy)
                    if ok_custom_offstart:
                        return "YES", "pass: offstart(dropped+energy)"

                    failures: list[str] = []
                    if not ok_dropped:
                        failures.append(f"dropped({float(delta_dropped):+.3f}>+{dropped_cap:.3f})")
                    if not ok_energy:
                        failures.append(f"energy({float(delta_energy):+.6f}>+{energy_cap:.6f})")
                    return "NO", "fail: " + ", ".join(failures)

                ok_custom, reason_custom = custom_gate(delta_reward, delta_energy, delta_dropped)
                return ("YES" if ok_custom else "NO"), reason_custom

            tols = TRACK_TOLS.get(track_name, EVAL_TOLS)
            ok_reward = float(delta_reward) >= -float(tols["reward"])
            ok_dropped = float(delta_dropped) <= float(tols["dropped"])
            # Stability gate should only fail on energy regression (worse than NOOP), not improvement.
            ok_energy = float(delta_energy) <= float(tols["energy"])
            ok_all = bool(ok_reward and ok_dropped and ok_energy)

            if ok_all:
                reason = "ok"
            else:
                failures: list[str] = []
                if not ok_reward:
                    failures.append(
                        f"reward({float(delta_reward):+.3f}<{-float(tols['reward']):+.3f})"
                    )
                if not ok_dropped:
                    failures.append(
                        f"dropped({float(delta_dropped):+.3f}>+{float(tols['dropped']):.3f})"
                    )
                if not ok_energy:
                    failures.append(
                        f"energy({float(delta_energy):+.6f}>+{float(tols['energy']):.6f})"
                    )
                reason = "fail: " + ", ".join(failures)

            better = "YES" if ok_all else "NO"
            return better, reason

        d_reward = trained_det["reward_mean"] - noop_det["reward_mean"]
        d_energy = trained_det["energy_mean"] - noop_det["energy_mean"]
        d_dropped = trained_det["dropped_mean"] - noop_det["dropped_mean"]
        better, reason = _better_with_tolerance(d_reward, d_energy, d_dropped, eval_track)
        print(
            f"\n[summary:{eval_track}] trained(det) vs noop: better={better} "
            f"Δreward={d_reward:+.3f} Δenergy={d_energy:+.6f} Δdropped={d_dropped:+.3f} "
            f"({reason})"
        )

        d_reward_s = trained_stoch["reward_mean"] - noop_det["reward_mean"]
        d_energy_s = trained_stoch["energy_mean"] - noop_det["energy_mean"]
        d_dropped_s = trained_stoch["dropped_mean"] - noop_det["dropped_mean"]
        better_s, reason_s = _better_with_tolerance(d_reward_s, d_energy_s, d_dropped_s, eval_track)
        print(
            f"[summary:{eval_track}] trained(stoch) vs noop: better={better_s} "
            f"Δreward={d_reward_s:+.3f} Δenergy={d_energy_s:+.6f} Δdropped={d_dropped_s:+.3f} "
            f"({reason_s})"
        )
        d_reward_r = random_masked_stoch["reward_mean"] - noop_det["reward_mean"]
        d_energy_r = random_masked_stoch["energy_mean"] - noop_det["energy_mean"]
        d_dropped_r = random_masked_stoch["dropped_mean"] - noop_det["dropped_mean"]
        better_r, reason_r = _better_with_tolerance(d_reward_r, d_energy_r, d_dropped_r, eval_track)
        print(
            f"[summary:{eval_track}] random_masked(stoch) vs noop: better={better_r} "
            f"Δreward={d_reward_r:+.3f} Δenergy={d_energy_r:+.6f} Δdropped={d_dropped_r:+.3f} "
            f"({reason_r})"
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
