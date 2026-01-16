from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from greennet.env import EnvConfig, GreenNetEnv
from greennet.utils.config import load_env_config_from_run, save_env_config, save_train_config, load_train_config_from_run


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "total_timesteps": 60_000,
    "ppo": {
        "policy": "MultiInputPolicy",
        "verbose": 1,
        "n_steps": 2048,
        "batch_size": 256,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "ent_coef": 0.005,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
    },
}


def load_config(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_env(seed: int, env_config: EnvConfig):
    def _init():
        # Use a fresh copy of the env config for each vectorized environment.
        env = GreenNetEnv(config=replace(env_config))
        env.reset(seed=seed)
        return Monitor(env)

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


def eval_policy(
    model: PPO | None,
    env_config: EnvConfig,
    episodes: int,
    seed: int,
    label: str,
    deterministic: bool,
) -> Dict[str, float]:
    """Evaluate a policy and print aggregate stats."""
    env = Monitor(GreenNetEnv(config=replace(env_config)))

    ep_rewards: list[float] = []
    ep_energy: list[float] = []
    ep_dropped: list[float] = []
    ep_delivered: list[float] = []
    ep_applied: list[int] = []
    ep_reverted: list[int] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_r = 0.0
        total_energy = 0.0
        total_dropped = 0.0
        total_delivered = 0.0
        applied = 0
        reverted = 0

        while not (terminated or truncated):
            if model is None:
                action = 0
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                try:
                    action = int(action)
                except Exception:
                    action = int(action[0])

            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)

            metrics = info.get("metrics")
            if metrics is not None:
                total_energy += float(getattr(metrics, "energy_kwh", 0.0))
                total_dropped += float(getattr(metrics, "dropped", 0.0))
                total_delivered += float(getattr(metrics, "delivered", 0.0))

            applied += int(bool(info.get("toggle_applied")))
            reverted += int(bool(info.get("toggle_reverted")))

        ep_rewards.append(total_r)
        ep_energy.append(total_energy)
        ep_dropped.append(total_dropped)
        ep_delivered.append(total_delivered)
        ep_applied.append(applied)
        ep_reverted.append(reverted)

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        arr = np.asarray(xs, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    r_m, r_s = _mean_std(ep_rewards)
    e_m, e_s = _mean_std(ep_energy)
    d_m, d_s = _mean_std(ep_dropped)
    dl_m, dl_s = _mean_std(ep_delivered)

    mode = "det" if deterministic else "stoch"
    print(f"\n=== Evaluation: {label} ({mode}, {episodes} episodes) ===")
    print(f"episode_reward: mean={r_m:.3f} std={r_s:.3f}")
    print(f"energy_kwh:     mean={e_m:.6f} std={e_s:.6f}")
    print(f"dropped:        mean={d_m:.3f} std={d_s:.3f}")
    print(f"delivered:      mean={dl_m:.3f} std={dl_s:.3f}")
    print(
        f"toggles applied mean={float(np.mean(ep_applied)):.2f} "
        f"reverted mean={float(np.mean(ep_reverted)):.2f}"
    )

    stats = {
        "reward_mean": r_m,
        "energy_mean": e_m,
        "dropped_mean": d_m,
    }

    env.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on GreenNet.")
    parser.add_argument("--config", type=Path, help="Path to JSON config to load.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training.")
    parser.add_argument("--model", type=Path, help="Path to saved PPO model zip (default: latest under runs/).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    args = parser.parse_args()

    config = load_config(args.config)
    # Ensure defaults exist if a custom config omits them.
    config.setdefault("ppo", {})
    config["ppo"].setdefault("ent_coef", DEFAULT_CONFIG["ppo"]["ent_coef"])
    config["ppo"].setdefault("n_steps", DEFAULT_CONFIG["ppo"]["n_steps"])

    seed = int(config.get("seed", 0))
    set_seeds(seed)

    if args.eval:
        model_path = args.model or find_latest_model()
        print(f"Evaluating model: {model_path}")

        model_run_dir = Path(model_path).parent
        env_config = load_env_config_from_run(model_run_dir, verbose=True)
        print(
            "[eval] using env_config keys: "
            f"{sorted(env_config.__dict__.keys())}"
        )
        train_cfg = load_train_config_from_run(model_run_dir, verbose=False)
        if train_cfg:
            print(f"[eval] loaded train_config keys: {sorted(train_cfg.keys())}")
        else:
            print("[eval] no train_config found; using defaults for PPO keys.")

        model = PPO.load(str(model_path))

        trained_det = eval_policy(
            model,
            env_config,
            episodes=int(args.episodes),
            seed=seed,
            label="trained",
            deterministic=True,
        )
        trained_stoch = eval_policy(
            model,
            env_config,
            episodes=int(args.episodes),
            seed=seed,
            label="trained",
            deterministic=False,
        )
        noop_det = eval_policy(
            None,
            env_config,
            episodes=int(args.episodes),
            seed=seed,
            label="always_noop",
            deterministic=True,
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

        return

    # --- Training run setup (only when not evaluating) ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    save_train_config(run_dir, config)
    save_requirements_copy(run_dir)

    env_config = EnvConfig(
        topology_randomize=False,
        topology_seed=0,

        # Reward shaping: balance QoS vs energy so the optimal deterministic policy isn't always NOOP.
        drop_penalty_lambda=0.4,
        normalize_drop=True,

        # Increase energy incentive now that we have per-edge observability + cooldown.
        energy_weight=10.0,

        # With cooldown enabled, we can make a single toggle a bit cheaper without allowing thrashing.
        toggle_penalty=0.0002,

        # Prevent per-edge flapping.
        toggle_cooldown_steps=5,

        util_block_threshold=0.10,
        global_toggle_cooldown_steps=5,
    )

    save_env_config(run_dir, env_config)
    print(f"[train] saved env_config keys: {sorted(env_config.__dict__.keys())}")
    print(f"[train] saved train_config keys: {sorted(config.keys())}")

    env = DummyVecEnv([make_env(seed, env_config)])
    model = PPO(env=env, **config["ppo"])

    class ActionEffectivenessCallback(BaseCallback):
        def __init__(self, log_every: int = 5000):
            super().__init__()
            self.log_every = log_every
            self.applied = 0
            self.reverted = 0
            self.noop = 0
            self.invalid = 0
            self.blocked = 0

        def _on_training_start(self) -> None:
            self.applied = 0
            self.reverted = 0
            self.noop = 0
            self.invalid = 0
            self.blocked = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                applied = bool(info.get("toggle_applied"))
                reverted = bool(info.get("toggle_reverted"))
                nooped = bool(info.get("action_is_noop"))
                invalid = bool(info.get("action_is_invalid"))
                blocked = (
                    bool(info.get("toggle_blocked_cooldown"))
                    or bool(info.get("toggle_blocked_high_util"))
                    or bool(info.get("toggle_blocked_global_cooldown"))
                )

                self.applied += int(applied)
                self.reverted += int(reverted)
                self.noop += int(nooped)
                self.invalid += int(invalid)
                self.blocked += int(blocked)

            total = self.applied + self.reverted + self.noop + self.invalid + self.blocked
            if total > 0 and self.num_timesteps % self.log_every == 0:
                applied_frac = self.applied / total
                reverted_frac = self.reverted / total
                noop_frac = self.noop / total
                invalid_frac = self.invalid / total
                blocked_frac = self.blocked / total
                print(
                    f"[action_stats] steps={self.num_timesteps} "
                    f"applied={applied_frac:.3f} reverted={reverted_frac:.3f} "
                    f"noop={noop_frac:.3f} invalid={invalid_frac:.3f} blocked={blocked_frac:.3f}"
                )
                self.logger.record("action_stats/applied_frac", applied_frac)
                self.logger.record("action_stats/reverted_frac", reverted_frac)
                self.logger.record("action_stats/noop_frac", noop_frac)
                self.logger.record("action_stats/invalid_frac", invalid_frac)
                self.logger.record("action_stats/blocked_frac", blocked_frac)
            return True

    callback = ActionEffectivenessCallback(log_every=5000)
    model.learn(total_timesteps=int(config["total_timesteps"]), callback=callback)
    model.save(str(run_dir / "ppo_greennet"))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
