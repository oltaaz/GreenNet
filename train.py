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
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None
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
        "device": "auto",
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


def parse_seed_list(seed_text: str) -> list[int]:
    """Parse comma-separated integers like '0,1,2,3,4'."""
    parts = [p.strip() for p in seed_text.split(",") if p.strip()]
    return [int(p) for p in parts]


def run_robustness_eval(
    model: PPO,
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
    ep_norm_drop: list[float] = []

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

        denom = max(total_delivered + total_dropped, 1e-9)
        ep_norm_drop.append(float(total_dropped / denom))

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
    nd_m, nd_s = _mean_std(ep_norm_drop)

    mode = "det" if deterministic else "stoch"
    print(f"\n=== Evaluation: {label} ({mode}, {episodes} episodes) ===")
    print(f"episode_reward: mean={r_m:.3f} std={r_s:.3f}")
    print(f"energy_kwh:     mean={e_m:.6f} std={e_s:.6f}")
    print(f"dropped:        mean={d_m:.3f} std={d_s:.3f}")
    print(f"delivered:      mean={dl_m:.3f} std={dl_s:.3f}")
    print(f"norm_drop:      mean={nd_m:.5f} std={nd_s:.5f}")
    print(
        f"toggles applied mean={float(np.mean(ep_applied)):.2f} "
        f"reverted mean={float(np.mean(ep_reverted)):.2f}"
    )

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
        "--robustness",
        action="store_true",
        help="Run multi-seed robustness eval (NOOP vs trained(det)) and write CSV+PNG under the model run dir.",
    )
    parser.add_argument(
        "--topology-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated topology seeds for --robustness (e.g., '0,1,2,3,4').",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Ensure defaults exist if a custom config omits them.
    config.setdefault("ppo", {})
    config["ppo"].setdefault("ent_coef", DEFAULT_CONFIG["ppo"]["ent_coef"])
    config["ppo"].setdefault("n_steps", DEFAULT_CONFIG["ppo"]["n_steps"])

    seed = int(config.get("seed", 0))
    set_seeds(seed)

    # ---- Device selection (Apple Silicon GPU via MPS) ----
    # SB3 defaults to "auto" which primarily checks CUDA; it will NOT automatically pick MPS.
    preferred_device = "mps" if torch.backends.mps.is_available() else "cpu"
    config.setdefault("ppo", {})
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
        model = PPO.load(str(model_path), device=load_device)
        print(f"[device] loaded PPO on: {model.policy.device}")

        seeds = parse_seed_list(args.topology_seeds)
        out_csv = model_run_dir / "robustness_eval.csv"
        out_png = model_run_dir / "robustness_energy_vs_drop.png"

        run_robustness_eval(
            model=model,
            base_env_config=env_config,
            episodes=int(args.episodes),
            seed=seed,
            topology_seeds=seeds,
            out_csv=out_csv,
            out_png=out_png,
        )
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
            seed=seed,
            label="always_noop",
            deterministic=True,
        )
        return

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

        load_device = config.get("ppo", {}).get("device", "cpu")
        model = PPO.load(str(model_path), device=load_device)
        print(f"[device] loaded PPO on: {model.policy.device}")

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
        drop_penalty_lambda=0.2,
        normalize_drop=True,

        # Increase energy incentive now that we have per-edge observability + cooldown.
        energy_weight=190.0,

        # With cooldown enabled, we can make a single toggle a bit cheaper without allowing thrashing.
        toggle_penalty=0.0002,

        # Prevent per-edge flapping.
        toggle_cooldown_steps=5,

        util_block_threshold=0.10,
        global_toggle_cooldown_steps=5,
        flows_per_step=6,
        base_capacity=10.0,
    )

    save_env_config(run_dir, env_config)
    print(f"[train] saved env_config keys: {sorted(env_config.__dict__.keys())}")
    print(f"[train] saved train_config keys: {sorted(config.keys())}")

    env = DummyVecEnv([make_env(seed, env_config)])
    model = PPO(env=env, **config["ppo"])
    print(f"[device] training PPO on: {model.policy.device}")

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
