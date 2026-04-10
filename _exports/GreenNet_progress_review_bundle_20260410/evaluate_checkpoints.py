#!/usr/bin/env python3
"""
evaluate_checkpoints.py

Evaluate every PPO checkpoint saved under runs/*/ppo_greennet(.zip).

Usage:
  python evaluate_checkpoints.py --runs_dir runs --episodes 10 --seed 123 --csv eval_results.csv

Notes:
- Assumes Stable-Baselines3 PPO checkpoints saved as "ppo_greennet.zip"
- Builds the same env used in training (GreenNetEnv), optionally using env_config.json if present.
- Collects mean/std reward, mean episode length, and averages any numeric info[*] keys if your env provides them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as e:
    raise SystemExit(
        "Stable-Baselines3 not found/importable. Install it in your venv:\n"
        "  pip install stable-baselines3\n\n"
        f"Import error: {e}"
    )

try:
    from greennet.env import GreenNetEnv, EnvConfig
    from greennet.utils.config import load_env_config_from_run, load_train_config_from_run
except Exception as e:
    raise SystemExit(
        "Could not import GreenNetEnv / EnvConfig. Make sure you run this from the project root "
        "and your venv is activated.\n\n"
        f"Import error: {e}"
    )


def make_vec_env(run_dir: Path, seed: int) -> DummyVecEnv:
    """
    Create a DummyVecEnv with Monitor wrapper. Attempts to pass EnvConfig if supported.
    """
    env_cfg = load_env_config_from_run(run_dir, verbose=True)

    def _init():
        try:
            env = GreenNetEnv(config=env_cfg)  # current signature in your repo
        except TypeError:
            # In case your GreenNetEnv signature changed
            env = GreenNetEnv()  # type: ignore[call-arg]
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_init])


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, (int, float, np.number)):
        xf = float(x)
        return math.isfinite(xf)
    return False


def rollout_eval(
    model: PPO,
    env: DummyVecEnv,
    n_episodes: int,
    deterministic: bool,
) -> Dict[str, Any]:
    """
    Manual rollout so we can also aggregate info keys.
    Returns mean/std reward, mean ep length, and mean of any numeric info keys.
    """
    ep_rewards: List[float] = []
    ep_lens: List[int] = []

    # Collect per-episode averages for each numeric info key
    info_key_ep_means: Dict[str, List[float]] = defaultdict(list)

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        ep_len = 0

        info_sums: Dict[str, float] = defaultdict(float)
        info_counts: Dict[str, int] = defaultdict(int)

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = env.step(action)

            ep_r += float(reward[0])
            ep_len += 1
            done = bool(dones[0])

            info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
            if isinstance(info, dict):
                for k, v in info.items():
                    if _is_finite_number(v):
                        info_sums[k] += float(v)
                        info_counts[k] += 1

        ep_rewards.append(ep_r)
        ep_lens.append(ep_len)

        for k, s in info_sums.items():
            c = info_counts.get(k, 0)
            if c > 0:
                info_key_ep_means[k].append(s / c)

    out: Dict[str, Any] = {
        "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else float("nan"),
        "std_reward": float(np.std(ep_rewards)) if ep_rewards else float("nan"),
        "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else float("nan"),
    }

    # Add info-key means across episodes (if present)
    for k, vals in info_key_ep_means.items():
        if vals:
            out[f"info_mean_{k}"] = float(np.mean(vals))

    return out


def find_checkpoints(runs_dir: Path) -> List[Path]:
    """
    Find ppo_greennet.zip files under runs_dir and also directly under runs_dir.
    """
    candidates = set()

    # Common patterns:
    # runs/<timestamp>/ppo_greennet.zip
    candidates.update(runs_dir.glob("*/ppo_greennet.zip"))
    candidates.update(runs_dir.glob("ppo_greennet.zip"))

    # If some saves are without .zip (rare), include them too
    candidates.update(runs_dir.glob("*/ppo_greennet"))
    candidates.update(runs_dir.glob("ppo_greennet"))

    # Keep only existing files
    out = [p for p in candidates if p.exists() and p.is_file()]
    return sorted(out)


def checkpoint_to_run_dir(ckpt_path: Path) -> Path:
    # runs/<timestamp>/ppo_greennet.zip -> runs/<timestamp>
    # runs/ppo_greennet.zip -> runs
    return ckpt_path.parent


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg = load_train_config_from_run(run_dir, verbose=False)
    return cfg if isinstance(cfg, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all PPO checkpoints under runs/.")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Path to runs directory.")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes per checkpoint.")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for evaluation.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (recommended).")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write CSV results.")
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists():
        raise SystemExit(f"runs_dir does not exist: {runs_dir}")

    ckpts = find_checkpoints(runs_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {runs_dir} (expected ppo_greennet.zip).")

    results: List[Dict[str, Any]] = []

    for i, ckpt in enumerate(ckpts):
        run_dir = checkpoint_to_run_dir(ckpt)
        run_name = run_dir.name if run_dir != runs_dir else str(runs_dir)

        # Make eval env with a per-run seed offset so runs don't share identical episode RNG streams
        env = make_vec_env(run_dir=run_dir, seed=args.seed + i)

        # SB3 expects path WITHOUT ".zip" sometimes, but it also accepts ".zip".
        # We'll pass the full path; if that fails, try without suffix.
        model = None
        try:
            model = PPO.load(str(ckpt), env=env)
        except Exception:
            try:
                model = PPO.load(str(ckpt.with_suffix("")), env=env)
            except Exception as e:
                print(f"[SKIP] Failed to load {ckpt}: {e}")
                continue

        metrics = rollout_eval(
            model=model,
            env=env,
            n_episodes=args.episodes,
            deterministic=bool(args.deterministic),
        )

        cfg = load_run_config(run_dir)
        ent_coef = cfg.get("ppo", {}).get("ent_coef", None) if isinstance(cfg, dict) else None
        total_ts = cfg.get("total_timesteps", None) if isinstance(cfg, dict) else None

        row: Dict[str, Any] = {
            "run": run_name,
            "checkpoint": str(ckpt),
            "episodes": args.episodes,
            "seed_base": args.seed,
            "deterministic": bool(args.deterministic),
            "ent_coef": ent_coef,
            "total_timesteps": total_ts,
            **metrics,
        }
        results.append(row)

        print(
            f"[OK] {run_name:>15} | mean_reward={row['mean_reward']:.4f} "
            f"(std={row['std_reward']:.4f}) | mean_len={row['mean_ep_len']:.1f}"
        )

    if not results:
        raise SystemExit("No checkpoints were successfully evaluated.")

    # Rank by mean_reward desc
    results_sorted = sorted(results, key=lambda r: float(r.get("mean_reward", float("-inf"))), reverse=True)

    # Print top 10
    print("\n=== Ranked (top 10 by mean_reward) ===")
    for rank, r in enumerate(results_sorted[:10], start=1):
        print(
            f"{rank:>2}. {r['run']:<15}  mean_reward={r['mean_reward']:.4f}  "
            f"std={r['std_reward']:.4f}  len={r['mean_ep_len']:.1f}  ent_coef={r.get('ent_coef')}"
        )

    # Optional CSV
    if args.csv:
        # union all keys so we don't lose info_mean_* columns
        all_keys = sorted({k for row in results_sorted for k in row.keys()})
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as f:
            f.write(",".join(all_keys) + "\n")
            for row in results_sorted:
                vals = []
                for k in all_keys:
                    v = row.get(k, "")
                    if isinstance(v, (dict, list)):
                        v = json.dumps(v, ensure_ascii=False)
                    vals.append(str(v).replace("\n", " ").replace(",", ";"))
                f.write(",".join(vals) + "\n")
        print(f"\nWrote CSV -> {args.csv}")

    # Quick “keep list” suggestion
    best = results_sorted[0]
    print("\n=== Keep suggestion ===")
    print(f"Keep BEST checkpoint: {best['checkpoint']} (run={best['run']}, mean_reward={best['mean_reward']:.4f})")
    # also keep one baseline-ish (smallest ent_coef if available)
    with_ent = [r for r in results_sorted if r.get("ent_coef") is not None]
    if with_ent:
        baseline = sorted(with_ent, key=lambda r: float(r["ent_coef"]))[0]
        if baseline["checkpoint"] != best["checkpoint"]:
            print(f"Keep BASELINE (lowest ent_coef): {baseline['checkpoint']} (ent_coef={baseline['ent_coef']})")


if __name__ == "__main__":
    main()
