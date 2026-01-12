# train.py
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from greennet.env import GreenNetEnv


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "total_timesteps": 200_000,
    "ppo": {
        "policy": "MultiInputPolicy",
        "verbose": 1,
        "n_steps": 1024,
        "batch_size": 256,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
    },
}


def load_config(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_env(seed: int):
    def _init():
        env = GreenNetEnv()
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


def save_run_artifacts(run_dir: Path, config: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    requirements_src = Path("ml-env") / "requirements.txt"
    if requirements_src.exists():
        requirements_dst = run_dir / "requirements.txt"
        requirements_dst.write_text(requirements_src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on GreenNet.")
    parser.add_argument("--config", type=Path, help="Path to JSON config to load.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    set_seeds(seed)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    save_run_artifacts(run_dir, config)

    env = DummyVecEnv([make_env(seed)])

    model = PPO(env=env, **config["ppo"])
    model.learn(total_timesteps=int(config["total_timesteps"]))
    model.save(str(run_dir / "ppo_greennet"))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
