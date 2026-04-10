# resume_latest.py
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from greennet.env import GreenNetEnv, EnvConfig
from greennet.utils.config import load_env_config_from_run, load_train_config_from_run, save_env_config, save_train_config


def find_latest_model(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*/ppo_greennet.zip"))
    if not candidates:
        raise FileNotFoundError("No runs/*/ppo_greennet.zip found.")
    return candidates[-1]


def make_env(env_cfg: EnvConfig):
    def _init():
        env = GreenNetEnv(config=replace(env_cfg))
        return Monitor(env)
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=Path, default=Path("runs"))
    p.add_argument("--model", type=Path, default=None, help="Override model path (ppo_greennet.zip).")
    p.add_argument("--timesteps", type=int, default=200_000)
    args = p.parse_args()

    model_path = args.model or find_latest_model(args.runs_dir)
    run_dir = model_path.parent

    env_cfg = load_env_config_from_run(run_dir, verbose=True)
    train_cfg = load_train_config_from_run(run_dir, verbose=False)
    if train_cfg:
        ppo_keys = sorted(train_cfg.get("ppo", {}).keys()) if isinstance(train_cfg.get("ppo"), dict) else []
        print(f"[resume] loaded train_config keys: {sorted(train_cfg.keys())} ppo_keys={ppo_keys}")
    else:
        print("[resume] no train_config found; continuing with defaults.")
    env = DummyVecEnv([make_env(env_cfg)])

    model = PPO.load(str(model_path), env=env)
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.runs_dir / f"{run_dir.name}_cont_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # keep env_config and train_config with the continued run too
    save_env_config(out_dir, env_cfg)

    cont_meta = {
        "continued_from": str(model_path),
        "timesteps_added": int(args.timesteps),
        "continued_at_utc": stamp,
    }

    # If we have the original train config, persist it alongside continuation metadata.
    # Otherwise, still write a minimal train_config.json so the run is self-describing.
    merged_train_cfg = dict(train_cfg) if isinstance(train_cfg, dict) else {}
    merged_train_cfg.setdefault("continuation", {})
    if isinstance(merged_train_cfg["continuation"], dict):
        merged_train_cfg["continuation"].update(cont_meta)
    else:
        merged_train_cfg["continuation"] = cont_meta

    save_train_config(out_dir, merged_train_cfg)

    model.save(str(out_dir / "ppo_greennet"))
    print(f"Saved env_config.json and train_config.json to: {out_dir}")
    print(f"Saved continued model to: {out_dir / 'ppo_greennet.zip'}")


if __name__ == "__main__":
    main()
