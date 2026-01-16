"""Shared config loading utilities for env and training configs."""
from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

from greennet.env import EnvConfig


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_env_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(EnvConfig)} if is_dataclass(EnvConfig) else set()
    return {k: v for k, v in raw.items() if not allowed or k in allowed}


def load_env_config_from_run(run_dir: Path, *, verbose: bool = True, fallback: EnvConfig | None = None) -> EnvConfig:
    """Load EnvConfig from run_dir with fallbacks.

    Priority:
      1) env_config.json (filtered to EnvConfig fields)
      2) train_config.json/config.json env keys under "env", "env_config", or "env_kwargs"
      3) fallback or EnvConfig() with a warning.
    """
    default_cfg = fallback or EnvConfig()
    cfg_path = run_dir / "env_config.json"
    if cfg_path.exists():
        try:
            raw = _load_json(cfg_path)
            env_data = _filter_env_fields(raw if isinstance(raw, dict) else {})
            cfg = EnvConfig(**env_data)
            if verbose:
                print(f"[env_config] Loaded from {cfg_path} (keys={sorted(env_data.keys())})")
            return cfg
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[env_config] Failed to load {cfg_path}: {exc}; falling back.")

    # Try train_config.json then config.json
    for name in ("train_config.json", "config.json"):
        cand = run_dir / name
        if not cand.exists():
            continue
        try:
            raw = _load_json(cand)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[env_config] Failed to load {cand}: {exc}; continuing.")
            continue
        env_data: Dict[str, Any] = {}
        if isinstance(raw, dict):
            for key in ("env", "env_config", "env_kwargs"):
                if isinstance(raw.get(key), dict):
                    env_data.update(raw[key])
        env_filtered = _filter_env_fields(env_data)
        if env_filtered:
            cfg = EnvConfig(**env_filtered)
            if verbose:
                print(f"[env_config] Loaded from {cand} (keys={sorted(env_filtered.keys())})")
            return cfg
        if verbose:
            print(f"[env_config] No usable env keys in {cand}; continuing.")

    if verbose:
        print("[env_config] Missing env_config.json and no env keys found in configs; using defaults (risk of mismatch).")
    return default_cfg


def load_train_config_from_run(run_dir: Path, *, verbose: bool = True) -> Dict[str, Any]:
    """Load training config (PPO hyperparams, seed, timesteps) with fallbacks."""
    for name in ("train_config.json", "config.json"):
        cand = run_dir / name
        if cand.exists():
            try:
                cfg = _load_json(cand)
                if verbose:
                    keys = sorted(cfg.keys()) if isinstance(cfg, dict) else []
                    print(f"[train_config] Loaded from {cand} (keys={keys})")
                return cfg if isinstance(cfg, dict) else {}
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[train_config] Failed to load {cand}: {exc}; continuing.")
    if verbose:
        print("[train_config] No train config found; returning empty dict.")
    return {}


def save_env_config(run_dir: Path, env_config: EnvConfig) -> None:
    """Persist the environment configuration used for this run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "env_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(env_config), handle, indent=2, sort_keys=True)


def save_train_config(run_dir: Path, train_config: Dict[str, Any]) -> None:
    """Persist the training (ppo/seed/timesteps) configuration used for this run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "train_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(train_config, handle, indent=2, sort_keys=True)

    # Keep legacy config.json for backward compatibility
    legacy_path = run_dir / "config.json"
    with legacy_path.open("w", encoding="utf-8") as handle:
        json.dump(train_config, handle, indent=2, sort_keys=True)
