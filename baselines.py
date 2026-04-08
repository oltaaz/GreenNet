"""Traditional controller baselines for GreenNet.

Runs three policies on the same seeded environment:
  A) Always-on (no toggles)
  B) Utilization-threshold controller (toggle off low-util edges, toggle on if congested)
  C) RL policy (if a PPO model path is provided)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from greennet.env import EnvConfig, GreenNetEnv

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore[misc,assignment]


ActionFn = Callable[[Dict[str, Any], Dict[str, Any], GreenNetEnv], int]


def canonical_controller_policy_name(policy: str | None) -> str:
    key = str(policy or "").strip().lower()
    if key in {"noop", "all_on"}:
        return "all_on"
    if key in {"baseline", "heuristic"}:
        return "utilization_threshold"
    if key == "ppo":
        return "ppo"
    return key or "unknown"


def controller_policy_class(policy: str | None) -> str:
    canonical = canonical_controller_policy_name(policy)
    if canonical in {"all_on", "utilization_threshold"}:
        return "traditional_baseline"
    if canonical == "ppo":
        return "ai_enhanced"
    return "other"


def action_always_on(_: Dict[str, Any], __: Dict[str, Any], ___: GreenNetEnv) -> int:
    return 0


def action_utilization_threshold(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    env: GreenNetEnv,
    low_thresh: float = 0.15,
    high_thresh: float = 0.8,
) -> int:
    sim = env.simulator
    if sim is None or not env.edge_list:
        return 0

    avg_util = float(info.get("metrics").avg_utilization if info.get("metrics") else obs["avg_util"][0])

    # Collect active/inactive edges with their utilization.
    active_edges: List[tuple[tuple[int, int], float]] = []
    inactive_edges: List[tuple[int, int]] = []
    for edge in env.edge_list:
        key = env._edge_key(edge[0], edge[1])  # type: ignore[attr-defined]
        util = float(sim.utilization.get(key, 0.0))
        if sim.active.get(key, True):
            active_edges.append((edge, util))
        else:
            inactive_edges.append(edge)

    # If underutilized, try to sleep the lowest-util active edge (keep at least one edge up).
    if avg_util < low_thresh and len(active_edges) > 1:
        edge_to_sleep = sorted(active_edges, key=lambda x: x[1])[0][0]
        return env.edge_list.index(edge_to_sleep) + 1

    # If congested and edges are off, turn one back on.
    if avg_util > high_thresh and inactive_edges:
        return env.edge_list.index(inactive_edges[0]) + 1

    return 0


# Backward-compatible alias used by older evaluation code paths.
action_sleep_if_idle = action_utilization_threshold


def action_rl(model: Any | None) -> ActionFn:
    def _fn(obs: Dict[str, Any], _: Dict[str, Any], env: GreenNetEnv) -> int:
        if model is None:
            return 0
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return _fn


def run_episode(
    policy_name: str,
    env_seed: int,
    steps: int,
    action_fn: ActionFn,
) -> Dict[str, Any]:
    env = GreenNetEnv(config=EnvConfig(topology_seed=env_seed))
    obs, info = env.reset(seed=env_seed)

    log: List[Dict[str, float]] = []

    for _ in range(steps):
        act = action_fn(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(act)
        metrics = info.get("metrics")
        log.append(
            {
                "reward": float(reward),
                "delivered": float(getattr(metrics, "delivered", 0.0)),
                "dropped": float(getattr(metrics, "dropped", 0.0)),
                "avg_delay_ms": float(getattr(metrics, "avg_delay_ms", 0.0)),
                "avg_utilization": float(getattr(metrics, "avg_utilization", 0.0)),
                "active_ratio": float(obs["active_ratio"][0]),
                "energy_kwh": float(getattr(metrics, "energy_kwh", 0.0)),
                "carbon_g": float(getattr(metrics, "carbon_g", 0.0)),
            }
        )
        if terminated or truncated:
            break

    env.close()
    return {"policy": policy_name, "log": log, "summary": summarize(log)}


def summarize(log: List[Dict[str, float]]) -> Dict[str, float]:
    if not log:
        return {}
    keys = ["reward", "delivered", "dropped", "avg_delay_ms", "avg_utilization", "active_ratio", "energy_kwh", "carbon_g"]
    return {k: fmean([row[k] for row in log]) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controller-baseline comparisons (all-on, utilization-threshold, RL)."
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of steps per policy.")
    parser.add_argument("--seed", type=int, default=7, help="Environment seed to align flows/topology.")
    parser.add_argument("--model", type=Path, default=None, help="Path to a trained PPO model (.zip) for RL baseline.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for logs.")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []

    results.append(run_episode("always_on", args.seed, args.steps, action_always_on))
    results.append(run_episode("utilization_threshold", args.seed, args.steps, action_utilization_threshold))

    rl_included = False
    if args.model and args.model.exists() and PPO is not None:
        model = PPO.load(str(args.model))
        results.append(run_episode("rl_policy", args.seed, args.steps, action_rl(model)))
        rl_included = True

    for res in results:
        name = res["policy"]
        summary = res["summary"]
        print(f"\nPolicy: {name}")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json

        with args.output.open("w", encoding="utf-8") as handle:
            json.dump({"results": results, "rl_included": rl_included}, handle, indent=2)
        print(f"\nSaved detailed logs to {args.output}")


if __name__ == "__main__":
    main()
