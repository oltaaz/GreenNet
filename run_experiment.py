#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from baselines import action_sleep_if_idle
from greennet.env import EnvConfig, GreenNetEnv
from greennet.utils.config import save_env_config

try:
    from sb3_contrib import MaskablePPO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MaskablePPO = None  # type: ignore


try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore


# --- Helper functions for reading JSON and inferring topology seed from PPO run folder ---
def _load_json_file(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def infer_topology_seed_from_model(model_path: Path) -> int | None:
    """Infer the topology seed used during training from env_config.json next to the model."""
    cfg = _load_json_file(model_path.parent / "env_config.json")
    if not cfg:
        return None
    topo = cfg.get("topology_seed")
    if isinstance(topo, (int, float, str)):
        try:
            return int(topo)
        except Exception:
            return None
    seeds = cfg.get("topology_seeds")
    if isinstance(seeds, list) and seeds:
        try:
            return int(seeds[0])
        except Exception:
            return None
    return None


StepRow = Dict[str, Any]
ActionFn = Callable[[Dict[str, Any], Dict[str, Any], GreenNetEnv], int]


FIELDNAMES = [
    "run_id",
    "policy",
    "scenario",
    "seed",
    "episode_seed",
    "episode",
    "step",
    "action",
    "reward",
    "terminated",
    "truncated",
    "avg_utilization",
    "active_ratio",
    "max_util",
    "min_util",
    "p95_util",
    "dropped_prev",
    "num_active_edges",
    "near_saturated_edges",
    "delivered",
    "dropped",
    "avg_delay_ms",
    "avg_path_latency_ms",
    "energy_kwh",
    "carbon_g",
    "delta_energy_kwh",
    "delta_delivered",
    "delta_dropped",
    "delta_carbon_g",
    "norm_drop_step",
    "norm_drop",
    "reward_energy",
    "reward_drop",
    "reward_qos",
    "reward_toggle",
    "toggle_applied",
    "toggle_reverted",
    "toggle_blocked_any",
    "toggle_blocked_cooldown",
    "toggle_blocked_high_util",
    "toggle_blocked_global_cooldown",
    "action_is_noop",
    "action_is_invalid",
    "flows_count",
    "flows_json",
]


def _flow_to_dict(flow: Any) -> Dict[str, Any]:
    if hasattr(flow, "source") and hasattr(flow, "destination") and hasattr(flow, "demand"):
        return {
            "source": int(getattr(flow, "source")),
            "destination": int(getattr(flow, "destination")),
            "demand": float(getattr(flow, "demand")),
        }
    if isinstance(flow, (list, tuple)) and len(flow) == 3:
        return {"source": int(flow[0]), "destination": int(flow[1]), "demand": float(flow[2])}
    return {"repr": repr(flow)}


def _serialize_flows(flows: Any) -> str:
    items: List[Dict[str, Any]] = []
    try:
        for flow in flows:
            items.append(_flow_to_dict(flow))
    except TypeError:
        items.append({"repr": repr(flows)})
    try:
        return json.dumps(items, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return json.dumps([{"repr": repr(items)}], separators=(",", ":"), ensure_ascii=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def build_env_config(scenario: str, max_steps: int | None) -> EnvConfig:
    cfg = EnvConfig()
    if hasattr(cfg, "traffic_seed"):
        cfg.traffic_seed = None

    if scenario == "normal":
        cfg.traffic_model = "uniform"
    elif scenario == "burst":
        cfg.traffic_model = "stochastic"
        cfg.traffic_avg_bursts_per_step = 4.0
        cfg.traffic_spike_prob = 0.05
        cfg.traffic_spike_multiplier_range = (2.0, 6.0)
        cfg.traffic_spike_duration_range = (3, 12)
    elif scenario == "hotspot":
        cfg.traffic_model = "stochastic"
        cfg.traffic_hotspots = ((0, 5, 3.0), (2, 7, 2.0))
        cfg.traffic_avg_bursts_per_step = 3.0
        cfg.traffic_spike_prob = 0.01
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if max_steps is not None:
        cfg.max_steps = int(max_steps)

    return cfg


def find_latest_model(runs_dir: Path) -> Path:
    candidates: List[Path] = []
    if runs_dir.exists():
        candidates.extend(sorted(runs_dir.glob("*/ppo_greennet.zip")))
        candidates.extend(sorted(runs_dir.glob("*/ppo_greennet")))
        candidates.extend(sorted(runs_dir.glob("ppo_greennet.zip")))
        candidates.extend(sorted(runs_dir.glob("ppo_greennet")))
    if not candidates:
        raise FileNotFoundError("No model found under runs/*/ppo_greennet(.zip).")
    return candidates[-1]


def load_policy(
    policy: str,
    *,
    runs_dir: Path,
    model_path: Path | None,
    deterministic: bool,
) -> Tuple[ActionFn, Dict[str, Any]]:
    if policy == "noop":
        return (lambda _obs, _info, _env: 0), {"policy_type": "noop"}

    if policy == "baseline":
        return action_sleep_if_idle, {"policy_type": "baseline"}

    if policy != "ppo":
        raise ValueError(f"Unknown policy: {policy}")

    if model_path is not None and not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    resolved_model_path = model_path or find_latest_model(runs_dir)
    model = None
    model_type = None
    load_errors: List[str] = []

    if MaskablePPO is not None:
        try:
            model = MaskablePPO.load(str(resolved_model_path))
            model_type = "MaskablePPO"
        except Exception as exc:  # pragma: no cover - optional dependency
            load_errors.append(f"MaskablePPO load failed: {exc}")
            model = None

    if model is None:
        if PPO is None:
            err = "stable-baselines3 not available; cannot load PPO model."
            if load_errors:
                err = f"{err} Previous errors: {load_errors}"
            raise SystemExit(err)
        model = PPO.load(str(resolved_model_path))
        model_type = "PPO"

    def _action_ppo(obs: Dict[str, Any], _info: Dict[str, Any], env: GreenNetEnv) -> int:
        if model is None:
            return 0
        if model_type == "MaskablePPO" and hasattr(env, "get_action_mask"):
            mask = env.get_action_mask()
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        return int(action)

    return _action_ppo, {
        "policy_type": "ppo",
        "model_path": str(resolved_model_path),
        "model_type": model_type,
    }


def build_step_row(
    run_id: str,
    policy: str,
    scenario: str,
    seed: int,
    episode_seed: int,
    episode_idx: int,
    step_idx: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    obs: Dict[str, Any],
    info: Dict[str, Any],
    *,
    save_flows: bool,
) -> StepRow:
    metrics = info.get("metrics")
    flows = info.get("flows") or ()
    try:
        flows_count = len(flows)
    except TypeError:
        flows_count = 0
    flows_json = _serialize_flows(flows) if save_flows else ""

    return {
        "run_id": run_id,
        "policy": policy,
        "scenario": scenario,
        "seed": int(seed),
        "episode_seed": int(episode_seed),
        "episode": int(episode_idx),
        "step": int(step_idx),
        "action": int(action),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "avg_utilization": float(getattr(metrics, "avg_utilization", 0.0)),
        "active_ratio": float(obs["active_ratio"][0]),
        "max_util": float(obs["max_util"][0]),
        "min_util": float(obs["min_util"][0]),
        "p95_util": float(obs["p95_util"][0]),
        "dropped_prev": float(obs["dropped_prev"][0]),
        "num_active_edges": int(obs["num_active_edges"][0]),
        "near_saturated_edges": int(obs["near_saturated_edges"][0]),
        "delivered": float(getattr(metrics, "delivered", 0.0)),
        "dropped": float(getattr(metrics, "dropped", 0.0)),
        "avg_delay_ms": float(getattr(metrics, "avg_delay_ms", 0.0)),
        "avg_path_latency_ms": float(getattr(metrics, "avg_path_latency_ms", 0.0)),
        "energy_kwh": float(getattr(metrics, "energy_kwh", 0.0)),
        "carbon_g": float(getattr(metrics, "carbon_g", 0.0)),
        "delta_energy_kwh": float(info.get("delta_energy_kwh", 0.0)),
        "delta_delivered": float(info.get("delta_delivered", 0.0)),
        "delta_dropped": float(info.get("delta_dropped", 0.0)),
        "delta_carbon_g": float(info.get("delta_carbon_g", 0.0)),
        "norm_drop_step": float(info.get("norm_drop_step", 0.0)),
        "norm_drop": float(info.get("norm_drop", 0.0)),
        "reward_energy": float(info.get("reward_energy", 0.0)),
        "reward_drop": float(info.get("reward_drop", 0.0)),
        "reward_qos": float(info.get("reward_qos", 0.0)),
        "reward_toggle": float(info.get("reward_toggle", 0.0)),
        "toggle_applied": bool(info.get("toggle_applied", False)),
        "toggle_reverted": bool(info.get("toggle_reverted", False)),
        "toggle_blocked_any": bool(info.get("toggle_blocked_any", False)),
        "toggle_blocked_cooldown": bool(info.get("toggle_blocked_cooldown", False)),
        "toggle_blocked_high_util": bool(info.get("toggle_blocked_high_util", False)),
        "toggle_blocked_global_cooldown": bool(info.get("toggle_blocked_global_cooldown", False)),
        "action_is_noop": bool(info.get("action_is_noop", False)),
        "action_is_invalid": bool(info.get("action_is_invalid", False)),
        "flows_count": int(flows_count),
        "flows_json": flows_json,
    }


def summarize_episodes(episode_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episode_rows:
        return {"episodes": [], "overall": {}}

    def _mean(vals: List[float]) -> float:
        return float(fmean(vals)) if vals else 0.0

    def _std(vals: List[float]) -> float:
        return float(pstdev(vals)) if len(vals) > 1 else 0.0

    totals = {
        "reward_total": [float(r["reward_total"]) for r in episode_rows],
        "delivered_total": [float(r["delivered_total"]) for r in episode_rows],
        "dropped_total": [float(r["dropped_total"]) for r in episode_rows],
        "energy_kwh_total": [float(r["energy_kwh_total"]) for r in episode_rows],
        "carbon_g_total": [float(r["carbon_g_total"]) for r in episode_rows],
        "steps": [int(r["steps"]) for r in episode_rows],
    }

    overall = {
        "episodes": len(episode_rows),
        "reward_total_mean": _mean(totals["reward_total"]),
        "reward_total_std": _std(totals["reward_total"]),
        "delivered_total_mean": _mean(totals["delivered_total"]),
        "delivered_total_std": _std(totals["delivered_total"]),
        "dropped_total_mean": _mean(totals["dropped_total"]),
        "dropped_total_std": _std(totals["dropped_total"]),
        "energy_kwh_total_mean": _mean(totals["energy_kwh_total"]),
        "energy_kwh_total_std": _std(totals["energy_kwh_total"]),
        "carbon_g_total_mean": _mean(totals["carbon_g_total"]),
        "carbon_g_total_std": _std(totals["carbon_g_total"]),
        "steps_mean": _mean([float(v) for v in totals["steps"]]),
        "steps_std": _std([float(v) for v in totals["steps"]]),
        "avg_utilization_mean": _mean([float(r["avg_utilization_mean"]) for r in episode_rows]),
        "active_ratio_mean": _mean([float(r["active_ratio_mean"]) for r in episode_rows]),
        "avg_delay_ms_mean": _mean([float(r["avg_delay_ms_mean"]) for r in episode_rows]),
    }

    return {"episodes": episode_rows, "overall": overall}


def run_episode(
    env: GreenNetEnv,
    action_fn: ActionFn,
    *,
    run_id: str,
    policy: str,
    scenario: str,
    seed: int,
    episode_seed: int,
    episode_idx: int,
    save_flows: bool,
    writer: csv.DictWriter,
) -> Dict[str, Any]:
    # Vary traffic per episode (if supported) while keeping topology fixed
    if hasattr(env.config, "traffic_seed"):
        env.config.traffic_seed = int(episode_seed) + 10_000
    obs, info = env.reset(seed=episode_seed)

    total_reward = 0.0
    total_delivered = 0.0
    total_dropped = 0.0
    total_energy = 0.0
    total_carbon = 0.0
    avg_util_sum = 0.0
    active_ratio_sum = 0.0
    avg_delay_sum = 0.0
    steps = 0
    carbon_prev = 0.0

    for step_idx in range(1, int(env.config.max_steps) + 1):
        action = action_fn(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)

        carbon_now = float(getattr(info.get("metrics"), "carbon_g", 0.0))
        delta_carbon = carbon_now - carbon_prev
        if delta_carbon < -1e-9:
            delta_carbon = 0.0
        carbon_prev = carbon_now
        info["delta_carbon_g"] = float(delta_carbon)

        writer.writerow(
            build_step_row(
                run_id,
                policy,
                scenario,
                seed,
                episode_seed,
                episode_idx,
                step_idx,
                int(action),
                float(reward),
                bool(terminated),
                bool(truncated),
                obs,
                info,
                save_flows=save_flows,
            )
        )

        total_reward += float(reward)
        total_delivered += float(info.get("delta_delivered", 0.0))
        total_dropped += float(info.get("delta_dropped", 0.0))
        total_energy += float(info.get("delta_energy_kwh", 0.0))
        total_carbon += float(delta_carbon)
        avg_util_sum += float(getattr(info.get("metrics"), "avg_utilization", 0.0))
        active_ratio_sum += float(obs["active_ratio"][0])
        avg_delay_sum += float(getattr(info.get("metrics"), "avg_delay_ms", 0.0))

        steps += 1
        if terminated or truncated:
            break

    denom = float(steps) if steps > 0 else 1.0
    return {
        "episode": int(episode_idx),
        "steps": int(steps),
        "reward_total": float(total_reward),
        "delivered_total": float(total_delivered),
        "dropped_total": float(total_dropped),
        "energy_kwh_total": float(total_energy),
        "carbon_g_total": float(total_carbon),
        "avg_utilization_mean": float(avg_util_sum / denom),
        "active_ratio_mean": float(active_ratio_sum / denom),
        "avg_delay_ms_mean": float(avg_delay_sum / denom),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GreenNet experiments and log per-step results.")
    parser.add_argument("--policy", choices=["noop", "baseline", "ppo"], required=True)
    parser.add_argument("--scenario", choices=["normal", "burst", "hotspot"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--save-flows", action="store_true", default=False)
    parser.add_argument("--topology-seed", type=int, default=None)
    parser.add_argument("--traffic-seed", type=int, default=None)
    args = parser.parse_args()

    seed_everything(int(args.seed))

    max_steps = args.max_steps
    if args.steps is not None:
        max_steps = args.steps

    env_config = build_env_config(args.scenario, max_steps)

    action_fn, policy_meta = load_policy(
        args.policy,
        runs_dir=args.runs_dir,
        model_path=args.model,
        deterministic=bool(args.deterministic),
    )

    # Choose topology seed: allow override; otherwise keep PPO on its training topology to avoid action-space mismatches.
    chosen_topology_seed: int | None = args.topology_seed
    if chosen_topology_seed is None and args.policy == "ppo":
        mp = policy_meta.get("model_path") if isinstance(policy_meta, dict) else None
        if mp:
            inferred = infer_topology_seed_from_model(Path(mp))
            if inferred is not None:
                chosen_topology_seed = inferred
    if chosen_topology_seed is None:
        chosen_topology_seed = int(args.seed)

    # Choose base traffic seed: allow override; otherwise seed+10000.
    chosen_traffic_seed_base: int | None = args.traffic_seed
    if chosen_traffic_seed_base is None:
        chosen_traffic_seed_base = int(args.seed) + 10_000

    if hasattr(env_config, "topology_seed"):
        env_config.topology_seed = int(chosen_topology_seed)
    if hasattr(env_config, "traffic_seed"):
        env_config.traffic_seed = int(chosen_traffic_seed_base)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")
    folder = f"{run_id}__policy-{args.policy}__scenario-{args.scenario}__seed-{args.seed}"
    if args.tag:
        folder = f"{folder}__tag-{args.tag}"
    out_dir = args.out_dir / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    save_env_config(out_dir, env_config)

    env = GreenNetEnv(config=env_config)

    per_step_path = out_dir / "per_step.csv"
    episode_summaries: List[Dict[str, Any]] = []

    with per_step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for ep in range(int(args.episodes)):
            episode_seed = int(args.seed) + ep
            episode_summaries.append(
                run_episode(
                    env,
                    action_fn,
                    run_id=run_id,
                    policy=args.policy,
                    scenario=args.scenario,
                    seed=int(args.seed),
                    episode_seed=episode_seed,
                    episode_idx=ep,
                    save_flows=bool(args.save_flows),
                    writer=writer,
                )
            )

    env.close()

    summary = summarize_episodes(episode_summaries)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_path_meta = policy_meta.get("model_path") if isinstance(policy_meta, dict) else None
    run_meta = {
        "run_id": run_id,
        "policy": args.policy,
        "scenario": args.scenario,
        "seed": int(args.seed),
        "topology_seed": int(chosen_topology_seed),
        "traffic_seed_base": int(chosen_traffic_seed_base),
        "episodes": int(args.episodes),
        "max_steps": int(env_config.max_steps),
        "deterministic": bool(args.deterministic),
        "save_flows": bool(args.save_flows),
        "model_path": model_path_meta,
        "runs_dir": str(args.runs_dir),
        "timestamp_utc": now.isoformat(),
        "command": " ".join(sys.argv),
    }
    run_meta.update(policy_meta)
    run_meta_path = out_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[run_experiment] results saved to {out_dir}")


if __name__ == "__main__":
    main()
