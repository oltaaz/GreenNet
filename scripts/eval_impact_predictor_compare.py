#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from greennet.env import EnvConfig, GreenNetEnv


def _unpack_reset(ret: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(ret, tuple):
        if len(ret) >= 2:
            return ret[0], ret[1] if isinstance(ret[1], dict) else {}
        if len(ret) == 1:
            return ret[0], {}
    return ret, {}


def _unpack_step(ret: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(ret, tuple):
        raise RuntimeError("env.step returned non-tuple")
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
    if len(ret) == 4:
        obs, reward, done, info = ret
        return obs, float(reward), bool(done), False, (info if isinstance(info, dict) else {})
    raise RuntimeError(f"Unsupported env.step output length={len(ret)}")


def _metric_get(metrics: Any, key: str, default: float = 0.0) -> float:
    if metrics is None:
        return float(default)
    if isinstance(metrics, dict):
        try:
            return float(metrics.get(key, default))
        except Exception:
            return float(default)
    try:
        return float(getattr(metrics, key, default))
    except Exception:
        return float(default)


def _choose_action(env: GreenNetEnv, rng: np.random.Generator, p_off: float) -> Tuple[int, int]:
    if not bool(env._is_decision_step()):  # type: ignore[attr-defined]
        return 0, 0
    if float(rng.random()) >= float(p_off):
        return 0, 0
    off_actions: list[int] = []
    for i in range(1, int(env.action_space.n)):
        try:
            if bool(env._is_off_toggle_action(i)):  # type: ignore[attr-defined]
                off_actions.append(i)
        except Exception:
            continue
    if not off_actions:
        return 0, 0
    return int(off_actions[int(rng.integers(0, len(off_actions)))]), 1


@dataclass
class RolloutSummary:
    off_proposals: int
    off_candidates: int
    off_applied: int
    off_masked: int
    energy_mean: float
    dropped_mean: float
    norm_drop_mean: float
    avg_path_latency_ms_mean: float
    impact_enabled_rate: float
    mask_reason_counts: Dict[str, int]


def _run_condition(
    *,
    enabled: bool,
    episodes: int,
    max_steps: int,
    p_off: float,
    seed: int,
    model_dir: str,
) -> RolloutSummary:
    cfg = EnvConfig(
        max_steps=int(max_steps),
        cost_estimator_enabled=bool(enabled),
        cost_estimator_model_dir=str(model_dir),
    )
    env = GreenNetEnv(cfg)
    rng = np.random.default_rng(int(seed))
    ep_seeds = [int(seed) + i for i in range(int(episodes))]

    off_proposals = 0
    off_candidates = 0
    off_applied = 0
    off_masked = 0
    impact_enabled_steps = 0
    mask_reason_counts: Counter[str] = Counter()

    energy_ep: list[float] = []
    dropped_ep: list[float] = []
    norm_drop_ep: list[float] = []
    path_lat_step_vals: list[float] = []

    for ep_seed in ep_seeds:
        obs, _ = _unpack_reset(env.reset(seed=int(ep_seed)))
        done = False
        t = 0
        ep_energy = 0.0
        ep_dropped = 0.0
        ep_norm_drop = 0.0
        while not done and t < int(max_steps):
            action, proposed_off = _choose_action(env, rng, p_off)
            off_proposals += int(proposed_off)

            obs, _reward, term, trunc, info = _unpack_step(env.step(action))
            done = bool(term or trunc)
            t += 1

            is_off_candidate = int(info.get("is_off_candidate", 0))
            off_candidates += int(is_off_candidate)
            off_masked += int(info.get("forced_noop_by_cost_estimator", 0))
            if is_off_candidate and bool(info.get("toggled_off", False)):
                off_applied += 1

            impact_enabled_steps += int(info.get("impact_enabled", 0))
            reason = info.get("mask_reason")
            if isinstance(reason, str) and reason:
                mask_reason_counts[reason] += 1

            ep_energy += float(info.get("delta_energy_kwh", 0.0))
            ep_dropped += float(info.get("delta_dropped", 0.0))
            ep_norm_drop = float(info.get("norm_drop", ep_norm_drop))
            metrics = info.get("metrics")
            path_lat_step_vals.append(_metric_get(metrics, "avg_path_latency_ms", 0.0))

        energy_ep.append(float(ep_energy))
        dropped_ep.append(float(ep_dropped))
        norm_drop_ep.append(float(ep_norm_drop))

    env.close()
    step_count = max(1, len(path_lat_step_vals))
    return RolloutSummary(
        off_proposals=int(off_proposals),
        off_candidates=int(off_candidates),
        off_applied=int(off_applied),
        off_masked=int(off_masked),
        energy_mean=float(np.mean(energy_ep)) if energy_ep else 0.0,
        dropped_mean=float(np.mean(dropped_ep)) if dropped_ep else 0.0,
        norm_drop_mean=float(np.mean(norm_drop_ep)) if norm_drop_ep else 0.0,
        avg_path_latency_ms_mean=float(np.mean(path_lat_step_vals)) if path_lat_step_vals else 0.0,
        impact_enabled_rate=float(impact_enabled_steps / step_count),
        mask_reason_counts=dict(mask_reason_counts),
    )


def _print_table(off: RolloutSummary, on: RolloutSummary) -> None:
    rows = [
        ("off_proposals", float(off.off_proposals), float(on.off_proposals), False),
        ("off_candidates", float(off.off_candidates), float(on.off_candidates), False),
        ("off_applied", float(off.off_applied), float(on.off_applied), False),
        ("off_masked", float(off.off_masked), float(on.off_masked), False),
        ("%masked", (100.0 * off.off_masked / max(1, off.off_candidates)), (100.0 * on.off_masked / max(1, on.off_candidates)), True),
        ("energy_mean", off.energy_mean, on.energy_mean, True),
        ("dropped_mean", off.dropped_mean, on.dropped_mean, True),
        ("norm_drop_mean", off.norm_drop_mean, on.norm_drop_mean, True),
        ("avg_path_latency_ms_mean", off.avg_path_latency_ms_mean, on.avg_path_latency_ms_mean, True),
    ]
    print("\nImpact Predictor Compare (OFF vs ON)")
    print(f"{'metric':<28} {'OFF':>14} {'ON':>14}")
    print("-" * 58)
    for name, off_v, on_v, is_float in rows:
        if is_float:
            print(f"{name:<28} {off_v:>14.6f} {on_v:>14.6f}")
        else:
            print(f"{name:<28} {int(off_v):>14d} {int(on_v):>14d}")
    print("-" * 58)
    print("mask_reason_counts_on", json.dumps(on.mask_reason_counts, sort_keys=True))
    print("mask_reason_counts_off", json.dumps(off.mask_reason_counts, sort_keys=True))
    print(
        "summary_json",
        json.dumps(
            {
                "off": off.__dict__,
                "on": on.__dict__,
            },
            sort_keys=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Impact Predictor OFF vs ON with OFF-biased action proposals.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--p-off", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default="models/impact_predictor")
    args = parser.parse_args()

    off = _run_condition(
        enabled=False,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        p_off=float(args.p_off),
        seed=int(args.seed),
        model_dir=str(args.model_dir),
    )
    on = _run_condition(
        enabled=True,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        p_off=float(args.p_off),
        seed=int(args.seed),
        model_dir=str(args.model_dir),
    )
    _print_table(off, on)
    if on.off_candidates > 0 and on.off_masked <= 0:
        print("[warn] Impact Predictor ON had off_candidates > 0 but off_masked == 0.")


if __name__ == "__main__":
    main()
