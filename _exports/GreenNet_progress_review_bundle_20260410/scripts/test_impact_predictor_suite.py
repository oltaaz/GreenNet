#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from greennet.env import EnvConfig, GreenNetEnv

try:
    from build_cost_dataset_graph import _apply_scenario
except Exception:  # pragma: no cover
    from scripts.build_cost_dataset_graph import _apply_scenario


def _parse_int_csv(text: str) -> list[int]:
    out = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer value")
    return out


def _parse_str_csv(text: str) -> list[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one value")
    return out


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
class RunSummary:
    off_proposals: int
    off_candidates: int
    off_applied: int
    off_masked: int
    internal_forced_noop: int
    energy_mean: float
    dropped_mean: float
    norm_drop_mean: float
    avg_path_latency_ms_mean: float
    mask_reason_counts: Dict[str, int]
    internal_noop_reason_counts: Dict[str, int]


def _args_to_jsonable(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    out["argv"] = list(sys.argv)
    return out


def _lock_artifacts(
    *,
    args: argparse.Namespace,
    rows: list[Dict[str, Any]],
) -> Path:
    root = Path(args.lock_root)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    lock_dir = root / ts
    lock_dir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv)
    if out_csv.exists():
        shutil.copy2(out_csv, lock_dir / "report.csv")

    model_dir = Path(args.model_dir)
    meta_path = model_dir / "meta.json"
    report_path = model_dir / "report.json"
    if meta_path.exists():
        shutil.copy2(meta_path, lock_dir / "meta.json")
    locked_report_path = lock_dir / "report.json"
    if report_path.exists():
        shutil.copy2(report_path, locked_report_path)
    else:
        with locked_report_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "missing": True,
                    "source": str(report_path),
                    "note": "Train with updated scripts/train_cost_estimator_torch.py to produce this file.",
                },
                f,
                indent=2,
                sort_keys=True,
            )

    run_args_payload = _args_to_jsonable(args)
    run_args_payload["row_count"] = int(len(rows))
    run_args_payload["timestamp"] = ts
    run_args_path = lock_dir / "run_args.json"
    with run_args_path.open("w", encoding="utf-8") as f:
        json.dump(run_args_payload, f, indent=2, sort_keys=True)
    return lock_dir


def _run_condition(
    *,
    enabled: bool,
    scenario: str,
    demand_scale: float,
    capacity_scale: float,
    episodes: int,
    max_steps: int,
    p_off: float,
    seed: int,
    model_dir: str,
    topology_seeds: Iterable[int],
    traffic_seeds: Iterable[int],
) -> RunSummary:
    topo = [int(x) for x in topology_seeds]
    traf = [int(x) for x in traffic_seeds]
    if not topo or not traf:
        raise ValueError("topology_seeds and traffic_seeds must be non-empty")

    cfg = EnvConfig(
        max_steps=int(max_steps),
        cost_estimator_enabled=bool(enabled),
        cost_estimator_model_dir=str(model_dir),
        traffic_model="stochastic",
        topology_randomize=True,
        topology_seed=int(topo[0]),
        topology_seeds=tuple(topo),
    )
    _apply_scenario(cfg, scenario)
    cfg.base_capacity = float(cfg.base_capacity) * float(capacity_scale)
    cfg.demand_min = float(cfg.demand_min) * float(demand_scale)
    cfg.demand_max = max(float(cfg.demand_min), float(cfg.demand_max) * float(demand_scale))
    # Keep internal OFF blocks from dominating the acceptance signal.
    cfg.qos_target_norm_drop = 1.0
    cfg.qos_guard_margin_off = 0.0
    cfg.off_calm_steps_required = 0
    cfg.util_block_threshold = max(float(cfg.util_block_threshold), 0.99)
    cfg.max_util_off_allow_threshold = max(float(cfg.max_util_off_allow_threshold), 0.99)
    cfg.max_off_toggles_per_episode = max(int(cfg.max_off_toggles_per_episode), 5)

    env = GreenNetEnv(cfg)
    rng = np.random.default_rng(int(seed))
    off_proposals = 0
    off_candidates = 0
    off_applied = 0
    off_masked = 0
    internal_forced_noop = 0
    mask_reason_counts: Counter[str] = Counter()
    internal_noop_reason_counts: Counter[str] = Counter()
    ep_energy: list[float] = []
    ep_dropped: list[float] = []
    ep_norm_drop: list[float] = []
    path_lat_vals: list[float] = []

    try:
        for ep in range(int(episodes)):
            topo_seed = int(topo[ep % len(topo)])
            traffic_seed = int(traf[ep % len(traf)])
            env.config.topology_seeds = (topo_seed,)
            env.config.traffic_seed = traffic_seed

            obs, _ = _unpack_reset(env.reset(seed=int(seed) + ep))
            done = False
            t = 0
            energy_ep = 0.0
            dropped_ep = 0.0
            norm_ep = 0.0
            while not done and t < int(max_steps):
                action, proposed_off = _choose_action(env, rng, float(p_off))
                off_proposals += int(proposed_off)
                obs, _reward, term, trunc, info = _unpack_step(env.step(action))
                done = bool(term or trunc)
                t += 1

                off_candidates += int(info.get("is_off_candidate", 0))
                off_masked += int(info.get("forced_noop_by_cost_estimator", 0))
                internal_forced_noop += int(info.get("internal_forced_noop", 0))
                internal_reason = info.get("internal_noop_reason")
                if isinstance(internal_reason, str) and internal_reason:
                    internal_noop_reason_counts[internal_reason] += 1
                was_active = info.get("target_edge_was_active")
                is_active = info.get("target_edge_is_active")
                if was_active == 1 and is_active == 0:
                    off_applied += 1

                reason = info.get("mask_reason")
                if isinstance(reason, str) and reason:
                    mask_reason_counts[reason] += 1

                energy_ep += float(info.get("delta_energy_kwh", 0.0))
                dropped_ep += float(info.get("delta_dropped", 0.0))
                norm_ep = float(info.get("norm_drop", norm_ep))
                path_lat_vals.append(_metric_get(info.get("metrics"), "avg_path_latency_ms", 0.0))

            ep_energy.append(float(energy_ep))
            ep_dropped.append(float(dropped_ep))
            ep_norm_drop.append(float(norm_ep))
    finally:
        env.close()

    return RunSummary(
        off_proposals=int(off_proposals),
        off_candidates=int(off_candidates),
        off_applied=int(off_applied),
        off_masked=int(off_masked),
        internal_forced_noop=int(internal_forced_noop),
        energy_mean=float(np.mean(ep_energy)) if ep_energy else 0.0,
        dropped_mean=float(np.mean(ep_dropped)) if ep_dropped else 0.0,
        norm_drop_mean=float(np.mean(ep_norm_drop)) if ep_norm_drop else 0.0,
        avg_path_latency_ms_mean=float(np.mean(path_lat_vals)) if path_lat_vals else 0.0,
        mask_reason_counts=dict(mask_reason_counts),
        internal_noop_reason_counts=dict(internal_noop_reason_counts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Impact Predictor acceptance suite across scenarios and stress buckets.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--p-off", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default="models/impact_predictor")
    parser.add_argument("--scenarios", type=str, default="normal,burst,hotspot")
    parser.add_argument("--topology-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--traffic-seeds", type=str, default="0,1,2")
    parser.add_argument("--abs-tol-lat-ms", type=float, default=2.0)
    parser.add_argument("--tol-norm-drop", type=float, default=0.002)
    parser.add_argument("--tol-dropped", type=float, default=5.0)
    parser.add_argument("--ood-abs-tol-lat-ms", type=float, default=5.0)
    parser.add_argument("--ood-tol-norm-drop", type=float, default=0.005)
    parser.add_argument("--ood-tol-dropped", type=float, default=15.0)
    parser.add_argument("--ood-demand-scale", type=float, default=1.8)
    parser.add_argument("--ood-capacity-scale", type=float, default=0.7)
    parser.add_argument("--include-ood", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("artifacts/impact_predictor_suite/report.csv"),
    )
    parser.add_argument("--lock-artifacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--lock-root",
        type=Path,
        default=Path("artifacts/locked/impact_predictor"),
    )
    args = parser.parse_args()

    scenarios = _parse_str_csv(args.scenarios)
    topology_seeds = _parse_int_csv(args.topology_seeds)
    traffic_seeds = _parse_int_csv(args.traffic_seeds)
    buckets: list[tuple[str, float, float, int]] = [
        ("mild", 1.0, 1.0, 0),
        ("stress", 1.4, 0.8, 0),
    ]
    if bool(args.include_ood):
        buckets.append(
            (
                "extreme",
                float(args.ood_demand_scale),
                float(args.ood_capacity_scale),
                1,
            )
        )

    rows: list[Dict[str, Any]] = []
    failures: list[str] = []

    for scenario in scenarios:
        for bucket_name, demand_scale, capacity_scale, ood_bucket in buckets:
            off = _run_condition(
                enabled=False,
                scenario=scenario,
                demand_scale=float(demand_scale),
                capacity_scale=float(capacity_scale),
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                p_off=float(args.p_off),
                seed=int(args.seed),
                model_dir=str(args.model_dir),
                topology_seeds=topology_seeds,
                traffic_seeds=traffic_seeds,
            )
            on = _run_condition(
                enabled=True,
                scenario=scenario,
                demand_scale=float(demand_scale),
                capacity_scale=float(capacity_scale),
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                p_off=float(args.p_off),
                seed=int(args.seed),
                model_dir=str(args.model_dir),
                topology_seeds=topology_seeds,
                traffic_seeds=traffic_seeds,
            )

            lat_tol_ms = float(args.ood_abs_tol_lat_ms) if int(ood_bucket) == 1 else float(args.abs_tol_lat_ms)
            norm_drop_tol = (
                float(args.ood_tol_norm_drop) if int(ood_bucket) == 1 else float(args.tol_norm_drop)
            )
            dropped_tol = (
                float(args.ood_tol_dropped) if int(ood_bucket) == 1 else float(args.tol_dropped)
            )
            lat_ok = bool(
                float(on.avg_path_latency_ms_mean)
                <= float(off.avg_path_latency_ms_mean) + float(lat_tol_ms)
            )
            norm_drop_ok = bool(
                float(on.norm_drop_mean) <= float(off.norm_drop_mean) + float(norm_drop_tol)
            )
            dropped_ok = bool(
                float(on.dropped_mean) <= float(off.dropped_mean) + float(dropped_tol)
            )
            safety_ok = bool(lat_ok and norm_drop_ok and dropped_ok)
            stress_mask_ok = True
            stress_reason_ok = True
            if bucket_name == "stress" and int(on.off_candidates) > 0:
                stress_mask_ok = bool(int(on.off_masked) > 0)
                stress_reason_ok = bool(len(on.mask_reason_counts) > 0)
            ood_off_candidates_ok = True
            ood_off_masked_ok = True
            if int(ood_bucket) == 1:
                ood_off_candidates_ok = bool(int(on.off_candidates) > 0)
                ood_off_masked_ok = bool(int(on.off_masked) > 0)
            mild_off_applied_ok = True
            if bucket_name == "mild":
                mild_off_applied_ok = bool(int(off.off_applied) > 0)
            passed = bool(
                safety_ok
                and stress_mask_ok
                and stress_reason_ok
                and ood_off_candidates_ok
                and ood_off_masked_ok
                and mild_off_applied_ok
            )

            row: Dict[str, Any] = {
                "scenario": scenario,
                "bucket": bucket_name,
                "ood_bucket": int(ood_bucket),
                "demand_scale": float(demand_scale),
                "capacity_scale": float(capacity_scale),
                "off_proposals": int(on.off_proposals),
                "off_candidates": int(on.off_candidates),
                "off_applied_off": int(off.off_applied),
                "off_applied_on": int(on.off_applied),
                "off_masked_on": int(on.off_masked),
                "internal_forced_noop_off": int(off.internal_forced_noop),
                "internal_forced_noop_on": int(on.internal_forced_noop),
                "masked_pct_on": float(100.0 * on.off_masked / max(1, on.off_candidates)),
                "energy_mean_off": float(off.energy_mean),
                "energy_mean_on": float(on.energy_mean),
                "dropped_mean_off": float(off.dropped_mean),
                "dropped_mean_on": float(on.dropped_mean),
                "norm_drop_mean_off": float(off.norm_drop_mean),
                "norm_drop_mean_on": float(on.norm_drop_mean),
                "avg_path_latency_ms_mean_off": float(off.avg_path_latency_ms_mean),
                "avg_path_latency_ms_mean_on": float(on.avg_path_latency_ms_mean),
                "lat_tol_ms_used": float(lat_tol_ms),
                "norm_drop_tol_used": float(norm_drop_tol),
                "dropped_tol_used": float(dropped_tol),
                "lat_ok": int(lat_ok),
                "norm_drop_ok": int(norm_drop_ok),
                "dropped_ok": int(dropped_ok),
                "safety_ok": int(safety_ok),
                "stress_mask_ok": int(stress_mask_ok),
                "stress_reason_ok": int(stress_reason_ok),
                "ood_off_candidates_ok": int(ood_off_candidates_ok),
                "ood_off_masked_ok": int(ood_off_masked_ok),
                "mild_off_applied_ok": int(mild_off_applied_ok),
                "pass": int(passed),
                "mask_reason_counts_on": json.dumps(on.mask_reason_counts, sort_keys=True),
                "internal_noop_reason_counts_off": json.dumps(off.internal_noop_reason_counts, sort_keys=True),
                "internal_noop_reason_counts_on": json.dumps(on.internal_noop_reason_counts, sort_keys=True),
            }
            rows.append(row)
            if not passed:
                failures.append(f"{scenario}:{bucket_name}")

    print("\nImpact Predictor Acceptance Suite")
    print(
        f"{'scenario':<10} {'bucket':<8} {'ood':>3} {'off_cand':>8} {'off_app':>8} {'off_mask':>8} {'int_noop':>8} {'%mask':>8} "
        f"{'lat_off':>10} {'lat_on':>10} {'nd_off':>10} {'nd_on':>10} {'PASS':>6}"
    )
    print("-" * 132)
    for row in rows:
        print(
            f"{row['scenario']:<10} {row['bucket']:<8} {int(row['ood_bucket']):>3d} "
            f"{int(row['off_candidates']):>8d} "
            f"{int(row['off_applied_off']):>8d} {int(row['off_masked_on']):>8d} "
            f"{int(row['internal_forced_noop_on']):>8d} {float(row['masked_pct_on']):>8.2f} "
            f"{float(row['avg_path_latency_ms_mean_off']):>10.3f} {float(row['avg_path_latency_ms_mean_on']):>10.3f} "
            f"{float(row['norm_drop_mean_off']):>10.4f} {float(row['norm_drop_mean_on']):>10.4f} "
            f"{('PASS' if int(row['pass']) else 'FAIL'):>6}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "bucket",
        "ood_bucket",
        "demand_scale",
        "capacity_scale",
        "off_proposals",
        "off_candidates",
        "off_applied_off",
        "off_applied_on",
        "off_masked_on",
        "internal_forced_noop_off",
        "internal_forced_noop_on",
        "masked_pct_on",
        "energy_mean_off",
        "energy_mean_on",
        "dropped_mean_off",
        "dropped_mean_on",
        "norm_drop_mean_off",
        "norm_drop_mean_on",
        "avg_path_latency_ms_mean_off",
        "avg_path_latency_ms_mean_on",
        "lat_tol_ms_used",
        "norm_drop_tol_used",
        "dropped_tol_used",
        "lat_ok",
        "norm_drop_ok",
        "dropped_ok",
        "safety_ok",
        "stress_mask_ok",
        "stress_reason_ok",
        "ood_off_candidates_ok",
        "ood_off_masked_ok",
        "mild_off_applied_ok",
        "pass",
        "mask_reason_counts_on",
        "internal_noop_reason_counts_off",
        "internal_noop_reason_counts_on",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"[saved] {args.out_csv}")
    if bool(args.lock_artifacts):
        lock_dir = _lock_artifacts(args=args, rows=rows)
        print(f"[saved] locked artifacts -> {lock_dir}")

    if failures:
        print(f"[FAIL] acceptance checks failed for: {', '.join(failures)}")
        raise SystemExit(1)
    print("[PASS] all acceptance checks passed")


if __name__ == "__main__":
    main()
