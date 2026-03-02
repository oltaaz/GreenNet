#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from greennet.env import EnvConfig, GreenNetEnv

try:
    from build_cost_dataset_graph import (
        _apply_scenario,
        _extract_env_overrides,
        _extract_state_graph_features,
        _load_config,
        _metric_get,
        _pick_action,
        _safe_float,
        _unpack_reset,
        _unpack_step,
    )
except Exception:  # pragma: no cover
    from scripts.build_cost_dataset_graph import (
        _apply_scenario,
        _extract_env_overrides,
        _extract_state_graph_features,
        _load_config,
        _metric_get,
        _pick_action,
        _safe_float,
        _unpack_reset,
        _unpack_step,
    )


def _parse_int_csv(text: str) -> list[int]:
    out: list[int] = []
    for raw in str(text).split(","):
        s = raw.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("Expected at least one integer value")
    return out


def _parse_str_csv(text: str) -> list[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one value")
    return out


def _scale_int_range(rng: tuple[int, int], scale: float) -> tuple[int, int]:
    lo = max(1, int(round(float(rng[0]) * float(scale))))
    hi = max(lo, int(round(float(rng[1]) * float(scale))))
    return (int(lo), int(hi))


def _apply_domain_randomization(
    cfg: EnvConfig,
    base_cfg: EnvConfig,
    *,
    demand_scale: float,
    capacity_scale: float,
    flows_scale: float,
) -> None:
    cfg.base_capacity = max(1e-6, float(base_cfg.base_capacity) * float(capacity_scale))
    cfg.demand_min = max(1e-6, float(base_cfg.demand_min) * float(demand_scale))
    cfg.demand_max = max(float(cfg.demand_min), float(base_cfg.demand_max) * float(demand_scale))
    cfg.flows_per_step = max(1, int(round(float(base_cfg.flows_per_step) * float(flows_scale))))
    cfg.traffic_avg_bursts_per_step = max(
        0.1, float(base_cfg.traffic_avg_bursts_per_step) * float(flows_scale)
    )
    cfg.traffic_mice_size_range = _scale_int_range(base_cfg.traffic_mice_size_range, float(demand_scale))
    cfg.traffic_elephant_size_range = _scale_int_range(base_cfg.traffic_elephant_size_range, float(demand_scale))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-scenario graph dataset for Impact Predictor.")
    parser.add_argument("--episodes-per-scenario", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--scenarios", type=str, default="normal,burst,hotspot")
    parser.add_argument("--topology-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--traffic-seeds", type=str, default="0,1,2")
    parser.add_argument("--config", type=Path, default=None, help="JSON config with env/env_config/env_kwargs.")
    parser.add_argument("--demand-scale-min", type=float, default=0.7)
    parser.add_argument("--demand-scale-max", type=float, default=1.6)
    parser.add_argument("--capacity-scale-min", type=float, default=0.7)
    parser.add_argument("--capacity-scale-max", type=float, default=1.3)
    parser.add_argument("--flows-scale-min", type=float, default=0.8)
    parser.add_argument("--flows-scale-max", type=float, default=1.5)
    parser.add_argument(
        "--qos-delay-ms",
        type=float,
        default=None,
        help="QoS delay threshold in ms. Default: 2x base_latency_ms from EnvConfig.",
    )
    parser.add_argument(
        "--qos-drop-max",
        type=float,
        default=1.0,
        help="QoS drop threshold (absolute dropped units per step).",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/cost_estimator/ds_sweep.npz"))
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    scenarios = _parse_str_csv(args.scenarios)
    topology_seeds = _parse_int_csv(args.topology_seeds)
    traffic_seeds = _parse_int_csv(args.traffic_seeds)

    config_blob = _load_config(args.config)
    base_cfg = EnvConfig()
    for key, value in _extract_env_overrides(config_blob).items():
        setattr(base_cfg, key, value)
    if args.max_steps is not None:
        base_cfg.max_steps = int(args.max_steps)
    base_cfg.cost_estimator_enabled = False
    base_cfg.traffic_model = "stochastic"
    base_cfg.topology_randomize = True
    base_cfg.topology_seed = int(topology_seeds[0])
    base_cfg.topology_seeds = tuple(int(x) for x in topology_seeds)

    qos_delay_ms = (
        float(args.qos_delay_ms)
        if args.qos_delay_ms is not None
        else float(base_cfg.base_latency_ms) * 2.0
    )
    qos_drop_max = float(args.qos_drop_max)

    xg_rows: list[np.ndarray] = []
    xe_rows: list[np.ndarray] = []
    a_rows: list[int] = []
    action_rows: list[int] = []
    is_off_rows: list[int] = []
    y_energy_rows: list[float] = []
    y_ddrop_rows: list[float] = []
    y_qos_rows: list[int] = []
    scenario_id_rows: list[int] = []
    topology_seed_rows: list[int] = []
    traffic_seed_rows: list[int] = []
    demand_scale_rows: list[float] = []
    capacity_scale_rows: list[float] = []
    flows_scale_rows: list[float] = []

    global_feature_names: list[str] | None = None
    edge_feature_names: list[str] | None = None
    scenario_names = [str(s).strip().lower() for s in scenarios]

    episodes_per_scenario = int(args.episodes_per_scenario)
    total_episodes = episodes_per_scenario * len(scenario_names)
    done_episodes = 0

    cfg = replace(base_cfg)
    env = GreenNetEnv(config=cfg)
    try:
        for scenario_id, scenario in enumerate(scenario_names):
            for _ in range(episodes_per_scenario):
                topo_seed = int(topology_seeds[int(rng.integers(0, len(topology_seeds)))])
                traffic_seed = int(traffic_seeds[int(rng.integers(0, len(traffic_seeds)))])
                demand_scale = float(
                    rng.uniform(float(args.demand_scale_min), float(args.demand_scale_max))
                )
                capacity_scale = float(
                    rng.uniform(float(args.capacity_scale_min), float(args.capacity_scale_max))
                )
                flows_scale = float(
                    rng.uniform(float(args.flows_scale_min), float(args.flows_scale_max))
                )

                env.config.topology_randomize = True
                env.config.topology_seed = int(topology_seeds[0])
                env.config.topology_seeds = (int(topo_seed),)
                env.config.traffic_seed = int(traffic_seed)
                _apply_scenario(env.config, scenario)
                _apply_domain_randomization(
                    env.config,
                    base_cfg,
                    demand_scale=demand_scale,
                    capacity_scale=capacity_scale,
                    flows_scale=flows_scale,
                )

                ep_seed = int(rng.integers(0, 1_000_000_000))
                obs, _ = _unpack_reset(env.reset(seed=ep_seed))
                prev_norm_drop_total = 0.0

                done = False
                while not done:
                    g_names, xg, e_names, xe = _extract_state_graph_features(env, obs)
                    if global_feature_names is None:
                        global_feature_names = list(g_names)
                    if edge_feature_names is None:
                        edge_feature_names = list(e_names)

                    action = _pick_action(env, rng)
                    action_edge_id = int(action) - 1 if int(action) > 0 else -1

                    is_off = 0
                    if int(action) > 0:
                        if hasattr(env, "_is_off_toggle_action"):
                            try:
                                is_off = int(bool(env._is_off_toggle_action(int(action))))  # type: ignore[attr-defined]
                            except Exception:
                                is_off = 0
                        if is_off == 0 and action_edge_id >= 0 and action_edge_id < int(xe.shape[0]):
                            active_col = 0
                            if "edge_active" in e_names:
                                active_col = int(e_names.index("edge_active"))
                            if xe.ndim == 2 and xe.shape[1] > active_col:
                                is_off = int(float(xe[action_edge_id, active_col]) > 0.5)

                    next_obs, _reward, terminated, truncated, info = _unpack_step(env.step(action))
                    done = bool(terminated or truncated)
                    metrics = info.get("metrics")

                    denergy = _safe_float(info.get("delta_energy_kwh", 0.0), 0.0)
                    norm_drop_total_now = _safe_float(
                        info.get("norm_drop", info.get("norm_drop_step", 0.0)), 0.0
                    )
                    ddrop = float(norm_drop_total_now - prev_norm_drop_total)
                    prev_norm_drop_total = float(norm_drop_total_now)
                    path_delay_ms = _metric_get(metrics, "avg_path_latency_ms", -1.0)
                    queue_delay_ms = _metric_get(metrics, "avg_delay_ms", 0.0)
                    delay_ms = float(path_delay_ms if path_delay_ms >= 0.0 else queue_delay_ms)
                    dropped = _metric_get(metrics, "dropped", 0.0)
                    qos_next = int((delay_ms > qos_delay_ms) or (dropped > qos_drop_max))

                    xg_rows.append(np.asarray(xg, dtype=np.float32))
                    xe_rows.append(np.asarray(xe, dtype=np.float32))
                    a_rows.append(int(action_edge_id))
                    action_rows.append(int(action))
                    is_off_rows.append(int(is_off))
                    y_energy_rows.append(float(denergy))
                    y_ddrop_rows.append(float(ddrop))
                    y_qos_rows.append(int(qos_next))
                    scenario_id_rows.append(int(scenario_id))
                    topology_seed_rows.append(int(topo_seed))
                    traffic_seed_rows.append(int(traffic_seed))
                    demand_scale_rows.append(float(demand_scale))
                    capacity_scale_rows.append(float(capacity_scale))
                    flows_scale_rows.append(float(flows_scale))

                    obs = next_obs

                done_episodes += 1
                if done_episodes == 1 or done_episodes % 10 == 0:
                    print(
                        f"[ds-sweep] episodes={done_episodes}/{total_episodes} "
                        f"scenario={scenario} rows={len(xg_rows)}"
                    )
    finally:
        env.close()

    if not xg_rows:
        raise RuntimeError("No samples collected")

    xg_arr = np.stack(xg_rows, axis=0).astype(np.float32, copy=False)
    xe_arr = np.stack(xe_rows, axis=0).astype(np.float32, copy=False)
    a_arr = np.asarray(a_rows, dtype=np.int64)
    action_arr = np.asarray(action_rows, dtype=np.int64)
    is_off_arr = np.asarray(is_off_rows, dtype=np.int8)
    y_energy_arr = np.asarray(y_energy_rows, dtype=np.float32)
    y_ddrop_arr = np.asarray(y_ddrop_rows, dtype=np.float32)
    y_qos_arr = np.asarray(y_qos_rows, dtype=np.int8)
    scenario_id_arr = np.asarray(scenario_id_rows, dtype=np.int16)
    topology_seed_arr = np.asarray(topology_seed_rows, dtype=np.int64)
    traffic_seed_arr = np.asarray(traffic_seed_rows, dtype=np.int64)
    demand_scale_arr = np.asarray(demand_scale_rows, dtype=np.float32)
    capacity_scale_arr = np.asarray(capacity_scale_rows, dtype=np.float32)
    flows_scale_arr = np.asarray(flows_scale_rows, dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        Xg=xg_arr,
        Xe=xe_arr,
        a=a_arr,
        action=action_arr,
        is_off=is_off_arr,
        y_delta_energy=y_energy_arr,
        y_delta_norm_drop=y_ddrop_arr,
        y_qos=y_qos_arr,
        scenario_id=scenario_id_arr,
        scenario_names=np.asarray(scenario_names, dtype=object),
        topology_seed=topology_seed_arr,
        traffic_seed=traffic_seed_arr,
        demand_scale=demand_scale_arr,
        capacity_scale=capacity_scale_arr,
        flows_scale=flows_scale_arr,
        qos_delay_ms=np.float32(qos_delay_ms),
        qos_drop_max=np.float32(qos_drop_max),
        global_feature_names=np.asarray(global_feature_names or [], dtype=object),
        edge_feature_names=np.asarray(edge_feature_names or [], dtype=object),
    )

    print(
        f"[ok] wrote {args.out} | rows={xg_arr.shape[0]} Xg={tuple(xg_arr.shape)} Xe={tuple(xe_arr.shape)} "
        f"scenarios={scenario_names} y_qos_mean={float(y_qos_arr.mean()):.4f}"
    )


if __name__ == "__main__":
    main()
