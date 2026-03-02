#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from greennet.env import EnvConfig, GreenNetEnv


_ENV_TUPLE_FIELDS = {
    "topology_seeds",
    "traffic_mice_size_range",
    "traffic_elephant_size_range",
    "traffic_duration_range",
    "traffic_spike_multiplier_range",
    "traffic_spike_duration_range",
}


def _coerce_env_value(key: str, value: Any) -> Any:
    if key == "traffic_hotspots" and isinstance(value, list):
        return tuple(tuple(item) for item in value)
    if key == "topology_seeds" and isinstance(value, list):
        return tuple(int(v) for v in value)
    if key in _ENV_TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)
    return value


def _extract_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("env", "env_config", "env_kwargs"):
        block = config.get(key)
        if isinstance(block, dict):
            out.update(block)

    for key in ("topology_seed", "topology_seeds", "topology_randomize", "traffic_seed"):
        if key in config and key not in out:
            out[key] = config[key]

    if "traffic_seed_base" in config and "traffic_seed" not in out:
        out["traffic_seed"] = config["traffic_seed_base"]

    if is_dataclass(EnvConfig):
        allowed = {f.name for f in fields(EnvConfig)}
        return {
            k: _coerce_env_value(k, v)
            for k, v in out.items()
            if k in allowed
        }
    return {k: _coerce_env_value(k, v) for k, v in out.items()}


def _load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def _apply_scenario(cfg: EnvConfig, scenario: str | None) -> None:
    if not scenario:
        return
    s = str(scenario).strip().lower()
    if s in ("normal", "diurnal"):
        cfg.traffic_model = "stochastic"
        cfg.traffic_scenario = "normal"
        return
    if s in ("burst", "hotspot", "anomaly"):
        cfg.traffic_model = "stochastic"
        cfg.traffic_scenario = s
        return
    raise ValueError(f"Unsupported scenario: {scenario}")


def _unpack_reset(ret: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(ret, tuple):
        if len(ret) >= 2:
            obs = ret[0]
            info = ret[1] if isinstance(ret[1], dict) else {}
            return obs, info
        if len(ret) == 1:
            return ret[0], {}
    return ret, {}


def _unpack_step(ret: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(ret, tuple):
        raise ValueError("env.step returned non-tuple")
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
    if len(ret) == 4:
        obs, reward, done, info = ret
        return obs, float(reward), bool(done), False, (info if isinstance(info, dict) else {})
    raise ValueError(f"Unsupported env.step output length: {len(ret)}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _metric_get(metrics: Any, key: str, default: float = 0.0) -> float:
    if metrics is None:
        return float(default)
    if isinstance(metrics, dict):
        return _safe_float(metrics.get(key, default), default)
    return _safe_float(getattr(metrics, key, default), default)


def _pick_action(env: GreenNetEnv, rng: np.random.Generator) -> int:
    # Bias sampling toward OFF toggles so the surrogate sees risky candidates.
    if getattr(env, "simulator", None) is not None and hasattr(env, "_edge_universe"):
        off_actions: list[int] = []
        try:
            for i, (u, v) in enumerate(env._edge_universe):  # type: ignore[attr-defined]
                key = env._edge_key(int(u), int(v))  # type: ignore[attr-defined]
                if bool(env.simulator.active.get(key, False)):
                    off_actions.append(i + 1)
        except Exception:
            off_actions = []
        if off_actions and float(rng.random()) < 0.70:
            return int(off_actions[int(rng.integers(0, len(off_actions)))])

    try:
        if hasattr(env, "get_action_mask"):
            mask = np.asarray(env.get_action_mask(), dtype=bool)
            valid = np.where(mask)[0]
            if valid.size > 0:
                return int(valid[int(rng.integers(0, valid.size))])
    except Exception:
        pass
    return int(env.action_space.sample())


def _extract_state_graph_features(env: GreenNetEnv, obs: Any) -> Tuple[list[str], np.ndarray, list[str], np.ndarray]:
    if hasattr(env, "get_cost_estimator_graph_state"):
        return env.get_cost_estimator_graph_state(obs)

    # Fallback for compatibility if helper is unavailable.
    g_names = [
        "time",
        "avg_util",
        "active_ratio",
        "max_util",
        "min_util",
        "p95_util",
        "demand_now",
        "demand_forecast",
        "dropped_prev",
        "num_active_edges",
        "near_saturated_edges",
    ]
    xg = np.zeros((len(g_names),), dtype=np.float32)
    if isinstance(obs, dict):
        for i, k in enumerate(g_names):
            arr = np.asarray(obs.get(k, [0.0]), dtype=np.float32).reshape(-1)
            xg[i] = float(arr[0]) if arr.size else 0.0

    e_names = ["edge_active", "edge_utilization"]
    if isinstance(obs, dict):
        a = np.asarray(obs.get("edge_active", []), dtype=np.float32).reshape(-1)
        u = np.asarray(obs.get("edge_util", []), dtype=np.float32).reshape(-1)
        e = min(a.size, u.size)
        xe = np.zeros((e, 2), dtype=np.float32)
        if e > 0:
            xe[:, 0] = a[:e]
            xe[:, 1] = u[:e]
    else:
        xe = np.zeros((0, 2), dtype=np.float32)

    return g_names, xg, e_names, xe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph-aware supervised dataset for Impact Predictor.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--traffic-seed-base", type=int, default=10_000)
    parser.add_argument("--scenario", type=str, default=None, help="normal|burst|hotspot|anomaly")
    parser.add_argument("--config", type=Path, default=None, help="JSON config with env/env_config/env_kwargs.")
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/cost_estimator/ds_graph.npz"),
    )
    args = parser.parse_args()

    config_blob = _load_config(args.config)
    cfg = EnvConfig()
    for key, value in _extract_env_overrides(config_blob).items():
        setattr(cfg, key, value)

    _apply_scenario(cfg, args.scenario)
    if args.max_steps is not None:
        cfg.max_steps = int(args.max_steps)

    qos_delay_ms = float(args.qos_delay_ms) if args.qos_delay_ms is not None else float(cfg.base_latency_ms) * 2.0
    qos_drop_max = float(args.qos_drop_max)

    # Never self-mask while collecting labels.
    cfg.cost_estimator_enabled = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    env = GreenNetEnv(config=cfg)

    xg_rows: list[np.ndarray] = []
    xe_rows: list[np.ndarray] = []
    a_rows: list[int] = []
    action_rows: list[int] = []
    is_off_rows: list[int] = []
    y_energy_rows: list[float] = []
    y_ddrop_rows: list[float] = []
    y_qos_rows: list[int] = []

    global_feature_names: list[str] | None = None
    edge_feature_names: list[str] | None = None

    for ep in range(int(args.episodes)):
        ep_seed = int(rng.integers(0, 1_000_000_000))
        if hasattr(env.config, "traffic_seed"):
            env.config.traffic_seed = int(args.traffic_seed_base) + ep

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
            norm_drop_total_now = _safe_float(info.get("norm_drop", info.get("norm_drop_step", 0.0)), 0.0)
            ddrop = float(norm_drop_total_now - prev_norm_drop_total)
            prev_norm_drop_total = float(norm_drop_total_now)
            path_delay_ms = _metric_get(metrics, "avg_path_latency_ms", -1.0)
            queue_delay_ms = _metric_get(metrics, "avg_delay_ms", 0.0)
            # Prefer path latency for QoS thresholding; fallback when unavailable.
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

            obs = next_obs

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[ds-graph] episodes={ep + 1}/{args.episodes} rows={len(xg_rows)}")

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
        qos_delay_ms=np.float32(qos_delay_ms),
        qos_drop_max=np.float32(qos_drop_max),
        global_feature_names=np.asarray(global_feature_names or [], dtype=object),
        edge_feature_names=np.asarray(edge_feature_names or [], dtype=object),
    )

    print(
        f"[ok] wrote {args.out} | rows={xg_arr.shape[0]} "
        f"Xg={tuple(xg_arr.shape)} Xe={tuple(xe_arr.shape)} "
        f"qos_delay_ms={qos_delay_ms:.3f} qos_drop_max={qos_drop_max:.3f} "
        f"y_qos_mean={float(y_qos_arr.mean()):.4f}"
    )


if __name__ == "__main__":
    main()
