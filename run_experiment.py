#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from greennet.baselines import (
    action_sleep_if_idle,
    canonical_controller_policy_name,
    controller_policy_class,
)
from greennet.env import EnvConfig, GreenNetEnv
from greennet.persistence import persist_run_directory
from greennet.routing import (
    canonicalize_routing_baseline_name,
    canonicalize_routing_link_cost_model,
)
from greennet.utils.config import load_env_config_from_run, resolve_env_paths_in_config, save_env_config

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


def _load_config_file(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    cfg = _load_json_file(path)
    if cfg is None:
        raise ValueError(f"Config file is not valid JSON: {path}")
    return resolve_env_paths_in_config(cfg, base_dir=path.parent)


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
    env_overrides: Dict[str, Any] = {}
    for key in ("env", "env_config", "env_kwargs"):
        block = config.get(key)
        if isinstance(block, dict):
            env_overrides.update(block)

    for key in (
        "topology_seed",
        "topology_seeds",
        "topology_randomize",
        "topology_name",
        "topology_path",
        "traffic_seed",
        "traffic_name",
        "traffic_path",
    ):
        if key in config and key not in env_overrides:
            env_overrides[key] = config[key]

    if "traffic_seed_base" in config and "traffic_seed" not in env_overrides:
        env_overrides["traffic_seed"] = config["traffic_seed_base"]

    if is_dataclass(EnvConfig):
        allowed = {f.name for f in fields(EnvConfig)}
        env_overrides = {
            k: _coerce_env_value(k, v)
            for k, v in env_overrides.items()
            if k in allowed
        }
    else:
        env_overrides = {k: _coerce_env_value(k, v) for k, v in env_overrides.items()}

    return env_overrides


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


def _resolve_impact_predictor_model_dir() -> Path | None:
    candidate = Path("models/impact_predictor")
    return candidate if (candidate / "meta.json").exists() else None


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
    "power_total_watts",
    "power_fixed_watts",
    "power_variable_watts",
    "power_device_watts",
    "power_link_watts",
    "active_devices",
    "inactive_devices",
    "active_links",
    "inactive_links",
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
    "qos_violation",
    "qos_excess",
    "toggle_applied",
    "toggle_reverted",
    "toggle_blocked_any",
    "toggle_blocked_cooldown",
    "toggle_blocked_high_util",
    "toggle_blocked_global_cooldown",
    "blocked_by_util_count",
    "blocked_by_cooldown_count",
    "allowed_toggle_count",
    "toggles_attempted_count",
    "toggles_applied_count",
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


def _resolve_reference_run_dir(model_path: Path | None, runs_dir: Path) -> Path | None:
    if model_path is not None and model_path.exists():
        return model_path.parent if model_path.is_file() else model_path
    try:
        return find_latest_model(runs_dir).parent
    except FileNotFoundError:
        return None


def build_env_config(scenario: str, max_steps: int | None, base_config: EnvConfig | None = None) -> EnvConfig:
    cfg = replace(base_config) if base_config is not None else EnvConfig()
    if hasattr(cfg, "traffic_seed"):
        cfg.traffic_seed = None

    scenario_key = scenario.strip().lower()
    if scenario_key in ("normal", "diurnal", "normal/diurnal", "normal diurnal"):
        cfg.traffic_model = "stochastic"
        # Keep "normal" aligned with the model's saved training environment when available.
        cfg.traffic_scenario = None if base_config is not None else "normal"
    elif scenario_key == "burst":
        cfg.traffic_model = "stochastic"
        cfg.traffic_scenario = "burst"
    elif scenario_key == "hotspot":
        cfg.traffic_model = "stochastic"
        cfg.traffic_scenario = "hotspot"
    elif scenario_key in ("anomaly", "failure"):
        cfg.traffic_model = "stochastic"
        cfg.traffic_scenario = "anomaly"
    elif scenario_key == "custom":
        pass
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
    controller_policy = canonical_controller_policy_name(policy)
    controller_class = controller_policy_class(policy)

    if policy in ("noop", "all_on"):
        return (lambda _obs, _info, _env: 0), {
            "policy_type": "all_on",
            "controller_policy": controller_policy,
            "controller_policy_class": controller_class,
        }

    if policy in ("baseline", "heuristic"):
        return action_sleep_if_idle, {
            "policy_type": "heuristic",
            "controller_policy": controller_policy,
            "controller_policy_class": controller_class,
        }

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

    def _action_mask(env: GreenNetEnv) -> np.ndarray | None:
        if not hasattr(env, "get_action_mask"):
            return None
        try:
            return np.asarray(env.get_action_mask(), dtype=bool)
        except Exception:
            return None

    def _action_allowed(env: GreenNetEnv, action_int: int, mask: np.ndarray | None = None) -> bool:
        if int(action_int) <= 0:
            return True
        if mask is None:
            mask = _action_mask(env)
        if mask is None:
            return True
        return 0 <= int(action_int) < len(mask) and bool(mask[int(action_int)])

    def _action_turns_on(env: GreenNetEnv, action_int: int) -> bool:
        sim = env.simulator
        if sim is None or int(action_int) <= 0:
            return False
        idx = int(action_int) - 1
        if idx < 0 or idx >= len(env.edge_list):
            return False
        u, v = env.edge_list[idx]
        key = env._edge_key(int(u), int(v))  # type: ignore[attr-defined]
        return not bool(sim.active.get(key, True))

    def _action_turns_off(env: GreenNetEnv, action_int: int) -> bool:
        sim = env.simulator
        if sim is None or int(action_int) <= 0:
            return False
        idx = int(action_int) - 1
        if idx < 0 or idx >= len(env.edge_list):
            return False
        u, v = env.edge_list[idx]
        key = env._edge_key(int(u), int(v))  # type: ignore[attr-defined]
        return bool(sim.active.get(key, True))

    def _edge_pressure(env: GreenNetEnv, action_int: int) -> float:
        sim = env.simulator
        if sim is None or int(action_int) <= 0:
            return 0.0
        idx = int(action_int) - 1
        if idx < 0 or idx >= len(env.edge_list):
            return 0.0
        u, v = env.edge_list[idx]
        key = env._edge_key(int(u), int(v))  # type: ignore[attr-defined]
        pressure = float(sim.utilization.get(key, 0.0)) * 2.0
        seen_edges: set[Tuple[int, int]] = {key}
        for node in (int(u), int(v)):
            for nbr in sim.graph.neighbors(node):
                edge_key = env._edge_key(int(node), int(nbr))  # type: ignore[attr-defined]
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                pressure += float(sim.utilization.get(edge_key, 0.0))
        return float(pressure)

    def _predict_off_impact(
        obs: Dict[str, Any],
        env: GreenNetEnv,
        action_int: int,
        *,
        precomputed_state: Tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Dict[str, float] | None:
        predict_fn = getattr(env, "_predict_cost", None)
        if not callable(predict_fn):
            return None
        try:
            pred = predict_fn(obs, int(action_int), precomputed_state=precomputed_state)
        except Exception:
            return None
        return pred if isinstance(pred, dict) else None

    def _set_controller_cost_estimator_override(env: GreenNetEnv, action_int: int | None) -> None:
        setter = getattr(env, "set_controller_cost_estimator_override", None)
        if callable(setter):
            try:
                setter(action_int)
            except Exception:
                pass

    def _edge_cooldown_ready(env: GreenNetEnv, action_int: int) -> bool:
        sim = env.simulator
        if sim is None or int(action_int) <= 0:
            return False
        idx = int(action_int) - 1
        if idx < 0 or idx >= len(env.edge_list):
            return False
        u, v = env.edge_list[idx]
        g = sim.graph
        if g.has_edge(u, v):
            graph_edge = (u, v)
        elif g.has_edge(v, u):
            graph_edge = (v, u)
        else:
            return False
        cooldown_steps = int(getattr(env.config, "toggle_cooldown_steps", 0) or 0)
        last_toggled = int(g.edges[graph_edge].get("last_toggled", -cooldown_steps))
        return (int(getattr(env, "_step_count", 0)) - last_toggled) >= max(0, cooldown_steps)

    def _scaled_cap(env: GreenNetEnv, base_cap: int) -> int:
        scaled_cap_fn = getattr(env, "_scaled_toggle_cap", None)
        if callable(scaled_cap_fn):
            try:
                return int(scaled_cap_fn(int(base_cap)))
            except Exception:
                return int(base_cap)
        return int(base_cap)

    def _controller_can_override_cost_estimator(
        env: GreenNetEnv,
        pred: Dict[str, float] | None,
        *,
        calm_factor: float,
        guard_p_qos_max: float,
        guard_ddrop_max: float,
        guard_tau: float,
    ) -> bool:
        if pred is None or calm_factor < 0.90:
            return False
        p_upper = float(pred.get("p_qos_mean", 0.0) + pred.get("p_qos_std", 0.0))
        ddrop_upper = float(pred.get("ddrop_mean", 0.0) + pred.get("ddrop_std", 0.0))
        risk_score = float(pred.get("risk_score", 0.0))
        ddrop_soft_limit = max(float(guard_ddrop_max) * 1.35, float(guard_ddrop_max) + 0.015)
        return bool(
            p_upper <= float(guard_p_qos_max)
            and risk_score <= float(guard_tau)
            and max(0.0, ddrop_upper) <= float(ddrop_soft_limit)
            and float(getattr(env, "_last_max_util", 0.0)) <= 0.10
        )

    def _recovery_action(env: GreenNetEnv, *, mask: np.ndarray | None = None) -> int:
        sim = env.simulator
        if sim is None or not env.edge_list:
            return 0

        candidates: List[Tuple[float, int]] = []
        for idx, (u, v) in enumerate(env.edge_list):
            key = env._edge_key(int(u), int(v))  # type: ignore[attr-defined]
            if bool(sim.active.get(key, True)):
                continue

            action_int = idx + 1
            if not _action_allowed(env, action_int, mask):
                continue

            candidates.append((_edge_pressure(env, action_int), action_int))

        if not candidates:
            return 0
        candidates.sort(key=lambda item: item[0], reverse=True)
        return int(candidates[0][1])

    def _calm_off_action(
        obs: Dict[str, Any],
        info: Dict[str, Any],
        env: GreenNetEnv,
        *,
        mask: np.ndarray | None = None,
    ) -> Tuple[int, bool]:
        sim = env.simulator
        if sim is None or not env.edge_list:
            return 0, False
        is_decision_step = getattr(env, "_is_decision_step", None)
        if callable(is_decision_step) and not bool(is_decision_step()):
            return 0, False

        qos_violation = bool(info.get("qos_violation", getattr(env, "_last_qos_viol_step", False)))
        if qos_violation:
            return 0, False

        guard_fn = getattr(env, "_cost_estimator_guard_params", None)
        guard_p_qos_max = float(info.get("cost_estimator_p_qos_max", 0.0) or 0.0)
        guard_ddrop_max = float(info.get("cost_estimator_ddrop_max", 0.0) or 0.0)
        guard_tau = float(info.get("cost_estimator_tau", 0.0) or 0.0)
        calm_factor = float(info.get("cost_estimator_calm_factor", 0.0) or 0.0)
        if callable(guard_fn):
            try:
                guard_p_qos_max, guard_ddrop_max, guard_tau, calm_factor = [
                    float(v) for v in guard_fn()
                ]
            except Exception:
                pass
        if calm_factor < 0.35:
            return 0, False

        metrics = info.get("metrics")
        qos_target = float(getattr(env.config, "qos_target_norm_drop", 0.0))
        qos_margin_off = float(getattr(env.config, "qos_guard_margin_off", 0.0))
        norm_drop_total = float(info.get("norm_drop", getattr(env, "_last_norm_drop_total", 0.0)))
        norm_drop_step = float(info.get("norm_drop_step", getattr(env, "_last_norm_drop_step", 0.0)))
        avg_delay_ms = float(getattr(metrics, "avg_delay_ms", 0.0)) if metrics is not None else 0.0
        avg_path_latency_ms = float(getattr(metrics, "avg_path_latency_ms", 0.0)) if metrics is not None else 0.0
        if norm_drop_total > max(0.0, qos_target - max(qos_margin_off * 1.5, 0.01)):
            return 0, False
        if norm_drop_step > max(0.03, qos_target * 0.75):
            return 0, False
        if avg_path_latency_ms > 0.0 and avg_delay_ms > max(avg_path_latency_ms * 4.0, avg_path_latency_ms + 15.0):
            return 0, False

        demand_now = float(np.asarray(obs.get("demand_now", [0.0]), dtype=np.float32).reshape(-1)[0])
        demand_forecast = float(np.asarray(obs.get("demand_forecast", [0.0]), dtype=np.float32).reshape(-1)[0])
        if max(demand_now, demand_forecast) > 0.60:
            return 0, False

        max_util = float(np.asarray(obs.get("max_util", [0.0]), dtype=np.float32).reshape(-1)[0])
        util_gate = float(
            getattr(
                env.config,
                "max_util_off_allow_threshold",
                getattr(env.config, "util_block_threshold", 0.80),
            )
        )
        if max_util >= max(0.05, util_gate * 0.90):
            return 0, False

        calm_required = int(getattr(env.config, "off_calm_steps_required", 0))
        calm_streak = int(getattr(env, "_calm_streak", 0))
        if calm_required > 0 and calm_streak < calm_required:
            return 0, False

        if int(getattr(env, "_global_toggle_cooldown_remaining", 0)) > 0:
            return 0, False

        total_cap = _scaled_cap(env, int(getattr(env.config, "max_total_toggles_per_episode", 0)))
        total_used = int(getattr(env, "_total_toggles_used", 0))
        if total_cap > 0 and total_used >= total_cap:
            return 0, False
        off_cap = _scaled_cap(env, int(getattr(env.config, "max_off_toggles_per_episode", 0)))
        off_used = int(getattr(env, "_off_toggles_used", 0))
        if off_cap > 0 and off_used >= off_cap:
            return 0, False

        if mask is None:
            mask = _action_mask(env)
        if mask is None:
            mask = np.zeros(len(env.edge_list) + 1, dtype=bool)

        precomputed_state: Tuple[np.ndarray, np.ndarray] | None = None
        graph_state_fn = getattr(env, "get_cost_estimator_graph_state", None)
        if callable(graph_state_fn) and getattr(env, "_impact_predictor", None) is not None:
            try:
                _, xg_impact, _, xe_impact = graph_state_fn(obs)
                precomputed_state = (xg_impact, xe_impact)
            except Exception:
                precomputed_state = None

        candidates: List[Tuple[float, int, bool]] = []
        for action_int in range(1, len(env.edge_list) + 1):
            if not _action_turns_off(env, action_int):
                continue
            if not _edge_cooldown_ready(env, action_int):
                continue

            score = _edge_pressure(env, action_int)
            pred = _predict_off_impact(obs, env, action_int, precomputed_state=precomputed_state)
            allowed_by_mask = _action_allowed(env, action_int, mask)
            override_needed = False
            if not allowed_by_mask:
                if not _controller_can_override_cost_estimator(
                    env,
                    pred,
                    calm_factor=calm_factor,
                    guard_p_qos_max=guard_p_qos_max,
                    guard_ddrop_max=guard_ddrop_max,
                    guard_tau=guard_tau,
                ):
                    continue
                override_needed = True
            if pred is not None:
                p_upper = float(pred.get("p_qos_mean", 0.0) + pred.get("p_qos_std", 0.0))
                ddrop_upper = float(pred.get("ddrop_mean", 0.0) + pred.get("ddrop_std", 0.0))
                risk_score = float(pred.get("risk_score", 0.0))
                score += 1.5 * (p_upper / max(guard_p_qos_max, 1e-6))
                score += 1.0 * (max(0.0, ddrop_upper) / max(guard_ddrop_max, 1e-6))
                score += 0.75 * (risk_score / max(guard_tau, 1e-6))
            if override_needed:
                score += 0.05
            candidates.append((float(score), int(action_int), bool(override_needed)))

        if not candidates:
            return 0, False
        candidates.sort(key=lambda item: (item[0], item[1]))
        return int(candidates[0][1]), bool(candidates[0][2])

    def _action_ppo_safe(obs: Dict[str, Any], info: Dict[str, Any], env: GreenNetEnv) -> int:
        _set_controller_cost_estimator_override(env, None)
        proposed = _action_ppo(obs, info, env)
        mask = _action_mask(env)
        metrics = info.get("metrics")
        qos_target = float(getattr(env.config, "qos_target_norm_drop", 0.0))
        qos_margin_off = float(getattr(env.config, "qos_guard_margin_off", 0.0))
        norm_drop_total = float(info.get("norm_drop", getattr(env, "_last_norm_drop_total", 0.0)))
        norm_drop_step = float(info.get("norm_drop_step", getattr(env, "_last_norm_drop_step", 0.0)))
        qos_violation = bool(info.get("qos_violation", getattr(env, "_last_qos_viol_step", False)))
        avg_delay_ms = float(getattr(metrics, "avg_delay_ms", 0.0)) if metrics is not None else 0.0
        avg_path_latency_ms = float(getattr(metrics, "avg_path_latency_ms", 0.0)) if metrics is not None else 0.0

        recovery_mode = bool(
            qos_violation
            or norm_drop_total > max(0.0, qos_target - qos_margin_off)
            or norm_drop_step > max(0.05, qos_target * 1.5)
            or (
                avg_path_latency_ms > 0.0
                and avg_delay_ms > max(avg_path_latency_ms * 6.0, avg_path_latency_ms + 20.0)
                and norm_drop_step > 0.0
            )
        )

        if recovery_mode:
            recovery = _recovery_action(env, mask=mask)
            if recovery > 0:
                return int(recovery)
            if proposed > 0 and _action_turns_on(env, proposed) and _action_allowed(env, proposed, mask):
                return int(proposed)
            return 0

        proposed_allowed = _action_allowed(env, proposed, mask)
        if proposed > 0 and _action_turns_off(env, proposed) and proposed_allowed:
            return int(proposed)

        calm_off, calm_override = _calm_off_action(obs, info, env, mask=mask)
        if calm_off > 0 and (proposed <= 0 or _action_turns_on(env, proposed) or not proposed_allowed):
            if calm_override:
                _set_controller_cost_estimator_override(env, calm_off)
            return int(calm_off)

        if proposed > 0 and not proposed_allowed:
            fallback = action_sleep_if_idle(obs, info, env)
            if fallback > 0 and _action_allowed(env, fallback, mask):
                return int(fallback)
            return 0

        return int(proposed)

    return _action_ppo_safe, {
        "policy_type": "ppo",
        "controller_policy": controller_policy,
        "controller_policy_class": controller_class,
        "model_path": str(resolved_model_path),
        "model_type": model_type,
        "control_mode": "ppo_safe_hybrid_calm_off",
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
        "power_total_watts": float(getattr(metrics, "power_total_watts", 0.0)),
        "power_fixed_watts": float(getattr(metrics, "power_fixed_watts", 0.0)),
        "power_variable_watts": float(getattr(metrics, "power_variable_watts", 0.0)),
        "power_device_watts": float(getattr(metrics, "power_device_watts", 0.0)),
        "power_link_watts": float(getattr(metrics, "power_link_watts", 0.0)),
        "active_devices": int(getattr(metrics, "active_devices", 0)),
        "inactive_devices": int(getattr(metrics, "inactive_devices", 0)),
        "active_links": int(getattr(metrics, "active_links", 0)),
        "inactive_links": int(getattr(metrics, "inactive_links", 0)),
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
        "qos_violation": bool(info.get("qos_violation", False)),
        "qos_excess": float(info.get("qos_excess", 0.0)),
        "toggle_applied": bool(info.get("toggle_applied", False)),
        "toggle_reverted": bool(info.get("toggle_reverted", False)),
        "toggle_blocked_any": bool(info.get("toggle_blocked_any", False)),
        "toggle_blocked_cooldown": bool(info.get("toggle_blocked_cooldown", False)),
        "toggle_blocked_high_util": bool(info.get("toggle_blocked_high_util", False)),
        "toggle_blocked_global_cooldown": bool(info.get("toggle_blocked_global_cooldown", False)),
        "blocked_by_util_count": int(info.get("blocked_by_util_count", 0)),
        "blocked_by_cooldown_count": int(info.get("blocked_by_cooldown_count", 0)),
        "allowed_toggle_count": int(info.get("allowed_toggle_count", 0)),
        "toggles_attempted_count": int(info.get("toggles_attempted_count", 0)),
        "toggles_applied_count": int(info.get("toggles_applied_count", 0)),
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
    if any("toggles_total" in r for r in episode_rows):
        totals["toggles_total"] = [float(r.get("toggles_total", 0.0)) for r in episode_rows]
        totals["toggles_applied_total"] = [float(r.get("toggles_applied_total", 0.0)) for r in episode_rows]
        totals["toggles_reverted_total"] = [float(r.get("toggles_reverted_total", 0.0)) for r in episode_rows]
    if any("toggles_attempted_count" in r for r in episode_rows):
        totals["toggles_attempted_count"] = [float(r.get("toggles_attempted_count", 0.0)) for r in episode_rows]
        totals["allowed_toggle_count"] = [float(r.get("allowed_toggle_count", 0.0)) for r in episode_rows]
        totals["blocked_by_util_count"] = [float(r.get("blocked_by_util_count", 0.0)) for r in episode_rows]
        totals["blocked_by_cooldown_count"] = [float(r.get("blocked_by_cooldown_count", 0.0)) for r in episode_rows]
        totals["toggles_applied_count"] = [float(r.get("toggles_applied_count", 0.0)) for r in episode_rows]
    if any("power_total_watts_mean" in r for r in episode_rows):
        totals["power_total_watts_mean"] = [float(r.get("power_total_watts_mean", 0.0)) for r in episode_rows]
        totals["power_fixed_watts_mean"] = [float(r.get("power_fixed_watts_mean", 0.0)) for r in episode_rows]
        totals["power_variable_watts_mean"] = [float(r.get("power_variable_watts_mean", 0.0)) for r in episode_rows]
        totals["power_device_watts_mean"] = [float(r.get("power_device_watts_mean", 0.0)) for r in episode_rows]
        totals["power_link_watts_mean"] = [float(r.get("power_link_watts_mean", 0.0)) for r in episode_rows]
        totals["active_devices_mean"] = [float(r.get("active_devices_mean", 0.0)) for r in episode_rows]
        totals["inactive_devices_mean"] = [float(r.get("inactive_devices_mean", 0.0)) for r in episode_rows]
        totals["active_links_mean"] = [float(r.get("active_links_mean", 0.0)) for r in episode_rows]
        totals["inactive_links_mean"] = [float(r.get("inactive_links_mean", 0.0)) for r in episode_rows]

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
        "avg_path_latency_ms_mean": _mean([float(r.get("avg_path_latency_ms_mean", 0.0)) for r in episode_rows]),
    }
    if "toggles_total" in totals:
        overall.update(
            {
                "toggles_total_mean": _mean([float(v) for v in totals["toggles_total"]]),
                "toggles_total_std": _std([float(v) for v in totals["toggles_total"]]),
                "toggles_applied_mean": _mean([float(v) for v in totals["toggles_applied_total"]]),
                "toggles_reverted_mean": _mean([float(v) for v in totals["toggles_reverted_total"]]),
            }
        )
    if "toggles_attempted_count" in totals:
        overall.update(
            {
                "toggles_attempted_count_mean": _mean([float(v) for v in totals["toggles_attempted_count"]]),
                "toggles_attempted_count_std": _std([float(v) for v in totals["toggles_attempted_count"]]),
                "allowed_toggle_count_mean": _mean([float(v) for v in totals["allowed_toggle_count"]]),
                "allowed_toggle_count_std": _std([float(v) for v in totals["allowed_toggle_count"]]),
                "blocked_by_util_count_mean": _mean([float(v) for v in totals["blocked_by_util_count"]]),
                "blocked_by_util_count_std": _std([float(v) for v in totals["blocked_by_util_count"]]),
                "blocked_by_cooldown_count_mean": _mean([float(v) for v in totals["blocked_by_cooldown_count"]]),
                "blocked_by_cooldown_count_std": _std([float(v) for v in totals["blocked_by_cooldown_count"]]),
                "toggles_applied_count_mean": _mean([float(v) for v in totals["toggles_applied_count"]]),
                "toggles_applied_count_std": _std([float(v) for v in totals["toggles_applied_count"]]),
            }
        )
    if "power_total_watts_mean" in totals:
        overall.update(
            {
                "power_total_watts_mean": _mean([float(v) for v in totals["power_total_watts_mean"]]),
                "power_total_watts_std": _std([float(v) for v in totals["power_total_watts_mean"]]),
                "power_fixed_watts_mean": _mean([float(v) for v in totals["power_fixed_watts_mean"]]),
                "power_variable_watts_mean": _mean([float(v) for v in totals["power_variable_watts_mean"]]),
                "power_device_watts_mean": _mean([float(v) for v in totals["power_device_watts_mean"]]),
                "power_link_watts_mean": _mean([float(v) for v in totals["power_link_watts_mean"]]),
                "active_devices_mean": _mean([float(v) for v in totals["active_devices_mean"]]),
                "inactive_devices_mean": _mean([float(v) for v in totals["inactive_devices_mean"]]),
                "active_links_mean": _mean([float(v) for v in totals["active_links_mean"]]),
                "inactive_links_mean": _mean([float(v) for v in totals["inactive_links_mean"]]),
            }
        )

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
    traffic_seed_base: int,
    save_flows: bool,
    writer: csv.DictWriter,
) -> Dict[str, Any]:
    # Vary traffic per episode (if supported) while keeping topology fixed
    if hasattr(env.config, "traffic_seed"):
        env.config.traffic_seed = int(traffic_seed_base) + int(episode_idx)
    obs, info = env.reset(seed=episode_seed)

    total_reward = 0.0
    total_delivered = 0.0
    total_dropped = 0.0
    total_energy = 0.0
    total_carbon = 0.0
    avg_util_sum = 0.0
    active_ratio_sum = 0.0
    avg_delay_sum = 0.0
    avg_path_latency_sum = 0.0
    power_total_watts_sum = 0.0
    power_fixed_watts_sum = 0.0
    power_variable_watts_sum = 0.0
    power_device_watts_sum = 0.0
    power_link_watts_sum = 0.0
    active_devices_sum = 0.0
    inactive_devices_sum = 0.0
    active_links_sum = 0.0
    inactive_links_sum = 0.0
    steps = 0
    toggles_applied = 0
    toggles_reverted = 0
    toggles_attempted = 0
    toggles_allowed = 0
    blocked_by_util = 0
    blocked_by_cooldown = 0
    toggles_applied_count = 0

    for step_idx in range(1, int(env.config.max_steps) + 1):
        action = action_fn(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        delta_carbon = float(info.get("delta_carbon_g", getattr(info.get("metrics"), "carbon_g", 0.0)))

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
        avg_path_latency_sum += float(getattr(info.get("metrics"), "avg_path_latency_ms", 0.0))
        power_total_watts_sum += float(getattr(info.get("metrics"), "power_total_watts", 0.0))
        power_fixed_watts_sum += float(getattr(info.get("metrics"), "power_fixed_watts", 0.0))
        power_variable_watts_sum += float(getattr(info.get("metrics"), "power_variable_watts", 0.0))
        power_device_watts_sum += float(getattr(info.get("metrics"), "power_device_watts", 0.0))
        power_link_watts_sum += float(getattr(info.get("metrics"), "power_link_watts", 0.0))
        active_devices_sum += float(getattr(info.get("metrics"), "active_devices", 0.0))
        inactive_devices_sum += float(getattr(info.get("metrics"), "inactive_devices", 0.0))
        active_links_sum += float(getattr(info.get("metrics"), "active_links", 0.0))
        inactive_links_sum += float(getattr(info.get("metrics"), "inactive_links", 0.0))
        toggles_applied += int(bool(info.get("toggle_applied")))
        toggles_reverted += int(bool(info.get("toggle_reverted")))
        toggles_attempted += int(info.get("toggles_attempted_count", 0))
        toggles_allowed += int(info.get("allowed_toggle_count", 0))
        blocked_by_util += int(info.get("blocked_by_util_count", 0))
        blocked_by_cooldown += int(info.get("blocked_by_cooldown_count", 0))
        toggles_applied_count += int(info.get("toggles_applied_count", 0))

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
        "avg_path_latency_ms_mean": float(avg_path_latency_sum / denom),
        "power_total_watts_mean": float(power_total_watts_sum / denom),
        "power_fixed_watts_mean": float(power_fixed_watts_sum / denom),
        "power_variable_watts_mean": float(power_variable_watts_sum / denom),
        "power_device_watts_mean": float(power_device_watts_sum / denom),
        "power_link_watts_mean": float(power_link_watts_sum / denom),
        "active_devices_mean": float(active_devices_sum / denom),
        "inactive_devices_mean": float(inactive_devices_sum / denom),
        "active_links_mean": float(active_links_sum / denom),
        "inactive_links_mean": float(inactive_links_sum / denom),
        "toggles_applied_total": int(toggles_applied),
        "toggles_reverted_total": int(toggles_reverted),
        "toggles_total": int(toggles_applied + toggles_reverted),
        "blocked_by_util_count": int(blocked_by_util),
        "blocked_by_cooldown_count": int(blocked_by_cooldown),
        "allowed_toggle_count": int(toggles_allowed),
        "toggles_attempted_count": int(toggles_attempted),
        "toggles_applied_count": int(toggles_applied_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GreenNet experiments and log per-step results.")
    parser.add_argument("--config", type=Path, help="Path to JSON config for eval/experiment.")
    parser.add_argument(
        "--policy",
        choices=["all_on", "heuristic", "baseline", "noop", "ppo"],
        default=None,
    )
    parser.add_argument(
        "--scenario",
        choices=["normal", "diurnal", "burst", "hotspot", "anomaly", "failure", "custom"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None, help="Base eval seed (overrides config).")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=None)
    parser.add_argument("--save-flows", action="store_true", default=None)
    parser.add_argument("--topology-seed", type=int, default=None)
    parser.add_argument("--topology-name", type=str, default=None)
    parser.add_argument("--topology-path", type=Path, default=None)
    parser.add_argument("--traffic-seed", type=int, default=None)
    parser.add_argument("--traffic-name", type=str, default=None)
    parser.add_argument("--traffic-path", type=Path, default=None)
    parser.add_argument("--routing-baseline", type=str, default=None)
    parser.add_argument("--routing-link-cost-model", type=str, default=None)
    args = parser.parse_args()

    config = _load_config_file(args.config)
    env_overrides = _extract_env_overrides(config)

    policy = args.policy or config.get("policy")
    if not policy:
        parser.error("Missing --policy (or set 'policy' in --config).")

    scenario = args.scenario or config.get("scenario")
    if not scenario:
        parser.error("Missing --scenario (or set 'scenario' in --config).")

    eval_seed_raw = args.seed if args.seed is not None else config.get("eval_seed", config.get("seed"))
    if eval_seed_raw is None:
        parser.error("Missing --seed (or set 'eval_seed' in --config).")
    eval_seed = int(eval_seed_raw)

    episodes = args.episodes if args.episodes is not None else int(config.get("episodes", 10))

    max_steps = args.max_steps if args.max_steps is not None else config.get("max_steps")
    if args.steps is not None:
        max_steps = args.steps
    elif "steps" in config and max_steps is None:
        max_steps = config.get("steps")
    if max_steps is not None:
        max_steps = int(max_steps)

    runs_dir = args.runs_dir or Path(config.get("runs_dir", "runs"))
    out_dir_root = args.out_dir or Path(config.get("out_dir", "results"))
    tag = args.tag or config.get("tag")
    tag_str = str(tag).strip() if tag is not None else ""
    if not tag_str:
        tag_str = None

    if args.deterministic is None:
        deterministic = bool(config.get("deterministic", True))
    else:
        deterministic = bool(args.deterministic)

    if args.save_flows is None:
        save_flows = bool(config.get("save_flows", False))
    else:
        save_flows = bool(args.save_flows)

    model_path = args.model
    if model_path is None and config.get("model"):
        model_path = Path(config["model"])

    seed_everything(int(eval_seed))

    reference_run_dir = _resolve_reference_run_dir(model_path, runs_dir)
    reference_env_config = (
        load_env_config_from_run(reference_run_dir, verbose=False) if reference_run_dir is not None else None
    )

    env_config = build_env_config(scenario, max_steps, reference_env_config)
    for key, value in env_overrides.items():
        setattr(env_config, key, value)
    if args.topology_name is not None:
        env_config.topology_name = str(args.topology_name).strip() or None
    if args.topology_path is not None:
        env_config.topology_path = str(Path(args.topology_path).expanduser().resolve())
    if args.traffic_name is not None:
        env_config.traffic_name = str(args.traffic_name).strip() or None
    if args.traffic_path is not None:
        env_config.traffic_path = str(Path(args.traffic_path).expanduser().resolve())

    impact_model_dir = _resolve_impact_predictor_model_dir()
    if policy == "ppo" and impact_model_dir is not None:
        env_config.cost_estimator_enabled = True
        env_config.cost_estimator_model_dir = str(impact_model_dir)

    action_fn, policy_meta = load_policy(
        policy,
        runs_dir=runs_dir,
        model_path=model_path,
        deterministic=bool(deterministic),
    )

    # Choose topology seed: allow override; otherwise keep PPO on its training topology to avoid action-space mismatches.
    chosen_topology_seed: int | None = args.topology_seed
    if chosen_topology_seed is None:
        chosen_topology_seed = config.get("topology_seed")
    if chosen_topology_seed is None:
        chosen_topology_seed = env_overrides.get("topology_seed")
    if chosen_topology_seed is None and policy == "ppo":
        mp = policy_meta.get("model_path") if isinstance(policy_meta, dict) else None
        if mp:
            inferred = infer_topology_seed_from_model(Path(mp))
            if inferred is not None:
                chosen_topology_seed = inferred
    if chosen_topology_seed is None:
        chosen_topology_seed = int(eval_seed)

    # Choose base traffic seed: allow override; otherwise seed+10000.
    chosen_traffic_seed_base: int | None = args.traffic_seed
    if chosen_traffic_seed_base is None:
        chosen_traffic_seed_base = config.get("traffic_seed")
    if chosen_traffic_seed_base is None:
        chosen_traffic_seed_base = env_overrides.get("traffic_seed")
    if chosen_traffic_seed_base is None:
        chosen_traffic_seed_base = int(eval_seed) + 10_000

    if hasattr(env_config, "topology_seed"):
        env_config.topology_seed = int(chosen_topology_seed)
    if hasattr(env_config, "traffic_seed"):
        env_config.traffic_seed = int(chosen_traffic_seed_base)
    if args.routing_baseline is not None:
        env_config.routing_baseline = str(args.routing_baseline)
    elif config.get("routing_baseline") is not None:
        env_config.routing_baseline = str(config.get("routing_baseline"))
    if args.routing_link_cost_model is not None:
        env_config.routing_link_cost_model = str(args.routing_link_cost_model)
    elif config.get("routing_link_cost_model") is not None:
        env_config.routing_link_cost_model = str(config.get("routing_link_cost_model"))

    env_config.routing_baseline = canonicalize_routing_baseline_name(
        getattr(env_config, "routing_baseline", "min_hop_single_path")
    )
    env_config.routing_link_cost_model = canonicalize_routing_link_cost_model(
        getattr(env_config, "routing_link_cost_model", "unit")
    )

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")
    folder = f"{run_id}__policy-{policy}__scenario-{scenario}__seed-{eval_seed}"
    if tag_str:
        folder = f"{folder}__tag-{tag_str}"
    out_dir = out_dir_root / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    save_env_config(out_dir, env_config)

    env = GreenNetEnv(config=env_config)

    per_step_path = out_dir / "per_step.csv"
    episode_summaries: List[Dict[str, Any]] = []

    with per_step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for ep in range(int(episodes)):
            episode_seed = int(eval_seed) + ep
            episode_summaries.append(
                run_episode(
                    env,
                    action_fn,
                    run_id=run_id,
                    policy=policy,
                    scenario=scenario,
                    seed=int(eval_seed),
                    episode_seed=episode_seed,
                    episode_idx=ep,
                    traffic_seed_base=int(chosen_traffic_seed_base),
                    save_flows=bool(save_flows),
                    writer=writer,
                )
            )

    routing_meta = env.get_routing_metadata() if hasattr(env, "get_routing_metadata") else {}
    env.close()

    summary = summarize_episodes(episode_summaries)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    count_keys = [
        "blocked_by_util_count",
        "blocked_by_cooldown_count",
        "allowed_toggle_count",
        "toggles_attempted_count",
        "toggles_applied_count",
    ]
    count_totals = {
        key: int(sum(int(ep.get(key, 0)) for ep in episode_summaries)) for key in count_keys
    }

    model_path_meta = policy_meta.get("model_path") if isinstance(policy_meta, dict) else None
    run_meta = {
        "run_id": run_id,
        "policy": policy,
        "scenario": scenario,
        "seed": int(eval_seed),
        "eval_seed": int(eval_seed),
        "topology_seed": int(chosen_topology_seed),
        "traffic_seed": int(chosen_traffic_seed_base),
        "traffic_seed_base": int(chosen_traffic_seed_base),
        "episodes": int(episodes),
        "max_steps": int(env_config.max_steps),
        "deterministic": bool(deterministic),
        "save_flows": bool(save_flows),
        "model_path": model_path_meta,
        "runs_dir": str(runs_dir),
        "timestamp_utc": now.isoformat(),
        "created_at_utc": now.isoformat(),
        "command": " ".join(sys.argv),
    }
    run_meta.update(count_totals)
    run_meta.update(routing_meta)
    if getattr(env_config, "topology_seeds", None):
        run_meta["topology_seeds"] = list(env_config.topology_seeds)
    if getattr(env_config, "topology_name", None):
        run_meta["topology_name"] = env_config.topology_name
    if getattr(env_config, "topology_path", None):
        run_meta["topology_path"] = env_config.topology_path
    if getattr(env_config, "traffic_scenario", None):
        run_meta["traffic_scenario"] = env_config.traffic_scenario
        run_meta["traffic_scenario_version"] = int(getattr(env_config, "traffic_scenario_version", 2))
        run_meta["traffic_scenario_intensity"] = float(getattr(env_config, "traffic_scenario_intensity", 1.0))
        run_meta["traffic_scenario_duration"] = float(getattr(env_config, "traffic_scenario_duration", 1.0))
        run_meta["traffic_scenario_frequency"] = float(getattr(env_config, "traffic_scenario_frequency", 1.0))
    if getattr(env_config, "traffic_name", None):
        run_meta["traffic_name"] = env_config.traffic_name
    if getattr(env_config, "traffic_path", None):
        run_meta["traffic_path"] = env_config.traffic_path
    run_meta.update(policy_meta)
    run_meta["tag"] = tag_str
    run_meta_path = out_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    try:
        persist_run_directory(out_dir, source="results")
    except Exception as exc:
        # Keep file artifacts authoritative enough for compatibility; DB backfill can repair later.
        print(f"[run_experiment] warning: failed to persist run to SQLite: {exc}")

    print(f"[run_experiment] results saved to {out_dir}")


if __name__ == "__main__":
    main()
