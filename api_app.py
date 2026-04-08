from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import networkx as nx

from greennet.env import GreenNetEnv
from greennet.persistence import get_run_repository, infer_run_source
from greennet.utils.config import load_env_config_from_run

app = FastAPI(title="GreenNet Metrics API")

# Figure out the repo root reliably.
# If this file is at repo root (GreenNet/api_app.py) -> repo root is this file's parent.
# If later moved into a package (GreenNet/greennet/api_app.py) -> repo root is one level up.
THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_FILE_DIR
if (REPO_ROOT / "runs").exists() is False and (REPO_ROOT / "results").exists() is False:
    # Likely inside a subpackage; fall back one directory.
    REPO_ROOT = THIS_FILE_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"
LOCKED_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "locked"

BaseChoice = Literal["results", "runs", "both"]
GROUP_BY_FIELDS = {"policy", "scenario", "tag", "topology_seed", "deterministic"}
KEY_FILES = ["per_step.csv", "summary.json", "run_meta.json", "env_config.json"]
OFFICIAL_LOCKED_SCENARIOS = ("normal", "burst", "hotspot")
FINAL_EVALUATION_SUMMARY_FILENAME = "final_evaluation_summary.json"
FINAL_EVALUATION_REPORT_FILENAME = "final_evaluation_report.md"
HIGHLIGHT_FIELDS = [
    "reward_total_mean",
    "energy_kwh_total_mean",
    "dropped_total_mean",
    "toggles_total_mean",
]
PACKET_EVENT_LIMIT = 40
INT_COLUMNS = {
    "t",
    "step",
    "episode",
    "seed",
    "episode_seed",
    "action",
    "num_active_edges",
    "near_saturated_edges",
    "flows_count",
    "active_devices",
    "inactive_devices",
    "active_links",
    "inactive_links",
    "blocked_by_util_count",
    "blocked_by_cooldown_count",
    "allowed_toggle_count",
    "toggles_attempted_count",
    "toggles_applied_count",
}
FLOAT_COLUMNS = {
    "reward",
    "avg_utilization",
    "active_ratio",
    "max_util",
    "min_util",
    "p95_util",
    "dropped_prev",
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
    "qos_excess",
}
RUN_PREFIX_RE = re.compile(r"^(?P<stamp>\d{8}_\d{6})")
RESULTS_SAVED_RE = re.compile(r"^\[run_experiment\] results saved to (?P<path>.+)$", re.MULTILINE)
DASHBOARD_RUN_TAG = "dashboard"
LOCKED_SUMMARY_RE = re.compile(
    r"^\[summary:[^\]]+\] trained\(det\) vs noop: better=(?P<better>YES|NO) "
    r"Δreward=(?P<delta_reward>[-+0-9.]+) "
    r"Δenergy=(?P<delta_energy>[-+0-9.]+) "
    r"Δdropped=(?P<delta_dropped>[-+0-9.]+) "
    r"\((?P<reason>.+)\)$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1):\d+$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRunRequest(BaseModel):
    policy: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    seed: int
    steps: int = Field(ge=1, le=5000)


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON object from disk; return None on missing/parse error."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _run_repository():
    try:
        return get_run_repository()
    except Exception:
        return None


def _run_identity_for_dir(run_dir: Path) -> Tuple[str, str]:
    return infer_run_source(run_dir, repo_root=REPO_ROOT), run_dir.name


def _db_run_snapshots(base: BaseChoice) -> Dict[Tuple[str, str], Dict[str, Any]]:
    repo = _run_repository()
    if repo is None:
        return {}

    try:
        snapshots = repo.list_run_snapshots(base=base)
    except Exception:
        return {}

    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for snapshot in snapshots:
        source = str(snapshot.get("source") or "")
        run_name = str(snapshot.get("run_id") or "")
        if source and run_name:
            indexed[(source, run_name)] = snapshot
    return indexed


def _load_db_json_payload(source: str, run_dir: Path, filename: str) -> Optional[Dict[str, Any]]:
    repo = _run_repository()
    if repo is None:
        return None

    try:
        if filename == "run_meta.json":
            return repo.get_run_meta(source, run_dir.name)
        if filename == "env_config.json":
            return repo.get_env_config(source, run_dir.name)
        if filename == "summary.json":
            return repo.get_run_summary(source, run_dir.name)
    except Exception:
        return None
    return None


def _edge_id(source: Any, target: Any) -> str:
    source_id = str(source)
    target_id = str(target)
    return f"{source_id}__{target_id}" if source_id < target_id else f"{target_id}__{source_id}"


def _node_id(node: Any) -> str:
    try:
        return str(int(node))
    except (TypeError, ValueError):
        return str(node)


def _normalize_layout(layout: Dict[Any, Any]) -> Dict[str, Tuple[float, float]]:
    if not layout:
        return {}

    xs = [float(point[0]) for point in layout.values()]
    ys = [float(point[1]) for point in layout.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    normalized: Dict[str, Tuple[float, float]] = {}
    for node, point in layout.items():
        x = 0.1 + 0.8 * ((float(point[0]) - min_x) / span_x)
        y = 0.12 + 0.76 * ((float(point[1]) - min_y) / span_y)
        normalized[_node_id(node)] = (x, y)
    return normalized


def _off_level_rank(value: Any) -> int:
    match = re.match(r"^off(?P<count>\d+)$", str(value or "").strip().lower())
    if not match:
        return -1
    return int(match.group("count"))


def _read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_locked_note(bundle_dir: Path) -> Optional[str]:
    text = _read_text_if_exists(bundle_dir / "notes.md")
    if not text:
        return None

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        lines.append(line)
    return " ".join(lines) if lines else None


def _final_evaluation_candidate_paths() -> List[Path]:
    preferred = [
        REPO_ROOT / "artifacts" / "final_evaluation" / "latest" / FINAL_EVALUATION_SUMMARY_FILENAME,
    ]

    candidates: List[Path] = [path for path in preferred if path.exists()]
    for root in (REPO_ROOT / "artifacts", REPO_ROOT / "experiments"):
        if not root.exists():
            continue
        candidates.extend(root.rglob(FINAL_EVALUATION_SUMMARY_FILENAME))

    deduped: Dict[str, Path] = {}
    for path in candidates:
        deduped[str(path.resolve())] = path.resolve()
    return list(deduped.values())


@lru_cache(maxsize=4)
def _latest_final_evaluation_artifact() -> Optional[Dict[str, Any]]:
    ranked: List[Tuple[float, int, str, Path, Optional[Path], Dict[str, Any]]] = []
    for summary_path in _final_evaluation_candidate_paths():
        payload = load_json(summary_path)
        if not payload or not isinstance(payload.get("summary_rows"), list):
            continue

        report_path = summary_path.with_name(FINAL_EVALUATION_REPORT_FILENAME)
        generated_at = _parse_iso_timestamp(payload.get("generated_at_utc"))
        ranked.append(
            (
                generated_at.timestamp() if generated_at is not None else float(summary_path.stat().st_mtime),
                1 if "artifacts" in summary_path.parts else 0,
                str(summary_path),
                summary_path,
                report_path if report_path.exists() else None,
                payload,
            )
        )

    if ranked:
        _, _, _, summary_path, report_path, payload = max(ranked)
        return {
            "summary_path": summary_path,
            "report_path": report_path,
            "payload": payload,
        }

    return None


@lru_cache(maxsize=16)
def _read_locked_eval_rows(bundle_dir_str: str, filename: str) -> Tuple[Dict[str, Any], ...]:
    bundle_dir = Path(bundle_dir_str)
    csv_path = bundle_dir / filename
    if not csv_path.exists():
        return tuple()

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "scenario": str(row.get("scenario") or "").strip().lower(),
                    "off_level": str(row.get("off_level") or "").strip().lower(),
                    "pass": str(row.get("PASS") or "").strip().upper() == "PASS",
                    "delta_energy_kwh": _to_float(row.get("Δenergy")),
                    "delta_dropped": _to_float(row.get("Δdropped")),
                    "delta_reward": _to_float(row.get("Δreward")),
                    "on_edges_mean": _to_float(row.get("on_edges_mean")),
                    "toggles_applied_mean": _to_float(row.get("toggles_applied_mean")),
                    "blocked_on_actions_mean": _to_float(row.get("blocked_on_actions_mean")),
                    "cap_used": (str(row.get("cap_used") or "").strip() or None),
                    "seeds": (str(row.get("seeds") or "").strip() or None),
                    "episodes": _to_int(row.get("episodes")),
                    "log_file": (str(row.get("log_file") or "").strip() or None),
                }
            )
    return tuple(rows)


def _select_locked_summary_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    ranked = sorted(
        rows,
        key=lambda row: (
            0 if bool(row.get("pass")) else 1,
            -_off_level_rank(row.get("off_level")),
            str(row.get("off_level") or ""),
        ),
    )
    return ranked[0]


@lru_cache(maxsize=16)
def _read_locked_log_stats(log_path_str: str) -> Dict[str, Any]:
    log_path = Path(log_path_str)
    text = _read_text_if_exists(log_path)
    if not text:
        return {}

    stat_patterns = {
        "reward_mean": re.compile(r"^episode_reward:\s+mean=([-+0-9.]+)\s+std="),
        "energy_kwh_mean": re.compile(r"^energy_kwh:\s+mean=([-+0-9.]+)\s+std="),
        "dropped_mean": re.compile(r"^dropped:\s+mean=([-+0-9.]+)\s+std="),
        "delivered_mean": re.compile(r"^delivered:\s+mean=([-+0-9.]+)\s+std="),
        "toggles_applied_mean": re.compile(r"^toggles applied mean=([-+0-9.]+)\s+reverted mean="),
        "blocked_on_actions_mean": re.compile(
            r"^on-edge budget\s+max_on_edges=\d+\s+blocked_on_actions_mean=([-+0-9.]+)"
        ),
        "on_edges_mean": re.compile(r"^edge state\s+on_edges_mean=([-+0-9.]+)\s+off_edges_mean="),
    }

    stats: Dict[str, Any] = {"trained_det": {}, "noop_det": {}}
    section: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("=== Evaluation: trained (det,"):
            section = "trained_det"
            continue
        if line.startswith("=== Evaluation: always_noop (det,"):
            section = "noop_det"
            continue
        if line.startswith("=== Evaluation: "):
            section = None
            continue

        if stats.get("delta_summary") is None:
            summary_match = LOCKED_SUMMARY_RE.match(line)
            if summary_match:
                stats["delta_summary"] = {
                    "better": summary_match.group("better") == "YES",
                    "reason": summary_match.group("reason"),
                    "delta_reward": _to_float(summary_match.group("delta_reward")),
                    "delta_energy_kwh": _to_float(summary_match.group("delta_energy")),
                    "delta_dropped": _to_float(summary_match.group("delta_dropped")),
                }

        if section not in {"trained_det", "noop_det"}:
            continue

        bucket = stats[section]
        for key, pattern in stat_patterns.items():
            match = pattern.match(line)
            if match:
                bucket[key] = float(match.group(1))

    for key in ("trained_det", "noop_det"):
        bucket = stats.get(key)
        if not isinstance(bucket, dict) or not bucket:
            continue

        delivered_mean = _to_float(bucket.get("delivered_mean")) or 0.0
        dropped_mean = _to_float(bucket.get("dropped_mean")) or 0.0
        sent_mean = delivered_mean + dropped_mean
        bucket["sent_mean"] = sent_mean
        bucket["drop_rate"] = (dropped_mean / sent_mean) if sent_mean > 0 else 0.0

    return stats


def _resolve_locked_log_path(bundle_dir: Path, log_file: Optional[str]) -> Optional[Path]:
    if not log_file:
        return None

    candidate = bundle_dir / Path(log_file).name
    if candidate.exists():
        return candidate

    repo_candidate = REPO_ROOT / log_file
    if repo_candidate.exists():
        return repo_candidate

    return None


@lru_cache(maxsize=8)
def _official_locked_result_for_scenario(scenario: str) -> Optional[Dict[str, Any]]:
    scenario_key = scenario.strip().lower()
    if scenario_key not in OFFICIAL_LOCKED_SCENARIOS:
        return None

    scenario_dir = LOCKED_ARTIFACTS_DIR / scenario_key
    if not scenario_dir.exists():
        return None

    bundle_dirs = sorted(path for path in scenario_dir.iterdir() if path.is_dir())
    if not bundle_dirs:
        return None

    bundle_dir = bundle_dirs[-1]
    eval_rows = list(_read_locked_eval_rows(str(bundle_dir), "eval_summary.csv"))
    selected_row = _select_locked_summary_row(eval_rows)

    log_path = _resolve_locked_log_path(bundle_dir, None if selected_row is None else selected_row.get("log_file"))
    log_stats = _read_locked_log_stats(str(log_path)) if log_path is not None else {}

    return {
        "scenario": scenario_key,
        "bundle_id": bundle_dir.name,
        "bundle_path": str(bundle_dir.relative_to(REPO_ROOT)),
        "pass_all": all(bool(row.get("pass")) for row in eval_rows) if eval_rows else None,
        "summary": selected_row,
        "eval_rows": eval_rows,
        "trained_det": log_stats.get("trained_det"),
        "noop_det": log_stats.get("noop_det"),
        "delta_summary": log_stats.get("delta_summary"),
        "notes": _read_locked_note(bundle_dir),
    }


@lru_cache(maxsize=64)
def _read_all_per_step_rows(run_dir_str: str) -> Tuple[Dict[str, Any], ...]:
    run_dir = Path(run_dir_str)
    source, run_name = _run_identity_for_dir(run_dir)
    repo = _run_repository()
    if repo is not None:
        try:
            db_rows = repo.get_step_rows(source, run_name)
        except Exception:
            db_rows = []
        if db_rows:
            return tuple({k: _coerce_per_step_value(k, v) for k, v in row.items()} for row in db_rows)

    csv_path = run_dir / "per_step.csv"
    if not csv_path.exists():
        return tuple()

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _coerce_per_step_value(k, v) for k, v in row.items()})
    return tuple(rows)


def _mean_or_zero(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std_or_zero(values: List[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _row_float(row: Dict[str, Any], key: str) -> float:
    return float(_to_float(row.get(key)) or 0.0)


@lru_cache(maxsize=64)
def _recompute_summary_from_per_step(run_dir_str: str) -> Optional[Dict[str, Any]]:
    raw_rows = list(_read_all_per_step_rows(run_dir_str))
    if not raw_rows:
        return None

    episodes: Dict[int, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        episode = _to_int(row.get("episode"))
        episodes.setdefault(episode if episode is not None else 0, []).append(row)

    episode_summaries: List[Dict[str, Any]] = []
    for episode, rows in sorted(episodes.items(), key=lambda item: item[0]):
        avg_path_latency_values = [
            float(value)
            for value in (_to_float(row.get("avg_path_latency_ms")) for row in rows)
            if value is not None
        ]

        episode_summaries.append(
            {
                "episode": int(episode),
                "steps": int(len(rows)),
                "reward_total": float(sum(_row_float(row, "reward") for row in rows)),
                "delivered_total": float(sum(_row_float(row, "delivered") for row in rows)),
                "dropped_total": float(sum(_row_float(row, "dropped") for row in rows)),
                "energy_kwh_total": float(sum(_row_float(row, "energy_kwh") for row in rows)),
                "carbon_g_total": float(sum(_row_float(row, "carbon_g") for row in rows)),
                "avg_utilization_mean": _mean_or_zero([_row_float(row, "avg_utilization") for row in rows]),
                "active_ratio_mean": _mean_or_zero([_row_float(row, "active_ratio") for row in rows]),
                "avg_delay_ms_mean": _mean_or_zero([_row_float(row, "avg_delay_ms") for row in rows]),
                "avg_path_latency_ms_mean": (
                    _mean_or_zero(avg_path_latency_values) if avg_path_latency_values else None
                ),
                "toggles_applied_total": int(sum(1 for row in rows if _to_bool(row.get("toggle_applied")) is True)),
                "toggles_reverted_total": int(sum(1 for row in rows if _to_bool(row.get("toggle_reverted")) is True)),
            }
        )
        episode_summaries[-1]["toggles_total"] = (
            int(episode_summaries[-1]["toggles_applied_total"]) + int(episode_summaries[-1]["toggles_reverted_total"])
        )

    overall: Dict[str, Any] = {
        "episodes": len(episode_summaries),
        "reward_total_mean": _mean_or_zero([float(item["reward_total"]) for item in episode_summaries]),
        "reward_total_std": _std_or_zero([float(item["reward_total"]) for item in episode_summaries]),
        "delivered_total_mean": _mean_or_zero([float(item["delivered_total"]) for item in episode_summaries]),
        "delivered_total_std": _std_or_zero([float(item["delivered_total"]) for item in episode_summaries]),
        "dropped_total_mean": _mean_or_zero([float(item["dropped_total"]) for item in episode_summaries]),
        "dropped_total_std": _std_or_zero([float(item["dropped_total"]) for item in episode_summaries]),
        "energy_kwh_total_mean": _mean_or_zero([float(item["energy_kwh_total"]) for item in episode_summaries]),
        "energy_kwh_total_std": _std_or_zero([float(item["energy_kwh_total"]) for item in episode_summaries]),
        "carbon_g_total_mean": _mean_or_zero([float(item["carbon_g_total"]) for item in episode_summaries]),
        "carbon_g_total_std": _std_or_zero([float(item["carbon_g_total"]) for item in episode_summaries]),
        "steps_mean": _mean_or_zero([float(item["steps"]) for item in episode_summaries]),
        "steps_std": _std_or_zero([float(item["steps"]) for item in episode_summaries]),
        "avg_utilization_mean": _mean_or_zero([float(item["avg_utilization_mean"]) for item in episode_summaries]),
        "active_ratio_mean": _mean_or_zero([float(item["active_ratio_mean"]) for item in episode_summaries]),
        "avg_delay_ms_mean": _mean_or_zero([float(item["avg_delay_ms_mean"]) for item in episode_summaries]),
        "toggles_total_mean": _mean_or_zero([float(item["toggles_total"]) for item in episode_summaries]),
        "toggles_total_std": _std_or_zero([float(item["toggles_total"]) for item in episode_summaries]),
        "toggles_applied_mean": _mean_or_zero([float(item["toggles_applied_total"]) for item in episode_summaries]),
        "toggles_applied_std": _std_or_zero([float(item["toggles_applied_total"]) for item in episode_summaries]),
        "toggles_reverted_mean": _mean_or_zero([float(item["toggles_reverted_total"]) for item in episode_summaries]),
        "toggles_reverted_std": _std_or_zero([float(item["toggles_reverted_total"]) for item in episode_summaries]),
    }

    avg_path_latency_episode_values = [
        float(item["avg_path_latency_ms_mean"])
        for item in episode_summaries
        if item.get("avg_path_latency_ms_mean") is not None
    ]
    if avg_path_latency_episode_values:
        overall["avg_path_latency_ms_mean"] = _mean_or_zero(avg_path_latency_episode_values)
        overall["avg_path_latency_ms_std"] = _std_or_zero(avg_path_latency_episode_values)

    return {"episodes": episode_summaries, "overall": overall}


def _summary_payload_for_run(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    source, _ = _run_identity_for_dir(run_dir)
    stored = _load_db_json_payload(source, run_dir, "summary.json") or load_json(summary_path) or {}
    recomputed = _recompute_summary_from_per_step(str(run_dir))
    if recomputed is None:
        return stored

    payload = dict(stored)
    stored_overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    payload["overall"] = {**stored_overall, **recomputed["overall"]}
    payload["episodes"] = recomputed["episodes"]
    return payload


@lru_cache(maxsize=64)
def _build_topology_bundle(run_dir_str: str) -> Dict[str, Any]:
    run_dir = Path(run_dir_str)
    cfg = load_env_config_from_run(run_dir, verbose=False)
    parsed = parse_run_dir_name(run_dir.name)
    meta = load_json(run_dir / "run_meta.json") or {}

    topology_seed = _to_int(meta.get("topology_seed"))
    if topology_seed is None:
        topology_seed = _to_int(parsed.get("topology_seed"))
    if topology_seed is None:
        topology_seed = _to_int(getattr(cfg, "topology_seed", None))
    if topology_seed is None:
        topology_seed = 0

    reset_seed = _to_int(meta.get("seed"))
    if reset_seed is None:
        reset_seed = _to_int(parsed.get("seed"))
    if reset_seed is None:
        reset_seed = int(topology_seed)

    if hasattr(cfg, "topology_seed"):
        cfg.topology_seed = int(topology_seed)

    env = GreenNetEnv(config=cfg)
    try:
        env.reset(seed=int(reset_seed))
        graph = env.simulator.graph if env.simulator is not None else getattr(env, "_base_graph", None)
        if graph is None:
            raise RuntimeError(f"Could not build topology for run: {run_dir.name}")

        raw_layout = nx.spring_layout(graph, seed=int(topology_seed)) if graph.number_of_nodes() > 0 else {}
        layout = _normalize_layout(raw_layout)

        nodes: List[Dict[str, Any]] = []
        for node in graph.nodes():
            node_id = _node_id(node)
            x, y = layout.get(node_id, (0.5, 0.5))
            nodes.append({"id": node_id, "label": node_id, "x": x, "y": y})

        edges: List[Dict[str, Any]] = []
        initial_links: Dict[str, bool] = {}
        for source, target in graph.edges():
            source_id = _node_id(source)
            target_id = _node_id(target)
            edge_key = _edge_id(source_id, target_id)
            edges.append({"id": edge_key, "source": source_id, "target": target_id})
            initial_links[edge_key] = bool(graph.edges[source, target].get("active", True))

        action_edges = [(_node_id(source), _node_id(target)) for (source, target) in getattr(env, "edge_list", [])]
        return {
            "nodes": nodes,
            "edges": edges,
            "initial_links": initial_links,
            "action_edges": action_edges,
        }
    finally:
        env.close()


def _stable_ratio(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _first_episode_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    first_episode = next((row.get("episode") for row in raw_rows if row.get("episode") is not None), None)
    if first_episode is None:
        return raw_rows
    return [row for row in raw_rows if row.get("episode") == first_episode]


def _resolve_replay_seed(run_dir: Path, episode_rows: List[Dict[str, Any]], cfg: Any) -> int:
    meta = load_json(run_dir / "run_meta.json") or {}
    parsed = parse_run_dir_name(run_dir.name)
    episode_seed = _to_int(episode_rows[0].get("episode_seed")) if episode_rows else None
    if episode_seed is not None:
        return int(episode_seed)

    for candidate in (meta.get("eval_seed"), meta.get("seed"), parsed.get("seed"), getattr(cfg, "topology_seed", None)):
        parsed_candidate = _to_int(candidate)
        if parsed_candidate is not None:
            return int(parsed_candidate)
    return 0


def _resolve_replay_traffic_seed(run_dir: Path, episode_rows: List[Dict[str, Any]], cfg: Any) -> Optional[int]:
    meta = load_json(run_dir / "run_meta.json") or {}
    episode_idx = _to_int(episode_rows[0].get("episode")) if episode_rows else None
    if episode_idx is None:
        episode_idx = 0

    for candidate in (meta.get("traffic_seed_base"), meta.get("traffic_seed"), getattr(cfg, "traffic_seed", None)):
        parsed_candidate = _to_int(candidate)
        if parsed_candidate is not None:
            return int(parsed_candidate) + int(episode_idx)
    return None


def _links_on_from_env(env: GreenNetEnv) -> Dict[str, bool]:
    if env.simulator is None:
        return {}

    links: Dict[str, bool] = {}
    for source, target in env.simulator.graph.edges():
        edge_key = env.simulator._edge_key(int(source), int(target))
        links[_edge_id(_node_id(source), _node_id(target))] = bool(env.simulator.active.get(edge_key, True))
    return links


def _distribute_budget(weights: List[float], budget: int) -> List[int]:
    counts = [0] * len(weights)
    if budget <= 0 or not weights:
        return counts

    cleaned = [max(0.0, float(weight)) for weight in weights]
    total = sum(cleaned)
    if total <= 0.0:
        return counts

    exact = [(budget * weight / total) if weight > 0.0 else 0.0 for weight in cleaned]
    positive_indices = [index for index, weight in enumerate(cleaned) if weight > 0.0]

    for index in positive_indices:
        counts[index] = int(exact[index])

    if budget >= len(positive_indices):
        for index in positive_indices:
            if counts[index] == 0:
                counts[index] = 1

    used = sum(counts)
    if used > budget:
        trim_order = sorted(positive_indices, key=lambda index: (exact[index] - counts[index], counts[index]))
        for index in trim_order:
            while used > budget and counts[index] > 1:
                counts[index] -= 1
                used -= 1
        if used > budget:
            for index in trim_order:
                while used > budget and counts[index] > 0:
                    counts[index] -= 1
                    used -= 1
    elif used < budget and positive_indices:
        add_order = sorted(positive_indices, key=lambda index: exact[index] - counts[index], reverse=True)
        pointer = 0
        while used < budget:
            index = add_order[pointer % len(add_order)]
            counts[index] += 1
            used += 1
            pointer += 1

    return counts


def _fallback_edge_key(env: GreenNetEnv, source: int, target: int) -> Optional[Tuple[int, int]]:
    if env.simulator is None:
        return None

    graph = env.simulator.graph
    for node in (source, target):
        if graph.has_node(node):
            neighbors = sorted(graph.neighbors(node))
            if neighbors:
                return env.simulator._edge_key(int(node), int(neighbors[0]))

    first_edge = next(iter(graph.edges()), None)
    if first_edge is None:
        return None
    return env.simulator._edge_key(int(first_edge[0]), int(first_edge[1]))


def _packet_event(edge_key: Tuple[int, int], *, packet_id: str, status: str, progress_key: str) -> Dict[str, Any]:
    source_id = _node_id(edge_key[0])
    target_id = _node_id(edge_key[1])
    return {
        "packet_id": packet_id,
        "edge_id": _edge_id(source_id, target_id),
        "source": source_id,
        "target": target_id,
        "progress": 0.08 + 0.84 * _stable_ratio(progress_key),
        "status": status,
    }


def _packet_events_from_env_step(env: GreenNetEnv, info: Dict[str, Any], step_t: int) -> List[Dict[str, Any]]:
    simulator = env.simulator
    if simulator is None:
        return []

    raw_flows = info.get("flows") or ()
    flows = [simulator._normalize_flow(flow) for flow in raw_flows]
    if not flows:
        return []

    routing_graph = simulator._active_routing_graph()
    desired_by_edge: Dict[Tuple[int, int], float] = {}
    allocations: List[Dict[str, Any]] = []
    unrouted: List[Dict[str, Any]] = []

    for flow_index, flow in enumerate(flows):
        source = int(getattr(flow, "source"))
        destination = int(getattr(flow, "destination"))
        demand = max(0.0, float(getattr(flow, "demand", 0.0)))
        if source == destination or demand <= 0.0:
            continue

        paths, weights = simulator._resolve_paths(source, destination, routing_graph)
        if not paths:
            unrouted.append(
                {
                    "flow_index": flow_index,
                    "source": source,
                    "destination": destination,
                    "demand": demand,
                }
            )
            continue

        for path_index, (path, weight) in enumerate(zip(paths, weights)):
            desired = demand * max(0.0, float(weight))
            edge_keys = simulator._path_edges(path)
            if desired <= 0.0 or not edge_keys:
                continue

            allocations.append(
                {
                    "flow_index": flow_index,
                    "path_index": path_index,
                    "source": source,
                    "destination": destination,
                    "desired": desired,
                    "edge_keys": edge_keys,
                }
            )
            for edge_key in edge_keys:
                desired_by_edge[edge_key] = desired_by_edge.get(edge_key, 0.0) + desired

    edge_scale: Dict[Tuple[int, int], float] = {}
    for edge_key, desired in desired_by_edge.items():
        if not simulator.active.get(edge_key, True):
            edge_scale[edge_key] = 0.0
            continue
        capacity = float(simulator.capacity.get(edge_key, 0.0))
        if capacity <= 0.0:
            edge_scale[edge_key] = 0.0
            continue
        edge_scale[edge_key] = min(1.0, capacity / desired) if desired > 0.0 else 1.0

    for allocation in allocations:
        scales = [edge_scale.get(edge_key, 0.0) for edge_key in allocation["edge_keys"]]
        scale = min(scales) if scales else 0.0
        delivered = allocation["desired"] * scale
        dropped = max(0.0, allocation["desired"] - delivered)
        bottlenecks = [
            edge_key
            for edge_key in allocation["edge_keys"]
            if abs(edge_scale.get(edge_key, 0.0) - scale) <= 1e-9
        ] or [allocation["edge_keys"][-1]]
        allocation["delivered"] = delivered
        allocation["dropped"] = dropped
        allocation["bottlenecks"] = bottlenecks

    delivered_total = sum(float(item.get("delivered", 0.0)) for item in allocations)
    dropped_total = sum(float(item.get("dropped", 0.0)) for item in allocations) + sum(
        float(item.get("demand", 0.0)) for item in unrouted
    )
    total_sent = delivered_total + dropped_total
    if total_sent <= 0.0:
        return []

    budget = max(6, min(PACKET_EVENT_LIMIT, int(round(total_sent))))
    delivered_budget = int(round(budget * (delivered_total / total_sent))) if delivered_total > 0 else 0
    delivered_budget = min(delivered_budget, budget)
    dropped_budget = budget - delivered_budget

    delivered_counts = _distribute_budget(
        [float(item.get("delivered", 0.0)) * max(1, len(item.get("edge_keys", []))) for item in allocations],
        delivered_budget,
    )
    dropped_counts = _distribute_budget(
        [float(item.get("dropped", 0.0)) for item in allocations] + [float(item.get("demand", 0.0)) for item in unrouted],
        dropped_budget,
    )

    events: List[Dict[str, Any]] = []
    for allocation_index, allocation in enumerate(allocations):
        count = delivered_counts[allocation_index] if allocation_index < len(delivered_counts) else 0
        edge_keys = allocation.get("edge_keys") or []
        if count <= 0 or not edge_keys:
            continue
        for sample_index in range(count):
            segment_index = sample_index % len(edge_keys)
            edge_key = edge_keys[segment_index]
            status = "delivered" if segment_index == len(edge_keys) - 1 else "in_transit"
            packet_id = (
                f"step-{step_t}-flow-{allocation['flow_index']}-path-{allocation['path_index']}"
                f"-seg-{segment_index}-ok-{sample_index}"
            )
            events.append(
                _packet_event(
                    edge_key,
                    packet_id=packet_id,
                    status=status,
                    progress_key=packet_id,
                )
            )

    dropped_offset = len(allocations)
    for allocation_index, allocation in enumerate(allocations):
        count_index = allocation_index
        count = dropped_counts[count_index] if count_index < len(dropped_counts) else 0
        bottlenecks = allocation.get("bottlenecks") or allocation.get("edge_keys") or []
        if count <= 0 or not bottlenecks:
            continue
        for sample_index in range(count):
            edge_key = bottlenecks[sample_index % len(bottlenecks)]
            packet_id = (
                f"step-{step_t}-flow-{allocation['flow_index']}-path-{allocation['path_index']}"
                f"-drop-{sample_index}"
            )
            events.append(
                _packet_event(
                    edge_key,
                    packet_id=packet_id,
                    status="dropped",
                    progress_key=packet_id,
                )
            )

    for unrouted_index, item in enumerate(unrouted):
        count_index = dropped_offset + unrouted_index
        count = dropped_counts[count_index] if count_index < len(dropped_counts) else 0
        if count <= 0:
            continue
        edge_key = _fallback_edge_key(env, int(item["source"]), int(item["destination"]))
        if edge_key is None:
            continue
        for sample_index in range(count):
            packet_id = f"step-{step_t}-flow-{item['flow_index']}-noroute-{sample_index}"
            events.append(
                _packet_event(
                    edge_key,
                    packet_id=packet_id,
                    status="dropped",
                    progress_key=packet_id,
                )
            )

    return events[:PACKET_EVENT_LIMIT]


def _build_step_payload_fallback(run_dir_str: str, episode_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    topology = _build_topology_bundle(run_dir_str)
    current_links = dict(topology["initial_links"])
    action_edges = list(topology["action_edges"])

    steps: List[Dict[str, Any]] = []
    for index, row in enumerate(episode_rows):
        metrics = dict(row)
        action = _to_int(metrics.get("action"))
        toggle_applied = _to_bool(metrics.get("toggle_applied")) is True
        if toggle_applied and action is not None:
            action_index = action - 1
            if 0 <= action_index < len(action_edges):
                source_id, target_id = action_edges[action_index]
                edge_key = _edge_id(source_id, target_id)
                current_links[edge_key] = not bool(current_links.get(edge_key, True))

        t = _to_int(metrics.get("t"))
        if t is None:
            t = _to_int(metrics.get("step"))
        if t is None:
            t = index

        steps.append(
            {
                "t": int(t),
                "metrics": metrics,
                "links_on": dict(current_links),
                "packet_events": [],
            }
        )

    return tuple(steps)


@lru_cache(maxsize=64)
def _build_step_payload(run_dir_str: str) -> Tuple[Dict[str, Any], ...]:
    raw_rows = list(_read_all_per_step_rows(run_dir_str))
    if not raw_rows:
        return tuple()

    episode_rows = _first_episode_rows(raw_rows)
    run_dir = Path(run_dir_str)
    cfg = load_env_config_from_run(run_dir, verbose=False)

    replay_seed = _resolve_replay_seed(run_dir, episode_rows, cfg)
    replay_traffic_seed = _resolve_replay_traffic_seed(run_dir, episode_rows, cfg)
    if replay_traffic_seed is not None and hasattr(cfg, "traffic_seed"):
        cfg.traffic_seed = int(replay_traffic_seed)

    env = GreenNetEnv(config=cfg)
    try:
        env.reset(seed=int(replay_seed))
        total_edges = max(1, int(env.simulator.graph.number_of_edges()) if env.simulator is not None else 0)

        steps: List[Dict[str, Any]] = []
        for index, row in enumerate(episode_rows):
            metrics = dict(row)
            action = _to_int(metrics.get("action")) or 0
            _, _, terminated, truncated, info = env.step(int(action))

            links_on = _links_on_from_env(env)
            on_count = sum(1 for is_on in links_on.values() if is_on)
            metrics["active_ratio"] = float(on_count / total_edges)

            t = _to_int(metrics.get("t"))
            if t is None:
                t = _to_int(metrics.get("step"))
            if t is None:
                t = index + 1

            steps.append(
                {
                    "t": int(t),
                    "metrics": metrics,
                    "links_on": links_on,
                    "packet_events": _packet_events_from_env_step(env, info, int(t)),
                }
            )

            if terminated or truncated:
                break

        return tuple(steps)
    except Exception:
        return _build_step_payload_fallback(run_dir_str, episode_rows)
    finally:
        env.close()


def _step_for_timestep(steps: Tuple[Dict[str, Any], ...], step: int) -> Optional[Dict[str, Any]]:
    if not steps:
        return None

    exact = next((item for item in steps if _to_int(item.get("t")) == int(step)), None)
    if exact is not None:
        return exact

    prior = [item for item in steps if (_to_int(item.get("t")) or 0) <= int(step)]
    if prior:
        return prior[-1]
    return steps[0]


def parse_run_dir_name(name: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "started_at": None,
        "policy": None,
        "scenario": None,
        "seed": None,
        "topology_seed": None,
        "tag": None,
    }

    prefix_match = RUN_PREFIX_RE.match(name)
    if prefix_match:
        try:
            dt = datetime.strptime(prefix_match.group("stamp"), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            parsed["started_at"] = dt.isoformat()
        except ValueError:
            parsed["started_at"] = None

    for token in name.split("__")[1:]:
        if token.startswith("policy-"):
            parsed["policy"] = token[len("policy-") :] or parsed["policy"]
        elif token.startswith("scenario-"):
            parsed["scenario"] = token[len("scenario-") :] or parsed["scenario"]
        elif token.startswith("seed-"):
            parsed["seed"] = _to_int(token[len("seed-") :])
        elif token.startswith("topology_seed-"):
            parsed["topology_seed"] = _to_int(token[len("topology_seed-") :])
        elif token.startswith("topology-seed-"):
            parsed["topology_seed"] = _to_int(token[len("topology-seed-") :])
        elif token.startswith("tag-"):
            parsed["tag"] = token[len("tag-") :] or parsed["tag"]

    return parsed


def list_available_files(run_dir: Path) -> List[str]:
    files = [p.name for p in run_dir.iterdir() if p.is_file()]
    files.sort(key=lambda name: (0 if name in KEY_FILES else 1, name))
    return files


def _key_file_flags(run_dir: Path) -> Dict[str, bool]:
    return {
        "per_step": (run_dir / "per_step.csv").exists(),
        "summary": (run_dir / "summary.json").exists(),
        "meta": (run_dir / "run_meta.json").exists(),
        "env_config": (run_dir / "env_config.json").exists(),
    }


def get_run_record(run_dir: Path, source: str, db_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parsed = parse_run_dir_name(run_dir.name)
    file_flags = _key_file_flags(run_dir)
    db_has = db_snapshot.get("has") if isinstance(db_snapshot, dict) and isinstance(db_snapshot.get("has"), dict) else {}
    flags = {key: bool(file_flags.get(key)) or bool(db_has.get(key)) for key in file_flags}

    record: Dict[str, Any] = {
        "run_id": run_dir.name,
        "source": source,
        "started_at": parsed.get("started_at"),
        "timestamp_utc": None,
        "policy": parsed.get("policy"),
        "scenario": parsed.get("scenario"),
        "seed": parsed.get("seed"),
        "topology_seed": parsed.get("topology_seed"),
        "tag": parsed.get("tag"),
        "episodes": None,
        "max_steps": None,
        "deterministic": None,
        "model_path": None,
        "has": flags,
        "highlights": {field: None for field in HIGHLIGHT_FIELDS},
    }

    if db_snapshot:
        record["timestamp_utc"] = db_snapshot.get("timestamp_utc")
        if record["started_at"] is None and isinstance(record["timestamp_utc"], str):
            record["started_at"] = record["timestamp_utc"]

        record["policy"] = db_snapshot.get("policy") or record["policy"]
        record["scenario"] = db_snapshot.get("scenario") or record["scenario"]
        record["tag"] = db_snapshot.get("tag") or record["tag"]

        if db_snapshot.get("seed") is not None:
            record["seed"] = _to_int(db_snapshot.get("seed"))
        if db_snapshot.get("topology_seed") is not None:
            record["topology_seed"] = _to_int(db_snapshot.get("topology_seed"))

        record["episodes"] = _to_int(db_snapshot.get("episodes"))
        record["max_steps"] = _to_int(db_snapshot.get("max_steps"))
        record["deterministic"] = _to_bool(db_snapshot.get("deterministic"))
        record["model_path"] = db_snapshot.get("model_path")

        highlights_payload = (
            db_snapshot.get("highlights")
            if isinstance(db_snapshot.get("highlights"), dict)
            else {}
        )
        record["highlights"] = {field: _to_float(highlights_payload.get(field)) for field in HIGHLIGHT_FIELDS}
        return record

    meta = load_json(run_dir / "run_meta.json")
    if meta:
        record["timestamp_utc"] = meta.get("timestamp_utc") or meta.get("created_at_utc")
        if record["started_at"] is None and isinstance(record["timestamp_utc"], str):
            record["started_at"] = record["timestamp_utc"]

        record["policy"] = meta.get("policy") or record["policy"]
        record["scenario"] = meta.get("scenario") or record["scenario"]
        record["tag"] = meta.get("tag") or record["tag"]

        if meta.get("seed") is not None:
            record["seed"] = _to_int(meta.get("seed"))
        if meta.get("topology_seed") is not None:
            record["topology_seed"] = _to_int(meta.get("topology_seed"))

        record["episodes"] = _to_int(meta.get("episodes"))
        record["max_steps"] = _to_int(meta.get("max_steps") if meta.get("max_steps") is not None else meta.get("steps"))
        record["deterministic"] = _to_bool(meta.get("deterministic"))
        record["model_path"] = meta.get("model_path")

    summary = load_json(run_dir / "summary.json")
    if summary and isinstance(summary.get("overall"), dict):
        overall = summary["overall"]
        for field in HIGHLIGHT_FIELDS:
            record["highlights"][field] = _to_float(overall.get(field))

    return record


def _scan_run_dirs(base: BaseChoice = "both") -> List[Tuple[str, Path]]:
    dirs: List[Tuple[str, Path]] = []
    roots: List[Tuple[str, Path]] = []

    if base in {"results", "both"}:
        roots.append(("results", RESULTS_DIR))
    if base in {"runs", "both"}:
        roots.append(("runs", RUNS_DIR))

    for source, root in roots:
        if root.exists():
            dirs.extend((source, p) for p in root.iterdir() if p.is_dir())

    # newest first (by folder name timestamp, usually)
    return sorted(dirs, key=lambda item: item[1].name, reverse=True)


def _find_run_dir(run_id: str, base: BaseChoice = "both") -> Optional[Tuple[str, Path]]:
    for source, run_dir in _scan_run_dirs(base=base):
        if run_dir.name == run_id:
            return source, run_dir
    return None


def _resolve_run_dir_or_404(run_id: str, base: BaseChoice) -> Tuple[str, Path]:
    if run_id == "latest":
        runs = _scan_run_dirs(base=base)
        if not runs:
            raise HTTPException(status_code=404, detail="No runs available")
        return runs[0]

    resolved = _find_run_dir(run_id, base=base)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return resolved


def _load_run_json_or_error(run_id: str, filename: str, base: BaseChoice) -> Tuple[str, Dict[str, Any]]:
    source, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    payload = _load_db_json_payload(source, run_dir, filename)
    if payload is not None:
        return source, payload

    file_path = run_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found for run_id: {run_id}")

    payload = load_json(file_path)
    if payload is None:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON file: {filename} for run_id: {run_id}")

    return source, payload


def _matches_text_filter(actual: Any, expected: Optional[str]) -> bool:
    if expected is None:
        return True
    if actual is None:
        return False
    return str(actual).lower() == expected.lower()


def _coerce_per_step_value(key: str, value: Optional[str]) -> Any:
    if value is None or value == "":
        return value
    if key in INT_COLUMNS:
        parsed_int = _to_int(value)
        return parsed_int if parsed_int is not None else value
    if key in FLOAT_COLUMNS or key.startswith("delta_"):
        parsed_float = _to_float(value)
        return parsed_float if parsed_float is not None else value
    return value


def _parse_group_by(group_by: str) -> List[str]:
    if not group_by.strip():
        return []

    fields: List[str] = []
    invalid: List[str] = []
    for raw in group_by.split(","):
        field = raw.strip()
        if not field:
            continue
        if field not in GROUP_BY_FIELDS:
            invalid.append(field)
            continue
        if field not in fields:
            fields.append(field)

    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by field(s): {', '.join(invalid)}. Allowed: {', '.join(sorted(GROUP_BY_FIELDS))}",
        )
    return fields


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _run_sort_key(record: Dict[str, Any]) -> Tuple[int, float, int, float, str, str]:
    ts = _parse_iso_timestamp(record.get("timestamp_utc"))
    started = _parse_iso_timestamp(record.get("started_at"))
    ts_epoch = ts.timestamp() if ts else float("-inf")
    started_epoch = started.timestamp() if started else float("-inf")
    return (
        1 if ts else 0,
        ts_epoch,
        1 if started else 0,
        started_epoch,
        str(record.get("run_id") or ""),
        str(record.get("source") or ""),
    )


def _list_runs_payload(
    base: BaseChoice,
    tag: Optional[str],
    policy: Optional[str],
    scenario: Optional[str],
    seed: Optional[int],
    topology_seed: Optional[int],
    deterministic: Optional[bool],
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    db_snapshots = _db_run_snapshots(base)
    for source, run_dir in _scan_run_dirs(base=base):
        try:
            record = get_run_record(run_dir, source, db_snapshot=db_snapshots.get((source, run_dir.name)))
        except Exception:
            # Never fail this endpoint because one folder is broken.
            continue

        if not any(record["has"].values()):
            continue
        if not _matches_text_filter(record.get("tag"), tag):
            continue
        if not _matches_text_filter(record.get("policy"), policy):
            continue
        if not _matches_text_filter(record.get("scenario"), scenario):
            continue
        if seed is not None and record.get("seed") != seed:
            continue
        if topology_seed is not None and record.get("topology_seed") != topology_seed:
            continue
        if deterministic is not None and record.get("deterministic") is not deterministic:
            continue

        # Keep response shape stable for UI consumers.
        record["run_id"] = record.get("run_id") or run_dir.name
        record["source"] = record.get("source") or source
        has_payload = record.get("has") if isinstance(record.get("has"), dict) else {}
        record["has"] = {
            "per_step": bool(has_payload.get("per_step")),
            "summary": bool(has_payload.get("summary")),
            "meta": bool(has_payload.get("meta")),
            "env_config": bool(has_payload.get("env_config")),
        }
        highlights_payload = record.get("highlights") if isinstance(record.get("highlights"), dict) else {}
        record["highlights"] = {field: highlights_payload.get(field) for field in HIGHLIGHT_FIELDS}
        runs.append(record)

    # Deterministic newest-first ordering:
    # 1) run_meta.timestamp_utc, 2) parsed started_at, 3) run folder name desc.
    runs.sort(key=_run_sort_key, reverse=True)
    total = len(runs)
    return {"total": total, "items": runs[offset : offset + limit]}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/version")
def version() -> Dict[str, str]:
    return {"version": _git_hash()}


@app.get("/api/runs")
def list_runs(
    base: BaseChoice = Query("both"),
    tag: Optional[str] = Query(None),
    policy: Optional[str] = Query(None),
    scenario: Optional[str] = Query(None),
    seed: Optional[int] = Query(None),
    topology_seed: Optional[int] = Query(None),
    deterministic: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    return _list_runs_payload(
        base=base,
        tag=tag,
        policy=policy,
        scenario=scenario,
        seed=seed,
        topology_seed=topology_seed,
        deterministic=deterministic,
        limit=limit,
        offset=offset,
    )


@app.get("/api/runs_flat")
def list_runs_flat(
    base: BaseChoice = Query("both"),
    tag: Optional[str] = Query(None),
    policy: Optional[str] = Query(None),
    scenario: Optional[str] = Query(None),
    seed: Optional[int] = Query(None),
    topology_seed: Optional[int] = Query(None),
    deterministic: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    payload = _list_runs_payload(
        base=base,
        tag=tag,
        policy=policy,
        scenario=scenario,
        seed=seed,
        topology_seed=topology_seed,
        deterministic=deterministic,
        limit=limit,
        offset=offset,
    )
    return payload["items"]


@app.get("/api/runs/{run_id}/meta")
def run_meta(run_id: str, base: BaseChoice = Query("both")) -> Dict[str, Any]:
    _, payload = _load_run_json_or_error(run_id, "run_meta.json", base=base)
    return payload


@app.get("/api/runs/{run_id}/env")
def run_env(run_id: str, base: BaseChoice = Query("both")) -> Dict[str, Any]:
    _, payload = _load_run_json_or_error(run_id, "env_config.json", base=base)
    return payload


@app.get("/api/runs/{run_id}/summary")
def run_summary(
    run_id: str,
    base: BaseChoice = Query("both"),
    mode: Literal["full", "overall"] = Query("full"),
) -> Dict[str, Any]:
    _, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    payload = _summary_payload_for_run(run_dir)
    if not payload:
        raise HTTPException(status_code=404, detail=f"summary.json not found for run_id: {run_id}")
    if mode == "overall":
        overall = payload.get("overall")
        if not isinstance(overall, dict):
            raise HTTPException(status_code=500, detail=f"summary.json missing valid overall section for run_id: {run_id}")
        return overall
    return payload


@app.get("/api/runs/{run_id}/files")
def run_files(run_id: str, base: BaseChoice = Query("both")) -> Dict[str, Any]:
    source, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    return {
        "run_id": run_id,
        "source": source,
        "files": list_available_files(run_dir),
        "has": _key_file_flags(run_dir),
    }


@app.get("/api/runs/{run_id}/per_step")
def run_per_step(
    run_id: str,
    base: BaseChoice = Query("both"),
    limit: Optional[int] = Query(None, ge=1),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    _, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    rows = list(_read_all_per_step_rows(str(run_dir)))
    if not rows:
        raise HTTPException(status_code=404, detail=f"per_step.csv not found for run_id: {run_id}")
    if limit is None:
        return rows[offset:]
    return rows[offset : offset + limit]


@app.post("/api/runs/start")
def start_run(payload: StartRunRequest) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_experiment.py"),
        "--policy",
        payload.policy.strip().lower(),
        "--scenario",
        payload.scenario.strip().lower(),
        "--seed",
        str(int(payload.seed)),
        "--steps",
        str(int(payload.steps)),
        "--episodes",
        "1",
        "--out-dir",
        str(RESULTS_DIR),
        "--tag",
        DASHBOARD_RUN_TAG,
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"Run timed out after {int(exc.timeout or 600)}s") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "run_experiment.py failed").strip()
        raise HTTPException(status_code=500, detail=detail[-2000:])

    stdout = completed.stdout or ""
    match = RESULTS_SAVED_RE.search(stdout)
    if match:
        output_dir = Path(match.group("path").strip())
    else:
        candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()]
        if not candidates:
            raise HTTPException(status_code=500, detail="Run completed but no output directory was found")
        output_dir = max(candidates, key=lambda path: path.stat().st_mtime)

    return {"run_id": output_dir.name}


@app.get("/api/runs/{run_id}/topology")
def run_topology(run_id: str, base: BaseChoice = Query("both")) -> Dict[str, Any]:
    source, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    bundle = _build_topology_bundle(str(run_dir))
    return {
        "run_id": run_dir.name,
        "source": source,
        "nodes": bundle["nodes"],
        "edges": bundle["edges"],
    }


@app.get("/api/runs/{run_id}/steps")
@app.get("/api/runs/{run_id}/timeline")
def run_steps(run_id: str, base: BaseChoice = Query("both")) -> List[Dict[str, Any]]:
    _, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    steps = list(_build_step_payload(str(run_dir)))
    if not steps:
        raise HTTPException(status_code=404, detail=f"No step timeline available for run_id: {run_id}")
    return steps


@app.get("/api/runs/{run_id}/link_state")
@app.get("/api/runs/{run_id}/links")
def run_link_state(run_id: str, step: int = Query(0, ge=0), base: BaseChoice = Query("both")) -> Dict[str, Any]:
    _, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    steps = _build_step_payload(str(run_dir))
    payload = _step_for_timestep(steps, step)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No link state available for run_id: {run_id}")
    return {
        "run_id": run_dir.name,
        "step": payload["t"],
        "links_on": payload["links_on"],
    }


@app.get("/api/runs/{run_id}/packet_events")
def run_packet_events(run_id: str, base: BaseChoice = Query("both"), step: int = Query(0, ge=0)) -> Dict[str, Any]:
    _, run_dir = _resolve_run_dir_or_404(run_id, base=base)
    steps = _build_step_payload(str(run_dir))
    payload = _step_for_timestep(steps, step)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No packet events available for run_id: {run_id}")
    return {"run_id": run_dir.name, "step": payload["t"], "events": payload.get("packet_events", [])}


@app.get("/api/official_results")
def official_results(scenario: Optional[str] = Query(None)) -> Dict[str, Any]:
    if scenario is None:
        scenarios = list(OFFICIAL_LOCKED_SCENARIOS)
    else:
        scenarios = [item.strip().lower() for item in scenario.split(",") if item.strip()]
        invalid = [item for item in scenarios if item not in OFFICIAL_LOCKED_SCENARIOS]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unsupported scenario filter: {', '.join(invalid)}")

    items: List[Dict[str, Any]] = []
    for name in scenarios:
        payload = _official_locked_result_for_scenario(name)
        if payload is not None:
            items.append(payload)

    return {"total": len(items), "items": items}


@app.get("/api/final_evaluation")
def final_evaluation() -> Dict[str, Any]:
    artifact = _latest_final_evaluation_artifact()
    if artifact is None:
        raise HTTPException(status_code=404, detail="No final evaluation summary artifact was found")

    payload = dict(artifact["payload"])
    summary_path = artifact["summary_path"]
    report_path = artifact.get("report_path")
    payload["artifact"] = {
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
        "report_path": None if report_path is None else str(report_path.relative_to(REPO_ROOT)),
    }
    return payload


@app.get("/api/aggregate")
def aggregate(
    base: BaseChoice = Query("both"),
    tag: Optional[str] = Query(None),
    policy: Optional[str] = Query(None),
    scenario: Optional[str] = Query(None),
    group_by: str = Query(""),
) -> List[Dict[str, Any]]:
    group_fields = _parse_group_by(group_by)
    buckets: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for source, run_dir in _scan_run_dirs(base=base):
        try:
            record = get_run_record(run_dir, source)
        except Exception:
            # Never fail aggregation because one folder is broken.
            continue

        if not _matches_text_filter(record.get("tag"), tag):
            continue
        if not _matches_text_filter(record.get("policy"), policy):
            continue
        if not _matches_text_filter(record.get("scenario"), scenario):
            continue

        metric_values = {field: _to_float(record["highlights"].get(field)) for field in HIGHLIGHT_FIELDS}
        if all(value is None for value in metric_values.values()):
            continue

        key = tuple(record.get(field) for field in group_fields)
        if key not in buckets:
            buckets[key] = {
                "group": {field: record.get(field) for field in group_fields},
                "n": 0,
                "_metrics": {field: [] for field in HIGHLIGHT_FIELDS},
            }

        bucket = buckets[key]
        bucket["n"] += 1
        for field, value in metric_values.items():
            if value is not None:
                bucket["_metrics"][field].append(value)

    # Uses statistics.pstdev (population standard deviation) for *_std fields.
    response: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        item: Dict[str, Any] = {
            "group": bucket["group"],
            "n": bucket["n"],
        }
        for field, values in bucket["_metrics"].items():
            item[f"{field}_mean"] = mean(values) if values else None
            item[f"{field}_std"] = pstdev(values) if values else None
        response.append(item)

    response.sort(
        key=lambda item: tuple(
            "" if item["group"].get(field) is None else str(item["group"].get(field)) for field in group_fields
        )
    )
    return response
