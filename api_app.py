from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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

BaseChoice = Literal["results", "runs", "both"]
GROUP_BY_FIELDS = {"policy", "scenario", "tag", "topology_seed", "deterministic"}
KEY_FILES = ["per_step.csv", "summary.json", "run_meta.json", "env_config.json"]
HIGHLIGHT_FIELDS = [
    "reward_total_mean",
    "energy_kwh_total_mean",
    "dropped_total_mean",
    "toggles_total_mean",
]
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def get_run_record(run_dir: Path, source: str) -> Dict[str, Any]:
    parsed = parse_run_dir_name(run_dir.name)
    flags = _key_file_flags(run_dir)

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
    resolved = _find_run_dir(run_id, base=base)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return resolved


def _load_run_json_or_error(run_id: str, filename: str, base: BaseChoice) -> Tuple[str, Dict[str, Any]]:
    source, run_dir = _resolve_run_dir_or_404(run_id, base=base)
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
    for source, run_dir in _scan_run_dirs(base=base):
        try:
            record = get_run_record(run_dir, source)
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
    _, payload = _load_run_json_or_error(run_id, "summary.json", base=base)
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

    csv_path = run_dir / "per_step.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"per_step.csv not found for run_id: {run_id}")

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < offset:
                continue

            parsed = {k: _coerce_per_step_value(k, v) for k, v in row.items()}
            rows.append(parsed)

            if limit is not None and len(rows) >= limit:
                break

    return rows


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
