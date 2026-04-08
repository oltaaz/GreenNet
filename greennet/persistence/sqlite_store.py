from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

DEFAULT_DB_ENV_VAR = "GREENNET_DB_PATH"
RUN_PREFIX_RE = re.compile(r"^(?P<stamp>\d{8}_\d{6})")
SUMMARY_HIGHLIGHT_FIELDS = (
    "reward_total_mean",
    "energy_kwh_total_mean",
    "dropped_total_mean",
    "toggles_total_mean",
)

_MIGRATIONS: Sequence[tuple[str, str]] = (
    (
        "001_core_run_store",
        """
        CREATE TABLE runs (
            source TEXT NOT NULL,
            run_name TEXT NOT NULL,
            run_path TEXT NOT NULL,
            meta_run_id TEXT,
            policy TEXT,
            scenario TEXT,
            seed INTEGER,
            eval_seed INTEGER,
            topology_seed INTEGER,
            traffic_seed INTEGER,
            traffic_seed_base INTEGER,
            tag TEXT,
            deterministic INTEGER,
            save_flows INTEGER,
            episodes INTEGER,
            max_steps INTEGER,
            model_path TEXT,
            runs_dir TEXT,
            policy_type TEXT,
            model_type TEXT,
            control_mode TEXT,
            traffic_scenario TEXT,
            traffic_scenario_version INTEGER,
            traffic_scenario_intensity REAL,
            traffic_scenario_duration REAL,
            traffic_scenario_frequency REAL,
            timestamp_utc TEXT,
            created_at_utc TEXT,
            command TEXT,
            has_meta INTEGER NOT NULL DEFAULT 0,
            has_summary INTEGER NOT NULL DEFAULT 0,
            has_env_config INTEGER NOT NULL DEFAULT 0,
            has_steps INTEGER NOT NULL DEFAULT 0,
            meta_json TEXT,
            env_config_json TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (source, run_name)
        );

        CREATE TABLE run_summaries (
            source TEXT NOT NULL,
            run_name TEXT NOT NULL,
            episodes_count INTEGER,
            reward_total_mean REAL,
            reward_total_std REAL,
            delivered_total_mean REAL,
            delivered_total_std REAL,
            dropped_total_mean REAL,
            dropped_total_std REAL,
            energy_kwh_total_mean REAL,
            energy_kwh_total_std REAL,
            carbon_g_total_mean REAL,
            carbon_g_total_std REAL,
            steps_mean REAL,
            steps_std REAL,
            avg_utilization_mean REAL,
            active_ratio_mean REAL,
            avg_delay_ms_mean REAL,
            avg_path_latency_ms_mean REAL,
            avg_path_latency_ms_std REAL,
            toggles_total_mean REAL,
            toggles_total_std REAL,
            summary_json TEXT NOT NULL,
            overall_json TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (source, run_name),
            FOREIGN KEY (source, run_name) REFERENCES runs(source, run_name) ON DELETE CASCADE
        );

        CREATE TABLE run_episode_summaries (
            source TEXT NOT NULL,
            run_name TEXT NOT NULL,
            episode INTEGER NOT NULL,
            steps INTEGER,
            reward_total REAL,
            delivered_total REAL,
            dropped_total REAL,
            energy_kwh_total REAL,
            carbon_g_total REAL,
            avg_utilization_mean REAL,
            active_ratio_mean REAL,
            avg_delay_ms_mean REAL,
            avg_path_latency_ms_mean REAL,
            toggles_total INTEGER,
            toggles_applied_total INTEGER,
            toggles_reverted_total INTEGER,
            blocked_by_util_count INTEGER,
            blocked_by_cooldown_count INTEGER,
            allowed_toggle_count INTEGER,
            toggles_attempted_count INTEGER,
            toggles_applied_count INTEGER,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (source, run_name, episode),
            FOREIGN KEY (source, run_name) REFERENCES runs(source, run_name) ON DELETE CASCADE
        );

        CREATE TABLE run_step_metrics (
            source TEXT NOT NULL,
            run_name TEXT NOT NULL,
            episode INTEGER NOT NULL,
            step INTEGER NOT NULL,
            episode_seed INTEGER,
            action INTEGER,
            reward REAL,
            terminated INTEGER,
            truncated INTEGER,
            avg_utilization REAL,
            active_ratio REAL,
            avg_delay_ms REAL,
            avg_path_latency_ms REAL,
            delivered REAL,
            dropped REAL,
            energy_kwh REAL,
            carbon_g REAL,
            delta_energy_kwh REAL,
            delta_delivered REAL,
            delta_dropped REAL,
            delta_carbon_g REAL,
            num_active_edges INTEGER,
            near_saturated_edges INTEGER,
            qos_violation INTEGER,
            toggle_applied INTEGER,
            toggle_reverted INTEGER,
            blocked_by_util_count INTEGER,
            blocked_by_cooldown_count INTEGER,
            allowed_toggle_count INTEGER,
            toggles_attempted_count INTEGER,
            toggles_applied_count INTEGER,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (source, run_name, episode, step),
            FOREIGN KEY (source, run_name) REFERENCES runs(source, run_name) ON DELETE CASCADE
        );

        CREATE INDEX idx_runs_filter
            ON runs (source, policy, scenario, tag, seed, topology_seed, deterministic);
        CREATE INDEX idx_run_episode_lookup
            ON run_episode_summaries (source, run_name, episode);
        CREATE INDEX idx_run_steps_lookup
            ON run_step_metrics (source, run_name, episode, step);
        """,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_db_path(repo_root: Path | None = None) -> Path:
    root = repo_root.resolve() if repo_root is not None else _repo_root()
    return root / "artifacts" / "db" / "greennet.sqlite3"


def _resolve_db_path(db_path: Path | str | None = None) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser().resolve()
    override = os.environ.get(DEFAULT_DB_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return default_db_path()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if value is None or value == "":
        return None
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


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None


def _first_float(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _bool_to_db(value: Any) -> Optional[int]:
    parsed = _to_bool(value)
    if parsed is None:
        return None
    return 1 if parsed else 0


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _json_load_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or value == "":
        return None
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _json_load_row(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or value == "":
        return None
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _run_mtime_iso(run_dir: Path) -> str:
    return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()


def _parse_run_dir_name(name: str) -> Dict[str, Any]:
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


def infer_run_source(run_dir: Path, repo_root: Path | None = None) -> str:
    root = repo_root.resolve() if repo_root is not None else _repo_root()
    resolved = run_dir.expanduser().resolve()
    for source in ("results", "runs"):
        candidate_root = (root / source).resolve()
        try:
            resolved.relative_to(candidate_root)
            return source
        except ValueError:
            continue
    return resolved.parent.name


@dataclass(frozen=True)
class RunDirectoryArtifacts:
    source: str
    run_name: str
    run_path: str
    run_dir: Path
    parsed: Dict[str, Any]
    meta: Optional[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]
    env_config: Optional[Dict[str, Any]]
    step_rows: List[Dict[str, Any]]


@dataclass(frozen=True)
class PersistRunResult:
    source: str
    run_name: str
    episodes: int
    steps: int
    has_summary: bool
    has_steps: bool


@dataclass(frozen=True)
class BackfillReport:
    scanned: int
    persisted: int
    skipped: int
    failed: int


class RunRepository(Protocol):
    def ensure_initialized(self) -> None: ...

    def upsert_run_directory(self, artifacts: RunDirectoryArtifacts) -> PersistRunResult: ...

    def list_run_snapshots(self, base: str = "both") -> List[Dict[str, Any]]: ...

    def get_run_meta(self, source: str, run_name: str) -> Optional[Dict[str, Any]]: ...

    def get_env_config(self, source: str, run_name: str) -> Optional[Dict[str, Any]]: ...

    def get_run_summary(self, source: str, run_name: str) -> Optional[Dict[str, Any]]: ...

    def get_step_rows(self, source: str, run_name: str) -> List[Dict[str, Any]]: ...


def discover_run_artifacts(
    run_dir: Path,
    *,
    source: str | None = None,
    repo_root: Path | None = None,
) -> Optional[RunDirectoryArtifacts]:
    root = repo_root.resolve() if repo_root is not None else _repo_root()
    resolved_dir = run_dir.expanduser().resolve()
    resolved_source = source or infer_run_source(resolved_dir, repo_root=root)

    meta = _load_json_object(resolved_dir / "run_meta.json")
    summary = _load_json_object(resolved_dir / "summary.json")
    env_config = _load_json_object(resolved_dir / "env_config.json")
    step_rows = _load_csv_rows(resolved_dir / "per_step.csv")

    if meta is None and summary is None and env_config is None and not step_rows:
        return None

    parsed = _parse_run_dir_name(resolved_dir.name)
    try:
        run_path = str(resolved_dir.relative_to(root))
    except ValueError:
        run_path = str(resolved_dir)

    return RunDirectoryArtifacts(
        source=resolved_source,
        run_name=resolved_dir.name,
        run_path=run_path,
        run_dir=resolved_dir,
        parsed=parsed,
        meta=meta,
        summary=summary,
        env_config=env_config,
        step_rows=step_rows,
    )


class SqliteRunRepository:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = _resolve_db_path(db_path)
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._connect() as conn:
            self._apply_migrations(conn)
        self._initialized = True

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at_utc TEXT NOT NULL
            )
            """
        )
        applied = {
            str(row["version"])
            for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")
        }
        for version, sql in _MIGRATIONS:
            if version in applied:
                continue
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at_utc) VALUES (?, ?)",
                (version, _now_utc_iso()),
            )

    def upsert_run_directory(self, artifacts: RunDirectoryArtifacts) -> PersistRunResult:
        self.ensure_initialized()

        meta = artifacts.meta or {}
        summary = artifacts.summary or {}
        overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
        episode_rows = summary.get("episodes") if isinstance(summary.get("episodes"), list) else []
        parsed = artifacts.parsed
        mtime_iso = _run_mtime_iso(artifacts.run_dir)
        updated_at_utc = _now_utc_iso()

        timestamp_utc = _first_text(meta.get("timestamp_utc"), meta.get("created_at_utc"), parsed.get("started_at"), mtime_iso)
        created_at_utc = _first_text(meta.get("created_at_utc"), meta.get("timestamp_utc"), parsed.get("started_at"), mtime_iso)
        episodes_value = _first_int(meta.get("episodes"), overall.get("episodes"), len(episode_rows))
        max_steps_value = _first_int(meta.get("max_steps"), meta.get("steps"))

        with self._connect() as conn, conn:
            self._apply_migrations(conn)
            conn.execute(
                "DELETE FROM runs WHERE source = ? AND run_name = ?",
                (artifacts.source, artifacts.run_name),
            )
            conn.execute(
                """
                INSERT INTO runs (
                    source,
                    run_name,
                    run_path,
                    meta_run_id,
                    policy,
                    scenario,
                    seed,
                    eval_seed,
                    topology_seed,
                    traffic_seed,
                    traffic_seed_base,
                    tag,
                    deterministic,
                    save_flows,
                    episodes,
                    max_steps,
                    model_path,
                    runs_dir,
                    policy_type,
                    model_type,
                    control_mode,
                    traffic_scenario,
                    traffic_scenario_version,
                    traffic_scenario_intensity,
                    traffic_scenario_duration,
                    traffic_scenario_frequency,
                    timestamp_utc,
                    created_at_utc,
                    command,
                    has_meta,
                    has_summary,
                    has_env_config,
                    has_steps,
                    meta_json,
                    env_config_json,
                    updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifacts.source,
                    artifacts.run_name,
                    artifacts.run_path,
                    _first_text(meta.get("run_id")),
                    _first_text(meta.get("policy"), parsed.get("policy")),
                    _first_text(meta.get("scenario"), parsed.get("scenario")),
                    _first_int(meta.get("seed"), parsed.get("seed")),
                    _first_int(meta.get("eval_seed"), meta.get("seed"), parsed.get("seed")),
                    _first_int(meta.get("topology_seed"), parsed.get("topology_seed")),
                    _first_int(meta.get("traffic_seed")),
                    _first_int(meta.get("traffic_seed_base"), meta.get("traffic_seed")),
                    _first_text(meta.get("tag"), parsed.get("tag")),
                    _bool_to_db(meta.get("deterministic")),
                    _bool_to_db(meta.get("save_flows")),
                    episodes_value,
                    max_steps_value,
                    _first_text(meta.get("model_path")),
                    _first_text(meta.get("runs_dir")),
                    _first_text(meta.get("policy_type")),
                    _first_text(meta.get("model_type")),
                    _first_text(meta.get("control_mode")),
                    _first_text(meta.get("traffic_scenario")),
                    _first_int(meta.get("traffic_scenario_version")),
                    _first_float(meta.get("traffic_scenario_intensity")),
                    _first_float(meta.get("traffic_scenario_duration")),
                    _first_float(meta.get("traffic_scenario_frequency")),
                    timestamp_utc,
                    created_at_utc,
                    _first_text(meta.get("command")),
                    1 if artifacts.meta is not None else 0,
                    1 if artifacts.summary is not None else 0,
                    1 if artifacts.env_config is not None else 0,
                    1 if artifacts.step_rows else 0,
                    _json_dumps(meta) if artifacts.meta is not None else None,
                    _json_dumps(artifacts.env_config) if artifacts.env_config is not None else None,
                    updated_at_utc,
                ),
            )

            if artifacts.summary is not None:
                conn.execute(
                    """
                    INSERT INTO run_summaries (
                        source,
                        run_name,
                        episodes_count,
                        reward_total_mean,
                        reward_total_std,
                        delivered_total_mean,
                        delivered_total_std,
                        dropped_total_mean,
                        dropped_total_std,
                        energy_kwh_total_mean,
                        energy_kwh_total_std,
                        carbon_g_total_mean,
                        carbon_g_total_std,
                        steps_mean,
                        steps_std,
                        avg_utilization_mean,
                        active_ratio_mean,
                        avg_delay_ms_mean,
                        avg_path_latency_ms_mean,
                        avg_path_latency_ms_std,
                        toggles_total_mean,
                        toggles_total_std,
                        summary_json,
                        overall_json,
                        updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifacts.source,
                        artifacts.run_name,
                        _first_int(overall.get("episodes"), len(episode_rows)),
                        _first_float(overall.get("reward_total_mean")),
                        _first_float(overall.get("reward_total_std")),
                        _first_float(overall.get("delivered_total_mean")),
                        _first_float(overall.get("delivered_total_std")),
                        _first_float(overall.get("dropped_total_mean")),
                        _first_float(overall.get("dropped_total_std")),
                        _first_float(overall.get("energy_kwh_total_mean")),
                        _first_float(overall.get("energy_kwh_total_std")),
                        _first_float(overall.get("carbon_g_total_mean")),
                        _first_float(overall.get("carbon_g_total_std")),
                        _first_float(overall.get("steps_mean")),
                        _first_float(overall.get("steps_std")),
                        _first_float(overall.get("avg_utilization_mean")),
                        _first_float(overall.get("active_ratio_mean")),
                        _first_float(overall.get("avg_delay_ms_mean")),
                        _first_float(overall.get("avg_path_latency_ms_mean")),
                        _first_float(overall.get("avg_path_latency_ms_std")),
                        _first_float(overall.get("toggles_total_mean")),
                        _first_float(overall.get("toggles_total_std")),
                        _json_dumps(summary),
                        _json_dumps(overall) if isinstance(overall, dict) else None,
                        updated_at_utc,
                    ),
                )

                episode_params: List[tuple[Any, ...]] = []
                for episode_index, episode_row in enumerate(episode_rows):
                    if not isinstance(episode_row, dict):
                        continue
                    episode_number = _first_int(episode_row.get("episode"), episode_index)
                    if episode_number is None:
                        continue
                    episode_params.append(
                        (
                            artifacts.source,
                            artifacts.run_name,
                            int(episode_number),
                            _first_int(episode_row.get("steps")),
                            _first_float(episode_row.get("reward_total")),
                            _first_float(episode_row.get("delivered_total")),
                            _first_float(episode_row.get("dropped_total")),
                            _first_float(episode_row.get("energy_kwh_total")),
                            _first_float(episode_row.get("carbon_g_total")),
                            _first_float(episode_row.get("avg_utilization_mean")),
                            _first_float(episode_row.get("active_ratio_mean")),
                            _first_float(episode_row.get("avg_delay_ms_mean")),
                            _first_float(episode_row.get("avg_path_latency_ms_mean")),
                            _first_int(episode_row.get("toggles_total")),
                            _first_int(episode_row.get("toggles_applied_total")),
                            _first_int(episode_row.get("toggles_reverted_total")),
                            _first_int(episode_row.get("blocked_by_util_count")),
                            _first_int(episode_row.get("blocked_by_cooldown_count")),
                            _first_int(episode_row.get("allowed_toggle_count")),
                            _first_int(episode_row.get("toggles_attempted_count")),
                            _first_int(episode_row.get("toggles_applied_count")),
                            _json_dumps(episode_row),
                        )
                    )

                if episode_params:
                    conn.executemany(
                        """
                        INSERT INTO run_episode_summaries (
                            source,
                            run_name,
                            episode,
                            steps,
                            reward_total,
                            delivered_total,
                            dropped_total,
                            energy_kwh_total,
                            carbon_g_total,
                            avg_utilization_mean,
                            active_ratio_mean,
                            avg_delay_ms_mean,
                            avg_path_latency_ms_mean,
                            toggles_total,
                            toggles_applied_total,
                            toggles_reverted_total,
                            blocked_by_util_count,
                            blocked_by_cooldown_count,
                            allowed_toggle_count,
                            toggles_attempted_count,
                            toggles_applied_count,
                            payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        episode_params,
                    )

            if artifacts.step_rows:
                step_params: List[tuple[Any, ...]] = []
                for row in artifacts.step_rows:
                    episode_number = _first_int(row.get("episode"), 0)
                    step_number = _first_int(row.get("step"))
                    if step_number is None:
                        continue
                    step_params.append(
                        (
                            artifacts.source,
                            artifacts.run_name,
                            int(episode_number or 0),
                            int(step_number),
                            _first_int(row.get("episode_seed")),
                            _first_int(row.get("action")),
                            _first_float(row.get("reward")),
                            _bool_to_db(row.get("terminated")),
                            _bool_to_db(row.get("truncated")),
                            _first_float(row.get("avg_utilization")),
                            _first_float(row.get("active_ratio")),
                            _first_float(row.get("avg_delay_ms")),
                            _first_float(row.get("avg_path_latency_ms")),
                            _first_float(row.get("delivered")),
                            _first_float(row.get("dropped")),
                            _first_float(row.get("energy_kwh")),
                            _first_float(row.get("carbon_g")),
                            _first_float(row.get("delta_energy_kwh")),
                            _first_float(row.get("delta_delivered")),
                            _first_float(row.get("delta_dropped")),
                            _first_float(row.get("delta_carbon_g")),
                            _first_int(row.get("num_active_edges")),
                            _first_int(row.get("near_saturated_edges")),
                            _bool_to_db(row.get("qos_violation")),
                            _bool_to_db(row.get("toggle_applied")),
                            _bool_to_db(row.get("toggle_reverted")),
                            _first_int(row.get("blocked_by_util_count")),
                            _first_int(row.get("blocked_by_cooldown_count")),
                            _first_int(row.get("allowed_toggle_count")),
                            _first_int(row.get("toggles_attempted_count")),
                            _first_int(row.get("toggles_applied_count")),
                            _json_dumps(row),
                        )
                    )

                if step_params:
                    conn.executemany(
                        """
                        INSERT INTO run_step_metrics (
                            source,
                            run_name,
                            episode,
                            step,
                            episode_seed,
                            action,
                            reward,
                            terminated,
                            truncated,
                            avg_utilization,
                            active_ratio,
                            avg_delay_ms,
                            avg_path_latency_ms,
                            delivered,
                            dropped,
                            energy_kwh,
                            carbon_g,
                            delta_energy_kwh,
                            delta_delivered,
                            delta_dropped,
                            delta_carbon_g,
                            num_active_edges,
                            near_saturated_edges,
                            qos_violation,
                            toggle_applied,
                            toggle_reverted,
                            blocked_by_util_count,
                            blocked_by_cooldown_count,
                            allowed_toggle_count,
                            toggles_attempted_count,
                            toggles_applied_count,
                            payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        step_params,
                    )

        return PersistRunResult(
            source=artifacts.source,
            run_name=artifacts.run_name,
            episodes=_first_int(len(episode_rows), episodes_value) or 0,
            steps=len(artifacts.step_rows),
            has_summary=artifacts.summary is not None,
            has_steps=bool(artifacts.step_rows),
        )

    def list_run_snapshots(self, base: str = "both") -> List[Dict[str, Any]]:
        self.ensure_initialized()
        sources = _sources_for_base(base)
        placeholders = ", ".join("?" for _ in sources)
        query = f"""
            SELECT
                r.source,
                r.run_name,
                r.run_path,
                r.timestamp_utc,
                r.created_at_utc,
                r.policy,
                r.scenario,
                r.seed,
                r.topology_seed,
                r.tag,
                r.episodes,
                r.max_steps,
                r.deterministic,
                r.model_path,
                r.has_meta,
                r.has_summary,
                r.has_env_config,
                r.has_steps,
                s.reward_total_mean,
                s.energy_kwh_total_mean,
                s.dropped_total_mean,
                s.toggles_total_mean
            FROM runs r
            LEFT JOIN run_summaries s
                ON s.source = r.source AND s.run_name = r.run_name
            WHERE r.source IN ({placeholders})
        """
        with self._connect() as conn:
            rows = conn.execute(query, sources).fetchall()

        snapshots: List[Dict[str, Any]] = []
        for row in rows:
            snapshots.append(
                {
                    "run_id": str(row["run_name"]),
                    "source": str(row["source"]),
                    "run_path": str(row["run_path"]),
                    "timestamp_utc": row["timestamp_utc"] or row["created_at_utc"],
                    "policy": row["policy"],
                    "scenario": row["scenario"],
                    "seed": row["seed"],
                    "topology_seed": row["topology_seed"],
                    "tag": row["tag"],
                    "episodes": row["episodes"],
                    "max_steps": row["max_steps"],
                    "deterministic": None if row["deterministic"] is None else bool(row["deterministic"]),
                    "model_path": row["model_path"],
                    "has": {
                        "per_step": bool(row["has_steps"]),
                        "summary": bool(row["has_summary"]),
                        "meta": bool(row["has_meta"]),
                        "env_config": bool(row["has_env_config"]),
                    },
                    "highlights": {
                        "reward_total_mean": row["reward_total_mean"],
                        "energy_kwh_total_mean": row["energy_kwh_total_mean"],
                        "dropped_total_mean": row["dropped_total_mean"],
                        "toggles_total_mean": row["toggles_total_mean"],
                    },
                }
            )
        return snapshots

    def get_run_meta(self, source: str, run_name: str) -> Optional[Dict[str, Any]]:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT meta_json FROM runs WHERE source = ? AND run_name = ?",
                (source, run_name),
            ).fetchone()
        return _json_load_dict(None if row is None else row["meta_json"])

    def get_env_config(self, source: str, run_name: str) -> Optional[Dict[str, Any]]:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT env_config_json FROM runs WHERE source = ? AND run_name = ?",
                (source, run_name),
            ).fetchone()
        return _json_load_dict(None if row is None else row["env_config_json"])

    def get_run_summary(self, source: str, run_name: str) -> Optional[Dict[str, Any]]:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json FROM run_summaries WHERE source = ? AND run_name = ?",
                (source, run_name),
            ).fetchone()
        return _json_load_dict(None if row is None else row["summary_json"])

    def get_step_rows(self, source: str, run_name: str) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM run_step_metrics
                WHERE source = ? AND run_name = ?
                ORDER BY episode ASC, step ASC
                """,
                (source, run_name),
            ).fetchall()
        payloads: List[Dict[str, Any]] = []
        for row in rows:
            payload = _json_load_row(row["payload_json"])
            if payload is not None:
                payloads.append(payload)
        return payloads


def _sources_for_base(base: str) -> tuple[str, ...]:
    if base == "both":
        return ("results", "runs")
    if base in {"results", "runs"}:
        return (base,)
    raise ValueError(f"Unsupported base: {base}")


@lru_cache(maxsize=4)
def _cached_repository(db_path_str: str) -> SqliteRunRepository:
    return SqliteRunRepository(Path(db_path_str))


def get_run_repository(db_path: Path | str | None = None) -> RunRepository:
    resolved = _resolve_db_path(db_path)
    return _cached_repository(str(resolved))


def persist_run_directory(
    run_dir: Path,
    *,
    source: str | None = None,
    db_path: Path | str | None = None,
) -> Optional[PersistRunResult]:
    artifacts = discover_run_artifacts(run_dir, source=source)
    if artifacts is None:
        return None
    repository = get_run_repository(db_path)
    return repository.upsert_run_directory(artifacts)


def backfill_run_directories(
    *,
    base: str = "both",
    db_path: Path | str | None = None,
    results_dir: Path | None = None,
    runs_dir: Path | None = None,
) -> BackfillReport:
    repository = get_run_repository(db_path)
    repository.ensure_initialized()

    root = _repo_root()
    source_roots: List[tuple[str, Path]] = []
    if base in {"results", "both"}:
        source_roots.append(("results", (results_dir or (root / "results")).resolve()))
    if base in {"runs", "both"}:
        source_roots.append(("runs", (runs_dir or (root / "runs")).resolve()))

    scanned = 0
    persisted = 0
    skipped = 0
    failed = 0

    for source, source_root in source_roots:
        if not source_root.exists():
            continue
        for run_dir in sorted((path for path in source_root.iterdir() if path.is_dir()), key=lambda path: path.name):
            scanned += 1
            artifacts = discover_run_artifacts(run_dir, source=source, repo_root=root)
            if artifacts is None:
                skipped += 1
                continue
            try:
                repository.upsert_run_directory(artifacts)
            except Exception:
                failed += 1
                continue
            persisted += 1

    return BackfillReport(scanned=scanned, persisted=persisted, skipped=skipped, failed=failed)
