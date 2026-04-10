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
    (
        "002_experiment_story_expansion",
        """
        ALTER TABLE runs ADD COLUMN matrix_id TEXT;
        ALTER TABLE runs ADD COLUMN matrix_name TEXT;
        ALTER TABLE runs ADD COLUMN matrix_manifest TEXT;
        ALTER TABLE runs ADD COLUMN matrix_case_id TEXT;
        ALTER TABLE runs ADD COLUMN matrix_case_label TEXT;
        ALTER TABLE runs ADD COLUMN policy_class TEXT;
        ALTER TABLE runs ADD COLUMN controller_policy TEXT;
        ALTER TABLE runs ADD COLUMN controller_policy_class TEXT;
        ALTER TABLE runs ADD COLUMN topology_name TEXT;
        ALTER TABLE runs ADD COLUMN topology_path TEXT;
        ALTER TABLE runs ADD COLUMN traffic_mode TEXT;
        ALTER TABLE runs ADD COLUMN traffic_model TEXT;
        ALTER TABLE runs ADD COLUMN traffic_name TEXT;
        ALTER TABLE runs ADD COLUMN traffic_path TEXT;
        ALTER TABLE runs ADD COLUMN energy_model_name TEXT;
        ALTER TABLE runs ADD COLUMN energy_model_signature TEXT;
        ALTER TABLE runs ADD COLUMN carbon_model_name TEXT;
        ALTER TABLE runs ADD COLUMN qos_policy_name TEXT;
        ALTER TABLE runs ADD COLUMN qos_policy_signature TEXT;
        ALTER TABLE runs ADD COLUMN stability_policy_name TEXT;
        ALTER TABLE runs ADD COLUMN stability_policy_signature TEXT;
        ALTER TABLE runs ADD COLUMN routing_baseline TEXT;
        ALTER TABLE runs ADD COLUMN routing_link_cost_model TEXT;

        ALTER TABLE run_summaries ADD COLUMN delivery_loss_rate_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN delivery_loss_rate_std REAL;
        ALTER TABLE run_summaries ADD COLUMN energy_steady_kwh_total_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN energy_transition_kwh_total_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN power_total_watts_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN power_fixed_watts_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN power_variable_watts_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN power_transition_watts_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN qos_violation_rate_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN qos_violation_count_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN qos_acceptance_status TEXT;
        ALTER TABLE run_summaries ADD COLUMN qos_acceptance_missing TEXT;
        ALTER TABLE run_summaries ADD COLUMN transition_count_total_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN transition_on_count_total_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN transition_off_count_total_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN transition_rate_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN flap_event_count_total_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN flap_rate_mean REAL;
        ALTER TABLE run_summaries ADD COLUMN stability_status TEXT;
        ALTER TABLE run_summaries ADD COLUMN stability_missing TEXT;

        ALTER TABLE run_episode_summaries ADD COLUMN delivery_loss_rate REAL;
        ALTER TABLE run_episode_summaries ADD COLUMN qos_violation_rate REAL;
        ALTER TABLE run_episode_summaries ADD COLUMN transition_count_total INTEGER;
        ALTER TABLE run_episode_summaries ADD COLUMN transition_on_count_total INTEGER;
        ALTER TABLE run_episode_summaries ADD COLUMN transition_off_count_total INTEGER;
        ALTER TABLE run_episode_summaries ADD COLUMN flap_event_count_total INTEGER;
        ALTER TABLE run_episode_summaries ADD COLUMN transition_rate REAL;
        ALTER TABLE run_episode_summaries ADD COLUMN flap_rate REAL;
        ALTER TABLE run_episode_summaries ADD COLUMN qos_acceptance_status TEXT;
        ALTER TABLE run_episode_summaries ADD COLUMN stability_status TEXT;

        ALTER TABLE run_step_metrics ADD COLUMN transition_count INTEGER;
        ALTER TABLE run_step_metrics ADD COLUMN flap_event INTEGER;
        ALTER TABLE run_step_metrics ADD COLUMN flap_event_count INTEGER;
        ALTER TABLE run_step_metrics ADD COLUMN stability_reversal_penalty REAL;
        ALTER TABLE run_step_metrics ADD COLUMN power_total_watts REAL;
        ALTER TABLE run_step_metrics ADD COLUMN power_fixed_watts REAL;
        ALTER TABLE run_step_metrics ADD COLUMN power_variable_watts REAL;
        ALTER TABLE run_step_metrics ADD COLUMN power_transition_watts REAL;
        ALTER TABLE run_step_metrics ADD COLUMN active_devices INTEGER;
        ALTER TABLE run_step_metrics ADD COLUMN active_links INTEGER;

        CREATE TABLE final_evaluations (
            output_dir TEXT PRIMARY KEY,
            generated_at_utc TEXT,
            matrix_id TEXT,
            matrix_name TEXT,
            matrix_manifest TEXT,
            matrix_tag TEXT,
            source_summary_csv TEXT,
            summary_path TEXT,
            report_path TEXT,
            payload_json TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE INDEX idx_runs_matrix_lookup
            ON runs (matrix_id, matrix_case_id, tag, policy, scenario, seed);
        CREATE INDEX idx_final_evaluations_lookup
            ON final_evaluations (generated_at_utc, matrix_id, matrix_tag);
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


def _sql_placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


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

    def list_summary_rows(self, base: str = "both", tag: str | None = None) -> List[Dict[str, Any]]: ...

    def upsert_final_evaluation(
        self,
        *,
        output_dir: str,
        payload: Dict[str, Any],
        summary_path: str | None,
        report_path: str | None,
        source_summary_csv: str | None,
    ) -> None: ...

    def get_latest_final_evaluation(self) -> Optional[Dict[str, Any]]: ...


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
            run_params = (
                artifacts.source,
                artifacts.run_name,
                artifacts.run_path,
                _first_text(meta.get("run_id")),
                _first_text(meta.get("matrix_id")),
                _first_text(meta.get("matrix_name")),
                _first_text(meta.get("matrix_manifest")),
                _first_text(meta.get("matrix_case_id")),
                _first_text(meta.get("matrix_case_label")),
                _first_text(meta.get("policy"), parsed.get("policy")),
                _first_text(meta.get("policy_class")),
                _first_text(meta.get("controller_policy")),
                _first_text(meta.get("controller_policy_class")),
                _first_text(meta.get("scenario"), parsed.get("scenario")),
                _first_int(meta.get("seed"), parsed.get("seed")),
                _first_int(meta.get("eval_seed"), meta.get("seed"), parsed.get("seed")),
                _first_int(meta.get("topology_seed"), parsed.get("topology_seed")),
                _first_text(meta.get("topology_name")),
                _first_text(meta.get("topology_path")),
                _first_int(meta.get("traffic_seed")),
                _first_int(meta.get("traffic_seed_base"), meta.get("traffic_seed")),
                _first_text(meta.get("traffic_mode")),
                _first_text(meta.get("traffic_model")),
                _first_text(meta.get("traffic_name")),
                _first_text(meta.get("traffic_path")),
                _first_text(meta.get("tag"), parsed.get("tag")),
                _bool_to_db(meta.get("deterministic")),
                _bool_to_db(meta.get("save_flows")),
                episodes_value,
                max_steps_value,
                _first_text(meta.get("model_path")),
                _first_text(meta.get("runs_dir")),
                _first_text(meta.get("energy_model_name")),
                _first_text(meta.get("energy_model_signature")),
                _first_text(meta.get("carbon_model_name")),
                _first_text(meta.get("qos_policy_name")),
                _first_text(meta.get("qos_policy_signature")),
                _first_text(meta.get("stability_policy_name")),
                _first_text(meta.get("stability_policy_signature")),
                _first_text(meta.get("routing_baseline")),
                _first_text(meta.get("routing_link_cost_model")),
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
            )
            conn.execute(
                f"""
                INSERT INTO runs (
                    source,
                    run_name,
                    run_path,
                    meta_run_id,
                    matrix_id,
                    matrix_name,
                    matrix_manifest,
                    matrix_case_id,
                    matrix_case_label,
                    policy,
                    policy_class,
                    controller_policy,
                    controller_policy_class,
                    scenario,
                    seed,
                    eval_seed,
                    topology_seed,
                    topology_name,
                    topology_path,
                    traffic_seed,
                    traffic_seed_base,
                    traffic_mode,
                    traffic_model,
                    traffic_name,
                    traffic_path,
                    tag,
                    deterministic,
                    save_flows,
                    episodes,
                    max_steps,
                    model_path,
                    runs_dir,
                    energy_model_name,
                    energy_model_signature,
                    carbon_model_name,
                    qos_policy_name,
                    qos_policy_signature,
                    stability_policy_name,
                    stability_policy_signature,
                    routing_baseline,
                    routing_link_cost_model,
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
                ) VALUES ({_sql_placeholders(len(run_params))})
                """,
                run_params,
            )

            if artifacts.summary is not None:
                summary_params = (
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
                    _first_float(overall.get("delivery_loss_rate_mean")),
                    _first_float(overall.get("delivery_loss_rate_std")),
                    _first_float(overall.get("energy_steady_kwh_total_mean")),
                    _first_float(overall.get("energy_transition_kwh_total_mean")),
                    _first_float(overall.get("carbon_g_total_mean")),
                    _first_float(overall.get("carbon_g_total_std")),
                    _first_float(overall.get("steps_mean")),
                    _first_float(overall.get("steps_std")),
                    _first_float(overall.get("avg_utilization_mean")),
                    _first_float(overall.get("active_ratio_mean")),
                    _first_float(overall.get("avg_delay_ms_mean")),
                    _first_float(overall.get("avg_path_latency_ms_mean")),
                    _first_float(overall.get("avg_path_latency_ms_std")),
                    _first_float(overall.get("power_total_watts_mean")),
                    _first_float(overall.get("power_fixed_watts_mean")),
                    _first_float(overall.get("power_variable_watts_mean")),
                    _first_float(overall.get("power_transition_watts_mean")),
                    _first_float(overall.get("qos_violation_rate_mean")),
                    _first_float(overall.get("qos_violation_count_mean")),
                    _first_text(overall.get("qos_acceptance_status")),
                    _first_text(overall.get("qos_acceptance_missing")),
                    _first_float(overall.get("toggles_total_mean")),
                    _first_float(overall.get("toggles_total_std")),
                    _first_float(overall.get("transition_count_total_mean")),
                    _first_float(overall.get("transition_on_count_total_mean")),
                    _first_float(overall.get("transition_off_count_total_mean")),
                    _first_float(overall.get("transition_rate_mean")),
                    _first_float(overall.get("flap_event_count_total_mean")),
                    _first_float(overall.get("flap_rate_mean")),
                    _first_text(overall.get("stability_status")),
                    _first_text(overall.get("stability_missing")),
                    _json_dumps(summary),
                    _json_dumps(overall) if isinstance(overall, dict) else None,
                    updated_at_utc,
                )
                conn.execute(
                    f"""
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
                        delivery_loss_rate_mean,
                        delivery_loss_rate_std,
                        energy_steady_kwh_total_mean,
                        energy_transition_kwh_total_mean,
                        carbon_g_total_mean,
                        carbon_g_total_std,
                        steps_mean,
                        steps_std,
                        avg_utilization_mean,
                        active_ratio_mean,
                        avg_delay_ms_mean,
                        avg_path_latency_ms_mean,
                        avg_path_latency_ms_std,
                        power_total_watts_mean,
                        power_fixed_watts_mean,
                        power_variable_watts_mean,
                        power_transition_watts_mean,
                        qos_violation_rate_mean,
                        qos_violation_count_mean,
                        qos_acceptance_status,
                        qos_acceptance_missing,
                        toggles_total_mean,
                        toggles_total_std,
                        transition_count_total_mean,
                        transition_on_count_total_mean,
                        transition_off_count_total_mean,
                        transition_rate_mean,
                        flap_event_count_total_mean,
                        flap_rate_mean,
                        stability_status,
                        stability_missing,
                        summary_json,
                        overall_json,
                        updated_at_utc
                    ) VALUES ({_sql_placeholders(len(summary_params))})
                    """,
                    summary_params,
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
                            _first_float(episode_row.get("delivery_loss_rate")),
                            _first_float(episode_row.get("qos_violation_rate")),
                            _first_int(episode_row.get("toggles_total")),
                            _first_int(episode_row.get("toggles_applied_total")),
                            _first_int(episode_row.get("toggles_reverted_total")),
                            _first_int(episode_row.get("blocked_by_util_count")),
                            _first_int(episode_row.get("blocked_by_cooldown_count")),
                            _first_int(episode_row.get("allowed_toggle_count")),
                            _first_int(episode_row.get("toggles_attempted_count")),
                            _first_int(episode_row.get("toggles_applied_count")),
                            _first_int(episode_row.get("transition_count_total")),
                            _first_int(episode_row.get("transition_on_count_total")),
                            _first_int(episode_row.get("transition_off_count_total")),
                            _first_int(episode_row.get("flap_event_count_total")),
                            _first_float(episode_row.get("transition_rate")),
                            _first_float(episode_row.get("flap_rate")),
                            _first_text(episode_row.get("qos_acceptance_status")),
                            _first_text(episode_row.get("stability_status")),
                            _json_dumps(episode_row),
                        )
                    )

                if episode_params:
                    conn.executemany(
                        f"""
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
                            delivery_loss_rate,
                            qos_violation_rate,
                            toggles_total,
                            toggles_applied_total,
                            toggles_reverted_total,
                            blocked_by_util_count,
                            blocked_by_cooldown_count,
                            allowed_toggle_count,
                            toggles_attempted_count,
                            toggles_applied_count,
                            transition_count_total,
                            transition_on_count_total,
                            transition_off_count_total,
                            flap_event_count_total,
                            transition_rate,
                            flap_rate,
                            qos_acceptance_status,
                            stability_status,
                            payload_json
                        ) VALUES ({_sql_placeholders(len(episode_params[0]))})
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
                            _first_float(row.get("power_total_watts")),
                            _first_float(row.get("power_fixed_watts")),
                            _first_float(row.get("power_variable_watts")),
                            _first_float(row.get("power_transition_watts")),
                            _first_int(row.get("active_devices")),
                            _first_int(row.get("active_links")),
                            _first_int(row.get("num_active_edges")),
                            _first_int(row.get("near_saturated_edges")),
                            _bool_to_db(row.get("qos_violation")),
                            _bool_to_db(row.get("toggle_applied")),
                            _bool_to_db(row.get("toggle_reverted")),
                            _first_int(row.get("transition_count")),
                            _bool_to_db(row.get("flap_event")),
                            _first_int(row.get("flap_event_count")),
                            _first_float(row.get("stability_reversal_penalty")),
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
                        f"""
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
                            power_total_watts,
                            power_fixed_watts,
                            power_variable_watts,
                            power_transition_watts,
                            active_devices,
                            active_links,
                            num_active_edges,
                            near_saturated_edges,
                            qos_violation,
                            toggle_applied,
                            toggle_reverted,
                            transition_count,
                            flap_event,
                            flap_event_count,
                            stability_reversal_penalty,
                            blocked_by_util_count,
                            blocked_by_cooldown_count,
                            allowed_toggle_count,
                            toggles_attempted_count,
                            toggles_applied_count,
                            payload_json
                        ) VALUES ({_sql_placeholders(len(step_params[0]))})
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
                r.matrix_id,
                r.matrix_name,
                r.matrix_manifest,
                r.matrix_case_id,
                r.matrix_case_label,
                r.policy,
                r.policy_class,
                r.controller_policy,
                r.controller_policy_class,
                r.scenario,
                r.seed,
                r.topology_seed,
                r.topology_name,
                r.topology_path,
                r.traffic_mode,
                r.traffic_model,
                r.traffic_name,
                r.traffic_path,
                r.traffic_scenario,
                r.energy_model_name,
                r.energy_model_signature,
                r.qos_policy_name,
                r.qos_policy_signature,
                r.stability_policy_name,
                r.stability_policy_signature,
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
                ,s.avg_delay_ms_mean
                ,s.qos_violation_rate_mean
                ,s.transition_rate_mean
                ,s.flap_rate_mean
                ,s.qos_acceptance_status
                ,s.qos_acceptance_missing
                ,s.stability_status
                ,s.stability_missing
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
                    "matrix_id": row["matrix_id"],
                    "matrix_name": row["matrix_name"],
                    "matrix_manifest": row["matrix_manifest"],
                    "matrix_case_id": row["matrix_case_id"],
                    "matrix_case_label": row["matrix_case_label"],
                    "policy": row["policy"],
                    "policy_class": row["policy_class"],
                    "controller_policy": row["controller_policy"],
                    "controller_policy_class": row["controller_policy_class"],
                    "scenario": row["scenario"],
                    "seed": row["seed"],
                    "topology_seed": row["topology_seed"],
                    "topology_name": row["topology_name"],
                    "topology_path": row["topology_path"],
                    "traffic_mode": row["traffic_mode"],
                    "traffic_model": row["traffic_model"],
                    "traffic_name": row["traffic_name"],
                    "traffic_path": row["traffic_path"],
                    "traffic_scenario": row["traffic_scenario"],
                    "energy_model_name": row["energy_model_name"],
                    "energy_model_signature": row["energy_model_signature"],
                    "qos_policy_name": row["qos_policy_name"],
                    "qos_policy_signature": row["qos_policy_signature"],
                    "stability_policy_name": row["stability_policy_name"],
                    "stability_policy_signature": row["stability_policy_signature"],
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
                        "avg_delay_ms_mean": row["avg_delay_ms_mean"],
                        "qos_violation_rate_mean": row["qos_violation_rate_mean"],
                        "transition_rate_mean": row["transition_rate_mean"],
                        "flap_rate_mean": row["flap_rate_mean"],
                    },
                    "qos_acceptance_status": row["qos_acceptance_status"],
                    "qos_acceptance_missing": row["qos_acceptance_missing"],
                    "stability_status": row["stability_status"],
                    "stability_missing": row["stability_missing"],
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

    def list_summary_rows(self, base: str = "both", tag: str | None = None) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        sources = _sources_for_base(base)
        placeholders = ", ".join("?" for _ in sources)
        query = f"""
            SELECT
                r.source,
                r.run_name,
                r.run_path,
                r.meta_json,
                s.summary_json
            FROM runs r
            LEFT JOIN run_summaries s
                ON s.source = r.source AND s.run_name = r.run_name
            WHERE r.source IN ({placeholders})
        """
        params: list[Any] = list(sources)
        if tag is not None:
            query += " AND r.tag = ?"
            params.append(tag)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        payloads: list[Dict[str, Any]] = []
        for row in rows:
            meta = _json_load_dict(row["meta_json"]) or {}
            summary = _json_load_dict(row["summary_json"]) or {}
            overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
            payloads.append(
                {
                    "run_id": meta.get("run_id", row["run_name"]),
                    "policy": meta.get("policy", ""),
                    "policy_class": meta.get("policy_class", ""),
                    "controller_policy": meta.get("controller_policy", ""),
                    "controller_policy_class": meta.get("controller_policy_class", ""),
                    "scenario": meta.get("scenario", ""),
                    "seed": meta.get("seed", ""),
                    "tag": meta.get("tag", ""),
                    "created_at_utc": meta.get("created_at_utc") or meta.get("timestamp_utc", ""),
                    "episodes": meta.get("episodes", ""),
                    "max_steps": meta.get("max_steps", ""),
                    "deterministic": meta.get("deterministic", ""),
                    "matrix_id": meta.get("matrix_id", ""),
                    "matrix_name": meta.get("matrix_name", ""),
                    "matrix_manifest": meta.get("matrix_manifest", ""),
                    "matrix_case_id": meta.get("matrix_case_id", ""),
                    "matrix_case_label": meta.get("matrix_case_label", ""),
                    "topology_seed": meta.get("topology_seed", ""),
                    "topology_name": meta.get("topology_name", ""),
                    "topology_path": meta.get("topology_path", ""),
                    "traffic_seed": meta.get("traffic_seed", ""),
                    "traffic_seed_base": meta.get("traffic_seed_base", ""),
                    "traffic_mode": meta.get("traffic_mode", ""),
                    "traffic_model": meta.get("traffic_model", ""),
                    "traffic_name": meta.get("traffic_name", ""),
                    "traffic_path": meta.get("traffic_path", ""),
                    "traffic_scenario": meta.get("traffic_scenario", ""),
                    "traffic_scenario_version": meta.get("traffic_scenario_version", ""),
                    "traffic_scenario_intensity": meta.get("traffic_scenario_intensity", ""),
                    "traffic_scenario_duration": meta.get("traffic_scenario_duration", ""),
                    "traffic_scenario_frequency": meta.get("traffic_scenario_frequency", ""),
                    "qos_policy_name": meta.get("qos_policy_name", ""),
                    "qos_policy_signature": meta.get("qos_policy_signature", ""),
                    "qos_target_norm_drop": meta.get("qos_target_norm_drop", ""),
                    "qos_min_volume": meta.get("qos_min_volume", ""),
                    "qos_avg_delay_guard_multiplier": meta.get("qos_avg_delay_guard_multiplier", ""),
                    "qos_avg_delay_guard_margin_ms": meta.get("qos_avg_delay_guard_margin_ms", ""),
                    "stability_policy_name": meta.get("stability_policy_name", ""),
                    "stability_policy_signature": meta.get("stability_policy_signature", ""),
                    "stability_reversal_window_steps": meta.get("stability_reversal_window_steps", ""),
                    "stability_reversal_penalty": meta.get("stability_reversal_penalty", ""),
                    "stability_max_transition_rate": meta.get("stability_max_transition_rate", ""),
                    "stability_max_flap_rate": meta.get("stability_max_flap_rate", ""),
                    "stability_max_flap_count": meta.get("stability_max_flap_count", ""),
                    "energy_model_name": meta.get("energy_model_name", ""),
                    "energy_model_signature": meta.get("energy_model_signature", ""),
                    "power_utilization_sensitive": meta.get("power_utilization_sensitive", ""),
                    "power_transition_on_joules": meta.get("power_transition_on_joules", ""),
                    "power_transition_off_joules": meta.get("power_transition_off_joules", ""),
                    "carbon_model_name": meta.get("carbon_model_name", ""),
                    "routing_baseline": meta.get("routing_baseline", ""),
                    "routing_link_cost_model": meta.get("routing_link_cost_model", ""),
                    "reward_total_mean": overall.get("reward_total_mean", ""),
                    "delivered_total_mean": overall.get("delivered_total_mean", ""),
                    "dropped_total_mean": overall.get("dropped_total_mean", ""),
                    "energy_kwh_total_mean": overall.get("energy_kwh_total_mean", ""),
                    "energy_steady_kwh_total_mean": overall.get("energy_steady_kwh_total_mean", ""),
                    "energy_transition_kwh_total_mean": overall.get("energy_transition_kwh_total_mean", ""),
                    "carbon_g_total_mean": overall.get("carbon_g_total_mean", ""),
                    "power_total_watts_mean": overall.get("power_total_watts_mean", ""),
                    "power_fixed_watts_mean": overall.get("power_fixed_watts_mean", ""),
                    "power_variable_watts_mean": overall.get("power_variable_watts_mean", ""),
                    "power_transition_watts_mean": overall.get("power_transition_watts_mean", ""),
                    "active_devices_mean": overall.get("active_devices_mean", ""),
                    "active_links_mean": overall.get("active_links_mean", ""),
                    "avg_utilization_mean": overall.get("avg_utilization_mean", ""),
                    "active_ratio_mean": overall.get("active_ratio_mean", ""),
                    "delivery_loss_rate_mean": overall.get("delivery_loss_rate_mean", ""),
                    "avg_delay_ms_mean": overall.get("avg_delay_ms_mean", ""),
                    "avg_path_latency_ms_mean": overall.get("avg_path_latency_ms_mean", ""),
                    "qos_violation_rate_mean": overall.get("qos_violation_rate_mean", ""),
                    "qos_violation_count_mean": overall.get("qos_violation_count_mean", ""),
                    "qos_acceptance_status": overall.get("qos_acceptance_status", ""),
                    "qos_acceptance_missing": overall.get("qos_acceptance_missing", ""),
                    "transition_count_total_mean": overall.get("transition_count_total_mean", ""),
                    "transition_on_count_total_mean": overall.get("transition_on_count_total_mean", ""),
                    "transition_off_count_total_mean": overall.get("transition_off_count_total_mean", ""),
                    "transition_rate_mean": overall.get("transition_rate_mean", ""),
                    "flap_event_count_total_mean": overall.get("flap_event_count_total_mean", ""),
                    "flap_rate_mean": overall.get("flap_rate_mean", ""),
                    "stability_status": overall.get("stability_status", ""),
                    "stability_missing": overall.get("stability_missing", ""),
                    "results_dir": row["run_path"],
                    "status": "ok" if summary else "partial",
                    "error": "" if summary else "missing summary_json",
                }
            )
        return payloads

    def upsert_final_evaluation(
        self,
        *,
        output_dir: str,
        payload: Dict[str, Any],
        summary_path: str | None,
        report_path: str | None,
        source_summary_csv: str | None,
    ) -> None:
        self.ensure_initialized()
        source_meta = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        with self._connect() as conn, conn:
            self._apply_migrations(conn)
            conn.execute(
                """
                INSERT INTO final_evaluations (
                    output_dir,
                    generated_at_utc,
                    matrix_id,
                    matrix_name,
                    matrix_manifest,
                    matrix_tag,
                    source_summary_csv,
                    summary_path,
                    report_path,
                    payload_json,
                    updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(output_dir) DO UPDATE SET
                    generated_at_utc=excluded.generated_at_utc,
                    matrix_id=excluded.matrix_id,
                    matrix_name=excluded.matrix_name,
                    matrix_manifest=excluded.matrix_manifest,
                    matrix_tag=excluded.matrix_tag,
                    source_summary_csv=excluded.source_summary_csv,
                    summary_path=excluded.summary_path,
                    report_path=excluded.report_path,
                    payload_json=excluded.payload_json,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    output_dir,
                    _first_text(payload.get("generated_at_utc")),
                    _first_text(source_meta.get("matrix_id")),
                    _first_text(source_meta.get("matrix_name")),
                    _first_text(source_meta.get("matrix_manifest")),
                    _first_text(source_meta.get("matrix_tag")),
                    source_summary_csv,
                    summary_path,
                    report_path,
                    _json_dumps(payload),
                    _now_utc_iso(),
                ),
            )

    def get_latest_final_evaluation(self) -> Optional[Dict[str, Any]]:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT output_dir, summary_path, report_path, payload_json
                FROM final_evaluations
                ORDER BY COALESCE(generated_at_utc, updated_at_utc) DESC, updated_at_utc DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        payload = _json_load_dict(row["payload_json"])
        if payload is None:
            return None
        payload["artifact"] = {
            "summary_path": row["summary_path"],
            "report_path": row["report_path"],
            "output_dir": row["output_dir"],
        }
        return payload


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


def persist_final_evaluation_bundle(
    *,
    output_dir: Path,
    payload: Dict[str, Any],
    summary_path: Path | None = None,
    report_path: Path | None = None,
    source_summary_csv: Path | None = None,
    db_path: Path | str | None = None,
) -> None:
    repository = get_run_repository(db_path)
    repository.upsert_final_evaluation(
        output_dir=str(output_dir),
        payload=payload,
        summary_path=None if summary_path is None else str(summary_path),
        report_path=None if report_path is None else str(report_path),
        source_summary_csv=None if source_summary_csv is None else str(source_summary_csv),
    )
