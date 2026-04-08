# Run Database

GreenNet now persists completed run data to a lightweight SQLite database in addition to the existing CSV/JSON artifacts.

## What is stored

- run metadata from `run_meta.json`
- env configuration from `env_config.json` when present
- full run summary payloads from `summary.json`
- episode summary rows from `summary.json`
- per-step rows from `per_step.csv`

The database keeps JSON payload copies for compatibility and also flattens a small set of commonly queried fields for the API.

## Default location

- default database file: `artifacts/db/greennet.sqlite3`
- override with: `GREENNET_DB_PATH=/custom/path/greennet.sqlite3`

SQLite sidecar files such as `-wal` and `-shm` will live beside the main database file.

## Initialization

Initialization is code-first and migration-backed.

- automatic: the database is created and migrated on first successful persistence or first backend DB access
- manual:

```bash
python3 -m greennet.persistence init
```

## Backfill existing artifact-only runs

If older runs already exist only as files under `results/` or `runs/`, import them into SQLite with:

```bash
python3 -m greennet.persistence backfill --base both
```

Common variants:

```bash
python3 -m greennet.persistence backfill --base results
python3 -m greennet.persistence backfill --base runs
python3 -m greennet.persistence backfill --db-path /tmp/greennet.sqlite3
```

## Runtime behavior

- `run_experiment.py` still writes `per_step.csv`, `summary.json`, `run_meta.json`, and `env_config.json`
- after artifact write, it upserts the completed run into SQLite
- if SQLite persistence fails, the artifact write still succeeds and the run can be repaired later with `backfill`
- the FastAPI backend now prefers SQLite for core run reads and falls back to files when a run has not been indexed yet

## Incremental migration notes

This change is intentionally incremental:

- file artifacts remain the compatibility layer and export surface
- SQLite becomes the primary structured store for core run data
- API integration is limited to core read paths first: run listing, run metadata, env payloads, summaries, and per-step-backed timelines
- the repository boundary is small so a PostgreSQL implementation can be added later behind the same interface without rewriting the API surface
