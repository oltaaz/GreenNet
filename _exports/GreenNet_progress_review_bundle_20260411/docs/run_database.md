# Run Database

GreenNet now persists completed run data to a lightweight SQLite database in addition to the existing CSV/JSON artifacts.

## What is stored

The database now covers the full official experiment story for ordinary runs and final evaluation bundles.

- run metadata from `run_meta.json`
- env configuration from `env_config.json` when present
- full run summary payloads from `summary.json`
- episode summary rows from `summary.json`
- per-step rows from `per_step.csv`
- final evaluation bundle payloads from the official final-pipeline path

The database keeps JSON payload copies for compatibility and also flattens the core identity and query fields used by the API and official reporting:

- matrix identity:
  `matrix_id`, `matrix_name`, `matrix_manifest`, `matrix_case_id`, `matrix_case_label`
- policy identity:
  `policy`, `policy_class`, `controller_policy`, `controller_policy_class`
- topology identity:
  `topology_seed`, `topology_name`, `topology_path`
- traffic identity:
  `traffic_seed`, `traffic_seed_base`, `traffic_mode`, `traffic_model`, `traffic_name`, `traffic_path`, `traffic_scenario`
- energy/carbon identity:
  `energy_model_name`, `energy_model_signature`, `carbon_model_name`
- QoS identity and status:
  `qos_policy_name`, `qos_policy_signature`, `qos_acceptance_status`
- stability identity and status:
  `stability_policy_name`, `stability_policy_signature`, `stability_status`
- run and step metrics:
  delivery loss, delay, QoS violation rate, power breakdowns, transition counts, flap counts, and transition/flap rates

The final evaluation table stores the latest official report payload plus the linked artifact paths, so `/api/final_evaluation` can be served from SQLite instead of scanning files first.

## Default location

- default database file: `artifacts/db/greennet.sqlite3`
- override with: `GREENNET_DB_PATH=/custom/path/greennet.sqlite3`

SQLite sidecar files such as `-wal` and `-shm` will live beside the main database file.

## Initialization

Initialization is code-first and migration-backed.

- automatic: the database is created and migrated on first successful persistence or first backend DB access
- automatic via the reviewer-facing rerun command: `python3 experiments/run_official_acceptance_matrix.py`
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

Export a DB-backed summary CSV without rescanning run folders:

```bash
python3 -m greennet.persistence export-summary --base both --output /tmp/results_summary.csv
```

## Runtime behavior

- `run_experiment.py` still writes `per_step.csv`, `summary.json`, `run_meta.json`, and `env_config.json`
- after artifact write, it upserts the completed run into SQLite
- the final pipeline still writes its CSV/JSON/Markdown bundle, then mirrors the final evaluation payload into SQLite
- if SQLite persistence fails, the artifact write still succeeds and the run can be repaired later with `backfill`
- the FastAPI backend now prefers SQLite for run listing, run metadata, summaries, per-step reads, and final evaluation retrieval; it falls back to files when a run has not been indexed yet
- the final submission evidence should still cite the curated matrix and verification bundles rather than assuming every historical run has been indexed

## Incremental migration notes

This change is intentionally incremental:

- file artifacts remain the compatibility layer and export surface
- SQLite becomes the primary structured store for official run data and final-evaluation payloads
- API integration now covers run listing, run metadata, env payloads, summaries, per-step-backed timelines, and final evaluation lookup
- the repository boundary is small so a PostgreSQL implementation can be added later behind the same interface without rewriting the API surface

## What remains intentionally file-based

The DB is the primary structured store, but some artifacts remain intentionally file-based:

- `per_step.csv`, `summary.json`, `run_meta.json`, and `env_config.json` are still written for compatibility and inspection
- official final reports still ship as CSV/JSON/Markdown files under `artifacts/final_pipeline/...`
- locked reviewer bundles under `artifacts/locked/` remain filesystem artifacts
- simulator reconstruction endpoints that need raw topology/config files may still read from the run directory when that is simpler than reconstructing from DB payloads
