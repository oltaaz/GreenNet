# GreenNet Commands

This file is a legacy command index. The official workflow is documented in [README.md](README.md) and [docs/final_submission_overview.md](docs/final_submission_overview.md).

## Official Commands

### Install

```bash
python3 -m pip install -e .[test,train]
```

### Train

```bash
python3 train.py --config configs/train_normal.json --timesteps 300000
python3 train.py --config configs/train_burst.json --timesteps 300000
python3 train.py --config configs/train_hotspot.json --timesteps 300000
python3 train.py --config configs/train_official_ppo.json --timesteps 100000
```

Regenerate the canonical official PPO family for the current env/topology definitions:

```bash
python3 experiments/regenerate_official_ppo_checkpoint.py --all-topologies --config configs/train_official_ppo.json --timesteps 100000
```

This writes the official reviewer-facing PPO checkpoints to:

- `artifacts/models/official_acceptance_v1/small/ppo_greennet.zip`
- `artifacts/models/official_acceptance_v1/medium/ppo_greennet.zip`
- `artifacts/models/official_acceptance_v1/large/ppo_greennet.zip`

### Run A Single Experiment

```bash
python3 run_experiment.py --policy heuristic --scenario normal --seed 0 --episodes 1 --steps 300
python3 run_experiment.py --policy ppo --scenario hotspot --seed 0 --episodes 1 --steps 300
python3 run_experiment.py --policy all_on --scenario normal --seed 0 --episodes 1 --steps 300 --topology-name medium
python3 run_experiment.py --policy all_on --scenario flash_crowd --seed 0 --episodes 1 --steps 300 --topology-name large
```

### Reproduce The Official Final Claim

```bash
python3 experiments/run_official_acceptance_matrix.py
```

This is now the canonical one-command reviewer path. It always uses `configs/acceptance_matrices/official_acceptance_v1.json`, initializes the SQLite run store, executes the official matrix, aggregates results, writes the final evaluation bundle, and mirrors indexed runs plus the final-evaluation payload into `artifacts/db/greennet.sqlite3`.

Important note:

- the reviewer-facing bundle currently pinned in `artifacts/final_pipeline/latest/` is a preserved historical `~1.49%` PPO bundle
- it is not regenerated exactly by the current code/checkpoint state
- use the rerun command for the current live reproducible benchmark path, and treat `latest` as the pinned historical review bundle

For PPO, this command now resolves the canonical topology-specific checkpoint family automatically. The older single checkpoint under `runs/20260220_111755/ppo_greennet.zip` is not compatible with the current observation space and should not be used as the official artifact.

Main outputs:

- `artifacts/final_pipeline/official_acceptance_v1/report/reviewer_start_here.md`
- `artifacts/final_pipeline/official_acceptance_v1/report/concise_report.md`
- `artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_report.md`
- `artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv`

Prerequisite-only check:

```bash
python3 experiments/run_official_acceptance_matrix.py --check-only
```

Use `python3 experiments/run_matrix.py --matrix-manifest <path.json>` for non-official benchmark variants.
Run summaries now also preserve `matrix_id`, `matrix_name`, `matrix_manifest`, `matrix_case_id`, and `matrix_case_label` so official benchmark identity survives aggregation and final reporting.

### Build The Final Evidence Bundle

```bash
python3 experiments/run_official_acceptance_matrix.py \
  --skip-eval \
  --summary-csv results/results_summary_official_acceptance_v1.csv
```

Optional QoS acceptance overrides for the final report remain available here:

- `--max-delivered-loss-pct`
- `--max-dropped-increase-pct`
- `--max-delay-increase-pct`
- `--max-path-latency-increase-pct`
- `--max-qos-violation-rate-increase`

### Run Database Helpers

```bash
python3 -m greennet.persistence init
python3 -m greennet.persistence backfill --base both
python3 -m greennet.persistence export-summary --base both --output /tmp/results_summary.csv
```

### Start The API And Frontend

```bash
npm run dev
```

This is the official local demo command from the repo root. It uses a small root `package.json` wrapper to run the existing backend and frontend dev commands together. If the repo-local `.venv` exists, the backend launcher will use it automatically.

Fallback direct commands:

```bash
uvicorn api_app:app --reload --port 8000
npm --prefix frontend run dev
```

### Start The Internal Dashboard

```bash
streamlit run dashboard/app.py
```

## Legacy And Deprecated Entries

The following patterns still appear in older notes and should not be treated as the official path for the final submission:

- `requirements.txt` based setup
- machine-specific virtualenv shell paths
- `results/` as the canonical final evidence store
- root-level historical configs such as `train_normal_v2.json`
- command snippets copied from older machine-specific notes

If you need a one-line answer for the current project state, use the official workflow in the README instead of this file.
