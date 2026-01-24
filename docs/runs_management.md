# Runs Management

Greennet training is continuous, so we keep curated runs as thesis evidence while pruning bulk artifacts. The goal is to preserve configs, logs, and robustness outputs for traceability while safely removing large model files or tiny throwaway runs.

## Generate training history
Create an evidence summary across all runs:

```bash
python experiments/summarize_runs.py
```

This writes `experiments/training_history.csv` with one row per run and prints a short summary to the terminal.

## Pruning runs safely
The pruning script is dry-run by default and prints an action plan (KEEP / STRIP / DELETE). Use `--apply` to perform changes. STRIP removes model artifacts only; DELETE removes tiny empty runs (when enabled).

Example commands:

1) Summarize:
```bash
python experiments/summarize_runs.py
```

2) Dry-run prune keeping final run:
```bash
python experiments/prune_runs.py --keep 20260123_125047 --keep-latest 5 --keep-robustness
```

3) Apply stripping models (recommended):
```bash
python experiments/prune_runs.py --keep 20260123_125047 --keep-latest 8 --keep-robustness --strip-only --apply
```

4) Apply delete empty:
```bash
python experiments/prune_runs.py --keep 20260123_125047 --keep-latest 8 --keep-robustness --strip-only --delete-empty --apply
```
