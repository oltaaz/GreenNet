# Project Structure

GreenNet is organized around a small set of canonical top-level areas:

- `greennet/`: Python package with the simulator, environment, baselines, evaluation, persistence, RL helpers, and shared utilities.
- `scripts/`: operational helper scripts that are not part of the importable library surface.
- `configs/`: canonical tracked configs for training and acceptance matrices.
- `configs/archive/root_legacy/`: archived historical root-level config snapshots kept for traceability, not as the recommended workflow.
- `experiments/`: reproducible matrix and reporting pipelines.
- `frontend/`: official React/Vite UI shell and source app.
- `dashboard/`: internal Streamlit tooling.
- `tests/`: unit and integration coverage.
- `artifacts/`, `results/`, `runs/`, `output/`, `tmp/`: generated outputs and working data.

Root-level compatibility entrypoints remain intentionally thin:

- `train.py`
- `run_experiment.py`
- `api_app.py`
- `eval.py`
- `evaluate_checkpoints.py`
- `resume_latest.py`

These wrappers keep existing commands working while the implementation lives in package or script directories.
