# GreenNet ChatGPT Analysis Bundle

This archive is a curated subset of the repository focused on source code and run configuration.

Included:
- Core package: `greennet/`
- Dashboard code: `dashboard/`
- Experiment scripts: `experiments/` (script-focused)
- Main entry scripts: `train.py`, `run_experiment.py`, `evaluate_checkpoints.py`, `eval.py`, `api_app.py`, `baselines.py`, `resume_latest.py`
- Project metadata/docs: `pyproject.toml`, `COMMANDS.md`, `docs/runs_management.md`
- Scenario configs: `train_*.json`, `eval_*.json`

Excluded intentionally:
- Heavy runtime outputs (`results/`, `runs/`, `artifacts/`, `demo_bundle/`)
- Virtual environments (`.venv/`, `ml-env/`)
- Frontend dependencies and large lock/build trees
- Binary docs and logs
