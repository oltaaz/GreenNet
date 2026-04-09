# GreenNet Verification Notes

This folder captures the lightweight verification steps executed during the post-fix pass on 2026-04-09.

## Commands that passed

- `python -c "import api_app; print('api_import_ok')"`
  - Result: FastAPI application imports successfully with the system Python in this environment.
- `python run_experiment.py --policy heuristic --scenario normal --seed 0 --episodes 1 --steps 5 --tag verify_smoke`
  - Result: completed and wrote `results/20260409_122527__policy-heuristic__scenario-normal__seed-0__tag-verify_smoke/`.
- `python experiments/final_evaluation.py --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv --primary-baseline-policy heuristic --ai-policies ppo --output-dir final_audit/verification/final_evaluation_smoke`
  - Result: completed after the summary-bundle fallback fix in `greennet/evaluation/final_report.py`.
  - Note: when raw historical run folders are absent, the regenerated report preserves the aggregate energy comparison but marks per-step-derived QoS/path-latency fields as `insufficient_data`.

## Commands blocked by this environment

- `python -m pytest -q`
  - Blocked because `pytest` is not installed in the active system Python used here.
- `python -c "from fastapi.testclient import TestClient; ..."`
  - Blocked because `httpx` is not installed in the active system Python used here.
- `npm run build`
  - Blocked because `node.exe` is not available in PATH on this machine.

## Interpretation

The core Python execution path is in better shape than the original audit indicated:

- backend import works
- experiment execution works
- final evaluation can now rebuild from the packaged matrix CSV instead of requiring missing historical run folders

The remaining verification gap is full environment/toolchain parity, not a newly discovered overclaim in the repository narrative.
