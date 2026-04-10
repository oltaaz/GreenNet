# GreenNet Verification Notes

This folder captures verification runs executed on 2026-04-09 after rechecking the earlier audit claims on this same machine.

## Environment used for the definitive proof run

- Fresh virtual environment: `.venv-verify`
- Interpreter: `python3.12`
- Install command: `python3.12 -m venv .venv-verify && .venv-verify/bin/python -m pip install -e '.[test,train]'`

## Commands that passed

- `.venv-verify/bin/python -c "import api_app; print('api_import_ok')"`
  - Result: backend imports successfully from the clean editable install.
- `.venv-verify/bin/python -c "from fastapi.testclient import TestClient; import api_app; client = TestClient(api_app.app); resp = client.get('/api/health'); print(resp.status_code); print(resp.json())"`
  - Result: API test client succeeds with `200` and `{"status": "ok"}`.
- `.venv-verify/bin/python run_experiment.py --policy heuristic --scenario normal --seed 0 --episodes 1 --steps 5 --tag verify_smoke_recheck`
  - Result: completed and wrote `results/20260409_164637__policy-heuristic__scenario-normal__seed-0__tag-verify_smoke_recheck/`.
- `.venv-verify/bin/python experiments/final_evaluation.py --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv --primary-baseline-policy heuristic --ai-policies ppo --output-dir final_audit/verification/final_evaluation_smoke_recheck`
  - Result: completed and regenerated the report bundle under `final_audit/verification/final_evaluation_smoke_recheck/`.
  - Note: on this machine the curated summary CSV resolves to present `results/...__tag-matrix_v6` folders, so the regenerated output includes populated QoS and path-latency fields rather than `insufficient_data`.
- `.venv-verify/bin/python -m pytest -q`
  - Result: `37 passed in 9.70s`.
- `npm ci`
  - Result: frontend dependencies installed successfully from `frontend/greennet-ui/package-lock.json`.
- `npm run build`
  - Result: production build succeeds after removing dead code in `frontend/greennet-ui/src/pages/SimulatorPage.tsx`.

## Interpretation

The earlier "remaining issues" list does not hold on this machine after a clean recheck:

- clean Python install with `test` and `train` extras is proven
- backend smoke checks and full pytest suite are proven
- frontend install and production build are proven
- final evaluation regeneration matches the shipped official report structure and does not degrade to summary-only fields here

The only notable environment nuance is interpreter selection: the system default Python here is `3.14`, but the verified clean toolchain run used `3.12`. The repository now documents that path explicitly and pins `requires-python = ">=3.10,<3.14"` so unsupported interpreter choices fail early.
