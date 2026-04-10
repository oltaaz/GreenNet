# GreenNet Repair Pass Summary

## Fixed in this pass

- Made the final `ppo` claim honest everywhere by relabeling reviewer-facing references as the **PPO-Based Hybrid (AI)** controller and updating docs, generated reports, and UI labels.
- Unified the canonical final-evidence path around `artifacts/final_pipeline/official_acceptance_v1/` and marked `latest/` plus `official_matrix_v6/` as archival.
- Fixed `/api/final_evaluation` state leakage so DB-backed final-evaluation payloads only override repo-local artifacts when they belong to the current `REPO_ROOT`.
- Resolved the documented-vs-actual PPO artifact lineage mismatch by aligning docs with the checked-in checkpoint metadata and demoting `configs/train_official_ppo.json` to an alternate longer-run recipe.
- Cleaned the quality gates so the full Python suite, frontend lint, and frontend build all pass.
- Added a submission-facing repo path with `SUBMISSION_INDEX.md` and archival/internal labels for `_exports/`, `results/`, `tmp/official_acceptance_backup/`, `experiments/`, `final_audit/`, and `artifacts/final_pipeline/archive/`.

## Remaining work

- Tighten reviewer-facing wording around networking realism and real-world applicability.
- Clarify how project-defined QoS acceptance gates relate to rough raw delay/drop outcomes.
- Soften or better validate forecasting / impact-predictor / cost-model claims.
- Consider whether the heavy checked-in SQLite and artifact footprint should be reduced further for the final delivered package.

## Commands run

- `python3 -m py_compile greennet/policy_taxonomy.py greennet/evaluation/final_pipeline.py greennet/evaluation/final_report.py`
- `python3 -m py_compile greennet/evaluation/reproduction.py`
- `python3 -m py_compile api_app.py`
- `python3 -m py_compile api_app.py greennet/policy_taxonomy.py greennet/evaluation/final_pipeline.py greennet/evaluation/final_report.py greennet/evaluation/reproduction.py`
- `.venv/bin/python -m pytest -q tests/integration/test_official_reproduction.py`
- `.venv/bin/python -m pytest -q tests/integration/test_api_app.py -k 'final_evaluation_endpoint'`
- `.venv/bin/python -m pytest -q tests/unit/test_official_ppo.py`
- `.venv/bin/python -m pytest -q`
- `npm --prefix frontend/greennet-ui run lint`
- `npm --prefix frontend/greennet-ui run build`

## Validation results

- `tests/integration/test_official_reproduction.py`: pass (`2 passed`)
- `tests/integration/test_api_app.py -k 'final_evaluation_endpoint'`: pass (`2 passed, 3 deselected`)
- `tests/unit/test_official_ppo.py`: pass (`2 passed`)
- full `pytest -q`: pass (`78 passed`)
- frontend lint: pass
- frontend build: pass

## Blocked items

- None of the critical audit items remain blocked.
- The remaining non-critical items are judgment and framing improvements rather than broken flows.
