# GreenNet Improvement Pass Changelog

## Files Created

- `artifacts/final_submission/matrix_v6/README.md`
  - Rationale: defines the canonical reviewer-facing evidence bundle and states the honest final claim.
- `artifacts/final_submission/matrix_v6/manifest.json`
  - Rationale: machine-readable manifest for the preserved final evidence package.
- `artifacts/final_submission/matrix_v6/traceability.csv`
  - Rationale: maps each reviewer-facing claim to concrete supporting artifacts.
- `docs/final_submission_overview.md`
  - Rationale: compact architecture, artifact glossary, official workflow, and limitations note for final submission review.
- `final_audit/IMPLEMENTATION_CHANGELOG.md`
  - Rationale: records the improvement pass in one place.
- `final_audit/FINAL_REAUDIT_SUMMARY.md`
  - Rationale: short post-fix verdict and remaining-risk note.
- `final_audit/verification/README.md`
  - Rationale: records the verification commands actually executed in this pass.
- `final_audit/verification/final_evaluation_smoke/final_evaluation_report.md`
  - Rationale: smoke output proving `experiments/final_evaluation.py` now runs from the packaged matrix CSV.
- `final_audit/verification/final_evaluation_smoke/final_evaluation_summary.csv`
  - Rationale: smoke output from the regenerated final-evaluation command.
- `final_audit/verification/final_evaluation_smoke/final_evaluation_summary.json`
  - Rationale: machine-readable smoke output from the regenerated final-evaluation command.
- `results/20260409_122527__policy-heuristic__scenario-normal__seed-0__tag-verify_smoke/`
  - Rationale: real experiment smoke artifact proving the core runner still executes.

## Files Modified

- `README.md`
  - Rationale: replaced overclaiming and drift with one official workflow, an honest final claim, a compact architecture summary, an artifact glossary, and limitations guidance.
- `COMMANDS.md`
  - Rationale: converted from an inconsistent command dump into a legacy index that points reviewers to the official path.
- `pyproject.toml`
  - Rationale: added tracked optional dependency groups for `test` and `train`.
- `configs/README.md`
  - Rationale: marked `configs/train_*.json` as the canonical config family.
- `docs/custom_inputs.md`
  - Rationale: aligned terminology and examples with the canonical config/policy path.
- `docs/runs_management.md`
  - Rationale: made the curated submission evidence path explicit and de-emphasized ad hoc run folders.
- `docs/run_database.md`
  - Rationale: aligned persistence docs with the curated evidence story.
- `experiments/official_matrix_v6/notes.md`
  - Rationale: documented the preserved matrix caveats and linked the canonical final bundle.
- `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`
  - Rationale: removed machine-specific paths and added an explicit mixed-result interpretation.
- `experiments/official_matrix_v6/final_evaluation/final_evaluation_summary.json`
  - Rationale: removed machine-specific paths and linked the canonical bundle.
- `greennet/evaluation/final_report.py`
  - Rationale: fixed `final_evaluation.py` so it can regenerate from the packaged matrix CSV when historical run folders are absent, and normalized reviewer-facing source paths.
- `frontend/greennet-ui/src/lib/api.ts`
  - Rationale: added explicit run-catalog source reporting so the UI can distinguish backend data from demo fallback.
- `frontend/greennet-ui/src/lib/demo.ts`
  - Rationale: removed the inconsistent local alias remap and labeled demo runs explicitly.
- `frontend/greennet-ui/src/pages/ComparePage.tsx`
  - Rationale: surfaces demo-data mode explicitly when live run catalog data is unavailable.
- `frontend/greennet-ui/src/pages/DashboardPage.tsx`
  - Rationale: surfaces demo-data and generated-topology fallback states explicitly.
- `frontend/greennet-ui/src/pages/SimulatorPage.tsx`
  - Rationale: surfaces demo-data and generated-topology fallback states explicitly.
- `final_audit/REPORT_FINAL_AUDIT.md`
  - Rationale: replaced the pre-fix audit with the post-fix re-audit.
- `final_audit/CHECKLIST_FINAL_FIXES.md`
  - Rationale: updated all GN-001..GN-015 statuses after implementation and verification.
- `final_audit/CHECKLIST_FINAL_FIXES.json`
  - Rationale: machine-readable status update for all checklist items.

## Files Archived Or Moved

- None in this pass.
- Historical files were intentionally retained and relabeled rather than moved or deleted.

## Intentionally Deferred Items

- Fresh-environment verification with `python -m pip install -e .[test,train]`
  - Deferred because this machine does not currently provide the full verified toolchain state.
- `pytest`-based verification and FastAPI `TestClient` smoke
  - Deferred because the active interpreter here does not have `pytest` and `httpx` installed.
- Frontend build verification
  - Deferred because `node.exe` is not available in PATH on this machine.
- Restoring missing historical run folders/checkpoints referenced by older matrix notes
  - Deferred because the improvement pass favored honest packaging and reproducible manifests over inventing or reconstructing missing artifacts.
