# GreenNet Final Fix Checklist

- [x] GN-001: Decide the final research claim
  - Priority: P0
  - Status: completed
  - What was done: Rewrote the reviewer-facing claim so it no longer suggests an AI energy win; the official story now says `heuristic` is best overall and `ppo` is the best AI policy with a mixed result.
  - Evidence: `README.md`; `experiments/official_matrix_v6/notes.md`; `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`
  - What remains: None for the final narrative.
  - Submission blocker now: no

- [x] GN-002: Package one reproducible final evidence bundle
  - Priority: P0
  - Status: completed
  - What was done: Added `artifacts/final_submission/matrix_v6/` with a canonical README, manifest, and traceability map for the final submission evidence.
  - Evidence: `artifacts/final_submission/matrix_v6/README.md`; `artifacts/final_submission/matrix_v6/manifest.json`; `artifacts/final_submission/matrix_v6/traceability.csv`
  - What remains: The bundle still references preserved artifacts instead of duplicating every historical raw file, which is intentional.
  - Submission blocker now: no

- [x] GN-003: Unify setup and run instructions
  - Priority: P1
  - Status: completed
  - What was done: Aligned the README, legacy command index, and submission overview around one install/train/evaluate/demo path.
  - Evidence: `README.md`; `COMMANDS.md`; `docs/final_submission_overview.md`
  - What remains: None for the official workflow docs.
  - Submission blocker now: no

- [x] GN-004: Define a complete reproducible environment
  - Priority: P0
  - Status: completed
  - What was done: Added tracked optional dependency groups for test and training dependencies, pinned the supported Python range in `pyproject.toml`, and aligned the docs with the verified `python3.12` environment path.
  - Evidence: `pyproject.toml`; `README.md`; `docs/final_submission_overview.md`; `final_audit/verification/README.md`
  - What remains: None for the documented submission environment.
  - Submission blocker now: no

- [x] GN-005: Make basic verification actually pass
  - Priority: P0
  - Status: completed
  - What was done: Verified clean editable install in a fresh `python3.12` venv, backend import, FastAPI `TestClient` health smoke, `run_experiment.py`, `final_evaluation.py`, `pytest`, and the frontend `npm ci` plus `npm run build` path.
  - Evidence: `final_audit/verification/README.md`; `final_audit/verification/final_evaluation_smoke_recheck/final_evaluation_report.md`; `results/20260409_164637__policy-heuristic__scenario-normal__seed-0__tag-verify_smoke_recheck/`; `frontend/greennet-ui/src/pages/SimulatorPage.tsx`
  - What remains: None for the machine-verified submission checklist.
  - Submission blocker now: no

- [x] GN-006: Resolve the Impact Predictor scope mismatch
  - Priority: P1
  - Status: completed
  - What was done: Downgraded Impact Predictor to exploratory scope unless it is explicitly bundled and verified.
  - Evidence: `README.md`; `docs/final_submission_overview.md`
  - What remains: If it is later promoted into the final story, it still needs a real acceptance bundle.
  - Submission blocker now: no

- [x] GN-007: Explain the real controller architecture honestly
  - Priority: P1
  - Status: completed
  - What was done: Documented the simulator, explicit baseline controllers, PPO controller, API layer, and demo/internal tooling split in reviewer-facing docs.
  - Evidence: `README.md`; `docs/final_submission_overview.md`
  - What remains: None in the final docs.
  - Submission blocker now: no

- [x] GN-008: Reconcile locked validation evidence with the final thesis result
  - Priority: P0
  - Status: completed
  - What was done: Aligned the final thesis narrative with the actual official matrix result and tied that narrative back to locked validation artifacts through the final bundle traceability map.
  - Evidence: `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`; `experiments/official_matrix_v6/final_evaluation/final_evaluation_summary.json`; `artifacts/final_submission/matrix_v6/traceability.csv`; `artifacts/traffic_verify/20260220_matrix/matrix_status.md`
  - What remains: None beyond continuing to cite the preserved official report as the authoritative shipped artifact.
  - Submission blocker now: no

- [x] GN-009: Choose one canonical config family
  - Priority: P1
  - Status: completed
  - What was done: Marked `configs/train_*.json` as canonical and moved reviewer guidance away from root-level historical config snapshots.
  - Evidence: `configs/README.md`; `docs/custom_inputs.md`; `README.md`
  - What remains: None for the official config story.
  - Submission blocker now: no

- [x] GN-010: Mark legacy entrypoints and internal tools explicitly
  - Priority: P1
  - Status: completed
  - What was done: Reframed `COMMANDS.md` as a legacy index and labeled the Streamlit dashboard as internal analyst tooling rather than the public demo path.
  - Evidence: `COMMANDS.md`; `README.md`; `docs/runs_management.md`
  - What remains: None for the reviewer-facing workflow.
  - Submission blocker now: no

- [x] GN-011: Fix demo fallback integrity
  - Priority: P1
  - Status: completed
  - What was done: The frontend now tracks whether run data came from the backend or demo fallback and displays explicit notices when it is showing generated data or a generated topology layout.
  - Evidence: `frontend/greennet-ui/src/lib/api.ts`; `frontend/greennet-ui/src/pages/ComparePage.tsx`; `frontend/greennet-ui/src/pages/DashboardPage.tsx`; `frontend/greennet-ui/src/pages/SimulatorPage.tsx`
  - What remains: None in the verified build path from this machine recheck.
  - Submission blocker now: no

- [x] GN-012: Fix policy alias consistency across the frontend demo path
  - Priority: P1
  - Status: completed
  - What was done: Demo normalization now uses the same alias map as the rest of the frontend, removing the previous `baseline`/`noop` mismatch.
  - Evidence: `frontend/greennet-ui/src/lib/demo.ts`; `frontend/greennet-ui/src/lib/data.ts`
  - What remains: None in the code path or build verification path.
  - Submission blocker now: no

- [x] GN-013: Clean reviewer-facing artifacts of machine-specific paths
  - Priority: P1
  - Status: completed
  - What was done: Cleaned the official final-evaluation artifacts and reporting code so reviewer-facing outputs use repo-relative paths instead of personal machine paths.
  - Evidence: `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`; `experiments/official_matrix_v6/final_evaluation/final_evaluation_summary.json`; `greennet/evaluation/final_report.py`
  - What remains: None in the cleaned reviewer-facing artifacts from this pass.
  - Submission blocker now: no

- [x] GN-014: Add a concise final architecture and artifact glossary
  - Priority: P1
  - Status: completed
  - What was done: Added a compact final submission overview and artifact glossary, and reflected the same architecture summary in the README.
  - Evidence: `README.md`; `docs/final_submission_overview.md`
  - What remains: None for the submission docs.
  - Submission blocker now: no

- [x] GN-015: Add a limitations and responsible-design section
  - Priority: P1
  - Status: completed
  - What was done: Added explicit responsible-design and limitations language around QoS gating, mixed results, and experimental scope.
  - Evidence: `README.md`; `docs/final_submission_overview.md`
  - What remains: None in the final docs.
  - Submission blocker now: no
