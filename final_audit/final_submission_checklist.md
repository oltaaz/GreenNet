# Final Submission Checklist

## Critical before submission

- [x] Make the final `ppo` claim honest everywhere.  
What is wrong: the evaluated `ppo` path is a PPO-based hybrid controller with rule-based override logic.  
Why it matters: this is the biggest claim-honesty risk.  
Files/modules: `run_experiment.py`, `README.md`, `docs/final_submission_overview.md`, final report outputs.  
Done: reviewer-facing language now matches the actual evaluated controller while preserving the stable `ppo` policy id for artifact compatibility.

- [x] Unify the canonical final-evidence path.  
What is wrong: `artifacts/final_pipeline/latest/`, `artifacts/final_pipeline/official_acceptance_v1/`, and `experiments/official_matrix_v6/` compete.  
Why it matters: reviewers need one answer.  
Files/modules: `README.md`, `docs/final_submission_overview.md`, `artifacts/final_pipeline/latest/README.md`, final-pipeline directories.  
Done: one clearly primary bundle now exists in both code defaults and docs at `artifacts/final_pipeline/official_acceptance_v1/`, while `latest/` and `official_matrix_v6/` are explicitly archival.

- [x] Fix `/api/final_evaluation` state leakage and make the failing integration test pass.  
What is wrong: API behavior is influenced by ambient SQLite state.  
Why it matters: reproducibility and trust.  
Files/modules: `api_app.py`, `greennet/persistence/sqlite_store.py`, `tests/integration/test_api_app.py`.  
Done: the API now only prefers DB-backed final-evaluation payloads when their artifact paths belong to the current `REPO_ROOT`, and the targeted integration coverage passes.

- [x] Resolve the documented-vs-actual PPO artifact mismatch.  
What is wrong: docs say one training recipe, checkpoint metadata says another.  
Why it matters: direct reproducibility gap.  
Files/modules: `README.md`, `configs/README.md`, `configs/train_official_ppo.json`, `artifacts/models/official_acceptance_v1/*/checkpoint_metadata.json`.  
Done: reviewer docs now match the checked-in checkpoint metadata, and `train_official_ppo.json` is explicitly labeled as an alternate longer-run recipe rather than the source of the current canonical family.

- [x] Get all quality gates clean.  
What is wrong: `pytest` was failing and frontend lint failed.  
Why it matters: final submission polish.  
Files/modules: `tests/integration/test_api_app.py`, `frontend/greennet-ui/src/hooks/useBackendStatus.tsx`, `frontend/greennet-ui/src/lib/api.ts`, `frontend/greennet-ui/src/pages/DashboardPage.tsx`, `frontend/greennet-ui/src/components/TopologyPanel.tsx`.  
Done: completed in this repair pass. `pytest -q`, `npm --prefix frontend/greennet-ui run lint`, and `npm --prefix frontend/greennet-ui run build` all pass.

- [x] Produce a submission-clean repo/package.  
What is wrong: duplicate exports, huge artifacts, backups, caches, and legacy files clutter the repo.  
Why it matters: reviewer experience.  
Files/modules: `_exports/`, `tmp/official_acceptance_backup/`, `experiments/official_matrix_v*`, `results/`, generated caches, root-level legacy configs.  
Done: completed in this repair pass. `SUBMISSION_INDEX.md` now gives one explicit reviewer path, and `_exports/`, `results/`, `tmp/official_acceptance_backup/`, `experiments/`, `final_audit/`, and `artifacts/final_pipeline/archive/` are now labeled in place as archival/internal rather than competing with the canonical submission bundle.

## Important but not blocking

- [ ] Tighten wording about simulator realism.  
What is wrong: some language can be overread as realistic routing/protocol simulation.  
Why it matters: oral defense risk.  
Files/modules: docs, report, poster/presentation materials.  
Done: materials consistently describe a flow-level abstract simulator.

- [ ] Clarify what “acceptable QoS” means relative to raw metrics.  
What is wrong: project-defined gates can look lenient without context.  
Why it matters: reviewers may challenge the acceptance logic.  
Files/modules: `greennet/qos.py`, final reports, thesis text.  
Done: clear explanation of internal gates vs absolute raw QoS severity.

- [ ] Reframe forecasting and impact prediction as supporting/exploratory unless stronger evidence is added.  
What is wrong: they are real but weakly validated.  
Why it matters: prevents overclaiming.  
Files/modules: `greennet/forecasting.py`, `greennet/impact_predictor.py`, model artifacts, docs/report.  
Done: either stronger evidence is added or claims are softened.

- [ ] Make frontend fallback states unmistakable.  
What is wrong: demo/generated runs and generated topology/layout can still be mistaken for canonical evidence.  
Why it matters: presentation trust.  
Files/modules: `frontend/greennet-ui/src/lib/api.ts`, `frontend/greennet-ui/src/pages/DashboardPage.tsx`, `frontend/greennet-ui/src/pages/SimulatorPage.tsx`.  
Done: fallback mode is impossible to miss.

- [ ] Curate the SQLite submission story.  
What is wrong: the DB is large and carries ambient historical state.  
Why it matters: portability and trust.  
Files/modules: `artifacts/db/greennet.sqlite3`, persistence docs/scripts.  
Done: curated DB or a clear policy not to rely on ambient DB state for reviewer-facing results.

## Nice-to-have polish

- [ ] Remove root-level legacy config snapshots from the submission copy.  
What is wrong: they dilute the canonical `configs/` path.  
Why it matters: cleaner navigation.  
Files/modules: repo root.  
Done: `configs/` is the only config family reviewers need.

- [ ] Remove machine-local/generated noise.  
What is wrong: `__pycache__`, `.DS_Store`, env dirs, and other local artifacts remain visible.  
Why it matters: professionalism.  
Files/modules: source tree broadly.  
Done: source tree looks deliberate and clean.

- [x] Add a root-level “reviewer quick path” note.  
What is wrong: the canonical story was spread across several docs.  
Why it matters: faster board onboarding.  
Files/modules: repo root docs.  
Done: completed in this repair pass as `SUBMISSION_INDEX.md`.
