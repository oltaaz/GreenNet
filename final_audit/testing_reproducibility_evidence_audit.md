# GreenNet Testing, Reproducibility, and Evidence Quality Audit

## 1. Executive Verdict

- Readiness score: **5.8/10**
- Submission recommendation: **Not ready yet**

GreenNet has real testing and a real final-evaluation pipeline, but its reviewer story is still unstable. The strongest problems are provenance and canonical-path ambiguity: the code-default reproduction target conflicts with the docs, the pinned `latest` bundle is explicitly historical rather than reproducible, the API can prefer ambient DB state over local artifact state, and the stored verification note is now contradicted by a current failing integration test.

## 2. What clearly works well

- There is meaningful test coverage across unit and integration layers.
  Evidence: [`tests/unit/test_acceptance_matrix.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/unit/test_acceptance_matrix.py), [`tests/unit/test_official_ppo.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/unit/test_official_ppo.py), [`tests/integration/test_run_experiment.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_run_experiment.py), [`tests/integration/test_acceptance_matrix_pipeline.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_acceptance_matrix_pipeline.py), [`tests/integration/test_official_reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_official_reproduction.py).

- The official benchmark is manifest-driven rather than hand-waved.
  Evidence: [`configs/acceptance_matrices/official_acceptance_v1.json`](/Users/oltazagraxha/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json) defines policies, seeds, steps, cases, and matrix identity; [`artifacts/final_pipeline/official_acceptance_v1/metadata/pipeline_config.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/metadata/pipeline_config.json) preserves the exact invocation state.

- The repo keeps machine-readable final outputs, not just screenshots or prose.
  Evidence: [`artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv), [`artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json), [`artifacts/final_pipeline/official_acceptance_v1/report/concise_report.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/report/concise_report.md).

- The repository is unusually honest in some key docs about historical-vs-current evidence.
  Evidence: [`README.md`](/Users/oltazagraxha/Desktop/GreenNet/README.md), [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md), [`artifacts/final_pipeline/latest/README.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest/README.md) all explicitly state that the pinned `latest` bundle is historical and not the output of the current rerun path.

## 3. Testing Audit

### Verdict

- Automated tests exist and cover important flows: **PASS**
- Tests fully validate the final submission story end-to-end: **PARTIAL**
- Tests are environment-isolated and reviewer-safe: **FAIL**

### Evidence

- Verified locally in this audit:
  - `.venv/bin/python -m pytest -q tests/integration/test_official_reproduction.py tests/integration/test_acceptance_matrix_pipeline.py tests/unit/test_official_ppo.py tests/unit/test_acceptance_matrix.py`
  - Result: `8 passed in 14.90s`

- Verified locally in this audit:
  - `.venv/bin/python -m pytest -q tests/integration/test_run_experiment.py tests/integration/test_env_integration.py tests/integration/test_api_app.py`
  - Result: `17 passed, 1 failed`
  - Failing test: [`tests/integration/test_api_app.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_api_app.py) `test_final_evaluation_endpoint_returns_latest_valid_artifact`

### Findings

- Test breadth is decent, but much of it is smoke-level.
  The official reproduction integration test in [`tests/integration/test_official_reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_official_reproduction.py) uses `--check-only` and a placeholder PPO zip. That proves command wiring and prerequisite checks, not that the official benchmark can run from a clean environment and produce the shipped results.

- The current API integration test failure exposes real state leakage.
  In [`api_app.py`](/Users/oltazagraxha/Desktop/GreenNet/api_app.py), `/api/final_evaluation` always returns `_latest_final_evaluation_from_db()` before scanning artifact files. Because the DB is global, a test that patches `REPO_ROOT` still receives the real repository’s newer DB payload. This is not just a brittle test; it means behavior depends on ambient machine state.

- The stored verification note is stale relative to current behavior.
  [`final_audit/verification/README.md`](/Users/oltazagraxha/Desktop/GreenNet/final_audit/verification/README.md) says `.venv-verify/bin/python -m pytest -q` passed with `37 passed in 9.70s`. In this audit, a targeted current integration batch already fails in the main repo environment. That weakens trust in the repo’s “verified final state” claim unless the verification note is updated and scoped.

- Tool invocation assumptions are inconsistent.
  Shell `pytest` is not on PATH in this environment, and `pytest -q ...` failed with `command not found`, while `.venv/bin/python -m pytest ...` worked. Reviewer docs should not assume global executables when the repo’s own workflow depends on `.venv`.

## 4. Reproducibility Audit

### Verdict

- There is a plausible canonical rerun path: **PARTIAL**
- One canonical command/path is consistently documented and implemented: **FAIL**
- A reviewer can reliably rerun and verify the final claim without interpretation work: **PARTIAL**

### Major findings

- The code-default reproduction output path conflicts with the docs.
  [`greennet/evaluation/reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/greennet/evaluation/reproduction.py) defaults `--output-dir` to `artifacts/final_pipeline/latest`, but [`README.md`](/Users/oltazagraxha/Desktop/GreenNet/README.md) and [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md) describe `artifacts/final_pipeline/official_acceptance_v1/` as the canonical runnable benchmark output bundle. This is a reviewer-facing contradiction.

- The repo currently maintains two “official” final bundles with different numbers.
  [`artifacts/final_pipeline/official_acceptance_v1/report/concise_report.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/report/concise_report.md) reports PPO vs `all_on` energy reduction of `1.33%`.
  [`artifacts/final_pipeline/latest/report/concise_report.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest/report/concise_report.md) reports the historical promoted `1.49%`.
  The repo explains this, but a reviewer still sees competing “final” answers.

- The pinned `latest` bundle is explicitly not reproducible from the current one-command rerun path.
  Evidence: [`README.md`](/Users/oltazagraxha/Desktop/GreenNet/README.md), [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md), [`artifacts/final_pipeline/latest/README.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest/README.md), [`artifacts/final_pipeline/archive/official_acceptance_v1_energy_1p49_reconstructed/README.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/archive/official_acceptance_v1_energy_1p49_reconstructed/README.md).
  This honesty is good, but it means the repo is not yet in a clean “rerun this and get the shipped final result” state.

- Historical and current evidence are not cleanly separated.
  The repo contains `experiments/official_matrix_v1` through `v6`, `artifacts/final_pipeline/latest`, `artifacts/final_pipeline/official_acceptance_v1`, `artifacts/final_pipeline/archive/...`, `results/`, `runs/`, `tmp/official_acceptance_backup/`, `_exports/...`, and `final_audit/verification/...`. That is a lot of provenance surface for reviewers to untangle.

- The canonical PPO artifacts are present and tracked, which helps.
  Evidence: [`artifacts/models/official_acceptance_v1/checkpoint_family.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/models/official_acceptance_v1/checkpoint_family.json) and topology-specific zips under `small/`, `medium/`, and `large/`.

## 5. Evidence Quality and Trustworthiness Audit

### Verdict

- Artifacts are machine-readable and traceable: **PASS**
- Artifact provenance is simple and unambiguous: **FAIL**
- Pinned historical results are clearly distinguished from current reproducible outputs: **PARTIAL**

### Findings

- The manifest/metadata discipline is strong.
  [`artifacts/final_pipeline/official_acceptance_v1/metadata/pipeline_manifest.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/metadata/pipeline_manifest.json) and [`artifacts/final_pipeline/official_acceptance_v1/metadata/pipeline_config.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/metadata/pipeline_config.json) preserve command history, thresholds, and matrix identity.

- The current official final summary is internally coherent.
  [`artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json) includes selected run count `90`, policy classification, QoS thresholds, stability thresholds, and case IDs.

- The historical promoted bundle is transparent but still risky as a pinned reviewer entrypoint.
  [`artifacts/final_pipeline/latest/README.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest/README.md) says `latest` is a promoted historical reconstructed bundle. That is acceptable as archive evidence, but it is weak as the default destination for the reproduction command.

- The DB is far too large for a clean final-submission evidence store.
  `artifacts/db/greennet.sqlite3` is approximately **33G** (`8699305` pages at `4096` bytes/page). For an honors review repo, that raises immediate trust and UX concerns:
  - reviewers may not be able to clone or handle it comfortably
  - it is not obvious which data is necessary for final submission
  - API behavior depends on this DB
  - the evidence layer is harder to audit because historical and current final-evaluation entries coexist

- The locked evidence folders are useful but noisy.
  `artifacts/locked/` contains both curated scenario bundles and older exploratory logs such as [`artifacts/locked/20260220_025833_baseline/`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/locked/20260220_025833_baseline). This mixes polished evidence with deep debugging history.

## 6. Canonical Command and Reviewer Experience Audit

### Verdict

- A reviewer can find a start path: **PARTIAL**
- There is one obvious and conflict-free reviewer path: **FAIL**

### Specific problems

- The README says the canonical one-command reviewer path is:
  - `.venv/bin/python experiments/run_official_acceptance_matrix.py`

- But the wrapper code in [`greennet/evaluation/reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/greennet/evaluation/reproduction.py) defaults to:
  - output in `artifacts/final_pipeline/latest`

- Meanwhile the docs and reviewer files present:
  - `artifacts/final_pipeline/official_acceptance_v1/` as the canonical runnable bundle

- And `latest` itself says it is historical and not from the most recent rerun.

This is the central reviewer-UX problem in the repository today.

## 7. Mismatch Audit: Historical Results vs Current Reproducible Outputs

### Verified mismatch

- Historical promoted bundle:
  - path: [`artifacts/final_pipeline/latest/summary/final_evaluation/final_evaluation_summary.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest/summary/final_evaluation/final_evaluation_summary.json)
  - PPO energy reduction vs `all_on`: `1.4937128295319209%`

- Current runnable official bundle:
  - path: [`artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json)
  - PPO energy reduction vs `all_on`: `1.3336912174367566%`

### Assessment

- The repo does not hide this mismatch. That is a strength.
- The repo is still not submission-clean because the default reproduction wrapper currently points at the path (`latest`) that is explicitly historical.
- The historical promoted bundle should be archive-only, not reviewer-default, unless you can truly regenerate it.

## 8. Priority Checklist

### Critical before submission

- [ ] Make the canonical rerun destination and the canonical shipped bundle the same path.
  Why it matters: right now code defaults to `artifacts/final_pipeline/latest`, while docs present `artifacts/final_pipeline/official_acceptance_v1/` as canonical.
  Files: [`greennet/evaluation/reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/greennet/evaluation/reproduction.py), [`README.md`](/Users/oltazagraxha/Desktop/GreenNet/README.md), [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md), [`artifacts/final_pipeline/latest/README.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest/README.md).
  Done looks like: one rerun command, one default output path, one reviewer bundle, no ambiguity.

- [ ] Fix the `/api/final_evaluation` state-leak problem or clearly remove the API from the submission-critical reviewer path.
  Why it matters: reviewer-visible output currently depends on ambient DB state rather than only the selected artifact tree.
  Files: [`api_app.py`](/Users/oltazagraxha/Desktop/GreenNet/api_app.py), [`tests/integration/test_api_app.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_api_app.py).
  Done looks like: the integration test passes and API behavior is deterministic under isolated temp data.

- [ ] Update or regenerate the verification note so it matches the current repo state.
  Why it matters: [`final_audit/verification/README.md`](/Users/oltazagraxha/Desktop/GreenNet/final_audit/verification/README.md) currently overstates confidence because it claims a fully passing state that is not what this audit observed.
  Files: [`final_audit/verification/README.md`](/Users/oltazagraxha/Desktop/GreenNet/final_audit/verification/README.md), any supporting verification artifacts.
  Done looks like: commands, dates, interpreter, and pass/fail status all match the current repo exactly.

- [ ] Shrink or exclude the 33G SQLite DB from the final submission package, or provide a clearly scoped final DB snapshot.
  Why it matters: the current DB is too large and too historically mixed for a clean reviewer artifact.
  Files: [`artifacts/db/greennet.sqlite3`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/db/greennet.sqlite3), [`docs/run_database.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/run_database.md).
  Done looks like: either a small canonical DB containing only final-submission rows, or the DB is excluded from the reviewer-critical path.

### Important but not blocking

- [ ] Separate archive evidence from reviewer-first evidence more aggressively.
  Why it matters: `latest`, `archive`, `official_acceptance_v1`, `official_matrix_v6`, and `final_audit/verification` all compete for attention.
  Files: [`README.md`](/Users/oltazagraxha/Desktop/GreenNet/README.md), [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md), [`artifacts/final_pipeline/latest/README.md`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest/README.md).
  Done looks like: one “open this first” path and explicit “archive only” labels elsewhere.

- [ ] Strengthen end-to-end reproducibility testing beyond `--check-only`.
  Why it matters: current official reproduction tests validate plumbing, not a full clean rerun of the official pipeline.
  Files: [`tests/integration/test_official_reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_official_reproduction.py).
  Done looks like: at least one minimized but real pipeline run that produces a final-evaluation bundle from scratch.

- [ ] Standardize docs on `.venv/bin/python -m pytest` instead of assuming `pytest` is globally available.
  Why it matters: the shell `pytest` command was unavailable in this audit environment.
  Files: [`README.md`](/Users/oltazagraxha/Desktop/GreenNet/README.md), [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md), [`docs/run_database.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/run_database.md), [`docs/runs_management.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/runs_management.md).
  Done looks like: reviewer commands work exactly as written.

### Nice-to-have polish

- [ ] Trim exploratory/debug evidence from `artifacts/locked/` or move it under a clearly non-submission subfolder.
  Why it matters: reviewer evidence currently includes many exploratory logs that dilute confidence.
  Files: [`artifacts/locked/`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/locked).
  Done looks like: locked evidence reads as curated, not accumulated.

- [ ] Add a short provenance table comparing current runnable result vs preserved historical promoted result.
  Why it matters: this would turn an existing liability into a controlled and honest narrative.
  Files: [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md), reviewer-facing report files under [`artifacts/final_pipeline/`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline).
  Done looks like: a reviewer can understand the `1.33%` vs `1.49%` discrepancy in under a minute.

## 9. Fastest path to submission credibility

1. Make `official_acceptance_v1` the actual default output of the reproduction command and demote `latest` to archive-only.
2. Fix the `api_app.py` final-evaluation precedence issue so the integration test passes deterministically.
3. Replace the stale verification note with a current verification pass using the exact commands reviewers are told to run.
4. Remove or slim the 33G DB from the submission-critical bundle.

Those four changes would materially improve reviewer trust without requiring new research work.

## 10. Likely reviewer impression if submitted today

### What looks strong

- The project has real experiments, real structured outputs, and real tests.
- The acceptance-matrix manifest and final pipeline metadata are stronger than typical capstone evidence handling.
- The docs are more honest than average about mixed results and historical reconstruction.

### What looks weak

- Too many “official/final/latest/archive” paths.
- The shipped evidence story is not single-source.
- Verification claims are not fully aligned with present repo behavior.
- The DB and artifact footprint look more like an active lab notebook than a final curated submission.

### What may come up in oral defense

- “Which exact command should we run, and which folder is the real final result?”
- “Why is `latest` historical if it is called `latest`?”
- “Can you explain why the current rerun gives `1.33%` but the pinned reviewer bundle says `1.49%`?”
- “Why does your API prefer DB state over local artifact state?”

## 11. Evidence Map

- [`README.md`](/Users/oltazagraxha/Desktop/GreenNet/README.md): top-level reviewer workflow and final-claim language.
- [`docs/final_submission_overview.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/final_submission_overview.md): canonical-submission framing and artifact glossary.
- [`docs/runs_management.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/runs_management.md): curated vs locked vs historical runs.
- [`docs/run_database.md`](/Users/oltazagraxha/Desktop/GreenNet/docs/run_database.md): DB role and persistence claims.
- [`pyproject.toml`](/Users/oltazagraxha/Desktop/GreenNet/pyproject.toml): Python requirements and pytest config.
- [`configs/README.md`](/Users/oltazagraxha/Desktop/GreenNet/configs/README.md): training/reproduction config guidance.
- [`configs/acceptance_matrices/official_acceptance_v1.json`](/Users/oltazagraxha/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json): authoritative benchmark definition.
- [`greennet/evaluation/reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/greennet/evaluation/reproduction.py): one-command reproduction implementation and default output path.
- [`greennet/evaluation/final_pipeline.py`](/Users/oltazagraxha/Desktop/GreenNet/greennet/evaluation/final_pipeline.py): pipeline orchestration and artifact writing.
- [`greennet/evaluation/final_report.py`](/Users/oltazagraxha/Desktop/GreenNet/greennet/evaluation/final_report.py): final evaluation generation.
- [`greennet/evaluation/official_ppo.py`](/Users/oltazagraxha/Desktop/GreenNet/greennet/evaluation/official_ppo.py): topology-specific PPO family handling.
- [`api_app.py`](/Users/oltazagraxha/Desktop/GreenNet/api_app.py): reviewer-facing API behavior and final-evaluation lookup precedence.
- [`tests/integration/test_api_app.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_api_app.py): exposed current integration failure.
- [`tests/integration/test_official_reproduction.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_official_reproduction.py): official reproduction smoke coverage.
- [`tests/integration/test_acceptance_matrix_pipeline.py`](/Users/oltazagraxha/Desktop/GreenNet/tests/integration/test_acceptance_matrix_pipeline.py): manifest-driven pipeline integration.
- [`artifacts/final_pipeline/official_acceptance_v1/`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1): current runnable official bundle.
- [`artifacts/final_pipeline/latest/`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/latest): pinned historical promoted bundle.
- [`artifacts/final_pipeline/archive/official_acceptance_v1_energy_1p49_reconstructed/`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/archive/official_acceptance_v1_energy_1p49_reconstructed): reconstructed historical archive.
- [`final_audit/verification/README.md`](/Users/oltazagraxha/Desktop/GreenNet/final_audit/verification/README.md): stored verification claim that now needs refresh.
- [`artifacts/db/greennet.sqlite3`](/Users/oltazagraxha/Desktop/GreenNet/artifacts/db/greennet.sqlite3): primary structured store, currently oversized for clean submission.
