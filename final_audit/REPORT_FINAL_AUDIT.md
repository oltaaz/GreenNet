# GreenNet Final Audit Report

## 1. Executive Summary
- Overall readiness score: 5/10
- Submission recommendation:
  - Needs important fixes before submission
- GreenNet has a credible simulator core, a non-trivial experiment/reporting stack, and a coherent backend/frontend shape. The problem is final-readiness, not raw ambition. The current repository mixes strong implementation with workflow drift, missing packaged evidence, inconsistent setup instructions, and a final evaluation artifact that does not support the intended headline claim that the AI controller materially outperforms the baseline on energy. In its current state, this is not a weak project, but it is not yet a strong final capstone submission.

## 2. What Is Already OK
- Modular simulator core
  - Why it is good: The project is not built around a fake or purely narrative "AI networking" demo. It contains explicit topology, traffic, routing, capacity, delay, power, and carbon modeling, which makes the technical story defensible for a capstone.
  - Evidence (file paths / artifacts / scripts): `greennet/env.py`, `greennet/simulator.py`, `greennet/routing.py`, `greennet/power.py`, `greennet/topology.py`, `greennet/traffic.py`

- Routing baselines are named and separated from controller policy
  - Why it is good: The repository distinguishes forwarding/routing assumptions from controller policy, which is technically more honest than treating "baseline routing" and "baseline controller" as the same thing.
  - Evidence (file paths / artifacts / scripts): `README.md`, `baselines.py`, `run_experiment.py`, `greennet/routing.py`

- Final reporting pipeline exists
  - Why it is good: There is already a real aggregation and reporting layer that can turn run artifacts into summary CSV/JSON/Markdown outputs. That is much closer to thesis-ready than a repo that only contains training code and raw logs.
  - Evidence (file paths / artifacts / scripts): `greennet/evaluation/final_report.py`, `greennet/evaluation/final_pipeline.py`, `experiments/final_evaluation.py`, `experiments/final_pipeline.py`, `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`

- API and React frontend are structurally aligned
  - Why it is good: The backend exposes data that the frontend is actually designed to consume, rather than the UI scraping files ad hoc. That makes the demo path more credible.
  - Evidence (file paths / artifacts / scripts): `api_app.py`, `greennet/persistence/sqlite_store.py`, `frontend/greennet-ui/src/lib/api.ts`, `frontend/greennet-ui/src/pages/DashboardPage.tsx`, `frontend/greennet-ui/src/pages/ComparePage.tsx`, `frontend/greennet-ui/src/pages/SimulatorPage.tsx`

- Test layout is meaningful
  - Why it is good: The repository includes unit and integration tests over the simulator, routing, reward/masking logic, API, and experiment artifact generation. That is a real software-quality strength.
  - Evidence (file paths / artifacts / scripts): `tests/unit/test_env_reward_and_masking.py`, `tests/unit/test_routing.py`, `tests/unit/test_forecasting.py`, `tests/unit/test_power.py`, `tests/integration/test_api_app.py`, `tests/integration/test_run_experiment.py`

- Locked artifacts and scenario validation evidence exist
  - Why it is good: The project has at least some curated, reviewer-facing evidence bundles rather than only transient development outputs.
  - Evidence (file paths / artifacts / scripts): `artifacts/locked/normal/20260220_111755_100k_ctrl_cap16/eval_summary.md`, `artifacts/traffic_verify/20260220_matrix/matrix_status.md`, `artifacts/traffic_verify/20260220_matrix/traffic_eval_summary.csv`

## 3. What Needs Improvement

### Technical correctness
- Final AI-vs-baseline claim is not currently supported
  - Severity: critical
  - Why it matters: The final evaluation artifact is the project's strongest evidence source, and it currently says the hypothesis was not achieved. If the written submission still claims meaningful AI energy gains, the repo contradicts the claim.
  - Evidence: `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md` shows `best overall policy: heuristic`, `best AI policy: ppo`, and only about `-4.01%` energy reduction vs heuristic against a `15.0%` target, with `hypothesis=not_achieved`.
  - Concrete recommendation: Either improve the model/evaluation until the claim is true, or reframe the capstone honestly as a negative or mixed result: "the RL controller matched or slightly trailed the best heuristic while preserving QoS, and the project's contribution is the evaluation framework plus safety-aware simulator."

- AI controller behavior is not cleanly separated from rule-based safety scaffolding
  - Severity: major
  - Why it matters: The RL path is intertwined with masking, QoS gates, cooldown logic, demand gates, and optional Impact Predictor overrides. That can be valid engineering, but it weakens any simplistic claim that PPO alone learned the control policy.
  - Evidence: `greennet/env.py`, `greennet/rl/eval.py`, `run_experiment.py`, `baselines.py`
  - Concrete recommendation: Document the actual control stack explicitly: RL policy + hard safety gates + optional learned risk gate. Make that architecture part of the writeup rather than implying a pure end-to-end RL controller.

- Impact Predictor capability is documented more strongly than it is demonstrated
  - Severity: major
  - Why it matters: README-level claims about a complete, accepted subsystem are risky if the repo does not include the referenced model bundle and locked acceptance artifacts.
  - Evidence: `README.md` references `models/impact_predictor` and `artifacts/locked/impact_predictor/<timestamp>/`; `Test-Path models\impact_predictor` is `False`; `Test-Path artifacts\locked\impact_predictor` is `False`.
  - Concrete recommendation: Either add the missing model bundle and locked acceptance evidence, or clearly mark the Impact Predictor as experimental/optional and not part of the final demonstrated scope.

### Evaluation / experiments
- Official matrix package is not reproducible from this checkout
  - Severity: critical
  - Why it matters: The repo contains summary CSVs and notes, but the paths they reference are missing. A reviewer cannot trace summary rows back to included run folders or the referenced PPO checkpoint.
  - Evidence: `experiments/official_matrix_v6/results_summary_matrix_v6.csv` points to `results/...` folders, but `Test-Path results` is `False`; `experiments/official_matrix_v6/notes.md` references `runs/20260205_222626/ppo_greennet.zip`, but that run is not present under `runs/`.
  - Concrete recommendation: Package one canonical final bundle that includes the summary tables, the referenced checkpoint, and either the full run folders or a traceability manifest that maps each summary row to included artifacts.

- There are too many experiment eras and canonicality is unclear
  - Severity: major
  - Why it matters: Multiple official matrix versions, many top-level config variants, legacy scripts, and demo bundles make it difficult to determine what the final authoritative result is.
  - Evidence: `experiments/official_matrix_v1`, `experiments/official_matrix_v2`, `experiments/official_matrix_v4`, `experiments/official_matrix_v6`, top-level `train_normal_v*.json`, `train_burst_v*.json`, `eval.py`, `evaluate_checkpoints.py`, `resume_latest.py`
  - Concrete recommendation: Pick one final tag and one final train/eval config family. Mark everything else as archived, superseded, or internal.

- Positive locked-scenario acceptance evidence and final matrix evidence tell different stories
  - Severity: major
  - Why it matters: The repository has PASS summaries for locked traffic verification, but the thesis-level final evaluation still says the main energy hypothesis failed. That difference needs explanation.
  - Evidence: `artifacts/traffic_verify/20260220_matrix/matrix_status.md` reports PASS for normal/burst/hotspot; `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md` reports `hypothesis=not_achieved`.
  - Concrete recommendation: Add a short "how these artifacts differ" note: traffic verification checks safe operation under fixed off-level/controller conditions; final evaluation tests the headline baseline-vs-AI hypothesis across the official matrix.

### Documentation / communication
- Setup and run instructions are inconsistent with the actual repo
  - Severity: critical
  - Why it matters: A final reviewer will judge the project partly on whether they can understand and reproduce it from the docs. The current docs disagree with the checkout.
  - Evidence: `COMMANDS.md` references `requirements.txt`, `ml-env/.venv`, and `results/`; there is no tracked `requirements.txt`, no `ml-env/`, and no tracked `results/` directory. `README.md` mixes `.venv/bin/python` and `ml-env/.venv/bin/python`.
  - Concrete recommendation: Replace the current mixed instruction set with one canonical path based on the actual repo, current dependencies, and the final supported workflows.

- Frontend docs contain machine-specific paths
  - Severity: moderate
  - Why it matters: This makes the repo look unpolished and copied from a local workstation instead of submission-ready.
  - Evidence: `frontend/greennet-ui/README.md` contains `/Users/enionismaili/Desktop/GreenNet/...`
  - Concrete recommendation: Replace all machine-specific absolute paths with relative, repository-based instructions.

- Final evidence artifacts still contain machine-specific provenance
  - Severity: moderate
  - Why it matters: Reviewer-facing reports should look packaged and portable.
  - Evidence: `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md` and `.../final_evaluation_summary.json` include `/Users/enionismaili/Desktop/GreenNet/...`
  - Concrete recommendation: Regenerate or sanitize the final bundle so paths are relative or described abstractly.

### Testing / reproducibility
- The environment definition does not support the repo’s actual toolchain
  - Severity: critical
  - Why it matters: The code uses packages that are not declared in the main project dependencies, and the current environment was not able to run basic verification.
  - Evidence: `pyproject.toml` omits packages used in code and scripts such as `sb3_contrib`, `torch`, `scikit-learn`, and optional `matplotlib`; imports appear in `greennet/cli/train_cli.py`, `greennet/rl/eval.py`, `greennet/rl/sweep.py`, `greennet/impact_predictor.py`, `scripts/train_cost_estimator_torch.py`, and `greennet/evaluation/final_pipeline.py`.
  - Concrete recommendation: Define one reproducible environment spec for final review, including core, RL, optional ML, testing, and frontend prerequisites.

- Basic verification could not be executed from the current checkout environment
  - Severity: major
  - Why it matters: Even if the code is good, inability to run tests/build/API in a fresh or shared environment damages final submission credibility.
  - Evidence: `.\\.venv\\Scripts\\python.exe -m pytest -q` failed with `No module named pytest`; `.\\.venv\\Scripts\\python.exe -c "import api_app"` failed with `ModuleNotFoundError: No module named 'fastapi'`; `npm run build` in `frontend/` failed because `node.exe` was unavailable.
  - Concrete recommendation: Make the documented environment actually runnable and include one verified smoke-test section in the docs with expected commands and outcomes.

### Architecture / maintainability
- Workflow sprawl creates ambiguity
  - Severity: major
  - Why it matters: There is no single clear answer to "how do I train, evaluate, and package the final result?" Too many overlapping entrypoints exist.
  - Evidence: `train.py`, `run_experiment.py`, `eval.py`, `evaluate_checkpoints.py`, `experiments/run_matrix.py`, `experiments/final_pipeline.py`, `COMMANDS.md`, `README.md`
  - Concrete recommendation: Keep one official train path, one official evaluation path, and one official packaging path. Mark legacy scripts explicitly.

- Several critical files are too large and too multi-purpose
  - Severity: moderate
  - Why it matters: Late-stage reviewers and project owners need to understand the system quickly. Giant orchestration files increase risk and explanation cost.
  - Evidence: `greennet/env.py`, `greennet/cli/train_cli.py`, `run_experiment.py`, `greennet/rl/eval.py`, `api_app.py`
  - Concrete recommendation: For submission, add architecture documentation and module role summaries even if no major refactor is done yet. If time permits later, split the largest files by concern.

- Repository root is cluttered with mixed concerns
  - Severity: moderate
  - Why it matters: The root directory contains code, configs, old artifacts, paper drafts, and legacy experiment files all at once. This weakens the final impression.
  - Evidence: root-level `train_*`, `eval_*`, `.docx` proposal/draft files, `Terminal_Output.txt`, `models/latest/ppo_greennet_old.zip`
  - Concrete recommendation: Move non-essential drafts and legacy files into archival folders or clearly label them as non-canonical.

### Frontend / demo / usability
- Demo fallback can silently switch to synthetic data
  - Severity: major
  - Why it matters: A presentation or reviewer session could appear to be showing live backend-backed results while actually showing locally generated demo data.
  - Evidence: `frontend/greennet-ui/src/lib/api.ts`, `frontend/greennet-ui/src/lib/demo.ts`, `frontend/greennet-ui/src/pages/ComparePage.tsx`, `frontend/greennet-ui/src/pages/SimulatorPage.tsx`
  - Concrete recommendation: Make demo mode explicit and non-silent. The UI should prominently show when it is using synthetic data and should not pretend those are real experiment artifacts.

- Demo alias mapping is inconsistent with canonical policy aliases
  - Severity: major
  - Why it matters: This risks mislabeling policies in fallback/demo paths, which is unacceptable for a final demo.
  - Evidence: `frontend/greennet-ui/src/lib/demo.ts` maps `baseline -> all_on` and `noop -> heuristic`; `frontend/greennet-ui/src/lib/data.ts` maps `noop -> all_on` and `baseline -> heuristic`
  - Concrete recommendation: Make alias handling consistent across backend, frontend, demo, and reporting code before any final presentation.

- Official UI split is understandable but still somewhat confusing
  - Severity: moderate
  - Why it matters: The repo says React is the official frontend and Streamlit is internal, but both still occupy visible product-like roles.
  - Evidence: `README.md`, `dashboard/README.md`, `dashboard/app.py`, `frontend/README.md`
  - Concrete recommendation: Keep the split, but label Streamlit everywhere as analyst/internal tooling and remove it from any final-user narrative unless explicitly needed.

### Ethical / responsible design
- QoS safeguards exist in code, but responsible framing needs improvement
  - Severity: major
  - Why it matters: The code does include QoS penalties and OFF-action safety gating, which is good. The submission still needs to explain those constraints and their effect on results, especially because the final AI hypothesis failed.
  - Evidence: `greennet/env.py`, `artifacts/traffic_verify/20260220_matrix/matrix_status.md`, `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`
  - Concrete recommendation: Add a limitations/responsible-design section that explicitly says the project optimizes under QoS constraints, not energy alone, and that safety constraints may reduce the energy gains achievable by RL.

- Some repo language overstates readiness
  - Severity: moderate
  - Why it matters: Overclaiming makes the project easier to challenge during viva/presentation or final review.
  - Evidence: `README.md` Impact Predictor completion wording versus missing bundled artifacts; general framing of AI-enhanced routing despite final evidence favoring heuristic
  - Concrete recommendation: Audit the README and final report language for claims that should be softened to "implemented", "explored", "evaluated", or "not fully demonstrated".

## 4. Submission Blockers
- The current final evaluation artifact does not support the intended AI energy-optimization headline; it explicitly reports `hypothesis=not_achieved` and only about `-4%` energy reduction vs heuristic in `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`.
- The official matrix package is not reproducible from this checkout because the referenced `results/...` folders and referenced PPO run are missing.
- Setup and verification are not reviewer-safe: current docs are inconsistent and basic checks failed in the present environment (`pytest`, `fastapi`, and frontend build prerequisites were unavailable).
- README-level Impact Predictor claims are not matched by bundled artifacts or a bundled runtime model directory.
- Demo fallback integrity is not safe enough for a final presentation because synthetic demo data can stand in for backend data and policy aliases are inconsistent.

## 5. Strong-but-Optional Improvements
- Add a one-page architecture diagram or workflow diagram that shows training, run generation, aggregation, final evaluation, API serving, and frontend consumption.
- Reduce root-level clutter by moving old drafts, legacy configs, and obsolete model artifacts into an archival folder.
- Add a small "artifact glossary" explaining the difference between `runs/`, `artifacts/locked/`, `artifacts/traffic_verify/`, `experiments/official_matrix_*`, and `demo_bundle/`.
- Regenerate docs with clean UTF-8 output and remove visible mojibake from user-facing text.
- Add one final smoke-test script that checks Python imports, API startup, key endpoints, and frontend build preconditions.

## 6. Evidence Gaps / Unverified Claims
- Claim: the AI controller achieves a strong energy win over the baseline. Current evidence gap: the final official artifact says the headline hypothesis was not achieved.
- Claim: Impact Predictor is complete and accepted. Current evidence gap: the repo does not include the referenced `models/impact_predictor` folder or locked acceptance bundle.
- Claim: the official matrix is reproducible from the repository. Current evidence gap: referenced `results/...` artifacts are missing from this checkout.
- Claim: the frontend/backend demo path is turnkey. Current evidence gap: API import and frontend build could not be verified in the current environment.
- Claim: there is one clean official workflow. Current evidence gap: multiple overlapping train/eval/report paths remain active and insufficiently labeled.

## 7. Recommended Order of Attack
1. Decide the final claim first. Either improve the final experiment result or rewrite the project narrative to match the current evidence honestly.
2. Package one canonical final evidence bundle that actually exists in the repo and can be traced from report tables back to included artifacts.
3. Fix the environment and setup story so backend, tests, and frontend can be launched from a clean checkout using one documented path.
4. Clean README, COMMANDS, and frontend docs so they all describe the same supported workflow and remove machine-specific paths.
5. Resolve the Impact Predictor scope mismatch by either bundling the missing evidence or downgrading the claim.
6. Fix demo integrity issues, especially silent synthetic fallback and alias mismatches, before any presentation.
7. Mark legacy scripts/configs/surfaces as archived or internal so the final repo has one obvious path through training, evaluation, and demo.
