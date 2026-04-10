# GreenNet Final Submission Audit

## 1. Executive Verdict

- Overall readiness score: **8.0 / 10**
- Submission recommendation: **Ready with minor fixes**

Brief explanation:
GreenNet is a real and technically substantial capstone repository. The simulator, experiment pipeline, API, persistence layer, and React UI are materially implemented, and the critical submission blockers from the audit have now been repaired. It is now reasonably submittable, but a strict Honors Board could still press on realism framing, QoS interpretation, and the breadth of secondary forecasting/cost-model claims.

The most important concerns are:
- forecasting / impact-predictor claims are still stronger than their current dedicated evidence
- some wording still needs discipline around realism and QoS interpretation
- the repository remains artifact-heavy even though the canonical reviewer path is now explicit

## 2. What the project clearly does well

- **Real simulator core**: The repository contains a coherent flow-level simulation stack with topology loading, traffic generation/replay, routing policies, capacity clipping, delay approximation, power/carbon accounting, QoS evaluation, and switching stability controls in `greennet/`.
- **Explicit baseline taxonomy**: The project now distinguishes the official traditional baseline (`all_on`), strongest non-AI heuristic (`heuristic`), and AI policy (`ppo`) in code and docs instead of collapsing everything into “baseline.”
- **Nontrivial evaluation pipeline**: The acceptance-matrix/final-pipeline path is real and produces summary CSVs, final evaluation JSON/Markdown, leaderboard outputs, plots, and DB persistence.
- **Claim honesty is better than average**: The current README and final-submission overview explicitly say the final matrix does not prove strong AI dominance, and they acknowledge that the historical `latest/` bundle is pinned and not the current rerun output.
- **Usable reviewer-facing UI exists**: The React frontend builds successfully and has meaningful pages for dashboarding, simulation playback, and final comparison. The `/results` / comparison surface is the strongest presentation artifact.
- **Tests exist across layers**: There is real unit and integration coverage for routing, topology, traffic, env behavior, official PPO packaging, acceptance matrix logic, and parts of the API/reproduction flow.

## 3. Requirement-by-requirement evaluation

### Proposal promises

| Item | Status | Why |
| --- | --- | --- |
| Routing / network simulator | PASS | Implemented in `greennet/simulator.py`, `greennet/env.py`, `greennet/topology.py`, `greennet/traffic.py`, `greennet/routing.py`. |
| AI-driven routing / optimization | PARTIAL | AI is real, but it is link-state control over a fixed routing baseline, not learned path routing. The evaluated `ppo` path is hybridized with rules. |
| Energy-aware behavior | PASS | Energy model, link/device sleep logic, transition costs, and energy-driven reward/reporting are implemented. |
| Dashboard / visualization | PASS | React app is real and buildable; Streamlit dashboard also exists, though it creates reviewer-path ambiguity. |
| Comparative evaluation | PASS | `all_on`, `heuristic`, and `ppo` are compared through the official acceptance matrix and final report outputs. |
| Documented methodology and research framing | PARTIAL | Methodology exists in proposal, draft, README, final overview, and reports; the critical PPO artifact drift is fixed, but some supporting ML/cost-model claims still need softer framing. |
| Hypothesis-driven results | PASS | The repo explicitly states the hypothesis was **not achieved** in the final benchmark. |

### Stage 2 / first-draft promises

| Item | Status | Why |
| --- | --- | --- |
| System architecture | PASS | There is a real package structure with simulator, API, persistence, evaluation pipeline, and UI separation. |
| Methodology | PARTIAL | The methodology is present, but some Stage 2 language implies stronger forecasting/cost-modeling/RL validation than the current repo supports. |
| Testing framework | PASS | Real unit/integration tests exist and the suite is currently green. |
| Reproducibility | PARTIAL | There is a canonical rerun command, a unified final bundle path, and the API state leak is fixed; remaining weakness is mainly repo bulk and historical layering. |
| Evaluation structure | PASS | Acceptance matrix, summary CSVs, final evaluation reports, and official policy taxonomy are implemented. |
| Dashboard/backend integration | PASS | FastAPI + React integration is real and working enough for build/demo use. |
| RL claims | PARTIAL | PPO is real and trainable, but the evaluated controller is not pure PPO and does not beat the strongest heuristic in the final aggregate. |
| Forecasting claims | PARTIAL | Forecasting exists and is integrated, but it is lightweight online forecasting rather than a strong standalone ML contribution with dedicated evidence. |
| Cost-modeling claims | PARTIAL | Impact predictor / cost estimator code exists, but validation is weak and should be treated as exploratory rather than a headline contribution. |

### Final-review expectations

#### Technical soundness

| Item | Status | Why |
| --- | --- | --- |
| Simulator architecture is coherent and defensible | PASS | Coherent at a flow-level abstraction. Not a packet/network-protocol simulator, but technically defensible if framed honestly. |
| Routing baselines are clearly defined and honestly named | PARTIAL | Mostly improved. Still risky if `ospf`/“routing” language is used loosely, because this is static shortest-path/ECMP-style forwarding, not protocol realism. |
| AI/RL part is real, integrated, and evaluated properly | PARTIAL | Real and integrated, and the repo now labels the final controller honestly as a PPO-based hybrid; the remaining concern is not fake AI but limited performance and realism scope. |
| Energy modeling is defensible enough | PASS | Explicit, simplified accounting model; acceptable for capstone comparison if not sold as hardware-calibrated measurement. |
| Metrics, QoS constraints, and comparison logic are coherent | PARTIAL | The logic exists and is explicit, but some “acceptable QoS” outputs sit beside very high absolute drop/delay values, which will require careful explanation. |
| Credible final experiment/evaluation pipeline | PASS | The pipeline is real and generates final outputs. |
| Reproducible enough for reviewers | PARTIAL | Better than average; the canonical final bundle path is unified, the API final-evaluation state leak is fixed, and quality gates are clean, but the repository is still large and historically layered. |

#### Ethical / responsible design

| Item | Status | Why |
| --- | --- | --- |
| Avoids misleading “AI saves energy” claims | PARTIAL | The main docs are honest, but repo structure and some UI wording still make overreading possible. |
| Tradeoffs, limitations, risks, assumptions stated honestly | PASS | README and final overview explicitly acknowledge simplified energy modeling, mixed AI outcome, and historical bundle limitations. |
| QoS and user impact are protected | PARTIAL | QoS constraints are explicit, but the final absolute QoS metrics are still rough enough that reviewers may question whether “acceptable” is too internal/project-defined. |
| Sustainability presented responsibly rather than as hype | PARTIAL | Stronger than most student repos, but still vulnerable if oral/written framing overstates real-world applicability. |
| Overclaim risk around AI/optimization/real-world applicability is controlled | PARTIAL | The biggest remaining risk area. |

#### Communication of complex ideas

| Item | Status | Why |
| --- | --- | --- |
| Reviewer can understand what the system is | PASS | The current README/docs explain the high-level system clearly. |
| Reviewer can distinguish baseline vs heuristic vs AI | PASS | Current docs do this much better than the proposal/draft. |
| Reviewer can understand what the final result actually is | PASS | The final result is now documented through a clearer canonical bundle path and a root-level submission index. |
| Reviewer can see whether the hypothesis was met | PASS | Final report and concise report clearly say it was not achieved. |
| Docs, folder structure, commands, and outputs are understandable | PARTIAL | The canonical reviewer path is now explicit and much better labeled; remaining confusion comes from repository bulk rather than a missing path. |
| One obvious reviewer path exists | PASS | `SUBMISSION_INDEX.md`, the final overview, and `artifacts/final_pipeline/official_acceptance_v1/` now define a single obvious path. |
| Final artifacts are presentation-ready | PARTIAL | The core bundle is presentation-ready; side materials are still broader and noisier than ideal. |

## 4. Honesty / overclaim / inconsistency audit

### Claims that are fully supported

- GreenNet contains a real **energy/QoS-aware network simulation and evaluation framework**.
- The repo includes explicit comparison among `all_on`, `heuristic`, and `ppo`.
- The final official evaluation does **not** support the original “AI wins strongly on energy” hypothesis.
- The React frontend and FastAPI backend are real and integrated.
- SQLite persistence and artifact export are both implemented.

### Claims that are only partially supported

- “AI-driven routing”: partially true. The learned component controls link states over a fixed routing baseline rather than learning end-to-end route computation.
- “Forecasting module”: true, but it is lightweight online forecasting rather than a major validated ML contribution.
- “Cost modeling / impact predictor”: real code and artifacts exist, but validation is weak enough that it should be framed as exploratory.
- “Reproducible official final benchmark”: partly true. The official pipeline reruns, but the pinned `latest/` bundle is historical, and some API/output behavior is affected by ambient DB state.

### Claims that should be softened or rewritten

- Any phrase implying the final `ppo` result is **pure PPO** should be rewritten as a **PPO-based hybrid controller with rule-based safety/override logic**.
- Any phrase implying the simulator is a **realistic Internet routing simulator** should be rewritten as a **flow-level, static-routing, abstract network simulator**.
- Any phrase implying **forecasting and cost modeling are headline validated contributions** should be softened to exploratory/supporting components unless stronger evidence is added.
- Any phrase implying **real-world deployable network savings** should be replaced with **comparative results within a simplified simulation model**.

### Wording likely to mislead reviewers

- The simulator page wording about “simulated packet flow and real link states” is too easy to overread.
- Historical `latest/` naming is risky because many reviewers will assume “latest” means “most recent rerun,” which is explicitly not true here.
- The presence of multiple “official” bundles and `_exports` snapshots can make it look like results were curated after the fact, even if the docs explain why.
- The use of `ppo` as the report label without explaining the hybrid safety wrapper is misleading.

## 5. Reproducibility and reviewer experience audit

- **How easy is it to run?** Moderately easy, and now substantially easier to navigate because the root `SUBMISSION_INDEX.md` and archival labels narrow the reviewer path.
- **Is there one canonical command/path?** Yes in docs and labels: `.venv/bin/python experiments/run_official_acceptance_matrix.py` feeding `artifacts/final_pipeline/official_acceptance_v1/`.
- **Are outputs easy to verify?** Partially. The final pipeline outputs are well structured and the canonical path is clear, but the repo still carries a lot of historical material.
- **Are artifacts trustworthy and well organized?** Partially. The canonical artifacts are reasonable, but trust is weakened by:
  - pinned historical `latest/`
  - a very large checked-in artifact/database footprint
  - broad backup/export duplication
  - large historical run directories

Verified evidence:
- `pytest -q` passes: `78 passed`.
- `npm --prefix frontend/greennet-ui run lint` passes.
- `npm --prefix frontend/greennet-ui run build` passes.
- `artifacts/db/greennet.sqlite3` exists and the local DB contains `1757` runs and `9` final-evaluation payloads.
- `artifacts/`, `results/`, `_exports/`, `tmp/official_acceptance_backup/`, and `experiments/official_matrix_v*` create substantial reviewer clutter.

## 6. Code and architecture audit

### Strong architectural decisions

- Installable Python package with layered modules.
- Thin wrapper entry points instead of huge single scripts.
- Clear separation of simulation, evaluation, persistence, API, and frontend concerns.
- Explicit baseline taxonomy and final evaluation pipeline.
- SQLite integration without removing file-backed artifacts.

### Weak spots

- The repository presentation is much weaker than the code architecture.
- Legacy root-level configs and scripts compete with canonical paths.
- The repo still carries a very heavy historical artifact footprint.
- Forecasting/impact-predictor side systems are easier to overread than their evidence supports.

### Duplication / dead / confusing areas

- `_exports/GreenNet_progress_review_bundle_20260411/` is a near-duplicate project snapshot.
- `artifacts/final_pipeline/latest/` remains a preserved historical bundle beside the canonical final bundle.
- `experiments/official_matrix_v1` through `v6` are preserved evidence but crowd the canonical path.
- `tmp/official_acceptance_backup/` adds another “official” evidence surface.
- Existing `final_audit/` materials are useful internally but also add another reviewer-facing narrative layer.
- Source hygiene is weakened by `__pycache__`, `.DS_Store`, local env directories, and checked-in bulk artifacts.

### Anything fragile or jury-rigged

- Historical layering and preserved bundle breadth.
- Heavy SQLite/artifact footprint.
- Demo/generated fallback behavior in the frontend.
- Historical bundle reconstruction/promotion into `latest/`.

## 7. Final checklist of improvements

### Critical before submission

- [x] **Make the final `ppo` claim honest in all reviewer-facing materials.**  
What is wrong: the evaluated controller is not clean raw PPO; `run_experiment.py` uses a PPO-based hybrid with rule-based override/recovery logic.  
Why it matters: presenting it as pure PPO is a material scientific honesty issue.  
Files/modules involved: `run_experiment.py`, `README.md`, `docs/final_submission_overview.md`, final report/concise report generation outputs.  
Done looks like: completed in this repair pass. Reviewer-facing docs, checked-in final report artifacts, report-generation code, and UI labels now explicitly describe `ppo` as the PPO-based hybrid controller while preserving the stable benchmark policy id.

- [x] **Fix the canonical final-evaluation story so one path is actually canonical.**  
What is wrong: `artifacts/final_pipeline/latest/` is pinned historical output, while the rerun path writes elsewhere, and `experiments/official_matrix_v6/` remains another “official” story.  
Why it matters: reviewers will not trust a repo with multiple competing final answers.  
Files/modules involved: `README.md`, `docs/final_submission_overview.md`, `artifacts/final_pipeline/latest/README.md`, `artifacts/final_pipeline/official_acceptance_v1/`, `experiments/official_matrix_v6/`.  
Done looks like: completed in this repair pass. The reproduction default, reviewer docs, and artifact landing notes now all point to `artifacts/final_pipeline/official_acceptance_v1/` as the canonical final bundle, while `latest/` and `official_matrix_v6/` are explicitly archival.

- [x] **Fix the API final-evaluation state-isolation defect.**  
What is wrong: `/api/final_evaluation` currently prefers ambient SQLite state; one integration test fails because of it.  
Why it matters: this undermines reproducibility and can return stale/out-of-scope “official” results.  
Files/modules involved: `api_app.py`, `tests/integration/test_api_app.py`, `greennet/persistence/sqlite_store.py`.  
Done looks like: completed in this repair pass. The endpoint now only prefers DB-backed final-evaluation payloads when their stored artifact paths belong to the current `REPO_ROOT`, and the targeted integration tests pass.

- [x] **Resolve the PPO artifact lineage mismatch.**  
What is wrong: docs describe `configs/train_official_ppo.json` at `100000` timesteps, while checked-in official checkpoint metadata points to `configs/train_normal.json` at `25000` timesteps.  
Why it matters: this is a direct reproducibility/claim-honesty problem.  
Files/modules involved: `README.md`, `configs/README.md`, `configs/train_official_ppo.json`, `artifacts/models/official_acceptance_v1/*/checkpoint_metadata.json`.  
Done looks like: completed in this repair pass. Reviewer docs now match the checked-in checkpoint metadata, and `train_official_ppo.json` is explicitly labeled as an alternate longer-run recipe rather than the source of the current canonical family.

- [x] **Get the repository into a submission-clean state.**  
What is wrong: the repo surface is cluttered with duplicate exports, historical bundles, backups, generated caches, and huge bulk artifacts.  
Why it matters: reviewer experience and trust.  
Files/modules involved: `_exports/`, `tmp/official_acceptance_backup/`, `experiments/official_matrix_v*`, `final_audit/`, `results/`, generated cache files, root-level legacy configs.  
Done looks like: completed in this repair pass. `SUBMISSION_INDEX.md` now defines one obvious reviewer path, and the highest-confusion directories are labeled in place as archival/internal rather than competing with the canonical final bundle.

- [x] **Fix the failing Python test and clean the frontend lint errors.**  
What is wrong: the test suite was not clean and frontend lint failed.  
Why it matters: a final submission should not ship with known failing quality gates.  
Files/modules involved: `tests/integration/test_api_app.py`, `api_app.py`, `frontend/greennet-ui/src/hooks/useBackendStatus.tsx`, `frontend/greennet-ui/src/lib/api.ts`, `frontend/greennet-ui/src/pages/DashboardPage.tsx`, `frontend/greennet-ui/src/components/TopologyPanel.tsx`.  
Done looks like: completed in this repair pass. `pytest -q`, `npm --prefix frontend/greennet-ui run lint`, and `npm --prefix frontend/greennet-ui run build` all pass.

### Important but not blocking

- [ ] **Tighten language about networking realism.**  
What is wrong: the simulator is flow-level and static-routing-based, not realistic Internet routing/protocol simulation.  
Why it matters: oral defense and thesis wording risk.  
Files/modules involved: proposal/thesis text, `README.md`, `docs/final_submission_overview.md`, any poster/demo script.  
Done looks like: all materials describe GreenNet as an abstract flow-level simulator with explicit simplifications.

- [ ] **Clarify absolute QoS meaning in the final report.**  
What is wrong: some scenarios are labeled “acceptable” under project-defined gates despite very rough absolute delay/drop figures, which may surprise reviewers.  
Why it matters: protects against accusations of moving the goalposts.  
Files/modules involved: `greennet/qos.py`, final evaluation reports, concise report, thesis text.  
Done looks like: reviewer-facing text distinguishes internal acceptance gates from absolute raw QoS severity and explains the rationale.

- [ ] **Demote forecasting and impact predictor claims unless stronger evidence is added.**  
What is wrong: these components are real but weakly validated relative to headline claims.  
Why it matters: claim discipline.  
Files/modules involved: `greennet/forecasting.py`, `greennet/impact_predictor.py`, `models/impact_predictor/`, docs/report text.  
Done looks like: they are either presented as exploratory/supporting components or backed by stronger dedicated evaluation.

- [ ] **Make demo/generated frontend fallback unmistakable.**  
What is wrong: the React app can fall back to demo runs or generated topology/layout.  
Why it matters: reviewers may misread placeholder/generated state as real run evidence.  
Files/modules involved: `frontend/greennet-ui/src/lib/api.ts`, `frontend/greennet-ui/src/pages/DashboardPage.tsx`, `frontend/greennet-ui/src/pages/SimulatorPage.tsx`.  
Done looks like: fallback mode is impossible to miss and is never confused with canonical evidence.

- [ ] **Reduce the burden of the SQLite artifact footprint.**  
What is wrong: the DB is very large and stores many historical runs/final evaluations.  
Why it matters: portability, trust, and state leakage.  
Files/modules involved: `artifacts/db/greennet.sqlite3`, persistence docs/scripts.  
Done looks like: the submission copy either ships a curated DB or clearly excludes heavyweight/internal DB state.

### Nice-to-have polish

- [ ] **Remove root-level legacy config snapshots from the submission copy.**  
What is wrong: many old `train_*.json` and `eval_*.json` files dilute the canonical config story.  
Why it matters: cleaner reviewer navigation.  
Files/modules involved: repo root, `configs/`.  
Done looks like: reviewers see `configs/` as the only config family that matters.

- [ ] **Prune generated files and local-machine noise.**  
What is wrong: `__pycache__`, `.DS_Store`, local env directories, and duplicate `node_modules` weaken polish.  
Why it matters: professionalism.  
Files/modules involved: source tree broadly.  
Done looks like: source tree is clean and intentional.

- [x] **Add a short “reviewer path in 5 minutes” file at repo root.**  
What is wrong: the canonical story was documented, but spread across several docs.  
Why it matters: smoother board review.  
Files/modules involved: repo root docs.  
Done looks like: completed in this repair pass as `SUBMISSION_INDEX.md`.

## 8. Fastest path to submission

The smallest remaining changes that would most increase credibility now are:

1. Tighten wording around networking realism so the project is consistently presented as a flow-level abstraction rather than realistic Internet protocol simulation.
2. Clarify how project-defined QoS acceptance gates relate to rough absolute raw QoS numbers in the final report.
3. Soften or better validate forecasting / impact-predictor / cost-modeling claims so side systems are not over-read as headline contributions.
4. Decide whether the final delivered package should further reduce the checked-in SQLite/artifact footprint.

Those changes are no longer fixing broken flows; they are mainly about sharpening reviewer confidence.

## 9. If reviewed today

### What looks strong

- Real codebase, not a mock-up.
- Honest mixed-outcome reporting is present in the main docs.
- Clear effort in evaluation structure, artifact packaging, and reviewer-oriented reporting.
- Stronger engineering depth than many capstone submissions.
- Quality gates are clean and the canonical reviewer path is explicit.

### What looks weak

- Repo bloat and artifact clutter still exist even though the canonical path is now labeled.
- Some AI/forecasting/cost-modeling claims are stronger than the evidence.
- Realism and QoS interpretation still need careful verbal framing.

### What might raise questions in an oral defense

- “Is this actually PPO, or PPO plus a lot of hand-coded guardrails?”
- “Why does `latest/` not match the canonical rerun bundle if it still exists in the repo?”
- “What exactly is being simulated here: packet routing, protocol behavior, or a flow-level abstraction?”
- “Why are absolute QoS numbers so rough if results are still labeled acceptable?”
- “How much of forecasting / impact prediction is core evidence versus exploratory support?”

## 10. Appendix: evidence map

- `README.md`: current canonical project story and reviewer workflow.
- `docs/final_submission_overview.md`: final architecture, artifact glossary, limitations.
- `GreenNet_Honors_Project_Proposal.docx`: original proposal promises and hypothesis.
- `GreenNet_FirstDraft_OltaZagraxha.docx`: Stage 2 commitments and claimed architecture/methodology.
- `pyproject.toml`: packaging, dependencies, test config.
- `run_experiment.py`: real evaluation path and PPO/hybrid policy behavior.
- `greennet/env.py`: environment design, reward shaping, action masking, forecasting/cost-estimator integration.
- `greennet/simulator.py`: capacity, delay, delivery/drop model.
- `greennet/routing.py`: routing baseline semantics and honesty of abstraction.
- `greennet/topology.py`: topology model and packaged topologies.
- `greennet/traffic.py`: traffic generation and replay abstractions.
- `greennet/qos.py`: runtime and acceptance QoS logic.
- `greennet/stability.py`: switching stability logic.
- `greennet/impact_predictor.py`: cost-model / impact predictor implementation.
- `greennet/evaluation/reproduction.py`: canonical official rerun path.
- `configs/acceptance_matrices/official_acceptance_v1.json`: authoritative benchmark definition.
- `artifacts/final_pipeline/official_acceptance_v1/...`: current canonical final-pipeline outputs.
- `artifacts/final_pipeline/latest/...`: pinned historical bundle and associated explanation.
- `experiments/official_matrix_v6/...`: preserved historical evaluation bundle.
- `artifacts/models/official_acceptance_v1/...`: checked-in official PPO artifact family and metadata.
- `api_app.py`: API behavior, especially final-evaluation retrieval and run browsing.
- `tests/`: unit/integration test coverage and current failures.
- `frontend/greennet-ui/...`: official public UI.
- `dashboard/...`: retained internal analyst tooling.
- `artifacts/db/greennet.sqlite3`: DB-backed run and final-evaluation state.
