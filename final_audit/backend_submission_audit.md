# GreenNet Final Submission Audit

## Scope
This pass audits the backend/simulator/experiment pipeline only. It is evidence-based against the live repository code and shipped artifacts, with emphasis on `greennet/env.py`, simulator/routing/topology/traffic/power/QoS/stability modules, `run_experiment.py`, training entrypoints, acceptance-matrix runners, evaluation packaging, persistence, and API integration.

## 1. Executive Verdict
- Overall readiness score: 6.9/10
- Submission recommendation: Ready with minor fixes

The backend is materially implemented, not just described. There is a real simulator, real routing baselines, a real PPO evaluation path, a manifest-driven acceptance matrix, persistence into SQLite, and a reviewer-facing API. The main risks are not “missing core code,” but credibility and reviewer-trust issues: the simulator is explicitly simplified, the PPO controller is wrapped in substantial rule-based safety logic, the training pathway is thinner and less evidenced than the evaluation pathway, and there are multiple historical result trees that can blur what is canonical.

## 2. What the project clearly does well
- The simulator architecture is coherent and inspectable. `greennet/env.py` builds a topology, annotates routing costs, instantiates a `Simulator`, injects traffic, applies edge-toggle actions, computes reward components, and emits per-step diagnostics with QoS and stability summaries. This is a real end-to-end environment, not placeholder wiring.
- Routing baselines are honestly framed as forwarding baselines rather than protocol simulation. `greennet/routing.py` explicitly states that OSPF-like behavior means static additive link-cost forwarding, not control-plane dynamics. That framing is technically modest and appropriate for a capstone.
- Energy-aware behavior is integrated into the environment and the metrics pipeline. `greennet/power.py` gives a simple but explicit active/sleep/linear-utilization model; `greennet/simulator.py` converts per-step power into energy and carbon; `run_experiment.py` records energy, power, carbon, and transition metrics into CSV and JSON outputs.
- Comparative evaluation is structured, not ad hoc. `configs/acceptance_matrices/official_acceptance_v1.json` defines a canonical benchmark matrix across policies, seeds, topologies, and traffic cases; `experiments/run_matrix.py` executes it; `greennet/evaluation/final_pipeline.py` filters, packages, and persists final evaluation outputs.
- Reproducibility support is stronger than typical student projects. `greennet/evaluation/reproduction.py` enforces a one-command official path, validates dependencies, initializes SQLite, and points reviewers to canonical outputs. `run_experiment.py` persists `env_config.json`, `run_meta.json`, `summary.json`, and `per_step.csv` per run.
- The API is more than a demo shell. `api_app.py` reads both filesystem artifacts and SQLite-backed snapshots, exposes run lists, summaries, topology/timeline endpoints, and a final-evaluation endpoint, with integration tests covering the expected shapes.

## 3. Requirement-by-requirement evaluation

### Proposal promises

| Item | Verdict | Why |
| --- | --- | --- |
| Routing/network simulator | PASS | `greennet/simulator.py` enforces per-edge capacities, path-based delivery scaling, utilization, and delay; `greennet/env.py` wraps it as a controllable RL environment. |
| AI-driven routing / optimization | PARTIAL | There is real PPO inference and training plumbing, but the “AI” policy in evaluation is a hybrid controller: PPO output is post-processed by rule-based safety and calm-off logic in `run_experiment.py:791-866`. This is AI-assisted edge-toggle control, not pure learned optimization. |
| Energy-aware behavior | PASS | Energy, power, carbon, and transition costs are implemented in `greennet/power.py`, `greennet/simulator.py`, and surfaced in run summaries and final evaluation artifacts. |
| Dashboard / visualization | PARTIAL | Backend/API support is real, but this pass did not validate frontend behavior end-to-end. From the backend side, the API appears sufficient; presentation quality should be judged separately. |
| Comparative evaluation | PASS | Manifest-driven matrix execution exists via `configs/acceptance_matrices/official_acceptance_v1.json`, `experiments/run_matrix.py`, and `greennet/evaluation/final_pipeline.py`. |
| Documented methodology / research framing | PARTIAL | The code contains many policy signatures and metadata hooks, but methodology clarity depends heavily on docs outside this audit scope. Backend naming is mostly honest, but reviewers still need a clean canonical path explanation. |
| Hypothesis-driven results | PARTIAL | `greennet/evaluation/final_pipeline.py` and shipped final-evaluation JSONs encode hypothesis thresholds and classification. However, the repo also contains older matrices and legacy summary files, so the canonical hypothesis result is not instantly obvious without guidance. |

### Stage 2 / first-draft commitments

| Item | Verdict | Why |
| --- | --- | --- |
| System architecture | PASS | The architecture is consistent: environment -> simulator/routing/traffic/power/QoS/stability -> experiment runner -> summary -> final pipeline -> SQLite/API. |
| Methodology | PARTIAL | The evaluation methodology is codified in manifests and summary logic, but the modeling assumptions remain simple and some terms like “official” rely on packaging discipline rather than a mathematically richer simulator. |
| Testing framework | PASS | There are unit and integration tests across simulator, routing, traffic, QoS, stability, impact predictor, run experiment, API, and reproduction paths. |
| Reproducibility | PASS | Strong by capstone standards: `run_experiment.py` writes full artifacts; `reproduction.py` offers a one-command official path; SQLite persistence exists. |
| Evaluation structure | PASS | Acceptance matrix schema, policy taxonomy, final pipeline, and final evaluation packaging are all implemented. |
| Dashboard/backend integration | PASS | `api_app.py` serves run summaries, timeline/topology data, and final evaluation payloads, and integration tests validate these endpoints. |
| RL / forecasting / cost-modeling claims | PARTIAL | Forecasting is implemented as lightweight online heuristics in `greennet/forecasting.py`, not a learned forecasting subsystem. The impact predictor exists and is integrated as a gating mechanism, but it is optional and currently disabled by default in evaluation settings. PPO exists, but evaluation wraps it with hand-coded safety logic. |

### Final-review expectations

| Item | Verdict | Why |
| --- | --- | --- |
| Technical soundness | PARTIAL | The code is internally coherent and better than a prototype script pile, but the networking and energy models are deliberately simplified. Acceptable for a capstone if presented modestly. |
| Ethical / responsible design | PASS | The backend explicitly protects QoS and stability through masks, vetoes, thresholds, cooldowns, and acceptance summaries. It does not blindly optimize energy at any cost. |
| Communication of complex ideas | PARTIAL | The code contains the right metadata and canonical paths, but the repository has many overlapping result directories (`experiments/official_matrix_*`, `artifacts/final_pipeline/*`, `results/*`, `_exports/*`) that can confuse reviewers. |

## 4. Honesty / overclaim / inconsistency audit

### Claims that are fully supported
- GreenNet has a functioning simulator environment with traffic, routing, power, and QoS instrumentation.
- GreenNet compares an always-on baseline, a heuristic energy-aware baseline, and a PPO-based controller.
- GreenNet has a canonical acceptance matrix and a final evaluation packaging pipeline.
- GreenNet persists runs to both file artifacts and SQLite, and exposes reviewer-facing API endpoints.

### Claims that are only partially supported
- “AI-driven optimization” is only partially supported as a standalone claim. The PPO path is real, but it is wrapped in deterministic controller logic in `run_experiment.py:791-866`, so the final evaluated behavior is hybrid rather than purely learned.
- “Forecasting” is implemented, but only as online EMA/Holt/adaptive-EMA demand forecasting in `greennet/forecasting.py`. If documents imply a trained forecaster, that would overstate what is present here.
- “Impact predictor / cost modeling” is implemented as an optional PyTorch ensemble in `greennet/impact_predictor.py` and environment gating in `greennet/env.py`, but it is not the dominant canonical evaluation path because `cost_estimator_enabled` is `False` by default and the tested PPO evaluation path explicitly expects it off in `tests/integration/test_run_experiment.py:84-95`.

### Claims that should be softened or rewritten
- Any wording that implies real-world routing protocol realism should be softened. `greennet/routing.py:120-124` explicitly says this is forwarding-time route selection, not OSPF control-plane simulation.
- Any wording that implies GreenNet proves deployable internet-scale energy savings should be softened. The power model is thesis-friendly and explicit, but still a lightweight abstraction in `greennet/power.py`.
- Any wording that presents PPO as an unconstrained learned controller should be rewritten to “PPO-based controller evaluated under rule-based QoS/stability safety constraints.”

### Wording likely to mislead reviewers
- Calling `ospf_ecmp` “OSPF” without the forwarding-only caveat.
- Describing the impact predictor or forecaster as core final-system intelligence unless the final submission makes clear they are auxiliary and, in the official path, not the primary differentiator.
- Describing the acceptance results as obviously canonical while leaving older matrices like `experiments/official_matrix_v1`, `v2`, `v4`, `v6`, and `official_matrix_accept` alongside them without a clear reviewer note.

## 5. Reproducibility and reviewer experience audit
- Reviewer run experience is above average if they follow the intended canonical path. `experiments/run_official_acceptance_matrix.py` delegates to `greennet/evaluation/reproduction.py`, which checks dependencies, requires canonical manifests, initializes SQLite, and routes into the final pipeline.
- There is one canonical manifest-backed path, but the repo still contains several competing “official” looking outputs. `configs/acceptance_matrices/official_acceptance_v1.json` is the cleanest authority; older `experiments/official_matrix_v*` folders are useful history but create review noise.
- Outputs are verifiable. Each experiment run writes `env_config.json`, `run_meta.json`, `per_step.csv`, and `summary.json`, and `persist_run_directory()` attempts SQLite ingestion afterward.
- Artifact trustworthiness is decent, but mixed. The current repo contains both fresh canonical assets and older summary CSVs with very different scales and settings, for example `experiments/official_matrix_accept/results_summary_matrix_accept.csv` is clearly an older tiny run shape. Without a reviewer note, this can undermine confidence.

## 6. Code and architecture audit

### Strong architectural decisions
- Clear separation of concerns: topology, routing, traffic, simulator, power, QoS, stability, persistence, evaluation, and API are distinct modules.
- Manifest-driven experiment structure is strong. Acceptance matrix config is not hard-coded into bash scripts.
- The environment emits rich step metadata, enabling both run summaries and downstream API visualization without recomputation.
- SQLite persistence is optional at write time but additive, preserving file artifacts as the authoritative fallback.

### Weak spots
- `greennet/env.py` and `run_experiment.py` are large and policy-heavy. They now contain a lot of intertwined environment logic, controller constraints, reward shaping, safety vetoes, forecasting, optional impact gating, and metadata handling. This is workable, but hard to explain cleanly in an oral defense.
- The PPO evaluation path is not a clean pure-policy benchmark. `load_policy()` returns `_action_ppo_safe`, which can override or suppress PPO choices based on masks, heuristics, and calm-off logic. That improves safety, but weakens clean attribution of results to learning alone.
- The training entrypoint is thin and the canonical evidence is stronger for evaluation than for training. `train.py` is just a shim, and the real training story lives in `greennet/cli/train_cli.py`, which is substantial but not as tightly tied into the official evaluation path as the reproduction pipeline is.

### Duplication / confusing areas
- `_exports/` contains large historical copies of the repo. `git status` shows deletion churn in `_exports/GreenNet_progress_review_bundle_20260410` and new `_exports/GreenNet_progress_review_bundle_20260411`, which is likely reviewer noise rather than core project value.
- There are multiple result families with “official” naming: `experiments/official_matrix_accept`, `experiments/official_matrix_v6`, `artifacts/final_pipeline/official_acceptance_v1`, and shipped verification folders under `final_audit/verification`.
- Several helper functions are duplicated conceptually across training/evaluation entrypoints, especially config extraction and model discovery.

### Fragile or jury-rigged areas
- The environment forcibly ensures the action-edge universe exists on the graph in `greennet/env.py:561-564` and `731-738`. This stabilizes action indexing, but it also means evaluation graphs can contain edge-universe padding rather than only original topology edges, which deserves clear explanation.
- Random topology support and fixed action spaces are handled with compatibility logic that is technically pragmatic but conceptually messy for a thesis-grade narrative.
- SQLite persistence is best-effort from `run_experiment.py:1723-1727`. That is the right operational choice, but it means DB state can lag filesystem artifacts.

## 7. Final checklist of improvements

### Critical before submission
- [ ] Add one reviewer-facing backend architecture note that explicitly states the modeling level.
Why it matters: reviewers need to know this is a capacity-and-toggle forwarding simulator, not a packet-level or protocol-control-plane simulator.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/greennet/routing.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/routing.py) conceptually; canonical reviewer docs should point to it.
Done looks like: a short “modeling assumptions” section naming static forwarding, simplified power model, and QoS/stability guards.

- [ ] Clarify in submission materials that the evaluated PPO controller is hybrid and safety-constrained.
Why it matters: otherwise the repo risks overstating what the learning component alone achieved.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/run_experiment.py`](file:///Users/enionismaili/Desktop/GreenNet/run_experiment.py) around `load_policy()` and `_action_ppo_safe`.
Done looks like: docs/report language changed to “PPO-based controller with rule-based QoS/stability safeguards,” not “pure RL optimizer.”

- [ ] Declare one canonical official result path and demote legacy matrices to archive/history status.
Why it matters: reviewer trust drops when multiple folders all look official.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json`](file:///Users/enionismaili/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json), [`/Users/enionismaili/Desktop/GreenNet/greennet/evaluation/reproduction.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/evaluation/reproduction.py), [`/Users/enionismaili/Desktop/GreenNet/experiments`](file:///Users/enionismaili/Desktop/GreenNet/experiments).
Done looks like: reviewers are told exactly which manifest, command, output folder, and report file matter.

- [ ] Remove or clearly label stale export trees and historical bundle copies from the submission-facing repo.
Why it matters: `_exports/` and overlapping historical bundles create the appearance of disorder and can send reviewers down the wrong path.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/_exports`](file:///Users/enionismaili/Desktop/GreenNet/_exports).
Done looks like: archive paths are excluded from the final submitted repo or labeled as non-canonical.

### Important but not blocking
- [ ] Add a backend evidence note for the optional impact predictor and forecasting components.
Why it matters: both exist, but they are easy to overclaim.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/greennet/impact_predictor.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/impact_predictor.py), [`/Users/enionismaili/Desktop/GreenNet/greennet/forecasting.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/forecasting.py), [`/Users/enionismaili/Desktop/GreenNet/greennet/env.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/env.py).
Done looks like: final materials explicitly say forecasting is online heuristic forecasting and impact prediction is an optional learned guard, not the main evaluation claim.

- [ ] Tighten the narrative around baseline naming.
Why it matters: `all_on` vs `noop`, and `heuristic` vs `baseline`, are canonically normalized in code, but a reviewer reading outputs may still think there are more distinct baselines than there are.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/greennet/policy_taxonomy.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/policy_taxonomy.py), [`/Users/enionismaili/Desktop/GreenNet/tests/integration/test_run_experiment.py`](file:///Users/enionismaili/Desktop/GreenNet/tests/integration/test_run_experiment.py).
Done looks like: one table in docs mapping requested names to canonical experimental roles.

- [ ] Explain why SQLite is secondary and file artifacts remain authoritative.
Why it matters: the code intentionally uses best-effort DB persistence, and reviewers may otherwise expect the DB to be the single source of truth.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/run_experiment.py`](file:///Users/enionismaili/Desktop/GreenNet/run_experiment.py), [`/Users/enionismaili/Desktop/GreenNet/greennet/persistence/sqlite_store.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/persistence/sqlite_store.py).
Done looks like: docs say file outputs are primary artifacts; SQLite is an index/query layer.

### Nice-to-have polish
- [ ] Refactor oversized control logic out of `run_experiment.py` and `greennet/env.py` after submission.
Why it matters: maintainability and oral-defense clarity.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/run_experiment.py`](file:///Users/enionismaili/Desktop/GreenNet/run_experiment.py), [`/Users/enionismaili/Desktop/GreenNet/greennet/env.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/env.py).
Done looks like: policy wrappers, safety controller logic, and metadata packaging live in smaller modules.

- [ ] Expand invariant-level tests around final evaluation semantics, not just endpoint/file shapes.
Why it matters: many integration tests validate successful wiring, but fewer validate scientific claims.
Files/modules: [`/Users/enionismaili/Desktop/GreenNet/tests/integration`](file:///Users/enionismaili/Desktop/GreenNet/tests/integration).
Done looks like: tests assert policy-class labeling, canonical filtering, and hypothesis status logic against controlled fixtures.

## 8. Fastest path to submission
The fastest credibility gain is not more code. It is tightening the explanation layer around what already exists:

1. State the simulator assumptions clearly.
2. State that PPO is evaluated with hard safety constraints and hybrid override logic.
3. Mark `official_acceptance_v1` as the only canonical final benchmark path.
4. Remove or quarantine stale export and legacy “official” folders from the submission package.

If you do only those four things, the backend will read as deliberate and defensible rather than sprawling.

## 9. If reviewed today
- What looks strong: a real environment, real evaluation machinery, real artifact generation, and unusual attention to QoS/stability guardrails for a student project.
- What looks weak: the system is more heuristic-heavy and more simplified than “AI sustainable routing simulator” language can imply.
- What raises oral-defense questions: why PPO needs hybrid safety logic, how realistic the routing/power model is, whether the impact predictor is central or optional, and which result folder is the true final one.

## 10. Appendix: evidence map
- [`/Users/enionismaili/Desktop/GreenNet/greennet/env.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/env.py): core environment, action masking, reward, QoS/stability integration, optional impact predictor and forecasting.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/simulator.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/simulator.py): actual delivery/drop/utilization/delay/energy simulation step.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/routing.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/routing.py): baseline routing taxonomy and honesty of forwarding-only modeling.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/topology.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/topology.py): packaged topology loading and validation.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/traffic.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/traffic.py): stochastic and replay traffic generation.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/power.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/power.py): explicit simplified energy model.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/qos.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/qos.py): runtime and acceptance QoS evaluation logic.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/stability.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/stability.py): stability policy metadata and post-run stability evaluation.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/forecasting.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/forecasting.py): online heuristic forecasting, not offline ML forecasting.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/impact_predictor.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/impact_predictor.py): optional learned guard model with ensemble uncertainty.
- [`/Users/enionismaili/Desktop/GreenNet/run_experiment.py`](file:///Users/enionismaili/Desktop/GreenNet/run_experiment.py): main experiment runner, policy loading, PPO safety wrapper, summary generation, artifact writing, SQLite persistence.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/cli/train_cli.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/cli/train_cli.py): real training path behind thin `train.py`.
- [`/Users/enionismaili/Desktop/GreenNet/experiments/run_matrix.py`](file:///Users/enionismaili/Desktop/GreenNet/experiments/run_matrix.py): matrix execution and result harvesting.
- [`/Users/enionismaili/Desktop/GreenNet/experiments/run_official_acceptance_matrix.py`](file:///Users/enionismaili/Desktop/GreenNet/experiments/run_official_acceptance_matrix.py): official wrapper into reproduction flow.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/evaluation/reproduction.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/evaluation/reproduction.py): canonical one-command reproduction path.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/evaluation/final_pipeline.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/evaluation/final_pipeline.py): packaging and final-evaluation assembly.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/evaluation/acceptance_matrix.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/evaluation/acceptance_matrix.py): acceptance-manifest schema and validation.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/evaluation/official_ppo.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/evaluation/official_ppo.py): topology-specific official PPO checkpoint family logic.
- [`/Users/enionismaili/Desktop/GreenNet/greennet/persistence/sqlite_store.py`](file:///Users/enionismaili/Desktop/GreenNet/greennet/persistence/sqlite_store.py): SQLite schema, run ingestion, final-evaluation persistence, snapshot queries.
- [`/Users/enionismaili/Desktop/GreenNet/api_app.py`](file:///Users/enionismaili/Desktop/GreenNet/api_app.py): backend API and reviewer-facing aggregation endpoints.
- [`/Users/enionismaili/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json`](file:///Users/enionismaili/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json): canonical final matrix definition.
- [`/Users/enionismaili/Desktop/GreenNet/tests`](file:///Users/enionismaili/Desktop/GreenNet/tests): unit/integration evidence for simulator, routing, traffic, QoS, stability, API, and reproduction.
