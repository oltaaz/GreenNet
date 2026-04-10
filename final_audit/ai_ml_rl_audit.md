# GreenNet AI/ML/RL Validity Audit

## Executive Verdict

Verdict: `PARTIAL / CAUTION`

Bottom line:
- PPO is real code, trainable, and integrated into the simulator.
- Forecasting is real but lightweight feature engineering, not a substantial predictive modeling contribution.
- The impact predictor / cost-modeling stack is real but exploratory and weakly validated.
- The most important honesty issue is that the reviewer-facing `ppo` evaluation path is not pure PPO. It is a hybrid controller with hand-coded safety and calm-off logic, and it can load the impact predictor opportunistically during evaluation.

If the repo is presented as:
- "an energy-aware simulator with a learned controller evaluated against explicit baselines" -> defensible.
- "AI-driven routing with PPO, forecasting, and cost modeling proving AI optimization" -> overstated.

## Scope Judged

This audit focused on:
- PPO training and evaluation integration
- observation / action / reward design
- `greennet/forecasting.py`
- `greennet/impact_predictor.py`
- cost-estimator training/data scripts and model artifacts
- configs, tests, and documentation claims touching AI/ML/RL

## What Is Real

### 1. PPO integration is real

Verified evidence:
- Training entrypoint is real in [greennet/cli/train_cli.py](/Users/enionismaili/Desktop/GreenNet/greennet/cli/train_cli.py).
- It constructs `GreenNetEnv`, uses SB3 `PPO`, and supports `MaskablePPO` via `ActionMasker`.
- The environment is a real Gymnasium env in [greennet/env.py](/Users/enionismaili/Desktop/GreenNet/greennet/env.py).
- Official checkpoint-family packaging exists in [greennet/evaluation/official_ppo.py](/Users/enionismaili/Desktop/GreenNet/greennet/evaluation/official_ppo.py).
- Canonical reviewer-facing PPO artifacts exist under [artifacts/models/official_acceptance_v1](/Users/enionismaili/Desktop/GreenNet/artifacts/models/official_acceptance_v1).

Assessment:
- `AI-driven controller` claim: `REAL`
- `PPO training/eval pipeline exists`: `REAL`

### 2. Observation / action / reward design is real and non-trivial

Verified evidence from [greennet/env.py](/Users/enionismaili/Desktop/GreenNet/greennet/env.py):
- Action space is a discrete link-toggle controller over a stable edge universe.
- Observation is a `spaces.Dict` with scalar network state plus per-edge activity/utilization.
- Reward has explicit components for energy, drops, QoS violation, and toggling.
- Action masking is extensive and tied to cooldowns, budgets, utilization, QoS stress, forecast gates, and optional cost-estimator gating.

Assessment:
- This is not a fake RL wrapper around static code.
- It is a heavily constrained control problem with substantial hand-designed shaping and guardrails.

Implication:
- The AI is not learning routing paths directly.
- It is learning network link-state control over a fixed routing baseline.

Claim status:
- `AI-driven routing` -> `PARTIAL`, only if phrased carefully as controller-level routing/energy management over a fixed baseline.
- `RL controller for energy/QoS tradeoffs` -> `SUPPORTED`

## Major Validity Issues

### 1. Reviewer-facing `ppo` is not pure PPO

This is the single biggest claim-risk issue.

Verified evidence:
- [run_experiment.py](/Users/enionismaili/Desktop/GreenNet/run_experiment.py) defines `_action_ppo_safe` and returns metadata with `control_mode = "ppo_safe_hybrid_calm_off"`.
- The wrapper applies:
  - PPO proposal
  - mask checks
  - heuristic recovery logic
  - heuristic calm-off selection
  - fallback to heuristic behavior when PPO proposals are disallowed
- `_calm_off_action` calls `_ensure_controller_predictor`, which can load `models/impact_predictor` even when the env config has `cost_estimator_enabled = false`.

Why it matters:
- The label `ppo` in reports can be interpreted by reviewers as "policy outputs come directly from the trained PPO controller."
- In reality, the evaluated controller is a hybrid policy stack with hand-coded override logic.

Assessment:
- `PPO is evaluated as a pure learned controller` -> `FAIL`
- `PPO is evaluated inside a hybrid safety wrapper` -> `SUPPORTED`

Required wording fix:
- Do not describe the final evaluated policy as raw PPO.
- Describe it as a `PPO-based hybrid controller with rule-based safety/override logic`.

### 2. Official PPO artifact lineage is inconsistent with the docs

Verified evidence:
- Docs claim the official regeneration path is `configs/train_official_ppo.json` at `100000` timesteps in [README.md](/Users/enionismaili/Desktop/GreenNet/README.md) and [configs/README.md](/Users/enionismaili/Desktop/GreenNet/configs/README.md).
- Actual canonical checkpoint metadata in [artifacts/models/official_acceptance_v1/small/checkpoint_metadata.json](/Users/enionismaili/Desktop/GreenNet/artifacts/models/official_acceptance_v1/small/checkpoint_metadata.json) and its `medium`/`large` peers point to:
  - `config_path = configs/train_normal.json`
  - `total_timesteps = 25000`
- The stored env configs for those artifacts use `forecast_model = "ema"`, not the `adaptive_ema` claimed in [configs/train_official_ppo.json](/Users/enionismaili/Desktop/GreenNet/configs/train_official_ppo.json).

Why it matters:
- This is a direct reproducibility and claim-honesty problem.
- A reviewer cannot tell which PPO configuration actually produced the official results unless they inspect artifact metadata.

Assessment:
- `official PPO artifact matches documented canonical config` -> `FAIL`
- `official PPO artifact is reproducibly identified` -> `PARTIAL`

### 3. Forecasting is real, but it is lightweight and not a demonstrated ML contribution

Verified evidence:
- [greennet/forecasting.py](/Users/enionismaili/Desktop/GreenNet/greennet/forecasting.py) implements:
  - EMA
  - damped Holt trend
  - adaptive EMA ensemble
- The env uses `demand_now` and `demand_forecast` in observations and mask gating.
- Unit tests in [tests/unit/test_forecasting.py](/Users/enionismaili/Desktop/GreenNet/tests/unit/test_forecasting.py) verify mechanics and one synthetic regime-shift advantage.

Limitations:
- No offline training pipeline exists for forecasting because it is not a trained model family.
- There is no serious ablation showing forecast features materially improve PPO or heuristic outcomes on the official benchmark.
- The official checkpoint artifacts currently use `forecast_model = "ema"`, not `adaptive_ema`.

Assessment:
- `forecasting exists in code` -> `PASS`
- `forecasting is integrated into env state and gates` -> `PASS`
- `forecasting is a validated project contribution` -> `PARTIAL`
- `forecasting as an ML contribution` -> `WEAK / OVERSTATED if highlighted`

Recommended framing:
- `lightweight online demand forecasting features`
- not `advanced forecasting module` unless backed by dedicated evaluation

### 4. Impact predictor / cost modeling is exploratory and weakly validated

Verified evidence:
- Runtime model loader is real in [greennet/impact_predictor.py](/Users/enionismaili/Desktop/GreenNet/greennet/impact_predictor.py).
- Dataset generation exists in:
  - [scripts/build_cost_dataset_graph.py](/Users/enionismaili/Desktop/GreenNet/scripts/build_cost_dataset_graph.py)
  - [scripts/build_cost_dataset_sweep.py](/Users/enionismaili/Desktop/GreenNet/scripts/build_cost_dataset_sweep.py)
- Training exists in [scripts/train_cost_estimator_torch.py](/Users/enionismaili/Desktop/GreenNet/scripts/train_cost_estimator_torch.py).
- Comparison script exists in [scripts/eval_impact_predictor_compare.py](/Users/enionismaili/Desktop/GreenNet/scripts/eval_impact_predictor_compare.py).
- Model artifacts exist in [models/impact_predictor](/Users/enionismaili/Desktop/GreenNet/models/impact_predictor).

But the validation quality is weak:
- `models/impact_predictor/meta.json` reports `qos_auc = 0.5799`, which is only slightly above random.
- Training args show only `epochs = 1`.
- Validation split mode is `random`, not topology/scenario-grouped, which weakens generalization claims.
- The checked-in report is only aggregate-level and does not demonstrate strong scenario robustness.

Additional concern:
- There are multiple model families:
  - `models/impact_predictor`
  - `models/cost_estimator_gnnlite`
  - legacy `models/cost_estimator/*.joblib`
- This creates reviewer confusion about which artifact is canonical.

Assessment:
- `cost-modeling code exists` -> `PASS`
- `cost-modeling is integrated as an optional guard` -> `PASS`
- `cost-modeling is strong enough to headline as a validated contribution` -> `FAIL`
- `impact predictor should be presented as exploratory` -> `PASS`

### 5. The RL problem is valid, but heavily hand-shaped

Verified evidence from [greennet/env.py](/Users/enionismaili/Desktop/GreenNet/greennet/env.py):
- Reward = energy penalty + drop penalty + QoS penalty + toggle penalties.
- Multiple non-learning constraints shape behavior:
  - cooldowns
  - budget caps
  - util thresholds
  - QoS guards
  - OFF gating by calm demand / forecast
  - optional cost-estimator veto

Why it matters:
- This is acceptable for a capstone controller problem.
- But it means gains are not attributable to PPO alone.
- The environment design contributes heavily to policy behavior.

Assessment:
- `well-shaped constrained RL environment` -> `PASS`
- `policy success can be attributed cleanly to learning rather than extensive control scaffolding` -> `PARTIAL`

## Tests and Reproducibility

Verified:
- `.venv/bin/python -m pytest -q tests/unit/test_forecasting.py tests/unit/test_impact_predictor.py tests/unit/test_official_ppo.py`
- Result: `7 passed`

What this does prove:
- forecasting code paths work
- impact predictor loader/aggregation works
- official PPO packaging helpers work

What it does not prove:
- PPO training quality
- official PPO performance validity
- forecasting benefit on benchmark outcomes
- impact predictor usefulness in final evaluation

Assessment:
- `mechanical coverage exists` -> `PASS`
- `scientific evidence for AI component quality exists` -> `PARTIAL`

## Claim-by-Claim Judgment

### PPO / AI-driven routing

- `There is a real PPO-based learned controller.` -> `SUPPORTED`
- `The final evaluated policy is pure PPO.` -> `NOT SUPPORTED`
- `The AI directly performs routing.` -> `PARTIAL`
  The learned action is link-state control over a fixed routing baseline, not path-computation learning.
- `The AI meaningfully beats strong non-AI baselines.` -> `NOT ESTABLISHED HERE`
  Repo docs themselves state PPO does not beat the heuristic on energy in the final aggregate.

### Forecasting

- `Forecasting is implemented and used.` -> `SUPPORTED`
- `Forecasting is a major validated contribution.` -> `NOT SUPPORTED`
- `Adaptive forecasting is part of the official reviewer-facing PPO artifact.` -> `NOT SUPPORTED by current artifact metadata`

### Cost modeling / impact prediction

- `A learned impact predictor exists.` -> `SUPPORTED`
- `It is integrated into decision-making paths.` -> `SUPPORTED`
- `It is strong enough to present as a validated model.` -> `NOT SUPPORTED`
- `It should be described as exploratory.` -> `SUPPORTED`

## Overclaim Risks

These phrasings are risky and should be softened:

- `AI-driven routing`
  Better: `PPO-based link-state controller over a fixed routing baseline`

- `PPO policy`
  Better for final evaluation: `PPO-based hybrid controller`

- `forecasting module improves optimization`
  Better: `forecast features are available and integrated, but contribution should be treated as exploratory unless ablations are shown`

- `cost model` or `impact predictor` as if production-grade
  Better: `exploratory learned guard model`

- Any implication that official PPO was regenerated with `configs/train_official_ppo.json` for `100000` steps
  Current checked-in canonical artifact metadata contradicts that.

## Final Classification

### Fully implemented

- PPO training code and env integration
- Dict observation space and discrete action space
- reward shaping and action masking
- online forecasting utilities
- impact predictor runtime loader
- cost-dataset and training scripts
- basic unit tests for forecasting / impact predictor / official PPO packaging

### Partially implemented / partially validated

- AI-driven routing claim
- forecasting as a meaningful research contribution
- cost modeling as a meaningful research contribution
- official reproducible PPO lineage
- clean attribution of gains to learning rather than wrapper logic

### Present but weak

- impact predictor predictive quality
- cost-model validation protocol
- evidence that forecasting helps
- documentation consistency around official PPO artifacts

### Misleading if not rewritten

- presenting evaluated `ppo` as raw PPO
- presenting official PPO artifacts as if they come from `train_official_ppo.json` at `100000` steps
- presenting forecasting or impact prediction as core validated breakthroughs

## Required Corrections Before Submission

1. Re-label the evaluated `ppo` policy everywhere the final report or README implies it is raw PPO.
   Done means: reviewer-facing docs explicitly state the final evaluated controller is a hybrid PPO + rule-based safety layer.

2. Resolve the canonical artifact/config mismatch.
   Done means: either regenerate the official PPO family from the documented config, or rewrite docs to match the actual checked-in artifacts.

3. Demote forecasting and impact prediction from headline contributions unless ablations/evidence are added.
   Done means: they are described as auxiliary or exploratory components.

4. Clarify that learning acts on link toggling under a fixed routing baseline.
   Done means: no reviewer can mistake this for learned packet-routing/path selection.

## Final Judgment

The AI layer is real, but it is not cleanly presentable as a pure PPO routing breakthrough.

Most defensible final-positioning:
- GreenNet includes a real constrained RL controller for energy/QoS-aware link-state control.
- The project honestly shows mixed AI results rather than claiming a decisive AI win.
- Forecasting and impact prediction exist, but should be framed as exploratory support components.

Least defensible final-positioning:
- claiming strong AI-driven routing optimization powered by PPO, forecasting, and cost modeling without qualifying the hybrid control stack and the weak auxiliary-model evidence.
