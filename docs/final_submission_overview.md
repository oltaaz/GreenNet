# GreenNet Final Submission Overview

This note is the compact reviewer guide for the final submission. It is written to match the evidence actually present in the repository.

## 1. Official Workflow

1. Install the project in the verified `python3.12` environment path:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -e '.[test,train]'
   ```
2. Train or rerun policies using `train.py` and the canonical configs in `configs/`.
3. Run single experiments with `run_experiment.py` or larger comparisons with `experiments/run_matrix.py`.
4. Run the canonical reviewer-facing reproduction command:

   ```bash
   .venv/bin/python experiments/run_official_acceptance_matrix.py
   ```
   The current official PPO artifact is a topology-specific family under `artifacts/models/official_acceptance_v1/`, not the historical single checkpoint in `runs/20260220_111755/`. If those reviewer-facing checkpoints are missing, regenerate them with:

   ```bash
   .venv/bin/python experiments/regenerate_official_ppo_checkpoint.py --all-topologies --timesteps 25000
   ```
5. Verify the final repo state with `.venv/bin/python -m pytest -q` and `cd frontend/greennet-ui && npm ci && npm run build`.
6. Use `artifacts/traffic_verify/20260220_matrix/` and `artifacts/locked/` for reproducibility evidence.
7. Use `artifacts/final_pipeline/official_acceptance_v1/metadata/acceptance_matrix_manifest.json` and `pipeline_config.json` as the reviewer-facing map for the final runnable claim.
8. Treat `artifacts/db/greennet.sqlite3` as the primary structured store for indexed official runs and persisted final-evaluation payloads; use the CSV/JSON/Markdown artifacts as reviewer-facing exports.

Short note on the pinned canonical bundle:

- the reviewer-facing bundle currently pinned under `artifacts/final_pipeline/latest/` reflects the preserved historical `~1.49%` PPO result
- it is historically pinned for review continuity and is not fully reproducible from the current code/checkpoint state
- the live one-command rerun path still reproduces the official benchmark flow, but not that exact historical bundle

## 2. What The System Actually Contains

- simulator core: routing, topology, delay, energy, and carbon accounting
- official traditional baseline controller: `all_on`
- energy-aware heuristic baseline controller: `heuristic`
- learned controller: `ppo`
- API layer: `api_app.py` plus the persistence layer under `greennet/`
- public demo: `frontend/greennet-ui/`
- internal analyst tooling: `dashboard/`

The routing layer is separate from the controller layer. The simulator's default traditional routing baseline is `min_hop_single_path` with `unit` link cost, while the controller comparison is run over `all_on`, `heuristic`, and `ppo`.

The PPO artifact should also be described honestly:

- the current env uses topology-dependent Dict observations
- the historical checkpoint at `runs/20260220_111755/ppo_greennet.zip` expected an older observation layout and is not compatible with the current codebase
- the official reviewer-facing PPO artifact is now a three-checkpoint family:
  - `artifacts/models/official_acceptance_v1/small/ppo_greennet.zip`
  - `artifacts/models/official_acceptance_v1/medium/ppo_greennet.zip`
  - `artifacts/models/official_acceptance_v1/large/ppo_greennet.zip`
- the official reproduction command resolves that family automatically per topology class

The energy model should be described as a lightweight accounting model, not as a hardware-calibrated measurement stack:

- links and devices have active and sleep power states
- total power is expressed as fixed plus dynamic power
- utilization sensitivity is optional through `power_utilization_sensitive`
- link transition energy is optional through `power_transition_on_joules` and `power_transition_off_joules`
- carbon is derived from total energy via the configured carbon-intensity profile

This remains backward-compatible with the legacy `power_*` and `carbon_*` config fields. If the optional transition-cost fields are absent, the transition-energy contribution is zero.

The QoS model should also be described explicitly:

- the env-level `qos_violation` signal is driven by cumulative normalized delivery loss, not by an opaque SLA oracle
- the official runtime loss threshold is `qos_target_norm_drop=0.072` once at least `qos_min_volume=500` traffic units have been observed
- the current architecture also uses an average-delay guard relative to path latency:
  `avg_delay_ms <= max(avg_path_latency_ms * 4.0, avg_path_latency_ms + 15.0)`
- p95 delay is not part of the official rule yet because the simulator and reporting stack do not compute it today
- the final evaluation bundle applies a second, baseline-relative QoS acceptance rule against the official traditional baseline `all_on`

The switching-stability model should now also be described explicitly:

- only links are directly switchable; node/device sleep is derived from incident active links
- the official stability controls are decision-step gating, per-edge cooldown, global OFF cooldown, calm-streak gating, and per-episode toggle budgets
- the model also applies an explicit reversal penalty when the same edge is toggled back in the opposite direction within the configured flap window
- a transition is any successful applied link state change
- a flap event is an opposite-direction transition on the same edge within the reversal window
- the final evidence bundle now reports `transition_rate`, `flap_rate`, and `stability_status`, and uses an operational status that requires both the existing energy/QoS gate and stable switching behavior

## 3. Honest Final Claim

The final matrix does not support a strong "AI wins on energy" claim.

The correct submission claim is:

- GreenNet demonstrates a reproducible energy-vs-QoS routing evaluation framework
- the official traditional baseline is `all_on`
- the heuristic baseline remains the strongest handcrafted non-AI controller in the official acceptance matrix
- PPO is the best AI policy in the final aggregate, but it does not beat the heuristic on energy
- the research value is the simulator, the comparison framework, and the clear reporting of a mixed outcome

## 4. Evidence Manifest

| Artifact | Role |
| --- | --- |
| `configs/acceptance_matrices/official_acceptance_v1.json` | authoritative benchmark definition |
| `artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv` | authoritative acceptance-matrix summary |
| `artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_by_seed_official_acceptance_v1.csv` | grouped seed traceability for the acceptance matrix |
| `artifacts/final_pipeline/official_acceptance_v1/summary/leaderboard_official_acceptance_v1.csv` | ranked comparison table |
| `artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_report.md` | final thesis-facing evaluation and hypothesis gate |
| `artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.csv` | machine-readable final summary |
| `artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.json` | machine-readable final summary |
| `final_audit/verification/final_evaluation_smoke_recheck/final_evaluation_report.md` | supplemental machine recheck proving the report regenerates in this environment |
| `artifacts/final_pipeline/official_acceptance_v1/metadata/acceptance_matrix_manifest.json` | copied runnable benchmark manifest |
| `artifacts/final_pipeline/official_acceptance_v1/metadata/pipeline_config.json` | exact final-pipeline invocation state |
| `artifacts/final_pipeline/official_acceptance_v1/report/reviewer_start_here.md` | reviewer-facing entry file created by the one-command reproduction path |
| `artifacts/final_pipeline/official_acceptance_v1/report/concise_report.md` | short final-claim summary |
| `artifacts/db/greennet.sqlite3` | primary structured store for indexed runs, summary identity, and persisted final-evaluation payloads |
| `artifacts/traffic_verify/20260220_matrix/matrix_status.md` | deterministic traffic verification status |
| `artifacts/traffic_verify/20260220_matrix/traffic_eval_summary.csv` | traffic verification results |
| `artifacts/locked/` | locked scenario evidence and retained run bundles |

This manifest is intentionally narrow. It does not pretend that every historical run or experimental branch is part of the final submission story.
The historical `experiments/official_matrix_v6/` bundle remains useful preserved evidence, but the manifest-driven `official_acceptance_v1` path is now the authoritative runnable benchmark definition. The `final_audit/verification/` bundle is supplemental verification material.

## 5. Artifact Glossary

- `configs/` - canonical training configs
- `configs/acceptance_matrices/` - canonical acceptance-matrix definitions
- `artifacts/final_pipeline/official_acceptance_v1/` - canonical runnable benchmark output bundle
- `artifacts/final_pipeline/official_acceptance_v1/report/reviewer_start_here.md` - first file a reviewer should open after rerunning the benchmark
- `artifacts/db/greennet.sqlite3` - canonical structured store for indexed official run metadata and final-evaluation payloads
- `experiments/official_matrix_v6/` - preserved historical comparison bundle
- `artifacts/traffic_verify/20260220_matrix/` - reproducibility verification logs
- `artifacts/locked/` - scenario-locked evidence folders
- `runs/` - operational working directory, not the only source of truth
- `models/` - saved checkpoints and compatibility artifacts

## 6. Responsible Design And Limitations

- QoS must remain visible whenever energy savings are discussed.
- The official QoS rule has two layers: runtime QoS status in ordinary runs and baseline-relative QoS acceptance in the final evaluation bundle.
- Stability must remain visible whenever energy savings are discussed. Transition-heavy or flap-heavy results should not be presented as responsible energy savings.
- The energy model is explicit but simplified; it is designed for consistent comparison, not for claiming hardware-accurate wattage.
- The final evidence supports a mixed outcome, not a headline of clear AI dominance.
- `Impact Predictor` is exploratory unless its artifacts are bundled and explicitly verified.
- Older command sheets and root-level config snapshots exist for compatibility, but they are not the canonical submission path.
- The canonical rerun command still assumes the project dependencies are installed and that a PPO checkpoint is available in `runs/` or passed via `--ppo-model`.
- The frontend has a public demo role, while the Streamlit dashboard is internal tooling.

## 7. What A Reviewer Should Check First

- the one-command rerun path: `.venv/bin/python experiments/run_official_acceptance_matrix.py`
- the reviewer entry file in `artifacts/final_pipeline/official_acceptance_v1/report/reviewer_start_here.md`
- the final report in `artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_report.md`
- the source table in `artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv`
- the scenario verification status in `artifacts/traffic_verify/20260220_matrix/matrix_status.md`
- the canonical config set in `configs/README.md`
