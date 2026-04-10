# GreenNet

GreenNet is an energy- and QoS-aware routing simulator with a baseline-vs-controller evaluation pipeline, a FastAPI backend, and a React demo UI. The project is strongest as a reproducible research/software artifact: the simulator is real, the baselines are explicit, and the final evidence bundle shows a mixed but technically honest AI result rather than a fabricated win.

## Demo Startup

For the local demo, the official repo-root command is:

```bash
npm run dev
```

This root script is intentionally lightweight: it reuses the existing backend command (`uvicorn api_app:app --reload --port 8000`) and the existing frontend command (`npm --prefix frontend run dev`) and starts them together for reviewers from one place.
If `.venv` exists, the backend side of this command uses that repo-local interpreter automatically, so reviewers do not need to activate the virtual environment in a separate shell first.

One-time setup before that command:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,train]'
npm install
```

If the root `npm run dev` flow is not available in your shell or platform, use the original commands separately:

```bash
uvicorn api_app:app --reload --port 8000
npm --prefix frontend run dev
```

## Official Path

Use this path for the final submission workflow:

1. Install the project with the tracked Python metadata:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,train]'
```

2. Use the canonical training configs in `configs/`.

3. Run experiments with `run_experiment.py`, `train.py`, or `experiments/run_matrix.py` as needed.

4. Run and package the official acceptance matrix from `configs/acceptance_matrices/official_acceptance_v1.json`.
   Treat `experiments/official_matrix_v6/` as the preserved historical evidence bundle, not the executable source of truth.

5. Use [docs/final_submission_overview.md](docs/final_submission_overview.md) for the final architecture, artifact glossary, and limitations summary.

6. Run the final verification checks before submission:

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend/greennet-ui ci
npm --prefix frontend/greennet-ui run build
```

## Current Final Claim

The repository now uses one explicit baseline taxonomy:

- official traditional baseline: `all_on` on top of the default routing baseline `min_hop_single_path` with `unit` link cost
- strongest non-AI heuristic baseline: `heuristic`
- AI policy: `ppo`

In the curated final evaluation:

- `heuristic` is the best overall non-AI controller
- `ppo` is the best AI policy
- `ppo` remains about 4% worse than `heuristic` on energy in the final aggregate
- the official traditional baseline remains the fixed comparison anchor for report deltas and hypothesis checks
- QoS stays within the acceptance gates used in the final report

The defensible submission claim is therefore:

- GreenNet provides a credible routing simulation and evaluation framework for energy/QoS tradeoffs
- the official traditional baseline is explicit and technically honest
- the energy-aware heuristic is not misrepresented as traditional routing
- the AI controller is measured against both the official traditional baseline and the strongest handcrafted heuristic under the same scenarios and seeds
- the final evidence explains a mixed outcome instead of overclaiming improvement

Important note on the canonical bundle:

- the reviewer-facing canonical bundle currently pinned under `artifacts/final_pipeline/latest/` is the preserved historical `~1.49%` PPO result
- it is a promoted historical reconstructed bundle, not the output of the current one-command rerun path from the current code/checkpoint state
- the current codebase still reproduces the official benchmark pipeline, but not that exact historical `1.49%` result end to end

## Canonical Workflow

### 1. Train or rerun a policy

Use the tracked scenario configs in `configs/`:

```bash
.venv/bin/python train.py --config configs/train_normal.json --timesteps 300000
.venv/bin/python train.py --config configs/train_burst.json --timesteps 300000
.venv/bin/python train.py --config configs/train_hotspot.json --timesteps 300000
.venv/bin/python train.py --config configs/train_official_ppo.json --timesteps 100000
```

The historical checkpoint at `runs/20260220_111755/ppo_greennet.zip` is no longer compatible with the current env because it was trained against an older observation layout. The current env emits topology-dependent Dict observations, so the canonical PPO artifact is now a topology-specific family:

- `artifacts/models/official_acceptance_v1/small/ppo_greennet.zip`
- `artifacts/models/official_acceptance_v1/medium/ppo_greennet.zip`
- `artifacts/models/official_acceptance_v1/large/ppo_greennet.zip`

To regenerate that official family with the current codebase:

```bash
.venv/bin/python experiments/regenerate_official_ppo_checkpoint.py --all-topologies --config configs/train_official_ppo.json --timesteps 100000
```

That config exists because the old single checkpoint under `runs/20260220_111755/` was both observation-incompatible and under the current benchmark materially worse than the traditional baseline. The official acceptance-matrix path uses the topology-specific family automatically. Use `--ppo-model` only when you intentionally want a non-official single-checkpoint override.

### 2. Run a single experiment

```bash
.venv/bin/python run_experiment.py --policy heuristic --scenario normal --seed 0 --episodes 1 --steps 300
.venv/bin/python run_experiment.py --policy ppo --scenario hotspot --seed 0 --episodes 1 --steps 300
.venv/bin/python run_experiment.py --policy all_on --scenario normal --seed 0 --episodes 1 --steps 300 --topology-name medium
.venv/bin/python run_experiment.py --policy all_on --scenario flash_crowd --seed 0 --episodes 1 --steps 300 --topology-name large
```

### 3. Reproduce the official final claim

```bash
.venv/bin/python experiments/run_official_acceptance_matrix.py
```

This is the canonical one-command reviewer path. It now does the full official flow:

- verifies the canonical acceptance matrix manifest
- checks the required Python dependencies
- initializes the SQLite run store
- verifies that the canonical topology-specific PPO checkpoint family exists, or uses an explicit `--ppo-model` override
- runs the official acceptance matrix
- aggregates results and writes the final evaluation bundle
- persists the final-evaluation payload into SQLite

Main outputs after a successful run:

- quick reviewer summary: `artifacts/final_pipeline/official_acceptance_v1/report/reviewer_start_here.md`
- concise claim summary: `artifacts/final_pipeline/official_acceptance_v1/report/concise_report.md`
- final thesis-facing report: `artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_report.md`
- authoritative summary CSV: `artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv`
- SQLite store: `artifacts/db/greennet.sqlite3`

If you only want to verify prerequisites first:

```bash
.venv/bin/python experiments/run_official_acceptance_matrix.py --check-only
```

If the canonical PPO family is missing, regenerate it with `experiments/regenerate_official_ppo_checkpoint.py --all-topologies`. That is the reviewer-safe fix; do not reuse the old single checkpoint from `runs/20260220_111755/`.

### 4. Run the official acceptance matrix manually if needed

```bash
.venv/bin/python experiments/run_official_acceptance_matrix.py
```

The authoritative definition lives in [official_acceptance_v1.json](/Users/enionismaili/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json).
It includes:

- policies: `all_on`, `heuristic`, `ppo`
- seeds: `0-4`
- episodes: `20`
- steps: `300`
- cases:
  - `small_normal`
  - `small_commuter_replay`
  - `medium_diurnal`
  - `medium_hotspot`
  - `large_flash_crowd`
  - `large_flash_replay`

If you need a custom benchmark, use `experiments/run_matrix.py --matrix-manifest <path.json>`. Do not describe that as the official final matrix.

### 5. Build the final evaluation bundle manually

```bash
.venv/bin/python experiments/run_official_acceptance_matrix.py \
  --skip-eval \
  --summary-csv results/results_summary_official_acceptance_v1.csv
```

The default output path for that wrapper is `artifacts/final_pipeline/official_acceptance_v1/`.
It copies the manifest into `metadata/acceptance_matrix_manifest.json` and preserves matrix identity in the authoritative summary and final evaluation payloads.

## Run Database

GreenNet now treats SQLite as the primary structured store for indexed runs and official final-evaluation payloads, while keeping file artifacts as compatible secondary outputs.

- default DB path: `artifacts/db/greennet.sqlite3`
- initialize manually: `python3 -m greennet.persistence init`
- backfill older file-only runs: `python3 -m greennet.persistence backfill --base both`
- export a DB-backed summary CSV: `python3 -m greennet.persistence export-summary --base both --output /tmp/results_summary.csv`

The DB stores:

- run metadata plus official acceptance-matrix identity
- policy, topology, traffic, energy-model, QoS-policy, and stability-policy identity
- full run summaries and key flattened aggregate metrics
- per-step metrics including power, transition, and flap fields
- persisted final-evaluation payloads for the official pipeline

The API now prefers SQLite for run retrieval and final evaluation lookup. CSV/JSON/Markdown artifacts are still written and remain the reviewer-facing export surface.

## Topology Selection

GreenNet supports three stable packaged topology classes through the existing config and experiment flow:

- `small`
- `medium`
- `large`

Use `env.topology_name` in config JSON or pass `--topology-name` to `run_experiment.py`, `experiments/run_matrix.py`, or `greennet.evaluation.final_pipeline`. `topology_path` still overrides `topology_name` when you need a custom JSON topology file. Legacy packaged names such as `regional_ring` and `metro_hub` are still accepted for compatibility, but new configs should prefer the size-class names.

Traffic selection is now first-class in the same flow. You can:

- use stochastic traffic scenarios: `normal`, `diurnal`, `burst`, `hotspot`, `anomaly`, `flash_crowd`, `multi_peak`
- use packaged replay traffic via `--traffic-name`
- use custom replay JSON via `--traffic-path`

`traffic_name` and `traffic_path` override the stochastic generator settings when both are present. Run outputs and aggregate summaries preserve `traffic_mode`, `traffic_model`, `traffic_name`, `traffic_path`, and the configured stochastic scenario metadata.

If the raw historical run folders are absent, you can still rerun the final packaging path against a preserved summary CSV. The historical `experiments/official_matrix_v6/final_evaluation/` bundle remains the preserved shipped evidence for the earlier repo state, while `official_acceptance_v1.json` is now the authoritative runnable benchmark definition.

## Architecture Overview

GreenNet is organized around four layers:

- routing/simulation core in `greennet/`
- baselines and experiment runners in `baselines.py`, `run_experiment.py`, `train.py`, `eval.py`, and `experiments/`
- API and persistence in `api_app.py` and `greennet/persistence`
- public demo and analyst tooling in `frontend/greennet-ui/` and `dashboard/`

The controller stack is intentionally simple:

- the simulator computes routing, load, delay, power, and carbon metrics
- the default routing baseline is `min_hop_single_path` with `unit` link cost
- `all_on` is the official traditional baseline controller
- `heuristic` is the explicit energy-aware heuristic baseline controller
- `ppo` is the learned controller, but it is not presented as universally superior

## Energy Model

GreenNet's energy model is intentionally lightweight and explicit:

- nodes and links each have active and sleep power states
- total network power is modeled as fixed power plus an optional utilization-sensitive dynamic component
- carbon emissions are derived from total energy, not estimated independently

The legacy config surface remains valid. Existing `power_*` fields still define the base active/sleep and dynamic power levels for devices and links. The clearer model adds these optional knobs:

- `power_utilization_sensitive`
- `power_transition_on_joules`
- `power_transition_off_joules`

For backward compatibility:

- omitting `power_utilization_sensitive` preserves the legacy fixed-plus-dynamic interpretation from the existing `power_*` fields
- omitting `power_transition_on_joules` and `power_transition_off_joules` keeps transition energy costs at zero
- existing configs that only set legacy `power_*` and `carbon_*` fields remain valid

In the official workflow, the tracked `configs/train_*.json` files are the config-facing source of truth for these assumptions. `EnvConfig` still provides the fallback defaults used when a config does not override them.

## QoS Policy

GreenNet now uses one explicit QoS policy instead of leaving “acceptable QoS” implicit.

The runtime QoS rule used by ordinary runs and dashboards is:

- normalized delivery-loss ratio threshold: `qos_target_norm_drop` with the default official value `0.072`
- minimum observed traffic volume before that loss rule is enforced: `qos_min_volume` with the default official value `500`
- average-delay guard used by controller veto logic and run summaries:
  `avg_delay_ms <= max(avg_path_latency_ms * 4.0, avg_path_latency_ms + 15.0)`
- p95 delay is not part of the official rule yet because the current simulator does not compute it cleanly

The final evaluation bundle keeps the repository’s thesis-facing acceptance gate relative to the official traditional baseline. The default official thresholds are:

- delivered traffic loss vs `all_on`: at most `2%`
- dropped traffic increase vs `all_on`: at most `5%`
- average delay increase vs `all_on`: at most `10%`
- average path latency increase vs `all_on`: at most `10%`
- QoS-violation-rate increase vs `all_on`: at most `0.02`

Run outputs and aggregate summaries now preserve QoS identity and status through fields such as `qos_policy_name`, `qos_policy_signature`, `qos_thresholds`, `delivery_loss_rate_mean`, `qos_violation_rate_mean`, and `qos_acceptance_status`. Final evaluation artifacts still preserve `qos_acceptability_status` for compatibility, and also expose the centralized `qos_thresholds` block.

## Stability Policy

GreenNet now treats switching stability as an explicit project rule instead of leaving it implicit in the controller code.

The official stability policy is built on the existing env guardrails:

- decision-step gating through `decision_interval_steps`
- per-edge minimum dwell through `toggle_cooldown_steps`
- global OFF cooldown through `global_toggle_cooldown_steps`
- calm-period gating through `off_calm_steps_required`
- per-episode OFF, total, and emergency-ON toggle budgets
- an explicit reversal penalty through `stability_reversal_penalty`

The centralized stability assessment then evaluates each run with these defaults:

- reversal/flap window: `stability_reversal_window_steps=20`
- minimum steps before assessment: `stability_min_steps_for_assessment=50`
- maximum transition rate: `stability_max_transition_rate=0.02`
- maximum flap rate: `stability_max_flap_rate=0.25`
- maximum flap count: `stability_max_flap_count=2`

A transition is any successful applied link state change. A flap event is a successful toggle that reverses the same edge's previous successful direction within the reversal window.

Run outputs, aggregate summaries, API responses, and final evaluation artifacts now preserve stability identity and status through fields such as `stability_policy_name`, `stability_policy_signature`, `transition_rate_mean`, `flap_rate_mean`, `stability_status`, and `stability_qualified_hypothesis_status`.

## Artifact Glossary

- `configs/` - canonical training configs used by the official workflow
- `experiments/official_matrix_v6/` - curated final matrix summary and final evaluation bundle
- `artifacts/traffic_verify/20260220_matrix/` - deterministic traffic verification evidence
- `artifacts/locked/` - locked scenario evidence and retained run bundles
- `runs/` - operational run directory used by the codebase and scripts
- `models/` - saved checkpoints and compatibility artifacts
- `docs/` - workflow notes, input formats, and submission guidance

Not every historical raw result is bundled in one place. The official submission story should cite the curated summary tables and locked verification artifacts rather than trying to imply a single monolithic `results/` folder exists.

## Canonical Config Family

The canonical config family is `configs/train_*.json`.

- use these files for current scenario training
- treat top-level `train_*.json` files as historical or compatibility snapshots
- do not mix root-level legacy configs into the final submission narrative unless you are explicitly discussing history

See [configs/README.md](configs/README.md) for the tracked config set and format notes.

## Legacy And Internal Tools

The following paths are useful, but they are not the public-facing submission path:

- `dashboard/` - internal analyst tooling
- `COMMANDS.md` - legacy command sheet kept for reference
- older experiment folders and ad hoc run folders under `runs/`

The public demo path is the React app in `frontend/greennet-ui/`, but the official reviewer startup entrypoint is now the repo root.

```bash
npm run dev
```

Fallback if you want to run the pieces directly:

```bash
uvicorn api_app:app --reload --port 8000
npm --prefix frontend run dev
```

## Responsible Design And Limitations

GreenNet evaluates energy against QoS, not energy alone. The project should be described with those constraints visible:

- energy savings are only meaningful if QoS remains acceptable
- the final matrix does not prove a large AI win on energy
- gating, thresholds, and scenario definitions matter to the result
- the Impact Predictor and some experimental scripts are exploratory and should be described as such unless you have bundled and verified evidence for them

Further detail is documented in [docs/final_submission_overview.md](docs/final_submission_overview.md).
