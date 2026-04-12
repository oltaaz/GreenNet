# GreenNet

## Repository Layout

The canonical structure is:

- `greennet/` for importable backend code
- `scripts/` for helper executables
- `configs/` for supported tracked configs
- `configs/archive/root_legacy/` for archived historical config snapshots
- `experiments/`, `tests/`, `frontend/`, and `dashboard/` for workflows around the core package

More detail is in [docs/project_structure.md](docs/project_structure.md).

## Frontend Strategy

GreenNet now has one official frontend path:

- Official demo/product UI: React/Vite app in `frontend/greennet-ui`
- Internal analyst tooling only: Streamlit dashboard in `dashboard/`

Use the React app for demos, product walkthroughs, and the primary workflow across:

- dashboard
- results
- simulator

The Streamlit app remains available for analysts/developers who want a filesystem-oriented run inspector and internal launcher, but it is no longer the public-facing interface.

### Run The Official Frontend

Start the backend API first:

```bash
uvicorn api_app:app --reload --port 8000
```

Then start the React frontend:

```bash
cd frontend
npm run dev
```

Notes:
- `frontend/package.json` is the supported entrypoint for frontend commands.
- The Vite source lives under `frontend/greennet-ui/`.
- The UI uses canonical controller-policy names `all_on`, `heuristic`, and `ppo`. Legacy `noop`/`baseline` aliases are still normalized for older runs.

## Routing Baselines

GreenNet now records the routing baseline separately from the controller policy.

- `min_hop_single_path` is the legacy default and matches the project’s earlier behavior: one deterministic shortest path over unit link cost, which is effectively minimum-hop routing.
- `ospf_ecmp` is a clearer traditional baseline: static link-state SPF with equal splitting across equal-cost shortest paths. It approximates OSPF forwarding behavior after convergence; it does not simulate the OSPF control plane.
- The existing `heuristic` policy is an edge-toggle controller baseline, not a routing protocol baseline.

Example:

```bash
python3 run_experiment.py \
  --policy heuristic \
  --scenario normal \
  --seed 0 \
  --routing-baseline ospf_ecmp
```

### Run The Internal Streamlit Tool

```bash
streamlit run dashboard/app.py
```

Use this only for internal analyst/developer workflows such as manually browsing result folders or launching ad hoc runs from the repo.

## Forecasting

GreenNet keeps the existing EMA demand forecaster as the default runtime baseline and now supports two additional lightweight options:

- `ema`: existing baseline behavior
- `adaptive_ema`: adaptive ensemble over multiple EMA experts, recommended for current traffic traces
- `holt`: damped Holt trend forecaster

Environment config fields:

```json
{
  "enable_forecasting": true,
  "forecast_model": "adaptive_ema",
  "forecast_alpha": 0.6,
  "forecast_horizon_steps": 3,
  "forecast_adaptive_alphas": [0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
  "forecast_adaptive_error_alpha": 0.02,
  "forecast_adaptive_temperature": 0.25
}
```

Notes:
- Default behavior is unchanged because `forecast_model` defaults to `ema`.
- `demand_forecast` remains the same observation key consumed by the environment and controller code.
- `forecast_alpha`, `forecast_beta`, and `forecast_trend_damping` are still used by single-model forecasters.

Run the forecasting comparison workflow:

```bash
.venv/bin/python scripts/eval_forecasters.py \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4 \
  --episodes-per-scenario 1 \
  --max-steps 300 \
  --baseline-model ema \
  --improved-model adaptive_ema \
  --output-dir artifacts/forecasting
```

Generated artifacts:
- `artifacts/forecasting/forecast_compare_summary.csv`
- `artifacts/forecasting/forecast_compare_series_metrics.csv`
- `artifacts/forecasting/forecast_compare_predictions.csv`
- `artifacts/forecasting/forecast_compare_summary.json`
- `artifacts/forecasting/forecast_compare_report.md`

`MAE` and `RMSE` are the primary metrics for aggregate demand quality. `MAPE` is reported on non-zero-demand targets only because zero-demand steps make percentage error undefined.

## Impact Predictor

Impact Predictor is a graph-aware ensemble that predicts next-step OFF-action impact (QoS risk via latency/drop proxies, normalized drop change, and energy change). Runtime gating uses calibrated QoS probability + uncertainty bounds + risk score to block risky OFF toggles.

```bash
# 1) Build sweep dataset (multi-scenario + domain randomization)
ml-env/.venv/bin/python scripts/build_cost_dataset_sweep.py \
  --episodes-per-scenario 200 \
  --topology-seeds 0,1,2,3,4,5,6,7,8,9 \
  --traffic-seeds 0,1,2 \
  --out artifacts/cost_estimator/ds_sweep.npz \
  --qos-delay-ms 24 \
  --qos-drop-max 2 \
  --demand-scale-min 0.7 \
  --demand-scale-max 1.6 \
  --capacity-scale-min 0.7 \
  --capacity-scale-max 1.3 \
  --flows-scale-min 0.8 \
  --flows-scale-max 1.5

# 2) Train Impact Predictor ensemble (group split by topology_seed when available)
ml-env/.venv/bin/python scripts/train_cost_estimator_torch.py \
  --dataset artifacts/cost_estimator/ds_sweep.npz \
  --out-dir models/impact_predictor \
  --ensemble-size 3 \
  --epochs 3

# 3) Acceptance suite (done gate)
ml-env/.venv/bin/python scripts/test_impact_predictor_suite.py \
  --episodes 10 \
  --max-steps 200 \
  --p-off 0.30 \
  --seed 0 \
  --abs-tol-lat-ms 2.0 \
  --tol-norm-drop 0.002 \
  --tol-dropped 5.0 \
  --lock-artifacts

# 4) Optional quick smoke
ml-env/.venv/bin/python scripts/smoke_impact_predictor.py
```

Impact Predictor is complete when `scripts/test_impact_predictor_suite.py` exits successfully and all rows pass.  
The suite includes an OOD `extreme` bucket (`ood_bucket=1`) with conservative checks and stronger masking expectations.
For thesis references, cite a locked acceptance run folder under `artifacts/locked/impact_predictor/<timestamp>/`.
Locked acceptance example: `artifacts/locked/impact_predictor/20260221_121739/`.

### Threshold presets

Defaults remain strict in code for safety-first behavior. Use env vars to pick a runtime preset.

```bash
# Strict (safety-first, code defaults)
export COST_ESTIMATOR_P_QOS_MAX=0.11
export COST_ESTIMATOR_DDROP_MAX=0.001
export COST_ESTIMATOR_TAU=0.11
```

```bash
# Balanced (recommended)
export COST_ESTIMATOR_P_QOS_MAX=0.55
export COST_ESTIMATOR_DDROP_MAX=0.04
export COST_ESTIMATOR_TAU=0.85
```

```bash
# Aggressive (energy-first exploratory preset; validate with acceptance suite)
export COST_ESTIMATOR_P_QOS_MAX=0.60
export COST_ESTIMATOR_DDROP_MAX=0.05
export COST_ESTIMATOR_TAU=0.90
```

Example output fields from the acceptance suite:

```text
scenario   bucket   ood off_cand off_mask %mask lat_off lat_on nd_off nd_on PASS
normal     mild      0   ...
normal     stress    0   ...
normal     extreme   1   ...
burst      extreme   1   ...
hotspot    extreme   1   ...
```

## Energy Model

GreenNet now uses a lightweight structured power model instead of a single `base + active_links` placeholder.

- Each device (node) has an `active` draw, a `sleep` draw, and a small utilization-sensitive term.
- Each link has an `active` draw, a `sleep` draw, and a small utilization-sensitive term.
- A device is considered active when it has at least one active incident link.
- `energy_kwh` is still computed per step from average power over `dt_seconds`, so reward and evaluation code remain compatible.
- `carbon_g` is still derived from `energy_kwh`, but the carbon intensity profile is now exposed through explicit config fields.

Relevant env config keys:
- `power_network_fixed_watts`
- `power_device_active_watts`
- `power_device_sleep_watts`
- `power_device_dynamic_watts`
- `power_link_active_watts`
- `power_link_sleep_watts`
- `power_link_dynamic_watts`
- `carbon_base_intensity_g_per_kwh`
- `carbon_amplitude_g_per_kwh`
- `carbon_period_seconds`

Per-step reports now also include power breakdown fields such as `power_total_watts`, `power_fixed_watts`, `power_variable_watts`, and active/inactive link/device counts.

## Metrics API

Example calls:

```bash
curl "http://localhost:8000/api/runs?limit=3"
curl "http://localhost:8000/api/runs_flat?limit=3"
curl "http://localhost:8000/api/runs/<run_id>/summary"
curl "http://localhost:8000/api/aggregate?tag=matrix_v4&group_by=policy,scenario"
```

Notes:
- CORS allows Vite dev origins `http://localhost:5173` and `http://127.0.0.1:5173`.
- `/api/runs_flat` is a compatibility endpoint for older/quick UIs expecting a plain array (same objects as `/api/runs.items`).
- `/api/aggregate` computes `*_std` using population standard deviation (`statistics.pstdev`).

## Custom Inputs

GreenNet can now load predefined or file-backed topologies and replayable traffic inputs in addition to its original synthetic generation.

- named bundled topologies: `metro_hub`, `regional_ring`
- named bundled traffic profiles: `commuter_bursts`, `commuter_matrices`
- runner support: `run_experiment.py --scenario custom --topology-name ... --traffic-name ...`
- config support: `env.topology_name`, `env.topology_path`, `env.traffic_name`, `env.traffic_path`

Format details and examples are documented in [docs/custom_inputs.md](docs/custom_inputs.md).

## Final Thesis Evaluation

Use the final evaluation layer to turn existing result artifacts into one thesis/reporting-ready baseline-vs-AI comparison bundle.

```bash
python3 experiments/final_evaluation.py \
  --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv \
  --primary-baseline-policy heuristic \
  --ai-policies ppo \
  --output-dir experiments/official_matrix_v6/final_evaluation
```

Generated files:
- `final_evaluation_summary.csv`
- `final_evaluation_summary.json`
- `final_evaluation_report.md`

The script reads existing `run_meta.json`, `summary.json`, and `per_step.csv` artifacts, aggregates across seeds/runs, derives QoS violation metrics from per-step logs when available, and reports whether the `>=15%` energy-reduction hypothesis is met under configurable QoS thresholds.
The exported evaluation bundle also carries routing-baseline metadata so baseline-vs-AI comparisons remain explicit about the forwarding assumptions used.

## Final Thesis Pipeline

Use the final pipeline runner when you want one repeatable command that:
- runs the policy matrix when needed
- re-aggregates authoritative results
- builds by-seed tables and leaderboard tables
- generates the final baseline-vs-AI evaluation bundle
- exports plot-ready CSVs and optional PNG plots
- writes a concise thesis-ready report

### Prerequisites

- Python 3.10+
- project dependencies installed in an active virtual environment
- a PPO checkpoint under `runs/*/ppo_greennet(.zip)` or passed explicitly with `--ppo-model` if you want to rerun evaluations
- optional: `matplotlib` for PNG plot export; without it, the pipeline still writes plot-ready CSV files

Example setup from a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m pip install matplotlib
```

### Full End-to-End Rerun

This reruns the evaluation matrix with deterministic settings and then packages the final thesis bundle.

```bash
python3 experiments/final_pipeline.py \
  --tag matrix_v7 \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --ppo-model runs/<RUN_ID>/ppo_greennet.zip \
  --output-dir artifacts/final_pipeline/matrix_v7
```

### Rebuild From Existing Artifacts

This is the fastest reproducible path when the matrix results already exist in the repo or have been shared separately.

```bash
python3 experiments/final_pipeline.py \
  --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv \
  --skip-eval \
  --tag matrix_v6 \
  --output-dir artifacts/final_pipeline/matrix_v6
```

### Output Layout

The runner writes a predictable bundle:

```text
artifacts/final_pipeline/<name>/
  logs/
  metadata/
    pipeline_config.json
    pipeline_manifest.json
  summary/
    results_summary_<tag>.csv
    results_summary_by_seed_<tag>.csv
    leaderboard_<tag>.csv
    leaderboard_source_<tag>.csv
    research_question_summary.csv
    final_evaluation/
      final_evaluation_summary.csv
      final_evaluation_summary.json
      final_evaluation_report.md
  plots/
    policy_tradeoff_overall.csv
    policy_tradeoff_by_scenario.csv
    research_question_tradeoff.csv
    *.png
  report/
    concise_report.md
```

### Notes

- The pipeline keeps evaluation logic centralized in `experiments/run_matrix.py`, `experiments/aggregate_results.py`, `experiments/make_leaderboard.py`, and `greennet/evaluation/final_report.py`.
- If a step fails, inspect the corresponding file under `logs/`; each step is isolated and logged separately.
- Determinism comes from explicit seeds, deterministic policy evaluation, and fixed scenario/policy selection on the CLI.
