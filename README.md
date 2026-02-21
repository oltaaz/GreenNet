# GreenNet

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
