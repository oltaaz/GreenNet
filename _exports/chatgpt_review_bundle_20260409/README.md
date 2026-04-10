# GreenNet

GreenNet is an energy- and QoS-aware routing simulator with a baseline-vs-controller evaluation pipeline, a FastAPI backend, and a React demo UI. The project is strongest as a reproducible research/software artifact: the simulator is real, the baselines are explicit, and the final evidence bundle shows a mixed but technically honest AI result rather than a fabricated win.

## Official Path

Use this path for the final submission workflow:

1. Install the project with the tracked Python metadata:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,train]'
```

2. Use the canonical training configs in `configs/`.

3. Run experiments with `run_experiment.py`, `train.py`, or `experiments/run_matrix.py` as needed.

4. Package and review the official evidence bundle in `experiments/official_matrix_v6/`, `artifacts/final_submission/matrix_v6/`, and `artifacts/traffic_verify/20260220_matrix/`.

5. Use [docs/final_submission_overview.md](docs/final_submission_overview.md) for the final architecture, artifact glossary, and limitations summary.

6. Run the final verification checks before submission:

```bash
.venv/bin/python -m pytest -q
cd frontend/greennet-ui
npm ci
npm run build
```

## Current Final Claim

The official matrix does not support the claim that the AI policy clearly beats the routing baseline on energy. In the curated final evaluation:

- `heuristic` is the best overall policy
- `ppo` is the best AI policy
- `ppo` is still about 4% worse than `heuristic` on energy in the final aggregate
- QoS stays within the acceptance gates used in the final report

The defensible submission claim is therefore:

- GreenNet provides a credible routing simulation and evaluation framework for energy/QoS tradeoffs
- the baseline is explicit and technically honest
- the AI controller is measured against the baseline under the same scenarios and seeds
- the final evidence explains a mixed outcome instead of overclaiming improvement

## Canonical Workflow

### 1. Train or rerun a policy

Use the tracked scenario configs in `configs/`:

```bash
.venv/bin/python train.py --config configs/train_normal.json --timesteps 300000
.venv/bin/python train.py --config configs/train_burst.json --timesteps 300000
.venv/bin/python train.py --config configs/train_hotspot.json --timesteps 300000
```

### 2. Run a single experiment

```bash
.venv/bin/python run_experiment.py --policy heuristic --scenario normal --seed 0 --episodes 1 --steps 300
.venv/bin/python run_experiment.py --policy ppo --scenario hotspot --seed 0 --episodes 1 --steps 300
```

### 3. Run the official matrix

```bash
.venv/bin/python experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v6
```

### 4. Build the final evaluation bundle

```bash
.venv/bin/python experiments/final_evaluation.py \
  --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv \
  --primary-baseline-policy heuristic \
  --ai-policies ppo \
  --output-dir experiments/official_matrix_v6/final_evaluation
```

The canonical reviewer bundle is `artifacts/final_submission/matrix_v6/`.
It contains `manifest.json` and `traceability.csv` so reviewers can map the submission claim back to the preserved artifacts.

If the raw historical run folders are absent, the command still rebuilds the aggregate comparison from the packaged summary CSV. The preserved report in `experiments/official_matrix_v6/final_evaluation/` remains the authoritative shipped artifact for the final thesis-facing QoS gate interpretation; any recheck bundle under `final_audit/verification/` should be treated as supplemental verification evidence.

## Architecture Overview

GreenNet is organized around four layers:

- routing/simulation core in `greennet/`
- baselines and experiment runners in `baselines.py`, `run_experiment.py`, `train.py`, `eval.py`, and `experiments/`
- API and persistence in `api_app.py` and `greennet/persistence`
- public demo and analyst tooling in `frontend/greennet-ui/` and `dashboard/`

The controller stack is intentionally simple:

- the simulator computes routing, load, delay, power, and carbon metrics
- `all_on`, `heuristic`, and `ppo` are compared on the same scenario set
- the heuristic is the explicit non-AI baseline
- PPO is the learned controller, but it is not presented as universally superior

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

The public demo path is the React app in `frontend/greennet-ui/`.

```bash
cd frontend/greennet-ui
npm run dev
```

## Responsible Design And Limitations

GreenNet evaluates energy against QoS, not energy alone. The project should be described with those constraints visible:

- energy savings are only meaningful if QoS remains acceptable
- the final matrix does not prove a large AI win on energy
- gating, thresholds, and scenario definitions matter to the result
- the Impact Predictor and some experimental scripts are exploratory and should be described as such unless you have bundled and verified evidence for them

Further detail is documented in [docs/final_submission_overview.md](docs/final_submission_overview.md).
