# GreenNet — Commands Cheat Sheet

Here are listed the most common commands that are required to run, train, evaluate and demo the project.
If any of these doesn't work, please just check the path of the folders or files you are trying to use.

If you see any error like "ModuleNotFoundError" or something like that, check firstly if you activated VENV, because these modules, libraries and dependencies work only in virtual environment (venv).

---------------------------------------------------------------------------------------------------------------------------------------

## 0) Go to project root

cd ~/Desktop/GreenNet/


If your project folder is in a different location, change the path to match your project directory.

---------------------------------------------------------------------------------------------------------------------------------------

## 1) Activate the virtual environment

If your venv is inside `ml-env/.venv`:

cd ml-env
source .venv/bin/activate
cd ..

Quick checks:

python3 -V
python3 -m pip show numpy


---------------------------------------------------------------------------------------------------------------------------------------

## 2) Install dependencies

python3 -m pip install -r requirements.txt

(Optional) freeze exact versions:

python3 -m pip freeze > requirements_freeze.txt

---------------------------------------------------------------------------------------------------------------------------------------

## 3) Train PPO

Example training run:

python3 train.py --timesteps 300000

Scenario-specific training configs (recommended):

python3 train.py --config configs/train_normal.json --timesteps 300000
python3 train.py --config configs/train_burst.json --timesteps 300000
python3 train.py --config configs/train_hotspot.json --timesteps 300000

---------------------------------------------------------------------------------------------------------------------------------------

## 4) Evaluate a trained model (quick sanity)

python3 train.py --eval --episodes 30

---------------------------------------------------------------------------------------------------------------------------------------

## 5) Run a single experiment (end-to-end runner)

This writes (inside a new folder under `results/`):
- `per_step.csv`
- `summary.json`
- `run_meta.json`
- `env_config.json`

Examples:

python3 run_experiment.py --policy all_on --scenario normal --seed 0 --episodes 1 --steps 300
python3 run_experiment.py --policy heuristic --scenario hotspot --seed 0 --episodes 1 --steps 300
python3 run_experiment.py --policy ppo --scenario normal --seed 0 --episodes 1 --steps 300

Stochastic PPO evaluation:

python3 run_experiment.py --policy ppo --scenario hotspot --seed 4 --episodes 1 --steps 300 --stochastic

---------------------------------------------------------------------------------------------------------------------------------------

## 6) Run the evaluation matrix (multi-seed / multi-scenario / multi-policy)

Writes:
- `results/results_summary.csv`

python3 experiments/run_matrix.py --episodes 1 --steps 300

Custom matrix:

python3 experiments/run_matrix.py --seeds 0,1,2 --scenarios normal,hotspot --policies noop,baseline,ppo --episodes 1 --steps 300

Force fixed topology seed:

python3 experiments/run_matrix.py --episodes 1 --steps 300 --topology-seed 0

---------------------------------------------------------------------------------------------------------------------------------------

## 6a) Official Matrix Run (canonical)

Use this command for the **official results** (do not change without bumping the tag/version):

python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v1

---------------------------------------------------------------------------------------------------------------------------------------

## 6b) Official Matrix Run (matrix_v2 — energy_weight +50%)

Train PPO v2 (energy_weight=1800 only; everything else unchanged):

python3 train.py --config train_normal_v2.json --timesteps 300000

Run the matrix using the exact PPO model you trained (pin model path):

python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v2 \
  --ppo-model <PATH_TO_PPO_ZIP>

---------------------------------------------------------------------------------------------------------------------------------------

## 6c) Official Matrix Run (matrix_v3 — lower toggle penalties)

Train PPO v3 (energy_weight=1800, toggle penalties reduced only):

python3 train.py --config train_normal_v3.json --timesteps 300000

Run the matrix using the exact PPO model you trained (pin model path):

python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v3 \
  --ppo-model <PATH_TO_PPO_ZIP>

Note: single knob change vs v2 = lower toggle_penalty (0.0005) and blocked_action_penalty (0.0003).

---------------------------------------------------------------------------------------------------------------------------------------

## 6d) Official Matrix Run (matrix_v4 — loosen cooldowns)

Train PPO v4 (keep v3 settings, reduce cooldowns only):

python3 train.py --config train_normal_v4.json --timesteps 300000

Run the matrix using the exact PPO model you trained (pin model path):

python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v4 \
  --ppo-model <PATH_TO_PPO_ZIP>

---------------------------------------------------------------------------------------------------------------------------------------

## 7) Final Thesis Evaluation Bundle

Build one authoritative baseline-vs-AI summary from existing result artifacts.

Using an already packaged matrix summary:

python3 experiments/final_evaluation.py \
  --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv \
  --primary-baseline-policy heuristic \
  --ai-policies ppo \
  --output-dir experiments/official_matrix_v6/final_evaluation

Or scan `results/` directly by tag:

python3 experiments/final_evaluation.py \
  --results-dir results \
  --tag matrix_v6 \
  --primary-baseline-policy heuristic \
  --ai-policies ppo \
  --output-dir artifacts/final_evaluation/matrix_v6

Writes:
- `final_evaluation_summary.csv`
- `final_evaluation_summary.json`
- `final_evaluation_report.md`

Note: single knob change vs v3 = toggle_cooldown_steps=4 (was 12) and global_toggle_cooldown_steps=10 (was 40).

---------------------------------------------------------------------------------------------------------------------------------------

## 7a) Final Thesis Pipeline (single end-to-end runner)

Use this when you want one command that handles matrix execution, aggregation, summary tables, final evaluation, plots, and a concise report bundle.

Full rerun:

python3 experiments/final_pipeline.py \
  --tag matrix_v7 \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --ppo-model runs/<RUN_ID>/ppo_greennet.zip \
  --output-dir artifacts/final_pipeline/matrix_v7

Rebuild from an existing packaged matrix:

python3 experiments/final_pipeline.py \
  --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv \
  --skip-eval \
  --tag matrix_v6 \
  --output-dir artifacts/final_pipeline/matrix_v6

Bundle outputs:
- `summary/results_summary_<tag>.csv`
- `summary/results_summary_by_seed_<tag>.csv`
- `summary/leaderboard_<tag>.csv`
- `summary/research_question_summary.csv`
- `summary/final_evaluation/final_evaluation_report.md`
- `plots/*.csv`
- `plots/*.png` (if `matplotlib` is installed)
- `report/concise_report.md`
- `metadata/pipeline_manifest.json`

---------------------------------------------------------------------------------------------------------------------------------------

## 6e) Scenario lock + eval matrix (thesis reproducibility)

Lock a run to scenario-specific artifacts:

scripts/lock_run.sh --scenario normal --run-id <RUN_ID>

Run fixed eval matrix (scenarios normal/burst/hotspot, off edges 0/2/4, held-out topology seeds 10..19):

scripts/eval_matrix.sh --model runs/<RUN_ID>/ppo_greennet.zip --lock-scenario normal --episodes 10

By default, matrix logs are written to:

artifacts/locked/<train_scenario>/<RUN_ID>/eval_<scenario>_off<k>.txt

---------------------------------------------------------------------------------------------------------------------------------------

## 7) Aggregate existing run folders into a CSV (optional helper)

python3 experiments/aggregate_results.py --out-dir results

Only runs with a tag:

python3 experiments/aggregate_results.py --out-dir results --tag matrix

---------------------------------------------------------------------------------------------------------------------------------------

## 8) Inspect a run quickly

List recent runs:

ls -lt results | head -n 20

Check columns in `per_step.csv`:

python3 -c "import csv; r=csv.DictReader(open('results/<RUN_FOLDER>/per_step.csv')); print(len(r.fieldnames), r.fieldnames[:10])"

Compute PPO “noop rate” / toggles:

python3 -c "import pandas as pd; df=pd.read_csv('results/<RUN_FOLDER>/per_step.csv'); print('noop_rate', df['action_is_noop'].mean()); print('toggle_applied_rate', df['toggle_applied'].mean()); print('active_ratio_mean', df['active_ratio'].mean()); print('unique_actions', df['action'].nunique()); print(df['action'].value_counts().head(10))"

Check whether toggles caused drops:

python3 - <<'PY'
import pandas as pd
df = pd.read_csv('results/<RUN_FOLDER>/per_step.csv')
print('toggles applied:', df['toggle_applied'].sum())
print('avg_util when toggled:', df.loc[df['toggle_applied'], 'avg_utilization'].mean())
print('max_util when toggled:', df.loc[df['toggle_applied'], 'max_util'].mean())
print('drops during toggle steps:', df.loc[df['toggle_applied'], 'delta_dropped'].sum())
PY

---------------------------------------------------------------------------------------------------------------------------------------

## 9) Decode which edge an action ID toggles (debug)

python3 - <<'PY'
from greennet.env import GreenNetEnv, EnvConfig
env = GreenNetEnv(EnvConfig())
env.reset(seed=0)
print('action_space_n:', env.action_space.n)
print('num_edges:', len(env.edge_list))
print('edge_list[0:10]:', list(env.edge_list)[:10])
env.close()
PY

Mapping rule:
- action 0 = NOOP
- action k (>= 1) toggles `edge_list[k-1]`

---------------------------------------------------------------------------------------------------------------------------------------

## 10) Dashboard (Streamlit)

streamlit run app.py

---------------------------------------------------------------------------------------------------------------------------------------

## 11) Frontend + Backend Integration

Start the backend API:

cd ~/Desktop/GreenNet/
source ml-env/.venv/bin/activate
uvicorn api_app:app --host 127.0.0.1 --port 8000

Start the frontend in a new terminal:

cd ~/Desktop/GreenNet/frontend/greennet-ui
npm run dev -- --host 127.0.0.1 --port 5173

Build the frontend:

cd ~/Desktop/GreenNet/frontend/greennet-ui
npm run build

Backend health check:

curl http://127.0.0.1:8000/api/health

List runs:

curl "http://127.0.0.1:8000/api/runs?limit=3"

Get topology for a run:

curl "http://127.0.0.1:8000/api/runs/<RUN_ID>/topology"

Get step timeline for a run:

curl "http://127.0.0.1:8000/api/runs/<RUN_ID>/steps"

Get link state at a specific step:

curl "http://127.0.0.1:8000/api/runs/<RUN_ID>/link_state?step=10"

Start a new run from the backend:

curl -X POST "http://127.0.0.1:8000/api/runs/start" \
  -H "Content-Type: application/json" \
  -d '{"policy":"baseline","scenario":"normal","seed":123,"steps":20}'

Frontend URL:

http://127.0.0.1:5173/

---------------------------------------------------------------------------------------------------------------------------------------

## Notes

- Run commands from the project root unless stated otherwise.
- If you see `ModuleNotFoundError`, activate `.venv` first.
- If you use any new script or command, add it here, maybe you will need them again... as you wish.
