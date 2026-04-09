# GreenNet Commands

This file is a legacy command index. The official workflow is documented in [README.md](README.md) and [docs/final_submission_overview.md](docs/final_submission_overview.md).

## Official Commands

### Install

```bash
python3 -m pip install -e .[test,train]
```

### Train

```bash
python3 train.py --config configs/train_normal.json --timesteps 300000
python3 train.py --config configs/train_burst.json --timesteps 300000
python3 train.py --config configs/train_hotspot.json --timesteps 300000
```

### Run A Single Experiment

```bash
python3 run_experiment.py --policy heuristic --scenario normal --seed 0 --episodes 1 --steps 300
python3 run_experiment.py --policy ppo --scenario hotspot --seed 0 --episodes 1 --steps 300
```

### Run The Official Matrix

```bash
python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v6
```

### Build The Final Evidence Bundle

```bash
python3 experiments/final_evaluation.py \
  --summary-csv experiments/official_matrix_v6/results_summary_matrix_v6.csv \
  --primary-baseline-policy heuristic \
  --ai-policies ppo \
  --output-dir experiments/official_matrix_v6/final_evaluation
```

### Start The API And Frontend

```bash
uvicorn api_app:app --reload --port 8000
cd frontend/greennet-ui
npm run dev
```

### Start The Internal Dashboard

```bash
streamlit run dashboard/app.py
```

## Legacy And Deprecated Entries

The following patterns still appear in older notes and should not be treated as the official path for the final submission:

- `requirements.txt` based setup
- machine-specific virtualenv shell paths
- `results/` as the canonical final evidence store
- root-level historical configs such as `train_normal_v2.json`
- command snippets copied from older machine-specific notes

If you need a one-line answer for the current project state, use the official workflow in the README instead of this file.
