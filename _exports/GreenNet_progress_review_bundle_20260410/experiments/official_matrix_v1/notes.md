# Official Matrix v1

## Commit
- `cdc87e112d61386e118f596301aa14ab0a2c48be`

## Command
```
python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v1
```

## Config files
- none (run_matrix uses CLI defaults + scenario presets; PPO model inferred from latest run)

## Seeds
- 0–9 (10 seeds)

## Scenarios
- normal (traffic_scenario=normal, version=2, intensity=1.0)
- burst (traffic_scenario=burst, version=2, intensity=1.0)
- hotspot (traffic_scenario=hotspot, version=2, intensity=1.0)

Scenario presets are defined in `greennet/traffic.py` under `_SCENARIO_PRESETS`.

## Policies
- all_on
- heuristic
- ppo

## PPO model
- uses latest model in `runs/` at matrix time
- resolved in this run as: `runs/20260123_125047/ppo_greennet.zip`

## Files in this pack
- `results_summary_matrix_v1.csv`
- `results_summary_by_seed_matrix_v1.csv`
- `leaderboard_matrix_v1.csv`
- `leaderboard_source_matrix_v1.csv`
