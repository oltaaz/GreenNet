# Official Matrix v3 (template)

## Commit
- `<COMMIT_HASH>`

## Command
```
python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --episodes 50 \
  --steps 300 \
  --tag matrix_v3 \
  --ppo-model <PATH_TO_PPO_ZIP>
```

## Seeds
- 0–9 (10 seeds)

## Scenarios
- normal (traffic_scenario=normal, version=2, intensity=1.0)
- burst (traffic_scenario=burst, version=2, intensity=1.0)
- hotspot (traffic_scenario=hotspot, version=2, intensity=1.0)

## Policies
- all_on
- heuristic
- ppo

## PPO model
- `<PATH_TO_PPO_ZIP>`

## Change vs v2
- single knob change: `toggle_penalty=0.0005`, `blocked_action_penalty=0.0003`
- keep: `energy_weight=1800`
