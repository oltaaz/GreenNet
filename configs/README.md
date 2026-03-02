# Scenario Training Configs

Use these tracked configs for scenario-specific training runs:

- `configs/train_normal.json`
- `configs/train_burst.json`
- `configs/train_hotspot.json`

Example:

```bash
python3 train.py --config configs/train_normal.json --timesteps 300000
```

These files define both PPO hyperparameters and training-time env overrides (`env.*`), including `traffic_scenario` and scenario-specific traffic/reward knobs.
