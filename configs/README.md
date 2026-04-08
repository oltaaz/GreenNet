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

Additional env overrides now supported for realistic inputs:

- `topology_name` or `topology_path`
- `traffic_name` or `traffic_path`

See [docs/custom_inputs.md](../docs/custom_inputs.md) for the JSON formats and examples.

Forecasting overrides can also live under `env.*`, for example:

```json
{
  "env": {
    "forecast_model": "adaptive_ema",
    "forecast_horizon_steps": 3,
    "forecast_adaptive_alphas": [0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
    "forecast_adaptive_error_alpha": 0.02,
    "forecast_adaptive_temperature": 0.25
  }
}
```

Energy/carbon model overrides can also be set in these `env.*` blocks. The main
ones are:

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

If omitted, GreenNet uses the lightweight defaults from `EnvConfig`.
