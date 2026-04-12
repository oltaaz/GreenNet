# Scenario Training Configs

Use these tracked configs for scenario-specific training runs. This directory is the canonical config family for the final submission:

- `configs/train_normal.json`
- `configs/train_burst.json`
- `configs/train_hotspot.json`
- `configs/train_official_ppo.json`

Example:

```bash
python3 train.py --config configs/train_normal.json --timesteps 300000
```

For the official acceptance-matrix rerun, PPO is no longer represented by one generic checkpoint. The current env emits topology-dependent Dict observations, so the canonical reviewer-facing PPO artifact is a three-checkpoint family:

- `artifacts/models/official_acceptance_v1/small/ppo_greennet.zip`
- `artifacts/models/official_acceptance_v1/medium/ppo_greennet.zip`
- `artifacts/models/official_acceptance_v1/large/ppo_greennet.zip`

Regenerate that family from the current codebase with:

```bash
python3 experiments/regenerate_official_ppo_checkpoint.py --all-topologies --config configs/train_official_ppo.json --timesteps 100000
```

This preserves the official acceptance matrix while avoiding brittle observation-space shims for older checkpoints. `train_official_ppo.json` is the current canonical PPO regeneration config for the acceptance-matrix family.

These files define both PPO hyperparameters and training-time env overrides (`env.*`), including `traffic_scenario` and scenario-specific traffic/reward knobs. Prefer these files over the historical root-level `train_*.json` snapshots when describing the official workflow.

Additional env overrides now supported for realistic inputs:

- `topology_name` or `topology_path`
- `traffic_name` or `traffic_path`
- `traffic_model`
- `traffic_scenario`
- `traffic_scenario_version`
- `traffic_scenario_intensity`
- `traffic_scenario_duration`
- `traffic_scenario_frequency`
- `qos_target_norm_drop`
- `qos_min_volume`
- `qos_violation_penalty_scale`
- `qos_guard_margin`
- `qos_guard_margin_off`
- `qos_guard_margin_on`
- `qos_avg_delay_guard_multiplier`
- `qos_avg_delay_guard_margin_ms`
- `qos_recovery_delay_multiplier`
- `qos_recovery_delay_guard_margin_ms`
- `qos_p95_delay_threshold_ms`
- `stability_reversal_window_steps`
- `stability_reversal_penalty`
- `stability_min_steps_for_assessment`
- `stability_max_transition_rate`
- `stability_max_flap_rate`
- `stability_max_flap_count`

For reusable packaged topologies, prefer these stable `topology_name` values in new configs:

- `small`
- `medium`
- `large`

The legacy packaged names (`regional_ring`, `metro_hub`, `backbone_large`) are still accepted as compatibility aliases.

See [docs/custom_inputs.md](../docs/custom_inputs.md) for the JSON formats and examples.

Supported stochastic traffic scenario names now include:

- `normal`
- `diurnal`
- `burst`
- `hotspot`
- `anomaly`
- `flash_crowd`
- `multi_peak`

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
- `power_utilization_sensitive`
- `power_transition_on_joules`
- `power_transition_off_joules`
- `carbon_base_intensity_g_per_kwh`
- `carbon_amplitude_g_per_kwh`
- `carbon_period_seconds`

Model semantics:

- devices and links each contribute active or sleep power
- total power is separated into fixed and dynamic components
- `power_utilization_sensitive` optionally gates whether utilization contributes to the dynamic component
- `power_transition_on_joules` and `power_transition_off_joules` optionally add one-time energy costs when links are toggled on or off
- carbon is computed from total energy using the configured carbon-intensity profile

QoS semantics:

- `qos_target_norm_drop` is the official runtime delivery-loss threshold used by the env-level `qos_violation` signal
- `qos_min_volume` prevents that loss rule from firing before enough traffic has been observed
- `qos_avg_delay_guard_multiplier` and `qos_avg_delay_guard_margin_ms` define the ordinary delay guard:
  `avg_delay_ms <= max(avg_path_latency_ms * multiplier, avg_path_latency_ms + margin_ms)`
- `qos_recovery_delay_multiplier` and `qos_recovery_delay_guard_margin_ms` define the looser recovery trigger used when the controller needs to turn links back on
- `qos_p95_delay_threshold_ms` is reserved for future use; the current simulator/reporting flow does not yet compute p95 delay, so the official default remains unset

The final thesis-facing QoS acceptance gate still lives in the final evaluation pipeline, but it now uses one shared repository rule set with these default thresholds:

- delivered traffic loss vs the official traditional baseline: `2%`
- dropped traffic increase vs the official traditional baseline: `5%`
- average delay increase vs the official traditional baseline: `10%`
- average path latency increase vs the official traditional baseline: `10%`
- QoS-violation-rate increase vs the official traditional baseline: `0.02`

Backward compatibility:

- legacy `power_*` fields remain valid and still define the base power assumptions
- if `power_utilization_sensitive` is omitted, the legacy interpretation is preserved
- if `power_transition_on_joules` or `power_transition_off_joules` are omitted, they default to zero additional transition cost

Stability semantics:

- the existing cooldown/budget/calm controls remain the real switching guardrails
- `stability_reversal_window_steps` defines when an opposite-direction toggle on the same edge counts as a flap
- `stability_reversal_penalty` adds explicit reward pressure against rapid reversals without blocking QoS recovery paths
- `stability_min_steps_for_assessment` prevents very short smoke runs from being over-interpreted
- `stability_max_transition_rate`, `stability_max_flap_rate`, and `stability_max_flap_count` define the official run-level stability gate

The final evaluation bundle now preserves `stability_policy_name`, `stability_policy_signature`, `transition_rate_mean`, `flap_rate_mean`, and `stability_status` so switching behavior stays attached to the energy/QoS interpretation.

If omitted, GreenNet uses the lightweight defaults from `EnvConfig`. For the final submission path, treat the tracked `configs/train_*.json` files as the primary config-facing statement of the intended power/carbon assumptions.

## Official Acceptance Matrix

The canonical final benchmark definition now lives in [official_acceptance_v1.json](/Users/oltazagraxha/Desktop/GreenNet/configs/acceptance_matrices/official_acceptance_v1.json).

It is the authoritative machine-readable source of truth for:

- included policies
- included topology-backed benchmark cases
- included traffic scenarios and replay inputs
- seed count
- episode count
- step horizon
- baseline taxonomy identity
- the official run-family identity that should be preserved in both file artifacts and the SQLite run database

The official matrix includes these six benchmark cases:

- `small_normal`
- `small_commuter_replay`
- `medium_diurnal`
- `medium_hotspot`
- `large_flash_crowd`
- `large_flash_replay`

Historical bundles under `experiments/official_matrix_v*` remain preserved evidence, but they should be described as legacy or historical artifacts unless they were generated from this manifest-driven path.
