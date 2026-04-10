# Custom Topologies And Traffic Inputs

GreenNet still supports its original synthetic/random topology and stochastic traffic generation.
The new file-backed inputs are optional overrides on top of that default behavior.

## What Was Added

You can now provide:

- a named topology bundled with the repo
- a custom topology JSON file
- a named traffic replay profile bundled with the repo
- a custom traffic replay JSON file

When no explicit topology or traffic input is provided, GreenNet keeps using the existing synthetic generators.

## Supported Config Fields

These `EnvConfig` fields are now supported in train/eval configs and saved `env_config.json` files:

- `topology_name`
- `topology_path`
- `traffic_model`
- `traffic_name`
- `traffic_path`
- `traffic_scenario`
- `traffic_scenario_version`
- `traffic_scenario_intensity`
- `traffic_scenario_duration`
- `traffic_scenario_frequency`

Resolution rules:

- `topology_path` overrides `topology_name`
- `traffic_path` overrides `traffic_name`
- explicit topology inputs override random topology generation
- explicit traffic inputs override `traffic_model` / `traffic_scenario`
- when documenting the final submission, keep `configs/` as the canonical config family and treat root-level `train_*.json` files as historical snapshots

Relative `topology_path` and `traffic_path` values in config files are resolved relative to the config file location.

## Named Inputs

Bundled named topologies:

- official reusable classes:
  - `small` -> loads the packaged `regional_ring` topology
  - `medium` -> loads the packaged `metro_hub` topology
  - `large` -> loads the packaged `backbone_large` topology
- compatibility aliases still accepted:
  - `regional_ring`
  - `metro_hub`
  - `backbone_large`

Bundled named traffic profiles:

- `commuter_bursts`
- `commuter_matrices`
- `regional_ring_commuter_matrices`
- `backbone_large_flash_crowd_bursts`

Built-in stochastic traffic scenarios:

- `normal`
- `diurnal`
- `burst`
- `hotspot`
- `anomaly`
- `flash_crowd`
- `multi_peak`

Compatibility aliases still accepted where already used:

- `failure` -> `anomaly`
- `normal/diurnal` -> `diurnal`
- `flash crowd` -> `flash_crowd`

## Experiment Runner Usage

Direct CLI usage:

```bash
python3 run_experiment.py \
  --policy all_on \
  --scenario flash_crowd \
  --seed 17 \
  --steps 50 \
  --topology-name medium \
  --traffic-scenario-intensity 1.2
```

Config-driven usage:

```json
{
  "policy": "heuristic",
  "scenario": "custom",
  "episodes": 3,
  "steps": 100,
  "env": {
    "topology_name": "large",
    "traffic_name": "backbone_large_flash_crowd_bursts"
  }
}
```

Train CLI usage works through the same env block:

```json
{
  "seed": 42,
  "total_timesteps": 20000,
  "env": {
    "topology_path": "inputs/my_topology.json",
    "traffic_path": "inputs/my_traffic.json"
  }
}
```

Matrix and official-pipeline usage support the same selectors:

```bash
python3 experiments/run_matrix.py \
  --policies all_on,heuristic,ppo \
  --scenarios normal,burst,hotspot \
  --seeds 0,1,2 \
  --topology-name small \
  --traffic-name regional_ring_commuter_matrices

python3 -m greennet.evaluation.final_pipeline \
  --tag topology_small_demo \
  --topology-name small \
  --traffic-name regional_ring_commuter_matrices
```

## Topology File Format

Topology files are JSON objects with:

- `format_version`: must be `1`
- `directed`: optional, defaults to `false`
- `nodes`: required list of contiguous integer node IDs starting at `0`
- `edge_defaults`: optional default edge attributes
- `edges`: required list of edge objects

Example:

```json
{
  "format_version": 1,
  "directed": false,
  "nodes": [0, 1, 2, 3],
  "edge_defaults": {
    "capacity": 15.0,
    "latency_ms": 5.0,
    "active": true
  },
  "edges": [
    {"source": 0, "target": 1},
    {"source": 1, "target": 2, "capacity": 20.0},
    {"source": 2, "target": 3},
    {"source": 3, "target": 0}
  ]
}
```

Validation rules:

- node IDs must be exactly `0..N-1`
- the graph must be connected as an undirected physical graph
- no duplicate edges
- no self-loops
- `capacity` and `weight` must be positive when provided
- `latency_ms` must be non-negative when provided

## Traffic File Format

Traffic replay files are JSON objects with:

- `format_version`: must be `1`
- `node_count`: optional, but if present it must match the active topology
- `repeat`: optional, defaults to `false`
- exactly one of:
  - `bursts`
  - `matrices`

These replay files are topology-specific by `node_count`. A named or custom traffic replay file must match the active topology's node count.

### Burst Trace Format

Each burst entry has:

- `source`
- `destination`
- `size`
- `start_time`
- `duration` (optional, defaults to `1`)

Example:

```json
{
  "format_version": 1,
  "node_count": 4,
  "repeat": true,
  "cycle_length": 8,
  "bursts": [
    {"source": 0, "destination": 2, "size": 6.0, "start_time": 0, "duration": 2},
    {"source": 1, "destination": 3, "size": 4.0, "start_time": 3, "duration": 1}
  ]
}
```

### Traffic Matrix Format

`matrices` is a list of square `node_count x node_count` demand matrices.
Each matrix index is treated as one simulator step.

Example:

```json
{
  "format_version": 1,
  "node_count": 4,
  "repeat": true,
  "matrices": [
    [
      [0, 3, 0, 0],
      [0, 0, 2, 0],
      [0, 0, 0, 4],
      [1, 0, 0, 0]
    ],
    [
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [2, 0, 0, 0]
    ]
  ]
}
```

Validation rules:

- `source` and `destination` must be valid node IDs
- `source != destination`
- burst `size` must be positive
- matrix values must be finite and non-negative
- matrix diagonal entries must be zero
- at least one positive demand entry must exist

## Notes

- Saved `env_config.json` files are normalized so file-backed topologies persist the correct `node_count` and `directed` values.
- Relative file paths are stored as absolute paths in saved run configs to keep result folders replayable.
- Prefer `small`, `medium`, and `large` in new configs and scripts. The older packaged names remain accepted as compatibility aliases.
- `traffic_name` / `traffic_path` take precedence over stochastic `traffic_model` / `traffic_scenario` settings and are recorded in run metadata alongside `traffic_mode`.
- The built-in `hotspot`, `flash_crowd`, and `multi_peak` stochastic scenarios now generate topology-safe hotspot pairs so they work across the D6 `small`, `medium`, and `large` topology classes.
- For PPO evaluation, a custom topology must still be action-space compatible with the checkpoint you load.
