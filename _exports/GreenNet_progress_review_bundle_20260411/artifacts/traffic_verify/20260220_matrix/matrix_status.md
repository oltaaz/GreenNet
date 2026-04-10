# Traffic Matrix Status (Deterministic trained vs noop)

| scenario | off0 | off2 | off4 | note |
|---|---|---|---|---|
| normal | PASS | PASS | PASS | off4 with controller cap=16 |
| burst | PASS | PASS | PASS | controller cap=16 |
| hotspot | PASS | PASS | PASS | controller cap=16 |

See `artifacts/traffic_verify/20260220_matrix/traffic_eval_summary.csv` for full deltas and telemetry.
