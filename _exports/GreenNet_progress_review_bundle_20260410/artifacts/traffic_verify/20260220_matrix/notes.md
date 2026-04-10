# Controller Reproducibility Notes

- Baseline model artifact: `artifacts/locked/normal/20260220_111755_100k/model.zip`
- Evaluation seeds: `10..19`
- Episodes per run: `100`
- Final controller setting for locked scenario artifacts:
  - normal: `--eval-max-on-edges 16` (off4 gate run)
  - burst: `--eval-max-on-edges 16`
  - hotspot: `--eval-max-on-edges 16`
- Controller is eval-only (no model weight updates).
- Normal off0/off2 remain from baseline full-matrix runs; normal off4 uses controller for gate compliance.
