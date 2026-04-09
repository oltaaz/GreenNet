# GreenNet Final Submission Overview

This note is the compact reviewer guide for the final submission. It is written to match the evidence actually present in the repository.

## 1. Official Workflow

1. Install the project with `python3 -m pip install -e .[test,train]`.
2. Train or rerun policies using `train.py` and the canonical configs in `configs/`.
3. Run single experiments with `run_experiment.py` or larger comparisons with `experiments/run_matrix.py`.
4. Review the curated matrix and final evaluation in `experiments/official_matrix_v6/`.
5. Use `artifacts/traffic_verify/20260220_matrix/` and `artifacts/locked/` for reproducibility evidence.
6. Use `artifacts/final_submission/matrix_v6/manifest.json` and `traceability.csv` as the reviewer-facing map for the final claim.

## 2. What The System Actually Contains

- simulator core: routing, topology, delay, energy, and carbon accounting
- baseline controllers: `all_on` and `heuristic`
- learned controller: `ppo`
- API layer: `api_app.py` plus the persistence layer under `greennet/`
- public demo: `frontend/greennet-ui/`
- internal analyst tooling: `dashboard/`

The official comparison is between a technical baseline and the learned controller under the same scenarios and seeds.

## 3. Honest Final Claim

The final matrix does not support a strong "AI wins on energy" claim.

The correct submission claim is:

- GreenNet demonstrates a reproducible energy-vs-QoS routing evaluation framework
- the heuristic baseline remains strongest in the official matrix
- PPO is the best AI policy in the final aggregate, but it does not beat the heuristic on energy
- the research value is the simulator, the comparison framework, and the clear reporting of a mixed outcome

## 4. Evidence Manifest

| Artifact | Role |
| --- | --- |
| `experiments/official_matrix_v6/results_summary_matrix_v6.csv` | source table for the final matrix aggregation |
| `experiments/official_matrix_v6/results_summary_by_seed_matrix_v6.csv` | seed-level traceability |
| `experiments/official_matrix_v6/leaderboard_matrix_v6.csv` | ranked comparison table |
| `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md` | final thesis-facing evaluation and hypothesis gate |
| `experiments/official_matrix_v6/final_evaluation/final_evaluation_summary.csv` | machine-readable final summary |
| `experiments/official_matrix_v6/final_evaluation/final_evaluation_summary.json` | machine-readable final summary |
| `artifacts/final_submission/matrix_v6/manifest.json` | canonical evidence manifest for the preserved final bundle |
| `artifacts/final_submission/matrix_v6/traceability.csv` | claim-to-artifact mapping for reviewer traceability |
| `artifacts/traffic_verify/20260220_matrix/matrix_status.md` | deterministic traffic verification status |
| `artifacts/traffic_verify/20260220_matrix/traffic_eval_summary.csv` | traffic verification results |
| `artifacts/locked/` | locked scenario evidence and retained run bundles |

This manifest is intentionally narrow. It does not pretend that every historical run or experimental branch is part of the final submission story.

## 5. Artifact Glossary

- `configs/` - canonical training configs
- `experiments/official_matrix_v6/` - final comparison bundle
- `artifacts/traffic_verify/20260220_matrix/` - reproducibility verification logs
- `artifacts/locked/` - scenario-locked evidence folders
- `runs/` - operational working directory, not the only source of truth
- `models/` - saved checkpoints and compatibility artifacts

## 6. Responsible Design And Limitations

- QoS must remain visible whenever energy savings are discussed.
- The final evidence supports a mixed outcome, not a headline of clear AI dominance.
- `Impact Predictor` is exploratory unless its artifacts are bundled and explicitly verified.
- Older command sheets and root-level config snapshots exist for compatibility, but they are not the canonical submission path.
- The frontend has a public demo role, while the Streamlit dashboard is internal tooling.

## 7. What A Reviewer Should Check First

- the headline in `experiments/official_matrix_v6/final_evaluation/final_evaluation_report.md`
- the source table in `experiments/official_matrix_v6/results_summary_matrix_v6.csv`
- the scenario verification status in `artifacts/traffic_verify/20260220_matrix/matrix_status.md`
- the canonical config set in `configs/README.md`
