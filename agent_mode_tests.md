# Agent Mode Tests

## 1) Compile
Command:
```bash
python3 -m py_compile run_experiment.py experiments/run_matrix.py experiments/make_leaderboard.py experiments/package_official_matrix.py dashboard/page_compare.py
```
Expected:
- Exit code `0`.
- No syntax/import errors.

## 2) Tiny tag smoke test (matrix_test)
Command:
```bash
python3 experiments/run_matrix.py --policies all_on --scenarios normal --seeds 0 --episodes 1 --steps 10 --tag matrix_test
```
Expected:
- `results/results_summary_matrix_test.csv` exists.
- `results/results_summary.csv` exists and is updated.
- Latest run folder contains `run_meta.json` with:
  - `"tag": "matrix_test"`
  - `"created_at_utc": "..."`

## 3) Package smoke test (matrix_test)
Command:
```bash
python3 experiments/package_official_matrix.py \
  --tag matrix_test \
  --out experiments/official_matrix_test \
  --matrix-command "python3 experiments/run_matrix.py --policies all_on --scenarios normal --seeds 0 --episodes 1 --steps 10 --tag matrix_test" \
  --train-command "N/A"
```
Expected:
- `experiments/official_matrix_test/` contains:
  - `leaderboard_matrix_test.csv`
  - `leaderboard_source_matrix_test.csv`
  - `results_summary_matrix_test.csv`
  - `results_summary_by_seed_matrix_test.csv`
  - `notes.md`
- `results/leaderboard_matrix_test.csv` has 1 data row (`wc -l = 2`).

## 4) Full matrix packaging test (matrix_v6)
Command:
```bash
python3 experiments/package_official_matrix.py \
  --tag matrix_v6 \
  --out experiments/official_matrix_v6 \
  --matrix-command "<canonical matrix_v6 command>" \
  --train-command "<canonical training reference>"
```
Expected:
- `results/results_summary_matrix_v6.csv` has 90 rows + header (`wc -l = 91`).
- `results/leaderboard_matrix_v6.csv` has 9 rows + header (`wc -l = 10`).
- Official pack contains matching files.

## 5) Dashboard behavior
Command:
```bash
streamlit run app.py
```
(From `dashboard/` directory)

Expected:
- Compare page loads.
- Leaderboard selector defaults to newest official pack first, e.g. `experiments/official_matrix_v6/leaderboard_matrix_v6.csv`.
- KPI delta sign/color follows direction rules:
  - reward: higher is better (`delta = policy - baseline`)
  - dropped/energy/qos/toggles: lower is better (`delta = baseline - policy`)
- Download buttons work for leaderboard table and summary table.

## Bugs Found And Fixed
- `run_meta.json` tag drift: folder tag and run_meta tag could diverge. Fixed by normalizing CLI/config tag and always storing `tag` + `created_at_utc`.
- Leaderboard schema drift: missing metric aliases caused opaque failures. Fixed via explicit alias resolution and clear `ValueError` with expected/available columns and file path.
- Tagged summary naming: matrix runs now write `results_summary_<tag>.csv` when tag is set and always refresh `results_summary.csv` for backward compatibility.
- Dashboard default source: Compare page now prioritizes newest `experiments/official_matrix_*/leaderboard_matrix_*.csv`, then `results/leaderboard_*.csv`.
