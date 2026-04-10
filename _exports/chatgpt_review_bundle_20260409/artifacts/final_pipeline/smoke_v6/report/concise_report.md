# GreenNet Final Pipeline Report

## Direct Answer
- No. ppo vs heuristic: energy -4.01%, delivered 3.12%, dropped -1.82%, delay 1.75%, path latency -0.77%, QoS rate delta -0.0001, QoS=acceptable, hypothesis=not_achieved.
- Source: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/summary/results_summary_matrix_v6.csv`
- Selected runs: `90`
- Primary baseline policy: `heuristic`

## Thesis Table
| Scope | Baseline | AI | Energy vs baseline | Delivered delta | Dropped delta | Delay delta | Path latency delta | QoS rate delta | QoS status | Hypothesis |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ALL | heuristic | ppo | -4.01% | 3.12% | -1.82% | 1.75% | -0.77% | -0.0001 | acceptable | not_achieved |
| burst | heuristic | ppo | -3.66% | 3.06% | -1.59% | -0.04% | -0.90% | +0.0003 | acceptable | not_achieved |
| hotspot | heuristic | ppo | -4.19% | 5.12% | -2.38% | 4.06% | -0.34% | +0.0003 | acceptable | not_achieved |
| normal | heuristic | ppo | -4.19% | 1.66% | -1.63% | -1.03% | -1.19% | -0.0008 | acceptable | not_achieved |

## Bundle Files
- Summary CSV: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/summary/results_summary_matrix_v6.csv`
- Leaderboard CSV: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/summary/leaderboard_matrix_v6.csv`
- Final evaluation report: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/summary/final_evaluation/final_evaluation_report.md`
- Plot: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/plots/policy_tradeoff_overall.csv`
- Plot: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/plots/policy_tradeoff_by_scenario.csv`
- Plot: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/plots/research_question_tradeoff.csv`
- Plot: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/plots/energy_vs_qos_tradeoff.png`
- Plot: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/plots/research_question_tradeoff.png`
