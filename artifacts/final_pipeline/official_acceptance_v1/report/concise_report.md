# GreenNet Final Pipeline Report

## Direct Answer
- No. PPO-Based Hybrid (AI) vs Traditional (All-On): energy 1.33%, delivered -1.22%, dropped 0.57%, delay 1.47%, path latency 1.95%, QoS rate delta +0.0000, QoS=acceptable, stability=stable, operational=not_achieved, hypothesis=not_achieved.
- Source: `artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv`
- Acceptance matrix: `official_acceptance_v1` (GreenNet Official Acceptance Matrix v1)
- Selected runs: `90`
- Official traditional baseline policy: `Traditional (All-On)`
- Strongest heuristic baseline policy: `Energy-Aware Heuristic`
- AI policy note: `ppo` in this benchmark denotes the PPO-based hybrid controller, not a raw unwrapped PPO policy.

## Thesis Table
| Scope | Baseline | AI | Baseline energy (kWh) | AI energy (kWh) | Energy vs baseline | Delivered delta | Dropped delta | Delay delta | Path latency delta | QoS rate delta | QoS status | Stability | Operational | Hypothesis |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| ALL | Traditional (All-On) | PPO-Based Hybrid (AI) | 0.017216 | 0.016986 | 1.33% | -1.22% | 0.57% | 1.47% | 1.95% | +0.0000 | acceptable | stable | not_achieved | not_achieved |
| custom | Traditional (All-On) | PPO-Based Hybrid (AI) | 0.017781 | 0.017781 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | +0.0000 | acceptable | stable | not_achieved | not_achieved |
| diurnal | Traditional (All-On) | PPO-Based Hybrid (AI) | 0.015552 | 0.014936 | 3.96% | -1.76% | 1.58% | 1.48% | 2.52% | +0.0000 | acceptable | stable | not_achieved | not_achieved |
| flash_crowd | Traditional (All-On) | PPO-Based Hybrid (AI) | 0.023940 | 0.023914 | 0.11% | 0.09% | -0.02% | 0.77% | 1.78% | +0.0000 | acceptable | stable | not_achieved | not_achieved |
| hotspot | Traditional (All-On) | PPO-Based Hybrid (AI) | 0.015687 | 0.015238 | 2.86% | -1.97% | 1.44% | 0.89% | 2.08% | +0.0000 | acceptable | stable | not_achieved | not_achieved |
| normal | Traditional (All-On) | PPO-Based Hybrid (AI) | 0.012554 | 0.012266 | 2.29% | -4.83% | 5.08% | 4.84% | 5.79% | +0.0000 | not_acceptable | stable | not_achieved | not_achieved |

## Bundle Files
- Summary CSV: `/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv`
- Leaderboard CSV: `/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/summary/leaderboard_official_acceptance_v1.csv`
- Final evaluation report: `/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_report.md`
- Rerun accepted final benchmark: `python -m greennet.evaluation.reproduction --output-dir /Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1`
- Plot: `/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/plots/policy_tradeoff_overall.csv`
- Plot: `/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/plots/policy_tradeoff_by_scenario.csv`
- Plot: `/Users/oltazagraxha/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1/plots/research_question_tradeoff.csv`
