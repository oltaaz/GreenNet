# GreenNet Final Evaluation

## Headline
- Generated at: `2026-04-10T15:38:31.696624+00:00`
- Source selection: `artifacts/final_pipeline/official_acceptance_v1/summary/results_summary_official_acceptance_v1.csv`
- Selected runs: `90`
- Official traditional baseline policy: `Traditional (All-On)`
- Strongest heuristic baseline policy: `Energy-Aware Heuristic`
- Non-AI policies in scope: `Traditional (All-On), Energy-Aware Heuristic`
- AI policies in scope: `PPO-Based Hybrid (AI)`
- Routing baselines in scope: `min_hop_single_path`
- Routing link-cost models in scope: `unit`
- Routing comparison consistency: `consistent`
- Best overall policy: `PPO-Based Hybrid (AI)`
- Best AI policy: `PPO-Based Hybrid (AI)`
- Overall best-policy summary: PPO-Based Hybrid (AI) (ai_policy) vs Traditional (All-On): baseline energy 0.017 kWh, ai energy 0.016986 kWh, energy 1.33%, delivered -1.22%, QoS violation rate delta +0.0000, QoS=acceptable, stability=stable, operational=not_achieved
- Overall best-AI summary: PPO-Based Hybrid (AI) (ai_policy) vs Traditional (All-On): baseline energy 0.017 kWh, ai energy 0.016986 kWh, energy 1.33%, delivered -1.22%, QoS violation rate delta +0.0000, QoS=acceptable, stability=stable, operational=not_achieved
- AI policy note: `ppo` in this repository's final benchmark denotes the PPO-based hybrid controller with rule-based safety, recovery, and calm-off overrides over the fixed routing baseline.
- Benchmark verdict: baseline `Traditional (All-On)` energy 0.017216 kWh vs AI `PPO-Based Hybrid (AI)` energy 0.016986 kWh; energy 1.33%; QoS=acceptable; hypothesis=not_achieved.

## Hypothesis Gate
- Target: `energy reduction >= 15.0%` with acceptable QoS defined as `delivered loss <= 2.0%`, `dropped increase <= 5.0%`, `delay increase <= 10.0%`, `path latency increase <= 10.0%`, `QoS violation rate increase <= 0.0200`.
- Stability gate: use the exported `stability_status` from the centralized stability policy. Operational success requires the energy/QoS hypothesis to be achieved while stability remains `stable`.

## Overall Comparison
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Transitions | Flap rate | Carbon (g) | QoS status | Stability | Operational | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Traditional (All-On) | traditional_baseline | 30 | 0.017216 | 0.00% | 8748.516 | 18679.552 | 2676.151 | 9.449 | 0.7824 | 4694.20 | 0.00 | 0.00% | 6.890917 | acceptable | stable | not_achieved | not_achieved |
| Energy-Aware Heuristic | heuristic_baseline | 30 | 0.016753 | 2.69% | 8497.837 | 18930.232 | 2739.338 | 9.773 | 0.7824 | 4694.20 | 1.49 | 0.56% | 6.705687 | not_acceptable | stable | not_achieved | not_achieved |
| PPO-Based Hybrid (AI) [best] | ai_policy | 30 | 0.016986 | 1.33% | 8642.122 | 18785.947 | 2715.440 | 9.634 | 0.7824 | 4694.20 | 0.60 | 0.00% | 6.799010 | acceptable | stable | not_achieved | not_achieved |

## Scenario: custom
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Transitions | Flap rate | Carbon (g) | QoS status | Stability | Operational | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| all_on | traditional_baseline | 10 | 0.017781 | 0.00% | 7695.000 | 630.000 | 1819.757 | 10.291 | 0.4750 | 2850.00 | 0.00 | 0.00% | 7.117114 | acceptable | stable | not_achieved | not_achieved |
| Energy-Aware Heuristic [best] | heuristic_baseline | 10 | 0.017333 | 2.52% | 7695.000 | 630.000 | 1819.757 | 10.291 | 0.4750 | 2850.00 | 1.00 | 0.00% | 6.938055 | acceptable | stable | not_achieved | not_achieved |
| PPO-Based Hybrid (AI) [best-ai] | ai_policy | 10 | 0.017781 | 0.00% | 7695.000 | 630.000 | 1819.757 | 10.291 | 0.4750 | 2850.00 | 0.00 | 0.00% | 7.117114 | acceptable | stable | not_achieved | not_achieved |

## Scenario: diurnal
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Transitions | Flap rate | Carbon (g) | QoS status | Stability | Operational | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| all_on | traditional_baseline | 5 | 0.015552 | 0.00% | 6532.255 | 7264.175 | 2626.050 | 7.849 | 0.9011 | 5406.60 | 0.00 | 0.00% | 6.224865 | acceptable | stable | not_achieved | not_achieved |
| heuristic | heuristic_baseline | 5 | 0.014696 | 5.50% | 6069.771 | 7726.659 | 2743.316 | 8.303 | 0.9011 | 5406.60 | 2.00 | 0.00% | 5.882429 | not_acceptable | stable | not_achieved | not_achieved |
| PPO-Based Hybrid (AI) [best] | ai_policy | 5 | 0.014936 | 3.96% | 6417.598 | 7378.832 | 2664.851 | 8.046 | 0.9011 | 5406.60 | 1.44 | 0.00% | 5.978611 | acceptable | stable | not_achieved | not_achieved |

## Scenario: flash_crowd
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Transitions | Flap rate | Carbon (g) | QoS status | Stability | Operational | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| all_on | traditional_baseline | 5 | 0.023940 | 0.00% | 14921.138 | 85467.692 | 3884.243 | 10.890 | 0.9697 | 5818.00 | 0.00 | 0.00% | 9.582446 | acceptable | stable | not_achieved | not_achieved |
| Energy-Aware Heuristic [best] | heuristic_baseline | 5 | 0.023813 | 0.53% | 14839.446 | 85549.384 | 3863.764 | 10.970 | 0.9697 | 5818.00 | 0.34 | 0.00% | 9.531605 | acceptable | stable | not_achieved | not_achieved |
| PPO-Based Hybrid (AI) [best-ai] | ai_policy | 5 | 0.023914 | 0.11% | 14934.790 | 85454.040 | 3914.073 | 11.083 | 0.9697 | 5818.00 | 0.17 | 0.00% | 9.572062 | acceptable | stable | not_achieved | not_achieved |

## Scenario: hotspot
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Transitions | Flap rate | Carbon (g) | QoS status | Stability | Operational | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| all_on | traditional_baseline | 5 | 0.015687 | 0.00% | 7650.130 | 10475.890 | 3012.927 | 7.778 | 0.9405 | 5643.20 | 0.00 | 0.00% | 6.279015 | acceptable | stable | not_achieved | not_achieved |
| heuristic | heuristic_baseline | 5 | 0.015138 | 3.50% | 7091.027 | 11034.993 | 3071.201 | 8.194 | 0.9405 | 5643.20 | 1.80 | 0.00% | 6.059260 | not_acceptable | stable | not_achieved | not_achieved |
| PPO-Based Hybrid (AI) [best] | ai_policy | 5 | 0.015238 | 2.86% | 7499.442 | 10626.578 | 3039.834 | 7.939 | 0.9405 | 5643.20 | 1.24 | 0.00% | 6.099526 | acceptable | stable | not_achieved | not_achieved |

## Scenario: normal
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Transitions | Flap rate | Carbon (g) | QoS status | Stability | Operational | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| all_on [best] | traditional_baseline | 5 | 0.012554 | 0.00% | 7997.576 | 7609.554 | 2894.173 | 9.596 | 0.9329 | 5597.40 | 0.00 | 0.00% | 5.024950 | acceptable | stable | not_achieved | not_achieved |
| heuristic | heuristic_baseline | 5 | 0.012203 | 2.79% | 7596.775 | 8010.355 | 3118.234 | 10.587 | 0.9329 | 5597.40 | 2.78 | 1.80% | 4.884719 | not_acceptable | stable | not_achieved | not_achieved |
| PPO-Based Hybrid (AI) [best-ai] | ai_policy | 5 | 0.012266 | 2.29% | 7610.900 | 7996.230 | 3034.368 | 10.152 | 0.9329 | 5597.40 | 0.73 | 0.00% | 4.909635 | not_acceptable | stable | not_achieved | not_achieved |
