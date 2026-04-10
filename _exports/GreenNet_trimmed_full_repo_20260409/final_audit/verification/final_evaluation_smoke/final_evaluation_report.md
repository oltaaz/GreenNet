# GreenNet Final Evaluation

## Headline
- Generated at: `2026-04-09T12:29:50.604134+00:00`
- Source selection: `experiments/official_matrix_v6/results_summary_matrix_v6.csv`
- Selected runs: `90`
- Primary baseline policy: `heuristic`
- Baseline policies in scope: `all_on, heuristic`
- AI policies in scope: `ppo`
- Routing baselines in scope: `min_hop_single_path`
- Routing link-cost models in scope: `unit`
- Routing comparison consistency: `consistent`
- Best overall policy: `heuristic`
- Best AI policy: `ppo`
- Overall best-policy summary: heuristic (baseline) vs heuristic: energy 0.00%, delivered 0.00%, QoS violation rate delta n/a, hypothesis=insufficient_data
- Overall best-AI summary: ppo (ai) vs heuristic: energy -4.01%, delivered 3.12%, QoS violation rate delta n/a, hypothesis=insufficient_data

## Hypothesis Gate
- Target: `energy reduction >= 15.0%` with acceptable QoS defined as `delivered loss <= 2.0%`, `dropped increase <= 5.0%`, `delay increase <= 10.0%`, `path latency increase <= 10.0%`, `QoS violation rate increase <= 0.0200`.

## Overall Comparison
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 30 | 0.020000 | -4.17% | 9061.355 | 14800.133 | 6771.284 | n/a | n/a | n/a | 0.087375 | insufficient_data | insufficient_data |
| heuristic [best] | baseline | 30 | 0.019199 | 0.00% | 8783.352 | 15077.288 | 6654.987 | n/a | n/a | n/a | 0.083907 | insufficient_data | insufficient_data |
| ppo [best-ai] | ai | 30 | 0.019969 | -4.01% | 9057.379 | 14802.364 | 6771.520 | n/a | n/a | n/a | 0.086987 | insufficient_data | insufficient_data |

## Scenario: burst
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 10 | 0.020000 | -3.82% | 13014.731 | 23309.939 | 5057.034 | n/a | n/a | n/a | 0.087375 | insufficient_data | insufficient_data |
| heuristic [best] | baseline | 10 | 0.019265 | 0.00% | 12622.697 | 23692.492 | 5058.973 | n/a | n/a | n/a | 0.084251 | insufficient_data | insufficient_data |
| ppo [best-ai] | ai | 10 | 0.019969 | -3.66% | 13008.704 | 23315.493 | 5057.184 | n/a | n/a | n/a | 0.086994 | insufficient_data | insufficient_data |

## Scenario: hotspot
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 10 | 0.020000 | -4.35% | 6328.347 | 12657.588 | 10324.954 | n/a | n/a | n/a | 0.087375 | insufficient_data | insufficient_data |
| heuristic [best] | baseline | 10 | 0.019167 | 0.00% | 6018.796 | 12966.997 | 9921.125 | n/a | n/a | n/a | 0.083735 | insufficient_data | insufficient_data |
| ppo [best-ai] | ai | 10 | 0.019969 | -4.19% | 6327.041 | 12658.550 | 10324.060 | n/a | n/a | n/a | 0.086989 | insufficient_data | insufficient_data |

## Scenario: normal
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 10 | 0.020000 | -4.35% | 7840.986 | 8432.871 | 4931.864 | n/a | n/a | n/a | 0.087375 | insufficient_data | insufficient_data |
| heuristic [best] | baseline | 10 | 0.019167 | 0.00% | 7708.561 | 8572.374 | 4984.862 | n/a | n/a | n/a | 0.083735 | insufficient_data | insufficient_data |
| ppo [best-ai] | ai | 10 | 0.019969 | -4.19% | 7836.392 | 8433.049 | 4933.316 | n/a | n/a | n/a | 0.086977 | insufficient_data | insufficient_data |
