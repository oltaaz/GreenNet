# GreenNet Final Evaluation

## Headline
- Generated at: `2026-03-21T23:59:16.895552+00:00`
- Source selection: `/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/smoke_v6/summary/results_summary_matrix_v6.csv`
- Selected runs: `90`
- Primary baseline policy: `heuristic`
- Baseline policies in scope: `all_on, heuristic`
- AI policies in scope: `ppo`
- Best overall policy: `heuristic`
- Best AI policy: `ppo`
- Overall best-policy summary: heuristic (baseline) vs heuristic: energy 0.00%, delivered 0.00%, QoS violation rate delta +0.0000, hypothesis=not_achieved
- Overall best-AI summary: ppo (ai) vs heuristic: energy -4.01%, delivered 3.12%, QoS violation rate delta -0.0001, hypothesis=not_achieved

## Hypothesis Gate
- Target: `energy reduction >= 15.0%` with acceptable QoS defined as `delivered loss <= 2.0%`, `dropped increase <= 5.0%`, `delay increase <= 10.0%`, `path latency increase <= 10.0%`, `QoS violation rate increase <= 0.0200`.

## Overall Comparison
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 30 | 0.020000 | -4.17% | 9061.355 | 14800.133 | 6771.284 | 17.907 | 0.7794 | 11690.87 | 0.087375 | acceptable | not_achieved |
| heuristic [best] | baseline | 30 | 0.019199 | 0.00% | 8783.352 | 15077.288 | 6654.987 | 18.057 | 0.7794 | 11691.63 | 0.083907 | acceptable | not_achieved |
| ppo [best-ai] | ai | 30 | 0.019969 | -4.01% | 9057.379 | 14802.364 | 6771.520 | 17.919 | 0.7794 | 11690.43 | 0.086987 | acceptable | not_achieved |

## Scenario: burst
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 10 | 0.020000 | -3.82% | 13014.731 | 23309.939 | 5057.034 | 15.987 | 0.8483 | 12725.00 | 0.087375 | acceptable | not_achieved |
| heuristic [best] | baseline | 10 | 0.019265 | 0.00% | 12622.697 | 23692.492 | 5058.973 | 16.146 | 0.8482 | 12723.10 | 0.084251 | acceptable | not_achieved |
| ppo [best-ai] | ai | 10 | 0.019969 | -3.66% | 13008.704 | 23315.493 | 5057.184 | 16.000 | 0.8485 | 12727.00 | 0.086994 | acceptable | not_achieved |

## Scenario: hotspot
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 10 | 0.020000 | -4.35% | 6328.347 | 12657.588 | 10324.954 | 21.385 | 0.7561 | 11341.10 | 0.087375 | acceptable | not_achieved |
| heuristic [best] | baseline | 10 | 0.019167 | 0.00% | 6018.796 | 12966.997 | 9921.125 | 21.463 | 0.7558 | 11336.40 | 0.083735 | acceptable | not_achieved |
| ppo [best-ai] | ai | 10 | 0.019969 | -4.19% | 6327.041 | 12658.550 | 10324.060 | 21.390 | 0.7561 | 11341.10 | 0.086989 | acceptable | not_achieved |

## Scenario: normal
| Policy | Class | Runs | Energy (kWh) | Energy vs baseline | Delivered | Dropped | Avg delay (ms) | Path latency (ms) | QoS rate | QoS count | Carbon (g) | QoS status | Hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| all_on | baseline | 10 | 0.020000 | -4.35% | 7840.986 | 8432.871 | 4931.864 | 16.349 | 0.7338 | 11006.50 | 0.087375 | acceptable | not_achieved |
| heuristic [best] | baseline | 10 | 0.019167 | 0.00% | 7708.561 | 8572.374 | 4984.862 | 16.563 | 0.7344 | 11015.40 | 0.083735 | acceptable | not_achieved |
| ppo [best-ai] | ai | 10 | 0.019969 | -4.19% | 7836.392 | 8433.049 | 4933.316 | 16.367 | 0.7335 | 11003.20 | 0.086977 | acceptable | not_achieved |
