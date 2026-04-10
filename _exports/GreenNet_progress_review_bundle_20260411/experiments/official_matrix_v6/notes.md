# Official Matrix matrix_v6

## Metadata
- tag: matrix_v6
- generated_at_utc: 2026-02-06T00:38:45.243976+00:00

## Commit
- 8158c772be0c682d7e1637d6cd27237334071f1f

## Commands
- train: python3 train.py --config <your_train_config.json> --timesteps 300000
- matrix: python3 experiments/run_matrix.py --policies all_on,heuristic,ppo --scenarios normal,burst,hotspot --seeds 0,1,2,3,4,5,6,7,8,9 --episodes 50 --steps 300 --tag matrix_v6 --ppo-model runs/20260205_222626/ppo_greennet.zip
- topology selection note: current code also supports `--topology-name small|medium|large` or `--topology-path <file.json>` through the same matrix runner, but the preserved `matrix_v6` bundle remains a historical reference.

## Policies
- all_on, heuristic, ppo

## Scenarios
- burst, hotspot, normal

## Seeds
- 0-9 (10 seeds)

## Episodes / Steps
- episodes: 50
- steps: 300

## PPO model path
- runs/20260205_222626/ppo_greennet.zip
- Historical reference from the original matrix run; the checkpoint is not bundled in this checkout.

## Canonical final bundle
- artifacts/final_submission/matrix_v6/
- Reviewer-facing package for the preserved official matrix evidence.
- Final claim: `all_on` is the official traditional baseline, `heuristic` is the strongest handcrafted heuristic baseline, PPO is best AI, and the hypothesis gate is not achieved.

## Headline result
- burst: reward +31.433, drops -376.999, energy +0.000704 | hotspot: reward +53.754, drops -308.447, energy +0.000803 | normal: reward +23.037, drops -139.325, energy +0.000803 | overall: reward +36.075, drops -274.924, energy +0.000770
