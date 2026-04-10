from __future__ import annotations

import csv
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

try:
    from sb3_contrib import MaskablePPO  # type: ignore
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
    MASKABLE_AVAILABLE = True
except Exception:  # pragma: no cover
    MaskablePPO = None  # type: ignore
    ActionMasker = None  # type: ignore
    MASKABLE_AVAILABLE = False

from greennet.utils.config import save_env_config, save_train_config
from greennet.rl.eval import eval_policy, custom_gate

def _sweep_tag_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def run_sweep(
    config: Dict[str, Any],
    train_seed: int,
    eval_seed: int,
    sweep_timesteps: int,
    sweep_episodes: int,
    train_drop_lambda: float,
    eval_drop_lambda: float,
    sweep_samples: int,
    progress_every: int,
) -> None:
    """Random-sample a reward/behavior sweep and write summary CSV."""
    from greennet.cli.train_cli import (
        ProgressBarCallback,
        build_train_env_config,
        make_env,
        save_requirements_copy,
        set_seeds,
    )

    sweep_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = Path("runs") / f"sweep_results_{sweep_timestamp}.csv"

    rng = np.random.default_rng(train_seed)

    # Reward shaping knobs
    energy_weights = [60.0, 80.0, 120.0, 160.0, 220.0]
    attempt_penalties = [0.0, 0.002, 0.005, 0.01, 0.02]
    off_penalties = [0.02, 0.05, 0.10, 0.15, 0.20]
    on_penalties = [0.2, 0.5, 1.0, 1.5]

    # Behavioral / feasibility knobs
    max_totals = [2, 4, 6, 8, 10]
    max_offs = [0, 1, 2, 3, 4, 5]
    off_start_guards = [0, 3, 6, 10]
    cooldowns = [2, 5, 10]
    global_cds = [0, 2, 5]
    util_thresholds = [0.75, 0.80, 0.85, 0.90]

    # Build only feasible combinations (MO <= MT).
    full: list[tuple[float, float, float, float, int, int, int, int, int, float]] = []
    for ew, tap, off_s, on_s in product(energy_weights, attempt_penalties, off_penalties, on_penalties):
        for mt in max_totals:
            for mo in [m for m in max_offs if m <= mt]:
                for g, cd, gcd, ut in product(off_start_guards, cooldowns, global_cds, util_thresholds):
                    full.append((ew, tap, off_s, on_s, mo, mt, g, cd, gcd, ut))

    k = max(1, min(int(sweep_samples), len(full)))
    picked_idx = rng.choice(len(full), size=k, replace=False)
    grid = [full[i] for i in picked_idx]
    total = len(grid)

    base_env_config = build_train_env_config(config)
    base_env_config = replace(
        base_env_config,
        drop_penalty_lambda=float(train_drop_lambda),
        debug_logs=False,
    )

    fieldnames = [
        "tag",
        "energy_weight",
        "toggle_attempt_penalty",
        "toggle_off_penalty_scale",
        "toggle_on_penalty_scale",
        "max_off_toggles_per_episode",
        "max_total_toggles_per_episode",
        "off_start_guard_decision_steps",
        "toggle_cooldown_steps",
        "global_toggle_cooldown_steps",
        "util_block_threshold",
        "trained_reward_mean",
        "trained_energy_mean",
        "trained_dropped_mean",
        "trained_norm_drop_mean",
        "trained_toggles_applied_mean",
        "trained_off_edges_mean",
        "noop_energy_mean",
        "noop_dropped_mean",
        "delta_energy",
        "delta_dropped",
        "trained_vs_random_same",
        "det_pass",
        "det_fail_reason",
        "det_delta_reward",
        "det_delta_energy",
        "det_delta_dropped",
        "stoch_pass",
        "stoch_fail_reason",
        "stoch_delta_reward",
        "stoch_delta_energy",
        "stoch_delta_dropped",
        "score",
    ]

    rows: list[Dict[str, Any]] = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (ew, tap, off_s, on_s, mo, mt, g, cd, gcd, ut) in enumerate(grid, start=1):
            if int(mo) > int(mt):
                # Feasibility guard; should never trigger because the grid is pre-filtered.
                continue
            tag = (
                f"E{_sweep_tag_value(ew)}"
                f"_A{_sweep_tag_value(tap)}"
                f"_OFF{_sweep_tag_value(off_s)}"
                f"_ON{_sweep_tag_value(on_s)}"
                f"_MO{int(mo)}"
                f"_MT{int(mt)}"
                f"_G{int(g)}"
                f"_CD{int(cd)}"
                f"_GCD{int(gcd)}"
                f"_UT{_sweep_tag_value(ut)}"
            )
            run_dir = Path("runs") / f"{sweep_timestamp}__sweep_{tag}"

            print(f"\n[sweep {i}/{total}] {tag}")

            set_seeds(train_seed)
            run_config: Dict[str, Any] = dict(config)
            run_config["ppo"] = dict(config.get("ppo", {}))
            run_config["total_timesteps"] = int(sweep_timesteps)

            env_cfg_train = replace(
                base_env_config,
                energy_weight=float(ew),
                toggle_attempt_penalty=float(tap),
                toggle_off_penalty_scale=float(off_s),
                toggle_on_penalty_scale=float(on_s),
                max_off_toggles_per_episode=int(mo),
                max_total_toggles_per_episode=int(mt),
                off_start_guard_decision_steps=int(g),
                toggle_cooldown_steps=int(cd),
                global_toggle_cooldown_steps=int(gcd),
                util_block_threshold=float(ut),
            )

            save_train_config(run_dir, run_config)
            save_requirements_copy(run_dir)
            save_env_config(run_dir, env_cfg_train)

            env = DummyVecEnv([make_env(train_seed, env_cfg_train)])
            if MASKABLE_AVAILABLE and MaskablePPO is not None:
                model = MaskablePPO(env=env, **run_config["ppo"])
            else:
                model = PPO(env=env, **run_config["ppo"])

            print(
                f"[sweep] training {type(model).__name__} "
                f"timesteps={int(sweep_timesteps)} device={model.policy.device}"
            )
            callback = ProgressBarCallback(
                total_timesteps=int(sweep_timesteps),
                every_steps=int(progress_every),
                bar_len=30,
                verbose=1,
            )
            model.learn(total_timesteps=int(sweep_timesteps), callback=callback, progress_bar=False)
            model.save(str(run_dir / "ppo_greennet"))
            env.close()

            eval_cfg = replace(
                env_cfg_train,
                initial_off_edges=0,
                drop_penalty_lambda=float(eval_drop_lambda),
                debug_logs=False,
            )
            trained_det = eval_policy(
                model,
                eval_cfg,
                episodes=int(sweep_episodes),
                seed=eval_seed,
                label="trained",
                deterministic=True,
                policy_mode="model",
            )
            noop_det = eval_policy(
                None,
                eval_cfg,
                episodes=int(sweep_episodes),
                seed=eval_seed,
                label="always_noop",
                deterministic=True,
                policy_mode="noop",
            )
            det_delta_reward = float(trained_det["reward_mean"] - noop_det["reward_mean"])
            det_delta_energy = float(trained_det["energy_mean"] - noop_det["energy_mean"])
            det_delta_dropped = float(trained_det["dropped_mean"] - noop_det["dropped_mean"])
            det_pass, det_reason = custom_gate(det_delta_reward, det_delta_energy, det_delta_dropped)

            stoch_seeds = [0, 1, 2, 3, 4]
            stoch_delta_rewards: list[float] = []
            stoch_delta_energies: list[float] = []
            stoch_delta_droppeds: list[float] = []
            stoch_toggles_applied_vals: list[float] = []
            stoch_off_edges_vals: list[float] = []
            stoch_noop_energy_vals: list[float] = []
            stoch_noop_dropped_vals: list[float] = []
            stoch_trained_energy_vals: list[float] = []
            stoch_trained_dropped_vals: list[float] = []
            stoch_random_energy_vals: list[float] = []
            stoch_random_dropped_vals: list[float] = []

            for st_seed in stoch_seeds:
                trained_stoch = eval_policy(
                    model,
                    eval_cfg,
                    episodes=int(sweep_episodes),
                    seed=int(st_seed),
                    label="trained",
                    deterministic=False,
                    policy_mode="model",
                )
                noop_stoch = eval_policy(
                    None,
                    eval_cfg,
                    episodes=int(sweep_episodes),
                    seed=int(st_seed),
                    label="always_noop",
                    deterministic=True,
                    policy_mode="noop",
                )
                random_masked = eval_policy(
                    model,
                    eval_cfg,
                    episodes=int(sweep_episodes),
                    seed=int(st_seed),
                    label="random_masked",
                    deterministic=False,
                    policy_mode="random_masked",
                )

                stoch_delta_rewards.append(float(trained_stoch["reward_mean"] - noop_stoch["reward_mean"]))
                stoch_delta_energies.append(float(trained_stoch["energy_mean"] - noop_stoch["energy_mean"]))
                stoch_delta_droppeds.append(float(trained_stoch["dropped_mean"] - noop_stoch["dropped_mean"]))
                stoch_toggles_applied_vals.append(float(trained_stoch.get("toggles_applied_mean", 0.0)))
                stoch_off_edges_vals.append(float(trained_stoch.get("mean_off_edges_count", 0.0)))
                stoch_noop_energy_vals.append(float(noop_stoch["energy_mean"]))
                stoch_noop_dropped_vals.append(float(noop_stoch["dropped_mean"]))
                stoch_trained_energy_vals.append(float(trained_stoch["energy_mean"]))
                stoch_trained_dropped_vals.append(float(trained_stoch["dropped_mean"]))
                stoch_random_energy_vals.append(float(random_masked["energy_mean"]))
                stoch_random_dropped_vals.append(float(random_masked["dropped_mean"]))

            stoch_delta_reward = float(np.mean(stoch_delta_rewards)) if stoch_delta_rewards else 0.0
            stoch_delta_energy = float(np.mean(stoch_delta_energies)) if stoch_delta_energies else 0.0
            stoch_delta_dropped = float(np.mean(stoch_delta_droppeds)) if stoch_delta_droppeds else 0.0
            stoch_pass, stoch_reason = custom_gate(stoch_delta_reward, stoch_delta_energy, stoch_delta_dropped)
            trained_toggles_applied_mean = float(np.mean(stoch_toggles_applied_vals)) if stoch_toggles_applied_vals else 0.0
            trained_off_edges_mean = float(np.mean(stoch_off_edges_vals)) if stoch_off_edges_vals else 0.0
            noop_energy_mean = float(np.mean(stoch_noop_energy_vals)) if stoch_noop_energy_vals else float(noop_det["energy_mean"])
            noop_dropped_mean = float(np.mean(stoch_noop_dropped_vals)) if stoch_noop_dropped_vals else float(noop_det["dropped_mean"])
            trained_stoch_energy_mean = float(np.mean(stoch_trained_energy_vals)) if stoch_trained_energy_vals else float(trained_det["energy_mean"])
            trained_stoch_dropped_mean = float(np.mean(stoch_trained_dropped_vals)) if stoch_trained_dropped_vals else float(trained_det["dropped_mean"])
            random_stoch_energy_mean = float(np.mean(stoch_random_energy_vals)) if stoch_random_energy_vals else 0.0
            random_stoch_dropped_mean = float(np.mean(stoch_random_dropped_vals)) if stoch_random_dropped_vals else 0.0

            trained_vs_random_same = bool(
                abs(trained_stoch_energy_mean - random_stoch_energy_mean) < 1e-6
                and abs(trained_stoch_dropped_mean - random_stoch_dropped_mean) < 1e-6
            )

            score = 0.0
            if not det_pass:
                score += 1e6
            if not stoch_pass:
                score += 1e6
            score += 1000.0 * float(stoch_delta_energy)
            score += 10.0 * max(0.0, float(stoch_delta_dropped))
            score += 1.0 * max(0.0, -float(stoch_delta_reward))

            row: Dict[str, Any] = {
                "tag": tag,
                "energy_weight": float(ew),
                "toggle_attempt_penalty": float(tap),
                "toggle_off_penalty_scale": float(off_s),
                "toggle_on_penalty_scale": float(on_s),
                "max_off_toggles_per_episode": int(mo),
                "max_total_toggles_per_episode": int(mt),
                "off_start_guard_decision_steps": int(g),
                "toggle_cooldown_steps": int(cd),
                "global_toggle_cooldown_steps": int(gcd),
                "util_block_threshold": float(ut),
                "trained_reward_mean": float(trained_det["reward_mean"]),
                "trained_energy_mean": float(trained_det["energy_mean"]),
                "trained_dropped_mean": float(trained_det["dropped_mean"]),
                "trained_norm_drop_mean": float(trained_det["norm_drop_mean"]),
                "trained_toggles_applied_mean": trained_toggles_applied_mean,
                "trained_off_edges_mean": trained_off_edges_mean,
                "noop_energy_mean": noop_energy_mean,
                "noop_dropped_mean": noop_dropped_mean,
                "delta_energy": stoch_delta_energy,
                "delta_dropped": stoch_delta_dropped,
                "trained_vs_random_same": trained_vs_random_same,
                "det_pass": det_pass,
                "det_fail_reason": det_reason,
                "det_delta_reward": det_delta_reward,
                "det_delta_energy": det_delta_energy,
                "det_delta_dropped": det_delta_dropped,
                "stoch_pass": stoch_pass,
                "stoch_fail_reason": stoch_reason,
                "stoch_delta_reward": stoch_delta_reward,
                "stoch_delta_energy": stoch_delta_energy,
                "stoch_delta_dropped": stoch_delta_dropped,
                "score": score,
            }
            writer.writerow(row)
            rows.append(row)

            print(
                f"[sweep] done {tag}: "
                f"det={'PASS' if det_pass else 'FAIL'} stoch={'PASS' if stoch_pass else 'FAIL'} "
                f"delta_energy(stoch)={stoch_delta_energy:+.6f} "
                f"delta_dropped(stoch)={stoch_delta_dropped:+.3f} "
                f"same_as_random={trained_vs_random_same}"
            )
            print(
                f"[sweep-candidate] MT={int(mt)} MO={int(mo)} CD={int(cd)} "
                f"GCD={int(gcd)} G={int(g)} UT={float(ut):.2f} "
                f"applied={trained_toggles_applied_mean:.2f} "
                f"off_edges={trained_off_edges_mean:.3f}"
            )

    print(f"\n[sweep] wrote CSV: {out_csv}")

    pass_both = [r for r in rows if bool(r.get("det_pass")) and bool(r.get("stoch_pass"))]
    pass_stoch = [r for r in rows if bool(r.get("stoch_pass"))]
    ranked_source = pass_both if pass_both else (pass_stoch if pass_stoch else rows)
    ranked = sorted(ranked_source, key=lambda r: float(r.get("score", 0.0)))

    print("\nTop 10 candidates:")
    for idx, row in enumerate(ranked[:10], start=1):
        print(
            f"{idx:02d}. {row['tag']} "
            f"score={float(row.get('score', 0.0)):+.3f} "
            f"det={'PASS' if bool(row.get('det_pass')) else 'FAIL'} "
            f"stoch={'PASS' if bool(row.get('stoch_pass')) else 'FAIL'} "
            f"ΔE(stoch)={float(row.get('stoch_delta_energy', 0.0)):+.6f} "
            f"ΔD(stoch)={float(row.get('stoch_delta_dropped', 0.0)):+.3f} "
            f"ΔR(stoch)={float(row.get('stoch_delta_reward', 0.0)):+.3f}"
        )
