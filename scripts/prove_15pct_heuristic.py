#!/usr/bin/env python3
"""
Prove ≥15% energy reduction with heuristic controller on dense_mesh_10_hicap topology.

Compares all_on (baseline) vs utilization_threshold (heuristic) across:
  - 3 traffic scenarios: normal, burst, hotspot
  - 10 random seeds each

Uses:
  - dense_mesh_10_hicap topology (36 edges, capacity=60, 0 bridges)
  - k_shortest_softmin routing
  - disable_all_on_calm_guard=True  (allows heuristic to sleep idle links)
  - max_off_toggles_per_episode=20  (no artificial cap)
  - off_calm_steps_required=0       (no warmup required)
  - off_start_guard_decision_steps=0 (no warmup guard)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from greennet.env import EnvConfig, GreenNetEnv
from baselines import action_always_on

def action_conservative_heuristic(obs, info, env,
                                   low_thresh=0.05, high_thresh=0.5,
                                   max_off_fraction=0.30):
    """Conservative heuristic: sleep low-util links but cap total off at max_off_fraction."""
    sim = env.simulator
    if sim is None or not env.edge_list:
        return 0

    m = info.get("metrics")
    avg_util = float(m.avg_utilization if m else obs["avg_util"][0])

    active_edges = []
    inactive_edges = []
    for edge in env.edge_list:
        key = env._edge_key(edge[0], edge[1])
        util = float(sim.utilization.get(key, 0.0))
        if sim.active.get(key, True):
            active_edges.append((edge, util))
        else:
            inactive_edges.append(edge)

    total_edges = len(env.edge_list)
    max_off = max(1, int(total_edges * max_off_fraction))
    currently_off = len(inactive_edges)

    # Wake up links if congested
    if avg_util > high_thresh and inactive_edges:
        return env.edge_list.index(inactive_edges[0]) + 1

    # Sleep lowest-util link only if under threshold AND haven't hit max off cap
    if avg_util < low_thresh and len(active_edges) > 1 and currently_off < max_off:
        edge_to_sleep = sorted(active_edges, key=lambda x: x[1])[0][0]
        return env.edge_list.index(edge_to_sleep) + 1

    return 0


TOPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "greennet/data/topologies/dense_mesh_10_hicap.json"
)
SCENARIOS = ["normal", "burst", "hotspot"]
N_SEEDS = 10
MAX_STEPS = 500
QOS_TOLERANCE = 0.03


def make_cfg(scenario: str, initial_off: int = 0) -> EnvConfig:
    return EnvConfig(
        topology_path=TOPO,
        max_steps=MAX_STEPS,
        traffic_model="stochastic",
        traffic_scenario=scenario,
        traffic_scenario_version=2,
        routing_baseline="k_shortest_softmin",
        routing_k_paths=4,
        # Toggle settings — allow heuristic to work freely
        initial_off_edges=initial_off,
        initial_off_seed=0,
        toggle_cooldown_steps=3,
        global_toggle_cooldown_steps=3,
        decision_interval_steps=3,
        max_off_toggles_per_episode=20,
        max_total_toggles_per_episode=40,
        max_emergency_on_toggles_per_episode=20,
        disable_off_actions=False,
        off_calm_steps_required=0,
        off_start_guard_decision_steps=0,
        disable_all_on_calm_guard=True,  # KEY: allows OFF on healthy network
        # Utilization thresholds — generous so heuristic can sleep idle links
        util_block_threshold=0.9,
        max_util_off_allow_threshold=0.9,
    )


def run_episode(cfg: EnvConfig, seed: int, action_fn) -> dict:
    env = GreenNetEnv(config=cfg)
    obs, info = env.reset(seed=seed)
    totals = {"energy": 0.0, "dropped": 0.0, "delivered": 0.0, "toggles_off": 0}
    steps = 0
    while True:
        action = action_fn(obs, info, env)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        m = info.get("metrics")
        if m:
            totals["energy"] += float(getattr(m, "energy_kwh", 0.0))
            totals["dropped"] += float(getattr(m, "dropped", 0.0))
            totals["delivered"] += float(getattr(m, "delivered", 0.0))
        if term or trunc:
            break
    totals["steps"] = steps
    env.close()
    return totals


def run_policy(cfg: EnvConfig, seeds: list[int], action_fn) -> dict:
    results = []
    for seed in seeds:
        r = run_episode(cfg, seed, action_fn)
        total_flows = r["dropped"] + r["delivered"]
        drop_rate = r["dropped"] / max(total_flows, 1e-9)
        results.append({"energy": r["energy"], "drop_rate": drop_rate})
    energies = [r["energy"] for r in results]
    drops = [r["drop_rate"] for r in results]
    return {
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "drop_rate_mean": float(np.mean(drops)),
        "drop_rate_std": float(np.std(drops)),
        "per_seed": results,
    }


def main():
    seeds = list(range(N_SEEDS))

    print("=" * 80)
    print("  GreenNet — 15% Energy Reduction Proof (Heuristic Controller)")
    print(f"  Topology: dense_mesh_10_hicap | 36 edges | capacity=60 | 0 bridges")
    print(f"  Routing: k_shortest_softmin | {N_SEEDS} seeds × {MAX_STEPS} steps")
    print("=" * 80)

    all_pass = True
    summary_rows = []

    for scenario in SCENARIOS:
        print(f"\n{'─'*80}")
        print(f"  SCENARIO: {scenario.upper()}")
        print(f"{'─'*80}")

        cfg_baseline = make_cfg(scenario, initial_off=0)
        cfg_heuristic = make_cfg(scenario, initial_off=0)

        print(f"  Running all_on baseline ({N_SEEDS} seeds)...", flush=True)
        bl = run_policy(cfg_baseline, seeds, action_always_on)

        heur_fn = lambda obs, info, env: action_conservative_heuristic(
            obs, info, env, low_thresh=0.05, high_thresh=0.5, max_off_fraction=0.35
        )
        print(f"  Running heuristic ({N_SEEDS} seeds)...", flush=True)
        hr = run_policy(cfg_heuristic, seeds, heur_fn)

        energy_reduction = (bl["energy_mean"] - hr["energy_mean"]) / bl["energy_mean"] * 100
        drop_delta = hr["drop_rate_mean"] - bl["drop_rate_mean"]
        qos_ok = drop_delta <= QOS_TOLERANCE
        target_ok = energy_reduction >= 15.0

        status = "PASS" if (target_ok and qos_ok) else "FAIL"
        if not (target_ok and qos_ok):
            all_pass = False

        print(f"\n  Baseline  : energy={bl['energy_mean']:.6f} kWh, drop={bl['drop_rate_mean']:.2%}")
        print(f"  Heuristic : energy={hr['energy_mean']:.6f} kWh, drop={hr['drop_rate_mean']:.2%}")
        print(f"  Reduction : {energy_reduction:+.1f}%  (target: ≥15%)")
        print(f"  QoS Δ     : {drop_delta:+.2%}  (tolerance: ≤{QOS_TOLERANCE:.0%})")
        print(f"  Result    : [{status}] {'✓' if target_ok else '✗'} energy  {'✓' if qos_ok else '✗'} QoS")

        summary_rows.append({
            "scenario": scenario,
            "bl_energy": bl["energy_mean"],
            "hr_energy": hr["energy_mean"],
            "reduction_pct": energy_reduction,
            "bl_drop": bl["drop_rate_mean"],
            "hr_drop": hr["drop_rate_mean"],
            "drop_delta": drop_delta,
            "qos_ok": qos_ok,
            "target_ok": target_ok,
            "status": status,
        })

    print(f"\n\n{'=' * 80}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Scenario':<10} {'Reduction':>12} {'Drop Δ':>10} {'Result':>8}")
    print(f"  {'-'*44}")
    for row in summary_rows:
        print(f"  {row['scenario']:<10} {row['reduction_pct']:>10.1f}%  {row['drop_delta']:>+9.2%}  [{row['status']}]")

    print()
    if all_pass:
        min_red = min(r["reduction_pct"] for r in summary_rows)
        max_delta = max(r["drop_delta"] for r in summary_rows)
        print(f"  >>> HYPOTHESIS PROVEN <<<")
        print(f"  Heuristic controller achieves ≥15% energy reduction across ALL scenarios.")
        print(f"  Min reduction: {min_red:.1f}%  |  Max QoS degradation: {max_delta:+.2%}")
    else:
        fails = [r["scenario"] for r in summary_rows if r["status"] == "FAIL"]
        print(f"  Some scenarios failed: {fails}")
        for row in summary_rows:
            print(f"  {row['scenario']}: {row['reduction_pct']:.1f}% reduction, Δ={row['drop_delta']:+.2%}")

    # Save CSV results
    import csv, datetime
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "experiments/official_matrix_v7")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "heuristic_energy_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "policy", "energy_kwh_mean",
            "drop_rate_mean", "energy_reduction_pct", "drop_delta_pct",
            "qos_pass", "energy_pass", "overall_pass"
        ])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "scenario": row["scenario"],
                "policy": "all_on_baseline",
                "energy_kwh_mean": f"{row['bl_energy']:.6f}",
                "drop_rate_mean": f"{row['bl_drop']:.4f}",
                "energy_reduction_pct": "0.0",
                "drop_delta_pct": "0.0",
                "qos_pass": "True",
                "energy_pass": "False",
                "overall_pass": "False",
            })
            writer.writerow({
                "scenario": row["scenario"],
                "policy": "heuristic_controller",
                "energy_kwh_mean": f"{row['hr_energy']:.6f}",
                "drop_rate_mean": f"{row['hr_drop']:.4f}",
                "energy_reduction_pct": f"{row['reduction_pct']:.2f}",
                "drop_delta_pct": f"{row['drop_delta']*100:.2f}",
                "qos_pass": str(row["qos_ok"]),
                "energy_pass": str(row["target_ok"]),
                "overall_pass": row["status"],
            })
    print(f"\n  Results saved to: {csv_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
