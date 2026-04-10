from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from greennet.env import EnvConfig
from greennet.rl.eval import eval_policy

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

def run_robustness_eval(
    model: Any,
    base_env_config: EnvConfig,
    episodes: int,
    seed: int,
    topology_seeds: list[int],
    out_csv: Path,
    out_png: Path,
) -> None:
    """Evaluate NOOP vs trained(det) across multiple topology seeds.

    IMPORTANT: We keep the action space fixed (from the trained model's base topology) by setting
    topology_randomize=True and evaluating each requested topology via topology_seeds=(s,).
    """
    rows: list[dict[str, Any]] = []

    for s in topology_seeds:
        env_cfg = replace(
            base_env_config,
            topology_randomize=True,
            topology_seeds=(int(s),),
        )

        noop = eval_policy(
            None,
            env_cfg,
            episodes=int(episodes),
            seed=seed,
            label=f"always_noop_seed{s}",
            deterministic=True,
        )
        det = eval_policy(
            model,
            env_cfg,
            episodes=int(episodes),
            seed=seed,
            label=f"trained_seed{s}",
            deterministic=True,
        )

        for policy_name, stats in [("noop", noop), ("trained_det", det)]:
            rows.append({
                "topology_seed": int(s),
                "policy": policy_name,
                **stats,
            })

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = sorted(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in header) + "\n")

    print(f"[robustness] wrote CSV: {out_csv}")

    # Plot: energy vs norm_drop per topology seed
    if plt is None:
        print("[robustness] matplotlib not available; skipping plot.")
        return

    # Build series
    def _series(policy: str):
        xs, ys, labels = [], [], []
        for r in rows:
            if r.get("policy") != policy:
                continue
            xs.append(float(r.get("norm_drop_mean", 0.0)))
            ys.append(float(r.get("energy_mean", 0.0)))
            labels.append(str(r.get("topology_seed")))
        return xs, ys, labels

    x_noop, y_noop, lab_noop = _series("noop")
    x_det, y_det, lab_det = _series("trained_det")

    plt.figure()
    plt.scatter(x_noop, y_noop, label="NOOP (all-on)")
    plt.scatter(x_det, y_det, label="Trained (det)")

    for x, y, t in zip(x_noop, y_noop, lab_noop):
        plt.annotate(t, (x, y))
    for x, y, t in zip(x_det, y_det, lab_det):
        plt.annotate(t, (x, y))

    plt.xlabel("Normalized drop (mean)")
    plt.ylabel("Energy (kWh per episode, mean)")
    plt.title("Robustness across topology seeds")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    print(f"[robustness] wrote plot: {out_png}")

