#!/usr/bin/env python3
"""
Create a thesis-friendly comparison CSV from a robustness_eval.csv file.

Input format (your file):
delivered_mean,...,energy_mean,...,norm_drop_mean,...,policy,...,toggles_applied_mean,...,topology_seed

Output:
experiments/robustness_comparison_<run_id>.csv

Usage:
  python experiments/make_robustness_table.py runs/20260123_125047/robustness_eval.csv --qos-target 0.072
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def run_id_from_path(p: Path) -> str:
    # If path is .../runs/<RUN_ID>/robustness_eval.csv, extract RUN_ID.
    parts = p.resolve().parts
    if "runs" in parts:
        i = parts.index("runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: file stem
    return p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robustness_csv", type=str, help="Path to robustness_eval.csv")
    ap.add_argument("--qos-target", type=float, default=0.072, help="QoS target for norm_drop (default: 0.072)")
    ap.add_argument("--trained-policy", type=str, default="trained_det", help="Policy name for trained model")
    ap.add_argument("--baseline-policy", type=str, default="noop", help="Policy name for baseline")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path (default: experiments/robustness_comparison_<run_id>.csv)",
    )
    args = ap.parse_args()

    in_path = Path(args.robustness_csv)
    rows = read_rows(in_path)

    # Index by (topology_seed, policy)
    by_seed_policy: Dict[Tuple[int, str], Dict[str, str]] = {}
    seeds = set()

    for r in rows:
        seed = int(to_float(r.get("topology_seed", "0"), 0.0))
        policy = (r.get("policy") or "").strip()
        seeds.add(seed)
        by_seed_policy[(seed, policy)] = r

    seeds = sorted(seeds)

    out_rows = []
    # Summary accumulators (across seeds)
    sum_noop_e = sum_tr_e = 0.0
    sum_noop_nd = sum_tr_nd = 0.0
    sum_noop_drop = sum_tr_drop = 0.0
    sum_tr_toggles = 0.0
    count = 0

    for seed in seeds:
        base = by_seed_policy.get((seed, args.baseline_policy))
        tr = by_seed_policy.get((seed, args.trained_policy))

        # Skip seeds missing either row
        if base is None or tr is None:
            continue

        noop_e = to_float(base.get("energy_mean", ""), 0.0)
        tr_e = to_float(tr.get("energy_mean", ""), 0.0)
        noop_nd = to_float(base.get("norm_drop_mean", ""), 0.0)
        tr_nd = to_float(tr.get("norm_drop_mean", ""), 0.0)
        noop_drop = to_float(base.get("dropped_mean", ""), 0.0)
        tr_drop = to_float(tr.get("dropped_mean", ""), 0.0)
        tr_toggles = to_float(tr.get("toggles_applied_mean", ""), 0.0)

        energy_saved_kwh = noop_e - tr_e
        energy_saved_pct = (energy_saved_kwh / noop_e * 100.0) if noop_e > 0 else 0.0
        delta_nd = tr_nd - noop_nd
        delta_drop = tr_drop - noop_drop

        qos_pass = (tr_nd <= args.qos_target)
        # Verdict logic (simple + thesis-friendly)
        if (abs(energy_saved_kwh) < 1e-12) and (abs(delta_nd) < 1e-12) and (abs(delta_drop) < 1e-9) and tr_toggles == 0:
            verdict = "Same as NOOP"
        elif qos_pass and energy_saved_kwh > 0 and delta_nd <= 0:
            verdict = "Better (energy↓, QoS ok)"
        elif qos_pass and energy_saved_kwh > 0:
            verdict = "Tradeoff (energy↓, drop↑)"
        elif qos_pass:
            verdict = "QoS ok, no energy gain"
        else:
            verdict = "Worse (QoS fail)"

        out_rows.append(
            {
                "topology_seed": seed,
                "energy_noop_kwh": f"{noop_e:.6f}",
                "energy_trained_kwh": f"{tr_e:.6f}",
                "energy_saved_kwh": f"{energy_saved_kwh:.6f}",
                "energy_saved_pct": f"{energy_saved_pct:.2f}",
                "norm_drop_noop": f"{noop_nd:.6f}",
                "norm_drop_trained": f"{tr_nd:.6f}",
                "delta_norm_drop": f"{delta_nd:.6f}",
                "dropped_noop": f"{noop_drop:.3f}",
                "dropped_trained": f"{tr_drop:.3f}",
                "delta_dropped": f"{delta_drop:.3f}",
                "toggles_trained_mean": f"{tr_toggles:.2f}",
                "qos_pass": str(bool(qos_pass)),
                "verdict": verdict,
            }
        )

        sum_noop_e += noop_e
        sum_tr_e += tr_e
        sum_noop_nd += noop_nd
        sum_tr_nd += tr_nd
        sum_noop_drop += noop_drop
        sum_tr_drop += tr_drop
        sum_tr_toggles += tr_toggles
        count += 1

    # Add overall mean row at bottom (nice for thesis)
    if count > 0:
        mean_noop_e = sum_noop_e / count
        mean_tr_e = sum_tr_e / count
        mean_noop_nd = sum_noop_nd / count
        mean_tr_nd = sum_tr_nd / count
        mean_noop_drop = sum_noop_drop / count
        mean_tr_drop = sum_tr_drop / count
        mean_tr_toggles = sum_tr_toggles / count

        energy_saved_kwh = mean_noop_e - mean_tr_e
        energy_saved_pct = (energy_saved_kwh / mean_noop_e * 100.0) if mean_noop_e > 0 else 0.0
        delta_nd = mean_tr_nd - mean_noop_nd
        delta_drop = mean_tr_drop - mean_noop_drop
        qos_pass = (mean_tr_nd <= args.qos_target)

        out_rows.append(
            {
                "topology_seed": "MEAN",
                "energy_noop_kwh": f"{mean_noop_e:.6f}",
                "energy_trained_kwh": f"{mean_tr_e:.6f}",
                "energy_saved_kwh": f"{energy_saved_kwh:.6f}",
                "energy_saved_pct": f"{energy_saved_pct:.2f}",
                "norm_drop_noop": f"{mean_noop_nd:.6f}",
                "norm_drop_trained": f"{mean_tr_nd:.6f}",
                "delta_norm_drop": f"{delta_nd:.6f}",
                "dropped_noop": f"{mean_noop_drop:.3f}",
                "dropped_trained": f"{mean_tr_drop:.3f}",
                "delta_dropped": f"{delta_drop:.3f}",
                "toggles_trained_mean": f"{mean_tr_toggles:.2f}",
                "qos_pass": str(bool(qos_pass)),
                "verdict": "Overall",
            }
        )

    run_id = run_id_from_path(in_path)
    default_out = Path("experiments") / f"robustness_comparison_{run_id}.csv"
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "topology_seed",
        "energy_noop_kwh",
        "energy_trained_kwh",
        "energy_saved_kwh",
        "energy_saved_pct",
        "norm_drop_noop",
        "norm_drop_trained",
        "delta_norm_drop",
        "dropped_noop",
        "dropped_trained",
        "delta_dropped",
        "toggles_trained_mean",
        "qos_pass",
        "verdict",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()