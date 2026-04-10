#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


# Canonical metrics we want in output and accepted aliases from historical summaries.
CANONICAL_METRICS: dict[str, list[str]] = {
    "energy_mean": ["energy_kwh_total_mean", "energy_mean", "energy_kwh_mean", "energy"],
    "dropped_mean": ["dropped_total_mean", "dropped_mean", "dropped"],
    "reward_mean": ["reward_total_mean", "reward_mean", "episode_reward"],
    "qos_violation_rate_mean": ["qos_violation_rate_mean", "qos_violation_mean", "qos_violation_rate", "qos_violation"],
    "toggles_total_mean": ["toggles_total_mean", "toggles_mean", "toggles_total", "toggles"],
}


def _safe_mean(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").mean())


def _resolve_metric_columns(df: pd.DataFrame, summary_path: Path) -> tuple[list[str], dict[str, str]]:
    metric_cols: list[str] = []
    metric_alias: dict[str, str] = {}

    for canonical, candidates in CANONICAL_METRICS.items():
        for candidate in candidates:
            if candidate in df.columns:
                metric_cols.append(candidate)
                metric_alias[canonical] = candidate
                break

    if metric_cols:
        return metric_cols, metric_alias

    expected = sorted({c for values in CANONICAL_METRICS.values() for c in values})
    available = [str(c) for c in df.columns]
    raise ValueError(
        "No expected metric columns found in summary file.\n"
        f"  file: {summary_path}\n"
        f"  expected any of: {expected}\n"
        f"  available: {available}"
    )


def build_leaderboard(df: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    required_keys = {"policy", "scenario"}
    if not required_keys.issubset(set(df.columns)):
        missing = sorted(required_keys.difference(set(df.columns)))
        raise ValueError(
            "Summary file is missing required columns for leaderboard build.\n"
            f"  file: {summary_path}\n"
            f"  missing: {missing}\n"
            f"  available: {[str(c) for c in df.columns]}"
        )

    metric_cols, metric_alias = _resolve_metric_columns(df, summary_path)

    keep_cols = ["scenario", "policy"] + metric_cols
    df = df[keep_cols].copy()

    # Coerce metric columns to numeric where possible
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate: mean across seeds/runs for each (scenario, policy)
    agg = (
        df.groupby(["scenario", "policy"], as_index=False)
        .agg({c: _safe_mean for c in metric_cols})
        .sort_values(["scenario", "policy"])
        .reset_index(drop=True)
    )

    # Build lookup tables for deltas
    def add_deltas(vs_policy: str, suffix: str) -> None:
        base = agg[agg["policy"] == vs_policy].set_index("scenario")
        for metric in metric_cols:
            # Prefer canonical names for stable output; fall back to the actual column name.
            canonical_name = None
            for k, v in metric_alias.items():
                if v == metric:
                    canonical_name = k
                    break
            base_name = (canonical_name or metric)
            col = base_name.replace("_mean", "")
            out = f"delta_{col}_vs_{suffix}"
            def _delta(row: pd.Series) -> float:
                sc = row["scenario"]
                if sc not in base.index:
                    return float("nan")
                return float(row[metric]) - float(base.loc[sc, metric])

            agg[out] = agg.apply(_delta, axis=1)

    add_deltas("heuristic", "heuristic")
    add_deltas("all_on", "all_on")

    # Column order to match your v1 file style as closely as possible
    ordered = ["scenario", "policy"] + [c for c in metric_cols if c in agg.columns]
    # deltas
    delta_cols = []
    for base in ("heuristic", "all_on"):
        for metric in metric_cols:
            canonical_name = None
            for k, v in metric_alias.items():
                if v == metric:
                    canonical_name = k
                    break
            base_name = (canonical_name or metric)
            name = base_name.replace("_mean", "")
            col = f"delta_{name}_vs_{base}"
            if col in agg.columns:
                delta_cols.append(col)
    ordered += delta_cols

    return agg[ordered]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build leaderboard CSV from results_summary CSV.")
    ap.add_argument("--summary", required=True, help="Path to results_summary_matrix_*.csv (90 rows).")
    ap.add_argument("--output", required=True, help="Path to output leaderboard CSV (9 rows).")
    ap.add_argument(
        "--output-source",
        default=None,
        help="Optional: path to write the source rows used (copy of summary, filtered to needed cols).",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_path = Path(args.output)
    out_src = Path(args.output_source) if args.output_source else None

    df = pd.read_csv(summary_path)

    # Write source if requested (keep as evidence)
    if out_src is not None:
        out_src.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_src, index=False)

    lb = build_leaderboard(df, summary_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out_path, index=False)

    print(f"[leaderboard] wrote: {out_path} (rows={len(lb)})")
    if out_src is not None:
        print(f"[leaderboard] wrote source: {out_src} (rows={len(df)})")


if __name__ == "__main__":
    main()
