from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard.data_io import RESULTS_DIR, load_run
from dashboard.models import RunData
from dashboard.plotting import plot_series
from dashboard.stats import behavior_stats, filter_episode, overall, safe_float


def _recompute_overall_from_per_step(df: pd.DataFrame) -> dict:
    metrics = {
        "reward_total_mean": float("nan"),
        "delivered_total_mean": float("nan"),
        "dropped_total_mean": float("nan"),
        "energy_kwh_total_mean": float("nan"),
        "carbon_g_total_mean": float("nan"),
        "avg_utilization_mean": float("nan"),
        "active_ratio_mean": float("nan"),
        "avg_delay_ms_mean": float("nan"),
        "steps_mean": float("nan"),
    }
    if df is None or df.empty:
        return metrics

    df_sorted = df.copy()
    sort_cols: List[str] = []
    if "episode" in df_sorted.columns:
        sort_cols.append("episode")
    if "step" in df_sorted.columns:
        sort_cols.append("step")
    if sort_cols:
        try:
            df_sorted = df_sorted.sort_values(by=sort_cols)
        except Exception:
            pass

    has_episode = "episode" in df_sorted.columns and not df_sorted["episode"].dropna().empty

    if has_episode:
        group = df_sorted.groupby("episode", dropna=True)

        def _group_sum(col: str) -> pd.Series:
            if col not in df_sorted.columns:
                return pd.Series(dtype="float64")
            try:
                return group[col].sum()
            except Exception:
                return pd.Series(dtype="float64")

        def _group_max(col: str) -> pd.Series:
            if col not in df_sorted.columns:
                return pd.Series(dtype="float64")
            try:
                return group[col].max()
            except Exception:
                return pd.Series(dtype="float64")

        def _group_mean(col: str) -> pd.Series:
            if col not in df_sorted.columns:
                return pd.Series(dtype="float64")
            try:
                return group[col].mean()
            except Exception:
                return pd.Series(dtype="float64")

        reward_total_ep = _group_sum("reward")
        energy_total_ep = (
            _group_sum("delta_energy_kwh")
            if "delta_energy_kwh" in df_sorted.columns
            else _group_max("energy_kwh")
        )
        dropped_total_ep = (
            _group_sum("delta_dropped")
            if "delta_dropped" in df_sorted.columns
            else _group_max("dropped")
        )
        delivered_total_ep = (
            _group_sum("delta_delivered")
            if "delta_delivered" in df_sorted.columns
            else _group_max("delivered")
        )
        carbon_total_ep = (
            _group_sum("delta_carbon_g")
            if "delta_carbon_g" in df_sorted.columns
            else _group_max("carbon_g")
        )
        avg_util_ep = _group_mean("avg_utilization")
        active_ratio_ep = _group_mean("active_ratio")
        delay_ep = _group_mean("avg_delay_ms")

        steps_mean = (
            group["step"].max().mean() if "step" in df_sorted.columns else float(len(df_sorted))
        )

        metrics.update(
            {
                "reward_total_mean": float(reward_total_ep.mean())
                if not reward_total_ep.empty
                else float("nan"),
                "energy_kwh_total_mean": float(energy_total_ep.mean())
                if not energy_total_ep.empty
                else float("nan"),
                "dropped_total_mean": float(dropped_total_ep.mean())
                if not dropped_total_ep.empty
                else float("nan"),
                "delivered_total_mean": float(delivered_total_ep.mean())
                if not delivered_total_ep.empty
                else float("nan"),
                "carbon_g_total_mean": float(carbon_total_ep.mean())
                if not carbon_total_ep.empty
                else float("nan"),
                "avg_utilization_mean": float(avg_util_ep.mean())
                if not avg_util_ep.empty
                else float("nan"),
                "active_ratio_mean": float(active_ratio_ep.mean())
                if not active_ratio_ep.empty
                else float("nan"),
                "avg_delay_ms_mean": float(delay_ep.mean())
                if not delay_ep.empty
                else float("nan"),
                "steps_mean": float(steps_mean),
            }
        )
        return metrics

    def _col_sum(col: str) -> float:
        if col not in df_sorted.columns:
            return float("nan")
        try:
            return float(pd.to_numeric(df_sorted[col], errors="coerce").sum())
        except Exception:
            return float("nan")

    def _col_max(col: str) -> float:
        if col not in df_sorted.columns:
            return float("nan")
        try:
            return float(pd.to_numeric(df_sorted[col], errors="coerce").max())
        except Exception:
            return float("nan")

    def _col_mean(col: str) -> float:
        if col not in df_sorted.columns:
            return float("nan")
        try:
            return float(pd.to_numeric(df_sorted[col], errors="coerce").mean())
        except Exception:
            return float("nan")

    metrics.update(
        {
            "reward_total_mean": _col_sum("reward"),
            "energy_kwh_total_mean": _col_sum("delta_energy_kwh")
            if "delta_energy_kwh" in df_sorted.columns
            else _col_max("energy_kwh"),
            "dropped_total_mean": _col_sum("delta_dropped")
            if "delta_dropped" in df_sorted.columns
            else _col_max("dropped"),
            "delivered_total_mean": _col_sum("delta_delivered")
            if "delta_delivered" in df_sorted.columns
            else _col_max("delivered"),
            "carbon_g_total_mean": _col_sum("delta_carbon_g")
            if "delta_carbon_g" in df_sorted.columns
            else _col_max("carbon_g"),
            "avg_utilization_mean": _col_mean("avg_utilization"),
            "active_ratio_mean": _col_mean("active_ratio"),
            "avg_delay_ms_mean": _col_mean("avg_delay_ms"),
            "steps_mean": _col_max("step") if "step" in df_sorted.columns else float(len(df_sorted)),
        }
    )
    return metrics


def _metric_cards_from_summary(run: RunData) -> None:
    meta = run.meta or {}
    ov = overall(run.summary)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Policy", str(meta.get("policy", "")))
    c2.metric("Scenario", str(meta.get("scenario", "")))
    c3.metric("Seed", str(meta.get("seed", "")))
    c4.metric("Episodes", str(meta.get("episodes", "")))
    c5.metric("Steps (mean)", str(ov.get("steps_mean", "")))

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Reward (mean)", str(ov.get("reward_total_mean", "")))
    c7.metric("Dropped (mean)", str(ov.get("dropped_total_mean", "")))
    c8.metric("Energy kWh (mean)", str(ov.get("energy_kwh_total_mean", "")))
    c9.metric("Avg util (mean)", str(ov.get("avg_utilization_mean", "")))
    c10.metric("Delay ms (mean)", str(ov.get("avg_delay_ms_mean", "")))


def render_single_run(runs: List[Path]) -> None:
    st.subheader("Single run")

    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_right:
        if st.button("Refresh run list", key="refresh_single"):
            st.cache_data.clear()
            st.rerun()

    run_names = [r.name for r in runs]
    if not run_names:
        st.info(
            "No runs found in ./results yet. Create one with: "
            "python3 run_experiment.py --policy noop --scenario normal --seed 0 --episodes 1 --steps 300"
        )
        return

    # Live demo mode can set this to auto-open the newest run.
    requested = st.session_state.get("autoselect_run")
    default_index = 0
    if isinstance(requested, str) and requested in run_names:
        default_index = run_names.index(requested)

    with ctrl_left:
        selected = st.selectbox(
            "Select a run folder",
            run_names,
            index=default_index,
            key="single_run",
        )

    # Clear after applying so it doesn't keep hijacking user selections.
    if requested == selected:
        st.session_state.pop("autoselect_run", None)

    run_dir = RESULTS_DIR / selected
    run = load_run(run_dir)
    if run is None:
        st.error(f"Failed to load run: {run_dir}")
        return

    st.markdown(f"### Selected run: `{run_dir.name}`")
    st.markdown(f"`{run_dir}`")

    _metric_cards_from_summary(run)

    df = run.per_step

    # Episode selection (if present)
    episode_value: Optional[int] = None
    if "episode" in df.columns:
        eps = sorted([int(x) for x in df["episode"].dropna().unique().tolist()])
        if eps:
            episode_value = st.selectbox("Episode", ["(all)"] + eps, index=0)
            if episode_value == "(all)":
                episode_value = None

    df_view = filter_episode(df, episode_value)

    # Choose plotting x-axis + cumulative columns:
    # - If a single episode is selected, use per-episode cumulative (resets each episode) with x=step.
    # - If viewing all episodes, use run-level cumulative with x=global_step (monotonic) when available.
    xcol = "step"
    cum_energy_col = "cum_energy_kwh"
    cum_dropped_col = "cum_dropped"
    if episode_value is not None:
        if "ep_cum_energy_kwh" in df_view.columns:
            cum_energy_col = "ep_cum_energy_kwh"
        if "ep_cum_dropped" in df_view.columns:
            cum_dropped_col = "ep_cum_dropped"
    else:
        if "global_step" in df_view.columns:
            xcol = "global_step"
        if "run_cum_energy_kwh" in df_view.columns:
            cum_energy_col = "run_cum_energy_kwh"
        if "run_cum_dropped" in df_view.columns:
            cum_dropped_col = "run_cum_dropped"

    st.divider()

    st.subheader("Time-series plots")
    left, right = st.columns(2)

    with left:
        if cum_energy_col in df_view.columns:
            plot_series(df_view, xcol, cum_energy_col, "Cumulative energy (kWh)")
        if "delta_energy_kwh" in df_view.columns:
            plot_series(df_view, xcol, "delta_energy_kwh", "Delta energy (kWh) per step")

        if "active_ratio" in df_view.columns:
            plot_series(df_view, xcol, "active_ratio", "Active link ratio")

    with right:
        if cum_dropped_col in df_view.columns:
            plot_series(df_view, xcol, cum_dropped_col, "Cumulative dropped")
        if "delta_dropped" in df_view.columns:
            plot_series(df_view, xcol, "delta_dropped", "Delta dropped per step")

        if "avg_delay_ms" in df_view.columns:
            plot_series(df_view, xcol, "avg_delay_ms", "Average delay (ms)")

    st.divider()

    st.subheader("Behavior stats")
    stats = behavior_stats(df_view)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Rows", str(stats.get("rows", 0)))
    s2.metric("No-op rate", f"{safe_float(stats.get('noop_rate')):.3f}")
    s3.metric("Toggle rate", f"{safe_float(stats.get('toggle_applied_rate')):.3f}")
    s4.metric("Invalid rate", f"{safe_float(stats.get('invalid_rate')):.3f}")
    s5.metric("Unique actions", str(stats.get("unique_actions", 0)))

    top_actions = stats.get("top_actions") or {}
    if top_actions:
        st.caption("Top actions")
        st.dataframe(
            pd.DataFrame({"action": list(top_actions.keys()), "count": list(top_actions.values())}),
            use_container_width=True,
        )

    with st.expander("Verification (sanity check)"):
        ov_json = overall(run.summary)
        ov_csv = _recompute_overall_from_per_step(run.per_step)

        metrics = [
            "reward_total_mean",
            "delivered_total_mean",
            "dropped_total_mean",
            "energy_kwh_total_mean",
            "carbon_g_total_mean",
            "avg_utilization_mean",
            "active_ratio_mean",
            "avg_delay_ms_mean",
            "steps_mean",
        ]
        tolerances = {
            "reward_total_mean": 1e-6,
            "delivered_total_mean": 1e-6,
            "dropped_total_mean": 1e-6,
            "energy_kwh_total_mean": 1e-6,
            "carbon_g_total_mean": 1e-6,
            "avg_utilization_mean": 1e-6,
            "active_ratio_mean": 1e-6,
            "avg_delay_ms_mean": 1e-6,
            "steps_mean": 0.0,
        }

        def _to_float(val: Any) -> float:
            try:
                return float(val)
            except Exception:
                return float("nan")

        rows: List[Dict[str, Any]] = []
        violations: List[str] = []
        for key in metrics:
            json_val = _to_float(ov_json.get(key, float("nan")))
            csv_val = _to_float(ov_csv.get(key, float("nan")))
            delta = (
                csv_val - json_val
                if math.isfinite(json_val) and math.isfinite(csv_val)
                else float("nan")
            )

            rows.append(
                {
                    "metric": key,
                    "summary_json": json_val,
                    "recomputed_from_csv": csv_val,
                    "delta": delta,
                }
            )

            tol = tolerances.get(key, 0.0)
            if math.isfinite(json_val) and math.isfinite(csv_val):
                if key == "steps_mean":
                    if delta != 0:
                        violations.append(key)
                else:
                    if abs(delta) > tol:
                        violations.append(key)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if violations:
            st.warning(f"Verification mismatch beyond tolerance for: {', '.join(violations)}")
        else:
            st.success("Verification passed: all metrics within tolerance.")

    with st.expander("Preview per_step.csv (first 50 rows)"):
        st.dataframe(df_view.head(50), use_container_width=True)

    st.divider()

    st.subheader("Comparison table (results_summary.csv)")
    summary_csv = RESULTS_DIR / "results_summary.csv"
    if summary_csv.exists():
        sumdf = pd.read_csv(summary_csv)

        f1, f2, f3 = st.columns(3)
        with f1:
            scenario_filter = st.selectbox(
                "Scenario",
                ["(all)"] + sorted(sumdf["scenario"].dropna().unique().tolist()),
                key="sum_scenario_single",
            )
        with f2:
            policy_filter = st.selectbox(
                "Policy",
                ["(all)"] + sorted(sumdf["policy"].dropna().unique().tolist()),
                key="sum_policy_single",
            )
        with f3:
            seed_values = sorted([str(s) for s in sumdf["seed"].dropna().unique().tolist()])
            seed_filter = st.selectbox("Seed", ["(all)"] + seed_values, key="sum_seed_single")

        view = sumdf.copy()
        if scenario_filter != "(all)":
            view = view[view["scenario"] == scenario_filter]
        if policy_filter != "(all)":
            view = view[view["policy"] == policy_filter]
        if seed_filter != "(all)":
            view = view[view["seed"].astype(str) == seed_filter]

        cols_first = [
            "policy",
            "scenario",
            "seed",
            "status",
            "reward_total_mean",
            "dropped_total_mean",
            "energy_kwh_total_mean",
            "avg_delay_ms_mean",
            "results_dir",
        ]
        cols = [c for c in cols_first if c in view.columns] + [c for c in view.columns if c not in cols_first]
        view = view[cols].sort_values(by=[c for c in ["scenario", "seed", "policy"] if c in view.columns])

        st.dataframe(view, use_container_width=True)
    else:
        st.info(
            "No results_summary.csv found yet. Generate it with: "
            "python3 experiments/run_matrix.py --episodes 1 --steps 300"
        )
