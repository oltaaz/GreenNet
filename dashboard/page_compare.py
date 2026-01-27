from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import streamlit as st

from dashboard.data_io import RESULTS_DIR, load_run
from dashboard.models import RunData
from dashboard.plotting import plot_overlay
from dashboard.stats import behavior_stats, filter_episode, overall, safe_float


def _find_matching_runs(
    runs: List[Path],
    *,
    policy: str,
    scenario: str,
    seed: int,
    tag_contains: str,
) -> List[Path]:
    pol_token = f"__policy-{policy}__"
    scn_token = f"__scenario-{scenario}__"
    seed_token = f"__seed-{seed}"

    out: List[Path] = []
    for r in runs:
        name = r.name
        if pol_token not in name:
            continue
        if scn_token not in name:
            continue
        if seed_token not in name:
            continue
        if tag_contains and tag_contains not in name:
            continue
        out.append(r)

    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("inf")


def _best_ppo_run_name_from_summary(sumdf: pd.DataFrame, *, scenario: str, seed: int) -> Optional[str]:
    """
    Pick best PPO run using results_summary.csv.
    Heuristic: minimize dropped_total_mean first, then energy_kwh_total_mean.
    Only considers rows with policy == 'ppo' and status == 'ok' (if present).
    Returns run folder name (basename of results_dir) when available.
    """
    if sumdf is None or sumdf.empty:
        return None
    if "policy" not in sumdf.columns:
        return None

    view = sumdf.copy()

    view = view[view["policy"].astype(str) == "ppo"]
    if "status" in view.columns:
        view = view[view["status"].astype(str) == "ok"]
    if "scenario" in view.columns:
        view = view[view["scenario"].astype(str) == str(scenario)]
    if "seed" in view.columns:
        view = view[view["seed"].astype(str) == str(seed)]

    if view.empty:
        return None

    drops_col = "dropped_total_mean" if "dropped_total_mean" in view.columns else None
    energy_col = "energy_kwh_total_mean" if "energy_kwh_total_mean" in view.columns else None

    def _score(row: pd.Series) -> Tuple[float, float]:
        drops = _to_float(row.get(drops_col)) if drops_col else float("inf")
        energy = _to_float(row.get(energy_col)) if energy_col else float("inf")
        return (drops, energy)

    scored = list(view.iterrows())
    scored.sort(key=lambda kv: _score(cast(pd.Series, kv[1])))
    best = cast(pd.Series, scored[0][1])

    if "results_dir" in best.index and isinstance(best.get("results_dir"), str):
        rd = str(best.get("results_dir"))
        if rd:
            return Path(rd).name
    if "run_id" in best.index and isinstance(best.get("run_id"), str):
        return str(best.get("run_id"))
    return None


def _prep_for_overlay(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    If df contains multiple episodes, create a monotonic x-axis (global_step) and
    compute run-level cumulative series from delta_* so the plot doesn't reset each episode.
    """
    df = df_in.copy()
    xcol = "step"

    if "episode" in df.columns and df["episode"].nunique() > 1:
        df = df.sort_values(["episode", "step"], kind="mergesort").reset_index(drop=True)
        df["global_step"] = range(1, len(df) + 1)
        xcol = "global_step"

        if "delta_energy_kwh" in df.columns:
            df["run_cum_energy_kwh"] = df["delta_energy_kwh"].fillna(0.0).cumsum()
        if "delta_dropped" in df.columns:
            df["run_cum_dropped"] = df["delta_dropped"].fillna(0.0).cumsum()

    return df, xcol


def render_compare(runs: List[Path]) -> None:
    st.subheader("Compare policies")
    st.caption("Pick a scenario + seed and compare NOOP vs baseline vs PPO (overlay plots + behavior stats).")

    # Default values from results_summary.csv if present
    summary_csv = RESULTS_DIR / "results_summary.csv"
    scenario_options = ["normal", "burst", "hotspot"]
    seed_options = [0, 1, 2, 3, 4]

    sumdf2: Optional[pd.DataFrame] = None
    if summary_csv.exists():
        try:
            sumdf = pd.read_csv(summary_csv)
            sumdf2 = sumdf.copy()
            if "scenario" in sumdf.columns:
                vals = [str(x) for x in sumdf["scenario"].dropna().unique().tolist()]
                if vals:
                    scenario_options = sorted(list(set(vals)))
            if "seed" in sumdf.columns:
                vals2: List[int] = []
                for x in sumdf["seed"].dropna().unique().tolist():
                    try:
                        vals2.append(int(x))
                    except Exception:
                        pass
                if vals2:
                    seed_options = sorted(list(set(vals2)))
        except Exception:
            sumdf2 = None

    c1, c2, c3 = st.columns(3)
    with c1:
        scenario = st.selectbox("Scenario", scenario_options, index=0, key="cmp_scenario")
    with c2:
        seed = st.selectbox("Seed", seed_options, index=0, key="cmp_seed")
    with c3:
        tag_contains = st.text_input(
            "Tag contains (optional)",
            value="",
            help="If you use tags in run folder names, filter runs by substring (e.g., 'matrix' or 'compare').",
            key="cmp_tag",
        ).strip()

    st.divider()

    best_ppo_name: Optional[str] = None
    if sumdf2 is not None:
        try:
            best_ppo_name = _best_ppo_run_name_from_summary(sumdf2, scenario=str(scenario), seed=int(seed))
        except Exception:
            best_ppo_name = None

    if best_ppo_name:
        bcol1, bcol2 = st.columns([2, 3])
        with bcol1:
            if st.button("Auto-pick best PPO", key="cmp_autopick_ppo"):
                st.session_state["cmp_pick_ppo"] = best_ppo_name
                st.rerun()
        with bcol2:
            st.caption(f"Best PPO from results_summary.csv: `{best_ppo_name}` (min drops → min energy)")

    policies = ["noop", "baseline", "ppo"]
    matched: Dict[str, List[Path]] = {
        p: _find_matching_runs(runs, policy=p, scenario=scenario, seed=int(seed), tag_contains=tag_contains)
        for p in policies
    }

    # Pick newest candidate by default; allow manual override.
    pick_cols = st.columns(3)
    chosen: Dict[str, Optional[Path]] = {p: (matched[p][0] if matched[p] else None) for p in policies}

    for idx, p in enumerate(policies):
        with pick_cols[idx]:
            opts = matched[p]
            if not opts:
                st.warning(f"No matching runs for policy={p}")
                chosen[p] = None
                continue
            names = [x.name for x in opts]
            default_idx = 0
            selected_name = st.selectbox(f"{p} run", names, index=default_idx, key=f"cmp_pick_{p}")
            chosen[p] = RESULTS_DIR / selected_name

    loaded: Dict[str, RunData] = {}
    for p in policies:
        if chosen[p] is None:
            continue
        rd = load_run(chosen[p])
        if rd is None:
            st.error(f"Failed to load run for {p}: {chosen[p]}")
            continue
        loaded[p] = rd

    if len(loaded) < 2:
        st.info("Select at least two policies that have matching runs to compare.")
        return

    # Episode selection (applies to all runs that have the episode column)
    episode_value: Optional[int] = None
    all_eps: List[int] = []
    for rd in loaded.values():
        if "episode" in rd.per_step.columns:
            all_eps.extend([int(x) for x in rd.per_step["episode"].dropna().unique().tolist()])
    all_eps = sorted(list(set(all_eps)))
    if all_eps:
        ep_sel = st.selectbox("Episode", ["(all)"] + all_eps, index=0, key="cmp_episode")
        episode_value = None if ep_sel == "(all)" else int(ep_sel)

    # Build filtered dfs for plotting
    series_energy: List[Tuple[str, pd.DataFrame]] = []
    series_drops: List[Tuple[str, pd.DataFrame]] = []
    series_active: List[Tuple[str, pd.DataFrame]] = []
    series_delay: List[Tuple[str, pd.DataFrame]] = []

    for p, rd in loaded.items():
        dfv_raw = filter_episode(rd.per_step, episode_value)
        dfv, _xcol = _prep_for_overlay(dfv_raw)
        label = f"{p}"

        # Use run-level cumulative columns when available; fallback to existing cum_* columns
        if "run_cum_energy_kwh" in dfv.columns:
            dfv = dfv.copy()
            dfv["cum_energy_kwh"] = dfv["run_cum_energy_kwh"]
        if "run_cum_dropped" in dfv.columns:
            dfv = dfv.copy()
            dfv["cum_dropped"] = dfv["run_cum_dropped"]

        series_energy.append((label, dfv))
        series_drops.append((label, dfv))
        series_active.append((label, dfv))
        series_delay.append((label, dfv))

    xcol_overlay = "global_step" if any("global_step" in df.columns for _, df in series_energy) else "step"

    st.subheader("Overlay plots")
    left, right = st.columns(2)

    with left:
        plot_overlay(series_energy, xcol_overlay, "cum_energy_kwh", "Cumulative energy (kWh) — overlay")
        plot_overlay(series_active, xcol_overlay, "active_ratio", "Active link ratio — overlay")

    with right:
        plot_overlay(series_drops, xcol_overlay, "cum_dropped", "Cumulative dropped — overlay")
        if any("avg_delay_ms" in df.columns for _, df in series_delay):
            plot_overlay(series_delay, xcol_overlay, "avg_delay_ms", "Average delay (ms) — overlay")

    st.divider()

    st.subheader("Summary comparison")
    rows: List[Dict[str, Any]] = []
    for p, rd in loaded.items():
        ov = overall(rd.summary)
        rows.append(
            {
                "policy": p,
                "run": rd.name,
                "reward_total_mean": ov.get("reward_total_mean", ""),
                "dropped_total_mean": ov.get("dropped_total_mean", ""),
                "energy_kwh_total_mean": ov.get("energy_kwh_total_mean", ""),
                "avg_delay_ms_mean": ov.get("avg_delay_ms_mean", ""),
                "active_ratio_mean": ov.get("active_ratio_mean", ""),
                "avg_utilization_mean": ov.get("avg_utilization_mean", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    st.subheader("Behavior stats (proves PPO isn’t just NOOP)")
    stat_cols = st.columns(len(loaded))
    for i, (p, rd) in enumerate(sorted(loaded.items(), key=lambda kv: kv[0])):
        with stat_cols[i]:
            st.markdown(f"#### {p}")
            dfv_raw = filter_episode(rd.per_step, episode_value)
            dfv, _ = _prep_for_overlay(dfv_raw)
            stats = behavior_stats(dfv)
            st.metric("No-op rate", f"{safe_float(stats.get('noop_rate')):.3f}")
            st.metric("Toggle rate", f"{safe_float(stats.get('toggle_applied_rate')):.3f}")
            st.metric("Invalid rate", f"{safe_float(stats.get('invalid_rate')):.3f}")
            st.metric("Unique actions", str(stats.get("unique_actions", 0)))
            top = stats.get("top_actions") or {}
            if top:
                st.caption("Top actions")
                st.dataframe(
                    pd.DataFrame({"action": list(top.keys()), "count": list(top.values())}),
                    use_container_width=True,
                    height=260,
                )


# Backwards-compatible alias (some app.py versions call page_compare.render(...))
def render(runs: List[Path]) -> None:
    render_compare(runs)