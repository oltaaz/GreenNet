from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import streamlit as st

from dashboard.data_io import ROOT, RESULTS_DIR, load_run
from dashboard.models import RunData
from dashboard.plotting import plot_overlay
from dashboard.stats import behavior_stats, filter_episode, overall, safe_float

POLICY_ALIASES = {
    "all_on": ["all_on", "noop"],
    "heuristic": ["heuristic", "baseline"],
    "ppo": ["ppo"],
}
SCENARIO_RE = re.compile(r"__scenario-(.+?)__seed-")
SEED_RE = re.compile(r"__seed-(\d+)")


def _find_matching_runs(
    runs: List[Path],
    *,
    policies: List[str],
    scenario: str,
    seed: int,
    tag_contains: str,
) -> List[Path]:
    policy_tokens = [f"__policy-{p}__" for p in policies]
    scn_token = f"__scenario-{scenario}__"
    seed_token = f"__seed-{seed}"

    out: List[Path] = []
    for r in runs:
        name = r.name
        if not any(tok in name for tok in policy_tokens):
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


def _extract_scenario_seed(name: str) -> Tuple[Optional[str], Optional[int]]:
    scenario_match = SCENARIO_RE.search(name)
    seed_match = SEED_RE.search(name)
    scenario = scenario_match.group(1).strip() if scenario_match else None
    seed: Optional[int] = None
    if seed_match:
        try:
            seed = int(seed_match.group(1))
        except Exception:
            seed = None
    return scenario, seed


def _scenario_seed_index(runs: List[Path], tag_contains: str) -> Dict[str, List[int]]:
    mapping: Dict[str, set[int]] = {}
    for run in runs:
        name = run.name
        if tag_contains and tag_contains not in name:
            continue
        scenario, seed = _extract_scenario_seed(name)
        if scenario is None or seed is None:
            continue
        mapping.setdefault(scenario, set()).add(seed)
    return {scenario: sorted(list(seeds)) for scenario, seeds in mapping.items()}


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


def _mean_episode_total(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    try:
        if "episode" in df.columns and df["episode"].notna().any():
            series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            grouped = series.groupby(df["episode"]).sum()
            return float(grouped.mean()) if not grouped.empty else float("nan")
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return float(series.sum())
    except Exception:
        return float("nan")


def _mean_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    try:
        return float(pd.to_numeric(df[col], errors="coerce").mean())
    except Exception:
        return float("nan")


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _qos_badge(label: str, tone: str, detail: str = "") -> str:
    palette = {
        "success": ("#0f5132", "#d1fae5", "#34d399"),
        "warning": ("#664d03", "#fef3c7", "#f59e0b"),
        "danger": ("#7f1d1d", "#fee2e2", "#ef4444"),
        "neutral": ("#1f2937", "#e5e7eb", "#9ca3af"),
    }
    fg, bg, border = palette.get(tone, palette["neutral"])
    suffix = f" {detail}" if detail else ""
    return (
        f"<span style='display:inline-flex;align-items:center;gap:0.35rem;padding:0.35rem 0.7rem;"
        f"border-radius:999px;background:{bg};color:{fg};border:1px solid {border};font-size:0.8rem;"
        f"font-weight:700;line-height:1.2;'>{label}{suffix}</span>"
    )


def _qos_tone(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"acceptable", "accepted", "pass", "passed", "met", "ok"}:
        return "success"
    if normalized in {"not_acceptable", "rejected", "fail", "failed", "violated", "violation", "breach"}:
        return "danger"
    if normalized in {"insufficient_data", "check", "pending", "unknown", "missing"}:
        return "warning"
    return "neutral"


def _compute_kpis(run: RunData) -> Dict[str, Any]:
    df = run.per_step
    ov = overall(run.summary)

    reward_mean = safe_float(ov.get("reward_total_mean")) if ov else float("nan")
    energy_mean = safe_float(ov.get("energy_kwh_total_mean")) if ov else float("nan")
    dropped_mean = safe_float(ov.get("dropped_total_mean")) if ov else float("nan")
    active_ratio_mean = safe_float(ov.get("active_ratio_mean")) if ov else float("nan")

    if not pd.isna(reward_mean):
        reward_val = reward_mean
    else:
        reward_val = _mean_episode_total(df, "reward")

    if not pd.isna(energy_mean):
        energy_val = energy_mean
    else:
        energy_val = _mean_episode_total(df, "delta_energy_kwh")

    if not pd.isna(dropped_mean):
        dropped_val = dropped_mean
    else:
        dropped_val = _mean_episode_total(df, "delta_dropped")

    if not pd.isna(active_ratio_mean):
        active_ratio_val = active_ratio_mean
    else:
        active_ratio_val = _mean_col(df, "active_ratio")

    if "toggle_applied" in df.columns:
        toggle_applied = pd.to_numeric(df["toggle_applied"], errors="coerce").fillna(0.0)
    else:
        toggle_applied = pd.Series([0.0] * len(df))
    if "toggle_reverted" in df.columns:
        toggle_reverted = pd.to_numeric(df["toggle_reverted"], errors="coerce").fillna(0.0)
    else:
        toggle_reverted = pd.Series([0.0] * len(df))
    toggles_total = float(toggle_applied.sum() + toggle_reverted.sum())
    steps = max(1, len(df))
    toggles_rate = toggles_total / float(steps)

    if "qos_violation" in df.columns:
        qos_violation = pd.to_numeric(df["qos_violation"], errors="coerce").fillna(0.0)
    else:
        qos_violation = pd.Series([0.0] * len(df))
    qos_violation_steps = float(qos_violation.sum())
    qos_violation_rate = qos_violation_steps / float(steps)
    qos_status = _first_text(ov.get("qos_acceptance_status"), ov.get("qos_acceptability_status"))
    qos_missing = _first_text(ov.get("qos_acceptance_missing"), ov.get("qos_acceptability_missing"))
    stability_status = _first_text(ov.get("stability_status"))
    stability_missing = _first_text(ov.get("stability_missing"))

    return {
        "reward_mean": float(reward_val),
        "energy_mean": float(energy_val),
        "dropped_mean": float(dropped_val),
        "active_ratio_mean": float(active_ratio_val),
        "toggles_rate": float(toggles_rate),
        "toggles_total": float(toggles_total),
        "qos_violation_rate": float(qos_violation_rate),
        "qos_violation_steps": float(qos_violation_steps),
        "qos_acceptance_status": qos_status,
        "qos_acceptance_missing": qos_missing,
        "stability_status": stability_status,
        "stability_missing": stability_missing,
        "qos_violation_rate_mean": float(qos_violation_rate),
        "qos_violation_count_total": float(qos_violation_steps),
    }


def _top_util_edges(df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, float]]:
    util_cols = [c for c in df.columns if c.startswith("edge_util_")]
    if not util_cols:
        return []
    vals = []
    for col in util_cols:
        try:
            vals.append((col, float(pd.to_numeric(df[col], errors="coerce").mean())))
        except Exception:
            continue
    vals.sort(key=lambda x: x[1], reverse=True)
    return vals[:top_k]


def render_compare(runs: List[Path]) -> None:
    st.subheader("Compare policies")
    st.caption("Pick a scenario + seed and compare a non-AI reference vs PPO (overlay plots + KPI cards).")

    st.subheader("Official Matrix Leaderboard")
    results_dir = ROOT / "results"
    official_dir = ROOT / "experiments"

    def _version_key(path: Path) -> int:
        match = re.search(r"matrix_v(\\d+)", str(path))
        return int(match.group(1)) if match else -1

    def _display_name(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except Exception:
            return str(path)

    # Preference order:
    # 1) latest official pack leaderboard
    # 2) latest results leaderboard
    official_lb = sorted(official_dir.glob("official_matrix_*/leaderboard_matrix_*.csv"))
    results_lb = sorted(results_dir.glob("leaderboard_*.csv"))
    lb_files = (official_lb + results_lb) if official_lb else results_lb
    leaderboard_path: Optional[Path] = None

    if not lb_files:
        st.warning("No leaderboard files found. Run matrix + packaging first.")
        return

    lb: Optional[pd.DataFrame] = None
    preferred_pool = official_lb if official_lb else results_lb
    latest = max(preferred_pool, key=lambda p: p.stat().st_mtime)
    option_map = {_display_name(p): p for p in lb_files}
    options = list(option_map.keys())
    default_index = options.index(_display_name(latest))
    chosen = st.selectbox(
        "Leaderboard file",
        options,
        index=default_index,
        key="lb_file_select",
    )
    leaderboard_path = option_map[chosen]

    if leaderboard_path is not None and leaderboard_path.exists():
        try:
            lb = pd.read_csv(leaderboard_path)
        except Exception as exc:
            st.warning(f"Failed to read leaderboard file: {leaderboard_path} ({exc})")
            lb = None

    if lb is not None and not lb.empty:
        scenario_filter = st.selectbox(
            "Leaderboard scenario",
            ["(all)"] + sorted(lb["scenario"].dropna().unique().tolist()) if "scenario" in lb.columns else ["(all)"],
            index=0,
            key="lb_scenario_filter",
        )
        if scenario_filter != "(all)" and "scenario" in lb.columns:
            lb = lb[lb["scenario"] == scenario_filter]

        compact = st.checkbox("Compact view", value=True, key="lb_compact")
        show_pct = st.checkbox("Show % deltas", value=False, key="lb_show_pct")

        def _add_pct_deltas(df: pd.DataFrame) -> pd.DataFrame:
            if "scenario" not in df.columns or "policy" not in df.columns:
                return df
            numeric = [
                "energy_mean",
                "dropped_mean",
                "qos_violation_rate_mean",
                "toggles_total_mean",
                "reward_mean",
            ]
            out = df.copy()
            for scenario in out["scenario"].dropna().unique().tolist():
                sc = out[out["scenario"] == scenario]
                base_all = sc[sc["policy"] == "all_on"].head(1)
                base_heur = sc[sc["policy"] == "heuristic"].head(1)
                for idx, row in sc.iterrows():
                    for metric in numeric:
                        val = row.get(metric)
                        if metric in out.columns:
                            if not base_all.empty:
                                denom = float(base_all.iloc[0].get(metric, 0.0))
                                pct = ((float(val) - denom) / denom * 100.0) if denom != 0 else 0.0
                                out.at[idx, f"{metric}_pct_vs_all_on"] = pct
                            if not base_heur.empty:
                                denom = float(base_heur.iloc[0].get(metric, 0.0))
                                pct = ((float(val) - denom) / denom * 100.0) if denom != 0 else 0.0
                                out.at[idx, f"{metric}_pct_vs_heuristic"] = pct
            return out

        if show_pct:
            lb = _add_pct_deltas(lb)

        if compact:
            preferred_cols = [
                c
                for c in [
                    "scenario",
                    "policy",
                    "energy_mean",
                    "energy_mean_pct_vs_all_on",
                    "energy_mean_pct_vs_heuristic",
                    "delta_energy_vs_all_on",
                    "delta_energy_vs_heuristic",
                    "dropped_mean",
                    "dropped_mean_pct_vs_all_on",
                    "dropped_mean_pct_vs_heuristic",
                    "delta_dropped_vs_all_on",
                    "delta_dropped_vs_heuristic",
                    "qos_violation_rate_mean",
                    "qos_acceptance_status",
                    "qos_violation_rate_mean_pct_vs_all_on",
                    "qos_violation_rate_mean_pct_vs_heuristic",
                    "delta_qos_violation_vs_all_on",
                    "delta_qos_violation_vs_heuristic",
                    "toggles_total_mean",
                    "toggles_total_mean_pct_vs_all_on",
                    "toggles_total_mean_pct_vs_heuristic",
                    "delta_toggles_vs_all_on",
                    "delta_toggles_vs_heuristic",
                    "reward_mean",
                    "reward_mean_pct_vs_all_on",
                    "reward_mean_pct_vs_heuristic",
                    "delta_reward_vs_all_on",
                    "delta_reward_vs_heuristic",
                ]
                if c in lb.columns
            ]
            if preferred_cols:
                lb = lb[preferred_cols]

        # Make the first view more readable.
        lb_view = lb.copy()
        rename = {
            "scenario": "Scenario",
            "policy": "Policy",
            "energy_mean": "Energy (kWh)",
            "delta_energy_vs_all_on": "ΔEnergy vs All-on",
            "delta_energy_vs_heuristic": "ΔEnergy vs Heuristic",
            "energy_mean_pct_vs_all_on": "%Energy vs All-on",
            "energy_mean_pct_vs_heuristic": "%Energy vs Heuristic",
            "dropped_mean": "Dropped",
            "delta_dropped_vs_all_on": "ΔDropped vs All-on",
            "delta_dropped_vs_heuristic": "ΔDropped vs Heuristic",
            "dropped_mean_pct_vs_all_on": "%Dropped vs All-on",
            "dropped_mean_pct_vs_heuristic": "%Dropped vs Heuristic",
            "qos_violation_rate_mean": "QoS viol rate",
            "qos_acceptance_status": "QoS acceptance",
            "delta_qos_violation_vs_all_on": "ΔQoS vs All-on",
            "delta_qos_violation_vs_heuristic": "ΔQoS vs Heuristic",
            "qos_violation_rate_mean_pct_vs_all_on": "%QoS vs All-on",
            "qos_violation_rate_mean_pct_vs_heuristic": "%QoS vs Heuristic",
            "toggles_total_mean": "Toggles (mean)",
            "delta_toggles_vs_all_on": "ΔToggles vs All-on",
            "delta_toggles_vs_heuristic": "ΔToggles vs Heuristic",
            "toggles_total_mean_pct_vs_all_on": "%Toggles vs All-on",
            "toggles_total_mean_pct_vs_heuristic": "%Toggles vs Heuristic",
            "reward_mean": "Reward (mean)",
            "delta_reward_vs_all_on": "ΔReward vs All-on",
            "delta_reward_vs_heuristic": "ΔReward vs Heuristic",
            "reward_mean_pct_vs_all_on": "%Reward vs All-on",
            "reward_mean_pct_vs_heuristic": "%Reward vs Heuristic",
        }
        lb_view = lb_view.rename(columns={k: v for k, v in rename.items() if k in lb_view.columns})

        # Numeric rounding for readability
        for col in lb_view.columns:
            if col in ("Scenario", "Policy"):
                continue
            try:
                lb_view[col] = pd.to_numeric(lb_view[col], errors="coerce").round(4)
            except Exception:
                continue

        if "Scenario" in lb_view.columns and "Policy" in lb_view.columns:
            lb_view = lb_view.sort_values(by=["Scenario", "Policy"])

        st.dataframe(lb_view, use_container_width=True, height=360)
        st.download_button(
            "Download leaderboard CSV",
            data=lb_view.to_csv(index=False).encode("utf-8"),
            file_name=f"{leaderboard_path.stem if leaderboard_path else 'leaderboard'}_view.csv",
            mime="text/csv",
            key="lb_download_csv",
        )
    else:
        st.warning("No leaderboard files found. Run matrix + packaging first.")
        return

    def _apply_tag_filter() -> None:
        st.session_state["cmp_tag"] = st.session_state.get("cmp_tag_input", "").strip()

    if "cmp_tag" not in st.session_state:
        st.session_state["cmp_tag"] = ""
    if "cmp_tag_input" not in st.session_state:
        st.session_state["cmp_tag_input"] = str(st.session_state.get("cmp_tag", ""))

    # Default values from results_summary.csv if present
    summary_csv = RESULTS_DIR / "results_summary.csv"
    sumdf2: Optional[pd.DataFrame] = None
    if summary_csv.exists():
        try:
            sumdf = pd.read_csv(summary_csv)
            sumdf2 = sumdf.copy()
        except Exception:
            sumdf2 = None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.text_input(
            "Tag contains (optional)",
            key="cmp_tag_input",
            help="Filter runs by tag substring, e.g. matrix_v6.",
            on_change=_apply_tag_filter,
        )
    with c2:
        if st.button("Apply tag filter", key="cmp_tag_apply"):
            _apply_tag_filter()
        tag_contains = str(st.session_state.get("cmp_tag", "")).strip()
        st.caption(f"Active tag filter: `{tag_contains or '(none)'}`")

    scenario_seed_map = _scenario_seed_index(runs, tag_contains)
    scenario_options = sorted(scenario_seed_map.keys())
    if not scenario_options:
        scenario_options = ["normal", "burst", "hotspot"]

    with c3:
        scenario_default = 0
        current_scenario = st.session_state.get("cmp_scenario")
        if current_scenario in scenario_options:
            scenario_default = scenario_options.index(current_scenario)
        scenario = st.selectbox("Scenario", scenario_options, index=scenario_default, key="cmp_scenario")

    seed_options = scenario_seed_map.get(str(scenario), [])
    if not seed_options:
        seed_options = [0]

    with c4:
        seed_default = 0
        current_seed = st.session_state.get("cmp_seed")
        if current_seed in seed_options:
            seed_default = seed_options.index(current_seed)
        seed = st.selectbox("Seed", seed_options, index=seed_default, key="cmp_seed")

    c5, c6 = st.columns(2)
    with c5:
        baseline_policy = st.selectbox(
            "Reference policy",
            ["heuristic", "all_on"],
            index=0,
            key="cmp_baseline_policy",
        )

    with c6:
        include_all_on = st.checkbox("Include all-on reference", value=False, key="cmp_include_all_on")

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

    baseline_aliases = POLICY_ALIASES.get(baseline_policy, [baseline_policy])
    ppo_aliases = POLICY_ALIASES["ppo"]
    all_on_aliases = POLICY_ALIASES["all_on"]

    matched_baseline = _find_matching_runs(
        runs,
        policies=baseline_aliases,
        scenario=scenario,
        seed=int(seed),
        tag_contains=tag_contains,
    )
    matched_ppo = _find_matching_runs(
        runs,
        policies=ppo_aliases,
        scenario=scenario,
        seed=int(seed),
        tag_contains=tag_contains,
    )
    matched_all_on: List[Path] = []
    if include_all_on:
        matched_all_on = _find_matching_runs(
            runs,
            policies=all_on_aliases,
            scenario=scenario,
            seed=int(seed),
            tag_contains=tag_contains,
        )

    pick_cols = st.columns(2 + (1 if include_all_on else 0))
    chosen: Dict[str, Optional[Path]] = {
        "baseline": matched_baseline[0] if matched_baseline else None,
        "ppo": matched_ppo[0] if matched_ppo else None,
    }
    if include_all_on:
        chosen["all_on"] = matched_all_on[0] if matched_all_on else None

    with pick_cols[0]:
        if not matched_baseline:
            st.warning(f"No matching runs for reference={baseline_policy}")
            chosen["baseline"] = None
        else:
            names = [x.name for x in matched_baseline]
            default_idx = 0
            selected_name = st.selectbox("Reference run", names, index=default_idx, key="cmp_pick_baseline")
            chosen["baseline"] = RESULTS_DIR / selected_name

    with pick_cols[1]:
        if not matched_ppo:
            st.warning("No matching PPO runs")
            chosen["ppo"] = None
        else:
            names = [x.name for x in matched_ppo]
            default_idx = 0
            requested = st.session_state.get("cmp_pick_ppo")
            if requested in names:
                default_idx = names.index(requested)
            selected_name = st.selectbox("PPO run", names, index=default_idx, key="cmp_pick_ppo_select")
            chosen["ppo"] = RESULTS_DIR / selected_name

    if include_all_on:
        with pick_cols[2]:
            if not matched_all_on:
                st.warning("No matching all_on runs")
                chosen["all_on"] = None
            else:
                names = [x.name for x in matched_all_on]
                default_idx = 0
                selected_name = st.selectbox("All-on run", names, index=default_idx, key="cmp_pick_all_on")
                chosen["all_on"] = RESULTS_DIR / selected_name

    loaded: Dict[str, RunData] = {}
    for key, path in chosen.items():
        if path is None:
            continue
        rd = load_run(path)
        if rd is None:
            st.error(f"Failed to load run for {key}: {path}")
            continue
        loaded[key] = rd

    if len(loaded) < 2:
        st.info("Select at least two policies that have matching runs to compare.")
        return

    label_map = {
        "baseline": f"{baseline_policy}",
        "ppo": "ppo",
        "all_on": "all_on",
    }

    st.subheader("KPI cards")
    kpis = {key: _compute_kpis(rd) for key, rd in loaded.items()}
    baseline_ref_key = "baseline" if "baseline" in kpis else ("all_on" if "all_on" in kpis else None)
    baseline_ref = kpis.get(baseline_ref_key) if baseline_ref_key else None

    metric_specs = [
        ("Reward (mean)", "reward_mean", True, ".3f"),
        ("Dropped (mean)", "dropped_mean", False, ".3f"),
        ("Energy (mean)", "energy_mean", False, ".6f"),
        ("Toggle rate", "toggles_rate", False, ".4f"),
        ("QoS viol rate", "qos_violation_rate", False, ".4f"),
    ]
    tolerances = {
        "reward_mean": 1e-2,
        "dropped_mean": 5.0,
        # Treat tiny changes as visually neutral (~0) for thesis screenshots.
        # Note: these are *display-only* tolerances; delta math stays the same.
        "energy_mean": 1e-3,          # ~0.001 kWh
        "qos_violation_rate": 1e-3,   # ~0.1% of steps if qos_violation is 0/1
        "toggles_rate": 2e-3,         # ~0.2% toggles per step
    }
    status_colors = {
        "good": "#16A34A",
        "bad": "#DC2626",
        "neutral": "#9CA3AF",
    }

    def _delta_display(
        metric_key: str,
        val: float,
        ref_val: float,
        *,
        higher_is_better: bool,
        fmt: str,
    ) -> tuple[str, str]:
        # Positive delta always means improvement by construction.
        delta = (val - ref_val) if higher_is_better else (ref_val - val)
        tol = float(tolerances.get(metric_key, 0.0))
        if abs(delta) <= tol:
            return "~0", "neutral"
        return (f"{delta:+{fmt}}", "good" if delta > 0 else "bad")

    order = [k for k in ["baseline", "ppo", "all_on"] if k in kpis]
    kpi_cols = st.columns(len(order))
    for idx, key in enumerate(order):
        vals = kpis[key]
        ref = baseline_ref if key != baseline_ref_key else None
        with kpi_cols[idx]:
            st.markdown(f"#### {label_map.get(key, key)}")
            for title, field, higher_is_better, fmt in metric_specs:
                value_text = f"{vals[field]:{fmt}}" if not pd.isna(vals[field]) else "nan"
                st.metric(
                    title,
                    value_text,
                    delta=None,
                )
                if ref and not pd.isna(vals[field]) and not pd.isna(ref[field]):
                    delta_text, status = _delta_display(
                        field,
                        float(vals[field]),
                        float(ref[field]),
                        higher_is_better=higher_is_better,
                        fmt=fmt,
                    )
                    st.markdown(
                        (
                            f"<div style='color:{status_colors[status]}; "
                            "font-size:0.85rem; margin-top:-0.25rem; margin-bottom:0.35rem;'>"
                            f"Δ vs reference: {delta_text}</div>"
                        ),
                        unsafe_allow_html=True,
                    )
            st.metric(
                "Active ratio",
                f"{vals['active_ratio_mean']:.3f}",
            )
            qos_status = _first_text(vals.get("qos_acceptance_status"), vals.get("qos_acceptability_status"))
            qos_missing = _first_text(vals.get("qos_acceptance_missing"), vals.get("qos_acceptability_missing"))
            stability_status = _first_text(vals.get("stability_status"))
            stability_missing = _first_text(vals.get("stability_missing"))
            badges: List[str] = []
            if qos_status:
                badges.append(_qos_badge("QoS", _qos_tone(qos_status), qos_status))
            elif qos_missing:
                badges.append(_qos_badge("QoS", "warning", qos_missing))
            if stability_status:
                badges.append(_qos_badge("Stability", _qos_tone(stability_status), stability_status))
            elif stability_missing:
                badges.append(_qos_badge("Stability", "warning", stability_missing))
            if badges:
                st.markdown(
                    "<div style='display:flex;gap:0.5rem;flex-wrap:wrap;'>" + "".join(badges) + "</div>",
                    unsafe_allow_html=True,
                )

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
    series_reward: List[Tuple[str, pd.DataFrame]] = []
    series_qos: List[Tuple[str, pd.DataFrame]] = []

    for p, rd in loaded.items():
        dfv_raw = filter_episode(rd.per_step, episode_value)
        dfv, _xcol = _prep_for_overlay(dfv_raw)
        label = label_map.get(p, p)

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
        series_reward.append((label, dfv))
        series_qos.append((label, dfv))

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

    left2, right2 = st.columns(2)
    with left2:
        if any("reward" in df.columns for _, df in series_reward):
            plot_overlay(series_reward, xcol_overlay, "reward", "Reward — overlay")
    with right2:
        if any("qos_violation" in df.columns for _, df in series_qos):
            plot_overlay(series_qos, xcol_overlay, "qos_violation", "QoS violation (0/1) — overlay")

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
                "qos_acceptance_status": _first_text(ov.get("qos_acceptance_status"), ov.get("qos_acceptability_status")),
                "qos_acceptance_missing": _first_text(ov.get("qos_acceptance_missing"), ov.get("qos_acceptability_missing")),
                "qos_violation_rate_mean": ov.get("qos_violation_rate_mean", ""),
                "qos_violation_count_total": ov.get("qos_violation_count_total", ""),
                "active_ratio_mean": ov.get("active_ratio_mean", ""),
                "avg_utilization_mean": ov.get("avg_utilization_mean", ""),
            }
        )
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True)
    st.download_button(
        "Download summary table CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="compare_summary_table.csv",
        mime="text/csv",
        key="cmp_summary_download_csv",
    )

    st.divider()

    st.subheader("Topology hints")
    hint_cols = st.columns(len(order))
    for idx, key in enumerate(order):
        rd = loaded.get(key)
        if rd is None:
            continue
        with hint_cols[idx]:
            st.markdown(f"#### {label_map.get(key, key)}")
            active_ratio = _mean_col(rd.per_step, "active_ratio")
            st.metric("Active link ratio (mean)", f"{active_ratio:.3f}")
            top_edges = _top_util_edges(rd.per_step)
            if top_edges:
                st.caption("Top-util edges (mean)")
                st.dataframe(
                    pd.DataFrame({"edge": [e for e, _ in top_edges], "mean_util": [v for _, v in top_edges]}),
                    use_container_width=True,
                    height=200,
                )
            else:
                st.caption("Top-util edges: not available (per-edge util not logged).")

    st.divider()

    st.subheader("Behavior stats (reference vs PPO)")
    stat_cols = st.columns(len(order))
    for i, key in enumerate(order):
        rd = loaded.get(key)
        if rd is None:
            continue
        with stat_cols[i]:
            st.markdown(f"#### {label_map.get(key, key)}")
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
