from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard.models import RunData, RunPaths


def project_root() -> Path:
    # dashboard/data_io.py -> project root is one level up from the dashboard folder
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
RESULTS_DIR = ROOT / "results"


@st.cache_data(show_spinner=False)
def _load_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def list_run_dirs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    runs = [p for p in results_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest first
    return runs


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add helpful derived columns for plotting (non-destructive)."""
    out = df.copy()

    # Step column must exist for plots.
    if "step" not in out.columns:
        return out

    # Episode column is optional; keep as-is.

    # Delta energy
    if "delta_energy_kwh" in out.columns:
        out["delta_energy_kwh"] = pd.to_numeric(out["delta_energy_kwh"], errors="coerce").fillna(0.0)
    elif "energy_kwh" in out.columns:
        e = pd.to_numeric(out["energy_kwh"], errors="coerce").fillna(0.0)
        out["delta_energy_kwh"] = e.diff().fillna(e.iloc[0]).clip(lower=0.0)
    else:
        out["delta_energy_kwh"] = 0.0

    # Delta dropped
    if "delta_dropped" in out.columns:
        out["delta_dropped"] = pd.to_numeric(out["delta_dropped"], errors="coerce").fillna(0.0)
    elif "dropped" in out.columns:
        d = pd.to_numeric(out["dropped"], errors="coerce").fillna(0.0)
        out["delta_dropped"] = d.diff().fillna(d.iloc[0]).clip(lower=0.0)
    else:
        out["delta_dropped"] = 0.0

    # If we have multiple episodes, sort for stable plotting/aggregation and create a monotonic x-axis.
    if "episode" in out.columns:
        out["episode"] = pd.to_numeric(out["episode"], errors="coerce")
        out = out.sort_values(["episode", "step"], kind="mergesort").reset_index(drop=True)
        out["global_step"] = range(1, len(out) + 1)

    # Cumulative series for plotting:
    # - per-episode cum (resets each episode)
    # - run-level cum (monotonic across the entire dataframe)
    if "episode" in out.columns and out["episode"].notna().any():
        out["ep_cum_energy_kwh"] = out.groupby("episode")["delta_energy_kwh"].cumsum()
        out["ep_cum_dropped"] = out.groupby("episode")["delta_dropped"].cumsum()
    else:
        out["ep_cum_energy_kwh"] = out["delta_energy_kwh"].cumsum()
        out["ep_cum_dropped"] = out["delta_dropped"].cumsum()

    out["run_cum_energy_kwh"] = out["delta_energy_kwh"].cumsum()
    out["run_cum_dropped"] = out["delta_dropped"].cumsum()

    # Backwards-compatible names used by existing pages:
    out["cum_energy_kwh"] = out["run_cum_energy_kwh"]
    out["cum_dropped"] = out["run_cum_dropped"]

    # Numeric cleaning for common columns (if present)
    for col in ["active_ratio", "avg_delay_ms", "avg_utilization", "max_util", "min_util", "p95_util"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def load_run(run_dir: Path) -> Optional[RunData]:
    paths = RunPaths(run_dir=run_dir)

    df = _load_csv(str(paths.per_step_csv))
    if df is None:
        return None

    meta = _load_json(str(paths.meta_json)) or {}
    summary = _load_json(str(paths.summary_json)) or {}

    df = _ensure_derived_columns(df)

    return RunData(
        name=run_dir.name,
        paths=paths,
        meta=meta,
        summary=summary,
        per_step=df,
    )
