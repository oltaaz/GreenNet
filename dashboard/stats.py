from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def behavior_stats(df: pd.DataFrame) -> Dict[str, Any]:
    def _mean_bool(col: str) -> float:
        if col not in df.columns:
            return 0.0
        s = df[col]
        # handle bools or 0/1
        try:
            return float(pd.to_numeric(s, errors="coerce").fillna(0.0).mean())
        except Exception:
            return 0.0

    stats: Dict[str, Any] = {
        "rows": int(len(df)),
        "noop_rate": _mean_bool("action_is_noop"),
        "toggle_applied_rate": _mean_bool("toggle_applied"),
        "invalid_rate": _mean_bool("action_is_invalid"),
    }

    if "action" in df.columns:
        try:
            stats["unique_actions"] = int(df["action"].nunique())
            stats["top_actions"] = df["action"].value_counts().head(10).to_dict()
        except Exception:
            stats["unique_actions"] = 0
            stats["top_actions"] = {}
    else:
        stats["unique_actions"] = 0
        stats["top_actions"] = {}

    return stats


def overall(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    ov = summary.get("overall")
    return ov if isinstance(ov, dict) else {}


def filter_episode(df: pd.DataFrame, episode: Optional[int]) -> pd.DataFrame:
    if episode is None:
        return df
    if "episode" not in df.columns:
        return df
    return df[df["episode"] == episode]
