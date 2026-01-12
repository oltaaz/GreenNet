"""Streamlit dashboard for GreenNet metrics."""
from __future__ import annotations

from typing import Dict

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional dependency
    st = None
    _streamlit_import_error = exc
else:
    _streamlit_import_error = None


def render(metrics: Dict[str, float]) -> None:
    if st is None:
        raise ImportError("streamlit is required to run the dashboard") from _streamlit_import_error

    st.set_page_config(page_title="GreenNet Dashboard", layout="wide")
    st.title("GreenNet Dashboard")
    for name, value in metrics.items():
        st.metric(label=name, value=value)


if __name__ == "__main__":
    render({"throughput": 0.0, "latency": 0.0, "carbon": 0.0})
