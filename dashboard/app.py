from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard import page_compare, page_single
from dashboard.data_io import RESULTS_DIR

RUN_DIR_RE = re.compile(r"\[run_experiment\] results saved to (.+)")


def _project_root() -> Path:
    # dashboard/ is inside project root
    return Path(__file__).resolve().parents[1]

def _list_runs(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    runs = [p for p in results_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)  # newest first
    return runs


def _extract_run_dir(stdout: str, stderr: str) -> str:
    match = None
    for line in (stdout + "\n" + stderr).splitlines():
        m = RUN_DIR_RE.search(line)
        if m:
            match = m
    return match.group(1).strip() if match else ""


def render_launcher() -> None:
    st.subheader("Live demo mode")
    st.caption("One click: run an experiment and auto-open the results in the Single run page.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        policy = st.selectbox("Policy", ["noop", "baseline", "ppo"], index=2, key="launch_policy")
    with c2:
        scenario = st.selectbox("Scenario", ["normal", "burst", "hotspot"], index=0, key="launch_scenario")
    with c3:
        seed = st.number_input("Seed", min_value=0, max_value=100000, value=0, step=1, key="launch_seed")
    with c4:
        episodes = st.number_input("Episodes", min_value=1, max_value=100, value=1, step=1, key="launch_episodes")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        steps = st.number_input("Steps", min_value=10, max_value=50000, value=300, step=10, key="launch_steps")
    with c6:
        tag = st.text_input("Tag", value="dashboard", key="launch_tag")
    with c7:
        deterministic = st.checkbox("Deterministic", value=True, key="launch_det")
    with c8:
        runs_dir = st.text_input("runs_dir", value="runs", key="launch_runs_dir")

    root = _project_root()
    script = root / "run_experiment.py"

    st.markdown("**Command preview**")
    cmd = [
        sys.executable,
        str(script),
        "--policy", str(policy),
        "--scenario", str(scenario),
        "--seed", str(int(seed)),
        "--episodes", str(int(episodes)),
        "--steps", str(int(steps)),
        "--tag", (str(tag).strip() or "dashboard"),
        "--runs-dir", (str(runs_dir).strip() or "runs"),
    ]
    if not deterministic:
        cmd.append("--stochastic")

    st.code(" ".join(cmd))

    if st.button("🚀 Run & open results", key="launch_run"):
        with st.spinner("Running experiment..."):
            res = subprocess.run(
                cmd,
                cwd=str(root),           # IMPORTANT: run from project root
                capture_output=True,
                text=True,
                check=False,
            )

        st.markdown("### Output")
        if res.stdout:
            st.code(res.stdout)
        if res.stderr:
            st.code(res.stderr)

        if res.returncode != 0:
            st.error(f"Experiment failed (exit {res.returncode}).")
            return

        run_dir = _extract_run_dir(res.stdout, res.stderr)
        if run_dir:
            run_name = Path(run_dir).name
            st.success(f"Run completed: {run_name}")

            # auto-load it on Single run page
            st.session_state["autoselect_run"] = run_name
            st.session_state["nav_page_request"] = "Single run"
            st.rerun()
        else:
            st.success("Run completed, but could not parse run folder from output.")
            st.caption("Open Single run and select the newest results folder.")


def main() -> None:
    st.set_page_config(page_title="GreenNet Dashboard", layout="wide")
    st.title("GreenNet Dashboard")

    # Apply any pending navigation request BEFORE the nav widget is created.
    if "nav_page_request" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("nav_page_request")

    # Default page state (separate from the widget key to avoid Streamlit state mutation errors).
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Single run"

    pages = ["Single run", "Compare policies", "Run launcher"]

    # optional: allow launcher to request navigation
    requested = st.session_state.pop("nav_page_request", None)
    if requested in pages:
        st.session_state["nav_page"] = requested

    current = st.session_state.get("nav_page", "Single run")
    if current not in pages:
        current = "Single run"

    page = st.radio(
        "Navigation",
        pages,
        index=pages.index(current),
        horizontal=True,
        key="nav_page_radio",   # IMPORTANT: different key
    )

    # persist user selection
    st.session_state["nav_page"] = page

    root = _project_root()
    results_dir = root / "results"
    runs: list[Path] = []
    if RESULTS_DIR.exists():
        runs = sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir()], reverse=True)

    if page == "Single run":
        page_single.render_single_run(runs)
    elif page == "Compare policies":
        page_compare.render_compare(runs)
    else:
        render_launcher()

    


if __name__ == "__main__":
    main()
