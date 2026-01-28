from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

app = FastAPI(title="GreenNet Metrics API")

# Figure out the repo root reliably.
# If this file is at repo root (GreenNet/api_app.py) → repo root is this file's parent.
# If later moved into a package (GreenNet/greennet/api_app.py) → repo root is one level up.
THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_FILE_DIR
if (REPO_ROOT / "runs").exists() is False and (REPO_ROOT / "results").exists() is False:
    # Likely inside a subpackage; fall back one directory.
    REPO_ROOT = THIS_FILE_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"


def _scan_run_dirs() -> List[Path]:
    dirs: List[Path] = []
    for base in [RESULTS_DIR, RUNS_DIR]:
        if base.exists():
            dirs.extend([p for p in base.iterdir() if p.is_dir()])
    # newest first (by folder name timestamp, usually)
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def _find_run_dir(run_id: str) -> Optional[Path]:
    for p in _scan_run_dirs():
        if p.name == run_id:
            return p
    return None


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for d in _scan_run_dirs():
        # Only include directories that look like an experiment/run
        has_per_step = (d / "per_step.csv").exists()
        has_summary = (d / "summary.json").exists()
        has_meta = (d / "run_meta.json").exists()
        if not (has_per_step or has_summary or has_meta):
            continue

        runs.append({
            "run_id": d.name,
            "started_at": None,      # optional, fill later if you want
            "policy": None,
            "scenario": None,
            "topology_seed": None,
        })
    return runs


@app.get("/api/runs/{run_id}/per_step")
def run_per_step(run_id: str) -> List[Dict[str, Any]]:
    d = _find_run_dir(run_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")

    csv_path = d / "per_step.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="per_step.csv not found for this run")

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert common numeric fields when present
            out: Dict[str, Any] = dict(r)
            for k in ["t", "energy", "dropped", "reward", "active_ratio"]:
                if k in out and out[k] is not None and out[k] != "":
                    try:
                        out[k] = float(out[k]) if k != "t" else int(float(out[k]))
                    except ValueError:
                        pass
            rows.append(out)
    return rows