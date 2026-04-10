from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_official_reproduction_check_only_initializes_sqlite_and_accepts_explicit_model(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    model_dir = runs_dir / "demo_run"
    model_dir.mkdir(parents=True)
    (model_dir / "ppo_greennet.zip").write_bytes(b"placeholder")

    results_dir = tmp_path / "results"
    output_dir = tmp_path / "artifacts" / "final_pipeline" / "official_acceptance_v1"
    db_path = tmp_path / "artifacts" / "db" / "greennet.sqlite3"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "run_official_acceptance_matrix.py"),
            "--check-only",
            "--runs-dir",
            str(runs_dir),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--db-path",
            str(db_path),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert db_path.exists()
    assert "official manifest:" in completed.stdout
    assert "check-only: prerequisites satisfied." in completed.stdout


def test_official_reproduction_fails_clearly_when_explicit_ppo_checkpoint_is_missing(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "run_official_acceptance_matrix.py"),
            "--check-only",
            "--ppo-model",
            str(tmp_path / "missing" / "ppo_greennet.zip"),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--ppo-model does not exist".lower() in completed.stderr.lower()
