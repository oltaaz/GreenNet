from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def run_experiment_cli(repo_root: Path):
    def _run(
        tmp_path: Path,
        *,
        policy: str = "noop",
        scenario: str = "normal",
        seed: int = 7,
        steps: int = 4,
        episodes: int = 1,
        extra_args: list[str] | None = None,
    ) -> tuple[Path, subprocess.CompletedProcess[str]]:
        out_dir = tmp_path / "results"
        cmd = [
            sys.executable,
            str(repo_root / "run_experiment.py"),
            "--policy",
            policy,
            "--scenario",
            scenario,
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            "--episodes",
            str(episodes),
            "--out-dir",
            str(out_dir),
        ]
        if extra_args:
            cmd.extend(extra_args)

        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.fail(
                "run_experiment.py failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        run_dirs = sorted(path for path in out_dir.iterdir() if path.is_dir())
        assert len(run_dirs) == 1
        return run_dirs[0], completed

    return _run


@pytest.fixture
def generated_run(tmp_path: Path, run_experiment_cli):
    run_dir, _ = run_experiment_cli(
        tmp_path,
        policy="noop",
        scenario="normal",
        seed=11,
        steps=4,
        episodes=1,
    )
    return run_dir
