#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _venv_python(repo_root: Path) -> Path:
    if os.name == "nt":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def main() -> int:
    repo_root = _repo_root()
    venv_python = _venv_python(repo_root)

    if venv_python.exists():
        python_executable = str(venv_python)
    else:
        python_executable = sys.executable

    command = [python_executable, "-m", "uvicorn", "api_app:app", "--reload", "--port", "8000"]

    try:
        completed = subprocess.run(command, cwd=repo_root, check=False)
    except FileNotFoundError:
        print(
            "GreenNet backend launcher could not find a usable Python interpreter. "
            "Create the local .venv first or install Python 3.12.",
            file=sys.stderr,
        )
        return 1

    if completed.returncode != 0 and not venv_python.exists():
        print(
            "GreenNet backend launcher could not start uvicorn from the current Python interpreter. "
            "Create the repo-local .venv with `python3.12 -m venv .venv` and install dependencies with "
            "`.venv/bin/python -m pip install -e '.[test,train]'`.",
            file=sys.stderr,
        )

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
