from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def per_step_csv(self) -> Path:
        return self.run_dir / "per_step.csv"

    @property
    def summary_json(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def meta_json(self) -> Path:
        return self.run_dir / "run_meta.json"


@dataclass
class RunData:
    name: str
    paths: RunPaths
    meta: Dict[str, Any]
    summary: Dict[str, Any]
    per_step: pd.DataFrame
