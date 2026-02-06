#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _git_head(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "<UNKNOWN>"
    except Exception:
        pass
    return "<UNKNOWN>"


def _format_seed_list(seeds: List[int]) -> str:
    if not seeds:
        return "<UNKNOWN>"
    seeds_sorted = sorted(seeds)
    if seeds_sorted == list(range(seeds_sorted[0], seeds_sorted[-1] + 1)):
        return f"{seeds_sorted[0]}–{seeds_sorted[-1]} ({len(seeds_sorted)} seeds)"
    return ", ".join(str(s) for s in seeds_sorted)


def _find_model_path(results_dir: Path, tag: str) -> str:
    token = f"__tag-{tag}"
    paths: List[str] = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if token not in run_dir.name:
            continue
        meta = _load_json(run_dir / "run_meta.json")
        if not meta:
            continue
        if meta.get("policy") != "ppo":
            continue
        mp = meta.get("model_path")
        if mp:
            paths.append(str(mp))
    if not paths:
        return "<UNKNOWN>"
    return Counter(paths).most_common(1)[0][0]


def _model_path_from_command(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except Exception:
        return None
    for idx, part in enumerate(parts):
        if part == "--ppo-model" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _write_by_seed(summary_path: Path, out_path: Path) -> None:
    df = pd.read_csv(summary_path)
    if df.empty:
        out_path.write_text("", encoding="utf-8")
        return

    group_cols = ["policy", "scenario"]
    metric_cols = [c for c in df.columns if c.endswith("_mean")]
    metric_cols = [c for c in metric_cols if c not in group_cols]

    # Coerce to numeric for aggregation
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()

    # Flatten columns: <metric>_<stat>
    flat_cols: List[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            if base in group_cols:
                flat_cols.append(base)
            else:
                flat_cols.append(f"{base}_{stat}")
        else:
            flat_cols.append(str(col))
    grouped.columns = flat_cols

    # Reorder: policy, scenario, then metrics in original order with mean/std/count
    ordered_cols: List[str] = ["policy", "scenario"]
    for metric in metric_cols:
        for stat in ("mean", "std", "count"):
            col = f"{metric}_{stat}"
            if col in grouped.columns:
                ordered_cols.append(col)
    ordered_cols += [c for c in grouped.columns if c not in ordered_cols]

    grouped = grouped[ordered_cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package an official matrix run (aggregate + leaderboard + notes).")
    parser.add_argument("--tag", required=True, help="Matrix tag (e.g., matrix_v6)")
    parser.add_argument("--out", required=True, help="Output folder (e.g., experiments/official_matrix_v6)")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--matrix-command", default="<FILL_MATRIX_COMMAND>")
    parser.add_argument("--train-command", default="<FILL_TRAIN_COMMAND>")
    parser.add_argument("--headline", default="<FILL_HEADLINE_RESULT>")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tag = str(args.tag)
    summary_path = results_dir / f"results_summary_{tag}.csv"
    by_seed_path = results_dir / f"results_summary_by_seed_{tag}.csv"
    leaderboard_path = results_dir / f"leaderboard_{tag}.csv"
    leaderboard_source_path = results_dir / f"leaderboard_source_{tag}.csv"

    # 1) Aggregate results
    _run(
        [
            sys.executable,
            str(repo_root / "experiments" / "aggregate_results.py"),
            "--out-dir",
            str(results_dir),
            "--tag",
            tag,
            "--output",
            str(summary_path),
        ]
    )

    # 2) By-seed summary
    _write_by_seed(summary_path, by_seed_path)

    # 3) Leaderboard
    _run(
        [
            sys.executable,
            str(repo_root / "experiments" / "make_leaderboard.py"),
            "--summary",
            str(summary_path),
            "--output",
            str(leaderboard_path),
            "--output-source",
            str(leaderboard_source_path),
        ]
    )

    # 4) Notes and package folder
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    policies = sorted(df.get("policy", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    scenarios = sorted(df.get("scenario", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    seeds = []
    if "seed" in df.columns:
        for val in df["seed"].dropna().tolist():
            try:
                seeds.append(int(val))
            except Exception:
                continue
    seeds = sorted(set(seeds))
    episodes = sorted(set(int(v) for v in df.get("episodes", pd.Series(dtype=int)).dropna().tolist()))
    steps = sorted(set(int(v) for v in df.get("max_steps", pd.Series(dtype=int)).dropna().tolist()))

    model_path = _find_model_path(results_dir, tag)
    model_path_from_cmd = _model_path_from_command(args.matrix_command)
    if model_path == "<UNKNOWN>" and model_path_from_cmd:
        model_path = model_path_from_cmd
    commit = _git_head(repo_root)
    generated_at_utc = datetime.now(timezone.utc).isoformat()

    notes = """# Official Matrix {tag}

## Metadata
- tag: {tag}
- generated_at_utc: {generated_at_utc}

## Commit
- {commit}

## Commands
- train: {train_cmd}
- matrix: {matrix_cmd}

## Policies
- {policies}

## Scenarios
- {scenarios}

## Seeds
- {seeds}

## Episodes / Steps
- episodes: {episodes}
- steps: {steps}

## PPO model path
- {model_path}

## Headline result
- {headline}
""".format(
        tag=tag,
        commit=commit,
        generated_at_utc=generated_at_utc,
        train_cmd=args.train_command,
        matrix_cmd=args.matrix_command,
        policies=", ".join(policies) if policies else "<UNKNOWN>",
        scenarios=", ".join(scenarios) if scenarios else "<UNKNOWN>",
        seeds=_format_seed_list(seeds),
        episodes=", ".join(str(e) for e in episodes) if episodes else "<UNKNOWN>",
        steps=", ".join(str(s) for s in steps) if steps else "<UNKNOWN>",
        model_path=model_path,
        headline=args.headline,
    )

    notes_path = out_dir / "notes.md"
    notes_path.write_text(notes, encoding="utf-8")

    for src in (summary_path, by_seed_path, leaderboard_path, leaderboard_source_path, notes_path):
        if not src.exists():
            continue
        dest = out_dir / src.name
        try:
            if src.resolve() == dest.resolve():
                continue
        except Exception:
            pass
        shutil.copy2(src, dest)

    print(f"[package] wrote official pack to {out_dir}")


if __name__ == "__main__":
    main()
