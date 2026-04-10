from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Sequence

from greennet.evaluation.acceptance_matrix import official_acceptance_matrix_path
from greennet.evaluation.final_pipeline import build_pipeline
from greennet.evaluation.official_ppo import (
    canonical_official_ppo_family_dir,
    canonical_official_ppo_model_path,
    missing_official_ppo_topologies,
    official_ppo_exists,
)
from greennet.persistence import default_db_path, get_run_repository


INSTALL_HINT = "Install project dependencies with: python3.12 -m venv .venv && .venv/bin/python -m pip install -e '.[test,train]'"


class ReproductionError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_text: str | Path) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (_repo_root() / raw).resolve()


def _find_latest_model(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates: list[Path] = []
    candidates.extend(sorted(runs_dir.glob("*/ppo_greennet.zip")))
    candidates.extend(sorted(runs_dir.glob("*/ppo_greennet")))
    candidates.extend(sorted(runs_dir.glob("ppo_greennet.zip")))
    candidates.extend(sorted(runs_dir.glob("ppo_greennet")))
    return candidates[-1] if candidates else None


def _require_modules(module_names: Sequence[str]) -> None:
    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ReproductionError(
            f"Missing required Python dependencies for the official reproduction path: {formatted}. {INSTALL_HINT}"
        )


def _preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical one-command GreenNet final reproduction path: "
            "preflight checks, SQLite initialization, official acceptance-matrix execution, "
            "aggregation, and final report packaging."
        ),
        epilog=(
            "This command always uses the canonical official acceptance matrix. "
            "Use experiments/run_matrix.py or greennet.evaluation.final_pipeline for custom benchmarks."
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_pipeline/official_acceptance_v1"))
    parser.add_argument("--ppo-model", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser


def _write_reviewer_summary(manifest: dict, *, db_path: Path) -> Path:
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
    report_dir = Path(
        str(
            outputs.get("concise_report")
            or _repo_root() / "artifacts" / "final_pipeline" / "official_acceptance_v1" / "report"
        )
    ).resolve().parent
    summary_path = report_dir / "reviewer_start_here.md"
    lines = [
        "# GreenNet Official Reproduction",
        "",
        "This bundle was produced by the canonical one-command reproduction path.",
        "",
        "## Main Outputs",
        f"- Quick claim summary: `{outputs.get('concise_report', 'n/a')}`",
        f"- Final thesis-facing report: `{outputs.get('final_evaluation_report', 'n/a')}`",
        f"- Authoritative summary CSV: `{outputs.get('summary_csv', 'n/a')}`",
        f"- Final evaluation JSON: `{outputs.get('final_evaluation_json', 'n/a')}`",
        f"- SQLite store: `{db_path}`",
        "",
        "## Review Order",
        f"1. `{outputs.get('concise_report', 'n/a')}`",
        f"2. `{outputs.get('final_evaluation_report', 'n/a')}`",
        f"3. `{outputs.get('summary_csv', 'n/a')}`",
        f"4. `{outputs.get('final_evaluation_json', 'n/a')}`",
        "",
        "## Notes",
        "- File artifacts remain the reviewer-facing export layer.",
        "- SQLite is the primary structured store for indexed runs and persisted final-evaluation payloads.",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path


def run_official_reproduction(argv: Sequence[str] | None = None) -> dict:
    parser = _preflight_parser()
    args, remaining = parser.parse_known_args(argv)
    if "--matrix-manifest" in remaining:
        raise ReproductionError(
            "The official reproduction command always uses the canonical acceptance matrix manifest. "
            "Do not pass --matrix-manifest here; use experiments/run_matrix.py or greennet.evaluation.final_pipeline for custom variants."
        )

    _require_modules(["gymnasium", "networkx", "numpy", "pandas", "stable_baselines3"])

    repo_root = _repo_root()
    results_dir = _resolve_path(args.results_dir)
    runs_dir = _resolve_path(args.runs_dir)
    summary_csv = _resolve_path(args.summary_csv) if args.summary_csv is not None else None
    ppo_model = _resolve_path(args.ppo_model) if args.ppo_model is not None else None
    explicit_ppo_model = ppo_model
    db_path = _resolve_path(args.db_path) if args.db_path is not None else default_db_path()

    if summary_csv is not None and not summary_csv.exists():
        raise ReproductionError(f"--summary-csv does not exist: {summary_csv}")
    if ppo_model is not None and not ppo_model.exists():
        raise ReproductionError(f"--ppo-model does not exist: {ppo_model}")

    if summary_csv is None and not args.skip_eval:
        resolved_model = ppo_model
        if resolved_model is None and not missing_official_ppo_topologies():
            resolved_model = canonical_official_ppo_model_path("medium")
        if resolved_model is None:
            resolved_model = _find_latest_model(runs_dir)
        if resolved_model is None:
            raise ReproductionError(
                "The official acceptance matrix includes policy=ppo, but no PPO checkpoint was found. "
                f"Expected the canonical checkpoint family under {canonical_official_ppo_family_dir()} or provide --ppo-model "
                f"or place a checkpoint under {runs_dir} (for example runs/<RUN_ID>/ppo_greennet.zip)."
            )
        if ppo_model is None:
            missing_topologies = missing_official_ppo_topologies()
            if missing_topologies:
                raise ReproductionError(
                    "The official acceptance matrix requires topology-specific PPO checkpoints for small, medium, and large. "
                    f"Missing canonical artifacts for: {', '.join(missing_topologies)}. "
                    f"Regenerate them into {canonical_official_ppo_family_dir()} or pass --ppo-model for a non-official single-topology run."
                )
        ppo_model = resolved_model

    os.environ["GREENNET_DB_PATH"] = str(db_path)
    repository = get_run_repository(db_path)
    repository.ensure_initialized()

    print("[reproduce_final_claim] official manifest:", official_acceptance_matrix_path())
    print("[reproduce_final_claim] results dir:", results_dir)
    print("[reproduce_final_claim] runs dir:", runs_dir)
    print("[reproduce_final_claim] sqlite db:", db_path)
    if ppo_model is not None:
        print("[reproduce_final_claim] ppo model:", ppo_model)
        if explicit_ppo_model is None and ppo_model == canonical_official_ppo_model_path("medium"):
            print("[reproduce_final_claim] ppo model family:", canonical_official_ppo_family_dir())
    if summary_csv is not None:
        print("[reproduce_final_claim] summary csv override:", summary_csv)

    if args.check_only:
        print("[reproduce_final_claim] check-only: prerequisites satisfied.")
        return {
            "status": "ok",
            "matrix_manifest": str(official_acceptance_matrix_path()),
            "results_dir": str(results_dir),
            "runs_dir": str(runs_dir),
            "db_path": str(db_path),
            "ppo_model": None if ppo_model is None else str(ppo_model),
            "summary_csv": None if summary_csv is None else str(summary_csv),
        }

    pipeline_argv = [
        "--matrix-manifest",
        str(official_acceptance_matrix_path()),
        "--results-dir",
        str(results_dir),
        "--runs-dir",
        str(runs_dir),
        "--output-dir",
        str(_resolve_path(args.output_dir)),
    ]
    if summary_csv is not None:
        pipeline_argv.extend(["--summary-csv", str(summary_csv)])
    if args.skip_eval:
        pipeline_argv.append("--skip-eval")
    if args.skip_plots:
        pipeline_argv.append("--skip-plots")
    if explicit_ppo_model is not None:
        pipeline_argv.extend(["--ppo-model", str(ppo_model)])
    pipeline_argv.extend(remaining)
    manifest = build_pipeline(pipeline_argv)
    reviewer_summary = _write_reviewer_summary(manifest, db_path=db_path)
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
    print("[reproduce_final_claim] bundle ready:", outputs.get("final_evaluation_report", "n/a"))
    print("[reproduce_final_claim] reviewer summary:", reviewer_summary)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    try:
        run_official_reproduction(argv)
    except ReproductionError as exc:
        raise SystemExit(str(exc)) from exc
