from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


DEFAULT_POLICIES = "all_on,heuristic,ppo"
DEFAULT_SCENARIOS = "normal,burst,hotspot"
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"
DEFAULT_BASELINE_POLICIES = "heuristic,utilization_threshold,baseline,all_on,noop"
DEFAULT_AI_POLICIES = "ppo"
DEFAULT_TAG = "final_pipeline"


class PipelineError(RuntimeError):
    pass


@dataclass
class StepRecord:
    name: str
    status: str
    log_path: str
    outputs: list[str] = field(default_factory=list)
    command: str | None = None
    details: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_text: str | Path) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (_repo_root() / raw).resolve()


def _clean_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_seed_csv(value: str | None) -> list[int]:
    if not value:
        return []
    seeds: list[int] = []
    for token in _clean_csv_list(value):
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            seeds.extend(list(range(start, end + step, step)))
        else:
            seeds.append(int(token))
    return seeds


def _bool_from_csv(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _tail_text(text: str, *, lines: int = 20) -> str:
    parts = [line for line in text.rstrip().splitlines() if line.strip()]
    if not parts:
        return "<no output captured>"
    return "\n".join(parts[-lines:])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise PipelineError(f"Expected JSON object at {path}")
    return data


def _git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


def _run_command(
    *,
    name: str,
    cmd: Sequence[str],
    log_path: Path,
    step_records: list[StepRecord],
    cwd: Path | None = None,
    outputs: Sequence[Path] | None = None,
) -> None:
    workdir = cwd or _repo_root()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = _quote_cmd(cmd)
    print(f"[final_pipeline] {name}: {command_text}")

    result = subprocess.run(
        list(cmd),
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )

    body = [
        f"$ {command_text}",
        "",
        "[stdout]",
        result.stdout.rstrip(),
        "",
        "[stderr]",
        result.stderr.rstrip(),
        "",
        f"[returncode] {result.returncode}",
    ]
    log_path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")

    if result.returncode != 0:
        step_records.append(
            StepRecord(
                name=name,
                status="failed",
                log_path=str(log_path),
                outputs=[str(path) for path in outputs or []],
                command=command_text,
                details=f"exit_code={result.returncode}",
            )
        )
        tail = _tail_text(result.stderr or result.stdout)
        raise PipelineError(
            f"{name} failed with exit code {result.returncode}. "
            f"See log: {log_path}\n{tail}"
        )

    step_records.append(
        StepRecord(
            name=name,
            status="ok",
            log_path=str(log_path),
            outputs=[str(path) for path in outputs or []],
            command=command_text,
        )
    )
    print(f"[final_pipeline] {name}: ok -> {log_path}")


def _run_python_step(
    *,
    name: str,
    log_path: Path,
    step_records: list[StepRecord],
    outputs: Sequence[Path] | None,
    fn: Any,
    description: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[final_pipeline] {name}: {description}")
    try:
        fn()
    except Exception as exc:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        step_records.append(
            StepRecord(
                name=name,
                status="failed",
                log_path=str(log_path),
                outputs=[str(path) for path in outputs or []],
                command=description,
                details=str(exc),
            )
        )
        raise PipelineError(f"{name} failed. See log: {log_path}\n{exc}") from exc

    if log_path.exists():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[status] ok\n{description}\n")
    else:
        log_path.write_text(f"{description}\n", encoding="utf-8")
    step_records.append(
        StepRecord(
            name=name,
            status="ok",
            log_path=str(log_path),
            outputs=[str(path) for path in outputs or []],
            command=description,
        )
    )
    print(f"[final_pipeline] {name}: ok -> {log_path}")


def _copy_rows_to_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames: list[str] = list(fieldnames or [])
    if not resolved_fieldnames:
        seen: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(str(key))
        resolved_fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in resolved_fieldnames})


def _write_by_seed_summary(summary_path: Path, output_path: Path) -> None:
    try:
        from experiments.package_official_matrix import _write_by_seed
    except ModuleNotFoundError as exc:
        raise PipelineError(
            "The by-seed summary step requires optional dependency 'pandas'. "
            "Install project dependencies before running the full pipeline."
        ) from exc
    _write_by_seed(summary_path, output_path)


def _filter_summary_csv(
    *,
    source_path: Path,
    output_path: Path,
    seed_filter: set[str],
    scenario_filter: set[str],
    policy_filter: set[str],
    deterministic_filter: bool | None,
) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise PipelineError(f"Summary CSV has no header: {source_path}")

        kept_rows: list[dict[str, Any]] = []
        non_ok_rows: list[dict[str, str]] = []
        observed_ok: set[tuple[str, str, str]] = set()

        for row in reader:
            policy = str(row.get("policy") or "").strip()
            scenario = str(row.get("scenario") or "").strip()
            seed = str(row.get("seed") or "").strip()
            status = str(row.get("status") or "").strip().lower()
            deterministic = _bool_from_csv(row.get("deterministic"))

            if policy_filter and policy not in policy_filter:
                continue
            if scenario_filter and scenario not in scenario_filter:
                continue
            if seed_filter and seed not in seed_filter:
                continue
            if deterministic_filter is not None and deterministic is not None and deterministic != deterministic_filter:
                continue

            if status and status != "ok":
                non_ok_rows.append(
                    {
                        "policy": policy,
                        "scenario": scenario,
                        "seed": seed,
                        "status": status,
                        "error": str(row.get("error") or ""),
                    }
                )
                continue

            kept_rows.append(dict(row))
            observed_ok.add((policy, scenario, seed))

    if not kept_rows:
        raise PipelineError(
            "No successful summary rows matched the requested filters. "
            f"source={source_path}"
        )

    expected = {
        (policy, scenario, seed)
        for policy, scenario, seed in product(
            sorted(policy_filter),
            sorted(scenario_filter),
            sorted(seed_filter),
        )
    }
    missing = sorted(expected.difference(observed_ok))
    if missing:
        preview = ", ".join(f"{policy}/{scenario}/seed={seed}" for policy, scenario, seed in missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise PipelineError(
            "The selected summary is incomplete for the requested matrix. "
            f"Missing successful rows: {preview}{suffix}"
        )

    _copy_rows_to_csv(output_path, kept_rows, fieldnames=fieldnames)
    return {
        "rows_written": len(kept_rows),
        "non_ok_rows": non_ok_rows,
        "expected_combinations": len(expected),
    }


def _final_eval_paths(final_eval_dir: Path) -> dict[str, Path]:
    return {
        "csv": final_eval_dir / "final_evaluation_summary.csv",
        "json": final_eval_dir / "final_evaluation_summary.json",
        "report": final_eval_dir / "final_evaluation_report.md",
    }


def _build_research_question_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary_rows = payload.get("summary_rows")
    if not isinstance(summary_rows, list):
        raise PipelineError("final_evaluation_summary.json is missing summary_rows")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw_row in summary_rows:
        if not isinstance(raw_row, dict):
            continue
        scope_type = str(raw_row.get("scope_type") or "")
        scope = str(raw_row.get("scope") or "")
        if scope_type not in {"overall", "scenario"}:
            continue
        grouped.setdefault((scope_type, scope), []).append(raw_row)

    output: list[dict[str, Any]] = []
    scope_order = sorted(grouped.keys(), key=lambda item: (0 if item[0] == "overall" else 1, item[1]))
    for scope_type, scope in scope_order:
        bucket = grouped[(scope_type, scope)]
        baseline_row = next((row for row in bucket if bool(row.get("is_primary_baseline"))), None)
        ai_row = next((row for row in bucket if bool(row.get("is_best_ai_policy_for_scope"))), None)
        if ai_row is None:
            ai_candidates = [row for row in bucket if str(row.get("policy_class") or "") == "ai"]
            ai_row = ai_candidates[0] if ai_candidates else None

        if baseline_row is None or ai_row is None:
            output.append(
                {
                    "scope_type": scope_type,
                    "scope": scope,
                    "baseline_policy": baseline_row.get("policy") if baseline_row else None,
                    "ai_policy": ai_row.get("policy") if ai_row else None,
                    "status": "missing_comparison_rows",
                }
            )
            continue

        output.append(
            {
                "scope_type": scope_type,
                "scope": scope,
                "baseline_policy": baseline_row.get("policy"),
                "ai_policy": ai_row.get("policy"),
                "baseline_run_count": baseline_row.get("run_count"),
                "ai_run_count": ai_row.get("run_count"),
                "baseline_seed_count": baseline_row.get("seed_count"),
                "ai_seed_count": ai_row.get("seed_count"),
                "baseline_energy_kwh_mean": baseline_row.get("energy_kwh_mean"),
                "ai_energy_kwh_mean": ai_row.get("energy_kwh_mean"),
                "baseline_delivered_traffic_mean": baseline_row.get("delivered_traffic_mean"),
                "ai_delivered_traffic_mean": ai_row.get("delivered_traffic_mean"),
                "baseline_dropped_traffic_mean": baseline_row.get("dropped_traffic_mean"),
                "ai_dropped_traffic_mean": ai_row.get("dropped_traffic_mean"),
                "baseline_avg_delay_ms_mean": baseline_row.get("avg_delay_ms_mean"),
                "ai_avg_delay_ms_mean": ai_row.get("avg_delay_ms_mean"),
                "baseline_avg_path_latency_ms_mean": baseline_row.get("avg_path_latency_ms_mean"),
                "ai_avg_path_latency_ms_mean": ai_row.get("avg_path_latency_ms_mean"),
                "baseline_qos_violation_rate_mean": baseline_row.get("qos_violation_rate_mean"),
                "ai_qos_violation_rate_mean": ai_row.get("qos_violation_rate_mean"),
                "energy_reduction_pct_vs_baseline": ai_row.get("energy_reduction_pct_vs_baseline"),
                "delivered_traffic_change_pct_vs_baseline": ai_row.get("delivered_traffic_change_pct_vs_baseline"),
                "dropped_traffic_change_pct_vs_baseline": ai_row.get("dropped_traffic_change_pct_vs_baseline"),
                "avg_delay_ms_change_pct_vs_baseline": ai_row.get("avg_delay_ms_change_pct_vs_baseline"),
                "avg_path_latency_ms_change_pct_vs_baseline": ai_row.get("avg_path_latency_ms_change_pct_vs_baseline"),
                "qos_violation_rate_delta_vs_baseline": ai_row.get("qos_violation_rate_delta_vs_baseline"),
                "qos_acceptability_status": ai_row.get("qos_acceptability_status"),
                "hypothesis_status": ai_row.get("hypothesis_status"),
                "comparison_available": ai_row.get("comparison_available"),
                "status": "ok",
            }
        )
    return output


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value in ("", None):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "n/a"


def _fmt_delta(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "n/a"
    try:
        return f"{float(value):+.{digits}f}"
    except Exception:
        return "n/a"


def _direct_answer(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "No overall AI-vs-baseline comparison was available."

    baseline = str(row.get("baseline_policy") or "baseline")
    ai = str(row.get("ai_policy") or "ai")
    hypothesis = str(row.get("hypothesis_status") or "insufficient_data")
    qos_status = str(row.get("qos_acceptability_status") or "insufficient_data")

    if hypothesis == "achieved":
        prefix = "Yes."
    elif hypothesis == "not_achieved":
        prefix = "No."
    else:
        prefix = "Inconclusive."

    return (
        f"{prefix} {ai} vs {baseline}: energy {_fmt_pct(row.get('energy_reduction_pct_vs_baseline'))}, "
        f"delivered {_fmt_pct(row.get('delivered_traffic_change_pct_vs_baseline'))}, "
        f"dropped {_fmt_pct(row.get('dropped_traffic_change_pct_vs_baseline'))}, "
        f"delay {_fmt_pct(row.get('avg_delay_ms_change_pct_vs_baseline'))}, "
        f"path latency {_fmt_pct(row.get('avg_path_latency_ms_change_pct_vs_baseline'))}, "
        f"QoS rate delta {_fmt_delta(row.get('qos_violation_rate_delta_vs_baseline'))}, "
        f"QoS={qos_status}, hypothesis={hypothesis}."
    )


def _research_question_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Scope | Baseline | AI | Energy vs baseline | Delivered delta | Dropped delta | Delay delta | "
        "Path latency delta | QoS rate delta | QoS status | Hypothesis |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {scope} | {baseline} | {ai} | {energy} | {delivered} | {dropped} | {delay} | {path} | {qos} | {qos_status} | {hypothesis} |".format(
                scope=row.get("scope", "n/a"),
                baseline=row.get("baseline_policy", "n/a"),
                ai=row.get("ai_policy", "n/a"),
                energy=_fmt_pct(row.get("energy_reduction_pct_vs_baseline")),
                delivered=_fmt_pct(row.get("delivered_traffic_change_pct_vs_baseline")),
                dropped=_fmt_pct(row.get("dropped_traffic_change_pct_vs_baseline")),
                delay=_fmt_pct(row.get("avg_delay_ms_change_pct_vs_baseline")),
                path=_fmt_pct(row.get("avg_path_latency_ms_change_pct_vs_baseline")),
                qos=_fmt_delta(row.get("qos_violation_rate_delta_vs_baseline")),
                qos_status=row.get("qos_acceptability_status", "n/a"),
                hypothesis=row.get("hypothesis_status", "n/a"),
            )
        )
    return "\n".join(lines)


def _write_concise_report(
    *,
    report_path: Path,
    research_rows: Sequence[Mapping[str, Any]],
    final_eval_report_path: Path,
    summary_csv_path: Path,
    leaderboard_path: Path,
    plots_written: Sequence[Path],
    payload: Mapping[str, Any],
) -> None:
    overall_row = next((row for row in research_rows if row.get("scope_type") == "overall"), None)
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    classification = payload.get("classification") if isinstance(payload.get("classification"), dict) else {}

    parts = [
        "# GreenNet Final Pipeline Report",
        "",
        "## Direct Answer",
        f"- {_direct_answer(overall_row)}",
        f"- Source: `{source.get('description', summary_csv_path)}`",
        f"- Selected runs: `{source.get('selected_run_count', 'n/a')}`",
        f"- Primary baseline policy: `{classification.get('primary_baseline_policy', 'n/a')}`",
        "",
        "## Thesis Table",
        _research_question_markdown(research_rows),
        "",
        "## Bundle Files",
        f"- Summary CSV: `{summary_csv_path}`",
        f"- Leaderboard CSV: `{leaderboard_path}`",
        f"- Final evaluation report: `{final_eval_report_path}`",
    ]

    if plots_written:
        parts.extend([f"- Plot: `{path}`" for path in plots_written])
    else:
        parts.append("- Plot export: `plot-ready CSV files only (matplotlib unavailable or disabled)`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _rows_to_plot_data(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = payload.get("summary_rows")
    if not isinstance(summary_rows, list):
        raise PipelineError("final_evaluation_summary.json is missing summary_rows")

    overall_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for raw_row in summary_rows:
        if not isinstance(raw_row, dict):
            continue
        scope_type = str(raw_row.get("scope_type") or "")
        if scope_type == "overall":
            overall_rows.append(raw_row)
        elif scope_type == "scenario":
            scenario_rows.append(raw_row)
    return overall_rows, scenario_rows


def _plot_color(status: str) -> str:
    mapping = {
        "achieved": "#2e8b57",
        "not_achieved": "#c0392b",
        "acceptable": "#d4a017",
        "insufficient_data": "#7f8c8d",
    }
    return mapping.get(status, "#4c566a")


def _write_plots(
    *,
    plots_dir: Path,
    research_rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    skip_plots: bool,
) -> list[Path]:
    overall_rows, scenario_rows = _rows_to_plot_data(payload)

    overall_csv = plots_dir / "policy_tradeoff_overall.csv"
    by_scenario_csv = plots_dir / "policy_tradeoff_by_scenario.csv"
    rq_csv = plots_dir / "research_question_tradeoff.csv"
    _copy_rows_to_csv(overall_csv, overall_rows)
    _copy_rows_to_csv(by_scenario_csv, scenario_rows)
    _copy_rows_to_csv(rq_csv, research_rows)

    written = [overall_csv, by_scenario_csv, rq_csv]
    if skip_plots or plt is None:
        return written

    eligible = [
        row
        for row in research_rows
        if str(row.get("status") or "") == "ok"
        and row.get("energy_reduction_pct_vs_baseline") not in ("", None)
        and row.get("qos_violation_rate_delta_vs_baseline") not in ("", None)
    ]
    if eligible:
        scatter_path = plots_dir / "energy_vs_qos_tradeoff.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in eligible:
            hypothesis = str(row.get("hypothesis_status") or "")
            ax.scatter(
                float(row["qos_violation_rate_delta_vs_baseline"]),
                float(row["energy_reduction_pct_vs_baseline"]),
                s=80 if row.get("scope_type") == "overall" else 55,
                color=_plot_color(hypothesis),
            )
            ax.annotate(str(row.get("scope") or ""), (
                float(row["qos_violation_rate_delta_vs_baseline"]),
                float(row["energy_reduction_pct_vs_baseline"]),
            ))
        ax.axhline(0.0, color="#d0d7de", linewidth=1.0)
        ax.axvline(0.0, color="#d0d7de", linewidth=1.0)
        ax.set_xlabel("QoS violation rate delta vs baseline")
        ax.set_ylabel("Energy reduction vs baseline (%)")
        ax.set_title("Energy vs QoS tradeoff")
        fig.tight_layout()
        fig.savefig(scatter_path)
        plt.close(fig)
        written.append(scatter_path)

    eligible_scope_rows = [
        row
        for row in research_rows
        if str(row.get("status") or "") == "ok"
        and row.get("energy_reduction_pct_vs_baseline") not in ("", None)
        and row.get("qos_violation_rate_delta_vs_baseline") not in ("", None)
    ]
    if eligible_scope_rows:
        bar_path = plots_dir / "research_question_tradeoff.png"
        labels = [str(row.get("scope") or "") for row in eligible_scope_rows]
        energy_vals = [float(row["energy_reduction_pct_vs_baseline"]) for row in eligible_scope_rows]
        qos_vals = [float(row["qos_violation_rate_delta_vs_baseline"]) for row in eligible_scope_rows]
        colors = [_plot_color(str(row.get("hypothesis_status") or "")) for row in eligible_scope_rows]

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].bar(labels, energy_vals, color=colors)
        axes[0].axhline(0.0, color="#d0d7de", linewidth=1.0)
        axes[0].set_ylabel("Energy reduction (%)")
        axes[0].set_title("Best AI vs baseline by scope")

        axes[1].bar(labels, qos_vals, color=colors)
        axes[1].axhline(0.0, color="#d0d7de", linewidth=1.0)
        axes[1].set_ylabel("QoS rate delta")
        axes[1].set_xlabel("Scope")

        fig.tight_layout()
        fig.savefig(bar_path)
        plt.close(fig)
        written.append(bar_path)

    return written


def _prepare_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "logs": output_dir / "logs",
        "summary": output_dir / "summary",
        "final_eval": output_dir / "summary" / "final_evaluation",
        "plots": output_dir / "plots",
        "report": output_dir / "report",
        "metadata": output_dir / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_pipeline(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run the GreenNet final thesis pipeline: matrix evaluation, aggregation, leaderboard, final report, and plots."
    )
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional existing results_summary*.csv to package instead of scanning results.")
    parser.add_argument("--skip-eval", action="store_true", help="Do not run experiments/run_matrix.py before packaging.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_pipeline/latest"))
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG, help="Tag used for matrix runs and result selection.")
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--scenarios", type=str, default=DEFAULT_SCENARIOS)
    parser.add_argument("--policies", type=str, default=DEFAULT_POLICIES)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ppo-model", type=Path, default=None, help="Optional PPO checkpoint for policy=ppo runs.")
    parser.add_argument("--topology-seed", type=int, default=None)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--baseline-policies", type=str, default=DEFAULT_BASELINE_POLICIES)
    parser.add_argument("--ai-policies", type=str, default=DEFAULT_AI_POLICIES)
    parser.add_argument("--primary-baseline-policy", type=str, default="heuristic")
    parser.add_argument("--energy-target-pct", type=float, default=15.0)
    parser.add_argument("--max-qos-violation-rate-increase", type=float, default=0.02)
    parser.add_argument("--max-delivered-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-dropped-increase-pct", type=float, default=5.0)
    parser.add_argument("--max-delay-increase-pct", type=float, default=10.0)
    parser.add_argument("--max-path-latency-increase-pct", type=float, default=10.0)
    parser.add_argument("--skip-plots", action="store_true", help="Only emit plot-ready CSV data.")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    output_dir = _resolve_path(args.output_dir)
    results_dir = _resolve_path(args.results_dir)
    runs_dir = _resolve_path(args.runs_dir)
    summary_csv = _resolve_path(args.summary_csv) if args.summary_csv is not None else None
    ppo_model = _resolve_path(args.ppo_model) if args.ppo_model is not None else None

    scenario_filter = set(_clean_csv_list(args.scenarios))
    policy_filter = set(_clean_csv_list(args.policies))
    seed_filter = {str(seed) for seed in _parse_seed_csv(args.seeds)}
    baseline_policies = _clean_csv_list(args.baseline_policies)
    ai_policies = _clean_csv_list(args.ai_policies)

    if summary_csv is None and args.skip_eval and not args.tag:
        raise PipelineError("--skip-eval requires --tag or --summary-csv so result selection stays authoritative.")
    if summary_csv is not None and not summary_csv.exists():
        raise PipelineError(f"--summary-csv does not exist: {summary_csv}")
    if ppo_model is not None and not ppo_model.exists():
        raise PipelineError(f"--ppo-model does not exist: {ppo_model}")

    dirs = _prepare_dirs(output_dir)
    step_records: list[StepRecord] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    config_payload = {
        "generated_at_utc": timestamp,
        "repo_root": str(repo_root),
        "results_dir": str(results_dir),
        "runs_dir": str(runs_dir),
        "output_dir": str(output_dir),
        "summary_csv": None if summary_csv is None else str(summary_csv),
        "tag": args.tag,
        "skip_eval": bool(args.skip_eval),
        "seeds": sorted(seed_filter, key=lambda value: int(value)),
        "scenarios": sorted(scenario_filter),
        "policies": sorted(policy_filter),
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "ppo_model": None if ppo_model is None else str(ppo_model),
        "topology_seed": args.topology_seed,
        "deterministic": args.deterministic,
        "baseline_policies": baseline_policies,
        "ai_policies": ai_policies,
        "primary_baseline_policy": args.primary_baseline_policy,
        "thresholds": {
            "energy_target_pct": args.energy_target_pct,
            "max_qos_violation_rate_increase_abs": args.max_qos_violation_rate_increase,
            "max_delivered_loss_pct": args.max_delivered_loss_pct,
            "max_dropped_increase_pct": args.max_dropped_increase_pct,
            "max_delay_increase_pct": args.max_delay_increase_pct,
            "max_path_latency_increase_pct": args.max_path_latency_increase_pct,
        },
        "skip_plots": bool(args.skip_plots),
    }
    _write_json(dirs["metadata"] / "pipeline_config.json", config_payload)

    authoritative_summary_path = dirs["summary"] / f"results_summary_{args.tag}.csv"
    raw_summary_path = dirs["summary"] / f"results_summary_{args.tag}_raw.csv"
    by_seed_path = dirs["summary"] / f"results_summary_by_seed_{args.tag}.csv"
    leaderboard_path = dirs["summary"] / f"leaderboard_{args.tag}.csv"
    leaderboard_source_path = dirs["summary"] / f"leaderboard_source_{args.tag}.csv"
    research_question_path = dirs["summary"] / "research_question_summary.csv"
    concise_report_path = dirs["report"] / "concise_report.md"
    final_eval_paths = _final_eval_paths(dirs["final_eval"])

    if summary_csv is None and not args.skip_eval:
        cmd = [
            sys.executable,
            str(repo_root / "experiments" / "run_matrix.py"),
            "--seeds",
            args.seeds,
            "--scenarios",
            args.scenarios,
            "--policies",
            args.policies,
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.steps),
            "--out-dir",
            str(results_dir),
            "--runs-dir",
            str(runs_dir),
            "--tag",
            args.tag,
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        else:
            cmd.append("--stochastic")
        if ppo_model is not None:
            cmd.extend(["--ppo-model", str(ppo_model)])
        if args.topology_seed is not None:
            cmd.extend(["--topology-seed", str(args.topology_seed)])
        _run_command(
            name="run_matrix",
            cmd=cmd,
            log_path=dirs["logs"] / "01_run_matrix.log",
            step_records=step_records,
            outputs=[results_dir],
        )

    if summary_csv is None:
        aggregate_cmd = [
            sys.executable,
            str(repo_root / "experiments" / "aggregate_results.py"),
            "--out-dir",
            str(results_dir),
            "--output",
            str(raw_summary_path),
        ]
        if args.tag:
            aggregate_cmd.extend(["--tag", args.tag])
        _run_command(
            name="aggregate_results",
            cmd=aggregate_cmd,
            log_path=dirs["logs"] / "02_aggregate_results.log",
            step_records=step_records,
            outputs=[raw_summary_path],
        )
        source_summary = raw_summary_path
    else:
        source_summary = summary_csv

    def _filter_step() -> None:
        details = _filter_summary_csv(
            source_path=source_summary,
            output_path=authoritative_summary_path,
            seed_filter=seed_filter,
            scenario_filter=scenario_filter,
            policy_filter=policy_filter,
            deterministic_filter=args.deterministic,
        )
        log_path = dirs["logs"] / "03_filter_summary.log"
        log_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    _run_python_step(
        name="filter_summary",
        log_path=dirs["logs"] / "03_filter_summary.log",
        step_records=step_records,
        outputs=[authoritative_summary_path],
        fn=_filter_step,
        description="Filter authoritative results_summary.csv to the requested seeds/scenarios/policies.",
    )

    _run_python_step(
        name="write_by_seed_summary",
        log_path=dirs["logs"] / "04_write_by_seed_summary.log",
        step_records=step_records,
        outputs=[by_seed_path],
        fn=lambda: _write_by_seed_summary(authoritative_summary_path, by_seed_path),
        description=f"_write_by_seed({authoritative_summary_path}, {by_seed_path})",
    )

    _run_command(
        name="make_leaderboard",
        cmd=[
            sys.executable,
            str(repo_root / "experiments" / "make_leaderboard.py"),
            "--summary",
            str(authoritative_summary_path),
            "--output",
            str(leaderboard_path),
            "--output-source",
            str(leaderboard_source_path),
        ],
        log_path=dirs["logs"] / "05_make_leaderboard.log",
        step_records=step_records,
        outputs=[leaderboard_path, leaderboard_source_path],
    )

    final_eval_cmd = [
        sys.executable,
        str(repo_root / "experiments" / "final_evaluation.py"),
        "--summary-csv",
        str(authoritative_summary_path),
        "--baseline-policies",
        ",".join(baseline_policies),
        "--ai-policies",
        ",".join(ai_policies),
        "--primary-baseline-policy",
        args.primary_baseline_policy,
        "--energy-target-pct",
        str(args.energy_target_pct),
        "--max-qos-violation-rate-increase",
        str(args.max_qos_violation_rate_increase),
        "--max-delivered-loss-pct",
        str(args.max_delivered_loss_pct),
        "--max-dropped-increase-pct",
        str(args.max_dropped_increase_pct),
        "--max-delay-increase-pct",
        str(args.max_delay_increase_pct),
        "--max-path-latency-increase-pct",
        str(args.max_path_latency_increase_pct),
        "--output-dir",
        str(dirs["final_eval"]),
    ]
    _run_command(
        name="final_evaluation",
        cmd=final_eval_cmd,
        log_path=dirs["logs"] / "06_final_evaluation.log",
        step_records=step_records,
        outputs=list(final_eval_paths.values()),
    )

    final_payload = _load_json(final_eval_paths["json"])
    research_rows = _build_research_question_rows(final_payload)

    _run_python_step(
        name="write_research_question_summary",
        log_path=dirs["logs"] / "07_research_question_summary.log",
        step_records=step_records,
        outputs=[research_question_path],
        fn=lambda: _copy_rows_to_csv(research_question_path, research_rows),
        description=f"Write {research_question_path}",
    )

    plots_written: list[Path] = []

    def _plot_step() -> None:
        nonlocal plots_written
        plots_written = _write_plots(
            plots_dir=dirs["plots"],
            research_rows=research_rows,
            payload=final_payload,
            skip_plots=bool(args.skip_plots),
        )

    _run_python_step(
        name="export_plots",
        log_path=dirs["logs"] / "08_export_plots.log",
        step_records=step_records,
        outputs=[dirs["plots"]],
        fn=_plot_step,
        description="Export plot-ready CSV files and optional PNG plots.",
    )

    _run_python_step(
        name="write_concise_report",
        log_path=dirs["logs"] / "09_write_concise_report.log",
        step_records=step_records,
        outputs=[concise_report_path],
        fn=lambda: _write_concise_report(
            report_path=concise_report_path,
            research_rows=research_rows,
            final_eval_report_path=final_eval_paths["report"],
            summary_csv_path=authoritative_summary_path,
            leaderboard_path=leaderboard_path,
            plots_written=plots_written,
            payload=final_payload,
        ),
        description=f"Write {concise_report_path}",
    )

    manifest = {
        "generated_at_utc": timestamp,
        "repo_root": str(repo_root),
        "git_head": _git_head(repo_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config_payload,
        "outputs": {
            "summary_csv": str(authoritative_summary_path),
            "summary_by_seed_csv": str(by_seed_path),
            "leaderboard_csv": str(leaderboard_path),
            "leaderboard_source_csv": str(leaderboard_source_path),
            "final_evaluation_csv": str(final_eval_paths["csv"]),
            "final_evaluation_json": str(final_eval_paths["json"]),
            "final_evaluation_report": str(final_eval_paths["report"]),
            "research_question_summary_csv": str(research_question_path),
            "concise_report": str(concise_report_path),
            "plots_dir": str(dirs["plots"]),
        },
        "steps": [asdict(step) for step in step_records],
    }
    _write_json(dirs["metadata"] / "pipeline_manifest.json", manifest)

    print(f"[final_pipeline] bundle ready: {output_dir}")
    print(f"[final_pipeline] concise report: {concise_report_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    try:
        build_pipeline(argv)
    except PipelineError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
