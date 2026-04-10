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

from greennet.evaluation.acceptance_matrix import acceptance_matrix_metadata, load_acceptance_matrix
from greennet.persistence import persist_final_evaluation_bundle
from greennet.qos import QoSAcceptanceThresholds

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


DEFAULT_POLICIES = "all_on,heuristic,ppo"
DEFAULT_SCENARIOS = "normal,burst,hotspot"
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"
DEFAULT_BASELINE_POLICIES = "all_on,noop,heuristic,baseline,utilization_threshold"
DEFAULT_AI_POLICIES = "ppo"
DEFAULT_TAG = "final_pipeline"
DEFAULT_QOS_THRESHOLDS = QoSAcceptanceThresholds()


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


def _parse_bool_arg(value: str) -> bool:
    parsed = _bool_from_csv(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")
    return parsed


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
    matrix_id_filter: str | None,
    matrix_case_filter: set[str],
    deterministic_filter: bool | None,
    topology_seed_filter: str | None,
    topology_name_filter: str | None,
    topology_path_filter: str | None,
    traffic_seed_filter: str | None,
    traffic_model_filter: str | None,
    traffic_name_filter: str | None,
    traffic_path_filter: str | None,
    traffic_scenario_filter: str | None,
    traffic_scenario_version_filter: str | None,
    stability_reversal_window_steps_filter: str | None,
    stability_reversal_penalty_filter: str | None,
    stability_max_transition_rate_filter: str | None,
    stability_max_flap_rate_filter: str | None,
    stability_max_flap_count_filter: str | None,
    power_utilization_sensitive_filter: str | None,
    power_transition_on_joules_filter: str | None,
    power_transition_off_joules_filter: str | None,
) -> dict[str, Any]:
    def _selection_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        policy = str(row.get("policy") or "").strip()
        seed = str(row.get("seed") or "").strip()
        matrix_case_id = str(row.get("matrix_case_id") or "").strip()
        scenario = str(row.get("scenario") or "").strip()
        case_key = matrix_case_id if matrix_case_filter else scenario
        return (policy, case_key, seed)

    def _row_rank(row: Mapping[str, Any]) -> tuple[str, str]:
        run_id = str(row.get("run_id") or "").strip()
        results_dir = str(row.get("results_dir") or "").strip()
        return (run_id, results_dir)

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise PipelineError(f"Summary CSV has no header: {source_path}")

        kept_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        non_ok_rows: list[dict[str, str]] = []
        observed_ok: set[tuple[str, str, str]] = set()

        for row in reader:
            policy = str(row.get("policy") or "").strip()
            matrix_id = str(row.get("matrix_id") or "").strip()
            matrix_case_id = str(row.get("matrix_case_id") or "").strip()
            scenario = str(row.get("scenario") or "").strip()
            seed = str(row.get("seed") or "").strip()
            status = str(row.get("status") or "").strip().lower()
            deterministic = _bool_from_csv(row.get("deterministic"))
            topology_seed = str(row.get("topology_seed") or "").strip()
            topology_name = str(row.get("topology_name") or "").strip()
            topology_path = str(row.get("topology_path") or "").strip()
            traffic_seed = str(row.get("traffic_seed") or "").strip()
            traffic_model = str(row.get("traffic_model") or "").strip()
            traffic_name = str(row.get("traffic_name") or "").strip()
            traffic_path = str(row.get("traffic_path") or "").strip()
            traffic_scenario = str(row.get("traffic_scenario") or "").strip()
            traffic_scenario_version = str(row.get("traffic_scenario_version") or "").strip()
            stability_reversal_window_steps = str(row.get("stability_reversal_window_steps") or "").strip()
            stability_reversal_penalty = str(row.get("stability_reversal_penalty") or "").strip()
            stability_max_transition_rate = str(row.get("stability_max_transition_rate") or "").strip()
            stability_max_flap_rate = str(row.get("stability_max_flap_rate") or "").strip()
            stability_max_flap_count = str(row.get("stability_max_flap_count") or "").strip()
            power_utilization_sensitive = str(row.get("power_utilization_sensitive") or "").strip().lower()
            power_transition_on_joules = str(row.get("power_transition_on_joules") or "").strip()
            power_transition_off_joules = str(row.get("power_transition_off_joules") or "").strip()

            if matrix_id_filter is not None and matrix_id != matrix_id_filter:
                continue
            if matrix_case_filter and matrix_case_id not in matrix_case_filter:
                continue
            if policy_filter and policy not in policy_filter:
                continue
            if scenario_filter and scenario not in scenario_filter:
                continue
            if seed_filter and seed not in seed_filter:
                continue
            if deterministic_filter is not None and deterministic is not None and deterministic != deterministic_filter:
                continue
            if topology_seed_filter is not None and topology_seed != topology_seed_filter:
                continue
            if topology_name_filter is not None and topology_name != topology_name_filter:
                continue
            if topology_path_filter is not None and topology_path != topology_path_filter:
                continue
            if traffic_seed_filter is not None and traffic_seed != traffic_seed_filter:
                continue
            if traffic_model_filter is not None and traffic_model != traffic_model_filter:
                continue
            if traffic_name_filter is not None and traffic_name != traffic_name_filter:
                continue
            if traffic_path_filter is not None and traffic_path != traffic_path_filter:
                continue
            if traffic_scenario_filter is not None and traffic_scenario != traffic_scenario_filter:
                continue
            if traffic_scenario_version_filter is not None and traffic_scenario_version != traffic_scenario_version_filter:
                continue
            if (
                stability_reversal_window_steps_filter is not None
                and stability_reversal_window_steps != stability_reversal_window_steps_filter
            ):
                continue
            if stability_reversal_penalty_filter is not None and stability_reversal_penalty != stability_reversal_penalty_filter:
                continue
            if (
                stability_max_transition_rate_filter is not None
                and stability_max_transition_rate != stability_max_transition_rate_filter
            ):
                continue
            if stability_max_flap_rate_filter is not None and stability_max_flap_rate != stability_max_flap_rate_filter:
                continue
            if stability_max_flap_count_filter is not None and stability_max_flap_count != stability_max_flap_count_filter:
                continue
            if power_utilization_sensitive_filter is not None and power_utilization_sensitive != power_utilization_sensitive_filter:
                continue
            if power_transition_on_joules_filter is not None and power_transition_on_joules != power_transition_on_joules_filter:
                continue
            if power_transition_off_joules_filter is not None and power_transition_off_joules != power_transition_off_joules_filter:
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

            selected_key = _selection_key(row)
            current_row = dict(row)
            existing_row = kept_rows.get(selected_key)
            if existing_row is None or _row_rank(current_row) >= _row_rank(existing_row):
                kept_rows[selected_key] = current_row
            observed_ok.add(selected_key)

    if not kept_rows:
        raise PipelineError(
            "No successful summary rows matched the requested filters. "
            f"source={source_path}"
        )

    if matrix_case_filter:
        expected = {
            (policy, case_id, seed)
            for policy, case_id, seed in product(
                sorted(policy_filter),
                sorted(matrix_case_filter),
                sorted(seed_filter),
            )
        }
    else:
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
        if matrix_case_filter:
            preview = ", ".join(f"{policy}/{case_id}/seed={seed}" for policy, case_id, seed in missing[:10])
        else:
            preview = ", ".join(f"{policy}/{scenario}/seed={seed}" for policy, scenario, seed in missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise PipelineError(
            "The selected summary is incomplete for the requested matrix. "
            f"Missing successful rows: {preview}{suffix}"
        )

    selected_rows = [kept_rows[key] for key in sorted(kept_rows)]
    _copy_rows_to_csv(output_path, selected_rows, fieldnames=fieldnames)
    return {
        "rows_written": len(selected_rows),
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
            ai_candidates = [row for row in bucket if str(row.get("policy_class") or "") == "ai_policy"]
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
                "stability_status": ai_row.get("stability_status"),
                "stability_qualified_hypothesis_status": ai_row.get("stability_qualified_hypothesis_status"),
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
    operational = str(row.get("stability_qualified_hypothesis_status") or hypothesis)
    qos_status = str(row.get("qos_acceptability_status") or "insufficient_data")
    stability_status = str(row.get("stability_status") or "insufficient_data")

    if operational == "achieved":
        prefix = "Yes."
    elif operational == "not_achieved":
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
        f"QoS={qos_status}, stability={stability_status}, operational={operational}, hypothesis={hypothesis}."
    )


def _research_question_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Scope | Baseline | AI | Energy vs baseline | Delivered delta | Dropped delta | Delay delta | "
        "Path latency delta | QoS rate delta | QoS status | Stability | Operational | Hypothesis |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {scope} | {baseline} | {ai} | {energy} | {delivered} | {dropped} | {delay} | {path} | {qos} | {qos_status} | {stability} | {operational} | {hypothesis} |".format(
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
                stability=row.get("stability_status", "n/a"),
                operational=row.get("stability_qualified_hypothesis_status", "n/a"),
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
        f"- Acceptance matrix: `{source.get('matrix_id', 'n/a')}` ({source.get('matrix_name', 'n/a')})",
        f"- Selected runs: `{source.get('selected_run_count', 'n/a')}`",
        f"- Official traditional baseline policy: `{classification.get('official_traditional_baseline_policy', classification.get('primary_baseline_policy', 'n/a'))}`",
        f"- Strongest heuristic baseline policy: `{classification.get('strongest_heuristic_baseline_policy', 'n/a')}`",
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
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        default=None,
        help="Optional acceptance-matrix JSON manifest. When provided, it becomes the authoritative final benchmark definition.",
    )
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
    parser.add_argument("--topology-name", type=str, default=None)
    parser.add_argument("--topology-path", type=Path, default=None)
    parser.add_argument("--traffic-seed", type=int, default=None)
    parser.add_argument("--traffic-model", type=str, default=None)
    parser.add_argument("--traffic-name", type=str, default=None)
    parser.add_argument("--traffic-path", type=Path, default=None)
    parser.add_argument("--traffic-scenario", type=str, default=None)
    parser.add_argument("--traffic-scenario-version", type=int, default=None)
    parser.add_argument("--traffic-scenario-intensity", type=float, default=None)
    parser.add_argument("--traffic-scenario-duration", type=float, default=None)
    parser.add_argument("--traffic-scenario-frequency", type=float, default=None)
    parser.add_argument("--stability-reversal-window-steps", type=int, default=None)
    parser.add_argument("--stability-reversal-penalty", type=float, default=None)
    parser.add_argument("--stability-min-steps-for-assessment", type=int, default=None)
    parser.add_argument("--stability-max-transition-rate", type=float, default=None)
    parser.add_argument("--stability-max-flap-rate", type=float, default=None)
    parser.add_argument("--stability-max-flap-count", type=int, default=None)
    parser.add_argument("--power-utilization-sensitive", type=_parse_bool_arg, default=None)
    parser.add_argument("--power-transition-on-joules", type=float, default=None)
    parser.add_argument("--power-transition-off-joules", type=float, default=None)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--baseline-policies", type=str, default=DEFAULT_BASELINE_POLICIES)
    parser.add_argument("--ai-policies", type=str, default=DEFAULT_AI_POLICIES)
    parser.add_argument("--primary-baseline-policy", type=str, default="all_on")
    parser.add_argument("--energy-target-pct", type=float, default=15.0)
    parser.add_argument(
        "--max-qos-violation-rate-increase",
        type=float,
        default=DEFAULT_QOS_THRESHOLDS.max_qos_violation_rate_increase_abs,
    )
    parser.add_argument("--max-delivered-loss-pct", type=float, default=DEFAULT_QOS_THRESHOLDS.max_delivered_loss_pct)
    parser.add_argument("--max-dropped-increase-pct", type=float, default=DEFAULT_QOS_THRESHOLDS.max_dropped_increase_pct)
    parser.add_argument("--max-delay-increase-pct", type=float, default=DEFAULT_QOS_THRESHOLDS.max_delay_increase_pct)
    parser.add_argument(
        "--max-path-latency-increase-pct",
        type=float,
        default=DEFAULT_QOS_THRESHOLDS.max_path_latency_increase_pct,
    )
    parser.add_argument("--skip-plots", action="store_true", help="Only emit plot-ready CSV data.")
    args = parser.parse_args(argv)

    matrix_manifest = load_acceptance_matrix(args.matrix_manifest) if args.matrix_manifest is not None else None

    repo_root = _repo_root()
    output_dir_arg = args.output_dir
    if matrix_manifest is not None and str(output_dir_arg) == "artifacts/final_pipeline/latest":
        output_dir_arg = Path(f"artifacts/final_pipeline/{matrix_manifest.tag}")
    output_dir = _resolve_path(output_dir_arg)
    results_dir = _resolve_path(args.results_dir)
    runs_dir = _resolve_path(args.runs_dir)
    summary_csv = _resolve_path(args.summary_csv) if args.summary_csv is not None else None
    ppo_model = _resolve_path(args.ppo_model) if args.ppo_model is not None else None
    topology_path = _resolve_path(args.topology_path) if args.topology_path is not None else None
    traffic_path = _resolve_path(args.traffic_path) if args.traffic_path is not None else None

    scenario_filter = (
        {case.scenario for case in matrix_manifest.cases}
        if matrix_manifest is not None
        else set(_clean_csv_list(args.scenarios))
    )
    policy_filter = set(matrix_manifest.policies) if matrix_manifest is not None else set(_clean_csv_list(args.policies))
    seed_filter = (
        {str(seed) for seed in matrix_manifest.seeds}
        if matrix_manifest is not None
        else {str(seed) for seed in _parse_seed_csv(args.seeds)}
    )
    matrix_case_filter = (
        {case.case_id for case in matrix_manifest.cases}
        if matrix_manifest is not None
        else set()
    )
    baseline_policies = (
        list(matrix_manifest.baseline_policies)
        if matrix_manifest is not None
        else _clean_csv_list(args.baseline_policies)
    )
    ai_policies = list(matrix_manifest.ai_policies) if matrix_manifest is not None else _clean_csv_list(args.ai_policies)
    tag_value = matrix_manifest.tag if matrix_manifest is not None else args.tag

    if summary_csv is None and args.skip_eval and not tag_value:
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
        "tag": tag_value,
        "skip_eval": bool(args.skip_eval),
        "seeds": sorted(seed_filter, key=lambda value: int(value)),
        "scenarios": sorted(scenario_filter),
        "policies": sorted(policy_filter),
        "episodes": int(matrix_manifest.episodes if matrix_manifest is not None else args.episodes),
        "steps": int(matrix_manifest.steps if matrix_manifest is not None else args.steps),
        "ppo_model": None if ppo_model is None else str(ppo_model),
        "topology_seed": args.topology_seed,
        "topology_name": args.topology_name,
        "topology_path": None if topology_path is None else str(topology_path),
        "traffic_seed": args.traffic_seed,
        "traffic_model": args.traffic_model,
        "traffic_name": args.traffic_name,
        "traffic_path": None if traffic_path is None else str(traffic_path),
        "traffic_scenario": args.traffic_scenario,
        "traffic_scenario_version": args.traffic_scenario_version,
        "traffic_scenario_intensity": args.traffic_scenario_intensity,
        "traffic_scenario_duration": args.traffic_scenario_duration,
        "traffic_scenario_frequency": args.traffic_scenario_frequency,
        "stability_reversal_window_steps": args.stability_reversal_window_steps,
        "stability_reversal_penalty": args.stability_reversal_penalty,
        "stability_min_steps_for_assessment": args.stability_min_steps_for_assessment,
        "stability_max_transition_rate": args.stability_max_transition_rate,
        "stability_max_flap_rate": args.stability_max_flap_rate,
        "stability_max_flap_count": args.stability_max_flap_count,
        "power_utilization_sensitive": args.power_utilization_sensitive,
        "power_transition_on_joules": args.power_transition_on_joules,
        "power_transition_off_joules": args.power_transition_off_joules,
        "deterministic": matrix_manifest.deterministic if matrix_manifest is not None else args.deterministic,
        "baseline_policies": baseline_policies,
        "ai_policies": ai_policies,
        "primary_baseline_policy": (
            matrix_manifest.primary_baseline_policy if matrix_manifest is not None else args.primary_baseline_policy
        ),
        "thresholds": {
            "energy_target_pct": args.energy_target_pct,
            "max_qos_violation_rate_increase_abs": args.max_qos_violation_rate_increase,
            "max_delivered_loss_pct": args.max_delivered_loss_pct,
            "max_dropped_increase_pct": args.max_dropped_increase_pct,
            "max_delay_increase_pct": args.max_delay_increase_pct,
            "max_path_latency_increase_pct": args.max_path_latency_increase_pct,
        },
        "qos_thresholds": {
            "max_qos_violation_rate_increase_abs": args.max_qos_violation_rate_increase,
            "max_delivered_loss_pct": args.max_delivered_loss_pct,
            "max_dropped_increase_pct": args.max_dropped_increase_pct,
            "max_delay_increase_pct": args.max_delay_increase_pct,
            "max_path_latency_increase_pct": args.max_path_latency_increase_pct,
        },
        "skip_plots": bool(args.skip_plots),
    }
    if matrix_manifest is not None:
        config_payload["acceptance_matrix"] = acceptance_matrix_metadata(matrix_manifest)
    _write_json(dirs["metadata"] / "pipeline_config.json", config_payload)
    if matrix_manifest is not None:
        _write_json(dirs["metadata"] / "acceptance_matrix_manifest.json", json.loads(Path(matrix_manifest.manifest_path).read_text(encoding="utf-8")))

    authoritative_summary_path = dirs["summary"] / f"results_summary_{tag_value}.csv"
    raw_summary_path = dirs["summary"] / f"results_summary_{tag_value}_raw.csv"
    by_seed_path = dirs["summary"] / f"results_summary_by_seed_{tag_value}.csv"
    leaderboard_path = dirs["summary"] / f"leaderboard_{tag_value}.csv"
    leaderboard_source_path = dirs["summary"] / f"leaderboard_source_{tag_value}.csv"
    research_question_path = dirs["summary"] / "research_question_summary.csv"
    concise_report_path = dirs["report"] / "concise_report.md"
    final_eval_paths = _final_eval_paths(dirs["final_eval"])

    if summary_csv is None and not args.skip_eval:
        cmd = [
            sys.executable,
            str(repo_root / "experiments" / "run_matrix.py"),
            "--seeds",
            ",".join(sorted(seed_filter, key=lambda value: int(value))),
            "--scenarios",
            ",".join(sorted(scenario_filter)),
            "--policies",
            ",".join(sorted(policy_filter)),
            "--episodes",
            str(matrix_manifest.episodes if matrix_manifest is not None else args.episodes),
            "--steps",
            str(matrix_manifest.steps if matrix_manifest is not None else args.steps),
            "--out-dir",
            str(results_dir),
            "--runs-dir",
            str(runs_dir),
            "--tag",
            tag_value,
        ]
        if matrix_manifest is not None:
            cmd.extend(["--matrix-manifest", str(matrix_manifest.manifest_path)])
        if matrix_manifest.deterministic if matrix_manifest is not None else args.deterministic:
            cmd.append("--deterministic")
        else:
            cmd.append("--stochastic")
        if ppo_model is not None:
            cmd.extend(["--ppo-model", str(ppo_model)])
        if args.topology_seed is not None and matrix_manifest is None:
            cmd.extend(["--topology-seed", str(args.topology_seed)])
        if args.topology_name is not None and matrix_manifest is None:
            cmd.extend(["--topology-name", str(args.topology_name)])
        if topology_path is not None and matrix_manifest is None:
            cmd.extend(["--topology-path", str(topology_path)])
        if args.traffic_seed is not None:
            cmd.extend(["--traffic-seed", str(args.traffic_seed)])
        if args.traffic_model is not None:
            cmd.extend(["--traffic-model", str(args.traffic_model)])
        if args.traffic_name is not None and matrix_manifest is None:
            cmd.extend(["--traffic-name", str(args.traffic_name)])
        if traffic_path is not None and matrix_manifest is None:
            cmd.extend(["--traffic-path", str(traffic_path)])
        if args.traffic_scenario is not None:
            cmd.extend(["--traffic-scenario", str(args.traffic_scenario)])
        if args.traffic_scenario_version is not None:
            cmd.extend(["--traffic-scenario-version", str(args.traffic_scenario_version)])
        if args.traffic_scenario_intensity is not None:
            cmd.extend(["--traffic-scenario-intensity", str(args.traffic_scenario_intensity)])
        if args.traffic_scenario_duration is not None:
            cmd.extend(["--traffic-scenario-duration", str(args.traffic_scenario_duration)])
        if args.traffic_scenario_frequency is not None:
            cmd.extend(["--traffic-scenario-frequency", str(args.traffic_scenario_frequency)])
        if args.stability_reversal_window_steps is not None:
            cmd.extend(["--stability-reversal-window-steps", str(args.stability_reversal_window_steps)])
        if args.stability_reversal_penalty is not None:
            cmd.extend(["--stability-reversal-penalty", str(args.stability_reversal_penalty)])
        if args.stability_min_steps_for_assessment is not None:
            cmd.extend(["--stability-min-steps-for-assessment", str(args.stability_min_steps_for_assessment)])
        if args.stability_max_transition_rate is not None:
            cmd.extend(["--stability-max-transition-rate", str(args.stability_max_transition_rate)])
        if args.stability_max_flap_rate is not None:
            cmd.extend(["--stability-max-flap-rate", str(args.stability_max_flap_rate)])
        if args.stability_max_flap_count is not None:
            cmd.extend(["--stability-max-flap-count", str(args.stability_max_flap_count)])
        if args.power_utilization_sensitive is not None:
            cmd.extend(["--power-utilization-sensitive", str(args.power_utilization_sensitive).lower()])
        if args.power_transition_on_joules is not None:
            cmd.extend(["--power-transition-on-joules", str(args.power_transition_on_joules)])
        if args.power_transition_off_joules is not None:
            cmd.extend(["--power-transition-off-joules", str(args.power_transition_off_joules)])
        routing_baseline_value = matrix_manifest.routing_baseline if matrix_manifest is not None else None
        routing_link_cost_model_value = matrix_manifest.routing_link_cost_model if matrix_manifest is not None else None
        if routing_baseline_value:
            cmd.extend(["--routing-baseline", str(routing_baseline_value)])
        if routing_link_cost_model_value:
            cmd.extend(["--routing-link-cost-model", str(routing_link_cost_model_value)])
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
        if tag_value:
            aggregate_cmd.extend(["--tag", tag_value])
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
            matrix_id_filter=matrix_manifest.matrix_id if matrix_manifest is not None else None,
            matrix_case_filter=matrix_case_filter,
            deterministic_filter=matrix_manifest.deterministic if matrix_manifest is not None else args.deterministic,
            topology_seed_filter=None if args.topology_seed is None else str(args.topology_seed),
            topology_name_filter=str(args.topology_name).strip() if args.topology_name else None,
            topology_path_filter=None if topology_path is None else str(topology_path),
            traffic_seed_filter=None if args.traffic_seed is None else str(args.traffic_seed),
            traffic_model_filter=str(args.traffic_model).strip() if args.traffic_model else None,
            traffic_name_filter=str(args.traffic_name).strip() if args.traffic_name else None,
            traffic_path_filter=None if traffic_path is None else str(traffic_path),
            traffic_scenario_filter=str(args.traffic_scenario).strip() if args.traffic_scenario else None,
            traffic_scenario_version_filter=None if args.traffic_scenario_version is None else str(args.traffic_scenario_version),
            stability_reversal_window_steps_filter=(
                None if args.stability_reversal_window_steps is None else str(args.stability_reversal_window_steps)
            ),
            stability_reversal_penalty_filter=(
                None if args.stability_reversal_penalty is None else str(args.stability_reversal_penalty)
            ),
            stability_max_transition_rate_filter=(
                None if args.stability_max_transition_rate is None else str(args.stability_max_transition_rate)
            ),
            stability_max_flap_rate_filter=(
                None if args.stability_max_flap_rate is None else str(args.stability_max_flap_rate)
            ),
            stability_max_flap_count_filter=(
                None if args.stability_max_flap_count is None else str(args.stability_max_flap_count)
            ),
            power_utilization_sensitive_filter=(
                None if args.power_utilization_sensitive is None else str(args.power_utilization_sensitive).lower()
            ),
            power_transition_on_joules_filter=(
                None if args.power_transition_on_joules is None else str(args.power_transition_on_joules)
            ),
            power_transition_off_joules_filter=(
                None if args.power_transition_off_joules is None else str(args.power_transition_off_joules)
            ),
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
        matrix_manifest.primary_baseline_policy if matrix_manifest is not None else args.primary_baseline_policy,
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

    try:
        persist_final_evaluation_bundle(
            output_dir=dirs["final_eval"],
            payload=final_payload,
            summary_path=final_eval_paths["json"],
            report_path=final_eval_paths["report"],
            source_summary_csv=authoritative_summary_path,
        )
    except Exception:
        pass

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
