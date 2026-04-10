#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from greennet.env import EnvConfig, GreenNetEnv
from greennet.forecasting import DemandForecastConfig, build_demand_forecaster


def _parse_csv_list(raw: str, cast) -> List[Any]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def _unpack_reset(ret: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(ret, tuple):
        if len(ret) >= 2:
            return ret[0], ret[1] if isinstance(ret[1], dict) else {}
        if len(ret) == 1:
            return ret[0], {}
    return ret, {}


def _unpack_step(ret: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(ret, tuple):
        raise RuntimeError("env.step returned non-tuple")
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
    if len(ret) == 4:
        obs, reward, done, info = ret
        return obs, float(reward), bool(done), False, (info if isinstance(info, dict) else {})
    raise RuntimeError(f"Unsupported env.step output length={len(ret)}")


def _collect_demand_series(*, scenario: str, episode_seed: int, max_steps: int, horizon_steps: int) -> List[float]:
    cfg = EnvConfig(
        max_steps=int(max_steps),
        traffic_model="stochastic",
        traffic_seed=None,
        traffic_scenario=str(scenario),
        enable_forecasting=True,
        forecast_model="ema",
        forecast_horizon_steps=int(horizon_steps),
        initial_off_edges=0,
        max_off_toggles_per_episode=0,
        max_total_toggles_per_episode=0,
        max_emergency_on_toggles_per_episode=0,
    )
    env = GreenNetEnv(cfg)
    try:
        _obs, _info = _unpack_reset(env.reset(seed=int(episode_seed)))
        series: List[float] = []
        done = False
        while not done:
            _obs, _reward, terminated, truncated, _info = _unpack_step(env.step(0))
            if env._demand_forecaster is None:
                raise RuntimeError("Demand forecaster not initialized while collecting demand series")
            series.append(float(env._demand_forecaster.last_observation))
            done = bool(terminated or truncated)
        return series
    finally:
        env.close()


def _evaluate_series(
    *,
    series: Sequence[float],
    scenario: str,
    episode_seed: int,
    forecaster_name: str,
    forecaster_cfg: DemandForecastConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    forecaster = build_demand_forecaster(forecaster_cfg)
    forecaster.reset(initial=0.0)
    horizon = max(1, int(forecaster_cfg.horizon_steps))

    prediction_rows: List[Dict[str, Any]] = []
    for step_idx, observed in enumerate(series):
        observed_value = float(observed)
        forecaster.update(observed_value)
        prediction = float(forecaster.predict())
        target_idx = int(step_idx) + int(horizon)
        if target_idx >= len(series):
            continue
        actual = float(series[target_idx])
        abs_error = abs(prediction - actual)
        squared_error = abs_error ** 2
        ape = (abs_error / abs(actual)) if actual > 1e-9 else np.nan
        prediction_rows.append(
            {
                "scenario": str(scenario),
                "episode_seed": int(episode_seed),
                "forecaster": str(forecaster_name),
                "horizon_steps": int(horizon),
                "input_step": int(step_idx + 1),
                "target_step": int(target_idx + 1),
                "observed_demand": observed_value,
                "predicted_demand": prediction,
                "actual_demand": actual,
                "abs_error": float(abs_error),
                "squared_error": float(squared_error),
                "ape": float(ape) if np.isfinite(ape) else np.nan,
            }
        )

    points = len(prediction_rows)
    if points == 0:
        raise RuntimeError(f"No forecast points were produced for scenario={scenario} seed={episode_seed}")

    errors = np.asarray([row["abs_error"] for row in prediction_rows], dtype=np.float64)
    squared_errors = np.asarray([row["squared_error"] for row in prediction_rows], dtype=np.float64)
    ape_vals = np.asarray([row["ape"] for row in prediction_rows], dtype=np.float64)
    finite_ape = np.isfinite(ape_vals)

    metrics = {
        "scenario": str(scenario),
        "episode_seed": int(episode_seed),
        "forecaster": str(forecaster_name),
        "horizon_steps": int(horizon),
        "points": int(points),
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(squared_errors.mean())),
        "mape_pct": float(np.mean(ape_vals[finite_ape]) * 100.0) if np.any(finite_ape) else None,
        "mape_points": int(np.sum(finite_ape)),
        "mean_actual_demand": float(np.mean([row["actual_demand"] for row in prediction_rows])),
        "mean_predicted_demand": float(np.mean([row["predicted_demand"] for row in prediction_rows])),
    }
    return prediction_rows, metrics


def _aggregate_summary(predictions: pd.DataFrame, *, baseline_model: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in list(sorted(predictions["scenario"].dropna().unique())) + ["all"]:
        subset = predictions if scenario == "all" else predictions[predictions["scenario"] == scenario]
        for forecaster in sorted(subset["forecaster"].dropna().unique()):
            frame = subset[subset["forecaster"] == forecaster]
            if frame.empty:
                continue
            ape = frame["ape"].to_numpy(dtype=np.float64)
            finite_ape = np.isfinite(ape)
            rows.append(
                {
                    "scenario": scenario,
                    "forecaster": forecaster,
                    "series_count": int(frame["episode_seed"].nunique()),
                    "points": int(len(frame)),
                    "horizon_steps": int(frame["horizon_steps"].max()),
                    "mae": float(frame["abs_error"].mean()),
                    "rmse": float(np.sqrt(frame["squared_error"].mean())),
                    "mape_pct": float(np.mean(ape[finite_ape]) * 100.0) if np.any(finite_ape) else np.nan,
                    "mape_points": int(np.sum(finite_ape)),
                    "mean_actual_demand": float(frame["actual_demand"].mean()),
                    "mean_predicted_demand": float(frame["predicted_demand"].mean()),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["scenario", "forecaster"]).reset_index(drop=True)
    if summary.empty:
        return summary

    baseline = summary[summary["forecaster"] == baseline_model][["scenario", "mae", "rmse", "mape_pct"]].rename(
        columns={
            "mae": "baseline_mae",
            "rmse": "baseline_rmse",
            "mape_pct": "baseline_mape_pct",
        }
    )
    summary = summary.merge(baseline, on="scenario", how="left")
    summary["mae_improvement_pct_vs_baseline"] = np.where(
        summary["baseline_mae"] > 0.0,
        100.0 * (summary["baseline_mae"] - summary["mae"]) / summary["baseline_mae"],
        np.nan,
    )
    summary["rmse_improvement_pct_vs_baseline"] = np.where(
        summary["baseline_rmse"] > 0.0,
        100.0 * (summary["baseline_rmse"] - summary["rmse"]) / summary["baseline_rmse"],
        np.nan,
    )
    valid_mape_baseline = summary["baseline_mape_pct"].notna() & (summary["baseline_mape_pct"] > 0.0)
    summary["mape_improvement_pct_vs_baseline"] = np.where(
        valid_mape_baseline,
        100.0 * (summary["baseline_mape_pct"] - summary["mape_pct"]) / summary["baseline_mape_pct"],
        np.nan,
    )
    return summary


def _write_report(
    *,
    summary: pd.DataFrame,
    series_metrics: pd.DataFrame,
    output_path: Path,
    baseline_model: str,
    improved_model: str,
) -> None:
    if summary.empty:
        report = "# Forecast Evaluation\n\nNo summary rows were produced.\n"
        output_path.write_text(report, encoding="utf-8")
        return

    lines = [
        "# Forecast Evaluation",
        "",
        f"- Baseline forecaster: `{baseline_model}`",
        f"- Improved forecaster: `{improved_model}`",
        f"- Scored metrics: `MAE`, `RMSE`, `MAPE`",
        "- MAPE ignores zero-demand targets to avoid undefined divisions.",
        "",
        "## Aggregate Summary",
        "",
        "```text",
        summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Per-Series Mean Metrics",
        "",
        "```text",
        series_metrics.sort_values(["scenario", "episode_seed", "forecaster"]).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        ),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GreenNet demand forecasters on env-generated traffic traces.")
    parser.add_argument("--scenarios", type=str, default="normal,burst,hotspot")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon-steps", type=int, default=3)
    parser.add_argument("--baseline-model", type=str, default="ema")
    parser.add_argument("--baseline-alpha", type=float, default=0.6)
    parser.add_argument("--improved-model", type=str, default="adaptive_ema")
    parser.add_argument("--improved-alpha", type=float, default=0.55)
    parser.add_argument("--improved-beta", type=float, default=0.25)
    parser.add_argument("--improved-trend-damping", type=float, default=0.9)
    parser.add_argument("--improved-adaptive-alphas", type=str, default="0.1,0.2,0.4,0.6,0.8,0.95")
    parser.add_argument("--improved-adaptive-error-alpha", type=float, default=0.02)
    parser.add_argument("--improved-adaptive-temperature", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forecasting"))
    args = parser.parse_args()

    scenarios = _parse_csv_list(args.scenarios, str)
    seeds = _parse_csv_list(args.seeds, int)
    if not scenarios:
        raise SystemExit("At least one scenario is required")
    if not seeds:
        raise SystemExit("At least one seed is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = DemandForecastConfig(
        model=str(args.baseline_model),
        alpha=float(args.baseline_alpha),
        horizon_steps=int(args.horizon_steps),
    )
    improved_cfg = DemandForecastConfig(
        model=str(args.improved_model),
        alpha=float(args.improved_alpha),
        beta=float(args.improved_beta),
        trend_damping=float(args.improved_trend_damping),
        adaptive_expert_alphas=tuple(_parse_csv_list(args.improved_adaptive_alphas, float)),
        adaptive_error_alpha=float(args.improved_adaptive_error_alpha),
        adaptive_temperature=float(args.improved_adaptive_temperature),
        horizon_steps=int(args.horizon_steps),
    )

    prediction_rows: List[Dict[str, Any]] = []
    series_metric_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        for seed in seeds:
            for episode_idx in range(int(args.episodes_per_scenario)):
                episode_seed = int(seed) + (episode_idx * 10_000)
                series = _collect_demand_series(
                    scenario=str(scenario),
                    episode_seed=int(episode_seed),
                    max_steps=int(args.max_steps),
                    horizon_steps=int(args.horizon_steps),
                )
                manifest_rows.append(
                    {
                        "scenario": str(scenario),
                        "seed": int(seed),
                        "episode_seed": int(episode_seed),
                        "steps": int(len(series)),
                        "mean_demand": float(np.mean(series)) if series else 0.0,
                        "max_demand": float(np.max(series)) if series else 0.0,
                    }
                )

                for name, cfg in (
                    (str(args.baseline_model), baseline_cfg),
                    (str(args.improved_model), improved_cfg),
                ):
                    pred_rows, metrics = _evaluate_series(
                        series=series,
                        scenario=str(scenario),
                        episode_seed=int(episode_seed),
                        forecaster_name=name,
                        forecaster_cfg=cfg,
                    )
                    prediction_rows.extend(pred_rows)
                    series_metric_rows.append(metrics)

    predictions_df = pd.DataFrame(prediction_rows)
    series_metrics_df = pd.DataFrame(series_metric_rows)
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["scenario", "episode_seed"]).reset_index(drop=True)
    summary_df = _aggregate_summary(predictions_df, baseline_model=str(args.baseline_model))

    predictions_path = output_dir / "forecast_compare_predictions.csv"
    series_metrics_path = output_dir / "forecast_compare_series_metrics.csv"
    summary_path = output_dir / "forecast_compare_summary.csv"
    manifest_path = output_dir / "forecast_compare_manifest.csv"
    json_path = output_dir / "forecast_compare_summary.json"
    report_path = output_dir / "forecast_compare_report.md"

    predictions_df.to_csv(predictions_path, index=False)
    series_metrics_df.to_csv(series_metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "baseline_model": str(args.baseline_model),
                "improved_model": str(args.improved_model),
                "scenarios": list(scenarios),
                "seeds": [int(seed) for seed in seeds],
                "episodes_per_scenario": int(args.episodes_per_scenario),
                "max_steps": int(args.max_steps),
                "horizon_steps": int(args.horizon_steps),
                "summary": summary_df.replace({np.nan: None}).to_dict(orient="records"),
                "series_metrics": series_metrics_df.replace({np.nan: None}).to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_report(
        summary=summary_df,
        series_metrics=series_metrics_df,
        output_path=report_path,
        baseline_model=str(args.baseline_model),
        improved_model=str(args.improved_model),
    )

    print(f"[forecasting] wrote {summary_path}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"[forecasting] wrote {report_path}")


if __name__ == "__main__":
    main()
