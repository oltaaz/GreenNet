from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "artifacts/final_pipeline/official_acceptance_v1/summary/final_evaluation/final_evaluation_summary.csv"
OUTPUT_DIR = ROOT / "output/doc/thesis_figures"

POLICY_LABELS = {
    "all_on": "All-On",
    "heuristic": "Heuristic",
    "ppo": "PPO",
}

POLICY_COLORS = {
    "all_on": "#1f3c88",
    "heuristic": "#e07a1f",
    "ppo": "#2a9d8f",
}


def load_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open() as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: list[dict[str, str]], scope_type: str, scope: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["scope_type"] == scope_type and row["scope"] == scope]
    order = {"all_on": 0, "ppo": 1, "heuristic": 2}
    return sorted(selected, key=lambda row: order.get(row["policy"], 99))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save_overall_comparison(rows: list[dict[str, str]]) -> Path:
    labels = [POLICY_LABELS[row["policy"]] for row in rows]
    colors = [POLICY_COLORS[row["policy"]] for row in rows]

    metrics = [
        ("energy_kwh_mean", "Total Energy (kWh)", "{:.3f}"),
        ("delivered_traffic_mean", "Delivered Traffic", "{:.0f}"),
        ("dropped_traffic_mean", "Dropped Traffic", "{:.0f}"),
        ("avg_delay_ms_mean", "Average Delay (ms)", "{:.1f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    fig.suptitle("Official Final Benchmark: Overall Policy Comparison", fontsize=15, fontweight="bold")

    for ax, (key, title, fmt) in zip(axes.flatten(), metrics):
        values = [to_float(row, key) for row in rows]
        bars = ax.barh(labels, values, color=colors, height=0.58)
        style_axes(ax)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
                fontsize=9,
            )

    fig.text(
        0.5,
        0.02,
        "Source: official_acceptance_v1 final evaluation artifact (30 runs, 5 scenarios).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.98, 0.95))

    output = OUTPUT_DIR / "figure_overall_benchmark_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def percent_benefit(value: float, baseline: float, higher_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    if higher_is_better:
        return ((value - baseline) / baseline) * 100.0
    return ((baseline - value) / baseline) * 100.0


def save_flash_crowd_case(rows: list[dict[str, str]]) -> Path:
    by_policy = {row["policy"]: row for row in rows}
    baseline = by_policy["all_on"]

    metrics = [
        ("energy_kwh_mean", "Energy", False),
        ("delivered_traffic_mean", "Delivered", True),
        ("dropped_traffic_mean", "Dropped", False),
        ("avg_delay_ms_mean", "Delay", False),
        ("avg_path_latency_ms_mean", "Path Latency", False),
    ]

    policies = ["heuristic", "ppo"]
    x = list(range(len(metrics)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for offset, policy in enumerate(policies):
        row = by_policy[policy]
        benefits = [
            percent_benefit(to_float(row, key), to_float(baseline, key), higher_is_better)
            for key, _, higher_is_better in metrics
        ]
        positions = [idx + (offset - 0.5) * width for idx in x]
        bars = ax.bar(
            positions,
            benefits,
            width=width,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
        for bar, value in zip(bars, benefits):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.12 if value >= 0 else -0.28),
                f"{value:+.2f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in metrics], fontsize=10)
    ax.set_ylabel("Benefit vs All-On (%)")
    ax.set_title("Selected PPO Case Study: Flash Crowd Scenario", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        0.02,
        "Positive values indicate improvement relative to All-On. Source: official_acceptance_v1 scenario summary.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.95))

    output = OUTPUT_DIR / "figure_flash_crowd_case_study.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def save_scenario_tradeoff(rows: list[dict[str, str]]) -> Path:
    scenarios = ["custom", "diurnal", "flash_crowd", "hotspot", "normal"]
    ppo_rows = {row["scenario"]: row for row in rows if row["policy"] == "ppo"}

    energy = [to_float(ppo_rows[scenario], "energy_reduction_pct_vs_baseline") for scenario in scenarios]
    delivered = [to_float(ppo_rows[scenario], "delivered_traffic_change_pct_vs_baseline") for scenario in scenarios]

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True)
    fig.suptitle("PPO vs All-On by Scenario in the Official Final Benchmark", fontsize=14, fontweight="bold")

    axes[0].bar(scenarios, energy, color=POLICY_COLORS["ppo"])
    axes[0].axhline(0, color="#444444", linewidth=1.0)
    axes[0].set_ylabel("Energy Reduction (%)")
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(scenarios, delivered, color=POLICY_COLORS["ppo"])
    axes[1].axhline(0, color="#444444", linewidth=1.0)
    axes[1].set_ylabel("Delivered Change (%)")
    axes[1].set_xlabel("Scenario")
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.02,
        "Appendix-supporting view derived from official_acceptance_v1 final evaluation summary.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.98, 0.93))

    output = OUTPUT_DIR / "figure_appendix_scenario_tradeoff.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    overall_rows = select_rows(rows, "overall", "ALL")
    flash_rows = select_rows(rows, "scenario", "flash_crowd")
    scenario_rows = [row for row in rows if row["scope_type"] == "scenario"]

    outputs = [
        save_overall_comparison(overall_rows),
        save_flash_crowd_case(flash_rows),
        save_scenario_tradeoff(scenario_rows),
    ]

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
