#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FINAL_EVAL_JSON = ROOT / "artifacts" / "final_evaluation" / "latest" / "final_evaluation_summary.json"
LOCKED_DIR = ROOT / "artifacts" / "locked"
OUT_DIR = ROOT / "output" / "thesis_figures_20260412"


def load_final_eval() -> dict:
    return json.loads(FINAL_EVAL_JSON.read_text())


def load_locked_row(scenario: str, off_level: str = "off4") -> dict:
    csv_path = LOCKED_DIR / scenario / "20260220_111755_100k_ctrl_cap16" / "eval_summary.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["off_level"].strip().lower() == off_level:
            return row
    raise ValueError(f"Missing {off_level} row for {scenario}")


def scenario_rows(final_eval: dict) -> list[dict]:
    rows = [
        row
        for row in final_eval["summary_rows"]
        if row["scope_type"] == "scenario" and row.get("is_best_ai_policy_for_scope")
    ]
    order = {"normal": 0, "burst": 1, "hotspot": 2}
    return sorted(rows, key=lambda row: order.get(str(row["scenario"]).lower(), 99))


def overall_rows(final_eval: dict) -> tuple[dict, dict]:
    baseline = next(row for row in final_eval["summary_rows"] if row["scope_type"] == "overall" and row["policy"] == "all_on")
    best_ai = next(row for row in final_eval["summary_rows"] if row["scope_type"] == "overall" and row.get("is_best_ai_policy_for_scope"))
    return baseline, best_ai


def pct(value: float, digits: int = 1) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def pts(value: float, digits: int = 2) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f} pts"


def draw_chip(ax, x: float, y: float, text: str, face: str = "#10352b", edge: str = "#1fd092", text_color: str = "#7ef0c0") -> None:
    ax.text(
        x,
        y,
        f"  {text}  ",
        color=text_color,
        fontsize=11,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.9", facecolor=face, edgecolor=edge, linewidth=1.1),
    )


def draw_metric_box(ax, x: float, y: float, w: float, h: float, label: str, value: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.007,rounding_size=0.02",
        facecolor="#0d1b2e",
        edgecolor="#29486d",
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(x + 0.03 * w + 0.01, y + h * 0.68, label.upper(), color="#9fb4cf", fontsize=10, va="center", ha="left")
    ax.text(x + 0.03 * w + 0.01, y + h * 0.28, value, color="#e7eef8", fontsize=18, fontweight="bold", va="center", ha="left")


def style_dark_canvas(fig, ax) -> None:
    fig.patch.set_facecolor("#061224")
    ax.set_facecolor("#061224")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def make_scenario_cards(final_eval: dict) -> Path:
    rows = scenario_rows(final_eval)
    fig, ax = plt.subplots(figsize=(20, 5.2), dpi=160)
    style_dark_canvas(fig, ax)

    ax.text(0.015, 0.95, "SCENARIO RESULTS", color="#8fa7c5", fontsize=12, va="top", ha="left")
    ax.text(0.015, 0.90, "Best AI result per scenario against the baseline", color="#f2f7fd", fontsize=20, fontweight="bold", va="top", ha="left")

    card_w = 0.31
    card_h = 0.72
    x_positions = [0.015, 0.345, 0.675]

    for x, row in zip(x_positions, rows):
        card = FancyBboxPatch(
            (x, 0.08),
            card_w,
            card_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#09182a",
            edgecolor="#29486d",
            linewidth=1.2,
        )
        ax.add_patch(card)

        ax.text(x + 0.02, 0.77, "SCENARIO", color="#9fb4cf", fontsize=12, ha="left", va="top")
        ax.text(x + 0.02, 0.72, str(row["scenario"]).capitalize(), color="#e7eef8", fontsize=20, fontweight="bold", ha="left", va="top")
        draw_chip(ax, x + 0.02, 0.63, str(row["hypothesis_status"]).replace("_", " ").upper())
        draw_chip(ax, x + 0.12, 0.63, str(row["qos_acceptability_status"]).upper())

        ax.text(
            x + 0.02,
            0.54,
            f"{str(row['policy']).capitalize()} vs All-On",
            color="#c6d3e6",
            fontsize=13,
            ha="left",
            va="center",
        )

        metrics = [
            ("Energy reduction", pct(float(row["energy_reduction_pct_vs_baseline"]))),
            ("Delivered change", pct(float(row["delivered_traffic_change_pct_vs_baseline"]))),
            ("Dropped change", pct(float(row["dropped_traffic_change_pct_vs_baseline"]))),
            ("Delay change", pct(float(row["avg_delay_ms_change_pct_vs_baseline"]))),
            ("QoS rate delta", pts(float(row["qos_violation_rate_delta_vs_baseline"]) * 100)),
            ("Run count", str(int(row["run_count"]))),
        ]
        box_w = card_w * 0.29
        box_h = 0.16
        xs = [x + 0.02, x + 0.12, x + 0.22]
        ys = [0.29, 0.12]
        for idx, (label, value) in enumerate(metrics):
            col = idx % 3
            row_idx = idx // 3
            draw_metric_box(ax, xs[col], ys[row_idx], box_w, box_h, label, value)

    out = OUT_DIR / "scenario_results_cards_generated.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_locked_cards() -> Path:
    burst = load_locked_row("burst")
    hotspot = load_locked_row("hotspot")
    rows = [burst, hotspot]
    titles = ["Burst", "Hotspot"]

    fig, ax = plt.subplots(figsize=(20, 6.5), dpi=160)
    style_dark_canvas(fig, ax)

    ax.text(0.015, 0.96, "SCENARIO ACCEPTANCE", color="#8fa7c5", fontsize=12, va="top", ha="left")
    ax.text(0.015, 0.92, "Official locked validation bundles", color="#f2f7fd", fontsize=20, fontweight="bold", va="top", ha="left")

    card_w = 0.47
    card_h = 0.80
    x_positions = [0.015, 0.515]

    for x, title, row in zip(x_positions, titles, rows):
        card = FancyBboxPatch(
            (x, 0.08),
            card_w,
            card_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#09182a",
            edgecolor="#29486d",
            linewidth=1.2,
        )
        ax.add_patch(card)

        ax.text(x + 0.02, 0.85, "OFFICIAL LOCKED RESULT", color="#9fb4cf", fontsize=12, ha="left", va="top")
        ax.text(x + 0.02, 0.80, title, color="#e7eef8", fontsize=20, fontweight="bold", ha="left", va="top")
        draw_chip(ax, x + card_w - 0.09, 0.825, "PASS")

        wrapped = textwrap.fill(
            "Locked acceptance bundle 20260220_111755_100k_ctrl_cap16 with controller-evaluated held-out traffic.",
            width=58,
        )
        ax.text(x + 0.02, 0.73, wrapped, color="#d8e3f3", fontsize=12.5, ha="left", va="top")
        for i, token in enumerate(["OFF4", "100 eps", "seeds 10-19", "cap 16"]):
            ax.text(
                x + 0.02 + i * 0.075,
                0.64,
                f" {token} ",
                color="#b6c8df",
                fontsize=10.5,
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.34,rounding_size=0.9", facecolor="#142640", edgecolor="#345173", linewidth=1.0),
            )

        metrics = [
            ("Energy vs All-On", f"{float(row['Δenergy']):+.3f} kWh"),
            ("Dropped vs All-On", f"{float(row['Δdropped']):+.1f} pkts"),
            ("Reward vs All-On", f"{float(row['Δreward']):+.1f}"),
            ("Mean ON edges", f"{float(row['on_edges_mean']):.2f}"),
            ("Toggles applied", f"{float(row['toggles_applied_mean']):.2f}"),
            ("Blocked ON actions", f"{float(row['blocked_on_actions_mean']):.2f}"),
        ]
        box_w = card_w * 0.45
        box_h = 0.14
        xs = [x + 0.02, x + 0.255]
        ys = [0.42, 0.25, 0.08]
        for idx, (label, value) in enumerate(metrics):
            col = idx % 2
            row_idx = idx // 2
            draw_metric_box(ax, xs[col], ys[row_idx], box_w, box_h, label, value)

    out = OUT_DIR / "locked_validation_cards_generated.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def style_light_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666")
    ax.spines["bottom"].set_color("#666")
    ax.tick_params(colors="#333", labelsize=11)
    ax.grid(axis="y", color="#d7dbe2", linewidth=1, alpha=0.7)


def make_energy_delivery_chart(final_eval: dict) -> Path:
    rows = scenario_rows(final_eval)
    scenarios = [str(row["scenario"]).capitalize() for row in rows]
    energy = [float(row["energy_reduction_pct_vs_baseline"]) for row in rows]
    delivered = [float(row["delivered_traffic_change_pct_vs_baseline"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    color = "#2f9e91"

    axes[0].bar(scenarios, energy, color=color)
    axes[0].set_title("Heuristic vs All-On by Scenario in the Official Final Benchmark", fontsize=19, fontweight="bold", pad=12)
    axes[0].set_ylabel("Energy Reduction (%)", fontsize=13)
    for idx, value in enumerate(energy):
        axes[0].text(idx, value + 0.35, f"{value:+.1f}%", ha="center", va="bottom", fontsize=11, color="#222")
    style_light_axes(axes[0])

    axes[1].bar(scenarios, delivered, color=color)
    axes[1].axhline(0, color="#666", linewidth=1.2)
    axes[1].set_ylabel("Delivered Change (%)", fontsize=13)
    axes[1].set_xlabel("Scenario", fontsize=13)
    for idx, value in enumerate(delivered):
        y = value - 0.15 if value < 0 else value + 0.1
        va = "top" if value < 0 else "bottom"
        axes[1].text(idx, y, f"{value:+.1f}%", ha="center", va=va, fontsize=11, color="#222")
    style_light_axes(axes[1])

    fig.text(0.5, 0.02, "Source: artifacts/final_evaluation/latest/final_evaluation_summary.json", ha="center", fontsize=11, color="#444")
    out = OUT_DIR / "benchmark_energy_delivery_by_scenario.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_qos_tradeoff_chart(final_eval: dict) -> Path:
    rows = scenario_rows(final_eval)
    scenarios = [str(row["scenario"]).capitalize() for row in rows]
    dropped = [float(row["dropped_traffic_change_pct_vs_baseline"]) for row in rows]
    qos_delta_pts = [float(row["qos_violation_rate_delta_vs_baseline"]) * 100 for row in rows]
    delay = [float(row["avg_delay_ms_change_pct_vs_baseline"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), dpi=160, sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    colors = ["#e68619", "#8b5cf6", "#d9485f"]
    series = [
        ("Dropped Change (%)", dropped, colors[0]),
        ("QoS Rate Delta (pts)", qos_delta_pts, colors[1]),
        ("Delay Change (%)", delay, colors[2]),
    ]

    axes[0].set_title("QoS Trade-Offs of Heuristic vs All-On by Scenario", fontsize=19, fontweight="bold", pad=12)
    for ax, (ylabel, values, color) in zip(axes, series):
        ax.bar(scenarios, values, color=color)
        ax.axhline(0, color="#666", linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=12.5)
        style_light_axes(ax)
        for idx, value in enumerate(values):
            offset = max(abs(value) * 0.04, 0.12)
            y = value + offset if value >= 0 else value - offset
            va = "bottom" if value >= 0 else "top"
            suffix = " pts" if "pts" in ylabel else "%"
            ax.text(idx, y, f"{value:+.2f}{suffix if suffix == ' pts' else ''}" if suffix == " pts" else f"{value:+.1f}%", ha="center", va=va, fontsize=10.5, color="#222")
    axes[-1].set_xlabel("Scenario", fontsize=13)

    fig.text(0.5, 0.015, "Positive values indicate worse QoS than All-On. Source: latest final evaluation summary.", ha="center", fontsize=11, color="#444")
    out = OUT_DIR / "benchmark_qos_tradeoffs_by_scenario.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_eval = load_final_eval()
    make_scenario_cards(final_eval)
    make_locked_cards()
    make_energy_delivery_chart(final_eval)
    make_qos_tradeoff_chart(final_eval)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
