#!/usr/bin/env python3
"""Visualize the completed 2x2 state-by-policy cross experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


CELLS = ("S8_P8", "S8_P12", "S12_P8", "S12_P12")
CELL_LABELS = {
    "S8_P8": r"$S_8, P_8$",
    "S8_P12": r"$S_8, P_{12}$",
    "S12_P8": r"$S_{12}, P_8$",
    "S12_P12": r"$S_{12}, P_{12}$",
}
CELL_COLORS = {
    "S8_P8": "#2F66A3",
    "S8_P12": "#D9772B",
    "S12_P8": "#73A9D8",
    "S12_P12": "#F0AE69",
}
REPLICATE_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
FAILURE_ORDER = ("成功", "未有效接近/接触", "接触或推动但未抬升", "抬升后仍未成功")
FAILURE_COLORS = {
    "成功": "#3A923A",
    "未有效接近/接触": "#B8B8B8",
    "接触或推动但未抬升": "#E9A23B",
    "抬升后仍未成功": "#C84C4C",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Matplotlib registers the installed Noto CJK TTC under its internal
            # JP family name; the font still contains simplified-Chinese and
            # Latin glyphs.  Using the registered name avoids a broken fallback
            # to the Chinese-only Droid font.
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, dict]], dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summaries = read_jsonl(run_dir / "cell_summaries.jsonl")
    analysis = json.loads((run_dir / "cross_analysis.json").read_text())
    expected_replicates = len(manifest["trial_seeds"])
    if len(summaries) != expected_replicates * len(CELLS):
        raise AssertionError("cell_summaries.jsonl is incomplete")
    by_rep: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in summaries:
        by_rep[int(row["replicate"])][str(row["cell"])] = row
    for replicate in range(expected_replicates):
        if set(by_rep[replicate]) != set(CELLS):
            raise AssertionError(f"Replicate {replicate} does not contain all four cells")
    if not analysis["restore_audit"]["passed"]:
        raise AssertionError("Restore audit did not pass")
    if not all(item["passed"] for item in analysis["common_random_number_audit"]):
        raise AssertionError("Common-random-number audit did not pass")
    return manifest, dict(by_rep), analysis


def load_events(run_dir: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(run_dir / "cross_events.jsonl"):
        if row.get("event") == "step":
            grouped[(int(row["replicate"]), str(row["cell"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return dict(grouped)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", facecolor="white")
    plt.close(fig)


def cell_success_values(by_rep: dict[int, dict[str, dict]], cell: str) -> list[float]:
    return [float(bool(by_rep[index][cell]["success"])) for index in sorted(by_rep)]


def plot_overview(by_rep: dict[int, dict[str, dict]], output_dir: Path) -> None:
    replicates = sorted(by_rep)
    rates = [np.mean(cell_success_values(by_rep, cell)) for cell in CELLS]
    fig = plt.figure(figsize=(14.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.35))

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(CELLS))
    ax.bar(x, rates, color=[CELL_COLORS[cell] for cell in CELLS], width=0.68)
    for index, (cell, rate) in enumerate(zip(CELLS, rates)):
        values = cell_success_values(by_rep, cell)
        jitter = np.linspace(-0.17, 0.17, len(values))
        ax.scatter(index + jitter, values, s=35, color="white", edgecolor="#222222", zorder=3)
        ax.text(index, rate + 0.055, f"{int(sum(values))}/{len(values)}", ha="center", weight="bold")
    ax.set_xticks(x, [CELL_LABELS[cell] for cell in CELLS])
    ax.set_ylim(-0.08, 1.12)
    ax.set_ylabel("第五任务成功率")
    ax.set_title("A  四个交叉条件的成功率")
    ax.grid(axis="y", alpha=0.2)

    ax = fig.add_subplot(grid[0, 1])
    matrix = np.array(
        [[int(bool(by_rep[replicate][cell]["success"])) for cell in CELLS] for replicate in replicates]
    )
    ax.imshow(matrix, cmap=ListedColormap(["#D95F59", "#53A451"]), vmin=0, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, "成功" if matrix[row, column] else "失败", ha="center", va="center", color="white", weight="bold")
    ax.set_xticks(np.arange(len(CELLS)), [CELL_LABELS[cell] for cell in CELLS])
    ax.set_yticks(np.arange(len(replicates)), [f"重复 {item}" for item in replicates])
    ax.set_title("B  每次配对重复的原始结果（不隐藏波动）")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")

    def paired_panel(axis: plt.Axes, factor: str) -> None:
        if factor == "policy":
            labels = [r"fixed-8  $P_8$", r"age-12  $P_{12}$"]
            pairs = (("S8_P8", "S12_P8"), ("S8_P12", "S12_P12"))
            title = "C  每次重复内的策略主效应"
            colors = ("#2F66A3", "#D9772B")
        else:
            labels = [r"状态 $S_8$", r"状态 $S_{12}$"]
            pairs = (("S8_P8", "S8_P12"), ("S12_P8", "S12_P12"))
            title = "D  每次重复内的初始状态主效应"
            colors = ("#696969", "#4A9A8A")
        values = np.array(
            [
                [np.mean([float(bool(by_rep[rep][cell]["success"])) for cell in pair]) for pair in pairs]
                for rep in replicates
            ]
        )
        for row, color in zip(values, REPLICATE_COLORS):
            axis.plot((0, 1), row, marker="o", color=color, alpha=0.78, linewidth=1.4)
        means = values.mean(axis=0)
        axis.plot((0, 1), means, color="#111111", marker="D", markersize=8, linewidth=3, zorder=5)
        axis.scatter((0, 1), means, s=90, color=colors, edgecolor="#111111", zorder=6)
        axis.text(0.5, 1.04, f"平均变化 {100 * (means[1] - means[0]):+.0f} 个百分点", ha="center", weight="bold")
        axis.set_xticks((0, 1), labels)
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylim(-0.08, 1.14)
        axis.set_ylabel("在两个另一因素水平上的平均成功率")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)

    paired_panel(fig.add_subplot(grid[1, 0]), "policy")
    paired_panel(fig.add_subplot(grid[1, 1]), "state")
    fig.suptitle("第五任务2×2交叉实验：策略差异大于初始状态差异", fontsize=17, weight="bold")
    save_figure(fig, output_dir, "01_factorial_success_overview")


def plot_effects(analysis: dict, output_dir: Path) -> None:
    contrast_keys = (
        "averaged_policy_main_effect__P12_minus_P8",
        "averaged_state_main_effect__S12_minus_S8",
        "interaction__policy_effect_S12_minus_policy_effect_S8",
    )
    labels = ("策略主效应\n$P_{12}-P_8$", "状态主效应\n$S_{12}-S_8$", "交互效应\nDiD")
    colors = ("#D9772B", "#4A9A8A", "#8C6BB1")
    panels = (
        ("success", 100.0, "成功率变化（百分点）", "越高越好"),
        ("performance_score", 1.0, "performance score变化", "越高越好"),
        ("pink_lift_max", 1000.0, "最大抬升变化（mm）", "越高越好"),
        ("min_tcp_pink_distance", 1000.0, "最小TCP-方块距离变化（mm）", "越低越好"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    for axis, (metric, scale, ylabel, direction) in zip(axes.flat, panels):
        for index, (key, color) in enumerate(zip(contrast_keys, colors)):
            record = analysis["paired_contrasts"][key][metric]
            values = np.asarray(record["values"], dtype=float) * scale
            jitter = np.linspace(-0.12, 0.12, len(values))
            axis.scatter(index + jitter, values, s=43, color=color, alpha=0.72, edgecolor="white", linewidth=0.6)
            mean = float(record["mean"]) * scale
            sd = float(record["sd"]) * scale
            axis.errorbar(index, mean, yerr=sd, fmt="D", color="#111111", markerfacecolor=color, capsize=5, markersize=7, linewidth=1.8, zorder=5)
            axis.annotate(f"{mean:+.1f}", (index, mean), xytext=(8, 5), textcoords="offset points", fontsize=9, weight="bold")
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel}（{direction}）")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("配对效应：均值菱形与±1 SD；圆点为5次重复", fontsize=16, weight="bold")
    save_figure(fig, output_dir, "02_paired_main_and_interaction_effects")


def extract_dynamics(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    initial_pink = np.asarray(rows[0]["pre_physics"]["scene"]["movable_objects"]["block_pink"]["current_pos"], dtype=float)
    distances = []
    lifts = []
    for row in rows:
        physics = row["post_physics"]
        pink = np.asarray(physics["scene"]["movable_objects"]["block_pink"]["current_pos"], dtype=float)
        tcp = np.asarray(physics["tcp"]["position"], dtype=float)
        distances.append(100.0 * np.linalg.norm(tcp - pink))
        lifts.append(100.0 * (pink[2] - initial_pink[2]))
    return np.asarray(distances), np.asarray(lifts)


def plot_dynamics(events: dict[tuple[int, str], list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True, constrained_layout=True)
    for column, cell in enumerate(CELLS):
        policy = "P8" if cell.endswith("P8") else "P12"
        # Match the live evaluators exactly: the legacy fixed-mod-8 sequence is
        # 0, 7, 15, ...; fixed-age-12 is 0, 12, 24, ....
        slow_steps = ([0] + list(range(7, 241, 8))) if policy == "P8" else range(0, 241, 12)
        for row_index in range(2):
            for slow_step in slow_steps:
                axes[row_index, column].axvline(slow_step, color="#888888", alpha=0.055, linewidth=0.8)
        for replicate, color in enumerate(REPLICATE_COLORS):
            rows = events[(replicate, cell)]
            distances, lifts = extract_dynamics(rows)
            success = bool(rows[-1]["task_success"])
            style = "-" if success else "--"
            label = f"重复{replicate} {'成功' if success else '失败'}"
            axes[0, column].plot(np.arange(len(distances)), distances, style, color=color, linewidth=1.45, alpha=0.85, label=label)
            axes[1, column].plot(np.arange(len(lifts)), lifts, style, color=color, linewidth=1.45, alpha=0.85)
        axes[0, column].axhline(5.0, color="#C84C4C", linestyle=":", linewidth=1.2)
        axes[1, column].axhline(1.0, color="#3A923A", linestyle=":", linewidth=1.2)
        axes[0, column].set_title(CELL_LABELS[cell])
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].set_xlabel("第五任务环境步")
        axes[0, column].grid(alpha=0.15)
        axes[1, column].grid(alpha=0.15)
    axes[0, 0].set_ylabel("TCP到粉色方块距离（cm）")
    axes[1, 0].set_ylabel("粉色方块相对抬升（cm）")
    axes[0, -1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        "逐步动力学：实线为成功，虚线为失败\n"
        "红色横线：5 cm接近阈值；绿色横线：1 cm抬升阈值；淡竖线：对应策略slow-call时刻",
        fontsize=15,
        weight="bold",
    )
    save_figure(fig, output_dir, "03_tcp_distance_and_block_lift_dynamics")


def classify_failure(rows: list[dict[str, Any]]) -> str:
    if bool(rows[-1]["task_success"]):
        return "成功"
    initial_pink = np.asarray(rows[0]["pre_physics"]["scene"]["movable_objects"]["block_pink"]["current_pos"], dtype=float)
    positions = np.asarray(
        [row["post_physics"]["scene"]["movable_objects"]["block_pink"]["current_pos"] for row in rows],
        dtype=float,
    )
    max_lift = float(np.max(positions[:, 2] - initial_pink[2]))
    max_move = float(np.max(np.linalg.norm(positions - initial_pink, axis=1)))
    if max_lift >= 0.01:
        return "抬升后仍未成功"
    if max_move >= 0.005:
        return "接触或推动但未抬升"
    return "未有效接近/接触"


def stacked_failure_axis(axis: plt.Axes, groups: list[tuple[str, list[tuple[int, str]]]], events: dict) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    bottom = np.zeros(len(groups), dtype=float)
    for category in FAILURE_ORDER:
        values = []
        for label, keys in groups:
            counts.setdefault(label, {item: 0 for item in FAILURE_ORDER})
            value = sum(classify_failure(events[key]) == category for key in keys)
            counts[label][category] = value
            values.append(value)
        axis.bar(np.arange(len(groups)), values, bottom=bottom, color=FAILURE_COLORS[category], label=category, width=0.68)
        for index, (value, start) in enumerate(zip(values, bottom)):
            if value:
                axis.text(index, start + value / 2, str(value), ha="center", va="center", color="white" if category != "未有效接近/接触" else "#333333", weight="bold")
        bottom += np.asarray(values)
    axis.set_xticks(np.arange(len(groups)), [label for label, _ in groups])
    axis.set_ylabel("运行次数")
    axis.set_ylim(0, max(bottom) + 0.8)
    axis.grid(axis="y", alpha=0.15)
    return counts


def plot_failures(events: dict[tuple[int, str], list[dict[str, Any]]], output_dir: Path) -> dict[str, dict[str, int]]:
    replicates = sorted({key[0] for key in events})
    cell_groups = [(CELL_LABELS[cell], [(rep, cell) for rep in replicates]) for cell in CELLS]
    pooled_groups = [
        (r"fixed-8  $P_8$", [(rep, cell) for rep in replicates for cell in ("S8_P8", "S12_P8")]),
        (r"age-12  $P_{12}$", [(rep, cell) for rep in replicates for cell in ("S8_P12", "S12_P12")]),
        (r"状态 $S_8$", [(rep, cell) for rep in replicates for cell in ("S8_P8", "S8_P12")]),
        (r"状态 $S_{12}$", [(rep, cell) for rep in replicates for cell in ("S12_P8", "S12_P12")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    counts = stacked_failure_axis(axes[0], cell_groups, events)
    pooled_counts = stacked_failure_axis(axes[1], pooled_groups, events)
    axes[0].set_title("A  四个交叉条件")
    axes[1].set_title("B  分别按策略和初始状态汇总")
    axes[1].axvline(1.5, color="#555555", linewidth=1)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("失败阶段分解：age-12增加的不只是单一类型失败", fontsize=16, weight="bold")
    save_figure(fig, output_dir, "04_failure_stage_decomposition")
    return {**counts, **pooled_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "figures")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    manifest, by_rep, analysis = load_run(run_dir)
    events = load_events(run_dir)
    expected_keys = {(rep, cell) for rep in by_rep for cell in CELLS}
    if set(events) != expected_keys:
        raise AssertionError("cross_events.jsonl does not contain exactly the expected 20 trajectories")

    plot_overview(by_rep, output_dir)
    plot_effects(analysis, output_dir)
    plot_dynamics(events, output_dir)
    failure_counts = plot_failures(events, output_dir)
    summary = {
        "run_dir": str(run_dir),
        "replicates": len(by_rep),
        "trial_seeds": manifest["trial_seeds"],
        "restore_audit_passed": analysis["restore_audit"]["passed"],
        "common_random_number_audit_passed": all(item["passed"] for item in analysis["common_random_number_audit"]),
        "failure_classification_thresholds": {"minimum_block_motion_m": 0.005, "minimum_block_lift_m": 0.01},
        "failure_counts": failure_counts,
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
    }
    (output_dir / "visualization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
