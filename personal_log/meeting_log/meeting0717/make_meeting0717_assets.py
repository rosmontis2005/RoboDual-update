#!/usr/bin/env python3
"""Generate figures and summary data for the 2026-07-17 meeting report.

The script intentionally separates canonical 100-sequence results from the
fixed-16 mechanism checks.  Values are loaded from committed experiment
summaries whenever available; the small offline-gain table comes from the
corresponding training summaries/logs and is recorded explicitly below.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
EVAL = REPO / "evaluation_results"

COLORS = {
    "base": "#4C78A8",
    "candidate": "#F58518",
    "safe": "#54A24B",
    "danger": "#E45756",
    "neutral": "#9D9DA1",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def canonical_results() -> list[dict]:
    v4 = load_json(EVAL / "exp0713_LoRA_v4_100seq/benchmark_summary.json")
    v7 = load_json(EVAL / "exp0716_LoRA_v7_gated_100seq/benchmark_summary.json")
    v11 = load_json(EVAL / "exp0715_LoRA_v11_gated_100seq/benchmark_summary.json")
    return [
        {
            "name": "Baseline",
            "avg": float(v4["baseline"]["avg_seq_len"]),
            "chain": list(v4["baseline"]["chain_sr"]),
        },
        {"name": "V4", "avg": float(v4["v4"]["avg_seq_len"]), "chain": list(v4["v4"]["chain_sr"])},
        {
            "name": "V7 age8",
            "avg": float(v7["candidate"]["avg_seq_len"]),
            "chain": list(v7["candidate"]["chain_sr"]),
        },
        {
            "name": "V11 age12",
            "avg": float(v11["candidate"]["avg_seq_len"]),
            "chain": list(v11["candidate"]["chain_sr"]),
        },
    ]


def plot_canonical(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={"width_ratios": [0.78, 1.55]})

    names = [item["name"] for item in results]
    avgs = [item["avg"] for item in results]
    colors = [COLORS["base"], COLORS["candidate"], COLORS["danger"], COLORS["safe"]]
    bars = axes[0].bar(names, avgs, color=colors)
    axes[0].axhline(avgs[0], color=COLORS["base"], linestyle="--", linewidth=1.3)
    axes[0].set_ylim(2.7, 3.55)
    axes[0].set_ylabel("Average completed subtasks")
    axes[0].set_title("Canonical 100-sequence average")
    axes[0].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, avgs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center")

    x = np.arange(1, 6)
    markers = ["o", "s", "^", "D"]
    for item, color, marker in zip(results, colors, markers):
        axes[1].plot(x, np.asarray(item["chain"]) * 100, marker=marker, linewidth=2, label=item["name"], color=color)
    axes[1].set_xticks(x, [f"Chain@{i}" for i in x])
    axes[1].set_ylim(25, 98)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Long-chain success profile")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Transition-LoRA canonical benchmark: later chain stages expose accumulated drift", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "01_canonical_100seq_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_v2_ablation() -> None:
    data = load_json(EVAL / "exp0713_LoRA_v_check/ablation_summary.json")["modes"]
    order = ["base", "history_only", "lora_only", "full"]
    labels = ["Base", "History only", "LoRA only", "History + LoRA"]
    avgs = [float(data[key]["avg_seq_len"]) for key in order]
    base_metrics = data["base"]["metrics"]
    metric_keys = ["action_norm_ee6", "expired_action_norm_ee6", "dp_ref_l2_ee6"]
    metric_labels = ["Action norm", "Expired-ref norm", "Slow-ref error"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = [COLORS["base"], "#72B7B2", COLORS["candidate"], COLORS["danger"]]
    bars = axes[0].bar(labels, avgs, color=colors)
    axes[0].axhline(avgs[0], color=COLORS["base"], linestyle="--", linewidth=1.2)
    axes[0].set_ylim(2.2, 3.65)
    axes[0].set_ylabel("Average completed subtasks")
    axes[0].set_title("V2 fixed-16 mechanism ablation")
    axes[0].tick_params(axis="x", rotation=16)
    for bar, value in zip(bars, avgs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}", ha="center", fontsize=9)

    x = np.arange(len(metric_keys))
    width = 0.22
    for idx, (key, label, color) in enumerate(zip(order[1:], labels[1:], colors[1:])):
        changes = []
        for metric in metric_keys:
            base = float(base_metrics[metric]["mean"])
            value = float(data[key]["metrics"][metric]["mean"])
            changes.append((value / base - 1.0) * 100)
        axes[1].bar(x + (idx - 1) * width, changes, width=width, label=label, color=color)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, metric_labels)
    axes[1].set_ylabel("Change relative to base (%)")
    axes[1].set_title("Smoother/smaller action did not mean better guidance")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "02_v2_ablation_and_action_drift.png", bbox_inches="tight")
    plt.close(fig)


def gate_result(path: str) -> dict:
    return load_json(EVAL / path)


def plot_gate_iterations() -> list[dict]:
    summaries = [
        gate_result("exp0714_LoRA_v7_gated_check/gate_summary.json"),
        gate_result("exp0714_LoRA_v8_gated_half_check_retry1/gate_summary.json"),
        gate_result("exp0715_LoRA_v9_gated_age10_check/gate_summary.json"),
        gate_result("exp0715_LoRA_v10_step500_gated_check/gate_summary.json"),
        gate_result("exp0715_LoRA_v11_gated_age12_check/gate_summary.json"),
    ]
    names = [str(item.get("version", f"V{idx + 7}")) for idx, item in enumerate(summaries)]
    avgs = [float(item["candidate_avg_seq_len"]) for item in summaries]
    baseline = float(summaries[0]["baseline_avg_seq_len"])
    coverage = []
    rows = []
    for name, item, avg in zip(names, summaries, avgs):
        steps = item["gate_steps"]
        total = sum(int(steps.get(key, 0)) for key in ("base", "transition", "expired_base"))
        ratio = int(steps["transition"]) / total if total else 0.0
        coverage.append(ratio * 100)
        rows.append(
            {
                "version": name,
                "scope": "fixed-16",
                "avg_seq_len": avg,
                "delta_vs_baseline": avg - baseline,
                "transition_coverage": ratio,
                "passed": bool(item.get("pass", False)),
            }
        )

    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(names))
    bars = ax1.bar(x, coverage, color="#B9D4E8", width=0.58, label="Transition coverage")
    ax1.set_ylabel("Transition-LoRA step coverage (%)", color="#4378A5")
    ax1.set_ylim(0, max(35, max(coverage) + 5))
    ax1.set_xticks(x, names)
    for bar, value in zip(bars, coverage):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(x, avgs, color=COLORS["danger"], marker="o", linewidth=2.2, label="Candidate average")
    ax2.axhline(baseline, color=COLORS["base"], linestyle="--", linewidth=1.4, label=f"Baseline {baseline:.3f}")
    ax2.set_ylim(2.1, 4.05)
    ax2.set_ylabel("Average completed subtasks", color=COLORS["danger"])
    for idx, value in enumerate(avgs):
        ax2.text(idx, value + 0.08, f"{value:.3f}", ha="center", fontsize=9, color=COLORS["danger"])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )
    ax1.set_title("V7-V11 fixed-16 gate iterations: lower exposure mainly restores the base policy")
    fig.tight_layout()
    fig.savefig(OUT / "03_gate_iteration_coverage.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_offline_gain() -> list[dict]:
    rows = [
        {"version": "V5", "gain": 3.63e-6, "threshold": None, "decision": "deployed; later failed full-100"},
        {"version": "V12", "gain": 2.04e-5, "threshold": 2.5e-4, "decision": "step-0 fallback"},
        {"version": "V13", "gain": 1.169e-4, "threshold": 2.0e-4, "decision": "step-0 fallback"},
    ]
    x = np.arange(len(rows))
    gains = np.asarray([item["gain"] for item in rows])
    colors = [COLORS["danger"], COLORS["neutral"], COLORS["candidate"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(x, gains, color=colors, width=0.58)
    for idx, item in enumerate(rows):
        if item["threshold"] is not None:
            ax.hlines(item["threshold"], idx - 0.34, idx + 0.34, color="black", linestyle="--", linewidth=1.4)
            ax.text(idx, item["threshold"] * 1.08, "admission threshold", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 5e-4)
    ax.set_xticks(x, [item["version"] for item in rows])
    ax.set_ylabel("Best absolute held-out diffusion-loss improvement")
    ax.set_title("Offline improvement remained too small to identify a reliable correction")
    for bar, item in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            item["gain"] * (1.18 if item["threshold"] is None else 1.08),
            f"{item['gain']:.2e}\n{item['decision']}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    fig.tight_layout()
    fig.savefig(OUT / "04_offline_gain_and_admission.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_paired_outcomes() -> None:
    v4 = load_json(EVAL / "exp0713_LoRA_v4_100seq/benchmark_summary.json")
    v7 = load_json(EVAL / "exp0716_LoRA_v7_gated_100seq/benchmark_summary.json")
    v11 = load_json(EVAL / "exp0715_LoRA_v11_gated_100seq/benchmark_summary.json")
    items = [
        ("V4", v4["paired_sequences"]),
        ("V7 age8", v7["paired_sequences"]),
        ("V11 age12", v11["paired_sequences"]),
    ]
    labels = [name for name, _ in items]
    improved = np.asarray([data["improved"] for _, data in items])
    equal = np.asarray([data["equal"] for _, data in items])
    worse = np.asarray([data["worse"] for _, data in items])
    fig, ax = plt.subplots(figsize=(8.7, 4.6))
    ax.barh(labels, improved, color=COLORS["safe"], label="Improved")
    ax.barh(labels, equal, left=improved, color="#D7D7D7", label="Equal")
    ax.barh(labels, worse, left=improved + equal, color=COLORS["danger"], label="Worse")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Canonical sequences")
    ax.set_title("Paired outcomes show high variance and little consistent positive shift")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    for idx, values in enumerate(zip(improved, equal, worse)):
        left = 0
        for value in values:
            ax.text(left + value / 2, idx, str(int(value)), ha="center", va="center", fontsize=9)
            left += value
    fig.tight_layout()
    fig.savefig(OUT / "05_paired_sequence_outcomes.png", bbox_inches="tight")
    plt.close(fig)


def save_summary(canonical: list[dict], gate_rows: list[dict], offline_rows: list[dict]) -> None:
    summary = {"canonical_100seq": canonical, "fixed16_gate_iterations": gate_rows, "offline_gain": offline_rows}
    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "experiment_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "scope", "avg_seq_len", "chain1", "chain2", "chain3", "chain4", "chain5"],
        )
        writer.writeheader()
        for item in canonical:
            writer.writerow(
                {
                    "experiment": item["name"],
                    "scope": "canonical-100",
                    "avg_seq_len": item["avg"],
                    **{f"chain{idx + 1}": value for idx, value in enumerate(item["chain"])},
                }
            )


def main() -> None:
    setup_style()
    canonical = canonical_results()
    plot_canonical(canonical)
    plot_v2_ablation()
    gate_rows = plot_gate_iterations()
    offline_rows = plot_offline_gain()
    plot_paired_outcomes()
    save_summary(canonical, gate_rows, offline_rows)


if __name__ == "__main__":
    main()
