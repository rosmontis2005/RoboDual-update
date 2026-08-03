#!/usr/bin/env python3
"""Generate figures and summary tables for the 2026-05-26 meeting report."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

EXPERIMENTS = [
    ("age12", "exp0523-0428-nostradegy-maxage12", "age12 baseline"),
    ("risk_bal", "exp0524-0428-risk_balanced-maxage12", "risk balanced"),
    ("risk_cons", "exp0524-0428-risk_conservative-maxage12", "risk conservative"),
    ("risk_score", "exp0524-0428-risk_score-maxage12", "risk score"),
    ("risk_cons_diag", "exp0525-0428-risk_conservative-diagnose-maxage12", "risk cons diag"),
    ("risk_bal_start10", "exp0526-0428-risk_balanced-minage10-maxage12", "risk bal start10"),
    ("risk_cons_start10", "exp0526-0428-risk_conservative-minage10-maxage12", "risk cons start10"),
    ("task_age", "exp0526-0525-task_age", "task-age"),
]

TASK_AGE_GROUPS = {
    "A": [
        "open_drawer",
        "move_slider_right",
        "turn_on_led",
        "turn_off_led",
        "turn_on_lightbulb",
        "turn_off_lightbulb",
        "lift_red_block_table",
        "push_into_drawer",
        "push_pink_block_left",
        "rotate_blue_block_left",
        "rotate_red_block_left",
        "lift_blue_block_drawer",
        "lift_pink_block_drawer",
        "lift_red_block_drawer",
    ],
    "B": [
        "close_drawer",
        "move_slider_left",
        "place_in_drawer",
        "place_in_slider",
        "lift_pink_block_table",
        "lift_red_block_slider",
        "push_pink_block_right",
        "push_red_block_right",
        "rotate_blue_block_right",
        "rotate_pink_block_left",
        "rotate_pink_block_right",
        "rotate_red_block_right",
        "unstack_block",
    ],
    "C": [
        "lift_blue_block_slider",
        "lift_pink_block_slider",
        "lift_blue_block_table",
        "push_blue_block_left",
        "push_blue_block_right",
        "push_red_block_left",
    ],
    "D": ["stack_block"],
}
TASK_TO_GROUP = {task: group for group, tasks in TASK_AGE_GROUPS.items() for task in tasks}


def pct(num, den):
    return None if den == 0 else num / den


def pctl(values, q):
    xs = sorted(v for v in values if v is not None and not math.isnan(v) and not math.isinf(v))
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[idx]


def profile_value(step, key):
    profile = step.get("profile") or {}
    if key in profile:
        return profile.get(key)
    return step.get(key)


def rms_delta(a, b, start=0, end=6):
    if a is None or b is None:
        return None
    if len(a) < end or len(b) < end:
        return None
    vals = [float(x) - float(y) for x, y in zip(a[start:end], b[start:end])]
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def is_risk_reason(reason):
    return isinstance(reason, str) and reason.startswith("risk_") and reason != "risk_skip"


def read_result(exp_dir):
    data = json.loads((exp_dir / "result_rank0.json").read_text())
    result = data.get("null")
    if result is None:
        result = next(iter(data.values()))
    task_info = result.get("task_info", {})
    successes = sum(item["success"] for item in task_info.values())
    total = sum(item["total"] for item in task_info.values())
    return {
        "avg_seq_len": result.get("avg_seq_len"),
        "chain_sr": result.get("chain_sr", {}),
        "task_info": task_info,
        "task_success": pct(successes, total),
        "successes": successes,
        "subtasks": total,
    }


def read_profile(exp_dir):
    run_config = {}
    steps = []
    ends = []
    profile_path = exp_dir / "specialist_profile_rank0.jsonl"
    with profile_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            event = rec.get("event")
            if event == "run_config":
                run_config = rec
            elif event == "step":
                steps.append(rec)
            elif event == "subtask_end":
                ends.append(rec)
    return run_config, steps, ends


def summarize_profile(steps):
    total = len(steps)
    slow = sum(1 for s in steps if profile_value(s, "slow_system"))
    expired = sum(1 for s in steps if profile_value(s, "ref_action_expired"))
    risk = sum(
        1
        for s in steps
        if profile_value(s, "slow_system") and is_risk_reason(profile_value(s, "slow_trigger_reason"))
    )
    return {
        "steps": total,
        "slow_steps": slow,
        "slow_rate": pct(slow, total),
        "expired_steps": expired,
        "expired_rate": pct(expired, total),
        "risk_refresh_steps": risk,
        "risk_refresh_rate": pct(risk, total),
    }


def refresh_action_jumps(steps):
    buckets = defaultdict(list)
    by_key = defaultdict(list)
    for step in steps:
        key = (step.get("sequence"), step.get("subtask_i"), step.get("task"))
        by_key[key].append(step)
    for bucket in by_key.values():
        bucket.sort(key=lambda s: int(s.get("step", 0)))
        for idx, step in enumerate(bucket):
            if idx == 0 or not profile_value(step, "slow_system"):
                continue
            reason = profile_value(step, "slow_trigger_reason")
            if is_risk_reason(reason):
                kind = "risk_refresh"
            elif reason in {"max_slow_age", "task_max_slow_age"}:
                kind = "age_refresh"
            else:
                continue
            prev_action = profile_value(bucket[idx - 1], "action_prediction")
            curr_action = profile_value(step, "action_prediction")
            buckets[kind].append(rms_delta(curr_action, prev_action))
    return {
        kind: {
            "count": len(values),
            "p50": pctl(values, 0.5),
            "p95": pctl(values, 0.95),
        }
        for kind, values in buckets.items()
    }


def group_task_rates(result):
    groups = defaultdict(lambda: [0, 0])
    for task, item in result["task_info"].items():
        group = TASK_TO_GROUP.get(task, "default")
        groups[group][0] += int(item["success"])
        groups[group][1] += int(item["total"])
    return {group: pct(success, total) for group, (success, total) in groups.items()}


def collect():
    rows = []
    details = {}
    for key, dirname, label in EXPERIMENTS:
        exp_dir = ROOT / dirname
        result = read_result(exp_dir)
        run_config, steps, ends = read_profile(exp_dir)
        profile = summarize_profile(steps)
        jumps = refresh_action_jumps(steps)
        row = {
            "key": key,
            "dirname": dirname,
            "label": label,
            "strategy": run_config.get("slow_call_strategy", run_config.get("slow_trigger_policy", "")),
            "risk_start_age": run_config.get("risk_start_age"),
            "task_success": result["task_success"],
            "avg_seq_len": result["avg_seq_len"],
            "chain1": result["chain_sr"].get("1"),
            "chain2": result["chain_sr"].get("2"),
            "chain3": result["chain_sr"].get("3"),
            "chain4": result["chain_sr"].get("4"),
            "chain5": result["chain_sr"].get("5"),
            "slow_rate": profile["slow_rate"],
            "expired_rate": profile["expired_rate"],
            "risk_refresh_rate": profile["risk_refresh_rate"],
            "subtasks": result["subtasks"],
            "successes": result["successes"],
            "steps": profile["steps"],
            "risk_jump_p95": jumps.get("risk_refresh", {}).get("p95"),
            "age_jump_p95": jumps.get("age_refresh", {}).get("p95"),
        }
        rows.append(row)
        details[key] = {
            "result": result,
            "run_config": run_config,
            "profile": profile,
            "jumps": jumps,
        }
    return rows, details


def save_csv(rows):
    path = OUT / "experiment_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (OUT / "experiment_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))


def annotate_bars(ax, values, percent=True):
    for idx, value in enumerate(values):
        if value is None:
            continue
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        ax.text(idx, value, label, ha="center", va="bottom", fontsize=8, rotation=0)


def plot_chain5(rows):
    labels = [r["label"] for r in rows]
    values = [r["chain5"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(range(len(values)), [v * 100 for v in values], marker="o", linewidth=2, color="#2f6fb0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Chain@5 success (%)")
    ax.set_title("Long-horizon success across scheduler variants")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v * 100 + 0.8, f"{v * 100:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "01_chain5_by_experiment.png", dpi=180)
    plt.close(fig)


def plot_task_success(rows):
    labels = [r["label"] for r in rows]
    values = [r["task_success"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#7a7a7a"] + ["#d95f5f"] * 6 + ["#2a9d68"]
    ax.bar(range(len(values)), [v * 100 for v in values], color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Subtask success (%)")
    ax.set_title("Subtask success by experiment")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(values):
        ax.text(i, v * 100 + 0.6, f"{v * 100:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "02_task_success_by_experiment.png", dpi=180)
    plt.close(fig)


def plot_slow_rate(rows):
    labels = [r["label"] for r in rows]
    values = [r["slow_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(values)), [v * 100 for v in values], color="#5b8cbe")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Slow call steps / all steps (%)")
    ax.set_title("Slow-call cost by experiment")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(values):
        ax.text(i, v * 100 + 0.12, f"{v * 100:.2f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "03_slow_rate_by_experiment.png", dpi=180)
    plt.close(fig)


def plot_risk_jump(rows):
    selected = [r for r in rows if r["risk_jump_p95"] is not None or r["key"] == "age12"]
    labels = []
    values = []
    for row in selected:
        if row["key"] == "age12":
            labels.append("age12 age-refresh")
            values.append(row["age_jump_p95"])
        elif row["risk_jump_p95"] is not None:
            labels.append(row["label"])
            values.append(row["risk_jump_p95"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(range(len(values)), values, color="#b65f5f")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("P95 action jump, ee6 RMS")
    ax.set_title("Risk refresh happens around larger action discontinuities")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "04_risk_refresh_action_jump_p95.png", dpi=180)
    plt.close(fig)


def plot_group_delta(details):
    task_age_rates = group_task_rates(details["task_age"]["result"])
    baseline_rates = group_task_rates(details["age12"]["result"])
    groups = ["A", "B", "C", "D"]
    deltas = [(task_age_rates[g] - baseline_rates[g]) * 100 for g in groups]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#2a9d68" if d >= 0 else "#d95f5f" for d in deltas]
    ax.bar(groups, deltas, color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Success delta vs age12 (percentage points)")
    ax.set_title("Task-age group effect vs uniform max_age=12")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(deltas):
        ax.text(i, v + (0.6 if v >= 0 else -1.5), f"{v:+.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "05_task_age_group_delta.png", dpi=180)
    plt.close(fig)


def plot_task_delta(details):
    task_age = details["task_age"]["result"]["task_info"]
    age12 = details["age12"]["result"]["task_info"]
    rows = []
    for task, item in task_age.items():
        if task not in age12:
            continue
        rate = pct(item["success"], item["total"])
        base = pct(age12[task]["success"], age12[task]["total"])
        rows.append((task, (rate - base) * 100, TASK_TO_GROUP.get(task, "default")))
    selected = sorted(rows, key=lambda x: x[1])[:8] + sorted(rows, key=lambda x: x[1], reverse=True)[:8]
    selected = sorted(selected, key=lambda x: x[1])
    labels = [f"{task} ({group})" for task, _, group in selected]
    values = [delta for _, delta, _ in selected]
    fig, ax = plt.subplots(figsize=(9, 7.2))
    colors = ["#d95f5f" if v < 0 else "#2a9d68" for v in values]
    ax.barh(range(len(values)), values, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Success delta vs age12 (percentage points)")
    ax.set_title("Task-level changes under task-age scheduling")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(values):
        ax.text(v + (0.8 if v >= 0 else -0.8), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "06_task_age_task_delta.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, details = collect()
    save_csv(rows)
    plot_chain5(rows)
    plot_task_success(rows)
    plot_slow_rate(rows)
    plot_risk_jump(rows)
    plot_group_delta(details)
    plot_task_delta(details)
    print(f"Wrote meeting assets to {OUT}")
    for row in rows:
        print(
            f"{row['label']}: task_success={row['task_success'] * 100:.2f}%, "
            f"avg_seq={row['avg_seq_len']:.2f}, chain5={row['chain5'] * 100:.1f}%, "
            f"slow={row['slow_rate'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
