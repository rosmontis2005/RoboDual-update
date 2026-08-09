#!/usr/bin/env python3
"""Validate and summarize the four-condition mechanism diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mechanism_common import CONDITIONS, CONDITION_FACTORS, condition_effects, json_safe, rms


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def aggregate(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "n": len(clean),
        "mean": None if not clean else float(np.mean(clean)),
        "sd": sample_sd(clean),
        "values": clean,
    }


def validate_events(manifest: dict[str, Any], observations: list[dict], events: list[dict]) -> dict:
    expected = set(CONDITIONS)
    by_observation: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_observation[str(event["observation_id"])].append(event)
        condition = event.get("condition")
        if condition not in expected:
            raise AssertionError(f"Unknown condition {condition!r}")
        factors = CONDITION_FACTORS[condition]
        factor_keys = {
            "hidden_source": "hidden",
            "ref_source": "ref",
            "intervention": "intervention",
        }
        for event_key, factor_key in factor_keys.items():
            if event.get(event_key) != factors[factor_key]:
                raise AssertionError(
                    f"Condition metadata mismatch for {condition}: "
                    f"{event_key}={event.get(event_key)!r}"
                )
    observation_ids = {str(item["observation_id"]) for item in observations}
    if set(by_observation) != observation_ids:
        raise AssertionError(
            f"Observation/event mismatch: observations={len(observation_ids)} events={len(by_observation)}"
        )
    noise_mismatch = []
    duplicate_observations = []
    for observation_id, rows in by_observation.items():
        conditions = [row["condition"] for row in rows]
        if set(conditions) != expected or len(conditions) != len(expected):
            raise AssertionError(f"{observation_id} does not have exactly four conditions: {conditions}")
        hashes = {row["initial_noise_sha256"] for row in rows}
        if len(hashes) != 1:
            noise_mismatch.append({"observation_id": observation_id, "hashes": sorted(hashes)})
        if len(conditions) != len(set(conditions)):
            duplicate_observations.append(observation_id)
    if noise_mismatch or duplicate_observations:
        raise AssertionError(
            f"Invalid paired diagnostic: noise_mismatch={noise_mismatch}, duplicates={duplicate_observations}"
        )
    expected_observations = int(manifest.get("expected_observations", len(observations)))
    expected_events = int(
        manifest.get("expected_condition_events", expected_observations * len(CONDITIONS))
    )
    if len(observations) != expected_observations:
        raise AssertionError(
            f"Incomplete observations: expected {expected_observations}, got {len(observations)}"
        )
    if len(events) != expected_events:
        raise AssertionError(f"Incomplete events: expected {expected_events}, got {len(events)}")
    return {
        "observations": len(observations),
        "events": len(events),
        "expected_observations": expected_observations,
        "expected_events": expected_events,
        "noise_hashes_verified": len(noise_mismatch) == 0,
        "condition_set": list(CONDITIONS),
    }


def action_metrics(event: dict, baseline: np.ndarray) -> dict[str, Any]:
    first = np.asarray(event["first_action"], dtype=np.float64)
    delta = first - baseline
    return {
        "first_action_delta_rms": rms(delta),
        "first_action_delta_ee6_rms": rms(delta[:6]),
        "first_action_delta_gripper_abs": float(abs(delta[6])),
        "gripper_sign_changed": int(np.sign(first[6]) != np.sign(baseline[6])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...], value_keys: tuple[str, ...]) -> dict:
    buckets: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        for key in value_keys:
            buckets[group][key].append(float(row[key]))
    result = {}
    for group, values in sorted(buckets.items(), key=lambda item: item[0]):
        label = "|".join(str(item) for item in group)
        result[label] = {key: aggregate(value) for key, value in values.items()}
    return result


def make_plot(run_dir: Path, condition_rows: list[dict], contrast_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ages = sorted({int(row["age"]) for row in condition_rows})
    colors = {
        "fresh_hidden_empty_ref": "#3B82F6",
        "stale_hidden_fresh_ref": "#F59E0B",
        "fresh_hidden_fresh_ref": "#10B981",
    }
    labels = {
        "fresh_hidden_empty_ref": "fresh H / empty R",
        "stale_hidden_fresh_ref": "stale H / fresh R",
        "fresh_hidden_fresh_ref": "fresh H / fresh R",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for condition in colors:
        values = []
        for age in ages:
            values.append(
                [
                    float(row["first_action_delta_ee6_rms"])
                    for row in condition_rows
                    if row["condition"] == condition and int(row["age"]) == age
                ]
            )
        means = [float(np.mean(value)) if value else np.nan for value in values]
        axes[0].plot(ages, means, marker="o", color=colors[condition], label=labels[condition])
    axes[0].axvline(8, color="#888888", linestyle=":", linewidth=1)
    axes[0].set_title("Relative to P12: first-action EE6 delta")
    axes[0].set_xlabel("P12 age")
    axes[0].set_ylabel("RMS action delta")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    contrast_colors = {
        "hidden_effect_at_empty_ref": "#2563EB",
        "ref_effect_at_stale_hidden": "#D97706",
        "hidden_ref_interaction": "#059669",
    }
    contrast_labels = {
        "hidden_effect_at_empty_ref": "hidden effect | empty R",
        "ref_effect_at_stale_hidden": "ref effect | stale H",
        "hidden_ref_interaction": "H×R interaction",
    }
    for effect, color in contrast_colors.items():
        means = []
        for age in ages:
            values = [
                float(row["effect_ee6_rms"])
                for row in contrast_rows
                if row["effect"] == effect and int(row["age"]) == age
            ]
            means.append(float(np.mean(values)) if values else np.nan)
        axes[1].plot(ages, means, marker="o", color=color, label=contrast_labels[effect])
    axes[1].axvline(8, color="#888888", linestyle=":", linewidth=1)
    axes[1].set_title("Paired channel contrasts")
    axes[1].set_xlabel("P12 age")
    axes[1].set_ylabel("RMS effect on first action (EE6)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(run_dir / "mechanism_first_action_attribution.png", dpi=180)
    fig.savefig(run_dir / "mechanism_first_action_attribution.svg")
    plt.close(fig)


def seed_cluster_mean_ci(
    rows: list[dict[str, Any]],
    effect: str,
    *,
    age: int | None = None,
    bootstrap_seed: int = 809,
    bootstrap_samples: int = 10000,
) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling trial seeds as independent clusters."""

    selected = [
        row
        for row in rows
        if row["effect"] == effect and (age is None or int(row["age"]) == int(age))
    ]
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        by_seed[int(row["trial_seed"])].append(float(row["effect_ee6_rms"]))
    seed_means = np.asarray(
        [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=np.float64
    )
    if not len(seed_means):
        raise ValueError(f"No rows for effect={effect!r}, age={age!r}")
    rng = np.random.default_rng(bootstrap_seed + (0 if age is None else int(age)))
    indices = rng.integers(0, len(seed_means), size=(bootstrap_samples, len(seed_means)))
    boot_means = seed_means[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(seed_means.mean()), float(low), float(high)


def paired_seed_cluster_ci(
    rows: list[dict[str, Any]],
    minuend: str,
    subtrahend: str,
    *,
    age_a: int | None = None,
    age_b: int | None = None,
    bootstrap_seed: int = 1809,
    bootstrap_samples: int = 10000,
) -> tuple[float, float, float]:
    """Cluster-bootstrap a paired effect or an age contrast."""

    lookup = {
        (str(row["observation_id"]), str(row["effect"])): float(row["effect_ee6_rms"])
        for row in rows
    }
    by_seed: dict[int, list[float]] = defaultdict(list)
    if age_a is None and age_b is None:
        for row in rows:
            if row["effect"] != minuend:
                continue
            key = (str(row["observation_id"]), subtrahend)
            by_seed[int(row["trial_seed"])].append(float(row["effect_ee6_rms"]) - lookup[key])
    else:
        if age_a is None or age_b is None or minuend != subtrahend:
            raise ValueError("Age contrast requires one effect and both ages")
        age_lookup = {
            (int(row["trial_seed"]), str(row["state"]), int(row["age"]), str(row["effect"])): float(
                row["effect_ee6_rms"]
            )
            for row in rows
        }
        seeds = sorted({int(row["trial_seed"]) for row in rows})
        states = sorted({str(row["state"]) for row in rows})
        for seed in seeds:
            for state in states:
                by_seed[seed].append(
                    age_lookup[(seed, state, age_a, minuend)]
                    - age_lookup[(seed, state, age_b, minuend)]
                )
    seed_means = np.asarray(
        [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=np.float64
    )
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seed_means), size=(bootstrap_samples, len(seed_means)))
    boot_means = seed_means[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(seed_means.mean()), float(low), float(high)


def make_key_conclusion_plot(run_dir: Path, contrast_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    context_effects = (
        ("hidden_effect_at_empty_ref", "Refresh hidden\n(empty ref)", "#2563EB"),
        ("hidden_effect_at_fresh_ref", "Refresh hidden\n(fresh ref)", "#93C5FD"),
        ("ref_effect_at_stale_hidden", "Add fresh ref\n(stale hidden)", "#D97706"),
        ("ref_effect_at_fresh_hidden", "Add fresh ref\n(fresh hidden)", "#FBBF24"),
    )
    age_effects = (
        ("hidden_effect_at_empty_ref", "Hidden refresh | empty ref", "#2563EB", "o"),
        ("ref_effect_at_stale_hidden", "Fresh ref | stale hidden", "#D97706", "s"),
        ("hidden_ref_interaction", "H×R interaction", "#059669", "^"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    x_positions = np.asarray([0.0, 1.0, 2.6, 3.6])
    for x, (effect, label, color) in zip(x_positions, context_effects):
        mean, low, high = seed_cluster_mean_ci(contrast_rows, effect)
        axes[0].bar(x, mean, width=0.72, color=color, alpha=0.88)
        axes[0].errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            color="#111827",
            capsize=4,
            linewidth=1.3,
        )
        axes[0].text(x, mean + 0.012, f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x_positions, [item[1] for item in context_effects], fontsize=9)
    axes[0].set_title("A. Channel effects depend strongly on the other channel")
    axes[0].set_ylabel("First-action EE6 RMS effect")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)

    ages = [8, 9, 10, 11]
    for effect, label, color, marker in age_effects:
        estimates = [seed_cluster_mean_ci(contrast_rows, effect, age=age) for age in ages]
        means = np.asarray([item[0] for item in estimates])
        lows = np.asarray([item[1] for item in estimates])
        highs = np.asarray([item[2] for item in estimates])
        axes[1].plot(ages, means, color=color, marker=marker, linewidth=2, label=label)
        axes[1].fill_between(ages, lows, highs, color=color, alpha=0.14)
    axes[1].set_xticks(ages)
    axes[1].set_title("B. Stale-hidden effect grows; empty-ref effect stays flat")
    axes[1].set_xlabel("P12 age (steps since slow refresh)")
    axes[1].set_ylabel("First-action EE6 RMS effect")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Seq060 / subtask 5 mechanism diagnostic (80 paired observations)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(run_dir / "mechanism_key_conclusion.png", dpi=220)
    fig.savefig(run_dir / "mechanism_key_conclusion.svg")
    plt.close(fig)


def write_brief_report(
    run_dir: Path,
    validation: dict[str, Any],
    condition_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
) -> None:
    effects = {
        effect: seed_cluster_mean_ci(contrast_rows, effect)
        for effect in (
            "hidden_effect_at_empty_ref",
            "hidden_effect_at_fresh_ref",
            "ref_effect_at_stale_hidden",
            "ref_effect_at_fresh_hidden",
            "hidden_ref_interaction",
        )
    }
    hidden_context = paired_seed_cluster_ci(
        contrast_rows, "hidden_effect_at_fresh_ref", "hidden_effect_at_empty_ref"
    )
    ref_context = paired_seed_cluster_ci(
        contrast_rows, "ref_effect_at_fresh_hidden", "ref_effect_at_stale_hidden"
    )
    hidden_age = paired_seed_cluster_ci(
        contrast_rows,
        "hidden_effect_at_empty_ref",
        "hidden_effect_at_empty_ref",
        age_a=11,
        age_b=8,
    )
    ref_age = paired_seed_cluster_ci(
        contrast_rows,
        "ref_effect_at_stale_hidden",
        "ref_effect_at_stale_hidden",
        age_a=11,
        age_b=8,
    )
    age_values = {
        effect: {
            age: seed_cluster_mean_ci(contrast_rows, effect, age=age)[0]
            for age in (8, 9, 10, 11)
        }
        for effect in ("hidden_effect_at_empty_ref", "ref_effect_at_stale_hidden")
    }
    state_values = {}
    for state in ("S8", "S12"):
        state_values[state] = {}
        for effect in ("hidden_effect_at_empty_ref", "ref_effect_at_stale_hidden"):
            values = [
                float(row["effect_ee6_rms"])
                for row in contrast_rows
                if row["state"] == state and row["effect"] == effect
            ]
            state_values[state][effect] = float(np.mean(values))
    gripper_flips = {}
    for condition in (
        "fresh_hidden_empty_ref",
        "stale_hidden_fresh_ref",
        "fresh_hidden_fresh_ref",
    ):
        rows = [row for row in condition_rows if row["condition"] == condition]
        gripper_flips[condition] = int(sum(int(row["gripper_sign_changed"]) for row in rows))

    findings = {
        "validation": validation,
        "n_trial_seeds": len({int(row["trial_seed"]) for row in contrast_rows}),
        "n_boundaries": len({str(row["state"]) for row in contrast_rows}),
        "ages": sorted({int(row["age"]) for row in contrast_rows}),
        "effects_mean_ci95_seed_cluster_bootstrap": effects,
        "context_contrasts_mean_ci95": {
            "hidden_effect_fresh_ref_minus_empty_ref": hidden_context,
            "ref_effect_fresh_hidden_minus_stale_hidden": ref_context,
        },
        "age11_minus_age8_mean_ci95": {
            "hidden_effect_at_empty_ref": hidden_age,
            "ref_effect_at_stale_hidden": ref_age,
        },
        "age_means": age_values,
        "state_means": state_values,
        "gripper_sign_changes_vs_p12_out_of_80": gripper_flips,
    }
    (run_dir / "mechanism_key_findings.json").write_text(
        json.dumps(json_safe(findings), indent=2) + "\n"
    )

    h_empty = effects["hidden_effect_at_empty_ref"][0]
    h_fresh = effects["hidden_effect_at_fresh_ref"][0]
    r_stale = effects["ref_effect_at_stale_hidden"][0]
    r_fresh = effects["ref_effect_at_fresh_hidden"][0]
    interaction = effects["hidden_ref_interaction"][0]
    report = f"""# Seq060 / Subtask 5 mechanism diagnostic 简报

## 结论

在现 P12 的 `stale hidden + empty ref` 上下文附近，刷新 hidden 与补回 fresh ref 对 specialist 首个动作的影响几乎同量级：EE6 RMS 分别为 **{h_empty:.3f}** 和 **{r_stale:.3f}**。因此，P12 在 age 8–11 的变化不能只归因于 empty ref 或只归因于 stale hidden；两个通道都产生了实质影响。

更关键的是，两通道并非可加的独立误差源。补回 fresh ref 后，hidden 的 stale→fresh 效应从 **{h_empty:.3f}** 降到 **{h_fresh:.3f}**（下降 {100 * (1 - h_fresh / h_empty):.1f}%）；反过来，在 fresh hidden 下补 ref 的效应从 **{r_stale:.3f}** 增至 **{r_fresh:.3f}**。二阶交互项 RMS 为 **{interaction:.3f}**，与两个 P12 局部主效应本身相当。这说明 fresh action chunk 与 fresh hidden 是一个强耦合、内部一致的条件对；fresh ref 基本屏蔽了 specialist 对 hidden 新旧的敏感性。

沿 age 观察能够进一步分开两种机制：

- `hidden effect | empty ref` 从 age 8 的 **{age_values['hidden_effect_at_empty_ref'][8]:.3f}** 增至 age 11 的 **{age_values['hidden_effect_at_empty_ref'][11]:.3f}**，配对增量为 **{hidden_age[0]:+.3f}**，seed-cluster bootstrap 95% CI [{hidden_age[1]:+.3f}, {hidden_age[2]:+.3f}]。
- `ref effect | stale hidden` 从 **{age_values['ref_effect_at_stale_hidden'][8]:.3f}** 变为 **{age_values['ref_effect_at_stale_hidden'][11]:.3f}**，配对增量仅 **{ref_age[0]:+.3f}**，95% CI [{ref_age[1]:+.3f}, {ref_age[2]:+.3f}]。

这支持一个更精确的描述：**进入 empty-ref 窗口会带来近似稳定的通道缺失效应，而 hidden staleness 的影响会随 slow age 继续累积；二者通过强交互共同决定动作。**

## 数据与核验

- 80 个冻结 observation：10 个 trial seed × 2 个 boundary（S8/S12）× 4 个 age（8–11）。
- 每个 observation 包含四个完整 condition，共 320 个 event；四格 condition、metadata 和数量均通过检查。
- 每组四次 specialist 调用使用相同 current observation、history、language 和 diffusion noise；固定噪声哈希检查无失败。
- S8/S12 上 hidden 局部效应均值分别为 {state_values['S8']['hidden_effect_at_empty_ref']:.3f}/{state_values['S12']['hidden_effect_at_empty_ref']:.3f}，ref 局部效应为 {state_values['S8']['ref_effect_at_stale_hidden']:.3f}/{state_values['S12']['ref_effect_at_stale_hidden']:.3f}，主要结论跨两个 boundary 方向一致。

## 解释边界

这里衡量的是 specialist **raw first diffusion action** 的配对变化，不经过 temporal aggregation，也不等价于 rollout success。除 `stale hidden + empty ref` 外，其余三格均是人为 channel intervention，不能解释为可部署策略的性能。实验只覆盖 seq060/subtask 5、一个固定 diffusion seed 和 age 8–11，因此结论应表述为该局部机制诊断的证据，而不是全任务总体因果结论。

关键图：`mechanism_key_conclusion.png`。误差带/误差条为按 10 个 trial seed 聚类重采样的 percentile bootstrap 95% CI。
"""
    (run_dir / "MECHANISM_REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    observations = read_jsonl(run_dir / "mechanism_observations.jsonl")
    events = read_jsonl(run_dir / "mechanism_events.jsonl")
    validation = validate_events(manifest, observations, events)

    condition_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    by_observation = defaultdict(dict)
    for event in events:
        by_observation[str(event["observation_id"])][str(event["condition"])] = event
    for observation in observations:
        observation_id = str(observation["observation_id"])
        grouped = by_observation[observation_id]
        baseline = np.asarray(grouped["stale_hidden_empty_ref"]["first_action"], dtype=np.float64)
        for condition in CONDITIONS:
            row = grouped[condition]
            metrics = action_metrics(row, baseline)
            condition_rows.append(
                {
                    "observation_id": observation_id,
                    "replicate": row["replicate"],
                    "trial_seed": row["trial_seed"],
                    "state": row["state"],
                    "age": row["age"],
                    "condition": condition,
                    **metrics,
                }
            )
        first_actions = {
            condition: np.asarray(grouped[condition]["first_action"], dtype=np.float64)
            for condition in CONDITIONS
        }
        for effect, vector in condition_effects(first_actions).items():
            contrast_rows.append(
                {
                    "observation_id": observation_id,
                    "replicate": grouped["stale_hidden_empty_ref"]["replicate"],
                    "trial_seed": grouped["stale_hidden_empty_ref"]["trial_seed"],
                    "state": grouped["stale_hidden_empty_ref"]["state"],
                    "age": grouped["stale_hidden_empty_ref"]["age"],
                    "effect": effect,
                    "effect_rms": rms(vector),
                    "effect_ee6_rms": rms(vector[:6]),
                    "effect_gripper_abs": float(abs(vector[6])),
                }
            )

    write_csv(run_dir / "mechanism_condition_rows.csv", condition_rows)
    write_csv(run_dir / "mechanism_contrast_rows.csv", contrast_rows)
    summary = {
        "validation": validation,
        "condition_metrics": summarize_rows(
            condition_rows,
            ("state", "age", "condition"),
            (
                "first_action_delta_rms",
                "first_action_delta_ee6_rms",
                "first_action_delta_gripper_abs",
                "gripper_sign_changed",
            ),
        ),
        "contrast_metrics": summarize_rows(
            contrast_rows,
            ("state", "age", "effect"),
            ("effect_rms", "effect_ee6_rms", "effect_gripper_abs"),
        ),
        "interpretation": {
            "baseline": "stale_hidden_empty_ref (P12)",
            "hidden_effect": "fresh minus stale at a fixed reference channel",
            "ref_effect": "fresh reference minus empty reference at a fixed hidden channel",
            "interaction": "fresh/fresh - fresh/empty - stale/fresh + stale/empty",
            "units": "specialist raw first diffusion action; no temporal aggregation",
        },
    }
    (run_dir / "mechanism_analysis.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")
    make_plot(run_dir, condition_rows, contrast_rows)
    make_key_conclusion_plot(run_dir, contrast_rows)
    write_brief_report(run_dir, validation, condition_rows, contrast_rows)
    print(json.dumps({"run_dir": str(run_dir), **validation}, indent=2))


if __name__ == "__main__":
    main()
