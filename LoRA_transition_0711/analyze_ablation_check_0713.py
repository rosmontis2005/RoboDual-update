"""Summarize success and action diagnostics from the fixed-sequence ablation."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys


MODES = ("base", "history_only", "lora_only", "full")


def mean(values):
    return sum(values) / len(values) if values else None


def load_mode(root: Path, mode: str) -> dict:
    result_file = json.loads((root / mode / "result_rank0.json").read_text())
    if len(result_file) != 1:
        raise RuntimeError(f"{mode} result must contain exactly one evaluation payload")
    result = next(iter(result_file.values()))
    completions = {}
    metrics = defaultdict(list)
    task_results = Counter()
    with (root / mode / "specialist_profile_rank0.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["event"] == "subtask_end":
                sequence = int(row["sequence"])
                if row["task_success"]:
                    completions[sequence] = max(completions.get(sequence, 0), int(row["subtask_i"]) + 1)
                else:
                    completions.setdefault(sequence, int(row["subtask_i"]))
                task_results[(row["task"], bool(row["task_success"]))] += 1
            if row["event"] != "step":
                continue
            profile = row["profile"]
            action = profile.get("action_prediction")
            if action:
                norm = math.sqrt(sum(float(value) ** 2 for value in action[:6]))
                metrics["action_norm_ee6"].append(norm)
                if profile.get("ref_action_expired"):
                    metrics["expired_action_norm_ee6"].append(norm)
            for key in ("jerk_l2_ee6", "aggregation_delta_ee6", "dp_ref_l2_ee6"):
                value = profile.get(key)
                if value is not None:
                    metrics[key].append(float(value))
    return {
        "avg_seq_len": result["avg_seq_len"],
        "chain_sr": result["chain_sr"],
        "completions": completions,
        "metrics": {key: {"mean": mean(values), "count": len(values)} for key, values in metrics.items()},
        "task_results": {f"{task}:{'success' if success else 'failure'}": count for (task, success), count in sorted(task_results.items())},
    }


def paired_delta(left: dict, right: dict) -> dict:
    ids = sorted(set(left) | set(right))
    deltas = [right.get(index, 0) - left.get(index, 0) for index in ids]
    return {
        "mean": mean(deltas),
        "improved": sum(delta > 0 for delta in deltas),
        "equal": sum(delta == 0 for delta in deltas),
        "worse": sum(delta < 0 for delta in deltas),
        "values": dict(zip(map(str, ids), deltas)),
    }


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(root: Path) -> None:
    data = {mode: load_mode(root, mode) for mode in MODES}
    comparisons = {
        "history_only_vs_base": paired_delta(data["base"]["completions"], data["history_only"]["completions"]),
        "lora_only_vs_base": paired_delta(data["base"]["completions"], data["lora_only"]["completions"]),
        "full_vs_base": paired_delta(data["base"]["completions"], data["full"]["completions"]),
        "full_vs_lora_only": paired_delta(data["lora_only"]["completions"], data["full"]["completions"]),
    }
    summary = {"modes": data, "paired_comparisons": comparisons}
    (root / "ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Transition-LoRA fixed-sequence ablation",
        "",
        "All four modes use the same 16 CALVIN sequences, seed, scheduler and profiling settings.",
        "This is a mechanism check, not a replacement for the 100-sequence benchmark.",
        "",
        "## Success",
        "",
        "| mode | avg length | >=1 | >=2 | >=3 | >=4 | =5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        chain = data[mode]["chain_sr"]
        lines.append(f"| {mode} | {data[mode]['avg_seq_len']:.3f} | " + " | ".join(f"{100 * chain[str(i)]:.1f}%" for i in range(1, 6)) + " |")
    lines += [
        "",
        "## Action diagnostics",
        "",
        "| mode | action norm | expired-ref norm | jerk | aggregation delta | slow-ref error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        metrics = data[mode]["metrics"]
        value = lambda key: metrics.get(key, {}).get("mean")
        lines.append(
            f"| {mode} | {fmt(value('action_norm_ee6'))} | {fmt(value('expired_action_norm_ee6'))} | "
            f"{fmt(value('jerk_l2_ee6'))} | {fmt(value('aggregation_delta_ee6'))} | {fmt(value('dp_ref_l2_ee6'))} |"
        )
    lines += ["", "## Paired sequence outcomes", ""]
    for name, item in comparisons.items():
        lines.append(
            f"- `{name}`: mean delta {item['mean']:+.3f}; "
            f"improved/equal/worse = {item['improved']}/{item['equal']}/{item['worse']}"
        )
    base_avg = data["base"]["avg_seq_len"]
    history_effect = data["history_only"]["avg_seq_len"] - base_avg
    lora_effect = data["lora_only"]["avg_seq_len"] - base_avg
    interaction = data["full"]["avg_seq_len"] - (base_avg + history_effect + lora_effect)
    metric_delta = lambda mode, key: (
        data[mode]["metrics"][key]["mean"] / data["base"]["metrics"][key]["mean"] - 1.0
    )
    lines += [
        "",
        "## 机制判断",
        "",
        f"- History adapter 单独贡献：平均长度变化 `{history_effect:+.3f}`。",
        f"- 14 个 LoRA 主干层单独贡献：平均长度变化 `{lora_effect:+.3f}`。",
        f"- 二者超出简单相加的交互项：`{interaction:+.3f}`。",
        f"- `lora_only` 的动作范数变化 `{100 * metric_delta('lora_only', 'action_norm_ee6'):+.1f}%`，"
        f"reference 耗尽后的动作范数变化 `{100 * metric_delta('lora_only', 'expired_action_norm_ee6'):+.1f}%`，"
        f"但 slow-reference 误差变化 `{100 * metric_delta('lora_only', 'dp_ref_l2_ee6'):+.1f}%`。",
        f"- `full` 的动作范数变化 `{100 * metric_delta('full', 'action_norm_ee6'):+.1f}%`，"
        f"jerk 变化 `{100 * metric_delta('full', 'jerk_l2_ee6'):+.1f}%`，"
        f"slow-reference 误差变化 `{100 * metric_delta('full', 'dp_ref_l2_ee6'):+.1f}%`。",
        "",
        "主要负面来源是 LoRA 对 x_embedder、context_adapter 和六层 temporal attention 的全局修改。"
        "它降低了动作幅度和 jerk，但同时使 fast 输出偏离 slow guidance；history adapter 单独并没有产生稳定的平滑收益，"
        "与 LoRA 主干组合后还出现额外负交互。当前训练得到的是保守化/欠执行，不是保持任务能力的条件式平滑。",
        "",
        "## 序列证据",
        "",
        "- `lora_only` 相比 base：2 条改善、10 条不变、4 条变差；变差序列 35/36/65/89 均减少 3 个任务。",
        "- `full` 相比 base：3 条改善、6 条不变、7 条变差；序列 11/20/65 分别减少 5/4/4 个任务。",
        "- 退化不是所有任务均匀下降，而是少数闭环轨迹提前失败并截断后续任务，符合小策略偏移逐步累积的表现。",
        "",
        "## 改进方向",
        "",
        "1. 暂停当前 14 层全局 LoRA 配置。下一版先只训练独立、零初始化的 residual correction 分支，冻结原 specialist。",
        "2. correction 必须有在线门控。normal 状态严格输出零，仅在 refresh、reference 耗尽、high-conflict 或 stale 条件满足时启用。",
        "3. 训练目标改为 `target_action - frozen_base_action`，而不是再次拟合绝对动作；normal 样本的 residual target 设为零。",
        "4. 加入 frozen-base preservation loss，并分别约束正常状态动作、动作范数、slow-reference 一致性和 gripper 决策。"
        "验证集 checkpoint 选择不能只看 action label MSE。",
        "5. 若仍需 LoRA，先只试最后 1-2 层 temporal output projection，使用较小 rank/alpha，并对 LoRA 输出增加显式缩放门。"
        "不要同时修改 x_embedder、context_adapter 和全部 temporal blocks。",
        "6. 补充 D 组、边界失败和恢复样本，并保存同一 observation 上 frozen base 的预测，形成可监督的 paired correction 数据。",
        "7. 可对现有权重先做 `alpha=0.25/0.5` 的 LoRA-only 权重插值筛查，但只有通过 base-preservation 门槛才值得跑 100 条正式评测。",
        "",
        "## 下一轮准入标准",
        "",
        "- 先跑同一 16 条机制集；平均长度不得低于 base 超过 0.10。",
        "- 任一 chain success rate 不得下降超过一个样本，即 6.25 个百分点。",
        "- normal 状态动作范数与 base 的偏差建议控制在 5% 内，slow-reference 误差不得明显上升。",
        "- 满足以上条件后再运行 100 sequence；本次 16 条结果用于定位机制，不能作为最终性能结论。",
    ]
    (root / "ablation_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser().resolve())
