# 单 sequence 基线选择

## 结论

后续研究基线确定为 **CALVIN sequence index 60（0-based；人工编号第 61 条）**。

任务顺序：

1. `open_drawer`
2. `push_blue_block_left`
3. `move_slider_left`
4. `turn_on_lightbulb`
5. `lift_pink_block_slider`

历史结果满足预设筛选条件：

| 调度 | 成功子任务数 | 失败位置 |
| --- | ---: | --- |
| 原始 `fixed_mod8`（0413） | 5/5 | 无 |
| `age_empty, max_age=7` 控制组（0424） | 5/5 | 无 |
| uniform `max_slow_age=12`（0425_1） | 4/5 | 第 5 步 |
| uniform `max_slow_age=12`（0523 归档） | 4/5 | 第 5 步 |

两个 age-12 目录的累计成功率文件 SHA256 完全相同，所选 sequence 的逐任务步数和诊断统计也相同；因此应把它们视为两个一致的历史 artifact，不能在没有运行来源记录的情况下声称是两个独立随机重复。

## 100-sequence 顺序确认

可以确认：仓库内标准的 100-sequence 评测面对相同且同序的 sequence。

- CALVIN 的 `get_sequences(100)` 在 `temp_seed(0)` 下生成候选，按 initial-state index 设置确定性 seed，最后在同一临时 seed 中 shuffle。
- 仓库中找到 24 个恰有 100 行的成功率结果。
- 其中 21 个具有完整 sequence id `0..99` 的逐步 profile，可直接审计任务身份。
- 21 个 profile 共核对 7,436 个实际到达的 `(sequence, subtask)` 位置，与导出的 canonical 100-sequence catalog **0 个不一致**。

这个结论只适用于相同调用 `get_sequences(num_sequences=100)` 的标准入口。改变 `num_sequences`、显式传入 `sequence_indices`、使用自定义 catalog，都会改变评测集合或顺序，不能默认与这里相同。

## 为什么选择 sequence 60

满足条件的轨迹共有两条：index 36 和 index 60，均为 fixed-mod-8 下 5/5、age-12 下 4/5。选择 60 而不是 36，原因是它的分叉更干净：

- 前四个子任务在两种调度下都成功。
- fixed-mod-8 / age-12 的前四步耗时分别为 `67/68`、`117/121`、`66/80`、`60/67`，没有 sequence 36 中 `move_slider_left` 的 `98 -> 150` 这种大幅退化。
- 第五步 `lift_pink_block_slider` 在 fixed-mod-8 下 70 步成功；age-12 给足 360 步仍失败，便于把研究焦点放在第五步开始前的累计状态误差和第五步内部的 stale/empty-guidance 窗口。
- sequence 结构存在明确的跨任务状态联系：初始 pink block 在 slider right，第三步移动 slider，第五步再从 slider 中抬起 pink block，适合研究早期状态偏差如何传递到末端任务。

## 初步 slow-call 对照信号

前四步合计：

| 指标 | fixed-mod-8 | uniform age-12 |
| --- | ---: | ---: |
| 环境步数 | 310 | 336 |
| slow calls | 41 | 30 |
| reference 已耗尽步数 | 0 | 104 |

第五步：

| 指标 | fixed-mod-8 | uniform age-12 |
| --- | ---: | ---: |
| 结果 | 70 步成功 | 360 步失败 |
| slow calls | 9 | 30 |
| reference 已耗尽步数 | 0 | 120 |
| `aggregation_delta_ee6` p95 | 0.109 | 0.187 |
| 最大 `gripper_flip_count` | 2 | 3 |

这些是候选机制，不是因果结论。后续应优先检验：reference 耗尽窗口的动作偏差、slow refresh 前后跳变、gripper 状态、第三步结束时 slider/pink block 的最终位姿，以及这些误差在第五步开始时是否已经存在。

## 实验使用约定

- 统一使用 `sequence_index=60`；日志和文件名写 `seq060`，同时注明它是人工编号第 61 条。
- 每个调度至少使用 3 个相同的 per-sequence diffusion seeds，禁止让前序 rollout 长度改变后续任务的随机数流。
- fixed-mod-8 与 age-12 使用完全相同的模型、初始状态、语言标注、`ep_len`、temporal aggregation 和 seed。
- 保存每个 subtask 的起始/结束 simulator state，尤其是第三步结束、第五步开始的 slider 与 pink block 位姿。
- 历史 0413 与 0425 数据用于筛选，不应直接当成已经完成的严格 paired causal experiment；0413 的 `ep_len=240`，age-12 artifact 的失败上限为 360。

## 文件说明

- `baseline_sequence.json`：机器可读的最终基线定义和来源哈希。
- `canonical_100_sequences.json`：当前 CALVIN 代码导出的固定 100 条 catalog。
- `sequence_order_audit.json`：顺序审计结果与所覆盖的结果文件。
- `candidate_outcome_matrix.csv`：两条合格候选的历史结果。
- `selected_sequence_profile_summary.csv`：所选 sequence 的逐子任务指标。
- `profiles/`：从三个关键历史 profile 提取的 seq060 原始逐步记录。
- `analyze_and_extract.py`：从仓库历史结果重新生成审计、摘要和切片。
- `export_canonical_sequences.py`：从 CALVIN `get_sequences(100)` 重新导出 catalog。
