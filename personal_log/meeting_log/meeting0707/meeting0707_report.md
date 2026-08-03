# 0707 组会汇报：面向 RoboDual 的任务相关 slow-call 调度

汇报日期：2026-07-07  
项目阶段：暑期集中实验第一阶段  
当前目标：在两个月内完善实验体系，形成可写论文的主线结果与消融分析。

依据材料：`exp_log/log0704.md`、`exp_log/log0706.md`，并核对了已有 `task_age_analysis.md` 和 `risk_refresh_diagnosis.md` 中的关键结果。

---

## 1. 本次汇报要点

本阶段我想把项目主线从“零散尝试不同改法”收束为一个更清楚的问题：

> 在双模型机械臂控制框架中，如何根据任务类型和状态稳定性，减少 slow model 调用，同时保持 fast specialist 的动作连续性和任务成功率？

目前比较确定的判断是：

1. `max_slow_age=12` 已经是一个很强的效率基线。
2. `task_age` 任务分组调度是目前最有希望形成正结果的方向。
3. naive risk trigger 和 naive hidden-state blend 结果不好，但它们暴露了一个重要问题：动态 slow-call 会带来 guidance / action chunk 切换突变。
4. 下一阶段应先把 task-aware scheduling 跑稳，再把 handover / safe commit 作为补充问题处理。

---

## 2. 项目背景

RoboDual 的基本结构是双系统控制：

- slow generalist：周期性输出动作 chunk 和 hidden states，提供高层视觉语言指导。
- fast specialist：每一步执行控制，利用 slow guidance 完成具体动作。

原始系统接近固定频率调用 slow model。这个设计比较稳，但 slow call 成本较高。因此本项目的核心优化空间是：

```text
在尽量不损失成功率的前提下，减少 slow model 调用次数。
```

这里的难点是：slow guidance 不是简单的外部提示，它会影响 specialist 的动作轨迹。延长 slow-call 间隔可以省算力，但也可能导致 stale / empty guidance；提前刷新又可能打断 fast specialist 已经形成的动作连续性。

---

## 3. 已有实验结果概览

| 实验方向 | task success | avg seq len | chain@5 | slow rate | 结论 |
|---|---:|---:|---:|---:|---|
| `age7` | 84.91% | n/a | 41% | 14.64% | 成功率稳，但 slow call 较多 |
| `age12 baseline` | 84.74% | 3.22 | 42% | 8.61% | 当前最重要的效率基线 |
| `age13 / age14 / age16` | 83.51% / 81.62% / 78.76% | n/a | 38% / 32% / 28% | 更低 | 继续延长 age 会明显损失成功率 |
| `risk conservative start10` | 83.74% | 3.09 | 40% | 8.73% | risk 延迟触发后有所恢复，但仍不超过 age12 |
| `task_age v1` | 85.42% | 3.34 | 43% | 9.05% | 当前唯一同时略升成功率、保持低 slow rate 的方向 |
| `hidden handover` | 78.70% | 2.66 | 28% | 8.98% | 当前实现明显负收益 |
| LoRA targeted | all-case 45% -> 40% | n/a | n/a | n/a | attempted 指标有改善，但整体副作用明显 |

从这些结果看，接下来不适合继续把 LoRA 或 hidden blend 当主线。更稳妥的主线是：

> task-aware slow-call scheduling：不同任务使用不同的 slow guidance 刷新频率。

---

## 4. 对已有结果的解释

### 4.1 `age12` 是强 baseline

固定 `max_slow_age=12` 的意义在于：它相对原始近似 8 步刷新延长了 slow guidance 的使用窗口，显著减少 slow call，同时 task success 基本不下降。

这说明 specialist 并不需要每 8 步都依赖新的 slow guidance；在一定范围内，它可以继续沿着已有 guidance 执行。

### 4.2 继续延长 age 不安全

当 `max_slow_age` 继续拉到 13、14、16 后，成功率和 chain@5 开始明显下降。说明 slow guidance 的有效期存在上界，简单地全局延长 slow-call 间隔不是可行主线。

### 4.3 risk trigger 是诊断信号，不是直接调度策略

risk 指标可以找到不稳定状态，但实验中 “检测到 risk 后立即刷新 slow guidance” 没有转化成成功率提升。

已有诊断显示，risk refresh 周围的 action jump 更大。也就是说，risk 触发点往往已经是不稳定区间，此时硬切新的 guidance 可能进一步打断 specialist 的连续控制。

因此，risk 更适合作为：

- 失败模式分析信号；
- handover / safe commit 的触发候选；
- 论文中解释动态调度风险的证据。

暂时不适合直接作为主调度器。

### 4.4 task-age 说明调度空间存在

`task_age v1` 按任务分组设置不同 `max_slow_age`，当前结果达到：

- task success：85.42%
- avg seq len：3.34
- chain@5：43%
- slow rate：9.05%

虽然提升幅度不大，但它是当前唯一比 `age12 baseline` 略好的结果。这个结果的科研意义不是“已经找到最优规则”，而是证明：

> 不同任务对 stale / empty slow guidance 的耐受度不同，slow-call 频率不应只用全局常数控制。

---

## 5. task-age v1 的任务级现象

已有 `task_age` 分组：

| 组 | max age | 任务类型 | 目的 |
|---|---:|---|---|
| A | 13 | 稳定任务、容易任务 | 进一步减少 slow call |
| B | 12 | 默认任务 | 保持 age12 baseline |
| C | 10 | 对空指导敏感的弱任务 | 增加 slow guidance |
| D | 8 | `stack_block` | 高频指导保护 |

任务级结果显示，这个方向有正信号，也有需要修正的任务：

| 任务 | age12 | task_age v1 | 变化 | 初步判断 |
|---|---:|---:|---:|---|
| `lift_pink_block_slider` | 50.00% | 100.00% | +50.00% | 增加 slow guidance 有效 |
| `lift_blue_block_table` | 68.75% | 92.86% | +24.11% | C 组保护有效 |
| `push_red_block_left` | 66.67% | 81.82% | +15.15% | C 组保护有效 |
| `lift_pink_block_table` | 85.71% | 100.00% | +14.29% | 稳定受益 |
| `place_in_slider` | 69.57% | 75.00% | +5.43% | 小幅受益 |
| `rotate_red_block_right` | 77.78% | 44.44% | -33.33% | 分组或样本波动需复查 |
| `lift_blue_block_slider` | 71.43% | 56.25% | -15.18% | 不能简单归入 C 组 |
| `push_blue_block_right` | 33.33% | 16.67% | -16.67% | 可能不是调度能解决的弱任务 |
| `turn_on_lightbulb` | 94.12% | 81.25% | -12.87% | A 组延长到 13 可能过激 |

这说明 v1 不是最终策略，但它提供了 v2 的修改依据。

---

## 6. 第一阶段实验矩阵

根据 0706 的计划，第一阶段只关注任务分级本身，不混入 risk / handover / LoRA。

统一入口：

```text
RoboDual/vla-scripts/task_age_v1_0706.py
```

统一输出目录：

```text
RoboDual/evaluation_results/repeat_task_age_v1/
```

实验矩阵：

| 实验 | 参数组 | 目的 |
|---|---|---|
| `age12_baseline` | 全任务 `max_slow_age=12` | 最重要基线；已有结果约 84.74%、slow rate 8.61% |
| `age11_budget` | 全任务 `max_slow_age=11` | 控制 slow-call 预算，判断 task-age 收益是否只是来自更多 slow call |
| `task_age_v1` | A=13, B=12, C=10, D=8 | 复跑当前最好分组，验证 85.42% 是否稳定 |
| `protect_only` | A=12, B=12, C=10, D=8 | 只测试“困难任务增加 slow guidance”是否带来收益 |
| `extend_only` | A=13, B=12, C=12, D=12 | 只测试“简单任务减少 slow call”是否安全 |
| `task_age_v2a` | A=13, B=12, C 拆分，D=7/8 | 在 v1 基础上微调，重点修正失败任务 |
| `task_age_v2b` | A=12/13, B=12, C=8/10, D=7 | 如果 v2a 有方向，再做更激进保护版 |

第一批建议先跑：

1. `age12_baseline`
2. `age11_budget`
3. `task_age_v1`

这三组可以先回答两个关键问题：

- `task_age_v1` 的提升是否可重复？
- `task_age_v1` 是否只是因为 slow call 预算略高于 age12？

---

## 7. task-age v2 的设计方向

v2 不应该大范围扫参，而应该根据 v1 的任务级结果做小改动。

优先保持的正向任务：

- `lift_pink_block_slider`
- `lift_blue_block_table`
- `push_red_block_left`
- `lift_pink_block_table`
- `place_in_slider`

需要重点修正的任务：

| 任务 | v1 问题 | v2 可能调整 |
|---|---|---|
| `stack_block` | D=8 只小幅优于 age12，且历史 age7 更好 | 单独测试 D=7 / 8 |
| `lift_blue_block_slider` | C=10 反而下降 | 单独测试 8 / 10 / 12，避免简单归入 C 组 |
| `turn_on_lightbulb` | A=13 明显下降 | 从 A 组移出，回到 12 或单独设定 |
| `rotate_red_block_right` | v1 大幅下降 | 先复查样本数和 seed 稳定性，再决定是否改组 |
| `push_blue_block_right` | 整体成功率低，调度改善有限 | 作为 failure mode 单独分析，不强行纳入 v2 收益 |

v2 的目标不是一次性找到最优表，而是验证两个更清楚的假设：

1. 保护困难任务是否真的能提升成功率？
2. 延长简单任务 age 是否真的能安全省 slow call？

---

## 8. 动态调度与 chunk 突变问题

导师关注的 handover / chunk 突变问题需要保留，因为它和 task-aware scheduling 是同一个大问题的另一面：

> 当 slow-call 频率变成动态后，新旧 guidance 如何安全切换？

当前 hidden-state blend 的负结果不能简单理解为“handover 不重要”。更合理的解释是：当前 naive blend 方式可能不对。

可能失败原因：

1. hidden state 不是动作空间变量，线性插值后可能落在 specialist 未见过的表示区域。
2. 当前实验同时混合 ref action 和 hidden state，无法定位是 action ramp 还是 hidden blend 造成负收益。
3. 当旧 ref 已经过期为空时，blend 会把新的 ref action 缩小成 25% / 50% 强度，这可能比直接使用新 guidance 更差。
4. gripper 维度接近离散开合，连续混合可能产生 OOD 的中间值。
5. risk refresh 发生在原本就不稳定的位置，在这些位置硬切或混合 guidance 更容易放大动作跳变。
6. 只处理 guidance，没有同步处理 temporal aggregation buffer，新旧动作预测仍可能互相拉扯。

---

## 9. handover 的下一步处理方式

我建议先不要把 hidden blend 混入第一阶段 task-age 实验。更稳妥的处理方式是：在 task-age 主线跑稳后，单独做最小消融。

| 实验 | 目的 |
|---|---|
| `risk_balanced + no handover` | 原始对照 |
| `risk_balanced + ref-only handover steps=2` | 判断 action-space ref ramp 是否有用 |
| `risk_balanced + ref-only handover steps=4` | 判断 handover 持续时间过长是否伤害性能 |
| `age12 + ref-only handover` | 去掉 risk 干扰，看固定刷新下 handover 是否有效 |
| `age12 + hidden blend only` | 判断 hidden blend 是普遍有害，还是只在 risk refresh 下有害 |

如果继续改实现，优先考虑两点：

1. 旧 ref 过期时不要 blend zero ref。  
   如果旧 chunk 已经为空，直接使用完整 new ref；平滑只放在最终动作限幅或 consistency gate 上。

2. hidden 不做线性 blend，改成 delayed commit / gated commit。  
   slow 可以先计算新 hidden，但只有当新条件下的 action 与当前轨迹差异小于阈值时才切换。否则暂存为 pending hidden，等待稳定窗口或 max age 再切。

这样可以把 handover 问题表述为：

```text
不是把两个 hidden state 平均，而是判断什么时候安全接入新的 slow guidance。
```

这个方向可以作为论文中的第二个分析点：naive latent interpolation 会造成 OOD conditioning，因此需要 action-aware safe commit。

---

## 10. 两个月内的实验节奏

| 时间 | 主要任务 | 产出 |
|---|---|---|
| 第 1-2 周 | 复跑 `age12`、`age11`、`task_age_v1`，跑 `protect_only` / `extend_only` | 稳定 baseline，确认 task-age 是否真实有效 |
| 第 3-4 周 | 做 `task_age_v2a/v2b`，按任务分析失败模式 | 任务分组表、弱任务诊断 |
| 第 5-6 周 | 视结果加入 risk diagnostic-only 或 safe commit 小消融 | 解释动态调度下的 chunk 突变问题 |
| 第 7-8 周 | 整理最终表格、画图、写论文初稿 | 主结果表、消融表、方法与实验章节 |

最终论文故事可以收束为：

> Slow guidance 不需要固定频率调用；任务相关调度能在低 slow-call 预算下保持甚至略微提升成功率。但动态调度引入 guidance 切换突变，简单 risk refresh 和 naive hidden blend 会失败，因此需要更谨慎的 action-aware handover / safe commit 机制。

---

## 11. 本次组会希望确认的问题

1. 第一阶段是否先只跑 task-age 相关矩阵，不混入 risk / handover / LoRA？
2. `age11_budget` 是否足够作为 slow-call 预算控制，还是需要再加 `age10`？
3. task-age v2 是否采用“小改动、少扫参”的策略，优先修正 `stack_block`、`lift_blue_block_slider`、`turn_on_lightbulb`？
4. handover 是否作为第二阶段分析点，先做 ref-only / delayed commit，而不是继续扩大 hidden-state blend？

---

## 12. 汇报时的简短结论

当前最有希望的主线是 task-aware slow-call scheduling。它不是为了大幅刷新成功率，而是为了证明双系统 VLA 控制中 slow guidance 的调用频率具有任务相关性：简单任务可以更少调用，困难任务需要更频繁指导。

`age12` 已经给出强效率基线，`task_age v1` 给出初步正结果。下一步最重要的是复跑和拆分消融，确认收益不是 seed 波动或 slow-call 预算变化造成的。

chunk 突变 / handover 问题仍然重要，但当前 hidden-state blend 负结果说明 naive latent interpolation 不可靠。后续应把它转化为 safe commit 问题：什么时候安全接入新的 slow guidance，而不是直接混合 hidden state。
