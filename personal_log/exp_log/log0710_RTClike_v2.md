# 0710 Transition-aware action smoothing 实验记录

## 1. 从前两次 RTC-like 实验发现的问题

0708-0709 的两次实验都基于 `task_age_v1`，只对 slow refresh 后的 `ref_action[..., :6]` 做 RTC-like ramp，gripper 不参与插值：

```text
step4:
RoboDual/evaluation_results/trial_smooth_RTConly_v1_0708/
seed42_old_last_ref_steps4_ee6_0708

step2:
RoboDual/evaluation_results/trial_smooth_RTConly_v1_0708/
seed42_old_last_ref_steps2_ee6_0709
```

对应基线为：

```text
RoboDual/evaluation_results/exp0526-0525-task_age
```

原始设想是：slow guidance 更新时，不立即从旧指导切换到新指导，而是在有限步内线性引入新 `ref_action`，降低 chunk 边界的不连续性。

实际检查后发现，这个构造存在两个根本问题。

第一，slow call 的间隔通常是 8-13 步，而 generalist action chunk 只有 8 步。触发刷新时旧 chunk 通常已经没有有效的 future overlap，因此 ramp 使用的并不是真正与当前时刻对齐的旧轨迹，而是：

```text
old_last_ref = 过期旧 chunk 的最后一个动作
```

然后把这个旧末值复制成一段伪旧指导，与新指导插值。它更接近“延迟采用新指导”，而不是真正的 RTC continuation。

第二，固定 ramp 对所有 refresh 无差别介入。`step4` 实际产生 3 个插值执行步，新指导权重依次为：

```text
25% -> 50% -> 75% -> 100%
```

它在 8-12 步左右的 slow-call 周期中占比过高。`step2` 实际只有 1 个插值步：

```text
50% old_last_ref + 50% new_ref
```

虽然缩短 ramp 后性能有所恢复，但仍没有超过 hard-switch baseline。

## 2. 成功率实验结果

三组实验使用相同的 seed 42、100 条 CALVIN 序列、`ep_len=360` 和相同的 ABCD task-age 配置。

| 实验 | avg seq len | SR1 | SR2 | SR3 | SR4 | SR5 |
|---|---:|---:|---:|---:|---:|---:|
| task-age baseline | **3.34** | 90% | 77% | **65%** | **59%** | **43%** |
| RTC step2 ee6 | 3.22 | **93%** | **81%** | 63% | 48% | 37% |
| RTC step4 ee6 | 2.96 | 88% | 73% | 58% | 44% | 33% |

从 `step4` 缩短到 `step2` 后：

- 平均完成长度从 2.96 提升到 3.22。
- SR1-SR5 分别提升 5、8、5、4、4 个百分点。
- 相对 baseline 的 0.38 平均长度损失，追回约 68%。
- 逐序列比较中，32 条改善、19 条退化、49 条不变，总计多完成 26 个子任务。

这支持“较长 ramp 拖慢新指导生效”的判断。但 `step2` 仍低于 baseline，特别是第 3 到第 4 个连续任务：

| 条件成功率 | baseline | step2 | step4 |
|---|---:|---:|---:|
| task 1 -> 2 | 85.6% | **87.1%** | 83.0% |
| task 2 -> 3 | **84.4%** | 77.8% | 79.5% |
| task 3 -> 4 | **90.8%** | 76.2% | 75.9% |
| task 4 -> 5 | 72.9% | **77.1%** | 75.0% |

因此，当前结果不能解释为“把 ramp 再调短就能解决问题”。失败的是 `fixed steps + expired old_last_ref interpolation` 这个具体构造。

## 3. 三组实验的动作分布分析

重新分析三个 `specialist_profile_rank0.jsonl` 时，必须按 `(sequence, subtask_i)` 独立计算动作差分。评测代码会在每个子任务开始时调用 `model.reset()`，不能把前一个子任务的末动作和下一个子任务的首动作相减。

按子任务正确重建后，slow-refresh 边界的动作分布如下。这里：

```text
delta = ||u_t - u_(t-1)||_2, only ee6
jerk  = ||u_t - 2*u_(t-1) + u_(t-2)||_2, only ee6
```

| 实验 | 成功任务 delta p95 | 失败任务 delta p95 | 成功任务 jerk p95 | 失败任务 jerk p95 |
|---|---:|---:|---:|---:|
| baseline | 0.154 | 0.184 | 0.193 | 0.216 |
| step2 | 0.114 | 0.123 | 0.179 | 0.186 |
| step4 | 0.097 | 0.107 | 0.179 | 0.180 |

这个结果非常关键：

> step2 和 step4 的动作在数值上确实更平滑，但成功率反而更差。

因此不能把降低所有 refresh 的 delta/jerk 当作目标。很多较大的动作变化是 slider、lift、stack 等任务推进所必需的；如果无条件削弱，就会得到“轨迹看起来更平滑，但机器人没有及时完成动作”的结果。

baseline 中仍然存在少量 slow-refresh 联合异常尾部。采用：

```text
slow refresh
AND delta_l2_ee6 > 0.18
AND jerk_l2_ee6 > 0.24
```

在 baseline 中只命中 66 个 refresh event：

- 占可计算历史动作的 refresh 约 1.53%。
- 占全部执行 step 约 0.13%。
- 这些 event 分布在 20/57 个失败子任务中。
- 同时分布在 23/334 个成功子任务中。

失败子任务通常会运行到 360 步，因此拥有更多 refresh 和更多被命中的机会。上述统计不能直接证明异常 jump 导致失败，但可以支持一个非常保守的保底触发器，而不支持全局平滑。

## 4. 新思路：Transition-aware minimal-intervention smoothing

新的思路不再修改 `ref_action`，也不插值 `hidden_states`。新 slow guidance 在 refresh 后立即完整生效：

```text
self.action        = new slow action chunk
self.hidden_states = new slow hidden states
```

specialist 正常使用新条件生成 action chunk，temporal aggregation 也保持不变。只有在最终候选动作即将送入环境前，才检查本次 refresh 是否同时出现异常 delta 和 jerk。

整体流程：

```text
task-age scheduler 触发 slow refresh
        |
        v
新 hidden/ref 立即完整生效
        |
        v
specialist + temporal aggregation 得到候选动作 u_new
        |
        v
检查 refresh AND delta > 0.18 AND jerk > 0.24
        |
        +-- 否：原样执行 u_new
        |
        +-- 是：只对 ee6 做一次最小 jerk 修正
```

修正使用上一时刻动作速度的 constant-velocity anchor：

```text
u_anchor = 2 * u_(t-1) - u_(t-2)
raw_jerk = u_new - u_anchor
```

把 `raw_jerk` 朝 `jerk_limit=0.24` 投影，但对候选动作的总修正量设置：

```text
||u_exec - u_new||_2 <= 0.18
```

这个 correction cap 来自 baseline 联合触发样本的修正量分布，约对应 p95。它的作用是避免极端情况下为了严格满足 jerk limit 而大幅改写模型动作。

实现原则：

- 只在 slow refresh 边界检查。
- `delta` 和 `jerk` 必须同时越界才触发。
- 只修改 temporal aggregation 后的最终 ee6 动作。
- gripper 直接采用 specialist 的离散结果，不参与平滑。
- 不修改 generalist `ref_action`。
- 不修改 generalist `hidden_states`。
- 不预先规定必须连续平滑 2 步或 4 步。
- 非触发步骤和关闭功能时应与 baseline 动作完全一致。

这条路线的目标不是让所有动作更平滑，而是：

> 保留正常任务动作的响应速度，只对 slow-refresh 的少量联合异常跳变做一次保底修正。

## 5. 新实验脚本

新脚本基于：

```text
RoboDual/vla-scripts/evaluate_calvin_task_age_0525.py
```

建立独立副本：

```text
RoboDual/vla-scripts/evaluate_calvin_task_age_transition_smooth_0710.py
```

原始 `evaluate_calvin_task_age_0525.py` 未修改。

新增类：

```text
TransitionSmoothTaskAgeDualSystemEvaluation
```

新增参数：

```text
--transition_smooth
--transition_delta_threshold
--transition_jerk_threshold
--transition_jerk_limit
--transition_max_correction_ee6
--output_dir
```

新增 profile 字段包括：

```text
transition_smooth_refresh
transition_smooth_eligible
transition_smooth_triggered
transition_smooth_reason
transition_raw_delta_l2_ee6
transition_raw_jerk_l2_ee6
transition_correction_l2_ee6
transition_post_delta_l2_ee6
transition_post_jerk_l2_ee6
```

代码检查结果：

- `python -m py_compile` 通过。
- conda 环境下完整导入和 `--help` 通过。
- 定向行为测试通过。
- 关闭功能时动作保持不变。
- 非 slow-refresh 步骤不触发。
- 只有 delta 或只有 jerk 越界时不触发。
- correction L2 cap 正确。
- gripper 在触发前后保持不变。

目前尚未运行完整 CALVIN 评测，因此该方法是否提升成功率仍需实验验证。

## 6. 推荐启动命令

当前只推荐先运行一组保守参数，不立即进行大范围阈值扫描：

```bash
cd /home/rosmontis/Projects/dualsys

CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/matplotlib \
conda run -n dualsys_env python \
  RoboDual/vla-scripts/evaluate_calvin_task_age_transition_smooth_0710.py \
  --dataset_subdir calvin_debug_dataset \
  --output_dir RoboDual/evaluation_results/task_age_transition_smooth_0710/seed42_delta018_jerk024_cap018 \
  --num_sequences 100 \
  --ep_len 360 \
  --slow_call_strategy task_age \
  --max_slow_age 12 \
  --empty_ref_after_age 8 \
  --task_age_default_max_slow_age 12 \
  --task_age_group_a_max_slow_age 13 \
  --task_age_group_b_max_slow_age 12 \
  --task_age_group_c_max_slow_age 10 \
  --task_age_group_d_max_slow_age 8 \
  --profile_sample_var_k 3 \
  --profile_sample_var_interval 8 \
  --profile_sample_var_ages 8,9,10,11,12 \
  --transition_smooth \
  --transition_delta_threshold 0.18 \
  --transition_jerk_threshold 0.24 \
  --transition_jerk_limit 0.24 \
  --transition_max_correction_ee6 0.18 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

注意：该脚本继承 baseline 的 `seed_everything(42)`，当前没有新增 `--seed` 参数。

## 7. 实验后优先检查的指标

成功率仍然是主指标：

```text
avg_seq_len
SR1-SR5
逐任务 success / total
```

同时检查新机制是否符合“少量、精准干预”的设计：

```text
transition_smooth_triggered rate
triggered / slow refresh
triggered / all steps
transition_correction_l2_ee6
transition_raw_jerk_l2_ee6
transition_post_jerk_l2_ee6
```

预期触发比例应明显低于固定 ramp 的 8.42% 或 25.16% active-step 占比。如果新方法触发过多，即使成功率提升，也需要重新检查是否已经变成另一种全局 limiter。

最终判断标准：

1. 如果成功率高于 baseline，同时 trigger rate 很低，说明 refresh-only 保底平滑有效。
2. 如果平滑指标下降但成功率不升，说明边界 jump 仍然主要是任务必要动作，而不是失败原因。
3. 如果成功率下降，应优先关闭或提高门限，不继续增加平滑步数。

codex resume 019f3773-e13e-7ba3-b8b4-7eb2f9772be0

## 8. 实验结果简要总结

本次 transition-aware smoothing 的结果为：

```text
avg_seq_len = 3.22
SR1-SR5 = 92% / 77% / 63% / 51% / 39%
```

它与 RTC step2 的平均完成长度相同，但 SR4 和 SR5 分别提高 3 和 2 个百分点，说明相比固定 ramp，新方法对长任务链的影响更小。与 task-age baseline 相比，平均完成长度仍低 0.12，SR4 和 SR5 分别低 8 和 4 个百分点，因此没有实现明确涨点。

平滑器共触发 82 次，占全部执行 step 的 0.155%、slow refresh 的 1.71%，说明“只处理少量联合异常边界”的设计正常生效。gripper 没有被修改，ee6 correction 中位数为 0.044，p95 接近设定上限 0.18。

82 次触发中有 63 次位于最终失败子任务。失败子任务的 eligible-refresh 触发率约为成功子任务的 3.8 倍，尤其集中在 `stack_block`、`lift_blue_block_table` 和 `rotate_pink_block_right`。因此 `delta + jerk` 联合门限可以作为有效的失败状态信号，但当前 constant-velocity jerk correction 没有证明能够挽救这些状态。

当前结论是：refresh-only sparse smoothing 明显比固定多步 ramp 更合理，且不会普遍拖慢新指导，但它仍不是有效的成功率提升方案。下一步不建议降低门限或增强 correction，而应先补充同一脚本关闭 `--transition_smooth` 的严格对照，并考虑将该联合异常信号用于重新规划或 safe commit gate，而不是继续强化动作插值。
