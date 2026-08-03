# 0708 RTC-like ref_action handover 实验记录

## 说明：这一部分是自己写的
RTC本质是动作插值，这里的基本逻辑是对指导动作做插值。但是，因为需要 slow call 的时候，老的指导动作通常已经过期了，所以这里采用的逻辑是用上一组指导的第八步动作和新的指导动作做插值。这是为了避免让 RTC-like 逻辑用 0 和新指导做插值得到缺少物理意义的结果。这个开关在参数 `--rtc_ref_expired_old_mode old_last_ref`，当前代码默认也是 `old_last_ref`；如果显式设置为 `zero`，才会用 0 作为旧指导参与插值。

0708 后续检查发现一个关键细节：原本 ramp 是对 7 维 action 整体插值，因此 gripper 维也参与了插值。如果旧指导 gripper=-1、新指导 gripper=1，ramp 过程中会产生 -0.5、0、0.5 这类中间值。虽然最终环境动作会被阈值化成 -1/1，但 specialist 的 `ref_action` conditioner 已经看到了这些中间 gripper 值，可能增加不确定性或 OOD 条件。

因此现在新增参数:

```text
--rtc_ref_blend_dims all7
--rtc_ref_blend_dims ee6
```

其中:

- `all7`: 保持原行为，对末端 6 维和 gripper 1 维一起插值。
- `ee6`: 只对前 6 维末端动作插值，gripper 维直接采用 new ref，不参与 ramp。

后续更推荐把 `ee6` 作为主实验，把 `all7` 作为对照，用来验证 gripper 中间值是否是 RTC-like handover 负收益来源之一。

## 1. 实验入口

本次新增入口:

```text
RoboDual/vla-scripts/trial_smooth_RTConly_v1_0708.py
```

它基于:

```text
RoboDual/vla-scripts/task_age_v1_0706.py
```

保留 task_age_v1 的 ABCD 任务分组和 slow-call scheduler，只额外插入一个 RTC-like 的 `ref_action` handover 模块。这个入口不启用 hidden blend，不启用 action limiter，也不改变 specialist、temporal aggregation 或 CALVIN rollout 流程。

## 2. 插入模块的处理对象

RoboDual 中 slow generalist 每次 slow call 输出两个条件:

- `self.action`: slow generalist 的 8 步 action chunk。
- `self.hidden_states`: slow generalist 的 latent condition，后续作为 specialist 的 `action_cond`。

fast specialist 每一步使用:

- 当前 observation。
- `hist_action`。
- `ref_action`，由 `self.action` 整理得到。
- `action_cond`，来自 `self.hidden_states`。

本次 RTC-like 模块只处理 `ref_action`，不处理 hidden:

```text
操作对象: old slow action chunk 和 new slow action chunk 构造出的 ref_action
不操作: hidden_states / action_cond
不操作: specialist 输出的 dp_action
不操作: temporal aggregation 后的最终 action_prediction
```

因此它不是完整 RTC。完整 RTC 会在 action chunk 生成过程中对已提交前缀做 freeze / inpainting；当前实现只是 wrapper 层的 slow guidance handover，用于缓解 slow guidance hard switch。

注意：`ref_action` 本身是 7 维动作条件。新增 `--rtc_ref_blend_dims ee6` 后，handover 仍然发生在 `ref_action` 上，但只对 `ref_action[..., :6]` 做线性 ramp，`ref_action[..., -1]` 的 gripper 维不做插值。

## 3. 模块逻辑

当触发 slow refresh 时，父类流程原本会直接把:

```text
self.action = new_action
self.hidden_states = new_hidden_states
```

这会导致 specialist 在下一次调用时立刻看到新的 `ref_action` 和新的 `action_cond`。

新入口中的 `RTCRefHandoverTaskAgeV1DualSystemEvaluation` 改成:

1. slow refresh 前保存旧 slow chunk:

```text
old_action
old_age_before
```

2. slow refresh 后 hidden 仍然 hard switch:

```text
action_cond = new_hidden_states
```

3. 仅在构造 `ref_action` 时做有限步 ramp:

```text
effective_ref = alpha * new_ref + (1 - alpha) * old_ref
```

如果使用:

```text
--rtc_ref_blend_dims ee6
```

实际逻辑变为:

```text
effective_ref[..., :6] = alpha * new_ref[..., :6] + (1 - alpha) * old_ref[..., :6]
effective_ref[..., -1] = new_ref[..., -1]
```

这样 gripper 不再产生 old/new 冲突下的中间值。

默认:

```text
--rtc_ref_handover_steps 4
```

所以 refresh 后 alpha 约为:

```text
slow_age_after = 0 -> alpha = 0.25
slow_age_after = 1 -> alpha = 0.50
slow_age_after = 2 -> alpha = 0.75
slow_age_after >= 4 -> alpha = 1.00
```

窗口结束后会清空 `_rtc_ref_handover`，避免 profile 继续记录 stale old chunk 信息。

## 4. old_ref 的来源

如果旧 slow chunk 仍有有效 overlap，优先使用旧 chunk 在当前时间对齐后的剩余 ref:

```text
old_ref_source = old_overlap
```

但本项目实验通常让 slow call 发生在 age >= 8，而 `empty_ref_after_age = 8`，此时旧 slow chunk 按原逻辑已经过期，`old_num_cond_actions = 0`。

旧实现默认使用 `zero` fallback:

```text
old_ref = 0
effective_ref = alpha * new_ref
```

这个操作的问题是，`0 ref_action` 在 specialist 中更像“空指导”占位，而不是物理上有意义的旧动作轨迹。把 new ref 和 0 线性插值，会变成缩放 new guidance，缺少明确物理语义，也可能让 specialist 看到训练分布外的半强度指导。

因此当前改为新增并默认使用:

```text
--rtc_ref_expired_old_mode old_last_ref
```

逻辑是:

```text
old_ref = repeat(old_action[:, -1], new_num_cond_actions)
effective_ref = alpha * new_ref + (1 - alpha) * old_ref
```

含义是当 old chunk 已经过期时，不再用空指导参与插值，而是把上一次 slow generalist 给出的最后一个非空指导值作为 stale guidance hold。它不是严格 RTC overlap，但比 zero fallback 更符合 slow guidance 的动作空间语义。

保留的 ablation 选项:

```text
--rtc_ref_expired_old_mode old_last_ref
--rtc_ref_expired_old_mode zero
--rtc_ref_expired_old_mode hold_prev_action
--rtc_ref_expired_old_mode none
```

## 5. profile 重点字段

新增 profile 字段主要用于判断 handover 是否真实生效:

```text
rtc_ref_handover_steps
rtc_ref_alpha
rtc_ref_in_window
rtc_ref_active
rtc_ref_reason
rtc_old_age_before
rtc_old_slow_age
rtc_new_num_cond_actions
rtc_old_num_cond_actions
rtc_ref_blend_len
rtc_old_ref_valid
rtc_old_ref_source
rtc_skipped_old_expired
rtc_ref_blend_dims
rtc_ref_gripper_source
old_new_ref_l2_ee6
old_new_ref_first_l2_ee6
old_new_ref_gripper_abs_mean
old_new_ref_first_gripper_abs
new_ref_first_vs_prev_action_l2_ee6
old_ref_first_vs_prev_action_l2_ee6
```

其中最重要的是:

- `rtc_old_ref_source`: 区分 `old_overlap`、`old_last_ref`、`zero`、`hold_prev_action`、`none`。
- `rtc_ref_active`: 是否实际发生了 ref_action blend。
- `rtc_ref_blend_len`: 参与 blend 的 ref horizon 长度。
- `rtc_skipped_old_expired`: 旧 ref 过期且没有 fallback 时是否跳过。
- `rtc_ref_blend_dims`: 当前使用 `all7` 还是 `ee6`。
- `rtc_ref_gripper_source`: gripper 是参与 `blend`，还是直接来自 `new` ref。
- `old_new_ref_gripper_abs_mean`: 新旧 ref 在 gripper 维上的平均差异，用于判断 gripper 冲突强度。

## 6. 推荐启动命令

统一运行目录:

```text
cd /home/rosmontis/Projects/dualsys
```

### 6.1 主实验: task_age_v1 + RTC-like old_last_ref + ee6 ramp

这是当前最推荐先跑的版本。它保留 RTC-like 的末端 6 维动作平滑，但避免 gripper 维参与线性插值。

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/matplotlib \
conda run -n dualsys_env python RoboDual/vla-scripts/trial_smooth_RTConly_v1_0708.py \
  --dataset_subdir calvin_debug_dataset \
  --output_dir RoboDual/evaluation_results/trial_smooth_RTConly_v1_0708/seed42_old_last_ref_steps4_ee6 \
  --seed 42 \
  --num_sequences 100 \
  --max_slow_age 12 \
  --task_age_default_max_slow_age 12 \
  --task_age_group_a_max_slow_age 13 \
  --task_age_group_b_max_slow_age 12 \
  --task_age_group_c_max_slow_age 10 \
  --task_age_group_d_max_slow_age 8 \
  --empty_ref_after_age 8 \
  --rtc_ref_handover_steps 4 \
  --rtc_ref_expired_old_mode old_last_ref \
  --rtc_ref_blend_dims ee6 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

### 6.2 关键对比: task_age_v1 + RTC-like old_last_ref + all7 ramp

这个实验只和 6.1 差一个参数:

```text
--rtc_ref_blend_dims all7
```

用途是检查 gripper 参与 ramp 是否带来负面影响。它保留旧行为，即末端 6 维和 gripper 维一起插值。

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/matplotlib \
conda run -n dualsys_env python RoboDual/vla-scripts/trial_smooth_RTConly_v1_0708.py \
  --dataset_subdir calvin_debug_dataset \
  --output_dir RoboDual/evaluation_results/trial_smooth_RTConly_v1_0708/seed42_old_last_ref_steps4_all7 \
  --seed 42 \
  --num_sequences 100 \
  --max_slow_age 12 \
  --task_age_default_max_slow_age 12 \
  --task_age_group_a_max_slow_age 13 \
  --task_age_group_b_max_slow_age 12 \
  --task_age_group_c_max_slow_age 10 \
  --task_age_group_d_max_slow_age 8 \
  --empty_ref_after_age 8 \
  --rtc_ref_handover_steps 4 \
  --rtc_ref_expired_old_mode old_last_ref \
  --rtc_ref_blend_dims all7 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

### 6.3 关闭 RTC-like handover 的对照

用于确认新入口在关闭 handover 时是否等价于 `task_age_v1_0706.py`。

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/matplotlib \
conda run -n dualsys_env python RoboDual/vla-scripts/trial_smooth_RTConly_v1_0708.py \
  --dataset_subdir calvin_debug_dataset \
  --output_dir RoboDual/evaluation_results/trial_smooth_RTConly_v1_0708/seed42_no_rtc_ref \
  --seed 42 \
  --num_sequences 100 \
  --max_slow_age 12 \
  --task_age_default_max_slow_age 12 \
  --task_age_group_a_max_slow_age 13 \
  --task_age_group_b_max_slow_age 12 \
  --task_age_group_c_max_slow_age 10 \
  --task_age_group_d_max_slow_age 8 \
  --empty_ref_after_age 8 \
  --rtc_ref_handover_steps 0 \
  --rtc_ref_blend_dims ee6 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

### 6.4 zero fallback 消融

用于验证“空指导参与线性插值”是否带来负面影响。

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/matplotlib \
conda run -n dualsys_env python RoboDual/vla-scripts/trial_smooth_RTConly_v1_0708.py \
  --dataset_subdir calvin_debug_dataset \
  --output_dir RoboDual/evaluation_results/trial_smooth_RTConly_v1_0708/seed42_zero_steps4_ee6 \
  --seed 42 \
  --num_sequences 100 \
  --max_slow_age 12 \
  --task_age_default_max_slow_age 12 \
  --task_age_group_a_max_slow_age 13 \
  --task_age_group_b_max_slow_age 12 \
  --task_age_group_c_max_slow_age 10 \
  --task_age_group_d_max_slow_age 8 \
  --empty_ref_after_age 8 \
  --rtc_ref_handover_steps 4 \
  --rtc_ref_expired_old_mode zero \
  --rtc_ref_blend_dims ee6 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

### 6.5 none fallback 消融

旧 ref 过期时直接不用 handover，只有 old overlap 存在时才插值。

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/matplotlib \
conda run -n dualsys_env python RoboDual/vla-scripts/trial_smooth_RTConly_v1_0708.py \
  --dataset_subdir calvin_debug_dataset \
  --output_dir RoboDual/evaluation_results/trial_smooth_RTConly_v1_0708/seed42_none_steps4_ee6 \
  --seed 42 \
  --num_sequences 100 \
  --max_slow_age 12 \
  --task_age_default_max_slow_age 12 \
  --task_age_group_a_max_slow_age 13 \
  --task_age_group_b_max_slow_age 12 \
  --task_age_group_c_max_slow_age 10 \
  --task_age_group_d_max_slow_age 8 \
  --empty_ref_after_age 8 \
  --rtc_ref_handover_steps 4 \
  --rtc_ref_expired_old_mode none \
  --rtc_ref_blend_dims ee6 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

## 6.6 当前最小对比矩阵

如果 GPU 时间有限，建议先只跑前三个:

| 实验 | 关键参数 | 目的 |
|---|---|---|
| no handover | `--rtc_ref_handover_steps 0` | 新入口等价性和基础对照 |
| old_last_ref + ee6 | `--rtc_ref_expired_old_mode old_last_ref --rtc_ref_blend_dims ee6` | 当前推荐主实验，避免 gripper 中间值 |
| old_last_ref + all7 | `--rtc_ref_expired_old_mode old_last_ref --rtc_ref_blend_dims all7` | 只测试 gripper 参与插值的影响 |

如果这三组里 `ee6` 明显好于 `all7`，可以支持一个具体结论:

```text
RTC-like ref_action ramp 中，gripper 维不应和末端连续控制量一起线性插值。
```

如果 `ee6` 仍然不如 no handover，则说明问题不只来自 gripper 中间值，wrapper-level ref ramp 本身或 old_last_ref fallback 仍可能破坏 specialist 的条件分布。

## 7. 预期观察

重点不要只看最终 success rate，还要看 profile 中的 handover 形态:

```text
rtc_old_ref_source 分布
rtc_ref_active rate
rtc_ref_blend_len 分布
rtc_ref_blend_dims
rtc_ref_gripper_source
old_new_ref_gripper_abs_mean
jerk_l2_ee6
aggregation_delta_ee6
dp_ref_l2_ee6
ref_action_expired rate
chain@5
slow rate
```

如果 `old_last_ref` 相对 `zero` 更好，说明把空指导当成物理旧 ref 参与插值是不合适的；如果 `none` 更好，说明当前 wrapper-level ramp 本身可能仍然破坏 specialist 的条件分布。

额外需要关注 `ee6` 和 `all7` 的差异:

- 如果 `ee6 > all7`，说明 gripper 参与线性插值确实引入了额外不确定因素。
- 如果 `ee6 ≈ all7`，说明 gripper 中间值不是主要瓶颈，问题更可能来自 ref_action ramp 本身、old_last_ref fallback 或 hidden hard switch。
- 如果 `all7 > ee6`，需要进一步检查 gripper hard switch 是否在某些任务中反而导致开合时机突变。

codex resume 019f3773-e13e-7ba3-b8b4-7eb2f9772be0
