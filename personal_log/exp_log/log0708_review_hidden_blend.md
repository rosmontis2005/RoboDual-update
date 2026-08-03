# 0708 review: hidden blend 结果复盘

## 1. 相关模块的处理对象、处理逻辑和预期结果

| 模块 | 处理对象 | 处理逻辑 | 输出 | 理论预期结果 | 当前阶段是否适合参考 |
|---|---|---|---|---|---|
| RTC / Real-Time Chunking | 新旧 action chunk，尤其是已经提交执行的动作前缀和新生成的动作后缀 | 推理时把已提交动作前缀固定，让新 chunk 在剩余 horizon 内和旧 chunk 保持连续，相当于 action-time continuation / inpainting | 更连续的新 action chunk | slow policy 推理延迟或 chunk 刷新时，降低 chunk 边界动作跳变 | 适合参考，但主要适合 action / ref_action，不直接支持 hidden state 线性插值 |
| AAC / Adaptive Action Chunking | chunk 长度、refresh 频率、动作不确定性 | 根据 action entropy 或类似不确定性信号，自适应决定什么时候短 chunk、什么时候长 chunk | 动态 chunk size 或动态刷新时机 | 稳定时少刷新，不稳定时提高反应性，避免固定 chunk size 的反应性和平滑性冲突 | 适合参考，但它是 scheduler 思路，不是 blend 公式 |
| IBTC / Interpolated Bumpless Transfer Control | switched system 中 old controller 到 new controller 的切换 | 检测到模式切换后，不让新 controller 立刻 100% 接管，而是在有限步内插值过渡 | 随时间变化的 transition controller / alpha 权重 | 降低 controller 切换时的 control input jump 和瞬态冲击 | 适合参考为有限步 handover 思路，但原对象是 control input / controller gain，不是 neural hidden state |
| Ruckig | 最终机器人轨迹或控制命令 | 在速度、加速度、jerk 约束下做在线轨迹生成或限幅 | jerk-limited trajectory / smoother action | 降低最终执行动作的一阶、二阶、三阶差分 | 可作为输出层 action limiter 参考；本次实验未启用 |
| ACT temporal ensembling | 多个重叠 action chunk 对当前步的预测 | 对当前步来自不同 chunk 的预测做加权平均 | 聚合后的当前动作 | 降低 action chunking 的单步噪声和动作抖动 | RoboDual 已有类似 temporal aggregation；它不能解决 slow guidance hard switch 本身 |
| Legato | 模型训练中的 continuation 能力 | 训练时让 policy 学会从已知动作前缀自然延续到后续动作 | 原生支持 continuation 的模型 | 比推理时外部修补更稳定，减少 mode switching | 当前时间和算力成本不允许参考训练期方案，暂不作为主线 |
| Training-Time RTC | 训练样本中的 committed action prefix 和 future action postfix | 训练时模拟 delay / prefix condition，让模型学会 prefix-conditioned prediction | 学会从已提交动作自然接续的模型 | 推理时减少 RTC 外部修补成本 | 同样属于训练期方案，当前不作为主线 |

关键区分:

- RTC、AAC、IBTC 都能支持“不要在 chunk 或 controller 边界硬切”的设计原则。
- RTC 和 IBTC 的直接对象仍然是 action / control input / controller gain 这类有控制意义的量。
- AAC 的直接对象是 refresh schedule，不是 hidden state。
- 因此，这些模块不能直接证明 old hidden state 和 new hidden state 可以线性平均。

## 2. 进入入口代码的 blend / smooth 模块如何运行

本次实验入口是:

- `RoboDual/vla-scripts/evaluate_calvin_0428.py`
- `RoboDual/vla-scripts/dual_sys_evaluation_0424test.py`

实际进入主代码的是 wrapper 级别的 `slow_handover`，不是直接调用 `search-smoothen-action/code/*` 中的 RTC、AAC、IBTC 官方代码。

### 2.1 slow-call scheduler

`evaluate_calvin_0428.py` 中的 `VariableSlowCallDualSystemEvaluation` 负责决定什么时候调用 slow generalist。

本次实验配置:

- `slow_call_strategy = risk_balanced`
- `effective_slow_trigger_policy = age_empty`
- `min_slow_age = 7`
- `max_slow_age = 12`
- `risk_start_age = 8`

逻辑上，它先保证 slow age 小于 `min_slow_age` 时不刷新；到 `max_slow_age` 必须刷新；中间区间根据上一 step 的风险指标决定是否提前刷新。

风险指标包括:

- `aggregation_delta_ee6`
- `jerk_l2_ee6`
- `gripper_flip_count`
- `sample_var_ee6`
- `sample_var_gripper`

这部分更接近 AAC 的启发: 用不稳定性指标影响刷新时机。但当前实现仍然是 risk trigger -> immediate slow refresh，不是 AAC 式的“先调整目标 chunk / refresh horizon，再通过安全接入机制切换”。

### 2.2 ref_action handover

在 `dual_sys_evaluation_0424test.py` 中，slow refresh 前保存:

- `old_action`
- `old_hidden_states`
- `old_age_before`

slow generalist 重新输出后，代码更新:

- `self.action = new_action`
- `self.hidden_states = new_hidden_states`

然后通过 `slow_handover_steps` 计算 alpha。

本次实验 `slow_handover_steps = 4`，所以 alpha 序列为:

- refresh 后第 0 步: `alpha = 0.25`
- refresh 后第 1 步: `alpha = 0.50`
- refresh 后第 2 步: `alpha = 0.75`
- refresh 后第 3 步及以后: `alpha = 1.00`

`ref_action` 的混合方式是:

```text
mixed_ref_action = alpha * new_ref_action + (1 - alpha) * old_ref_action
```

这部分可以理解为 RTC / IBTC 思路的低成本 wrapper 近似:

- RTC 贡献的是“新旧 chunk 交接不要硬切”的问题表述。
- IBTC 贡献的是“有限步 alpha transition”的控制直觉。

### 2.3 hidden state blend

本次实验额外打开了:

```text
--slow_handover_blend_hidden
```

因此代码对 `hidden_states` 也使用同一个 alpha:

```text
mixed_hidden = alpha * new_hidden + (1 - alpha) * old_hidden
```

这个操作对象是 fast specialist 的 `action_cond`，也就是 slow generalist 输出的 latent condition。

这里是本次实验最关键的问题:

- `ref_action` 有动作坐标意义，线性 ramp 至少有 action-space 解释。
- `hidden_states` 没有明确物理坐标意义，线性平均不一定仍然表示一个有效的 slow guidance。
- 因此，hidden blend 不是 RTC / AAC / IBTC 的直接结论，而是把 action-space handover 思路外推到 latent-space 的一次经验性尝试。

### 2.4 action limiter / smooth 输出层

代码中还存在 Ruckig-inspired 的输出层 limiter:

- `action_delta_limit_ee6`
- `action_jerk_limit_ee6`

但本次实验配置中二者都是 `0.0`，所以没有启用。profile 中 `action_slew_applied = 0.00%` 也确认最终 action limiter 没有介入。

## 3. 本次实验结果及不理想程度

实验目录:

```text
RoboDual/evaluation_results/exp0609-0428(edited)-slow_handover_steps 4 --slow_handover_blend_hidden
```

最终结果:

| 指标 | 数值 |
|---|---:|
| subtasks | 338 |
| task success rate | 78.70% |
| avg sequence length | 2.66 |
| chain@1 | 85.00% |
| chain@2 | 64.00% |
| chain@3 | 51.00% |
| chain@4 | 38.00% |
| chain@5 | 28.00% |

profile 聚合:

| 指标 | 数值 |
|---|---:|
| step records | 52278 |
| slow calls | 4694 |
| slow rate | 8.98% |
| ref expired rate | 29.73% |
| risk refresh rate | 1.58% |
| handover active rate | 24.81% |
| action slew applied | 0.00% |

slow call 原因:

| 原因 | 次数 |
|---|---:|
| initial | 338 |
| max_slow_age | 3530 |
| risk_balanced | 826 |

active handover 来源:

| 来源 | active handover step |
|---|---:|
| max_slow_age | 10524 |
| risk_balanced | 2448 |

这个结果不理想，程度不是轻微波动，而是明显负收益:

- 与 `age=12` 附近的省 slow-call baseline 相比，成功率从约 `84.74%` 降到 `78.70%`，下降约 `6.04` 个百分点。
- 与 `task_age` 结果相比，成功率从 `85.42%` 降到 `78.70%`，下降约 `6.72` 个百分点。
- `chain@5` 从 `task_age` 的 `43.00%` 降到 `28.00%`，下降 `15` 个百分点。
- avg sequence length 从 `task_age` 的 `3.34` 降到 `2.66`，说明不仅单个 subtask 成功率下降，连续完成长链任务的能力也明显变差。

因此，这次实验不能被解释为“平滑没有帮助但影响很小”。更准确的结论是:

```text
当前实现的 ref_action ramp + hidden_state linear blend 组合，在 risk_balanced slow-call 策略下表现为明显负收益。
```

## 4. 可能不理想的原因

第一，hidden state 不是 action-space 变量。

RTC、IBTC 的插值对象通常是动作、控制输入或 controller gain。它们至少有明确控制意义。RoboDual 的 `hidden_states` 是 slow generalist 的 latent condition。对它做:

```text
0.25 * new_hidden + 0.75 * old_hidden
```

不保证得到一个 specialist 训练时见过的有效 condition。它可能落到 latent manifold 之外，让 fast specialist 收到 OOD 条件。

第二，本次实验同时混合了 `ref_action` 和 `hidden_states`。

`--slow_handover_steps 4 --slow_handover_blend_hidden` 不是纯 hidden ablation，而是:

```text
ref_action ramp + hidden_state ramp
```

因此负结果不能单独归因于 hidden，也不能证明 ref_action ramp 一定无效。它只能说明这个组合不好。

第三，risk trigger 发生在本来就不稳定的位置。

`risk_balanced` 会在 aggregation delta、jerk、sample variance 或 gripper flip 风险较高时提前刷新 slow guidance。也就是说，handover 往往发生在 specialist 已经不稳定的阶段。在这种阶段再引入 latent interpolation，失败概率会被放大。

第四，当前实现仍然是 immediate refresh 后再 wrapper blend。

slow refresh 后，代码已经把 `self.action` 和 `self.hidden_states` 更新为 new guidance，再在传给 fast specialist 前做混合。这不是 delayed commit，也不是 gated commit。它没有先判断 new guidance 是否和当前执行轨迹 coherent。

第五，hidden blend 缺少安全接入条件。

更合理的 hidden 处理可能不是:

```text
mixed_hidden = alpha * new_hidden + (1 - alpha) * old_hidden
```

而是:

```text
old_hidden -> keep
new_hidden -> pending
if new_ref / predicted action 与当前轨迹足够一致:
    commit new_hidden
else:
    delay or force commit at max age
```

也就是说，hidden 适合做 delayed commit / gated commit，不适合直接做线性 blend。

当前结论:

```text
RTC / AAC / IBTC 仍然值得作为 inference-time 主线参考，但它们应主要作用在 ref_action、最终 action、slow-call schedule 和 commit timing 上。
hidden state 不应再作为线性插值对象；下一步若继续处理 hidden，应把问题改成什么时候安全切换到 new hidden，而不是如何平均 old/new hidden。
```


codex resume 019f405d-93fe-7810-b77b-d90e1697a22a
