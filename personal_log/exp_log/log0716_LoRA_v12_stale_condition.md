# 0716 Transition LoRA V12：部署一致的 stale-condition 训练

## 1. 为什么不继续微调 V5 gate

V7 与 V11 使用相同 V5 checkpoint，但完整 100-sequence 平均长度分别为 `3.01` 与 `3.35`。
把 gate 从 age 8 推迟到 age 12 的主要效果是减少 LoRA 生效步数，使系统回到 baseline；V8 的半幅权重
和 V10 的早期 checkpoint 均未形成单调收益。因此继续调整缩放比例、训练步数或 age threshold 缺乏新的
因果假设。

全流程纵向检查得到三个更根本的问题：

1. 数据只包含 baseline 成功 rollout，target 是 baseline 自己实际执行的 action。它可以做成功条件下的
   再蒸馏，但没有专家纠正或失败恢复信息，理论上不能保证超过 baseline。
2. V5 训练集为 `50% normal + 30% refresh + 10% high_conflict + 10% stale`，而在线 LoRA 只在
   reference 耗尽的 stale 状态启用。大量 supervised gradient 来自部署时不会启用 transition 权重的 age 0
   状态。
3. V5 只修改最后两层 temporal projection。该通路不直接读取 slow hidden，更容易改变所有 action token
   的时间传播和动作幅度；V7 短测 action norm `+9.07%`、完整 chain@4 `-17 pp` 与该风险一致。

独立审阅进一步指出：560 个 stale 训练窗口中，age 8/9/10/11/12 分别只有
`157/131/127/89/56` 条；D 组仅采到 3 条成功轨迹。V5 validation transition loss 的准入改善仅
`3.63e-6`，与闭环大幅变化不成比例，说明原 checkpoint selection metric 缺乏闭环辨识力。

## 2. V12 假设

V12 先使用现有数据进行一个成本受控的机制验证：

> 当训练监督、插层通路和在线 gate 三者都只针对 reference-expired stale 状态时，LoRA 是否能在不放大
> 动作幅度的前提下，复现成功 rollout 中更可靠的 stale continuation？

它仍然没有超过 baseline 的专家标签，因此这不是最终数据方案。如果 V12 不能得到稳定正收益，应停止在
现有成功轨迹上继续调参，转向专家重标注或失败状态 recovery/DAgger。

## 3. 数据与采样

数据继续使用已修复并验证 committed-action target 的：

```text
LoRA_transition_0711/collected_transition_v1_repaired
```

只加载 `normal` 与 `stale`：

- `stale`：提供 action-label diffusion supervision，并使用较弱 frozen-base preservation；
- `normal`：supervised weight 为零，只做严格 frozen-base prediction preservation；
- `refresh/high_conflict`：本轮不进入训练，因为 age 0 状态与 age>=8 部署 gate 不一致。

训练不再按窗口均匀 shuffle。sampler 先给 normal/stale 各 50% 概率质量，再在每个类别内让每条 source
trajectory 获得相同概率质量，最后在轨迹内部均分到窗口。这样避免长轨迹的高度重叠窗口主导梯度。

## 4. 插层与冻结

LoRA rank 2、alpha 2，只插入五个 slow-condition 权重：

```text
model.context_adapter
model.blocks.4.cross_attn.attn.l_proj
model.blocks.4.cross_attn.attn.values_l_proj
model.blocks.5.cross_attn.attn.l_proj
model.blocks.5.cross_attn.attn.values_l_proj
```

冻结 temporal attention、`x_embedder`、final action head、视觉/深度/gripper encoder、history adapter 和
其余 specialist 参数。选择最后两层 cross-attention 是为了直接调整 slow hidden 的 key/value 解释，同时
限制影响范围；不再通过 temporal projection 直接改变动作 token 动力学。

评测器新增显式 `--transition_gate_target_profile stale_condition`。gate off 时，上述五个权重逐 tensor
恢复原 specialist；gate on 时才切换为 merged V12 权重。默认 `temporal_proj` profile 保持 V7-V11 行为
不变。两组 target profile 明确且互不重叠。

## 5. 目标与 checkpoint 准入

默认目标：

```text
normal: 0.0 * supervised + 4.0 * frozen-base prediction drift
stale:  1.0 * supervised + 0.25 * frozen-base prediction drift
drift:  ee6 prediction MSE + 2.0 * gripper prediction MSE
```

teacher/student 使用相同 noise、timestep 和 condition mask，policy 固定为 eval mode，避免 attention dropout
产生伪 drift。checkpoint 必须同时满足：

1. held-out stale supervised loss 至少绝对改善 `2.5e-4`，约为当前 baseline stale loss 的 0.5%；
2. normal prediction drift、overall drift 和 normal gripper drift 不超过既定上限；
3. 在合格 checkpoint 中优先选择 held-out stale supervised loss 最低者；loss 在 `1e-6` 内相同时，
   才使用加权 prediction drift 作为 tie-breaker。

相比 V5 的 `3e-6` 准入改善，新阈值要求可辨认的 stale 拟合收益。训练脚本还强制
`normal_supervised_weight=0` 和 `empty_ref_after_age=8`，不能通过 CLI 绕开部署一致性契约。
若没有 checkpoint 合格，脚本应保留
step-0 base fallback，不能把不合格 final 权重伪装成候选。

## 6. 训练命令

输出目录在启动前必须不存在：

```text
LoRA_transition_0711/lora_runs/transition_lora_v12_stale_condition
```

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
test ! -e LoRA_transition_0711/lora_runs/transition_lora_v12_stale_condition
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  LoRA_transition_0711/train_transition_lora_stale_condition_0716.py \
  --data_dir LoRA_transition_0711/collected_transition_v1_repaired \
  --output_dir LoRA_transition_0711/lora_runs/transition_lora_v12_stale_condition \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --bf16
```

默认 `batch_size=1`、`gradient_accumulation=2`、`max_steps=1200`、learning rate `1e-5`。输出目录
非空时脚本拒绝启动，不使用 `--overwrite_output`。

## 7. 分阶段验证与停止规则

1. 先检查 training summary：若 best step 为 0、未达到 stale improvement 或 merged checkpoint 与 base
   没有五个预期 delta，则停止，不运行闭环。
2. 通过离线检查后，先跑原固定 16 条作为灾难性退化 smoke test，但不再据此选择最佳版本。
3. smoke test 合格后，另选未参与 V7-V11 调参的 holdout sequence，并对 baseline/V12 使用相同 seed
   做配对评测；动作范数、jerk、slow-reference error 与成功率同时准入。
4. 只有 holdout 通过后才运行 canonical 100 条。每个结果使用独立目录，禁止覆盖。

代码与测试：

```text
LoRA_transition_0711/train_transition_lora_stale_condition_0716.py
LoRA_transition_0711/audit_stale_condition_checkpoint_0716.py
LoRA_transition_0711/test_stale_condition_trainer_0716.py
vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py
LoRA_transition_0711/test_gated_evaluator_contract.py
```

## 8. 训练结果与停止决定

V12 于 1200 optimizer steps 正常结束，结果目录：

```text
LoRA_transition_0711/lora_runs/transition_lora_v12_stale_condition
```

baseline stale validation loss 为 `0.03366846`。训练过程中最低值出现在 step 200，为 `0.03364804`，
绝对改善仅 `2.04e-5`；此后指标在 baseline 附近上下波动，step 1200 为 `0.03368018`。所有
preservation constraints 均通过，但没有任何 step 达到 `2.5e-4` 的最低 stale 改善要求。

因此：

```text
best_step = 0
merged_from_adapter_step = 0
best_transition_improvement = 0
```

输出的 merged checkpoint 是明确的 base fallback，不是 V12 candidate。按照预先停止规则，本轮不执行
短序列或 100-sequence 闭环，避免降低阈值后测试近似 no-op。V12 说明：仅把训练类别与部署状态对齐，
在 rank-2 condition path 上仍无法从现有 baseline 成功轨迹中提取可辨认的 stale supervision gain。

下一轮 V13 不修改 gate threshold，而验证另一个独立假设：增加 condition-path 表达能力并显式提高动作
chunk 前两步的训练权重。若 V13 仍不能在独立离线指标上取得实质改善，则停止使用现有数据训练，转向
专家重标注或 recovery 数据采集。
