# 0712 Transition-History LoRA 训练说明

## 1. 训练目标

前面的 RTC/条件插值实验表明，在推理阶段人为延长旧指导、构造固定 ramp，容易降低新 slow guidance 的生效速度，并且难以严格复现一个适合平滑插值的条件。当前实验因此不再直接修改动作输出，而是尝试让 fast specialist 从成功轨迹中学习：在 slow condition 不同年龄、动作历史不同以及新旧指导可能冲突时，如何生成更连续且仍能完成任务的动作。

训练只更新 fast specialist，不训练 generalist。每个样本包含：

```text
当前与前一帧 RGB
当前 depth、gripper RGB/depth、proprio
最近 4 步实际执行动作 hist_action_before [4,7]
采集时真实 slow action [8,7]
采集时真实 slow hidden [tokens,4096]
未来 8 步成功动作 target [8,7]
slow age 与 normal/refresh/high_conflict/stale 类别
```

history 使用的是已经发送给环境的 committed actions，而不是 fast specialist 尚未执行的预测 chunk，避免未来动作泄漏。slow action 和 slow hidden 直接读取采集时保存的 condition，不离线重新运行 generalist。

## 2. condition 与训练样本

训练脚本按照 task-age 评测逻辑重建 ref action：

```text
age = 0:     使用完整 8 步 slow action
age = 1..7:  使用 slow chunk 最后的 8-age 步，并写入 ref 前部
age >= 8:    使用全零 empty ref
```

训练、验证和测试均覆盖四类样本：

```text
normal
refresh
high_conflict
stale
```

目标不是只优化冲突或 stale 样本，而是在学习 transition behavior 的同时保留原 specialist 的正常动作能力。当前数据集共 8000 个样本，按 trajectory 划分为：

```text
train:      5600
validation: 1200
test:       1200
```

需要保留限制说明：任务组 D 的成功轨迹只有 3 条，低于目标 20 条，因此该数据可以用于第一轮训练和整体方法验证，但不能据此稳定判断 `stack_block` 或 D 组是否提升。

## 3. History adapter

新增的 history adapter 将最近 4 步实际动作映射到 DiT hidden size：

```text
[B,4,7] -> Flatten -> Linear -> SiLU -> Linear -> scalar gate
```

最后一个 Linear 采用零初始化，gate 初始为 1。因此 adapter 安装后、训练前是严格的零残差，不会立刻改变原 specialist 输出；第一步反向传播仍可更新输出层。训练时 history adapter 全参数更新。

## 4. LoRA 插入位置

LoRA 仅插入 fast specialist 的以下 14 个线性层：

```text
model.x_embedder
model.context_adapter
model.blocks.0-5.attn_temporal.qkv
model.blocks.0-5.attn_temporal.proj
```

参数为：

```text
rank = 4
alpha = 8
dropout = 0.05
```

完整训练：

```text
model.history_adapter
```

冻结范围包括 vision/depth/gripper adapter、proprio embedder、MLP、cross-attention、final layer、其余 DiT 层以及 generalist。脚本会严格检查 LoRA target 集合和 trainable 参数名，避免误训练视觉编码器、输出头或共享大模块。当前可训练参数约为 128569。

## 5. Batch 与 hidden 长度修正

采集得到的 slow hidden token 数存在 87、88 等长度。原先的 batch=2 方案会复制最后一个 token 做 padding，但当前 DiT 没有 context mask，补出的 token 会参与 `context.mean()` 和 cross-attention，使同一个样本的 condition 随 batch 组成发生变化。

当前训练固定为：

```text
physical batch size = 1
gradient accumulation steps = 2
effective batch size = 2
validation batch size = 1
```

每个样本直接保留原始 slow hidden，不再 padding。collator 如果收到多样本 batch 会直接报错。每两个 micro-batch 的 loss 分别除以 2 后反向传播，再执行一次 gradient clipping 和 optimizer step。

因此 `max_steps=3000` 表示 3000 个 optimizer steps，对应 6000 个训练样本前向/反向，而不是 3000 个 micro-batch。当前 train split 为 5600 条，3000 steps 约为 1.07 个数据遍历周期。

## 6. Validation 与模型选择

训练前先计算冻结 base specialist 的 validation loss。validation 从四类样本中固定抽取各 64 条，并使用固定 diffusion seed，分别记录：

```text
loss_overall
loss_normal
loss_refresh
loss_high_conflict
loss_stale
normal_vs_base_ratio
```

每 100 optimizer steps 验证一次，early-stopping patience 为 5。只有满足：

```text
normal loss / base normal loss <= 1.05
```

的候选才能写入 `adapter_best.pt`。同时保存 `adapter_best_unconstrained.pt`，用于没有候选满足 normal 约束时明确记录最优但有退化风险的模型。最终 merged checkpoint 从选中的 best adapter 生成，不直接使用最后一步权重。

训练日志还记录 effective-batch loss、最近 100 个 micro-batch 的平均 loss、gradient norm、history adapter output norm 和 history gate。best 模型选定后，在独立 test subset 上报告四类 loss，但 test 不参与模型选择。

## 7. 脚本与输出

训练脚本：

```text
/home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/train_transition_lora.py
```

主要输出：

```text
training_config.json
validation_baseline.json
metrics.jsonl
adapter_best.pt
adapter_best_unconstrained.pt
adapter_final.pt
specialist_transition_lora_merged_policy.pt
specialist_transition_lora_merged_ema.pt
training_summary.json
```

正式评测优先使用 `specialist_transition_lora_merged_ema.pt`。

## 8. 训练启动命令

在 `RoboDual` 目录和 `dualsys_env` 环境中执行：

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual

CUDA_VISIBLE_DEVICES=0 \
python LoRA_transition_0711/train_transition_lora.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --data_dir LoRA_transition_0711/collected_transition_v1_repaired \
  --output_dir LoRA_transition_0711/lora_runs/transition_history_lora_v2_repaired \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --max_steps 3000 \
  --learning_rate 3e-5 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --validation_interval 100 \
  --validation_samples_per_category 64 \
  --validation_batch_size 1 \
  --early_stopping_patience 5 \
  --max_normal_loss_ratio 1.05 \
  --save_adapter_steps 500 \
  --bf16
```

首次运行不要加入 `--overwrite_output`。如果输出目录非空，脚本会拒绝启动；只有确认放弃已有 run 时才能使用该参数清空同名目录。

## 9. 专用 task-age 评测

普通 `evaluate_calvin_task_age_0525.py` 没有安装 history adapter，不能直接用于新 checkpoint，否则 adapter 权重可能无法进入模型。新增专用评测脚本：

```text
/home/rosmontis/Projects/dualsys/RoboDual/vla-scripts/evaluate_calvin_task_age_transition_lora_0712.py
```

它会在构造 `DualSystem` 前安装 adapter，使 online 和 EMA 模型结构一致；随后严格加载 checkpoint，检查 history key，并逐 tensor 确认权重进入实际推理使用的 `ema_fast_system.ema_model`。

训练完成后的评测命令：

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual

CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
python vla-scripts/evaluate_calvin_task_age_transition_lora_0712.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_history_lora_v2_repaired/specialist_transition_lora_merged_ema.pt \
  --dataset_subdir calvin_debug_dataset \
  --num_sequences 100 \
  --slow_call_strategy task_age \
  --load_in_4bit \
  --low_cpu_mem_usage
```

正式对比应保持 `--with_cfg` 关闭，与本轮 `cond_drop_chance=0` 的训练配置一致，并使用和 baseline 相同的 seed、任务序列、task-age 分组及 slow-call 参数。

## 10. 当前检查状态

已完成：

```text
训练与评测脚本 py_compile
batch=1 与禁止 hidden padding 的契约测试
raw policy checkpoint 严格加载测试
EMA wrapper checkpoint 严格加载测试
缺少 history adapter key 时失败测试
共 19 个 CPU contract tests
独立 subagent 代码审阅
```

独立审阅未发现阻止正式训练或评测的高严重度问题。正式长训练前仍建议先运行少量 optimizer steps，生成真实 merged checkpoint，并完成一次完整模型的 GPU 加载/短序列评测，以覆盖简化 checkpoint 测试未覆盖的显存、Accelerate 和真实 EMA state-dict 行为。
