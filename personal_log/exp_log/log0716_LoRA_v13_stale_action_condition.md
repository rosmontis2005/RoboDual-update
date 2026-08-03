# 0716 Transition LoRA V13：前置动作加权的 stale action-condition

## 1. 从 V12 得到的约束

V12 已证明部署一致的 normal/stale 采样和 rank-2 condition-path LoRA 本身不足以产生可辨认收益：最低
stale validation loss 只改善 `2.04e-5`，最终按规则回退 step 0，未进入闭环。V13 是现有 baseline
成功轨迹数据上的最后一次训练机制验证；若仍失败，后续不再降低 loss threshold 或调 gate，而转向专家或
recovery 标签。

## 2. 新假设

fast specialist 每次预测 8 个 action token，但闭环最先执行 chunk 前部，前一到两步的方向与幅度错误会
立即改变下一观测。V5/V12 对 8 步和 7 维平均计算 diffusion loss，可能使与实际闭环最相关的信号被后续
token 稀释。

V13 在保留 V12 normal/stale 轨迹平衡采样的基础上：

1. 前两个 action token 的 diffusion MSE 权重设为 `2.0`，其余为 `1.0`，按权重和归一化；
2. LoRA 增加 `model.x_embedder`，使 stale 分支能适配 noisy action 与全零 reference 的融合；
3. 保留 `context_adapter` 与 blocks 4/5 cross-attention key/value projection；
4. 继续冻结 temporal attention 和 final action head，避免重现 V7 的 temporal 动作幅度漂移；
5. rank/alpha 提升为 `4/4`，learning rate `3e-5`，stale preservation 从 `0.25` 降到 `0.1`；
6. normal supervision 仍强制为零，age 仍强制为 8。

评测时必须使用：

```text
--transition_gate_start_age 8
--transition_gate_target_profile stale_action_condition
```

该 profile 只切换六个训练权重，gate off 恢复原 EMA 权重。

## 3. 准入规则

V13 的 weighted stale validation loss 必须至少改善 `2.0e-4`，同时满足 normal、overall 和 gripper
prediction drift 上限。合格 checkpoint 按最低 weighted stale loss 选择，drift 只用于 loss tie-break。

训练后必须依次满足：

1. `best_step > 0` 且 merged checkpoint 来自该 step；
2. 独立 test stale weighted loss 相对 test base 确有改善；
3. tensor audit 证明 deployed EMA 只有六个预期 weight delta，history 输出精确为零，gate off 全量恢复；
4. 之后才允许固定短序列 smoke test。

## 4. 训练命令

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
test ! -e LoRA_transition_0711/lora_runs/transition_lora_v13_stale_action_condition
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  LoRA_transition_0711/train_transition_lora_stale_action_condition_0716.py \
  --data_dir LoRA_transition_0711/collected_transition_v1_repaired \
  --output_dir LoRA_transition_0711/lora_runs/transition_lora_v13_stale_action_condition \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --bf16
```

输出目录非空时拒绝启动，不使用覆盖参数。

## 5. 训练结果与停止决定

V13 于 1200 optimizer steps 正常结束。weighted stale validation baseline 为 `0.03520992`，最低值出现在
step 1000，为 `0.03509302`，绝对改善 `1.169e-4`，只达到预设最低改善 `2.0e-4` 的 58.5%。
其余多数 validation point 低于 baseline，说明更高 rank、`x_embedder` 和前两步加权增加了可学习性，
但收益仍小且不稳定。

最终状态：

```text
best_unconstrained_step = 1000
best_unconstrained_validation_loss = 0.03509302
best_step = 0
merged_from_adapter_step = 0
best_transition_improvement = 0
```

由于没有 checkpoint 达到准入阈值，merged checkpoint 仍是 base fallback。按训练前约定，不对 step-1000
unconstrained adapter 进行短序列测试，也不事后把阈值从 `2e-4` 降到 `1e-4`。否则是在看到结果后改变
准入规则，并会继续放大 checkpoint-selection overfitting。

V12/V13 连续失败后，现有 baseline 成功轨迹训练路线到此停止。下一轮优化必须给训练引入 baseline 本身
不具备的新信息：专家动作、失败后的成功 recovery branch，或至少成功/失败配对排序信号。
