# 0713 Transition-History LoRA 数据修复与 V2 训练

## 1. V1 在线评测暴露的问题

第一次 LoRA 训练虽然离线 loss 明显下降，但在线成功率严重退化：

```text
平均链长：3.34 -> 1.87
1-chain： 90%  -> 74%
5-chain： 43%  -> 7%
```

进一步检查发现，采集脚本把 action 的 NumPy 引用写入 frame 后，又将同一个数组传给 `env.step()`。CALVIN 环境会原地转换 action，导致保存的 `rel_actions` 从归一化 command 变成了约小 29 倍的物理位姿增量。V1 因此学习了错误尺度的 target，loss 下降实际对应动作幅度收缩。

## 2. 数据清洗

采集端已经改为保存和执行独立的 action copy，防止后续采集再次出现原地修改：

```python
np.asarray(action, dtype=np.float32).reshape(7).copy()
env.step(action.copy())
```

已有数据不需要重新采集。第 `t+1` 帧的：

```text
hist_action_before[-1]
```

保存了第 `t` 步在进入环境前的正确 committed action。因此修复脚本用该关系恢复 target，并丢弃每条轨迹无法恢复的最后一步以及所有不能覆盖完整 8-step target 的窗口。

修复数据目录：

```text
/home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/collected_transition_v1_repaired
```

修复结果：

```text
轨迹：                     482
frames：                   47586
可恢复 committed actions：47104
history 最大连续误差：      0.0
样本：                     8000 个唯一窗口
train/validation/test：    5600/1200/1200

污染 action ee6 L2 均值：  0.01255
修复 action ee6 L2 均值：  0.38621
```

trainer 现在强制读取 `committed_actions/*.npy`，不会静默回退到污染的 `rel_actions`。图像与 slow condition 通过符号链接只读复用原始采集目录，没有重复复制约 11GB 数据。

## 3. V2 训练结果

V2 输出目录：

```text
/home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/lora_runs/transition_history_lora_v2_repaired
```

主要离线结果：

```text
baseline validation overall：0.039107
best validation overall：    0.033318
整体下降：                   14.8%
best step：                  3000
normal_vs_base_ratio：       0.902
```

分类改善：

```text
normal：        9.8%
refresh：       18.5%
high_conflict： 28.4%
stale：         2.6%
```

改善主要集中在 refresh 和 high-conflict，符合 transition-history LoRA 的设计目标；normal 离线 loss 没有退化。Merged EMA 只改变了预定的 14 个 LoRA target，其他 434 个原 specialist tensor 保持不变。离线结果健康，但是否改善真实成功率仍必须由 CALVIN rollout 确认。

## 4. V2 训练启动命令

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

本轮已经训练完成。上述命令仅用于实验复现；不要对已有输出目录使用 `--overwrite_output`。

## 5. V2 在线评测启动命令

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

建议将本轮结果独立归档为：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0713_LoRA_v2_repaired
```

## 6. CALVIN 数据目录检查

这里必须区分两个概念：

```text
A/B/C/D：task-age 调度中的任务分组
calvin_debug_dataset：本机真实 CALVIN 数据集目录
```

评测命令中的参数必须是：

```bash
--dataset_subdir calvin_debug_dataset
```

不能写成：

```bash
--dataset_subdir task_ABC_D
```

本机已经确认以下文件存在：

```text
/home/rosmontis/Projects/dualsys/calvin/dataset/calvin_debug_dataset/validation/.hydra/merged_config.yaml
```

专用评测脚本的 `--dataset_subdir` 默认值也已经改为 `calvin_debug_dataset`。因此当前命令不会再尝试访问不存在的 CALVIN ABCD 数据目录。
