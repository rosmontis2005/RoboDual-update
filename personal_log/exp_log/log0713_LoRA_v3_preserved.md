# 0713 Transition LoRA V3：窄插层与基线保持训练

## 1. 上一轮结果为什么不理想

固定 16 sequence 消融中，baseline 平均完成长度为 3.438，`history_only` 为 3.312，`lora_only` 为 2.938，`full` 为 2.562。LoRA 主干单独造成 -0.500，history 与 LoRA 组合还有 -0.250 的额外负交互。

V2 同时修改 `x_embedder`、`context_adapter` 和六层 temporal attention 的 qkv/proj，共 14 个线性层。其离线标签 MSE 下降，但 `lora_only` 的动作范数下降 7.7%、reference 耗尽后的动作范数下降 11.5%，slow-reference 误差上升 109.2%。`full` 的动作范数下降 13.4%，slow-reference 误差上升 98.8%。这说明训练把全局策略推向了保守和欠执行，而不是只在 transition 条件下改善平滑性。

旧 checkpoint 筛选只要求 normal 标签 loss 不超过 baseline 的 1.05 倍。这个条件不足以保护原策略：标签 MSE 接近并不代表同一 diffusion 输入上的预测保持一致，更不能阻止小偏移在闭环 rollout 中累积。history adapter 单独也没有产生稳定平滑收益，组合后反而放大退化。

## 2. 修正思路

本轮先采用可直接接入现有评测链路的低风险窄 LoRA，不同时引入新的在线推理结构。

1. 基础 specialist 全部冻结，只在最后两层 temporal attention 的输出投影插入 LoRA：

   - `model.blocks.4.attn_temporal.proj`
   - `model.blocks.5.attn_temporal.proj`

2. 不再训练 `x_embedder`、`context_adapter`、temporal qkv 和前四层 block。LoRA 默认 `rank=2, alpha=2, dropout=0`，学习率从 `3e-5` 降到 `1e-5`。
3. history adapter 仍以零输出形式写入 checkpoint，以兼容 transition 专用评测脚本，但参数完全冻结。该版本在线行为因此属于 `lora_only`，不再承担 V2 的负交互风险。
4. 每个 batch 先运行 student，再临时关闭 LoRA 和 history residual；使用完全相同的 noise、timestep 和可选 classifier-free condition mask 运行 frozen-base teacher。这样得到的 prediction drift 只反映适配器造成的策略偏移，不混入 diffusion 随机误差。
5. `normal` 样本默认不重新拟合 action label，只最小化与 frozen base 的 prediction drift。`refresh/high_conflict/stale` 样本同时优化 action-label diffusion loss 和 base-preservation loss。
6. gripper drift 单独计算并赋予更高权重，防止连续动作看似平滑但夹爪语义被破坏。
7. 训练全程保持 `policy.eval()` 以关闭模型内部 attention dropout，但不会关闭 autograd。否则两次 forward 即使共享 noise/timestep，也会因不同 dropout mask 产生伪 prediction drift；脚本会主动拒绝在 training mode 下计算该目标。

默认目标为：

```text
normal:     4.0 * preservation
transition: 1.0 * supervised + 1.0 * preservation
preservation = ee6 prediction MSE + 2.0 * gripper prediction MSE
```

## 19. V11 结果、候选选择与 100-sequence

V11 平均完成长度 `3.3750`，相对 baseline `-0.0625`；chain 变化为
`0/0/-12.5/0/+6.25` 个百分点，逐序列 improved/equal/worse 为 `4/9/3`。
action norm `+2.09%`、expired norm `+0.97%`、jerk `+3.11%`、aggregation delta `+4.68%`、
slow-reference error `+7.29%`。总共 `7994` 个 base steps、`217` 个 transition steps、
`2318` 个 reference 已过期但仍走 base 的延迟步，提前激活为零。V11 通过平均长度与动作范数门槛，
但 chain@3 和 slow-reference error 未通过，因此 V7-V11 没有一版满足全部短测标准。

按约定从 V7-V11 选择 V11 进入 100-sequence。选择依据是 base-preservation：V11 的平均长度
最接近 baseline，action norm 处于 5% 范围内；V7 虽成功率最高，但 action norm `+9.07%`，
正是本轮要规避的系统性风险。100 条将检验 V11 的 chain@3 与 slow-reference error 是否只是固定 16 条
小样本波动。

gated evaluator 新增 `--full_benchmark`：启用后只接受严格按顺序的 canonical `0..99`，否则拒绝启动；
输出目录仍拒绝非空覆盖。正式结果目录：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0715_LoRA_v11_gated_100seq
```

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0715_LoRA_v11_gated_100seq
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/specialist_transition_lora_merged_ema.pt \
  --transition_gate_start_age 12 \
  --ablation_mode full \
  --save_dir evaluation_results/exp0715_LoRA_v11_gated_100seq \
  --sequence_indices "$(seq -s, 0 99)" \
  --full_benchmark \
  --dataset_subdir calvin_debug_dataset \
  --log_dir exp0715_LoRA_v11_gated_100seq \
  --slow_call_strategy task_age \
  --no_profile_steps \
  --load_in_4bit \
  --low_cpu_mem_usage
```

## 20. V11 100-sequence 完整结果

V11 完整评测正常退出，结果目录包含 100 条 canonical sequence（`0..99`），完整性检查通过。
平均完成长度为 `3.35`，baseline 为 `3.34`，差值 `+0.01`；chain@1/2/3/4/5 为
`92/81/63/55/44%`，baseline 为 `90/77/65/59/43%`，差值为
`+2/+4/-2/-4/+1` 个百分点。

逐序列配对 improved/equal/worse 为 `23/57/20`，净增 `+1` 个完成子任务。配对均值差的
bootstrap 95% 区间为 `[-0.35, +0.38]`，跨过零且范围较宽。因此 V11 达到预设的 baseline
recovery 门槛，但不能据此认为 LoRA 产生了可靠成功率增益。短测中 action norm 仅变化 `+2.09%`，
说明 V11 避免了 V7 的大幅动作偏移；但 slow-reference error 仍为 `+7.29%`，且完整测试未启用
逐步 profile，不能宣称平滑性得到改善。

完整分析产物：

```text
evaluation_results/exp0715_LoRA_v11_gated_100seq/benchmark_summary.json
evaluation_results/exp0715_LoRA_v11_gated_100seq/benchmark_report.md
personal_log/exp_log/log0715_LoRA_v7_v11_summary.md
```

复现分析命令：

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
python LoRA_transition_0711/analyze_full_benchmark_0715.py \
  --candidate_name "V11 gated age12" \
  --baseline_dir evaluation_results/exp0526-0525-task_age \
  --candidate_dir evaluation_results/exp0715_LoRA_v11_gated_100seq \
  --output_dir evaluation_results/exp0715_LoRA_v11_gated_100seq
```

## 18. V10 结果与 V11 极晚门控

V10 平均完成长度 `2.3750`，相对 baseline `-1.0625`；chain 变化为
`-12.5/-18.75/-25/-18.75/-31.25` 个百分点，逐序列 improved/equal/worse 为 `3/7/6`。
action norm `-3.68%`、expired norm `-2.46%`、jerk `+3.47%`、slow-reference error `-3.59%`，
动作保持通过，但成功率严重失败。step-500 adapter 说明实际优化轨迹中的早期参数同样不能由离线 drift
推断闭环行为；它不是 V8 post-hoc 插值问题的简单替代。

V11 是本轮最后一版。恢复 V7/step-1000 full-delta checkpoint，但把 gate 推迟到 age 12：
B 组 max age 12 时每个 slow 周期最多激活 1 步，A 组 max age 13 时最多激活 2 步，
C/D 组 max age 10/8 时完全不激活。该版本最大限度保留 baseline，只在最 stale 尾部保留非零 correction。
若 V11 仍不通过，则按预先约定从 V7-V11 中选择综合表现最佳者进入 100-sequence。

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0715_LoRA_v11_gated_age12_check
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/specialist_transition_lora_merged_ema.pt \
  --transition_gate_start_age 12 \
  --ablation_mode full \
  --save_dir evaluation_results/exp0715_LoRA_v11_gated_age12_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

## 17. V9 结果与 V10 早停 adapter

V9 平均完成长度 `3.0625`，相对 baseline `-0.3750`；chain 变化为
`0/-6.25/-18.75/0/-12.5` 个百分点，逐序列 improved/equal/worse 为 `3/9/4`。
动作保持已恢复：action norm `-1.51%`、expired norm `-3.21%`、jerk `+4.19%`、
aggregation delta `+2.97%`、slow-reference error `+1.36%`。base/transition/expired-base 步数为
`7275/1232/1346`，没有 reference 有效时提前激活。V9 通过动作和 slow-reference 门槛，
但未通过平均成功长度和 chain 门槛。

V10 不再改变 gate，恢复 V7 的 age 8 激活条件；训练侧改为部署 V5 训练轨迹中真实保存的
`adapter_step_500.pt`。该点的 validation transition improvement 为约 `7.28e-7`，
小于 step 1000 的 `3.63e-6`，prediction drift 也更小。与 V8 的 0.5 权重插值不同，step 500
是优化器实际到达的参数点，可用于检验更早停止训练是否能同时保留 correction 与基线能力。

新增 `finalize_transition_adapter_0715.py` 直接按 `B @ A * alpha/rank` 合并 adapter，且只允许
EMA/online 的 blocks 4/5 temporal projection 四个 base tensor 改变；history adapter 输出保持严格为零。

```bash
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  LoRA_transition_0711/finalize_transition_adapter_0715.py \
  --base_checkpoint /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --adapter_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/adapter_step_500.pt \
  --output_dir LoRA_transition_0711/lora_runs/transition_lora_v10_step500

CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v10_step500/specialist_transition_lora_adapter_ema.pt \
  --transition_gate_start_age 8 \
  --ablation_mode full \
  --save_dir evaluation_results/exp0715_LoRA_v10_step500_gated_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

## 16. V8 结果与 V9 延迟门控

V8 retry1 正常完成全部固定 16 条。平均完成长度为 `2.9375`，相对 contemporaneous baseline
`3.4375` 下降 `0.5000`；chain@1/2/3/4/5 变化为
`-6.25/-6.25/-18.75/-12.50/-6.25` 个百分点，逐序列 improved/equal/worse 为 `2/7/7`。
action norm `+5.98%`、expired-reference norm `+8.69%`、jerk `+7.09%`、aggregation delta
`+10.91%`、slow-reference error `+2.21%`。门控 base/transition 步数为 `5870/2269`，
不存在提前激活。V8 同时未通过成功率、chain 和动作范数门槛。

该结果说明对 diffusion 权重做 0.5 插值不能视为闭环行为的单调缩放：它既没有把 expired 区间动作
漂移压到安全范围，也破坏了 V7 的成功率增益。V9 因此恢复使用 V7 的 full-delta V5 checkpoint，
但把门控起点从 age 8 延后到 age 10。age 0-9 均使用精确 base，仅在 reference 已过期至少两步后
启用 LoRA，以减少在线占空比并保留最 stale 区间的 correction。

评测脚本现允许 `transition_gate_start_age=8..13`，安全条件由“激活状态必须等于 reference 过期”
收紧为实际需要的单向约束“激活时 reference 必须已经过期”。这允许 age 8-9 有意继续走 base，
但仍禁止 reference 有效时提前启用 transition 权重。

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0715_LoRA_v9_gated_age10_check
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/specialist_transition_lora_merged_ema.pt \
  --transition_gate_start_age 10 \
  --ablation_mode full \
  --save_dir evaluation_results/exp0715_LoRA_v9_gated_age10_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

## 15. V8 中断后的固定 16 条重跑

V8 首次检查由用户为恢复电脑使用而主动中断，只完成 `1/16` 条，旧目录
`evaluation_results/exp0714_LoRA_v8_gated_half_check/full` 不参与任何统计。重跑使用全新目录
`evaluation_results/exp0714_LoRA_v8_gated_half_check_retry1/full`；启动前必须确认该目录不存在，
避免覆盖或混合中间结果。

V8 仍使用 V7 的 `slow_age_after >= 8` 条件门控，但 transition 分支改用 V6 的 0.5 delta
checkpoint。固定序列、baseline 和准入标准均保持不变：平均完成长度相对 baseline 下降不超过
`0.10`，任一 chain 降幅不超过 `6.25` 个百分点，action norm 变化绝对值不超过 `5%`，
slow-reference error 增幅不超过约 `5%`。只有全部通过才进入 100-sequence。

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v8_gated_half_check_retry1
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v6_scaled_050/specialist_transition_lora_scaled_ema.pt \
  --transition_gate_start_age 8 \
  --ablation_mode full \
  --save_dir evaluation_results/exp0714_LoRA_v8_gated_half_check_retry1/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

## 3. Checkpoint 选择和退出保护

验证集继续对四类样本各固定抽取 64 条。checkpoint 的优化指标改为 refresh、high_conflict、stale 三类 supervised diffusion loss 的平均值，但只有同时满足以下条件才可成为 best：

```text
normal prediction drift <= 2e-4
overall prediction drift <= 5e-4
normal gripper prediction drift <= 1e-4
```

训练开始前会保存严格零 LoRA 的 step 0 checkpoint 作为安全回退。如果后续权重没有同时改善 transition loss 并通过保持约束，最终合并的就是 step 0，脚本不会自动使用 unconstrained checkpoint。`adapter_best_unconstrained.pt` 仅用于诊断。

这一机制解决的是上一轮已经确认的全局策略漂移风险，但不能保证任务成功率上涨。由于窄 LoRA 在推理时仍持续生效，它不是报告中理想的“normal 严格为零、仅 transition 激活”的在线 gated residual。只有本轮先通过固定 16 sequence 的 base-preservation 门槛，才值得进一步实现显式在线门控。

## 4. 实现位置

训练脚本：

```text
/home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/train_transition_lora_preserved_0713.py
```

为支持 matched teacher/student，`DiffusionDiTImagePolicy.compute_loss()` 新增了可选的 `noise`、`timesteps`、`cond_mask` 和 `return_details` 参数；默认调用方式和返回值不变，不影响已有训练及评测脚本。

契约测试：

```text
/home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/test_preserved_training_contract.py
```

## 5. 启动命令

在 `/home/rosmontis/Projects/dualsys/RoboDual` 下运行：

```bash
CUDA_VISIBLE_DEVICES=0 /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  LoRA_transition_0711/train_transition_lora_preserved_0713.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --data_dir LoRA_transition_0711/collected_transition_v1_repaired \
  --output_dir LoRA_transition_0711/lora_runs/transition_lora_v3_preserved \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --max_steps 1500 \
  --learning_rate 1e-5 \
  --lora_rank 2 \
  --lora_alpha 2 \
  --lora_dropout 0 \
  --normal_supervised_weight 0 \
  --transition_supervised_weight 1 \
  --normal_preservation_weight 4 \
  --transition_preservation_weight 1 \
  --gripper_preservation_weight 2 \
  --max_normal_prediction_drift 2e-4 \
  --max_overall_prediction_drift 5e-4 \
  --max_normal_gripper_drift 1e-4 \
  --validation_interval 100 \
  --validation_samples_per_category 64 \
  --early_stopping_patience 8 \
  --bf16
```

### V5 训练结果与固定序列命令

V5 于 `2026-07-14 09:08:25` 正常退出，共 1500 optimizer steps / 3000 micro steps。最终选择 step 1000，而不是最低 validation loss 的 step 1500：

```text
baseline transition loss:     0.0552289008
selected transition loss:     0.0552252718
selected improvement:         3.629e-6
selected drift score:         3.748e-9
unconstrained best step:      1500
merged adapter step:          1000
```

本次 baseline loss 与 V4 不同，说明 diffusion validation 的绝对 loss 不能跨训练进程直接比较。V5 selector 只比较同一进程、同一 validation noise 轨迹中的相对改善，因此仍满足设计契约。输出目录和 V4 完全隔离：

```text
/home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected
```

固定 16 条测试使用新目录，启动前确认该目录不存在：

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v5_check

cd /home/rosmontis/Projects/dualsys/RoboDual
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_ablation_0713.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/specialist_transition_lora_merged_ema.pt \
  --ablation_mode full \
  --save_dir evaluation_results/exp0714_LoRA_v5_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

V8 首次运行因需要恢复电脑使用而由用户主动中断。进程退出码为 `1`，只完成 `1/16` 条，未生成 `result_rank0.json`；该结果无效但保留现场，不删除、不覆盖。后续完整重跑必须使用：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v8_gated_half_check_retry1/full
```

### V7 结果与 V8 gated half-scale

V7 平均长度 `3.875`，相对 baseline `+0.4375`；chain@1/2/3/4/5 变化 `+6.25/+6.25/0/+18.75/+12.5` 个百分点，成功率恢复。gate 审计为 base 5541 步、transition 2362 步，`transition_gate_active` 与 `ref_action_expired` 零错配。

但 action norm `+9.07%`、expired norm `+8.40%`、jerk `+6.96%` 超过既定 5% 动作保持范围；slow-reference error `+4.20%` 合格。因此尚不进入 100 sequence。V8 保持完全相同的 age>=8 gate，只把 transition 分支换为已审计的 0.5 delta checkpoint；正常 age 0-7 仍为精确 base。

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v8_gated_half_check

cd /home/rosmontis/Projects/dualsys/RoboDual
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v6_scaled_050/specialist_transition_lora_scaled_ema.pt \
  --transition_gate_start_age 8 \
  --ablation_mode full \
  --save_dir evaluation_results/exp0714_LoRA_v8_gated_half_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

### V6 结果与 V7 expired-reference gate

V6 fixed-check 平均长度为 `2.9375`，相对 baseline `-0.50`；chain@1/2/3/4/5 分别下降 `6.25/12.5/18.75/6.25/6.25` 个百分点。action norm `+12.08%`、slow-reference error `+8.41%`、jerk `+12.52%`。权重减半没有产生单调恢复，反而放大了闭环动作差异，因此停止继续做 0.25/0.125 权重插值。

V7 改为条件部署：使用 V5 原始 transition 权重，但只在 `slow_age_after >= 8`、即 `num_cond_actions == 0` 时启用；age 0-7 每一步都严格复制 base 的两个 projection weight。gate 条件与 reference-expired 状态不一致时 evaluator 立即报错，逐步 profile 记录 `transition_gate_active`。专用脚本：

```text
/home/rosmontis/Projects/dualsys/RoboDual/vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py
```

固定 16 条命令：

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v7_gated_check

cd /home/rosmontis/Projects/dualsys/RoboDual
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_gated_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/specialist_transition_lora_merged_ema.pt \
  --transition_gate_start_age 8 \
  --ablation_mode full \
  --save_dir evaluation_results/exp0714_LoRA_v7_gated_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

### V5 同期对照结论与 V6 scaled deployment

同期 baseline 正常完成且与历史 fixed baseline 完全一致：平均长度 `3.4375`，逐序列 completion 和动作统计也一致。因此 V5 的 `-0.125` 是有效退化，不再归因于评测随机性。

checkpoint 差分审查发现：V5 的 `ema_model` 只有 block 4/5 temporal projection 两个 weight 改变，符合训练目标；旧保存函数却把 EMA 初始化的 merged policy 整体写入 `online_model`，使 online 分支 290 个非目标 tensor 被 EMA 权重覆盖。评测使用 `ema_model`，所以该打包问题不是 V5 退化原因，但 V6 finalizer 同时修复它。

V6 不重跑相同训练，而把 V5 已选 LoRA delta 缩放到 `0.5`。输出从原 specialist checkpoint 重建：EMA 和 online 各自保留原始权重，只在两个目标 projection 上加入同一个 scaled delta；history compatibility output 保持严格为零。脚本会验证相对 base 恰好只有 4 个 tensor 改变，否则拒绝保存。

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/lora_runs/transition_lora_v6_scaled_050

cd /home/rosmontis/Projects/dualsys/RoboDual
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  LoRA_transition_0711/finalize_transition_lora_scaled_0714.py \
  --base_checkpoint /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --merged_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/specialist_transition_lora_merged_ema.pt \
  --output_dir LoRA_transition_0711/lora_runs/transition_lora_v6_scaled_050 \
  --scale 0.5
```

V6 finalizer 正常退出，相对 base 的 changed keys 恰好为 EMA/online 各自的 block 4/5 temporal projection weight。两个 EMA delta 的 L2 范数由 `0.027362/0.020702` 降为 `0.013681/0.010351`。

固定 16 条评测命令：

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v6_check

cd /home/rosmontis/Projects/dualsys/RoboDual
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_ablation_0713.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v6_scaled_050/specialist_transition_lora_scaled_ema.pt \
  --ablation_mode full \
  --save_dir evaluation_results/exp0714_LoRA_v6_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

首次 V5 fixed-check 在 sequence 开始前失败：沙箱进程无法看到 CUDA，导致 bitsandbytes 拒绝 4-bit 加载。该目录未产生实验结果，已写入 `load_environment_failure.json`，不删除也不原地覆盖。宿主机 `nvidia-smi` 正常，因此改在宿主 GPU 环境使用新目录：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v5_check_retry1/full
```

### V5 fixed-check 初步结果与同期 baseline

V5 fixed-check 正常完成，平均长度 `3.3125`，相对历史 fixed baseline `3.4375` 为 `-0.125`，暂未通过 `-0.10` 门槛。chain@1/2/3/4/5 为 `87.5/68.75/62.5/62.5/50.0%`。逐 sequence improved/equal/worse=`4/9/3`，其中 sequence 65 和 75 分别下降 4、5 个 subtask，主导了净下降；同时也有 4 条改善。

动作诊断相对历史 baseline：action norm `+2.40%`、expired-reference norm `+2.85%`、jerk `+3.07%`、aggregation delta `+3.42%`、slow-reference error `+2.44%`，均在 5% 范围内，不支持系统性动作分布破坏。鉴于训练 validation 在相同 seed 下也出现明显跨进程差异，先在当前环境重跑原始 baseline，而不是直接把少数闭环失败归因于 V5。

同期 baseline 使用独立目录：

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0714_LoRA_v5_check_retry1/base_repeat

cd /home/rosmontis/Projects/dualsys/RoboDual
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_ablation_0713.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/specialist_transition_lora_merged_ema.pt \
  --ablation_mode base \
  --save_dir evaluation_results/exp0714_LoRA_v5_check_retry1/base_repeat \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir base_repeat \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

## 10. V4 100-sequence 结果与下一轮依据

测试进程正常退出，canonical sequence `0..99` 共 100 条全部完成；`success_rate_rank0.txt` 恰好 100 行，`result_rank0.json` 可正常解析。

| experiment | avg seq len | chain@1 | chain@2 | chain@3 | chain@4 | chain@5 |
|---|---:|---:|---:|---:|---:|---:|
| 0525 task-age baseline | 3.34 | 90% | 77% | 65% | 59% | 43% |
| V4 transition-LoRA | 3.19 | 86% | 75% | 60% | 54% | 44% |
| delta | -0.15 | -4pp | -2pp | -5pp | -5pp | +1pp |

逐 sequence 比较为 improved/equal/worse=`25/49/26`，净少完成 15 个 subtask。V4 没有复现早期 LoRA 的根本性退化，且完整五任务链增加 1 条；但零成功序列由 10 条增至 14 条，四任务成功序列由 16 条降至 10 条，因此平均长度和 chain@1-4 仍低于 baseline。按照事先约定的 `avg_seq_len delta >= -0.10` 标准，本轮未恢复 baseline。

完整报告：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0713_LoRA_v4_100seq/benchmark_report.md
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0713_LoRA_v4_100seq/benchmark_summary.json
```

V4 selector 选择 step 2500，因为 transition validation loss 从 baseline `0.0403511` 降至 `0.0403282`；但 normal prediction drift 也从 step 500 的 `1.28e-6` 增至 step 2500 的 `2.93e-6`。虽然没有触及旧的宽松 drift 上限，100-sequence 闭环结果说明“约束内 transition loss 最小”不足以选择 checkpoint。下一轮不扩大插入层和 rank，也不继续增大学习率，而应把较小 drift 纳入选择分数，优先保留较弱、非零的 adapter，再走固定 16 条准入。

## 11. V5 drift-aware checkpoint selection

新增脚本：

```text
/home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/train_transition_lora_drift_selected_0714.py
```

V5 保持 V4 的数据、最后两层 temporal attention output projection、rank 2、学习率和 preservation objective 不变，只修改 checkpoint selection。候选先满足：

```text
transition improvement >= 3e-6
原有三项 preservation constraints 全部通过
```

再最小化：

```text
normal_drift + overall_drift + 2 * normal_gripper_drift
```

该规则避免用很小的离线 loss 优势交换更大的基线行为漂移。根据 V4 的确定性验证轨迹，step 500 的 selection score 约 `3.12e-6`，step 2500 约 `7.86e-6`，因此预期选择 step 500。max steps 缩短为 1500；这不是增加训练强度，而是验证新 selector 能稳定选择较弱的非零 adapter。

启动前必须确认输出目录不存在，且不使用 `--overwrite_output`：

```bash
test ! -e /home/rosmontis/Projects/dualsys/RoboDual/LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected

cd /home/rosmontis/Projects/dualsys/RoboDual
CUDA_VISIBLE_DEVICES=0 /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  LoRA_transition_0711/train_transition_lora_drift_selected_0714.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --data_dir LoRA_transition_0711/collected_transition_v1_repaired \
  --output_dir LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --max_steps 1500 \
  --learning_rate 3e-5 \
  --lora_rank 2 \
  --lora_alpha 2 \
  --lora_dropout 0 \
  --normal_supervised_weight 0 \
  --transition_supervised_weight 1 \
  --normal_preservation_weight 2 \
  --transition_preservation_weight 0.5 \
  --gripper_preservation_weight 2 \
  --selection_min_transition_improvement 3e-6 \
  --selection_normal_drift_weight 1 \
  --selection_overall_drift_weight 1 \
  --selection_normal_gripper_drift_weight 2 \
  --validation_interval 100 \
  --validation_samples_per_category 64 \
  --min_steps_before_early_stopping 1500 \
  --bf16
```

## 9. V4 训练结果与在线准入检查

V4 完整运行 3,000 optimizer steps，没有提前停止；`best_step=2500`，最终 merged EMA 也来自 step 2500，不再是 step 0 回退。相对 frozen baseline，best validation 的 transition 三类均值下降约 0.0567%，独立 test subset 的 overall supervised loss 下降约 0.0431%，四类 test loss 方向均为改善，其中 high-conflict 改善约 0.1307%。

best checkpoint 的 normal/overall/normal-gripper prediction drift 分别为 `2.93e-6`、`3.02e-6`、`9.52e-7`，仅占对应门槛的 1.47%、0.60%、0.95%。EMA 合并后仅 blocks 4/5 的 temporal projection weight 发生变化，history adapter 输出仍为零。

上述改善幅度很小，不能直接推断闭环成功率。下一步使用与 `exp0713_LoRA_v_check/base` 完全相同的 16 条固定序列：

```text
3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95
```

V4 精简检查输出保存到：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0713_LoRA_v4_check/full
```

启动前已经确认该目录不存在，不会覆盖旧 baseline、V3/V4 训练结果或前一轮消融结果。由于 V4 history adapter 严格为零，只运行完整加载 checkpoint 的 `full` 模式；此时其行为等价于本 checkpoint 的 `lora_only`。

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_ablation_0713.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v4_preserved/specialist_transition_lora_merged_ema.pt \
  --ablation_mode full \
  --save_dir evaluation_results/exp0713_LoRA_v4_check/full \
  --sequence_indices 3,11,20,21,28,35,36,53,59,65,75,83,86,89,91,95 \
  --dataset_subdir calvin_debug_dataset \
  --log_dir full \
  --slow_call_strategy task_age \
  --profile_sample_var_ages '' \
  --load_in_4bit \
  --low_cpu_mem_usage
```

### V4 固定 16 条准入结果

V4 平均完成长度为 `3.5000`，baseline 为 `3.4375`，变化 `+0.0625`。逐序列 improved/equal/worse 为 `3/8/5`。chain-1/2/3/4/5 的变化分别为 `+6.25/0/-6.25/+12.5/-6.25` 个百分点，没有超过“任一 chain 下降不超过一个样本”的限制。

动作统计相对 baseline：action norm `+0.87%`，expired-reference action norm `+3.80%`，jerk `-0.14%`，aggregation delta `+2.51%`，slow-reference error `+3.69%`。未出现 V2 的动作幅度收缩或 slow-reference error 翻倍。四项预设 gate 全部通过，因此不进入 V5 重训，转入 100-sequence 正式基准。

完整准入报告：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0713_LoRA_v4_check/gate_report.md
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0713_LoRA_v4_check/gate_summary.json
```

### 100-sequence 基准

旧 0712 evaluator 会直接覆盖共享 `evaluation_results/result_rank0.json` 等文件，因此新增专用脚本：

```text
/home/rosmontis/Projects/dualsys/RoboDual/vla-scripts/evaluate_calvin_task_age_transition_lora_100seq_0713.py
```

它继承固定序列 evaluator 的独立 `--save_dir` 和非空目录拒绝覆盖机制，并严格要求 canonical indices `0..99` 恰好 100 条。正式输出目录在启动前确认不存在：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0713_LoRA_v4_100seq
```

启动命令：

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual-matplotlib \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  vla-scripts/evaluate_calvin_task_age_transition_lora_100seq_0713.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_lora_v4_preserved/specialist_transition_lora_merged_ema.pt \
  --ablation_mode full \
  --save_dir evaluation_results/exp0713_LoRA_v4_100seq \
  --sequence_indices "$(seq -s, 0 99)" \
  --dataset_subdir calvin_debug_dataset \
  --log_dir exp0713_LoRA_v4_100seq \
  --slow_call_strategy task_age \
  --no_profile_steps \
  --load_in_4bit \
  --low_cpu_mem_usage
```

输出目录非空时脚本会拒绝启动；只有明确加入 `--overwrite_output` 才会清空重跑。训练只读取修复后的 transition 数据和 specialist/generalist 模型，不访问 CALVIN `task_ABC_D` 或 `task_ABCD` 数据目录，因此不会复现此前错误访问 CALVIN ABCD validation 配置的问题。

## 6. 预期效果和评测顺序

预期的好结果不是离线 loss 大幅下降，而是 transition loss 小幅改善、normal prediction drift 接近零，并且固定 16 sequence 中动作范数保持在 baseline 的 5% 范围内、slow-reference 误差不再明显上升。成功率平均长度相对 baseline 的下降不得超过 0.10，任一 chain success rate 不得下降超过 6.25 个百分点。

坏结果包括：best 始终停留在 step 0；transition loss 改善但保持约束不通过；离线 drift 很小但固定序列仍下降；或者动作范数再次系统性收缩。前两种说明现有数据/目标无法在窄参数空间中提供有效修正，后两种说明 diffusion prediction drift 仍不足以替代闭环保持指标，此时不应扩大 LoRA 层数，而应转向带在线条件门控的独立 residual correction 分支。

训练完成后先复用 `exp0713_LoRA_v_check` 的同一组 16 sequence 做 base 与新 checkpoint 的配对检查；通过准入标准后再运行 100 sequence 正式评测。

## 7. 实现验证

实现完成后执行了真实 GPU 1-step 冒烟测试，覆盖修复数据读取、两次 matched forward、反向传播、梯度累积、验证、adapter 选择及 merged policy/EMA 保存。实际可训练参数只有最后两层 proj 的 4 个 LoRA tensor，共 2,048 参数；history output norm 为 0。该次测试没有达到 transition improvement 的 `min_delta`，最终正确选择并合并 step 0，而不是 unconstrained 权重，验证了安全回退路径。

独立审阅确认默认启动命令无阻断问题，并发现非零 `cond_drop_chance` 下 condition mask 原先没有 matched。现已将 student 生成的 `cond_mask` 与 noise/timestep 一同传给 teacher。最终 Python 测试覆盖 target 集合、deterministic eval-mode、student-only LoRA 梯度、history 零输出、采样/数据契约和 checkpoint 加载契约。

## 8. V3 训练结果与 V4 参数修正

V3 实际在 step 800 提前停止，最终 `best_step=0`、`merged_from_adapter_step=0`。其 validation transition loss 从 baseline 的 `0.0403510974` 最多只下降到 step 400 的 `0.0403506190`，改善约 `4.78e-7`，远小于旧 `early_stopping_min_delta=1e-4`。训练只读取了 1,600 个 micro-batch，相当于约 0.286 个训练集 epoch，因此最终 merged EMA 是安全回退的原始 specialist EMA，不是有效 LoRA。

V4 保留两层 proj、rank 2、matched teacher、gripper 保护和 step 0 回退，只修改优化强度与退出逻辑：

```text
output_dir:                         transition_lora_v4_preserved
learning_rate:                      1e-5 -> 3e-5
max_steps:                          1500 -> 3000
normal_preservation_weight:         4.0 -> 2.0
transition_preservation_weight:     1.0 -> 0.5
early_stopping_min_delta:           1e-4 -> 1e-6
early_stopping_patience:            8 -> 10
min_steps_before_early_stopping:    2000
```

step 2000 对应约 4,000 个 micro-batch，即约 0.714 epoch；完整 step 3000 对应约 1.071 epoch。warmup 期间 stale counter 不累计，避免再次只训练不到三分之一轮就退出。prediction drift 准入阈值保持不变，因此增大学习强度不等于放弃基线保护。

V4 启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  LoRA_transition_0711/train_transition_lora_preserved_0713.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --data_dir LoRA_transition_0711/collected_transition_v1_repaired \
  --output_dir LoRA_transition_0711/lora_runs/transition_lora_v4_preserved \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --max_steps 3000 \
  --learning_rate 3e-5 \
  --lora_rank 2 \
  --lora_alpha 2 \
  --lora_dropout 0 \
  --normal_supervised_weight 0 \
  --transition_supervised_weight 1 \
  --normal_preservation_weight 2 \
  --transition_preservation_weight 0.5 \
  --gripper_preservation_weight 2 \
  --max_normal_prediction_drift 2e-4 \
  --max_overall_prediction_drift 5e-4 \
  --max_normal_gripper_drift 1e-4 \
  --validation_interval 100 \
  --validation_samples_per_category 64 \
  --early_stopping_min_delta 1e-6 \
  --early_stopping_patience 10 \
  --min_steps_before_early_stopping 2000 \
  --bf16
```
