# 0716 Transition LoRA V7-V11 完整对比

## 1. 对比范围

V7 和 V11 使用完全相同的 V5 LoRA checkpoint：

```text
LoRA_transition_0711/lora_runs/transition_lora_v5_drift_selected/
specialist_transition_lora_merged_ema.pt
```

两者唯一核心差异是部署门控：V7 在 `slow_age_after >= 8` 时启用 transition 权重，V11 推迟到
`slow_age_after >= 12`。两次完整评测均使用 seed 42、task-age baseline 配置以及 canonical
sequence `0..99`，因此可以进行逐序列配对。

结果目录：

```text
V7:  evaluation_results/exp0716_LoRA_v7_gated_100seq
V11: evaluation_results/exp0715_LoRA_v11_gated_100seq
base: evaluation_results/exp0526-0525-task_age
```

V7 进程正常退出，`result_rank0.json` 和 100 行 `success_rate_rank0.txt` 完整生成；分析器确认
sequence ID 严格覆盖 `0..99`。

## 2. 完整 100-sequence 结果

| 策略 | 平均长度 | chain@1 | chain@2 | chain@3 | chain@4 | chain@5 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3.34 | 90% | 77% | 65% | 59% | 43% |
| V7，gate age 8 | 3.01 | 91% | 77% | 57% | 42% | 34% |
| V11，gate age 12 | 3.35 | 92% | 81% | 63% | 55% | 44% |
| V7 - baseline | -0.33 | +1 pp | 0 pp | -8 pp | -17 pp | -9 pp |
| V11 - baseline | +0.01 | +2 pp | +4 pp | -2 pp | -4 pp | +1 pp |
| V7 - V11 | -0.34 | -1 pp | -4 pp | -6 pp | -13 pp | -10 pp |

V7 相对 baseline 的逐序列 improved/equal/worse 为 `25/47/28`，净减少 `33` 个完成子任务；
平均差 bootstrap 95% 区间为 `[-0.71, +0.05]`。V7 相对 V11 为 `25/41/34`，净减少
`34` 个完成子任务，区间为 `[-0.73, +0.05]`。区间因单条序列完成长度的高方差仍覆盖零附近，
但 chain@3-5 的一致下降说明 V7 的主要问题是长链累积退化，而不是第一项任务无法启动。

## 3. 固定 16 条为何产生相反判断

固定 16 条中，V7 平均长度为 `3.875`，比同集合 baseline 高 `0.4375`；完整 100 条中却比 baseline
低 `0.33`。这不是小幅回归，而是结论方向反转。

固定集合原本用于快速发现灾难性退化，但经过 V7-V11 多轮 checkpoint、缩放比例和 gate age 选择后，
它也参与了候选决策，已不再是独立测试集。16 条里一次成败会改变 chain success rate `6.25` 个百分点，
且闭环 diffusion rollout 的序列级差值标准差约为 `1.9` 个子任务。V7 的短测优势因此不能外推为总体优势。

后续固定 16 条只能作为安全 smoke test。候选选择至少需要：

1. 将开发序列、准入序列和最终 100 条严格分离；
2. 对短测使用多个 diffusion seed，报告配对均值而不是单次最高值；
3. 不再根据同一 16 条反复调整 checkpoint 或 gate 后仍把它称为验证集。

## 4. V7-V11 门控迭代带来的效果

| 版本 | 权重版本 | gate | 短测 transition 覆盖 | 主要观察 |
|---|---|---:|---:|---|
| V7 | V5 full delta | age 8 | 29.9% | 短测成功率高，但动作范数 +9.07%；完整长链明显退化 |
| V8 | V5 0.5 delta | age 8 | 27.9% | 权重半幅没有产生可预测的行为半幅，成功率下降 |
| V9 | V5 full delta | age 10 | 14.5% | 动作分布接近 baseline，但短测成功率仍未恢复 |
| V10 | step-500 delta | age 8 | 29.2% | 更小离线 drift 的早期 checkpoint 闭环表现最差 |
| V11 | V5 full delta | age 12 | 2.64% | 动作范数恢复，100 条与 baseline 基本持平 |

这些版本没有形成“delta 越小或 gate 越晚，correction 质量越好”的单调关系：V8 的权重缩放和 V10 的
早期 checkpoint 都失败。唯一稳定趋势是 gate 越晚，LoRA 影响的动作步越少，最终行为越接近 baseline。

因此 V11 的作用应解释为 **blast-radius control**：task group C/D 完全不启用，A/B 只在最晚的 stale
尾部启用一到两步。它有效限制了 LoRA 对原策略的损害，但没有证明 V5 correction 本身更准确。V11
完整结果净增仅一个子任务，配对区间跨零，也支持“主要恢复 baseline”而非“LoRA 带来增益”的判断。

## 5. 对训练策略的直接启示

V5 的训练类别为 `50% normal + 30% refresh + 10% high_conflict + 10% stale`，而 V7-V11 的 LoRA
只在 reference 耗尽后的 stale 状态部署。训练与部署不一致：大量 supervised gradient 来自 age 0 的
refresh/high-conflict，真正匹配 V7 age>=8 的 stale 训练样本只有 560 条；直接匹配 V11 age 12 的只有
56 条。

同时 V5 只修改最后两层 temporal attention output projection。该通路改变所有 action token 的时间传播，
但不直接读取 slow hidden，容易表现为动作幅度和长链动力学改变。V7 的 action norm `+9.07%` 与完整
chain@4 `-17 pp` 正是这一风险的闭环表现。

下一轮不再继续调整 V5 的缩放比例或 age threshold，而应验证新的因果假设：

1. 只用与部署一致的 stale 状态提供 LoRA supervision；normal/refresh 仅用于 base preservation；
2. 优先适配 `x_embedder` 与最后两层 cross-attention 的 slow-condition projection，冻结 temporal
   projection 和 final action head；
3. 按轨迹而不是按高度重叠窗口平衡采样，降低少数长成功轨迹的重复权重；
4. checkpoint 选择要求 stale held-out loss 有实质改善，同时约束 normal drift、首步动作误差、动作范数
   和 gripper 符号，而不是接受 `3.63e-6` 量级的 diffusion loss 改善；
5. 若现有 baseline 成功轨迹监督仍不能提高闭环表现，再转向专家重标注或失败状态 recovery/DAgger，
   因为现有标签本身不包含超过 baseline 的纠正信息。

## 6. 复现命令与分析产物

V7 完整评测命令：

```bash
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
  --save_dir evaluation_results/exp0716_LoRA_v7_gated_100seq \
  --sequence_indices "$(seq -s, 0 99)" \
  --full_benchmark \
  --dataset_subdir calvin_debug_dataset \
  --log_dir exp0716_LoRA_v7_gated_100seq \
  --slow_call_strategy task_age \
  --no_profile_steps \
  --load_in_4bit \
  --low_cpu_mem_usage
```

分析产物：

```text
evaluation_results/exp0716_LoRA_v7_gated_100seq/benchmark_summary.json
evaluation_results/exp0716_LoRA_v7_gated_100seq/benchmark_report.md
evaluation_results/exp0716_LoRA_v7_gated_100seq/comparison_vs_v11/benchmark_summary.json
evaluation_results/exp0716_LoRA_v7_gated_100seq/comparison_vs_v11/benchmark_report.md
```
