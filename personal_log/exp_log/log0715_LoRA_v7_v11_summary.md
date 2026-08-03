# 0715 Transition LoRA V7-V11 实验总结

## 1. 实验目标与停止条件

本轮目标不是仅提高固定短序列成功率，而是在不破坏 baseline 动作分布的前提下部署 transition LoRA。
固定 16 sequence 同时检查成功率、chain success rate、动作范数、slow-reference error 和 gate 状态一致性。
若 V8-V11 均未通过，则从 V7-V11 中选择最接近 baseline 且动作漂移可控的版本完成 canonical
100-sequence 评测。

## 2. 固定 16 sequence 结果

| 版本 | 主要设置 | 平均长度 | 相对 baseline | chain@1/2/3/4/5 (%) | action norm | slow-ref error | 结论 |
|---|---|---:|---:|---|---:|---:|---|
| baseline | 无 LoRA | 3.4375 | 0 | 93.75/81.25/75.00/50.00/43.75 | 0 | 0 | 参照 |
| V7 | full delta，age 8 | 3.8750 | +0.4375 | 100.00/87.50/75.00/68.75/56.25 | +9.07% | +4.20% | 成功率通过，动作幅度失败 |
| V8 | 0.5 delta，age 8 | 2.9375 | -0.5000 | 87.50/75.00/56.25/37.50/37.50 | +5.98% | +2.21% | 失败 |
| V9 | full delta，age 10 | 3.0625 | -0.3750 | 93.75/75.00/56.25/50.00/31.25 | -1.51% | +1.36% | 成功率失败 |
| V10 | step-500 adapter，age 8 | 2.3750 | -1.0625 | 81.25/62.50/50.00/31.25/12.50 | -3.68% | -3.59% | 严重失败 |
| V11 | full delta，age 12 | 3.3750 | -0.0625 | 93.75/81.25/62.50/50.00/50.00 | +2.09% | +7.29% | 最接近 baseline，仍未全通过 |

V7 的 transition gate 覆盖约 `29.9%` 的短测动作步，成功率提高但动作范数增加 `9.07%`，与本轮
“恢复基线且控制动作偏移”的目标冲突。V8 的半幅权重并没有形成半幅行为效果，反而同时损失成功率和
动作保持。V9 通过推迟 gate 将覆盖率降至约 `14.5%`，仍未恢复成功率。V10 证明更早训练 checkpoint
的较小离线参数漂移不能预测闭环表现。

V11 仅在 age 12 之后启用 full delta，短测中 `217/8211` 步进入 transition 分支，覆盖率约 `2.64%`；
提前激活计数为零。它的成功率和动作范数最接近 baseline，因此在 V7-V11 均未满足全部短测门槛后，
按预设停止条件选为完整评测候选。该选择优先控制 V7 已暴露的动作幅度风险，并非宣称 V11 的短测成功率
优于 V7。

## 3. V11 100-sequence 结果

评测目录：

```text
/home/rosmontis/Projects/dualsys/RoboDual/evaluation_results/exp0715_LoRA_v11_gated_100seq
```

进程正常退出，`result_rank0.json` 和 `success_rate_rank0.txt` 均生成。分析脚本从累计成功率恢复逐序列
完成数，并确认 candidate 与 baseline 均严格覆盖 canonical sequence `0..99`。

| 模型 | 平均长度 | chain@1 | chain@2 | chain@3 | chain@4 | chain@5 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3.34 | 90% | 77% | 65% | 59% | 43% |
| V11 | 3.35 | 92% | 81% | 63% | 55% | 44% |
| V11 - baseline | +0.01 | +2 pp | +4 pp | -2 pp | -4 pp | +1 pp |

逐序列 improved/equal/worse 为 `23/57/20`，净增 `+1` 个子任务。序列级完成长度差的标准差为
`1.86`，均值差 `+0.01` 的配对 bootstrap 95% 区间为 `[-0.35, +0.38]`。V11 通过预设的
baseline recovery 门槛：平均长度下降不超过 `0.10`，任一 chain 下降不超过 `6.25` 个百分点。
不过该结果不支持“LoRA 提升成功率”的结论，因为净变化极小，置信区间跨零，chain@3 和 chain@4
仍分别低于 baseline `2`、`4` 个百分点。

历史 V4 完整结果为平均长度 `3.19`、chain `86/75/60/54/44%`。V11 明显恢复了 V4 的整体退化，
说明极晚条件门控能够限制 LoRA 对正常策略的破坏；但 V11 只有极少数动作实际使用 transition 权重，
其“恢复”主要来自保留 baseline，而不是证明 transition correction 本身有效。

## 4. 结论

1. V7 展示了成功率与动作幅度之间的明显交换关系，不能作为平滑改进版本使用。
2. 权重缩放、提前 checkpoint 和简单调整 age gate 均没有产生稳定、单调的闭环收益，离线 drift 或 loss
   不能替代 rollout 评估。
3. V11 在 100 sequence 上可以视为恢复 baseline，且短测 action norm 处于 5% 约束内；但短测
   slow-reference error 增加 `7.29%`，完整测试又未记录逐步 profile，因此不能声称达到平滑目标。
4. 本轮最严格的结论是：极稀疏门控可以将 LoRA 的行为损伤压回基线波动范围，但尚未证实 LoRA 对成功率
   或平滑性存在可靠正向效果。

## 5. 后续改进方向

不建议继续在同一固定 16 条上调 checkpoint 比例或 age 阈值，这会逐渐把候选选择拟合到短测集合。
下一轮应先明确可检验的平滑目标，再训练与评测：

1. 从实际部署 rollout 中收集 reference 刚耗尽及 slow guidance 刚刷新附近的状态，构造相同状态、噪声和
   condition 下的 base/target 配对 residual，减少离线样本与闭环触发状态错位。
2. 将 correction 目标限制为连续 6 维，并为 gripper 保留离散语义；对 normal 和未触发状态施加强制
   zero-residual 约束。
3. checkpoint 选择同时使用 transition 标签误差、normal prediction drift、动作范数、jerk 和
   slow-reference error，不能只看训练 loss。
4. 若要主张小幅收益，应对 baseline 与候选使用相同 canonical 100 条运行多个随机种子，并报告配对区间；
   单次 `+0.01` 不足以区分真实效果与闭环随机波动。

## 6. 分析产物

```text
LoRA_transition_0711/analyze_full_benchmark_0715.py
evaluation_results/exp0715_LoRA_v11_gated_100seq/benchmark_summary.json
evaluation_results/exp0715_LoRA_v11_gated_100seq/benchmark_report.md
```

复现命令：

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
python LoRA_transition_0711/analyze_full_benchmark_0715.py \
  --candidate_name "V11 gated age12" \
  --baseline_dir evaluation_results/exp0526-0525-task_age \
  --candidate_dir evaluation_results/exp0715_LoRA_v11_gated_100seq \
  --output_dir evaluation_results/exp0715_LoRA_v11_gated_100seq
```
