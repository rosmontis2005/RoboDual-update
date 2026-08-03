# 0716 Transition LoRA 全局复盘与下一阶段方案

## 1. 结论摘要

本轮完成了 V7 canonical 100-sequence、V7/V11 严格配对，以及 V12/V13 两次新训练机制验证。

- V7：平均长度 `3.01`，相对 baseline `-0.33`，chain@3/4/5 下降 `8/17/9` pp；固定 16 条的
  `+0.4375` 优势没有泛化。
- V11：平均长度 `3.35`，与 baseline `3.34` 基本相同；其恢复来自 age 12 极晚 gate 将 LoRA
  覆盖率压到短测约 `2.64%`，而不是证明 correction 有效。
- V12：stale-only、轨迹平衡、rank-2 condition-path，最好 validation 改善仅 `2.04e-5`，回退 base。
- V13：加入 x-embedding、rank 4 和前两 action token 2 倍权重，最好改善 `1.169e-4`，仍未达到
  `2e-4` 准入阈值，回退 base。

没有新 checkpoint 通过离线准入，因此没有启动额外闭环或 100-sequence。这个停止决定比对不合格权重
反复做 rollout 更重要：当前瓶颈已从模型容量和 gate 参数，转移到监督信息本身。

## 2. LoRA 路线的纵向变化

| 阶段 | 主要做法 | 得到的证据 |
|---|---|---|
| 初始 LoRA | 困难任务成功样本，多层 LoRA/history | 数据强调成功而非平滑，行为明显受损 |
| 数据修复 | 从下一帧 committed history 恢复真实执行 target | 修复了错误 target，但没有增加专家纠正信息 |
| V2 | 14 层 LoRA + history | `lora_only` 和组合均退化，动作趋于欠执行 |
| V3/V4 | 仅最后两层 temporal proj，base preservation | 完整 100 条仍低于 baseline，离线 MSE 与闭环不一致 |
| V5 | matched-noise teacher drift + checkpoint selection | validation gain 只有 `3.63e-6`，却能显著改变闭环动作 |
| V7-V11 | full/half/early checkpoint 与 age gate | 唯一稳定效果是 gate 越晚越接近 baseline |
| V12 | stale-only condition path | 条件与部署对齐仍无法产生可辨认 gain |
| V13 | 更强 condition/action path + front weighting | 可学习性略增，但改善仍小且不稳定 |

这条纵向证据排除了几个简单解释：问题不只是原始数据损坏、不只是 batch hidden mask、不只是 LoRA 插层
太宽、不只是 learning rate 或 checkpoint 选择，也不只是在线 gate 过早。它们都是风险放大器，但不是
缺少正收益的根因。

## 3. 根因判断

### 3.1 最强根因：target 没有超过 baseline 的信息

现有 repaired target 是 baseline 在成功 rollout 中实际 committed 的 action。训练可以提高这些动作在
成功分布上的似然，但无法回答失败状态下“应该怎样改”。失败 trajectory 又没有进入监督数据，因此
模型看不到 baseline 的负例。

V12/V13 是直接证据：即使只训练与部署严格一致的 stale 状态，并把模型容量和首步权重提高，held-out
gain 仍只有 `1e-5` 到 `1e-4`。base specialist 对自身成功动作已经拟合充分，继续 self-imitation 的边际
信息接近零。

### 3.2 训练窗口不等于独立样本

5600 个 train window 只来自 347 条 trajectory，同一轨迹最多产生大量相邻重叠窗口。D 组只有 3 条成功
轨迹，远低于目标 20 条。窗口数量看似充足，但困难状态、失败边界和任务覆盖不足。

### 3.3 diffusion validation 不能单独选择闭环 checkpoint

V5 的 `3.63e-6` loss gain 对应 V7 完整平均长度 `-0.33`；V10 的更早、更小离线 drift 反而是短测最差
版本。固定 noise/timestep validation 适合比较优化过程，却不能替代 denoising rollout、动作分布和环境
闭环评估。

### 3.4 gate 只能控制风险，不能创造 correction

age/ref-valid gate 是确定性安全边界。V11 证明它能将破坏限制到很少的 stale 尾步，但不能判断 LoRA
action 是否比 base 更好。未来 gate 仍应保留，不过其职责应是“何时允许已验证 correction 生效”，而
不是用阈值搜索补偿训练质量。

## 4. 新的优先方向：failure-recovery DAgger

下一阶段唯一优先建议是采集 **失败状态上的成功 recovery branch**，而不是继续收集 baseline 成功轨迹。

### 4.1 数据单元

每个样本以 baseline rollout 的失败边界为中心，保存：

```text
environment state / observation
instruction and task group
slow hidden and slow action
slow age, ref_valid_count, base-ref disagreement
last 4 committed actions
baseline failed action chunk
K 个不同 diffusion seed 或强制 slow refresh 的候选 branch
candidate branch 是否在限定 horizon 内完成当前 subtask
```

只在至少存在一个成功 recovery branch 时形成正负配对。正标签来自同一状态的成功 branch，负标签是
baseline 原失败 branch；这比从其他 demonstration 匹配动作更能控制 observation covariate shift。

### 4.2 采集规模与划分

最小可行版本先选择约 20 个可重复失败状态，每个状态尝试 8-16 个 branch，目标得到 100-200 个成功/失败
配对。确认环境 state restore 可重复后，再扩展到 1000-3000 个窗口。

split 必须按原始 failure episode/state 分组，不能把同一状态的不同 seed 分到 train/test。D 组和 V7/V11
共同退化的长链任务应设最低 state quota，而不是用普通窗口补齐总数。

### 4.3 训练目标

base specialist 永久冻结，优先使用 V13 的 action-condition 六个 LoRA 权重，但从 rank 2 开始：

```text
positive recovery diffusion loss
+ first-two-action weighted loss
+ margin ranking: successful branch score better than failed branch
+ normal/base prediction preservation
+ gripper sign preservation
```

normal 与非触发状态继续要求 zero residual。不要重新启用当前 history adapter；若需要状态特征，单独输入
normalized age、ref_valid_count、base/ref disagreement 和 diffusion variance 到一个小 gate，而不是把 4 步
history 压成单向量后广播到所有 token。

### 4.4 准入评测

1. 离线：held-out failure state recovery ranking、first-action error、action norm、gripper sign；
2. 分支回放：同一 failure state 上的 paired recovery SR；
3. 新 holdout short sequences，至少 3 个 diffusion seed；
4. canonical 100 条；若平均增益很小，baseline 与 candidate 都应重复多个 seed并报告配对区间。

## 5. 次优方向：CALVIN 专家重标注

本地存在 CALVIN training demonstration，可提供真正专家 action；但要用于当前双模型 specialist，需要在同一
demo observation 上重新生成 generalist slow hidden/slow action，并模拟 age>=8 的 empty reference。
这会产生较高 slow-call 预计算成本，且 demonstration observation 与在线失败状态仍有分布差异。

因此专家重标注适合作为 recovery 数据不足时的辅助监督：先采 300-500 个跨任务 stale expert window 验证
pipeline，再扩展到 1500-3000；不应直接替代 failure-state collection。

## 6. 当前可用成果与停止边界

可继续使用：

- repaired committed-action 数据：作为 normal preservation 和成功状态正则；
- V12/V13 condition-path 插层与显式 gate profile；
- trajectory-category balanced sampler；
- front-weighted loss；
- full EMA tensor audit；
- canonical 100-sequence 配对分析器。

不应继续：

- 对 V5/V13 adapter 做 post-hoc 比例缩放；
- 在同一固定 16 条上选择更多 gate age；
- 降低 V13 threshold 后测试 step 1000；
- 仅凭 diffusion loss 或 parameter drift 宣称平滑/成功率改善；
- 在没有新监督信息时继续增加 rank、层数或训练步数。

当前最可信的论文式负结果是：成功轨迹 self-imitation LoRA 和条件门控无法稳定提升该双模型系统；极晚
gate 可以恢复 baseline，但正向 correction 需要 failure-state recovery 或专家监督。
