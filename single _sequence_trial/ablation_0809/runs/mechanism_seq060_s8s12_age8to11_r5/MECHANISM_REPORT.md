# Seq060 / Subtask 5 mechanism diagnostic 简报

## 结论

在现 P12 的 `stale hidden + empty ref` 上下文附近，刷新 hidden 与补回 fresh ref 对 specialist 首个动作的影响几乎同量级：EE6 RMS 分别为 **0.125** 和 **0.130**。因此，P12 在 age 8–11 的变化不能只归因于 empty ref 或只归因于 stale hidden；两个通道都产生了实质影响。

更关键的是，两通道并非可加的独立误差源。补回 fresh ref 后，hidden 的 stale→fresh 效应从 **0.125** 降到 **0.022**（下降 82.4%）；反过来，在 fresh hidden 下补 ref 的效应从 **0.130** 增至 **0.222**。二阶交互项 RMS 为 **0.127**，与两个 P12 局部主效应本身相当。这说明 fresh action chunk 与 fresh hidden 是一个强耦合、内部一致的条件对；fresh ref 基本屏蔽了 specialist 对 hidden 新旧的敏感性。

沿 age 观察能够进一步分开两种机制：

- `hidden effect | empty ref` 从 age 8 的 **0.092** 增至 age 11 的 **0.155**，配对增量为 **+0.063**，seed-cluster bootstrap 95% CI [+0.043, +0.082]。
- `ref effect | stale hidden` 从 **0.139** 变为 **0.126**，配对增量仅 **-0.013**，95% CI [-0.025, +0.001]。

这支持一个更精确的描述：**进入 empty-ref 窗口会带来近似稳定的通道缺失效应，而 hidden staleness 的影响会随 slow age 继续累积；二者通过强交互共同决定动作。**

## 数据与核验

- 80 个冻结 observation：10 个 trial seed × 2 个 boundary（S8/S12）× 4 个 age（8–11）。
- 每个 observation 包含四个完整 condition，共 320 个 event；四格 condition、metadata 和数量均通过检查。
- 每组四次 specialist 调用使用相同 current observation、history、language 和 diffusion noise；固定噪声哈希检查无失败。
- S8/S12 上 hidden 局部效应均值分别为 0.140/0.111，ref 局部效应为 0.133/0.127，主要结论跨两个 boundary 方向一致。

## 解释边界

这里衡量的是 specialist **raw first diffusion action** 的配对变化，不经过 temporal aggregation，也不等价于 rollout success。除 `stale hidden + empty ref` 外，其余三格均是人为 channel intervention，不能解释为可部署策略的性能。实验只覆盖 seq060/subtask 5、一个固定 diffusion seed 和 age 8–11，因此结论应表述为该局部机制诊断的证据，而不是全任务总体因果结论。

关键图：`mechanism_key_conclusion.png`。误差带/误差条为按 10 个 trial seed 聚类重采样的 percentile bootstrap 95% CI。
