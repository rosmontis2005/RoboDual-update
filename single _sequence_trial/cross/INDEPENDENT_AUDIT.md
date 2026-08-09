# Independent agent audit

最终结论：**通过，可启动 GPU pilot；无阻塞项。**

## 已确认

- 所有 Python 文件通过编译，CPU 契约验证退出码为 0。
- S8/S12 来源 step 文件及 SHA256 provenance 有效；当前 `boundary_states.pt` SHA256 为 `78bfa90361b319fa18fd6414d25da47cbeeebcd91f8d10ac26b10c78f2f6c8d8`。
- controller target 重建已正确使用采集后 action 的物理增量，禁止重复乘 `0.02/0.05`。S8/S12 target–TCP lag 分别约 1.94 cm、1.15 cm，符合 target-pose 控制器特征。
- robot/scene observation、TCP 位姿和速度、controller target/姿态、gripper action、arm/gripper joint 位置速度、三个物体位姿与线/角速度、door/button/switch/light 均通过显式恢复审计；重复恢复误差为 0。
- P8/P12 slow-call 和 reference-action 数量与原 collector 一致，并有逐步 fail-fast。
- 每个 cell 在状态恢复后重新设置 Python/NumPy/Torch CPU/全部 CUDA RNG，并调用 `evaluator.reset()`；initial-noise hook 记录首个真实 DiT trajectory tensor，不改变输入或 RNG。
- 四个 simple effects、两个跨因素平均主效应和 difference-in-differences 交互公式正确。
- 紧凑记录与可选完整 tensor trace 的结构合理。

## 审查期间修正

- 初版 controller target 重建错误地对已经被 CALVIN 原地缩放的 saved action 再乘一次 `max_rel_pos/max_rel_orn`。现已改为直接累加米/弧度物理增量，重生成 bundle 并增加 target–TCP lag 防回归检查。
- 恢复审计增加 controller target/姿态/gripper action 以及 door/button/switch/light 显式比较。
- 分析器增加标准 averaged policy main effect 和 averaged state main effect。
- 分析器增加 replicate ID、trial seed 和 restore-audit 文件集合与 manifest 的严格一致性检查。

## 非阻塞边界

历史采集没有保存原生 PyBullet checkpoint，因此 solver warm-start、contact manifold/cache、door/button/switch joint velocity 和未采集 motor target 不能位级恢复。README 已明确说明。本实现精确恢复所有已采集且可设置量，是当前数据允许的最强状态重建，但不声称为 Bullet 内部 bitwise continuation。
