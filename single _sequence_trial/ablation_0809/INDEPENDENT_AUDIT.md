# Independent subagent audit

审查日期：2026-08-09。审查代理：独立 subagent `Tesla`。审查方式：只读源码检查和轻量静态推理；没有修改工作区，也没有运行 GPU rollout。

## 结论

四格 mechanism runner 的 channel 定义、frozen-observation 输入复用、固定 diffusion generator 和 noise-hash 检查符合实验目标。offline age/ref 构造、固定 noise/timestep 和 epsilon-derived x0 计算也符合当前 specialist 实现。

当前目录没有正式 GPU mechanism run artifact；因此 boundary restore、四次实际 noise pairing 和模型输出仍须由用户在有 CUDA 的机器上运行命令验证。这是运行状态限制，不是脚本契约的通过证明。

## 审查中确认的关键点

- `stale_hidden` 来自 P12 age-0 slow call；`fresh_hidden` 来自同一 frozen current observation 的额外 slow call。
- `fresh_ref` 是该额外 slow call 的完整 8-action chunk；所有 fresh/fresh、fresh/empty、stale/fresh 格子都标记为 intervention。
- 四个 specialist call 复用同一 current/previous image、proprio、4-action history、language，并各自注入同一个 diffusion seed；首个 DiT trajectory hash 必须相同，且 global RNG 前后不变。
- P12 reference tail contract 为 age `d<8` 使用 `8-d` 个 slow-action 尾部 action，age `d>=8` 使用全零 reference；因此 7→8 正好是 1→0。
- offline 每个 sample 使用 `noise_seed + sample_id` 的 CPU float32 noise，并固定 `prediction_type=epsilon` 的 diffusion timestep。

## 审查意见与处理

### Gripper / loss 语义

reviewer 提醒：如果启用 `decoupled_loss=True`，最后一维会走 BCE/logit 语义，不能直接用 epsilon x0 公式。当前脚本显式传入 `decoupled_loss=False`，并在 policy load 时要求 scheduler `prediction_type == "epsilon"`；因此本实验的 7 个 channel（包括 gripper）统一是 epsilon MSE，gripper accuracy 是 epsilon-derived `x0_hat` 的 sign accuracy。README 和 manifest 已将这个前提写明。

### Offline provenance

已增加逐行检查：condition 必须标记 `online_current_observation`；`sample_step - condition_step` 必须等于 stored `slow_age`；slow action/hidden/history shape 也必须匹配。不能只靠 collection summary 的文字字段。

### Mechanism completeness

manifest 现在记录 expected observation/event 数；analyzer 在读取结果时要求 observation 和四 condition event 数量完全匹配，并再次检查每个 observation 的四个 noise hash 相同。

## 本地验证

- `verify_ablation_contract.py`：通过；覆盖 reference tail、四格 contrast 公式和 d=0..11 validation coverage。
- 五个新增 Python 文件：通过 `py_compile`。
- `run_mechanism_diagnostic.py --dry_run`：通过，默认生成 1 个 observation / 4 个 condition event 的设计计划。
- `run_offline_age_curve.py --dry_run`：通过，validation d=0..11 均有样本。
- 单样本、age=8、CPU offline smoke：实际完成，生成 `age_curve_summary.json`、CSV/JSONL 和 PNG/SVG；该 smoke 不是正式 age curve 结果。

