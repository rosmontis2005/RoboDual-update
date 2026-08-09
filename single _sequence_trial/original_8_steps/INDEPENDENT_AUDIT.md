# Independent agent audit

审查范围：`original_8_steps/` 采集实现、历史 `vla-scripts/dual_sys_evaluation.py`、generalist remote code、specialist diffusion/DiT 路径。

## 最终结论

**有条件通过。** 当前实现可以作为新的 standalone fixed-mod-8 采集基线；未发现采集 hook 会改变执行动作、temporal aggregation 或 Torch RNG 流的问题。它不能被称为历史 0413 seq60 的 bitwise replay，正式确认本次 seq60 仍然 5/5 需要真实 GPU rollout。

## 已确认

- `OriginalFixedMod8Evaluation.step` 与旧版 `DualSystemCalvinEvaluation.step` 是同一函数对象。
- slow call 为 `0, 7, 15, ...`；reference tail 对齐、action buffer/mask/weights 和夹爪阈值逻辑未修改。
- 合成对照中，trace 开关前后模型输出与 Torch RNG state bitwise 相同。
- 环境状态覆盖 TCP 位姿/速度、机械臂与夹爪关节、接触和具名 scene info。
- generalist 的实际 action、transfer hidden、instruction/generated IDs、vision/projector features 和 latent 有覆盖。
- specialist raw conditions、encoders、adapters、RNG、initial noise、DiT calls、scheduler outputs 和最终 action chunk 有覆盖。

## 审查后已修正

- 默认环境改为本地存在的 `calvin_debug_dataset`。
- 默认 generalist 改为历史 profile 对应的 4-bit / FP16 compute。
- 增加 `condition_data` 和每次 `scheduler.step` 的实际输入/`prev_sample`。
- 每步写盘前增加必需字段、DiT/scheduler 调用数和 encoder 调用数的 fail-fast 检查。
- manifest 补齐 inference steps、CFG、tactile、EGL、precision、device map、attention implementation 和源码哈希。

## 未完成的集成验证

审查环境无法连接 NVIDIA driver，因此没有运行真实 7B generalist + specialist 的 GPU rollout。已完成：源码审查、Python 编译、CPU 仿真状态接口测试、合成模型 output/RNG 非干扰测试。
