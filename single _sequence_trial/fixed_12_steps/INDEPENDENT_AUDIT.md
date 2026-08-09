# Independent agent audit

最终评级：**有条件通过**。当前没有已知代码级 blocker；条件是尚未完成真实 GPU/CALVIN rollout 集成验证。

## 已确认

- 模型加载、checkpoint、4-bit/FP16、diffusion steps、dataset、sequence 60、catalog 100、seed、环境工厂、初始状态、annotation 和 rollout 均复用 paired `original_8_steps` 实现。
- `FixedAge12Evaluation.step`、`_should_call_slow_system` 和 `_build_ref_actions_from` 直接继承历史 evaluator，没有重写动作路径。
- 策略固定为 `age_empty / max_slow_age=12 / empty_ref_after_age=8`，无 handover、delta limit 或 jerk limit。
- slow calls 为 `0,12,24,...`；age 0–11 的 reference condition 数为 `8,7,6,5,4,3,2,1,0,0,0,0`。
- trace 保存结构与 8-step collector 共用同一实现；环境、generalist、specialist、RNG、initial noise、DiT、scheduler 和完整 action chunk 均有覆盖。
- `FixedAge12TraceCapture` 会逐步 fail-fast 检查 live policy、max age、empty-ref age、slow age 和 condition 数。
- shared trace 的默认 slow-call 判定仍是原 fixed-mod-8，原 collector 的回归验证通过。
- fixed-age-12 合成验证、原 fixed-mod-8 回归验证和 Python 编译均通过；trace 开关前后输出与 Torch RNG state 相同。

## 审查后修正

- manifest 中历史 age-12 入口已从错误的 `evaluate_calvin_0428.py` 修正为 artifact 实际记录的 `evaluate_calvin_codex_test_0424test.py`；`0428.py` 仅保留为环境工厂参考。
- RNG 说明已明确：确定性 slow generalist 通常不消耗 RNG；两个策略在 subtask 长度分叉后，后续 subtask 才会从不同全局 RNG 位置开始。
- 明确记录历史 artifact 使用 `ep_len=360`，当前 paired collector 为保持与本次 8-step 数据一致而使用 `ep_len=240`。

## 未完成的集成验证

独立审查环境无法连接 NVIDIA driver，因此没有执行真实 7B generalist、specialist、PyBullet 和完整 sequence 60。正式采集结束后，应同时检查逐步 fail-fast 未触发、`summary.json` 完整，以及最后一个 `.pt` 可读取且 SHA256 与 `events.jsonl` 一致。
