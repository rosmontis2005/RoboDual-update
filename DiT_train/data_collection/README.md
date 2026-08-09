# Age-Extended Expert Dataset（contract/test collector）

本目录只负责从 CALVIN **training split 的 language-annotated expert demonstrations** 构造小规模、可审计的 specialist 监督数据。它不运行 specialist、不训练模型，也不改变 P8/P12 evaluator。

## 数据语义

每个合法 anchor `t` 在 `rgb_static[t] + instruction` 上只调用一次 frozen generalist：

```text
model.eval()
torch.inference_mode()
processor(P12 prompt, anchor RGB)
predict_action(..., do_sample=False)
```

同一次调用返回的 `slow_action [1,8,7]` 与 `slow_hidden [1,T,H]` 被存入一个 `condition_XXXXXX.pt`。同一 condition 派生 age `d=0..11` 的 12 个 samples；每个 sample 的 current observation 是 frame `t+d`，label 是 CALVIN expert 的 `t+d .. t+d+7` relative actions。

P12 reference contract：

```text
d=0..7: ref[:8-d] = slow_action[:, d:8]
d=8..11: ref = exact zero [1,8,7]
```

history 是 current action 发生前的最近最多 4 个 task-local expert actions，左侧补零；anchor 位于 subtask 中部时，`d=0` 会保留 anchor 前的真实 history。task-local step 0 的 previous RGB 等于 current RGB，其他位置使用 current-1，绝不跨 annotation boundary。

CALVIN language annotation 的 `(start,end)` 是 inclusive。最大 age 11 的 8-action target 最后位置是 `t+18`，所以合法范围严格为：

```text
start <= t <= end - 18
```

anchor selection 还会检查 union source range `max(task_start,t-4)..t+18` 中每个 `.npz` 是否真实存在。annotation 合法但 source frame 缺失的候选会被跳过并计入 dry-run shortfall，因此可直接用于只保留部分 episode/frame 的 partial source dataset。

正式采集时，这段 union range 对每个 anchor 只加载一次；12 个 ages 的 observation、history 和 target 全部复用同一个内存 cache，不会按 age 重复读取 frame。

## 输出

```text
<output_dir>/
  manifest.json
  anchors.jsonl
  samples.jsonl
  audit_summary.json
  conditions/condition_XXXXXX.pt
  observations/*.npz                 # 仅 --materialize_observations
```

默认 samples 只存原始 CALVIN frame path/index/key reference，避免复制图像与深度。每行同时保存 ref、mask、history、expert target、proprio/scene freshness delta 及全部 source frame indices。`with_tactile=false` 被显式写入 manifest。

collector 先写同目录下的隐藏 staging directory，所有内存 audit 通过后才原子发布。非空 output 默认拒绝；显式 `--overwrite` 时旧目录会改名为 timestamped backup，不会直接删除。

## 不加载模型的检查

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual

PYTHONDONTWRITEBYTECODE=1 python \
  DiT_train/data_collection/collect_age_extended_expert.py \
  --cpu_contract_test

PYTHONDONTWRITEBYTECODE=1 python \
  DiT_train/data_collection/collect_age_extended_expert.py \
  --dry_run \
  --dataset_root /home/rosmontis/Projects/dualsys/calvin/dataset/calvin_debug_dataset \
  --max_anchors 50 --max_anchors_per_episode 2 --seed 42
```

`--dry_run` 只扫描 annotation/index、计算合法 anchor、stable trajectory split 和 task distribution；不会加载 processor/model，也不会创建 output。它同时报告 probed candidate 数、required-file 检查数和缺失 frame 示例。

## 1–2 anchor GPU smoke

当前部署/最新 mechanism diagnostic 使用 4-bit NF4、FP16 compute，因此 collector 默认启用相同配置：

generalist loader 与 `vla-scripts/task_age_v1_0706.py::load_generalist` 对齐：直接执行 `AutoModelForVision2Seq.from_pretrained(...).eval()`，4-bit placement 交给 Transformers/BitsAndBytes，不额外调用 `Accelerator.prepare()`。

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/data_collection/collect_age_extended_expert.py \
  --dataset_root /home/rosmontis/Projects/dualsys/calvin/dataset/calvin_debug_dataset \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --max_anchors 2 --max_anchors_per_episode 1 --seed 42 \
  --load_in_4bit --low_cpu_mem_usage \
  --materialize_observations \
  --output_dir DiT_train/data_collection/runs/smoke_2anchors

PYTHONDONTWRITEBYTECODE=1 python \
  DiT_train/data_collection/verify_age_extended_contract.py \
  DiT_train/data_collection/runs/smoke_2anchors
```

## 建议的 50-anchor test collection

需把 `--dataset_root` 指向含至少 25 个合格 language episodes 的 CALVIN training dataset；当前仓库默认 debug dataset 只有 9 个 language episodes，在每 episode cap=2 时最多只能选择 18 anchors。

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/data_collection/collect_age_extended_expert.py \
  --dataset_root /path/to/full/calvin_dataset \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --max_anchors 50 --max_anchors_per_episode 2 --seed 42 \
  --load_in_4bit --low_cpu_mem_usage --hash_model_files \
  --output_dir DiT_train/data_collection/runs/test_50anchors
```

`--hash_model_files` 会对所有 checkpoint shard 做 SHA256，较慢但提供最强 provenance；默认仍记录目录 metadata digest，并对 model config/index、processor/tokenizer/code 文件做内容 SHA256。

## 与旧实现的差异

- `DiskCalvinDataset` 为原始训练随机选择 `pred_actions=0..7`，其训练覆盖约 d=0..7，且原实现的 previous/history 是窗口内部构造；本 collector 显式使用 annotation subtask boundary，保留 anchor 之前的真实 expert history，并扩展到 d=11。
- transition collector 保存 successful online policy rollout、scheduler 自然产生的 age bucket 与 committed policy action target；本 collector 不做 rollout，固定一个 expert anchor condition 后成对派生完整 12 ages，target 始终来自官方 expert demonstration。
- official CALVIN benchmark sequence fingerprint 是 simulator initial-state/task sequence 的 fingerprint，无法可靠映射到 language annotation expert episode；因此这里不伪造 exclusion match，而在 manifest 中明确记录“不适用”。数据源被硬限制为 training split。
- partial source dataset 被显式支持：缺少任一 required frame 的窗口会在 anchor selection/dry-run 阶段被排除，不会进入 condition inference。

## 独立 verifier 检查

verifier 会重新加载 condition 与原始 `.npz`，检查：数量；完整 age set；action/hidden shape；same-call IDs；P12 suffix 与 7→8 transition；expired ref exact zero；history indices/value 与 no leakage；previous boundary；expert target indices/value；observation references；proprio/scene delta；training source；trajectory stable split 与 leakage；manifest/stored audit 一致性。任一失败以非零退出。
