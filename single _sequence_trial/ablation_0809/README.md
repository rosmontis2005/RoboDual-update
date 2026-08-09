# 0809 mechanism diagnostic：empty reference 与 stale hidden

本目录是在 `single _sequence_trial/cross` 的 seq060 / subtask 5 restore framework 上做的小型机制消融。它分成两个互补部分：

1. 在固定 simulator boundary、P12 roll-in 得到的当前 observation、以及 diffusion seed 下，做四格 channel attribution；
2. 在已有离线 CALVIN transition validation manifest 上，画 `d=0,…,11` 的 specialist age curve。

这里的 `fresh` 只表示人为把该 channel 在 frozen observation 上刷新。`fresh/fresh`、`fresh/empty`、`stale/fresh` 都不是部署策略，不能拿来报告 deployment success；它们只用于判断输出变化来自 hidden 还是 reference。

## 1. Mechanism diagnostic

### 条件定义

P12 在 age 0 做 slow call，之后每 12 步 refresh；age 8–11 的 reference 已耗尽。对一个冻结 observation，脚本先读取 P12 age-0 的 stale condition，再在同一个 current observation 上额外做一次 deterministic slow call，得到 fresh action chunk 和 fresh hidden。

| condition | hidden | ref | deployment meaning |
|---|---|---|---|
| `stale_hidden_empty_ref` | age-0 hidden | 全零 `[1,8,7]` | 当前 P12 |
| `fresh_hidden_empty_ref` | 当前 observation fresh hidden | 全零 `[1,8,7]` | intervention |
| `stale_hidden_fresh_ref` | age-0 hidden | 当前 observation fresh 的完整 8-action chunk | intervention |
| `fresh_hidden_fresh_ref` | 当前 observation fresh hidden | 同一次 fresh slow call 的完整 8-action chunk | intervention |

四次 specialist call 共享：

- 同一个 S8/S12 restored boundary 和 P12 roll-in seed；
- 当前 observation、previous static image、proprio 和四步 action history；
- 同一个 `torch.Generator` diffusion seed。

`InitialNoiseDigest` 会记录每个 call 的首个 DiT trajectory，并要求四个 SHA256 相同；同时检查四次 specialist call 没有消耗 global RNG。输出使用 specialist raw first diffusion action，不经过 temporal aggregation，避免把 channel attribution 和 aggregation history 混在一起。

默认只做 S12、age 8、1 个 roll-in replicate，适合先做 smoke/pilot。需要扩展到两个 boundary 和完整空 ref 窗口时使用 `--states S8,S12 --ages 8,9,10,11`。

### 运行

先做不加载模型的契约检查：

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
PYTHONDONTWRITEBYTECODE=1 python \
  'single _sequence_trial/ablation_0809/run_mechanism_diagnostic.py' \
  --dry_run
```

推荐的 GPU pilot：

```bash
CUDA_VISIBLE_DEVICES=0 \
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/robodual_ablation_mpl \
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/ablation_0809/run_mechanism_diagnostic.py' \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --dataset_subdir calvin_debug_dataset \
  --states S12 --ages 8 \
  --replicates 1 --base_trial_seed 42000 --diffusion_seed 809000 \
  --load_in_4bit --low_cpu_mem_usage \
  --output_dir 'single _sequence_trial/ablation_0809/runs/mechanism_seq060_s12_age8_r1'
```

完整小 pilot：

```text
--states S8,S12 --ages 8,9,10,11 --replicates 3
```

分析一个已经完成的 run：

```bash
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/ablation_0809/analyze_mechanism_diagnostic.py' \
  'single _sequence_trial/ablation_0809/runs/mechanism_seq060_s12_age8_r1'
```

主要输出：

```text
manifest.json
restore_audits/rep_XX_S12.json
observations/rep_XX_S12_age_08.npz
mechanism_observations.jsonl       # frozen observation 与 hidden/ref provenance
mechanism_events.jsonl             # 四个 condition 的 action/noise 记录
mechanism_condition_rows.csv       # 相对 P12 的 paired output delta
mechanism_contrast_rows.csv        # hidden/ref/interactions
mechanism_analysis.json
mechanism_first_action_attribution.png/.svg
```

推荐先看 `mechanism_analysis.json` 中的：

- `hidden_effect_at_empty_ref`：`fresh H / empty R - stale H / empty R`；
- `ref_effect_at_stale_hidden`：`stale H / fresh R - stale H / empty R`；
- `hidden_ref_interaction`：二阶差分 `FF-FE-SF+SE`。

这些是 paired frozen-observation contrasts，不是 task success 的因果估计。

## 2. Offline age curve

默认数据是：

```text
LoRA_transition_0711/collected_transition_v1_repaired/
```

脚本读取其中 `split=validation` 的 samples。每一行已经保存：

- current observation 与 previous observation；
- 当前 sample 的实际 `slow_age`；
- 最后一次 current-observation slow call 的 `slow_action` 和 `slow_hidden`；
- 未来 8 个 repaired committed actions 作为 target。

因此 age curve 使用自然的 age bucket，而不是拿一个 observation 人为复制成 12 个 age。当前仓库中 `validation` split 的 `d=0,…,11` 数量为：

```text
d:  0   1   2   3   4   5   6   7   8   9  10  11
n: 476  71  78  85  94  95  92  86  30  31  27  22
```

这个 offline manifest 的 condition provenance 是已保存的 online current-observation slow condition；它不是把 hidden 在事后用错误 observation 重建。注意：它是 transition 数据集的 trajectory-level `validation` split。若要严格重采官方 CALVIN `dataset/.../validation`，需要先用相同 collector 保存对应的 slow condition cache，再把新的 manifest 通过 `--data_dir` 传入；官方原始 `.npz` 本身没有 slow hidden，不能直接用于本曲线。

### 指标定义

默认 `--diffusion_timestep 50`，每个 sample 的 noise 为 CPU float32 `[1,8,7]`，seed 为 `noise_seed + sample_id`。同一个 sample 的所有复跑都使用相同 noise；脚本还把 noise SHA256 写入每行结果。

- `loss`：固定 timestep 下 epsilon prediction 的 `[8,7]` mean MSE；脚本显式使用 `decoupled_loss=False`，所以这里的 `with_gripper=True` 只表示输入了 gripper RGB/depth，不会切换到 BCE gripper objective；
- `first_action_error_rmse_ee6`：根据 epsilon prediction 反推 `x0_hat` 后，`x0_hat[0,0,:6]` 与 target first action 的 RMSE；
- `first_action_gripper_accuracy`：在同一个 epsilon-derived `x0_hat` 上，`x0_hat[0,0,6]` 与 target first gripper action 的 sign accuracy。由于 scheduler 固定为 `prediction_type=epsilon`，gripper channel 也按 epsilon 做 x0 反推。

加载每个 sample 时还会逐行核验 `condition.source == online_current_observation`、`sample_step - condition_step == slow_age`、`slow_action=[1,8,7]`、`slow_hidden=[1,T,4096]` 和 `hist_action_before=[4,7]`，因此不是只依赖 collection summary 的字符串标记。

运行 dry-run：

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  'single _sequence_trial/ablation_0809/run_offline_age_curve.py' \
  --dry_run
```

先用每个 age 两条样本检查管线：

```bash
CUDA_VISIBLE_DEVICES=0 \
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/robodual_ablation_mpl \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/ablation_0809/run_offline_age_curve.py' \
  --max_samples_per_age 2 \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --output_dir 'single _sequence_trial/ablation_0809/runs/age_curve_smoke'
```

正式曲线去掉 `--max_samples_per_age`（默认 0，使用所有可用样本）：

```text
--output_dir single _sequence_trial/ablation_0809/runs/age_curve_validation_all
```

输出：

```text
manifest.json
age_curve_samples.jsonl
age_curve_samples.csv
age_curve_summary.json
age_curve.png/.svg
```

`age_curve_summary.json` 会直接给出 `jump_7_to_8_*` 和 `slope_8_to_11_*`。解释时应同时查看每个 age 的 `n` 和 `sem`：7→8 的离散变化与 ref 从 1 个 action 变成 0 的时间点对齐；8→11 都处于 empty-ref 窗口，若 loss/error 仍随 age 增长，其斜率更符合 hidden staleness 的候选机制。但这仍是 offline age-bin 证据，不能单凭斜率排除 task mix、observation drift 或 target difficulty。

## 3. CPU contract check

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  'single _sequence_trial/ablation_0809/verify_ablation_contract.py'
```

它会检查：P12 的 ref tail 对齐、age 7→8 的 `8-age`/empty contract、四格二阶差分公式，以及 validation manifest 是否覆盖 d=0..11。

## 文件

- `mechanism_common.py`：四格命名、P12 reference 构造、hash 和 contrast 公式；
- `run_mechanism_diagnostic.py`：seq060 restore、P12 roll-in、frozen-observation 四格调用；
- `analyze_mechanism_diagnostic.py`：完整性审计、paired contrasts 和图；
- `run_offline_age_curve.py`：固定 noise/timestep 的离线 age curve；
- `verify_ablation_contract.py`：不加载模型的 CPU 契约测试；
- `INDEPENDENT_AUDIT.md`：独立 subagent 审查记录（实验脚本完成后补写）。
