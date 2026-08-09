# Fixed 12-step online trace collector

本目录是 `original_8_steps/` 的平行对照，采集同一个 canonical `sequence_index=60`，只将 slow-call 策略改成历史 uniform age-12：每个 subtask 在 `0, 12, 24, ...` 刷新 generalist。

## 配对一致性

模型 checkpoint、4-bit/FP16 精度、10-step diffusion、环境、sequence catalog、初始状态、指令、seed、temporal aggregation、夹爪处理、每步采集字段和落盘 schema 均与 8-step collector 相同。差异只有：

- evaluator 使用历史 `dual_sys_evaluation_0424test.py` 的 `age_empty` 策略；
- `max_slow_age=12`，slow calls 为 `0, 12, 24, ...`；
- generalist action chunk 仍为 8 步，reference 有效长度按 age 为 `8,7,6,5,4,3,2,1,0,0,0,0`，因此 age 8–11 为全零；
- 无 handover、action delta limit 或 jerk limit。

注意：固定 12 步不是 `(step + 1) % 12 == 0`。历史 age-12 artifact 的第一个周期是 step 0 到 step 12，必须使用 `step % 12 == 0` 才能复现其调用节奏。

采集 hook 仍不设置 RNG、不额外采样。slow generalist 使用 `do_sample=False`，通常不消耗随机数；只要两边累计执行的环境步数相同，specialist diffusion noise 会按调用序号保持对齐。若某个 subtask 的完成步数不同，后续 subtask 会从不同的全局 RNG 位置开始。每一步实际 RNG state 和 initial noise 均会保存。

历史筛选用的 age-12 artifact 设置 `ep_len=360`。本脚本为了与刚完成的 8-step 数据严格配对，默认继续使用 `ep_len=240`；它复现相同的 age-12 调度和动作策略，但不声称复现历史 run 的 360-step timeout 边界。

## 启动

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual

CUDA_VISIBLE_DEVICES=0 \
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual_mpl \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/fixed_12_steps/collect_fixed_12_steps.py' \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --dataset_subdir calvin_debug_dataset \
  --sequence_index 60 \
  --catalog_size 100 \
  --max_subtasks 5 \
  --ep_len 240 \
  --seed 42 \
  --fast_num_inference_steps 10 \
  --load_in_4bit \
  --low_cpu_mem_usage \
  --device_map none \
  --attn_implementation none \
  --output_dir '/home/rosmontis/Projects/dualsys/RoboDual/single _sequence_trial/fixed_12_steps/runs/seq060_seed42_fixed_age12'
```

输出目录必须不存在或为空，脚本拒绝覆盖已有数据。

## 验证

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/robodual_mpl \
  /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/fixed_12_steps/verify_collection_contract.py'
```

## 保存结构

run 目录仍包含 `manifest.json`、`sequence.json`、`events.jsonl`、`summary.json` 和 `tensors/seq_060/subtask_XX_*/step_XXXX.pt`。单步 payload 的 `environment/generalist/specialist/evaluator_profile` 字段与 `original_8_steps` 相同；具体字段说明参见配对目录的 README。

12-step 策略减少 generalist slow calls，但每个环境步的 specialist、DiT、图像、物理状态仍完整保存，因此总容量主要由实际轨迹长度决定。按相同步数估计，与 8-step 数据基本同量级、略小。
