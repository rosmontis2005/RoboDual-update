# LoRA Trial Data Collection

This folder contains standalone utilities for collecting small specialist-LoRA
training data. It does not modify `vla-scripts`.

## Collector

`collect_lora_rollouts.py` runs the current RoboDual policy in CALVIN, keeps only
successful target-task rollouts, and writes them in CALVIN-style per-frame
format:

```text
collected_lora_rollouts/
  manifest.jsonl
  collection_summary.json
  training/
    episode_0000000.npz
    episode_0000001.npz
    ...
    ep_start_end_ids.npy
    lang_annotations/
      auto_lang_ann.npy
```

Each saved frame includes:

```text
actions, rel_actions, robot_obs, scene_obs,
rgb_static, rgb_gripper, rgb_tactile,
depth_static, depth_gripper, depth_tactile
```

The current specialist training path uses `rel_actions`; `actions` is filled as a
best-effort format-compatible field. The current CALVIN evaluation environment
does not instantiate the tactile camera, so `rgb_tactile` and `depth_tactile`
are zero placeholders for schema compatibility.

The default first-round collection target is intentionally small:

```text
place_in_slider: 20
lift_blue_block_slider: 10
stack_block: 10
rotate_red_block_right: 5
push_pink_block_right: 5
```

`push_blue_block_right` and `push_red_block_right` are not collected by default.
The collector also applies two safe runtime reductions:

```text
1. Skip evaluation sequences that contain no still-needed target task.
2. Stop a sequence once its last still-needed target task has been attempted.
```

## Small Sanity Run

Run a short collection first:

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
conda run -n dualsys_env python LoRA_trial/collect_lora_rollouts.py \
  --dataset_subdir calvin_debug_dataset \
  --target_task_quotas place_in_slider:2,lift_blue_block_slider:2,stack_block:2,rotate_red_block_right:1,push_pink_block_right:1 \
  --num_sequences 20 \
  --load_in_4bit \
  --low_cpu_mem_usage \
  --overwrite
```

## Larger Collection

For the first LoRA exploration pass, use the reduced quota set:

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
conda run -n dualsys_env python LoRA_trial/collect_lora_rollouts.py \
  --dataset_subdir calvin_debug_dataset \
  --target_task_quotas place_in_slider:20,lift_blue_block_slider:10,stack_block:10,rotate_red_block_right:5,push_pink_block_right:5 \
  --num_sequences 100 \
  --slow_call_strategy task_age \
  --load_in_4bit \
  --low_cpu_mem_usage \
  --output_dir LoRA_trial/collected_lora_rollouts_task_age_v1 \
  --overwrite
```

If the 100-sequence pass does not fill all quotas, increase `--num_sequences` to
150 or 200 and keep the same `--target_task_quotas`. The skip and early-stop
logic means extra sequences mostly add opportunities for missing target tasks,
not full five-subtask evaluations.

## Specialist LoRA Training

`train_lora_specialist.py` is a minimal first-pass trainer for the collected
rollouts. It freezes the generalist and the base specialist, injects small LoRA
adapters into specialist linear layers, and trains on stale-age samples. By
default it samples ages `8,9,10,11`, which correspond to empty reference actions
when `empty_ref_after_age=8`.

Run a very short smoke test after collecting data:

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual
conda run -n dualsys_env python LoRA_trial/train_lora_specialist.py \
  --data_dir LoRA_trial/collected_lora_rollouts_task_age_v1 \
  --output_dir LoRA_trial/lora_runs/smoke_empty_ref_lora \
  --stale_ages 8,9 \
  --samples_per_rollout_per_age 1 \
  --batch_size 1 \
  --max_steps 5 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

First actual trial:

```bash
python -u train_lora_specialist.py \
  --data_dir LoRA_trial/collected_lora_rollouts_task_age_v1 \
  --output_dir LoRA_trial/lora_runs/specialist_empty_ref_lora_v1 \
  --stale_ages 8,9,10,11 \
  --samples_per_rollout_per_age 4 \
  --batch_size 2 \
  --max_steps 300 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

Outputs:

```text
lora_runs/specialist_empty_ref_lora_v1/
  adapter_final.pt
  adapter_step_*.pt
  specialist_lora_merged_policy.pt
  specialist_lora_merged_ema.pt
  metrics.jsonl
  training_config.json
  training_summary.json
```

Use `specialist_lora_merged_ema.pt` with the existing evaluation scripts via
`--specialist_path`. It is written in the same EMA-wrapper checkpoint format as
the original specialist checkpoint.

codex resume 019eaa3f-c26b-74f3-bd31-80d48b705e2e
