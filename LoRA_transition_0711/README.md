# Transition LoRA data collection

This directory implements the data side of `personal_log/exp_log/log0711_LoRA.md`.
It deliberately leaves the 0525 evaluator and the previous `LoRA_trial` intact.

## Collection contract

- Scheduler and A/B/C/D ages are imported from `evaluate_calvin_task_age_0525.py`.
- Groups A-C have a default cap of eight saved trajectories per task so their
  quotas cannot be filled by only one easy task; D contains only `stack_block`.
- A refresh sample uses `new_ref/new_hidden` produced by the real slow call on
  that step's current observation. Conditions are not reconstructed offline.
- `hist_action_before` is the last four temporally aggregated actions already
  sent to the environment; the current action is never leaked into history.
- Targets are the next eight executed actions and are retained only when the
  complete online subtask succeeds.
- Trajectories, not windows, receive a stable 70/15/15 split.
- The environment defaults explicitly to the CALVIN `training` split. Exact
  fingerprints of the 100 official benchmark sequences are also excluded.
- Final windows use the agreed 50/30/10/10 normal/refresh/high-conflict/stale
  distribution inside every split. High-conflict is tracked as a subset of the
  complete refresh pool; selected high-conflict windows are removed before the
  remaining 30% refresh sample is drawn, so no window is duplicated.

Conditions are stored once under `conditions/<trajectory>/`. A sample refers to
one by `condition_id` and records `slow_age`, allowing the trainer to reproduce
the 0525 reference shift exactly. Refresh samples also record
`old_condition_id` for optional two-view consistency training.

The collector does not stop when only the ABCD trajectory quotas are complete.
It continues saving successful trajectories until all per-split normal,
refresh-total, high-conflict, and stale requirements are also complete. If the
sequence budget is exhausted first, the summary is marked `incomplete` and the
command exits non-zero unless `--allow_incomplete` is used for diagnostics.

## Recommended collection command

Run from `RoboDual` in the `dualsys_env` environment:

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
python LoRA_transition_0711/collect_transition_rollouts.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --dataset_subdir calvin_debug_dataset \
  --dataset_split training \
  --output_dir LoRA_transition_0711/collected_transition_v1 \
  --num_sequences 1000 \
  --sequence_start 100 \
  --target_samples 8000 \
  --group_trajectory_quotas A:60,B:60,C:30,D:20 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

Do not use `--overwrite` for a resumable production run: this first version is
transactional per run and intentionally refuses to merge partially collected
conditions from a different model/configuration.

## History adapter

`history_adapter.py` provides a validated `[B,4,7]` adapter with a zero-initialized
output projection and unit scalar gate. It is therefore an exact no-op before
training while the output projection still receives gradients on the first
optimizer step. `install_history_adapter(policy)` registers it; the DiT forward
path consumes the adapter residually and remains unchanged while it is `None`.
Install before constructing `DualSystem` when possible. For an already-created
system, `install_dual_system_history_adapters()` registers matching online and
EMA copies; checkpoint loading must likewise install the structure first.

## Training

`train_transition_lora.py` consumes the finalized manifests and persisted slow
conditions directly. It does not rerun or train the generalist; the
`generalist_path` is used only to load the matching image processor.

The first experiment trains exactly these fast-specialist paths:

```text
model.history_adapter                         full training
model.x_embedder                              LoRA rank 4
model.context_adapter                         LoRA rank 4
model.blocks.0-5.attn_temporal.qkv/proj       LoRA rank 4
```

Vision/depth/gripper adapters, proprio, MLP, cross-attention, final head and
the rest of the specialist remain frozen. The trainer refuses to start if its
resolved LoRA target set differs from the expected 14 linear modules.

Recommended command from `RoboDual`:

```bash
CUDA_VISIBLE_DEVICES=0 \
python LoRA_transition_0711/train_transition_lora.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --data_dir LoRA_transition_0711/collected_transition_v1_repaired \
  --output_dir LoRA_transition_0711/lora_runs/transition_history_lora_v2_repaired \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --max_steps 3000 \
  --learning_rate 3e-5 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --validation_interval 100 \
  --validation_samples_per_category 64 \
  --early_stopping_patience 5 \
  --max_normal_loss_ratio 1.05 \
  --bf16
```

Validation uses a fixed, category-balanced subset and deterministic diffusion
noise. Metrics include overall and normal/refresh/high-conflict/stale losses,
history output norm, history gate and gradient norm. `adapter_best.pt` is only
written for checkpoints whose normal validation loss is at most 1.05 times the
frozen baseline. An unconstrained best is retained as a fallback. Final merged
policy/EMA checkpoints are always produced from the selected best adapter, not
blindly from the last optimizer step.

Training and validation are intentionally restricted to physical batch size 1.
Persisted slow conditions contain both 87- and 88-token hidden states, while the
current specialist has no context padding mask. Two micro-batches are accumulated
per optimizer step, preserving effective batch size 2 without synthetic tokens.

## Task-age evaluation

Use the dedicated evaluator for transition checkpoints. It installs the history
adapter before constructing `DualSystem`, strictly loads the merged checkpoint,
and verifies every loaded history tensor. The regular 0525 evaluator must not be
used because it has no adapter structure to receive these weights.

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
python vla-scripts/evaluate_calvin_task_age_transition_lora_0712.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --transition_checkpoint LoRA_transition_0711/lora_runs/transition_history_lora_v2_repaired/specialist_transition_lora_merged_ema.pt \
  --dataset_subdir calvin_debug_dataset \
  --num_sequences 100 \
  --slow_call_strategy task_age \
  --load_in_4bit \
  --low_cpu_mem_usage
```
