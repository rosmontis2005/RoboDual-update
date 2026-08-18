# M1 Expert-History Invariance Ablation

## Question

Does the current M1 specialist actually depend on expert-quality `hist_action`?

## Static expectation

The current `DiffusionDiTImagePolicy` constructs `DiT_Tiny_STA` with
`with_hist_action_num=0`. `DiT_SingleTokenAction.forward()` can consume
`hist_action` through either the history-token path or an optional residual
`history_adapter`, but the current construction disables the former and leaves
the latter as `None`. The expected static result is therefore
`static_history_path_active=false`.

The script verifies this again after loading the exact EMA checkpoint. It also
records checkpoint load mismatches, all history-related policy/checkpoint keys,
and the actual imported policy and transformer source files. A statically
inactive path does not skip the runtime test.

## Dynamic experiment

The experiment uses the existing `AgeExtendedExpertDataset`, policy builder,
EMA loader, processor loader, deterministic validation noise, autocast policy,
and epsilon-prediction x0 reconstruction from
`DiT_train/train_age_extended_expert.py`. It evaluates the `validation` split
with a strict paired contract: RGB, previous RGB, both depths, gripper RGB,
proprio, instruction, slow hidden state, reference action, target trajectory,
timestep, noise, and `cond_mask=torch.ones((1,1))` remain fixed. Only
`hist_action` changes among expert, zero, time-reversed, deterministic same-age
donor, and deterministic channel-scaled Gaussian corruption.

Donors come only from selected validation rows. Selection first prefers the same
task and age with a different condition, then any different selected sample of
the same age. If neither exists, that variant is explicitly unavailable.

All outputs are written below `runs/<run_name>/`. A non-empty run directory is
never overwritten unless `--overwrite` is explicit.

Quick contract check without loading the model:

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual-update
python DiT_train/0818_expert_history_ablation/run_history_invariance.py \
  --data_dir DiT_train/data_collection/runs/ageext_expert_600_s42 \
  --checkpoint DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt \
  --processor_path /home/rosmontis/Projects/dualsys/models/generalist \
  --split validation --max_samples_per_age 10 --diffusion_timestep 50 \
  --run_name m1_history_invariance_dry --dry_run
```

Model preflight on one deterministic sample at each boundary age 0, 7, 8, and
11:

```bash
python DiT_train/0818_expert_history_ablation/run_history_invariance.py \
  --data_dir DiT_train/data_collection/runs/ageext_expert_600_s42 \
  --checkpoint DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt \
  --processor_path /home/rosmontis/Projects/dualsys/models/generalist \
  --split validation --diffusion_timestep 50 \
  --run_name m1_history_invariance_preflight --preflight_only
```

Recommended 10-per-age run:

```bash
python DiT_train/0818_expert_history_ablation/run_history_invariance.py \
  --data_dir DiT_train/data_collection/runs/ageext_expert_600_s42 \
  --checkpoint DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt \
  --processor_path /home/rosmontis/Projects/dualsys/models/generalist \
  --split validation --max_samples_per_age 10 --diffusion_timestep 50 \
  --run_name m1_history_invariance_10perage
```

For full validation, use `--max_samples_per_age 0` and a new run name.

## Interpretation

If changed histories yield prediction and x0 deltas no larger than the selected
tolerance (default `1e-6`), the conclusion is:

**Current M1 is history-invariant; the expert-vs-policy mismatch cannot currently
enter through the explicit `hist_action` argument.**

This does not rule out train-deployment mismatch. The more important remaining
issue is expert-state training versus the specialist's policy-induced,
closed-loop state distribution: RGB, proprioception, and scene state can drift
because of the specialist's own execution errors.

If corruption produces nontrivial deltas, first audit the recorded runtime
architecture, checkpoint keys/load result, and import paths rather than
immediately claiming learned history dependence.
