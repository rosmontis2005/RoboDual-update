# M2a: Reference-Robust / Autonomous Specialist

## Scientific motivation

M2a tests one narrow hypothesis: after the slow action reference expires, M1's main weakness is an overly strong shortcut dependence on slow condition, leaving insufficient zero-reference competence. Prior diagnostics have already made history corruption, accumulated closed-loop drift as the primary action-error source, stale condition as a simple recovery explanation, and temporal-buffer masking as the primary recovery blocker less plausible. Fresh condition strongly changes raw M1 action, temporal aggregation reduces its first-step transmission to about 0.172812, and flushing restores transmission to about 1.0 without improving task recovery.

M2a therefore does not change the scheduler, slow-call strategy, temporal buffer, history-action path, or simulator data. It asks whether the same expert state and target can be learned under both persisted and counterfactual missing-reference conditions.

This is an expert-state-manifold experiment. It does **not** establish arbitrary policy-induced off-manifold recovery.

## M1 versus M2a

M1 couples condition with trajectory age: ages 0–7 have `8-age` valid reference actions, while ages 8–11 have an exact-zero reference. Late-age oversampling changes frequency but not that correlation.

M2a retains one persisted view and adds one same-state counterfactual view per micro sample. Observation, previous observation, depth, gripper inputs, proprio, instruction, history, slow age, and expert target remain identical. Only slow reference validity/action and, for one view, slow hidden values change. No new trajectories or generalist calls are needed; the immutable `robodual_age_extended_expert_v1` dataset already contains every required expert observation, target, old `slow_hidden`, `slow_action`, age, mask provenance, and split.

## Architecture

`DiffusionDiTImagePolicy` and `DiT_SingleTokenAction` accept `use_ref_validity=False` by default. M1 and all existing evaluators therefore retain their original call and state-dict contract.

M2a builds with `use_ref_validity=True` and requires `ref_valid_mask [B,8]` on every `predict_action`, `conditional_sample`, `compute_loss`, and DiT `forward` path. Semantics are fixed:

- `1`: this action token is a real, valid slow reference;
- `0`: invalid, padded, shortened, or expired.

The persisted mask is `mask[:ref_valid_count]=1`, `mask[ref_valid_count:]=0`, and every invalid `ref_action` position must be exact zero. The model does not infer validity from action values.

The only new layer is `model.ref_valid_embedder = Linear(1, hidden_size, bias=False)`. Its token output is added immediately after the unchanged 14-input `x_embedder`. The adapter is exact-zero initialized. Loading M1 EMA is audited so the only allowed missing key is `model.ref_valid_embedder.weight`; all shared keys must load and no unexpected key is accepted. Thus persisted-condition output at optimizer step 0 must match M1 within `1e-6`. Slow-age model conditioning is intentionally absent.

## Counterfactual views

For persisted count `C`:

1. `persisted`: original reference, persisted mask, original hidden.
2. `zero_ref_hidden_kept`: exact-zero reference, all-zero mask, original hidden.
3. `zero_ref_hidden_null`: exact-zero reference, all-zero mask, `zeros_like(original_hidden)`; token length and shape are unchanged. This is hidden shortcut prevention, not a claim about the only deployment state.
4. `shortened_reference`: only when `C>0`; sample `k` uniformly from `[0,C-1]`, retain the first `k` currently valid actions, and zero the suffix/mask. It never adds future reference.

Default probabilities for `C>0` are 0.60 kept-zero, 0.25 null-zero, and 0.15 shortened. For `C=0`, persisted is already identical to kept-zero, so the paired view is always `zero_ref_hidden_null`; an identical second forward is never performed.

The view layer never edits persisted dataset files. `slow_age` remains available for sampling, audits, and validation groups, but is not embedded.

## Paired diffusion contract and objective

Each micro sample performs persisted and counterfactual forwards with the exact same sampled diffusion noise, timestep, and CFG mask. Because DiT attention contains training dropout, Torch CPU/all-CUDA RNG state is also restored before the second forward and then advanced exactly once. M1's `cond_drop_chance=0.1` remains CFG training and is distinct from counterfactual reference removal.

The default objective is:

```text
L = 1.0 * L_persisted + 1.0 * L_counterfactual
```

Both use the unchanged M1 epsilon diffusion target for the same expert action chunk. Optional predicted-noise consistency is available through `--consistency_weight`, default `0.0`.

Physical dataset batch size is fixed to 1 because hidden token lengths vary and there is no context padding mask. Gradient accumulation remains 8 by default. Late-age sampling weight defaults to 2.0, matching M1's broad policy, but is not the M2a method.

## Preflight

`--dry_run` reads metadata, all persisted reference/mask contracts, split/age distributions, view eligibility, and hidden token lengths. It loads neither processor nor model, generalist, or simulator.

`--preflight_only` additionally performs:

- strict M1 EMA loading with the one-key allowlist and adapter-zero audit;
- fixed-sample M1 architecture versus M2a step-0 comparison using identical observation, target, persisted condition, noise, timestep, and CFG mask;
- prediction, diffusion loss, reconstructed first action, EE6, and gripper parity at tolerance `1e-6`;
- persisted mask/count and invalid-zero checks;
- zero-reference exact-zero mask/action checks;
- kept-hidden equality and null-hidden zero/shape checks;
- shortened-prefix, `k<C`, suffix-zero, and no-future-reference checks;
- observation, proprio, and expert-target invariance;
- exact paired noise/timestep/CFG equality;
- real forward/backward finite-loss and finite-gradient checks;
- nonzero validity-adapter gradient under different masks;
- unchanged frozen/eval vision encoder state;
- architecture/state-key sanity.

It writes `preflight.json` and one `sample_view_audit.jsonl` record to a new output directory, then exits without an optimizer update.

## Validation protocol and success criteria

Before training, the EMA initialized from M1 is evaluated on the unchanged full validation split and saved as `baseline_validation.json`. Every later validation uses the same sample population, SHA-derived per-sample noise, `--validation_timestep`, CFG mask 1, and three condition modes:

- `persisted`;
- `zero_ref_hidden_kept`;
- `zero_ref_hidden_null`.

Results are grouped by ages 0–7, each of 8/9/10/11, ages 8–11, and all. Each group reports diffusion noise MSE, first-action EE6 RMSE, and first-action gripper-sign accuracy. `reference_gap` is target-error/accuracy in a zero-reference mode minus persisted performance; raw prediction difference is not treated as performance. Per-sample records preserve the fixed noise hash, timestep, target, and reconstructed first action.

Primary offline success requires lower `zero_ref_hidden_kept` expert-target error than step-0/M1, especially for artificial age-0–7 zero-reference views and real expired ages 8–11, without material persisted degradation. Improvement for `zero_ref_hidden_null` is secondary. Closed-loop or off-manifold claims require a later evaluator.

## Outputs and checkpoint contract

Every run directory must be new and non-empty directories are rejected. A formal run writes:

```text
config.json
preflight.json
baseline_validation.json
metrics.jsonl
latest_validation.json
latest_checkpoint.json
sample_view_audit.jsonl
validation_per_sample_step_XXXXXX.jsonl
specialist_m2a_ema_step_XXXXXX.pt
m2a_training_step_XXXXXX.pt
```

Training checkpoints include online/EMA weights, optimizer, scheduler, step, args, git commit, dataset-manifest hash, M1 checkpoint SHA256, architecture/adapter config, counterfactual policy, and validation metrics. Evaluator-style checkpoints retain M1's flattened online/EMA layout, add the adapter weights, and carry `_m2a_metadata` with `model_variant=m2a_reference_robust_v1`. Save-time round-trip builds a new M2a architecture, strictly loads EMA with no missing/unexpected keys, confirms adapter presence, and compares a fixed forward within `1e-6`.

An old M1 evaluator cannot automatically use M2a. Later closed-loop evaluation must build `use_ref_validity=True` and pass the correct `[B,8]` validity mask on every step. This training task intentionally does not rewrite deployment evaluation.

## Commands

Dry run (CPU-safe, no processor/model):

```bash
python DiT_train/0820_m2a_reference_robust/train_m2a_reference_robust.py \
  --dry_run \
  --data_dir DiT_train/data_collection/runs/ageext_expert_600_s42
```

Full preflight (use a new output directory):

```bash
CUDA_VISIBLE_DEVICES=0 /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0820_m2a_reference_robust/train_m2a_reference_robust.py \
  --preflight_only --device cuda \
  --data_dir DiT_train/data_collection/runs/ageext_expert_600_s42 \
  --processor_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt \
  --output_dir DiT_train/0820_m2a_reference_robust/runs/preflight_s42
```

One-step smoke train (includes full step-0 validation and checkpoint round-trip):

```bash
CUDA_VISIBLE_DEVICES=0 /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0820_m2a_reference_robust/train_m2a_reference_robust.py \
  --device cuda --max_optimizer_steps 1 --grad_accumulation_steps 1 \
  --validate_every 1 --save_every 1 \
  --data_dir DiT_train/data_collection/runs/ageext_expert_600_s42 \
  --processor_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt \
  --output_dir DiT_train/0820_m2a_reference_robust/runs/smoke1_s42
```

Formal M2a training (does not run automatically):

```bash
CUDA_VISIBLE_DEVICES=0 /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0820_m2a_reference_robust/train_m2a_reference_robust.py \
  --device cuda --seed 42 \
  --max_optimizer_steps 1500 --grad_accumulation_steps 8 \
  --learning_rate 2e-5 --weight_decay 1e-3 \
  --warmup_optimizer_steps 100 --max_grad_norm 1.0 \
  --validate_every 250 --save_every 100 \
  --validation_timestep 50 --validation_seed 20260810 \
  --persisted_loss_weight 1.0 --counterfactual_loss_weight 1.0 \
  --zero_ref_kept_probability 0.60 --zero_ref_null_probability 0.25 \
  --shortened_ref_probability 0.15 --late_age_sample_weight 2.0 \
  --consistency_weight 0.0 \
  --data_dir DiT_train/data_collection/runs/ageext_expert_600_s42 \
  --processor_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt \
  --output_dir DiT_train/0820_m2a_reference_robust/runs/m2a_reference_robust_s42
```
