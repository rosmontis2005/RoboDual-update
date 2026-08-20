# Drifted-state slow-condition intervention

This directory contains a diagnosis-only counterfactual experiment. A fresh
generalist observation at a real policy-induced drift state is an intervention,
teacher, or oracle probe. It is **not** a proposed deployment policy, does not
change the evaluator's slow-call schedule, and must not be interpreted as an
adaptive scheduler.

The script reuses all anchors and frozen anchor conditions from a completed
50-anchor `0818_expert_specialist_trajectory` run. For intervention age `a`, it
resets to the persisted task-start anchor, captures the canonical oracle
`start_info`, executes specialist actions `k=0..a-1`, and intervenes immediately
before action `a`. Thus age 8 is exactly the first zero-reference step, with no
off-by-one shift.

## Branches

All four branches start from the same captured simulator state through the same
`env.reset(robot_obs=..., scene_obs=...)` path, from independent deep copies of
the same controller runtime snapshot, and from the same paired Torch CPU/CUDA RNG
snapshot. Temporal aggregation is retained.

- `frozen_baseline`: original slow action and hidden, `last_slow_step=0`.
- `fresh_hidden_only`: fresh hidden but original action and
  `last_slow_step=0`. At ages 8/11 `num_cond_actions=0`, so this branch receives
  no fresh explicit action reference.
- `fresh_ref_only`: fresh action, original hidden, and
  `last_slow_step=intervention_age`; the first post-intervention step has eight
  explicit references.
- `full_refresh`: fresh action and hidden with
  `last_slow_step=intervention_age`; this is a one-call diagnostic upper bound.

There is exactly one `do_sample=False` generalist call per `(anchor, age)`. Its
action and hidden originate in the same `predict_action` call and are shared by
all branches. The frozen wrapper forbids every later generalist call.

## Interpretation boundaries

These are patterns for subsequent analysis, not automatic claims or proof:

A. If `fresh_hidden_only` materially changes raw/final actions and improves the
branch outcome, stale slow latent is causally implicated within this experiment.
The next direction is deployment-state/on-policy training or distillation, not
simply increasing runtime generalist frequency.

B. If hidden-only is ineffective while `fresh_ref_only` is similar to
`full_refresh` and improves outcomes, the main issue points toward zero-reference
specialist competence rather than stale hidden. The next step is training the
specialist to be more reliable without explicit references.

C. If hidden-only and ref-only both help and full refresh is strongest, both
mechanisms may contribute.

D. If raw fast output is sensitive but the final executed action is not,
temporal aggregation is suppressing the corrective signal. Inspect buffer and
aggregation behavior first. `temporal_transmission_ratio` is diagnostic only,
not a strict causal coefficient.

E. If all interventions have almost no effect, fresh generalist condition does
not explain the observed phenomenon. Stop the stale-condition direction and
return to specialist/task competence.

Task success excludes anchor-age cases already successful during the common
prefix from the primary aggregate. Per-task results are descriptive when sample
counts are small. Persisted expert trajectories, when separately consulted, are
a reference and not a unique optimal ground truth.

## Commands

Set `SOURCE_RUN` to the completed 50-anchor source run.

```bash
python DiT_train/0820_drift_condition_intervention/run_drift_condition_intervention.py \
  --source_run_dir "$SOURCE_RUN" \
  --run_name drift_intervention_dry \
  --dry_run \
  --intervention_ages 8,11
```

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0820_drift_condition_intervention/run_drift_condition_intervention.py \
  --source_run_dir "$SOURCE_RUN" \
  --run_name drift_intervention_preflight_2 \
  --preflight_only \
  --preflight_anchors 2 \
  --intervention_ages 8,11 \
  --post_steps 8 \
  --device cuda \
  --use_egl
```

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0820_drift_condition_intervention/run_drift_condition_intervention.py \
  --source_run_dir "$SOURCE_RUN" \
  --run_name drift_intervention_validation50_s42 \
  --intervention_ages 8,11 \
  --post_steps 8 \
  --device cuda \
  --use_egl
```

Run directories are never silently overwritten. Formal output includes the
manifest and source hashes/fingerprints, JSONL intervention and step records,
CSV summaries, reset-fidelity audit, one fresh condition per anchor-age, and
compressed branch trajectories.
