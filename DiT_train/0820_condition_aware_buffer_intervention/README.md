# Condition-aware temporal-buffer intervention (M2 preflight)

This is a diagnosis-only causal experiment. It tests whether the existing
eight-step temporal-aggregation buffer suppresses the control authority of a
fresh generalist condition, and whether removing that suppression improves
recovery. It is neither an M2 training recipe nor a deployment scheduler.

The scientific population and model contract are inherited and revalidated
from the completed 50-anchor validation run:

`DiT_train/0818_expert_specialist_trajectory/runs/trajectory_validation_s42_n50`

The script reuses its persisted CALVIN task-start anchors and same-call frozen
conditions, seed 42, M1 EMA checkpoint, generalist checkpoint/fingerprint, NF4
4-bit loader, evaluator architecture, and all non-intervened control settings.
It imports contracts from the completed drift-condition experiment but never
writes into that directory or changes the global evaluator.

## Scientific design

For each anchor and age 8 or 11, one common prefix is executed. The environment
is reset to the persisted task-start state, canonical oracle `start_info` is
captured, the old frozen condition is injected, and real M1 closed-loop actions
`k=0..age-1` are executed. The intervention occurs immediately before action
`age`. Age 8 is asserted to be the first zero-reference state; age 11 is the
maximally stale tested state.

At the branchpoint, the generalist is called exactly once on the current
policy-induced RGB state and instruction with `do_sample=False`. Fresh slow
action and hidden state must share one inference-call ID. The one result is
saved and shared by every branch. The frozen wrapper forbids every later slow
call.

The six branches form a 3 × 2 factorial:

| Branch | Slow action | Slow hidden | `last_slow_step` | Buffer |
|---|---|---|---:|---|
| `old_keep` | old | old | 0 | keep |
| `old_flush` | old | old | 0 | flush |
| `ref_keep` | fresh | old | age | keep |
| `ref_flush` | fresh | old | age | flush |
| `full_keep` | fresh | fresh | age | keep |
| `full_flush` | fresh | fresh | age | flush |

`old_*` must have zero explicit references on the first intervention step.
`ref_*` and `full_*` must have eight.

`old_flush` is the critical control. If flushing improves recovery on its own,
an apparent `full_flush` benefit cannot be attributed to unmasking a fresh
condition. All fresh-condition comparisons therefore use buffer-matched
controls: `old_keep` for keep and `old_flush` for flush.

## Exact flush semantics

Every branch starts from an independent deep copy of the complete common-prefix
controller snapshot. After applying the branch condition, `flush` performs
only:

```python
wrapper.action_buffer[...] = 0
wrapper.action_buffer_mask[...] = False
```

It does not call `wrapper.reset()`. It preserves `hist_action`, `obs_buffer`,
`gripper_window`, `prev_action`, `prev_prev_action`, `prev_proprio`,
`prev_obs_tensor`, slow action, slow hidden, `last_slow_step`, and every other
snapshotted controller field. A value/digest audit compares each keep/flush
pair immediately before rollout and fails unless only the two buffer fields
differ. The keep arrays must exactly match the branchpoint snapshot; the flush
arrays must be zero/all-false.

## Mechanism versus outcome

The mechanism endpoint asks whether flush restores the fresh-condition action
signal. First-step reference and full-refresh transmission are computed under
matched buffer policy:

```text
raw effect       = ||dp(fresh) - dp(old)||_EE6
keep effect      = ||action(fresh_keep) - action(old_keep)||_EE6
flush effect     = ||action(fresh_flush) - action(old_flush)||_EE6
transmission     = matched executed effect / raw effect
```

The expected sanity pattern is keep transmission near `0.172812` and flush
transmission near `1.0`. This confirms control authority only. It is not
evidence that the fresh action is corrective.

The outcome endpoint separately reports success within the first 8 and 16 post
steps, first-success step, per-step success, state evolution, and explicit
paired contrasts. Recovery claims must be based on `full_flush` versus
`old_flush` (and related outcomes), not on transmission alone.

## Same-index expert proximity

For each valid source index through `anchor+18`, the policy state at
`age+j+1` is compared with the persisted expert state at the same index. The
output is named `same_index_expert_proximity` and includes robot EE6/full and
scene distances plus existing RGB/depth metrics.

This metric is **descriptive**. After policy-induced drift, the persisted expert
trajectory is not a unique optimal recovery trajectory. These values are not
called optimality or ground-truth recovery error.

## Hard preflight contracts

The run fails loudly unless all of the following hold:

- exact branch action/hidden/age/reference definitions;
- keep/flush runtime independence and exact flush isolation;
- identical Torch CPU/all-CUDA RNG restoration before each branch;
- uniform `env.reset(robot_obs=..., scene_obs=...)` branchpoint restoration;
- one value-identical captured branchpoint observation, verified by SHA-256,
  is used as the first specialist input in all six branches;
- same-condition keep/flush first raw specialist outputs equal with
  `atol=rtol=1e-6`, including EE6 and gripper deltas;
- every flush branch has first-step `aggregation_delta_ee6 <= 1e-6`, with raw
  and final EE6 predictions equal within `1e-6`;
- exactly one fresh same-call generalist inference per anchor-age;
- no post-intervention generalist call;
- `env.step(action.copy())` for prefix and every branch step;
- matched-buffer controls for both transmission calculations.

Branch execution order is deterministically shuffled from
`(seed, condition_id, intervention_age)`. The six simulator branches share the
same captured branchpoint and each restores the same paired RNG state.

CALVIN's required `env.reset(robot_obs=..., scene_obs=...)` path does not return
bit-identical rendered observations on repeated resets: tiny scene differences
and occasional RGB differences are retained in `reset_fidelity.json`. To keep
the first-step raw-action equality test causally about the temporal buffer, the
first specialist call in every branch consumes an independent copy of the
captured common-prefix branchpoint observation. The action is still applied to
that branch's reset simulator via `env.step(action.copy())`; all later policy
steps consume the real observation returned by the preceding `env.step`.

GPU runs also enable strict deterministic Torch algorithms, deterministic
cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before model loading. The
experiment-local wrapper records raw action diagnostics to nine decimal places;
this changes profiling precision only, not evaluator actions. The `1e-6`
raw-equality tolerance is intentionally unchanged. If raw equality or flush
aggregation still fails, the run writes
`preflight_failure_<condition_id>_age_<age>.json` with raw values, branch order,
RNG provenance, reset fidelity, condition/isolation audits, and deterministic
runtime settings before raising.

## Outputs

Each non-dry run writes a new, non-overwritable run directory containing:

- `manifest.json`: git/source/model fingerprints, M1 provenance, exact branch
  and flush definitions, model settings, call counts, tolerances, and all
  preflight results;
- `anchors.jsonl`: selected source anchors;
- `interventions.jsonl`: branchpoint provenance, fresh inference ID,
  condition/transmission metrics, reset/isolation/raw-equality/aggregation
  audits, and branch outcomes;
- `branch_steps.jsonl`: raw `dp_action_first`, raw aggregate prediction,
  executed action, aggregation delta, voter masks/counts, gripper sign,
- `preflight_failure_*.json`: emitted only on a hard first-step invariant
  failure so the numerical/reset context survives the exception;
  per-step success, paired action/state differences, and descriptive expert
  proximity;
- `branch_summary.csv`: per-age/branch descriptive outcome summaries;
- `paired_contrasts.csv`: all required paired condition and pure-buffer
  contrasts, with paired N, mean/median/std, win/tie/loss, deterministic
  10,000-draw paired-bootstrap 95% CI, and paired binary outcome counts;
- `task_summary.csv`: descriptive per-task success summaries;
- `summary.json`: separate mechanism/outcome endpoints and non-automatic
  interpretation criteria;
- `reset_fidelity.json`: all branch-to-captured and pairwise reset metrics;
- `fresh_conditions/*.pt`: one provenance/fingerprinted same-call condition per
  anchor-age;
- `trajectories/*.npz`: prefix, branch states, actions, raw predictions, and
  success sequences.

Primary aggregate outcomes exclude anchor-age pairs whose common prefix had
already succeeded. Raw records remain persisted.

## Interpretation cases (criteria, not proof labels)

- **A — corrective fresh condition suppressed by stale buffer:** keep
  transmission ≈0.173, flush ≈1, `old_flush` near `old_keep`, and
  `full_flush` improves over both `full_keep` and, crucially, `old_flush` in
  success and/or descriptive proximity. Condition-aware invalidation is then a
  justified M2 mechanism candidate.
- **B — flush helps independently:** `old_flush` improves strongly and fresh
  flush branches add little. Redesign temporal aggregation before claiming
  generalist refresh is useful.
- **C — fresh signal transmits but is not corrective:** transmission rises and
  actions change, but `full_flush` does not improve over `old_flush`. Prioritize
  zero-reference specialist competence and on-policy/recovery training.
- **D — fresh reference is sufficient:** `ref_flush` approximately matches
  `full_flush`; fresh slow action/reference is the dominant useful channel.
- **E — fresh hidden matters:** `full_flush` consistently outperforms
  `ref_flush`; M2 should train full slow-condition transitions.

The script records these criteria and never emits an automatic proof label or a
claim based solely on transmission.

## Commands

From the repository root, the CPU-safe source/contract dry run is:

```bash
python DiT_train/0820_condition_aware_buffer_intervention/run_condition_aware_buffer_intervention.py \
  --source_run_dir DiT_train/0818_expert_specialist_trajectory/runs/trajectory_validation_s42_n50 \
  --run_name condition_buffer_dry_s42 \
  --dry_run \
  --intervention_ages 8,11 \
  --post_steps 16 \
  --seed 42
```

One- or two-anchor GPU/CALVIN preflight (shown with two anchors):

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0820_condition_aware_buffer_intervention/run_condition_aware_buffer_intervention.py \
  --source_run_dir DiT_train/0818_expert_specialist_trajectory/runs/trajectory_validation_s42_n50 \
  --run_name condition_buffer_preflight2_s42 \
  --preflight_only \
  --preflight_anchors 2 \
  --intervention_ages 8,11 \
  --post_steps 16 \
  --seed 42 \
  --device cuda \
  --use_egl
```

Formal 50-anchor launch:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0820_condition_aware_buffer_intervention/run_condition_aware_buffer_intervention.py \
  --source_run_dir DiT_train/0818_expert_specialist_trajectory/runs/trajectory_validation_s42_n50 \
  --run_name condition_buffer_validation50_s42 \
  --intervention_ages 8,11 \
  --post_steps 16 \
  --seed 42 \
  --device cuda \
  --use_egl
```

Run names are never reused silently. Choose a new name if a previous dry run or
preflight directory already exists.
