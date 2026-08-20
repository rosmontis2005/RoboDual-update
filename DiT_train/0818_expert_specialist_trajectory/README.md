# Expert vs specialist closed-loop trajectory diagnostic

This is a diagnosis-only experiment. It does not train or modify M1, the core
evaluator, or CALVIN. It asks whether the current M1 specialist drifts away from
the persisted CALVIN expert trajectory when both start at the same language
subtask state and use the same frozen slow-system condition.

The experiment uses four branches:

- **A — expert dataset reference:** persisted states `anchor+k`, `k=0..12`, and
  persisted `rel_actions[anchor+k]`, `k=0..11`.
- **B — expert-action replay:** reset CALVIN with the persisted anchor
  `robot_obs` and `scene_obs`, then execute the 12 persisted expert actions.
- **C — specialist closed loop:** reset to the same persisted state, inject the
  frozen anchor condition, and execute the final action returned by the current
  evaluator. This retains temporal aggregation and gripper post-processing.
- **D — teacher-forced observations:** use an independent evaluator runtime,
  inject the same condition values, and call `step()` on successive persisted
  expert observations without controlling the simulator.

Branches C and D use separate temporal buffers, history deques, previous-image
buffers, and action state. Their initial diffusion RNG stream is paired so their
action difference is not merely caused by different sampled diffusion noise.
The normal evaluator history evolves in each branch, but history is not corrupted
or treated as an experimental variable.

## Contracts

Anchor selection reads
`task_D_D/training/lang_annotations/auto_lang_ann.npy`. The `trajectory_id` and
stable split are byte-equivalent to
`DiT_train/data_collection/collect_age_extended_expert.py`:

```text
trajectory_id = calvin_training_lang_{episode_i:06d}_{start}_{end}_{task}
SHA256(trajectory_id) bucket: train < 70, validation < 85, otherwise test
```

Only the language subtask start is used as an anchor. An episode is eligible
only if every persisted frame `anchor..anchor+18` exists within its inclusive
annotation bounds. Selection is deterministic, task-balanced round robin with
seed 42.

For every anchor, the generalist is called exactly once with the persisted
anchor RGB and `do_sample=False`. The `slow_action [1,8,7]` and `slow_hidden`
from that same call are saved to `conditions/`. No rollout step may refresh the
generalist. The evaluator receives `last_slow_step=0`, producing:

```text
age:               0 1 2 3 4 5 6 7 8 9 10 11
num_cond_actions:  8 7 6 5 4 3 2 1 0 0  0  0
```

Time indexing is explicit: `s_expert[k]` is persisted frame `anchor+k`,
`a_expert[k]` is that frame's `rel_actions`, M1 generates `a_policy[k]` at
`s_policy[k]`, and `env.step(a_policy[k].copy())` yields `s_policy[k+1]`.
There are 12 actions (`k=0..11`) and 13 states (`k=0..12`). State 12 is the
state after executing the age-11 action. Age 7 to age 8 is the reference
expiration boundary.

The phrase `expert_trajectory_action_difference` is used in records: an expert
action is not claimed to be the strictly optimal action after the policy state
has already drifted.

## Commands

Use the repository's `dualsys_env` environment. Run from the repository root.
Run names are never silently overwritten; choose a new name if a directory
already exists.

Dry-run smoke test (dataset parsing and 10 anchors only; no model or env):

```bash
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0818_expert_specialist_trajectory/run_expert_specialist_trajectory.py \
  --run_name trajectory_dry_s42_n10 \
  --dry_run --max_anchors 10 --save_visual_anchors 0
```

One-anchor preflight:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0818_expert_specialist_trajectory/run_expert_specialist_trajectory.py \
  --run_name trajectory_preflight_s42_n1 \
  --preflight_only --preflight_anchors 1 --device cuda --use_egl \
  --save_visual_anchors 1
```

Two-anchor preflight uses the same command with a fresh run name and
`--preflight_anchors 2`.

Formal 50-anchor run:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  DiT_train/0818_expert_specialist_trajectory/run_expert_specialist_trajectory.py \
  --run_name trajectory_validation_s42_n50 \
  --max_anchors 50 --device cuda --use_egl \
  --save_visual_anchors 2
```

Defaults are validation stable split, seed 42, M1 checkpoint
`DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt`,
generalist `/home/rosmontis/Projects/dualsys/models/generalist`, CALVIN dataset
`/home/rosmontis/Projects/dualsys/calvin/dataset/task_D_D`, 4-bit NF4,
`low_cpu_mem_usage`, 10 fast inference steps, depth and gripper enabled, tactile
and CFG disabled, no handover, and no slew limit.

`--use_egl` is appropriate for the normal headless GPU host. Omit it only when
the local CALVIN/PyBullet setup intentionally uses another rendering path.

## Preflight checks

`--preflight_only` runs one or two deterministic anchors through all branches
and asserts:

1. the environment resets with persisted `robot_obs` and `scene_obs`;
2. both reset-vs-anchor differences are measured and reported;
3. all 12 expert replay actions execute and produce 13 states;
4. neither specialist branch refreshes the frozen slow condition;
5. both branch age sequences are exactly `0..11`;
6. both `num_cond_actions` sequences equal `8,7,6,5,4,3,2,1,0,0,0,0`;
7. branches C and D use equal condition values and the same condition ID;
8. their mutable controller state and injected tensor storage are independent;
9. the configured M1 checkpoint is loaded through the evaluation `DualSystem`
   EMA wrapper;
10. checkpoint missing and unexpected keys are emitted in manifest and terminal
    output;
11. history corruption/ablation is absent.

The current M1 EMA checkpoint contains the wrapper-only placeholder keys
`online_model._dummy_variable` and `ema_model._dummy_variable`. They are loaded
with the evaluator's existing `strict=False` contract, recorded under
`raw_unexpected_keys` and `ignored_ema_compatibility_keys`, and do not count as
architecture mismatches. Any other missing or unexpected key remains fatal.

Replay fidelity has no hidden pass threshold. Actual errors are always written
to the output. In particular, the script does not relax a tolerance to make a
preflight pass.

If CUDA, bitsandbytes, CALVIN, EGL/PyBullet, model weights, or simulator assets
are unavailable, non-dry modes fail with the missing dependency or path. They do
not fabricate trajectory results.

## Outputs

Each run is written under `runs/<run_name>/`:

- `manifest.json`: full paths and hashes, git revision, selection and split
  rules, model architecture/settings, checkpoint missing/unexpected keys,
  condition/generalist-call audit, all per-anchor preflight contracts, and file
  provenance.
- `anchors.jsonl`: one row per selected language episode: trajectory ID,
  language episode index, task/instruction, inclusive task bounds, anchor frame,
  all source indices `anchor..anchor+18`, condition ID, and trajectory artifact.
- `conditions/*.pt`: detached CPU `slow_action`, `slow_hidden`, same-call IDs,
  normalization metadata, and anchor provenance. Load with `map_location="cpu"`.
- `trajectories/*.npz`: complete Branch A/B/C state arrays for robot, selected
  proprio, scene, both RGB streams, and both depth streams; also expert,
  teacher-forced, and closed-loop action arrays.
- `trajectory_steps.jsonl`: one row per anchor/state index. It contains compact
  expert/replay/policy robot and scene states, all raw state metrics, replay-
  baseline-adjusted diagnostics, and (for `k=0..11`) all three actions plus the
  evaluator profile, `dp_action_first`, raw/aggregated action prediction,
  age/count, gripper agreement, and action differences.
- `age_summary.csv`: mean, median, p90, and count for replay divergence,
  closed-loop divergence, baseline-adjusted divergence, and action metrics at
  every state index. `k=12` is labeled `state_12_after_age11_action`.
- `task_summary.csv`: the same aggregate families grouped by task.
- `summary.json`: nested overall, per-age, per-task, and highlighted age 7,
  age 8, age 11, and state 12 summaries. `boundary_7_to_8` is explicit.
- `replay_fidelity.json`: reset records plus overall/per-age Branch B versus
  Branch A fidelity. This is the validity gate for interpreting policy drift.
- `visuals/` (optional): PNG frame directories for the first requested anchors,
  separately for expert, replay, and closed-loop static/gripper RGB. GIF support
  is deliberately not required.

Robot metrics include full-vector L2/max-absolute and end-effector-first-six
L2/max-absolute. Scene metrics include L2/max-absolute. RGB metrics are raw pixel
MAE. Depth metrics include `depth_static_mae`, `depth_static_rmse`,
`depth_gripper_mae`, and `depth_gripper_rmse`; they compare the raw persisted and
simulator depth values before the evaluator's normalization. Action metrics use
the first six continuous channels; gripper agreement compares signs.
`aggregation_delta_ee6` is taken from the deployment evaluator's profile, not
recomputed from an unexecuted action shortcut.

## Interpretation criteria

### Case A — accumulated closed-loop drift supported

If expert replay approximately matches the dataset, teacher-forced M1 actions
approximately match expert actions, closed-loop state divergence grows with age,
and closed-loop versus teacher-forced action difference also grows, the result
supports **policy-induced closed-loop state distribution shift / accumulated
drift**. The next M2 direction is deployment-aware or on-policy roll-in data.

### Case B — expert-state imitation quality is already weak

If teacher-forced M1 differs materially from expert actions at early ages,
prioritize M1 optimization, checkpoint correctness, task imbalance, expert
target learning, and model capacity before immediately pursuing on-policy
training.

### Case C — reference expiration is the leading suspect

If teacher-forced and closed-loop actions both degrade near age 8 while
policy-versus-expert state divergence was small beforehand, prioritize the
**reference expiration mechanism**, not accumulated state shift.

### Case D — state-shift hypothesis not supported

If the specialist closed-loop trajectory remains near the expert trajectory,
return to full 100-sequence low-success-task analysis, task-specific weaknesses,
and scheduling/slow-call strategy.

### Case E — replay control invalidates causal interpretation

If expert-action replay itself does not reproduce the persisted expert
trajectory adequately, this experiment cannot establish policy-induced drift.
Resolve reset/simulator replay fidelity first. Always inspect raw replay metrics;
the baseline-adjusted specialist metric is supplementary and never replaces
them.
