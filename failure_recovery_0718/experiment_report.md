# Failure-recovery experiment, 2026-07-18

## Route

The experiment adapts the intervention and same-state comparison semantics of
[Sirius](https://github.com/UT-Austin-RPL/sirius) to the existing CALVIN evaluator.
[robomimic](https://github.com/ARISE-Initiative/robomimic) is the reference for
state-grouped imitation datasets. [DPPO](https://github.com/irom-princeton/dppo)
was deliberately deferred because the collected recovery coverage is still too small
for an on-policy RL stage.

The copied evaluator collects snapshots only from subtasks that subsequently fail,
restores the simulator and evaluator runtime, and launches `base_seed`,
`forced_refresh`, and `slow_override` branches. CALVIN's task oracle supplies the
branch label. The first 100 canonical sequences are excluded, and all branches from
one failure state remain in a SHA256-derived train/validation/test group.

## Dataset and audit

- 20 failure states, 180 branches, 21 successful branches, 86 positive/negative pairs.
- 8 branchable states (40%): train 4, validation 3, test 1.
- Trainable positive branches: train 7, validation 2, test 4.
- Strategy success was 7/60 (11.7%) for each strategy. The data does not establish an
  advantage for forced refresh or slow override over resampling.
- Fresh-process persistent replay passed on 3/3 audited states: RGB mean absolute
  error 0, end-effector 6D max error 6.43e-4, scene max error 5.69e-8, and fixed
  branch oracle outcomes matched.
- In-memory fixed-action replay reproduced action count and oracle label but not the
  exact terminal pose. PyBullet contact rollouts must not be described as bitwise
  deterministic.

## Training

Both runs use rank-2, alpha-2 LoRA on the six existing V13 action-conditioning
targets, positive recovery behavior cloning, and frozen-base normal replay.

- v1: 1000 steps with an absolute 2e-4 validation-improvement gate. No checkpoint
  qualified; the output intentionally fell back to the base model.
- v2: 800 steps with a scale-adjusted 4e-5 gate (about 2.1% of validation baseline).
  Step 700 was selected. Validation recovery loss improved from 0.00194291 to
  0.00187969 (3.25%), while normal prediction drift remained within the protection
  constraint.
- Unseen offline test recovery loss worsened from 0.0194342 to 0.0197573 (+1.66%).

## Online replay result

The base and v2 candidate used the same three unseen test states, three diffusion
seeds per state, and an 80-step horizon.

- Base: 1/9 successful rollouts (11.1%).
- v2 step 700: 0/9 successful rollouts (0%).
- The candidate lost the base model's only successful `lift_pink_block_table` rollout.

The candidate is rejected and must not be deployed. The next iteration should expand
the number of independent branchable states and successful recovery branches,
especially for tasks with zero successful branches. Preference or DPPO training should
remain deferred until held-out recovery coverage is materially larger.
