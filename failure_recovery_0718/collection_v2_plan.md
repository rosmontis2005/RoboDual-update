# Failure-recovery v2 collection plan

## Diagnosis carried forward from v1

The v1 candidate failed because its positive recovery source was too small and too
correlated: only 8 branchable failure states existed, of which 4 were in train and 1
was in test. Training used only 7 independent positive train branches. The selected
LoRA improved validation diffusion loss but worsened unseen test loss and reduced
online recovery from 1/9 to 0/9.

All three v1 branch strategies succeeded on exactly 7/60 rollouts. There is no
evidence that `forced_refresh` or `slow_override` adds a useful causal intervention;
their budget is reassigned to independent states.

## v2 target and stopping contract

- Start at canonical sequence index 200 so no v1 state is reused.
- Collect 60 states from 60 distinct failed subtasks; admit at most one state from each
  failed subtask.
- Require at least 24 branchable states (both a successful and a failed branch).
- Require at least 4 branchable states in each of train, validation, and test.
- Use six `base_seed` diffusion branches per state, horizon 80.
- Split only by SHA256 of `failure_state_id`; no state may cross splits.
- Keep the first 100 benchmark sequences excluded from collection/training.
- Stop after scanning 120 states if the quality target cannot be reached; report the
  dataset as incomplete rather than weakening admission.

At the observed 11.7% per-branch success rate, six seeds give roughly a 52.5% chance
of at least one success per state. Sixty independent states should therefore yield
about 31 branchable states while using 360 branches, versus v1's 180 branches over
only 20 states. The one-state-per-failed-subtask rule prevents adjacent snapshots
from one doomed rollout from inflating the apparent source count.

## Training admission after collection

Training may start only after integrity and fresh-process persistent replay pass,
there are at least 24 branchable states, all three state-grouped splits meet the
minimum, and each split contains a positive branch of at least eight actions.

The next LoRA run retains rank 2 / alpha 2 and base-behavior preservation. Checkpoint
selection uses validation only; unseen test and online replay remain final rejection
gates.

## Adaptive phase after 15 independent states

The first 15 states produced only 3 branchable states. Repeated zero-recovery tasks
were consuming most branch compute. Subsequent collection therefore branches only on
tasks with positive recovery evidence in v1 train/validation or v2 train:
`rotate_red_block_left`, `rotate_red_block_right`, `push_blue_block_left`,
`push_red_block_left`, `lift_pink_block_table`, `push_pink_block_left`,
`move_slider_left`, and `rotate_blue_block_left`. V2 test labels are deliberately not
used to define this allowlist. All sequences remain independent and failed tasks
outside the allowlist still terminate their source sequence; they are simply not
branched.

After sequential scanning of 223--230 produced no selected failure state, remaining
sequences are prioritized without changing their contents: first run sequences whose
first task is in the allowlist, then (if needed) those whose second task is in the
allowlist. Explicit phases checkpoint without finalizing so the same audited dataset
can be extended safely.

## Expert-guided acquisition update (2026-07-19)

Random base-seed recovery was confirmed to be the limiting data source rather than
the LoRA optimizer. The local `calvin_debug_dataset` contains 4,446 expert frames and
language-aligned trajectories. Demo-guided Cartesian controllers were audited from
fresh-process persisted failure states before admission:

- Lift recovery passed the standard CALVIN oracle on 3/3 existing states after fixing
  persistent restoration of the robot controller's `target_pos` / `target_orn`.
- Persisted lift augmentation added seven successful expert branches across three
  states; all three became branchable.
- One stack state passed the standard oracle and was augmented with 2/3 successful
  branches. Two other stack states remain excluded from expert positives.
- Place recovery is too contact-sensitive: two new deterministic place failures
  produced 0/3 successful expert branches each. This route is stopped.
- Full-trajectory translation for lift was rejected at 0/3; an explicit expert-pose
  approach/grasp/lift controller replaced it and passed 3/3.

Collection provenance is now locked by `sequence_catalogs.json`. Legacy states use
the implicit `legacy300` namespace; new catalog-1000 states use `c1000_*` IDs and
record catalog ID, catalog size, and a SHA256-derived per-sequence seed. This removes
both catalog-size aliasing and batch-order-dependent baseline sampling.

Current in-progress assessment after the resumed lift scan: 25 states, 177 branches,
110 preference pairs, and 12 branchable states (train/validation/test = 9/1/2).
Integrity passes, but the 24-state admission minimum, the four-per-split minimum, and
fresh-process replay have not passed. Training is therefore still prohibited. The
dataset remains non-finalized and safely resumable; no weakened admission threshold
is permitted.

## Table-lift acquisition phase (2026-07-19)

The next real collection phase excludes slider lift. Three newly collected slider
states remained at 0/3 even after fresh-process persisted expert augmentation, while
new table-lift failures produced branchable states. The acquisition unit remains one
natural baseline failure from one source subtask; successful baseline lift attempts
are discarded rather than converted into artificial failures.

- Use catalog `c1000` with its SHA256-derived per-sequence seed and indices >= 300.
- Target only `lift_{red,blue,pink}_block_table` as the second subtask.
- Prefer sources whose first task is drawer, slider, LED/lightbulb so the policy has a
  high probability of reaching the selected lift task.
- Scan the validation-sensitive candidates `615,642,806,841,936` first, followed by
  `519,566,590,637,638,666,675,686,724,747,912,913,984`.
- Run three `base_seed` and three `demo_guided` branches per captured state, horizon
  80, and save at most one state per failed subtask.
- Keep finalization deferred. Re-run grouped assessment after the batch; do not train
  unless 24 branchable states, four per split, integrity, and replay gates all pass.

The split hash includes the actual candidate step (`k032`, `k040`, or `k048`), so the
listed validation-sensitive sequences improve coverage but do not guarantee the
resulting split. No state is reassigned after observing its outcome.

The first reviewed table-only batch scanned 18 sources and found one natural failure,
`c1000_s0806_t1_k032` in validation. All six online branches and all three persisted
expert branches recovered successfully, so the state had positives but no negative
branch. The collector is therefore extended for future states to retain the first 80
actions of the actual baseline continuation from each captured snapshot. This branch
is labelled negative only when the source subtask is subsequently confirmed to fail;
successful source subtasks still discard the candidate. This supplies a real
same-state negative without no-op controls, artificial perturbations, cross-state
pairing, or retrospective reconstruction of old states.

## Validation-focused table-lift phase (2026-07-19)

After enabling the real baseline continuation, a 15-source table batch produced two
natural failures. `c1000_s0658_t1_k032` formed ten same-state preference pairs and
validated the new negative path; `c1000_s0468_t1_k032` remained a hard negative after
0/3 persisted expert recovery. The in-progress dataset now has 28 states, 203
branches, 120 pairs, and 13 branchable states (train/validation/test = 10/1/2).

The next phase targets the validation bottleneck without changing outcomes after
observation:

- Scan the remaining second-subtask `k032 -> validation` source `963` first.
- Then scan third-subtask table lifts whose source hash maps `k032` to validation,
  ordered by easy prerequisite count: `683,342,349,360,444,452,662,684,812,842,891,899`.
- Allow only red/blue/pink table lift to create a failure state. A failed prerequisite
  terminates its sequence and is not stored.
- Preserve the actual baseline failed continuation (up to 80 actions), plus three
  base-seed and three demo-guided branches. Persisted expert augmentation is attempted
  once only for a new state that has a real negative but no positive.
- Keep one state per failed table-lift subtask and keep finalization deferred. Do not
  reassign a state if its actual eligible step is `k040` or `k048` and hashes to a
  different split.

Phase result: 13 sources produced five table-lift failures. States 683 and 891 formed
12 same-state pairs each; both are validation. States 444, 684, and 899 retained real
failed continuations but remained at 0/3 persisted expert success and are kept only as
hard negatives. The batch increased the dataset to 33 states, 247 branches, and 144
pairs without reassigning outcomes or admitting non-table prerequisite failures.

## Test/validation split repair (2026-07-19--20)

Split repair was performed sequentially and retained the original natural-failure
contract. No successful baseline task was converted into a failure state, and no
state was reassigned after observing its split.

- Validation was repaired first by `c1000_s0375_t2_k048`. It contains a real failed
  baseline continuation and successful same-state recovery branches, increasing the
  branchable validation count from 3 to 4.
- Candidate selection initially reproduced the split with the full SHA256 integer.
  Review found that the collector actually uses only `hexdigest()[:8]`; subsequent
  candidates were selected with the collector's exact `_state_split` implementation.
- Nine first-subtask table-lift candidates all succeeded naturally and were discarded.
  Short-prerequisite candidates were then scanned without forced failures.
- `c1000_s0906_t1_k048` produced a natural blue-table-lift failure, a real negative
  continuation, and successful same-state recovery branches. It increased the
  branchable test count from 3 to 4.

Post-repair assessment: 41 failure states, 309 branches, 216 same-state preference
pairs, and 22 branchable states with train/validation/test = 14/4/4. Dataset integrity
passes with no reported errors, and all six collector/resume contract tests pass.
The dataset remains deferred and non-finalized. Training remains prohibited because
the overall 24-branchable-state target and fresh-process replay gates have not yet
passed.

## Final acquisition, admission, and protected training (2026-07-20)

The final acquisition phases shifted from low-yield deep table-lift sources to
second- and third-subtask `stack_block` failures, including slider-lift-to-stack
sources. The finalized independent dataset contains 62 failure states, 456 branches,
306 same-state preference pairs, and 31 branchable states. The grouped branchable
split is train/validation/test = 20/4/7; the full state split is 35/10/17.

Admission passed without weakening the reviewed thresholds:

- Integrity reported no errors and every split contains trainable positives and
  both positive and negative outcomes.
- Two in-process exact fixed-action audits reproduced both outcome and branch length.
- A fresh-process persistent replay audit passed on three states, including fixed
  action outcome/length and the configured robot, scene, and RGB restoration bounds.
- `analyze_recovery_dataset.py --min_branchable_states 24` recorded
  `training_admitted: true` in `dataset_assessment.json`.

Protected stage-1 LoRA used rank 2 / alpha 2, 800 steps, positive recovery behavior
cloning, matched normal replay preservation, and validation-only checkpoint
selection. Step 200 was selected. Recovery validation supervised loss changed from
0.2783193 to 0.2767965 (improvement 0.0015228, about 0.55%). Selected-checkpoint
normal drift was 0.0001512 and gripper drift was 0.0000627, below the respective
0.0002 and 0.0001 gates. The unseen candidate test recovery supervised loss is
0.4117187; normal test drift is 0.0001565 with gripper drift 0.0000650. This small
offline improvement is not sufficient for acceptance without online replay.

The online replay path was hardened after a bad `task_ABC_D` dataset argument caused
an immediate `FileNotFoundError` whose Rich exception formatting appeared to hang.
Replay now atomically checkpoints every rollout, reports progress, records a
`collecting`/`complete` status, and resumes only after exact data/model/split/horizon/
seed configuration validation. The correct environment source is
`calvin_debug_dataset`.

The complete base-model test replay covers all 17 test states with three fixed seeds
each and horizon 80. It achieved 4/51 (7.84%): 3/9 on
`lift_red_block_slider`, 1/24 on `stack_block`, and 0 on all other represented tasks.
The result is stored in `eval_base_v3_test_replay.json` with `status: complete`.

Candidate online replay subsequently completed with the same 51 state/seed pairs and
the selected `specialist_transition_lora_merged_ema.pt`. It also achieved 4/51
(7.84%), for zero overall success-rate change. Paired analysis found one gain and one
loss (two-sided exact sign-test p = 1.0): `lift_red_block_slider` improved from 3/9
to 4/9, while the targeted `stack_block` task regressed from 1/24 to 0/24. All other
46 failures and three successes were unchanged.

The checkpoint therefore passes offline validation selection and normal-behavior
preservation, but fails the final online net-improvement and target-task
non-regression gates. It is rejected for promotion. The authoritative decision is
stored in `recovery_stage1_v3_evaluation_decision.json`; base, candidate, and paired
online evidence are stored in `eval_base_v3_test_replay.json`,
`eval_candidate_v3_test_replay.json`, and `eval_v3_paired_comparison.json`.
