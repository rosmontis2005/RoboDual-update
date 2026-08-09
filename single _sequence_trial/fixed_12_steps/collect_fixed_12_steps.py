#!/usr/bin/env python3
"""Collect sequence 60 with the historical uniform age-12 slow-call policy.

All model loading, rollout, and trace hooks are shared with the paired original
8-step collector. Only the evaluator policy changes: slow calls happen at ages
0, 12, 24, ... and the 8-action reference is empty at ages 8 through 11.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAIR_ROOT = HERE.parent / "original_8_steps"
REPO_ROOT = HERE.parents[1]
VLA_SCRIPTS = REPO_ROOT / "vla-scripts"
CALVIN_ROOT = REPO_ROOT.parent / "calvin"
for path in reversed((
    HERE,
    PAIR_ROOT,
    VLA_SCRIPTS,
    CALVIN_ROOT / "calvin_models",
    CALVIN_ROOT / "calvin_env",
    CALVIN_ROOT / "calvin_env" / "tacto",
)):
    sys.path.insert(0, str(path))
os.environ.setdefault("CALVIN_ROOT", str(CALVIN_ROOT))

import hydra
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import collect_original_8_steps as paired
import dual_sys_evaluation_0424test as age_evaluator
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from trace_capture import OnlineTraceCapture, TraceWriter, cpu_clone, sha256_file


SLOW_PERIOD = 12
EMPTY_REF_AFTER_AGE = 8


def is_fixed_age12_slow_step(step: int) -> bool:
    return int(step) % SLOW_PERIOD == 0


class FixedAge12Evaluation(age_evaluator.DualSystemCalvinEvaluation):
    """Historical age-empty evaluator pinned to a uniform 12-step refresh."""

    def __init__(self, *args, **kwargs):
        kwargs.update(
            profile_sample_var_k=1,
            slow_trigger_policy="age_empty",
            max_slow_age=SLOW_PERIOD,
            empty_ref_after_age=EMPTY_REF_AFTER_AGE,
            slow_handover_steps=0,
            slow_handover_blend_hidden=False,
            action_delta_limit_ee6=0.0,
            action_jerk_limit_ee6=0.0,
        )
        super().__init__(*args, **kwargs)
        assert self.temporal_size == 8


class FixedAge12TraceCapture(OnlineTraceCapture):
    """Fail fast if the live evaluator deviates from the age-12 contract."""

    def finalize_step(self, *, profile, **kwargs):
        if self.current is None:
            raise RuntimeError("No active trace step")
        step = int(self.current["meta"]["step"])
        age = step % SLOW_PERIOD
        expected_cond = max(0, 8 - age) if age < EMPTY_REF_AFTER_AGE else 0
        required_profile = {
            "slow_trigger_policy": "age_empty",
            "max_slow_age": SLOW_PERIOD,
            "empty_ref_after_age": EMPTY_REF_AFTER_AGE,
            "slow_age_after": age,
            "num_cond_actions": expected_cond,
        }
        mismatches = {
            key: {"expected": expected, "actual": profile.get(key)}
            for key, expected in required_profile.items()
            if profile.get(key) != expected
        }
        if mismatches:
            raise AssertionError(f"fixed_age12 profile contract mismatch at step {step}: {mismatches}")
        return super().finalize_step(profile=profile, **kwargs)


def build_parser():
    parser = paired.build_parser()
    parser.description = "Collect the paired fixed-age-12 trace for canonical sequence 60"
    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = "Must be new/empty. Default: fixed_12_steps/runs/seq060_seed42_<timestamp>"
    return parser


def make_run_dir(args) -> Path:
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = HERE / "runs" / f"seq{args.sequence_index:03d}_seed{args.seed}_{stamp}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def source_contract() -> dict:
    files = {
        "evaluator": VLA_SCRIPTS / "dual_sys_evaluation_0424test.py",
        "historical_age12_entrypoint": VLA_SCRIPTS / "evaluate_calvin_codex_test_0424test.py",
        "environment_factory_reference": VLA_SCRIPTS / "evaluate_calvin_0428.py",
        "specialist": REPO_ROOT / "prismatic/models/policy/diffusion_policy.py",
        "dit": REPO_ROOT / "prismatic/models/policy/diffusion_transformer.py",
        "generalist_remote_code": REPO_ROOT.parent / "models/generalist/modeling_prismatic.py",
        "collector": HERE / "collect_fixed_12_steps.py",
        "trace_capture_bridge": HERE / "trace_capture.py",
        "shared_trace_capture": PAIR_ROOT / "trace_capture.py",
        "paired_fixed8_collector": PAIR_ROOT / "collect_original_8_steps.py",
    }
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in files.items()}


def main(args) -> None:
    if not (0 <= args.sequence_index < args.catalog_size):
        raise ValueError("sequence_index must be within the generated catalog")
    if args.max_subtasks < 1 or args.max_subtasks > 5:
        raise ValueError("max_subtasks must be in [1, 5]")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one of 4-bit and 8-bit generalist loading")

    run_dir = make_run_dir(args)
    seed_everything(args.seed)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=12))])
    if accelerator.num_processes != 1:
        raise RuntimeError("Full tensor collection currently requires a single Accelerator process")

    processor, action_tokenizer, dual_system = paired.load_models(args, accelerator)
    evaluator = FixedAge12Evaluation(
        dual_system,
        processor,
        action_tokenizer,
        profile_steps=True,
        profile_sample_var_interval=SLOW_PERIOD,
    )

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    env = paired.original.make_env(
        str(CALVIN_ROOT / "dataset" / args.dataset_subdir),
        observation_space,
        accelerator.device,
        args.use_egl,
    )
    sequences = list(get_sequences(args.catalog_size))
    initial_state, tasks = sequences[args.sequence_index]
    tasks = list(tasks[: args.max_subtasks])
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    conf_dir = CALVIN_ROOT / "calvin_models/conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    sequence_payload = {
        "sequence_index_zero_based": args.sequence_index,
        "catalog_size": args.catalog_size,
        "initial_state_symbolic": cpu_clone(initial_state),
        "initial_robot_obs": cpu_clone(robot_obs),
        "initial_scene_obs": cpu_clone(scene_obs),
        "tasks": tasks,
        "instructions": {task: str(annotations[task][0]) for task in tasks},
    }
    (run_dir / "sequence.json").write_text(
        json.dumps(sequence_payload, indent=2, default=lambda value: value.tolist()) + "\n"
    )

    writer = TraceWriter(
        run_dir,
        manifest={
            "entrypoint": str(Path(__file__).resolve()),
            "paired_baseline": str((PAIR_ROOT / "collect_original_8_steps.py").resolve()),
            "sequence_index": args.sequence_index,
            "catalog_size": args.catalog_size,
            "seed": args.seed,
            "ep_len": args.ep_len,
            "max_subtasks": args.max_subtasks,
            "dataset_subdir": args.dataset_subdir,
            "runtime_options": {
                "fast_num_inference_steps": int(args.fast_num_inference_steps),
                "with_cfg": bool(args.with_cfg),
                "with_tactile": bool(args.with_tactile),
                "with_depth": True,
                "with_gripper": True,
                "use_egl": bool(args.use_egl),
                "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
                "device_map": args.device_map,
                "attn_implementation": args.attn_implementation,
            },
            "generalist_runtime": {
                "load_in_4bit": bool(args.load_in_4bit),
                "load_in_8bit": bool(args.load_in_8bit),
                "expected_compute_dtype": (
                    "torch.float16" if (args.load_in_4bit or args.load_in_8bit) else "torch.bfloat16"
                ),
            },
            "generalist_path": str(Path(args.generalist_path).resolve()),
            "specialist_path": str(Path(args.specialist_path).resolve()),
            "specialist_checkpoint_sha256": sha256_file(Path(args.specialist_path)),
            "diffusion_scheduler": cpu_clone(dict(evaluator._fast_system().noise_scheduler.config)),
            "fixed_age12_contract": {
                "first_slow_step": 0,
                "later_condition": "step % 12 == 0",
                "expected_steps": list(range(0, args.ep_len, SLOW_PERIOD)),
                "slow_period": SLOW_PERIOD,
                "temporal_size": 8,
                "empty_ref_after_age": EMPTY_REF_AFTER_AGE,
                "empty_reference_ages": [8, 9, 10, 11],
                "slow_handover_steps": 0,
                "action_delta_limit_ee6": 0.0,
                "action_jerk_limit_ee6": 0.0,
                "profile_sample_var_k": 1,
            },
            "source_contract": source_contract(),
            "rng_note": (
                "Uses the same original global torch RNG mechanism as the paired fixed-mod-8 collector after "
                "seed_everything(seed). The slow generalist runs with do_sample=False and normally consumes no RNG, so "
                "specialist diffusion noise remains aligned by cumulative specialist-call ordinal while both rollouts have "
                "executed the same number of environment steps. If a subtask finishes at different lengths, later subtasks "
                "start at different global RNG positions. Every actual RNG state and initial-noise tensor is recorded."
            ),
            "historical_comparison_note": (
                "The historical age-12 selection artifact used ep_len=360. This collector intentionally keeps ep_len from "
                "the paired original-8 collection (default 240), so it reproduces the age-12 policy but not the historical "
                "run's timeout boundary."
            ),
            "subtask_token_note": (
                "The current model has no standalone subtask-token tensor. The canonical subtask id, language string, "
                "processor input_ids and generated token ids are stored instead."
            ),
        },
    )
    capture = FixedAge12TraceCapture(
        evaluator,
        writer,
        expected_slow_call=is_fixed_age12_slow_step,
        schedule_label="fixed_age12",
    )
    outcomes = []
    try:
        for subtask_index, subtask in enumerate(tasks):
            success, steps = paired.rollout_subtask(
                env=env,
                evaluator=evaluator,
                capture=capture,
                task_oracle=task_oracle,
                instruction=str(annotations[subtask][0]),
                subtask=subtask,
                sequence_index=args.sequence_index,
                subtask_index=subtask_index,
                ep_len=args.ep_len,
            )
            outcomes.append(
                {"subtask_index": subtask_index, "task": subtask, "success": success, "steps": steps}
            )
            if not success:
                break
    finally:
        capture.close()
        env.close()
    writer.finalize(
        {
            "successful_subtasks": sum(int(item["success"]) for item in outcomes),
            "outcomes": outcomes,
        }
    )
    print(f"Trace complete: {run_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
