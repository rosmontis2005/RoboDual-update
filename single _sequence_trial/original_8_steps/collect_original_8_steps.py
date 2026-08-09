#!/usr/bin/env python3
"""Collect a complete online trace for canonical CALVIN sequence 60.

Inference is delegated to the repository's existing evaluator and model
implementations.  This entry point fixes the scheduling arguments to the
original RoboDual fixed-mod-8 contract and adds observation/tensor tracing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
VLA_SCRIPTS = REPO_ROOT / "vla-scripts"
CALVIN_ROOT = REPO_ROOT.parent / "calvin"
for path in (
    VLA_SCRIPTS,
    CALVIN_ROOT / "calvin_models",
    CALVIN_ROOT / "calvin_env",
    CALVIN_ROOT / "calvin_env" / "tacto",
):
    sys.path.insert(0, str(path))
os.environ.setdefault("CALVIN_ROOT", str(CALVIN_ROOT))

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

import evaluate_calvin_0428 as original
import dual_sys_evaluation as legacy_fixed8
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from trace_capture import OnlineTraceCapture, TraceWriter, cpu_clone, physics_state, sha256_file


class OriginalFixedMod8Evaluation(legacy_fixed8.DualSystemCalvinEvaluation):
    """Original evaluator with immutable fixed-mod-8 configuration."""

    def __init__(self, *args, **kwargs):
        kwargs["profile_sample_var_k"] = 1
        super().__init__(*args, **kwargs)
        assert self.temporal_size == 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generalist_path", default=str(REPO_ROOT.parent / "models/generalist"))
    parser.add_argument(
        "--specialist_path",
        default=str(REPO_ROOT.parent / "models/specialist/Specialist+Depth+Gripper.pt"),
    )
    parser.add_argument(
        "--dataset_subdir",
        default="calvin_debug_dataset",
        help="Environment config source; this is the locally available CALVIN ABC-D debug dataset.",
    )
    parser.add_argument("--sequence_index", type=int, default=60)
    parser.add_argument("--catalog_size", type=int, default=100)
    parser.add_argument("--ep_len", type=int, default=240)
    parser.add_argument("--max_subtasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_num_inference_steps", type=int, default=10)
    parser.add_argument("--with_cfg", action="store_true")
    parser.add_argument("--with_tactile", action="store_true")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="none")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Must be new/empty. Default: original_8_steps/runs/seq060_seed42_<timestamp>",
    )
    parser.set_defaults(load_in_4bit=True)
    return parser


def source_contract() -> dict:
    files = {
        "evaluator": REPO_ROOT / "vla-scripts/dual_sys_evaluation.py",
        "legacy_evaluation_entrypoint": REPO_ROOT / "vla-scripts/evaluate_calvin_codex_test.py",
        "environment_factory_reference": REPO_ROOT / "vla-scripts/evaluate_calvin_0428.py",
        "specialist": REPO_ROOT / "prismatic/models/policy/diffusion_policy.py",
        "dit": REPO_ROOT / "prismatic/models/policy/diffusion_transformer.py",
        "generalist_remote_code": REPO_ROOT.parent / "models/generalist/modeling_prismatic.py",
        "collector": HERE / "collect_original_8_steps.py",
        "trace_capture": HERE / "trace_capture.py",
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path)} for name, path in files.items()
    }


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = HERE / "runs" / f"seq{args.sequence_index:03d}_seed{args.seed}_{stamp}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_models(args: argparse.Namespace, accelerator: Accelerator):
    quantization_config = None
    model_dtype = torch.bfloat16
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model_dtype = torch.float16
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_dtype = torch.float16
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": model_dtype,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "trust_remote_code": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation
    generalist = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **model_kwargs).eval()

    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from train_spacialist_calvin import DualSystem

    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    specialist = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}},
        noise_scheduler=scheduler,
        n_action_steps=8,
        num_inference_steps=args.fast_num_inference_steps,
        vision_encoder="DINO",
        # The specialist checkpoint below contains both the online and EMA DINO
        # encoder weights.  Asking timm for pretrained initialization here is
        # redundant and makes an otherwise local evaluation depend on a live
        # Hugging Face connection.
        vision_encoder_pretrained=False,
        with_depth=True,
        progressive_noise=False,
        with_gripper=True,
        with_tactile=args.with_tactile,
        cond_drop_chance=0.1 if args.with_cfg else 0.0,
    ).eval().to(accelerator.device)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dual_system = DualSystem(generalist, specialist, action_tokenizer)
    specialist_state = torch.load(args.specialist_path)
    incompatible = dual_system.ema_fast_system.load_state_dict(specialist_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.endswith("._dummy_variable")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Specialist checkpoint does not fully cover the inference model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    dual_system = accelerator.prepare(dual_system, device_placement=[True])
    dual_system.eval()
    return processor, action_tokenizer, dual_system


def rollout_subtask(
    *,
    env,
    evaluator,
    capture,
    task_oracle,
    instruction: str,
    subtask: str,
    sequence_index: int,
    subtask_index: int,
    ep_len: int,
) -> tuple[bool, int]:
    obs = env.get_obs()
    evaluator.reset()
    start_info = env.get_info()
    for step in range(ep_len):
        pre_info = env.get_info()
        capture.begin_step(
            sequence_index=sequence_index,
            subtask_index=subtask_index,
            subtask=subtask,
            instruction=instruction,
            step=step,
            pre_obs=obs,
            pre_info=pre_info,
            pre_physics=physics_state(env),
        )
        action = evaluator.step(obs, instruction, step)
        obs, _, _, current_info = env.step(action)
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        capture.finalize_step(
            executed_action=action,
            post_obs=obs,
            post_info=current_info,
            post_physics=physics_state(env),
            task_success=bool(current_task_info),
            profile=evaluator.last_step_profile,
        )
        if current_task_info:
            return True, step + 1
    return False, ep_len


def main(args: argparse.Namespace) -> None:
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

    processor, action_tokenizer, dual_system = load_models(args, accelerator)
    evaluator = OriginalFixedMod8Evaluation(
        dual_system,
        processor,
        action_tokenizer,
        profile_steps=True,
        profile_sample_var_interval=8,
    )

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    env = original.make_env(
        str(CALVIN_ROOT / "dataset" / args.dataset_subdir),
        observation_space,
        accelerator.device,
        args.use_egl,
    )
    sequences = list(get_sequences(args.catalog_size))
    initial_state, tasks = sequences[args.sequence_index]
    tasks = list(tasks[: args.max_subtasks])

    # CALVIN derives a deterministic temporary numpy seed from each symbolic
    # initial condition, so this state is identical whether seq60 is evaluated
    # alone or after the preceding catalog entries.
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
    (run_dir / "sequence.json").write_text(json.dumps(sequence_payload, indent=2, default=lambda x: x.tolist()) + "\n")

    writer = TraceWriter(
        run_dir,
        manifest={
            "entrypoint": str(Path(__file__).resolve()),
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
                "expected_compute_dtype": "torch.float16" if (args.load_in_4bit or args.load_in_8bit) else "torch.bfloat16",
            },
            "generalist_path": str(Path(args.generalist_path).resolve()),
            "specialist_path": str(Path(args.specialist_path).resolve()),
            "specialist_checkpoint_sha256": sha256_file(Path(args.specialist_path)),
            "diffusion_scheduler": cpu_clone(dict(evaluator._fast_system().noise_scheduler.config)),
            "fixed_mod8_contract": {
                "first_slow_step": 0,
                "later_condition": "(step + 1) % 8 == 0",
                "expected_steps": [0] + list(range(7, args.ep_len, 8)),
                "temporal_size": 8,
                "slow_handover_steps": 0,
                "action_delta_limit_ee6": 0.0,
                "action_jerk_limit_ee6": 0.0,
                "profile_sample_var_k": 1,
            },
            "source_contract": source_contract(),
            "rng_note": (
                "Uses the original global torch RNG mechanism after seed_everything(seed); no per-step generator is injected. "
                "This targeted run does not consume the torch/CUDA random draws from canonical sequences 0..59, so it is a "
                "new standalone seed-42 trace rather than a bitwise replay of historical 0413 sequence 60. Every actual RNG "
                "state and initial noise tensor used in this run is recorded."
            ),
            "subtask_token_note": (
                "The current model has no standalone subtask-token tensor. The canonical subtask id, language string, "
                "processor input_ids and generated token ids are stored instead."
            ),
        },
    )
    capture = OnlineTraceCapture(evaluator, writer)
    outcomes = []
    try:
        for subtask_index, subtask in enumerate(tasks):
            success, steps = rollout_subtask(
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
            outcomes.append({"subtask_index": subtask_index, "task": subtask, "success": success, "steps": steps})
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
