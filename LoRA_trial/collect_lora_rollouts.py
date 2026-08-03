#!/usr/bin/env python3
"""Collect successful CALVIN rollouts for specialist LoRA stale-ref training.

This script intentionally lives outside ``vla-scripts`` and does not modify the
existing evaluation entrypoints. It reuses their model wrappers and CALVIN env
setup, then writes successful target-task rollouts as CALVIN-style per-frame
``episode_XXXXXXX.npz`` files plus ``lang_annotations/auto_lang_ann.npy``.
"""

import argparse
import fnmatch
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
VLA_SCRIPTS = REPO_ROOT / "vla-scripts"
DEFAULT_CALVIN_ROOT = REPO_ROOT.parent / "calvin"
CALVIN_ROOT_PATH = Path(os.environ.get("CALVIN_ROOT", DEFAULT_CALVIN_ROOT)).expanduser().resolve()
os.environ.setdefault("CALVIN_ROOT", CALVIN_ROOT_PATH.as_posix())

for dependency_path in (
    VLA_SCRIPTS,
    REPO_ROOT.parent.as_posix(),
    CALVIN_ROOT_PATH / "calvin_models",
    CALVIN_ROOT_PATH / "calvin_env",
    CALVIN_ROOT_PATH / "calvin_env" / "tacto",
):
    path_str = dependency_path.as_posix() if isinstance(dependency_path, Path) else str(dependency_path)
    if Path(path_str).exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from evaluate_calvin_task_age_0525 import (  # noqa: E402
    DEFAULT_GENERALIST_PATH,
    DEFAULT_SPECIALIST_PATH,
    DEFAULT_TASK_AGE_GROUP_A,
    DEFAULT_TASK_AGE_GROUP_B,
    DEFAULT_TASK_AGE_GROUP_C,
    DEFAULT_TASK_AGE_GROUP_D,
    TaskAgeDualSystemEvaluation,
    build_task_age_config,
    make_env,
)


DEFAULT_TARGET_QUOTAS = {
    "place_in_slider": 20,
    "lift_blue_block_slider": 10,
    "stack_block": 10,
    "rotate_red_block_right": 5,
    "push_pink_block_right": 5,
}
DEFAULT_TARGET_TASKS = tuple(DEFAULT_TARGET_QUOTAS)
DEFAULT_TARGET_QUOTA_ARG = ",".join(f"{task}:{quota}" for task, quota in DEFAULT_TARGET_QUOTAS.items())


@dataclass
class SavedRollout:
    task: str
    instruction: str
    sequence_i: int
    subtask_i: int
    steps: int
    start_idx: int
    end_idx: int


def parse_csv(value):
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_task_quotas(value):
    quotas = {}
    for item in parse_csv(value):
        if ":" in item:
            task, count = item.split(":", 1)
        elif "=" in item:
            task, count = item.split("=", 1)
        else:
            raise ValueError(f"Quota item must look like task:count, got {item!r}")
        task = task.strip()
        if not task:
            raise ValueError(f"Empty task name in quota item {item!r}")
        quotas[task] = int(count)
    return quotas


def discover_tasks(num_sequences):
    tasks = set()
    for _, eval_sequence in get_sequences(num_sequences):
        tasks.update(eval_sequence)
    return sorted(tasks)


def expand_target_tasks(patterns, known_tasks):
    expanded = []
    for pattern in patterns:
        matches = [task for task in known_tasks if fnmatch.fnmatch(task, pattern)]
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pattern)
    return sorted(set(expanded))


def build_target_quotas(args, known_tasks):
    explicit_quotas = parse_task_quotas(args.target_task_quotas)
    if explicit_quotas:
        quotas = {}
        for pattern, quota in explicit_quotas.items():
            matches = [task for task in known_tasks if fnmatch.fnmatch(task, pattern)]
            if not matches:
                matches = [pattern]
            for task in matches:
                quotas[task] = int(quota)
    else:
        target_patterns = parse_csv(args.target_tasks)
        target_tasks = expand_target_tasks(target_patterns, known_tasks)
        quotas = {task: int(args.target_per_task) for task in target_tasks}

    quotas = {task: quota for task, quota in quotas.items() if quota > 0}
    return dict(sorted(quotas.items()))


def active_target_tasks(counts, target_quotas):
    return {task for task, quota in target_quotas.items() if counts[task] < quota}


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_nested(mapping, group, name, default):
    group_value = mapping.get(group, {})
    if isinstance(group_value, dict) and name in group_value:
        return group_value[name]
    return default


def approximate_absolute_action(rel_action, robot_obs):
    """Best-effort CALVIN ``actions`` field from a relative action.

    The specialist and current training path consume ``rel_actions``. The raw
    CALVIN format also contains absolute ``actions``; this approximation keeps
    the file schema complete without being used as a LoRA target.
    """

    rel_action = np.asarray(rel_action, dtype=np.float64).copy()
    robot_obs = np.asarray(robot_obs, dtype=np.float64)
    abs_action = np.zeros(7, dtype=np.float64)
    abs_action[:3] = robot_obs[:3] + np.clip(rel_action[:3], -1.0, 1.0) * 0.02
    abs_action[3:6] = robot_obs[3:6] + np.clip(rel_action[3:6], -1.0, 1.0) * 0.05
    abs_action[3:6] = (abs_action[3:6] + np.pi) % (2 * np.pi) - np.pi
    abs_action[6] = -1.0 if rel_action[6] < 0 else 1.0
    return abs_action


def capture_frame(obs, action):
    rel_action = np.asarray(to_numpy(action), dtype=np.float64).reshape(-1)
    if rel_action.shape[0] != 7:
        raise ValueError(f"Expected 7D relative action, got shape {rel_action.shape}")

    robot_obs = np.asarray(to_numpy(obs["robot_obs"]), dtype=np.float64)
    scene_obs = np.asarray(to_numpy(obs.get("scene_obs", np.zeros(24))), dtype=np.float64)

    rgb_static = np.asarray(to_numpy(get_nested(obs, "rgb_obs", "rgb_static", np.zeros((200, 200, 3), dtype=np.uint8))))
    rgb_gripper = np.asarray(to_numpy(get_nested(obs, "rgb_obs", "rgb_gripper", np.zeros((84, 84, 3), dtype=np.uint8))))
    rgb_tactile = np.asarray(to_numpy(get_nested(obs, "rgb_obs", "rgb_tactile", np.zeros((160, 120, 6), dtype=np.uint8))))
    depth_static = np.asarray(to_numpy(get_nested(obs, "depth_obs", "depth_static", np.zeros((200, 200), dtype=np.float32))), dtype=np.float32)
    depth_gripper = np.asarray(to_numpy(get_nested(obs, "depth_obs", "depth_gripper", np.zeros((84, 84), dtype=np.float32))), dtype=np.float32)
    depth_tactile = np.asarray(to_numpy(get_nested(obs, "depth_obs", "depth_tactile", np.zeros((160, 120, 2), dtype=np.float32))), dtype=np.float32)

    return {
        "actions": approximate_absolute_action(rel_action, robot_obs),
        "rel_actions": rel_action,
        "robot_obs": robot_obs,
        "scene_obs": scene_obs,
        "rgb_static": rgb_static.astype(np.uint8, copy=False),
        "rgb_gripper": rgb_gripper.astype(np.uint8, copy=False),
        "rgb_tactile": rgb_tactile.astype(np.uint8, copy=False),
        "depth_static": depth_static,
        "depth_gripper": depth_gripper,
        "depth_tactile": depth_tactile,
    }


class CalvinRolloutWriter:
    def __init__(self, output_dir, overwrite=False):
        self.output_dir = Path(output_dir)
        self.training_dir = self.output_dir / "training"
        self.lang_dir = self.training_dir / "lang_annotations"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "collection_summary.json"
        self.next_episode_idx = 0
        self.saved = []

        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"{self.output_dir} is not empty. Pass --overwrite or choose a new --output_dir."
                )
            shutil.rmtree(self.output_dir)

        self.lang_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text("")

    def add_rollout(self, task, instruction, sequence_i, subtask_i, frames):
        start_idx = self.next_episode_idx
        for frame in frames:
            out_path = self.training_dir / f"episode_{self.next_episode_idx:07d}.npz"
            np.savez_compressed(out_path, **frame)
            self.next_episode_idx += 1

        end_idx = self.next_episode_idx - 1
        record = SavedRollout(
            task=task,
            instruction=instruction,
            sequence_i=int(sequence_i),
            subtask_i=int(subtask_i),
            steps=len(frames),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
        )
        self.saved.append(record)
        with self.manifest_path.open("a") as file:
            file.write(json.dumps(record.__dict__, sort_keys=True) + "\n")
        self.write_annotations()
        return record

    def write_annotations(self):
        lang_data = {
            "info": {"indx": [(record.start_idx, record.end_idx) for record in self.saved]},
            "language": {
                "ann": [record.instruction for record in self.saved],
                "task": [record.task for record in self.saved],
            },
        }
        np.save(self.lang_dir / "auto_lang_ann.npy", lang_data, allow_pickle=True)
        np.save(self.training_dir / "ep_start_end_ids.npy", np.array(lang_data["info"]["indx"], dtype=np.int64))

    def write_summary(self, args, attempts, failed, skipped_short, target_quotas, skipped_sequences, stopped_subtasks):
        by_task = Counter(record.task for record in self.saved)
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": self.output_dir.as_posix(),
            "num_saved_rollouts": len(self.saved),
            "num_saved_frames": int(self.next_episode_idx),
            "saved_by_task": dict(sorted(by_task.items())),
            "target_quotas": dict(sorted(target_quotas.items())),
            "attempts_by_task": dict(sorted(attempts.items())),
            "failed_by_task": dict(sorted(failed.items())),
            "skipped_short_by_task": dict(sorted(skipped_short.items())),
            "skipped_sequences_without_active_target": int(skipped_sequences),
            "stopped_subtasks_after_last_active_target": int(stopped_subtasks),
            "args": vars(args),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        return summary


def load_dual_system(args, device):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from train_spacialist_calvin import DualSystem
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

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

    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    slow_model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **model_kwargs)
    slow_model.eval()
    if quantization_config is None and args.device_map == "none":
        slow_model = slow_model.to(device)

    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    diffusion_policy = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}},
        noise_scheduler=scheduler,
        n_action_steps=8,
        num_inference_steps=args.fast_num_inference_steps,
        vision_encoder="DINO",
        with_depth=args.with_depth,
        progressive_noise=False,
        with_gripper=args.with_gripper,
        with_tactile=args.with_tactile,
        cond_drop_chance=0.1 if args.with_cfg else 0.0,
    ).eval().to(device)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dual_system = DualSystem(slow_model, diffusion_policy, action_tokenizer)
    specialist_state = torch.load(args.specialist_path, map_location="cpu")
    dual_system.ema_fast_system.load_state_dict(specialist_state, strict=False)
    dual_system.eval()
    return dual_system, processor, action_tokenizer


def build_eval_wrapper(args, dual_system, processor, action_tokenizer):
    if args.slow_call_strategy == "fixed_mod8":
        slow_trigger_policy = "fixed_mod8"
    else:
        slow_trigger_policy = "age_empty"

    task_age_config = build_task_age_config(args)
    return TaskAgeDualSystemEvaluation(
        dual_system,
        processor,
        action_tokenizer,
        profile_steps=False,
        profile_sample_var_k=1,
        profile_sample_var_interval=8,
        profile_sample_var_ages="",
        slow_trigger_policy=slow_trigger_policy,
        max_slow_age=args.max_slow_age,
        empty_ref_after_age=args.empty_ref_after_age,
        slow_handover_steps=0,
        slow_handover_blend_hidden=False,
        action_delta_limit_ee6=0.0,
        action_jerk_limit_ee6=0.0,
        task_age_config=task_age_config,
        slow_call_strategy=args.slow_call_strategy,
        min_slow_age=args.min_slow_age,
        risk_start_age=args.risk_start_age,
    )


def rollout_one_subtask(env, model, task_oracle, task, instruction, ep_len):
    obs = env.get_obs()
    model.reset()
    if hasattr(model, "set_current_task"):
        model.set_current_task(task)
    start_info = env.get_info()
    frames = []

    for step in range(ep_len):
        action = model.step(obs, instruction, step)
        frames.append(capture_frame(obs, action))
        obs, _, _, current_info = env.step(action)
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
        if len(current_task_info) > 0:
            return True, frames
    return False, frames


def collect(args):
    seed_everything(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    known_tasks = discover_tasks(args.num_sequences)
    target_quotas = build_target_quotas(args, known_tasks)
    target_tasks = sorted(target_quotas)
    if not target_quotas:
        raise ValueError("No target tasks selected.")

    writer = CalvinRolloutWriter(args.output_dir, overwrite=args.overwrite)
    print(f"[collector] target_quotas={target_quotas}")
    print(f"[collector] output_dir={Path(args.output_dir).resolve()}")

    dual_system, processor, action_tokenizer = load_dual_system(args, device)
    dual_system = accelerator.prepare(dual_system, device_placement=[True])
    model = build_eval_wrapper(args, dual_system, processor, action_tokenizer)

    observation_space = {
        # Match the existing evaluation scripts: tactile camera instantiation is
        # unavailable in the current CALVIN environment. capture_frame() still
        # writes zero tactile placeholders for CALVIN-format compatibility.
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    dataset_path = CALVIN_ROOT_PATH / "dataset" / args.dataset_subdir
    env = make_env(dataset_path.as_posix(), observation_space, device, args.use_egl)

    conf_dir = CALVIN_ROOT_PATH / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    counts = Counter()
    attempts = Counter()
    skipped_short = Counter()
    failed = Counter()
    skipped_sequences = 0
    stopped_subtasks = 0

    eval_sequences = list(get_sequences(args.num_sequences))
    progress = tqdm(enumerate(eval_sequences), total=len(eval_sequences), desc="collect")
    for sequence_i, (initial_state, eval_sequence) in progress:
        active_targets = active_target_tasks(counts, target_quotas)
        if not active_targets:
            break
        target_positions = [idx for idx, task in enumerate(eval_sequence) if task in active_targets]
        if not target_positions:
            skipped_sequences += 1
            progress.set_postfix({task: counts[task] for task in target_tasks})
            continue
        last_active_target_i = max(target_positions)

        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        for subtask_i, subtask in enumerate(eval_sequence):
            instruction = val_annotations[subtask][0]
            should_save = subtask in active_targets and counts[subtask] < target_quotas[subtask]
            if subtask in target_quotas:
                attempts[subtask] += 1

            success, frames = rollout_one_subtask(env, model, task_oracle, subtask, instruction, args.ep_len)
            if not success:
                if subtask in target_quotas:
                    failed[subtask] += 1
                break

            if should_save:
                if len(frames) < args.min_steps:
                    skipped_short[subtask] += 1
                else:
                    record = writer.add_rollout(subtask, instruction, sequence_i, subtask_i, frames)
                    counts[subtask] += 1
                    print(
                        f"[collector] saved task={subtask} count={counts[subtask]}/{target_quotas[subtask]} "
                        f"steps={record.steps} frames=[{record.start_idx},{record.end_idx}]"
                    )
            if subtask_i >= last_active_target_i:
                stopped_subtasks += max(0, len(eval_sequence) - subtask_i - 1)
                break

        progress.set_postfix({task: counts[task] for task in target_tasks})

    summary = writer.write_summary(
        args,
        attempts,
        failed,
        skipped_short,
        target_quotas,
        skipped_sequences,
        stopped_subtasks,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    missing = {task: target_quotas[task] - counts[task] for task in target_tasks if counts[task] < target_quotas[task]}
    if missing:
        print(f"[collector] incomplete targets: {missing}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist_path", default=DEFAULT_GENERALIST_PATH.as_posix(), type=str)
    parser.add_argument("--specialist_path", default=DEFAULT_SPECIALIST_PATH.as_posix(), type=str)
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset", type=str)
    parser.add_argument(
        "--output_dir",
        default=(REPO_ROOT / "LoRA_trial" / "collected_lora_rollouts").as_posix(),
        type=str,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target_tasks", default=",".join(DEFAULT_TARGET_TASKS), type=str)
    parser.add_argument(
        "--target_task_quotas",
        default=DEFAULT_TARGET_QUOTA_ARG,
        type=str,
        help=(
            "Comma-separated per-task quotas, e.g. place_in_slider:20,stack_block:10. "
            "When non-empty, this is the authoritative target set."
        ),
    )
    parser.add_argument("--target_per_task", default=20, type=int)
    parser.add_argument("--num_sequences", default=100, type=int)
    parser.add_argument("--ep_len", default=360, type=int)
    parser.add_argument(
        "--min_steps",
        default=20,
        type=int,
        help="Skip successful rollouts shorter than this; stale age 11 + 8-step target needs about 20 frames.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_egl", action="store_true")

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none", type=str)
    parser.add_argument("--attn_implementation", default="none", type=str)
    parser.add_argument("--fast_num_inference_steps", default=10, type=int)
    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--with_cfg", default=False, action="store_true")

    parser.add_argument(
        "--slow_call_strategy",
        default="task_age",
        choices=["fixed_mod8", "age_empty", "task_age"],
        type=str,
    )
    parser.add_argument("--max_slow_age", default=12, type=int)
    parser.add_argument("--empty_ref_after_age", default=8, type=int)
    parser.add_argument("--min_slow_age", default=7, type=int)
    parser.add_argument("--risk_start_age", default=8, type=int)

    parser.add_argument("--task_age_default_max_slow_age", default=12, type=int)
    parser.add_argument("--task_age_group_a_max_slow_age", default=13, type=int)
    parser.add_argument("--task_age_group_b_max_slow_age", default=12, type=int)
    parser.add_argument("--task_age_group_c_max_slow_age", default=10, type=int)
    parser.add_argument("--task_age_group_d_max_slow_age", default=8, type=int)
    parser.add_argument("--task_age_group_a_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_A), type=str)
    parser.add_argument("--task_age_group_b_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_B), type=str)
    parser.add_argument("--task_age_group_c_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_C), type=str)
    parser.add_argument("--task_age_group_d_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_D), type=str)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
