#!/usr/bin/env python3
"""Collect successful task-age rollouts for transition-conditioned LoRA.

Unlike the earlier stale-ref collector, this script records conditions from the
actual online 0525 scheduler.  Every saved refresh condition was produced from
the current observation, every history contains actions already sent to CALVIN,
and samples are committed only after the subtask succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
VLA_SCRIPTS = REPO_ROOT / "vla-scripts"
DEFAULT_CALVIN_ROOT = REPO_ROOT.parent / "calvin"
CALVIN_ROOT_PATH = Path(os.environ.get("CALVIN_ROOT", DEFAULT_CALVIN_ROOT)).expanduser().resolve()
os.environ.setdefault("CALVIN_ROOT", CALVIN_ROOT_PATH.as_posix())

for dependency_path in (
    VLA_SCRIPTS,
    REPO_ROOT.parent,
    CALVIN_ROOT_PATH / "calvin_models",
    CALVIN_ROOT_PATH / "calvin_env",
    CALVIN_ROOT_PATH / "calvin_env" / "tacto",
):
    path = Path(dependency_path).resolve().as_posix()
    if Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)

from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: E402
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition  # noqa: E402
from evaluate_calvin_task_age_0525 import (  # noqa: E402
    DEFAULT_GENERALIST_PATH,
    DEFAULT_SPECIALIST_PATH,
    DEFAULT_TASK_AGE_GROUP_A,
    DEFAULT_TASK_AGE_GROUP_B,
    DEFAULT_TASK_AGE_GROUP_C,
    DEFAULT_TASK_AGE_GROUP_D,
    TaskAgeDualSystemEvaluation,
    build_task_age_config,
)


DEFAULT_GROUP_TRAJECTORY_QUOTAS = {"A": 60, "B": 60, "C": 30, "D": 20}
SAMPLE_RATIOS = {"normal": 0.50, "refresh": 0.30, "high_conflict": 0.10, "stale": 0.10}


@dataclass(frozen=True)
class Candidate:
    trajectory_id: str
    split: str
    step: int
    category: str
    condition_id: int
    old_condition_id: int | None
    slow_age: int
    refresh_age: int | None
    conflict_prev_l2_ee6: float | None
    conflict_old_new_l2_ee6: float | None
    action_delta_l2_ee6: float | None
    action_jerk_l2_ee6: float | None
    gripper_intent_change: bool


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_group_quotas(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in parse_csv(value):
        name, count = item.replace("=", ":").split(":", 1)
        name = name.strip().upper()
        if name not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unknown task-age group {name!r}")
        result[name] = int(count)
    if set(result) != {"A", "B", "C", "D"} or any(value < 0 for value in result.values()):
        raise ValueError("--group_trajectory_quotas must define non-negative A, B, C, and D quotas")
    return result


def stable_split(trajectory_id: str) -> str:
    bucket = int(hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def sequence_fingerprint(sequence: tuple) -> str:
    initial_state, tasks = sequence
    return hashlib.sha256(repr((initial_state, tuple(tasks))).encode("utf-8")).hexdigest()


def tensor_cpu(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    result = torch.as_tensor(value).detach().cpu()
    return result.to(dtype=dtype) if dtype is not None else result


def history_before_step(model: TaskAgeDualSystemEvaluation) -> np.ndarray:
    history = np.zeros((4, 7), dtype=np.float32)
    if model.hist_action:
        stacked = torch.stack(list(model.hist_action), dim=0).detach().cpu().numpy().astype(np.float32)
        history[-len(stacked) :] = stacked[-4:]
    return history


def nested(mapping: dict, group: str, key: str, default: Any) -> Any:
    value = mapping.get(group, {})
    return value.get(key, default) if isinstance(value, dict) else default


def capture_frame(obs: dict, action: np.ndarray, hist_action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(7).copy()
    robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32)
    return {
        "rel_actions": action,
        "hist_action_before": np.asarray(hist_action, dtype=np.float32),
        "robot_obs": robot_obs,
        "scene_obs": np.asarray(obs.get("scene_obs", np.zeros(24)), dtype=np.float32),
        "rgb_static": np.asarray(nested(obs, "rgb_obs", "rgb_static", np.zeros((200, 200, 3))), dtype=np.uint8),
        "rgb_gripper": np.asarray(nested(obs, "rgb_obs", "rgb_gripper", np.zeros((84, 84, 3))), dtype=np.uint8),
        "depth_static": np.asarray(nested(obs, "depth_obs", "depth_static", np.zeros((200, 200))), dtype=np.float32),
        "depth_gripper": np.asarray(nested(obs, "depth_obs", "depth_gripper", np.zeros((84, 84))), dtype=np.float32),
    }


def l2_ee6(left: Any, right: Any) -> float:
    left_np = np.asarray(left, dtype=np.float32).reshape(-1)[:6]
    right_np = np.asarray(right, dtype=np.float32).reshape(-1)[:6]
    return float(np.linalg.norm(left_np - right_np))


def load_dual_system(args: argparse.Namespace, device: torch.device):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from train_spacialist_calvin import DualSystem
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    quantization_config = None
    model_dtype = torch.bfloat16
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model_dtype = torch.float16
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_dtype = torch.float16

    kwargs = {
        "torch_dtype": model_dtype,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "trust_remote_code": True,
    }
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map
    if args.attn_implementation != "none":
        kwargs["attn_implementation"] = args.attn_implementation
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    slow_model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **kwargs).eval()
    if quantization_config is None and args.device_map == "none":
        slow_model = slow_model.to(device)

    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    fast_model = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}}, noise_scheduler=scheduler, n_action_steps=8,
        num_inference_steps=args.fast_num_inference_steps, vision_encoder="DINO",
        with_depth=True, with_gripper=True, with_tactile=False, cond_drop_chance=0.0,
        progressive_noise=False,
    ).eval().to(device)
    tokenizer = ActionTokenizer(processor.tokenizer)
    dual_system = DualSystem(slow_model, fast_model, tokenizer)
    dual_system.ema_fast_system.load_state_dict(torch.load(args.specialist_path, map_location="cpu"), strict=False)
    dual_system.eval()
    return dual_system, processor, tokenizer


def build_wrapper(args, dual_system, processor, tokenizer):
    return TaskAgeDualSystemEvaluation(
        dual_system, processor, tokenizer, task_age_config=build_task_age_config(args),
        slow_call_strategy="task_age", slow_trigger_policy="age_empty",
        max_slow_age=args.task_age_default_max_slow_age,
        empty_ref_after_age=args.empty_ref_after_age, profile_steps=False,
        profile_sample_var_k=1, slow_handover_steps=0,
        slow_handover_blend_hidden=False, action_delta_limit_ee6=0.0,
        action_jerk_limit_ee6=0.0,
    )


class DatasetWriter:
    def __init__(self, output_dir: Path, overwrite: bool):
        self.root = output_dir.expanduser().resolve()
        if self.root.exists() and any(self.root.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.root} is not empty; use --overwrite")
            shutil.rmtree(self.root)
        self.trajectories = self.root / "trajectories"
        self.conditions = self.root / "conditions"
        self.trajectories.mkdir(parents=True)
        self.conditions.mkdir(parents=True)

    def save_success(self, trajectory_id: str, frames: list[dict], conditions: list[dict]) -> None:
        trajectory_dir = self.trajectories / trajectory_id
        trajectory_dir.mkdir()
        for step, frame in enumerate(frames):
            np.savez_compressed(trajectory_dir / f"step_{step:04d}.npz", **frame)
        condition_dir = self.conditions / trajectory_id
        condition_dir.mkdir()
        for condition in conditions:
            condition_id = int(condition.pop("condition_id"))
            torch.save(condition, condition_dir / f"condition_{condition_id:03d}.pt")


def rollout_success(
    env, model, task_oracle, task: str, instruction: str, ep_len: int,
    trajectory_id: str, args: argparse.Namespace,
) -> tuple[bool, list[dict], list[dict], list[Candidate]]:
    obs = env.get_obs()
    model.reset()
    model.set_current_task(task)
    start_info = env.get_info()
    split = stable_split(trajectory_id)
    frames: list[dict] = []
    conditions: list[dict] = []
    candidates: list[Candidate] = []
    active_condition_id: int | None = None
    previous_condition_id: int | None = None
    previous_action: np.ndarray | None = None

    for step in range(ep_len):
        hist_action = history_before_step(model)
        old_slow_action = tensor_cpu(model.action, torch.float32)
        old_condition_id = active_condition_id
        action = np.asarray(model.step(obs, instruction, step), dtype=np.float32).reshape(7)
        profile = dict(model.last_step_profile)
        is_refresh = bool(profile.get("slow_system"))
        refresh_age = profile.get("slow_age_before") if is_refresh else None

        if is_refresh:
            previous_condition_id = old_condition_id
            active_condition_id = len(conditions)
            conditions.append({
                "condition_id": active_condition_id,
                "step": int(step),
                "refresh_age": None if refresh_age is None else int(refresh_age),
                "slow_action": tensor_cpu(model.action, torch.float32),
                "slow_hidden": tensor_cpu(model.hidden_states, torch.float16),
                "old_condition_id": old_condition_id,
                "source": "online_current_observation",
            })

        if active_condition_id is None:
            raise RuntimeError("Task-age scheduler did not create an initial slow condition")

        frames.append(capture_frame(obs, action, hist_action))
        slow_age = int(profile["slow_age_after"])
        conflict_prev = None
        conflict_old_new = None
        gripper_change = False
        if is_refresh and step > 0:
            new_action = tensor_cpu(model.action, torch.float32).numpy()[0]
            if previous_action is not None:
                conflict_prev = l2_ee6(new_action[0], previous_action)
                gripper_change = bool(np.sign(new_action[0, 6]) != np.sign(previous_action[6]))
            if old_slow_action is not None and refresh_age is not None:
                old_index = min(max(int(refresh_age), 0), old_slow_action.shape[1] - 1)
                conflict_old_new = l2_ee6(old_slow_action.numpy()[0, old_index], new_action[0])

        category = "refresh" if is_refresh and step > 0 else "normal"
        if category == "refresh" and (
            (conflict_prev is not None and conflict_prev >= args.high_conflict_prev_threshold)
            or (conflict_old_new is not None and conflict_old_new >= args.high_conflict_old_new_threshold)
            or (profile.get("jerk_l2_ee6") is not None and float(profile["jerk_l2_ee6"]) >= args.high_conflict_jerk_threshold)
        ):
            category = "high_conflict"
        elif not is_refresh and slow_age >= args.empty_ref_after_age:
            category = "stale"

        if step >= args.history_steps and step + args.action_chunk_size <= ep_len:
            candidates.append(Candidate(
                trajectory_id=trajectory_id, split=split, step=step, category=category,
                condition_id=active_condition_id,
                old_condition_id=previous_condition_id if is_refresh else None,
                slow_age=slow_age, refresh_age=None if refresh_age is None else int(refresh_age),
                conflict_prev_l2_ee6=conflict_prev,
                conflict_old_new_l2_ee6=conflict_old_new,
                action_delta_l2_ee6=None if previous_action is None else l2_ee6(action, previous_action),
                action_jerk_l2_ee6=None if profile.get("jerk_l2_ee6") is None else float(profile["jerk_l2_ee6"]),
                gripper_intent_change=gripper_change,
            ))

        # CALVIN may transform its input array in place. Keep the normalized
        # committed command immutable for targets, history, and diagnostics.
        obs, _, _, current_info = env.step(action.copy())
        previous_action = action.copy()
        if task_oracle.get_task_info_for_set(start_info, current_info, {task}):
            valid_end = len(frames) - args.action_chunk_size
            candidates = [candidate for candidate in candidates if candidate.step <= valid_end]
            return True, frames, conditions, candidates
    return False, [], [], []


def _select_category_mix(candidates: list[Candidate], target: int, seed: int) -> tuple[list[Candidate], dict]:
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)
    refresh_pool = by_category["refresh"] + by_category["high_conflict"]
    limiting_total = min(
        target,
        int(len(by_category["normal"]) / SAMPLE_RATIOS["normal"]),
        int(len(refresh_pool) / (SAMPLE_RATIOS["refresh"] + SAMPLE_RATIOS["high_conflict"])),
        int(len(by_category["high_conflict"]) / SAMPLE_RATIOS["high_conflict"]),
        int(len(by_category["stale"]) / SAMPLE_RATIOS["stale"]),
    )
    # Keep exact 50/30/10/10 proportions; round down to a multiple of ten.
    selected_total = max(0, limiting_total - limiting_total % 10)
    rng = random.Random(seed)
    requested = {category: int(selected_total * ratio) for category, ratio in SAMPLE_RATIOS.items()}
    high_pool = sorted(by_category["high_conflict"], key=lambda item: (item.trajectory_id, item.step))
    selected_high = rng.sample(high_pool, requested["high_conflict"])
    selected_high_keys = {(item.trajectory_id, item.step) for item in selected_high}
    remaining_refresh = [
        item for item in sorted(refresh_pool, key=lambda item: (item.trajectory_id, item.step))
        if (item.trajectory_id, item.step) not in selected_high_keys
    ]
    selected_refresh = [replace(item, category="refresh") for item in rng.sample(remaining_refresh, requested["refresh"])]
    selected = selected_high + selected_refresh
    for category in ("normal", "stale"):
        pool = sorted(by_category[category], key=lambda item: (item.trajectory_id, item.step))
        selected.extend(rng.sample(pool, requested[category]))
    rng.shuffle(selected)
    stats = {
        "target_total": int(target), "selected_total": len(selected),
        "available_by_category": {
            "normal": len(by_category["normal"]),
            "refresh_total": len(refresh_pool),
            "high_conflict": len(by_category["high_conflict"]),
            "stale": len(by_category["stale"]),
        },
        "selected_by_category": requested,
    }
    return selected, stats


def select_samples(candidates: list[Candidate], target: int, seed: int) -> tuple[list[Candidate], dict]:
    """Apply the category mix separately to train, validation, and test."""

    split_targets = {
        "train": (target * 70 // 100) // 10 * 10,
        "validation": (target * 15 // 100) // 10 * 10,
    }
    split_targets["test"] = target - split_targets["train"] - split_targets["validation"]
    selected: list[Candidate] = []
    by_split = {}
    for offset, split in enumerate(("train", "validation", "test")):
        pool = [candidate for candidate in candidates if candidate.split == split]
        split_selected, stats = _select_category_mix(pool, split_targets[split], seed + offset)
        selected.extend(split_selected)
        by_split[split] = stats
    random.Random(seed).shuffle(selected)
    return selected, {
        "target_total": target,
        "selected_total": len(selected),
        "target_by_split": split_targets,
        "by_split": by_split,
        "selected_by_category": dict(Counter(item.category for item in selected)),
    }


def sample_requirements(target: int) -> dict[tuple[str, str], int]:
    """Return exact per-split, per-category counts for a complete dataset."""

    split_targets = {"train": target * 70 // 100, "validation": target * 15 // 100}
    split_targets["test"] = target - split_targets["train"] - split_targets["validation"]
    requirements = {}
    for split, split_target in split_targets.items():
        requirements[(split, "normal")] = int(split_target * SAMPLE_RATIOS["normal"])
        requirements[(split, "refresh_total")] = int(
            split_target * (SAMPLE_RATIOS["refresh"] + SAMPLE_RATIOS["high_conflict"])
        )
        requirements[(split, "high_conflict")] = int(split_target * SAMPLE_RATIOS["high_conflict"])
        requirements[(split, "stale")] = int(split_target * SAMPLE_RATIOS["stale"])
    return requirements


def candidate_requirement_keys(candidate: Candidate) -> tuple[tuple[str, str], ...]:
    if candidate.category == "high_conflict":
        return ((candidate.split, "refresh_total"), (candidate.split, "high_conflict"))
    if candidate.category == "refresh":
        return ((candidate.split, "refresh_total"),)
    return ((candidate.split, candidate.category),)


def sample_deficits(available: Counter, requirements: dict[tuple[str, str], int]) -> dict[str, int]:
    return {
        f"{split}:{category}": required - available[(split, category)]
        for (split, category), required in requirements.items()
        if available[(split, category)] < required
    }


def collect(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    accelerator = Accelerator()
    group_quotas = parse_group_quotas(args.group_trajectory_quotas)
    writer = DatasetWriter(Path(args.output_dir), args.overwrite)
    dual_system, processor, tokenizer = load_dual_system(args, accelerator.device)
    dual_system = accelerator.prepare(dual_system, device_placement=[True])
    model = build_wrapper(args, dual_system, processor, tokenizer)

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"], "actions": ["rel_actions"], "language": ["language"],
    }
    dataset_path = CALVIN_ROOT_PATH / "dataset" / args.dataset_subdir / args.dataset_split
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"CALVIN split directory does not exist: {dataset_path}")
    from calvin_env_wrapper import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(dataset_path, observation_space, accelerator.device, use_egl=args.use_egl)
    conf_dir = CALVIN_ROOT_PATH / "calvin_models" / "conf"
    task_oracle = hydra.utils.instantiate(OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    task_config = build_task_age_config(args)
    group_map = task_config["task_group_map"]

    group_counts = Counter()
    quota_task_counts = Counter()
    task_counts = Counter()
    failures = Counter()
    all_candidates: list[Candidate] = []
    candidate_counts = Counter()
    requirements = sample_requirements(args.target_samples)
    trajectory_records: list[dict] = []
    benchmark_sequences = list(get_sequences(args.exclude_benchmark_sequences))
    excluded = {sequence_fingerprint(sequence) for sequence in benchmark_sequences}
    generated = list(get_sequences(args.sequence_start + args.num_sequences + args.exclude_benchmark_sequences))
    sequences = [
        sequence for sequence in generated[args.sequence_start :]
        if sequence_fingerprint(sequence) not in excluded
    ][: args.num_sequences]
    progress = tqdm(enumerate(sequences), total=len(sequences), desc="transition-collect")
    for local_sequence_i, (initial_state, tasks) in progress:
        sequence_i = args.sequence_start + local_sequence_i
        groups_complete = all(group_counts[group] >= quota for group, quota in group_quotas.items())
        if groups_complete and not sample_deficits(candidate_counts, requirements):
            break
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        for subtask_i, task in enumerate(tasks):
            group = group_map.get(task)
            if group is None:
                raise ValueError(f"Task {task!r} is absent from the 0525 A/B/C/D grouping")
            instruction = str(annotations[task][0])
            trajectory_id = f"seq{sequence_i:05d}_sub{subtask_i}_{task}"
            success, frames, conditions, candidates = rollout_success(
                env, model, task_oracle, task, instruction, args.ep_len,
                trajectory_id, args,
            )
            if not success:
                failures[task] += 1
                break
            # Cap multi-task groups per task so easy tasks cannot consume an
            # entire group quota. Group D contains only stack_block.
            below_task_cap = group == "D" or quota_task_counts[task] < args.max_trajectories_per_task
            qualifies_for_group = group_counts[group] < group_quotas[group] and below_task_cap
            needs_samples = bool(sample_deficits(candidate_counts, requirements))
            if len(frames) >= args.min_trajectory_steps and (qualifies_for_group or needs_samples):
                writer.save_success(trajectory_id, frames, conditions)
                task_counts[task] += 1
                all_candidates.extend(candidates)
                candidate_counts.update(
                    key for candidate in candidates for key in candidate_requirement_keys(candidate)
                )
                if qualifies_for_group:
                    group_counts[group] += 1
                    quota_task_counts[task] += 1
                trajectory_records.append({
                    "trajectory_id": trajectory_id, "split": stable_split(trajectory_id),
                    "sequence_i": sequence_i, "subtask_i": subtask_i, "task": task,
                    "task_group": group, "task_max_slow_age": task_config["task_age_map"][task],
                    "instruction": instruction, "steps": len(frames),
                    "conditions": len(conditions), "candidate_samples": len(candidates),
                    "counted_toward_group_quota": bool(qualifies_for_group),
                })
                groups_complete = all(group_counts[name] >= quota for name, quota in group_quotas.items())
                if groups_complete and not sample_deficits(candidate_counts, requirements):
                    break
        deficits = sample_deficits(candidate_counts, requirements)
        progress.set_postfix({
            **{group: group_counts[group] for group in "ABCD"},
            "sample_gaps": len(deficits),
            "saved": len(trajectory_records),
        })

    selected, selection_stats = select_samples(all_candidates, args.target_samples, args.seed)
    selected_by_split = Counter(candidate.split for candidate in selected)
    missing_groups = {
        group: group_quotas[group] - group_counts[group]
        for group in group_quotas if group_counts[group] < group_quotas[group]
    }
    category_deficits = sample_deficits(candidate_counts, requirements)
    complete = not missing_groups and not category_deficits and len(selected) == args.target_samples
    with (writer.root / "samples.jsonl").open("w") as file:
        for sample_id, candidate in enumerate(selected):
            payload = asdict(candidate)
            payload["sample_id"] = sample_id
            payload["history_steps"] = args.history_steps
            payload["action_chunk_size"] = args.action_chunk_size
            file.write(json.dumps(payload, sort_keys=True) + "\n")
    with (writer.root / "trajectories.jsonl").open("w") as file:
        for record in trajectory_records:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "format": "robodual_transition_lora_v1", "status": "complete" if complete else "incomplete",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": writer.root.as_posix(), "successful_trajectories": len(trajectory_records),
        "trajectory_group_quotas": group_quotas, "saved_by_group": dict(group_counts),
        "saved_by_task": dict(sorted(task_counts.items())), "failures_by_task": dict(sorted(failures.items())),
        "split_trajectories": dict(Counter(record["split"] for record in trajectory_records)),
        "selection": selection_stats, "sample_ratios": SAMPLE_RATIOS,
        "selected_by_split": dict(selected_by_split),
        "candidate_requirements": {f"{split}:{category}": count for (split, category), count in requirements.items()},
        "candidate_available": {f"{split}:{category}": candidate_counts[(split, category)] for split, category in requirements},
        "missing_groups": missing_groups, "category_deficits": category_deficits,
        "task_age_config": task_config, "args": vars(args),
        "integrity": {
            "conditions": "captured only from online slow calls on the current observation",
            "history": "four actions sent to env before the sample step",
            "targets": "future actions from the same successful online rollout",
            "split": "stable trajectory-level SHA256 split; adjacent windows never cross splits",
        },
    }
    (writer.root / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.allow_incomplete and not complete:
        raise RuntimeError(
            "Collection targets were not met: "
            f"missing_groups={missing_groups}, selected={len(selected)}, "
            f"target={args.target_samples}, category_deficits={category_deficits}. "
            "The summary is marked incomplete; "
            "use --allow_incomplete only for diagnostics."
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist_path", default=DEFAULT_GENERALIST_PATH.as_posix())
    parser.add_argument("--specialist_path", default=DEFAULT_SPECIALIST_PATH.as_posix())
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset")
    parser.add_argument("--dataset_split", default="training", choices=["training", "validation"])
    parser.add_argument("--output_dir", default=(EXPERIMENT_ROOT / "collected_transition_v1").as_posix())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument(
        "--sequence_start", type=int, default=100,
        help="Skip deterministic CALVIN sequences before this index; 100 avoids the standard evaluation set.",
    )
    parser.add_argument("--exclude_benchmark_sequences", type=int, default=100)
    parser.add_argument("--ep_len", type=int, default=360)
    parser.add_argument("--min_trajectory_steps", type=int, default=16)
    parser.add_argument("--target_samples", type=int, default=8000)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--group_trajectory_quotas", default="A:60,B:60,C:30,D:20")
    parser.add_argument(
        "--max_trajectories_per_task", type=int, default=8,
        help="Diversity cap for groups A-C; group D has only stack_block and is exempt.",
    )
    parser.add_argument("--history_steps", type=int, default=4, choices=[4])
    parser.add_argument("--action_chunk_size", type=int, default=8, choices=[8])
    parser.add_argument("--empty_ref_after_age", type=int, default=8)
    parser.add_argument("--high_conflict_prev_threshold", type=float, default=0.18)
    parser.add_argument("--high_conflict_old_new_threshold", type=float, default=0.18)
    parser.add_argument("--high_conflict_jerk_threshold", type=float, default=0.24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="none")
    parser.add_argument("--fast_num_inference_steps", type=int, default=10)
    parser.add_argument("--task_age_default_max_slow_age", type=int, default=12)
    parser.add_argument("--task_age_group_a_max_slow_age", type=int, default=13)
    parser.add_argument("--task_age_group_b_max_slow_age", type=int, default=12)
    parser.add_argument("--task_age_group_c_max_slow_age", type=int, default=10)
    parser.add_argument("--task_age_group_d_max_slow_age", type=int, default=8)
    parser.add_argument("--task_age_group_a_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_A))
    parser.add_argument("--task_age_group_b_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_B))
    parser.add_argument("--task_age_group_c_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_C))
    parser.add_argument("--task_age_group_d_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_D))
    args = parser.parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        parser.error("--load_in_4bit and --load_in_8bit are mutually exclusive")
    if args.target_samples < 200 or args.target_samples % 200:
        parser.error("--target_samples must be a positive multiple of 200 for exact 70/15/15 splits")
    if (
        args.sequence_start < 0 or args.exclude_benchmark_sequences < 0
        or args.num_sequences <= 0 or args.max_trajectories_per_task <= 0
    ):
        parser.error("sequence_start must be non-negative and sequence/task counts must be positive")
    return args


if __name__ == "__main__":
    collect(parse_args())
