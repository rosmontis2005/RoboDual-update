#!/usr/bin/env python3
"""Compare CALVIN expert and frozen-condition M1 closed-loop trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
VLA_SCRIPTS = REPO_ROOT / "vla-scripts"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET = REPO_ROOT.parent / "calvin" / "dataset" / "task_D_D"
DEFAULT_GENERALIST = REPO_ROOT.parent / "models" / "generalist"
DEFAULT_SPECIALIST = (
    REPO_ROOT / "DiT_train" / "runs" / "ageext_m1_long1500_b97f005"
    / "specialist_ema_step_001500.pt"
)

SCHEMA_VERSION = "expert_specialist_trajectory_v1"
ACTION_STEPS = 12
STATE_STEPS = 13
ACTION_DIM = 7
ACTION_HORIZON = 8
EXPECTED_AGES = list(range(ACTION_STEPS))
EXPECTED_COUNTS = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0]
ALLOWED_EMA_COMPATIBILITY_KEYS = {
    "online_model._dummy_variable",
    "ema_model._dummy_variable",
}
STATE_METRICS = (
    "robot_full_l2", "robot_max_abs", "robot_ee6_l2", "robot_ee6_max_abs",
    "scene_l2", "scene_max_abs", "rgb_static_mae", "rgb_gripper_mae",
    "depth_static_mae", "depth_static_rmse",
    "depth_gripper_mae", "depth_gripper_rmse",
)
ACTION_METRICS = (
    "closed_loop_action_ee6_l2_to_expert",
    "teacher_forced_action_ee6_l2_to_expert",
    "closed_loop_vs_teacher_forced_ee6_l2",
    "closed_loop_gripper_sign_agreement_to_expert",
    "teacher_forced_gripper_sign_agreement_to_expert",
    "aggregation_delta_ee6",
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def classify_checkpoint_keys(
    missing_keys: Iterable[str], unexpected_keys: Iterable[str],
) -> dict[str, list[str]]:
    raw_missing = list(missing_keys)
    raw_unexpected = list(unexpected_keys)
    ignored = sorted(
        key for key in raw_unexpected if key in ALLOWED_EMA_COMPATIBILITY_KEYS
    )
    return {
        "missing_keys": raw_missing,
        "unexpected_keys": [
            key for key in raw_unexpected if key not in ALLOWED_EMA_COMPATIBILITY_KEYS
        ],
        "raw_missing_keys": raw_missing,
        "raw_unexpected_keys": raw_unexpected,
        "ignored_ema_compatibility_keys": ignored,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stable_split(trajectory_id: str) -> str:
    """Byte-equivalent split rule from collect_age_extended_expert.py."""
    bucket = int(hashlib.sha256(trajectory_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def selected_proprio(robot_obs: Any) -> np.ndarray:
    robot = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    if robot.size < 7:
        raise ValueError(f"robot_obs has fewer than seven elements: {robot.shape}")
    return np.concatenate((robot[:6], robot[-1:])).astype(np.float32)


def as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).copy()


def nested(obs: Mapping[str, Any], group: str, key: str) -> np.ndarray:
    if group not in obs or key not in obs[group]:
        raise KeyError(f"Observation lacks {group}.{key}")
    return np.asarray(obs[group][key]).copy()


def capture_env_state(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = ("robot_obs", "scene_obs")
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"CALVIN observation lacks {missing}; scene state is never zero-filled")
    return {
        "robot_obs": as_float_array(obs["robot_obs"]),
        "selected_proprio": selected_proprio(obs["robot_obs"]),
        "scene_obs": as_float_array(obs["scene_obs"]),
        "rgb_static": nested(obs, "rgb_obs", "rgb_static"),
        "rgb_gripper": nested(obs, "rgb_obs", "rgb_gripper"),
        "depth_static": nested(obs, "depth_obs", "depth_static").astype(np.float32),
        "depth_gripper": nested(obs, "depth_obs", "depth_gripper").astype(np.float32),
    }


def dataset_observation(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the exact nested observation schema consumed by evaluator.step."""
    return {
        "rgb_obs": {
            "rgb_static": np.asarray(frame["rgb_static"]).copy(),
            "rgb_gripper": np.asarray(frame["rgb_gripper"]).copy(),
        },
        "depth_obs": {
            "depth_static": np.asarray(frame["depth_static"], dtype=np.float32).copy(),
            "depth_gripper": np.asarray(frame["depth_gripper"], dtype=np.float32).copy(),
        },
        "robot_obs": as_float_array(frame["robot_obs"]),
        "scene_obs": as_float_array(frame["scene_obs"]),
    }


def dataset_state(frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "robot_obs": as_float_array(frame["robot_obs"]),
        "selected_proprio": selected_proprio(frame["robot_obs"]),
        "scene_obs": as_float_array(frame["scene_obs"]),
        "rgb_static": np.asarray(frame["rgb_static"]).copy(),
        "rgb_gripper": np.asarray(frame["rgb_gripper"]).copy(),
        "depth_static": np.asarray(frame["depth_static"], dtype=np.float32).copy(),
        "depth_gripper": np.asarray(frame["depth_gripper"], dtype=np.float32).copy(),
    }


class CalvinLanguageIndex:
    """Language-episode index aligned with the Age-Extended Expert collector."""

    REQUIRED_KEYS = {
        "rgb_static", "rgb_gripper", "depth_static", "depth_gripper",
        "robot_obs", "scene_obs", "rel_actions",
    }

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.expanduser().resolve()
        self.training_dir = self.dataset_root / "training"
        self.validation_dir = self.dataset_root / "validation"
        self.annotation_path = self.training_dir / "lang_annotations" / "auto_lang_ann.npy"
        if not self.training_dir.is_dir():
            raise FileNotFoundError(f"Missing CALVIN training directory: {self.training_dir}")
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"Missing language annotation: {self.annotation_path}")
        annotation = np.load(self.annotation_path, allow_pickle=True).item()
        self.bounds = [(int(start), int(end)) for start, end in annotation["info"]["indx"]]
        self.instructions = [str(item) for item in annotation["language"]["ann"]]
        self.tasks = [str(item) for item in annotation["language"]["task"]]
        if not (len(self.bounds) == len(self.instructions) == len(self.tasks)):
            raise AssertionError("CALVIN language annotation arrays have inconsistent lengths")
        example = next(self.training_dir.glob("episode_*.npz"), None)
        if example is None:
            raise FileNotFoundError(f"No episode_*.npz in {self.training_dir}")
        match = re.match(r"^(.*?)(\d+)(\.npz)$", example.name)
        if match is None:
            raise ValueError(f"Cannot infer episode filename schema from {example.name}")
        self.prefix, digits, self.suffix = match.groups()
        self.digits = len(digits)

    def frame_path(self, frame: int) -> Path:
        return self.training_dir / f"{self.prefix}{frame:0{self.digits}d}{self.suffix}"

    def load_frame(self, frame: int) -> dict[str, np.ndarray]:
        path = self.frame_path(frame)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            missing = self.REQUIRED_KEYS.difference(archive.files)
            if missing:
                raise KeyError(f"{path} lacks {sorted(missing)}")
            return {key: np.asarray(archive[key]).copy() for key in self.REQUIRED_KEYS}

    def episode_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for episode_i, ((start, end), instruction, task) in enumerate(
            zip(self.bounds, self.instructions, self.tasks)
        ):
            trajectory_id = f"calvin_training_lang_{episode_i:06d}_{start}_{end}_{task}"
            rows.append({
                "trajectory_id": trajectory_id,
                "language_episode_index": episode_i,
                "task": task,
                "instruction": instruction,
                "task_start_frame": start,
                "task_end_frame_inclusive": end,
                "anchor_frame": start,
                "split": stable_split(trajectory_id),
            })
        return rows


def select_anchors(
    index: CalvinLanguageIndex, split: str, count: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select only language-subtask starts, round-robin balanced by task."""
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    checked_files = 0
    for row in index.episode_rows():
        if row["split"] != split:
            rejected["stable_split_mismatch"] += 1
            continue
        start, end = row["task_start_frame"], row["task_end_frame_inclusive"]
        if start + 18 > end:
            rejected["cannot_supply_anchor_through_anchor_plus_18"] += 1
            continue
        source_frames = list(range(start, start + 19))
        missing = []
        for frame in source_frames:
            checked_files += 1
            if not index.frame_path(frame).is_file():
                missing.append(frame)
        if missing:
            rejected["missing_source_frame"] += 1
            continue
        candidate = dict(row)
        candidate["source_frame_indices"] = source_frames
        candidate["condition_id"] = ""  # assigned after balanced selection
        pools[candidate["task"]].append(candidate)

    task_order = sorted(
        pools, key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).hexdigest()
    )
    for task in task_order:
        rng.shuffle(pools[task])
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progress = False
        for task in task_order:
            if pools[task] and len(selected) < count:
                selected.append(pools[task].pop())
                progress = True
        if not progress:
            break
    for anchor_i, row in enumerate(selected):
        row["condition_id"] = f"condition_{anchor_i:06d}"
        row["trajectory_artifact"] = f"trajectories/{row['condition_id']}.npz"
    report = {
        "requested_anchors": count,
        "selected_anchors": len(selected),
        "split": split,
        "seed": seed,
        "anchor_contract": "language_subtask_start_only",
        "eligibility_contract": "persisted frames anchor..anchor+18 inclusive",
        "task_counts": dict(sorted(Counter(row["task"] for row in selected).items())),
        "stable_split_counts": dict(sorted(Counter(row["split"] for row in selected).items())),
        "language_episodes_total": len(index.bounds),
        "rejected": dict(rejected),
        "required_source_files_checked": checked_files,
        "shortfall": max(0, count - len(selected)),
    }
    if not selected:
        raise RuntimeError("No eligible task-start anchors were selected")
    return selected, report


def prepare_run_dir(run_name: str) -> Path:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run_name must be one non-empty path component")
    run_dir = (RUNS_ROOT / run_name).resolve()
    if run_dir.parent != RUNS_ROOT.resolve():
        raise ValueError(f"Run path escapes experiment runs root: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def observation_space() -> dict[str, list[str]]:
    return {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }


def configure_calvin_imports(dataset_root: Path) -> Path:
    """Apply evaluate_calvin_0428.py's CALVIN import-path bootstrap."""
    calvin_root = dataset_root.expanduser().resolve().parents[1]
    os.environ.setdefault("CALVIN_ROOT", str(calvin_root))
    for dependency in (
        calvin_root / "calvin_models", calvin_root / "calvin_env",
        calvin_root / "calvin_env" / "tacto",
    ):
        if dependency.exists() and str(dependency) not in sys.path:
            sys.path.insert(0, str(dependency))
    if str(VLA_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(VLA_SCRIPTS))
    return calvin_root


def make_env(index: CalvinLanguageIndex, use_egl: bool):
    import torch

    configure_calvin_imports(index.dataset_root)
    try:
        from calvin_env_wrapper import CalvinEnvWrapperRaw
    except Exception as exc:
        raise RuntimeError(
            "CALVIN environment import failed. Activate dualsys_env and verify calvin_env dependencies."
        ) from exc
    if not index.validation_dir.is_dir():
        raise FileNotFoundError(
            f"CALVIN simulator assets expect the validation directory: {index.validation_dir}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CalvinEnvWrapperRaw(
        index.validation_dir, observation_space(), device, use_egl=use_egl,
    )


def load_models(args: argparse.Namespace):
    """Mirror evaluate_calvin_0428.py's generalist, specialist, DualSystem loader."""
    import torch
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    if str(VLA_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(VLA_SCRIPTS))
    from train_spacialist_calvin import DualSystem

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.load_in_4bit and device.type != "cuda":
        raise RuntimeError("4-bit NF4 generalist loading requires a CUDA-capable runtime")
    quantization = None
    model_dtype = torch.bfloat16
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model_dtype = torch.float16
    model_kwargs: dict[str, Any] = {
        "torch_dtype": model_dtype,
        "quantization_config": quantization,
        "trust_remote_code": True,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    generalist = AutoModelForVision2Seq.from_pretrained(
        args.generalist_path, **model_kwargs,
    ).eval()
    if quantization is None and args.device_map == "none":
        generalist = generalist.to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon"
    )
    specialist = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}}, noise_scheduler=scheduler,
        n_action_steps=8, num_inference_steps=args.fast_num_inference_steps,
        vision_encoder="DINO", with_depth=True, progressive_noise=False,
        with_gripper=True, with_tactile=False, cond_drop_chance=0.0,
    ).eval().to(device)
    tokenizer = ActionTokenizer(processor.tokenizer)
    dual_system = DualSystem(generalist, specialist, tokenizer)
    checkpoint = torch.load(args.specialist_path, map_location="cpu", weights_only=False)
    incompatible = dual_system.ema_fast_system.load_state_dict(checkpoint, strict=False)
    key_audit = classify_checkpoint_keys(
        incompatible.missing_keys, incompatible.unexpected_keys
    )
    checkpoint_audit = {
        "path": str(args.specialist_path),
        "sha256": sha256_file(args.specialist_path),
        **key_audit,
        "allowed_ema_compatibility_keys": sorted(ALLOWED_EMA_COMPATIBILITY_KEYS),
        "compatibility_key_rationale": (
            "EMA wrapper placeholder keys are ignored by the current strict=False evaluator loader; "
            "train_age_extended_expert.py also excludes ema_model._dummy_variable."
        ),
        "checkpoint_type": type(checkpoint).__name__,
        "specialist_architecture_source": "vla-scripts/evaluate_calvin_0428.py",
    }
    from DiT_train.data_collection.collect_age_extended_expert import (
        artifact_fingerprint, processor_fingerprint,
    )
    checkpoint_audit["generalist_checkpoint_fingerprint"] = artifact_fingerprint(
        args.generalist_path, hash_model_files=False
    )
    checkpoint_audit["processor_tokenizer_fingerprint"] = processor_fingerprint(
        args.generalist_path
    )
    dual_system.eval()
    return dual_system, processor, tokenizer, model_dtype, device, checkpoint_audit


def frozen_wrapper_class():
    if str(VLA_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(VLA_SCRIPTS))
    from dual_sys_evaluation_0424test import DualSystemCalvinEvaluation

    class FrozenAnchorConditionEvaluation(DualSystemCalvinEvaluation):
        """Base deployment evaluator with an injected, never-refreshed slow condition."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.frozen_condition_id: str | None = None
            self.forbidden_slow_call_count = 0
            self._condition_injected = False

        def inject_frozen_condition(self, condition: Mapping[str, Any]) -> None:
            import torch

            self.reset()
            runtime_device = self._runtime_device()
            action = torch.as_tensor(condition["slow_action"]).detach().clone()
            hidden = torch.as_tensor(condition["slow_hidden"]).detach().clone()
            if tuple(action.shape) != (1, ACTION_HORIZON, ACTION_DIM):
                raise ValueError(f"slow_action must be [1,8,7], got {tuple(action.shape)}")
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError(f"slow_hidden must be [1,T,D], got {tuple(hidden.shape)}")
            self.action = action.to(runtime_device, dtype=torch.float32)
            self.hidden_states = hidden.to(runtime_device)
            self.last_slow_step = 0
            self.frozen_condition_id = str(condition["condition_id"])
            self._condition_injected = True

        def _should_call_slow_system(self, step):
            if not self._condition_injected:
                raise RuntimeError("Frozen slow condition must be injected before step()")
            return False, "frozen_anchor_condition"

    return FrozenAnchorConditionEvaluation


def build_wrapper(wrapper_type, dual_system, processor, tokenizer):
    return wrapper_type(
        dual_system, processor, tokenizer,
        profile_steps=False, profile_sample_var_k=1,
        slow_trigger_policy="age_empty", max_slow_age=12,
        empty_ref_after_age=8, slow_handover_steps=0,
        slow_handover_blend_hidden=False,
        action_delta_limit_ee6=0.0, action_jerk_limit_ee6=0.0,
    )


def tensor_cpu(value: Any, dtype=None):
    import torch

    result = torch.as_tensor(value).detach().cpu()
    return result.to(dtype=dtype) if dtype is not None else result


def infer_frozen_condition(
    dual_system, processor, model_dtype, anchor: Mapping[str, Any],
    anchor_rgb: np.ndarray,
) -> dict[str, Any]:
    """Reuse the collector's single-call prompt, inference, and normalization contract."""
    import torch

    from DiT_train.data_collection.collect_age_extended_expert import infer_condition

    slow_action, slow_hidden, call_id, normalization = infer_condition(
        dual_system.slow_system, processor, model_dtype,
        anchor_rgb, str(anchor["instruction"]),
    )
    return {
        "condition_id": anchor["condition_id"],
        "trajectory_id": anchor["trajectory_id"],
        "task": anchor["task"],
        "instruction": anchor["instruction"],
        "anchor_frame": anchor["anchor_frame"],
        "slow_action": tensor_cpu(slow_action, torch.float32),
        "slow_hidden": tensor_cpu(slow_hidden, torch.float16),
        "slow_action_normalization": normalization,
        "inference_call_id": call_id,
        "slow_action_inference_call_id": call_id,
        "slow_hidden_inference_call_id": call_id,
        "same_inference_call": True,
        "do_sample": False,
        "source": "persisted_expert_anchor_runtime_predict_action",
        "generalist_path": str(Path(dual_system.slow_system.name_or_path).expanduser().resolve()),
        "generalist_inference_source": (
            "DiT_train/data_collection/collect_age_extended_expert.py:infer_condition"
        ),
        "generalist_calls_for_anchor": 1,
    }


def state_metrics(actual: Mapping[str, Any], expert: Mapping[str, Any]) -> dict[str, float]:
    robot_delta = as_float_array(actual["robot_obs"]) - as_float_array(expert["robot_obs"])
    scene_delta = as_float_array(actual["scene_obs"]) - as_float_array(expert["scene_obs"])
    static_delta = (
        np.asarray(actual["rgb_static"], dtype=np.float32)
        - np.asarray(expert["rgb_static"], dtype=np.float32)
    )
    gripper_delta = (
        np.asarray(actual["rgb_gripper"], dtype=np.float32)
        - np.asarray(expert["rgb_gripper"], dtype=np.float32)
    )
    depth_static_delta = (
        np.asarray(actual["depth_static"], dtype=np.float32)
        - np.asarray(expert["depth_static"], dtype=np.float32)
    )
    depth_gripper_delta = (
        np.asarray(actual["depth_gripper"], dtype=np.float32)
        - np.asarray(expert["depth_gripper"], dtype=np.float32)
    )
    return {
        "robot_full_l2": float(np.linalg.norm(robot_delta)),
        "robot_max_abs": float(np.max(np.abs(robot_delta))),
        "robot_ee6_l2": float(np.linalg.norm(robot_delta[:6])),
        "robot_ee6_max_abs": float(np.max(np.abs(robot_delta[:6]))),
        "scene_l2": float(np.linalg.norm(scene_delta)),
        "scene_max_abs": float(np.max(np.abs(scene_delta))),
        "rgb_static_mae": float(np.mean(np.abs(static_delta))),
        "rgb_gripper_mae": float(np.mean(np.abs(gripper_delta))),
        "depth_static_mae": float(np.mean(np.abs(depth_static_delta))),
        "depth_static_rmse": float(np.sqrt(np.mean(np.square(depth_static_delta)))),
        "depth_gripper_mae": float(np.mean(np.abs(depth_gripper_delta))),
        "depth_gripper_rmse": float(np.sqrt(np.mean(np.square(depth_gripper_delta)))),
    }


def state_json(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "robot_obs": as_float_array(state["robot_obs"]).tolist(),
        "selected_proprio": as_float_array(state["selected_proprio"]).tolist(),
        "scene_obs": as_float_array(state["scene_obs"]).tolist(),
    }


def action_l2_ee6(left: Any, right: Any) -> float:
    delta = as_float_array(left).reshape(-1)[:6] - as_float_array(right).reshape(-1)[:6]
    return float(np.linalg.norm(delta))


def gripper_sign_agreement(left: Any, right: Any) -> bool:
    left_value = float(as_float_array(left).reshape(-1)[-1])
    right_value = float(as_float_array(right).reshape(-1)[-1])
    return bool(np.sign(left_value) == np.sign(right_value))


def snapshot_rng():
    import torch

    return {
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state) -> None:
    import torch

    torch.random.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def runtime_independence(wrapper_c, wrapper_d) -> dict[str, Any]:
    import torch

    checks = {
        "different_wrapper_objects": wrapper_c is not wrapper_d,
        "different_action_buffers": not np.shares_memory(wrapper_c.action_buffer, wrapper_d.action_buffer),
        "different_action_buffer_masks": not np.shares_memory(
            wrapper_c.action_buffer_mask, wrapper_d.action_buffer_mask
        ),
        "different_hist_action_deques": wrapper_c.hist_action is not wrapper_d.hist_action,
        "different_slow_action_storage": wrapper_c.action.data_ptr() != wrapper_d.action.data_ptr(),
        "different_slow_hidden_storage": wrapper_c.hidden_states.data_ptr() != wrapper_d.hidden_states.data_ptr(),
        "same_slow_action_value": bool(torch.equal(wrapper_c.action, wrapper_d.action)),
        "same_slow_hidden_value": bool(torch.equal(wrapper_c.hidden_states, wrapper_d.hidden_states)),
        "same_condition_id": wrapper_c.frozen_condition_id == wrapper_d.frozen_condition_id,
    }
    checks["passed"] = all(checks.values())
    return checks


def profile_json(profile: Mapping[str, Any]) -> dict[str, Any]:
    wanted = (
        "step", "slow_system", "slow_trigger_reason", "slow_age_before",
        "slow_age_after", "num_cond_actions", "ref_action_expired",
        "dp_action_first", "raw_action_prediction", "action_prediction",
        "aggregation_delta_ee6", "raw_aggregation_delta_ee6",
        "action_slew_applied", "action_delta_limit_ee6", "action_jerk_limit_ee6",
    )
    return {key: profile.get(key) for key in wanted}


def rollout_anchor(
    args: argparse.Namespace, index: CalvinLanguageIndex, env, anchor: Mapping[str, Any],
    frames: Mapping[int, Mapping[str, Any]], condition: Mapping[str, Any],
    wrapper_type, dual_system, processor, tokenizer,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import torch

    expert_states = [dataset_state(frames[anchor["anchor_frame"] + k]) for k in range(STATE_STEPS)]
    expert_actions = [
        as_float_array(frames[anchor["anchor_frame"] + k]["rel_actions"]).reshape(ACTION_DIM)
        for k in range(ACTION_STEPS)
    ]

    # Branch B: exact expert actions from a reset to the persisted expert state.
    reset_replay_obs = env.reset(
        robot_obs=expert_states[0]["robot_obs"].copy(),
        scene_obs=expert_states[0]["scene_obs"].copy(),
    )
    replay_states = [capture_env_state(reset_replay_obs)]
    for action in expert_actions:
        next_obs, _, _, _ = env.step(action.copy())
        replay_states.append(capture_env_state(next_obs))
    if len(replay_states) != STATE_STEPS:
        raise AssertionError("Expert replay did not produce 13 states")

    # Branch C and D have separate controller state but share tensor values.
    wrapper_c = build_wrapper(wrapper_type, dual_system, processor, tokenizer)
    wrapper_d = build_wrapper(wrapper_type, dual_system, processor, tokenizer)
    wrapper_c.inject_frozen_condition(condition)
    wrapper_d.inject_frozen_condition(condition)
    independence = runtime_independence(wrapper_c, wrapper_d)
    if not independence["passed"]:
        raise AssertionError(f"Branch C/D runtime state is not independent: {independence}")

    reset_policy_obs = env.reset(
        robot_obs=expert_states[0]["robot_obs"].copy(),
        scene_obs=expert_states[0]["scene_obs"].copy(),
    )
    policy_obs = reset_policy_obs
    policy_states = [capture_env_state(policy_obs)]
    closed_actions: list[np.ndarray] = []
    closed_profiles: list[dict[str, Any]] = []
    paired_rng = snapshot_rng()
    for k in range(ACTION_STEPS):
        action = np.asarray(
            wrapper_c.step(policy_obs, str(anchor["instruction"]), k), dtype=np.float32
        ).reshape(ACTION_DIM)
        closed_actions.append(action.copy())
        closed_profiles.append(profile_json(dict(wrapper_c.last_step_profile)))
        policy_obs, _, _, _ = env.step(action.copy())
        policy_states.append(capture_env_state(policy_obs))

    # Prove C did not mutate the unused D runtime before teacher forcing begins.
    d_pristine_after_c = {
        "hist_action_length": len(wrapper_d.hist_action),
        "action_buffer_nonzero": int(np.count_nonzero(wrapper_d.action_buffer)),
        "last_step_profile": dict(wrapper_d.last_step_profile),
    }
    if d_pristine_after_c != {
        "hist_action_length": 0, "action_buffer_nonzero": 0, "last_step_profile": {}
    }:
        raise AssertionError(f"Branch C mutated Branch D runtime: {d_pristine_after_c}")

    # Pair the diffusion RNG stream across branches; only observations/runtime differ.
    restore_rng(paired_rng)
    teacher_actions: list[np.ndarray] = []
    teacher_profiles: list[dict[str, Any]] = []
    for k in range(ACTION_STEPS):
        expert_obs = dataset_observation(frames[anchor["anchor_frame"] + k])
        action = np.asarray(
            wrapper_d.step(expert_obs, str(anchor["instruction"]), k), dtype=np.float32
        ).reshape(ACTION_DIM)
        teacher_actions.append(action.copy())
        teacher_profiles.append(profile_json(dict(wrapper_d.last_step_profile)))

    closed_ages = [int(profile["slow_age_after"]) for profile in closed_profiles]
    teacher_ages = [int(profile["slow_age_after"]) for profile in teacher_profiles]
    closed_counts = [int(profile["num_cond_actions"]) for profile in closed_profiles]
    teacher_counts = [int(profile["num_cond_actions"]) for profile in teacher_profiles]
    slow_flags = [bool(profile["slow_system"]) for profile in closed_profiles + teacher_profiles]
    if closed_ages != EXPECTED_AGES or teacher_ages != EXPECTED_AGES:
        raise AssertionError(f"Age contract failed: C={closed_ages}, D={teacher_ages}")
    if closed_counts != EXPECTED_COUNTS or teacher_counts != EXPECTED_COUNTS:
        raise AssertionError(f"Reference count contract failed: C={closed_counts}, D={teacher_counts}")
    if any(slow_flags) or wrapper_c.forbidden_slow_call_count or wrapper_d.forbidden_slow_call_count:
        raise AssertionError("Frozen-condition rollout attempted a generalist refresh")

    step_rows: list[dict[str, Any]] = []
    for k in range(STATE_STEPS):
        replay_metric = state_metrics(replay_states[k], expert_states[k])
        policy_metric = state_metrics(policy_states[k], expert_states[k])
        row: dict[str, Any] = {
            "condition_id": anchor["condition_id"],
            "trajectory_id": anchor["trajectory_id"],
            "language_episode_index": anchor["language_episode_index"],
            "task": anchor["task"],
            "instruction": anchor["instruction"],
            "anchor_frame": anchor["anchor_frame"],
            "k": k,
            "state_semantics": (
                "state_after_age11_action" if k == 12 else f"state_before_action_age_{k}"
            ),
            "source_frame_index": anchor["anchor_frame"] + k,
            "trajectory_artifact": anchor["trajectory_artifact"],
            "expert_state": state_json(expert_states[k]),
            "replay_state": state_json(replay_states[k]),
            "closed_loop_state": state_json(policy_states[k]),
            "replay_vs_expert": replay_metric,
            "closed_loop_vs_expert": policy_metric,
            "closed_loop_minus_replay_baseline": {
                metric: policy_metric[metric] - replay_metric[metric] for metric in STATE_METRICS
            },
        }
        if k < ACTION_STEPS:
            expert_action = expert_actions[k]
            teacher_action = teacher_actions[k]
            closed_action = closed_actions[k]
            row.update({
                "action_semantics": "expert_trajectory_action_difference",
                "expert_action": expert_action.tolist(),
                "teacher_forced_action": teacher_action.tolist(),
                "closed_loop_action": closed_action.tolist(),
                "dp_action_first": closed_profiles[k].get("dp_action_first"),
                "teacher_forced_dp_action_first": teacher_profiles[k].get("dp_action_first"),
                "raw_action_prediction": closed_profiles[k].get("raw_action_prediction"),
                "temporal_aggregated_action_prediction": closed_profiles[k].get("action_prediction"),
                "teacher_forced_raw_action_prediction": teacher_profiles[k].get("raw_action_prediction"),
                "teacher_forced_temporal_aggregated_action_prediction": teacher_profiles[k].get("action_prediction"),
                "last_step_profile": closed_profiles[k],
                "teacher_forced_last_step_profile": teacher_profiles[k],
                "num_cond_actions": closed_counts[k],
                "slow_age_after": closed_ages[k],
                "closed_loop_action_ee6_l2_to_expert": action_l2_ee6(closed_action, expert_action),
                "teacher_forced_action_ee6_l2_to_expert": action_l2_ee6(teacher_action, expert_action),
                "closed_loop_vs_teacher_forced_ee6_l2": action_l2_ee6(closed_action, teacher_action),
                "closed_loop_gripper_sign_agreement_to_expert": gripper_sign_agreement(closed_action, expert_action),
                "teacher_forced_gripper_sign_agreement_to_expert": gripper_sign_agreement(teacher_action, expert_action),
                "aggregation_delta_ee6": closed_profiles[k].get("aggregation_delta_ee6"),
            })
        step_rows.append(row)

    contract = {
        "condition_id": anchor["condition_id"],
        "env_reset_succeeded": True,
        "expert_action_replay_steps": len(replay_states) - 1,
        "closed_loop_age_sequence": closed_ages,
        "teacher_forced_age_sequence": teacher_ages,
        "closed_loop_num_cond_actions": closed_counts,
        "teacher_forced_num_cond_actions": teacher_counts,
        "frozen_slow_refresh_count": int(sum(slow_flags)),
        "same_condition_c_d": wrapper_c.frozen_condition_id == wrapper_d.frozen_condition_id,
        "runtime_independence": independence,
        "teacher_runtime_pristine_after_closed_loop": d_pristine_after_c,
        "reset_replay_vs_expert": state_metrics(replay_states[0], expert_states[0]),
        "reset_policy_vs_expert": state_metrics(policy_states[0], expert_states[0]),
        "paired_diffusion_rng_stream": True,
        "history_path_variable": False,
    }
    arrays = {
        "expert_states": expert_states, "replay_states": replay_states,
        "policy_states": policy_states, "expert_actions": expert_actions,
        "teacher_actions": teacher_actions, "closed_actions": closed_actions,
    }
    return step_rows, contract, arrays


def save_trajectory_artifact(path: Path, arrays: Mapping[str, Any]) -> None:
    payload: dict[str, np.ndarray] = {}
    for branch in ("expert", "replay", "policy"):
        states = arrays[f"{branch}_states"]
        for key in (
            "robot_obs", "selected_proprio", "scene_obs", "rgb_static",
            "rgb_gripper", "depth_static", "depth_gripper",
        ):
            payload[f"{branch}_{key}"] = np.stack([state[key] for state in states])
    payload["expert_rel_actions"] = np.stack(arrays["expert_actions"])
    payload["teacher_forced_actions"] = np.stack(arrays["teacher_actions"])
    payload["closed_loop_actions"] = np.stack(arrays["closed_actions"])
    np.savez_compressed(path, **payload)


def save_visual_frames(
    visual_root: Path, anchor: Mapping[str, Any], arrays: Mapping[str, Any]
) -> None:
    from PIL import Image

    root = visual_root / str(anchor["condition_id"])
    for branch in ("expert", "replay", "policy"):
        branch_dir = root / branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        for k, state in enumerate(arrays[f"{branch}_states"]):
            Image.fromarray(np.asarray(state["rgb_static"], dtype=np.uint8)).save(
                branch_dir / f"state_{k:02d}_static.png"
            )
            Image.fromarray(np.asarray(state["rgb_gripper"], dtype=np.uint8)).save(
                branch_dir / f"state_{k:02d}_gripper.png"
            )


def finite_values(rows: Iterable[Mapping[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            number = float(value)
            if np.isfinite(number):
                values.append(number)
    return values


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_state_rows": len(rows)}
    for branch_key in ("replay_vs_expert", "closed_loop_vs_expert", "closed_loop_minus_replay_baseline"):
        result[branch_key] = {
            metric: stats(finite_values(rows, (branch_key, metric))) for metric in STATE_METRICS
        }
    action_rows = [row for row in rows if int(row["k"]) < ACTION_STEPS]
    result["actions"] = {
        metric: stats(finite_values(action_rows, (metric,))) for metric in ACTION_METRICS
    }
    return result


def flatten_summary_row(prefix: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(prefix)
    for branch in ("replay_vs_expert", "closed_loop_vs_expert", "closed_loop_minus_replay_baseline"):
        for metric, metric_stats in summary[branch].items():
            for stat_name, value in metric_stats.items():
                row[f"{branch}.{metric}.{stat_name}"] = value
    for metric, metric_stats in summary["actions"].items():
        for stat_name, value in metric_stats.items():
            row[f"actions.{metric}.{stat_name}"] = value
    return row


def summarize(step_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_age: dict[str, Any] = {}
    age_csv: list[dict[str, Any]] = []
    for k in range(STATE_STEPS):
        group = [row for row in step_rows if int(row["k"]) == k]
        summary = group_summary(group)
        label = "state_12_after_age11_action" if k == 12 else f"age_{k}"
        by_age[label] = summary
        age_csv.append(flatten_summary_row({"k": k, "age_label": label}, summary))

    tasks = sorted({str(row["task"]) for row in step_rows})
    by_task: dict[str, Any] = {}
    task_csv: list[dict[str, Any]] = []
    for task in tasks:
        group = [row for row in step_rows if row["task"] == task]
        summary = group_summary(group)
        by_task[task] = summary
        task_csv.append(flatten_summary_row({"task": task}, summary))

    overall = group_summary(step_rows)
    highlights = {
        label: by_age[label]
        for label in ("age_7", "age_8", "age_11", "state_12_after_age11_action")
    }
    return age_csv, task_csv, {
        "status": "complete",
        "state_index_contract": {
            "s_expert[k]": "persisted frame anchor+k",
            "a_expert[k]": "rel_actions at frame anchor+k",
            "a_policy[k]": "final evaluator action generated at s_policy[k]",
            "s_policy[k+1]": "CALVIN state after env.step(a_policy[k].copy())",
            "actions": "k=0..11",
            "states": "k=0..12",
        },
        "overall": overall,
        "by_age": by_age,
        "by_task": by_task,
        "highlights": highlights,
        "boundary_7_to_8": {"age_7": by_age["age_7"], "age_8": by_age["age_8"]},
        "expert_action_note": (
            "Expert actions are expert-trajectory references, not asserted optimal actions at a drifted policy state."
        ),
    }


def make_manifest(
    args: argparse.Namespace, index: CalvinLanguageIndex, selection: Mapping[str, Any],
    anchors: list[dict[str, Any]], status: str, checkpoint_audit: Mapping[str, Any] | None,
    contracts: list[dict[str, Any]], generalist_calls: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_git_commit": git_commit(),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "dataset_root": str(index.dataset_root),
        "dataset_source_split": "training",
        "language_annotation": str(index.annotation_path),
        "language_annotation_sha256": sha256_file(index.annotation_path),
        "stable_split": args.split,
        "split_rule": "SHA256(trajectory_id) bucket: train<70, validation<85, else test",
        "trajectory_id_rule": "calvin_training_lang_{episode_i:06d}_{start}_{end}_{task}",
        "selection": dict(selection),
        "selected_condition_ids": [row["condition_id"] for row in anchors],
        "seed": args.seed,
        "mode": "dry_run" if args.dry_run else "preflight" if args.preflight_only else "formal",
        "generalist_path": str(args.generalist_path),
        "specialist_path": str(args.specialist_path),
        "checkpoint_loading": checkpoint_audit,
        "generalist": {
            "load_in_4bit": args.load_in_4bit,
            "bnb_4bit_quant_type": "nf4" if args.load_in_4bit else None,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "predict_action_do_sample": False,
            "calls": generalist_calls,
            "expected_calls": 0 if args.dry_run else len(anchors),
            "rollout_refresh_forbidden": True,
            "checkpoint_fingerprint": (
                None if checkpoint_audit is None
                else checkpoint_audit.get("generalist_checkpoint_fingerprint")
            ),
            "processor_tokenizer_fingerprint": (
                None if checkpoint_audit is None
                else checkpoint_audit.get("processor_tokenizer_fingerprint")
            ),
        },
        "specialist": {
            "architecture_source": "vla-scripts/evaluate_calvin_0428.py",
            "fast_num_inference_steps": args.fast_num_inference_steps,
            "with_depth": True, "with_gripper": True, "with_tactile": False,
            "with_cfg": False, "slow_handover_steps": 0,
            "action_delta_limit_ee6": 0.0, "action_jerk_limit_ee6": 0.0,
            "temporal_aggregation_bypassed": False,
            "executed_action_source": "DualSystemCalvinEvaluation.step return value",
        },
        "condition_contract": {
            "source": "persisted expert anchor RGB",
            "one_generalist_call_per_anchor": True,
            "same_call_slow_action_and_hidden": True,
            "shared_exact_values_branch_c_d": True,
            "age_empty_num_cond_actions": EXPECTED_COUNTS,
            "last_slow_step_at_injection": 0,
        },
        "branch_contract": {
            "A": "persisted expert dataset states/actions",
            "B": "expert-action CALVIN replay from persisted robot_obs+scene_obs",
            "C": "specialist closed loop using final evaluator action and temporal aggregation",
            "D": "teacher-forced expert observations with independent evaluator runtime",
            "paired_diffusion_rng_stream_c_d": True,
        },
        "history": {
            "studied_as_variable": False,
            "corruption_or_ablation": False,
            "note": "Evaluator runtime history evolves normally; history is not an experimental variable.",
        },
        "replay_tolerance": None,
        "contracts": contracts,
        "visual_anchor_count": args.save_visual_anchors,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "files": {
            "anchors": "anchors.jsonl", "steps": "trajectory_steps.jsonl",
            "age_summary": "age_summary.csv", "task_summary": "task_summary.csv",
            "summary": "summary.json", "replay_fidelity": "replay_fidelity.json",
            "conditions": "conditions/*.pt", "trajectories": "trajectories/*.npz",
            "visuals": "visuals/<condition_id>/{expert,replay,policy}/*.png (optional)",
        },
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.max_anchors <= 0:
        raise ValueError("--max_anchors must be positive")
    if args.preflight_anchors not in (1, 2):
        raise ValueError("--preflight_anchors must be 1 or 2")
    if args.save_visual_anchors < 0:
        raise ValueError("--save_visual_anchors must be non-negative")
    if args.fast_num_inference_steps != 10:
        raise ValueError("This M1 experiment fixes --fast_num_inference_steps=10")
    if args.dry_run and args.preflight_only:
        raise ValueError("--dry_run and --preflight_only are mutually exclusive")
    if not args.dataset_root.expanduser().resolve().is_dir():
        raise FileNotFoundError(args.dataset_root)
    # Dry-run intentionally does not require model/checkpoint availability.
    if not args.dry_run:
        if not args.generalist_path.expanduser().resolve().is_dir():
            raise FileNotFoundError(args.generalist_path)
        if not args.specialist_path.expanduser().resolve().is_file():
            raise FileNotFoundError(args.specialist_path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.generalist_path = args.generalist_path.expanduser().resolve()
    args.specialist_path = args.specialist_path.expanduser().resolve()
    seed_everything(args.seed)
    index = CalvinLanguageIndex(args.dataset_root)
    selection_count = args.preflight_anchors if args.preflight_only else args.max_anchors
    anchors, selection = select_anchors(index, args.split, selection_count, args.seed)
    run_dir = prepare_run_dir(args.run_name)

    if args.dry_run:
        write_jsonl(run_dir / "anchors.jsonl", anchors)
        write_jsonl(run_dir / "trajectory_steps.jsonl", [])
        write_csv(run_dir / "age_summary.csv", [])
        write_csv(run_dir / "task_summary.csv", [])
        write_json(run_dir / "summary.json", {
            "status": "dry_run_complete", "selection": selection,
            "note": "Dataset/annotation/frame ranges only; no model, checkpoint, or environment loaded.",
        })
        write_json(run_dir / "replay_fidelity.json", {
            "status": "not_run", "reason": "dry_run"
        })
        manifest = make_manifest(
            args, index, selection, anchors, "dry_run_complete", None, [], 0,
        )
        write_json(run_dir / "manifest.json", manifest)
        result = {"run_dir": str(run_dir), "status": "dry_run_complete", **selection}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    import torch

    configure_calvin_imports(index.dataset_root)
    (run_dir / "conditions").mkdir()
    (run_dir / "trajectories").mkdir()
    if args.save_visual_anchors:
        (run_dir / "visuals").mkdir()
    write_json(run_dir / "manifest.json", make_manifest(
        args, index, selection, anchors, "initializing", None, [], 0,
    ))

    dual_system, processor, tokenizer, model_dtype, device, checkpoint_audit = load_models(args)
    checkpoint_exact = not (
        checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]
    )
    checkpoint_audit["exact_architecture_match"] = checkpoint_exact
    if not checkpoint_exact:
        write_json(run_dir / "manifest.json", make_manifest(
            args, index, selection, anchors, "checkpoint_mismatch",
            checkpoint_audit, [], 0,
        ))
        raise RuntimeError(
            "M1 checkpoint does not exactly match the evaluation architecture: "
            f"missing={checkpoint_audit['missing_keys']}, "
            f"unexpected={checkpoint_audit['unexpected_keys']}"
        )
    wrapper_type = frozen_wrapper_class()
    env = make_env(index, args.use_egl)
    all_rows: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    generalist_calls = 0
    try:
        for anchor_i, anchor in enumerate(anchors):
            frames = {
                frame: index.load_frame(frame) for frame in anchor["source_frame_indices"]
            }
            condition = infer_frozen_condition(
                dual_system, processor, model_dtype, anchor,
                frames[anchor["anchor_frame"]]["rgb_static"],
            )
            generalist_calls += 1
            torch.save(condition, run_dir / "conditions" / f"{anchor['condition_id']}.pt")
            rows, contract, arrays = rollout_anchor(
                args, index, env, anchor, frames, condition, wrapper_type,
                dual_system, processor, tokenizer,
            )
            save_trajectory_artifact(
                run_dir / "trajectories" / f"{anchor['condition_id']}.npz", arrays
            )
            if anchor_i < args.save_visual_anchors:
                save_visual_frames(run_dir / "visuals", anchor, arrays)
            all_rows.extend(rows)
            contracts.append(contract)
            print(
                f"[{anchor_i + 1}/{len(anchors)}] {anchor['condition_id']} "
                f"task={anchor['task']} complete", flush=True,
            )
    finally:
        env.close()

    if generalist_calls != len(anchors):
        raise AssertionError(
            f"Generalist call count {generalist_calls} != anchor count {len(anchors)}"
        )
    age_csv, task_csv, summary = summarize(all_rows)
    summary["mode"] = "preflight" if args.preflight_only else "formal"
    summary["anchors"] = len(anchors)
    summary["checkpoint_loading"] = checkpoint_audit
    replay = {
        "status": "measured_no_automatic_tolerance",
        "interpretation_gate": (
            "If replay is not close to persisted expert states, do not attribute specialist divergence to policy drift."
        ),
        "reset_records": [
            {
                "condition_id": contract["condition_id"],
                "reset_replay_vs_expert": contract["reset_replay_vs_expert"],
                "reset_policy_vs_expert": contract["reset_policy_vs_expert"],
            }
            for contract in contracts
        ],
        "overall": summary["overall"]["replay_vs_expert"],
        "by_age": {
            label: value["replay_vs_expert"] for label, value in summary["by_age"].items()
        },
    }
    write_jsonl(run_dir / "anchors.jsonl", anchors)
    write_jsonl(run_dir / "trajectory_steps.jsonl", all_rows)
    write_csv(run_dir / "age_summary.csv", age_csv)
    write_csv(run_dir / "task_summary.csv", task_csv)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "replay_fidelity.json", replay)
    status = "preflight_complete" if args.preflight_only else "complete"
    write_json(run_dir / "manifest.json", make_manifest(
        args, index, selection, anchors, status, checkpoint_audit,
        contracts, generalist_calls,
    ))
    result = {
        "run_dir": str(run_dir), "status": status, "anchors": len(anchors),
        "device": str(device), "generalist_calls": generalist_calls,
        "checkpoint_missing_keys": checkpoint_audit["missing_keys"],
        "checkpoint_unexpected_keys": checkpoint_audit["unexpected_keys"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--generalist_path", type=Path, default=DEFAULT_GENERALIST)
    parser.add_argument("--specialist_path", type=Path, default=DEFAULT_SPECIALIST)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_anchors", type=int, default=50)
    parser.add_argument("--preflight_anchors", type=int, choices=(1, 2), default=1)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--save_visual_anchors", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--fast_num_inference_steps", type=int, default=10)
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--low_cpu_mem_usage", dest="low_cpu_mem_usage", action="store_true")
    parser.add_argument("--no_low_cpu_mem_usage", dest="low_cpu_mem_usage", action="store_false")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="none")
    parser.set_defaults(load_in_4bit=True, low_cpu_mem_usage=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
