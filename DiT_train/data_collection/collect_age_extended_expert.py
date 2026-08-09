#!/usr/bin/env python3
"""Collect deployment-matched age-extended specialist data from CALVIN experts.

One frozen generalist call is made at each selected expert anchor.  The action
chunk and hidden state returned by that *same call* are then shared by twelve
samples whose current expert frames are anchor + age, for ages 0..11.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "robodual_age_extended_expert_v1"
CONDITION_SOURCE = "expert_anchor_runtime_predict_action"
TARGET_SOURCE = "calvin_expert"
AGES = tuple(range(12))
ACTION_CHUNK = 8
ACTION_DIM = 7
HISTORY = 4
DEFAULT_CALVIN_ROOT = Path(__file__).resolve().parents[3] / "calvin"
DEFAULT_DATASET_ROOT = DEFAULT_CALVIN_ROOT / "dataset" / "calvin_debug_dataset"
DEFAULT_GENERALIST = Path(__file__).resolve().parents[3] / "models" / "generalist"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "test_50anchors"
PROMPT_SOURCE = "vla-scripts/dual_sys_evaluation_0424test.py:get_openvla_prompt"


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_fingerprint(path: Path, hash_model_files: bool = False) -> dict[str, Any]:
    """Fingerprint a checkpoint without silently hashing only its directory name."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        stat = path.stat()
        return {"path": str(path), "kind": "file", "size": stat.st_size, "sha256": sha256_file(path)}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    metadata = hashlib.sha256()
    content_hashes: dict[str, str] = {}
    small_names = {
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "tokenizer.json", "tokenizer.model",
        "configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py",
    }
    for item in files:
        relative = item.relative_to(path).as_posix()
        stat = item.stat()
        metadata.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        if hash_model_files or item.name in small_names:
            content_hashes[relative] = sha256_file(item)
    return {
        "path": str(path),
        "kind": "directory",
        "file_count": len(files),
        "tree_metadata_sha256": metadata.hexdigest(),
        "content_sha256": content_hashes,
        "weight_files_content_hashed": bool(hash_model_files),
    }


def processor_fingerprint(model_path: Path) -> dict[str, Any]:
    names = {
        "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "tokenizer.json", "tokenizer.model",
        "processing_prismatic.py",
    }
    files = sorted(item for item in model_path.expanduser().resolve().iterdir() if item.name in names)
    return {"files": {item.name: sha256_file(item) for item in files}}


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stable_split(trajectory_id: str) -> str:
    bucket = int(hashlib.sha256(trajectory_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def deployment_prompt(instruction: str) -> str:
    # Kept byte-for-byte equivalent to the current P12 evaluator and recorded
    # in the manifest.  It deliberately has no target/action text.
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def build_reference(slow_action: Any, age: int):
    """Build the exact age_empty P12 reference and mask on CPU."""
    import torch

    action = torch.as_tensor(slow_action)
    if tuple(action.shape) != (1, ACTION_CHUNK, ACTION_DIM):
        raise AssertionError(f"slow_action must be [1,8,7], got {tuple(action.shape)}")
    if age not in AGES:
        raise ValueError(f"age must be 0..11, got {age}")
    ref = torch.zeros_like(action)
    count = max(ACTION_CHUNK - age, 0)
    if count:
        ref[:, :count] = action[:, -count:]
    mask = torch.zeros(ACTION_CHUNK, dtype=torch.bool)
    mask[:count] = True
    return ref, count, mask


def cpu_contract_test() -> dict[str, Any]:
    import torch

    slow = torch.arange(56, dtype=torch.float32).reshape(1, 8, 7)
    rows = []
    for age in AGES:
        ref, count, mask = build_reference(slow, age)
        expected = max(8 - age, 0)
        assert count == expected and int(mask.sum()) == expected
        if count:
            assert torch.equal(ref[:, :count], slow[:, -count:])
            assert torch.count_nonzero(ref[:, count:]).item() == 0
        else:
            assert torch.count_nonzero(ref).item() == 0
        rows.append({"age": age, "ref_valid_count": count})
    assert rows[7]["ref_valid_count"] == 1
    assert rows[8]["ref_valid_count"] == 0
    return {"status": "passed", "ages": rows, "transition_7_to_8": [1, 0]}


class CalvinExpertIndex:
    def __init__(self, dataset_root: Path, dataset_split: str):
        if dataset_split != "training":
            raise ValueError("Age-Extended Expert collection is restricted to CALVIN training split")
        self.dataset_root = dataset_root.expanduser().resolve()
        self.split = dataset_split
        self.split_dir = self.dataset_root / dataset_split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(self.split_dir)
        self.annotation_path = self.split_dir / "lang_annotations" / "auto_lang_ann.npy"
        if not self.annotation_path.is_file():
            fallback = self.split_dir / "auto_lang_ann.npy"
            if not fallback.is_file():
                raise FileNotFoundError(self.annotation_path)
            self.annotation_path = fallback
        data = np.load(self.annotation_path, allow_pickle=True).item()
        self.bounds = [(int(start), int(end)) for start, end in data["info"]["indx"]]
        self.instructions = [str(item) for item in data["language"]["ann"]]
        self.tasks = [str(item) for item in data["language"]["task"]]
        if not (len(self.bounds) == len(self.instructions) == len(self.tasks)):
            raise AssertionError("CALVIN language annotation arrays have inconsistent lengths")
        examples = sorted(self.split_dir.glob("episode_*.npz"))
        if not examples:
            raise FileNotFoundError(f"No episode_*.npz files in {self.split_dir}")
        match = re.match(r"^(.*?)(\d+)(\.npz)$", examples[0].name)
        if not match:
            raise ValueError(f"Cannot infer CALVIN frame naming from {examples[0]}")
        self.prefix, digits, self.suffix = match.groups()
        self.n_digits = len(digits)

    def frame_path(self, frame: int) -> Path:
        return self.split_dir / f"{self.prefix}{frame:0{self.n_digits}d}{self.suffix}"

    def frame_ref(self, frame: int, key: str) -> dict[str, Any]:
        path = self.frame_path(frame)
        return {
            "frame_index": int(frame),
            "relative_path": path.relative_to(self.dataset_root).as_posix(),
            "key": key,
        }

    def load_frame(self, frame: int) -> dict[str, np.ndarray]:
        path = self.frame_path(frame)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            required = {
                "rgb_static", "rgb_gripper", "depth_static", "depth_gripper",
                "robot_obs", "scene_obs", "rel_actions",
            }
            missing = required.difference(archive.files)
            if missing:
                raise KeyError(f"{path} lacks {sorted(missing)}")
            return {key: np.asarray(archive[key]).copy() for key in required}

    def episode_rows(self) -> list[dict[str, Any]]:
        rows = []
        for episode_i, ((start, end), instruction, task) in enumerate(
            zip(self.bounds, self.instructions, self.tasks)
        ):
            trajectory_id = f"calvin_training_lang_{episode_i:06d}_{start}_{end}_{task}"
            # CALVIN annotation end is inclusive.  age=11 then 8 targets uses
            # positions anchor+11 .. anchor+18, hence anchor <= end-18.
            valid_last_anchor = end - ((max(AGES)) + ACTION_CHUNK - 1)
            rows.append({
                "language_episode_index": episode_i,
                "trajectory_id": trajectory_id,
                "split": stable_split(trajectory_id),
                "task": task,
                "instruction": instruction,
                "task_start_frame": start,
                "task_end_frame_inclusive": end,
                "valid_anchor_first": start,
                "valid_anchor_last_inclusive": valid_last_anchor,
                "valid_anchor_count": max(0, valid_last_anchor - start + 1),
            })
        return rows


def select_anchors(index: CalvinExpertIndex, max_anchors: int, per_episode: int, seed: int):
    if max_anchors <= 0 or per_episode <= 0:
        raise ValueError("max_anchors and max_anchors_per_episode must be positive")
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid = Counter()
    for episode in index.episode_rows():
        if episode["valid_anchor_count"] <= 0:
            invalid["subtask_shorter_than_19_positions"] += 1
            continue
        frames = list(range(episode["valid_anchor_first"], episode["valid_anchor_last_inclusive"] + 1))
        rng.shuffle(frames)
        for frame in frames[:per_episode]:
            row = dict(episode)
            row["anchor_frame"] = frame
            row["anchor_task_local_step"] = frame - episode["task_start_frame"]
            pools[episode["task"]].append(row)
    task_order = sorted(pools, key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).hexdigest())
    for task in task_order:
        rng.shuffle(pools[task])
    selected: list[dict[str, Any]] = []
    while len(selected) < max_anchors:
        made_progress = False
        for task in task_order:
            if pools[task] and len(selected) < max_anchors:
                selected.append(pools[task].pop())
                made_progress = True
        if not made_progress:
            break
    for i, row in enumerate(selected):
        row["condition_id"] = f"condition_{i:06d}"
    report = {
        "requested_anchors": max_anchors,
        "selected_anchors": len(selected),
        "expected_samples": len(selected) * len(AGES),
        "max_anchors_per_language_episode": per_episode,
        "task_counts": dict(sorted(Counter(row["task"] for row in selected).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in selected).items())),
        "language_episodes_scanned": len(index.bounds),
        "ineligible_episode_reasons": dict(invalid),
        "shortfall": max(0, max_anchors - len(selected)),
        "shortfall_reason": None if len(selected) == max_anchors else "eligible anchors exhausted under per-episode cap",
    }
    return selected, report


def selected_proprio(robot_obs: np.ndarray) -> np.ndarray:
    robot_obs = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    if robot_obs.size < 7:
        raise AssertionError(f"robot_obs too short: {robot_obs.shape}")
    return np.concatenate([robot_obs[:6], robot_obs[-1:]]).astype(np.float32)


def vector_delta(anchor: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(current, dtype=np.float32) - np.asarray(anchor, dtype=np.float32)
    return {"vector": difference.tolist(), "l2_norm": float(np.linalg.norm(difference))}


def history_for(index: CalvinExpertIndex, task_start: int, current: int):
    history = np.zeros((HISTORY, ACTION_DIM), dtype=np.float32)
    source_indices = list(range(max(task_start, current - HISTORY), current))
    if source_indices:
        actions = [np.asarray(index.load_frame(frame)["rel_actions"], dtype=np.float32) for frame in source_indices]
        history[-len(actions):] = np.stack(actions)
    return history, source_indices


def load_generalist(args: argparse.Namespace):
    import torch
    from accelerate import Accelerator
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    quantization = None
    model_dtype = torch.bfloat16
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
        model_dtype = torch.float16
    elif args.load_in_8bit:
        quantization = BitsAndBytesConfig(load_in_8bit=True)
        model_dtype = torch.float16
    kwargs: dict[str, Any] = {
        "torch_dtype": model_dtype,
        "quantization_config": quantization,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "trust_remote_code": True,
    }
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map
    if args.attn_implementation != "none":
        kwargs["attn_implementation"] = args.attn_implementation
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **kwargs).eval()
    accelerator = Accelerator()
    if args.device_map == "none":
        if not torch.cuda.is_available():
            raise RuntimeError("Generalist collection requires CUDA; use --dry_run or --cpu_contract_test on CPU")
        model = accelerator.prepare(model, device_placement=[True])
    model.eval()
    return model, processor, model_dtype


def model_device(model):
    import torch
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def infer_condition(model, processor, model_dtype, rgb_static: np.ndarray, instruction: str):
    import torch
    from PIL import Image

    prompt = deployment_prompt(instruction)
    inputs = processor(prompt, Image.fromarray(rgb_static)).to(model_device(model), dtype=model_dtype)
    forbidden = {"labels", "actions", "rel_actions", "target", "target_rel_actions"}.intersection(inputs.keys())
    if forbidden:
        raise AssertionError(f"Processor produced forbidden teacher-forcing fields: {sorted(forbidden)}")
    call_id = str(uuid.uuid4())
    model.eval()
    with torch.inference_mode():
        output = model.predict_action(**inputs, do_sample=False)
    if not isinstance(output, tuple) or len(output) < 2:
        raise RuntimeError("Generalist predict_action must return (action, hidden_states) from one call")
    raw_action, raw_hidden = output[0], output[1]
    action = torch.as_tensor(raw_action).detach()
    if action.numel() != ACTION_CHUNK * ACTION_DIM:
        raise AssertionError(f"Generalist action has {action.numel()} values, expected 56")
    action = action.reshape(1, ACTION_CHUNK, -1)[:, :, :ACTION_DIM].to(torch.float32).cpu()
    hidden = torch.as_tensor(raw_hidden).detach().cpu()
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] <= 0 or hidden.shape[2] <= 0:
        raise AssertionError(f"Illegal slow_hidden shape {tuple(hidden.shape)}")
    return action, hidden.to(torch.float16), call_id


def make_sample(index: CalvinExpertIndex, anchor: dict[str, Any], age: int, slow_action, materialize_dir: Path | None):
    current = anchor["anchor_frame"] + age
    start, end = anchor["task_start_frame"], anchor["task_end_frame_inclusive"]
    previous = current if current == start else current - 1
    current_data = index.load_frame(current)
    anchor_data = index.load_frame(anchor["anchor_frame"])
    history, history_indices = history_for(index, start, current)
    target_indices = list(range(current, current + ACTION_CHUNK))
    if target_indices[-1] > end:
        raise AssertionError("Target crosses language subtask boundary")
    target = np.stack([
        np.asarray(index.load_frame(frame)["rel_actions"], dtype=np.float32) for frame in target_indices
    ])
    ref, count, mask = build_reference(slow_action, age)
    sample_id = f"{anchor['condition_id']}_age_{age:02d}"
    refs = {
        "current_rgb_static": index.frame_ref(current, "rgb_static"),
        "previous_rgb_static": index.frame_ref(previous, "rgb_static"),
        "current_depth_static": index.frame_ref(current, "depth_static"),
        "current_rgb_gripper": index.frame_ref(current, "rgb_gripper"),
        "current_depth_gripper": index.frame_ref(current, "depth_gripper"),
    }
    materialized = None
    if materialize_dir is not None:
        previous_data = index.load_frame(previous)
        materialized = f"observations/{sample_id}.npz"
        np.savez_compressed(
            materialize_dir / f"{sample_id}.npz",
            rgb_static=current_data["rgb_static"],
            previous_rgb_static=previous_data["rgb_static"],
            depth_static=current_data["depth_static"],
            rgb_gripper=current_data["rgb_gripper"],
            depth_gripper=current_data["depth_gripper"],
            robot_obs=current_data["robot_obs"], scene_obs=current_data["scene_obs"],
            hist_action_before=history, target_rel_actions=target,
        )
    anchor_proprio = selected_proprio(anchor_data["robot_obs"])
    current_proprio = selected_proprio(current_data["robot_obs"])
    sample = {
        "sample_id": sample_id,
        "condition_id": anchor["condition_id"],
        "condition_path": f"conditions/{anchor['condition_id']}.pt",
        "trajectory_id": anchor["trajectory_id"], "split": anchor["split"],
        "dataset_source_split": "training", "task": anchor["task"],
        "instruction": anchor["instruction"],
        "anchor_frame": anchor["anchor_frame"], "current_frame": current,
        "previous_frame": previous, "task_start_frame": start,
        "task_end_frame_inclusive": end, "task_local_step": current - start,
        "slow_age": age, "ref_valid_count": count,
        "ref_valid_mask": mask.tolist(), "ref_expired": age >= ACTION_CHUNK,
        "ref_action": ref.to(torch_float32()).tolist(),
        "hist_action_before": history.tolist(), "history_source_frames": history_indices,
        "history_zero_pad_count": HISTORY - len(history_indices),
        "current_proprio": current_proprio.tolist(), "anchor_proprio": anchor_proprio.tolist(),
        "anchor_to_current_proprio_delta": vector_delta(anchor_proprio, current_proprio),
        "current_robot_obs": np.asarray(current_data["robot_obs"], dtype=np.float32).tolist(),
        "anchor_robot_obs": np.asarray(anchor_data["robot_obs"], dtype=np.float32).tolist(),
        "current_scene_state": np.asarray(current_data["scene_obs"], dtype=np.float32).tolist(),
        "anchor_scene_state": np.asarray(anchor_data["scene_obs"], dtype=np.float32).tolist(),
        "anchor_to_current_scene_delta": vector_delta(anchor_data["scene_obs"], current_data["scene_obs"]),
        **refs, "materialized_observation": materialized,
        "target_rel_actions": target.tolist(), "target_source": TARGET_SOURCE,
        "target_source_frames": target_indices,
        "source_frame_indices": sorted(set([anchor["anchor_frame"], current, previous] + history_indices + target_indices)),
    }
    return sample


def torch_float32():
    import torch
    return torch.float32


def audit_in_memory(anchors, conditions, samples, inference_calls: int) -> dict[str, Any]:
    import torch

    failures: list[str] = []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        by_condition[sample["condition_id"]].append(sample)
        trajectory_splits[sample["trajectory_id"]].add(sample["split"])
        age = sample["slow_age"]
        if sample["current_frame"] - sample["anchor_frame"] != age:
            failures.append(f"{sample['sample_id']}: age/frame mismatch")
        if np.asarray(sample["target_rel_actions"]).shape != (8, 7):
            failures.append(f"{sample['sample_id']}: target shape")
        if sample["target_source_frames"] != list(range(sample["current_frame"], sample["current_frame"] + 8)):
            failures.append(f"{sample['sample_id']}: target indices")
        if any(frame >= sample["current_frame"] for frame in sample["history_source_frames"]):
            failures.append(f"{sample['sample_id']}: action leakage")
        expected_history = list(range(max(sample["task_start_frame"], sample["current_frame"] - 4), sample["current_frame"]))
        if sample["history_source_frames"] != expected_history:
            failures.append(f"{sample['sample_id']}: history source mismatch")
        expected_previous = sample["current_frame"] if sample["current_frame"] == sample["task_start_frame"] else sample["current_frame"] - 1
        if sample["previous_frame"] != expected_previous:
            failures.append(f"{sample['sample_id']}: previous crosses boundary")
        condition = conditions[sample["condition_id"]]
        ref, count, mask = build_reference(condition["slow_action"], age)
        if count != sample["ref_valid_count"] or mask.tolist() != sample["ref_valid_mask"]:
            failures.append(f"{sample['sample_id']}: ref count/mask")
        if not torch.equal(ref.to(torch.float32), torch.tensor(sample["ref_action"], dtype=torch.float32)):
            failures.append(f"{sample['sample_id']}: ref content")
        if age >= 8 and torch.count_nonzero(ref).item() != 0:
            failures.append(f"{sample['sample_id']}: expired ref nonzero")
        if sample["task_end_frame_inclusive"] < sample["target_source_frames"][-1]:
            failures.append(f"{sample['sample_id']}: target crosses task")
    for anchor in anchors:
        cid = anchor["condition_id"]
        ages = sorted(sample["slow_age"] for sample in by_condition[cid])
        if ages != list(AGES):
            failures.append(f"{cid}: incomplete ages {ages}")
        condition = conditions[cid]
        if tuple(condition["slow_action"].shape) != (1, 8, 7):
            failures.append(f"{cid}: slow_action shape")
        if condition["slow_hidden"].ndim != 3 or condition["hidden_seq_len"] != condition["slow_hidden"].shape[1]:
            failures.append(f"{cid}: slow_hidden shape/length")
        if condition["source"] != CONDITION_SOURCE or not condition["same_inference_call"]:
            failures.append(f"{cid}: condition provenance")
    if inference_calls != len(anchors) or len(conditions) != len(anchors):
        failures.append("generalist call/anchor/condition count mismatch")
    if any(len(splits) != 1 for splits in trajectory_splits.values()):
        failures.append("trajectory split leakage")
    if len(samples) != len(anchors) * len(AGES):
        failures.append("sample count mismatch")
    if failures:
        raise AssertionError("Contract audit failed:\n- " + "\n- ".join(failures))
    return {
        "status": "passed", "checks": 18, "failures": [],
        "unique_anchors": len(anchors), "unique_conditions": len(conditions),
        "total_samples": len(samples), "expected_samples": len(anchors) * 12,
        "generalist_inference_calls": inference_calls,
        "age7_valid_count": next(row["ref_valid_count"] for row in samples if row["slow_age"] == 7),
        "age8_valid_count": next(row["ref_valid_count"] for row in samples if row["slow_age"] == 8),
        "trajectory_split_leakage": False,
    }


def inspect_anchor(samples: list[dict[str, Any]]) -> dict[str, Any]:
    cid = samples[0]["condition_id"]
    chosen = {row["slow_age"]: row for row in samples if row["condition_id"] == cid}
    return {
        "condition_id": cid,
        "ages": [{
            "age": age, "anchor_frame": chosen[age]["anchor_frame"],
            "current_frame": chosen[age]["current_frame"], "previous_frame": chosen[age]["previous_frame"],
            "history_indices": chosen[age]["history_source_frames"],
            "target_indices": chosen[age]["target_source_frames"],
            "ref_valid_count": chosen[age]["ref_valid_count"],
        } for age in (0, 7, 8, 11)],
    }


def prepare_stage(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent))


def publish_stage(stage: Path, output_dir: Path, overwrite: bool) -> Path | None:
    output_dir = output_dir.expanduser().resolve()
    backup = None
    if output_dir.exists():
        if output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        elif not overwrite:
            raise FileExistsError(f"{output_dir} is non-empty; use --overwrite")
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = output_dir.with_name(f"{output_dir.name}.backup-{stamp}")
            if backup.exists():
                raise FileExistsError(backup)
            output_dir.rename(backup)
    stage.rename(output_dir)
    return backup


def run_independent_verifier(stage: Path) -> dict[str, Any]:
    verifier_path = Path(__file__).resolve().with_name("verify_age_extended_contract.py")
    spec = importlib.util.spec_from_file_location("age_extended_contract_verifier", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier.verify_run(stage)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    index = CalvinExpertIndex(Path(args.dataset_root), args.dataset_split)
    anchors, scan = select_anchors(index, args.max_anchors, args.max_anchors_per_episode, args.seed)
    if args.dry_run:
        result = {"mode": "dry_run", "dataset_root": str(index.dataset_root), "dataset_split": index.split, **scan}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    if not anchors:
        raise RuntimeError("No legal anchors; refusing to create an empty dataset")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.overwrite:
        raise FileExistsError(f"{output} is non-empty; use --overwrite")
    stage = prepare_stage(output)
    (stage / "conditions").mkdir()
    observation_dir = None
    if args.materialize_observations:
        observation_dir = stage / "observations"
        observation_dir.mkdir()
    model, processor, model_dtype = load_generalist(args)
    repo_root = Path(__file__).resolve().parents[2]
    fingerprints = {
        "generalist_checkpoint": artifact_fingerprint(Path(args.generalist_path), args.hash_model_files),
        "processor_tokenizer": processor_fingerprint(Path(args.generalist_path)),
    }
    shared_provenance = {
        "generalist_path": str(Path(args.generalist_path).expanduser().resolve()),
        "generalist_checkpoint_fingerprint": fingerprints["generalist_checkpoint"],
        "processor_tokenizer_fingerprint": fingerprints["processor_tokenizer"],
        "generalist_dtype": str(model_dtype),
        "generalist_quantization": {
            "load_in_4bit": args.load_in_4bit, "load_in_8bit": args.load_in_8bit,
            "bnb_4bit_quant_type": "nf4" if args.load_in_4bit else None,
            "bnb_4bit_compute_dtype": "torch.float16" if args.load_in_4bit else None,
            "device_map": args.device_map, "low_cpu_mem_usage": args.low_cpu_mem_usage,
            "attn_implementation": args.attn_implementation,
        },
        "code_git_commit": git_commit(repo_root), "collector_schema_version": SCHEMA_VERSION,
    }
    conditions: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    inference_calls = 0
    for anchor in anchors:
        anchor_data = index.load_frame(anchor["anchor_frame"])
        slow_action, slow_hidden, call_id = infer_condition(
            model, processor, model_dtype, anchor_data["rgb_static"], anchor["instruction"]
        )
        inference_calls += 1
        condition = {
            "condition_id": anchor["condition_id"], "trajectory_id": anchor["trajectory_id"],
            "task": anchor["task"], "instruction": anchor["instruction"], "split": anchor["split"],
            "dataset_source_split": "training",
            "language_annotation_provenance": str(index.annotation_path),
            "language_episode_index": anchor["language_episode_index"],
            "anchor_absolute_frame_index": anchor["anchor_frame"],
            "anchor_task_local_step": anchor["anchor_task_local_step"],
            "anchor_observation_reference": index.frame_ref(anchor["anchor_frame"], "rgb_static"),
            "slow_action": slow_action, "slow_hidden": slow_hidden,
            "hidden_seq_len": int(slow_hidden.shape[1]), "hidden_width": int(slow_hidden.shape[2]),
            "source": CONDITION_SOURCE, "inference_call_id": call_id,
            "slow_action_inference_call_id": call_id, "slow_hidden_inference_call_id": call_id,
            "same_inference_call": True, "do_sample": False, "teacher_forcing": False,
            "future_expert_actions_supplied_to_generalist": False,
            **shared_provenance,
        }
        conditions[anchor["condition_id"]] = condition
        torch.save(condition, stage / "conditions" / f"{anchor['condition_id']}.pt")
        anchor_rows.append({**anchor, "condition_path": f"conditions/{anchor['condition_id']}.pt"})
        for age in AGES:
            samples.append(make_sample(index, anchor, age, slow_action, observation_dir))
    audit = audit_in_memory(anchors, conditions, samples, inference_calls)
    inspection = inspect_anchor(samples)
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "dataset_root": str(index.dataset_root),
        "dataset_source_split": "training", "language_annotation": str(index.annotation_path),
        "annotation_sha256": sha256_file(index.annotation_path),
        "unique_anchors": len(anchors), "unique_conditions": len(conditions),
        "total_samples": len(samples), "expected_samples": len(anchors) * 12,
        "ages": list(AGES), "action_chunk_size": 8, "history_size": 4,
        "with_tactile": False, "materialize_observations": args.materialize_observations,
        "condition_source": CONDITION_SOURCE, "target_source": TARGET_SOURCE,
        "prompt_template": "In: What action should the robot take to {instruction.lower()}?\\nOut:",
        "prompt_source": PROMPT_SOURCE, "processor_call": "processor(prompt, PIL.Image.fromarray(anchor_rgb_static))",
        "generalist_inference": "model.eval(); torch.inference_mode(); predict_action(**inputs, do_sample=False)",
        "generalist_path": str(Path(args.generalist_path).expanduser().resolve()),
        "generalist_dtype": str(model_dtype),
        "generalist_quantization": {
            "load_in_4bit": args.load_in_4bit, "load_in_8bit": args.load_in_8bit,
            "bnb_4bit_quant_type": "nf4" if args.load_in_4bit else None,
            "bnb_4bit_compute_dtype": "torch.float16" if args.load_in_4bit else None,
            "device_map": args.device_map, "low_cpu_mem_usage": args.low_cpu_mem_usage,
            "attn_implementation": args.attn_implementation,
        },
        "fingerprints": fingerprints, "code_git_commit": shared_provenance["code_git_commit"],
        "collector_file_sha256": sha256_file(Path(__file__).resolve()),
        "selection": scan,
        "split_rule": "SHA256(trajectory_id) bucket: train<70, validation<85, else test",
        "anchor_legality": "inclusive CALVIN subtask end; anchor <= end-(11+8-1) = end-18",
        "benchmark_exclusion": "not applicable: official generated benchmark sequence fingerprints do not identify expert annotation episodes",
        "files": {"conditions": "conditions/*.pt", "samples": "samples.jsonl", "anchors": "anchors.jsonl"},
    }
    jsonl_dump(stage / "samples.jsonl", samples)
    jsonl_dump(stage / "anchors.jsonl", anchor_rows)
    audit_payload = {**audit, "inspection": inspection, "cpu_contract": cpu_contract_test()}
    json_dump(stage / "audit_summary.json", audit_payload)
    json_dump(stage / "manifest.json", manifest)
    independent = run_independent_verifier(stage)
    audit_payload["independent_verifier"] = {
        "status": independent["status"],
        "checks_executed": independent["checks_executed"],
        "checks_by_category": independent["checks_by_category"],
    }
    json_dump(stage / "audit_summary.json", audit_payload)
    backup = publish_stage(stage, output, args.overwrite)
    result = {"output_dir": str(output), "backup_of_previous_output": None if backup is None else str(backup), **audit, "inspection": inspection}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset_split", default="training", choices=["training"], help="Hard-restricted to expert training data")
    parser.add_argument("--generalist_path", default=str(DEFAULT_GENERALIST))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max_anchors", type=int, default=50)
    parser.add_argument("--max_anchors_per_episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--materialize_observations", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Preserve old output as a timestamped backup, then publish")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="none")
    parser.add_argument("--hash_model_files", action="store_true", help="SHA256 all weight shards (slow but strongest provenance)")
    parser.add_argument("--cpu_contract_test", action="store_true", help="Run only the synthetic d=0..11 reference test")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args(argv)
    if args.load_in_4bit and args.load_in_8bit:
        parser.error("--load_in_4bit and --load_in_8bit are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    if args.cpu_contract_test:
        print(json.dumps(cpu_contract_test(), indent=2, sort_keys=True))
        return
    collect(args)


if __name__ == "__main__":
    main()
