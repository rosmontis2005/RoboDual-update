#!/usr/bin/env python3
"""Stage-1 recovery LoRA: positive recovery BC plus frozen-base normal replay.

Preference pairs are used as an admission requirement and to identify positive
branches. Pairwise optimization is intentionally deferred until this protected
BC stage demonstrates held-out recovery gain without normal-state drift.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
EXPERIMENT_ROOT = REPO_ROOT / "LoRA_transition_0711"
for path in (REPO_ROOT, REPO_ROOT.parent, EXPERIMENT_ROOT, REPO_ROOT / "LoRA_trial"):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from history_adapter import install_history_adapter  # noqa: E402
from train_lora_specialist import count_trainable_params, inject_lora, load_specialist_checkpoint  # noqa: E402
from train_transition_lora_stale_action_condition_0716 import (  # noqa: E402
    DEFAULT_LORA_TARGETS,
    TransitionCollator,
    TransitionManifestDataset,
    build_transition_policy,
    compute_preserved_objective,
    evaluate_objective,
    front_weighted_mse,
    load_transition_adapter,
    move_batch,
    policy_loss_details,
    save_merged_checkpoints,
    save_transition_adapter,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


class RecoveryDataset(Dataset):
    def __init__(self, root: Path, split: str, action_chunk_size: int = 8):
        self.root = root.expanduser().resolve()
        self.split = split
        self.action_chunk_size = action_chunk_size
        pairs = [item for item in read_jsonl(self.root / "pairs.jsonl") if item["split"] == split]
        branches = {item["branch_id"]: item for item in read_jsonl(self.root / "branches.jsonl")}
        states = {item["failure_state_id"]: item for item in read_jsonl(self.root / "failure_states.jsonl")}
        positive_ids = [
            branch_id
            for branch_id in dict.fromkeys(item["positive_branch_id"] for item in pairs)
            if int(branches[branch_id]["steps"]) >= self.action_chunk_size
        ]
        positive_id_set = set(positive_ids)
        chunks_path = self.root / "trajectory_chunks.jsonl"
        chunks = read_jsonl(chunks_path) if chunks_path.is_file() else []
        chunk_samples = [
            item for item in chunks
            if item["branch_id"] in positive_id_set
            and int(item["steps"]) >= self.action_chunk_size
        ]
        chunk_branch_ids = {item["branch_id"] for item in chunk_samples}
        self.samples = [
            (
                "chunk",
                item,
                branches[item["branch_id"]],
                states[item["failure_state_id"]],
            )
            for item in chunk_samples
        ]
        self.samples.extend(
            (
                "legacy",
                None,
                branches[branch_id],
                states[branches[branch_id]["failure_state_id"]],
            )
            for branch_id in positive_ids
            if branch_id not in chunk_branch_ids
        )
        if not self.samples:
            raise ValueError(f"No positive recovery branches for split={split!r}")
        self.depth_min, self.depth_max = 3.5, 6.2
        self.gripper_depth_min, self.gripper_depth_max = 0.0, 2.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_kind, chunk, branch, state = self.samples[index]
        state_id = state["failure_state_id"]
        branch_id = branch["branch_id"]
        if sample_kind == "chunk":
            payload_path = self.root / "trajectory_chunks" / f"{chunk['chunk_id']}.npz"
            condition_path = (
                self.root / "trajectory_conditions" / f"{chunk['chunk_id']}.pt"
            )
        else:
            payload_path = self.root / "states" / f"{state_id}.npz"
            condition_path = self.root / "conditions" / f"{branch_id}.pt"
        with np.load(payload_path, allow_pickle=False) as payload:
            snapshot = {key: payload[key] for key in payload.files}
        if sample_kind == "chunk":
            actions = np.asarray(snapshot["actions"], dtype=np.float32)
        else:
            with np.load(self.root / "branches" / f"{branch_id}.npz", allow_pickle=False) as payload:
                actions = np.asarray(payload["actions"], dtype=np.float32)
        if len(actions) < self.action_chunk_size:
            raise ValueError(f"Successful branch is shorter than one action chunk: {branch_id}")
        condition = torch.load(condition_path, map_location="cpu", weights_only=False)
        robot_obs = np.asarray(snapshot["robot_obs"], dtype=np.float32)
        return {
            "sample_id": index,
            "sample_key": (
                state_id,
                branch_id if chunk is None else chunk["chunk_id"],
            ),
            "trajectory_id": branch_id,
            "task": state["task"],
            "category": "recovery",
            "instruction": state["instruction"],
            "slow_age": int(condition["slow_age"]),
            "current_image": Image.fromarray(np.asarray(snapshot["rgb_static"], dtype=np.uint8)),
            "previous_image": Image.fromarray(np.asarray(snapshot["previous_rgb"], dtype=np.uint8)),
            "gripper_image": Image.fromarray(np.asarray(snapshot["rgb_gripper"], dtype=np.uint8)),
            "depth_image": torch.from_numpy(
                (np.asarray(snapshot["depth_static"], dtype=np.float32) - self.depth_min)
                / (self.depth_max - self.depth_min)
            ).float(),
            "depth_gripper": torch.from_numpy(
                (np.asarray(snapshot["depth_gripper"], dtype=np.float32) - self.gripper_depth_min)
                / (self.gripper_depth_max - self.gripper_depth_min)
            ).float(),
            "raw_action": torch.from_numpy(actions[: self.action_chunk_size].copy()).float(),
            "hist_action": torch.from_numpy(np.asarray(snapshot["hist_action"], dtype=np.float32)).float(),
            "proprio": torch.from_numpy(np.concatenate([robot_obs[:6], robot_obs[-1:]])).float(),
            "slow_action": torch.as_tensor(condition["slow_action"], dtype=torch.float32).squeeze(0),
            "slow_hidden": torch.as_tensor(condition["slow_hidden"], dtype=torch.float32).squeeze(0),
        }


class RecoveryCollator(TransitionCollator):
    def __call__(self, instances):
        batch = super().__call__(instances)
        if "negative_raw_action" in instances[0]:
            batch["negative_raw_action"] = torch.stack([
                item["negative_raw_action"] for item in instances
            ])
        return batch


class RecoveryPreferenceDataset(Dataset):
    """Same-state positive/negative first chunks for explicit ranking."""

    def __init__(self, root: Path, split: str, action_chunk_size: int = 8):
        self.root = root.expanduser().resolve()
        self.split = split
        self.positive = RecoveryDataset(
            self.root, split, action_chunk_size=action_chunk_size
        )
        pairs = [
            item for item in read_jsonl(self.root / "pairs.jsonl")
            if item["split"] == split
        ]
        chunks = read_jsonl(self.root / "trajectory_chunks.jsonl")
        initial_chunks = {
            item["branch_id"]: item
            for item in chunks
            if int(item["start_offset"]) == 0
            and int(item["steps"]) >= action_chunk_size
        }
        positive_indices = {}
        for index, (_, chunk, branch, _) in enumerate(self.positive.samples):
            if chunk is not None and int(chunk["start_offset"]) == 0:
                positive_indices[branch["branch_id"]] = index
        self.samples = []
        for pair in pairs:
            positive_id = pair["positive_branch_id"]
            negative_id = pair["negative_branch_id"]
            if (
                positive_id in positive_indices
                and negative_id in initial_chunks
            ):
                self.samples.append((
                    positive_indices[positive_id],
                    initial_chunks[negative_id],
                    pair["pair_id"],
                ))
        if not self.samples:
            raise ValueError(
                f"No aligned positive/negative initial chunks for split={split!r}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        positive_index, negative_chunk, pair_id = self.samples[index]
        result = dict(self.positive[positive_index])
        payload_path = (
            self.root
            / "trajectory_chunks"
            / f"{negative_chunk['chunk_id']}.npz"
        )
        with np.load(payload_path, allow_pickle=False) as payload:
            negative_actions = np.asarray(
                payload["actions"], dtype=np.float32
            )
        result["negative_raw_action"] = torch.from_numpy(
            negative_actions[: self.positive.action_chunk_size].copy()
        ).float()
        result["sample_key"] = (pair_id, "preference")
        return result


def infinite(loader):
    while True:
        yield from loader


def recovery_state_task_balanced_sampler(dataset: RecoveryDataset, seed: int):
    """Equal task mass, then equal state mass, independent of chunk count."""

    sample_keys = [
        (state["task"], state["failure_state_id"])
        for _, _, _, state in dataset.samples
    ]
    task_states = {}
    state_samples = Counter(sample_keys)
    for task, state_id in sample_keys:
        task_states.setdefault(task, set()).add(state_id)
    weights = [
        1.0
        / (
            len(task_states)
            * len(task_states[task])
            * state_samples[(task, state_id)]
        )
        for task, state_id in sample_keys
    ]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def compute_preference_objective(policy, batch: dict, args) -> dict:
    """Rank a successful first chunk above its same-state failed continuation."""

    positive = policy_loss_details(policy, batch, args.bf16)
    negative_batch = {
        **batch,
        "raw_action": batch["negative_raw_action"],
    }
    negative = policy_loss_details(
        policy,
        negative_batch,
        args.bf16,
        noise=positive["noise"],
        timesteps=positive["timesteps"],
        cond_mask=positive["cond_mask"],
    )
    positive_loss = front_weighted_mse(
        positive["prediction"].float(),
        positive["target"].float(),
        args.front_action_steps,
        args.front_action_weight,
    )
    negative_loss = front_weighted_mse(
        negative["prediction"].float(),
        negative["target"].float(),
        args.front_action_steps,
        args.front_action_weight,
    )
    logit = (
        positive_loss - negative_loss + args.preference_margin
    ) / args.preference_temperature
    ranking = F.softplus(logit) * args.preference_temperature
    return {
        "preference": ranking,
        "preference_positive": positive_loss,
        "preference_negative": negative_loss,
        "preference_margin_observed": negative_loss - positive_loss,
        "preference_correct": (negative_loss > positive_loss).float(),
    }


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Recovery LoRA training requires CUDA")
    device = torch.device("cuda")
    data_root = Path(args.data_dir).expanduser().resolve()
    assessment = json.loads((data_root / "dataset_assessment.json").read_text())
    if not assessment.get("training_admitted") and not args.allow_unadmitted_data:
        raise RuntimeError("Dataset assessment did not admit training; expand or repair the dataset first")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    recovery_collator = RecoveryCollator(processor, empty_ref_after_age=args.empty_ref_after_age)
    normal_collator = TransitionCollator(processor, empty_ref_after_age=args.empty_ref_after_age)
    recovery = {split: RecoveryDataset(data_root, split) for split in ("train", "validation", "test")}
    normal_root = Path(args.normal_replay_dir).expanduser().resolve()
    normal = {
        split: TransitionManifestDataset(normal_root, split, categories=("normal",))
        for split in ("train", "validation", "test")
    }
    recovery_loaders = {
        split: DataLoader(
            dataset,
            batch_size=1,
            shuffle=(split == "train" and args.recovery_sampling == "uniform"),
            sampler=(
                recovery_state_task_balanced_sampler(dataset, args.seed)
                if split == "train" and args.recovery_sampling == "state_task_balanced"
                else None
            ),
            collate_fn=recovery_collator,
        )
        for split, dataset in recovery.items()
    }
    normal_loaders = {
        split: DataLoader(dataset, batch_size=1, shuffle=(split == "train"), collate_fn=normal_collator)
        for split, dataset in normal.items()
    }

    policy = build_transition_policy(args, device)
    load_info = load_specialist_checkpoint(policy, args.specialist_path, source=args.checkpoint_source)
    policy.requires_grad_(False)
    targets = inject_lora(policy, args.lora_rank, args.lora_alpha, 0.0, list(DEFAULT_LORA_TARGETS))
    if set(targets) != set(DEFAULT_LORA_TARGETS):
        raise RuntimeError("Recovery LoRA target mismatch")
    history_adapter = install_history_adapter(policy, history_steps=4)
    history_adapter.requires_grad_(False)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    policy.eval()

    baseline_recovery = evaluate_objective(policy, recovery_loaders["validation"], device, args, args.validation_seed)
    baseline_normal = evaluate_objective(policy, normal_loaders["validation"], device, args, args.validation_seed + 1)
    baseline_loss = baseline_recovery["recovery_supervised"]
    best_loss = baseline_loss
    best_step = 0
    config = {
        "args": vars(args),
        "dataset_assessment": assessment,
        "recovery_samples": {split: len(dataset) for split, dataset in recovery.items()},
        "recovery_train_sampling": args.recovery_sampling,
        "normal_samples": {split: len(dataset) for split, dataset in normal.items()},
        "lora_targets": targets,
        "trainable_params": count_trainable_params(policy),
        "specialist_load_info": load_info,
        "stage": "positive recovery BC + matched base preservation; preference optimization deferred",
    }
    write_json(output / "training_config.json", config)
    write_json(output / "validation_baseline.json", {"recovery": baseline_recovery, "normal": baseline_normal})
    save_transition_adapter(policy, output / "adapter_best.pt", config | {"step": 0, "base_fallback": True})

    recovery_iter = infinite(recovery_loaders["train"])
    normal_iter = infinite(normal_loaders["train"])
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("")
    progress = tqdm(range(1, args.max_steps + 1), desc="train_recovery_lora_stage1")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        batch_metrics = []
        for micro in range(args.grad_accumulation_steps):
            use_normal = micro == args.grad_accumulation_steps - 1
            batch = next(normal_iter if use_normal else recovery_iter)
            batch = move_batch(batch, device)
            details = compute_preserved_objective(policy, batch, args)
            (details["loss"] / args.grad_accumulation_steps).backward()
            batch_metrics.append({key: float(value.detach().cpu()) for key, value in details.items()})
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm).detach().cpu())
        optimizer.step()
        train_metric = {
            "event": "train", "step": step, "grad_norm": grad_norm,
            "loss": float(np.mean([item["loss"] for item in batch_metrics])),
            "supervised": float(np.mean([item["supervised"] for item in batch_metrics])),
            "drift": float(np.mean([item["drift"] for item in batch_metrics])),
        }
        with metrics_path.open("a") as file:
            file.write(json.dumps(train_metric, sort_keys=True) + "\n")
        progress.set_postfix(loss=f"{train_metric['loss']:.4f}", best=f"{best_loss:.4f}")

        if step == 1 or step % args.validation_interval == 0:
            recovery_val = evaluate_objective(policy, recovery_loaders["validation"], device, args, args.validation_seed)
            normal_val = evaluate_objective(policy, normal_loaders["validation"], device, args, args.validation_seed + 1)
            recovery_loss = recovery_val["recovery_supervised"]
            improvement = baseline_loss - recovery_loss
            admitted = (
                improvement >= args.selection_min_recovery_improvement
                and normal_val["normal_drift"] <= args.max_normal_prediction_drift
                and normal_val["normal_drift_gripper"] <= args.max_normal_gripper_drift
            )
            metric = {
                "event": "validation", "step": step, "recovery": recovery_val,
                "normal": normal_val, "recovery_improvement": improvement,
                "selection_admitted": admitted,
            }
            with metrics_path.open("a") as file:
                file.write(json.dumps(metric, sort_keys=True) + "\n")
            if admitted and recovery_loss < best_loss:
                best_loss, best_step = recovery_loss, step
                save_transition_adapter(policy, output / "adapter_best.pt", config | metric)

    progress.close()
    metadata = load_transition_adapter(policy, output / "adapter_best.pt")
    test_recovery = evaluate_objective(policy, recovery_loaders["test"], device, args, args.validation_seed + 2)
    test_normal = evaluate_objective(policy, normal_loaders["test"], device, args, args.validation_seed + 3)
    checkpoints = save_merged_checkpoints(policy, Path(args.specialist_path), output)
    summary = {
        **config,
        "best_step": best_step,
        "base_fallback": best_step == 0,
        "baseline_recovery_validation": baseline_recovery,
        "best_recovery_validation_loss": best_loss,
        "recovery_validation_improvement": baseline_loss - best_loss,
        "selected_metadata": metadata,
        "test_recovery": test_recovery,
        "test_normal": test_normal,
        "outputs": checkpoints | {"adapter_best": (output / "adapter_best.pt").as_posix()},
    }
    write_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--normal_replay_dir", default=(EXPERIMENT_ROOT / "collected_transition_v1_repaired").as_posix())
    parser.add_argument("--output_dir", default=(THIS_DIR / "lora_runs" / "recovery_stage1_v1").as_posix())
    parser.add_argument("--generalist_path", default=(REPO_ROOT.parent / "models" / "generalist").as_posix())
    parser.add_argument("--specialist_path", default=(REPO_ROOT.parent / "models" / "specialist" / "Specialist+Depth+Gripper.pt").as_posix())
    parser.add_argument("--allow_unadmitted_data", action="store_true")
    parser.add_argument("--action_chunk_size", type=int, default=8, choices=[8])
    parser.add_argument("--history_steps", type=int, default=4, choices=[4])
    parser.add_argument("--empty_ref_after_age", type=int, default=8, choices=[8])
    parser.add_argument("--fast_num_inference_steps", type=int, default=10)
    parser.add_argument("--vision_encoder", default="DINO", choices=["DINO"])
    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--cond_drop_chance", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=2, choices=[2, 4, 8])
    parser.add_argument("--lora_alpha", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4, choices=[4])
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument(
        "--recovery_sampling",
        default="state_task_balanced",
        choices=["state_task_balanced", "uniform"],
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--validation_seed", type=int, default=20260718)
    parser.add_argument("--normal_supervised_weight", type=float, default=0.0)
    parser.add_argument("--transition_supervised_weight", type=float, default=1.0)
    parser.add_argument("--normal_preservation_weight", type=float, default=4.0)
    parser.add_argument("--transition_preservation_weight", type=float, default=0.2)
    parser.add_argument("--gripper_preservation_weight", type=float, default=2.0)
    parser.add_argument("--front_action_steps", type=int, default=2, choices=[2])
    parser.add_argument("--front_action_weight", type=float, default=2.0)
    parser.add_argument("--selection_min_recovery_improvement", type=float, default=2e-4)
    parser.add_argument("--max_normal_prediction_drift", type=float, default=2e-4)
    parser.add_argument("--max_normal_gripper_drift", type=float, default=1e-4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpoint_source", default="ema_model", choices=["ema_model", "online_model", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.validation_interval <= 0:
        parser.error("training step counts must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
