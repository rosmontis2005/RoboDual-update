#!/usr/bin/env python3
"""Train a narrow, base-preserving transition LoRA for the fast specialist.

Unlike the earlier 14-layer experiment, this variant only adapts the final two
temporal attention output projections.  It distils the frozen base prediction
on every sample and uses action-label supervision only for transition samples.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
OLD_LORA_ROOT = REPO_ROOT / "LoRA_trial"
DEFAULT_DATA_DIR = EXPERIMENT_ROOT / "collected_transition_v1_repaired"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "lora_runs" / "transition_lora_v4_preserved"
DEFAULT_GENERALIST_PATH = REPO_ROOT.parent / "models" / "generalist"
DEFAULT_SPECIALIST_PATH = REPO_ROOT.parent / "models" / "specialist" / "Specialist+Depth+Gripper.pt"

for path in (REPO_ROOT, REPO_ROOT.parent, OLD_LORA_ROOT, EXPERIMENT_ROOT):
    value = path.as_posix()
    if value not in sys.path:
        sys.path.insert(0, value)

from history_adapter import install_history_adapter  # noqa: E402
from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy  # noqa: E402
from train_lora_specialist import (  # noqa: E402
    adapter_state_dict,
    count_trainable_params,
    inject_lora,
    load_specialist_checkpoint,
    merge_lora_modules,
    set_frozen_feature_extractors_eval,
)


CATEGORIES = ("normal", "refresh", "high_conflict", "stale")
DEFAULT_LORA_TARGETS = (
    "model.blocks.4.attn_temporal.proj",
    "model.blocks.5.attn_temporal.proj",
)


def build_transition_policy(args, device: torch.device) -> DiffusionDiTImagePolicy:
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    return DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}},
        noise_scheduler=scheduler,
        n_action_steps=args.action_chunk_size,
        num_inference_steps=args.fast_num_inference_steps,
        vision_encoder=args.vision_encoder,
        vision_encoder_pretrained=False,
        with_depth=args.with_depth,
        with_gripper=args.with_gripper,
        with_tactile=args.with_tactile,
        cond_drop_chance=args.cond_drop_chance,
        progressive_noise=False,
    ).to(device)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def frame_path(root: Path, trajectory_id: str, step: int) -> Path:
    return root / "trajectories" / trajectory_id / f"step_{step:04d}.npz"


class TransitionManifestDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str, categories: Sequence[str] = CATEGORIES):
        self.root = Path(data_dir).expanduser().resolve()
        self.split = split
        self.categories = set(categories)
        self.trajectory_info = {
            item["trajectory_id"]: item for item in read_jsonl(self.root / "trajectories.jsonl")
        }
        all_samples = read_jsonl(self.root / "samples.jsonl")
        self.samples = [
            sample for sample in all_samples
            if sample["split"] == split and sample["category"] in self.categories
        ]
        if not self.samples:
            raise ValueError(f"No samples for split={split!r}, categories={sorted(self.categories)}")
        self.depth_min = 3.5
        self.depth_max = 6.2
        self.gripper_depth_min = 0.0
        self.gripper_depth_max = 2.0
        self._condition_cache: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self.condition_cache_size = 256
        self._action_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.action_cache_size = 32
        summary = json.loads((self.root / "collection_summary.json").read_text())
        if summary.get("target_action_source") != "next_frame_hist_action_before[-1]":
            raise ValueError(
                "Dataset has no verified repaired committed-action targets; "
                "run repair_transition_dataset.py and train from its output directory"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, trajectory_id: str, step: int) -> dict[str, np.ndarray]:
        path = frame_path(self.root, trajectory_id, step)
        with np.load(path, allow_pickle=False) as frame:
            return {key: frame[key] for key in frame.files}

    def _load_condition(self, trajectory_id: str, condition_id: int) -> dict:
        key = (trajectory_id, int(condition_id))
        if key not in self._condition_cache:
            path = self.root / "conditions" / trajectory_id / f"condition_{condition_id:03d}.pt"
            self._condition_cache[key] = torch.load(path, map_location="cpu", weights_only=False)
            if len(self._condition_cache) > self.condition_cache_size:
                self._condition_cache.popitem(last=False)
        else:
            self._condition_cache.move_to_end(key)
        return self._condition_cache[key]

    def _load_actions(self, trajectory_id: str) -> np.ndarray:
        if trajectory_id not in self._action_cache:
            path = self.root / "committed_actions" / f"{trajectory_id}.npy"
            actions = np.load(path, allow_pickle=False, mmap_mode="r")
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"Bad repaired action array in {path}: {actions.shape}")
            self._action_cache[trajectory_id] = actions
            if len(self._action_cache) > self.action_cache_size:
                self._action_cache.popitem(last=False)
        else:
            self._action_cache.move_to_end(trajectory_id)
        return self._action_cache[trajectory_id]

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        trajectory_id = sample["trajectory_id"]
        step = int(sample["step"])
        action_chunk_size = int(sample["action_chunk_size"])
        current = self._load_frame(trajectory_id, step)
        previous = self._load_frame(trajectory_id, step - 1)
        repaired_actions = self._load_actions(trajectory_id)
        if step + action_chunk_size > len(repaired_actions):
            raise ValueError(f"Sample target exceeds repaired actions: {trajectory_id}:{step}")
        future = np.asarray(repaired_actions[step : step + action_chunk_size], dtype=np.float32).copy()
        condition = self._load_condition(trajectory_id, int(sample["condition_id"]))

        robot_obs = np.asarray(current["robot_obs"], dtype=np.float32)
        proprio = np.concatenate([robot_obs[:6], robot_obs[-1:]], axis=0)
        depth = (np.asarray(current["depth_static"], dtype=np.float32) - self.depth_min) / (
            self.depth_max - self.depth_min
        )
        gripper_depth = (
            np.asarray(current["depth_gripper"], dtype=np.float32) - self.gripper_depth_min
        ) / (self.gripper_depth_max - self.gripper_depth_min)
        return {
            "sample_id": int(sample["sample_id"]),
            "sample_key": (trajectory_id, step),
            "trajectory_id": trajectory_id,
            "task": self.trajectory_info[trajectory_id]["task"],
            "category": sample["category"],
            "instruction": self.trajectory_info[trajectory_id]["instruction"],
            "slow_age": int(sample["slow_age"]),
            "current_image": Image.fromarray(np.asarray(current["rgb_static"], dtype=np.uint8)),
            "previous_image": Image.fromarray(np.asarray(previous["rgb_static"], dtype=np.uint8)),
            "gripper_image": Image.fromarray(np.asarray(current["rgb_gripper"], dtype=np.uint8)),
            "depth_image": torch.from_numpy(depth).float(),
            "depth_gripper": torch.from_numpy(gripper_depth).float(),
            "raw_action": torch.from_numpy(future).float(),
            "hist_action": torch.from_numpy(np.asarray(current["hist_action_before"], dtype=np.float32)).float(),
            "proprio": torch.from_numpy(proprio).float(),
            "slow_action": torch.as_tensor(condition["slow_action"], dtype=torch.float32).squeeze(0),
            "slow_hidden": torch.as_tensor(condition["slow_hidden"], dtype=torch.float32).squeeze(0),
        }


class TransitionCollator:
    def __init__(self, processor, empty_ref_after_age: int = 8):
        self.processor = processor
        self.empty_ref_after_age = int(empty_ref_after_age)

    def _image(self, image: Image.Image) -> torch.Tensor:
        return self.processor.image_processor.apply_transform(image)[:3]

    def _ref_action(self, slow_action: torch.Tensor, age: int) -> torch.Tensor:
        ref = torch.zeros_like(slow_action)
        num_cond_actions = 0 if age >= self.empty_ref_after_age else max(0, slow_action.shape[0] - age)
        if num_cond_actions > 0:
            # Match DualSystemCalvinEvaluation._build_ref_actions_from exactly.
            ref[:num_cond_actions] = slow_action[-num_cond_actions:]
        return ref

    def __call__(self, instances: Sequence[dict]) -> dict:
        if len(instances) != 1:
            raise ValueError(
                "Transition LoRA requires batch_size=1 because persisted slow_hidden "
                "has variable token length and the specialist has no context padding mask"
            )
        return {
            "sample_id": torch.tensor([item["sample_id"] for item in instances], dtype=torch.long),
            "sample_key": [item["sample_key"] for item in instances],
            "task": [item["task"] for item in instances],
            "category": [item["category"] for item in instances],
            "lang": [item["instruction"] for item in instances],
            "pixel_values_dp": torch.stack([self._image(item["current_image"]) for item in instances]),
            "prev_pixel_values_dp": torch.stack([self._image(item["previous_image"]) for item in instances]),
            "gripper_image": torch.stack([self._image(item["gripper_image"]) for item in instances]),
            "depth_image": torch.stack([item["depth_image"] for item in instances]),
            "depth_gripper": torch.stack([item["depth_gripper"] for item in instances]),
            "raw_action": torch.stack([item["raw_action"] for item in instances]),
            "hist_action": torch.stack([item["hist_action"] for item in instances]),
            "proprio": torch.stack([item["proprio"] for item in instances]),
            "ref_action": torch.stack([
                self._ref_action(item["slow_action"], item["slow_age"]) for item in instances
            ]),
            "action_cond": instances[0]["slow_hidden"].unsqueeze(0),
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return result


@contextmanager
def base_policy_mode(policy):
    """Temporarily disable every trainable residual without changing weights."""

    lora_scalings = []
    for module in policy.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "scaling"):
            lora_scalings.append((module, module.scaling))
            module.scaling = 0.0
    adapter = policy.model.history_adapter
    gate = adapter.history_gate.detach().clone()
    adapter.history_gate.data.zero_()
    try:
        yield
    finally:
        adapter.history_gate.data.copy_(gate)
        for module, scaling in lora_scalings:
            module.scaling = scaling


def policy_loss_details(
    policy,
    batch: dict,
    bf16: bool,
    *,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    cond_mask: torch.Tensor | None = None,
) -> dict:
    device = batch["raw_action"].device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        return policy.compute_loss(
            trajectory=batch["raw_action"].float(),
            ref_action=batch["ref_action"].float(),
            action_cond=batch["action_cond"].float(),
            obs=(batch["pixel_values_dp"].float(), batch["prev_pixel_values_dp"].float()),
            depth_obs=batch["depth_image"].float(),
            gripper_obs=(batch["gripper_image"].float(), batch["depth_gripper"].float()),
            tactile_obs=None,
            lang=batch["lang"],
            proprio=batch["proprio"].float(),
            hist_action=batch["hist_action"].float(),
            decoupled_loss=False,
            noise=noise,
            timesteps=timesteps,
            cond_mask=cond_mask,
            return_details=True,
        )


def compute_preserved_objective(policy, batch: dict, args, *, need_teacher: bool = True) -> dict:
    """Compute supervised transition loss plus matched frozen-base distillation."""

    if policy.training:
        raise RuntimeError(
            "Base-preserving distillation requires policy.eval() so teacher and student "
            "do not receive different internal attention-dropout masks"
        )
    student = policy_loss_details(policy, batch, args.bf16)
    if need_teacher:
        with torch.no_grad(), base_policy_mode(policy):
            teacher = policy_loss_details(
                policy,
                batch,
                args.bf16,
                noise=student["noise"],
                timesteps=student["timesteps"],
                cond_mask=student["cond_mask"],
            )["prediction"].detach()
    else:
        teacher = student["prediction"].detach()

    prediction = student["prediction"].float()
    teacher = teacher.float()
    drift_ee = F.mse_loss(prediction[..., :6], teacher[..., :6])
    drift_gripper = F.mse_loss(prediction[..., 6], teacher[..., 6])
    drift = drift_ee + args.gripper_preservation_weight * drift_gripper
    is_normal = batch["category"][0] == "normal"
    supervised_weight = args.normal_supervised_weight if is_normal else args.transition_supervised_weight
    preservation_weight = (
        args.normal_preservation_weight if is_normal else args.transition_preservation_weight
    )
    objective = supervised_weight * student["loss"] + preservation_weight * drift
    return {
        "loss": objective,
        "supervised": student["loss"],
        "drift": drift,
        "drift_ee": drift_ee,
        "drift_gripper": drift_gripper,
    }


def deterministic_validation_subset(dataset: TransitionManifestDataset, per_category: int, seed: int):
    rng = random.Random(seed)
    indices_by_category = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        indices_by_category[sample["category"]].append(index)
    indices = []
    for category in CATEGORIES:
        pool = indices_by_category[category]
        if not pool:
            raise ValueError(f"Validation split has no {category} samples")
        indices.extend(rng.sample(pool, min(per_category, len(pool))))
    return torch.utils.data.Subset(dataset, indices)


def evaluate_objective(policy, dataloader, device: torch.device, args, seed: int) -> dict:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = policy.training
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    policy.eval()
    totals = Counter()
    sums = Counter()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            details = compute_preserved_objective(policy, batch, args)
            values = {key: float(value.detach().cpu()) for key, value in details.items()}
            batch_size = len(batch["category"])
            category_counts = Counter(batch["category"])
            if len(category_counts) != 1:
                raise RuntimeError(
                    "Validation batches must contain one category so batch-mean loss can be attributed correctly"
                )
            category = next(iter(category_counts))
            for metric, value in values.items():
                for scope in ("overall", category):
                    key = f"{scope}_{metric}"
                    sums[key] += value * batch_size
                    totals[key] += batch_size
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    if was_training:
        policy.train()
        set_frozen_feature_extractors_eval(policy)
    return {key: sums[key] / totals[key] for key in sorted(totals)}


def history_stats(policy, hist_action: torch.Tensor) -> dict:
    adapter = policy.model.history_adapter
    with torch.no_grad():
        feature = adapter(hist_action.to(device=next(adapter.parameters()).device, dtype=next(adapter.parameters()).dtype))
    return {
        "history_output_norm": float(torch.linalg.vector_norm(feature.float(), dim=-1).mean().cpu()),
        "history_gate": float(adapter.history_gate.detach().float().cpu()),
    }


def transition_supervised_loss(metrics: dict) -> float:
    return float(np.mean([metrics[f"{category}_supervised"] for category in CATEGORIES if category != "normal"]))


def preservation_constraints(metrics: dict, args) -> tuple[bool, dict]:
    checks = {
        "normal_drift": metrics["normal_drift"] <= args.max_normal_prediction_drift,
        "overall_drift": metrics["overall_drift"] <= args.max_overall_prediction_drift,
        "normal_gripper_drift": (
            metrics["normal_drift_gripper"] <= args.max_normal_gripper_drift
        ),
    }
    return all(checks.values()), checks


def should_stop_early(optimizer_step: int, stale_evaluations: int, args) -> bool:
    return (
        optimizer_step >= args.min_steps_before_early_stopping
        and stale_evaluations >= args.early_stopping_patience
    )


def save_transition_adapter(policy, path: Path, metadata: dict) -> None:
    payload = {
        "format": "robodual_transition_history_lora_v1",
        "metadata": metadata,
        "lora_state": adapter_state_dict(policy),
        "history_adapter_state": OrderedDict(
            (key, value.detach().cpu()) for key, value in policy.model.history_adapter.state_dict().items()
        ),
    }
    torch.save(payload, path)


def load_transition_adapter(policy, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "robodual_transition_history_lora_v1":
        raise ValueError(f"Unsupported transition adapter: {path}")
    lora_state = payload["lora_state"]
    current_lora = adapter_state_dict(policy)
    if set(lora_state) != set(current_lora):
        raise ValueError("Saved LoRA keys do not match the installed target modules")
    modules = dict(policy.named_modules())
    for key, value in lora_state.items():
        module_name, parameter_name = key.rsplit(".", 1)
        getattr(modules[module_name], parameter_name).data.copy_(value.to(
            device=getattr(modules[module_name], parameter_name).device,
            dtype=getattr(modules[module_name], parameter_name).dtype,
        ))
    policy.model.history_adapter.load_state_dict(payload["history_adapter_state"])
    return payload.get("metadata", {})


def save_merged_checkpoints(policy, base_path: Path, output_dir: Path) -> dict:
    merge_lora_modules(policy)
    policy.cpu()
    merged_state = OrderedDict((key, value.detach().cpu()) for key, value in policy.state_dict().items())
    raw_path = output_dir / "specialist_transition_lora_merged_policy.pt"
    torch.save(merged_state, raw_path)

    base_state = torch.load(base_path, map_location="cpu")
    ema_path = output_dir / "specialist_transition_lora_merged_ema.pt"
    if isinstance(base_state, dict) and any(str(key).startswith("ema_model.") for key in base_state):
        compatible = OrderedDict(base_state)
        for prefix in ("ema_model", "online_model"):
            for key, value in merged_state.items():
                full_key = f"{prefix}.{key}"
                if full_key in compatible or key.startswith("model.history_adapter."):
                    compatible[full_key] = value
        torch.save(compatible, ema_path)
        checkpoint_format = "ema_compatible_with_history_adapter"
    else:
        torch.save(merged_state, ema_path)
        checkpoint_format = "raw_policy"
    return {"merged_policy": raw_path.as_posix(), "merged_ema": ema_path.as_posix(), "format": checkpoint_format}


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"{output_dir} is not empty; use --overwrite_output only for a fresh rerun")
    if output_dir.exists() and any(output_dir.iterdir()) and args.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")

    data_summary = json.loads((Path(args.data_dir) / "collection_summary.json").read_text())
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    collator = TransitionCollator(processor, empty_ref_after_age=args.empty_ref_after_age)
    train_dataset = TransitionManifestDataset(args.data_dir, "train")
    validation_dataset = TransitionManifestDataset(args.data_dir, "validation")
    test_dataset = TransitionManifestDataset(args.data_dir, "test")
    validation_subset = deterministic_validation_subset(
        validation_dataset, args.validation_samples_per_category, args.validation_seed
    )
    test_subset = deterministic_validation_subset(
        test_dataset, args.validation_samples_per_category, args.validation_seed + 1
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_subset, batch_size=args.validation_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collator,
    )
    test_loader = DataLoader(
        test_subset, batch_size=args.validation_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collator,
    )

    policy = build_transition_policy(args, device)
    load_info = load_specialist_checkpoint(policy, args.specialist_path, source=args.checkpoint_source)
    policy.requires_grad_(False)
    requested_targets = list(DEFAULT_LORA_TARGETS)
    lora_targets = inject_lora(policy, args.lora_rank, args.lora_alpha, args.lora_dropout, requested_targets)
    if set(lora_targets) != set(requested_targets):
        raise RuntimeError(
            f"LoRA target mismatch; missing={sorted(set(requested_targets)-set(lora_targets))}, "
            f"unexpected={sorted(set(lora_targets)-set(requested_targets))}"
        )
    history_adapter = install_history_adapter(policy, history_steps=args.history_steps)
    # The ablation found no stable benefit from the history residual and a
    # negative interaction with LoRA. Keep a zero adapter only for checkpoint
    # compatibility with the transition evaluator.
    history_adapter.requires_grad_(False)
    if (
        torch.count_nonzero(history_adapter.net[-1].weight).item()
        or torch.count_nonzero(history_adapter.net[-1].bias).item()
    ):
        raise RuntimeError("The compatibility history adapter must start at exact zero output")
    trainable_params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    forbidden = ("visual_adapter", "depth_adapter", "gripper_", "proprio_embedder", ".mlp.", ".cross_attn.", "final_layer")
    if any(any(token in name for token in forbidden) for name in trainable_names):
        raise RuntimeError(f"Forbidden trainable parameter detected: {trainable_names}")

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "data_status": data_summary.get("status"),
        "data_missing_groups": data_summary.get("missing_groups", {}),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "validation_subset_samples": len(validation_subset),
        "test_samples": len(test_dataset),
        "test_subset_samples": len(test_subset),
        "train_category_counts": dict(Counter(sample["category"] for sample in train_dataset.samples)),
        "validation_category_counts": dict(Counter(sample["category"] for sample in validation_dataset.samples)),
        "lora_targets": lora_targets,
        "trainable_parameter_names": trainable_names,
        "trainable_params": count_trainable_params(policy),
        "specialist_load_info": load_info,
    }
    write_json(output_dir / "training_config.json", config)

    baseline_validation = evaluate_objective(
        policy, validation_loader, device, args, args.validation_seed
    )
    write_json(output_dir / "validation_baseline.json", baseline_validation)
    best_validation = transition_supervised_loss(baseline_validation)
    best_step = 0
    best_unconstrained_validation = float("inf")
    best_unconstrained_step = 0
    stale_evaluations = 0
    optimizer_step = 0
    micro_step = 0
    accumulated_loss = 0.0
    recent_losses = []
    optimizer.zero_grad(set_to_none=True)
    save_transition_adapter(
        policy,
        output_dir / "adapter_best.pt",
        config | {"step": 0, "validation": baseline_validation, "base_fallback": True},
    )
    # Gradients work in eval mode. Keeping the complete policy deterministic is
    # required because teacher and student are separate forwards.
    policy.eval()
    progress = tqdm(total=args.max_steps, desc="train_transition_lora")

    while optimizer_step < args.max_steps:
        for batch in train_loader:
            batch = move_batch(batch, device)
            loss_details = compute_preserved_objective(policy, batch, args)
            loss = loss_details["loss"]
            (loss / args.grad_accumulation_steps).backward()
            micro_step += 1
            loss_value = float(loss.detach().cpu())
            accumulated_loss += loss_value
            recent_losses.append(loss_value)
            recent_losses = recent_losses[-100:]
            if micro_step % args.grad_accumulation_steps:
                continue

            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm).detach().cpu())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            effective_batch_loss = accumulated_loss / args.grad_accumulation_steps
            accumulated_loss = 0.0
            hist_stats = history_stats(policy, batch["hist_action"][: min(8, batch["hist_action"].shape[0])])
            metric = {
                "event": "train",
                "step": optimizer_step,
                "loss": effective_batch_loss,
                "loss_ma100": float(np.mean(recent_losses)),
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "category": batch["category"][0],
                "supervised_loss": float(loss_details["supervised"].detach().cpu()),
                "prediction_drift": float(loss_details["drift"].detach().cpu()),
                "prediction_drift_ee": float(loss_details["drift_ee"].detach().cpu()),
                "prediction_drift_gripper": float(loss_details["drift_gripper"].detach().cpu()),
                **hist_stats,
            }
            with metrics_path.open("a") as file:
                file.write(json.dumps(metric, sort_keys=True) + "\n")
            progress.update(1)
            progress.set_postfix(loss=f"{effective_batch_loss:.4f}", best=f"{best_validation:.4f}")

            should_validate = optimizer_step == 1 or optimizer_step % args.validation_interval == 0
            if should_validate:
                validation = evaluate_objective(
                    policy, validation_loader, device, args, args.validation_seed
                )
                transition_loss = transition_supervised_loss(validation)
                constraints_met, constraint_checks = preservation_constraints(validation, args)
                val_metric = {
                    "event": "validation", "step": optimizer_step,
                    **validation,
                    "transition_supervised_loss": transition_loss,
                    "preservation_constraints_met": constraints_met,
                    "preservation_checks": constraint_checks,
                }
                with metrics_path.open("a") as file:
                    file.write(json.dumps(val_metric, sort_keys=True) + "\n")
                improved_unconstrained = (
                    transition_loss < best_unconstrained_validation - args.early_stopping_min_delta
                )
                if improved_unconstrained:
                    best_unconstrained_validation = transition_loss
                    best_unconstrained_step = optimizer_step
                    save_transition_adapter(
                        policy, output_dir / "adapter_best_unconstrained.pt",
                        config | {"step": optimizer_step, "validation": validation},
                    )
                improved = (
                    constraints_met
                    and transition_loss < best_validation - args.early_stopping_min_delta
                )
                if improved:
                    best_validation = transition_loss
                    best_step = optimizer_step
                    stale_evaluations = 0
                    save_transition_adapter(
                        policy, output_dir / "adapter_best.pt",
                        config | {"step": optimizer_step, "validation": validation},
                    )
                elif optimizer_step >= args.min_steps_before_early_stopping:
                    stale_evaluations += 1
                else:
                    stale_evaluations = 0
                if not constraints_met:
                    print(
                        f"[warning] preservation constraint failed: {constraint_checks}", flush=True,
                    )
                if should_stop_early(optimizer_step, stale_evaluations, args):
                    break

            if optimizer_step % args.save_adapter_steps == 0:
                save_transition_adapter(
                    policy, output_dir / f"adapter_step_{optimizer_step}.pt",
                    config | {"step": optimizer_step},
                )
            if optimizer_step >= args.max_steps:
                break
        if optimizer_step >= args.max_steps or should_stop_early(optimizer_step, stale_evaluations, args):
            break

    progress.close()
    save_transition_adapter(policy, output_dir / "adapter_final.pt", config | {"step": optimizer_step})
    constrained_best = output_dir / "adapter_best.pt"
    selected_best_path = constrained_best if constrained_best.exists() else output_dir / "adapter_best_unconstrained.pt"
    best_metadata = load_transition_adapter(policy, selected_best_path)
    best_test = evaluate_objective(policy, test_loader, device, args, args.validation_seed + 1)
    checkpoint_outputs = save_merged_checkpoints(
        policy, Path(args.specialist_path).expanduser().resolve(), output_dir
    )
    summary = {
        **config,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "stopped_early": should_stop_early(optimizer_step, stale_evaluations, args),
        "best_step": best_step,
        "best_validation_loss": best_validation,
        "best_validation_metric": "mean transition-category supervised diffusion loss",
        "best_unconstrained_step": best_unconstrained_step,
        "best_unconstrained_validation_loss": best_unconstrained_validation,
        "selected_best_adapter": selected_best_path.as_posix(),
        "selected_best_meets_preservation_constraints": preservation_constraints(
            best_metadata.get("validation", baseline_validation), args
        )[0],
        "baseline_validation": baseline_validation,
        "best_test": best_test,
        "merged_from_adapter_step": best_metadata.get("step"),
        "final_loss_ma100": float(np.mean(recent_losses)) if recent_losses else None,
        "outputs": {
            "adapter_best": (output_dir / "adapter_best.pt").as_posix(),
            "adapter_best_unconstrained": (output_dir / "adapter_best_unconstrained.pt").as_posix(),
            "adapter_final": (output_dir / "adapter_final.pt").as_posix(),
            "metrics": metrics_path.as_posix(),
            **checkpoint_outputs,
        },
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist_path", default=DEFAULT_GENERALIST_PATH.as_posix())
    parser.add_argument("--specialist_path", default=DEFAULT_SPECIALIST_PATH.as_posix())
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR.as_posix())
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--action_chunk_size", type=int, default=8, choices=[8])
    parser.add_argument("--history_steps", type=int, default=4, choices=[4])
    parser.add_argument("--empty_ref_after_age", type=int, default=8)
    parser.add_argument("--fast_num_inference_steps", type=int, default=10)
    parser.add_argument("--vision_encoder", default="DINO", choices=["DINO"])
    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--cond_drop_chance", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--lora_alpha", type=float, default=2.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1, choices=[1])
    parser.add_argument("--grad_accumulation_steps", type=int, default=2, choices=[2])
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--validation_samples_per_category", type=int, default=64)
    parser.add_argument("--validation_batch_size", type=int, default=1, choices=[1])
    parser.add_argument("--validation_seed", type=int, default=20260711)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-6)
    parser.add_argument("--min_steps_before_early_stopping", type=int, default=2000)
    parser.add_argument("--normal_supervised_weight", type=float, default=0.0)
    parser.add_argument("--transition_supervised_weight", type=float, default=1.0)
    parser.add_argument("--normal_preservation_weight", type=float, default=2.0)
    parser.add_argument("--transition_preservation_weight", type=float, default=0.5)
    parser.add_argument("--gripper_preservation_weight", type=float, default=2.0)
    parser.add_argument("--max_normal_prediction_drift", type=float, default=2e-4)
    parser.add_argument("--max_overall_prediction_drift", type=float, default=5e-4)
    parser.add_argument("--max_normal_gripper_drift", type=float, default=1e-4)
    parser.add_argument("--save_adapter_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpoint_source", default="ema_model", choices=["ema_model", "online_model", "auto"])
    args = parser.parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.grad_accumulation_steps <= 0:
        parser.error("step and batch parameters must be positive")
    if not 0 < args.min_steps_before_early_stopping <= args.max_steps:
        parser.error("--min_steps_before_early_stopping must be in [1, --max_steps]")
    if args.batch_size != 1 or args.validation_batch_size != 1:
        parser.error("variable-length slow_hidden requires train and validation batch size 1")
    if args.grad_accumulation_steps != 2:
        parser.error("this experiment fixes gradient accumulation at 2 (effective batch size 2)")
    if args.validation_samples_per_category % args.validation_batch_size:
        parser.error("--validation_samples_per_category must be divisible by --validation_batch_size")
    positive = (
        "transition_supervised_weight", "normal_preservation_weight",
        "transition_preservation_weight", "gripper_preservation_weight",
        "max_normal_prediction_drift", "max_overall_prediction_drift",
        "max_normal_gripper_drift",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error(f"these arguments must be positive: {', '.join(positive)}")
    if args.normal_supervised_weight < 0:
        parser.error("--normal_supervised_weight must be non-negative")
    return args


if __name__ == "__main__":
    train(parse_args())
