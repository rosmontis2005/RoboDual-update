#!/usr/bin/env python3
"""Minimal specialist LoRA training for collected RoboDual rollout data.

The script consumes the CALVIN-style output from ``collect_lora_rollouts.py``.
It freezes the generalist and the base specialist, inserts small LoRA adapters
into selected specialist linear layers, and trains only those adapters on
stale/empty-reference samples.
"""

import argparse
import copy
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
DEFAULT_GENERALIST_PATH = REPO_ROOT.parent / "models" / "generalist"
DEFAULT_SPECIALIST_PATH = REPO_ROOT.parent / "models" / "specialist" / "Specialist+Depth+Gripper.pt"
DEFAULT_DATA_DIR = REPO_ROOT / "LoRA_trial" / "collected_lora_rollouts_task_age_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "LoRA_trial" / "lora_runs" / "specialist_empty_ref_lora_v1"

for dependency_path in (REPO_ROOT, REPO_ROOT.parent):
    path_str = dependency_path.as_posix()
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy  # noqa: E402


def parse_csv(value: str | None) -> List[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_csv(value: str | None) -> List[int]:
    return [int(item) for item in parse_csv(value)]


def get_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def resolve_training_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    if (data_dir / "training").is_dir():
        return data_dir / "training"
    if data_dir.name == "training" and data_dir.is_dir():
        return data_dir
    raise FileNotFoundError(f"Could not find training directory under {data_dir}")


def load_lang_data(training_dir: Path) -> Dict:
    candidates = [
        training_dir / "lang_annotations" / "auto_lang_ann.npy",
        training_dir / "auto_lang_ann.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path, allow_pickle=True).item()
    raise FileNotFoundError(f"Could not find auto_lang_ann.npy under {training_dir}")


def episode_path(training_dir: Path, frame_idx: int) -> Path:
    return training_dir / f"episode_{frame_idx:07d}.npz"


def load_frame(training_dir: Path, frame_idx: int) -> Dict[str, np.ndarray]:
    path = episode_path(training_dir, frame_idx)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path.as_posix()) as data:
        return {key: data[key] for key in data.files}


@dataclass(frozen=True)
class StaleSample:
    rollout_i: int
    task: str
    instruction: str
    slow_idx: int
    current_idx: int
    stale_age: int


class StaleRefRolloutDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        stale_ages: Sequence[int],
        action_chunk_size: int,
        tasks: Sequence[str] | None = None,
        samples_per_rollout_per_age: int = 4,
        sampling_stride: int = 1,
        seed: int = 42,
    ):
        self.training_dir = resolve_training_dir(data_dir)
        self.stale_ages = sorted(set(int(age) for age in stale_ages))
        self.action_chunk_size = int(action_chunk_size)
        self.samples_per_rollout_per_age = int(samples_per_rollout_per_age)
        self.sampling_stride = max(1, int(sampling_stride))
        self.task_filter = set(tasks or [])
        self.lang_data = load_lang_data(self.training_dir)

        self.depth_max = 6.2
        self.depth_min = 3.5
        self.gripper_depth_max = 2.0
        self.gripper_depth_min = 0.0

        self.samples = self._build_samples(seed=seed)
        if not self.samples:
            raise ValueError(
                "No trainable stale-ref samples found. Check rollout length, tasks, and stale_ages."
            )

    def _build_samples(self, seed: int) -> List[StaleSample]:
        rng = random.Random(seed)
        indices = self.lang_data["info"]["indx"]
        anns = self.lang_data["language"]["ann"]
        tasks = self.lang_data["language"]["task"]

        samples: List[StaleSample] = []
        for rollout_i, (bounds, instruction, task) in enumerate(zip(indices, anns, tasks)):
            task = str(task)
            if self.task_filter and task not in self.task_filter:
                continue
            start_idx, end_idx = int(bounds[0]), int(bounds[1])
            rollout_len = end_idx - start_idx + 1
            for age in self.stale_ages:
                max_offset = rollout_len - age - self.action_chunk_size
                if max_offset < 0:
                    continue
                offsets = list(range(0, max_offset + 1, self.sampling_stride))
                if self.samples_per_rollout_per_age > 0 and len(offsets) > self.samples_per_rollout_per_age:
                    offsets = sorted(rng.sample(offsets, self.samples_per_rollout_per_age))
                for offset in offsets:
                    slow_idx = start_idx + offset
                    current_idx = slow_idx + age
                    samples.append(
                        StaleSample(
                            rollout_i=rollout_i,
                            task=task,
                            instruction=str(instruction),
                            slow_idx=slow_idx,
                            current_idx=current_idx,
                            stale_age=age,
                        )
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _rgb_image(frame: Dict[str, np.ndarray], key: str) -> Image.Image:
        return Image.fromarray(np.asarray(frame[key]).astype(np.uint8))

    def _target_actions(self, current_idx: int) -> torch.Tensor:
        actions = []
        for frame_idx in range(current_idx, current_idx + self.action_chunk_size):
            frame = load_frame(self.training_dir, frame_idx)
            actions.append(np.asarray(frame["rel_actions"], dtype=np.float32))
        return torch.from_numpy(np.stack(actions, axis=0)).float()

    def _hist_actions(self, current_idx: int, lower_bound_idx: int) -> torch.Tensor:
        hist = torch.zeros((4, 7), dtype=torch.float32)
        start = max(lower_bound_idx, current_idx - 4)
        frame_indices = list(range(start, current_idx))
        if not frame_indices:
            return hist
        actions = []
        for frame_idx in frame_indices:
            frame = load_frame(self.training_dir, frame_idx)
            actions.append(np.asarray(frame["rel_actions"], dtype=np.float32))
        stacked = torch.from_numpy(np.stack(actions, axis=0)).float()
        hist[-stacked.shape[0] :] = stacked[-4:]
        return hist

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        slow_frame = load_frame(self.training_dir, sample.slow_idx)
        current_frame = load_frame(self.training_dir, sample.current_idx)
        prev_idx = max(sample.slow_idx, sample.current_idx - 1)
        prev_frame = load_frame(self.training_dir, prev_idx)

        robot_obs = np.asarray(current_frame["robot_obs"], dtype=np.float32)
        proprio = torch.from_numpy(np.concatenate([robot_obs[:6], robot_obs[-1:]], axis=0)).float()

        depth_static = (np.asarray(current_frame["depth_static"], dtype=np.float32) - self.depth_min) / (
            self.depth_max - self.depth_min
        )
        depth_gripper = (np.asarray(current_frame["depth_gripper"], dtype=np.float32) - self.gripper_depth_min) / (
            self.gripper_depth_max - self.gripper_depth_min
        )

        rgb_tactile = np.asarray(
            current_frame.get("rgb_tactile", np.zeros((160, 120, 6), dtype=np.uint8)),
            dtype=np.float32,
        )
        if rgb_tactile.max() > 1.0:
            rgb_tactile = rgb_tactile / 255.0

        return {
            "sample_key": (
                sample.rollout_i,
                sample.slow_idx,
                sample.current_idx,
                sample.stale_age,
            ),
            "task": sample.task,
            "instruction": sample.instruction,
            "stale_age": sample.stale_age,
            "slow_image": self._rgb_image(slow_frame, "rgb_static"),
            "current_image": self._rgb_image(current_frame, "rgb_static"),
            "prev_image": self._rgb_image(prev_frame, "rgb_static"),
            "gripper_image": self._rgb_image(current_frame, "rgb_gripper"),
            "depth_image": torch.from_numpy(depth_static).float(),
            "depth_gripper": torch.from_numpy(depth_gripper).float(),
            "tactile_image": torch.from_numpy(rgb_tactile).permute(2, 0, 1).float(),
            "raw_action": self._target_actions(sample.current_idx),
            "hist_action": self._hist_actions(sample.current_idx, sample.slow_idx),
            "proprio": proprio,
        }


class StaleRefCollator:
    def __init__(self, processor, vision_encoder: str = "DINO"):
        self.processor = processor
        self.vision_encoder = vision_encoder
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def _process_slow_inputs(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        encoded = []
        for instance in instances:
            prompt = get_openvla_prompt(instance["instruction"])
            encoded.append(self.processor(prompt, instance["slow_image"], return_tensors="pt"))

        input_ids = [item["input_ids"].squeeze(0).long() for item in encoded]
        pixel_values = [item["pixel_values"].squeeze(0) for item in encoded]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "input_lengths": torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long),
            "pixel_values": torch.stack(pixel_values),
        }

    def _dp_image(self, image: Image.Image) -> torch.Tensor:
        tensor = self.processor.image_processor.apply_transform(image)
        if self.vision_encoder == "DINO":
            return tensor[:3]
        return tensor[-3:]

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        return {
            "slow_inputs": self._process_slow_inputs(instances),
            "pixel_values_dp": torch.stack([self._dp_image(item["current_image"]) for item in instances]),
            "prev_pixel_values_dp": torch.stack([self._dp_image(item["prev_image"]) for item in instances]),
            "gripper_image": torch.stack([self._dp_image(item["gripper_image"]) for item in instances]),
            "depth_image": torch.stack([item["depth_image"] for item in instances]),
            "depth_gripper": torch.stack([item["depth_gripper"] for item in instances]),
            "tactile_image": torch.stack([item["tactile_image"] for item in instances]),
            "raw_action": torch.stack([item["raw_action"] for item in instances]),
            "hist_action": torch.stack([item["hist_action"] for item in instances]),
            "proprio": torch.stack([item["proprio"] for item in instances]),
            "stale_age": torch.tensor([item["stale_age"] for item in instances], dtype=torch.long),
            "lang": [item["instruction"] for item in instances],
            "task": [item["task"] for item in instances],
            "sample_key": [item["sample_key"] for item in instances],
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        device = base.weight.device

        self.lora_A = nn.Parameter(torch.empty((self.rank, base.in_features), device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.rank), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora = torch.nn.functional.linear(self.dropout(x).to(torch.float32), self.lora_A)
        lora = torch.nn.functional.linear(lora, self.lora_B) * self.scaling
        return result + lora.to(dtype=result.dtype)

    def merged_linear(self) -> nn.Linear:
        merged = copy.deepcopy(self.base)
        delta = torch.matmul(self.lora_B.detach(), self.lora_A.detach()) * self.scaling
        merged.weight.data = merged.weight.data + delta.to(device=merged.weight.device, dtype=merged.weight.dtype)
        return merged


def get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def should_lora_module(module_name: str, scopes: Sequence[str]) -> bool:
    if not scopes:
        return True
    return any(module_name == scope or module_name.startswith(scope + ".") for scope in scopes)


def inject_lora(policy: nn.Module, rank: int, alpha: float, dropout: float, target_scopes: Sequence[str]) -> List[str]:
    targets = [
        name
        for name, module in policy.named_modules()
        if isinstance(module, nn.Linear) and should_lora_module(name, target_scopes)
    ]
    for name in targets:
        parent, child_name = get_parent_module(policy, name)
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank, alpha, dropout))
    return targets


def iter_lora_modules(module: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def adapter_state_dict(policy: nn.Module) -> Dict[str, torch.Tensor]:
    state = OrderedDict()
    for name, module in iter_lora_modules(policy):
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def merge_lora_modules(module: nn.Module) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, child_name, child.merged_linear())
        else:
            merge_lora_modules(child)


def count_trainable_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def set_frozen_feature_extractors_eval(policy: nn.Module) -> None:
    for name in ("vision_encoder", "depth_encoder", "tactile_encoder"):
        module = getattr(policy, name, None)
        if module is not None:
            module.eval()


def build_policy(args, device: torch.device) -> DiffusionDiTImagePolicy:
    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    policy = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}},
        noise_scheduler=scheduler,
        n_action_steps=args.action_chunk_size,
        num_inference_steps=args.fast_num_inference_steps,
        vision_encoder=args.vision_encoder,
        with_depth=args.with_depth,
        with_gripper=args.with_gripper,
        with_tactile=args.with_tactile,
        cond_drop_chance=args.cond_drop_chance,
        progressive_noise=False,
    )
    return policy.to(device)


def strip_prefixed_state(state: Dict[str, torch.Tensor], prefix: str) -> OrderedDict:
    prefix = prefix + "."
    return OrderedDict((key[len(prefix) :], value) for key, value in state.items() if key.startswith(prefix))


def load_specialist_checkpoint(policy: nn.Module, specialist_path: str | Path, source: str = "ema_model") -> Dict:
    checkpoint = torch.load(Path(specialist_path).expanduser().as_posix(), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported specialist checkpoint type: {type(checkpoint)}")

    loaded_from = "raw"
    load_state = checkpoint
    if any(str(key).startswith("ema_model.") for key in checkpoint):
        if source == "auto":
            source = "ema_model"
        if source not in {"ema_model", "online_model"}:
            raise ValueError("--checkpoint_source must be ema_model, online_model, or auto")
        load_state = strip_prefixed_state(checkpoint, source)
        loaded_from = source

    missing, unexpected = policy.load_state_dict(load_state, strict=False)
    return {
        "loaded_from": loaded_from,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "ema_checkpoint": any(str(key).startswith("ema_model.") for key in checkpoint),
    }


def load_generalist(args, device: torch.device):
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

    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    slow_model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **model_kwargs)
    slow_model.eval()
    slow_model.requires_grad_(False)
    if quantization_config is None and args.device_map == "none":
        slow_model = slow_model.to(device)
    return slow_model, processor, model_dtype


def move_slow_inputs(slow_inputs: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": slow_inputs["input_ids"].to(device),
        "input_lengths": slow_inputs["input_lengths"].to(device),
        "pixel_values": slow_inputs["pixel_values"].to(device=device, dtype=dtype),
    }


def move_fast_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key in (
        "pixel_values_dp",
        "prev_pixel_values_dp",
        "gripper_image",
        "depth_image",
        "depth_gripper",
        "tactile_image",
        "raw_action",
        "hist_action",
        "proprio",
        "stale_age",
    ):
        moved[key] = batch[key].to(device)
    moved["lang"] = batch["lang"]
    moved["task"] = batch["task"]
    moved["sample_key"] = batch["sample_key"]
    return moved


def normalize_predict_output(output, action_chunk_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, tuple) and len(output) >= 2:
        action, hidden_states = output[0], output[1]
    else:
        raise RuntimeError(
            "The loaded generalist predict_action() did not return hidden states. "
            "This LoRA trial needs the RoboDual generalist that returns (action, hidden_states)."
        )

    action = torch.as_tensor(action, device=device, dtype=torch.float32)
    action = action.reshape(1, action_chunk_size, -1)[..., :7]
    hidden_states = torch.as_tensor(hidden_states, device=device, dtype=torch.float32)
    return action, hidden_states


def run_slow_batch(
    slow_model,
    slow_inputs: Dict[str, torch.Tensor],
    sample_keys: Sequence,
    action_chunk_size: int,
    device: torch.device,
    cache: Dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = []
    hidden_states = []
    for batch_i, sample_key in enumerate(sample_keys):
        cache_key = tuple(sample_key)
        if cache is not None and cache_key in cache:
            cached_action, cached_hidden = cache[cache_key]
            actions.append(cached_action.to(device))
            hidden_states.append(cached_hidden.to(device))
            continue

        input_len = int(slow_inputs["input_lengths"][batch_i].item())
        single_inputs = {
            "input_ids": slow_inputs["input_ids"][batch_i : batch_i + 1, :input_len],
            "pixel_values": slow_inputs["pixel_values"][batch_i : batch_i + 1],
        }
        with torch.inference_mode():
            output = slow_model.predict_action(**single_inputs, do_sample=False)
        action, hidden = normalize_predict_output(output, action_chunk_size, device)
        if cache is not None:
            cache[cache_key] = (action.detach().cpu(), hidden.detach().cpu())
        actions.append(action)
        hidden_states.append(hidden)

    return torch.cat(actions, dim=0), pad_hidden_state_batch(hidden_states)


def pad_hidden_state_batch(hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
    max_len = max(int(hidden.shape[1]) for hidden in hidden_states)
    padded = []
    for hidden in hidden_states:
        if hidden.shape[1] < max_len:
            pad_len = max_len - hidden.shape[1]
            pad = hidden[:, -1:, :].expand(hidden.shape[0], pad_len, hidden.shape[2]).clone()
            hidden = torch.cat([hidden, pad], dim=1)
        padded.append(hidden)
    return torch.cat(padded, dim=0)


def build_ref_actions(slow_actions: torch.Tensor, stale_ages: torch.Tensor, empty_ref_after_age: int) -> torch.Tensor:
    batch_size, action_chunk_size, action_dim = slow_actions.shape
    ref_actions = torch.zeros_like(slow_actions)
    for batch_i in range(batch_size):
        age = int(stale_ages[batch_i].item())
        if age >= empty_ref_after_age:
            num_cond_actions = 0
        else:
            num_cond_actions = max(0, action_chunk_size - age)
        if num_cond_actions > 0:
            ref_actions[batch_i, :num_cond_actions] = slow_actions[batch_i, -num_cond_actions:]
    return ref_actions


def save_adapter(policy: nn.Module, path: Path, metadata: Dict) -> None:
    payload = {
        "format": "robodual_specialist_lora_linear_v1",
        "metadata": metadata,
        "lora_state": adapter_state_dict(policy),
    }
    torch.save(payload, path.as_posix())


def save_ema_compatible_checkpoint(
    base_specialist_path: str | Path,
    merged_policy_state: Dict[str, torch.Tensor],
    output_path: Path,
) -> str:
    base_state = torch.load(Path(base_specialist_path).expanduser().as_posix(), map_location="cpu")
    if not isinstance(base_state, dict) or not any(str(key).startswith("ema_model.") for key in base_state):
        torch.save(OrderedDict((key, value.detach().cpu()) for key, value in merged_policy_state.items()), output_path.as_posix())
        return "raw_policy"

    new_state = OrderedDict(base_state)
    for prefix in ("ema_model", "online_model"):
        for key, value in merged_policy_state.items():
            full_key = f"{prefix}.{key}"
            if full_key in new_state:
                new_state[full_key] = value.detach().cpu()
    torch.save(new_state, output_path.as_posix())
    return "ema_compatible"


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def train(args) -> Dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")

    tasks = parse_csv(args.tasks)
    stale_ages = parse_int_csv(args.stale_ages)
    if not stale_ages:
        raise ValueError("--stale_ages must contain at least one age")

    slow_model, processor, slow_dtype = load_generalist(args, device)
    dataset = StaleRefRolloutDataset(
        data_dir=args.data_dir,
        stale_ages=stale_ages,
        action_chunk_size=args.action_chunk_size,
        tasks=tasks,
        samples_per_rollout_per_age=args.samples_per_rollout_per_age,
        sampling_stride=args.sampling_stride,
        seed=args.seed,
    )
    collator = StaleRefCollator(processor, vision_encoder=args.vision_encoder)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=False,
    )

    policy = build_policy(args, device)
    load_info = load_specialist_checkpoint(policy, args.specialist_path, source=args.checkpoint_source)
    policy.requires_grad_(False)
    lora_targets = inject_lora(
        policy,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_scopes=parse_csv(args.lora_target_scopes),
    )
    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were created")

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    slow_cache = {} if args.cache_slow == "cpu" else None

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dataset_size": len(dataset),
        "tasks": sorted(set(sample.task for sample in dataset.samples)),
        "stale_ages": stale_ages,
        "lora_targets": lora_targets,
        "num_lora_targets": len(lora_targets),
        "trainable_params": count_trainable_params(policy),
        "specialist_load_info": load_info,
    }
    write_json(output_dir / "training_config.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))

    policy.train()
    set_frozen_feature_extractors_eval(policy)
    global_step = 0
    recent_losses: List[float] = []
    progress = tqdm(total=args.max_steps, desc="train_lora")

    while global_step < args.max_steps:
        for batch in dataloader:
            slow_inputs = move_slow_inputs(batch["slow_inputs"], device, slow_dtype)
            fast_batch = move_fast_batch(batch, device)

            slow_actions, action_cond = run_slow_batch(
                slow_model=slow_model,
                slow_inputs=slow_inputs,
                sample_keys=fast_batch["sample_key"],
                action_chunk_size=args.action_chunk_size,
                device=device,
                cache=slow_cache,
            )
            ref_actions = build_ref_actions(
                slow_actions=slow_actions,
                stale_ages=fast_batch["stale_age"],
                empty_ref_after_age=args.empty_ref_after_age,
            )

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                loss = policy.compute_loss(
                    trajectory=fast_batch["raw_action"].float(),
                    ref_action=ref_actions.float(),
                    action_cond=action_cond.float(),
                    obs=(fast_batch["pixel_values_dp"].float(), fast_batch["prev_pixel_values_dp"].float()),
                    depth_obs=fast_batch["depth_image"].float(),
                    gripper_obs=(fast_batch["gripper_image"].float(), fast_batch["depth_gripper"].float()),
                    tactile_obs=fast_batch["tactile_image"].float() if args.with_tactile else None,
                    lang=fast_batch["lang"],
                    proprio=fast_batch["proprio"].float(),
                    hist_action=fast_batch["hist_action"].float(),
                    decoupled_loss=False,
                )

            loss = loss / args.grad_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_value = float(loss.detach().cpu().item() * args.grad_accumulation_steps)
            recent_losses.append(loss_value)
            if len(recent_losses) > 100:
                recent_losses.pop(0)

            metric = {
                "step": global_step,
                "loss": loss_value,
                "loss_ma100": float(sum(recent_losses) / len(recent_losses)),
                "lr": optimizer.param_groups[0]["lr"],
                "cache_size": 0 if slow_cache is None else len(slow_cache),
            }
            with metrics_path.open("a") as file:
                file.write(json.dumps(metric, sort_keys=True) + "\n")
            progress.set_postfix(loss=f"{loss_value:.4f}", cache=metric["cache_size"])
            progress.update(1)

            if global_step % args.save_adapter_steps == 0:
                save_adapter(policy, output_dir / f"adapter_step_{global_step}.pt", metadata | {"step": global_step})

            if global_step >= args.max_steps:
                break

    progress.close()
    save_adapter(policy, output_dir / "adapter_final.pt", metadata | {"step": global_step})

    merge_lora_modules(policy)
    merged_policy_state = OrderedDict((key, value.detach().cpu()) for key, value in policy.state_dict().items())
    torch.save(merged_policy_state, (output_dir / "specialist_lora_merged_policy.pt").as_posix())
    checkpoint_format = save_ema_compatible_checkpoint(
        args.specialist_path,
        merged_policy_state,
        output_dir / "specialist_lora_merged_ema.pt",
    )

    summary = {
        **metadata,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "global_step": global_step,
        "final_loss_ma100": float(sum(recent_losses) / len(recent_losses)) if recent_losses else None,
        "checkpoint_format": checkpoint_format,
        "outputs": {
            "adapter_final": (output_dir / "adapter_final.pt").as_posix(),
            "merged_policy": (output_dir / "specialist_lora_merged_policy.pt").as_posix(),
            "merged_ema": (output_dir / "specialist_lora_merged_ema.pt").as_posix(),
            "metrics": metrics_path.as_posix(),
        },
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist_path", default=DEFAULT_GENERALIST_PATH.as_posix(), type=str)
    parser.add_argument("--specialist_path", default=DEFAULT_SPECIALIST_PATH.as_posix(), type=str)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR.as_posix(), type=str)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR.as_posix(), type=str)

    parser.add_argument("--tasks", default="", type=str, help="Optional comma-separated task filter.")
    parser.add_argument("--stale_ages", default="8,9,10,11", type=str)
    parser.add_argument("--empty_ref_after_age", default=8, type=int)
    parser.add_argument("--samples_per_rollout_per_age", default=4, type=int)
    parser.add_argument("--sampling_stride", default=1, type=int)

    parser.add_argument("--action_chunk_size", default=8, type=int)
    parser.add_argument("--fast_num_inference_steps", default=10, type=int)
    parser.add_argument("--vision_encoder", default="DINO", choices=["DINO", "Theia"], type=str)
    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--cond_drop_chance", default=0.0, type=float)

    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=8.0, type=float)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument(
        "--lora_target_scopes",
        default="model",
        type=str,
        help="Comma-separated module-name prefixes. Default trains LoRA under specialist.model only.",
    )

    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--max_steps", default=300, type=int)
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--save_adapter_steps", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], type=str)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none", type=str)
    parser.add_argument("--checkpoint_source", default="ema_model", choices=["ema_model", "online_model", "auto"], type=str)
    parser.add_argument("--cache_slow", default="cpu", choices=["cpu", "none"], type=str)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
