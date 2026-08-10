#!/usr/bin/env python3
"""M1 Age-Extended Expert post-training for the existing RoboDual specialist.

This trainer consumes the immutable ``robodual_age_extended_expert_v1``
collector contract.  It never loads or forwards OpenVLA: the persisted
``slow_action`` and ``slow_hidden`` tensors are the only slow-system inputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "robodual_age_extended_expert_v1"
TARGET_SOURCE = "calvin_expert"
SPLITS = ("train", "validation", "test")
AGES = tuple(range(12))
ACTION_HORIZON = 8
ACTION_DIM = 7
HISTORY_LENGTH = 4
DEFAULT_DATA_DIR = REPO_ROOT / "DiT_train/data_collection/runs/ageext_expert_600_s42"
DEFAULT_PROCESSOR_PATH = REPO_ROOT.parent / "models/generalist"
DEFAULT_SPECIALIST_PATH = REPO_ROOT.parent / "models/specialist/Specialist+Depth+Gripper.pt"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN deterministic mode is slower but makes paired M1 comparisons safer.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resolve_under(root: Path, relative: str, *, kind: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{kind} escapes its root: {relative!r}")
    return path


def normalize_ref(value: Any, sample_id: str) -> torch.Tensor:
    ref = torch.as_tensor(value, dtype=torch.float32)
    if tuple(ref.shape) == (1, ACTION_HORIZON, ACTION_DIM):
        ref = ref.squeeze(0)
    if tuple(ref.shape) != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"{sample_id}: ref_action has shape {tuple(ref.shape)}, expected [1,8,7] or [8,7]")
    return ref.contiguous()


class AgeExtendedExpertDataset(Dataset):
    """A split-preserving view over the Age-Extended Expert collector output."""

    REQUIRED_REFERENCES = (
        "current_rgb_static", "previous_rgb_static", "current_depth_static",
        "current_rgb_gripper", "current_depth_gripper",
    )

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        processor: Any | None,
        *,
        condition_cache_size: int = 128,
        verify_reference_contract: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if condition_cache_size <= 0:
            raise ValueError("condition_cache_size must be positive")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.manifest_path = self.data_dir / "manifest.json"
        self.samples_path = self.data_dir / "samples.jsonl"
        for required in (self.manifest_path, self.samples_path, self.data_dir / "anchors.jsonl", self.data_dir / "audit_summary.json", self.data_dir / "conditions"):
            if not required.exists():
                raise FileNotFoundError(required)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "target_source": TARGET_SOURCE,
            "dataset_source_split": "training",
        }
        mismatches = {
            key: {"expected": value, "actual": self.manifest.get(key)}
            for key, value in expected.items() if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Manifest contract mismatch: {json.dumps(mismatches, sort_keys=True)}")
        if self.manifest.get("action_chunk_size") != ACTION_HORIZON:
            raise ValueError("Manifest action_chunk_size must be 8")
        if self.manifest.get("with_tactile") is not False:
            raise ValueError("M1 requires manifest with_tactile=false")
        self.dataset_root = Path(self.manifest["dataset_root"]).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(self.dataset_root)
        all_samples = read_jsonl(self.samples_path)
        self.samples = [row for row in all_samples if row.get("split") == split]
        if not self.samples:
            raise ValueError(f"No {split!r} samples in {self.samples_path}")
        illegal = sorted({str(row.get("split")) for row in self.samples if row.get("split") != split})
        if illegal:
            raise AssertionError(f"Split filtering failed: {illegal}")
        self.split = split
        self.processor = processor
        self.condition_cache_size = int(condition_cache_size)
        self._condition_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        if verify_reference_contract:
            self.preflight_reference_contract()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_condition(self, condition_path: str) -> dict[str, Any]:
        cached = self._condition_cache.pop(condition_path, None)
        if cached is not None:
            self._condition_cache[condition_path] = cached
            return cached
        path = resolve_under(self.data_dir, condition_path, kind="condition_path")
        if not path.is_file():
            raise FileNotFoundError(path)
        condition = torch_load_cpu(path)
        if not isinstance(condition, dict):
            raise TypeError(f"Condition is not a dict: {path}")
        slow_action = torch.as_tensor(condition.get("slow_action"), dtype=torch.float32)
        slow_hidden = torch.as_tensor(condition.get("slow_hidden"), dtype=torch.float32)
        if tuple(slow_action.shape) != (1, ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Bad slow_action shape in {path}: {tuple(slow_action.shape)}")
        if slow_hidden.ndim != 3 or slow_hidden.shape[0] != 1 or slow_hidden.shape[1] <= 0:
            raise ValueError(f"Bad slow_hidden shape in {path}: {tuple(slow_hidden.shape)}")
        condition = dict(condition)
        condition["slow_action"] = slow_action
        condition["slow_hidden"] = slow_hidden
        self._condition_cache[condition_path] = condition
        while len(self._condition_cache) > self.condition_cache_size:
            self._condition_cache.popitem(last=False)
        return condition

    @staticmethod
    def _expected_ref(slow_action: torch.Tensor, age: int) -> torch.Tensor:
        expected = torch.zeros_like(slow_action)
        count = max(ACTION_HORIZON - age, 0)
        if count:
            expected[:, :count] = slow_action[:, -count:]
        return expected

    def _assert_reference_contract(self, row: Mapping[str, Any]) -> None:
        sample_id = str(row.get("sample_id", "<missing>"))
        age = int(row["slow_age"])
        if age not in AGES:
            raise ValueError(f"{sample_id}: slow_age must be 0..11, got {age}")
        condition = self._load_condition(str(row["condition_path"]))
        expected = self._expected_ref(condition["slow_action"], age)
        actual = torch.as_tensor(row["ref_action"], dtype=torch.float32)
        if tuple(actual.shape) == (ACTION_HORIZON, ACTION_DIM):
            actual = actual.unsqueeze(0)
        if tuple(actual.shape) != (1, ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"{sample_id}: persisted ref_action shape is {tuple(actual.shape)}")
        if not torch.equal(actual, expected):
            raise AssertionError(f"{sample_id}: persisted ref_action violates slow_action suffix/zero contract")
        count = max(ACTION_HORIZON - age, 0)
        if int(row["ref_valid_count"]) != count:
            raise AssertionError(f"{sample_id}: ref_valid_count mismatch")
        if age >= ACTION_HORIZON and torch.count_nonzero(actual).item() != 0:
            raise AssertionError(f"{sample_id}: expired reference is not exact zero")

    def preflight_reference_contract(self) -> None:
        # Always cover the discontinuity and both ends, then add stable samples.
        selected: list[dict[str, Any]] = []
        for age in (0, 7, 8, 11):
            row = next((item for item in self.samples if int(item["slow_age"]) == age), None)
            if row is not None:
                selected.append(row)
        stable = sorted(
            self.samples,
            key=lambda row: hashlib.sha256(str(row["sample_id"]).encode()).hexdigest(),
        )[:16]
        seen: set[str] = set()
        for row in selected + stable:
            if row["sample_id"] not in seen:
                self._assert_reference_contract(row)
                seen.add(row["sample_id"])

    def _load_reference(self, reference: Mapping[str, Any], expected_key: str) -> np.ndarray:
        if reference.get("key") != expected_key:
            raise ValueError(f"Observation reference key mismatch: expected {expected_key}, got {reference.get('key')}")
        path = resolve_under(self.dataset_root, str(reference["relative_path"]), kind="observation reference")
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            if expected_key not in archive.files:
                raise KeyError(f"{path} lacks {expected_key}")
            return np.asarray(archive[expected_key]).copy()

    def _image(self, array: np.ndarray) -> torch.Tensor:
        if self.processor is None:
            raise RuntimeError("Dataset image access requires a processor")
        transformed = self.processor.image_processor.apply_transform(Image.fromarray(array))
        transformed = torch.as_tensor(transformed)[:3]
        if transformed.ndim != 3 or transformed.shape[0] != 3:
            raise ValueError(f"Processor returned invalid DINO image shape {tuple(transformed.shape)}")
        return transformed.to(torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        sample_id = str(row["sample_id"])
        if row.get("target_source") != TARGET_SOURCE or row.get("dataset_source_split") != "training":
            raise ValueError(f"{sample_id}: sample source contract mismatch")
        condition = self._load_condition(str(row["condition_path"]))
        current_rgb = self._load_reference(row["current_rgb_static"], "rgb_static")
        previous_rgb = self._load_reference(row["previous_rgb_static"], "rgb_static")
        static_depth = self._load_reference(row["current_depth_static"], "depth_static").astype(np.float32)
        gripper_rgb = self._load_reference(row["current_rgb_gripper"], "rgb_gripper")
        gripper_depth = self._load_reference(row["current_depth_gripper"], "depth_gripper").astype(np.float32)
        raw_action = torch.as_tensor(row["target_rel_actions"], dtype=torch.float32)
        hist_action = torch.as_tensor(row["hist_action_before"], dtype=torch.float32)
        proprio = torch.as_tensor(row["current_proprio"], dtype=torch.float32)
        if tuple(raw_action.shape) != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"{sample_id}: target_rel_actions shape {tuple(raw_action.shape)}")
        if tuple(hist_action.shape) != (HISTORY_LENGTH, ACTION_DIM):
            raise ValueError(f"{sample_id}: hist_action_before shape {tuple(hist_action.shape)}")
        if tuple(proprio.shape) != (ACTION_DIM,):
            raise ValueError(f"{sample_id}: current_proprio shape {tuple(proprio.shape)}")
        return {
            "sample_id": sample_id,
            "condition_id": str(row["condition_id"]),
            "trajectory_id": str(row["trajectory_id"]),
            "age": int(row["slow_age"]),
            "instruction": str(row["instruction"]),
            "current_rgb": self._image(current_rgb),
            "previous_rgb": self._image(previous_rgb),
            "depth_image": torch.from_numpy((static_depth - 3.5) / (6.2 - 3.5)).float(),
            "gripper_image": self._image(gripper_rgb),
            "depth_gripper": torch.from_numpy(gripper_depth / 2.0).float(),
            "raw_action": raw_action,
            "ref_action": normalize_ref(row["ref_action"], sample_id),
            "ref_valid_count": int(row["ref_valid_count"]),
            "hist_action": hist_action,
            "proprio": proprio,
            # Remove only the collector's singleton batch dimension.  The
            # DataLoader restores it; batch_size is intentionally fixed at 1.
            "slow_hidden": condition["slow_hidden"].squeeze(0),
            "slow_action": condition["slow_action"].squeeze(0),
        }


def build_policy(device: torch.device) -> torch.nn.Module:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy

    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    policy = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [ACTION_DIM]}},
        noise_scheduler=scheduler,
        n_action_steps=ACTION_HORIZON,
        num_inference_steps=10,
        vision_encoder="DINO",
        vision_encoder_pretrained=False,
        with_depth=True,
        with_gripper=True,
        with_tactile=False,
        cond_drop_chance=0.1,
        progressive_noise=False,
    )
    return policy.to(device)


def extract_prefixed_policy_state(checkpoint: Mapping[str, Any], prefix: str) -> OrderedDict[str, torch.Tensor]:
    marker = prefix + "."
    state = OrderedDict(
        (str(key)[len(marker):], value)
        for key, value in checkpoint.items()
        if str(key).startswith(marker) and str(key) != f"{prefix}._dummy_variable"
    )
    if not state:
        raise ValueError(f"Checkpoint has no {prefix}.* policy tensors")
    return state


def load_baseline_ema(policy: torch.nn.Module, path: str | Path) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch_load_cpu(path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported specialist checkpoint type: {type(checkpoint)}")
    state = extract_prefixed_policy_state(checkpoint, "ema_model")
    incompatible = policy.load_state_dict(state, strict=False)
    missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "loaded_from": "ema_model",
        "ema_tensor_count": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    if missing or unexpected:
        raise RuntimeError(f"Specialist checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return OrderedDict(checkpoint), audit


class PolicyEMA:
    """EMA matching the baseline ema-pytorch power=0.75 warm-start."""

    def __init__(
        self, online_model: torch.nn.Module, beta: float = 0.9999,
        power: float = 0.75, update_after_step: int = 100,
    ) -> None:
        self.beta = float(beta)
        self.power = float(power)
        self.update_after_step = int(update_after_step)
        self.ema_model = copy.deepcopy(online_model).eval()
        self.ema_model.requires_grad_(False)
        self.updates = 0
        self.current_decay = 0.0

    @torch.no_grad()
    def update(self, online_model: torch.nn.Module) -> None:
        self.updates += 1
        if self.updates <= self.update_after_step:
            decay = 0.0
        else:
            epoch = self.updates - self.update_after_step
            decay = 1.0 - (1.0 + epoch) ** (-self.power)
            decay = min(max(decay, 0.0), self.beta)
        self.current_decay = float(decay)
        online = online_model.state_dict()
        for name, ema_value in self.ema_model.state_dict().items():
            source = online[name].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value) and decay > 0.0:
                ema_value.lerp_(source.to(dtype=ema_value.dtype), 1.0 - decay)
            else:
                ema_value.copy_(source.to(dtype=ema_value.dtype))


def freeze_invariants(policy: torch.nn.Module) -> None:
    policy.vision_encoder.requires_grad_(False)
    policy.vision_encoder.eval()


def set_train_mode(policy: torch.nn.Module) -> None:
    policy.train()
    # nn.Module.train() is recursive, so restore the deployment-frozen DINO.
    policy.vision_encoder.eval()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def autocast_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def standard_loss(policy: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    return policy.compute_loss(
        trajectory=batch["raw_action"].float(),
        ref_action=batch["ref_action"].float(),
        action_cond=batch["slow_hidden"].float(),
        obs=(batch["current_rgb"].float(), batch["previous_rgb"].float()),
        depth_obs=batch["depth_image"].float(),
        gripper_obs=(batch["gripper_image"].float(), batch["depth_gripper"].float()),
        tactile_obs=None,
        lang=batch["instruction"],
        proprio=batch["proprio"].float(),
        hist_action=batch["hist_action"].float(),
        decoupled_loss=False,
    )


def deterministic_noise(sample_id: str, base_seed: int, device: torch.device) -> tuple[torch.Tensor, int, str]:
    stable = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    derived = (int(base_seed) + stable) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    noise = torch.randn((1, ACTION_HORIZON, ACTION_DIM), generator=generator, dtype=torch.float32)
    digest = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
    return noise.to(device), derived, digest


def metric_group(rows: Iterable[dict[str, float]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        return {"n": 0, "diffusion_noise_mse": None, "first_action_ee6_rmse": None, "first_action_gripper_sign_accuracy": None}
    return {
        "n": len(values),
        "diffusion_noise_mse": float(np.mean([row["diffusion_noise_mse"] for row in values])),
        "first_action_ee6_rmse": float(math.sqrt(np.mean([row["first_action_ee6_mse"] for row in values]))),
        "first_action_gripper_sign_accuracy": float(np.mean([row["gripper_correct"] for row in values])),
    }


@torch.no_grad()
def validate(
    policy: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer_step: int,
    timestep: int,
    validation_seed: int,
) -> dict[str, Any]:
    policy.eval()
    rows: list[dict[str, float]] = []
    noise_protocol: list[dict[str, Any]] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        sample_id = str(cpu_batch["sample_id"][0])
        age = int(cpu_batch["age"].item())
        noise, derived_seed, noise_sha = deterministic_noise(sample_id, validation_seed, device)
        timesteps = torch.tensor([timestep], dtype=torch.long, device=device)
        cond_mask = torch.ones((1, 1), dtype=torch.float32, device=device)
        with autocast_context(device):
            details = policy.compute_loss(
                trajectory=batch["raw_action"].float(),
                ref_action=batch["ref_action"].float(),
                action_cond=batch["slow_hidden"].float(),
                obs=(batch["current_rgb"].float(), batch["previous_rgb"].float()),
                depth_obs=batch["depth_image"].float(),
                gripper_obs=(batch["gripper_image"].float(), batch["depth_gripper"].float()),
                tactile_obs=None,
                lang=batch["instruction"],
                proprio=batch["proprio"].float(),
                hist_action=batch["hist_action"].float(),
                decoupled_loss=False,
                noise=noise,
                timesteps=timesteps,
                cond_mask=cond_mask,
                return_details=True,
            )
        prediction = details["prediction"].to(torch.float32)
        target = details["target"].to(torch.float32)
        trajectory = batch["raw_action"].to(torch.float32)
        noisy = policy.noise_scheduler.add_noise(trajectory, noise, timesteps)
        alpha_bar = policy.noise_scheduler.alphas_cumprod[timestep].to(device=device, dtype=torch.float32)
        x0_hat = (noisy.float() - torch.sqrt(1.0 - alpha_bar) * prediction) / torch.sqrt(alpha_bar)
        first_error = x0_hat[0, 0, :6] - trajectory[0, 0, :6]
        true_gripper = bool(trajectory[0, 0, 6].item() >= 0.0)
        predicted_gripper = bool(x0_hat[0, 0, 6].item() >= 0.0)
        rows.append({
            "age": float(age),
            "diffusion_noise_mse": float(torch.mean(torch.square(prediction - target)).cpu()),
            "first_action_ee6_mse": float(torch.mean(torch.square(first_error)).cpu()),
            "gripper_correct": float(predicted_gripper == true_gripper),
        })
        if len(noise_protocol) < 4:
            noise_protocol.append({"sample_id": sample_id, "derived_seed": derived_seed, "noise_sha256": noise_sha})
    by_age = {age: [row for row in rows if int(row["age"]) == age] for age in AGES}
    metrics = {
        "optimizer_step": int(optimizer_step),
        "validation_timestep": int(timestep),
        "validation_seed": int(validation_seed),
        "fixed_sample_count": len(rows),
        "noise_protocol_examples": noise_protocol,
        "groups": {
            "age_0_7": metric_group(row for row in rows if int(row["age"]) <= 7),
            "age_8": metric_group(by_age[8]),
            "age_9": metric_group(by_age[9]),
            "age_10": metric_group(by_age[10]),
            "age_11": metric_group(by_age[11]),
            "age_8_11": metric_group(row for row in rows if int(row["age"]) >= 8),
        },
    }
    return metrics


def cosine_schedule(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return LambdaLR(optimizer, multiplier)


def evaluator_state(
    baseline_template: Mapping[str, Any],
    online_policy: torch.nn.Module,
    ema_policy: torch.nn.Module,
) -> OrderedDict[str, Any]:
    state = OrderedDict(baseline_template)
    online_state, ema_state = online_policy.state_dict(), ema_policy.state_dict()
    for name, value in online_state.items():
        state[f"online_model.{name}"] = value.detach().cpu()
    for name, value in ema_state.items():
        state[f"ema_model.{name}"] = value.detach().cpu()
    return state


def checkpoint_round_trip(path: Path, device: torch.device, expected_ema: torch.nn.Module) -> dict[str, Any]:
    saved = torch_load_cpu(path)
    reloaded = build_policy(torch.device("cpu"))
    incompatible = reloaded.load_state_dict(extract_prefixed_policy_state(saved, "ema_model"), strict=False)
    missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint round-trip mismatch: missing={missing}, unexpected={unexpected}")
    expected = expected_ema.state_dict()
    actual = reloaded.state_dict()
    names = list(expected)
    probes = sorted(set((names[0], names[len(names) // 2], names[-1])))
    comparisons = {}
    for name in probes:
        left = expected[name].detach().cpu()
        right = actual[name].detach().cpu()
        exact = torch.equal(left, right)
        close = bool(torch.allclose(left, right)) if torch.is_floating_point(left) else exact
        if not (exact or close):
            raise AssertionError(f"Round-trip tensor differs: {name}")
        comparisons[name] = {"exact": exact, "allclose": close}
    del reloaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"missing_keys": missing, "unexpected_keys": unexpected, "tensor_comparisons": comparisons}


def save_checkpoint(
    output_dir: Path,
    step: int,
    policy: torch.nn.Module,
    ema: PolicyEMA,
    optimizer: AdamW,
    scheduler: LambdaLR,
    args: argparse.Namespace,
    baseline_template: Mapping[str, Any],
    manifest_sha256: str,
    commit: str,
    validation_metrics: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    evaluator_path = output_dir / f"specialist_ema_step_{step:06d}.pt"
    training_path = output_dir / f"training_state_step_{step:06d}.pt"
    torch.save(evaluator_state(baseline_template, policy, ema.ema_model), evaluator_path)
    training_payload = {
        "format": "robodual_m1_age_extended_expert_training_v1",
        "online_policy": OrderedDict((name, value.detach().cpu()) for name, value in policy.state_dict().items()),
        "ema_policy": OrderedDict((name, value.detach().cpu()) for name, value in ema.ema_model.state_dict().items()),
        "ema_beta": ema.beta,
        "ema_power": ema.power,
        "ema_update_after_step": ema.update_after_step,
        "ema_current_decay": ema.current_decay,
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_optimizer_step": int(step),
        "args": serialized_args(args),
        "dataset_manifest_sha256": manifest_sha256,
        "git_commit": commit,
        "validation_metrics": dict(validation_metrics),
        "evaluator_checkpoint": evaluator_path.name,
    }
    torch.save(training_payload, training_path)
    audit = checkpoint_round_trip(evaluator_path, device, ema.ema_model)
    latest = {
        "global_optimizer_step": int(step),
        "evaluator_checkpoint": str(evaluator_path),
        "training_checkpoint": str(training_path),
        "round_trip": audit,
    }
    write_json(output_dir / "latest_checkpoint.json", latest)
    return latest


def dataset_statistics(data_dir: Path) -> dict[str, Any]:
    samples = read_jsonl(data_dir / "samples.jsonl")
    split_counts = Counter(str(row["split"]) for row in samples)
    age_counts = Counter(int(row["slow_age"]) for row in samples)
    return {
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "unique_trajectories": len({str(row["trajectory_id"]) for row in samples}),
        "unique_conditions": len({str(row["condition_id"]) for row in samples}),
        "age_counts": {str(age): age_counts[age] for age in AGES},
        "old_count": sum(count for age, count in age_counts.items() if age <= 7),
        "new_count": sum(count for age, count in age_counts.items() if age >= 8),
    }


def hidden_length_distribution(data_dir: Path) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for path in sorted((data_dir / "conditions").glob("*.pt")):
        condition = torch_load_cpu(path)
        hidden = torch.as_tensor(condition["slow_hidden"])
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError(f"Bad slow_hidden shape in {path}: {tuple(hidden.shape)}")
        counts[int(hidden.shape[1])] += 1
    return {str(length): counts[length] for length in sorted(counts)}


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())) and not overwrite:
        raise FileExistsError(f"{path} is non-empty; use --overwrite to permit replacing trainer outputs")
    path.mkdir(parents=True, exist_ok=True)


def make_loaders(args: argparse.Namespace, processor: Any):
    datasets = {
        split: AgeExtendedExpertDataset(args.data_dir, split, processor)
        for split in SPLITS
    }
    train_weights = torch.tensor(
        [1.0 if int(row["slow_age"]) <= 7 else 2.0 for row in datasets["train"].samples],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_weights,
        num_samples=len(datasets["train"]),
        replacement=True,
        generator=generator,
    )
    common = dict(batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    loaders = {
        "train": DataLoader(datasets["train"], sampler=sampler, shuffle=False, **common),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return datasets, loaders


def load_processor(path: str | Path) -> Any:
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        str(Path(path).expanduser().resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )


def preflight_samples(dataset: AgeExtendedExpertDataset) -> dict[str, Any]:
    result = {}
    for age in (0, 7, 8, 11):
        index = next(i for i, row in enumerate(dataset.samples) if int(row["slow_age"]) == age)
        item = dataset[index]
        result[str(age)] = {
            "sample_id": item["sample_id"],
            "slow_hidden_shape": list(item["slow_hidden"].shape),
            "ref_valid_count": item["ref_valid_count"],
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size != 1:
        raise ValueError("M1 physical batch_size must be exactly 1: slow_hidden token lengths vary and no context padding mask exists")
    if args.grad_accumulation_steps <= 0 or args.max_optimizer_steps <= 0:
        raise ValueError("grad_accumulation_steps and max_optimizer_steps must be positive")
    if not 0 <= args.validation_timestep < 100:
        raise ValueError("validation_timestep must be in [0, 99]")
    if args.validate_every <= 0 or args.save_every <= 0:
        raise ValueError("validate_every and save_every must be positive")
    seed_everything(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    stats = dataset_statistics(data_dir)
    if args.dry_run:
        result = {"mode": "dry_run", "data_dir": str(data_dir), **stats}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    device = choose_device(args.device)
    processor = load_processor(args.processor_path)
    datasets, loaders = make_loaders(args, processor)
    token_lengths = hidden_length_distribution(data_dir)
    policy = build_policy(device)
    freeze_invariants(policy)
    baseline_template, load_audit = load_baseline_ema(policy, args.specialist_path)
    freeze_invariants(policy)
    ema = PolicyEMA(policy, beta=0.9999)
    trainable_count = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in policy.parameters())
    vision_trainable = sum(parameter.numel() for parameter in policy.vision_encoder.parameters() if parameter.requires_grad)
    audit = {
        "mode": "preflight" if args.preflight_only else "train",
        "device": str(device),
        "dataset": {**stats, "slow_hidden_token_length_distribution": token_lengths},
        "checkpoint_loading": load_audit,
        "parameters": {
            "trainable": trainable_count,
            "total": total_count,
            "vision_encoder_trainable": vision_trainable,
        },
        "loaded_age_examples": preflight_samples(datasets["train"]),
    }
    first_batch = move_batch(next(iter(loaders["train"])), device)
    policy.eval()
    with autocast_context(device):
        loss = standard_loss(policy, first_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Preflight compute_loss is not finite: {loss}")
    audit["compute_loss"] = float(loss.detach().float().cpu())
    audit["compute_loss_finite"] = True
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.preflight_only:
        return audit

    output_dir = Path(args.output_dir).expanduser().resolve()
    prepare_output(output_dir, args.overwrite)
    commit = git_commit()
    manifest_sha = sha256_file(data_dir / "manifest.json")
    config = {
        "args": serialized_args(args),
        "git_commit": commit,
        "dataset_manifest_sha256": manifest_sha,
        "dataset_statistics": stats,
        "slow_hidden_token_length_distribution": token_lengths,
        "checkpoint_loading": load_audit,
        "trainable_parameter_count": trainable_count,
    }
    write_json(output_dir / "config.json", config)
    metrics_path = output_dir / "metrics.jsonl"
    if args.overwrite:
        metrics_path.write_text("", encoding="utf-8")

    optimizer = AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_schedule(optimizer, args.warmup_optimizer_steps, args.max_optimizer_steps)

    # Required paired baseline: no optimizer, scheduler, or EMA update occurs first.
    baseline_metrics = validate(
        ema.ema_model, loaders["validation"], device,
        optimizer_step=0, timestep=args.validation_timestep,
        validation_seed=args.validation_seed,
    )
    append_jsonl(metrics_path, baseline_metrics)
    write_json(output_dir / "latest_validation.json", baseline_metrics)
    latest_validation = baseline_metrics

    optimizer.zero_grad(set_to_none=True)
    global_optimizer_step = 0
    micro_step = 0
    epoch = 0
    train_iterator = iter(loaders["train"])
    while global_optimizer_step < args.max_optimizer_steps:
        try:
            cpu_batch = next(train_iterator)
        except StopIteration:
            epoch += 1
            train_iterator = iter(loaders["train"])
            cpu_batch = next(train_iterator)
        batch = move_batch(cpu_batch, device)
        set_train_mode(policy)
        with autocast_context(device):
            raw_loss = standard_loss(policy, batch)
            scaled_loss = raw_loss / args.grad_accumulation_steps
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(f"Non-finite training loss at micro_step={micro_step}: {raw_loss}")
        scaled_loss.backward()
        micro_step += 1
        if micro_step % args.grad_accumulation_steps != 0:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            max_norm=args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_optimizer_step += 1
        ema.update(policy)
        train_record = {
            "type": "train",
            "global_optimizer_step": global_optimizer_step,
            "micro_step": micro_step,
            "logical_epoch": epoch,
            "loss": float(raw_loss.detach().float().cpu()),
            "gradient_norm": float(torch.as_tensor(grad_norm).float().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        append_jsonl(metrics_path, train_record)
        print(json.dumps(train_record, sort_keys=True), flush=True)

        if global_optimizer_step % args.validate_every == 0:
            latest_validation = validate(
                ema.ema_model, loaders["validation"], device,
                optimizer_step=global_optimizer_step,
                timestep=args.validation_timestep,
                validation_seed=args.validation_seed,
            )
            append_jsonl(metrics_path, latest_validation)
            write_json(output_dir / "latest_validation.json", latest_validation)
        if global_optimizer_step % args.save_every == 0:
            save_checkpoint(
                output_dir, global_optimizer_step, policy, ema, optimizer, scheduler,
                args, baseline_template, manifest_sha, commit, latest_validation, device,
            )

    if global_optimizer_step % args.save_every != 0:
        save_checkpoint(
            output_dir, global_optimizer_step, policy, ema, optimizer, scheduler,
            args, baseline_template, manifest_sha, commit, latest_validation, device,
        )
    return {
        "output_dir": str(output_dir),
        "global_optimizer_step": global_optimizer_step,
        "micro_step": micro_step,
        "latest_validation": latest_validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--processor_path", type=Path, default=DEFAULT_PROCESSOR_PATH)
    parser.add_argument("--specialist_path", type=Path, default=DEFAULT_SPECIALIST_PATH)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "runs/m1_age_extended_expert")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_optimizer_steps", type=int, default=300)
    parser.add_argument("--warmup_optimizer_steps", type=int, default=100)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--validation_timestep", type=int, default=50)
    parser.add_argument("--validation_seed", type=int, default=20260810)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true", help="Audit JSON metadata only; load no model or processor")
    parser.add_argument("--preflight_only", action="store_true", help="Run contract/checkpoint/sample/full-forward audits, then exit")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    if not args.preflight_only and not args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
