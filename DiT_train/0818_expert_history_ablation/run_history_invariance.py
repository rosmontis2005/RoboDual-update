#!/usr/bin/env python3
"""Paired fixed-noise expert-history invariance ablation for M1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DiT_train.train_age_extended_expert import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    AGES,
    AgeExtendedExpertDataset,
    autocast_context,
    build_policy,
    choose_device,
    deterministic_noise,
    git_commit,
    load_baseline_ema,
    load_processor,
    move_batch,
    seed_everything,
    sha256_file,
)


SCHEMA_VERSION = "m1_expert_history_invariance_v1"
VARIANTS = ("expert", "zero", "reverse", "donor_same_age", "noise_0.5")
HISTORY_CHANGED_THRESHOLD = 1e-8
DEFAULT_DATA = REPO_ROOT / "DiT_train/data_collection/runs/ageext_expert_600_s42"
DEFAULT_CHECKPOINT = REPO_ROOT / "DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt"
DEFAULT_PROCESSOR = REPO_ROOT.parent / "models/generalist"
CSV_FIELDS = (
    "sample_id", "condition_id", "trajectory_id", "task", "age", "variant",
    "available", "unavailable_reason", "donor_sample_id", "donor_condition_id",
    "donor_task", "donor_age", "history_l2_delta", "history_rmse_delta",
    "history_max_abs_delta", "history_changed", "prediction_max_abs_delta",
    "prediction_rmse_delta", "prediction_exact_equal", "x0_max_abs_delta",
    "x0_chunk_rmse_delta", "first_action_ee6_delta_l2",
    "first_action_ee6_delta_rmse", "first_action_gripper_changed", "loss",
    "diffusion_noise_mse", "first_action_ee6_rmse_to_target",
    "first_action_gripper_accuracy", "loss_delta_vs_expert",
    "first_action_rmse_delta_vs_expert", "diffusion_timestep", "noise_seed",
    "noise_sha256",
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> None:
    field_list = list(fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in field_list})


def stable_hex(*parts: Any) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(base_seed: int, *parts: Any) -> int:
    value = int.from_bytes(bytes.fromhex(stable_hex(*parts))[:8], "big")
    return (int(base_seed) + value) % (2**63 - 1)


def parse_ages(value: str) -> tuple[int, ...]:
    try:
        ages = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ages must be comma-separated integers") from exc
    if not ages or any(age not in AGES for age in ages):
        raise argparse.ArgumentTypeError("--ages must be a non-empty subset of 0..11")
    return ages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processor_path", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--ages", type=parse_ages, default=tuple(AGES))
    parser.add_argument("--max_samples_per_age", type=int, default=10, help="0 uses all selected-split samples for each age")
    parser.add_argument("--selection_seed", type=int, default=20260818)
    parser.add_argument("--corruption_seed", type=int, default=20260819)
    parser.add_argument("--validation_noise_seed", type=int, default=20260810)
    parser.add_argument("--diffusion_timestep", type=int, default=50)
    parser.add_argument("--noise_scale", type=float, default=0.5)
    parser.add_argument("--invariance_atol", type=float, default=1e-6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Check contracts and selection without loading processor/model")
    parser.add_argument("--preflight_only", action="store_true", help="Evaluate one deterministic sample at ages 0,7,8,11")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("--run_name must be one non-empty path component")
    if args.max_samples_per_age < 0:
        raise ValueError("--max_samples_per_age must be non-negative")
    if not 0 <= args.diffusion_timestep < 100:
        raise ValueError("--diffusion_timestep must be in [0, 99]")
    if args.noise_scale < 0 or not math.isfinite(args.noise_scale):
        raise ValueError("--noise_scale must be finite and non-negative")
    if args.invariance_atol < 0 or not math.isfinite(args.invariance_atol):
        raise ValueError("--invariance_atol must be finite and non-negative")
    if args.dry_run and args.preflight_only:
        raise ValueError("--dry_run and --preflight_only are mutually exclusive")
    for path, kind in ((args.data_dir, "data directory"), (args.processor_path, "processor directory")):
        if not path.expanduser().resolve().is_dir():
            raise FileNotFoundError(f"Missing {kind}: {path}")
    if not args.checkpoint.expanduser().resolve().is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")


def prepare_run_dir(run_name: str, overwrite: bool) -> Path:
    run_dir = (RUNS_ROOT / run_name).resolve()
    if run_dir.parent != RUNS_ROOT.resolve():
        raise ValueError(f"Run directory escapes experiment runs root: {run_dir}")
    known = {
        "manifest.json", "static_audit.json", "sample_results.jsonl",
        "sample_results.csv", "summary.json", "age_summary.csv",
    }
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in known:
            target = run_dir / name
            if target.is_file():
                target.unlink()
    return run_dir


def selected_rows(dataset: AgeExtendedExpertDataset, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.preflight_only:
        wanted = (0, 7, 8, 11)
        result = []
        for age in wanted:
            candidates = [row for row in dataset.samples if int(row["slow_age"]) == age]
            if not candidates:
                raise ValueError(f"Preflight requires an age {age} {args.split} sample")
            result.append(min(candidates, key=lambda row: stable_hex(args.selection_seed, row["sample_id"])))
        return result
    result = []
    for age in args.ages:
        candidates = [row for row in dataset.samples if int(row["slow_age"]) == age]
        candidates.sort(key=lambda row: stable_hex(args.selection_seed, row["sample_id"]))
        if args.max_samples_per_age:
            candidates = candidates[: args.max_samples_per_age]
        result.extend(candidates)
    if not result:
        raise ValueError("Selection produced no samples")
    return result


def select_donors(rows: list[dict[str, Any]], selection_seed: int) -> dict[str, dict[str, Any] | None]:
    donors: dict[str, dict[str, Any] | None] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        age = int(row["slow_age"])
        same_task_condition = [
            candidate for candidate in rows
            if candidate["sample_id"] != sample_id
            and int(candidate["slow_age"]) == age
            and candidate.get("task") == row.get("task")
            and candidate.get("condition_id") != row.get("condition_id")
        ]
        same_age = [
            candidate for candidate in rows
            if candidate["sample_id"] != sample_id and int(candidate["slow_age"]) == age
        ]
        pool = same_task_condition or same_age
        donors[sample_id] = min(
            pool,
            key=lambda candidate: stable_hex(selection_seed, sample_id, candidate["sample_id"]),
        ) if pool else None
    return donors


def channel_std(rows: list[dict[str, Any]]) -> torch.Tensor:
    histories = torch.as_tensor(
        np.asarray([row["hist_action_before"] for row in rows], dtype=np.float32)
    )
    if histories.ndim != 3 or tuple(histories.shape[1:]) != (4, ACTION_DIM):
        raise ValueError(f"Selected history array has invalid shape {tuple(histories.shape)}")
    return histories.reshape(-1, ACTION_DIM).std(dim=0, unbiased=False)


def donor_report(rows: list[dict[str, Any]], donors: Mapping[str, dict[str, Any] | None]) -> dict[str, Any]:
    unavailable = [str(row["sample_id"]) for row in rows if donors[str(row["sample_id"])] is None]
    return {
        "selected_samples": len(rows),
        "donor_available": len(rows) - len(unavailable),
        "donor_unavailable": len(unavailable),
        "unavailable_sample_ids": unavailable,
    }


def static_audit(policy: torch.nn.Module, checkpoint: Mapping[str, Any], load_audit: Mapping[str, Any]) -> dict[str, Any]:
    import prismatic.models.policy.diffusion_policy as diffusion_policy_module
    import prismatic.models.policy.diffusion_transformer as diffusion_transformer_module

    terms = ("hist", "history", "hist_act_embed", "history_adapter")
    policy_keys = sorted(key for key in policy.state_dict() if any(term in key.lower() for term in terms))
    checkpoint_keys = sorted(
        str(key) for key in checkpoint
        if str(key).startswith("ema_model.") and any(term in str(key).lower() for term in terms)
    )
    history_adapter = getattr(policy.model, "history_adapter", None)
    with_hist = int(getattr(policy.model, "with_hist_action_num"))
    return {
        "status": "complete",
        "policy_model_with_hist_action_num": with_hist,
        "policy_model_history_adapter_is_none": history_adapter is None,
        "policy_model_history_adapter_type": None if history_adapter is None else f"{type(history_adapter).__module__}.{type(history_adapter).__qualname__}",
        "policy_history_related_state_dict_keys": policy_keys,
        "checkpoint_ema_history_related_keys": checkpoint_keys,
        "checkpoint_load_missing_keys": list(load_audit["missing_keys"]),
        "checkpoint_load_unexpected_keys": list(load_audit["unexpected_keys"]),
        "checkpoint_loaded_from": load_audit["loaded_from"],
        "diffusion_policy_file": str(Path(diffusion_policy_module.__file__).resolve()),
        "diffusion_transformer_file": str(Path(diffusion_transformer_module.__file__).resolve()),
        "static_history_path_active": not (with_hist == 0 and history_adapter is None),
        "decision_rule": "inactive iff with_hist_action_num == 0 and history_adapter is None",
    }


def dry_static_audit() -> dict[str, Any]:
    return {
        "status": "not_run_dry_run",
        "static_history_path_active": None,
        "reason": "--dry_run intentionally does not load the policy or checkpoint",
    }


def make_manifest(
    args: argparse.Namespace,
    dataset: AgeExtendedExpertDataset,
    rows: list[dict[str, Any]],
    donors: Mapping[str, dict[str, Any] | None],
    std: torch.Tensor,
    static: Mapping[str, Any],
    *,
    status: str,
    device: torch.device | None,
) -> dict[str, Any]:
    counts = Counter(int(row["slow_age"]) for row in rows)
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_path = args.data_dir.expanduser().resolve() / "manifest.json"
    cuda_device = None
    if device is not None and device.type == "cuda":
        cuda_device = torch.cuda.get_device_name(device)
    return {
        "experiment_schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": "dry_run" if args.dry_run else "preflight_only" if args.preflight_only else "evaluation",
        "git_commit": git_commit(),
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_dataset_path": str(args.data_dir.expanduser().resolve()),
        "source_dataset_manifest_sha256": sha256_file(manifest_path),
        "source_dataset_schema_version": dataset.manifest.get("schema_version"),
        "processor_path": str(args.processor_path.expanduser().resolve()),
        "split": args.split,
        "ages": list(args.ages),
        "selected_sample_counts_per_age": {str(age): counts.get(age, 0) for age in AGES},
        "selected_sample_count": len(rows),
        "selection_seed": args.selection_seed,
        "corruption_seed": args.corruption_seed,
        "validation_noise_seed": args.validation_noise_seed,
        "diffusion_timestep": args.diffusion_timestep,
        "invariance_atol": args.invariance_atol,
        "history_changed_threshold": HISTORY_CHANGED_THRESHOLD,
        "corruptions": {
            "expert": "unmodified hist_action_before",
            "zero": "zeros_like(expert)",
            "reverse": "flip expert along history time dimension 0",
            "donor_same_age": "deterministic selected-split donor: same task+age+different condition, else same age+different sample, else unavailable",
            "noise_0.5": f"expert + {args.noise_scale} * selected-history channel_std * deterministic N(0,1)",
        },
        "noise_scale": args.noise_scale,
        "selected_history_channel_std": [float(value) for value in std.tolist()],
        "donor_selection": donor_report(rows, donors),
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": None if device is None else str(device),
            "cuda_device_name": cuda_device,
        },
        "static_history_audit_summary": {
            "status": static.get("status"),
            "static_history_path_active": static.get("static_history_path_active"),
            "with_hist_action_num": static.get("policy_model_with_hist_action_num"),
            "history_adapter_is_none": static.get("policy_model_history_adapter_is_none"),
        },
    }


def tensor_delta(value: torch.Tensor, baseline: torch.Tensor) -> tuple[float, float]:
    delta = value.float() - baseline.float()
    return float(delta.abs().max().cpu()), float(torch.sqrt(torch.mean(delta.square())).cpu())


def history_delta(value: torch.Tensor, baseline: torch.Tensor) -> dict[str, Any]:
    delta = value.float() - baseline.float()
    max_abs = float(delta.abs().max().cpu())
    return {
        "history_l2_delta": float(torch.linalg.vector_norm(delta).cpu()),
        "history_rmse_delta": float(torch.sqrt(torch.mean(delta.square())).cpu()),
        "history_max_abs_delta": max_abs,
        "history_changed": max_abs > HISTORY_CHANGED_THRESHOLD,
    }


def make_history(
    variant: str,
    expert: torch.Tensor,
    row: Mapping[str, Any],
    donor: Mapping[str, Any] | None,
    std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor | None:
    if variant == "expert":
        return expert.clone()
    if variant == "zero":
        return torch.zeros_like(expert)
    if variant == "reverse":
        return torch.flip(expert, dims=(1,)).clone()
    if variant == "donor_same_age":
        if donor is None:
            return None
        value = torch.as_tensor(donor["hist_action_before"], dtype=torch.float32, device=device)
        return value.unsqueeze(0).clone()
    if variant == "noise_0.5":
        seed = stable_seed(args.corruption_seed, row["sample_id"], variant)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        epsilon = torch.randn(expert.shape, generator=generator, dtype=torch.float32)
        perturbation = args.noise_scale * std.view(1, 1, ACTION_DIM) * epsilon
        return expert + perturbation.to(device)
    raise ValueError(f"Unknown variant {variant}")


@torch.no_grad()
def forward_variant(
    policy: torch.nn.Module,
    batch: Mapping[str, Any],
    history: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    cond_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
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
            hist_action=history.float(),
            decoupled_loss=False,
            noise=noise,
            timesteps=timesteps,
            cond_mask=cond_mask,
            return_details=True,
        )
    prediction = details["prediction"].float()
    target = details["target"].float()
    trajectory = batch["raw_action"].float()
    noisy = policy.noise_scheduler.add_noise(trajectory, noise, timesteps)
    timestep = int(timesteps.item())
    alpha_bar = policy.noise_scheduler.alphas_cumprod[timestep].to(device=device, dtype=torch.float32)
    x0_hat = (noisy.float() - torch.sqrt(1.0 - alpha_bar) * prediction) / torch.sqrt(alpha_bar)
    first_error = x0_hat[0, 0, :6] - trajectory[0, 0, :6]
    predicted_gripper = bool(x0_hat[0, 0, 6].item() >= 0.0)
    target_gripper = bool(trajectory[0, 0, 6].item() >= 0.0)
    return {
        "prediction": prediction,
        "target": target,
        "x0_hat": x0_hat,
        "loss": float(details["loss"].float().cpu()),
        "diffusion_noise_mse": float(torch.mean((prediction - target).square()).cpu()),
        "first_action_ee6_rmse_to_target": float(torch.sqrt(torch.mean(first_error.square())).cpu()),
        "first_action_gripper_accuracy": float(predicted_gripper == target_gripper),
        "predicted_gripper": predicted_gripper,
    }


def unavailable_row(row: Mapping[str, Any], variant: str, reason: str) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "condition_id": str(row["condition_id"]),
        "trajectory_id": str(row["trajectory_id"]),
        "task": str(row.get("task", "")),
        "age": int(row["slow_age"]),
        "variant": variant,
        "available": False,
        "unavailable_reason": reason,
        "history_changed": None,
    }


def evaluate(
    policy: torch.nn.Module,
    dataset: AgeExtendedExpertDataset,
    rows: list[dict[str, Any]],
    donors: Mapping[str, dict[str, Any] | None],
    std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    index_by_id = {str(row["sample_id"]): index for index, row in enumerate(dataset.samples)}
    results: list[dict[str, Any]] = []
    policy.eval()
    for selected_index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        item = dataset[index_by_id[sample_id]]
        cpu_batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else [value] if key == "instruction" else value
            for key, value in item.items()
        }
        batch = move_batch(cpu_batch, device)
        expert_history = batch["hist_action"].float().clone()
        noise, noise_seed, noise_sha = deterministic_noise(sample_id, args.validation_noise_seed, device)
        timesteps = torch.tensor([args.diffusion_timestep], dtype=torch.long, device=device)
        cond_mask = torch.ones((1, 1), dtype=torch.float32, device=device)
        baseline = forward_variant(policy, batch, expert_history, noise, timesteps, cond_mask, device)
        donor = donors[sample_id]
        for variant in VARIANTS:
            history = make_history(variant, expert_history, row, donor, std, args, device)
            if history is None:
                results.append(unavailable_row(row, variant, "no different selected validation sample of the same age"))
                continue
            current = baseline if variant == "expert" else forward_variant(
                policy, batch, history, noise, timesteps, cond_mask, device
            )
            pred_max, pred_rmse = tensor_delta(current["prediction"], baseline["prediction"])
            x0_max, x0_rmse = tensor_delta(current["x0_hat"], baseline["x0_hat"])
            first_delta = current["x0_hat"][0, 0, :6] - baseline["x0_hat"][0, 0, :6]
            result = {
                "sample_id": sample_id,
                "condition_id": str(row["condition_id"]),
                "trajectory_id": str(row["trajectory_id"]),
                "task": str(row.get("task", "")),
                "age": int(row["slow_age"]),
                "variant": variant,
                "available": True,
                "unavailable_reason": None,
                "donor_sample_id": None if donor is None or variant != "donor_same_age" else str(donor["sample_id"]),
                "donor_condition_id": None if donor is None or variant != "donor_same_age" else str(donor["condition_id"]),
                "donor_task": None if donor is None or variant != "donor_same_age" else str(donor.get("task", "")),
                "donor_age": None if donor is None or variant != "donor_same_age" else int(donor["slow_age"]),
                **history_delta(history, expert_history),
                "prediction_max_abs_delta": pred_max,
                "prediction_rmse_delta": pred_rmse,
                "prediction_exact_equal": bool(torch.equal(current["prediction"], baseline["prediction"])),
                "x0_max_abs_delta": x0_max,
                "x0_chunk_rmse_delta": x0_rmse,
                "first_action_ee6_delta_l2": float(torch.linalg.vector_norm(first_delta).cpu()),
                "first_action_ee6_delta_rmse": float(torch.sqrt(torch.mean(first_delta.square())).cpu()),
                "first_action_gripper_changed": bool(current["predicted_gripper"] != baseline["predicted_gripper"]),
                "loss": current["loss"],
                "diffusion_noise_mse": current["diffusion_noise_mse"],
                "first_action_ee6_rmse_to_target": current["first_action_ee6_rmse_to_target"],
                "first_action_gripper_accuracy": current["first_action_gripper_accuracy"],
                "loss_delta_vs_expert": current["loss"] - baseline["loss"],
                "first_action_rmse_delta_vs_expert": current["first_action_ee6_rmse_to_target"] - baseline["first_action_ee6_rmse_to_target"],
                "diffusion_timestep": args.diffusion_timestep,
                "noise_seed": noise_seed,
                "noise_sha256": noise_sha,
            }
            results.append(result)
        print(f"[{selected_index}/{len(rows)}] evaluated {sample_id}", flush=True)
    return results


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("available")]
    changed = [row for row in available if row.get("history_changed")]
    return {
        "n": len(available),
        "n_unavailable": len(rows) - len(available),
        "n_history_changed": len(changed),
        "prediction_max_abs_delta": maximum(available, "prediction_max_abs_delta"),
        "prediction_rmse_delta_mean": mean(available, "prediction_rmse_delta"),
        "prediction_rmse_delta_max": maximum(available, "prediction_rmse_delta"),
        "x0_max_abs_delta": maximum(available, "x0_max_abs_delta"),
        "x0_chunk_rmse_delta_mean": mean(available, "x0_chunk_rmse_delta"),
        "x0_chunk_rmse_delta_max": maximum(available, "x0_chunk_rmse_delta"),
        "diffusion_noise_mse_mean": mean(available, "diffusion_noise_mse"),
        "first_action_ee6_rmse_to_target_mean": mean(available, "first_action_ee6_rmse_to_target"),
        "first_action_gripper_accuracy_mean": mean(available, "first_action_gripper_accuracy"),
        "loss_delta_vs_expert_mean": mean(available, "loss_delta_vs_expert"),
        "first_action_rmse_delta_vs_expert_mean": mean(available, "first_action_rmse_delta_vs_expert"),
        "prediction_exact_equal_n": sum(bool(row.get("prediction_exact_equal")) for row in available),
    }


def age_summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, Any]] = [(f"age_{age}", lambda row, age=age: int(row["age"]) == age) for age in AGES]
    groups.extend((
        ("age_0_7", lambda row: 0 <= int(row["age"]) <= 7),
        ("age_8_11", lambda row: 8 <= int(row["age"]) <= 11),
    ))
    rows = []
    for variant in VARIANTS:
        variant_rows = [row for row in results if row["variant"] == variant]
        for group_name, predicate in groups:
            rows.append({"variant": variant, "age_group": group_name, **aggregate_group([row for row in variant_rows if predicate(row)])})
    return rows


def witness(row: Mapping[str, Any], metric: str) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"], "age": row["age"], "task": row["task"],
        "variant": row["variant"], "history_max_abs_delta": row["history_max_abs_delta"],
        metric: row[metric],
    }


def summarize(results: list[dict[str, Any]], args: argparse.Namespace, static: Mapping[str, Any]) -> dict[str, Any]:
    available = [row for row in results if row.get("available")]
    changed = [row for row in available if row.get("variant") != "expert" and row.get("history_changed")]
    pred_winner = max(changed, key=lambda row: float(row["prediction_max_abs_delta"])) if changed else None
    x0_winner = max(changed, key=lambda row: float(row["x0_max_abs_delta"])) if changed else None
    pred_max = None if pred_winner is None else float(pred_winner["prediction_max_abs_delta"])
    x0_max = None if x0_winner is None else float(x0_winner["x0_max_abs_delta"])
    invariant = bool(changed) and pred_max <= args.invariance_atol and x0_max <= args.invariance_atol
    age_rows = age_summary_rows(results)
    age_lookup = {(row["variant"], row["age_group"]): row for row in age_rows}
    comparison = {
        variant: {
            "age_0_7_prediction_max_abs_delta": age_lookup[(variant, "age_0_7")]["prediction_max_abs_delta"],
            "age_8_11_prediction_max_abs_delta": age_lookup[(variant, "age_8_11")]["prediction_max_abs_delta"],
            "age_0_7_x0_max_abs_delta": age_lookup[(variant, "age_0_7")]["x0_max_abs_delta"],
            "age_8_11_x0_max_abs_delta": age_lookup[(variant, "age_8_11")]["x0_max_abs_delta"],
        }
        for variant in VARIANTS if variant != "expert"
    }
    return {
        "status": "complete",
        "primary_endpoint": "paired output change relative to expert-history baseline",
        "static_history_path_active": static.get("static_history_path_active"),
        "invariance_atol": args.invariance_atol,
        "total_n": len(available),
        "unavailable_n": len(results) - len(available),
        "history_changed_n": len(changed),
        "global_prediction_max_abs_delta_on_changed_inputs": pred_max,
        "global_x0_max_abs_delta_on_changed_inputs": x0_max,
        "dynamic_history_invariant": invariant,
        "dynamic_test_valid": bool(changed),
        "prediction_exact_equal_n_on_changed_inputs": sum(bool(row["prediction_exact_equal"]) for row in changed),
        "prediction_exact_equal_all_on_changed_inputs": bool(changed) and all(bool(row["prediction_exact_equal"]) for row in changed),
        "max_prediction_delta_witness": None if pred_winner is None else witness(pred_winner, "prediction_max_abs_delta"),
        "max_x0_delta_witness": None if x0_winner is None else witness(x0_winner, "x0_max_abs_delta"),
        "by_variant": {variant: aggregate_group([row for row in results if row["variant"] == variant]) for variant in VARIANTS},
        "age_8_11_vs_age_0_7": comparison,
        "interpretation": (
            "Current M1 is history-invariant; the expert-vs-policy mismatch cannot currently enter through the explicit hist_action argument."
            if invariant else
            "History-dependent output was detected; audit runtime architecture, checkpoint, and import paths before attributing causal history dependence."
        ),
    }


def dry_summary(rows: list[dict[str, Any]], donors: Mapping[str, dict[str, Any] | None]) -> dict[str, Any]:
    return {
        "status": "dry_run_complete",
        "dynamic_history_invariant": None,
        "dynamic_test_valid": False,
        "selected_sample_count": len(rows),
        "donor_selection": donor_report(rows, donors),
        "note": "No processor, model, checkpoint tensors, observations, or diffusion forwards were loaded.",
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.processor_path = args.processor_path.expanduser().resolve()
    run_dir = prepare_run_dir(args.run_name, args.overwrite)
    seed_everything(args.selection_seed)

    dataset = AgeExtendedExpertDataset(args.data_dir, args.split, processor=None)
    rows = selected_rows(dataset, args)
    donors = select_donors(rows, args.selection_seed)
    std = channel_std(rows)

    if args.dry_run:
        static = dry_static_audit()
        manifest = make_manifest(args, dataset, rows, donors, std, static, status="dry_run_complete", device=None)
        write_json(run_dir / "static_audit.json", static)
        write_jsonl(run_dir / "sample_results.jsonl", [])
        write_csv(run_dir / "sample_results.csv", [], CSV_FIELDS)
        write_json(run_dir / "summary.json", dry_summary(rows, donors))
        write_csv(run_dir / "age_summary.csv", [], ("variant", "age_group", "n"))
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"run_dir": str(run_dir), "status": "dry_run_complete", **donor_report(rows, donors)}, indent=2))
        return

    device = choose_device(args.device)
    processor = load_processor(args.processor_path)
    dataset.processor = processor
    policy = build_policy(device)
    checkpoint, load_audit = load_baseline_ema(policy, args.checkpoint)
    policy.eval()
    static = static_audit(policy, checkpoint, load_audit)
    write_json(run_dir / "static_audit.json", static)
    write_json(run_dir / "manifest.json", make_manifest(args, dataset, rows, donors, std, static, status="running", device=device))
    results = evaluate(policy, dataset, rows, donors, std, args, device)
    summary = summarize(results, args, static)
    age_rows = age_summary_rows(results)
    write_jsonl(run_dir / "sample_results.jsonl", results)
    write_csv(run_dir / "sample_results.csv", results, CSV_FIELDS)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "age_summary.csv", age_rows, age_rows[0].keys())
    write_json(run_dir / "manifest.json", make_manifest(args, dataset, rows, donors, std, static, status="complete", device=device))
    print(json.dumps({
        "run_dir": str(run_dir),
        "static_history_path_active": static["static_history_path_active"],
        "dynamic_history_invariant": summary["dynamic_history_invariant"],
        "prediction_max_abs_delta": summary["global_prediction_max_abs_delta_on_changed_inputs"],
        "x0_max_abs_delta": summary["global_x0_max_abs_delta_on_changed_inputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
