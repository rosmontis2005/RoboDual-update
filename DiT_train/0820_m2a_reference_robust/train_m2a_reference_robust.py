#!/usr/bin/env python3
"""M2a Reference-Robust / Autonomous Specialist training.

Uses the immutable age-extended expert dataset. Counterfactual reference views
are constructed in memory; this module never loads OpenVLA or a simulator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import tempfile
from collections import Counter, OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DiT_train.train_age_extended_expert import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    AGES,
    SPLITS,
    AgeExtendedExpertDataset,
    PolicyEMA,
    autocast_context,
    choose_device,
    cosine_schedule,
    dataset_statistics,
    extract_prefixed_policy_state,
    freeze_invariants,
    git_commit,
    hidden_length_distribution,
    load_processor,
    metric_group,
    move_batch,
    read_jsonl,
    seed_everything,
    serialized_args,
    set_train_mode,
    sha256_file,
    torch_load_cpu,
)

MODEL_VARIANT = "m2a_reference_robust_v1"
ADAPTER_KEY = "model.ref_valid_embedder.weight"
DEFAULT_DATA_DIR = REPO_ROOT / "DiT_train/data_collection/runs/ageext_expert_600_s42"
DEFAULT_PROCESSOR_PATH = REPO_ROOT.parent / "models/generalist"
DEFAULT_M1_CHECKPOINT = REPO_ROOT / "DiT_train/runs/ageext_m1_long1500_b97f005/specialist_ema_step_001500.pt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs/m2a_reference_robust_s42"
VALIDATION_MODES = ("persisted", "zero_ref_hidden_kept")
SELECTABLE_COUNTERFACTUAL_VIEWS = ("zero_ref_hidden_kept", "shortened_reference")
PARITY_TOLERANCE = 1e-6
ROUND_TRIP_TOLERANCE = 1e-6


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class M2aDataset(AgeExtendedExpertDataset):
    """M1 split-preserving dataset plus an explicit persisted validity mask."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        count = int(item["ref_valid_count"])
        mask = torch.zeros(ACTION_HORIZON, dtype=torch.float32)
        mask[:count] = 1.0
        invalid = item["ref_action"][count:]
        if torch.count_nonzero(invalid).item() != 0:
            raise AssertionError(f"{item['sample_id']}: invalid persisted ref positions are not exact zero")
        row_mask = self.samples[index].get("ref_valid_mask")
        if row_mask is not None and list(map(bool, row_mask)) != list(map(bool, mask.tolist())):
            raise AssertionError(f"{item['sample_id']}: persisted ref_valid_mask disagrees with ref_valid_count")
        item["ref_valid_mask"] = mask
        return item


def build_policy(device: torch.device, *, use_ref_validity: bool) -> torch.nn.Module:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy

    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    return DiffusionDiTImagePolicy(
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
        use_ref_validity=use_ref_validity,
    ).to(device)


def load_m1_checkpoint(
    policy: torch.nn.Module,
    path: str | Path,
    *,
    allow_m2a_adapter_missing: bool,
) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch_load_cpu(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    state = extract_prefixed_policy_state(checkpoint, "ema_model")
    incompatible = policy.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed = [ADAPTER_KEY] if allow_m2a_adapter_missing else []
    if missing != allowed or unexpected:
        raise RuntimeError(
            "M1 checkpoint audit failed: "
            f"missing={missing}, allowed_missing={allowed}, unexpected={unexpected}"
        )
    adapter_zero = None
    if allow_m2a_adapter_missing:
        adapter = policy.model.ref_valid_embedder.weight
        adapter_zero = bool(torch.count_nonzero(adapter.detach()).item() == 0)
        if not adapter_zero:
            raise AssertionError("M2a ref validity adapter is not exact-zero after M1 initialization")
    audit = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "loaded_from": "ema_model",
        "ema_tensor_count": len(state),
        "shared_parameter_tensor_count": len(state),
        "missing_keys": missing,
        "allowed_missing_keys": allowed,
        "unexpected_keys": unexpected,
        "adapter_exact_zero": adapter_zero,
    }
    return OrderedDict(checkpoint), audit


def invariant_view_fields(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: batch[key] for key in (
            "current_rgb", "previous_rgb", "depth_image", "gripper_image",
            "depth_gripper", "proprio", "raw_action", "hist_action", "instruction",
        )
    }


def persisted_view(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **invariant_view_fields(batch),
        "name": "persisted",
        "ref_action": batch["ref_action"],
        "ref_valid_mask": batch["ref_valid_mask"],
        "slow_hidden": batch["slow_hidden"],
        "shortened_k": None,
    }


def make_named_view(batch: Mapping[str, Any], name: str, *, k: int | None = None) -> dict[str, Any]:
    if name == "persisted":
        return persisted_view(batch)
    count = int(batch["ref_valid_count"].item())
    ref = batch["ref_action"]
    mask = batch["ref_valid_mask"]
    hidden = batch["slow_hidden"]
    if name == "zero_ref_hidden_kept":
        return {
            **invariant_view_fields(batch),
            "name": name, "ref_action": torch.zeros_like(ref),
            "ref_valid_mask": torch.zeros_like(mask), "slow_hidden": hidden,
            "shortened_k": 0,
        }
    if name != "shortened_reference":
        raise ValueError(f"Unknown view {name!r}")
    if count <= 0:
        raise ValueError("shortened_reference is not applicable when ref_valid_count=0")
    if k is None:
        raise ValueError("shortened_reference requires k")
    if not 0 <= int(k) < count:
        raise ValueError(f"shortened k must satisfy 0 <= k < {count}, got {k}")
    shortened_ref = torch.zeros_like(ref)
    shortened_mask = torch.zeros_like(mask)
    if k:
        shortened_ref[:, :k] = ref[:, :k]
        shortened_mask[:, :k] = 1.0
    return {
        **invariant_view_fields(batch),
        "name": name, "ref_action": shortened_ref,
        "ref_valid_mask": shortened_mask, "slow_hidden": hidden,
        "shortened_k": int(k),
    }


def choose_counterfactual_view(
    batch: Mapping[str, Any], args: argparse.Namespace, rng: random.Random,
) -> dict[str, Any] | None:
    count = int(batch["ref_valid_count"].item())
    if count == 0:
        return None
    if rng.random() < args.zero_ref_kept_probability:
        return make_named_view(batch, "zero_ref_hidden_kept")
    return make_named_view(batch, "shortened_reference", k=rng.randrange(count))


def assert_view_contracts(
    batch: Mapping[str, Any], base: Mapping[str, Any], counter: Mapping[str, Any],
) -> dict[str, Any]:
    count = int(batch["ref_valid_count"].item())
    expected_mask = torch.zeros_like(batch["ref_valid_mask"])
    expected_mask[:, :count] = 1.0
    checks = {
        "persisted_mask_matches_count": torch.equal(base["ref_valid_mask"], expected_mask),
        "persisted_invalid_exact_zero": torch.count_nonzero(base["ref_action"][:, count:]).item() == 0,
        "persisted_hidden_exact": torch.equal(base["slow_hidden"], batch["slow_hidden"]),
        "counterfactual_hidden_exact": torch.equal(counter["slow_hidden"], batch["slow_hidden"]),
        "observations_unchanged": all(
            torch.equal(base[key], batch[key]) and torch.equal(counter[key], batch[key])
            for key in ("current_rgb", "previous_rgb", "depth_image", "gripper_image", "depth_gripper")
        ),
        "proprio_unchanged": torch.equal(base["proprio"], batch["proprio"]) and torch.equal(counter["proprio"], batch["proprio"]),
        "target_unchanged": torch.equal(base["raw_action"], batch["raw_action"]) and torch.equal(counter["raw_action"], batch["raw_action"]),
        "history_unchanged": torch.equal(base["hist_action"], batch["hist_action"]) and torch.equal(counter["hist_action"], batch["hist_action"]),
        "instruction_unchanged": base["instruction"] == batch["instruction"] and counter["instruction"] == batch["instruction"],
    }
    if counter["name"].startswith("zero_ref"):
        checks.update({
            "zero_ref_exact_zero": torch.count_nonzero(counter["ref_action"]).item() == 0,
            "zero_mask_all_false": torch.count_nonzero(counter["ref_valid_mask"]).item() == 0,
        })
    if counter["name"] == "zero_ref_hidden_kept":
        checks["hidden_kept_exact"] = torch.equal(counter["slow_hidden"], batch["slow_hidden"])
    if counter["name"] == "shortened_reference":
        k = int(counter["shortened_k"])
        checks.update({
            "shortened_k_not_future": 0 <= k < count,
            "shortened_prefix_equal": torch.equal(counter["ref_action"][:, :k], batch["ref_action"][:, :k]),
            "shortened_suffix_zero": torch.count_nonzero(counter["ref_action"][:, k:]).item() == 0,
            "shortened_mask_exact": int(counter["ref_valid_mask"].sum().item()) == k,
        })
    failed = sorted(key for key, passed in checks.items() if not bool(passed))
    if failed:
        raise AssertionError(f"Counterfactual view contract failed: {failed}")
    return {key: bool(value) for key, value in checks.items()}


def snapshot_rng() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def loss_details(
    policy: torch.nn.Module,
    batch: Mapping[str, Any],
    view: Mapping[str, Any],
    *,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    cond_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    return policy.compute_loss(
        trajectory=view["raw_action"].float(),
        ref_action=view["ref_action"].float(),
        ref_valid_mask=view["ref_valid_mask"].float(),
        action_cond=view["slow_hidden"].float(),
        obs=(view["current_rgb"].float(), view["previous_rgb"].float()),
        depth_obs=view["depth_image"].float(),
        gripper_obs=(view["gripper_image"].float(), view["depth_gripper"].float()),
        tactile_obs=None,
        lang=view["instruction"],
        proprio=view["proprio"].float(),
        hist_action=view["hist_action"].float(),
        decoupled_loss=False,
        noise=noise,
        timesteps=timesteps,
        cond_mask=cond_mask,
        return_details=True,
    )


def weighted_supervised_loss(
    persisted_loss: torch.Tensor,
    counterfactual_loss: torch.Tensor | None,
    persisted_weight: float,
    counterfactual_weight: float,
) -> tuple[torch.Tensor, float]:
    if counterfactual_loss is None:
        return persisted_loss, 1.0
    denominator = float(persisted_weight + counterfactual_weight)
    if denominator <= 0:
        raise ValueError("persisted_loss_weight + counterfactual_loss_weight must be positive")
    supervised = (
        persisted_weight * persisted_loss + counterfactual_weight * counterfactual_loss
    ) / denominator
    return supervised, denominator


def paired_loss(
    policy: torch.nn.Module,
    batch: Mapping[str, Any],
    counter: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, torch.Tensor] | None, dict[str, Any]]:
    base = persisted_view(batch)
    count = int(batch["ref_valid_count"].item())
    if counter is None and count != 0:
        raise AssertionError("C>0 samples require one selectable counterfactual view")
    if counter is not None and count == 0:
        raise AssertionError("C==0 samples must use the single persisted-only loss path")
    noise = torch.randn_like(batch["raw_action"], dtype=torch.float32)
    timesteps = torch.randint(
        0, policy.noise_scheduler.config.num_train_timesteps,
        (batch["raw_action"].shape[0],), device=batch["raw_action"].device,
    ).long()
    cond_mask = (
        torch.rand((batch["raw_action"].shape[0], 1), device=batch["raw_action"].device)
        > policy.cond_drop_chance
    ).float()

    if counter is None:
        persisted = loss_details(policy, batch, base, noise=noise, timesteps=timesteps, cond_mask=cond_mask)
        counterfactual = None
        supervised, denominator = weighted_supervised_loss(
            persisted["loss"], None,
            args.persisted_loss_weight, args.counterfactual_loss_weight,
        )
        consistency = torch.zeros((), device=supervised.device, dtype=supervised.dtype)
        pair_audit = {
            "has_counterfactual": False,
            "paired_rng_not_applicable": True,
            "persisted_ref_exact_zero": torch.count_nonzero(base["ref_action"]).item() == 0,
            "persisted_mask_all_zero": torch.count_nonzero(base["ref_valid_mask"]).item() == 0,
            "persisted_hidden_exact": torch.equal(base["slow_hidden"], batch["slow_hidden"]),
            "noise_sha256": tensor_sha256(noise),
            "timestep": int(timesteps.item()),
            "cfg_mask": float(cond_mask.item()),
            "counterfactual_view": None,
            "shortened_k": None,
        }
        if not all(pair_audit[key] for key in (
            "persisted_ref_exact_zero", "persisted_mask_all_zero", "persisted_hidden_exact",
        )):
            raise AssertionError("Single persisted expired-reference contract failed")
    else:
        contracts = assert_view_contracts(batch, base, counter)
        pair_rng = snapshot_rng()
        persisted = loss_details(policy, batch, base, noise=noise, timesteps=timesteps, cond_mask=cond_mask)
        after_first = snapshot_rng()
        restore_rng(pair_rng)
        counterfactual = loss_details(policy, batch, counter, noise=noise, timesteps=timesteps, cond_mask=cond_mask)
        restore_rng(after_first)
        supervised, denominator = weighted_supervised_loss(
            persisted["loss"], counterfactual["loss"],
            args.persisted_loss_weight, args.counterfactual_loss_weight,
        )
        consistency = F.mse_loss(persisted["prediction"], counterfactual["prediction"])
        pair_audit = {
            **contracts,
            "has_counterfactual": True,
            "paired_rng_not_applicable": False,
            "noise_exact_equal": torch.equal(persisted["noise"], counterfactual["noise"]),
            "timestep_exact_equal": torch.equal(persisted["timesteps"], counterfactual["timesteps"]),
            "cfg_mask_exact_equal": torch.equal(persisted["cond_mask"], counterfactual["cond_mask"]),
            "noise_sha256": tensor_sha256(noise),
            "timestep": int(timesteps.item()),
            "cfg_mask": float(cond_mask.item()),
            "counterfactual_view": counter["name"],
            "shortened_k": counter["shortened_k"],
        }
        for key in ("noise_exact_equal", "timestep_exact_equal", "cfg_mask_exact_equal"):
            if not pair_audit[key]:
                raise AssertionError(f"Paired diffusion invariant failed: {key}")

    total = supervised + args.consistency_weight * consistency
    pair_audit.update({
        "supervised_loss": float(supervised.detach().float().cpu()),
        "persisted_loss": float(persisted["loss"].detach().float().cpu()),
        "counterfactual_loss": None if counterfactual is None else float(counterfactual["loss"].detach().float().cpu()),
        "supervised_weight_denominator": denominator,
        "consistency_loss": float(consistency.detach().float().cpu()),
    })
    return total, persisted, counterfactual, pair_audit


def deterministic_noise(sample_id: str, base_seed: int, device: torch.device) -> tuple[torch.Tensor, int, str]:
    stable = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    derived = (int(base_seed) + stable) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    noise = torch.randn((1, ACTION_HORIZON, ACTION_DIM), generator=generator, dtype=torch.float32)
    return noise.to(device), derived, hashlib.sha256(noise.numpy().tobytes()).hexdigest()


def x0_from_prediction(
    policy: torch.nn.Module,
    trajectory: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    noisy = policy.noise_scheduler.add_noise(trajectory, noise, timesteps)
    alpha = policy.noise_scheduler.alphas_cumprod.to(trajectory.device)[timesteps]
    while alpha.ndim < trajectory.ndim:
        alpha = alpha.unsqueeze(-1)
    return (noisy.float() - torch.sqrt(1.0 - alpha.float()) * prediction.float()) / torch.sqrt(alpha.float())


def fixed_indices(dataset: M2aDataset, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("fixed sample count must be positive")
    ordered = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(str(dataset.samples[index]["sample_id"]).encode()).hexdigest(),
    )
    chosen: list[int] = []
    for age in (0, 7, 8, 11):
        match = next((i for i in ordered if int(dataset.samples[i]["slow_age"]) == age), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    for index in ordered:
        if len(chosen) >= count:
            break
        if index not in chosen:
            chosen.append(index)
    return chosen[:count]


@torch.no_grad()
def m1_init_parity(
    m2a: torch.nn.Module,
    checkpoint_path: Path,
    dataset: M2aDataset,
    device: torch.device,
    *,
    sample_count: int,
    timestep: int,
    seed: int,
) -> dict[str, Any]:
    m1 = build_policy(device, use_ref_validity=False)
    freeze_invariants(m1)
    _, m1_load = load_m1_checkpoint(m1, checkpoint_path, allow_m2a_adapter_missing=False)
    m1.eval()
    m2a.eval()
    rows = []
    for index in fixed_indices(dataset, sample_count):
        cpu_batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1)))
        batch = move_batch(cpu_batch, device)
        sample_id = str(cpu_batch["sample_id"][0])
        noise, _, _ = deterministic_noise(sample_id, seed, device)
        timesteps = torch.tensor([timestep], dtype=torch.long, device=device)
        cond_mask = torch.ones((1, 1), dtype=torch.float32, device=device)
        common = dict(
            trajectory=batch["raw_action"].float(), ref_action=batch["ref_action"].float(),
            action_cond=batch["slow_hidden"].float(),
            obs=(batch["current_rgb"].float(), batch["previous_rgb"].float()),
            depth_obs=batch["depth_image"].float(),
            gripper_obs=(batch["gripper_image"].float(), batch["depth_gripper"].float()),
            tactile_obs=None, lang=batch["instruction"], proprio=batch["proprio"].float(),
            hist_action=batch["hist_action"].float(), decoupled_loss=False,
            noise=noise, timesteps=timesteps, cond_mask=cond_mask, return_details=True,
        )
        left = m1.compute_loss(**common)
        right = m2a.compute_loss(**common, ref_valid_mask=batch["ref_valid_mask"].float())
        left_x0 = x0_from_prediction(m1, batch["raw_action"].float(), noise, timesteps, left["prediction"])
        right_x0 = x0_from_prediction(m2a, batch["raw_action"].float(), noise, timesteps, right["prediction"])
        row = {
            "sample_id": sample_id,
            "age": int(batch["age"].item()),
            "prediction_max_abs_delta": float((left["prediction"] - right["prediction"]).abs().max().cpu()),
            "diffusion_loss_abs_delta": float((left["loss"] - right["loss"]).abs().cpu()),
            "x0_first_action_max_abs_delta": float((left_x0[:, 0] - right_x0[:, 0]).abs().max().cpu()),
            "x0_first_action_ee6_max_abs_delta": float((left_x0[:, 0, :6] - right_x0[:, 0, :6]).abs().max().cpu()),
            "x0_first_action_gripper_abs_delta": float((left_x0[:, 0, 6] - right_x0[:, 0, 6]).abs().max().cpu()),
        }
        rows.append(row)
    maxima = {
        key: max(row[key] for row in rows)
        for key in rows[0]
        if key.endswith("delta")
    }
    passed = all(value <= PARITY_TOLERANCE for value in maxima.values())
    if not passed:
        raise AssertionError(f"M1 init parity exceeds {PARITY_TOLERANCE}: {maxima}")
    del m1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "passed": passed,
        "tolerance": PARITY_TOLERANCE,
        "sample_count": len(rows),
        "fixed_timestep": timestep,
        "fixed_seed": seed,
        "m1_checkpoint_loading": m1_load,
        "maxima": maxima,
        "per_sample": rows,
    }


def preflight_all_views(dataset: M2aDataset) -> dict[str, Any]:
    records = []
    for age in (0, 7, 8, 11):
        index = next(i for i, row in enumerate(dataset.samples) if int(row["slow_age"]) == age)
        batch = next(iter(DataLoader(
            torch.utils.data.Subset(dataset, [index]), batch_size=1, num_workers=0,
        )))
        base = persisted_view(batch)
        count = int(batch["ref_valid_count"].item())
        if count == 0:
            planned = choose_counterfactual_view(
                batch,
                argparse.Namespace(zero_ref_kept_probability=0.8),
                random.Random(0),
            )
            checks = {
                "persisted_ref_exact_zero": torch.count_nonzero(base["ref_action"]).item() == 0,
                "persisted_mask_all_zero": torch.count_nonzero(base["ref_valid_mask"]).item() == 0,
                "persisted_hidden_exact": torch.equal(base["slow_hidden"], batch["slow_hidden"]),
                "no_counterfactual_generated": planned is None,
                "only_one_forward_planned": planned is None,
            }
            failed = sorted(key for key, passed in checks.items() if not passed)
            if failed:
                raise AssertionError(f"Expired-reference training-plan contract failed: {failed}")
            records.append({
                "sample_id": str(batch["sample_id"][0]), "age": age,
                "persisted_ref_valid_count": count, "view": None,
                "has_counterfactual": False, "planned_forward_count": 1,
                "shortened_k": None, "checks": checks,
            })
            continue
        names_and_k = [
            ("zero_ref_hidden_kept", None),
            ("shortened_reference", 0),
            ("shortened_reference", count - 1),
        ]
        for name, k in names_and_k:
            view = make_named_view(batch, name, k=k)
            checks = assert_view_contracts(batch, base, view)
            records.append({
                "sample_id": str(batch["sample_id"][0]), "age": age,
                "persisted_ref_valid_count": count, "view": name,
                "shortened_k": k, "checks": checks,
            })
    return {
        "passed": True, "real_dataset_samples": 4, "records": records,
        "shortened_boundaries_checked": ["k=0", "k=C-1"],
        "artificial_zero_ref_ages_0_7_checked": True,
        "expired_ages_8_11_checked": True,
        "all_selectable_views_keep_slow_hidden_exact": True,
        "no_training_view_may_zero_slow_hidden": True,
        "C_eq_0_single_persisted_forward_checked": True,
    }


def loss_scale_preflight() -> dict[str, Any]:
    persisted = torch.tensor(2.0)
    counterfactual = torch.tensor(4.0)
    equal, equal_denominator = weighted_supervised_loss(persisted, counterfactual, 1.0, 1.0)
    unequal, unequal_denominator = weighted_supervised_loss(persisted, counterfactual, 1.0, 3.0)
    single, single_denominator = weighted_supervised_loss(persisted, None, 1.0, 1.0)
    checks = {
        "equal_weights_is_mean_not_sum": bool(torch.equal(equal, torch.tensor(3.0))),
        "unequal_weights_is_weighted_mean": bool(torch.equal(unequal, torch.tensor(3.5))),
        "single_persisted_scale_unchanged": bool(torch.equal(single, persisted)),
        "equal_denominator": equal_denominator == 2.0,
        "unequal_denominator": unequal_denominator == 4.0,
        "single_denominator": single_denominator == 1.0,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise AssertionError(f"Loss-scale preflight failed: {failed}")
    return {"passed": True, "checks": checks}


def gradient_preflight(
    policy: torch.nn.Module,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    count = int(batch["ref_valid_count"].item())
    if count <= 0:
        raise AssertionError("Gradient preflight requires a C>0 sample to exercise the paired adapter gradient")
    counter_name = "zero_ref_hidden_kept"
    counter = make_named_view(batch, counter_name)
    set_train_mode(policy)
    policy.zero_grad(set_to_none=True)
    with autocast_context(policy.device):
        total, persisted, counterfactual, pair_audit = paired_loss(policy, batch, counter, args)
    total.backward()
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.requires_grad and parameter.grad is not None]
    adapter_grad = policy.model.ref_valid_embedder.weight.grad
    result = {
        "persisted_loss": float(persisted["loss"].detach().float().cpu()),
        "counterfactual_loss": float(counterfactual["loss"].detach().float().cpu()),
        "supervised_loss": pair_audit["supervised_loss"],
        "supervised_weight_denominator": pair_audit["supervised_weight_denominator"],
        "total_loss": float(total.detach().float().cpu()),
        "all_losses_finite": all(torch.isfinite(value).item() for value in (persisted["loss"], counterfactual["loss"], total)),
        "gradient_tensor_count": len(gradients),
        "all_gradients_finite": bool(gradients) and all(torch.isfinite(grad).all().item() for grad in gradients),
        "adapter_gradient_nonzero": adapter_grad is not None and torch.count_nonzero(adapter_grad).item() > 0,
        "adapter_gradient_norm": None if adapter_grad is None else float(torch.linalg.vector_norm(adapter_grad.detach().float()).cpu()),
        "vision_encoder_trainable": sum(p.numel() for p in policy.vision_encoder.parameters() if p.requires_grad),
        "vision_encoder_training": bool(policy.vision_encoder.training),
        "paired_contract": pair_audit,
        "counterfactual_view": counter_name,
    }
    policy.zero_grad(set_to_none=True)
    failures = [
        key for key in ("all_losses_finite", "all_gradients_finite", "adapter_gradient_nonzero")
        if not result[key]
    ]
    if result["vision_encoder_trainable"] != 0 or result["vision_encoder_training"]:
        failures.append("vision_encoder_freeze")
    if failures:
        raise AssertionError(f"Gradient preflight failed: {failures}")
    return result


def metric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return metric_group(rows)


def grouped_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "age_0_7": metric_rows(row for row in rows if int(row["age"]) <= 7),
        "age_8": metric_rows(row for row in rows if int(row["age"]) == 8),
        "age_9": metric_rows(row for row in rows if int(row["age"]) == 9),
        "age_10": metric_rows(row for row in rows if int(row["age"]) == 10),
        "age_11": metric_rows(row for row in rows if int(row["age"]) == 11),
        "age_8_11": metric_rows(row for row in rows if int(row["age"]) >= 8),
        "all": metric_rows(rows),
    }


def mode_view(batch: Mapping[str, Any], mode: str) -> dict[str, Any]:
    return persisted_view(batch) if mode == "persisted" else make_named_view(batch, mode)


@torch.no_grad()
def validate(
    policy: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer_step: int,
    timestep: int,
    validation_seed: int,
    per_sample_path: Path | None,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy.eval()
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in VALIDATION_MODES}
    protocol = []
    if per_sample_path is not None:
        per_sample_path.write_text("", encoding="utf-8")
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        sample_id = str(cpu_batch["sample_id"][0])
        age = int(cpu_batch["age"].item())
        noise, derived_seed, noise_sha = deterministic_noise(sample_id, validation_seed, device)
        timesteps = torch.tensor([timestep], dtype=torch.long, device=device)
        cond_mask = torch.ones((1, 1), dtype=torch.float32, device=device)
        if len(protocol) < 4:
            protocol.append({"sample_id": sample_id, "derived_seed": derived_seed, "noise_sha256": noise_sha})
        for mode in VALIDATION_MODES:
            view = mode_view(batch, mode)
            with autocast_context(device):
                details = loss_details(policy, batch, view, noise=noise, timesteps=timesteps, cond_mask=cond_mask)
            prediction = details["prediction"].float()
            target = details["target"].float()
            trajectory = batch["raw_action"].float()
            x0 = x0_from_prediction(policy, trajectory, noise, timesteps, prediction)
            first_error = x0[0, 0, :6] - trajectory[0, 0, :6]
            row = {
                "sample_id": sample_id,
                "mode": mode,
                "age": age,
                "slow_age": age,
                "ref_valid_count": int(batch["ref_valid_count"].item()),
                "diffusion_noise_mse": float(torch.mean((prediction - target) ** 2).cpu()),
                "first_action_ee6_mse": float(torch.mean(first_error ** 2).cpu()),
                "first_action_ee6_rmse": float(torch.sqrt(torch.mean(first_error ** 2)).cpu()),
                "gripper_correct": float(bool(x0[0, 0, 6].item() >= 0) == bool(trajectory[0, 0, 6].item() >= 0)),
                "predicted_first_action": x0[0, 0].cpu().tolist(),
                "target_first_action": trajectory[0, 0].cpu().tolist(),
                "noise_sha256": noise_sha,
                "timestep": timestep,
                "cfg_mask": 1.0,
            }
            rows_by_mode[mode].append(row)
            if per_sample_path is not None:
                append_jsonl(per_sample_path, row)
    modes = {mode: grouped_metrics(rows) for mode, rows in rows_by_mode.items()}
    gaps = {}
    for mode in ("zero_ref_hidden_kept",):
        gaps[mode] = {}
        for group in modes["persisted"]:
            persisted = modes["persisted"][group]
            other = modes[mode][group]
            gaps[mode][group] = {
                "diffusion_noise_mse_gap": None if persisted["n"] == 0 else other["diffusion_noise_mse"] - persisted["diffusion_noise_mse"],
                "first_action_ee6_rmse_gap": None if persisted["n"] == 0 else other["first_action_ee6_rmse"] - persisted["first_action_ee6_rmse"],
                "gripper_sign_accuracy_gap": None if persisted["n"] == 0 else other["first_action_gripper_sign_accuracy"] - persisted["first_action_gripper_sign_accuracy"],
            }
    result = {
        "type": "validation",
        "optimizer_step": int(optimizer_step),
        "validation_timestep": int(timestep),
        "validation_seed": int(validation_seed),
        "fixed_sample_count": sum(group["n"] for group in [modes["persisted"]["all"]]),
        "condition_modes": list(VALIDATION_MODES),
        "noise_protocol_examples": protocol,
        "modes": modes,
        "reference_gap": gaps,
    }
    if baseline is not None:
        deltas = {}
        for mode in VALIDATION_MODES:
            deltas[mode] = {}
            for group, current in modes[mode].items():
                initial = baseline["modes"][mode][group]
                deltas[mode][group] = {
                    "diffusion_noise_mse": None if current["n"] == 0 else current["diffusion_noise_mse"] - initial["diffusion_noise_mse"],
                    "first_action_ee6_rmse": None if current["n"] == 0 else current["first_action_ee6_rmse"] - initial["first_action_ee6_rmse"],
                    "gripper_sign_accuracy": None if current["n"] == 0 else current["first_action_gripper_sign_accuracy"] - initial["first_action_gripper_sign_accuracy"],
                }
        result["delta_vs_step0_baseline"] = deltas
    return result


def evaluator_state(
    template: Mapping[str, Any], online: torch.nn.Module, ema: torch.nn.Module,
    metadata: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    state = OrderedDict(template)
    for name, value in online.state_dict().items():
        state[f"online_model.{name}"] = value.detach().cpu()
    for name, value in ema.state_dict().items():
        state[f"ema_model.{name}"] = value.detach().cpu()
    state["_m2a_metadata"] = dict(metadata)
    return state


@torch.no_grad()
def checkpoint_round_trip(
    path: Path,
    device: torch.device,
    expected_ema: torch.nn.Module,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    saved = torch_load_cpu(path)
    reloaded = build_policy(device, use_ref_validity=True)
    state = extract_prefixed_policy_state(saved, "ema_model")
    incompatible = reloaded.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"M2a round-trip mismatch: {incompatible}")
    reloaded.eval()
    expected_ema.eval()
    sample_id = str(batch["sample_id"][0])
    noise, _, _ = deterministic_noise(sample_id, args.validation_seed, device)
    timesteps = torch.tensor([args.validation_timestep], dtype=torch.long, device=device)
    cond_mask = torch.ones((1, 1), dtype=torch.float32, device=device)
    view = persisted_view(batch)
    left = loss_details(expected_ema, batch, view, noise=noise, timesteps=timesteps, cond_mask=cond_mask)
    right = loss_details(reloaded, batch, view, noise=noise, timesteps=timesteps, cond_mask=cond_mask)
    delta = float((left["prediction"].float() - right["prediction"].float()).abs().max().cpu())
    if delta > ROUND_TRIP_TOLERANCE:
        raise AssertionError(f"Checkpoint forward round-trip delta {delta} > {ROUND_TRIP_TOLERANCE}")
    adapter_present = ADAPTER_KEY in state
    del reloaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "strict_missing_keys": [], "strict_unexpected_keys": [],
        "adapter_present": adapter_present,
        "forward_max_abs_delta": delta, "tolerance": ROUND_TRIP_TOLERANCE,
    }


def save_checkpoint(
    output_dir: Path,
    step: int,
    policy: torch.nn.Module,
    ema: PolicyEMA,
    optimizer: AdamW,
    scheduler: Any,
    args: argparse.Namespace,
    baseline_template: Mapping[str, Any],
    manifest_sha256: str,
    source_sha256: str,
    validation_metrics: Mapping[str, Any],
    device: torch.device,
    round_trip_batch: Mapping[str, Any],
) -> dict[str, Any]:
    evaluator_path = output_dir / f"specialist_m2a_ema_step_{step:06d}.pt"
    training_path = output_dir / f"m2a_training_step_{step:06d}.pt"
    architecture = {
        "model_variant": MODEL_VARIANT,
        "use_ref_validity": True,
        "slow_age_conditioning": False,
        "adapter": {"type": "Linear", "input_dim": 1, "output_dim": policy.model.hidden_size, "bias": False, "insertion": "add_after_x_embedder"},
        "ref_valid_mask_semantics": "[B,8] float; 1=real valid slow reference token, 0=invalid/padding/expired",
    }
    evaluator_metadata = {
        "model_variant": MODEL_VARIANT,
        "global_optimizer_step": int(step),
        "architecture": architecture,
        "m1_source_checkpoint_sha256": source_sha256,
    }
    torch.save(evaluator_state(baseline_template, policy, ema.ema_model, evaluator_metadata), evaluator_path)
    payload = {
        "format": "robodual_m2a_reference_robust_training_v1",
        "model_variant": MODEL_VARIANT,
        "online_policy": OrderedDict((k, v.detach().cpu()) for k, v in policy.state_dict().items()),
        "ema_policy": OrderedDict((k, v.detach().cpu()) for k, v in ema.ema_model.state_dict().items()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_optimizer_step": int(step),
        "ema": {"beta": ema.beta, "power": ema.power, "update_after_step": ema.update_after_step, "current_decay": ema.current_decay, "updates": ema.updates},
        "training_args": serialized_args(args),
        "git_commit": git_commit(),
        "dataset_manifest_sha256": manifest_sha256,
        "m1_source_checkpoint_sha256": source_sha256,
        "m2a_architecture_config": architecture,
        "ref_validity_adapter_config": architecture["adapter"],
        "counterfactual_policy": {
            "zero_ref_kept_probability": args.zero_ref_kept_probability,
            "shortened_ref_probability": args.shortened_ref_probability,
            "C_eq_0_rule": "single_persisted_only_no_counterfactual",
            "slow_hidden_policy": "original_persisted_hidden_in_all_views",
            "supervised_loss": "weighted_mean_for_C_gt_0; persisted_loss_for_C_eq_0",
            "paired_same_noise_timestep_cfg_and_torch_rng": True,
        },
        "validation_metrics": dict(validation_metrics),
        "evaluator_checkpoint": evaluator_path.name,
    }
    torch.save(payload, training_path)
    audit = checkpoint_round_trip(evaluator_path, device, ema.ema_model, round_trip_batch, args)
    latest = {
        "model_variant": MODEL_VARIANT,
        "global_optimizer_step": int(step),
        "evaluator_checkpoint": str(evaluator_path),
        "training_checkpoint": str(training_path),
        "round_trip": audit,
    }
    write_json(output_dir / "latest_checkpoint.json", latest)
    return latest


def make_loaders(args: argparse.Namespace, processor: Any):
    datasets = {split: M2aDataset(args.data_dir, split, processor) for split in SPLITS}
    weights = torch.tensor([
        1.0 if int(row["slow_age"]) <= 7 else args.late_age_sample_weight
        for row in datasets["train"].samples
    ], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(datasets["train"]), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    common = dict(batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return datasets, {
        "train": DataLoader(datasets["train"], sampler=sampler, shuffle=False, **common),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def dry_run_audit(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    stats = dataset_statistics(data_dir)
    rows = read_jsonl(data_dir / "samples.jsonl")
    failures = []
    view_counts = Counter()
    age_zero_candidates = 0
    for row in rows:
        count = int(row["ref_valid_count"])
        mask = [index < count for index in range(ACTION_HORIZON)]
        if row.get("ref_valid_mask") is not None and list(map(bool, row["ref_valid_mask"])) != mask:
            failures.append(f"{row['sample_id']}: persisted mask mismatch")
        ref = torch.as_tensor(row["ref_action"])
        if torch.count_nonzero(ref[..., count:, :]).item() != 0:
            failures.append(f"{row['sample_id']}: invalid ref nonzero")
        if int(row["slow_age"]) <= 7:
            age_zero_candidates += 1
        view_counts["persisted"] += 1
        if count == 0:
            view_counts["single_persisted_only_no_counterfactual"] += 1
        else:
            view_counts["counterfactual_eligible_zero_ref_hidden_kept"] += 1
            view_counts["counterfactual_eligible_shortened_reference"] += 1
    if failures:
        raise AssertionError(f"Dataset dry-run contracts failed: {failures[:10]}")
    return {
        "mode": "dry_run", "model_loaded": False, "processor_loaded": False,
        "generalist_loaded": False, "simulator_loaded": False,
        "data_dir": str(data_dir), "dataset": stats,
        "dataset_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "slow_hidden_token_length_distribution": hidden_length_distribution(data_dir),
        "view_eligibility": dict(view_counts),
        "age_0_7_counterfactual_zero_ref_candidates": age_zero_candidates,
        "probabilities": {
            "zero_ref_hidden_kept": args.zero_ref_kept_probability,
            "shortened_reference": args.shortened_ref_probability,
        },
        "selectable_counterfactual_views": list(SELECTABLE_COUNTERFACTUAL_VIEWS),
        "slow_hidden_policy": "original_persisted_hidden_in_all_views",
        "expired_persisted_rule": "single_persisted_only_no_counterfactual",
        "all_contracts_passed": True,
    }


def preflight_architecture_round_trip(policy: torch.nn.Module) -> dict[str, Any]:
    scratch = build_policy(torch.device("cpu"), use_ref_validity=True)
    state = OrderedDict((name, value.detach().cpu()) for name, value in policy.state_dict().items())
    incompatible = scratch.load_state_dict(state, strict=True)
    result = {
        "strict_missing_keys": list(incompatible.missing_keys),
        "strict_unexpected_keys": list(incompatible.unexpected_keys),
        "adapter_present": ADAPTER_KEY in scratch.state_dict(),
        "adapter_shape": list(scratch.model.ref_valid_embedder.weight.shape),
        "state_tensor_count": len(scratch.state_dict()),
    }
    if result["strict_missing_keys"] or result["strict_unexpected_keys"] or not result["adapter_present"]:
        raise AssertionError(f"Preflight architecture round-trip failed: {result}")
    del scratch, state
    return result


def prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path} is non-empty; choose a new M2a output directory")
    path.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("M2a physical batch_size must be exactly 1 because slow_hidden token lengths vary")
    if args.grad_accumulation_steps <= 0 or args.max_optimizer_steps <= 0:
        raise ValueError("grad_accumulation_steps and max_optimizer_steps must be positive")
    if not 0 <= args.validation_timestep < 100:
        raise ValueError("validation_timestep must be in [0,99]")
    probabilities = [args.zero_ref_kept_probability, args.shortened_ref_probability]
    if any(value < 0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError(f"Counterfactual probabilities must be nonnegative and sum to 1, got {probabilities}")
    if min(args.persisted_loss_weight, args.counterfactual_loss_weight, args.consistency_weight) < 0:
        raise ValueError("Loss weights must be nonnegative")
    if args.persisted_loss_weight + args.counterfactual_loss_weight <= 0:
        raise ValueError("persisted_loss_weight + counterfactual_loss_weight must be positive")
    if args.validate_every <= 0 or args.save_every <= 0 or args.late_age_sample_weight <= 0:
        raise ValueError("validate_every, save_every, and late_age_sample_weight must be positive")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    seed_everything(args.seed)
    if args.dry_run:
        result = dry_run_audit(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    device = choose_device(args.device)
    processor = load_processor(args.processor_path)
    datasets, loaders = make_loaders(args, processor)
    policy = build_policy(device, use_ref_validity=True)
    freeze_invariants(policy)
    baseline_template, load_audit = load_m1_checkpoint(policy, args.specialist_path, allow_m2a_adapter_missing=True)
    freeze_invariants(policy)

    parity = m1_init_parity(
        policy, Path(args.specialist_path), datasets["validation"], device,
        sample_count=args.parity_samples, timestep=args.validation_timestep,
        seed=args.validation_seed,
    )
    preflight_index = next(
        index for index, row in enumerate(datasets["train"].samples)
        if int(row["ref_valid_count"]) > 0
    )
    first_cpu = next(iter(DataLoader(
        torch.utils.data.Subset(datasets["train"], [preflight_index]), batch_size=1, num_workers=0,
    )))
    first_batch = move_batch(first_cpu, device)
    view_audit = preflight_all_views(datasets["train"])
    grad_audit = gradient_preflight(policy, first_batch, args)
    loss_scale_audit = loss_scale_preflight()
    state_names = sorted(policy.state_dict())
    architecture_sanity = {
        "use_ref_validity": policy.use_ref_validity,
        "adapter_key": ADAPTER_KEY,
        "adapter_key_present": ADAPTER_KEY in state_names,
        "adapter_shape": list(policy.model.ref_valid_embedder.weight.shape),
        "x_embedder_shape": list(policy.model.x_embedder.weight.shape),
        "slow_age_conditioning": False,
        "state_tensor_count": len(state_names),
    }
    if not architecture_sanity["adapter_key_present"] or architecture_sanity["x_embedder_shape"][1] != 14:
        raise AssertionError(f"M2a architecture sanity failed: {architecture_sanity}")
    architecture_round_trip = preflight_architecture_round_trip(policy)
    preflight = {
        "mode": "preflight" if args.preflight_only else "train",
        "model_variant": MODEL_VARIANT,
        "device": str(device),
        "dataset": {**dataset_statistics(Path(args.data_dir)), "slow_hidden_token_length_distribution": hidden_length_distribution(Path(args.data_dir))},
        "checkpoint_loading": load_audit,
        "m1_init_parity": parity,
        "counterfactual_view_preflight": view_audit,
        "gradient_preflight": grad_audit,
        "loss_scale_preflight": loss_scale_audit,
        "architecture_sanity": architecture_sanity,
        "checkpoint_architecture_round_trip": architecture_round_trip,
        "parameters": {
            "total": sum(p.numel() for p in policy.parameters()),
            "trainable": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "vision_encoder_trainable": sum(p.numel() for p in policy.vision_encoder.parameters() if p.requires_grad),
        },
    }
    if args.preflight_only:
        output_dir = Path(args.output_dir).expanduser().resolve()
        prepare_output(output_dir)
        write_json(output_dir / "preflight.json", preflight)
        append_jsonl(output_dir / "sample_view_audit.jsonl", grad_audit["paired_contract"])
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return preflight

    output_dir = Path(args.output_dir).expanduser().resolve()
    prepare_output(output_dir)
    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_sha = sha256_file(data_dir / "manifest.json")
    config = {
        "model_variant": MODEL_VARIANT,
        "args": serialized_args(args),
        "git_commit": git_commit(),
        "dataset_manifest_sha256": manifest_sha,
        "m1_source_checkpoint_sha256": load_audit["sha256"],
        "checkpoint_loading": load_audit,
        "architecture": {
            "use_ref_validity": True, "slow_age_conditioning": False,
            "adapter_key": ADAPTER_KEY, "adapter_zero_initialized": True,
            "x_embedder_unchanged_input_dim": 14,
        },
        "counterfactual_policy": {
            "probabilities_for_C_gt_0": {
                "zero_ref_hidden_kept": args.zero_ref_kept_probability,
                "shortened_reference": args.shortened_ref_probability,
            },
            "C_eq_0_rule": "single persisted forward; no counterfactual",
            "slow_hidden_policy": "original persisted slow_hidden in every formal view",
            "supervised_loss": "weighted mean for C>0; persisted loss for C==0",
        },
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "preflight.json", preflight)
    metrics_path = output_dir / "metrics.jsonl"
    view_audit_path = output_dir / "sample_view_audit.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    view_audit_path.write_text("", encoding="utf-8")

    ema = PolicyEMA(policy, beta=0.9999)
    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = cosine_schedule(optimizer, args.warmup_optimizer_steps, args.max_optimizer_steps)

    baseline_per_sample = output_dir / "validation_per_sample_step_000000.jsonl"
    baseline = validate(
        ema.ema_model, loaders["validation"], device,
        optimizer_step=0, timestep=args.validation_timestep,
        validation_seed=args.validation_seed, per_sample_path=baseline_per_sample,
    )
    write_json(output_dir / "baseline_validation.json", baseline)
    write_json(output_dir / "latest_validation.json", baseline)
    append_jsonl(metrics_path, baseline)
    latest_validation = baseline

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    epoch = 0
    train_iterator = iter(loaders["train"])
    counter_rng = random.Random(args.seed + 20260820)
    view_counter: Counter[str] = Counter()
    window_sums: Counter[str] = Counter()
    window_counterfactual_loss_sum = 0.0
    window_counterfactual_loss_count = 0
    window_paired_count = 0
    window_single_count = 0
    window_view_counts: Counter[str] = Counter()
    while global_step < args.max_optimizer_steps:
        try:
            cpu_batch = next(train_iterator)
        except StopIteration:
            epoch += 1
            train_iterator = iter(loaders["train"])
            cpu_batch = next(train_iterator)
        batch = move_batch(cpu_batch, device)
        counter = choose_counterfactual_view(batch, args, counter_rng)
        set_train_mode(policy)
        with autocast_context(device):
            total, persisted, counterfactual, pair_audit = paired_loss(policy, batch, counter, args)
            scaled = total / args.grad_accumulation_steps
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite M2a loss at micro_step={micro_step}")
        scaled.backward()
        micro_step += 1
        counter_name = None if counter is None else counter["name"]
        if counter_name is None:
            view_counter["single_persisted"] += 1
            window_single_count += 1
        else:
            view_counter[counter_name] += 1
            window_view_counts[counter_name] += 1
            window_paired_count += 1
        persisted_value = float(persisted["loss"].detach().float().cpu())
        counterfactual_value = (
            None if counterfactual is None
            else float(counterfactual["loss"].detach().float().cpu())
        )
        total_value = float(total.detach().float().cpu())
        window_sums["total_loss"] += total_value
        window_sums["persisted_loss"] += persisted_value
        window_sums["supervised_loss"] += pair_audit["supervised_loss"]
        window_sums["consistency_loss"] += pair_audit["consistency_loss"]
        if counterfactual_value is not None:
            window_counterfactual_loss_sum += counterfactual_value
            window_counterfactual_loss_count += 1
        pair_audit.update({
            "sample_id": str(cpu_batch["sample_id"][0]),
            "age": int(cpu_batch["age"].item()),
            "micro_step": micro_step,
        })
        append_jsonl(view_audit_path, pair_audit)
        if micro_step % args.grad_accumulation_steps:
            continue
        gradients = [p for p in policy.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(gradients, args.max_grad_norm)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at micro_step={micro_step}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        ema.update(policy)
        window_count = window_paired_count + window_single_count
        if window_count != args.grad_accumulation_steps:
            raise AssertionError(
                f"Optimizer-step micro-sample accounting mismatch: {window_count} "
                f"!= {args.grad_accumulation_steps}"
            )
        record = {
            "type": "train", "model_variant": MODEL_VARIANT,
            "global_optimizer_step": global_step, "micro_step": micro_step,
            "logical_epoch": epoch,
            "loss": window_sums["total_loss"] / window_count,
            "total_loss": window_sums["total_loss"] / window_count,
            "persisted_loss": window_sums["persisted_loss"] / window_count,
            "counterfactual_loss": (
                None if window_counterfactual_loss_count == 0
                else window_counterfactual_loss_sum / window_counterfactual_loss_count
            ),
            "supervised_loss": window_sums["supervised_loss"] / window_count,
            "consistency_loss": window_sums["consistency_loss"] / window_count,
            "consistency_weight": args.consistency_weight,
            "paired_micro_samples": window_paired_count,
            "single_persisted_micro_samples": window_single_count,
            "counterfactual_view_counts": dict(window_view_counts),
            "gradient_norm": float(torch.as_tensor(grad_norm).float().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "view_counts_so_far": dict(view_counter),
        }
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        window_sums.clear()
        window_counterfactual_loss_sum = 0.0
        window_counterfactual_loss_count = 0
        window_paired_count = 0
        window_single_count = 0
        window_view_counts.clear()
        if global_step % args.validate_every == 0:
            per_sample = output_dir / f"validation_per_sample_step_{global_step:06d}.jsonl"
            latest_validation = validate(
                ema.ema_model, loaders["validation"], device,
                optimizer_step=global_step, timestep=args.validation_timestep,
                validation_seed=args.validation_seed, per_sample_path=per_sample,
                baseline=baseline,
            )
            append_jsonl(metrics_path, latest_validation)
            write_json(output_dir / "latest_validation.json", latest_validation)
        if global_step % args.save_every == 0:
            save_checkpoint(
                output_dir, global_step, policy, ema, optimizer, scheduler, args,
                baseline_template, manifest_sha, load_audit["sha256"], latest_validation,
                device, first_batch,
            )
    if global_step % args.save_every:
        save_checkpoint(
            output_dir, global_step, policy, ema, optimizer, scheduler, args,
            baseline_template, manifest_sha, load_audit["sha256"], latest_validation,
            device, first_batch,
        )
    return {"output_dir": str(output_dir), "global_optimizer_step": global_step, "latest_validation": latest_validation}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--specialist_path", type=Path, default=DEFAULT_M1_CHECKPOINT)
    parser.add_argument("--processor_path", type=Path, default=DEFAULT_PROCESSOR_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_optimizer_steps", type=int, default=1500)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_optimizer_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--validate_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--validation_timestep", type=int, default=50)
    parser.add_argument("--validation_seed", type=int, default=20260810)
    parser.add_argument("--persisted_loss_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_loss_weight", type=float, default=1.0)
    parser.add_argument("--zero_ref_kept_probability", type=float, default=0.80)
    parser.add_argument("--shortened_ref_probability", type=float, default=0.20)
    parser.add_argument("--late_age_sample_weight", type=float, default=2.0)
    parser.add_argument("--consistency_weight", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--parity_samples", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    if not args.dry_run and not args.preflight_only:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
