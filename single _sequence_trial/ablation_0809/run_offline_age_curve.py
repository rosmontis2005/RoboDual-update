#!/usr/bin/env python3
"""Measure the offline age curve for the frozen specialist.

The default input is the repaired transition dataset's trajectory-level
``validation`` split.  Each row already contains the slow action/hidden state
captured online at the last refresh and the current observation at its stored
age.  The script therefore evaluates the natural d=0,...,11 age bins without
reconstructing slow conditions offline.

For every sample, a deterministic noise tensor and one fixed diffusion
timestep are used.  The reported first-action error is the EE6 RMSE between
the target first action and the epsilon-prediction-derived x0 estimate; the
gripper metric is the sign accuracy of that same first action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TRANSITION_ROOT = REPO_ROOT / "LoRA_transition_0711"
DEFAULT_DATA = TRANSITION_ROOT / "collected_transition_v1_repaired"
DEFAULT_OUTPUT = HERE / "runs" / "offline_age_curve"
for import_path in (HERE, REPO_ROOT, REPO_ROOT.parent):
    sys.path.insert(0, str(import_path))

import mechanism_common as common  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class OfflineTransitionValidation:
    """Minimal reader for the repaired v2 transition manifest."""

    def __init__(self, root: Path, split: str, age_min: int, age_max: int):
        self.root = root.expanduser().resolve()
        summary_path = self.root / "collection_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        self.summary = json.loads(summary_path.read_text())
        if not str(self.summary.get("format", "")).startswith("robodual_transition_lora_repaired"):
            raise ValueError(f"Unexpected transition dataset format: {self.summary.get('format')!r}")
        integrity = self.summary.get("integrity", {})
        if "current-observation" not in str(integrity.get("conditions", "")):
            raise ValueError("Dataset conditions are not verified current-observation slow conditions")
        target_provenance = str(integrity.get("targets", ""))
        if "target" not in target_provenance and "recovered" not in target_provenance:
            raise ValueError("Dataset target provenance is missing")
        trajectory_rows = read_jsonl(self.root / "trajectories.jsonl")
        self.trajectories = {row["trajectory_id"]: row for row in trajectory_rows}
        self.samples = [
            row
            for row in read_jsonl(self.root / "samples.jsonl")
            if row.get("split") == split and age_min <= int(row["slow_age"]) <= age_max
        ]
        if not self.samples:
            raise ValueError(f"No samples for split={split!r}, age range=[{age_min},{age_max}]")

    def age_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(int(row["slow_age"]) for row in self.samples).items()))

    def select(self, max_per_age: int, seed: int) -> list[dict[str, Any]]:
        by_age: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.samples:
            by_age[int(row["slow_age"])].append(row)
        selected = []
        for age in sorted(by_age):
            rows = sorted(by_age[age], key=lambda row: int(row["sample_id"]))
            if max_per_age > 0 and len(rows) > max_per_age:
                rows = random.Random(seed + age).sample(rows, max_per_age)
                rows.sort(key=lambda row: int(row["sample_id"]))
            selected.extend(rows)
        return selected

    def load(self, row: dict[str, Any]) -> dict[str, Any]:
        trajectory_id = str(row["trajectory_id"])
        step = int(row["step"])
        age = int(row["slow_age"])
        condition_id = int(row["condition_id"])
        frame_path = self.root / "trajectories" / trajectory_id / f"step_{step:04d}.npz"
        previous_path = self.root / "trajectories" / trajectory_id / f"step_{step - 1:04d}.npz"
        actions_path = self.root / "committed_actions" / f"{trajectory_id}.npy"
        condition_path = self.root / "conditions" / trajectory_id / f"condition_{condition_id:03d}.pt"
        for path in (frame_path, previous_path, actions_path, condition_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(frame_path, allow_pickle=False) as current:
            current_data = {key: current[key] for key in current.files}
        with np.load(previous_path, allow_pickle=False) as previous:
            previous_data = {key: previous[key] for key in previous.files}
        actions = np.load(actions_path, allow_pickle=False, mmap_mode="r")
        future = np.asarray(actions[step : step + 8], dtype=np.float32).copy()
        if future.shape != (8, 7):
            raise ValueError(f"Target action shape for {trajectory_id}:{step} is {future.shape}")
        condition = load_torch(condition_path)
        condition_step = int(condition.get("step", -1))
        if condition.get("source") != "online_current_observation":
            raise ValueError(
                f"Condition {condition_path} is not marked online_current_observation: "
                f"{condition.get('source')!r}"
            )
        if condition_step < 0 or step - condition_step != age:
            raise ValueError(
                f"Age provenance mismatch for {trajectory_id}:{step}: "
                f"sample_age={age}, condition_step={condition_step}"
            )
        slow_action = torch.as_tensor(condition["slow_action"], dtype=torch.float32)
        slow_hidden = torch.as_tensor(condition["slow_hidden"], dtype=torch.float32)
        if tuple(slow_action.shape) != (1, 8, 7):
            raise ValueError(f"Bad slow_action shape in {condition_path}: {tuple(slow_action.shape)}")
        if slow_hidden.ndim != 3 or slow_hidden.shape[0] != 1 or slow_hidden.shape[2] != 4096:
            raise ValueError(f"Bad slow_hidden shape in {condition_path}: {tuple(slow_hidden.shape)}")
        history = np.asarray(current_data.get("hist_action_before"))
        if history.shape != (4, 7):
            raise ValueError(f"Bad current history shape for {trajectory_id}:{step}: {history.shape}")
        return {
            "sample": row,
            "trajectory": self.trajectories[trajectory_id],
            "current": current_data,
            "previous": previous_data,
            "future": future,
            "slow_action": slow_action,
            "slow_hidden": slow_hidden,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--age_min", type=int, default=0)
    parser.add_argument("--age_max", type=int, default=11)
    parser.add_argument("--max_samples_per_age", type=int, default=0, help="0 means all available rows")
    parser.add_argument("--selection_seed", type=int, default=809080)
    parser.add_argument("--noise_seed", type=int, default=809081)
    parser.add_argument("--diffusion_timestep", type=int, default=50)
    parser.add_argument("--generalist_path", default=str(REPO_ROOT.parent / "models/generalist"))
    parser.add_argument(
        "--specialist_path",
        default=str(REPO_ROOT.parent / "models/specialist/Specialist+Depth+Gripper.pt"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.age_min <= args.age_max <= 11:
        raise ValueError("Require 0 <= age_min <= age_max <= 11")
    if args.max_samples_per_age < 0:
        raise ValueError("--max_samples_per_age must be non-negative")
    if not 0 <= args.diffusion_timestep < 100:
        raise ValueError("The specialist scheduler has timesteps 0..99")


def make_run_dir(path: Path) -> Path:
    run_dir = path.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(path: Path, device: torch.device) -> Any:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy

    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    policy = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}},
        noise_scheduler=scheduler,
        n_action_steps=8,
        num_inference_steps=10,
        vision_encoder="DINO",
        vision_encoder_pretrained=False,
        with_depth=True,
        with_gripper=True,
        with_tactile=False,
        cond_drop_chance=0.0,
        progressive_noise=False,
    )
    state = load_torch(path)
    ema_state = {
        str(key)[len("ema_model.") :]: value
        for key, value in state.items()
        if str(key).startswith("ema_model.") and str(key) != "ema_model._dummy_variable"
    }
    incompatible = policy.load_state_dict(ema_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Specialist checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if str(policy.noise_scheduler.config.prediction_type) != "epsilon":
        raise RuntimeError(
            "The age-curve x0 reconstruction requires prediction_type='epsilon', "
            f"got {policy.noise_scheduler.config.prediction_type!r}"
        )
    return policy.to(device).eval()


def make_batch(item: dict[str, Any], processor: Any, device: torch.device, age: int) -> dict[str, Any]:
    current = item["current"]
    previous = item["previous"]
    current_image = processor.image_processor.apply_transform(Image.fromarray(current["rgb_static"]))[:3]
    previous_image = processor.image_processor.apply_transform(Image.fromarray(previous["rgb_static"]))[:3]
    gripper_image = processor.image_processor.apply_transform(Image.fromarray(current["rgb_gripper"]))[:3]
    depth = (torch.from_numpy(np.asarray(current["depth_static"], dtype=np.float32)) - 3.5) / (6.2 - 3.5)
    gripper_depth = torch.from_numpy(np.asarray(current["depth_gripper"], dtype=np.float32)) / 2.0
    robot_obs = np.asarray(current["robot_obs"], dtype=np.float32)
    proprio = np.concatenate([robot_obs[:6], robot_obs[-1:]], axis=0)
    slow_action = item["slow_action"]
    if slow_action.ndim == 3:
        slow_action = slow_action.squeeze(0)
    slow_hidden = item["slow_hidden"]
    if slow_hidden.ndim == 2:
        slow_hidden = slow_hidden.unsqueeze(0)
    return {
        "trajectory": torch.from_numpy(item["future"]).unsqueeze(0).to(device),
        "slow_action": slow_action.unsqueeze(0).to(device),
        "action_cond": slow_hidden.to(device),
        "obs": (
            current_image.unsqueeze(0).to(device=device, dtype=torch.float32),
            previous_image.unsqueeze(0).to(device=device, dtype=torch.float32),
        ),
        "depth_obs": depth.unsqueeze(0).to(device),
        "gripper_obs": (
            gripper_image.unsqueeze(0).to(device=device, dtype=torch.float32),
            gripper_depth.unsqueeze(0).to(device),
        ),
        "proprio": torch.from_numpy(proprio).unsqueeze(0).to(device),
        "hist_action": torch.from_numpy(
            np.asarray(current["hist_action_before"], dtype=np.float32)
        ).unsqueeze(0).to(device),
        "lang": [item["trajectory"]["instruction"]],
        "age": int(age),
    }


def fixed_noise(seed: int, sample_id: int, device: torch.device) -> tuple[torch.Tensor, int, str]:
    derived_seed = int(seed) + int(sample_id)
    generator = torch.Generator(device="cpu").manual_seed(derived_seed)
    cpu_noise = torch.randn((1, 8, 7), generator=generator, dtype=torch.float32)
    digest = hashlib.sha256(cpu_noise.numpy().tobytes()).hexdigest()
    return cpu_noise.to(device), derived_seed, digest


def x0_from_epsilon_prediction(policy: Any, trajectory: torch.Tensor, noise: torch.Tensor, prediction: torch.Tensor, timestep: int) -> torch.Tensor:
    scheduler = policy.noise_scheduler
    noisy = scheduler.add_noise(trajectory, noise, torch.tensor([timestep], device=trajectory.device))
    alpha_bar = scheduler.alphas_cumprod[timestep].to(device=trajectory.device, dtype=torch.float32)
    return (noisy.to(torch.float32) - torch.sqrt(1.0 - alpha_bar) * prediction.to(torch.float32)) / torch.sqrt(alpha_bar)


def evaluate_item(policy: Any, item: dict[str, Any], processor: Any, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    row = item["sample"]
    age = int(row["slow_age"])
    batch = make_batch(item, processor, device, age)
    ref_action = common.reference_for_age(batch["slow_action"], age=age)
    noise, derived_seed, noise_digest = fixed_noise(args.noise_seed, int(row["sample_id"]), device)
    timesteps = torch.tensor([args.diffusion_timestep], dtype=torch.long, device=device)
    with torch.inference_mode():
        details = policy.compute_loss(
            trajectory=batch["trajectory"].float(),
            ref_action=ref_action.float(),
            action_cond=batch["action_cond"].float(),
            obs=batch["obs"],
            depth_obs=batch["depth_obs"].float(),
            gripper_obs=(batch["gripper_obs"][0].float(), batch["gripper_obs"][1].float()),
            tactile_obs=None,
            lang=batch["lang"],
            proprio=batch["proprio"].float(),
            hist_action=batch["hist_action"].float(),
            decoupled_loss=False,
            noise=noise,
            timesteps=timesteps,
            return_details=True,
        )
    prediction = details["prediction"].float()
    target = details["target"].float()
    x0_hat = x0_from_epsilon_prediction(policy, batch["trajectory"].float(), noise, prediction, args.diffusion_timestep)
    first_error = x0_hat[0, 0, :6] - batch["trajectory"][0, 0, :6].float()
    target_gripper = batch["trajectory"][0, 0, 6].item() >= 0.0
    predicted_gripper = x0_hat[0, 0, 6].item() >= 0.0
    return {
        "sample_id": int(row["sample_id"]),
        "trajectory_id": row["trajectory_id"],
        "task": item["trajectory"]["task"],
        "category": row["category"],
        "age": age,
        "step": int(row["step"]),
        "loss": float(details["loss"].cpu()),
        "loss_ee6": float(torch.mean(torch.square(prediction[..., :6] - target[..., :6])).cpu()),
        "loss_gripper": float(torch.mean(torch.square(prediction[..., 6] - target[..., 6])).cpu()),
        "first_action_error_l2_ee6": float(torch.linalg.vector_norm(first_error).cpu()),
        "first_action_error_rmse_ee6": float(torch.sqrt(torch.mean(torch.square(first_error))).cpu()),
        "first_action_gripper_accuracy": int(predicted_gripper == target_gripper),
        "target_gripper": int(target_gripper),
        "predicted_gripper": int(predicted_gripper),
        "diffusion_timestep": int(args.diffusion_timestep),
        "noise_seed": derived_seed,
        "noise_sha256": noise_digest,
        "ref_action_count": int(max(0, 8 - age) if age < 8 else 0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "loss",
        "loss_ee6",
        "loss_gripper",
        "first_action_error_l2_ee6",
        "first_action_error_rmse_ee6",
        "first_action_gripper_accuracy",
    )
    by_age: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_age[int(row["age"])].append(row)
    age_summary = {}
    for age, age_rows in sorted(by_age.items()):
        age_summary[str(age)] = {}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in age_rows], dtype=np.float64)
            age_summary[str(age)][metric] = {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": None if len(values) < 2 else float(np.std(values, ddof=1)),
                "sem": None if len(values) < 2 else float(np.std(values, ddof=1) / np.sqrt(len(values))),
            }

    focus = {}
    for metric in metrics:
        means = {int(age): values[metric]["mean"] for age, values in age_summary.items()}
        if 7 in means and 8 in means:
            focus[f"jump_7_to_8_{metric}"] = float(means[8] - means[7])
        late = [(age, means[age]) for age in range(8, 12) if age in means]
        if len(late) >= 2:
            x = np.asarray([item[0] for item in late], dtype=np.float64)
            y = np.asarray([item[1] for item in late], dtype=np.float64)
            focus[f"slope_8_to_11_{metric}"] = float(np.polyfit(x, y, deg=1)[0])
    return {"ages": age_summary, "focus_7_to_8_and_8_to_11": focus}


def make_plot(path: Path, summary: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = [
        ("loss", "fixed-noise diffusion loss"),
        ("first_action_error_rmse_ee6", "first-action EE6 RMSE"),
        ("first_action_gripper_accuracy", "first-action gripper accuracy"),
    ]
    ages = sorted(int(age) for age in summary["ages"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        means = [summary["ages"][str(age)][metric]["mean"] for age in ages]
        errors = [
            1.96 * (summary["ages"][str(age)][metric]["sem"] or 0.0)
            for age in ages
        ]
        axis.axvspan(8, 11.5, color="#F3F4F6", zorder=0)
        axis.errorbar(ages, means, yerr=errors, marker="o", color="#1D4ED8", capsize=3)
        axis.axvline(7.5, color="#B91C1C", linestyle=":", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("age d")
        axis.grid(alpha=0.2)
        if metric == "first_action_gripper_accuracy":
            axis.set_ylim(-0.02, 1.02)
        axis.text(
            0.98,
            0.04,
            "7→8: ref 1→0\n8–11: stale hidden window",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4B5563",
        )
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def dry_run(dataset: OfflineTransitionValidation, args: argparse.Namespace) -> None:
    selected = dataset.select(args.max_samples_per_age, args.selection_seed)
    print(
        json.dumps(
            {
                "data_dir": str(dataset.root),
                "split": args.split,
                "available_age_counts": dataset.age_counts(),
                "selected_age_counts": dict(sorted(Counter(int(row["slow_age"]) for row in selected).items())),
                "diffusion_timestep": args.diffusion_timestep,
                "noise_rule": "torch.randn([1,8,7], CPU, seed=noise_seed+sample_id)",
                "metrics": [
                    "fixed-noise epsilon MSE loss",
                    "epsilon-derived x0 first-action EE6 RMSE",
                    "epsilon-derived x0 first-action gripper sign accuracy",
                ],
            },
            indent=2,
        )
    )


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    dataset = OfflineTransitionValidation(args.data_dir, args.split, args.age_min, args.age_max)
    selected = dataset.select(args.max_samples_per_age, args.selection_seed)
    if args.dry_run:
        dry_run(dataset, args)
        return
    run_dir = make_run_dir(args.output_dir)
    device = choose_device(args.device)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    policy = load_policy(Path(args.specialist_path).expanduser().resolve(), device)
    manifest = {
        "schema_version": 1,
        "design": "offline CALVIN transition validation age curve",
        "data_dir": str(dataset.root),
        "split": args.split,
        "age_range": [args.age_min, args.age_max],
        "available_age_counts": dataset.age_counts(),
        "selected_age_counts": dict(sorted(Counter(int(row["slow_age"]) for row in selected).items())),
        "max_samples_per_age": args.max_samples_per_age,
        "selection_seed": args.selection_seed,
        "noise_seed": args.noise_seed,
        "diffusion_timestep": args.diffusion_timestep,
        "noise_contract": "fixed CPU float32 [1,8,7] noise with seed noise_seed + sample_id; same sample never changes noise",
        "condition_contract": "slow_action and slow_hidden are persisted current-observation conditions from the repaired manifest",
        "specialist_path": str(Path(args.specialist_path).expanduser().resolve()),
        "specialist_checkpoint_sha256": hashlib.sha256(Path(args.specialist_path).read_bytes()).hexdigest(),
        "generalist_path_for_processor": str(Path(args.generalist_path).expanduser().resolve()),
        "metric_contract": {
            "loss": "mean epsilon MSE over [8,7] at fixed timestep; decoupled_loss=False",
            "first_action_error": "EE6 RMSE of x0 estimate recovered from epsilon prediction",
            "gripper_accuracy": "sign(x0_hat first gripper) equals sign(target first gripper); the gripper channel is also epsilon-MSE here",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    rows_path = run_dir / "age_curve_samples.jsonl"
    rows_path.touch()
    for index, row in enumerate(selected, start=1):
        item = dataset.load(row)
        result = evaluate_item(policy, item, processor, device, args)
        rows.append(result)
        with rows_path.open("a") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        if index % 50 == 0 or index == len(selected):
            print(f"evaluated {index}/{len(selected)}", flush=True)
    curve = summarize(rows)
    write_csv(run_dir / "age_curve_samples.csv", rows)
    (run_dir / "age_curve_summary.json").write_text(json.dumps(curve, indent=2) + "\n")
    make_plot(run_dir / "age_curve", curve)
    print(json.dumps({"run_dir": str(run_dir), **curve["focus_7_to_8_and_8_to_11"]}, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
