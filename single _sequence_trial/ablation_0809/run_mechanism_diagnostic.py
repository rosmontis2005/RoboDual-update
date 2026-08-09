#!/usr/bin/env python3
"""Run the small seq060/subtask-5 hidden-vs-reference channel diagnostic.

The environment is rolled in with the historical P12 policy until a selected
age.  At that exact current observation, one fresh slow call produces a fresh
action chunk and hidden state.  Four specialist calls then use the same
observation, history, boundary-derived rollout context, and a fresh
diffusion generator with the same seed.  The fresh/fresh, fresh/empty, and
stale/fresh cells are interventions for attribution; only stale/empty is P12.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
TRIAL_ROOT = HERE.parent
REPO_ROOT = HERE.parents[1]
CROSS_ROOT = TRIAL_ROOT / "cross"
FIXED8_ROOT = TRIAL_ROOT / "original_8_steps"
AGE12_ROOT = TRIAL_ROOT / "fixed_12_steps"
VLA_SCRIPTS = REPO_ROOT / "vla-scripts"
CALVIN_ROOT = REPO_ROOT.parent / "calvin"
for import_path in (
    HERE,
    CROSS_ROOT,
    FIXED8_ROOT,
    AGE12_ROOT,
    VLA_SCRIPTS,
    CALVIN_ROOT / "calvin_models",
    CALVIN_ROOT / "calvin_env",
    CALVIN_ROOT / "calvin_env" / "tacto",
):
    sys.path.insert(0, str(import_path))
os.environ.setdefault("CALVIN_ROOT", str(CALVIN_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import collect_fixed_12_steps as age12_collector  # noqa: E402
import collect_original_8_steps as fixed8_collector  # noqa: E402
import run_cross_experiment as cross_runner  # noqa: E402
from mechanism_common import (  # noqa: E402
    CONDITIONS,
    CONDITION_FACTORS,
    json_safe,
    observation_sha256,
    normalize_generalist_action,
    parse_int_csv,
    parse_str_csv,
    reference_for_age,
    tensor_sha256,
    validate_ages,
)


TASK = "lift_pink_block_slider"
INSTRUCTION_KEY = TASK
DEFAULT_OUTPUT = HERE / "runs" / "mechanism_seq060"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def trial_seed_list(args: argparse.Namespace) -> list[int]:
    if args.trial_seeds:
        seeds = [int(item.strip()) for item in args.trial_seeds.split(",") if item.strip()]
    else:
        seeds = [args.base_trial_seed + index for index in range(args.replicates)]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Trial seeds must be non-empty and unique")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_bundle", type=Path, default=CROSS_ROOT / "boundary_states.pt")
    parser.add_argument("--states", default="S12", help="Boundary labels, e.g. S12 or S8,S12")
    parser.add_argument("--ages", default="8", help="P12 ages at which to freeze the observation")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base_trial_seed", type=int, default=42000)
    parser.add_argument("--trial_seeds", default="", help="Comma-separated roll-in seeds")
    parser.add_argument("--diffusion_seed", type=int, default=809000)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--generalist_path", default=str(REPO_ROOT.parent / "models/generalist"))
    parser.add_argument(
        "--specialist_path",
        default=str(REPO_ROOT.parent / "models/specialist/Specialist+Depth+Gripper.pt"),
    )
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset")
    parser.add_argument("--fast_num_inference_steps", type=int, default=10)
    parser.add_argument("--with_cfg", action="store_true")
    parser.add_argument("--with_tactile", action="store_true")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="none")
    parser.add_argument("--restore_observation_atol", type=float, default=2e-5)
    parser.add_argument("--restore_position_atol", type=float, default=2e-5)
    parser.add_argument("--restore_velocity_atol", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry_run", action="store_true")
    parser.set_defaults(load_in_4bit=True)
    return parser


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[int], list[int]]:
    states = parse_str_csv(args.states)
    if any(state not in {"S8", "S12"} for state in states):
        raise ValueError(f"--states must contain only S8/S12, got {states}")
    if len(states) != len(set(states)):
        raise ValueError("--states must not contain duplicates")
    ages = validate_ages(parse_int_csv(args.ages), minimum=8, maximum=11)
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    if args.fast_num_inference_steps < 1:
        raise ValueError("--fast_num_inference_steps must be positive")
    seeds = trial_seed_list(args)
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one generalist quantization mode")
    if not args.state_bundle.expanduser().is_file() and not args.dry_run:
        raise FileNotFoundError(args.state_bundle)
    return states, ages, seeds


def make_run_dir(path: Path) -> Path:
    run_dir = path.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "restore_audits").mkdir(exist_ok=True)
    (run_dir / "observations").mkdir(exist_ok=True)
    return run_dir


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def seed_rollin(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_inputs(evaluator: Any, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Build exactly the current/previous fast inputs used by the P12 evaluator."""

    image = observation["rgb_obs"]["rgb_static"]
    gripper_image = observation["rgb_obs"]["rgb_gripper"]
    runtime_device = evaluator._runtime_device()
    runtime_dtype = evaluator._runtime_dtype()
    evaluator._ensure_fast_system_device(runtime_device)
    processor = evaluator.processor

    gripper_tensor = processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3]
    gripper_tensor = gripper_tensor.unsqueeze(0).to(runtime_device)
    depth_image = (
        torch.as_tensor(observation["depth_obs"]["depth_static"], device=runtime_device).unsqueeze(0)
        - evaluator.depth_min
    ) / (evaluator.depth_max - evaluator.depth_min)
    depth_gripper = (
        torch.as_tensor(observation["depth_obs"]["depth_gripper"], device=runtime_device).unsqueeze(0)
        - evaluator.gripper_depth_min
    ) / (evaluator.gripper_depth_max - evaluator.gripper_depth_min)

    prompt = age12_collector.age_evaluator.get_openvla_prompt(instruction)
    slow_inputs = processor(prompt, Image.fromarray(image)).to(runtime_device, dtype=runtime_dtype)
    current_image = slow_inputs["pixel_values"][:, :3].to(torch.float32)
    previous_source = image if evaluator.obs_buffer is None else evaluator.obs_buffer
    previous_image = processor.image_processor.apply_transform(Image.fromarray(previous_source))[:3]
    previous_image = previous_image.unsqueeze(0).to(runtime_device)

    state = torch.as_tensor(observation["robot_obs"], device=runtime_device, dtype=torch.float32)
    proprio = torch.cat([state[:6], state[[-1]]], dim=-1).unsqueeze(0)
    hist_action = torch.zeros((1, 4, 7), device=runtime_device)
    if evaluator.hist_action:
        history = torch.stack(list(evaluator.hist_action), dim=0).unsqueeze(0).to(runtime_device)
        hist_action[:, -history.shape[1] :] = history

    return {
        "slow_inputs": slow_inputs,
        "obs": (current_image, previous_image),
        "depth_obs": depth_image,
        "gripper_obs": (gripper_tensor, depth_gripper),
        "tactile_obs": None,
        "lang": instruction,
        "proprio": proprio,
        "hist_action": hist_action,
        "runtime_device": runtime_device,
        "previous_image": np.asarray(previous_source).copy(),
    }


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator(device=str(device))
    return generator.manual_seed(int(seed))


def run_fast_condition(
    fast_system: Any,
    probe: Any,
    prepared: dict[str, Any],
    ref_action: torch.Tensor,
    hidden_states: torch.Tensor,
    instruction: str,
    condition: str,
    diffusion_seed: int,
) -> dict[str, Any]:
    device = prepared["runtime_device"]
    kwargs = {
        "ref_action": ref_action.to(device=device, dtype=torch.float32),
        "action_cond": hidden_states.to(device=device, dtype=torch.float32),
        "obs": prepared["obs"],
        "depth_obs": prepared["depth_obs"],
        "gripper_obs": prepared["gripper_obs"],
        "tactile_obs": prepared["tactile_obs"],
        "lang": instruction,
        "proprio": prepared["proprio"],
        "hist_action": prepared["hist_action"],
    }
    generator = make_generator(device, diffusion_seed)
    missing = object()
    previous_generator = fast_system.kwargs.get("generator", missing)
    fast_system.kwargs["generator"] = generator
    rng_before = cross_runner.rng_digest()
    probe.arm()
    try:
        with torch.inference_mode():
            predicted = fast_system.predict_action(**kwargs)
    finally:
        if previous_generator is missing:
            fast_system.kwargs.pop("generator", None)
        else:
            fast_system.kwargs["generator"] = previous_generator
    rng_after = cross_runner.rng_digest()
    if rng_before != rng_after:
        raise AssertionError(f"Condition {condition} consumed the global RNG despite fixed generator")
    if probe.current is None:
        raise AssertionError(f"No initial diffusion noise captured for {condition}")
    predicted = torch.as_tensor(predicted).detach().to(torch.float32).cpu()
    if predicted.ndim != 3 or tuple(predicted.shape[1:]) != (8, 7):
        raise AssertionError(f"Unexpected specialist output shape: {tuple(predicted.shape)}")
    return {
        "condition": condition,
        "hidden_source": CONDITION_FACTORS[condition]["hidden"],
        "ref_source": CONDITION_FACTORS[condition]["ref"],
        "intervention": CONDITION_FACTORS[condition]["intervention"],
        "first_action": predicted[0, 0].tolist(),
        "action_chunk": predicted[0].tolist(),
        "initial_noise": json_safe(probe.current),
        "initial_noise_sha256": probe.current["sha256"],
        "rng_before": rng_before,
        "rng_after": rng_after,
    }


def save_observation_snapshot(
    path: Path,
    observation: dict[str, Any],
    prepared: dict[str, Any],
    evaluator: Any,
) -> str:
    current_rgb = np.asarray(observation["rgb_obs"]["rgb_static"])
    current_gripper = np.asarray(observation["rgb_obs"]["rgb_gripper"])
    current_depth = np.asarray(observation["depth_obs"]["depth_static"])
    current_depth_gripper = np.asarray(observation["depth_obs"]["depth_gripper"])
    robot_obs = np.asarray(observation["robot_obs"])
    scene_obs = np.asarray(observation.get("scene_obs", np.zeros(0, dtype=np.float32)))
    history = (
        np.stack([item.detach().cpu().numpy() for item in evaluator.hist_action], axis=0)
        if evaluator.hist_action
        else np.zeros((0, 7), dtype=np.float32)
    )
    previous_rgb = np.asarray(prepared["previous_image"])
    digest = observation_sha256(
        current_rgb,
        current_gripper,
        current_depth,
        current_depth_gripper,
        robot_obs,
        scene_obs,
        previous_rgb,
        history,
    )
    np.savez_compressed(
        path,
        rgb_static=current_rgb,
        rgb_gripper=current_gripper,
        depth_static=current_depth,
        depth_gripper=current_depth_gripper,
        robot_obs=robot_obs,
        scene_obs=scene_obs,
        previous_rgb_static=previous_rgb,
        hist_action=history,
    )
    return digest


def source_contract(bundle_path: Path) -> dict[str, dict[str, str]]:
    files = {
        "diagnostic_runner": HERE / "run_mechanism_diagnostic.py",
        "shared_contract": HERE / "mechanism_common.py",
        "cross_runner": CROSS_ROOT / "run_cross_experiment.py",
        "boundary_bundle": bundle_path,
        "age12_collector": AGE12_ROOT / "collect_fixed_12_steps.py",
        "age12_evaluator": VLA_SCRIPTS / "dual_sys_evaluation_0424test.py",
        "fast_policy": REPO_ROOT / "prismatic/models/policy/diffusion_policy.py",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": cross_runner.sha256_file(path)}
        for name, path in files.items()
    }


def dry_run_payload(args: argparse.Namespace, states: list[str], ages: list[int], seeds: list[int]) -> dict:
    return {
        "design": "seq060/subtask-5 P12 current-observation channel attribution",
        "states": states,
        "ages": ages,
        "trial_seeds": seeds,
        "diffusion_seed": args.diffusion_seed,
        "conditions": [
            {"condition": condition, **CONDITION_FACTORS[condition]} for condition in CONDITIONS
        ],
        "expected_observations": len(states) * len(seeds) * len(ages),
        "expected_specialist_calls": len(states) * len(seeds) * len(ages) * len(CONDITIONS),
        "note": "All four calls at one observation use the same generator seed and must share noise SHA256.",
    }


def main(args: argparse.Namespace) -> None:
    states, ages, seeds = validate_args(args)
    if args.dry_run:
        print(json.dumps(dry_run_payload(args, states, ages, seeds), indent=2))
        return

    bundle_path = args.state_bundle.expanduser().resolve()
    bundle = cross_runner.load_bundle(bundle_path)
    run_dir = make_run_dir(args.output_dir)
    manifest = {
        "schema_version": 1,
        "design": "seq060/subtask-5 P12 current-observation hidden/ref channel attribution",
        "sequence_index": 60,
        "subtask_index": 4,
        "task": TASK,
        "policy_rollin": {
            "name": "P12",
            "slow_steps": "step % 12 == 0",
            "empty_ref_after_age": 8,
            "target_ages": ages,
        },
        "boundary_states": states,
        "trial_seeds": seeds,
        "diffusion_seed": int(args.diffusion_seed),
        "expected_observations": len(states) * len(seeds) * len(ages),
        "expected_condition_events": len(states) * len(seeds) * len(ages) * len(CONDITIONS),
        "fixed_inputs": [
            "restored simulator boundary per state",
            "P12 roll-in current observation",
            "previous static image, proprio, and four-action history",
            "same language instruction",
            "same specialist initial noise tensor within each frozen observation",
        ],
        "conditions": [
            {"condition": condition, **CONDITION_FACTORS[condition]} for condition in CONDITIONS
        ],
        "condition_semantics": {
            "stale_hidden": "hidden_states captured by the P12 slow call at age 0",
            "fresh_hidden": "hidden_states from one intervention slow call on the frozen current observation",
            "empty_ref": "all-zero [1, 8, 7] local condition",
            "fresh_ref": "the full [1, 8, 7] action chunk from that same intervention slow call",
            "intervention_warning": "fresh/fresh, fresh/empty, and stale/fresh are attribution interventions, not deployment policies",
        },
        "generalist_path": str(Path(args.generalist_path).expanduser().resolve()),
        "specialist_path": str(Path(args.specialist_path).expanduser().resolve()),
        "specialist_checkpoint_sha256": cross_runner.sha256_file(Path(args.specialist_path)),
        "runtime_options": {
            "dataset_subdir": args.dataset_subdir,
            "fast_num_inference_steps": args.fast_num_inference_steps,
            "with_cfg": args.with_cfg,
            "with_tactile": args.with_tactile,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
        },
        "restore_tolerances": {
            "observation_atol": args.restore_observation_atol,
            "position_atol": args.restore_position_atol,
            "velocity_atol": args.restore_velocity_atol,
        },
        "source_contract": source_contract(bundle_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    events_path = run_dir / "mechanism_events.jsonl"
    observations_path = run_dir / "mechanism_observations.jsonl"
    events_path.touch()
    observations_path.touch()

    seed_everything = fixed8_collector.seed_everything
    seed_everything(args.model_seed)
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=12))])
    if accelerator.num_processes != 1:
        raise RuntimeError("Mechanism diagnostic requires one Accelerator process")
    model_args = args
    processor, action_tokenizer, dual_system = fixed8_collector.load_models(model_args, accelerator)
    evaluator = age12_collector.FixedAge12Evaluation(
        dual_system,
        processor,
        action_tokenizer,
        profile_steps=True,
        profile_sample_var_interval=12,
    )
    fast_system = evaluator._fast_system()
    noise_probe = cross_runner.InitialNoiseDigest(fast_system)

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    env = fixed8_collector.original.make_env(
        str(CALVIN_ROOT / "dataset" / args.dataset_subdir),
        observation_space,
        accelerator.device,
        args.use_egl,
    )
    conf_dir = CALVIN_ROOT / "calvin_models/conf"
    from omegaconf import OmegaConf

    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    instruction = str(annotations[INSTRUCTION_KEY][0])
    all_observations = 0
    all_events = 0
    noise_failures = 0
    try:
        for replicate, trial_seed in enumerate(seeds):
            for state_label in states:
                snapshot = bundle["states"][state_label]
                observation, restore_record = cross_runner.restore_boundary(env, snapshot, args)
                restore_path = run_dir / "restore_audits" / f"rep_{replicate:02d}_{state_label}.json"
                restore_path.write_text(
                    json.dumps(
                        {
                            "replicate": replicate,
                            "trial_seed": trial_seed,
                            "state": state_label,
                            **json_safe(restore_record),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                seed_rollin(trial_seed)
                env.seed(trial_seed)
                evaluator.reset()
                evaluator.action = None
                evaluator.hidden_states = None
                obs = observation

                for step in range(max(ages) + 1):
                    if step in ages:
                        if evaluator.last_slow_step != 0:
                            raise AssertionError(
                                f"Expected P12 stale condition from step 0 at age {step}, "
                                f"got last_slow_step={evaluator.last_slow_step}"
                            )
                        if evaluator.action is None or evaluator.hidden_states is None:
                            raise AssertionError("P12 roll-in did not create a stale action/hidden condition")
                        prepared = prepare_inputs(evaluator, obs, instruction)
                        observation_id = f"rep_{replicate:02d}_{state_label}_age_{step:02d}"
                        snapshot_path = run_dir / "observations" / f"{observation_id}.npz"
                        observation_hash = save_observation_snapshot(
                            snapshot_path, obs, prepared, evaluator
                        )

                        stale_action = evaluator.action.detach().clone().to(torch.float32)
                        stale_hidden = evaluator.hidden_states.detach().clone().to(torch.float32)
                        with torch.inference_mode():
                            fresh_output = evaluator._slow_system().predict_action(
                                **prepared["slow_inputs"], do_sample=False
                            )
                        if not isinstance(fresh_output, tuple) or len(fresh_output) < 2:
                            raise RuntimeError("Generalist predict_action did not return (action, hidden_states)")
                        fresh_action = normalize_generalist_action(
                            fresh_output[0], device=stale_hidden.device
                        )
                        fresh_hidden = torch.as_tensor(
                            fresh_output[1], device=stale_hidden.device, dtype=torch.float32
                        )
                        if tuple(fresh_action.shape) != (1, 8, 7):
                            raise AssertionError(f"Unexpected fresh action shape {tuple(fresh_action.shape)}")

                        empty_ref = torch.zeros_like(fresh_action)
                        fresh_ref = reference_for_age(fresh_action, age=0)
                        hidden_by_condition = {
                            "stale_hidden_empty_ref": stale_hidden,
                            "fresh_hidden_empty_ref": fresh_hidden,
                            "stale_hidden_fresh_ref": stale_hidden,
                            "fresh_hidden_fresh_ref": fresh_hidden,
                        }
                        ref_by_condition = {
                            "stale_hidden_empty_ref": empty_ref,
                            "fresh_hidden_empty_ref": empty_ref,
                            "stale_hidden_fresh_ref": fresh_ref,
                            "fresh_hidden_fresh_ref": fresh_ref,
                        }
                        condition_results = {}
                        for condition in CONDITIONS:
                            condition_results[condition] = run_fast_condition(
                                fast_system=fast_system,
                                probe=noise_probe,
                                prepared=prepared,
                                ref_action=ref_by_condition[condition],
                                hidden_states=hidden_by_condition[condition],
                                instruction=instruction,
                                condition=condition,
                                diffusion_seed=args.diffusion_seed,
                            )

                        noise_hashes = {
                            condition: result["initial_noise_sha256"]
                            for condition, result in condition_results.items()
                        }
                        if len(set(noise_hashes.values())) != 1:
                            noise_failures += 1
                            raise AssertionError(
                                f"Fixed-noise violation at {observation_id}: {noise_hashes}"
                            )
                        append_jsonl(
                            observations_path,
                            {
                                "observation_id": observation_id,
                                "replicate": replicate,
                                "trial_seed": trial_seed,
                                "state": state_label,
                                "age": step,
                                "observation_sha256": observation_hash,
                                "snapshot": str(snapshot_path.relative_to(run_dir)),
                                "restore_audit": str(restore_path.relative_to(run_dir)),
                                "last_slow_step": int(evaluator.last_slow_step),
                                "stale_action_sha256": tensor_sha256(stale_action),
                                "stale_hidden_sha256": tensor_sha256(stale_hidden),
                                "fresh_action_sha256": tensor_sha256(fresh_action),
                                "fresh_hidden_sha256": tensor_sha256(fresh_hidden),
                                "fresh_hidden_shape": list(fresh_hidden.shape),
                                "p12_num_cond_actions": 0,
                                "fresh_ref_num_cond_actions": 8,
                                "initial_noise_sha256": next(iter(noise_hashes.values())),
                            },
                        )
                        for condition, result in condition_results.items():
                            append_jsonl(
                                events_path,
                                {
                                    "observation_id": observation_id,
                                    "replicate": replicate,
                                    "trial_seed": trial_seed,
                                    "state": state_label,
                                    "age": step,
                                    **result,
                                },
                            )
                        all_observations += 1
                        all_events += len(condition_results)

                    if step == max(ages):
                        break
                    action = evaluator.step(obs, instruction, step)
                    obs, _, _, _ = env.step(action)
    finally:
        noise_probe.close()
        cross_runner.close_env_once(env)

    summary = {
        "completed_observations": all_observations,
        "expected_observations": len(states) * len(seeds) * len(ages),
        "completed_condition_events": all_events,
        "expected_condition_events": len(states) * len(seeds) * len(ages) * len(CONDITIONS),
        "fixed_noise_failures": noise_failures,
        "trial_seeds": seeds,
        "states": states,
        "ages": ages,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if all_observations != summary["expected_observations"] or noise_failures:
        raise RuntimeError(f"Mechanism diagnostic incomplete: {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
