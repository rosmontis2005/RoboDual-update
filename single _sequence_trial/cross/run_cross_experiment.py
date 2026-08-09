#!/usr/bin/env python3
"""Run a paired 2x2 state-by-policy cross experiment on sequence 60 subtask 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRIAL_ROOT = HERE.parent
FIXED8_ROOT = TRIAL_ROOT / "original_8_steps"
AGE12_ROOT = TRIAL_ROOT / "fixed_12_steps"
REPO_ROOT = HERE.parents[1]
VLA_SCRIPTS = REPO_ROOT / "vla-scripts"
CALVIN_ROOT = REPO_ROOT.parent / "calvin"
for path in reversed(
    (
        HERE,
        FIXED8_ROOT,
        AGE12_ROOT,
        VLA_SCRIPTS,
        CALVIN_ROOT / "calvin_models",
        CALVIN_ROOT / "calvin_env",
        CALVIN_ROOT / "calvin_env" / "tacto",
    )
):
    sys.path.insert(0, str(path))
os.environ.setdefault("CALVIN_ROOT", str(CALVIN_ROOT))
# This experiment is defined entirely by the two local checkpoints.  Prevent
# accidental network access (and initialization failures on offline machines).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hydra
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import collect_fixed_12_steps as age12_collector
import collect_original_8_steps as fixed8_collector
from trace_capture import OnlineTraceCapture, TraceWriter, cpu_clone, physics_state, sha256_file


TASK = "lift_pink_block_slider"
INSTRUCTION_KEY = TASK
CELL_BASE_ORDER = (("S8", "P8"), ("S8", "P12"), ("S12", "P12"), ("S12", "P8"))


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rng_digest() -> dict[str, Any]:
    cpu = torch.random.get_rng_state().cpu().numpy().tobytes()
    result: dict[str, Any] = {"torch_cpu_sha256": sha256_bytes(cpu)}
    if torch.cuda.is_available():
        result["torch_cuda_sha256"] = [
            sha256_bytes(state.cpu().numpy().tobytes()) for state in torch.cuda.get_rng_state_all()
        ]
    return result


def seed_trial(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_set(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


class InitialNoiseDigest:
    """Capture the first DiT trajectory tensor of each specialist call by hash."""

    def __init__(self, fast_system: Any):
        self.current: dict[str, Any] | None = None
        self.handle = fast_system.model.register_forward_pre_hook(self._hook, with_kwargs=True)

    def arm(self) -> None:
        self.current = None

    def _hook(self, _module: Any, args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        if self.current is not None:
            return
        tensor = args[0].detach().to(torch.float32).cpu().contiguous()
        self.current = {
            "sha256": sha256_bytes(tensor.numpy().tobytes()),
            "shape": list(tensor.shape),
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "values_float32": tensor.tolist(),
        }

    def close(self) -> None:
        self.handle.remove()


def build_parser() -> argparse.ArgumentParser:
    parser = fixed8_collector.build_parser()
    parser.description = "Run the sequence-60 subtask-5 state x policy cross experiment"
    parser.add_argument("--state_bundle", type=Path, default=HERE / "boundary_states.pt")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--base_trial_seed", type=int, default=42000)
    parser.add_argument(
        "--trial_seeds",
        default="",
        help="Optional comma-separated seeds; overrides --replicates and --base_trial_seed.",
    )
    parser.add_argument(
        "--full_trace_replicates",
        default="",
        help="Optional replicate indices for original full tensor tracing, e.g. 0 or 0,1. Very large.",
    )
    parser.add_argument("--restore_observation_atol", type=float, default=2e-5)
    parser.add_argument("--restore_position_atol", type=float, default=2e-5)
    parser.add_argument("--restore_velocity_atol", type=float, default=2e-5)
    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = "Must be new/empty. Default: cross/runs/cross_<timestamp>"
    return parser


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_dir = HERE / "runs" / f"cross_{time.strftime('%Y%m%d_%H%M%S')}"
    if run_dir.exists():
        entries = list(run_dir.iterdir())
        # Model-loading failures used to leave only this empty directory behind.
        # It is safe to reuse, while any real output still remains protected.
        reusable_init_stub = (
            len(entries) == 1
            and entries[0].name == "restore_audits"
            and entries[0].is_dir()
            and not any(entries[0].iterdir())
        )
        if entries and not reusable_init_stub:
            raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "restore_audits").mkdir(exist_ok=True)
    return run_dir


def load_bundle(path: Path) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    if int(bundle.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported boundary-state schema")
    if bundle.get("task") != TASK or set(bundle.get("states", {})) != {"S8", "S12"}:
        raise ValueError("Boundary bundle does not contain the expected S8/S12 subtask-5 states")
    return bundle


def max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    return 0.0 if a.size == 0 else float(np.max(np.abs(a - b)))


def compact_physics_state(env: Any) -> dict[str, Any]:
    raw_env = env.unwrapped
    robot = raw_env.robot
    physics = raw_env.p

    def joint_record(joint_id: int) -> dict[str, Any]:
        state = physics.getJointState(
            robot.robot_uid, joint_id, physicsClientId=robot.cid
        )
        return {"joint_id": int(joint_id), "position": float(state[0]), "velocity": float(state[1])}

    link = physics.getLinkState(
        robot.robot_uid,
        robot.tcp_link_id,
        computeLinkVelocity=1,
        computeForwardKinematics=1,
        physicsClientId=robot.cid,
    )
    scene = raw_env.scene.get_info()
    return {
        "arm_joints": [joint_record(joint_id) for joint_id in robot.arm_joint_ids],
        "gripper_joints": [joint_record(joint_id) for joint_id in robot.gripper_joint_ids],
        "tcp": {
            "position": list(link[4]),
            "orientation_quaternion": list(link[5]),
            "linear_velocity": list(link[6]),
            "angular_velocity": list(link[7]),
        },
        "controller": {
            "use_target_pose": bool(robot.use_target_pose),
            "target_pos": json_safe(robot.target_pos),
            "target_orn_euler": json_safe(robot.target_orn),
            "gripper_action": int(robot.gripper_action),
        },
        "scene": {
            "movable_objects": {
                name: {
                    key: json_safe(value)
                    for key, value in record.items()
                    if key in {"current_pos", "current_orn", "current_lin_vel", "current_ang_vel"}
                }
                for name, record in scene["movable_objects"].items()
            },
            "doors": json_safe(scene["doors"]),
            "buttons": json_safe(scene["buttons"]),
            "switches": json_safe(scene["switches"]),
            "lights": json_safe(scene["lights"]),
        },
    }


def apply_exact_physics(env: Any, snapshot: dict[str, Any]) -> None:
    raw_env = env.unwrapped
    robot = raw_env.robot
    physics = raw_env.p
    target = snapshot["pre_physics"]

    for record in target["arm_joints"] + target["gripper_joints"]:
        physics.resetJointState(
            robot.robot_uid,
            int(record["joint_id"]),
            targetValue=float(record["position"]),
            targetVelocity=float(record["velocity"]),
            physicsClientId=robot.cid,
        )
    controller = snapshot["controller_state"]
    if not robot.use_target_pose or not controller["use_target_pose"]:
        raise AssertionError("Cross restoration requires the recorded use_target_pose=true controller")
    for name in (
        "max_rel_pos",
        "max_rel_orn",
        "magic_scaling_factor_pos",
        "magic_scaling_factor_orn",
    ):
        if not np.isclose(float(getattr(robot, name)), float(controller[name]), atol=0.0, rtol=0.0):
            raise AssertionError(
                f"Robot controller configuration changed for {name}: "
                f"runtime={getattr(robot, name)} saved={controller[name]}"
            )
    robot.gripper_action = int(controller["gripper_action"])
    robot.target_pos = np.asarray(controller["target_pos"], dtype=np.float64).copy()
    robot.target_orn = np.asarray(controller["target_orn_euler"], dtype=np.float64).copy()

    target_scene = target["scene_info"]
    movable_by_name = {obj.name: obj for obj in raw_env.scene.movable_objects}
    for name, record in target_scene["movable_objects"].items():
        obj = movable_by_name[name]
        physics.resetBasePositionAndOrientation(
            obj.uid,
            record["current_pos"],
            record["current_orn"],
            physicsClientId=robot.cid,
        )
        physics.resetBaseVelocity(
            obj.uid,
            linearVelocity=record["current_lin_vel"],
            angularVelocity=record["current_ang_vel"],
            physicsClientId=robot.cid,
        )

    for collection_name, state_key in (
        ("doors", "current_state"),
        ("buttons", "joint_state"),
        ("switches", "joint_state"),
    ):
        collection = {item.name: item for item in getattr(raw_env.scene, collection_name)}
        for name, record in target_scene[collection_name].items():
            item = collection[name]
            physics.resetJointState(
                item.uid,
                item.joint_index,
                targetValue=float(record[state_key]),
                targetVelocity=0.0,
                physicsClientId=robot.cid,
            )
    # Synchronize switch/button logical state and their light effects without
    # advancing physics. Scene poses and velocities remain at the saved instant.
    raw_env.scene.step()


def audit_restoration(
    snapshot: dict[str, Any],
    observation: dict[str, Any],
    observed: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    target = snapshot["pre_physics"]
    errors: dict[str, float] = {
        "robot_obs_max_abs": max_abs(snapshot["robot_obs"], observation["robot_obs"]),
        "scene_obs_max_abs": max_abs(snapshot["scene_obs"], observation["scene_obs"]),
        "tcp_position_max_abs": max_abs(target["tcp"]["world_link_frame_position"], observed["tcp"]["position"]),
        "tcp_orientation_max_abs": max_abs(
            target["tcp"]["world_link_frame_orientation_quaternion"],
            observed["tcp"]["orientation_quaternion"],
        ),
        "tcp_linear_velocity_max_abs": max_abs(
            target["tcp"]["world_linear_velocity"], observed["tcp"]["linear_velocity"]
        ),
        "tcp_angular_velocity_max_abs": max_abs(
            target["tcp"]["world_angular_velocity"], observed["tcp"]["angular_velocity"]
        ),
        "controller_target_pos_max_abs": max_abs(
            snapshot["controller_state"]["target_pos"], observed["controller"]["target_pos"]
        ),
        "controller_target_orn_max_abs": max_abs(
            snapshot["controller_state"]["target_orn_euler"],
            observed["controller"]["target_orn_euler"],
        ),
        "controller_gripper_action_max_abs": abs(
            float(snapshot["controller_state"]["gripper_action"])
            - float(observed["controller"]["gripper_action"])
        ),
    }
    for group in ("arm_joints", "gripper_joints"):
        target_by_id = {int(item["joint_id"]): item for item in target[group]}
        observed_by_id = {int(item["joint_id"]): item for item in observed[group]}
        errors[f"{group}_position_max_abs"] = max(
            abs(float(target_by_id[joint]["position"]) - float(observed_by_id[joint]["position"]))
            for joint in target_by_id
        )
        errors[f"{group}_velocity_max_abs"] = max(
            abs(float(target_by_id[joint]["velocity"]) - float(observed_by_id[joint]["velocity"]))
            for joint in target_by_id
        )

    target_objects = target["scene_info"]["movable_objects"]
    observed_objects = observed["scene"]["movable_objects"]
    for name in target_objects:
        for target_key, observed_key, kind in (
            ("current_pos", "current_pos", "position"),
            ("current_orn", "current_orn", "position"),
            ("current_lin_vel", "current_lin_vel", "velocity"),
            ("current_ang_vel", "current_ang_vel", "velocity"),
        ):
            errors[f"object.{name}.{target_key}_max_abs"] = max_abs(
                target_objects[name][target_key], observed_objects[name][observed_key]
            )

    for collection, keys in (
        ("doors", ("current_state",)),
        ("buttons", ("joint_state", "logical_state")),
        ("switches", ("joint_state", "logical_state")),
        ("lights", ("logical_state",)),
    ):
        target_collection = target["scene_info"][collection]
        observed_collection = observed["scene"][collection]
        if set(target_collection) != set(observed_collection):
            errors[f"scene.{collection}.name_set"] = math.inf
            continue
        for name in target_collection:
            for key in keys:
                errors[f"scene.{collection}.{name}.{key}_abs"] = abs(
                    float(target_collection[name][key]) - float(observed_collection[name][key])
                )

    violations = {}
    for key, error in errors.items():
        if key in {"robot_obs_max_abs", "scene_obs_max_abs"}:
            tolerance = args.restore_observation_atol
        elif "velocity" in key:
            tolerance = args.restore_velocity_atol
        else:
            tolerance = args.restore_position_atol
        if error > tolerance:
            violations[key] = {"error": error, "tolerance": tolerance}
    return {"passed": not violations, "errors": errors, "violations": violations}


def restore_boundary(
    env: Any, snapshot: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    env.reset(robot_obs=np.asarray(snapshot["robot_obs"]), scene_obs=np.asarray(snapshot["scene_obs"]))
    apply_exact_physics(env, snapshot)
    observation = env.get_obs()
    observed = compact_physics_state(env)
    audit = audit_restoration(snapshot, observation, observed, args)
    if not audit["passed"]:
        raise AssertionError(f"Boundary restoration failed for {snapshot['label']}: {audit['violations']}")
    return observation, {"audit": audit, "restored_physics": observed}


def expected_contract(policy: str, step: int) -> tuple[bool, int]:
    if policy == "P8":
        slow = step == 0 or (step + 1) % 8 == 0
        count = 8 if step == 0 else 8 - ((step + 1) % 8)
        return slow, count
    age = step % 12
    return step % 12 == 0, max(0, 8 - age) if age < 8 else 0


def assert_live_policy_contract(policy: str, step: int, profile: dict[str, Any]) -> None:
    expected_slow, expected_count = expected_contract(policy, step)
    actual = (bool(profile.get("slow_system")), int(profile.get("num_cond_actions") or 0))
    expected = (expected_slow, expected_count)
    if actual != expected:
        raise AssertionError(
            f"{policy} contract mismatch at step {step}: expected {expected}, observed {actual}"
        )


def trial_seed_list(args: argparse.Namespace) -> list[int]:
    if args.trial_seeds:
        seeds = [int(item.strip()) for item in args.trial_seeds.split(",") if item.strip()]
    else:
        if args.replicates < 1:
            raise ValueError("--replicates must be positive")
        seeds = [args.base_trial_seed + index for index in range(args.replicates)]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Trial seeds must be unique")
    return seeds


def cell_order(replicate: int) -> list[tuple[str, str]]:
    offset = replicate % len(CELL_BASE_ORDER)
    return list(CELL_BASE_ORDER[offset:] + CELL_BASE_ORDER[:offset])


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def close_env_once(env: Any) -> None:
    """Work around CALVIN's destructor calling close() a second time."""
    raw_env = env.unwrapped
    env.close()
    if hasattr(raw_env, "ownsPhysicsClient"):
        raw_env.ownsPhysicsClient = False


def make_full_capture(
    evaluator: Any,
    policy: str,
    trace_dir: Path,
    manifest: dict[str, Any],
) -> tuple[TraceWriter, OnlineTraceCapture]:
    trace_dir.mkdir(parents=True, exist_ok=False)
    writer = TraceWriter(trace_dir, manifest)
    if policy == "P8":
        capture = OnlineTraceCapture(evaluator, writer)
    else:
        capture = age12_collector.FixedAge12TraceCapture(
            evaluator,
            writer,
            expected_slow_call=age12_collector.is_fixed_age12_slow_step,
            schedule_label="fixed_age12",
        )
    return writer, capture


def source_contract(bundle_path: Path) -> dict[str, Any]:
    files = {
        "runner": HERE / "run_cross_experiment.py",
        "state_preparer": HERE / "prepare_boundary_states.py",
        "fixed8_collector": FIXED8_ROOT / "collect_original_8_steps.py",
        "age12_collector": AGE12_ROOT / "collect_fixed_12_steps.py",
        "shared_trace_capture": FIXED8_ROOT / "trace_capture.py",
        "fixed8_evaluator": VLA_SCRIPTS / "dual_sys_evaluation.py",
        "age12_evaluator": VLA_SCRIPTS / "dual_sys_evaluation_0424test.py",
        "boundary_bundle": bundle_path,
    }
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in files.items()}


def main(args: argparse.Namespace) -> None:
    if args.sequence_index != 60 or args.catalog_size != 100:
        raise ValueError("The frozen cross experiment requires sequence_index=60 and catalog_size=100")
    if args.ep_len < 1:
        raise ValueError("--ep_len must be positive")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one generalist quantization mode")

    bundle_path = args.state_bundle.expanduser().resolve()
    bundle = load_bundle(bundle_path)
    seeds = trial_seed_list(args)
    full_trace_replicates = parse_int_set(args.full_trace_replicates)
    if any(index < 0 or index >= len(seeds) for index in full_trace_replicates):
        raise ValueError("--full_trace_replicates contains an out-of-range replicate index")
    run_dir = make_run_dir(args)

    # Match paired data collection for model initialization; every cell later
    # receives its own common-random-number trial seed.
    seed_everything(args.seed)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=12))])
    if accelerator.num_processes != 1:
        raise RuntimeError("Cross collection requires one Accelerator process")
    processor, action_tokenizer, dual_system = fixed8_collector.load_models(args, accelerator)
    evaluators = {
        "P8": fixed8_collector.OriginalFixedMod8Evaluation(
            dual_system,
            processor,
            action_tokenizer,
            profile_steps=True,
            profile_sample_var_interval=8,
        ),
        "P12": age12_collector.FixedAge12Evaluation(
            dual_system,
            processor,
            action_tokenizer,
            profile_steps=True,
            profile_sample_var_interval=12,
        ),
    }
    fast_system = evaluators["P8"]._fast_system()
    noise_probe = InitialNoiseDigest(fast_system)

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
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    instruction = str(annotations[INSTRUCTION_KEY][0])

    manifest = {
        "schema_version": 1,
        "design": "2x2 crossed initial-state x slow-call-policy with common random numbers",
        "cells": [
            {"state": state, "policy": policy, "cell": f"{state}_{policy}"}
            for state, policy in CELL_BASE_ORDER
        ],
        "state_meaning": {
            "S8": "fixed-8 run boundary after successful subtask 4",
            "S12": "age-12 run boundary after successful subtask 4",
        },
        "policy_meaning": {"P8": "original fixed-mod-8", "P12": "fixed age-12 / empty-ref"},
        "sequence_index": 60,
        "task": TASK,
        "instruction": instruction,
        "ep_len": args.ep_len,
        "model_initialization_seed": args.seed,
        "trial_seeds": seeds,
        "cell_order_by_replicate": {
            str(index): [f"{state}_{policy}" for state, policy in cell_order(index)]
            for index in range(len(seeds))
        },
        "common_random_numbers": (
            "All four cells in a replicate reset Python, NumPy, torch CPU and all CUDA RNGs to the same trial seed "
            "immediately after state restoration and immediately before evaluator.reset(). Initial DiT noise hashes "
            "are saved per step."
        ),
        "policy_state_reset": (
            "evaluator.reset() is called for every cell, matching the original start of each subtask; no policy cache "
            "from subtask 4 is restored."
        ),
        "restore_tolerances": {
            "observation_atol": args.restore_observation_atol,
            "position_atol": args.restore_position_atol,
            "velocity_atol": args.restore_velocity_atol,
        },
        "full_trace_replicates": sorted(full_trace_replicates),
        "source_contract": source_contract(bundle_path),
        "generalist_path": str(Path(args.generalist_path).resolve()),
        "specialist_path": str(Path(args.specialist_path).resolve()),
        "specialist_checkpoint_sha256": sha256_file(Path(args.specialist_path)),
        "runtime_options": {
            "dataset_subdir": args.dataset_subdir,
            "fast_num_inference_steps": args.fast_num_inference_steps,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "with_cfg": args.with_cfg,
            "with_tactile": args.with_tactile,
            "use_egl": args.use_egl,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    events_path = run_dir / "cross_events.jsonl"
    cells_path = run_dir / "cell_summaries.jsonl"
    events_path.touch()
    cells_path.touch()

    all_summaries: list[dict[str, Any]] = []
    try:
        for replicate, trial_seed in enumerate(seeds):
            for order_index, (state_label, policy_label) in enumerate(cell_order(replicate)):
                cell = f"{state_label}_{policy_label}"
                evaluator = evaluators[policy_label]
                snapshot = bundle["states"][state_label]
                observation, restore_record = restore_boundary(env, snapshot, args)
                restore_path = run_dir / "restore_audits" / f"rep_{replicate:02d}_{cell}.json"
                restore_path.write_text(
                    json.dumps(
                        {
                            "replicate": replicate,
                            "trial_seed": trial_seed,
                            "state": state_label,
                            "policy": policy_label,
                            **json_safe(restore_record),
                        },
                        indent=2,
                    )
                    + "\n"
                )

                # Restore first, then reset every RNG. This makes accidental RNG
                # consumption inside restoration irrelevant to the policy trial.
                seed_trial(trial_seed)
                env.seed(trial_seed)
                evaluator.reset()
                start_info = env.get_info()
                initial_core = compact_physics_state(env)
                initial_pink = np.asarray(
                    initial_core["scene"]["movable_objects"]["block_pink"]["current_pos"],
                    dtype=np.float64,
                )
                max_pink_z = float(initial_pink[2])
                min_tcp_pink_distance = math.inf
                noise_hashes: list[str] = []
                successful = False
                steps_executed = 0

                full_writer = None
                full_capture = None
                if replicate in full_trace_replicates:
                    trace_dir = run_dir / "full_traces" / f"rep_{replicate:02d}_{cell}"
                    full_writer, full_capture = make_full_capture(
                        evaluator,
                        policy_label,
                        trace_dir,
                        {
                            "replicate": replicate,
                            "trial_seed": trial_seed,
                            "state": state_label,
                            "policy": policy_label,
                            "cell": cell,
                            "parent_manifest": str(run_dir / "manifest.json"),
                        },
                    )

                try:
                    for step in range(args.ep_len):
                        pre_observation = observation
                        pre_info = env.get_info()
                        pre_core = compact_physics_state(env)
                        if full_capture is not None:
                            full_capture.begin_step(
                                sequence_index=60,
                                subtask_index=4,
                                subtask=TASK,
                                instruction=instruction,
                                step=step,
                                pre_obs=pre_observation,
                                pre_info=pre_info,
                                pre_physics=physics_state(env),
                            )
                        rng_pre = rng_digest()
                        noise_probe.arm()
                        action = evaluator.step(pre_observation, instruction, step)
                        if noise_probe.current is None:
                            raise AssertionError("Failed to capture initial DiT noise")
                        noise_hashes.append(str(noise_probe.current["sha256"]))
                        observation, _, _, current_info = env.step(action)
                        task_info = task_oracle.get_task_info_for_set(
                            start_info, current_info, {TASK}
                        )
                        successful = bool(task_info)
                        post_core = compact_physics_state(env)
                        profile = cpu_clone(evaluator.last_step_profile)
                        assert_live_policy_contract(policy_label, step, profile)
                        pink = np.asarray(
                            post_core["scene"]["movable_objects"]["block_pink"]["current_pos"],
                            dtype=np.float64,
                        )
                        tcp = np.asarray(post_core["tcp"]["position"], dtype=np.float64)
                        max_pink_z = max(max_pink_z, float(pink[2]))
                        min_tcp_pink_distance = min(
                            min_tcp_pink_distance, float(np.linalg.norm(tcp - pink))
                        )
                        steps_executed = step + 1
                        append_jsonl(
                            events_path,
                            {
                                "event": "step",
                                "replicate": replicate,
                                "trial_seed": trial_seed,
                                "cell_order_index": order_index,
                                "cell": cell,
                                "state": state_label,
                                "policy": policy_label,
                                "step": step,
                                "task_success": successful,
                                "executed_action": action,
                                "rng_pre": rng_pre,
                                "initial_noise": noise_probe.current,
                                "profile": profile,
                                "pre_robot_obs": pre_observation["robot_obs"],
                                "pre_scene_obs": pre_observation["scene_obs"],
                                "post_robot_obs": observation["robot_obs"],
                                "post_scene_obs": observation["scene_obs"],
                                "pre_physics": pre_core,
                                "post_physics": post_core,
                            },
                        )
                        if full_capture is not None:
                            full_capture.finalize_step(
                                executed_action=action,
                                post_obs=observation,
                                post_info=current_info,
                                post_physics=physics_state(env),
                                task_success=successful,
                                profile=profile,
                            )
                        if successful:
                            break
                finally:
                    if full_capture is not None:
                        full_capture.close()

                final_core = compact_physics_state(env)
                final_pink = np.asarray(
                    final_core["scene"]["movable_objects"]["block_pink"]["current_pos"],
                    dtype=np.float64,
                )
                cell_summary = {
                    "event": "cell_summary",
                    "replicate": replicate,
                    "trial_seed": trial_seed,
                    "cell_order_index": order_index,
                    "cell": cell,
                    "state": state_label,
                    "policy": policy_label,
                    "success": successful,
                    "steps": steps_executed,
                    "max_pink_z": max_pink_z,
                    "pink_lift_max": max_pink_z - float(initial_pink[2]),
                    "pink_final_displacement": float(np.linalg.norm(final_pink - initial_pink)),
                    "min_tcp_pink_distance": min_tcp_pink_distance,
                    "slow_calls": sum(
                        1 for step in range(steps_executed) if expected_contract(policy_label, step)[0]
                    ),
                    "empty_reference_steps": sum(
                        1 for step in range(steps_executed) if expected_contract(policy_label, step)[1] == 0
                    ),
                    "initial_noise_sha256_by_step": noise_hashes,
                    "restore_audit": str(restore_path.relative_to(run_dir)),
                    "initial_physics": initial_core,
                    "final_physics": final_core,
                }
                append_jsonl(cells_path, cell_summary)
                all_summaries.append(cell_summary)
                if full_writer is not None:
                    full_writer.finalize(
                        {
                            "success": successful,
                            "steps": steps_executed,
                            "pink_lift_max": cell_summary["pink_lift_max"],
                        }
                    )
                print(
                    f"rep={replicate} seed={trial_seed} cell={cell} "
                    f"success={successful} steps={steps_executed}",
                    flush=True,
                )
    finally:
        noise_probe.close()
        close_env_once(env)

    final_summary = {
        "completed_cells": len(all_summaries),
        "expected_cells": 4 * len(seeds),
        "replicates": len(seeds),
        "trial_seeds": seeds,
        "outcomes": [
            {
                key: summary[key]
                for key in (
                    "replicate",
                    "trial_seed",
                    "cell",
                    "state",
                    "policy",
                    "success",
                    "steps",
                    "pink_lift_max",
                    "pink_final_displacement",
                    "min_tcp_pink_distance",
                )
            }
            for summary in all_summaries
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(f"Cross experiment complete: {run_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
