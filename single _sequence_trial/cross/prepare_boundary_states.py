#!/usr/bin/env python3
"""Freeze the two subtask-5 boundary states from the completed paired runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pybullet
import torch


HERE = Path(__file__).resolve().parent
TRIAL_ROOT = HERE.parent
DEFAULT_SOURCES = {
    "S8": TRIAL_ROOT
    / "original_8_steps/runs/seq060_seed42_original_fixed8/tensors/seq_060/"
    "subtask_04_lift_pink_block_slider/step_0000.pt",
    "S12": TRIAL_ROOT
    / "fixed_12_steps/runs/seq060_seed42_fixed_age12/tensors/seq_060/"
    "subtask_04_lift_pink_block_slider/step_0000.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): cpu_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_clone(item) for item in value]
    return value


def reconstruct_controller_state(boundary_step: Path) -> dict[str, Any]:
    """Replay saved relative actions into Robot.target_pos/target_orn."""
    run_dir = boundary_step.parents[3]
    seq_dir = run_dir / "tensors/seq_060"
    prior_directories = sorted(seq_dir.glob("subtask_0[0-3]_*"))
    if len(prior_directories) != 4:
        raise RuntimeError(f"Expected four prior subtask directories in {seq_dir}")
    prior_steps = [path for directory in prior_directories for path in sorted(directory.glob("step_*.pt"))]
    if not prior_steps:
        raise RuntimeError(f"No prior actions found in {seq_dir}")

    first = load_torch(prior_steps[0])
    first_meta = first["meta"]
    if int(first_meta["subtask_index"]) != 0 or int(first_meta["step"]) != 0:
        raise AssertionError("Controller reconstruction must begin at sequence subtask 0 / step 0")
    tcp = first["environment"]["pre_physics"]["tcp"]
    target_pos = np.asarray(tcp["world_com_position"], dtype=np.float64).copy()
    target_orn = np.asarray(
        pybullet.getEulerFromQuaternion(tcp["world_com_orientation_quaternion"]), dtype=np.float64
    )
    action_digest = hashlib.sha256()
    last_gripper_action = None
    subtask_counts: dict[str, int] = {}
    for path in prior_steps:
        payload = load_torch(path)
        meta = payload["meta"]
        subtask = str(meta["subtask"])
        subtask_counts[subtask] = subtask_counts.get(subtask, 0) + 1
        action = np.asarray(payload["environment"]["executed_action"], dtype=np.float64).reshape(-1)
        if action.shape != (7,):
            raise AssertionError(f"Unexpected action shape in {path}: {action.shape}")
        # The collector finalizes the trace after env.step(action). CALVIN's
        # relative_to_absolute() uses np.split views and scales the first six
        # values in-place, so the saved executed_action already contains the
        # physical position/orientation increments (metres/radians).
        target_pos += action[:3]
        target_orn += action[3:6]
        last_gripper_action = int(np.sign(action[-1]))
        action_digest.update(str(path.relative_to(run_dir)).encode())
        action_digest.update(action.tobytes())
    return {
        "use_target_pose": True,
        "target_pos": target_pos,
        "target_orn_euler": target_orn,
        "gripper_action": last_gripper_action,
        "max_rel_pos": 0.02,
        "max_rel_orn": 0.05,
        "magic_scaling_factor_pos": 1.0,
        "magic_scaling_factor_orn": 1.0,
        "saved_action_units": "post-env.step physical increments: metres, radians, gripper sign",
        "prior_action_count": len(prior_steps),
        "prior_subtask_step_counts": subtask_counts,
        "prior_action_trace_sha256": action_digest.hexdigest(),
        "source_run_dir": str(run_dir.resolve()),
    }


def extract_state(label: str, source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = load_torch(source)
    meta = payload["meta"]
    if int(meta["sequence_index"]) != 60 or int(meta["subtask_index"]) != 4 or int(meta["step"]) != 0:
        raise AssertionError(f"{source} is not sequence 60 / subtask 4 / step 0")
    if str(meta["subtask"]) != "lift_pink_block_slider":
        raise AssertionError(f"Unexpected task in {source}: {meta['subtask']}")

    environment = payload["environment"]
    observation = environment["pre_observation"]
    return {
        "label": label,
        "meaning": (
            "Simulator boundary after successful subtask 4 and immediately before the first policy call "
            "of subtask 5"
        ),
        "source_step_file": str(source.resolve()),
        "source_step_sha256": sha256_file(source),
        "source_meta": cpu_clone(meta),
        "robot_obs": cpu_clone(observation["robot_obs"]),
        "scene_obs": cpu_clone(observation["scene_obs"]),
        "pre_info": cpu_clone(environment["pre_info"]),
        "pre_physics": cpu_clone(environment["pre_physics"]),
        "controller_state": cpu_clone(reconstruct_controller_state(source)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s8_step", type=Path, default=DEFAULT_SOURCES["S8"])
    parser.add_argument("--s12_step", type=Path, default=DEFAULT_SOURCES["S12"])
    parser.add_argument("--output", type=Path, default=HERE / "boundary_states.pt")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "sequence_index": 60,
        "subtask_index": 4,
        "task": "lift_pink_block_slider",
        "policy_state_contract": (
            "Evaluator state is deliberately reset for every cell, matching the original rollout_subtask boundary. "
            "No action/history/generalist cache from subtask 4 is carried into subtask 5."
        ),
        "states": {
            "S8": extract_state("S8", args.s8_step.expanduser().resolve()),
            "S12": extract_state("S12", args.s12_step.expanduser().resolve()),
        },
    }
    temporary = output.with_suffix(output.suffix + ".incomplete")
    torch.save(bundle, temporary)
    os.replace(temporary, output)
    sidecar = {
        "bundle": str(output),
        "bundle_sha256": sha256_file(output),
        "schema_version": bundle["schema_version"],
        "sources": {
            label: {
                "path": state["source_step_file"],
                "sha256": state["source_step_sha256"],
            }
            for label, state in bundle["states"].items()
        },
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
