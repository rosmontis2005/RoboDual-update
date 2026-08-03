#!/usr/bin/env python3
"""Read-only integrity checks for an interrupted transition collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


FRAME_KEYS = {
    "rel_actions", "hist_action_before", "robot_obs", "scene_obs",
    "rgb_static", "rgb_gripper", "depth_static", "depth_gripper",
}
CONDITION_KEYS = {
    "step", "refresh_age", "slow_action", "slow_hidden",
    "old_condition_id", "source",
}


def numbered_files(directory: Path, pattern: str, template: str) -> list[Path]:
    files = sorted(directory.glob(pattern))
    expected = [template.format(index) for index in range(len(files))]
    actual = [path.name for path in files]
    if actual != expected:
        raise ValueError(f"Non-contiguous files in {directory}: expected tail={expected[-3:]}, actual tail={actual[-3:]}")
    return files


def check_frame(path: Path) -> None:
    with np.load(path, allow_pickle=False) as frame:
        if set(frame.files) != FRAME_KEYS:
            raise ValueError(f"Unexpected frame keys in {path}: {frame.files}")
        if frame["rel_actions"].shape != (7,):
            raise ValueError(f"Bad rel_actions shape in {path}: {frame['rel_actions'].shape}")
        if frame["hist_action_before"].shape != (4, 7):
            raise ValueError(f"Bad history shape in {path}: {frame['hist_action_before'].shape}")
        # Force CRC/decompression for every stored array.
        for key in frame.files:
            np.asarray(frame[key])


def check_condition(path: Path, condition_id: int, frame_count: int) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if set(data) != CONDITION_KEYS:
        raise ValueError(f"Unexpected condition keys in {path}: {sorted(data)}")
    if tuple(data["slow_action"].shape) != (1, 8, 7):
        raise ValueError(f"Bad slow_action shape in {path}: {tuple(data['slow_action'].shape)}")
    if data["slow_hidden"].ndim != 3:
        raise ValueError(f"Bad slow_hidden shape in {path}: {tuple(data['slow_hidden'].shape)}")
    if not 0 <= int(data["step"]) < frame_count:
        raise ValueError(f"Condition step outside trajectory in {path}: {data['step']} / {frame_count}")
    expected_old = None if condition_id == 0 else condition_id - 1
    if data["old_condition_id"] != expected_old:
        raise ValueError(f"Bad old_condition_id in {path}: {data['old_condition_id']} != {expected_old}")
    if data["source"] != "online_current_observation":
        raise ValueError(f"Unexpected condition source in {path}: {data['source']!r}")
    return {
        "condition_id": condition_id,
        "step": int(data["step"]),
        "refresh_age": data["refresh_age"],
        "old_condition_id": data["old_condition_id"],
        "slow_action_shape": list(data["slow_action"].shape),
        "slow_hidden_shape": list(data["slow_hidden"].shape),
    }


def main(args: argparse.Namespace) -> None:
    root = Path(args.data_dir).expanduser().resolve()
    trajectory_root = root / "trajectories"
    condition_root = root / "conditions"
    trajectory_names = sorted(path.name for path in trajectory_root.iterdir() if path.is_dir())
    condition_names = sorted(path.name for path in condition_root.iterdir() if path.is_dir())
    if trajectory_names != condition_names:
        raise ValueError("Trajectory and condition directory sets differ")
    if not trajectory_names:
        raise ValueError("No trajectories found")

    structural = []
    for name in trajectory_names:
        frames = numbered_files(trajectory_root / name, "step_*.npz", "step_{:04d}.npz")
        conditions = numbered_files(condition_root / name, "condition_*.pt", "condition_{:03d}.pt")
        if not frames or not conditions:
            raise ValueError(f"Empty trajectory or condition set: {name}")
        if any(path.stat().st_size == 0 for path in frames + conditions):
            raise ValueError(f"Zero-byte file found in {name}")
        structural.append((name, len(frames), len(conditions)))

    selected_name = args.trajectory or max(
        trajectory_names,
        key=lambda name: (trajectory_root / name).stat().st_mtime,
    )
    if selected_name not in set(trajectory_names):
        raise ValueError(f"Unknown trajectory: {selected_name}")
    frame_files = numbered_files(trajectory_root / selected_name, "step_*.npz", "step_{:04d}.npz")
    condition_files = numbered_files(condition_root / selected_name, "condition_*.pt", "condition_{:03d}.pt")
    for path in frame_files:
        check_frame(path)
    condition_summary = [
        check_condition(path, index, len(frame_files))
        for index, path in enumerate(condition_files)
    ]

    print(json.dumps({
        "status": "ok",
        "data_dir": root.as_posix(),
        "trajectory_directories": len(trajectory_names),
        "condition_directories": len(condition_names),
        "total_frame_files": sum(item[1] for item in structural),
        "total_condition_files": sum(item[2] for item in structural),
        "fully_loaded_trajectory": selected_name,
        "fully_loaded_frames": len(frame_files),
        "fully_loaded_conditions": len(condition_files),
        "first_frame": frame_files[0].name,
        "last_frame": frame_files[-1].name,
        "conditions": condition_summary,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--trajectory", default=None)
    main(parser.parse_args())
