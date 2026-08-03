#!/usr/bin/env python3
"""Audit a CALVIN-demo-guided controller from persisted failure states.

This is deliberately separate from the formal collector: no recovery manifest is
modified until a controller has demonstrated oracle-labelled recovery after a
fresh persistent restore.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CALVIN_ROOT = Path(os.environ.get("CALVIN_ROOT", REPO_ROOT.parent / "calvin")).resolve()
for path in (
    CALVIN_ROOT / "calvin_env",
    CALVIN_ROOT / "calvin_models",
    REPO_ROOT / "vla-scripts",
):
    sys.path.insert(0, path.as_posix())

from calvin_env.utils.utils import angle_between_angles  # noqa: E402
from calvin_env_wrapper import CalvinEnvWrapperRaw  # noqa: E402


def bullet_env(env):
    current = env
    for _ in range(6):
        if hasattr(current, "p") and hasattr(current, "cid"):
            return current
        current = current.env
    raise RuntimeError("CALVIN PyBullet environment not found")


def relative_action(target, robot_obs):
    """CALVIN's absolute target pose converted to its normalized relative action."""
    target = np.asarray(target, dtype=np.float64)
    action = np.empty(7, dtype=np.float32)
    action[:3] = np.clip((target[:3] - robot_obs[:3]) / 0.02, -1.0, 1.0)
    action[3:6] = np.clip(angle_between_angles(robot_obs[3:6], target[3:6]) / 0.05, -1.0, 1.0)
    action[6] = -1.0 if target[6] < 0 else 1.0
    return action


def interpolate_pose(start, end, steps, gripper):
    for alpha in np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[1:]:
        pose = start.copy()
        pose[:3] = start[:3] + alpha * (end[:3] - start[:3])
        pose[3:6] = start[3:6] + alpha * angle_between_angles(start[3:6], end[3:6])
        yield np.concatenate((pose, [gripper]))


def place_in_slider_targets(robot_obs, release_pose):
    """Retarget a demonstrated release pose with collision-safe Cartesian legs."""
    current = np.asarray(robot_obs[:6], dtype=np.float64)
    safe_z = max(float(current[2]), 0.58)
    raised = current.copy()
    raised[2] = safe_z
    above = np.asarray(release_pose[:6], dtype=np.float64).copy()
    above[2] = safe_z
    release = np.asarray(release_pose[:6], dtype=np.float64).copy()
    retreat = release.copy()
    retreat[2] = safe_z
    yield from interpolate_pose(current, raised, 8, -1.0)
    yield from interpolate_pose(raised, above, 24, -1.0)
    yield from interpolate_pose(above, release, 12, -1.0)
    for _ in range(8):
        yield np.concatenate((release, [1.0]))
    yield from interpolate_pose(release, retreat, 12, 1.0)


def load_demo_release(dataset_root):
    """Use the validation demo whose slider joint setting is 0.28."""
    annotation = np.load(
        dataset_root / "validation" / "lang_annotations" / "auto_lang_ann.npy",
        allow_pickle=True,
    ).item()
    for task, (start, end) in zip(annotation["language"]["task"], annotation["info"]["indx"]):
        if task != "place_in_slider":
            continue
        frames = []
        for frame_id in range(int(start), int(end) + 1):
            with np.load(dataset_root / "validation" / f"episode_{frame_id:07d}.npz") as frame:
                frames.append(np.asarray(frame["actions"], dtype=np.float64))
        closed = [action for action in frames if action[-1] < 0]
        if closed:
            return closed[-1], {"split": "validation", "start": int(start), "end": int(end)}
    raise RuntimeError("No place_in_slider demonstration found")


def restore_state(env, bullet, root, state_id):
    bullet.p.restoreState(
        fileName=(root / "states" / f"{state_id}.bullet").as_posix(),
        physicsClientId=bullet.cid,
    )
    simulator = torch.load(
        root / "states" / f"{state_id}_simulator.pt",
        map_location="cpu",
        weights_only=False,
    )
    bullet.robot.reset_from_storage(simulator["robot"])
    bullet.scene.reset_from_storage(simulator["scene"])
    obs = env.get_obs()
    bullet.robot.target_pos = np.asarray(obs["robot_obs"][:3], dtype=np.float64).copy()
    bullet.robot.target_orn = np.asarray(obs["robot_obs"][3:6], dtype=np.float64).copy()
    return obs


def place_recovery_postcondition(start_info, end_info):
    """Check the place goal without pretending the failure snapshot is task start.

    The canonical CALVIN oracle must be anchored before the place subtask. Older
    recovery payloads did not persist that anchor, so this audit identifies the
    movable object held at the snapshot and checks the same required end contact.
    """
    robot_uid = start_info["robot_info"]["uid"]
    held = [
        (name, info)
        for name, info in start_info["scene_info"]["movable_objects"].items()
        if robot_uid in {contact[2] for contact in info["contacts"]}
    ]
    if len(held) != 1:
        return False
    name, _ = held[0]
    end_object = end_info["scene_info"]["movable_objects"][name]
    if robot_uid in {contact[2] for contact in end_object["contacts"]}:
        return False
    table = end_info["scene_info"]["fixed_objects"]["table"]
    target = (table["uid"], table["links"]["plank_link"])
    return target in {(contact[2], contact[4]) for contact in end_object["contacts"]}


def audit(args):
    root = Path(args.data_dir).expanduser().resolve()
    dataset_root = CALVIN_ROOT / "dataset" / args.dataset_subdir
    states = [
        json.loads(line)
        for line in (root / "failure_states.jsonl").read_text().splitlines()
        if line.strip()
    ]
    states = [state for state in states if state["task"] == "place_in_slider"]
    release_pose, demo = load_demo_release(dataset_root)
    task_cfg = OmegaConf.load(
        CALVIN_ROOT / "calvin_models" / "conf" / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    env = CalvinEnvWrapperRaw(
        dataset_root / "validation",
        observation_space,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_egl=args.use_egl,
    )
    bullet = bullet_env(env)
    records = []
    try:
        for state in states:
            obs = restore_state(env, bullet, root, state["failure_state_id"])
            start_info = env.get_info()
            initial_robot = np.asarray(obs["robot_obs"], dtype=np.float64).copy()
            standard_oracle_success = False
            recovery_postcondition_success = False
            executed = []
            for target in place_in_slider_targets(initial_robot, release_pose):
                action = relative_action(target, obs["robot_obs"])
                executed.append(action.copy())
                obs, _, _, current_info = env.step(action)
                if task_oracle.get_task_info_for_set(start_info, current_info, {state["task"]}):
                    standard_oracle_success = True
                    recovery_postcondition_success = True
                    break
                if place_recovery_postcondition(start_info, current_info):
                    recovery_postcondition_success = True
                    break
            final_info = env.get_info()
            robot_uid = start_info["robot_info"]["uid"]
            records.append({
                "failure_state_id": state["failure_state_id"],
                "task": state["task"],
                "success": recovery_postcondition_success,
                "standard_oracle_success_with_snapshot_anchor": standard_oracle_success,
                "recovery_postcondition_success": recovery_postcondition_success,
                "steps": len(executed),
                "initial_tcp": initial_robot[:7].tolist(),
                "final_tcp": np.asarray(obs["robot_obs"][:7], dtype=np.float64).tolist(),
                "start_robot_contact_uids": sorted(set(
                    int(contact[2]) for contact in start_info["robot_info"]["contacts"]
                )),
                "start_objects": {
                    name: {
                        "position": np.asarray(info["current_pos"], dtype=np.float64).tolist(),
                        "contact_uids": sorted(set(int(contact[2]) for contact in info["contacts"])),
                        "robot_contact": robot_uid in set(contact[2] for contact in info["contacts"]),
                    }
                    for name, info in start_info["scene_info"]["movable_objects"].items()
                },
                "final_objects": {
                    name: {
                        "position": np.asarray(info["current_pos"], dtype=np.float64).tolist(),
                        "contact_links": sorted(set(
                            (int(contact[2]), int(contact[4])) for contact in info["contacts"]
                        )),
                        "robot_contact": robot_uid in set(contact[2] for contact in info["contacts"]),
                    }
                    for name, info in final_info["scene_info"]["movable_objects"].items()
                },
                "actions": np.asarray(executed, dtype=np.float32).tolist(),
            })
    finally:
        env.close()
    result = {
        "format": "robodual_demo_guided_recovery_audit_v1",
        "data_dir": root.as_posix(),
        "demo": demo,
        "release_pose": release_pose.tolist(),
        "records": records,
        "successes": sum(record["success"] for record in records),
        "states": len(records),
    }
    output = Path(args.output).expanduser().resolve()
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--use_egl", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
