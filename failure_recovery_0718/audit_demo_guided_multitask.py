#!/usr/bin/env python3
"""Fresh-process physical audit of retargeted CALVIN demo trajectories."""

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
for path in (REPO_ROOT / "vla-scripts", CALVIN_ROOT / "calvin_env", CALVIN_ROOT / "calvin_models"):
    sys.path.insert(0, path.as_posix())

import evaluate_calvin_failure_recovery_scale_0718 as recovery  # noqa: E402
from calvin_env_wrapper import CalvinEnvWrapperRaw  # noqa: E402


def bullet_env(env):
    current = env
    for _ in range(6):
        if hasattr(current, "p") and hasattr(current, "cid"):
            return current
        current = current.env
    raise RuntimeError("CALVIN PyBullet environment not found")


def restore_state(env, bullet, root, state_id):
    bullet.p.restoreState(
        fileName=(root / "states" / f"{state_id}.bullet").as_posix(),
        physicsClientId=bullet.cid,
    )
    simulator = torch.load(
        root / "states" / f"{state_id}_simulator.pt", map_location="cpu", weights_only=False
    )
    bullet.robot.reset_from_storage(simulator["robot"])
    bullet.scene.reset_from_storage(simulator["scene"])
    obs = env.get_obs()
    # Robot.reset_from_storage restores joints but not the relative controller's
    # accumulated target pose. Synchronize it before replaying relative actions.
    bullet.robot.target_pos = np.asarray(obs["robot_obs"][:3], dtype=np.float64).copy()
    bullet.robot.target_orn = np.asarray(obs["robot_obs"][3:6], dtype=np.float64).copy()
    return obs


def held_object(start_info):
    robot_uid = start_info["robot_info"]["uid"]
    names = [
        name
        for name, info in start_info["scene_info"]["movable_objects"].items()
        if robot_uid in {contact[2] for contact in info["contacts"]}
    ]
    return names[0] if len(names) == 1 else None


def physical_postcondition(task, start_info, end_info):
    robot_uid = start_info["robot_info"]["uid"]
    if task == "place_in_slider":
        name = held_object(start_info)
        if name is None:
            return False
        obj = end_info["scene_info"]["movable_objects"][name]
        table = end_info["scene_info"]["fixed_objects"]["table"]
        target = (table["uid"], table["links"]["plank_link"])
        contacts = {(contact[2], contact[4]) for contact in obj["contacts"]}
        return robot_uid not in {contact[2] for contact in obj["contacts"]} and target in contacts
    if task == "stack_block":
        object_uids = {
            item["uid"] for item in start_info["scene_info"]["movable_objects"].values()
        }
        for name, start_obj in start_info["scene_info"]["movable_objects"].items():
            end_obj = end_info["scene_info"]["movable_objects"][name]
            start_contacts = {contact[2] for contact in start_obj["contacts"]}
            end_contacts = {contact[2] for contact in end_obj["contacts"]}
            if (
                not (object_uids & start_contacts)
                and bool(object_uids & end_contacts)
                and not (end_contacts - object_uids)
            ):
                return True
        return False
    object_name, _ = recovery._task_object_name(task)
    start_obj = start_info["scene_info"]["movable_objects"][object_name]
    end_obj = end_info["scene_info"]["movable_objects"][object_name]
    start_pos = np.asarray(start_obj["current_pos"], dtype=np.float64)
    end_pos = np.asarray(end_obj["current_pos"], dtype=np.float64)
    if task.startswith("lift_"):
        threshold = 0.03 if task.endswith("_slider") else 0.05
        return end_pos[2] - start_pos[2] > threshold and robot_uid in {
            contact[2] for contact in end_obj["contacts"]
        }
    if task == "push_pink_block_right":
        return end_pos[0] - start_pos[0] > 0.1
    return False


def audit(args):
    root = Path(args.data_dir).expanduser().resolve()
    states = [json.loads(line) for line in (root / "failure_states.jsonl").read_text().splitlines() if line]
    states = [state for state in states if recovery._demo_guidance_key(state["task"]) is not None]
    selected_tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    if selected_tasks:
        states = [state for state in states if state["task"] in selected_tasks]
    guidance = recovery.load_demo_guidance(args.dataset_subdir)
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
    dataset_root = CALVIN_ROOT / "dataset" / args.dataset_subdir
    env = CalvinEnvWrapperRaw(
        dataset_root / "validation", observation_space, torch.device("cpu"), use_egl=False
    )
    bullet = bullet_env(env)
    records = []
    try:
        for state in states:
            obs = restore_state(env, bullet, root, state["failure_state_id"])
            oracle_path = root / "states" / f"{state['failure_state_id']}_oracle_start.pt"
            start_info = (
                torch.load(oracle_path, map_location="cpu", weights_only=False)
                if oracle_path.exists()
                else env.get_info()
            )
            standard = False
            physical = False
            steps = 0
            for target in recovery._retargeted_demo_targets(env, state["task"], guidance, args.seed):
                action = recovery._relative_demo_action(target, obs["robot_obs"])
                obs, _, _, current_info = env.step(action)
                steps += 1
                standard = bool(task_oracle.get_task_info_for_set(start_info, current_info, {state["task"]}))
                physical = bool(physical_postcondition(state["task"], start_info, current_info))
                if standard or physical or steps >= args.horizon:
                    break
            records.append({
                "failure_state_id": state["failure_state_id"],
                "task": state["task"],
                "steps": steps,
                "snapshot_anchor_standard_oracle": standard,
                "physical_postcondition": physical,
                "object_diagnostic": None if state["task"] in {"place_in_slider", "stack_block"} else (lambda name: {
                    "name": name,
                    "start_position": np.asarray(
                        start_info["scene_info"]["movable_objects"][name]["current_pos"], dtype=float
                    ).tolist(),
                    "end_position": np.asarray(
                        current_info["scene_info"]["movable_objects"][name]["current_pos"], dtype=float
                    ).tolist(),
                    "end_robot_contact": bool(
                        start_info["robot_info"]["uid"] in {
                            contact[2]
                            for contact in current_info["scene_info"]["movable_objects"][name]["contacts"]
                        }
                    ),
                })(recovery._task_object_name(state["task"])[0]),
            })
    finally:
        env.close()
    result = {
        "format": "robodual_demo_guided_multitask_audit_v1",
        "records": records,
        "states": len(records),
        "physical_successes": int(sum(record["physical_postcondition"] for record in records)),
    }
    output = Path(args.output).expanduser().resolve()
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset")
    parser.add_argument("--horizon", default=80, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
