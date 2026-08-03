#!/usr/bin/env python3
"""Audit persisted PyBullet recovery worlds in a fresh CALVIN environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CALVIN_ROOT = Path(os.environ.get("CALVIN_ROOT", REPO_ROOT.parent / "calvin")).resolve()
for path in (
    REPO_ROOT / "vla-scripts",
    CALVIN_ROOT / "calvin_env",
    CALVIN_ROOT / "calvin_models",
    CALVIN_ROOT / "calvin_env" / "tacto",
):
    sys.path.insert(0, path.as_posix())

from calvin_env_wrapper import CalvinEnvWrapperRaw  # noqa: E402


def bullet_env(env):
    current = env
    for _ in range(6):
        if hasattr(current, "p") and hasattr(current, "cid"):
            return current
        current = current.env
    raise RuntimeError("CALVIN PyBullet environment not found")


def branchable_audit_plan(states, branches, max_states=0):
    """Select every trainable positive and one negative per branchable state."""

    branch_groups = {}
    for branch in branches:
        branch_groups.setdefault(branch["failure_state_id"], []).append(branch)
    planned = []
    for state in states:
        group = branch_groups.get(state["failure_state_id"], [])
        positives = [
            item for item in group
            if item.get("success") and int(item.get("steps", 0)) >= 8
        ]
        negatives = [item for item in group if not item.get("success")]
        if positives and negatives:
            planned.append((state, positives, negatives[0]))
    if max_states:
        planned = planned[:max_states]
    return planned


def restore_persisted_failure_state(root, state_id, env, bullet):
    """Match the audited bullet-reset-bullet restore contract."""

    bullet_path = root / "states" / f"{state_id}.bullet"
    bullet.p.restoreState(
        fileName=bullet_path.as_posix(),
        physicsClientId=bullet.cid,
    )
    simulator = torch.load(
        root / "states" / f"{state_id}_simulator.pt",
        map_location="cpu",
        weights_only=False,
    )
    bullet.robot.reset_from_storage(simulator["robot"])
    bullet.scene.reset_from_storage(simulator["scene"])
    bullet.p.restoreState(
        fileName=bullet_path.as_posix(),
        physicsClientId=bullet.cid,
    )
    gripper_action = int(np.asarray(simulator["robot"]["gripper_action"]).item())
    bullet.robot.gripper_action = gripper_action
    bullet.robot.control_gripper(gripper_action)
    actual = env.get_obs()
    bullet.robot.target_pos = np.asarray(
        actual["robot_obs"][:3], dtype=np.float64
    ).copy()
    bullet.robot.target_orn = np.asarray(
        actual["robot_obs"][3:6], dtype=np.float64
    ).copy()
    return actual


def audit(args):
    root = Path(args.data_dir).expanduser().resolve()
    states = [json.loads(line) for line in (root / "failure_states.jsonl").read_text().splitlines() if line]
    branches = [json.loads(line) for line in (root / "branches.jsonl").read_text().splitlines() if line]
    plan = branchable_audit_plan(states, branches, args.max_states)
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
        CALVIN_ROOT / "dataset" / args.dataset_subdir / "validation",
        observation_space,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_egl=args.use_egl,
    )
    bullet = bullet_env(env)
    records = []
    missing_oracle_state_ids = []
    try:
        for state, positives, negative in plan:
            state_id = state["failure_state_id"]
            with np.load(root / "states" / f"{state_id}.npz", allow_pickle=False) as payload:
                expected = {key: payload[key] for key in payload.files}
            oracle_start_path = root / "states" / f"{state_id}_oracle_start.pt"
            if not oracle_start_path.is_file():
                missing_oracle_state_ids.append(state_id)
                continue
            start_info = torch.load(
                oracle_start_path, map_location="cpu", weights_only=False
            )
            cases = [("positive", branch) for branch in positives]
            cases.append(("negative", negative))
            for outcome, branch in cases:
                actual = restore_persisted_failure_state(root, state_id, env, bullet)
                with np.load(
                    root / "branches" / f"{branch['branch_id']}.npz",
                    allow_pickle=False,
                ) as payload:
                    actions = np.asarray(payload["actions"], dtype=np.float32)
                first_success_step = None
                replay_steps = 0
                for action in actions:
                    _, _, _, current_info = env.step(action.copy())
                    replay_steps += 1
                    if (
                        first_success_step is None
                        and task_oracle.get_task_info_for_set(
                            start_info, current_info, {state["task"]}
                        )
                    ):
                        first_success_step = replay_steps
                replay_success = first_success_step is not None
                expected_steps = int(branch["steps"])
                records.append({
                    "failure_state_id": state_id,
                    "split": state["split"],
                    "task": state["task"],
                    "audited_outcome": outcome,
                    "oracle_source": "persisted_subtask_start",
                    "restore_contract": "bullet_reset_bullet_v2_gripper_v3",
                    "robot_diff": [
                        float(value)
                        for value in (
                            actual["robot_obs"] - expected["robot_obs"]
                        ).reshape(-1)
                    ],
                    "robot_max_abs": float(np.max(np.abs(
                        actual["robot_obs"] - expected["robot_obs"]
                    ))),
                    "robot_ee6_max_abs": float(np.max(np.abs(
                        actual["robot_obs"][:6] - expected["robot_obs"][:6]
                    ))),
                    "scene_max_abs": float(np.max(np.abs(
                        actual["scene_obs"] - expected["scene_obs"]
                    ))),
                    "rgb_static_mean_abs": float(np.mean(np.abs(
                        actual["rgb_obs"]["rgb_static"].astype(np.float32)
                        - expected["rgb_static"].astype(np.float32)
                    ))),
                    "fixed_branch_id": branch["branch_id"],
                    "fixed_branch_expected_success": bool(branch["success"]),
                    "fixed_branch_replay_success": replay_success,
                    "fixed_branch_first_success_step": first_success_step,
                    "fixed_branch_same_outcome": bool(
                        branch["success"] == replay_success
                    ),
                    "fixed_branch_expected_steps": expected_steps,
                    "fixed_branch_replay_steps": replay_steps,
                    "fixed_branch_same_length": bool(
                        expected_steps == replay_steps
                    ),
                    "fixed_action_replay": True,
                })
    finally:
        env.close()
    planned_state_ids = [state["failure_state_id"] for state, _, _ in plan]
    planned_positive_branch_ids = [
        branch["branch_id"]
        for _, positives, _ in plan
        for branch in positives
    ]
    audited_outcomes = {
        (item["failure_state_id"], item["audited_outcome"]) for item in records
    }
    coverage_complete = (
        not missing_oracle_state_ids
        and all(
            (state_id, outcome) in audited_outcomes
            for state_id in planned_state_ids
            for outcome in ("positive", "negative")
        )
    )
    passed = bool(records) and coverage_complete and all(
        item["robot_max_abs"] <= args.state_tolerance
        and item["scene_max_abs"] <= args.state_tolerance
        and item["rgb_static_mean_abs"] <= args.rgb_tolerance
        and item["fixed_branch_same_outcome"]
        and item["fixed_branch_same_length"]
        for item in records
    )
    result = {
        "data_dir": root.as_posix(),
        "passed": passed,
        "coverage_complete": coverage_complete,
        "planned_branchable_state_ids": planned_state_ids,
        "planned_positive_branch_ids": planned_positive_branch_ids,
        "missing_oracle_state_ids": missing_oracle_state_ids,
        "positive_records": sum(
            item["audited_outcome"] == "positive" for item in records
        ),
        "negative_records": sum(
            item["audited_outcome"] == "negative" for item in records
        ),
        "state_tolerance": args.state_tolerance,
        "rgb_tolerance": args.rgb_tolerance,
        "records": records,
    }
    (root / "persistent_replay_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Persistent PyBullet replay audit failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset")
    parser.add_argument(
        "--max_states",
        type=int,
        default=0,
        help="Diagnostic cap on branchable states; 0 audits all branchable states.",
    )
    parser.add_argument("--state_tolerance", type=float, default=1e-3)
    parser.add_argument("--rgb_tolerance", type=float, default=0.05)
    parser.add_argument("--use_egl", action="store_true")
    args = parser.parse_args()
    if args.max_states < 0 or args.state_tolerance <= 0 or args.rgb_tolerance <= 0:
        parser.error("audit limits and tolerances must be positive")
    return args


if __name__ == "__main__":
    audit(parse_args())
