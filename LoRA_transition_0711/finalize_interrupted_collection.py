#!/usr/bin/env python3
"""Finalize manifests from successfully persisted interrupted rollouts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from collect_transition_rollouts import (
    CALVIN_ROOT_PATH,
    DEFAULT_TASK_AGE_GROUP_A,
    DEFAULT_TASK_AGE_GROUP_B,
    DEFAULT_TASK_AGE_GROUP_C,
    DEFAULT_TASK_AGE_GROUP_D,
    SAMPLE_RATIOS,
    Candidate,
    candidate_requirement_keys,
    l2_ee6,
    parse_group_quotas,
    sample_deficits,
    sample_requirements,
    select_samples,
    stable_split,
)


NAME_PATTERN = re.compile(r"^seq(?P<sequence>\d+)_sub(?P<subtask>\d+)_(?P<task>.+)$")
GROUP_TASKS = {
    "A": DEFAULT_TASK_AGE_GROUP_A,
    "B": DEFAULT_TASK_AGE_GROUP_B,
    "C": DEFAULT_TASK_AGE_GROUP_C,
    "D": DEFAULT_TASK_AGE_GROUP_D,
}
GROUP_AGES = {"A": 13, "B": 12, "C": 10, "D": 8}


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def parse_trajectory_name(name: str) -> tuple[int, int, str]:
    match = NAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unexpected trajectory directory name: {name}")
    return int(match["sequence"]), int(match["subtask"]), match["task"]


def build_task_maps() -> tuple[dict[str, str], dict[str, int]]:
    group_map = {}
    age_map = {}
    for group, tasks in GROUP_TASKS.items():
        for task in tasks:
            if task in group_map:
                raise ValueError(f"Duplicate task assignment: {task}")
            group_map[task] = group
            age_map[task] = GROUP_AGES[group]
    return group_map, age_map


def load_actions(frame_files: list[Path]) -> list[np.ndarray]:
    actions = []
    for path in frame_files:
        with np.load(path, allow_pickle=False) as frame:
            action = np.asarray(frame["rel_actions"], dtype=np.float32)
            history = np.asarray(frame["hist_action_before"], dtype=np.float32)
        if action.shape != (7,) or history.shape != (4, 7):
            raise ValueError(f"Bad action/history shape in {path}: {action.shape}, {history.shape}")
        actions.append(action)
    return actions


def load_conditions(condition_files: list[Path], frame_count: int) -> list[dict]:
    conditions = []
    for condition_id, path in enumerate(condition_files):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if int(data["step"]) >= frame_count:
            raise ValueError(f"Condition outside trajectory: {path}")
        conditions.append({
            "condition_id": condition_id,
            "step": int(data["step"]),
            "refresh_age": data["refresh_age"],
            "slow_action": np.asarray(data["slow_action"], dtype=np.float32),
            "old_condition_id": data["old_condition_id"],
        })
    if not conditions or conditions[0]["step"] != 0:
        raise ValueError("Every saved trajectory must start with condition 0 at step 0")
    return conditions


def reconstruct_candidates(
    trajectory_id: str,
    actions: list[np.ndarray],
    conditions: list[dict],
    args: argparse.Namespace,
) -> list[Candidate]:
    split = stable_split(trajectory_id)
    condition_by_step = {condition["step"]: condition for condition in conditions}
    active_condition = conditions[0]
    candidates = []
    for step, action in enumerate(actions):
        refresh_condition = condition_by_step.get(step)
        is_refresh = refresh_condition is not None
        if is_refresh:
            active_condition = refresh_condition
        slow_age = step - active_condition["step"]
        refresh_age = active_condition["refresh_age"] if is_refresh else None
        conflict_prev = None
        conflict_old_new = None
        gripper_change = False
        jerk = None
        if step >= 2:
            jerk = float(np.linalg.norm(action[:6] - 2 * actions[step - 1][:6] + actions[step - 2][:6]))
        if is_refresh and step > 0:
            new_action = active_condition["slow_action"][0]
            conflict_prev = l2_ee6(new_action[0], actions[step - 1])
            gripper_change = bool(np.sign(new_action[0, 6]) != np.sign(actions[step - 1][6]))
            old_id = active_condition["old_condition_id"]
            if old_id is not None and refresh_age is not None:
                old_action = conditions[int(old_id)]["slow_action"][0]
                old_index = min(max(int(refresh_age), 0), old_action.shape[0] - 1)
                conflict_old_new = l2_ee6(old_action[old_index], new_action[0])

        category = "refresh" if is_refresh and step > 0 else "normal"
        if category == "refresh" and (
            conflict_prev >= args.high_conflict_prev_threshold
            or conflict_old_new >= args.high_conflict_old_new_threshold
            or (jerk is not None and jerk >= args.high_conflict_jerk_threshold)
        ):
            category = "high_conflict"
        elif not is_refresh and slow_age >= args.empty_ref_after_age:
            category = "stale"

        if step >= args.history_steps and step + args.action_chunk_size <= len(actions):
            candidates.append(Candidate(
                trajectory_id=trajectory_id,
                split=split,
                step=step,
                category=category,
                condition_id=int(active_condition["condition_id"]),
                old_condition_id=active_condition["old_condition_id"] if is_refresh else None,
                slow_age=int(slow_age),
                refresh_age=None if refresh_age is None else int(refresh_age),
                conflict_prev_l2_ee6=conflict_prev,
                conflict_old_new_l2_ee6=conflict_old_new,
                action_delta_l2_ee6=None if step == 0 else l2_ee6(action, actions[step - 1]),
                action_jerk_l2_ee6=jerk,
                gripper_intent_change=gripper_change,
            ))
    return candidates


def finalize(args: argparse.Namespace) -> dict:
    root = Path(args.data_dir).expanduser().resolve()
    trajectory_root = root / "trajectories"
    condition_root = root / "conditions"
    names = sorted(path.name for path in trajectory_root.iterdir() if path.is_dir())
    condition_names = sorted(path.name for path in condition_root.iterdir() if path.is_dir())
    if names != condition_names:
        raise ValueError("Trajectory and condition directory sets differ")
    for manifest_name in ("samples.jsonl", "trajectories.jsonl", "collection_summary.json"):
        if (root / manifest_name).exists() and not args.overwrite_manifests:
            raise FileExistsError(f"{root / manifest_name} already exists; use --overwrite_manifests")

    annotation_path = CALVIN_ROOT_PATH / "calvin_models/conf/annotations/new_playtable_validation.yaml"
    annotations = OmegaConf.load(annotation_path)
    group_map, age_map = build_task_maps()
    group_quotas = parse_group_quotas(args.group_trajectory_quotas)
    group_counts = Counter()
    quota_task_counts = Counter()
    task_counts = Counter()
    candidate_counts = Counter()
    all_candidates = []
    trajectory_records = []

    for trajectory_id in names:
        sequence_i, subtask_i, task = parse_trajectory_name(trajectory_id)
        group = group_map[task]
        frame_files = sorted((trajectory_root / trajectory_id).glob("step_*.npz"))
        condition_files = sorted((condition_root / trajectory_id).glob("condition_*.pt"))
        actions = load_actions(frame_files)
        conditions = load_conditions(condition_files, len(actions))
        candidates = reconstruct_candidates(trajectory_id, actions, conditions, args)
        all_candidates.extend(candidates)
        candidate_counts.update(key for candidate in candidates for key in candidate_requirement_keys(candidate))
        task_counts[task] += 1

        below_cap = group == "D" or quota_task_counts[task] < args.max_trajectories_per_task
        counted_toward_quota = group_counts[group] < group_quotas[group] and below_cap
        if counted_toward_quota:
            group_counts[group] += 1
            quota_task_counts[task] += 1
        trajectory_records.append({
            "trajectory_id": trajectory_id,
            "split": stable_split(trajectory_id),
            "sequence_i": sequence_i,
            "subtask_i": subtask_i,
            "task": task,
            "task_group": group,
            "task_max_slow_age": age_map[task],
            "instruction": str(annotations[task][0]),
            "steps": len(actions),
            "conditions": len(conditions),
            "candidate_samples": len(candidates),
            "counted_toward_group_quota": counted_toward_quota,
        })

    requirements = sample_requirements(args.target_samples)
    category_deficits = sample_deficits(candidate_counts, requirements)
    selected, selection_stats = select_samples(all_candidates, args.target_samples, args.seed)
    if category_deficits or len(selected) != args.target_samples:
        raise RuntimeError(
            f"Cannot finalize requested dataset: selected={len(selected)}, "
            f"target={args.target_samples}, deficits={category_deficits}"
        )
    sample_records = []
    for sample_id, candidate in enumerate(selected):
        payload = asdict(candidate)
        payload.update({"sample_id": sample_id, "history_steps": args.history_steps, "action_chunk_size": args.action_chunk_size})
        sample_records.append(payload)

    missing_groups = {
        group: group_quotas[group] - group_counts[group]
        for group in group_quotas if group_counts[group] < group_quotas[group]
    }
    selected_by_split = Counter(candidate.split for candidate in selected)
    status = "complete" if not missing_groups else "usable_with_group_undercoverage"
    summary = {
        "format": "robodual_transition_lora_v1",
        "status": status,
        "finalized_from_interrupted_collection": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": root.as_posix(),
        "successful_trajectories": len(trajectory_records),
        "trajectory_group_quotas": group_quotas,
        "saved_by_group": dict(group_counts),
        "saved_by_task": dict(sorted(task_counts.items())),
        "split_trajectories": dict(Counter(record["split"] for record in trajectory_records)),
        "selection": selection_stats,
        "sample_ratios": SAMPLE_RATIOS,
        "selected_by_split": dict(selected_by_split),
        "candidate_requirements": {f"{split}:{category}": count for (split, category), count in requirements.items()},
        "candidate_available": {f"{split}:{category}": candidate_counts[(split, category)] for split, category in requirements},
        "missing_groups": missing_groups,
        "category_deficits": category_deficits,
        "args": vars(args),
        "integrity": {
            "conditions": "reloaded from persisted online current-observation slow calls",
            "history": "validated as [4,7] in every persisted frame",
            "targets": "future actions from the same successful online rollout",
            "split": "stable trajectory-level SHA256 split",
        },
    }
    atomic_jsonl(root / "samples.jsonl", sample_records)
    atomic_jsonl(root / "trajectories.jsonl", trajectory_records)
    atomic_json(root / "collection_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--target_samples", type=int, default=8000)
    parser.add_argument("--group_trajectory_quotas", default="A:60,B:60,C:30,D:20")
    parser.add_argument("--max_trajectories_per_task", type=int, default=8)
    parser.add_argument("--history_steps", type=int, default=4)
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--empty_ref_after_age", type=int, default=8)
    parser.add_argument("--high_conflict_prev_threshold", type=float, default=0.18)
    parser.add_argument("--high_conflict_old_new_threshold", type=float, default=0.18)
    parser.add_argument("--high_conflict_jerk_threshold", type=float, default=0.24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite_manifests", action="store_true")
    finalize(parser.parse_args())
