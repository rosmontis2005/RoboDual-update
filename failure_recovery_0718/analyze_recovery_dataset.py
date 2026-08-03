#!/usr/bin/env python3
"""Validate and summarize a state-grouped failure-recovery branch dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 10000) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def analyze(
    data_dir: Path,
    min_branchable_states: int,
    seed: int,
    required_validation_tasks: set[str] | None = None,
) -> dict:
    root = data_dir.expanduser().resolve()
    required_validation_tasks = required_validation_tasks or set()
    summary_path = root / "collection_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    else:
        progress_path = root / "collection_progress.json"
        summary = {
            "format": "robodual_failure_recovery_branch_v1_in_progress",
            "status": "collecting",
            "progress": json.loads(progress_path.read_text()) if progress_path.is_file() else {},
            "exact_branch_audits": [],
        }
    states = read_jsonl(root / "failure_states.jsonl")
    branches = read_jsonl(root / "branches.jsonl")
    pairs = read_jsonl(root / "pairs.jsonl")
    chunks_path = root / "trajectory_chunks.jsonl"
    trajectory_chunks = read_jsonl(chunks_path) if chunks_path.is_file() else []
    state_by_id = {item["failure_state_id"]: item for item in states}
    branch_by_id = {item["branch_id"]: item for item in branches}
    branch_actions = {}
    chunks_by_branch = defaultdict(list)
    errors = []

    if len(state_by_id) != len(states):
        errors.append("duplicate failure_state_id")
    if len(branch_by_id) != len(branches):
        errors.append("duplicate branch_id")
    for state_id in state_by_id:
        state_payload = root / "states" / f"{state_id}.npz"
        bullet_payload = root / "states" / f"{state_id}.bullet"
        model_payload = root / "states" / f"{state_id}_model.pt"
        simulator_payload = root / "states" / f"{state_id}_simulator.pt"
        if not state_payload.is_file():
            errors.append(f"missing state payload: {state_id}")
        if not bullet_payload.is_file() or not model_payload.is_file() or not simulator_payload.is_file():
            errors.append(f"missing persistent replay payload: {state_id}")

    branch_groups = defaultdict(list)
    for branch in branches:
        state_id = branch["failure_state_id"]
        branch_groups[state_id].append(branch)
        if state_id not in state_by_id:
            errors.append(f"branch references missing state: {branch['branch_id']}")
            continue
        if branch["split"] != state_by_id[state_id]["split"]:
            errors.append(f"split leakage: {branch['branch_id']}")
        branch_path = root / "branches" / f"{branch['branch_id']}.npz"
        condition_path = root / "conditions" / f"{branch['branch_id']}.pt"
        if not branch_path.is_file() or not condition_path.is_file():
            errors.append(f"missing branch payload: {branch['branch_id']}")
            continue
        with np.load(branch_path, allow_pickle=False) as payload:
            actions = payload["actions"]
            branch_actions[branch["branch_id"]] = np.asarray(
                actions, dtype=np.float32
            ).copy()
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                errors.append(f"bad actions: {branch['branch_id']} {actions.shape}")
            if len(actions) != int(branch["steps"]):
                errors.append(f"step mismatch: {branch['branch_id']}")
        condition = torch.load(condition_path, map_location="cpu", weights_only=False)
        required = {"slow_action", "slow_hidden", "slow_age", "strategy"}
        if not required.issubset(condition):
            errors.append(f"bad condition: {branch['branch_id']}")

    chunk_ids = set()
    for chunk in trajectory_chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in chunk_ids:
            errors.append(f"duplicate trajectory chunk: {chunk_id}")
        chunk_ids.add(chunk_id)
        if chunk["branch_id"] not in branch_by_id:
            errors.append(f"trajectory chunk references missing branch: {chunk_id}")
            continue
        branch = branch_by_id[chunk["branch_id"]]
        state = state_by_id.get(branch["failure_state_id"])
        chunks_by_branch[chunk["branch_id"]].append(chunk)
        if chunk.get("failure_state_id") != branch["failure_state_id"]:
            errors.append(f"trajectory chunk state mismatch: {chunk_id}")
        if chunk.get("split") != branch["split"]:
            errors.append(f"trajectory chunk split mismatch: {chunk_id}")
        if state is not None and chunk.get("task") != state["task"]:
            errors.append(f"trajectory chunk task mismatch: {chunk_id}")
        if chunk.get("strategy") != branch["strategy"]:
            errors.append(f"trajectory chunk strategy mismatch: {chunk_id}")
        payload_path = root / "trajectory_chunks" / f"{chunk_id}.npz"
        condition_path = root / "trajectory_conditions" / f"{chunk_id}.pt"
        if not payload_path.is_file() or not condition_path.is_file():
            errors.append(f"missing trajectory chunk payload: {chunk_id}")
            continue
        with np.load(payload_path, allow_pickle=False) as payload:
            actions = payload["actions"]
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                errors.append(f"bad trajectory actions: {chunk_id} {actions.shape}")
            if len(actions) != int(chunk["steps"]):
                errors.append(f"trajectory step mismatch: {chunk_id}")
            start = int(chunk["start_offset"])
            expected = branch_actions.get(chunk["branch_id"])
            if (
                expected is None
                or start < 0
                or start + len(actions) > len(expected)
                or not np.allclose(
                    actions,
                    expected[start:start + len(actions)],
                    rtol=0.0,
                    atol=1e-6,
                )
            ):
                errors.append(f"trajectory action slice mismatch: {chunk_id}")
            required_arrays = {
                "robot_obs", "rgb_static", "rgb_gripper", "depth_static",
                "depth_gripper", "previous_rgb", "hist_action",
            }
            if not required_arrays.issubset(payload.files):
                errors.append(f"bad trajectory observation payload: {chunk_id}")
        condition = torch.load(
            condition_path, map_location="cpu", weights_only=False
        )
        required = {"slow_action", "slow_hidden", "slow_age", "strategy"}
        if not required.issubset(condition):
            errors.append(f"bad trajectory condition: {chunk_id}")
        elif condition["strategy"] != chunk["strategy"]:
            errors.append(f"trajectory condition strategy mismatch: {chunk_id}")

    pair_states = set()
    for pair in pairs:
        state_id = pair["failure_state_id"]
        positive = branch_by_id.get(pair["positive_branch_id"])
        negative = branch_by_id.get(pair["negative_branch_id"])
        if positive is None or negative is None:
            errors.append(f"pair references missing branch: {pair['pair_id']}")
            continue
        if positive["failure_state_id"] != state_id or negative["failure_state_id"] != state_id:
            errors.append(f"cross-state pair: {pair['pair_id']}")
        if not positive["success"] or negative["success"]:
            errors.append(f"invalid pair labels: {pair['pair_id']}")
        if pair["split"] != state_by_id[state_id]["split"]:
            errors.append(f"pair split leakage: {pair['pair_id']}")
        pair_states.add(state_id)

    strategy_success = Counter()
    strategy_total = Counter()
    task_success = Counter()
    task_total = Counter()
    for branch in branches:
        strategy_total[branch["strategy"]] += 1
        strategy_success[branch["strategy"]] += int(branch["success"])
        task = state_by_id[branch["failure_state_id"]]["task"]
        task_total[task] += 1
        task_success[task] += int(branch["success"])

    state_strategy_advantages = []
    for state_id, group in branch_groups.items():
        base = [float(item["success"]) for item in group if item["strategy"] == "base_seed"]
        refresh = [float(item["success"]) for item in group if item["strategy"] == "forced_refresh"]
        if base and refresh:
            state_strategy_advantages.append(float(np.mean(refresh) - np.mean(base)))

    persistent_audit_path = root / "persistent_replay_audit.json"
    persistent_audit = (
        json.loads(persistent_audit_path.read_text())
        if persistent_audit_path.is_file()
        else {"passed": False, "reason": "persistent replay audit has not been run"}
    )
    exact_audits = summary.get("exact_branch_audits", [])
    # Bullet contact dynamics are not bitwise deterministic after saveState/restoreState:
    # even a fixed action trace can end at a numerically different contact pose.  The
    # contract needed by this dataset is reproducible actions, rollout length, and
    # oracle label.  Persistent state/observation fidelity is audited independently
    # below with explicit tolerances and in a fresh process.
    persistent_records = persistent_audit.get("records", [])
    audit_records_by_branch = {
        item.get("fixed_branch_id"): item for item in persistent_records
    }
    pair_branch_ids = {
        branch_id
        for pair in pairs
        for branch_id in (
            pair.get("positive_branch_id"),
            pair.get("negative_branch_id"),
        )
        if branch_id
    }
    fixed_action_label_replay_pass = (
        bool(persistent_audit.get("passed"))
        and bool(persistent_audit.get("coverage_complete"))
        and int(persistent_audit.get("positive_records", 0)) > 0
        and int(persistent_audit.get("negative_records", 0)) > 0
        and bool(persistent_records)
        and all(
            item.get("fixed_branch_same_outcome")
            and item.get("fixed_branch_same_length")
            and item.get("fixed_action_replay")
            and item.get("robot_max_abs", float("inf"))
            <= persistent_audit.get("state_tolerance", 0.0)
            and item.get("oracle_source") == "persisted_subtask_start"
            and item.get("restore_contract")
            == "bullet_reset_bullet_v2_gripper_v3"
            for item in persistent_records
        )
        and pair_branch_ids.issubset(audit_records_by_branch)
    )
    split_branchable = Counter(state_by_id[state_id]["split"] for state_id in pair_states)
    eligible_positive_ids = {
        pair["positive_branch_id"]
        for pair in pairs
        if pair.get("positive_branch_id") in branch_by_id
        and int(branch_by_id[pair["positive_branch_id"]]["steps"]) >= 8
    }
    terminal_complete_positive_ids = {
        branch_id
        for branch_id in eligible_positive_ids
        if any(
            int(chunk["steps"]) == 8
            and int(chunk["start_offset"]) + 8
            == int(branch_by_id[branch_id]["steps"])
            for chunk in chunks_by_branch.get(branch_id, [])
        )
        and all(
            int(chunk["steps"]) == 8
            for chunk in chunks_by_branch.get(branch_id, [])
        )
    }
    missing_terminal_positive_ids = sorted(
        eligible_positive_ids - terminal_complete_positive_ids
    )
    eligible_positive_splits = Counter(
        branch_by_id[branch_id]["split"] for branch_id in eligible_positive_ids
    )
    eligible_positive_tasks_by_split = defaultdict(set)
    for branch_id in eligible_positive_ids:
        branch = branch_by_id[branch_id]
        eligible_positive_tasks_by_split[branch["split"]].add(
            state_by_id[branch["failure_state_id"]]["task"]
        )
    untrusted_positive_ids = sorted(
        branch_id
        for branch_id in eligible_positive_ids
        if branch_by_id[branch_id].get("strategy") == "demo_guided_persisted"
        and (
            branch_by_id[branch_id].get("oracle_source")
            != "persisted_subtask_start"
            or branch_by_id[branch_id].get("restore_contract")
            != "bullet_reset_bullet_v2_gripper_v3"
        )
    )
    stable_manifest_path = root / "stable_filter_manifest.json"
    frozen_inventory_match = False
    if stable_manifest_path.is_file():
        stable_manifest = json.loads(stable_manifest_path.read_text())
        expected_state_ids = set(stable_manifest.get("kept_state_ids", []))
        expected_branch_ids = set(
            stable_manifest.get("kept_positive_branch_ids", [])
        ) | set(stable_manifest.get("kept_negative_branch_ids", []))
        expected_result = stable_manifest.get("result", {})
        frozen_inventory_match = (
            set(state_by_id) == expected_state_ids
            and set(branch_by_id) == expected_branch_ids
            and int(expected_result.get("failure_states", -1)) == len(states)
            and int(expected_result.get("branches", -1)) == len(branches)
            and int(expected_result.get("trajectory_chunks", -1))
            == len(trajectory_chunks)
        )
    training_admission = {
        "integrity": not errors,
        "fixed_action_label_replay": fixed_action_label_replay_pass,
        "persistent_replay": (
            bool(persistent_audit.get("passed"))
            and bool(persistent_audit.get("coverage_complete"))
        ),
        "minimum_branchable_states": len(pair_states) >= min_branchable_states,
        "has_train_validation_test": all(split_branchable[name] > 0 for name in ("train", "validation", "test")),
        "trainable_positive_in_each_split": all(
            eligible_positive_splits[name] > 0 for name in ("train", "validation", "test")
        ),
        "complete_terminal_window_for_every_positive": (
            bool(eligible_positive_ids) and not missing_terminal_positive_ids
        ),
        "has_positive_and_negative": bool(pair_states),
        "trusted_positive_provenance": not untrusted_positive_ids,
        "frozen_inventory_matches_manifest": frozen_inventory_match,
        "required_validation_task_coverage": required_validation_tasks.issubset(
            eligible_positive_tasks_by_split["validation"]
        ),
    }
    result = {
        "data_dir": root.as_posix(),
        "format": summary.get("format"),
        "integrity_errors": errors,
        "failure_states": len(states),
        "branchable_failure_states": len(pair_states),
        "branchability_rate": len(pair_states) / len(states) if states else 0.0,
        "branches": len(branches),
        "preference_pairs": len(pairs),
        "states_by_split": dict(Counter(item["split"] for item in states)),
        "branchable_states_by_split": dict(split_branchable),
        "trainable_positive_branches_by_split": dict(eligible_positive_splits),
        "trainable_positive_tasks_by_split": {
            split: sorted(tasks)
            for split, tasks in eligible_positive_tasks_by_split.items()
        },
        "required_validation_tasks": sorted(required_validation_tasks),
        "untrusted_positive_branch_ids": untrusted_positive_ids,
        "missing_terminal_positive_branch_ids": missing_terminal_positive_ids,
        "states_by_task": dict(Counter(item["task"] for item in states)),
        "strategy_success_rate": {
            name: {
                "success": strategy_success[name],
                "total": strategy_total[name],
                "rate": strategy_success[name] / strategy_total[name],
            }
            for name in sorted(strategy_total)
        },
        "task_branch_success_rate": {
            task: {
                "success": task_success[task],
                "total": task_total[task],
                "rate": task_success[task] / task_total[task],
            }
            for task in sorted(task_total)
        },
        "forced_refresh_minus_base_state_mean": (
            float(np.mean(state_strategy_advantages)) if state_strategy_advantages else None
        ),
        "forced_refresh_minus_base_bootstrap_95ci": bootstrap_mean_ci(
            state_strategy_advantages, seed
        ),
        "fixed_action_label_replay_pass": fixed_action_label_replay_pass,
        "fixed_action_terminal_drift": [
            {
                "failure_state_id": item.get("failure_state_id"),
                "robot_max_abs": item.get("final_robot_max_abs"),
                "scene_max_abs": item.get("final_scene_max_abs"),
            }
            for item in exact_audits
        ],
        "persistent_replay_audit": persistent_audit,
        "training_admission": training_admission,
        "training_admitted": all(training_admission.values()),
    }
    (root / "dataset_assessment.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = [
        "# Failure-recovery dataset assessment",
        "",
        f"- Integrity: `{'PASS' if not errors else 'FAIL'}`",
        f"- Fixed-action oracle-label replay: `{'PASS' if fixed_action_label_replay_pass else 'FAIL'}`",
        f"- Failure states: `{len(states)}`",
        f"- Branchable states: `{len(pair_states)}` ({result['branchability_rate']:.1%})",
        f"- Branches / preference pairs: `{len(branches)}` / `{len(pairs)}`",
        f"- Training admission: `{'PASS' if result['training_admitted'] else 'FAIL'}`",
        "",
        "Training is admitted only when integrity, fixed-action label replay, and persistent",
        "state replay pass, the requested",
        "minimum number of branchable states exists, and all three state-grouped splits are populated.",
    ]
    (root / "dataset_assessment.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--min_branchable_states", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--required_validation_tasks",
        default="",
        help="Comma-separated tasks that must have trainable validation positives.",
    )
    args = parser.parse_args()
    if args.min_branchable_states <= 0:
        parser.error("--min_branchable_states must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    analyze(
        parsed.data_dir,
        parsed.min_branchable_states,
        parsed.seed,
        {
            item.strip()
            for item in parsed.required_validation_tasks.split(",")
            if item.strip()
        },
    )
