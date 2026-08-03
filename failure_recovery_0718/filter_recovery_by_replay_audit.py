#!/usr/bin/env python3
"""Keep only states and branches proven stable by comprehensive replay audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from repair_recovery_dataset_v3 import rebuild_pairs, read_jsonl, write_jsonl


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def filter_dataset(source: Path, output: Path, max_pairs: int) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Filtered output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "states",
        "branches",
        "conditions",
        "trajectory_chunks",
        "trajectory_conditions",
    ):
        (output / name).mkdir()

    audit_path = source / "persistent_replay_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("coverage_complete"):
        raise RuntimeError("Cannot filter an incomplete persistent replay audit")
    records = audit["records"]
    state_fidelity = {}
    stable_positive_ids = set()
    stable_negative_ids = set()
    for record in records:
        state_id = record["failure_state_id"]
        fidelity = (
            record["robot_max_abs"] <= audit["state_tolerance"]
            and record["scene_max_abs"] <= audit["state_tolerance"]
            and record["rgb_static_mean_abs"] <= audit["rgb_tolerance"]
        )
        state_fidelity[state_id] = state_fidelity.get(state_id, True) and fidelity
        stable = (
            record["fixed_branch_same_outcome"]
            and record["fixed_branch_same_length"]
            and record.get("fixed_action_replay") is True
            and record["oracle_source"] == "persisted_subtask_start"
            and record["restore_contract"] == "bullet_reset_bullet_v2_gripper_v3"
        )
        if stable and record["audited_outcome"] == "positive":
            stable_positive_ids.add(record["fixed_branch_id"])
        if stable and record["audited_outcome"] == "negative":
            stable_negative_ids.add(record["fixed_branch_id"])

    branches = read_jsonl(source / "branches.jsonl")
    branch_by_id = {item["branch_id"]: item for item in branches}
    positive_states = {
        branch_by_id[branch_id]["failure_state_id"]
        for branch_id in stable_positive_ids
    }
    negative_states = {
        branch_by_id[branch_id]["failure_state_id"]
        for branch_id in stable_negative_ids
    }
    kept_state_ids = {
        state_id
        for state_id in positive_states & negative_states
        if state_fidelity.get(state_id, False)
    }
    states = [
        item for item in read_jsonl(source / "failure_states.jsonl")
        if item["failure_state_id"] in kept_state_ids
    ]
    kept_branch_ids = {
        branch_id
        for branch_id in stable_positive_ids | stable_negative_ids
        if branch_by_id[branch_id]["failure_state_id"] in kept_state_ids
    }
    kept_branches = [
        item for item in branches if item["branch_id"] in kept_branch_ids
    ]

    for state in states:
        state_id = state["failure_state_id"]
        for suffix in (
            ".npz",
            ".bullet",
            "_model.pt",
            "_simulator.pt",
            "_oracle_start.pt",
        ):
            copy(
                source / "states" / f"{state_id}{suffix}",
                output / "states" / f"{state_id}{suffix}",
            )
    for branch in kept_branches:
        branch_id = branch["branch_id"]
        copy(
            source / "branches" / f"{branch_id}.npz",
            output / "branches" / f"{branch_id}.npz",
        )
        copy(
            source / "conditions" / f"{branch_id}.pt",
            output / "conditions" / f"{branch_id}.pt",
        )
    source_chunks_path = source / "trajectory_chunks.jsonl"
    source_chunks = (
        read_jsonl(source_chunks_path) if source_chunks_path.is_file() else []
    )
    kept_chunks = [
        item for item in source_chunks if item["branch_id"] in kept_branch_ids
    ]
    for chunk in kept_chunks:
        chunk_id = chunk["chunk_id"]
        copy(
            source / "trajectory_chunks" / f"{chunk_id}.npz",
            output / "trajectory_chunks" / f"{chunk_id}.npz",
        )
        copy(
            source / "trajectory_conditions" / f"{chunk_id}.pt",
            output / "trajectory_conditions" / f"{chunk_id}.pt",
        )

    pairs = rebuild_pairs(states, kept_branches, max_pairs)
    write_jsonl(output / "failure_states.jsonl", states)
    write_jsonl(output / "branches.jsonl", kept_branches)
    write_jsonl(output / "pairs.jsonl", pairs)
    write_jsonl(output / "trajectory_chunks.jsonl", kept_chunks)
    filtered_audit = {
        **audit,
        "data_dir": output.as_posix(),
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "records": [
            item for item in records
            if item["fixed_branch_id"] in kept_branch_ids
        ],
        "planned_branchable_state_ids": sorted(kept_state_ids),
        "planned_positive_branch_ids": sorted(
            kept_branch_ids & stable_positive_ids
        ),
    }
    filtered_audit["positive_records"] = sum(
        item["audited_outcome"] == "positive"
        for item in filtered_audit["records"]
    )
    filtered_audit["negative_records"] = sum(
        item["audited_outcome"] == "negative"
        for item in filtered_audit["records"]
    )
    filtered_audit["coverage_complete"] = (
        not filtered_audit.get("missing_oracle_state_ids")
        and {
            item["failure_state_id"]
            for item in filtered_audit["records"]
            if item["audited_outcome"] == "positive"
        }
        == kept_state_ids
        and {
            item["failure_state_id"]
            for item in filtered_audit["records"]
            if item["audited_outcome"] == "negative"
        }
        == kept_state_ids
    )
    filtered_audit["passed"] = (
        bool(filtered_audit["records"])
        and filtered_audit["coverage_complete"]
        and all(
            item["robot_max_abs"] <= audit["state_tolerance"]
            and item["scene_max_abs"] <= audit["state_tolerance"]
            and item["rgb_static_mean_abs"] <= audit["rgb_tolerance"]
            and item["fixed_branch_same_outcome"]
            and item["fixed_branch_same_length"]
            and item.get("fixed_action_replay") is True
            and item["restore_contract"]
            == "bullet_reset_bullet_v2_gripper_v3"
            for item in filtered_audit["records"]
        )
    )
    (output / "persistent_replay_audit.json").write_text(
        json.dumps(filtered_audit, indent=2, sort_keys=True)
    )
    for name in ("sequence_catalog.json", "sequence_catalogs.json"):
        if (source / name).is_file():
            copy(source / name, output / name)

    branchable = {item["failure_state_id"] for item in pairs}
    progress = {
        "status": "collecting",
        "failure_states": len(states),
        "branchable_failure_states": len(branchable),
        "branches": len(kept_branches),
        "preference_pairs": len(pairs),
        "trajectory_chunks": len(kept_chunks),
    }
    (output / "collection_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True)
    )
    manifest = {
        "format": "robodual_failure_recovery_stable_filter_v1",
        "source": source.as_posix(),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "rules": {
            "state_fidelity_required": True,
            "positive_fixed_action_replay_required": True,
            "negative_fixed_action_replay_required": True,
            "full_robot_state_fidelity_required": True,
            "oracle_source": "persisted_subtask_start",
            "restore_contract": "bullet_reset_bullet_v2_gripper_v3",
            "max_pairs_per_state": max_pairs,
        },
        "kept_state_ids": sorted(kept_state_ids),
        "kept_positive_branch_ids": sorted(
            kept_branch_ids & stable_positive_ids
        ),
        "kept_negative_branch_ids": sorted(
            kept_branch_ids & stable_negative_ids
        ),
        "result": {
            **progress,
            "branchable_by_split": dict(Counter(
                item["split"]
                for item in states
                if item["failure_state_id"] in branchable
            )),
            "trainable_trajectory_chunks": sum(
                int(item["steps"]) >= 8 for item in kept_chunks
            ),
        },
    }
    (output / "stable_filter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max_pairs_per_state", type=int, default=16)
    args = parser.parse_args()
    if args.max_pairs_per_state <= 0:
        parser.error("--max_pairs_per_state must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    filter_dataset(args.source, args.output, args.max_pairs_per_state)
