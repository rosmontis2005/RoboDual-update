#!/usr/bin/env python3
"""Create a non-destructive recovery dataset with only auditable v2 provenance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def copy_payload(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def rebuild_pairs(states: list[dict], branches: list[dict], max_pairs: int) -> list[dict]:
    groups = defaultdict(list)
    for branch in branches:
        groups[branch["failure_state_id"]].append(branch)
    pairs = []
    for state in states:
        state_id = state["failure_state_id"]
        positives = [item for item in groups[state_id] if item["success"]]
        negatives = [item for item in groups[state_id] if not item["success"]]
        count = 0
        for positive in positives:
            for negative in negatives:
                if count >= max_pairs:
                    break
                pairs.append({
                    "pair_id": f"pair_{len(pairs):06d}",
                    "failure_state_id": state_id,
                    "split": state["split"],
                    "positive_branch_id": positive["branch_id"],
                    "negative_branch_id": negative["branch_id"],
                })
                count += 1
            if count >= max_pairs:
                break
    return pairs


def repair(source: Path, output: Path, max_pairs: int) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Repair output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    states_dir = output / "states"
    branches_dir = output / "branches"
    conditions_dir = output / "conditions"
    for directory in (states_dir, branches_dir, conditions_dir):
        directory.mkdir()

    source_states = read_jsonl(source / "failure_states.jsonl")
    source_branches = read_jsonl(source / "branches.jsonl")
    kept_states = []
    excluded_states = []
    for state in source_states:
        state_id = state["failure_state_id"]
        if not (source / "states" / f"{state_id}_oracle_start.pt").is_file():
            excluded_states.append({
                "failure_state_id": state_id,
                "reason": "missing_persisted_subtask_oracle_start",
            })
            continue
        kept_states.append(state)
        for suffix in (
            ".npz",
            ".bullet",
            "_model.pt",
            "_simulator.pt",
            "_oracle_start.pt",
        ):
            copy_payload(
                source / "states" / f"{state_id}{suffix}",
                states_dir / f"{state_id}{suffix}",
            )

    kept_state_ids = {item["failure_state_id"] for item in kept_states}
    kept_branches = []
    excluded_branches = []
    for branch in source_branches:
        branch_id = branch["branch_id"]
        if branch["failure_state_id"] not in kept_state_ids:
            excluded_branches.append({
                "branch_id": branch_id,
                "reason": "state_quarantined",
            })
            continue
        if (
            branch.get("strategy") == "demo_guided_persisted"
            and (
                branch.get("oracle_source") != "persisted_subtask_start"
                or branch.get("restore_contract") != "bullet_reset_bullet_v2"
            )
        ):
            excluded_branches.append({
                "branch_id": branch_id,
                "reason": "legacy_persisted_restore_or_oracle_contract",
            })
            continue
        kept_branches.append(branch)
        copy_payload(
            source / "branches" / f"{branch_id}.npz",
            branches_dir / f"{branch_id}.npz",
        )
        copy_payload(
            source / "conditions" / f"{branch_id}.pt",
            conditions_dir / f"{branch_id}.pt",
        )

    pairs = rebuild_pairs(kept_states, kept_branches, max_pairs)
    write_jsonl(output / "failure_states.jsonl", kept_states)
    write_jsonl(output / "branches.jsonl", kept_branches)
    write_jsonl(output / "pairs.jsonl", pairs)
    for name in ("sequence_catalog.json", "sequence_catalogs.json"):
        if (source / name).is_file():
            copy_payload(source / name, output / name)

    branchable = {item["failure_state_id"] for item in pairs}
    progress = {
        "status": "collecting",
        "failure_states": len(kept_states),
        "branchable_failure_states": len(branchable),
        "branches": len(kept_branches),
        "preference_pairs": len(pairs),
    }
    (output / "collection_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True)
    )
    source_manifest_hash = hashlib.sha256(
        (source / "failure_states.jsonl").read_bytes()
        + (source / "branches.jsonl").read_bytes()
        + (source / "pairs.jsonl").read_bytes()
    ).hexdigest()
    repair_manifest = {
        "format": "robodual_failure_recovery_repair_v1",
        "source": source.as_posix(),
        "source_manifest_sha256": source_manifest_hash,
        "rules": {
            "require_persisted_subtask_oracle_start": True,
            "require_v2_provenance_for_demo_guided_persisted": True,
            "max_pairs_per_state": max_pairs,
        },
        "excluded_states": excluded_states,
        "excluded_branches": excluded_branches,
        "result": {
            **progress,
            "branchable_by_split": dict(Counter(
                item["split"]
                for item in kept_states
                if item["failure_state_id"] in branchable
            )),
        },
    }
    (output / "repair_manifest.json").write_text(
        json.dumps(repair_manifest, indent=2, sort_keys=True)
    )
    print(json.dumps(repair_manifest, indent=2, sort_keys=True))
    return repair_manifest


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
    parsed = parse_args()
    repair(parsed.source, parsed.output, parsed.max_pairs_per_state)
