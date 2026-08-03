#!/usr/bin/env python3
"""Normalize redundant recovery branch/chunk metadata from authoritative states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def repair(root: Path) -> dict:
    root = root.expanduser().resolve()
    states = {
        item["failure_state_id"]: item
        for item in read_jsonl(root / "failure_states.jsonl")
    }
    branches = read_jsonl(root / "branches.jsonl")
    branch_changes = 0
    for branch in branches:
        expected = states[branch["failure_state_id"]]["task"]
        if branch.get("task") != expected:
            branch["task"] = expected
            branch_changes += 1
    branch_by_id = {item["branch_id"]: item for item in branches}
    chunks = read_jsonl(root / "trajectory_chunks.jsonl")
    chunk_changes = 0
    for chunk in chunks:
        branch = branch_by_id[chunk["branch_id"]]
        expected = {
            "failure_state_id": branch["failure_state_id"],
            "split": branch["split"],
            "task": states[branch["failure_state_id"]]["task"],
            "strategy": branch["strategy"],
        }
        if any(chunk.get(key) != value for key, value in expected.items()):
            chunk.update(expected)
            chunk_changes += 1
    write_jsonl_atomic(root / "branches.jsonl", branches)
    write_jsonl_atomic(root / "trajectory_chunks.jsonl", chunks)
    result = {
        "format": "robodual_recovery_manifest_metadata_repair_v1",
        "data_dir": root.as_posix(),
        "branch_changes": branch_changes,
        "chunk_changes": chunk_changes,
    }
    (root / "manifest_metadata_repair.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    for stale_name in ("dataset_assessment.json", "dataset_assessment.md"):
        stale_path = root / stale_name
        if stale_path.is_file():
            stale_path.unlink()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    repair(parse_args().data_dir)
