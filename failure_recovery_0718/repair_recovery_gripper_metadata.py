#!/usr/bin/env python3
"""Align persisted CALVIN gripper controller metadata with each state snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def repair(root: Path) -> dict:
    root = root.expanduser().resolve()
    changes = []
    for state in read_jsonl(root / "failure_states.jsonl"):
        state_id = state["failure_state_id"]
        with np.load(
            root / "states" / f"{state_id}.npz", allow_pickle=False
        ) as payload:
            expected = int(
                np.asarray(payload["robot_obs"], dtype=np.float32)[-1].item()
            )
        simulator_path = root / "states" / f"{state_id}_simulator.pt"
        simulator = torch.load(
            simulator_path, map_location="cpu", weights_only=False
        )
        previous = int(
            np.asarray(simulator["robot"]["gripper_action"]).item()
        )
        if previous == expected:
            continue
        simulator["robot"]["gripper_action"] = expected
        temporary = simulator_path.with_suffix(".pt.tmp")
        torch.save(simulator, temporary)
        temporary.replace(simulator_path)
        changes.append({
            "failure_state_id": state_id,
            "previous_gripper_action": previous,
            "expected_gripper_action": expected,
        })
    result = {
        "format": "robodual_recovery_gripper_metadata_repair_v1",
        "data_dir": root.as_posix(),
        "states": len(read_jsonl(root / "failure_states.jsonl")),
        "changed_states": len(changes),
        "changes": changes,
        "status": "persistent_replay_audit_required",
    }
    (root / "gripper_metadata_repair.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    for stale_name in (
        "persistent_replay_audit.json",
        "dataset_assessment.json",
        "dataset_assessment.md",
    ):
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
