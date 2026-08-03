#!/usr/bin/env python3
"""Validate and compare paired base/candidate recovery replay results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


MATCHED_CONFIG_KEYS = (
    "format",
    "data_dir",
    "split",
    "states",
    "horizon",
    "seeds_per_state",
    "seed",
)


def load_complete(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"Replay is not complete: {path}")
    if len(payload.get("records", [])) != payload.get("rollouts"):
        raise ValueError(f"Replay record count mismatch: {path}")
    return payload


def record_map(payload: dict) -> dict[tuple[str, int], dict]:
    records = {}
    for item in payload["records"]:
        key = (item["failure_state_id"], int(item["seed"]))
        if key in records:
            raise ValueError(f"Duplicate replay key: {key}")
        records[key] = item
    return records


def exact_sign_test_two_sided(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare(base: dict, candidate: dict) -> dict:
    mismatches = {
        key: (base.get(key), candidate.get(key))
        for key in MATCHED_CONFIG_KEYS
        if base.get(key) != candidate.get(key)
    }
    if mismatches:
        raise ValueError(f"Replay configuration mismatch: {mismatches}")
    base_records = record_map(base)
    candidate_records = record_map(candidate)
    if base_records.keys() != candidate_records.keys():
        missing = sorted(base_records.keys() - candidate_records.keys())
        extra = sorted(candidate_records.keys() - base_records.keys())
        raise ValueError(f"Paired replay key mismatch: missing={missing}, extra={extra}")

    task_counts = defaultdict(lambda: {"rollouts": 0, "base": 0, "candidate": 0, "gains": 0, "losses": 0})
    gains = losses = unchanged_success = unchanged_failure = 0
    changed = []
    for key, base_item in base_records.items():
        candidate_item = candidate_records[key]
        if (base_item["task"], base_item["split"]) != (
            candidate_item["task"], candidate_item["split"]
        ):
            raise ValueError(f"Paired replay metadata mismatch: {key}")
        base_success = bool(base_item["success"])
        candidate_success = bool(candidate_item["success"])
        counts = task_counts[base_item["task"]]
        counts["rollouts"] += 1
        counts["base"] += int(base_success)
        counts["candidate"] += int(candidate_success)
        if not base_success and candidate_success:
            gains += 1
            counts["gains"] += 1
            changed.append({"failure_state_id": key[0], "seed": key[1], "task": base_item["task"], "change": "gain"})
        elif base_success and not candidate_success:
            losses += 1
            counts["losses"] += 1
            changed.append({"failure_state_id": key[0], "seed": key[1], "task": base_item["task"], "change": "loss"})
        elif base_success:
            unchanged_success += 1
        else:
            unchanged_failure += 1

    rollouts = len(base_records)
    base_successes = sum(bool(item["success"]) for item in base_records.values())
    candidate_successes = sum(bool(item["success"]) for item in candidate_records.values())
    return {
        "format": "robodual_failure_recovery_paired_comparison_v1",
        "rollouts": rollouts,
        "base_specialist_path": base["specialist_path"],
        "candidate_specialist_path": candidate["specialist_path"],
        "base_successes": base_successes,
        "candidate_successes": candidate_successes,
        "base_success_rate": base_successes / rollouts,
        "candidate_success_rate": candidate_successes / rollouts,
        "success_rate_delta": (candidate_successes - base_successes) / rollouts,
        "paired_gains": gains,
        "paired_losses": losses,
        "unchanged_success": unchanged_success,
        "unchanged_failure": unchanged_failure,
        "exact_sign_test_two_sided_p": exact_sign_test_two_sided(gains, losses),
        "tasks": dict(sorted(task_counts.items())),
        "changed_records": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(load_complete(args.base), load_complete(args.candidate))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
