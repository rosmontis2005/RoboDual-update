#!/usr/bin/env python3
"""Repair normalized committed-action targets from next-frame history snapshots."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from collect_transition_rollouts import (
    candidate_requirement_keys,
    sample_deficits,
    sample_requirements,
    select_samples,
)
from finalize_interrupted_collection import load_conditions, reconstruct_candidates


ACTION_SOURCE = "next_frame_hist_action_before[-1]"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def recover_actions(frame_files: list[Path]) -> tuple[np.ndarray, dict]:
    if len(frame_files) < 2:
        raise ValueError("A trajectory needs at least two frames to recover one action")
    histories = []
    corrupted = []
    for expected_step, path in enumerate(frame_files):
        if path.name != f"step_{expected_step:04d}.npz":
            raise ValueError(f"Non-contiguous frame sequence at {path}")
        with np.load(path, allow_pickle=False) as frame:
            history = np.asarray(frame["hist_action_before"], dtype=np.float32)
            old_action = np.asarray(frame["rel_actions"], dtype=np.float32)
        if history.shape != (4, 7) or old_action.shape != (7,):
            raise ValueError(f"Bad action/history shape in {path}: {old_action.shape}, {history.shape}")
        histories.append(history.copy())
        corrupted.append(old_action.copy())

    recovered = np.stack([history[-1] for history in histories[1:]]).astype(np.float32)
    # At t+1 the first three history slots must equal the last three slots at t.
    shift_errors = [
        float(np.max(np.abs(histories[index + 1][:-1] - histories[index][1:])))
        for index in range(len(histories) - 1)
    ]
    max_shift_error = max(shift_errors, default=0.0)
    if max_shift_error > 1e-6:
        raise ValueError(f"Committed history is not temporally continuous: max error {max_shift_error}")
    if not np.isfinite(recovered).all():
        raise ValueError("Recovered actions contain non-finite values")
    if np.max(np.abs(recovered)) > 1.0001:
        raise ValueError(f"Recovered normalized action exceeds [-1,1]: {np.max(np.abs(recovered))}")

    corrupted = np.stack(corrupted[:-1])
    return recovered, {
        "frames": len(frame_files),
        "recoverable_actions": len(recovered),
        "max_history_shift_error": max_shift_error,
        "recovered_ee6_l2_sum": float(np.linalg.norm(recovered[:, :6], axis=1).sum()),
        "corrupted_ee6_l2_sum": float(np.linalg.norm(corrupted[:, :6], axis=1).sum()),
    }


def fill_split_shortfalls(candidates, selected, target_by_split, seed: int):
    """Fill small category shortfalls without changing split totals or reusing a window."""
    selected_keys = {(item.trajectory_id, item.step) for item in selected}
    rng = random.Random(seed)
    added = []
    for split, target in target_by_split.items():
        current = sum(item.split == split for item in selected)
        pool = [
            item for item in candidates
            if item.split == split and (item.trajectory_id, item.step) not in selected_keys
        ]
        pool.sort(key=lambda item: (item.trajectory_id, item.step))
        needed = target - current
        if needed < 0 or len(pool) < needed:
            raise RuntimeError(
                f"Cannot fill repaired split {split}: current={current}, target={target}, pool={len(pool)}"
            )
        for item in rng.sample(pool, needed):
            selected.append(item)
            added.append(item)
            selected_keys.add((item.trajectory_id, item.step))
    rng.shuffle(selected)
    return selected, added


def repair(args: argparse.Namespace) -> dict:
    source = Path(args.source_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output == source:
        raise ValueError("Repair output must differ from the source dataset")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output} is not empty; use --overwrite for a fresh rebuild")
        shutil.rmtree(output)
    if not (source / "trajectories").is_dir() or not (source / "conditions").is_dir():
        raise FileNotFoundError(f"Incomplete source dataset: {source}")

    output.mkdir(parents=True, exist_ok=True)
    actions_dir = output / "committed_actions"
    actions_dir.mkdir()
    os.symlink((source / "trajectories").as_posix(), output / "trajectories", target_is_directory=True)
    os.symlink((source / "conditions").as_posix(), output / "conditions", target_is_directory=True)

    original_summary = json.loads((source / "collection_summary.json").read_text())
    original_trajectories = {
        record["trajectory_id"]: record for record in read_jsonl(source / "trajectories.jsonl")
    }
    all_candidates = []
    candidate_counts = Counter()
    trajectory_records = []
    totals = Counter()
    max_history_shift_error = 0.0

    for trajectory_id in sorted(original_trajectories):
        frame_files = sorted((source / "trajectories" / trajectory_id).glob("step_*.npz"))
        condition_files = sorted((source / "conditions" / trajectory_id).glob("condition_*.pt"))
        actions, stats = recover_actions(frame_files)
        np.save(actions_dir / f"{trajectory_id}.npy", actions, allow_pickle=False)
        conditions = load_conditions(condition_files, len(frame_files))
        candidates = reconstruct_candidates(trajectory_id, list(actions), conditions, args)
        all_candidates.extend(candidates)
        candidate_counts.update(
            key for candidate in candidates for key in candidate_requirement_keys(candidate)
        )
        record = dict(original_trajectories[trajectory_id])
        record.update({
            "steps": len(frame_files),
            "recoverable_actions": len(actions),
            "candidate_samples": len(candidates),
            "target_action_source": ACTION_SOURCE,
        })
        trajectory_records.append(record)
        for key, value in stats.items():
            totals[key] += value
        max_history_shift_error = max(max_history_shift_error, stats["max_history_shift_error"])

    requirements = sample_requirements(args.target_samples)
    deficits = sample_deficits(candidate_counts, requirements)
    selected, selection_stats = select_samples(all_candidates, args.target_samples, args.seed)
    category_shortfall = sum(deficits.values())
    if category_shortfall > args.max_category_shortfall:
        raise RuntimeError(
            f"Repaired category shortfall exceeds limit {args.max_category_shortfall}: {deficits}"
        )
    selected, redistributed = fill_split_shortfalls(
        all_candidates, selected, selection_stats["target_by_split"], args.seed + 1000
    )
    if len(redistributed) > args.max_category_shortfall:
        raise RuntimeError(
            f"Actual category redistribution {len(redistributed)} exceeds limit "
            f"{args.max_category_shortfall}"
        )
    if len(selected) != args.target_samples:
        raise RuntimeError(f"Repaired selection has {len(selected)} samples, expected {args.target_samples}")
    selection_stats["selected_total"] = len(selected)
    selection_stats["selected_by_category"] = dict(Counter(item.category for item in selected))
    selection_stats["selected_by_split"] = dict(Counter(item.split for item in selected))
    selection_stats["redistributed_samples"] = len(redistributed)
    selection_stats["redistributed_by_category"] = dict(Counter(item.category for item in redistributed))
    sample_records = []
    for sample_id, candidate in enumerate(selected):
        payload = asdict(candidate)
        payload.update({
            "sample_id": sample_id,
            "history_steps": args.history_steps,
            "action_chunk_size": args.action_chunk_size,
            "target_action_source": ACTION_SOURCE,
        })
        sample_records.append(payload)

    recovered_mean = totals["recovered_ee6_l2_sum"] / totals["recoverable_actions"]
    corrupted_mean = totals["corrupted_ee6_l2_sum"] / totals["recoverable_actions"]
    if recovered_mean < args.min_recovered_ee6_l2_mean:
        raise RuntimeError(
            f"Recovered action scale is still suspicious: ee6 L2 mean={recovered_mean:.6f}"
        )
    summary = dict(original_summary)
    summary.update({
        "format": "robodual_transition_lora_repaired_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output.as_posix(),
        "repaired_from": source.as_posix(),
        "target_action_source": ACTION_SOURCE,
        "unrecoverable_terminal_actions_per_trajectory": 1,
        "selected_by_split": dict(Counter(item["split"] for item in sample_records)),
        "selection": selection_stats,
        "candidate_available": {
            f"{split}:{category}": candidate_counts[(split, category)]
            for split, category in requirements
        },
        "category_deficits": deficits,
        "category_redistribution_limit": args.max_category_shortfall,
        "repair_statistics": {
            "trajectories": len(trajectory_records),
            "frames": int(totals["frames"]),
            "recoverable_actions": int(totals["recoverable_actions"]),
            "max_history_shift_error": max_history_shift_error,
            "recovered_ee6_l2_mean": recovered_mean,
            "corrupted_ee6_l2_mean": corrupted_mean,
        },
        "integrity": {
            **original_summary.get("integrity", {}),
            "targets": "recovered from next-frame committed history; terminal unrecoverable action excluded",
            "storage": "images/conditions are symlinked to the original collection; repair and trainer only read them",
        },
    })
    write_jsonl(output / "samples.jsonl", sample_records)
    write_jsonl(output / "trajectories.jsonl", trajectory_records)
    (output / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", default="LoRA_transition_0711/collected_transition_v1")
    parser.add_argument("--output_dir", default="LoRA_transition_0711/collected_transition_v1_repaired")
    parser.add_argument("--target_samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_steps", type=int, default=4, choices=[4])
    parser.add_argument("--action_chunk_size", type=int, default=8, choices=[8])
    parser.add_argument("--empty_ref_after_age", type=int, default=8)
    parser.add_argument("--high_conflict_prev_threshold", type=float, default=0.18)
    parser.add_argument("--high_conflict_old_new_threshold", type=float, default=0.18)
    parser.add_argument("--high_conflict_jerk_threshold", type=float, default=0.24)
    parser.add_argument("--min_recovered_ee6_l2_mean", type=float, default=0.05)
    parser.add_argument("--max_category_shortfall", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    repair(parse_args())
