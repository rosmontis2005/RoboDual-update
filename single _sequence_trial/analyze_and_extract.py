#!/usr/bin/env python3
"""Audit CALVIN sequence order and extract the selected baseline evidence.

This script only reads existing evaluation artifacts.  It deliberately avoids
loading the models or CALVIN environment, so it can be rerun on a CPU login
node with the repository's default Python.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

RUNS = {
    "fixed_mod8_original": ROOT / "evaluation_results/0413exp",
    "age7_control": ROOT / "evaluation_results/0424exp",
    "uniform_age12_run1": ROOT / "evaluation_results/0425_1exp_command2",
    "uniform_age12_run2": ROOT / "evaluation_results/exp0523-0428-nostradegy-maxage12",
}

SELECTED_SEQUENCE = 60
CANDIDATES = (36, 60)

CANONICAL_SELECTED = {
    "sequence_index_zero_based": 60,
    "sequence_number_human": 61,
    "initial_state": {
        "led": 0,
        "lightbulb": 0,
        "slider": "right",
        "drawer": "closed",
        "red_block": "slider_left",
        "blue_block": "table",
        "pink_block": "slider_right",
        "grasped": 0,
    },
    "tasks": [
        "open_drawer",
        "push_blue_block_left",
        "move_slider_left",
        "turn_on_lightbulb",
        "lift_pink_block_slider",
    ],
}


def iter_steps(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "step":
                yield row


def load_profile(path: Path) -> tuple[dict[int, dict[int, str]], Counter[int]]:
    tasks: dict[int, dict[int, str]] = defaultdict(dict)
    successes: Counter[int] = Counter()
    for row in iter_steps(path):
        seq = int(row["sequence"])
        subtask = int(row["subtask_i"])
        tasks[seq].setdefault(subtask, row["task"])
        if row.get("terminal_step") and row.get("step_success"):
            successes[seq] += 1
    return tasks, successes


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def summarize_selected(path: Path, sequence: int) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_steps(path):
        if int(row["sequence"]) == sequence:
            groups[int(row["subtask_i"])].append(row)

    output = []
    for subtask, rows in sorted(groups.items()):
        profiles = [row["profile"] for row in rows]
        agg = [p["aggregation_delta_ee6"] for p in profiles if p.get("aggregation_delta_ee6") is not None]
        jerk = [p["jerk_l2_ee6"] for p in profiles if p.get("jerk_l2_ee6") is not None]
        output.append(
            {
                "subtask_index_zero_based": subtask,
                "task": rows[0]["task"],
                "steps": len(rows),
                "success": bool(rows[-1].get("step_success")),
                "slow_calls": sum(bool(p.get("slow_system")) for p in profiles),
                "expired_reference_steps": sum((p.get("num_cond_actions") or 0) == 0 for p in profiles),
                "aggregation_delta_ee6_p95": percentile(agg, 0.95),
                "jerk_l2_ee6_p95": percentile(jerk, 0.95),
                "gripper_flip_count_max": max((p.get("gripper_flip_count") or 0 for p in profiles), default=0),
            }
        )
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_order(reference_tasks: dict[int, dict[int, str]]) -> dict[str, Any]:
    profile_paths = sorted((ROOT / "evaluation_results").glob("**/specialist_profile_rank0.jsonl"))
    audited = []
    mismatches = []
    full_100_profiles = 0

    for path in profile_paths:
        tasks, _ = load_profile(path)
        sequence_ids = sorted(tasks)
        is_100_sequence = sequence_ids == list(range(100))
        if not is_100_sequence:
            continue
        full_100_profiles += 1
        compared = 0
        local_mismatches = []
        for seq, subtasks in tasks.items():
            for subtask, task in subtasks.items():
                expected = reference_tasks.get(seq, {}).get(subtask)
                if expected is None:
                    continue
                compared += 1
                if task != expected:
                    local_mismatches.append(
                        {"sequence": seq, "subtask": subtask, "expected": expected, "actual": task}
                    )
        rel = str(path.relative_to(ROOT))
        audited.append({"profile": rel, "compared_reached_tasks": compared, "mismatches": len(local_mismatches)})
        mismatches.extend({"profile": rel, **item} for item in local_mismatches)

    success_files = sorted((ROOT / "evaluation_results").glob("**/success_rate_rank0.txt"))
    result_100 = []
    for path in success_files:
        with path.open() as handle:
            lines = [line for line in handle if line.strip()]
        if len(lines) == 100:
            result_100.append(str(path.relative_to(ROOT)))

    return {
        "generator_contract": {
            "source": "/home/rosmontis/Projects/dualsys/calvin/calvin_models/calvin_agent/evaluation/multistep_sequences.py",
            "function": "get_sequences(num_sequences)",
            "determinism": "temp_seed(0), deterministic per-state seeds, then deterministic shuffle",
            "scope": "Calls with the same num_sequences=100 return the same ordered list; changing num_sequences can change the list/order.",
        },
        "100_line_result_files": len(result_100),
        "100_line_result_file_paths": result_100,
        "full_100_sequence_profiles_audited": full_100_profiles,
        "profile_audit": audited,
        "total_task_mismatches_on_shared_reached_positions": len(mismatches),
        "mismatch_details": mismatches,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    loaded = {}
    for name, directory in RUNS.items():
        profile = directory / "specialist_profile_rank0.jsonl"
        tasks, successes = load_profile(profile)
        loaded[name] = {"profile": profile, "tasks": tasks, "successes": successes}

    candidate_rows = []
    for sequence in CANDIDATES:
        row: dict[str, Any] = {"sequence_index_zero_based": sequence, "sequence_number_human": sequence + 1}
        for name in RUNS:
            row[name] = loaded[name]["successes"][sequence]
        row["tasks"] = " -> ".join(loaded["fixed_mod8_original"]["tasks"][sequence].values())
        candidate_rows.append(row)
    write_csv(OUT / "candidate_outcome_matrix.csv", candidate_rows)

    summary_rows = []
    for run_name in ("fixed_mod8_original", "uniform_age12_run1", "uniform_age12_run2"):
        for row in summarize_selected(loaded[run_name]["profile"], SELECTED_SEQUENCE):
            summary_rows.append({"run": run_name, **row})
    write_csv(OUT / "selected_sequence_profile_summary.csv", summary_rows)

    extracted_dir = OUT / "profiles"
    extracted_dir.mkdir(exist_ok=True)
    for run_name in ("fixed_mod8_original", "uniform_age12_run1", "uniform_age12_run2"):
        target = extracted_dir / f"sequence_060_{run_name}.jsonl"
        with target.open("w") as handle:
            for row in iter_steps(loaded[run_name]["profile"]):
                if int(row["sequence"]) == SELECTED_SEQUENCE:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    catalog_path = OUT / "canonical_100_sequences.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
        reference_tasks = {
            int(item["sequence_index_zero_based"]): dict(enumerate(item["tasks"])) for item in catalog
        }
    else:
        reference_tasks = loaded["fixed_mod8_original"]["tasks"]
    order_audit = audit_order(reference_tasks)
    with (OUT / "sequence_order_audit.json").open("w") as handle:
        json.dump(order_audit, handle, indent=2)
        handle.write("\n")

    selected = {
        **CANONICAL_SELECTED,
        "selection_status": "selected_baseline",
        "criterion": "5/5 under original fixed-mod-8 and 4/5 under both uniform age-12 artifacts",
        "outcomes": {
            name: {
                "successful_subtasks": loaded[name]["successes"][SELECTED_SEQUENCE],
                "profile_source": str(loaded[name]["profile"].relative_to(ROOT)),
                "profile_sha256": sha256(loaded[name]["profile"]),
            }
            for name in RUNS
        },
        "important_caveat": "Historical evaluations are not a paired multi-seed causal test; freeze per-sequence diffusion seeds for follow-up experiments.",
    }
    with (OUT / "baseline_sequence.json").open("w") as handle:
        json.dump(selected, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
