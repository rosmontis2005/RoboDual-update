#!/usr/bin/env python3
"""Summarize completed 2x2 cross runs and compute paired factorial contrasts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


CELLS = ("S8_P8", "S8_P12", "S12_P8", "S12_P12")
RAW_METRICS = (
    "success",
    "performance_score",
    "steps",
    "pink_lift_max",
    "pink_final_displacement",
    "min_tcp_pink_distance",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def difference(left: dict[str, float], right: dict[str, float], metric: str) -> float:
    return float(left[metric]) - float(right[metric])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = read_jsonl(run_dir / "cell_summaries.jsonl")
    ep_len = int(manifest["ep_len"])
    expected_replicates = len(manifest["trial_seeds"])
    if len(rows) != 4 * expected_replicates:
        raise AssertionError(f"Expected {4 * expected_replicates} cells, found {len(rows)}")

    by_replicate: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        cell = str(row["cell"])
        if cell not in CELLS:
            raise AssertionError(f"Unknown cell {cell}")
        replicate = int(row["replicate"])
        row["success"] = float(bool(row["success"]))
        row["performance_score"] = (
            (ep_len + 1 - int(row["steps"])) / ep_len if row["success"] else 0.0
        )
        if cell in by_replicate[replicate]:
            raise AssertionError(f"Duplicate {cell} in replicate {replicate}")
        by_replicate[replicate][cell] = row
    for replicate, cells in by_replicate.items():
        if set(cells) != set(CELLS):
            raise AssertionError(f"Incomplete replicate {replicate}: {sorted(cells)}")
    expected_ids = set(range(expected_replicates))
    if set(by_replicate) != expected_ids:
        raise AssertionError(
            f"Replicate IDs differ from manifest: expected {sorted(expected_ids)}, "
            f"found {sorted(by_replicate)}"
        )
    for replicate, cells in by_replicate.items():
        expected_seed = int(manifest["trial_seeds"][replicate])
        observed_seeds = {int(row["trial_seed"]) for row in cells.values()}
        if observed_seeds != {expected_seed}:
            raise AssertionError(
                f"Replicate {replicate} seed mismatch: expected {expected_seed}, "
                f"observed {sorted(observed_seeds)}"
            )

    cell_statistics: dict[str, Any] = {}
    cell_csv = []
    for cell in CELLS:
        cell_rows = [by_replicate[index][cell] for index in sorted(by_replicate)]
        cell_statistics[cell] = {
            metric: mean_sd([float(row[metric]) for row in cell_rows]) for metric in RAW_METRICS
        }
        cell_csv.append(
            {
                "cell": cell,
                "n": len(cell_rows),
                "successes": int(sum(row["success"] for row in cell_rows)),
                **{
                    f"{metric}_mean": cell_statistics[cell][metric]["mean"]
                    for metric in RAW_METRICS
                },
                **{
                    f"{metric}_sd": cell_statistics[cell][metric]["sd"]
                    for metric in RAW_METRICS
                },
            }
        )

    contrast_definitions = {
        "policy_effect_at_S8__P12_minus_P8": ("S8_P12", "S8_P8"),
        "policy_effect_at_S12__P12_minus_P8": ("S12_P12", "S12_P8"),
        "state_effect_under_P8__S12_minus_S8": ("S12_P8", "S8_P8"),
        "state_effect_under_P12__S12_minus_S8": ("S12_P12", "S8_P12"),
    }
    contrasts: dict[str, Any] = {}
    contrast_csv = []
    for name, (left_cell, right_cell) in contrast_definitions.items():
        contrasts[name] = {}
        for metric in RAW_METRICS:
            values = [
                difference(cells[left_cell], cells[right_cell], metric)
                for _, cells in sorted(by_replicate.items())
            ]
            contrasts[name][metric] = mean_sd(values)
            contrast_csv.append(
                {
                    "contrast": name,
                    "metric": metric,
                    "mean": contrasts[name][metric]["mean"],
                    "sd": contrasts[name][metric]["sd"],
                    "paired_values": json.dumps(values),
                }
            )

    averaged_main_definitions = {
        "averaged_policy_main_effect__P12_minus_P8": (
            ("S8_P12", "S8_P8"),
            ("S12_P12", "S12_P8"),
        ),
        "averaged_state_main_effect__S12_minus_S8": (
            ("S12_P8", "S8_P8"),
            ("S12_P12", "S8_P12"),
        ),
    }
    for name, component_pairs in averaged_main_definitions.items():
        contrasts[name] = {}
        for metric in RAW_METRICS:
            values = []
            for _, cells in sorted(by_replicate.items()):
                component_differences = [
                    difference(cells[left], cells[right], metric)
                    for left, right in component_pairs
                ]
                values.append(statistics.fmean(component_differences))
            contrasts[name][metric] = mean_sd(values)
            contrast_csv.append(
                {
                    "contrast": name,
                    "metric": metric,
                    "mean": contrasts[name][metric]["mean"],
                    "sd": contrasts[name][metric]["sd"],
                    "paired_values": json.dumps(values),
                }
            )

    interaction_name = "interaction__policy_effect_S12_minus_policy_effect_S8"
    contrasts[interaction_name] = {}
    for metric in RAW_METRICS:
        values = []
        for _, cells in sorted(by_replicate.items()):
            effect_s12 = difference(cells["S12_P12"], cells["S12_P8"], metric)
            effect_s8 = difference(cells["S8_P12"], cells["S8_P8"], metric)
            values.append(effect_s12 - effect_s8)
        contrasts[interaction_name][metric] = mean_sd(values)
        contrast_csv.append(
            {
                "contrast": interaction_name,
                "metric": metric,
                "mean": contrasts[interaction_name][metric]["mean"],
                "sd": contrasts[interaction_name][metric]["sd"],
                "paired_values": json.dumps(values),
            }
        )

    noise_audit = []
    for replicate, cells in sorted(by_replicate.items()):
        hash_lists = [cells[cell]["initial_noise_sha256_by_step"] for cell in CELLS]
        common_steps = min(map(len, hash_lists))
        mismatches = [
            step for step in range(common_steps) if len({hashes[step] for hashes in hash_lists}) != 1
        ]
        noise_audit.append(
            {
                "replicate": replicate,
                "common_steps": common_steps,
                "mismatch_count": len(mismatches),
                "first_mismatches": mismatches[:10],
                "passed": not mismatches,
            }
        )

    restore_files = sorted((run_dir / "restore_audits").glob("*.json"))
    expected_restore_names = {
        f"rep_{replicate:02d}_{cell}.json"
        for replicate in range(expected_replicates)
        for cell in CELLS
    }
    observed_restore_names = {path.name for path in restore_files}
    if observed_restore_names != expected_restore_names:
        raise AssertionError(
            "Restore audit file set mismatch: "
            f"missing={sorted(expected_restore_names - observed_restore_names)}, "
            f"extra={sorted(observed_restore_names - expected_restore_names)}"
        )
    restore_failures = []
    for path in restore_files:
        record = json.loads(path.read_text())
        if not record["audit"]["passed"]:
            restore_failures.append(path.name)

    result = {
        "run_dir": str(run_dir),
        "replicates": expected_replicates,
        "interpretation": {
            "success_and_performance_score": "higher is better",
            "steps_and_min_tcp_pink_distance": "lower is better",
            "pink_lift_max": "higher is better",
            "interaction": "difference-in-differences: policy effect at S12 minus policy effect at S8",
        },
        "cell_statistics": cell_statistics,
        "paired_contrasts": contrasts,
        "common_random_number_audit": noise_audit,
        "restore_audit": {
            "files": len(restore_files),
            "expected": 4 * expected_replicates,
            "failures": restore_failures,
            "passed": len(restore_files) == 4 * expected_replicates and not restore_failures,
        },
        "caution": (
            "With five replicates, report paired values and variability rather than treating asymptotic p-values as decisive. "
            "The two initial states are fixed exemplars from one upstream run each, not samples from an upstream-state population."
        ),
    }
    (run_dir / "cross_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(run_dir / "cross_cell_statistics.csv", cell_csv)
    write_csv(run_dir / "cross_paired_contrasts.csv", contrast_csv)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
