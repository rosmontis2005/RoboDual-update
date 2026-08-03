#!/usr/bin/env python3
"""Validate and compare a canonical 100-sequence CALVIN benchmark."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path


LINE_RE = re.compile(
    r"^(?P<index>\d+)/100(?: \[sequence=(?P<sequence>\d+)\])?: "
    r"(?P<rates>.+?)\s*\|\s*$"
)


def load_summary(directory: Path) -> dict:
    data = json.loads((directory / "result_rank0.json").read_text())
    if set(data) != {"null"}:
        raise RuntimeError(f"unexpected result keys in {directory}: {sorted(data)}")
    return data["null"]


def load_completions(directory: Path) -> dict[int, int]:
    completions: dict[int, int] = {}
    previous_counts = [0] * 5
    lines = (directory / "success_rate_rank0.txt").read_text().splitlines()
    for line in lines:
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        index = int(match.group("index"))
        sequence = int(match.group("sequence") or index)
        rates = [float(value.strip()) for value in match.group("rates").split("|")]
        if len(rates) != 5:
            raise RuntimeError(f"expected five rates in line: {line}")
        denominator = index + 1
        counts = [round(rate * denominator) for rate in rates]
        deltas = [count - previous for count, previous in zip(counts, previous_counts)]
        if any(delta not in (0, 1) for delta in deltas):
            raise RuntimeError(f"invalid cumulative delta at sequence {sequence}: {deltas}")
        if deltas != sorted(deltas, reverse=True):
            raise RuntimeError(f"non-prefix completion at sequence {sequence}: {deltas}")
        if sequence in completions:
            raise RuntimeError(f"duplicate sequence ID {sequence} in {directory}")
        completions[sequence] = sum(deltas)
        previous_counts = counts

    expected = set(range(100))
    if set(completions) != expected:
        missing = sorted(expected - set(completions))
        extra = sorted(set(completions) - expected)
        raise RuntimeError(f"non-canonical sequence set in {directory}: missing={missing}, extra={extra}")
    return completions


def main(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = load_summary(baseline_dir)
    candidate_summary = load_summary(candidate_dir)
    baseline_completions = load_completions(baseline_dir)
    candidate_completions = load_completions(candidate_dir)

    baseline_chain = [float(baseline_summary["chain_sr"][str(i)]) for i in range(1, 6)]
    candidate_chain = [float(candidate_summary["chain_sr"][str(i)]) for i in range(1, 6)]
    paired_values = {
        sequence: candidate_completions[sequence] - baseline_completions[sequence]
        for sequence in range(100)
    }
    paired = {
        "improved": sum(value > 0 for value in paired_values.values()),
        "equal": sum(value == 0 for value in paired_values.values()),
        "worse": sum(value < 0 for value in paired_values.values()),
        "net_completed_subtasks": sum(paired_values.values()),
        "values": paired_values,
    }
    paired_samples = list(paired_values.values())
    rng = random.Random(42)
    bootstrap_means = sorted(
        sum(rng.choice(paired_samples) for _ in paired_samples) / len(paired_samples)
        for _ in range(20_000)
    )
    paired["mean"] = statistics.mean(paired_samples)
    paired["sample_stddev"] = statistics.stdev(paired_samples)
    paired["bootstrap_95_ci"] = [bootstrap_means[500], bootstrap_means[19_499]]
    avg_delta = float(candidate_summary["avg_seq_len"] - baseline_summary["avg_seq_len"])
    chain_deltas = [candidate - baseline for baseline, candidate in zip(baseline_chain, candidate_chain)]
    recovery_gate = {
        "average_length_delta_at_least_minus_0_10": avg_delta >= -0.10,
        "no_chain_decline_over_6_25pp": min(chain_deltas) >= -0.0625,
    }
    result = {
        "status": "complete",
        "sequence_count": 100,
        "canonical_sequence_ids": True,
        "candidate": {
            "name": args.candidate_name,
            "directory": candidate_dir.as_posix(),
            "avg_seq_len": candidate_summary["avg_seq_len"],
            "chain_sr": candidate_chain,
        },
        "baseline": {
            "directory": baseline_dir.as_posix(),
            "avg_seq_len": baseline_summary["avg_seq_len"],
            "chain_sr": baseline_chain,
        },
        "delta": {
            "avg_seq_len": avg_delta,
            "chain_sr_percentage_points": [100 * value for value in chain_deltas],
        },
        "paired_sequences": paired,
        "baseline_recovery_gate": recovery_gate,
        "baseline_recovery_pass": all(recovery_gate.values()),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        f"# {args.candidate_name} 100-sequence benchmark",
        "",
        "- Integrity: `PASS` (100 canonical sequence IDs, 0-99).",
        f"- Average completed length: `{candidate_summary['avg_seq_len']:.2f}` vs "
        f"baseline `{baseline_summary['avg_seq_len']:.2f}` (`{avg_delta:+.2f}`).",
        "- Chain SR candidate: `" + "/".join(f"{100 * value:.0f}" for value in candidate_chain) + "%`.",
        "- Chain SR baseline: `" + "/".join(f"{100 * value:.0f}" for value in baseline_chain) + "%`.",
        "- Chain delta: `" + "/".join(f"{100 * value:+.0f}" for value in chain_deltas) + " pp`.",
        f"- Paired improved/equal/worse: `{paired['improved']}/{paired['equal']}/{paired['worse']}`; "
        f"net completed subtasks `{paired['net_completed_subtasks']:+d}`.",
        f"- Paired mean bootstrap 95% CI: `[{paired['bootstrap_95_ci'][0]:+.2f}, "
        f"{paired['bootstrap_95_ci'][1]:+.2f}]` completed subtasks per sequence.",
        f"- Baseline recovery gate: `{'PASS' if result['baseline_recovery_pass'] else 'FAIL'}`.",
    ]
    (output_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_name", default="V11 gated age12")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
