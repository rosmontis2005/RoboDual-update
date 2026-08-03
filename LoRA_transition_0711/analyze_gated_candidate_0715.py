#!/usr/bin/env python3
"""Compare one gated fixed-sequence run with the contemporary baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_ablation_check_0713 import load_mode, paired_delta


EXPECTED_SEQUENCE_IDS = (3, 11, 20, 21, 28, 35, 36, 53, 59, 65, 75, 83, 86, 89, 91, 95)


def relative_change(candidate: dict, baseline: dict, key: str) -> float:
    base_value = baseline["metrics"][key]["mean"]
    candidate_value = candidate["metrics"][key]["mean"]
    if base_value in (None, 0) or candidate_value is None:
        raise RuntimeError(f"metric {key} cannot be compared")
    return candidate_value / base_value - 1.0


def audit_gate(profile_path: Path) -> dict:
    base_steps = 0
    transition_steps = 0
    state_mismatches = 0
    expired_base_steps = 0
    with profile_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "step":
                continue
            profile = row["profile"]
            active = bool(profile.get("transition_gate_active"))
            expired = bool(profile.get("ref_action_expired"))
            transition_steps += int(active)
            base_steps += int(not active)
            state_mismatches += int(active and not expired)
            expired_base_steps += int(expired and not active)
    if base_steps + transition_steps == 0:
        raise RuntimeError("candidate profile contains no step rows")
    return {
        "base": base_steps,
        "transition": transition_steps,
        "expired_base": expired_base_steps,
        "state_mismatches": state_mismatches,
    }


def main(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_mode(baseline_dir.parent, baseline_dir.name)
    candidate = load_mode(candidate_dir.parent, candidate_dir.name)
    expected = set(EXPECTED_SEQUENCE_IDS)
    for name, item in (("baseline", baseline), ("candidate", candidate)):
        actual = set(item["completions"])
        if actual != expected:
            raise RuntimeError(f"{name} sequence IDs differ: expected={sorted(expected)}, actual={sorted(actual)}")

    paired = paired_delta(baseline["completions"], candidate["completions"])
    metric_keys = (
        "action_norm_ee6",
        "expired_action_norm_ee6",
        "jerk_l2_ee6",
        "aggregation_delta_ee6",
        "dp_ref_l2_ee6",
    )
    metric_changes = {
        key: relative_change(candidate, baseline, key)
        for key in metric_keys
    }
    baseline_chain = [float(baseline["chain_sr"][str(index)]) for index in range(1, 6)]
    candidate_chain = [float(candidate["chain_sr"][str(index)]) for index in range(1, 6)]
    chain_deltas = [right - left for left, right in zip(baseline_chain, candidate_chain)]
    avg_delta = float(candidate["avg_seq_len"] - baseline["avg_seq_len"])
    gate_steps = audit_gate(candidate_dir / "specialist_profile_rank0.jsonl")

    gates = {
        "average_length": avg_delta >= -0.10,
        "chain_decline": min(chain_deltas) >= -0.0625,
        "action_norm_within_5_percent": abs(metric_changes["action_norm_ee6"]) <= 0.05,
        "slow_reference": metric_changes["dp_ref_l2_ee6"] <= 0.05,
        "gate_state_consistency": gate_steps["state_mismatches"] == 0,
    }
    summary = {
        "version": args.version,
        "baseline_dir": baseline_dir.as_posix(),
        "candidate_dir": candidate_dir.as_posix(),
        "baseline_avg_seq_len": baseline["avg_seq_len"],
        "candidate_avg_seq_len": candidate["avg_seq_len"],
        "avg_seq_len_delta": avg_delta,
        "baseline_chain_sr": baseline_chain,
        "candidate_chain_sr": candidate_chain,
        "chain_deltas": chain_deltas,
        "paired": paired,
        "metric_relative_changes": metric_changes,
        "gate_steps": gate_steps,
        "gates": gates,
        "pass": all(gates.values()),
    }
    (output_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# {args.version} fixed-16 gate report",
        "",
        f"- Average length: `{candidate['avg_seq_len']:.4f}` vs baseline `{baseline['avg_seq_len']:.4f}` "
        f"(`{avg_delta:+.4f}`).",
        f"- Paired improved/equal/worse: `{paired['improved']}/{paired['equal']}/{paired['worse']}`.",
        f"- Chain deltas: `{'/'.join(f'{100 * value:+.2f}' for value in chain_deltas)}` percentage points.",
        f"- Action norm change: `{100 * metric_changes['action_norm_ee6']:+.2f}%`; "
        f"expired norm: `{100 * metric_changes['expired_action_norm_ee6']:+.2f}%`.",
        f"- Jerk: `{100 * metric_changes['jerk_l2_ee6']:+.2f}%`; aggregation delta: "
        f"`{100 * metric_changes['aggregation_delta_ee6']:+.2f}%`; slow-reference error: "
        f"`{100 * metric_changes['dp_ref_l2_ee6']:+.2f}%`.",
        f"- Gate steps base/transition/mismatch: `{gate_steps['base']}/{gate_steps['transition']}/"
        f"{gate_steps['state_mismatches']}`.",
        f"- Gate result: `{'PASS' if summary['pass'] else 'FAIL'}`.",
    ]
    (output_dir / "gate_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
