#!/usr/bin/env python3
"""Verify the paired collector's fixed-age-12 schedule and trace neutrality."""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PAIR_ROOT = HERE.parent / "original_8_steps"
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PAIR_ROOT))
sys.path.insert(2, str(REPO_ROOT / "vla-scripts"))

import dual_sys_evaluation_0424test as age_evaluator
from collect_fixed_12_steps import (
    FixedAge12Evaluation,
    FixedAge12TraceCapture,
    is_fixed_age12_slow_step,
)
from trace_capture import TraceWriter

PAIR_SPEC = importlib.util.spec_from_file_location(
    "robodual_paired_fixed8_checks", PAIR_ROOT / "verify_collection_contract.py"
)
if PAIR_SPEC is None or PAIR_SPEC.loader is None:
    raise ImportError("Cannot load paired fixed-mod-8 checks")
paired_checks = importlib.util.module_from_spec(PAIR_SPEC)
PAIR_SPEC.loader.exec_module(paired_checks)


def main() -> None:
    # The subclass pins constructor settings but does not replace action logic.
    assert FixedAge12Evaluation.step is age_evaluator.DualSystemCalvinEvaluation.step
    assert (
        FixedAge12Evaluation._should_call_slow_system
        is age_evaluator.DualSystemCalvinEvaluation._should_call_slow_system
    )
    assert (
        FixedAge12Evaluation._build_ref_actions_from
        is age_evaluator.DualSystemCalvinEvaluation._build_ref_actions_from
    )

    schedule_probe = object.__new__(FixedAge12Evaluation)
    schedule_probe.slow_trigger_policy = "age_empty"
    schedule_probe.max_slow_age = 12
    schedule_probe.last_slow_step = None
    observed_slow_steps = []
    observed_reasons = []
    for step in range(240):
        called, reason = schedule_probe._should_call_slow_system(step)
        if called:
            observed_slow_steps.append(step)
            observed_reasons.append(reason)
            schedule_probe.last_slow_step = step
    expected_slow_steps = list(range(0, 240, 12))
    assert observed_slow_steps == expected_slow_steps
    assert observed_reasons == ["initial"] + ["max_slow_age"] * (len(expected_slow_steps) - 1)
    assert [step for step in range(240) if is_fixed_age12_slow_step(step)] == expected_slow_steps

    schedule_probe.temporal_size = 8
    schedule_probe.empty_ref_after_age = 8
    action_chunk = torch.arange(56, dtype=torch.float32).reshape(1, 8, 7)
    cond_counts = []
    for age in range(12):
        ref, count = schedule_probe._build_ref_actions_from(action_chunk, age)
        cond_counts.append(count)
        if count:
            assert torch.equal(ref[:, :count], action_chunk[:, -count:])
        assert torch.count_nonzero(ref[:, count:]) == 0
    assert cond_counts == [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0]

    # Reuse the paired collector's deterministic fake models to prove that the
    # schedule-aware trace hooks neither change output nor advance RNG.
    torch.manual_seed(1234)
    control = paired_checks.FakeEvaluator(paired_checks.FakeSlow(), paired_checks.FakeFast())
    control_result = paired_checks.run_models(control)

    torch.manual_seed(1234)
    traced = paired_checks.FakeEvaluator(paired_checks.FakeSlow(), paired_checks.FakeFast())
    with tempfile.TemporaryDirectory(prefix="robodual_age12_trace_verify_") as temp:
        writer = TraceWriter(Path(temp) / "run", {"test": True})
        capture = FixedAge12TraceCapture(
            traced,
            writer,
            expected_slow_call=is_fixed_age12_slow_step,
            schedule_label="fixed_age12",
        )
        capture.begin_step(
            sequence_index=60,
            subtask_index=0,
            subtask="test",
            instruction="test",
            step=0,
            pre_obs={},
            pre_info={},
            pre_physics={},
        )
        traced_result = paired_checks.run_models(traced)
        step_path = capture.finalize_step(
            executed_action=torch.zeros(2),
            post_obs={"robot_obs": torch.zeros(1), "scene_obs": torch.zeros(1)},
            post_info={},
            post_physics={},
            task_success=False,
            profile={
                "slow_system": True,
                "slow_trigger_policy": "age_empty",
                "max_slow_age": 12,
                "empty_ref_after_age": 8,
                "slow_age_after": 0,
                "num_cond_actions": 8,
            },
        )
        try:
            saved = torch.load(step_path, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(step_path, map_location="cpu")
        capture.close()

    assert torch.equal(control_result[0][0], traced_result[0][0])
    assert torch.equal(control_result[0][1], traced_result[0][1])
    assert torch.equal(control_result[1], traced_result[1])
    assert torch.equal(control_result[2], traced_result[2])
    assert saved["generalist"]["called"]
    assert "initial_noise" in saved["specialist"]
    assert "dit_common_inputs" in saved["specialist"]
    assert "output_action_chunk" in saved["specialist"]

    print(
        json.dumps(
            {
                "age12_step_identity": True,
                "fixed_age12_schedule_steps_0_239": expected_slow_steps,
                "reference_condition_counts_ages_0_11": cond_counts,
                "empty_reference_ages": [8, 9, 10, 11],
                "trace_output_equal": True,
                "trace_rng_state_equal": True,
                "synthetic_capture_complete": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
