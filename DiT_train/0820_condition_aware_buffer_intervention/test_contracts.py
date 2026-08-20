#!/usr/bin/env python3
"""CPU-safe unit tests for M2 buffer-intervention contracts."""

from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("run_condition_aware_buffer_intervention.py")
SPEC = importlib.util.spec_from_file_location("condition_buffer_m2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fake_snapshot():
    values = {
        "action_buffer": np.arange(8 * 8 * 7, dtype=np.float64).reshape(8, 8, 7),
        "action_buffer_mask": np.tri(8, 8, dtype=np.bool_),
        "obs_buffer": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "hist_action": deque([np.arange(7, dtype=np.float32)], maxlen=4),
        "gripper_window": deque([1.0, -1.0], maxlen=8),
        "action": np.ones((1, 8, 7), dtype=np.float32),
        "hidden_states": np.ones((1, 3, 4), dtype=np.float16),
        "last_slow_step": 0,
        "prev_action": np.arange(7, dtype=np.float32),
        "prev_prev_action": np.arange(7, dtype=np.float32) - 1,
        "prev_proprio": np.arange(7, dtype=np.float32)[None],
        "prev_obs_tensor": np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        "last_step_profile": {"step": 7},
        "_slow_handover": None,
        "_fast_device": "cpu",
        "frozen_condition_id": "condition_000000",
        "forbidden_slow_call_count": 0,
        "_condition_injected": True,
    }
    assert set(values) == set(M.SNAPSHOT_FIELDS)
    return values


def wrapper_from(snapshot):
    return SimpleNamespace(**{key: M.clone_runtime(value) for key, value in snapshot.items()})


class BufferIsolationTests(unittest.TestCase):
    def test_flush_changes_only_two_fields(self):
        snapshot = fake_snapshot()
        keep = wrapper_from(snapshot)
        flush = wrapper_from(snapshot)
        M.flush_temporal_buffer(flush)
        audit = M.flush_isolation_audit(keep, flush, snapshot)
        self.assertTrue(audit["passed"])
        self.assertEqual(np.count_nonzero(flush.action_buffer), 0)
        self.assertFalse(np.any(flush.action_buffer_mask))
        for field in M.SNAPSHOT_FIELDS:
            if field not in M.FLUSH_FIELDS:
                self.assertTrue(M.exact_value_equal(getattr(keep, field), getattr(flush, field)), field)

    def test_non_buffer_mutation_fails_isolation(self):
        snapshot = fake_snapshot()
        keep = wrapper_from(snapshot)
        flush = wrapper_from(snapshot)
        M.flush_temporal_buffer(flush)
        flush.prev_action[0] += 1
        self.assertFalse(M.flush_isolation_audit(keep, flush, snapshot)["passed"])

    def test_keep_buffer_mutation_fails_isolation(self):
        snapshot = fake_snapshot()
        keep = wrapper_from(snapshot)
        flush = wrapper_from(snapshot)
        M.flush_temporal_buffer(flush)
        keep.action_buffer[0, 0, 0] += 1
        self.assertFalse(M.flush_isolation_audit(keep, flush, snapshot)["passed"])


class FactorialSummaryTests(unittest.TestCase):
    def test_first_step_tolerances_remain_strict(self):
        self.assertEqual(M.RAW_EQUALITY_ATOL, 1e-6)
        self.assertEqual(M.RAW_EQUALITY_RTOL, 1e-6)
        self.assertEqual(M.FLUSH_AGGREGATION_ATOL, 1e-6)

    def test_canonical_first_observation_roundtrip_hash(self):
        robot = np.arange(15, dtype=np.float32)
        branchpoint = {
            "robot_obs": robot,
            "selected_proprio": np.concatenate((robot[:6], robot[-1:])),
            "scene_obs": np.arange(6, dtype=np.float32),
            "rgb_static": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "rgb_gripper": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "depth_static": np.arange(4, dtype=np.float32).reshape(2, 2),
            "depth_gripper": np.arange(4, dtype=np.float32).reshape(2, 2),
        }
        policy_obs = M.T.dataset_observation(branchpoint)
        recaptured = M.T.capture_env_state(policy_obs)
        self.assertEqual(M.value_digest(branchpoint), M.value_digest(recaptured))

    def test_required_matched_control_contrasts_exist(self):
        rows = []
        for branch_i, branch in enumerate(M.BRANCHES):
            rows.append({
                "condition_id": "condition_000000", "intervention_age": 8,
                "branch": branch, "success_within_8": branch_i % 2 == 0,
                "success_within_16": branch_i % 3 == 0,
                **{key: float(branch_i) for key in M.CONTINUOUS_OUTCOMES},
            })
        contrasts = M.make_paired_contrasts(rows, seed=42)
        names = {row["contrast"] for row in contrasts}
        required = {
            "ref_keep_minus_old_keep", "full_keep_minus_old_keep",
            "ref_flush_minus_old_flush", "full_flush_minus_old_flush",
            "old_flush_minus_old_keep", "ref_flush_minus_ref_keep",
            "full_flush_minus_full_keep",
        }
        self.assertTrue(required.issubset(names))
        flush_condition = [
            row for row in contrasts
            if row["contrast"] == "full_flush_minus_old_flush"
        ]
        self.assertTrue(flush_condition)
        self.assertTrue(all(row["right_branch"] == "old_flush" for row in flush_condition))

    def test_branch_order_seed_is_deterministic(self):
        first = list(M.BRANCHES)
        second = list(M.BRANCHES)
        seed = M.intervention_seed(42, "condition_000001", 11)
        __import__("random").Random(seed).shuffle(first)
        __import__("random").Random(seed).shuffle(second)
        self.assertEqual(first, second)
        self.assertCountEqual(first, M.BRANCHES)


if __name__ == "__main__":
    unittest.main()
