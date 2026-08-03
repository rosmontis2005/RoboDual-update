"""Fast contract tests for failure-recovery state grouping and snapshots."""

from collections import deque
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "vla-scripts" / "evaluate_calvin_failure_recovery_0718.py"
sys.path.insert(0, SCRIPT.parent.as_posix())
SPEC = importlib.util.spec_from_file_location("recovery_eval_0718", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DummyModel:
    def __init__(self):
        self.action = torch.tensor([1.0])
        self.hidden_states = torch.tensor([2.0])
        self.action_buffer = np.ones((2, 2), dtype=np.float32)
        self.hist_action = deque([torch.tensor([3.0])], maxlen=4)
        self.last_step_profile = {"slow_age_after": 8}


class FailureRecoveryContractTest(unittest.TestCase):
    def test_runtime_snapshot_isolation_and_restore(self):
        model = DummyModel()
        snapshot = MODULE.capture_recovery_model_state(model)
        model.action.add_(10)
        model.action_buffer[:] = 0
        model.hist_action[0].add_(10)
        MODULE.restore_recovery_model_state(model, snapshot)
        self.assertEqual(model.action.item(), 1.0)
        self.assertTrue(np.all(model.action_buffer == 1))
        self.assertEqual(model.hist_action[0].item(), 3.0)

    def test_pairs_never_cross_failure_states(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MODULE.FailureRecoveryWriter(Path(directory) / "data")
            writer.branches = [
                {"branch_id": "a_pos", "failure_state_id": "a", "split": "train", "success": True},
                {"branch_id": "a_neg", "failure_state_id": "a", "split": "train", "success": False},
                {"branch_id": "b_pos", "failure_state_id": "b", "split": "test", "success": True},
                {"branch_id": "b_neg", "failure_state_id": "b", "split": "test", "success": False},
            ]
            writer.finalize_state_pairs("a", 4)
            writer.finalize_state_pairs("b", 4)
            self.assertEqual(len(writer.pairs), 2)
            for pair in writer.pairs:
                positive = next(item for item in writer.branches if item["branch_id"] == pair["positive_branch_id"])
                negative = next(item for item in writer.branches if item["branch_id"] == pair["negative_branch_id"])
                self.assertEqual(positive["failure_state_id"], pair["failure_state_id"])
                self.assertEqual(negative["failure_state_id"], pair["failure_state_id"])

    def test_failed_baseline_continuation_can_pair_with_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MODULE.FailureRecoveryWriter(Path(directory) / "data")
            writer.branches = [
                {
                    "branch_id": "state_demo_guided_00",
                    "failure_state_id": "state",
                    "split": "validation",
                    "strategy": "demo_guided",
                    "success": True,
                },
                {
                    "branch_id": "state_baseline_failed_continuation_00",
                    "failure_state_id": "state",
                    "split": "validation",
                    "strategy": "baseline_failed_continuation",
                    "success": False,
                    "source_subtask_failure_confirmed": True,
                },
            ]
            self.assertEqual(writer.finalize_state_pairs("state", 4), 1)
            self.assertEqual(
                writer.pairs[0]["negative_branch_id"],
                "state_baseline_failed_continuation_00",
            )

    def test_state_split_is_stable(self):
        ids = [f"state_{index}" for index in range(100)]
        first = [MODULE._state_split(item) for item in ids]
        second = [MODULE._state_split(item) for item in ids]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
