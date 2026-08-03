"""Fast contracts for comprehensive positive/negative persistent replay audits."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parent / "audit_persistent_replay.py"
SPEC = importlib.util.spec_from_file_location("persistent_replay_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PersistentAuditContractTest(unittest.TestCase):
    def test_plan_selects_both_outcomes_for_each_trainable_branchable_state(self):
        states = [
            {"failure_state_id": "a", "split": "train"},
            {"failure_state_id": "b", "split": "validation"},
            {"failure_state_id": "c", "split": "test"},
        ]
        branches = [
            {"branch_id": "a_pos", "failure_state_id": "a", "success": True, "steps": 8},
            {"branch_id": "a_neg", "failure_state_id": "a", "success": False, "steps": 80},
            # A positive shorter than one action chunk must not admit the state.
            {"branch_id": "b_pos", "failure_state_id": "b", "success": True, "steps": 7},
            {"branch_id": "b_neg", "failure_state_id": "b", "success": False, "steps": 80},
            {"branch_id": "c_pos", "failure_state_id": "c", "success": True, "steps": 9},
            {"branch_id": "c_neg", "failure_state_id": "c", "success": False, "steps": 80},
        ]
        plan = MODULE.branchable_audit_plan(states, branches)
        self.assertEqual([item[0]["failure_state_id"] for item in plan], ["a", "c"])
        for _, positives, negative in plan:
            self.assertTrue(positives)
            self.assertTrue(all(item["success"] for item in positives))
            self.assertFalse(negative["success"])

    def test_diagnostic_cap_is_state_based(self):
        states = [
            {"failure_state_id": name, "split": "train"}
            for name in ("a", "b")
        ]
        branches = [
            {
                "branch_id": f"{name}_{outcome}",
                "failure_state_id": name,
                "success": outcome == "pos",
                "steps": 8,
            }
            for name in ("a", "b")
            for outcome in ("pos", "neg")
        ]
        self.assertEqual(
            len(MODULE.branchable_audit_plan(states, branches, max_states=1)),
            1,
        )


if __name__ == "__main__":
    unittest.main()
