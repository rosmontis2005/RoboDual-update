import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CALVIN_ROOT", (ROOT / "calvin").as_posix())
SCRIPT = ROOT / "RoboDual" / "vla-scripts" / "evaluate_calvin_task_age_transition_lora_gated_0714.py"
sys.path.insert(0, SCRIPT.parent.as_posix())
spec = importlib.util.spec_from_file_location("gated_evaluator_0714", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GatedEvaluatorContractTest(unittest.TestCase):
    def test_gate_activates_only_at_expired_reference_age(self):
        self.assertFalse(module.transition_gate_should_activate(None, 8))
        self.assertFalse(module.transition_gate_should_activate(0, 8))
        self.assertFalse(module.transition_gate_should_activate(7, 8))
        self.assertTrue(module.transition_gate_should_activate(8, 8))
        self.assertTrue(module.transition_gate_should_activate(13, 8))

    def test_gate_can_be_delayed_after_reference_expiration(self):
        self.assertFalse(module.transition_gate_should_activate(8, 10))
        self.assertFalse(module.transition_gate_should_activate(9, 10))
        self.assertTrue(module.transition_gate_should_activate(10, 10))
        self.assertTrue(module.transition_gate_should_activate(13, 10))

    def test_full_benchmark_requires_canonical_manifest(self):
        module.validate_sequence_selection(list(range(100)), full_benchmark=True)
        with self.assertRaises(ValueError):
            module.validate_sequence_selection(list(range(1, 101)), full_benchmark=True)
        with self.assertRaises(ValueError):
            module.validate_sequence_selection(list(range(21)), full_benchmark=False)

    def test_gate_target_profiles_are_explicit_and_disjoint(self):
        temporal = set(module.TRANSITION_GATE_TARGET_PROFILES["temporal_proj"])
        condition = set(module.TRANSITION_GATE_TARGET_PROFILES["stale_condition"])
        action_condition = set(
            module.TRANSITION_GATE_TARGET_PROFILES["stale_action_condition"]
        )
        self.assertEqual(len(temporal), 2)
        self.assertEqual(len(condition), 5)
        self.assertEqual(len(action_condition), 6)
        self.assertEqual(action_condition - condition, {"model.x_embedder.weight"})
        self.assertFalse(temporal & condition)
        self.assertTrue(
            all(name.endswith(".weight") for name in temporal | condition | action_condition)
        )


if __name__ == "__main__":
    unittest.main()
