"""Contracts for drift-aware transition-LoRA checkpoint selection."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR.parent / "LoRA_trial", THIS_DIR.parent, THIS_DIR.parent.parent):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from train_transition_lora_drift_selected_0714 import checkpoint_selection_score


class DriftSelectedTrainingContractTest(unittest.TestCase):
    def test_selection_score_uses_all_preservation_terms(self):
        args = SimpleNamespace(
            selection_normal_drift_weight=1.0,
            selection_overall_drift_weight=1.0,
            selection_normal_gripper_drift_weight=2.0,
        )
        metrics = {
            "normal_drift": 1e-6,
            "overall_drift": 2e-6,
            "normal_drift_gripper": 3e-7,
        }
        self.assertAlmostEqual(checkpoint_selection_score(metrics, args), 3.6e-6)

    def test_lower_drift_wins_after_equal_transition_eligibility(self):
        args = SimpleNamespace(
            selection_normal_drift_weight=1.0,
            selection_overall_drift_weight=1.0,
            selection_normal_gripper_drift_weight=2.0,
        )
        weak = {
            "normal_drift": 1.28e-6,
            "overall_drift": 1.17e-6,
            "normal_drift_gripper": 0.33e-6,
        }
        strong = {
            "normal_drift": 2.93e-6,
            "overall_drift": 3.02e-6,
            "normal_drift_gripper": 0.95e-6,
        }
        self.assertLess(
            checkpoint_selection_score(weak, args),
            checkpoint_selection_score(strong, args),
        )


if __name__ == "__main__":
    unittest.main()
