"""CPU-only contracts for the base-preserving transition LoRA trainer."""

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR.parent / "LoRA_trial", THIS_DIR.parent, THIS_DIR.parent.parent):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from history_adapter import CommittedHistoryAdapter
from train_lora_specialist import LoRALinear
from train_transition_lora_preserved_0713 import (
    DEFAULT_LORA_TARGETS,
    base_policy_mode,
    compute_preserved_objective,
    preservation_constraints,
    should_stop_early,
)
from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2.0, dropout=0.0)
        self.model = SimpleNamespace(history_adapter=CommittedHistoryAdapter(hidden_size=4))


class DummyObjectivePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        base = nn.Linear(7, 7, bias=False)
        base.weight.data.copy_(torch.eye(7))
        self.projection = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
        self.projection.lora_B.data.fill_(0.1)
        self.model = nn.Module()
        self.model.history_adapter = CommittedHistoryAdapter(hidden_size=4)

    def compute_loss(self, trajectory, noise=None, timesteps=None, cond_mask=None, **kwargs):
        prediction = self.projection(trajectory)
        target = torch.zeros_like(prediction)
        return {
            "loss": torch.nn.functional.mse_loss(prediction, target),
            "prediction": prediction,
            "target": target,
            "noise": torch.zeros_like(trajectory) if noise is None else noise,
            "timesteps": torch.zeros(trajectory.shape[0], dtype=torch.long) if timesteps is None else timesteps,
            "cond_mask": cond_mask,
        }


class PreservedTrainingContractTest(unittest.TestCase):
    def test_only_final_two_temporal_output_projections_are_targeted(self):
        self.assertEqual(
            DEFAULT_LORA_TARGETS,
            (
                "model.blocks.4.attn_temporal.proj",
                "model.blocks.5.attn_temporal.proj",
            ),
        )

    def test_base_policy_mode_disables_and_restores_residuals(self):
        policy = DummyPolicy()
        policy.projection.scaling = 0.75
        policy.model.history_adapter.history_gate.data.fill_(1.25)
        with base_policy_mode(policy):
            self.assertEqual(policy.projection.scaling, 0.0)
            self.assertEqual(policy.model.history_adapter.history_gate.item(), 0.0)
        self.assertEqual(policy.projection.scaling, 0.75)
        self.assertEqual(policy.model.history_adapter.history_gate.item(), 1.25)

    def test_preservation_constraints_are_conjunctive(self):
        args = SimpleNamespace(
            max_normal_prediction_drift=2e-4,
            max_overall_prediction_drift=5e-4,
            max_normal_gripper_drift=1e-4,
        )
        good = {
            "normal_drift": 1e-4,
            "overall_drift": 4e-4,
            "normal_drift_gripper": 9e-5,
        }
        passed, checks = preservation_constraints(good, args)
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))
        good["normal_gripper_drift"] = 2e-4
        good["normal_drift_gripper"] = 2e-4
        self.assertFalse(preservation_constraints(good, args)[0])

    def test_early_stopping_waits_for_minimum_training_steps(self):
        args = SimpleNamespace(
            min_steps_before_early_stopping=2000,
            early_stopping_patience=10,
        )
        self.assertFalse(should_stop_early(1900, 10, args))
        self.assertFalse(should_stop_early(2000, 9, args))
        self.assertTrue(should_stop_early(2000, 10, args))

    def test_policy_loss_supports_matched_noise_and_details(self):
        parameters = inspect.signature(DiffusionDiTImagePolicy.compute_loss).parameters
        self.assertIn("noise", parameters)
        self.assertIn("timesteps", parameters)
        self.assertIn("cond_mask", parameters)
        self.assertIn("return_details", parameters)

    def test_objective_rejects_stochastic_training_mode(self):
        policy = DummyPolicy()
        policy.train()
        with self.assertRaisesRegex(RuntimeError, "attention-dropout"):
            compute_preserved_objective(policy, {}, SimpleNamespace())

    def test_preservation_gradient_reaches_only_student_lora(self):
        policy = DummyObjectivePolicy().eval()
        args = SimpleNamespace(
            bf16=False,
            gripper_preservation_weight=2.0,
            normal_supervised_weight=0.0,
            transition_supervised_weight=1.0,
            normal_preservation_weight=4.0,
            transition_preservation_weight=1.0,
        )
        action = torch.randn(1, 8, 7)
        batch = {
            "raw_action": action,
            "ref_action": action,
            "action_cond": action,
            "pixel_values_dp": action,
            "prev_pixel_values_dp": action,
            "depth_image": action,
            "gripper_image": action,
            "depth_gripper": action,
            "lang": ["dummy"],
            "proprio": action,
            "hist_action": torch.zeros(1, 4, 7),
            "category": ["normal"],
        }
        details = compute_preserved_objective(policy, batch, args)
        self.assertGreater(details["drift"].item(), 0)
        details["loss"].backward()
        self.assertGreater(policy.projection.lora_A.grad.abs().sum().item(), 0)
        self.assertGreater(policy.projection.lora_B.grad.abs().sum().item(), 0)
        self.assertIsNone(policy.projection.base.weight.grad)


if __name__ == "__main__":
    unittest.main()
