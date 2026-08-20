#!/usr/bin/env python3
"""CPU-safe unit tests for M2a counterfactual-view contracts."""

from __future__ import annotations

import argparse
import importlib.util
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT = Path(__file__).with_name("train_m2a_reference_robust.py")
SPEC = importlib.util.spec_from_file_location("m2a_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class FakePolicy:
    def __init__(self):
        self.noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=100))
        self.cond_drop_chance = 0.1
        self.calls = 0

    def compute_loss(self, **kwargs):
        self.calls += 1
        loss = torch.tensor(2.0 if torch.count_nonzero(kwargs["ref_valid_mask"]) else 4.0)
        return {
            "loss": loss,
            "prediction": torch.full_like(kwargs["trajectory"], float(self.calls)),
            "noise": kwargs["noise"],
            "timesteps": kwargs["timesteps"],
            "cond_mask": kwargs["cond_mask"],
        }


def fake_batch(count: int) -> dict:
    ref = torch.zeros(1, 8, 7)
    ref[:, :count] = torch.arange(max(1, count * 7), dtype=torch.float32)[:count * 7].reshape(1, count, 7) + 1
    mask = torch.zeros(1, 8)
    mask[:, :count] = 1
    return {
        "ref_action": ref,
        "ref_valid_mask": mask,
        "ref_valid_count": torch.tensor([count]),
        "slow_hidden": torch.randn(1, 83, 4096),
        "current_rgb": torch.randn(1, 3, 8, 8),
        "previous_rgb": torch.randn(1, 3, 8, 8),
        "depth_image": torch.randn(1, 8, 8),
        "gripper_image": torch.randn(1, 3, 8, 8),
        "depth_gripper": torch.randn(1, 8, 8),
        "proprio": torch.randn(1, 7),
        "raw_action": torch.randn(1, 8, 7),
        "hist_action": torch.randn(1, 4, 7),
        "instruction": ["test instruction"],
    }


class ViewContracts(unittest.TestCase):
    def test_zero_ref_kept_preserves_all_non_condition_fields(self):
        batch = fake_batch(6)
        base = M.persisted_view(batch)
        view = M.make_named_view(batch, "zero_ref_hidden_kept")
        checks = M.assert_view_contracts(batch, base, view)
        self.assertTrue(all(checks.values()))
        self.assertEqual(torch.count_nonzero(view["ref_action"]).item(), 0)
        self.assertEqual(torch.count_nonzero(view["ref_valid_mask"]).item(), 0)
        self.assertTrue(torch.equal(view["slow_hidden"], batch["slow_hidden"]))

    def test_hidden_null_is_not_a_formal_view(self):
        batch = fake_batch(2)
        self.assertNotIn("zero_ref_hidden_null", M.SELECTABLE_COUNTERFACTUAL_VIEWS)
        with self.assertRaises(ValueError):
            M.make_named_view(batch, "zero_ref_hidden_null")

    def test_shortened_never_adds_reference(self):
        batch = fake_batch(7)
        for k in range(7):
            view = M.make_named_view(batch, "shortened_reference", k=k)
            checks = M.assert_view_contracts(batch, M.persisted_view(batch), view)
            self.assertTrue(all(checks.values()))
            self.assertLess(k, 7)
            self.assertEqual(int(view["ref_valid_mask"].sum()), k)
            self.assertTrue(torch.equal(view["ref_action"][:, :k], batch["ref_action"][:, :k]))
            self.assertEqual(torch.count_nonzero(view["ref_action"][:, k:]).item(), 0)
            self.assertTrue(torch.equal(view["slow_hidden"], batch["slow_hidden"]))

    def test_expired_sample_has_no_counterfactual(self):
        batch = fake_batch(0)
        args = argparse.Namespace(
            zero_ref_kept_probability=1.0,
            shortened_ref_probability=0.0,
        )
        for seed in range(10):
            view = M.choose_counterfactual_view(batch, args, random.Random(seed))
            self.assertIsNone(view)

    def test_selectable_training_views_never_zero_hidden(self):
        batch = fake_batch(4)
        for name, k in (("zero_ref_hidden_kept", None), ("shortened_reference", 2)):
            view = M.make_named_view(batch, name, k=k)
            self.assertTrue(torch.equal(view["slow_hidden"], batch["slow_hidden"]))

    def test_counterfactual_selector_only_returns_formal_views(self):
        batch = fake_batch(4)
        args = argparse.Namespace(zero_ref_kept_probability=0.8, shortened_ref_probability=0.2)
        selected = {
            M.choose_counterfactual_view(batch, args, random.Random(seed))["name"]
            for seed in range(100)
        }
        self.assertTrue(selected)
        self.assertTrue(selected.issubset(set(M.SELECTABLE_COUNTERFACTUAL_VIEWS)))

    def test_shortened_rejects_zero_count_and_future_tokens(self):
        with self.assertRaises(ValueError):
            M.make_named_view(fake_batch(0), "shortened_reference", k=0)
        with self.assertRaises(ValueError):
            M.make_named_view(fake_batch(3), "shortened_reference", k=3)


class LossContracts(unittest.TestCase):
    @staticmethod
    def loss_args():
        return argparse.Namespace(
            persisted_loss_weight=1.0,
            counterfactual_loss_weight=1.0,
            consistency_weight=0.0,
        )

    def test_paired_supervised_loss_is_weighted_mean(self):
        persisted = torch.tensor(2.0)
        counterfactual = torch.tensor(4.0)
        equal, denominator = M.weighted_supervised_loss(persisted, counterfactual, 1.0, 1.0)
        self.assertEqual(equal.item(), 3.0)
        self.assertEqual(denominator, 2.0)
        unequal, denominator = M.weighted_supervised_loss(persisted, counterfactual, 1.0, 3.0)
        self.assertEqual(unequal.item(), 3.5)
        self.assertEqual(denominator, 4.0)

    def test_single_persisted_loss_keeps_scale(self):
        persisted = torch.tensor(2.0)
        supervised, denominator = M.weighted_supervised_loss(persisted, None, 1.0, 1.0)
        self.assertIs(supervised, persisted)
        self.assertEqual(supervised.item(), 2.0)
        self.assertEqual(denominator, 1.0)

    def test_expired_paired_loss_plans_exactly_one_forward(self):
        policy = FakePolicy()
        total, persisted, counterfactual, audit = M.paired_loss(
            policy, fake_batch(0), None, self.loss_args(),
        )
        self.assertEqual(policy.calls, 1)
        self.assertIsNone(counterfactual)
        self.assertEqual(total.item(), persisted["loss"].item())
        self.assertFalse(audit["has_counterfactual"])
        self.assertTrue(audit["paired_rng_not_applicable"])
        self.assertIsNone(audit["counterfactual_loss"])
        self.assertEqual(audit["supervised_weight_denominator"], 1.0)

    def test_valid_reference_paired_loss_uses_weighted_mean(self):
        policy = FakePolicy()
        batch = fake_batch(4)
        counter = M.make_named_view(batch, "zero_ref_hidden_kept")
        total, persisted, counterfactual, audit = M.paired_loss(
            policy, batch, counter, self.loss_args(),
        )
        self.assertEqual(policy.calls, 2)
        self.assertEqual(persisted["loss"].item(), 2.0)
        self.assertEqual(counterfactual["loss"].item(), 4.0)
        self.assertEqual(total.item(), 3.0)
        self.assertTrue(audit["has_counterfactual"])
        self.assertFalse(audit["paired_rng_not_applicable"])
        self.assertEqual(audit["supervised_weight_denominator"], 2.0)

    def test_nonpositive_paired_denominator_fails_loudly(self):
        with self.assertRaises(ValueError):
            M.weighted_supervised_loss(torch.tensor(2.0), torch.tensor(4.0), 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
