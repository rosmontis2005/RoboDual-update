#!/usr/bin/env python3
"""CPU-safe unit tests for M2a counterfactual-view contracts."""

from __future__ import annotations

import argparse
import importlib.util
import random
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).with_name("train_m2a_reference_robust.py")
SPEC = importlib.util.spec_from_file_location("m2a_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


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
        self.assertTrue(torch.equal(view["slow_hidden"], batch["slow_hidden"]))

    def test_zero_ref_null_keeps_hidden_shape_and_zeros_values(self):
        batch = fake_batch(2)
        view = M.make_named_view(batch, "zero_ref_hidden_null")
        checks = M.assert_view_contracts(batch, M.persisted_view(batch), view)
        self.assertTrue(all(checks.values()))
        self.assertEqual(view["slow_hidden"].shape, batch["slow_hidden"].shape)
        self.assertEqual(torch.count_nonzero(view["slow_hidden"]).item(), 0)

    def test_shortened_never_adds_reference(self):
        batch = fake_batch(7)
        for k in range(7):
            view = M.make_named_view(batch, "shortened_reference", k=k)
            checks = M.assert_view_contracts(batch, M.persisted_view(batch), view)
            self.assertTrue(all(checks.values()))
            self.assertEqual(int(view["ref_valid_mask"].sum()), k)
            self.assertEqual(torch.count_nonzero(view["ref_action"][:, k:]).item(), 0)

    def test_expired_sample_never_duplicates_persisted(self):
        batch = fake_batch(0)
        args = argparse.Namespace(
            zero_ref_kept_probability=1.0,
            zero_ref_null_probability=0.0,
            shortened_ref_probability=0.0,
        )
        for seed in range(10):
            view = M.choose_counterfactual_view(batch, args, random.Random(seed))
            self.assertEqual(view["name"], "zero_ref_hidden_null")
            self.assertEqual(torch.count_nonzero(view["slow_hidden"]).item(), 0)

    def test_shortened_rejects_zero_count_and_future_tokens(self):
        with self.assertRaises(ValueError):
            M.make_named_view(fake_batch(0), "shortened_reference", k=0)
        with self.assertRaises(ValueError):
            M.make_named_view(fake_batch(3), "shortened_reference", k=3)


if __name__ == "__main__":
    unittest.main()
