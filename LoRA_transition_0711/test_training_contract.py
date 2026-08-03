"""CPU-only contract tests for transition LoRA training."""

import unittest

import torch

from train_transition_lora import (
    CATEGORIES,
    DEFAULT_LORA_TARGETS,
    TransitionCollator,
    TransitionManifestDataset,
    deterministic_validation_subset,
)


class RefActionTest(unittest.TestCase):
    def setUp(self):
        self.collator = object.__new__(TransitionCollator)
        self.collator.empty_ref_after_age = 8
        self.action = torch.arange(56, dtype=torch.float32).reshape(8, 7)

    def test_full_ref_at_refresh(self):
        self.assertTrue(torch.equal(self.collator._ref_action(self.action, 0), self.action))

    def test_age_seven_uses_last_action(self):
        ref = self.collator._ref_action(self.action, 7)
        self.assertTrue(torch.equal(ref[0], self.action[-1]))
        self.assertEqual(torch.count_nonzero(ref[1:]).item(), 0)

    def test_age_eight_is_empty(self):
        self.assertEqual(torch.count_nonzero(self.collator._ref_action(self.action, 8)).item(), 0)


class TrainingSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation = TransitionManifestDataset(
            "RoboDual/LoRA_transition_0711/collected_transition_v1_repaired", "validation"
        )

    def test_validation_subset_has_equal_categories_and_homogeneous_batches(self):
        subset = deterministic_validation_subset(self.validation, per_category=8, seed=123)
        categories = [self.validation.samples[index]["category"] for index in subset.indices]
        self.assertEqual(len(categories), 32)
        for category in CATEGORIES:
            self.assertEqual(categories.count(category), 8)
        for start in range(0, len(categories), 2):
            self.assertEqual(len(set(categories[start : start + 2])), 1)

    def test_default_lora_target_contract(self):
        self.assertEqual(len(DEFAULT_LORA_TARGETS), 14)
        self.assertIn("model.x_embedder", DEFAULT_LORA_TARGETS)
        self.assertIn("model.context_adapter", DEFAULT_LORA_TARGETS)
        self.assertIn("model.blocks.5.attn_temporal.qkv", DEFAULT_LORA_TARGETS)
        self.assertIn("model.blocks.5.attn_temporal.proj", DEFAULT_LORA_TARGETS)

    def test_collator_rejects_batch_padding(self):
        collator = object.__new__(TransitionCollator)
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            collator([{}, {}])


if __name__ == "__main__":
    unittest.main()
