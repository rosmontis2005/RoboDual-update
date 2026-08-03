import unittest
from collections import Counter

from train_transition_lora_stale_condition_0716 import (
    CATEGORIES,
    DEFAULT_LORA_TARGETS,
    trajectory_category_balanced_sampler,
)


class DummyDataset:
    samples = [
        {"category": "normal", "trajectory_id": "normal_long"},
        {"category": "normal", "trajectory_id": "normal_long"},
        {"category": "normal", "trajectory_id": "normal_short"},
        {"category": "stale", "trajectory_id": "stale_only"},
    ]

    def __len__(self):
        return len(self.samples)


class StaleConditionTrainerTest(unittest.TestCase):
    def test_targets_only_touch_slow_condition_path(self):
        self.assertEqual(CATEGORIES, ("normal", "stale"))
        self.assertEqual(len(DEFAULT_LORA_TARGETS), 5)
        self.assertTrue(all("attn_temporal" not in name for name in DEFAULT_LORA_TARGETS))
        self.assertTrue(all("final_layer" not in name for name in DEFAULT_LORA_TARGETS))

    def test_balanced_sampler_equalizes_category_and_trajectory_mass(self):
        sampler = trajectory_category_balanced_sampler(DummyDataset(), seed=7)
        weights = list(sampler.weights.tolist())
        category_mass = Counter()
        trajectory_mass = Counter()
        for sample, weight in zip(DummyDataset.samples, weights):
            category_mass[sample["category"]] += weight
            trajectory_mass[(sample["category"], sample["trajectory_id"])] += weight
        self.assertAlmostEqual(category_mass["normal"], category_mass["stale"])
        self.assertAlmostEqual(
            trajectory_mass[("normal", "normal_long")],
            trajectory_mass[("normal", "normal_short")],
        )


if __name__ == "__main__":
    unittest.main()
