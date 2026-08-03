import unittest

import torch

from train_transition_lora_stale_action_condition_0716 import (
    CATEGORIES,
    DEFAULT_LORA_TARGETS,
    front_weighted_mse,
)


class StaleActionConditionTrainerTest(unittest.TestCase):
    def test_targets_add_action_embedding_without_temporal_or_head(self):
        self.assertEqual(CATEGORIES, ("normal", "stale"))
        self.assertEqual(len(DEFAULT_LORA_TARGETS), 6)
        self.assertIn("model.x_embedder", DEFAULT_LORA_TARGETS)
        self.assertTrue(all("attn_temporal" not in name for name in DEFAULT_LORA_TARGETS))
        self.assertTrue(all("final_layer" not in name for name in DEFAULT_LORA_TARGETS))

    def test_front_weighted_mse_value_and_gradient_ratio(self):
        prediction = torch.zeros((1, 3, 1), requires_grad=True)
        target = torch.ones_like(prediction)
        loss = front_weighted_mse(prediction, target, 1, 2.0)
        self.assertAlmostEqual(float(loss.detach()), 1.0)
        loss.backward()
        gradient = prediction.grad.detach().abs().flatten()
        self.assertAlmostEqual(float(gradient[0] / gradient[1]), 2.0)
        self.assertAlmostEqual(float(gradient[1]), float(gradient[2]))


if __name__ == "__main__":
    unittest.main()
