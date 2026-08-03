import copy
import unittest

import torch

from finalize_transition_adapter_0715 import EXPECTED_MODULES, build_checkpoint


class FinalizeTransitionAdapterTest(unittest.TestCase):
    def test_only_expected_weights_change(self):
        base = {"untouched": torch.ones(2)}
        lora_state = {}
        for module in EXPECTED_MODULES:
            for prefix in ("ema_model", "online_model"):
                base[f"{prefix}.{module}.weight"] = torch.zeros(3, 4)
            lora_state[f"{module}.lora_A"] = torch.ones(2, 4)
            lora_state[f"{module}.lora_B"] = torch.ones(3, 2)
        payload = {
            "format": "robodual_transition_history_lora_v1",
            "metadata": {"step": 500, "args": {"lora_rank": 2, "lora_alpha": 2}},
            "lora_state": lora_state,
            "history_adapter_state": {
                "net.2.weight": torch.zeros(2, 2),
                "net.2.bias": torch.zeros(2),
            },
        }
        original = copy.deepcopy(base)
        output, summary = build_checkpoint(base, payload)
        self.assertTrue(torch.equal(output["untouched"], original["untouched"]))
        self.assertEqual(summary["adapter_step"], 500)
        self.assertEqual(len(summary["changed_base_keys"]), 4)
        for module in EXPECTED_MODULES:
            self.assertTrue(torch.equal(output[f"ema_model.{module}.weight"], torch.full((3, 4), 2.0)))


if __name__ == "__main__":
    unittest.main()
