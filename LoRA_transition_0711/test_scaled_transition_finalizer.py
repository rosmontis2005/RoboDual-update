import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, THIS_DIR.as_posix())

from finalize_transition_lora_scaled_0714 import TARGET_SUFFIXES, build_scaled_checkpoint


class ScaledTransitionFinalizerTest(unittest.TestCase):
    def test_only_targets_change_and_online_keeps_its_own_base(self):
        base = OrderedDict()
        merged = OrderedDict()
        for prefix, offset in (("ema_model", 1.0), ("online_model", 10.0)):
            for suffix in TARGET_SUFFIXES:
                key = f"{prefix}.{suffix}"
                base[key] = torch.full((2, 2), offset)
                merged[key] = torch.full((2, 2), 3.0 if prefix == "ema_model" else 99.0)
            base[f"{prefix}.untouched"] = torch.tensor([offset])
            merged[f"{prefix}.untouched"] = torch.tensor([99.0])
            for name in ("net.2.weight", "net.2.bias"):
                merged[f"{prefix}.model.history_adapter.{name}"] = torch.zeros(1)
        output, summary = build_scaled_checkpoint(base, merged, 0.5)
        for suffix in TARGET_SUFFIXES:
            self.assertTrue(torch.equal(output[f"ema_model.{suffix}"], torch.full((2, 2), 2.0)))
            self.assertTrue(torch.equal(output[f"online_model.{suffix}"], torch.full((2, 2), 11.0)))
        self.assertEqual(output["online_model.untouched"].item(), 10.0)
        self.assertEqual(len(summary["changed_base_keys"]), 4)

    def test_rejects_nonzero_history_output(self):
        base = OrderedDict()
        merged = OrderedDict()
        for prefix in ("ema_model", "online_model"):
            for suffix in TARGET_SUFFIXES:
                key = f"{prefix}.{suffix}"
                base[key] = torch.zeros(1)
                merged[key] = torch.ones(1)
        merged["ema_model.model.history_adapter.net.2.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "history output"):
            build_scaled_checkpoint(base, merged, 0.5)


if __name__ == "__main__":
    unittest.main()
