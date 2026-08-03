"""Small CPU tests for sampling and history-adapter invariants."""

import unittest
from collections import Counter

import torch

from collect_transition_rollouts import (
    Candidate,
    candidate_requirement_keys,
    parse_group_quotas,
    sample_deficits,
    sample_requirements,
    select_samples,
    stable_split,
)
from history_adapter import (
    CommittedHistoryAdapter,
    add_history_condition,
    install_dual_system_history_adapters,
    install_history_adapter,
)


class HistoryAdapterTest(unittest.TestCase):
    def test_zero_output_has_live_gradient(self):
        adapter = CommittedHistoryAdapter(hidden_size=32)
        history = torch.randn(3, 4, 7)
        output = adapter(history)
        self.assertEqual(tuple(output.shape), (3, 32))
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        output.sum().backward()
        self.assertGreater(adapter.net[-1].weight.grad.abs().sum().item(), 0)

    def test_condition_broadcast_and_shape_validation(self):
        feature = torch.zeros(2, 32)
        self.assertEqual(tuple(add_history_condition(torch.randn(2, 5, 32), feature).shape), (2, 5, 32))
        with self.assertRaises(ValueError):
            CommittedHistoryAdapter(32)(torch.randn(2, 3, 7))

    def test_install_registers_trainable_module(self):
        class DummyDiT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 32
                self.anchor = torch.nn.Parameter(torch.ones(1))
                self.history_adapter = None

        dit = DummyDiT()
        adapter = install_history_adapter(dit)
        self.assertIs(dit.history_adapter, adapter)
        self.assertIn("history_adapter.history_gate", dict(dit.named_parameters()))

    def test_dual_installer_covers_online_and_ema(self):
        class DummyDiT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 32
                self.anchor = torch.nn.Parameter(torch.ones(1))
                self.history_adapter = None

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = DummyDiT()

        class Holder:
            pass

        dual = Holder()
        dual.fast_system = Wrapper()
        dual.ema_fast_system = Holder()
        dual.ema_fast_system.ema_model = Wrapper()
        online, ema = install_dual_system_history_adapters(dual)
        self.assertIs(dual.fast_system.model.history_adapter, online)
        self.assertIs(dual.ema_fast_system.ema_model.model.history_adapter, ema)
        self.assertFalse(any(parameter.requires_grad for parameter in ema.parameters()))


class SamplingTest(unittest.TestCase):
    def test_exact_mix_without_duplicates(self):
        candidates = []
        for split in ("train", "validation", "test"):
            for category in ("normal", "refresh", "high_conflict", "stale"):
                for index in range(100):
                    trajectory = f"{split}_{category}_trajectory_{index}"
                    candidates.append(Candidate(
                        trajectory, split, index, category, 0, None,
                        0, None, None, None, None, None, False,
                    ))
        selected, stats = select_samples(candidates, target=200, seed=42)
        self.assertEqual(len(selected), 200)
        self.assertEqual(
            stats["selected_by_category"],
            {"normal": 100, "refresh": 60, "high_conflict": 20, "stale": 20},
        )
        keys = {(item.trajectory_id, item.step) for item in selected}
        self.assertEqual(len(keys), len(selected))

    def test_high_only_refresh_pool_is_sampled_without_reuse(self):
        candidates = []
        for split in ("train", "validation", "test"):
            for category, count in (("normal", 100), ("high_conflict", 100), ("stale", 100)):
                for index in range(count):
                    candidates.append(Candidate(
                        f"{split}_{category}_{index}", split, index, category, 0, None,
                        0, None, None, None, None, None, False,
                    ))
        selected, stats = select_samples(candidates, target=200, seed=7)
        self.assertEqual(len(selected), 200)
        self.assertEqual(stats["selected_by_category"]["refresh"], 60)
        self.assertEqual(stats["selected_by_category"]["high_conflict"], 20)
        self.assertEqual(len({(item.trajectory_id, item.step) for item in selected}), 200)

    def test_group_quota_parser(self):
        self.assertEqual(parse_group_quotas("A:60,B=60,C:30,D:20")["D"], 20)
        with self.assertRaises(ValueError):
            parse_group_quotas("A:1,B:1,C:1")

    def test_category_deficit_blocks_completion_even_when_total_is_large(self):
        requirements = sample_requirements(200)
        available = Counter(requirements)
        available[("test", "high_conflict")] = 0
        available[("train", "normal")] += requirements[("test", "high_conflict")] + 100
        self.assertGreaterEqual(sum(available.values()), sum(requirements.values()))
        self.assertEqual(
            sample_deficits(available, requirements),
            {"test:high_conflict": requirements[("test", "high_conflict")]},
        )

    def test_no_deficit_only_when_every_partition_is_full(self):
        requirements = sample_requirements(8000)
        self.assertEqual(sample_deficits(Counter(requirements), requirements), {})
        self.assertEqual(requirements[("train", "refresh_total")], 2240)
        self.assertEqual(requirements[("test", "high_conflict")], 120)

    def test_high_conflict_counts_as_refresh_without_duplicate_sample(self):
        candidate = Candidate(
            "trajectory", "test", 10, "high_conflict", 1, 0,
            0, 8, 0.3, 0.4, 0.2, 0.3, False,
        )
        self.assertEqual(
            candidate_requirement_keys(candidate),
            (("test", "refresh_total"), ("test", "high_conflict")),
        )


if __name__ == "__main__":
    unittest.main()
