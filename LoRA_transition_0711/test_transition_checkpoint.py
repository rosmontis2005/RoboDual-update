"""Contract tests for strict transition-checkpoint loading."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from transition_checkpoint import apply_transition_ablation, load_transition_specialist


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.base = nn.Linear(2, 2)
        self.model.history_adapter = nn.Linear(2, 2)


class FakeEMA(nn.Module):
    def __init__(self, online, ema):
        super().__init__()
        self.online_model = online
        self.ema_model = ema


class FakeDualSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_system = FakePolicy()
        self.ema_fast_system = FakeEMA(self.fast_system, FakePolicy())


class FakeHistoryAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_gate = nn.Parameter(torch.ones(()))
        self.net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))


class FakeAblationPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.base = nn.Linear(2, 2)
        self.model.history_adapter = FakeHistoryAdapter()


class FakeAblationDualSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_system = FakeAblationPolicy()
        self.ema_fast_system = FakeEMA(self.fast_system, FakeAblationPolicy())


class TransitionCheckpointTest(unittest.TestCase):
    def _save(self, state, directory, name):
        path = Path(directory) / name
        torch.save(state, path)
        return path

    def test_loads_raw_policy_into_online_and_ema(self):
        source = FakePolicy()
        source.model.history_adapter.weight.data.fill_(3.0)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(source.state_dict(), directory, "raw.pt")
            target = FakeDualSystem()
            info = load_transition_specialist(target, path)
        self.assertEqual(info["format"], "raw_policy")
        self.assertTrue(torch.equal(
            target.fast_system.model.history_adapter.weight,
            target.ema_fast_system.ema_model.model.history_adapter.weight,
        ))
        self.assertTrue(torch.all(target.ema_fast_system.ema_model.model.history_adapter.weight == 3.0))

    def test_loads_ema_wrapper_checkpoint(self):
        source = FakeDualSystem()
        source.ema_fast_system.ema_model.model.history_adapter.bias.data.fill_(4.0)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(source.ema_fast_system.state_dict(), directory, "ema.pt")
            target = FakeDualSystem()
            info = load_transition_specialist(target, path)
        self.assertEqual(info["format"], "ema_wrapper")
        self.assertTrue(torch.all(target.ema_fast_system.ema_model.model.history_adapter.bias == 4.0))

    def test_ignores_only_legacy_ema_dummy_keys(self):
        source = FakeDualSystem()
        state = source.ema_fast_system.state_dict()
        state["online_model._dummy_variable"] = torch.tensor(0.0)
        state["ema_model._dummy_variable"] = torch.tensor(0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(state, directory, "legacy_ema.pt")
            info = load_transition_specialist(FakeDualSystem(), path)
        self.assertEqual(
            info["ignored_legacy_keys"],
            ["ema_model._dummy_variable", "online_model._dummy_variable"],
        )

    def test_rejects_unknown_extra_ema_key(self):
        source = FakeDualSystem()
        state = source.ema_fast_system.state_dict()
        state["ema_model.unexpected_parameter"] = torch.tensor(0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(state, directory, "bad_ema.pt")
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                load_transition_specialist(FakeDualSystem(), path)

    def test_rejects_checkpoint_without_history_adapter(self):
        state = FakePolicy().state_dict()
        state = {key: value for key, value in state.items() if "history_adapter" not in key}
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(state, directory, "missing.pt")
            with self.assertRaisesRegex(RuntimeError, "missing policy history adapter"):
                load_transition_specialist(FakeDualSystem(), path)

    def test_ablation_modes_restore_only_requested_components(self):
        for mode, expected_base, expected_history in (
            ("full", 9.0, 7.0),
            ("lora_only", 9.0, 0.0),
            ("history_only", 2.0, 7.0),
            ("base", 2.0, 0.0),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                base = FakeAblationDualSystem()
                base.ema_fast_system.online_model.model.base.weight.data.fill_(2.0)
                base.ema_fast_system.ema_model.model.base.weight.data.fill_(2.0)
                base_state = {
                    key: value
                    for key, value in base.ema_fast_system.state_dict().items()
                    if "history_adapter" not in key
                }
                base_path = self._save(base_state, directory, "base.pt")

                target = FakeAblationDualSystem()
                target.ema_fast_system.online_model.model.base.weight.data.fill_(9.0)
                target.ema_fast_system.ema_model.model.base.weight.data.fill_(9.0)
                for policy in (target.fast_system, target.ema_fast_system.ema_model):
                    policy.model.history_adapter.net[-1].weight.data.fill_(7.0)
                    policy.model.history_adapter.net[-1].bias.data.fill_(7.0)

                info = apply_transition_ablation(target, base_path, mode)
                ema_policy = target.ema_fast_system.ema_model
                self.assertTrue(torch.all(ema_policy.model.base.weight == expected_base))
                self.assertTrue(torch.all(
                    ema_policy.model.history_adapter.net[-1].weight == expected_history
                ))
                self.assertEqual(info["trained_history_enabled"], expected_history != 0.0)


if __name__ == "__main__":
    unittest.main()
