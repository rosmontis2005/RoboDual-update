"""Committed-action history adapter for the Legato-like specialist.

The zero-initialized output projection makes this module an exact no-op at
construction time.  A trainer should optimize the adapter and ``history_gate``
while keeping the base specialist frozen (apart from the selected LoRA paths).
"""

from __future__ import annotations

import torch
from torch import nn


class CommittedHistoryAdapter(nn.Module):
    """Map the last ``history_steps`` 7-D executed actions to DiT condition."""

    def __init__(self, hidden_size: int, history_steps: int = 4, action_dim: int = 7):
        super().__init__()
        if hidden_size <= 0 or history_steps <= 0 or action_dim <= 0:
            raise ValueError("hidden_size, history_steps, and action_dim must be positive")
        self.history_steps = int(history_steps)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.history_steps * self.action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Keep the residual exactly zero while allowing gradients to reach the
        # zero-initialized output projection on the first optimizer step.
        self.history_gate = nn.Parameter(torch.ones(()))

    def forward(self, hist_action: torch.Tensor) -> torch.Tensor:
        expected = (self.history_steps, self.action_dim)
        if hist_action.ndim != 3 or tuple(hist_action.shape[-2:]) != expected:
            raise ValueError(
                f"hist_action must have shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(hist_action.shape)}"
            )
        feature = self.net(hist_action.reshape(hist_action.shape[0], -1))
        return self.history_gate * feature


def add_history_condition(global_condition: torch.Tensor, history_feature: torch.Tensor) -> torch.Tensor:
    """Broadcast one history feature over a token sequence when necessary."""

    if global_condition.ndim == 2:
        if global_condition.shape != history_feature.shape:
            raise ValueError("2-D global condition and history feature shapes do not match")
        return global_condition + history_feature
    if global_condition.ndim == 3:
        if global_condition.shape[0] != history_feature.shape[0] or global_condition.shape[2] != history_feature.shape[1]:
            raise ValueError("3-D global condition is incompatible with history feature")
        return global_condition + history_feature[:, None, :]
    raise ValueError(f"Expected a 2-D or 3-D global condition, got {tuple(global_condition.shape)}")


def install_history_adapter(policy: nn.Module, history_steps: int = 4) -> CommittedHistoryAdapter:
    """Register this adapter on a DiffusionDiTImagePolicy (or its DiT model)."""

    dit = getattr(policy, "model", policy)
    hidden_size = getattr(dit, "hidden_size", None)
    if hidden_size is None or not hasattr(dit, "history_adapter"):
        raise TypeError("Expected a RoboDual DiT model with optional history_adapter support")
    adapter = CommittedHistoryAdapter(hidden_size=int(hidden_size), history_steps=history_steps)
    reference = next(dit.parameters())
    adapter.to(device=reference.device, dtype=reference.dtype)
    dit.history_adapter = adapter
    return adapter


def install_dual_system_history_adapters(dual_system: nn.Module, history_steps: int = 4) -> tuple:
    """Install matching adapters on a DualSystem online model and EMA copy.

    Prefer installing on the fast policy before constructing ``DualSystem`` so
    EMA's deepcopy includes it automatically. This helper safely handles the
    common post-construction/checkpoint-loading case.
    """

    system = getattr(dual_system, "module", dual_system)
    fast_policy = getattr(system, "fast_system", None)
    ema_wrapper = getattr(system, "ema_fast_system", None)
    ema_policy = getattr(ema_wrapper, "ema_model", None)
    if fast_policy is None or ema_policy is None:
        raise TypeError("Expected DualSystem.fast_system and ema_fast_system.ema_model")
    online_adapter = install_history_adapter(fast_policy, history_steps=history_steps)
    ema_adapter = install_history_adapter(ema_policy, history_steps=history_steps)
    ema_adapter.load_state_dict(online_adapter.state_dict())
    ema_adapter.requires_grad_(False)
    return online_adapter, ema_adapter
