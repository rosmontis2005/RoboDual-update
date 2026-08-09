#!/usr/bin/env python3
"""Small, dependency-light contracts shared by the 0809 diagnostics."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


CONDITIONS = (
    "stale_hidden_empty_ref",
    "fresh_hidden_empty_ref",
    "stale_hidden_fresh_ref",
    "fresh_hidden_fresh_ref",
)

CONDITION_FACTORS = {
    "stale_hidden_empty_ref": {"hidden": "stale", "ref": "empty", "intervention": False},
    "fresh_hidden_empty_ref": {"hidden": "fresh", "ref": "empty", "intervention": True},
    "stale_hidden_fresh_ref": {"hidden": "stale", "ref": "fresh", "intervention": True},
    "fresh_hidden_fresh_ref": {"hidden": "fresh", "ref": "fresh", "intervention": True},
}


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_str_csv(value: str) -> list[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one string")
    return values


def validate_ages(ages: list[int], *, minimum: int = 0, maximum: int = 11) -> list[int]:
    result = sorted(set(int(age) for age in ages))
    if any(age < minimum or age > maximum for age in result):
        raise ValueError(f"Ages must lie in [{minimum}, {maximum}], got {result}")
    return result


def normalize_generalist_action(
    value: Any,
    *,
    device: torch.device | str | None = None,
    action_chunk_size: int = 8,
    action_dim: int = 7,
) -> torch.Tensor:
    """Normalize one generalist action result to ``[B, F, action_dim]``.

    The online evaluator receives the generalist result as a flat ``[F * D]``
    vector for a single observation.  Some wrappers retain a batch dimension
    (``[B, F * D]``), while cached callers may already provide ``[B, F, D]``.
    Accept those three public forms without silently inventing a batch size for
    any other shape.
    """

    action = torch.as_tensor(value, device=device, dtype=torch.float32)
    if action.ndim == 1:
        action = action.unsqueeze(0)
    if action.ndim == 2:
        flattened_dim = int(action.shape[1])
        if flattened_dim % int(action_chunk_size) != 0:
            raise ValueError(
                f"Generalist action width {flattened_dim} is not divisible by "
                f"chunk size {action_chunk_size}"
            )
        action = action.reshape(action.shape[0], action_chunk_size, -1)
    if action.ndim != 3 or int(action.shape[1]) != int(action_chunk_size):
        raise ValueError(
            "Expected generalist action shape [F*D], [B,F*D], or [B,F,D] "
            f"with F={action_chunk_size}; got {tuple(action.shape)}"
        )
    if int(action.shape[2]) < int(action_dim):
        raise ValueError(
            f"Generalist action has {action.shape[2]} channels; expected at least {action_dim}"
        )
    return action[:, :, :action_dim]


def reference_for_age(slow_action: Any, age: int, empty_ref_after_age: int = 8) -> torch.Tensor:
    """Match DualSystemCalvinEvaluation._build_ref_actions_from exactly.

    ``slow_action`` is the action chunk emitted at the last slow refresh.  The
    valid tail is left-aligned in the specialist's 8-step local condition.
    """

    action = torch.as_tensor(slow_action).detach().clone().to(torch.float32)
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3 or action.shape[1:] != (8, 7):
        raise ValueError(f"Expected slow action shape [B, 8, 7], got {tuple(action.shape)}")
    age = int(age)
    ref = torch.zeros_like(action)
    if age < int(empty_ref_after_age):
        count = max(0, min(8, 8 - age))
        if count:
            ref[:, :count] = action[:, -count:]
    return ref


def tensor_sha256(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().to(torch.float32).cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def observation_sha256(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def rms(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0


def condition_effects(first_actions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return paired two-factor contrasts for one frozen observation."""

    missing = set(CONDITIONS) - set(first_actions)
    if missing:
        raise ValueError(f"Missing conditions: {sorted(missing)}")
    stale_empty = np.asarray(first_actions["stale_hidden_empty_ref"], dtype=np.float64)
    fresh_empty = np.asarray(first_actions["fresh_hidden_empty_ref"], dtype=np.float64)
    stale_fresh = np.asarray(first_actions["stale_hidden_fresh_ref"], dtype=np.float64)
    fresh_fresh = np.asarray(first_actions["fresh_hidden_fresh_ref"], dtype=np.float64)
    return {
        "hidden_effect_at_empty_ref": fresh_empty - stale_empty,
        "hidden_effect_at_fresh_ref": fresh_fresh - stale_fresh,
        "ref_effect_at_stale_hidden": stale_fresh - stale_empty,
        "ref_effect_at_fresh_hidden": fresh_fresh - fresh_empty,
        "hidden_ref_interaction": fresh_fresh - fresh_empty - stale_fresh + stale_empty,
    }


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)
