#!/usr/bin/env python3
"""CPU-only contract checks for the 0809 mechanism and age diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mechanism_common import (
    CONDITIONS,
    condition_effects,
    normalize_generalist_action,
    reference_for_age,
)
from run_offline_age_curve import DEFAULT_DATA, OfflineTransitionValidation


def main() -> None:
    flat_action = torch.arange(64, dtype=torch.float32)
    expected_action = flat_action.reshape(1, 8, 8)[:, :, :7]
    action_forms = (
        flat_action,
        flat_action.unsqueeze(0),
        flat_action.reshape(1, 8, 8),
    )
    for action_form in action_forms:
        normalized = normalize_generalist_action(action_form)
        if normalized.shape != (1, 8, 7) or not torch.equal(normalized, expected_action):
            raise AssertionError(
                f"Generalist action normalization failed for shape {tuple(action_form.shape)}"
            )
    try:
        normalize_generalist_action(torch.zeros(63))
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed generalist action width was not rejected")

    action = torch.arange(56, dtype=torch.float32).reshape(1, 8, 7)
    expected_counts = {age: (8 - age if age < 8 else 0) for age in range(12)}
    for age, expected_count in expected_counts.items():
        ref = reference_for_age(action, age)
        nonzero = int((ref.abs().sum(dim=-1) > 0).sum().item())
        if nonzero != expected_count:
            raise AssertionError(f"age {age}: expected {expected_count} ref actions, got {nonzero}")
        if age < 8 and expected_count:
            if not torch.equal(ref[0, :expected_count], action[0, -expected_count:]):
                raise AssertionError(f"age {age}: reference tail alignment differs from P12 contract")
        if age >= 8 and float(ref.abs().sum()) != 0.0:
            raise AssertionError(f"age {age}: empty ref is not zero")

    vectors = {
        condition: np.arange(7, dtype=np.float64) + index
        for index, condition in enumerate(CONDITIONS)
    }
    effects = condition_effects(vectors)
    if not np.allclose(effects["hidden_effect_at_empty_ref"], np.ones(7)):
        raise AssertionError("Hidden contrast formula is wrong")
    if not np.allclose(effects["ref_effect_at_stale_hidden"], np.full(7, 2.0)):
        raise AssertionError("Reference contrast formula is wrong")
    if not np.allclose(effects["hidden_ref_interaction"], np.zeros(7)):
        raise AssertionError("Interaction formula is wrong")

    dataset = OfflineTransitionValidation(DEFAULT_DATA, "validation", 0, 11)
    counts = dataset.age_counts()
    missing = [age for age in range(12) if counts.get(age, 0) == 0]
    if missing:
        raise AssertionError(f"Offline validation has no samples for ages {missing}")
    selected = dataset.select(max_per_age=2, seed=809080)
    selected_counts = {}
    for row in selected:
        selected_counts[str(row["slow_age"])] = selected_counts.get(str(row["slow_age"]), 0) + 1
    print(
        json.dumps(
            {
                "generalist_action_shape_contract": "passed",
                "reference_contract": "passed",
                "condition_effect_contract": "passed",
                "offline_validation_age_counts": counts,
                "offline_validation_two_per_age_counts": selected_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
