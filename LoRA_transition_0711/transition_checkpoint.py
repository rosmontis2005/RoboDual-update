"""Strict loading for merged transition-history specialist checkpoints."""

from pathlib import Path

import torch


LEGACY_EMA_DUMMY_KEYS = {
    "online_model._dummy_variable",
    "ema_model._dummy_variable",
}

TRANSITION_ABLATION_MODES = ("base", "history_only", "lora_only", "full")


def load_transition_specialist(dual_system, checkpoint_path):
    """Load a merged checkpoint without silently dropping history weights."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Transition specialist checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Expected a non-empty state dict in {path}")

    system = getattr(dual_system, "module", dual_system)
    ema_wrapper = system.ema_fast_system
    is_ema_checkpoint = any(str(key).startswith("ema_model.") for key in state)
    if is_ema_checkpoint:
        required = {
            f"ema_model.{key}"
            for key in ema_wrapper.ema_model.state_dict()
            if key.startswith("model.history_adapter.")
        }
        missing_history = sorted(required - set(state))
        if missing_history:
            raise RuntimeError(f"Checkpoint is missing EMA history adapter keys: {missing_history}")
        runtime_keys = set(ema_wrapper.state_dict())
        ignored_legacy_keys = sorted(
            key for key in LEGACY_EMA_DUMMY_KEYS if key in state and key not in runtime_keys
        )
        load_state = type(state)(
            (key, value) for key, value in state.items() if key not in ignored_legacy_keys
        )
        ema_wrapper.load_state_dict(load_state, strict=True)
        loaded_target = ema_wrapper.ema_model
        checkpoint_format = "ema_wrapper"
    else:
        required = {
            key for key in system.fast_system.state_dict()
            if key.startswith("model.history_adapter.")
        }
        missing_history = sorted(required - set(state))
        if missing_history:
            raise RuntimeError(f"Checkpoint is missing policy history adapter keys: {missing_history}")
        system.fast_system.load_state_dict(state, strict=True)
        ema_wrapper.ema_model.load_state_dict(state, strict=True)
        loaded_target = ema_wrapper.ema_model
        checkpoint_format = "raw_policy"

    loaded_history = loaded_target.model.history_adapter.state_dict()
    prefix = "ema_model.model.history_adapter." if is_ema_checkpoint else "model.history_adapter."
    for key, value in loaded_history.items():
        expected = state[f"{prefix}{key}"].to(device=value.device, dtype=value.dtype)
        if not torch.equal(value, expected):
            raise RuntimeError(f"Loaded history adapter tensor differs from checkpoint: {key}")
    return {
        "path": path.as_posix(),
        "format": checkpoint_format,
        "history_keys": len(loaded_history),
        "ignored_legacy_keys": ignored_legacy_keys if is_ema_checkpoint else [],
    }


def _zero_history_residual(policy):
    adapter = policy.model.history_adapter
    output_layer = adapter.net[-1]
    output_layer.weight.data.zero_()
    output_layer.bias.data.zero_()


def apply_transition_ablation(dual_system, base_checkpoint_path, mode):
    """Keep or remove the trained backbone/history components for evaluation.

    The full transition checkpoint must be loaded first. Restoring the original
    EMA wrapper leaves trained history keys untouched because the base checkpoint
    predates the adapter; the residual is then explicitly zeroed when disabled.
    """

    if mode not in TRANSITION_ABLATION_MODES:
        raise ValueError(f"Unsupported transition ablation mode: {mode}")
    system = getattr(dual_system, "module", dual_system)
    ema_wrapper = system.ema_fast_system
    restored_base = mode in {"base", "history_only"}
    if restored_base:
        path = Path(base_checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Base specialist checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Expected a non-empty base state dict in {path}")
        runtime_keys = set(ema_wrapper.state_dict())
        load_state = type(state)(
            (key, value)
            for key, value in state.items()
            if key in runtime_keys and key not in LEGACY_EMA_DUMMY_KEYS
        )
        incompatible = ema_wrapper.load_state_dict(load_state, strict=False)
        allowed_missing = {
            key for key in runtime_keys if ".model.history_adapter." in key
        }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != allowed_missing or unexpected:
            raise RuntimeError(
                "Base checkpoint restoration mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

    history_enabled = mode in {"history_only", "full"}
    if not history_enabled:
        _zero_history_residual(system.fast_system)
        _zero_history_residual(ema_wrapper.online_model)
        _zero_history_residual(ema_wrapper.ema_model)

    ema_history = ema_wrapper.ema_model.model.history_adapter
    output_norm = float(
        torch.linalg.vector_norm(ema_history.net[-1].weight.detach().float()).cpu()
    )
    if not history_enabled and output_norm != 0.0:
        raise RuntimeError("Disabled history adapter still has a non-zero output projection")
    return {
        "mode": mode,
        "base_checkpoint": Path(base_checkpoint_path).expanduser().resolve().as_posix(),
        "base_backbone_restored": restored_base,
        "trained_history_enabled": history_enabled,
        "history_output_weight_norm": output_norm,
    }
