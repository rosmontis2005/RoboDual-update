#!/usr/bin/env python3
"""Package a saved narrow transition adapter as an EMA-compatible checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch


EXPECTED_MODULES = (
    "model.blocks.4.attn_temporal.proj",
    "model.blocks.5.attn_temporal.proj",
)


def build_checkpoint(base: dict, payload: dict) -> tuple[OrderedDict, dict]:
    if payload.get("format") != "robodual_transition_history_lora_v1":
        raise ValueError("unsupported transition adapter format")
    metadata = payload.get("metadata", {})
    args = metadata.get("args", {})
    rank = int(args["lora_rank"])
    alpha = float(args["lora_alpha"])
    lora_state = payload["lora_state"]
    expected_keys = {
        f"{module}.{suffix}"
        for module in EXPECTED_MODULES
        for suffix in ("lora_A", "lora_B")
    }
    if set(lora_state) != expected_keys:
        raise ValueError(f"unexpected LoRA keys: {sorted(set(lora_state) ^ expected_keys)}")

    output = OrderedDict((key, value) for key, value in base.items())
    delta_stats = {}
    for module in EXPECTED_MODULES:
        lora_a = lora_state[f"{module}.lora_A"].float()
        lora_b = lora_state[f"{module}.lora_B"].float()
        if lora_a.shape[0] != rank or lora_b.shape[1] != rank:
            raise ValueError(f"rank metadata does not match {module}")
        delta = lora_b @ lora_a * (alpha / rank)
        if not torch.count_nonzero(delta):
            raise RuntimeError(f"adapter has zero delta: {module}")
        for prefix in ("ema_model", "online_model"):
            key = f"{prefix}.{module}.weight"
            if key not in base or base[key].shape != delta.shape:
                raise KeyError(f"base checkpoint target mismatch: {key}")
            output[key] = (base[key].float() + delta).to(dtype=base[key].dtype)
        delta_stats[module] = {
            "l2": float(torch.linalg.vector_norm(delta)),
            "max_abs": float(delta.abs().max()),
        }

    history_state = payload["history_adapter_state"]
    for key, value in history_state.items():
        for prefix in ("ema_model", "online_model"):
            output[f"{prefix}.model.history_adapter.{key}"] = value
        if key in ("net.2.weight", "net.2.bias") and torch.count_nonzero(value):
            raise RuntimeError(f"history output must remain exactly zero: {key}")

    changed = sorted(
        key for key in base
        if torch.is_tensor(base[key]) and not torch.equal(base[key], output[key])
    )
    expected_changed = sorted(
        f"{prefix}.{module}.weight"
        for prefix in ("ema_model", "online_model")
        for module in EXPECTED_MODULES
    )
    if changed != expected_changed:
        raise RuntimeError(f"unexpected changed base tensors: {changed}")
    return output, {
        "adapter_step": metadata.get("step"),
        "rank": rank,
        "alpha": alpha,
        "changed_base_keys": changed,
        "history_output_zero": True,
        "delta_stats": delta_stats,
    }


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    adapter_path = Path(args.adapter_checkpoint).expanduser().resolve()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    output, summary = build_checkpoint(base, payload)
    checkpoint_path = output_dir / "specialist_transition_lora_adapter_ema.pt"
    torch.save(output, checkpoint_path)
    summary.update({
        "base_checkpoint": base_path.as_posix(),
        "adapter_checkpoint": adapter_path.as_posix(),
        "output_checkpoint": checkpoint_path.as_posix(),
    })
    (output_dir / "finalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
