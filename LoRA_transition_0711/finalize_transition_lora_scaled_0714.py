#!/usr/bin/env python3
"""Build a scaled, EMA/online-safe deployment checkpoint from a merged LoRA."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch


TARGET_SUFFIXES = (
    "model.blocks.4.attn_temporal.proj.weight",
    "model.blocks.5.attn_temporal.proj.weight",
)
HISTORY_TOKEN = ".model.history_adapter."


def build_scaled_checkpoint(base: dict, merged: dict, scale: float) -> tuple[OrderedDict, dict]:
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    if set(base) - set(merged):
        raise ValueError(f"merged checkpoint is missing base keys: {sorted(set(base) - set(merged))[:5]}")

    output = OrderedDict((key, value) for key, value in base.items())
    delta_stats = {}
    for suffix in TARGET_SUFFIXES:
        ema_key = f"ema_model.{suffix}"
        if ema_key not in base or ema_key not in merged:
            raise KeyError(f"missing target tensor: {ema_key}")
        ema_delta = merged[ema_key].float() - base[ema_key].float()
        if not torch.count_nonzero(ema_delta):
            raise RuntimeError(f"merged checkpoint has a zero LoRA delta: {ema_key}")
        for prefix in ("ema_model", "online_model"):
            key = f"{prefix}.{suffix}"
            value = base[key].float() + scale * ema_delta
            output[key] = value.to(dtype=base[key].dtype)
        delta_stats[suffix] = {
            "source_l2": float(torch.linalg.vector_norm(ema_delta)),
            "scaled_l2": float(torch.linalg.vector_norm(output[ema_key].float() - base[ema_key].float())),
        }

    history_keys = sorted(key for key in merged if HISTORY_TOKEN in key)
    if not history_keys:
        raise RuntimeError("merged checkpoint has no history-adapter compatibility keys")
    for key in history_keys:
        output[key] = merged[key]
    for key in history_keys:
        if key.endswith(("net.2.weight", "net.2.bias")) and torch.count_nonzero(output[key]):
            raise RuntimeError(f"history output must remain exactly zero: {key}")

    changed = sorted(
        key for key in base
        if torch.is_tensor(base[key]) and not torch.equal(base[key], output[key])
    )
    expected_changed = sorted(
        f"{prefix}.{suffix}"
        for prefix in ("ema_model", "online_model")
        for suffix in TARGET_SUFFIXES
    )
    if changed != expected_changed:
        raise RuntimeError(f"unexpected changed base tensors: {changed}")
    return output, {
        "scale": scale,
        "changed_base_keys": changed,
        "added_history_keys": history_keys,
        "delta_stats": delta_stats,
    }


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    merged_path = Path(args.merged_checkpoint).expanduser().resolve()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    merged = torch.load(merged_path, map_location="cpu", weights_only=False)
    if not isinstance(base, dict) or not isinstance(merged, dict):
        raise TypeError("both checkpoints must be state dictionaries")
    output, summary = build_scaled_checkpoint(base, merged, args.scale)
    checkpoint_path = output_dir / "specialist_transition_lora_scaled_ema.pt"
    torch.save(output, checkpoint_path)
    summary.update({
        "base_checkpoint": base_path.as_posix(),
        "source_merged_checkpoint": merged_path.as_posix(),
        "output_checkpoint": checkpoint_path.as_posix(),
        "checkpoint_keys": len(output),
    })
    (output_dir / "finalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--merged_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
