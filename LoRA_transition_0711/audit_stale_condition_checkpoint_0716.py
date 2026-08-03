#!/usr/bin/env python3
"""Audit that a merged stale-condition EMA changes only explicitly gated weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

TARGET_PROFILES = {
    "stale_condition": (
        "model.context_adapter",
        "model.blocks.4.cross_attn.attn.l_proj",
        "model.blocks.4.cross_attn.attn.values_l_proj",
        "model.blocks.5.cross_attn.attn.l_proj",
        "model.blocks.5.cross_attn.attn.values_l_proj",
    ),
    "stale_action_condition": (
        "model.x_embedder",
        "model.context_adapter",
        "model.blocks.4.cross_attn.attn.l_proj",
        "model.blocks.4.cross_attn.attn.values_l_proj",
        "model.blocks.5.cross_attn.attn.l_proj",
        "model.blocks.5.cross_attn.attn.values_l_proj",
    ),
}


def main(args: argparse.Namespace) -> None:
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    candidate_path = Path(args.candidate_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    expected = {
        f"ema_model.{name}.weight" for name in TARGET_PROFILES[args.target_profile]
    }
    common_ema = {
        key for key in base.keys() & candidate.keys()
        if str(key).startswith("ema_model.") and not str(key).endswith("._dummy_variable")
    }
    changed = {
        key for key in common_ema
        if not torch.equal(base[key], candidate[key])
    }
    if changed != expected:
        raise RuntimeError(
            "unexpected deployed EMA deltas: "
            f"missing={sorted(expected - changed)}, unexpected={sorted(changed - expected)}"
        )

    restored_mismatches = []
    for key in common_ema:
        restored = base[key] if key in expected else candidate[key]
        if not torch.equal(restored, base[key]):
            restored_mismatches.append(key)
    if restored_mismatches:
        raise RuntimeError(f"gate-off state does not restore base: {sorted(restored_mismatches)}")

    history_keys = sorted(
        key for key in candidate if str(key).startswith("ema_model.model.history_adapter.")
    )
    output_keys = [key for key in history_keys if ".net.2." in key]
    if len(output_keys) != 2 or any(torch.count_nonzero(candidate[key]).item() for key in output_keys):
        raise RuntimeError("history adapter output projection is not exact zero")

    result = {
        "status": "pass",
        "base_checkpoint": base_path.as_posix(),
        "candidate_checkpoint": candidate_path.as_posix(),
        "deployed_ema_common_tensors": len(common_ema),
        "changed_deployed_ema_tensors": sorted(changed),
        "gate_off_restored_mismatches": restored_mismatches,
        "history_adapter_keys": len(history_keys),
        "history_output_zero_keys": output_keys,
        "online_model_note": "not audited; CALVIN evaluation executes ema_fast_system.ema_model",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--target_profile", choices=sorted(TARGET_PROFILES), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
