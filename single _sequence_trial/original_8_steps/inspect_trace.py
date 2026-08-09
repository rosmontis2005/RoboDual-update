#!/usr/bin/env python3
"""Inspect a collected step without loading models or the CALVIN environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trace_capture import tensor_descriptor


def load_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step_file", type=Path)
    parser.add_argument("--show_values", action="store_true")
    args = parser.parse_args()
    payload = load_payload(args.step_file)
    if args.show_values:
        print(payload)
    else:
        print(json.dumps(tensor_descriptor(payload), indent=2))


if __name__ == "__main__":
    main()
