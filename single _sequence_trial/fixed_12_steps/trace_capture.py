#!/usr/bin/env python3
"""Expose the shared trace implementation used by both paired collectors."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SHARED_PATH = Path(__file__).resolve().parents[1] / "original_8_steps" / "trace_capture.py"
SPEC = importlib.util.spec_from_file_location("robodual_shared_trace_capture", SHARED_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load shared trace implementation: {SHARED_PATH}")
SHARED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARED)

cpu_clone = SHARED.cpu_clone
OnlineTraceCapture = SHARED.OnlineTraceCapture
physics_state = SHARED.physics_state
rng_snapshot = SHARED.rng_snapshot
sha256_file = SHARED.sha256_file
tensor_descriptor = SHARED.tensor_descriptor
TraceWriter = SHARED.TraceWriter

__all__ = [
    "cpu_clone",
    "OnlineTraceCapture",
    "physics_state",
    "rng_snapshot",
    "sha256_file",
    "tensor_descriptor",
    "TraceWriter",
]
