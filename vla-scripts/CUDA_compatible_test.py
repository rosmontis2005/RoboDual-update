#!/usr/bin/env python3
"""Short, diagnostic CUDA >= 13 compatibility test for the CALVIN evaluator.

The actual rollout is delegated to the stable exp0525 task-age entry point so
this test exercises the same model, environment, and policy code.  This wrapper
fixes the workload at three sequences, streams all profiling output, and writes
a machine-readable compatibility report even when initialization fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STABLE_SCRIPT = SCRIPT_DIR / "evaluate_calvin_task_age_0525.py"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation_results"
REQUIRED_CUDA = (13, 0)
TEST_SEQUENCES = 3
TEST_DATASET_SUBDIR = "calvin_debug_dataset"
TEST_EPISODE_LENGTH = 120
TEST_MAX_SUBTASKS = 1


def version_tuple(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def nvidia_smi_info() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        query = subprocess.run(command, check=True, capture_output=True, text=True)
        header = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True
        ).stdout
        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
        return {
            "available": True,
            "gpus": [line.strip() for line in query.stdout.splitlines() if line.strip()],
            "driver_max_cuda": cuda_match.group(1) if cuda_match else None,
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc), "gpus": [], "driver_max_cuda": None}


def collect_cuda_preflight() -> dict[str, Any]:
    report: dict[str, Any] = {
        "required_cuda": ">=13.0",
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi": nvidia_smi_info(),
        "checks": {},
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        devices = []
        for index in range(torch.cuda.device_count() if cuda_available else 0):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "is_blackwell_class": properties.major >= 10,
                }
            )
        torch_cuda = torch.version.cuda
        compiled_arches = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        driver_cuda = report["nvidia_smi"].get("driver_max_cuda")
        report.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_build": torch_cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_available": cuda_available,
                "devices": devices,
                "torch_compiled_arches": compiled_arches,
            }
        )
        report["checks"] = {
            "cuda_available": cuda_available,
            "torch_built_for_cuda_13_plus": (version_tuple(torch_cuda) or (0, 0)) >= REQUIRED_CUDA,
            "driver_supports_cuda_13_plus": (version_tuple(driver_cuda) or (0, 0)) >= REQUIRED_CUDA,
            # This is target-readiness evidence, not a host requirement: the test
            # is intentionally expected to run on Ada (sm_89).
            "torch_contains_blackwell_cubin": any(
                arch.startswith(("sm_100", "sm_101", "sm_120")) for arch in compiled_arches
            ),
        }
    except Exception as exc:  # Keep a useful JSON artifact for broken migrations.
        report.update({"cuda_available": False, "torch_import_error": repr(exc), "devices": []})
        report["checks"] = {
            "cuda_available": False,
            "torch_built_for_cuda_13_plus": False,
            "driver_supports_cuda_13_plus": False,
            "torch_contains_blackwell_cubin": False,
        }
    report["host_runnable"] = bool(report["checks"]["cuda_available"])
    report["cuda13_test_environment"] = bool(
        report["checks"]["cuda_available"]
        and report["checks"]["torch_built_for_cuda_13_plus"]
        and report["checks"]["driver_supports_cuda_13_plus"]
    )
    report["blackwell_target_prediction"] = {
        "running_on_blackwell": bool(report.get("devices"))
        and all(device["is_blackwell_class"] for device in report["devices"]),
        "native_cubin_present": report["checks"]["torch_contains_blackwell_cubin"],
        "note": (
            "Ada execution can detect CUDA/API and application-level structural failures, "
            "but cannot prove Blackwell kernel-code generation or performance."
        ),
    }
    # Kept as a concise summary for report consumers. It deliberately does not
    # require the host GPU itself to be Blackwell.
    report["compatible"] = report["cuda13_test_environment"]
    return report


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"read_error": str(exc), "path": str(path)}


def summarize_profile(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "records": 0,
        "steps": 0,
        "subtasks": 0,
        "successful_subtasks": 0,
        "model_time_s": 0.0,
        "env_time_s": 0.0,
        "oracle_time_s": 0.0,
        "parse_errors": 0,
    }
    if not path.exists():
        summary["missing"] = True
        return summary
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            summary["parse_errors"] += 1
            continue
        summary["records"] += 1
        if record.get("event") == "step":
            summary["steps"] += 1
            for source, target in (
                ("model_s", "model_time_s"),
                ("env_s", "env_time_s"),
                ("oracle_s", "oracle_time_s"),
            ):
                summary[target] += float(record.get(source, 0.0))
        elif record.get("event") == "subtask_end":
            summary["subtasks"] += 1
            summary["successful_subtasks"] += int(bool(record.get("task_success")))
    for key in ("model_time_s", "env_time_s", "oracle_time_s"):
        summary[key] = round(summary[key], 6)
    return summary


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run exactly three exp0525 task-age sequences with live profiling.",
        epilog="Unknown arguments are forwarded to evaluate_calvin_task_age_0525.py.",
    )
    parser.add_argument("--analysis_json", type=Path, default=DEFAULT_RESULTS_DIR / "CUDA_compatible_analysis.json")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument(
        "--full_precision",
        action="store_true",
        help=(
            "Do not enable the Ada-friendly 4-bit model-loading default. "
            "Use this only when host and GPU memory are sufficient."
        ),
    )
    parser.add_argument(
        "--allow_unsupported_cuda",
        action="store_true",
        help="Run rollouts even when the active PyTorch environment is not CUDA >= 13.",
    )
    args, stable_args = parser.parse_known_args()

    started = time.time()
    calvin_root = Path(os.environ.get("CALVIN_ROOT", REPO_ROOT.parent / "calvin")).expanduser().resolve()
    dataset_validation = calvin_root / "dataset" / TEST_DATASET_SUBDIR / "validation"
    dataset_config = dataset_validation / ".hydra" / "merged_config.yaml"
    report: dict[str, Any] = {
        "schema_version": 1,
        "test": "exp0525task_age_cuda_compatibility_smoke_test",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stable_entrypoint": str(STABLE_SCRIPT),
        "requested_sequences": TEST_SEQUENCES,
        "test_dataset_subdir": TEST_DATASET_SUBDIR,
        "episode_length": TEST_EPISODE_LENGTH,
        "max_subtasks": TEST_MAX_SUBTASKS,
        "dataset_preflight": {
            "calvin_root": str(calvin_root),
            "validation_path": str(dataset_validation),
            "merged_config": str(dataset_config),
            "available": dataset_config.is_file(),
        },
        "preflight": collect_cuda_preflight(),
        "status": "preflight_complete",
    }
    atomic_write_json(args.analysis_json, report)
    print("[cuda-test] " + json.dumps(report["preflight"], ensure_ascii=False), flush=True)

    if args.preflight_only:
        ready = report["preflight"]["compatible"] and report["dataset_preflight"]["available"]
        report["status"] = "cuda13_environment_ready" if ready else "cuda13_environment_not_ready"
        report["elapsed_s"] = round(time.time() - started, 3)
        atomic_write_json(args.analysis_json, report)
        return 0 if ready else 2
    if not report["preflight"]["compatible"] and not args.allow_unsupported_cuda:
        report["status"] = "blocked_by_cuda_preflight"
        report["recommendation"] = (
            "Run this script with the Python executable from the CUDA 13 test environment; "
            "use --allow_unsupported_cuda only for a non-CUDA-13 diagnostic run."
        )
        report["elapsed_s"] = round(time.time() - started, 3)
        atomic_write_json(args.analysis_json, report)
        print(f"[cuda-test] incompatible platform; report written to {args.analysis_json}", flush=True)
        return 2
    if not report["dataset_preflight"]["available"]:
        report["status"] = "blocked_by_missing_calvin_dataset"
        report["recommendation"] = f"Expected CALVIN config: {dataset_config}"
        report["elapsed_s"] = round(time.time() - started, 3)
        atomic_write_json(args.analysis_json, report)
        print(f"[cuda-test] missing dataset config: {dataset_config}", flush=True)
        return 2

    result_path = DEFAULT_RESULTS_DIR / "result_rank0.json"
    profile_path = DEFAULT_RESULTS_DIR / "specialist_profile_rank0.jsonl"
    success_rate_path = DEFAULT_RESULTS_DIR / "success_rate_rank0.txt"
    console_log_path = DEFAULT_RESULTS_DIR / "CUDA_compatible_console.log"
    for stale_path in (result_path, profile_path, success_rate_path):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass

    command = [sys.executable, "-u", str(STABLE_SCRIPT), *stable_args]
    # Appending these makes the short-test invariants win over forwarded options.
    command += [
        "--num_sequences",
        str(TEST_SEQUENCES),
        "--profile_steps",
        "--profile_init",
        "--low_cpu_mem_usage",
        "--dataset_subdir",
        TEST_DATASET_SUBDIR,
        "--ep_len",
        str(TEST_EPISODE_LENGTH),
        "--max_subtasks",
        str(TEST_MAX_SUBTASKS),
    ]
    # A 7B-class BF16 checkpoint can exceed the 15 GiB host RAM available on
    # the Ada test machine before it is ever copied to VRAM. Quantized loading
    # also exercises bitsandbytes under CUDA 13, which is useful migration
    # coverage. Larger Blackwell hosts can request the original BF16 path.
    if not args.full_precision:
        command.append("--load_in_4bit")
    report["command"] = command
    report["status"] = "running"
    atomic_write_json(args.analysis_json, report)
    print(f"[cuda-test] launching {TEST_SEQUENCES} sequences", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        console_log_path.parent.mkdir(parents=True, exist_ok=True)
        with console_log_path.open("w") as console_log:
            process = subprocess.Popen(
                command,
                cwd=SCRIPT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                console_log.write(line)
                console_log.flush()
            return_code = process.wait()
        report["return_code"] = return_code
        if return_code < 0:
            signal_number = -return_code
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = f"SIGNAL_{signal_number}"
            report["termination"] = {
                "kind": "signal",
                "signal_number": signal_number,
                "signal_name": signal_name,
                "likely_cause": (
                    "The OS or job supervisor killed the process; SIGKILL during checkpoint "
                    "loading commonly indicates host-memory OOM."
                    if signal_number == signal.SIGKILL
                    else None
                ),
            }
        report["evaluation_result"] = read_json(result_path) if result_path.exists() else None
        report["profile_summary"] = summarize_profile(profile_path)
        report["artifacts"] = {
            "evaluation_result": str(result_path),
            "live_profile_jsonl": str(profile_path),
            "success_rate": str(success_rate_path),
            "console_log": str(console_log_path),
            "analysis_json": str(args.analysis_json.resolve()),
        }
        usable = (
            return_code == 0
            and result_path.exists()
            and report["profile_summary"]["steps"] > 0
            and report["profile_summary"]["parse_errors"] == 0
        )
        report["checks"] = {
            "stable_script_exit_success": return_code == 0,
            "result_json_generated": result_path.exists(),
            "profile_jsonl_generated": profile_path.exists(),
            "profile_contains_steps": report["profile_summary"]["steps"] > 0,
            "profile_json_valid": report["profile_summary"]["parse_errors"] == 0,
        }
        report["usable"] = usable
        report["status"] = "passed" if usable and report["preflight"]["compatible"] else "failed"
    except Exception as exc:
        report.update(
            {
                "status": "runner_exception",
                "usable": False,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_s"] = round(time.time() - started, 3)
    atomic_write_json(args.analysis_json, report)
    print(f"[cuda-test] status={report['status']} analysis={args.analysis_json}", flush=True)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
