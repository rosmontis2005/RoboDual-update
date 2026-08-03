"""Run four transition-LoRA ablations serially without overwriting results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "evaluation_results" / "exp0713_LoRA_v_check"
EVALUATOR = REPO_ROOT / "vla-scripts" / "evaluate_calvin_task_age_transition_lora_ablation_0713.py"
ANALYZER = REPO_ROOT / "LoRA_transition_0711" / "analyze_ablation_check_0713.py"
PYTHON = Path("/home/rosmontis/miniconda3/envs/dualsys_env/bin/python")
GENERALIST = REPO_ROOT.parent / "models" / "generalist"
SPECIALIST = REPO_ROOT.parent / "models" / "specialist" / "Specialist+Depth+Gripper.pt"
TRANSITION_CHECKPOINT = (
    REPO_ROOT
    / "LoRA_transition_0711/lora_runs/transition_history_lora_v2_repaired"
    / "specialist_transition_lora_merged_ema.pt"
)
CALVIN_ROOT = REPO_ROOT.parent / "calvin"
MODES = ("base", "history_only", "lora_only", "full")
# Selected only from task composition. Together these 16 sequences contain all
# 34 benchmark tasks and four chains containing stack_block.
SEQUENCE_INDICES = (3, 11, 20, 21, 28, 35, 36, 53, 59, 65, 75, 83, 86, 89, 91, 95)


def read_run_config(profile_path: Path) -> dict:
    with profile_path.open() as handle:
        first = json.loads(next(handle))
    if first.get("event") != "run_config":
        raise RuntimeError(f"First profile record is not run_config: {profile_path}")
    return first


def validate_result(mode: str) -> dict:
    result_dir = OUTPUT_ROOT / mode
    required = {
        "result": result_dir / "result_rank0.json",
        "success": result_dir / "success_rate_rank0.txt",
        "profile": result_dir / "specialist_profile_rank0.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"{mode} result is incomplete; missing/non-empty check failed: {missing}")
    result = json.loads(required["result"].read_text())
    if len(result) != 1:
        raise RuntimeError(f"{mode} result JSON must contain exactly one evaluation payload")
    result_payload = next(iter(result.values()))
    lines = [line for line in required["success"].read_text().splitlines() if line.strip()]
    if len(lines) != len(SEQUENCE_INDICES):
        raise RuntimeError(f"{mode} has {len(lines)} progress rows, expected {len(SEQUENCE_INDICES)}")
    config = read_run_config(required["profile"])
    if config.get("sequence_indices") != list(SEQUENCE_INDICES):
        raise RuntimeError(f"{mode} sequence list differs from the fixed manifest")
    ablation = config.get("transition_ablation", {})
    if ablation.get("mode") != mode:
        raise RuntimeError(f"{mode} profile reports ablation {ablation.get('mode')}")
    return {
        "mode": mode,
        "avg_seq_len": result_payload["avg_seq_len"],
        "chain_sr": result_payload["chain_sr"],
        "profile_bytes": required["profile"].stat().st_size,
    }


def build_command(mode: str) -> list[str]:
    return [
        PYTHON.as_posix(),
        EVALUATOR.as_posix(),
        "--generalist_path", GENERALIST.as_posix(),
        "--specialist_path", SPECIALIST.as_posix(),
        "--transition_checkpoint", TRANSITION_CHECKPOINT.as_posix(),
        "--ablation_mode", mode,
        "--save_dir", (OUTPUT_ROOT / mode).as_posix(),
        "--sequence_indices", ",".join(map(str, SEQUENCE_INDICES)),
        "--dataset_subdir", "calvin_debug_dataset",
        "--log_dir", mode,
        "--slow_call_strategy", "task_age",
        "--profile_sample_var_ages", "",
        "--load_in_4bit",
        "--low_cpu_mem_usage",
    ]


def write_manifest() -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": "task-composition-only set cover; all 34 tasks; four stack_block chains",
        "sequence_indices": list(SEQUENCE_INDICES),
        "modes_in_order": list(MODES),
        "common_seed": 42,
        "profile_sample_var_ages": "",
        "dataset_subdir": "calvin_debug_dataset",
        "transition_checkpoint": TRANSITION_CHECKPOINT.as_posix(),
        "base_specialist": SPECIALIST.as_posix(),
    }
    path = OUTPUT_ROOT / "experiment_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("sequence_indices", "modes_in_order", "common_seed", "transition_checkpoint"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"Existing experiment manifest differs at {key}")
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    for path in (PYTHON, EVALUATOR, ANALYZER, GENERALIST, SPECIALIST, TRANSITION_CHECKPOINT, CALVIN_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(exist_ok=True)
    write_manifest()

    previous = None
    for mode in MODES:
        if previous is not None:
            validate_result(previous)
        result_dir = OUTPUT_ROOT / mode
        if result_dir.exists() and any(result_dir.iterdir()):
            summary = validate_result(mode)
            completion_path = result_dir / "completion.json"
            if not completion_path.exists():
                completion_path.write_text(
                    json.dumps({**summary, "validated_at": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n"
                )
            print(f"[resume] validated completed {mode}: {summary}", flush=True)
            previous = mode
            continue
        command = build_command(mode)
        log_path = OUTPUT_ROOT / "logs" / f"{mode}.log"
        command_path = OUTPUT_ROOT / "logs" / f"{mode}.command.json"
        command_path.write_text(json.dumps(command, indent=2) + "\n")
        env = os.environ.copy()
        env.update({
            "CALVIN_ROOT": CALVIN_ROOT.as_posix(),
            "MPLCONFIGDIR": "/tmp/robodual-matplotlib",
            "TOKENIZERS_PARALLELISM": "false",
        })
        print(f"[start] {mode}: {' '.join(command)}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{mode} failed with exit code {completed.returncode}; inspect {log_path}")
        summary = validate_result(mode)
        (result_dir / "completion.json").write_text(
            json.dumps({**summary, "completed_at": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n"
        )
        print(f"[complete] {mode}: {summary}", flush=True)
        previous = mode

    subprocess.run([PYTHON.as_posix(), ANALYZER.as_posix(), OUTPUT_ROOT.as_posix()], check=True)
    print(f"[report] {OUTPUT_ROOT / 'ablation_report.md'}", flush=True)


if __name__ == "__main__":
    main()
