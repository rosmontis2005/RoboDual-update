#!/usr/bin/env python3
"""Targeted CALVIN evaluation for specialist LoRA checkpoints.

This entrypoint focuses on the empty-reference-sensitive tasks used by the
LoRA trial. It preserves prefix subtasks before each target task, stops at the
target, and reports target-task success separately from prefix failures.

Typical usage:

  python vla-scripts/evaluate_lora_targeted.py \
    --dataset_subdir calvin_debug_dataset \
    --specialist_paths /path/to/base.pt,/path/to/lora_merged_ema.pt \
    --specialist_names base,lora \
    --load_in_4bit --low_cpu_mem_usage
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
DEFAULT_CALVIN_ROOT = REPO_ROOT.parent / "calvin"
CALVIN_ROOT_PATH = Path(os.environ.get("CALVIN_ROOT", DEFAULT_CALVIN_ROOT)).expanduser().resolve()
os.environ.setdefault("CALVIN_ROOT", CALVIN_ROOT_PATH.as_posix())

for dependency_path in (
    THIS_FILE.parent,
    REPO_ROOT.parent,
    CALVIN_ROOT_PATH / "calvin_models",
    CALVIN_ROOT_PATH / "calvin_env",
    CALVIN_ROOT_PATH / "calvin_env" / "tacto",
):
    path_str = dependency_path.as_posix()
    if dependency_path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: E402
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition  # noqa: E402
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402
from evaluate_calvin_task_age_0525 import (  # noqa: E402
    DEFAULT_GENERALIST_PATH,
    DEFAULT_SPECIALIST_PATH,
    DEFAULT_TASK_AGE_GROUP_A,
    DEFAULT_TASK_AGE_GROUP_B,
    DEFAULT_TASK_AGE_GROUP_C,
    DEFAULT_TASK_AGE_GROUP_D,
    TaskAgeDualSystemEvaluation,
    build_task_age_config,
    emit_profile_record,
    make_env,
    rollout,
)
from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig  # noqa: E402


DEFAULT_TARGET_CASE_QUOTAS = {
    "place_in_slider": 10,
    "lift_blue_block_slider": 10,
    "stack_block": 10,
    "rotate_red_block_right": 5,
    "push_pink_block_right": 5,
}
DEFAULT_TARGET_CASE_QUOTAS_ARG = ",".join(f"{task}:{count}" for task, count in DEFAULT_TARGET_CASE_QUOTAS.items())


@dataclass(frozen=True)
class TargetCase:
    case_id: int
    sequence_i: int
    target_task: str
    target_i: int
    initial_state: object
    eval_sequence: Sequence[str]


class EmaOnlyWrapper(torch.nn.Module):
    """Tiny wrapper matching the evaluation API without keeping an online copy."""

    def __init__(self, ema_model):
        super().__init__()
        self.ema_model = ema_model


class EvalOnlyDualSystem(torch.nn.Module):
    """Minimal dual-system holder used only by DualSystemCalvinEvaluation."""

    def __init__(self, slow_system, fast_system):
        super().__init__()
        self.slow_system = slow_system
        self.ema_fast_system = EmaOnlyWrapper(fast_system)


def parse_csv(value):
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_task_quotas(value):
    quotas = {}
    for item in parse_csv(value):
        if ":" in item:
            task, count = item.split(":", 1)
        elif "=" in item:
            task, count = item.split("=", 1)
        else:
            raise ValueError(f"Quota item must look like task:count, got {item!r}")
        task = task.strip()
        if not task:
            raise ValueError(f"Empty task name in quota item {item!r}")
        quotas[task] = int(count)
    return {task: count for task, count in quotas.items() if count > 0}


def parse_specialist_specs(paths_arg, names_arg):
    paths = parse_csv(paths_arg)
    names = parse_csv(names_arg)
    if not paths:
        raise ValueError("--specialist_paths must contain at least one checkpoint")
    if names and len(names) != len(paths):
        raise ValueError("--specialist_names length must match --specialist_paths")
    if not names:
        names = [Path(path).stem for path in paths]
    return list(zip(names, paths))


def build_target_cases(args):
    quotas = parse_task_quotas(args.target_case_quotas)
    if not quotas:
        raise ValueError("No target case quotas selected.")

    counts = Counter()
    cases: List[TargetCase] = []
    for sequence_i, (initial_state, eval_sequence) in enumerate(get_sequences(args.search_num_sequences)):
        for target_i, task in enumerate(eval_sequence):
            if task not in quotas or counts[task] >= quotas[task]:
                continue
            cases.append(
                TargetCase(
                    case_id=len(cases),
                    sequence_i=sequence_i,
                    target_task=task,
                    target_i=target_i,
                    initial_state=initial_state,
                    eval_sequence=tuple(eval_sequence),
                )
            )
            counts[task] += 1
        if all(counts[task] >= quotas[task] for task in quotas):
            break

    missing = {task: quotas[task] - counts[task] for task in quotas if counts[task] < quotas[task]}
    if missing:
        print(f"[targeted-eval] warning: could not fill all target cases from search range: {missing}", flush=True)
    return cases, dict(sorted(quotas.items()))


def load_generalist(args, device):
    quantization_config = None
    model_dtype = torch.bfloat16
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model_dtype = torch.float16
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_dtype = torch.float16

    model_kwargs = {
        "torch_dtype": model_dtype,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "trust_remote_code": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation

    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    slow_model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **model_kwargs)
    slow_model.eval()
    slow_model.requires_grad_(False)
    if quantization_config is None and args.device_map == "none":
        slow_model = slow_model.to(device)
    return slow_model, processor


def build_specialist_policy(args, device):
    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    policy = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": [7]}},
        noise_scheduler=scheduler,
        n_action_steps=8,
        num_inference_steps=args.fast_num_inference_steps,
        vision_encoder="DINO",
        with_depth=args.with_depth,
        progressive_noise=False,
        with_gripper=args.with_gripper,
        with_tactile=args.with_tactile,
        cond_drop_chance=0.1 if args.with_cfg else 0.0,
    )
    return policy.eval().to(device)


def load_specialist_into_dual_system(dual_system, specialist_path):
    state = torch.load(Path(specialist_path).expanduser().as_posix(), map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported specialist checkpoint type: {type(state)}")
    has_ema_keys = any(str(key).startswith("ema_model.") for key in state)
    has_online_keys = any(str(key).startswith("online_model.") for key in state)
    if has_ema_keys or has_online_keys:
        missing, unexpected = dual_system.ema_fast_system.load_state_dict(state, strict=False)
        return {
            "checkpoint_format": "ema_wrapper",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    missing, unexpected = dual_system.ema_fast_system.ema_model.load_state_dict(state, strict=False)
    return {
        "checkpoint_format": "raw_policy",
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def build_eval_wrapper(args, dual_system, processor, action_tokenizer):
    slow_trigger_policy = "fixed_mod8" if args.slow_call_strategy == "fixed_mod8" else "age_empty"
    return TaskAgeDualSystemEvaluation(
        dual_system,
        processor,
        action_tokenizer,
        task_age_config=build_task_age_config(args),
        profile_steps=args.profile_steps,
        profile_sample_var_k=args.profile_sample_var_k,
        profile_sample_var_interval=args.profile_sample_var_interval,
        profile_sample_var_ages=args.profile_sample_var_ages,
        slow_trigger_policy=slow_trigger_policy,
        max_slow_age=args.max_slow_age,
        empty_ref_after_age=args.empty_ref_after_age,
        slow_call_strategy=args.slow_call_strategy,
        risk_start_age=args.risk_start_age,
        min_slow_age=args.min_slow_age,
        risk_score_threshold=args.risk_score_threshold,
        risk_late_age=args.risk_late_age,
        risk_late_score_threshold=args.risk_late_score_threshold,
        aggregation_delta_ee6_threshold=args.aggregation_delta_ee6_threshold,
        aggregation_delta_ee6_medium_threshold=args.aggregation_delta_ee6_medium_threshold,
        jerk_l2_ee6_threshold=args.jerk_l2_ee6_threshold,
        gripper_flip_count_threshold=args.gripper_flip_count_threshold,
        sample_var_ee6_threshold=args.sample_var_ee6_threshold,
        sample_var_gripper_threshold=args.sample_var_gripper_threshold,
    )


def evaluate_target_case(
    env,
    model,
    task_oracle,
    val_annotations,
    case: TargetCase,
    eval_dir,
    ep_len,
    profile_steps=False,
    profile_output=None,
    profile_rank=0,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(case.initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    prefix_success = 0
    for subtask_i, subtask in enumerate(case.eval_sequence[: case.target_i + 1]):
        success = rollout(
            env,
            model,
            task_oracle,
            subtask,
            val_annotations,
            debug=False,
            eval_dir=eval_dir,
            subtask_i=subtask_i,
            sequence_i=case.sequence_i,
            ep_len=ep_len,
            profile_steps=profile_steps,
            profile_output=profile_output,
            profile_rank=profile_rank,
        )
        if success:
            if subtask_i < case.target_i:
                prefix_success += 1
            else:
                return {
                    "case_id": int(case.case_id),
                    "sequence_i": int(case.sequence_i),
                    "target_task": case.target_task,
                    "target_i": int(case.target_i),
                    "eval_sequence": list(case.eval_sequence),
                    "prefix_success": int(prefix_success),
                    "target_attempted": True,
                    "target_success": True,
                    "failed_task": None,
                    "failed_i": None,
                    "status": "target_success",
                }
        else:
            target_attempted = subtask_i == case.target_i
            return {
                "case_id": int(case.case_id),
                "sequence_i": int(case.sequence_i),
                "target_task": case.target_task,
                "target_i": int(case.target_i),
                "eval_sequence": list(case.eval_sequence),
                "prefix_success": int(prefix_success),
                "target_attempted": bool(target_attempted),
                "target_success": False,
                "failed_task": subtask,
                "failed_i": int(subtask_i),
                "status": "target_failed" if target_attempted else "prefix_failed",
            }

    raise RuntimeError(f"Unexpected target-case fallthrough: {case}")


def summarize_records(records):
    total = Counter()
    attempted = Counter()
    success = Counter()
    prefix_failed = Counter()
    target_failed = Counter()
    for record in records:
        task = record["target_task"]
        total[task] += 1
        if record["target_attempted"]:
            attempted[task] += 1
        if record["target_success"]:
            success[task] += 1
        if record["status"] == "prefix_failed":
            prefix_failed[task] += 1
        if record["status"] == "target_failed":
            target_failed[task] += 1

    per_task = {}
    for task in sorted(total):
        per_task[task] = {
            "cases": int(total[task]),
            "target_attempted": int(attempted[task]),
            "target_success": int(success[task]),
            "prefix_failed": int(prefix_failed[task]),
            "target_failed": int(target_failed[task]),
            "target_sr_all_cases": float(success[task] / total[task]) if total[task] else None,
            "target_sr_attempted": float(success[task] / attempted[task]) if attempted[task] else None,
        }

    total_cases = sum(total.values())
    total_attempted = sum(attempted.values())
    total_success = sum(success.values())
    return {
        "overall": {
            "cases": int(total_cases),
            "target_attempted": int(total_attempted),
            "target_success": int(total_success),
            "prefix_failed": int(sum(prefix_failed.values())),
            "target_failed": int(sum(target_failed.values())),
            "target_sr_all_cases": float(total_success / total_cases) if total_cases else None,
            "target_sr_attempted": float(total_success / total_attempted) if total_attempted else None,
        },
        "per_task": per_task,
    }


def build_comparison(run_summaries):
    if len(run_summaries) < 2:
        return {}
    baseline_name = run_summaries[0]["name"]
    baseline = run_summaries[0]["summary"]["per_task"]
    comparison = {
        "baseline": baseline_name,
        "runs": {},
    }
    for run in run_summaries:
        name = run["name"]
        comparison["runs"][name] = {
            "overall_target_sr_all_cases": run["summary"]["overall"]["target_sr_all_cases"],
            "overall_target_sr_attempted": run["summary"]["overall"]["target_sr_attempted"],
            "per_task_delta_vs_baseline": {},
        }
        for task, stats in run["summary"]["per_task"].items():
            base_stats = baseline.get(task, {})
            base_all = base_stats.get("target_sr_all_cases")
            base_attempted = base_stats.get("target_sr_attempted")
            cur_all = stats.get("target_sr_all_cases")
            cur_attempted = stats.get("target_sr_attempted")
            comparison["runs"][name]["per_task_delta_vs_baseline"][task] = {
                "target_sr_all_cases_delta": None if base_all is None or cur_all is None else cur_all - base_all,
                "target_sr_attempted_delta": (
                    None if base_attempted is None or cur_attempted is None else cur_attempted - base_attempted
                ),
            }
    return comparison


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.timestamp_output_dir:
        output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, target_quotas = build_target_cases(args)
    if not cases:
        raise ValueError("No target cases selected.")
    print(f"[targeted-eval] selected {len(cases)} cases: {dict(Counter(case.target_task for case in cases))}")
    print(f"[targeted-eval] output_dir={output_dir}")

    slow_model, processor = load_generalist(args, device)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    dataset_path = CALVIN_ROOT_PATH / "dataset" / args.dataset_subdir
    env = make_env(dataset_path.as_posix(), observation_space, device, args.use_egl)

    conf_dir = CALVIN_ROOT_PATH / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    specs = parse_specialist_specs(args.specialist_paths, args.specialist_names)
    run_summaries = []
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "target_case_quotas": target_quotas,
        "selected_cases": [
            {
                "case_id": case.case_id,
                "sequence_i": case.sequence_i,
                "target_task": case.target_task,
                "target_i": case.target_i,
                "eval_sequence": list(case.eval_sequence),
            }
            for case in cases
        ],
        "runs": [],
    }

    for run_i, (name, specialist_path) in enumerate(specs):
        run_start = time.perf_counter()
        print(f"[targeted-eval] evaluating {name}: {specialist_path}", flush=True)
        policy = build_specialist_policy(args, device)
        dual_system = EvalOnlyDualSystem(slow_model, policy)
        load_info = load_specialist_into_dual_system(dual_system, specialist_path)
        dual_system.eval()
        model = build_eval_wrapper(args, dual_system, processor, action_tokenizer)

        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        detail_path = run_dir / "targeted_records.jsonl"
        detail_path.write_text("")
        profile_output = None
        if args.profile_steps:
            profile_output = run_dir / "profile.jsonl"
            profile_output.write_text("")
            emit_profile_record(
                profile_output,
                {
                    "event": "run_config",
                    "name": name,
                    "specialist_path": specialist_path,
                    "target_case_quotas": target_quotas,
                },
            )

        records = []
        progress = tqdm(cases, desc=f"targeted:{name}")
        for case in progress:
            record = evaluate_target_case(
                env,
                model,
                task_oracle,
                val_annotations,
                case,
                eval_dir=run_dir,
                ep_len=args.ep_len,
                profile_steps=args.profile_steps,
                profile_output=profile_output,
                profile_rank=run_i,
            )
            records.append(record)
            with detail_path.open("a") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
            progress.set_postfix(
                {
                    "task": case.target_task,
                    "status": record["status"],
                }
            )

        summary = summarize_records(records)
        run_payload = {
            "name": name,
            "specialist_path": specialist_path,
            "load_info": load_info,
            "elapsed_s": round(time.perf_counter() - run_start, 3),
            "summary": summary,
            "records_path": detail_path.as_posix(),
            "profile_path": None if profile_output is None else profile_output.as_posix(),
        }
        (run_dir / "summary.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))
        print(json.dumps(run_payload, indent=2, sort_keys=True), flush=True)
        run_summaries.append(run_payload)
        manifest["runs"].append(run_payload)

        del model
        del dual_system
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["comparison"] = build_comparison(run_summaries)
    (output_dir / "targeted_eval_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[targeted-eval] wrote summary: {output_dir / 'targeted_eval_summary.json'}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist_path", default=DEFAULT_GENERALIST_PATH.as_posix(), type=str)
    parser.add_argument("--specialist_paths", default=DEFAULT_SPECIALIST_PATH.as_posix(), type=str)
    parser.add_argument("--specialist_names", default="", type=str)
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset", type=str)
    parser.add_argument(
        "--output_dir",
        default=(REPO_ROOT / "evaluation_results" / "lora_targeted_eval").as_posix(),
        type=str,
    )
    parser.add_argument("--timestamp_output_dir", action="store_true")
    parser.add_argument("--target_case_quotas", default=DEFAULT_TARGET_CASE_QUOTAS_ARG, type=str)
    parser.add_argument("--search_num_sequences", default=100, type=int)
    parser.add_argument("--ep_len", default=360, type=int)
    parser.add_argument("--use_egl", action="store_true")

    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--with_cfg", default=False, action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none", type=str)
    parser.add_argument("--attn_implementation", default="none", type=str)
    parser.add_argument("--fast_num_inference_steps", default=10, type=int)

    parser.add_argument("--profile_steps", action="store_true")
    parser.add_argument("--profile_sample_var_k", default=3, type=int)
    parser.add_argument("--profile_sample_var_interval", default=8, type=int)
    parser.add_argument("--profile_sample_var_ages", default="", type=str)

    parser.add_argument(
        "--slow_call_strategy",
        default="task_age",
        choices=["fixed_mod8", "age_empty", "task_age", "risk_balanced", "risk_score", "risk_conservative", "risk_aggressive"],
        type=str,
    )
    parser.add_argument("--max_slow_age", default=12, type=int)
    parser.add_argument("--empty_ref_after_age", default=8, type=int)
    parser.add_argument("--min_slow_age", default=7, type=int)
    parser.add_argument("--risk_start_age", default=8, type=int)
    parser.add_argument("--risk_score_threshold", default=2, type=int)
    parser.add_argument("--risk_late_age", default=12, type=int)
    parser.add_argument("--risk_late_score_threshold", default=1, type=int)
    parser.add_argument("--aggregation_delta_ee6_threshold", default=0.22, type=float)
    parser.add_argument("--aggregation_delta_ee6_medium_threshold", default=0.12, type=float)
    parser.add_argument("--jerk_l2_ee6_threshold", default=0.32, type=float)
    parser.add_argument("--gripper_flip_count_threshold", default=2, type=int)
    parser.add_argument("--sample_var_ee6_threshold", default=0.012, type=float)
    parser.add_argument("--sample_var_gripper_threshold", default=0.86, type=float)
    parser.add_argument("--task_age_default_max_slow_age", default=12, type=int)
    parser.add_argument("--task_age_group_a_max_slow_age", default=13, type=int)
    parser.add_argument("--task_age_group_b_max_slow_age", default=12, type=int)
    parser.add_argument("--task_age_group_c_max_slow_age", default=10, type=int)
    parser.add_argument("--task_age_group_d_max_slow_age", default=8, type=int)
    parser.add_argument("--task_age_group_a_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_A), type=str)
    parser.add_argument("--task_age_group_b_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_B), type=str)
    parser.add_argument("--task_age_group_c_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_C), type=str)
    parser.add_argument("--task_age_group_d_tasks", default=",".join(DEFAULT_TASK_AGE_GROUP_D), type=str)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
