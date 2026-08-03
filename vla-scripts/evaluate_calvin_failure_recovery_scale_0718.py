# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Scale failure-recovery collection from the reviewed task-age baseline.

Lineage: evaluate_calvin_task_age_0525.py -> reviewed failure-recovery v1 collector
-> this v2 collection entry point. Normal evaluation remains the baseline path;
collection and replay require explicit opt-in flags.
"""
import argparse
from collections import Counter, deque
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import resource
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALVIN_ROOT = REPO_ROOT.parent / "calvin"
CALVIN_ROOT_PATH = Path(os.environ.get("CALVIN_ROOT", DEFAULT_CALVIN_ROOT)).expanduser().resolve()
os.environ.setdefault("CALVIN_ROOT", CALVIN_ROOT_PATH.as_posix())
for dependency_path in (
    CALVIN_ROOT_PATH / "calvin_models",
    CALVIN_ROOT_PATH / "calvin_env",
    CALVIN_ROOT_PATH / "calvin_env" / "tacto",
):
    if dependency_path.exists():
        sys.path.insert(0, dependency_path.as_posix())

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from dual_sys_evaluation_0424test import DualSystemCalvinEvaluation as BaseDualSystemCalvinEvaluation

from ema_pytorch import EMA
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
DEFAULT_GENERALIST_PATH = REPO_ROOT.parent / "models" / "generalist"
DEFAULT_SPECIALIST_PATH = REPO_ROOT.parent / "models" / "specialist" / "Specialist+Depth+Gripper.pt"
CALVIN_ROOT = os.environ["CALVIN_ROOT"]

DEFAULT_TASK_AGE_GROUP_A = [
    "open_drawer",
    "move_slider_right",
    "turn_on_led",
    "turn_off_led",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "lift_red_block_table",
    "push_into_drawer",
    "push_pink_block_left",
    "rotate_blue_block_left",
    "rotate_red_block_left",
    "lift_blue_block_drawer",
    "lift_pink_block_drawer",
    "lift_red_block_drawer",
]
DEFAULT_TASK_AGE_GROUP_B = [
    "close_drawer",
    "move_slider_left",
    "place_in_drawer",
    "place_in_slider",
    "lift_pink_block_table",
    "lift_red_block_slider",
    "push_pink_block_right",
    "push_red_block_right",
    "rotate_blue_block_right",
    "rotate_pink_block_left",
    "rotate_pink_block_right",
    "rotate_red_block_right",
    "unstack_block",
]
DEFAULT_TASK_AGE_GROUP_C = [
    "lift_blue_block_slider",
    "lift_pink_block_slider",
    "lift_blue_block_table",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_red_block_left",
]
DEFAULT_TASK_AGE_GROUP_D = [
    "stack_block",
]
BENCHMARK_NUM_SEQUENCES = 100


def read_proc_io():
    io_path = Path("/proc/self/io")
    if not io_path.exists():
        return {}
    stats = {}
    for line in io_path.read_text().splitlines():
        key, value = line.split(":")
        stats[key.strip()] = int(value.strip())
    return stats


def runtime_snapshot():
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    snapshot = {
        "rss_mb": round(rss_kb / 1024, 2),
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_alloc_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
                "cuda_max_alloc_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "cuda_max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
            }
        )
    snapshot.update(read_proc_io())
    return snapshot


def emit_profile_record(profile_output, record):
    line = json.dumps(record, sort_keys=True)
    print(f"[specialist-profile] {line}", flush=True)
    if profile_output is None:
        return
    with open(profile_output, "a") as file:
        file.write(line + "\n")


class InitProfiler:
    def __init__(self, enabled):
        self.enabled = enabled
        self.start = time.perf_counter()
        self.last = self.start
        self.prev_io = read_proc_io()

    def mark(self, stage, extra=None):
        if not self.enabled:
            return
        now = time.perf_counter()
        current_io = read_proc_io()
        io_delta = {}
        for key, value in current_io.items():
            io_delta[f"{key}_delta"] = value - self.prev_io.get(key, 0)
        payload = {
            "stage": stage,
            "stage_s": round(now - self.last, 4),
            "total_s": round(now - self.start, 4),
            "runtime": runtime_snapshot(),
            "io_delta": io_delta,
        }
        if extra:
            payload["extra"] = extra
        print(f"[init-profile] {json.dumps(payload, sort_keys=True)}", flush=True)
        self.last = now
        self.prev_io = current_io


class VariableSlowCallDualSystemEvaluation(BaseDualSystemCalvinEvaluation):
    """Evaluation wrapper with age-based and risk-triggered slow call policies.

    Risk-triggered policies use the previous step's profile to decide whether the
    current step should refresh the slow system. This avoids running the fast
    policy twice in one step while still reacting to empty-ref instability.
    """

    def __init__(
        self,
        *args,
        slow_call_strategy="risk_balanced",
        risk_start_age=8,
        min_slow_age=7,
        risk_score_threshold=2,
        risk_late_age=12,
        risk_late_score_threshold=1,
        aggregation_delta_ee6_threshold=0.22,
        aggregation_delta_ee6_medium_threshold=0.12,
        jerk_l2_ee6_threshold=0.32,
        gripper_flip_count_threshold=2,
        sample_var_ee6_threshold=0.012,
        sample_var_gripper_threshold=0.86,
        **kwargs,
    ):
        base_policy = kwargs.pop("slow_trigger_policy", "age_empty")
        if slow_call_strategy == "fixed_mod8":
            base_policy = "fixed_mod8"
        else:
            base_policy = "age_empty"
        super().__init__(*args, slow_trigger_policy=base_policy, **kwargs)
        self.slow_call_strategy = str(slow_call_strategy)
        self.risk_start_age = int(risk_start_age)
        self.min_slow_age = int(min_slow_age)
        self.risk_score_threshold = int(risk_score_threshold)
        self.risk_late_age = int(risk_late_age)
        self.risk_late_score_threshold = int(risk_late_score_threshold)
        self.aggregation_delta_ee6_threshold = float(aggregation_delta_ee6_threshold)
        self.aggregation_delta_ee6_medium_threshold = float(aggregation_delta_ee6_medium_threshold)
        self.jerk_l2_ee6_threshold = float(jerk_l2_ee6_threshold)
        self.gripper_flip_count_threshold = int(gripper_flip_count_threshold)
        self.sample_var_ee6_threshold = float(sample_var_ee6_threshold)
        self.sample_var_gripper_threshold = float(sample_var_gripper_threshold)
        self._slow_decision_details = {}

    @staticmethod
    def _as_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    def _risk_from_previous_step(self):
        profile = self.last_step_profile or {}
        prev_age = profile.get("slow_age_after", profile.get("step_since_slow"))
        if prev_age is None:
            return {
                "source_age": None,
                "score": 0,
                "flags": {},
                "trigger": False,
                "reason": "no_previous_profile",
            }
        prev_age = int(prev_age)
        if prev_age < self.risk_start_age:
            return {
                "source_age": prev_age,
                "score": 0,
                "flags": {},
                "trigger": False,
                "reason": "before_risk_age",
            }

        agg = self._as_float(profile.get("aggregation_delta_ee6"))
        jerk = self._as_float(profile.get("jerk_l2_ee6"))
        sample_ee6 = self._as_float(profile.get("sample_var_ee6"))
        sample_gripper = self._as_float(profile.get("sample_var_gripper"))
        flip = self._as_float(profile.get("gripper_flip_count"))

        flags = {
            "aggregation_delta_ee6_high": agg is not None and agg > self.aggregation_delta_ee6_threshold,
            "jerk_l2_ee6_high": jerk is not None and jerk > self.jerk_l2_ee6_threshold,
            "gripper_flip_count_high": flip is not None and flip >= self.gripper_flip_count_threshold,
            "sample_var_ee6_high": sample_ee6 is not None and sample_ee6 > self.sample_var_ee6_threshold,
            "sample_var_gripper_high": sample_gripper is not None and sample_gripper > self.sample_var_gripper_threshold,
            "aggregation_delta_ee6_medium": agg is not None and agg > self.aggregation_delta_ee6_medium_threshold,
        }
        score_flags = [
            flags["aggregation_delta_ee6_high"],
            flags["jerk_l2_ee6_high"],
            flags["gripper_flip_count_high"],
            flags["sample_var_ee6_high"],
            flags["sample_var_gripper_high"],
        ]
        score = int(sum(1 for flag in score_flags if flag))

        direct_balanced = (
            flags["aggregation_delta_ee6_high"]
            or flags["jerk_l2_ee6_high"]
            or flags["gripper_flip_count_high"]
            or flags["sample_var_ee6_high"]
            or (flags["sample_var_gripper_high"] and flags["aggregation_delta_ee6_medium"])
        )
        risk_score_trigger = score >= self.risk_score_threshold
        late_score_trigger = prev_age >= self.risk_late_age and score >= self.risk_late_score_threshold

        if self.slow_call_strategy == "risk_balanced":
            trigger = direct_balanced
            reason = "risk_balanced" if trigger else "risk_clear"
        elif self.slow_call_strategy == "risk_score":
            trigger = risk_score_trigger
            reason = "risk_score" if trigger else "risk_clear"
        elif self.slow_call_strategy == "risk_conservative":
            trigger = score >= 1
            reason = "risk_conservative" if trigger else "risk_clear"
        elif self.slow_call_strategy == "risk_aggressive":
            trigger = risk_score_trigger or late_score_trigger
            reason = "risk_aggressive" if trigger else "risk_clear"
        else:
            trigger = False
            reason = "strategy_without_risk"

        return {
            "source_age": prev_age,
            "score": score,
            "flags": flags,
            "trigger": bool(trigger),
            "reason": reason,
            "values": {
                "aggregation_delta_ee6": agg,
                "jerk_l2_ee6": jerk,
                "gripper_flip_count": flip,
                "sample_var_ee6": sample_ee6,
                "sample_var_gripper": sample_gripper,
            },
        }

    def _should_call_slow_system(self, step):
        self._slow_decision_details = {
            "slow_call_strategy": self.slow_call_strategy,
            "risk_start_age": self.risk_start_age,
            "min_slow_age": self.min_slow_age,
            "risk_score_threshold": self.risk_score_threshold,
            "risk_late_age": self.risk_late_age,
            "risk_late_score_threshold": self.risk_late_score_threshold,
        }

        if self.last_slow_step is None:
            self._slow_decision_details["slow_risk"] = None
            return True, "initial"

        if self.slow_call_strategy == "fixed_mod8":
            if (step + 1) % self.temporal_size == 0:
                self._slow_decision_details["slow_risk"] = None
                return True, "fixed_mod8"
            self._slow_decision_details["slow_risk"] = None
            return False, "fixed_mod8_skip"

        slow_age_before = int(step - self.last_slow_step)
        self._slow_decision_details["slow_age_before_decision"] = slow_age_before
        if slow_age_before < self.min_slow_age:
            self._slow_decision_details["slow_risk"] = None
            return False, "min_slow_age_skip"
        if slow_age_before >= self.max_slow_age:
            self._slow_decision_details["slow_risk"] = None
            return True, "max_slow_age"
        if self.slow_call_strategy == "age_empty":
            self._slow_decision_details["slow_risk"] = None
            return False, "age_skip"

        risk = self._risk_from_previous_step()
        self._slow_decision_details["slow_risk"] = risk
        if slow_age_before >= self.risk_start_age and risk["trigger"]:
            return True, risk["reason"]
        return False, "risk_skip"

    def step(self, obs, instruction, step):
        action = super().step(obs, instruction, step)
        if self.last_step_profile is not None:
            self.last_step_profile.update(self._slow_decision_details)
        return action


def parse_task_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    normalized = str(value).replace("\n", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def build_task_age_config(args):
    groups = {
        "A": {
            "max_slow_age": int(args.task_age_group_a_max_slow_age),
            "tasks": parse_task_list(args.task_age_group_a_tasks),
            "description": "extended_age",
        },
        "B": {
            "max_slow_age": int(args.task_age_group_b_max_slow_age),
            "tasks": parse_task_list(args.task_age_group_b_tasks),
            "description": "default_age",
        },
        "C": {
            "max_slow_age": int(args.task_age_group_c_max_slow_age),
            "tasks": parse_task_list(args.task_age_group_c_tasks),
            "description": "protected_age",
        },
        "D": {
            "max_slow_age": int(args.task_age_group_d_max_slow_age),
            "tasks": parse_task_list(args.task_age_group_d_tasks),
            "description": "high_guidance",
        },
    }
    task_age_map = {}
    task_group_map = {}
    duplicate_tasks = {}
    for group_name, group in groups.items():
        for task in group["tasks"]:
            if task in task_age_map:
                duplicate_tasks.setdefault(task, []).append(task_group_map[task])
                duplicate_tasks[task].append(group_name)
            task_age_map[task] = int(group["max_slow_age"])
            task_group_map[task] = group_name
    if duplicate_tasks:
        raise ValueError(f"Duplicate task_age task assignments: {duplicate_tasks}")
    return {
        "default_max_slow_age": int(args.task_age_default_max_slow_age),
        "groups": groups,
        "task_age_map": task_age_map,
        "task_group_map": task_group_map,
    }


class TaskAgeDualSystemEvaluation(VariableSlowCallDualSystemEvaluation):
    """Task-conditioned age scheduler.

    The slow system is still called with the same age_empty mechanics as the
    0424/0428 scripts. Only max_slow_age is selected per CALVIN subtask.
    """

    def __init__(
        self,
        *args,
        task_age_config=None,
        slow_call_strategy="task_age",
        **kwargs,
    ):
        requested_strategy = str(slow_call_strategy)
        parent_strategy = "age_empty" if requested_strategy == "task_age" else requested_strategy
        super().__init__(*args, slow_call_strategy=parent_strategy, **kwargs)
        self.slow_call_strategy = requested_strategy
        self.task_age_config = task_age_config or {
            "default_max_slow_age": int(self.max_slow_age),
            "groups": {},
            "task_age_map": {},
            "task_group_map": {},
        }
        self.global_max_slow_age = int(self.max_slow_age)
        self.current_task = None
        self._active_task_age_info = self._resolve_task_age_info(None)

    def set_current_task(self, task):
        self.current_task = None if task is None else str(task)
        self._active_task_age_info = self._resolve_task_age_info(self.current_task)

    def _resolve_task_age_info(self, task):
        task_age_map = self.task_age_config.get("task_age_map", {})
        task_group_map = self.task_age_config.get("task_group_map", {})
        default_max_slow_age = int(self.task_age_config.get("default_max_slow_age", self.global_max_slow_age if hasattr(self, "global_max_slow_age") else self.max_slow_age))
        if task is not None and task in task_age_map:
            group = task_group_map.get(task, "custom")
            max_slow_age = int(task_age_map[task])
        else:
            group = "default"
            max_slow_age = default_max_slow_age
        return {
            "task": task,
            "task_age_group": group,
            "task_max_slow_age": int(max_slow_age),
            "task_age_default_max_slow_age": int(default_max_slow_age),
        }

    def _should_call_slow_system(self, step):
        if self.slow_call_strategy != "task_age":
            return super()._should_call_slow_system(step)

        info = self._active_task_age_info or self._resolve_task_age_info(self.current_task)
        task_max_slow_age = int(info["task_max_slow_age"])
        self._slow_decision_details = {
            "slow_call_strategy": self.slow_call_strategy,
            "task": self.current_task,
            "task_age_group": info["task_age_group"],
            "task_max_slow_age": task_max_slow_age,
            "task_age_default_max_slow_age": info["task_age_default_max_slow_age"],
            "slow_risk": None,
        }

        if self.last_slow_step is None:
            return True, "initial"

        slow_age_before = int(step - self.last_slow_step)
        self._slow_decision_details["slow_age_before_decision"] = slow_age_before
        if slow_age_before >= task_max_slow_age:
            return True, "task_max_slow_age"
        return False, "task_age_skip"

    def step(self, obs, instruction, step):
        if self.slow_call_strategy != "task_age":
            return super().step(obs, instruction, step)

        self._active_task_age_info = self._resolve_task_age_info(self.current_task)
        previous_max_slow_age = self.max_slow_age
        self.max_slow_age = int(self._active_task_age_info["task_max_slow_age"])
        try:
            action = super().step(obs, instruction, step)
        finally:
            self.max_slow_age = previous_max_slow_age
        if self.last_step_profile is not None:
            self.last_step_profile.update(self._active_task_age_info)
        return action


def print_and_save(results, sequences, eval_result_path, task_name=None, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    # model_name = 'vla-test'
    split_dir = Path(eval_result_path).parent / str(task_name)
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f'split_{torch.cuda.current_device()}.json', "w") as file:
        json.dump(chain_sr, file)

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )



def make_env(dataset_path, observation_space, device, use_egl):
    val_folder = Path(dataset_path) / "validation"
    from calvin_env_wrapper import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device, use_egl=use_egl)
    return env


def evaluate_policy(
    model,
    env,
    eval_sr_path,
    eval_result_path,
    num_procs,
    procs_id,
    eval_dir,
    ep_len,
    num_sequences,
    task_name='test',
    enrich_lang=False,
    debug=False,
    max_subtasks=None,
    profile_steps=False,
    profile_output=None,
    profile_rank=0,
):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    if enrich_lang:
        with open('vla-scripts/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = list(get_sequences(num_sequences))
    num_seq_per_procs = int(np.ceil(num_sequences / num_procs))
    start_idx = num_seq_per_procs * procs_id
    end_idx = min(num_sequences, num_seq_per_procs * (procs_id + 1))
    eval_sequences = eval_sequences[start_idx:end_idx]
    eval_sequences_for_report = list(eval_sequences)
    if profile_steps:
        print(
            f"[profile] rank={profile_rank} sequence_range=[{start_idx}, {end_idx}) "
            f"profile_output={profile_output}",
            flush=True,
        )

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = start_idx
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_dir,
            sequence_i,
            ep_len,
            max_subtasks=max_subtasks,
            profile_steps=profile_steps,
            profile_output=profile_output,
            profile_rank=profile_rank,
        )
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences_for_report, eval_result_path, task_name, None)
    return results


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug,
    eval_dir,
    sequence_i,
    ep_len,
    max_subtasks=None,
    profile_steps=False,
    profile_output=None,
    profile_rank=0,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if max_subtasks is not None:
        eval_sequence = eval_sequence[:max_subtasks]
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(
            env,
            model,
            task_checker,
            subtask,
            val_annotations,
            debug,
            eval_dir,
            subtask_i,
            sequence_i,
            ep_len,
            profile_steps=profile_steps,
            profile_output=profile_output,
            profile_rank=profile_rank,
        )
        if success:
            # print('success: ', subtask_i)
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    eval_dir,
    subtask_i,
    sequence_i,
    ep_len,
    profile_steps=False,
    profile_output=None,
    profile_rank=0,
):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    if hasattr(model, "set_current_task"):
        model.set_current_task(subtask)
    start_info = env.get_info()
    if profile_steps:
        print(
            f"[profile] rank={profile_rank} sequence={sequence_i} subtask={subtask_i} "
            f"name={subtask} ep_len={ep_len}",
            flush=True,
        )
    if debug:
        img_dict = {
            'static': [],
            'gripper': [],
        }

    for step in range(ep_len):
        model_start = time.perf_counter()
        action = model.step(obs, lang_annotation, step)
        model_step_s = time.perf_counter() - model_start
        env_start = time.perf_counter()
        obs, _, _, current_info = env.step(action)
        env_step_s = time.perf_counter() - env_start

        if debug:
            img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
            img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        # check if current step solves a task
        oracle_start = time.perf_counter()
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        oracle_step_s = time.perf_counter() - oracle_start
        if profile_steps:
            step_profile = getattr(model, "last_step_profile", {})
            emit_profile_record(
                profile_output,
                {
                    "event": "step",
                    "rank": int(profile_rank),
                    "sequence": int(sequence_i),
                    "subtask_i": int(subtask_i),
                    "task": subtask,
                    "step": int(step),
                    "ep_len": int(ep_len),
                    "model_s": round(float(model_step_s), 6),
                    "env_s": round(float(env_step_s), 6),
                    "oracle_s": round(float(oracle_step_s), 6),
                    "step_success": bool(len(current_task_info) > 0),
                    "terminal_step": bool(len(current_task_info) > 0),
                    "profile": step_profile,
                },
            )
        if len(current_task_info) > 0:
            if profile_steps:
                emit_profile_record(
                    profile_output,
                    {
                        "event": "subtask_end",
                        "rank": int(profile_rank),
                        "sequence": int(sequence_i),
                        "subtask_i": int(subtask_i),
                        "task": subtask,
                        "task_success": True,
                        "steps": int(step + 1),
                    },
                )
            if debug:
                print(colored("success", "green"), end=" ")
                for key in img_dict.keys():
                    clip = ImageSequenceClip(img_dict[key], fps=30)
                    clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.gif'), fps=30)
            return True

    if profile_steps:
        emit_profile_record(
            profile_output,
            {
                "event": "subtask_end",
                "rank": int(profile_rank),
                "sequence": int(sequence_i),
                "subtask_i": int(subtask_i),
                "task": subtask,
                "task_success": False,
                "steps": int(ep_len),
            },
        )
    if debug:
        print(colored("fail", "red"), end=" ")
        for key in img_dict.keys():
            clip = ImageSequenceClip(img_dict[key], fps=30)
            clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-fail.gif'), fps=30)
    return False


# Failure-recovery collection -------------------------------------------------
#
# The collector deliberately lives in a byte-for-byte copy of the stable 0525
# evaluator.  The normal evaluation path above is unchanged.  In collection
# mode, PyBullet saveState/restoreState gives every candidate branch the exact
# same in-memory physics state, while the portable robot_obs/scene_obs snapshot
# is audited separately because it cannot preserve every contact constraint.

RECOVERY_STATE_FIELDS = (
    "action",
    "hidden_states",
    "obs_buffer",
    "action_buffer",
    "action_buffer_mask",
    "hist_action",
    "gripper_window",
    "last_slow_step",
    "prev_action",
    "prev_prev_action",
    "prev_proprio",
    "prev_obs_tensor",
    "last_step_profile",
    "_slow_handover",
    "_active_task",
    "_active_task_age_info",
)


def _clone_runtime_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, deque):
        return deque((_clone_runtime_value(item) for item in value), maxlen=value.maxlen)
    return copy.deepcopy(value)


def _cpu_runtime_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, deque):
        return deque((_cpu_runtime_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: _cpu_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_runtime_value(item) for item in value)
    return copy.deepcopy(value)


def capture_recovery_model_state(model):
    return {
        name: _clone_runtime_value(getattr(model, name))
        for name in RECOVERY_STATE_FIELDS
        if hasattr(model, name)
    }


def restore_recovery_model_state(model, state):
    for name, value in state.items():
        setattr(model, name, _clone_runtime_value(value))


def restore_persistent_model_state(model, state):
    """Restore CPU-persisted runtime, moving slow condition tensors to inference device."""

    runtime_device = model._runtime_device()
    restored = {name: _clone_runtime_value(value) for name, value in state.items()}
    for name in ("action", "hidden_states"):
        if torch.is_tensor(restored.get(name)):
            restored[name] = restored[name].to(runtime_device)
    handover = restored.get("_slow_handover")
    if isinstance(handover, dict):
        for name in ("old_action", "old_hidden_states"):
            if torch.is_tensor(handover.get(name)):
                handover[name] = handover[name].to(runtime_device)
    restore_recovery_model_state(model, restored)


def restore_persisted_failure_state(env, bullet, states_dir, state_id):
    """Restore controller/logical state without perturbing persisted Bullet contacts."""

    states_dir = Path(states_dir)
    bullet_path = states_dir / f"{state_id}.bullet"
    simulator_path = states_dir / f"{state_id}_simulator.pt"
    bullet.p.restoreState(
        fileName=bullet_path.as_posix(),
        physicsClientId=bullet.cid,
    )
    simulator = torch.load(
        simulator_path,
        map_location="cpu",
        weights_only=False,
    )
    bullet.robot.reset_from_storage(simulator["robot"])
    bullet.scene.reset_from_storage(simulator["scene"])
    # reset_from_storage restores controller/logical state but can move physical
    # bodies and contacts. Reload the persisted Bullet world after those resets.
    bullet.p.restoreState(
        fileName=bullet_path.as_posix(),
        physicsClientId=bullet.cid,
    )
    # reset_from_storage restores the gripper motor command, but CALVIN's
    # implementation does not restore the Python-side gripper_action field.
    # get_obs() exposes that field as robot_obs[-1], so leaving it stale creates
    # an offline/online proprio mismatch even when the Bullet joints are exact.
    gripper_action = int(np.asarray(simulator["robot"]["gripper_action"]).item())
    bullet.robot.gripper_action = gripper_action
    bullet.robot.control_gripper(gripper_action)
    restored_obs = env.get_obs()
    bullet.robot.target_pos = np.asarray(
        restored_obs["robot_obs"][:3], dtype=np.float64
    ).copy()
    bullet.robot.target_orn = np.asarray(
        restored_obs["robot_obs"][3:6], dtype=np.float64
    ).copy()
    return restored_obs


def _terminal_padded_actions(actions, chunk_size=8):
    """Pad a successful trace to a complete action chunk with a no-motion hold."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("Recovery actions must be a non-empty [T,7] array")
    remainder = len(actions) % int(chunk_size)
    if not remainder:
        return actions.copy(), 0
    padding_steps = int(chunk_size) - remainder
    hold = np.zeros((padding_steps, 7), dtype=np.float32)
    hold[:, -1] = actions[-1, -1]
    return np.concatenate((actions, hold), axis=0), padding_steps


def load_persisted_oracle_start(states_dir, state_id, env, allow_missing=False):
    """Load the original subtask oracle baseline used to label collection branches."""

    oracle_path = Path(states_dir) / f"{state_id}_oracle_start.pt"
    if oracle_path.is_file():
        return (
            torch.load(oracle_path, map_location="cpu", weights_only=False),
            "persisted_subtask_start",
        )
    if allow_missing:
        return env.get_info(), "failure_state_fallback"
    raise FileNotFoundError(
        f"Missing original oracle baseline for {state_id}: {oracle_path}. "
        "Quarantine/recollect this legacy state or explicitly opt into the "
        "non-comparable failure-state fallback."
    )


def _bullet_env(env):
    candidate = env
    for _ in range(6):
        if hasattr(candidate, "p") and hasattr(candidate, "cid"):
            return candidate
        if not hasattr(candidate, "env"):
            break
        candidate = candidate.env
    raise RuntimeError("Could not locate the CALVIN PlayTableEnv / PyBullet client")


def _portable_obs(obs):
    return {
        "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32).copy(),
        "scene_obs": np.asarray(obs["scene_obs"], dtype=np.float32).copy(),
        "rgb_static": np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8).copy(),
        "rgb_gripper": np.asarray(obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8).copy(),
        "depth_static": np.asarray(obs["depth_obs"]["depth_static"], dtype=np.float32).copy(),
        "depth_gripper": np.asarray(obs["depth_obs"]["depth_gripper"], dtype=np.float32).copy(),
    }


def _history_array(model):
    result = np.zeros((4, 7), dtype=np.float32)
    if model.hist_action:
        values = torch.stack(list(model.hist_action), dim=0).detach().cpu().numpy().astype(np.float32)
        result[-len(values) :] = values[-4:]
    return result


def _state_split(state_id):
    bucket = int(hashlib.sha256(state_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def _set_branch_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe_profile(profile):
    result = {}
    for key, value in dict(profile or {}).items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            result[key] = value
        elif isinstance(value, np.generic):
            result[key] = value.item()
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
    return result


def parse_branch_strategies(value):
    allowed = {"base_seed", "forced_refresh", "slow_override", "demo_guided"}
    result = [item.strip() for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or not set(result).issubset(allowed):
        raise ValueError(f"branch strategies must be unique values from {sorted(allowed)}")
    return result


def _angle_delta(start, end):
    return (np.asarray(end) - np.asarray(start) + np.pi) % (2 * np.pi) - np.pi


def _interpolate_demo_pose(start, end, steps, gripper):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    for alpha in np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float64)[1:]:
        pose = start.copy()
        pose[:3] = start[:3] + alpha * (end[:3] - start[:3])
        pose[3:6] = start[3:6] + alpha * _angle_delta(start[3:6], end[3:6])
        yield np.concatenate((pose, [gripper]))


def _demo_place_targets(robot_obs, release_pose, seed):
    """Retarget a CALVIN place demo with small, deterministic endpoint diversity."""
    rng = np.random.default_rng(seed)
    current = np.asarray(robot_obs[:6], dtype=np.float64)
    release = np.asarray(release_pose[:6], dtype=np.float64).copy()
    release[:2] += np.array([0.0004, -0.0026]) + rng.uniform(-0.0005, 0.0005, size=2)
    safe_z = max(float(current[2]), 0.58)
    raised = current.copy()
    raised[2] = safe_z
    above = release.copy()
    above[2] = safe_z
    retreat = release.copy()
    retreat[2] = safe_z
    yield from _interpolate_demo_pose(current, raised, 8, -1.0)
    yield from _interpolate_demo_pose(raised, above, 24, -1.0)
    yield from _interpolate_demo_pose(above, release, 12, -1.0)
    for _ in range(8):
        yield np.concatenate((release, [1.0]))
    yield from _interpolate_demo_pose(release, retreat, 12, 1.0)


def _relative_demo_action(target, robot_obs):
    target = np.asarray(target, dtype=np.float64)
    action = np.empty(7, dtype=np.float32)
    action[:3] = np.clip((target[:3] - robot_obs[:3]) / 0.02, -1.0, 1.0)
    action[3:6] = np.clip(_angle_delta(robot_obs[3:6], target[3:6]) / 0.05, -1.0, 1.0)
    action[6] = -1.0 if target[6] < 0 else 1.0
    return action


def load_place_demo_release(dataset_subdir):
    root = Path(CALVIN_ROOT) / "dataset" / dataset_subdir / "validation"
    annotation = np.load(
        root / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True
    ).item()
    for task, (start, end) in zip(
        annotation["language"]["task"], annotation["info"]["indx"]
    ):
        if task != "place_in_slider":
            continue
        closed = []
        for frame_id in range(int(start), int(end) + 1):
            with np.load(root / f"episode_{frame_id:07d}.npz") as frame:
                action = np.asarray(frame["actions"], dtype=np.float64)
            if action[-1] < 0:
                closed.append(action)
        if closed:
            return closed[-1]
    raise RuntimeError("No place_in_slider demonstration is available")


def load_demo_guidance(dataset_subdir):
    """Load compact expert trajectories for recovery tasks represented locally."""
    dataset = Path(CALVIN_ROOT) / "dataset" / dataset_subdir
    requested = {
        "place_in_slider",
        "lift_red_block_table",
        "lift_blue_block_slider",
        "push_pink_block_right",
    }
    guidance = {}
    for split in ("training", "validation"):
        root = dataset / split
        annotation = np.load(
            root / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True
        ).item()
        for task, (start, end) in zip(
            annotation["language"]["task"], annotation["info"]["indx"]
        ):
            if task not in requested or task in guidance:
                continue
            actions = []
            first_scene = None
            for frame_id in range(int(start), int(end) + 1):
                with np.load(root / f"episode_{frame_id:07d}.npz") as frame:
                    actions.append(np.asarray(frame["actions"], dtype=np.float64))
                    if first_scene is None:
                        first_scene = np.asarray(frame["scene_obs"], dtype=np.float64)
            guidance[task] = {
                "actions": np.asarray(actions, dtype=np.float64),
                "first_scene": first_scene,
                "source": {"split": split, "start": int(start), "end": int(end)},
            }
    missing = requested - set(guidance)
    if missing:
        raise RuntimeError(f"Missing local CALVIN demonstrations: {sorted(missing)}")
    return guidance


def _demo_guidance_key(task):
    if task == "place_in_slider":
        return task
    if task.startswith("lift_") and task.endswith("_table"):
        return "lift_red_block_table"
    if task.startswith("lift_") and task.endswith("_slider"):
        return "lift_blue_block_slider"
    if task == "push_pink_block_right":
        return task
    if task == "stack_block":
        return task
    return None


def _task_object_name(task):
    for color in ("red", "blue", "pink"):
        if f"_{color}_block_" in f"_{task}_":
            return f"block_{color}", color
    raise ValueError(f"Cannot identify task object: {task}")


def _stack_release_offset(seed):
    """Small deterministic placement sweep; consecutive branch seeds are distinct."""
    offsets = (
        (0.0, 0.0, 0.0),
        (0.004, 0.0, 0.0),
        (-0.004, 0.0, 0.0),
        (0.0, 0.004, 0.0),
        (0.0, -0.004, 0.0),
        (0.0, 0.0, 0.006),
        (0.0, 0.0, -0.006),
        (0.003, 0.003, 0.003),
        (-0.003, -0.003, 0.003),
    )
    return np.asarray(offsets[int(seed) % len(offsets)], dtype=np.float64)


def _retargeted_demo_targets(env, task, guidance, seed):
    key = _demo_guidance_key(task)
    if key is None:
        raise ValueError(f"demo_guided has no audited local demonstration for task={task}")
    if task == "place_in_slider":
        closed = guidance[key]["actions"][guidance[key]["actions"][:, -1] < 0]
        yield from _demo_place_targets(env.get_obs()["robot_obs"], closed[-1], seed)
        return
    if task == "stack_block":
        info = env.get_info()
        robot_uid = info["robot_info"]["uid"]
        objects = info["scene_info"]["movable_objects"]
        held = [
            name for name, item in objects.items()
            if robot_uid in {contact[2] for contact in item["contacts"]}
        ]
        if len(held) != 1:
            tcp = np.asarray(env.get_obs()["robot_obs"][:3], dtype=np.float64)
            held = [min(objects, key=lambda name: np.linalg.norm(
                np.asarray(objects[name]["current_pos"], dtype=np.float64) - tcp
            ))]
        held_name = held[0]
        held_pos = np.asarray(objects[held_name]["current_pos"], dtype=np.float64)
        supports = [name for name in objects if name != held_name]
        support_name = min(
            supports,
            key=lambda name: np.linalg.norm(
                np.asarray(objects[name]["current_pos"], dtype=np.float64)[:2] - held_pos[:2]
            ),
        )
        support_pos = np.asarray(objects[support_name]["current_pos"], dtype=np.float64)
        current = np.asarray(env.get_obs()["robot_obs"][:6], dtype=np.float64)
        release = current.copy()
        release[:2] = support_pos[:2]
        release[2] = support_pos[2] + 0.055
        release[:3] += _stack_release_offset(seed)
        above = release.copy()
        above[2] = max(float(current[2]), float(support_pos[2] + 0.15))
        retreat = above.copy()
        yield from _interpolate_demo_pose(current, above, 24, -1.0)
        yield from _interpolate_demo_pose(above, release, 16, -1.0)
        for _ in range(10):
            yield np.concatenate((release, [1.0]))
        yield from _interpolate_demo_pose(release, retreat, 14, 1.0)
        return
    object_name, _ = _task_object_name(task)
    current_pos = np.asarray(
        env.get_info()["scene_info"]["movable_objects"][object_name]["current_pos"],
        dtype=np.float64,
    )
    if task.startswith("lift_"):
        key_actions = guidance[key]["actions"]
        close_indices = np.where(key_actions[:, -1] < 0)[0]
        if not len(close_indices):
            raise RuntimeError(f"Lift demonstration has no close-gripper phase: {key}")
        grasp = key_actions[int(close_indices[0]), :6].copy()
        grasp[:3] = current_pos + np.array([0.0, 0.0, 0.013])
        current = np.asarray(env.get_obs()["robot_obs"][:6], dtype=np.float64)
        above = grasp.copy()
        above[2] = current_pos[2] + 0.15
        lifted = grasp.copy()
        lifted[2] = current_pos[2] + 0.12
        yield from _interpolate_demo_pose(current, above, 18, 1.0)
        yield from _interpolate_demo_pose(above, grasp, 18, 1.0)
        for _ in range(10):
            yield np.concatenate((grasp, [-1.0]))
        yield from _interpolate_demo_pose(grasp, lifted, 20, -1.0)
        return
    canonical_color = {
        "lift_red_block_table": "red",
        "lift_blue_block_slider": "blue",
        "push_pink_block_right": "pink",
    }[key]
    scene_offset = {"red": 6, "blue": 12, "pink": 18}[canonical_color]
    demo_object_pos = guidance[key]["first_scene"][scene_offset : scene_offset + 3]
    translation = current_pos - demo_object_pos
    for action in guidance[key]["actions"]:
        target = action.copy()
        target[:3] += translation
        yield target


class FailureRecoveryWriter:
    """State-grouped branch dataset inspired by Sirius intervention segments."""

    def __init__(self, output_dir, resume=False):
        self.root = Path(output_dir).expanduser().resolve()
        nonempty = self.root.exists() and any(self.root.iterdir())
        if nonempty and not resume:
            raise FileExistsError(f"Recovery output is not empty: {self.root}")
        self.states_dir = self.root / "states"
        self.branches_dir = self.root / "branches"
        self.conditions_dir = self.root / "conditions"
        self.trajectory_chunks_dir = self.root / "trajectory_chunks"
        self.trajectory_conditions_dir = self.root / "trajectory_conditions"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.branches_dir.mkdir(parents=True, exist_ok=True)
        self.conditions_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_conditions_dir.mkdir(parents=True, exist_ok=True)
        self.states = self._read_manifest("failure_states.jsonl") if nonempty else []
        self.branches = self._read_manifest("branches.jsonl") if nonempty else []
        self.pairs = self._read_manifest("pairs.jsonl") if nonempty else []
        chunks_path = self.root / "trajectory_chunks.jsonl"
        self.trajectory_chunks = (
            self._read_manifest("trajectory_chunks.jsonl")
            if nonempty and chunks_path.is_file()
            else []
        )
        if nonempty:
            self._validate_resume_payloads()

    def _read_manifest(self, name):
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"Resume requires manifest: {path}")
        with path.open() as file:
            return [json.loads(line) for line in file if line.strip()]

    def _validate_resume_payloads(self):
        if (self.root / "collection_summary.json").exists():
            raise FileExistsError("Refusing to resume a finalized recovery dataset")
        state_ids = [item["failure_state_id"] for item in self.states]
        branch_ids = [item["branch_id"] for item in self.branches]
        if len(state_ids) != len(set(state_ids)) or len(branch_ids) != len(set(branch_ids)):
            raise ValueError("Resume manifests contain duplicate state or branch IDs")
        for state_id in state_ids:
            required = (
                self.states_dir / f"{state_id}.npz",
                self.states_dir / f"{state_id}.bullet",
                self.states_dir / f"{state_id}_model.pt",
                self.states_dir / f"{state_id}_simulator.pt",
            )
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(f"Resume state payload is incomplete: {state_id}")
        for branch_id in branch_ids:
            if not (self.branches_dir / f"{branch_id}.npz").is_file() or not (
                self.conditions_dir / f"{branch_id}.pt"
            ).is_file():
                raise FileNotFoundError(f"Resume branch payload is incomplete: {branch_id}")
        chunk_ids = [item["chunk_id"] for item in self.trajectory_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Resume trajectory manifest contains duplicate chunk IDs")
        for chunk_id in chunk_ids:
            if not (self.trajectory_chunks_dir / f"{chunk_id}.npz").is_file() or not (
                self.trajectory_conditions_dir / f"{chunk_id}.pt"
            ).is_file():
                raise FileNotFoundError(f"Resume trajectory payload is incomplete: {chunk_id}")

    def save_state(self, record, obs, previous_rgb, hist_action):
        state_id = record["failure_state_id"]
        np.savez_compressed(
            self.states_dir / f"{state_id}.npz",
            **_portable_obs(obs),
            previous_rgb=np.asarray(previous_rgb, dtype=np.uint8),
            hist_action=np.asarray(hist_action, dtype=np.float32),
        )
        self.states.append(record)

    def save_persistent_runtime(
        self,
        state_id,
        bullet,
        model_state,
        oracle_start_info=None,
        expected_robot_obs=None,
    ):
        if expected_robot_obs is not None:
            gripper_action = int(
                np.asarray(expected_robot_obs, dtype=np.float32)[-1].item()
            )
            bullet.robot.gripper_action = gripper_action
            bullet.robot.control_gripper(gripper_action)
        bullet_path = self.states_dir / f"{state_id}.bullet"
        bullet.p.saveBullet(bullet_path.as_posix(), physicsClientId=bullet.cid)
        torch.save(
            {name: _cpu_runtime_value(value) for name, value in model_state.items()},
            self.states_dir / f"{state_id}_model.pt",
        )
        torch.save(bullet.serialize(), self.states_dir / f"{state_id}_simulator.pt")
        if oracle_start_info is not None:
            torch.save(
                _cpu_runtime_value(oracle_start_info),
                self.states_dir / f"{state_id}_oracle_start.pt",
            )
        return bullet_path

    def save_branch(self, record, actions, final_obs, condition, trajectory_chunks=None):
        branch_id = record["branch_id"]
        np.savez_compressed(
            self.branches_dir / f"{branch_id}.npz",
            actions=np.asarray(actions, dtype=np.float32),
            final_robot_obs=np.asarray(final_obs["robot_obs"], dtype=np.float32),
            final_scene_obs=np.asarray(final_obs["scene_obs"], dtype=np.float32),
        )
        torch.save(condition, self.conditions_dir / f"{branch_id}.pt")
        saved_chunks = []
        for chunk_index, chunk in enumerate(trajectory_chunks or []):
            chunk_actions = np.asarray(chunk["actions"], dtype=np.float32)
            if not len(chunk_actions):
                continue
            chunk_id = f"{branch_id}_chunk_{chunk_index:03d}"
            np.savez_compressed(
                self.trajectory_chunks_dir / f"{chunk_id}.npz",
                **_portable_obs(chunk["obs"]),
                previous_rgb=np.asarray(chunk["previous_rgb"], dtype=np.uint8),
                hist_action=np.asarray(chunk["hist_action"], dtype=np.float32),
                actions=chunk_actions,
            )
            torch.save(
                chunk["condition"],
                self.trajectory_conditions_dir / f"{chunk_id}.pt",
            )
            self.trajectory_chunks.append({
                "chunk_id": chunk_id,
                "branch_id": branch_id,
                "failure_state_id": record["failure_state_id"],
                "split": record["split"],
                "task": record.get("task"),
                "strategy": record["strategy"],
                "chunk_index": chunk_index,
                "start_offset": int(chunk["start_offset"]),
                "steps": len(chunk_actions),
            })
            saved_chunks.append(chunk_id)
        record = dict(record)
        record["trajectory_chunk_ids"] = saved_chunks
        self.branches.append(record)

    def finalize_state_pairs(self, state_id, max_pairs):
        state_branches = [item for item in self.branches if item["failure_state_id"] == state_id]
        positives = [item for item in state_branches if item["success"]]
        negatives = [item for item in state_branches if not item["success"]]
        pair_count = 0
        for positive in positives:
            for negative in negatives:
                if pair_count >= max_pairs:
                    return pair_count
                self.pairs.append({
                    "pair_id": f"pair_{len(self.pairs):06d}",
                    "failure_state_id": state_id,
                    "split": positive["split"],
                    "positive_branch_id": positive["branch_id"],
                    "negative_branch_id": negative["branch_id"],
                })
                pair_count += 1
        return pair_count

    def checkpoint(self):
        for filename, rows in (
            ("failure_states.jsonl", self.states),
            ("branches.jsonl", self.branches),
            ("pairs.jsonl", self.pairs),
            ("trajectory_chunks.jsonl", self.trajectory_chunks),
        ):
            with (self.root / filename).open("w") as file:
                for row in rows:
                    file.write(json.dumps(row, sort_keys=True) + "\n")
        progress = {
            "status": "collecting",
            "failure_states": len(self.states),
            "branchable_failure_states": len(self.branchable_state_ids()),
            "branches": len(self.branches),
            "preference_pairs": len(self.pairs),
            "trajectory_chunks": len(self.trajectory_chunks),
        }
        (self.root / "collection_progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True)
        )

    def branchable_state_ids(self):
        return {item["failure_state_id"] for item in self.pairs}

    def branchable_split_counts(self):
        branchable = self.branchable_state_ids()
        return Counter(
            item["split"] for item in self.states if item["failure_state_id"] in branchable
        )

    def finalize(self, args, restore_audits, exact_branch_audits):
        self.checkpoint()
        branchable = self.branchable_state_ids()
        branchable_splits = self.branchable_split_counts()
        target_met = (
            len(self.states) >= args.target_failure_states
            and len(branchable) >= args.min_branchable_failure_states
            and all(
                branchable_splits[name] >= args.min_branchable_states_per_split
                for name in ("train", "validation", "test")
            )
        )
        summary = {
            "format": "robodual_failure_recovery_branch_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "complete" if target_met else "incomplete",
            "failure_states": len(self.states),
            "branchable_failure_states": len(branchable),
            "branches": len(self.branches),
            "successful_branches": sum(bool(item["success"]) for item in self.branches),
            "failed_branches": sum(not bool(item["success"]) for item in self.branches),
            "preference_pairs": len(self.pairs),
            "states_by_split": dict(Counter(item["split"] for item in self.states)),
            "branchable_states_by_split": dict(branchable_splits),
            "states_by_task": dict(Counter(item["task"] for item in self.states)),
            "branches_by_strategy": dict(Counter(item["strategy"] for item in self.branches)),
            "portable_restore_audits": restore_audits,
            "exact_branch_audits": exact_branch_audits,
            "args": vars(args),
            "integrity": {
                "branch_restore": "PyBullet in-memory saveState/restoreState plus evaluator runtime snapshot",
                "persistent_replay": "PyBullet .bullet world plus CPU evaluator runtime state",
                "portable_snapshot": "robot_obs/scene_obs and sensor arrays; reset reproducibility audited",
                "split": "SHA256 failure_state_id group split; no state crosses train/validation/test",
                "benchmark_exclusion": (
                    f"collection starts at sequence {args.recovery_sequence_start}; "
                    f"first {args.exclude_benchmark_sequences} canonical sequences excluded"
                ),
                "labels": "CALVIN task oracle success within branch_horizon",
            },
        }
        (self.root / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        progress = {
            "status": summary["status"],
            "failure_states": len(self.states),
            "branchable_failure_states": len(branchable),
            "branches": len(self.branches),
            "preference_pairs": len(self.pairs),
        }
        (self.root / "collection_progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True)
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary


def audit_portable_restore(env, snapshot_obs, repeats=3):
    results = []
    for _ in range(repeats):
        restored = env.reset(
            robot_obs=snapshot_obs["robot_obs"].copy(),
            scene_obs=snapshot_obs["scene_obs"].copy(),
        )
        results.append({
            "robot_max_abs": float(np.max(np.abs(restored["robot_obs"] - snapshot_obs["robot_obs"]))),
            "robot_ee6_max_abs": float(np.max(np.abs(
                restored["robot_obs"][:6] - snapshot_obs["robot_obs"][:6]
            ))),
            "robot_last_abs": float(abs(
                restored["robot_obs"][-1] - snapshot_obs["robot_obs"][-1]
            )),
            "scene_max_abs": float(np.max(np.abs(restored["scene_obs"] - snapshot_obs["scene_obs"]))),
            "rgb_static_mean_abs": float(np.mean(np.abs(
                restored["rgb_obs"]["rgb_static"].astype(np.float32)
                - snapshot_obs["rgb_obs"]["rgb_static"].astype(np.float32)
            ))),
        })
    return results


def recovery_collection_target_met(writer, args):
    counts = writer.branchable_split_counts()
    return (
        len(writer.states) >= args.target_failure_states
        and len(writer.branchable_state_ids()) >= args.min_branchable_failure_states
        and all(
            counts[name] >= args.min_branchable_states_per_split
            for name in ("train", "validation", "test")
        )
    )


def run_recovery_branch(
    env,
    model,
    bullet,
    candidate,
    task_oracle,
    start_info,
    task,
    instruction,
    strategy,
    seed,
    horizon,
    demo_guidance=None,
    required_success_streak=1,
):
    bullet.p.restoreState(stateId=candidate["bullet_state_id"], physicsClientId=bullet.cid)
    restore_recovery_model_state(model, candidate["model_state"])
    _set_branch_seed(seed)
    if strategy in {"forced_refresh", "slow_override"}:
        model.last_slow_step = None
    obs = env.get_obs()
    actions = []
    first_profile = None
    condition = None
    slow_override_chunk = None
    success = False
    success_streak = 0
    trajectory_chunks = []
    if strategy == "demo_guided":
        if demo_guidance is None or _demo_guidance_key(task) is None:
            raise ValueError(f"demo_guided is unsupported for task={task}")
        # Materialize model conditions at every trainable 8-step boundary.  Between
        # boundaries, keep image/action history synchronized with the expert action
        # that is actually executed in the environment.
        active_chunk = None
        for offset, target in enumerate(
            _retargeted_demo_targets(env, task, demo_guidance, seed)
        ):
            if offset % 8 == 0:
                active_chunk = {
                    "start_offset": offset,
                    "obs": copy.deepcopy(obs),
                    "previous_rgb": (
                        np.asarray(model.obs_buffer, dtype=np.uint8).copy()
                        if model.obs_buffer is not None
                        else np.asarray(
                            obs["rgb_obs"]["rgb_static"], dtype=np.uint8
                        ).copy()
                    ),
                    "hist_action": _history_array(model),
                    "condition": None,
                    "actions": [],
                }
                trajectory_chunks.append(active_chunk)
                model.step(obs, instruction, candidate["step"] + offset)
                chunk_condition = {
                    "slow_action": model.action.detach().to(torch.float32).cpu(),
                    "slow_hidden": model.hidden_states.detach().to(torch.float16).cpu(),
                    "slow_age": int(model.last_step_profile["slow_age_after"]),
                    "strategy": strategy,
                }
                active_chunk["condition"] = chunk_condition
                if offset == 0:
                    first_profile = _json_safe_profile(model.last_step_profile)
                    condition = chunk_condition
            action = _relative_demo_action(target, obs["robot_obs"])
            actions.append(action.copy())
            active_chunk["actions"].append(action.copy())
            action_tensor = torch.from_numpy(action.copy()).to(torch.float32)
            if offset % 8 == 0:
                model.hist_action[-1] = action_tensor
            else:
                model.hist_action.append(action_tensor)
            model.prev_action = action.copy()
            model.obs_buffer = np.asarray(
                obs["rgb_obs"]["rgb_static"], dtype=np.uint8
            ).copy()
            obs, _, _, current_info = env.step(action.copy())
            current_success = bool(
                task_oracle.get_task_info_for_set(start_info, current_info, {task})
            )
            success_streak = success_streak + 1 if current_success else 0
            if success_streak >= required_success_streak:
                success = True
            # Remember the event-like oracle success, but finish the current
            # trainable chunk so the terminal gripper phase is supervised.
            if success and len(active_chunk["actions"]) >= 8:
                break
            if len(actions) >= horizon:
                break
        return {
            "success": success,
            "actions": actions,
            "final_obs": obs,
            "first_profile": first_profile,
            "condition": condition,
            "trajectory_chunks": trajectory_chunks,
        }
    active_chunk = None
    for offset in range(horizon):
        absolute_step = candidate["step"] + offset
        if offset % 8 == 0:
            active_chunk = {
                "start_offset": offset,
                "obs": copy.deepcopy(obs),
                "previous_rgb": (
                    np.asarray(model.obs_buffer, dtype=np.uint8).copy()
                    if model.obs_buffer is not None
                    else np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8).copy()
                ),
                "hist_action": _history_array(model),
                "condition": None,
                "actions": [],
            }
            trajectory_chunks.append(active_chunk)
        action = np.asarray(model.step(obs, instruction, absolute_step), dtype=np.float32).reshape(7)
        if active_chunk["condition"] is None:
            active_chunk["condition"] = {
                "slow_action": model.action.detach().to(torch.float32).cpu(),
                "slow_hidden": model.hidden_states.detach().to(torch.float16).cpu(),
                "slow_age": int(model.last_step_profile["slow_age_after"]),
                "strategy": strategy,
            }
        if offset == 0:
            first_profile = _json_safe_profile(model.last_step_profile)
            condition = {
                "slow_action": model.action.detach().to(torch.float32).cpu(),
                "slow_hidden": model.hidden_states.detach().to(torch.float16).cpu(),
                "slow_age": int(model.last_step_profile["slow_age_after"]),
                "strategy": strategy,
            }
            if strategy == "slow_override":
                slow_override_chunk = model.action.detach().to(torch.float32).cpu().numpy()[0]
        if slow_override_chunk is not None and offset < len(slow_override_chunk):
            action = np.asarray(slow_override_chunk[offset], dtype=np.float32).copy()
            action[-1] = -1.0 if action[-1] < -0.5 else 1.0
            model.hist_action[-1] = torch.from_numpy(action.copy()).to(torch.float32)
            model.prev_action = action.copy()
        actions.append(action.copy())
        active_chunk["actions"].append(action.copy())
        obs, _, _, current_info = env.step(action.copy())
        current_success = bool(
            task_oracle.get_task_info_for_set(start_info, current_info, {task})
        )
        success_streak = success_streak + 1 if current_success else 0
        if success_streak >= required_success_streak:
            success = True
        if success and len(active_chunk["actions"]) >= 8:
            break
    return {
        "success": success,
        "actions": actions,
        "final_obs": obs,
        "first_profile": first_profile,
        "condition": condition,
        "trajectory_chunks": trajectory_chunks,
    }


def replay_recorded_action_branch(
    env, bullet, candidate, task_oracle, start_info, task, actions
):
    """Replay fixed actions to audit physics restore without resampling a policy."""

    bullet.p.restoreState(stateId=candidate["bullet_state_id"], physicsClientId=bullet.cid)
    obs = env.get_obs()
    first_success_step = None
    steps = 0
    for action in actions:
        obs, _, _, current_info = env.step(np.asarray(action, dtype=np.float32).copy())
        steps += 1
        if (
            first_success_step is None
            and task_oracle.get_task_info_for_set(start_info, current_info, {task})
        ):
            first_success_step = steps
    return {
        "success": first_success_step is not None,
        "steps": steps,
        "first_success_step": first_success_step,
        "final_obs": obs,
    }


def collect_failure_recovery_dataset(model, env, args):
    conf_dir = Path(CALVIN_ROOT) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    bullet = _bullet_env(env)
    writer = FailureRecoveryWriter(
        args.recovery_output_dir, resume=args.resume_recovery_collection
    )
    sequence_end = args.recovery_sequence_start + args.num_sequences
    if sequence_end > args.recovery_sequence_catalog_size:
        raise ValueError(
            "recovery sequence range exceeds --recovery_sequence_catalog_size"
        )
    catalog_path = writer.root / "sequence_catalog.json"
    catalogs_path = writer.root / "sequence_catalogs.json"
    catalog_id = args.recovery_sequence_catalog_id.strip()
    catalog_record = {
        "catalog_size": int(args.recovery_sequence_catalog_size),
        "generator": "calvin_agent.evaluation.multistep_sequences.get_sequences",
    }
    if catalogs_path.exists():
        catalogs = json.loads(catalogs_path.read_text())
    elif catalog_path.exists():
        legacy = json.loads(catalog_path.read_text())
        catalogs = {"legacy300" if legacy.get("catalog_size") == 300 else f"catalog{legacy['catalog_size']}": legacy}
    elif writer.states:
        raise FileNotFoundError(
            "Legacy recovery dataset has no sequence_catalog.json; explicitly backfill "
            "its original catalog before resuming"
        )
    else:
        catalogs = {}
    if not catalog_id:
        matching_ids = [name for name, record in catalogs.items() if record == catalog_record]
        catalog_id = matching_ids[0] if len(matching_ids) == 1 else f"catalog{args.recovery_sequence_catalog_size}"
    if catalog_id in catalogs and catalogs[catalog_id] != catalog_record:
        raise ValueError(
            f"Recovery sequence catalog mismatch for {catalog_id}: "
            f"{catalogs[catalog_id]} != {catalog_record}"
        )
    catalogs[catalog_id] = catalog_record
    catalogs_path.write_text(json.dumps(catalogs, indent=2, sort_keys=True))
    all_sequences = list(get_sequences(args.recovery_sequence_catalog_size))
    existing_sequence_keys = {
        (item.get("sequence_catalog_id", "legacy300"), int(item["sequence_i"]))
        for item in writer.states
    }
    if args.recovery_sequence_indices:
        requested_indices = [
            int(item.strip())
            for item in args.recovery_sequence_indices.split(",")
            if item.strip()
        ]
        if len(requested_indices) != len(set(requested_indices)):
            raise ValueError("recovery sequence indices must be unique")
        invalid = [
            index for index in requested_indices
            if index < args.recovery_sequence_start or index >= sequence_end
        ]
        if invalid:
            raise ValueError(f"recovery sequence indices outside collection range: {invalid}")
        eval_sequences = [
            (index, all_sequences[index])
            for index in requested_indices
            if (catalog_id, index) not in existing_sequence_keys
        ]
    else:
        resume_start = args.recovery_sequence_start
        catalog_sequence_ids = [
            sequence_i for item_catalog, sequence_i in existing_sequence_keys
            if item_catalog == catalog_id
        ]
        if catalog_sequence_ids:
            resume_start = max(catalog_sequence_ids) + 1
        eval_sequences = [
            (index, all_sequences[index]) for index in range(resume_start, sequence_end)
        ]
    restore_audits = []
    exact_branch_audits = []
    branch_strategies = parse_branch_strategies(args.branch_strategies)
    demo_guidance = (
        load_demo_guidance(args.dataset_subdir)
        if "demo_guided" in branch_strategies
        else None
    )
    recovery_task_allowlist = {
        item.strip() for item in args.recovery_task_allowlist.split(",") if item.strip()
    }
    recovery_stop_after_tasks = {
        item.strip() for item in args.recovery_stop_after_tasks.split(",") if item.strip()
    }
    if not recovery_stop_after_tasks:
        recovery_stop_after_tasks = set(recovery_task_allowlist)
    failure_count = 0

    for local_sequence_i, (sequence_i, (initial_state, sequence)) in enumerate(eval_sequences):
        sequence_seed = (
            args.seed
            + int(hashlib.sha256(f"{catalog_id}:{sequence_i}".encode()).hexdigest()[:8], 16)
        ) % (2**31 - 1)
        _set_branch_seed(sequence_seed)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        for subtask_i, task in enumerate(sequence[: args.max_subtasks or len(sequence)]):
            instruction = annotations[task][0]
            model.reset()
            model.set_current_task(task)
            obs = env.get_obs()
            start_info = env.get_info()
            candidates = deque()
            succeeded = False
            for step in range(args.ep_len):
                slow_age = (
                    None
                    if model.last_slow_step is None
                    else int(step - model.last_slow_step)
                )
                age_eligible = (
                    step == 0 and args.failure_state_min_step == 0
                ) or (
                    slow_age is not None
                    and int(slow_age) >= args.failure_state_min_age
                )
                eligible = (
                    step >= args.failure_state_min_step
                    and age_eligible
                    and step % args.failure_state_stride == 0
                    and len(candidates) < args.states_per_failed_subtask
                    and (
                        not candidates
                        or step - candidates[-1]["step"] >= args.failure_state_spacing
                    )
                )
                if eligible:
                    state_prefix = "" if catalog_id == "legacy300" else f"{catalog_id}_"
                    state_id = f"{state_prefix}s{sequence_i:04d}_t{subtask_i}_k{step:03d}"
                    saved = {
                        "failure_state_id": state_id,
                        "sequence_i": sequence_i,
                        "sequence_catalog_id": catalog_id,
                        "sequence_catalog_size": int(args.recovery_sequence_catalog_size),
                        "sequence_seed": int(sequence_seed),
                        "subtask_i": subtask_i,
                        "task": task,
                        "instruction": instruction,
                        "step": step,
                        "slow_age": -1 if slow_age is None else int(slow_age),
                        "bullet_state_id": bullet.p.saveState(physicsClientId=bullet.cid),
                        "model_state": capture_recovery_model_state(model),
                        "obs": copy.deepcopy(obs),
                        "previous_rgb": (
                            np.asarray(model.obs_buffer, dtype=np.uint8).copy()
                            if model.obs_buffer is not None
                            else np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8).copy()
                        ),
                        "hist_action": _history_array(model),
                        # Preserve the actual baseline continuation from this
                        # exact snapshot. If the subtask ultimately fails, its
                        # first recovery-horizon actions are a real same-state
                        # negative branch rather than a synthetic no-op.
                        "original_actions": [],
                        "original_condition": None,
                        "original_first_profile": None,
                        "original_final_obs": None,
                    }
                    candidates.append(saved)

                action = np.asarray(model.step(obs, instruction, step), dtype=np.float32).reshape(7)
                for original_candidate in candidates:
                    if len(original_candidate["original_actions"]) >= args.branch_horizon:
                        continue
                    if not original_candidate["original_actions"]:
                        original_candidate["original_first_profile"] = _json_safe_profile(
                            model.last_step_profile
                        )
                        original_candidate["original_condition"] = {
                            "slow_action": model.action.detach().to(torch.float32).cpu(),
                            "slow_hidden": model.hidden_states.detach().to(torch.float16).cpu(),
                            "slow_age": int(model.last_step_profile["slow_age_after"]),
                            "strategy": "baseline_failed_continuation",
                        }
                    original_candidate["original_actions"].append(action.copy())
                obs, _, _, current_info = env.step(action.copy())
                for original_candidate in candidates:
                    if (
                        len(original_candidate["original_actions"]) == args.branch_horizon
                        and original_candidate["original_final_obs"] is None
                    ):
                        original_candidate["original_final_obs"] = copy.deepcopy(obs)
                if task_oracle.get_task_info_for_set(start_info, current_info, {task}):
                    succeeded = True
                    break

            if succeeded:
                while candidates:
                    dropped = candidates.popleft()
                    bullet.p.removeState(dropped["bullet_state_id"], physicsClientId=bullet.cid)
                if recovery_stop_after_tasks and task in recovery_stop_after_tasks:
                    print(
                        f"[recovery] target_task_succeeded sequence={sequence_i} task={task} "
                        "reason=no_failure_state",
                        flush=True,
                    )
                    # Explicit targeted acquisition admits at most one state per
                    # source sequence. Once the selected task succeeds, later
                    # subtasks cannot contribute an eligible state for this phase.
                    break
                continue

            failure_count += 1
            if recovery_task_allowlist and task not in recovery_task_allowlist:
                while candidates:
                    dropped = candidates.popleft()
                    bullet.p.removeState(dropped["bullet_state_id"], physicsClientId=bullet.cid)
                print(
                    f"[recovery] skip_failed_task sequence={sequence_i} task={task} "
                    "reason=not_in_allowlist",
                    flush=True,
                )
                # A failed subtask terminates the original CALVIN sequence even when
                # its task is not selected for expensive recovery branching.
                break

            for candidate in list(candidates):
                if (
                    recovery_collection_target_met(writer, args)
                    or len(writer.states) >= args.max_failure_states_scanned
                ):
                    break
                state_id = candidate["failure_state_id"]
                split = _state_split(state_id)
                exact_state_id = candidate["bullet_state_id"]
                if len(restore_audits) < args.restore_audit_states:
                    audit = audit_portable_restore(env, candidate["obs"], args.restore_audit_repeats)
                    restore_audits.append({"failure_state_id": state_id, "repeats": audit})
                    bullet.p.restoreState(stateId=exact_state_id, physicsClientId=bullet.cid)

                state_record = {
                    "failure_state_id": state_id,
                    "split": split,
                    "sequence_i": int(candidate["sequence_i"]),
                    "sequence_catalog_id": candidate["sequence_catalog_id"],
                    "sequence_catalog_size": int(candidate["sequence_catalog_size"]),
                    "sequence_seed": int(candidate["sequence_seed"]),
                    "subtask_i": int(candidate["subtask_i"]),
                    "task": task,
                    "instruction": instruction,
                    "step": int(candidate["step"]),
                    "slow_age": int(candidate["slow_age"]),
                    "baseline_subtask_failed": True,
                }
                bullet.p.restoreState(stateId=exact_state_id, physicsClientId=bullet.cid)
                writer.save_persistent_runtime(
                    state_id,
                    bullet,
                    candidate["model_state"],
                    oracle_start_info=start_info,
                    expected_robot_obs=candidate["obs"]["robot_obs"],
                )
                writer.save_state(
                    state_record,
                    candidate["obs"],
                    candidate["previous_rgb"],
                    candidate["hist_action"],
                )
                original_actions = candidate["original_actions"]
                if original_actions:
                    original_branch_id = f"{state_id}_baseline_failed_continuation_00"
                    writer.save_branch(
                        {
                            "branch_id": original_branch_id,
                            "failure_state_id": state_id,
                            "split": split,
                            "strategy": "baseline_failed_continuation",
                            "seed": int(candidate["sequence_seed"]),
                            "success": False,
                            "steps": len(original_actions),
                            "first_profile": candidate["original_first_profile"],
                            "source_subtask_failure_confirmed": True,
                        },
                        original_actions,
                        candidate["original_final_obs"] or obs,
                        candidate["original_condition"],
                    )
                for branch_index in range(args.branches_per_strategy):
                    stable_positive_found = False
                    for strategy_index, strategy in enumerate(branch_strategies):
                        branch_seed = (
                            args.seed
                            + sequence_i * 100003
                            + subtask_i * 1009
                            + candidate["step"] * 31
                            + branch_index * len(branch_strategies)
                            + strategy_index
                        )
                        result = run_recovery_branch(
                            env, model, bullet, candidate, task_oracle, start_info,
                            task, instruction, strategy, branch_seed, args.branch_horizon,
                            demo_guidance=demo_guidance,
                            required_success_streak=args.recovery_success_streak,
                        )
                        repeated = None
                        branch_success = bool(result["success"])
                        if branch_success and args.require_in_process_stable_positive:
                            repeated = replay_recorded_action_branch(
                                env, bullet, candidate, task_oracle, start_info,
                                task, result["actions"],
                            )
                            branch_success = bool(
                                repeated["success"]
                                and repeated["steps"] == len(result["actions"])
                            )
                        branch_id = f"{state_id}_{strategy}_{branch_index:02d}"
                        writer.save_branch(
                            {
                                "branch_id": branch_id,
                                "failure_state_id": state_id,
                                "split": split,
                                "strategy": strategy,
                                "task": task,
                                "seed": int(branch_seed),
                                "success": branch_success,
                                "rollout_success": bool(result["success"]),
                                "in_process_fixed_action_replay": (
                                    None
                                    if repeated is None
                                    else {
                                        "same_outcome": bool(
                                            result["success"] == repeated["success"]
                                        ),
                                        "same_length": bool(
                                            len(result["actions"]) == repeated["steps"]
                                        ),
                                    }
                                ),
                                "steps": len(result["actions"]),
                                "first_profile": result["first_profile"],
                            },
                            result["actions"],
                            result["final_obs"],
                            result["condition"],
                            trajectory_chunks=result["trajectory_chunks"],
                        )
                        if (
                            len(exact_branch_audits) < args.exact_branch_audit_states
                            and branch_index == 0
                            and strategy == "base_seed"
                        ):
                            if repeated is None:
                                repeated = replay_recorded_action_branch(
                                    env, bullet, candidate, task_oracle, start_info,
                                    task, result["actions"],
                                )
                            robot_max_abs = float(np.max(np.abs(
                                result["final_obs"]["robot_obs"] - repeated["final_obs"]["robot_obs"]
                            )))
                            scene_max_abs = float(np.max(np.abs(
                                result["final_obs"]["scene_obs"] - repeated["final_obs"]["scene_obs"]
                            )))
                            exact_branch_audits.append({
                                "failure_state_id": state_id,
                                "strategy": strategy,
                                "seed": int(branch_seed),
                                "same_outcome": bool(result["success"] == repeated["success"]),
                                "same_length": bool(len(result["actions"]) == repeated["steps"]),
                                "fixed_action_replay": True,
                                "final_robot_max_abs": robot_max_abs,
                                "final_scene_max_abs": scene_max_abs,
                            })
                        stable_positive_found = stable_positive_found or branch_success
                    if (
                        stable_positive_found
                        and args.stop_after_stable_positive_per_state
                    ):
                        break
                writer.finalize_state_pairs(state_id, args.max_pairs_per_state)
                writer.checkpoint()
                bullet.p.restoreState(stateId=exact_state_id, physicsClientId=bullet.cid)
                bullet.p.removeState(exact_state_id, physicsClientId=bullet.cid)
            for candidate in list(candidates):
                try:
                    bullet.p.removeState(candidate["bullet_state_id"], physicsClientId=bullet.cid)
                except Exception:
                    # Processed candidates were already removed after branching.
                    pass
            if (
                recovery_collection_target_met(writer, args)
                or len(writer.states) >= args.max_failure_states_scanned
            ):
                break
            # A failed subtask terminates the original CALVIN sequence.
            break
        if (
            recovery_collection_target_met(writer, args)
            or len(writer.states) >= args.max_failure_states_scanned
        ):
            break
        print(
            f"[recovery] sequences={local_sequence_i + 1} failed_subtasks={failure_count} "
            f"states={len(writer.states)} pairs={len(writer.pairs)}",
            flush=True,
        )
    if args.defer_recovery_finalize:
        writer.checkpoint()
        deferred = {
            "status": "collecting",
            "deferred_finalize": True,
            "failure_states": len(writer.states),
            "branchable_failure_states": len(writer.branchable_state_ids()),
            "branches": len(writer.branches),
            "preference_pairs": len(writer.pairs),
        }
        print(json.dumps(deferred, indent=2, sort_keys=True), flush=True)
        return deferred
    return writer.finalize(args, restore_audits, exact_branch_audits)


def replay_failure_recovery_dataset(model, env, args):
    root = Path(args.replay_recovery_dir).expanduser().resolve()
    output = Path(args.replay_output).expanduser().resolve()
    specialist_path = Path(args.specialist_path).expanduser().resolve().as_posix()
    stable_manifest_path = root / "stable_filter_manifest.json"
    assessment_path = root / "dataset_assessment.json"
    if not stable_manifest_path.is_file() or not assessment_path.is_file():
        raise FileNotFoundError(
            "Online replay requires a frozen stable_filter_manifest.json and "
            "a post-filter dataset_assessment.json"
        )
    stable_manifest = json.loads(stable_manifest_path.read_text())
    assessment = json.loads(assessment_path.read_text())
    if not assessment.get("training_admitted"):
        raise RuntimeError("Online replay dataset did not pass training admission")
    with (root / "failure_states.jsonl").open() as file:
        all_states = [json.loads(line) for line in file if line.strip()]
    frozen_state_ids = set(stable_manifest.get("kept_state_ids", []))
    actual_state_ids = {item["failure_state_id"] for item in all_states}
    if actual_state_ids != frozen_state_ids:
        raise RuntimeError(
            "Frozen replay inventory mismatch: "
            f"manifest_only={sorted(frozen_state_ids - actual_state_ids)} "
            f"data_only={sorted(actual_state_ids - frozen_state_ids)}"
        )
    states = all_states
    states = [item for item in states if item["split"] == args.replay_split]
    if args.replay_max_states is not None:
        states = states[: args.replay_max_states]
    if not states:
        raise ValueError(f"No recovery states for split={args.replay_split!r}")
    conf_dir = Path(CALVIN_ROOT) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    bullet = _bullet_env(env)
    replay_config = {
        "data_dir": root.as_posix(),
        "generalist_path": Path(args.generalist_path).expanduser().resolve().as_posix(),
        "specialist_path": specialist_path,
        "dataset_subdir": args.dataset_subdir,
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
        "fast_num_inference_steps": int(args.fast_num_inference_steps),
        "slow_call_strategy": args.slow_call_strategy,
        "slow_trigger_policy": args.slow_trigger_policy,
        "max_slow_age": int(args.max_slow_age),
        "empty_ref_after_age": int(args.empty_ref_after_age),
        "split": args.replay_split,
        "horizon": args.replay_horizon,
        "seeds_per_state": args.replay_seeds_per_state,
        "seed": args.seed,
        "states": len(states),
        "stable_manifest_sha256": hashlib.sha256(
            stable_manifest_path.read_bytes()
        ).hexdigest(),
        "restore_contract": "bullet_reset_bullet_v2_gripper_v3",
        "oracle_contract": (
            "persisted_subtask_start_or_explicit_fallback"
            if args.replay_allow_missing_oracle_start
            else "persisted_subtask_start_required"
        ),
    }
    records = []
    if args.resume_recovery_replay and output.exists():
        existing = json.loads(output.read_text())
        mismatches = {
            key: (existing.get(key), expected)
            for key, expected in replay_config.items()
            if existing.get(key) != expected
        }
        if existing.get("format") != "robodual_failure_recovery_replay_v1":
            mismatches["format"] = (
                existing.get("format"),
                "robodual_failure_recovery_replay_v1",
            )
        if mismatches:
            raise ValueError(f"Cannot resume replay with changed configuration: {mismatches}")
        records = list(existing.get("records", []))
    completed = {
        (item["failure_state_id"], int(item["seed"]))
        for item in records
    }

    def checkpoint_replay(status):
        summary = {
            "format": "robodual_failure_recovery_replay_v1",
            **replay_config,
            "status": status,
            "rollouts": len(records),
            "successes": sum(item["success"] for item in records),
            "success_rate": (
                float(np.mean([item["success"] for item in records]))
                if records else 0.0
            ),
            "records": records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True))
        temporary.replace(output)
        return summary

    for state_index, state in enumerate(states, start=1):
        state_id = state["failure_state_id"]
        for seed_index in range(args.replay_seeds_per_state):
            seed = args.seed + int(state["sequence_i"]) * 100003 + int(state["step"]) * 31 + seed_index
            if (state_id, seed) in completed:
                continue
            restored_obs = restore_persisted_failure_state(
                env, bullet, root / "states", state_id
            )
            runtime = torch.load(
                root / "states" / f"{state_id}_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            restore_persistent_model_state(model, runtime)
            model.set_current_task(state["task"])
            _set_branch_seed(seed)
            obs = env.get_obs()
            start_info, oracle_source = load_persisted_oracle_start(
                root / "states",
                state_id,
                env,
                allow_missing=args.replay_allow_missing_oracle_start,
            )
            success = False
            for offset in range(args.replay_horizon):
                action = np.asarray(
                    model.step(obs, state["instruction"], int(state["step"]) + offset),
                    dtype=np.float32,
                ).reshape(7)
                obs, _, _, current_info = env.step(action.copy())
                if task_oracle.get_task_info_for_set(start_info, current_info, {state["task"]}):
                    success = True
                    break
            records.append({
                "failure_state_id": state_id,
                "split": state["split"],
                "task": state["task"],
                "seed": seed,
                "success": success,
                "steps": offset + 1,
                "oracle_source": oracle_source,
            })
            completed.add((state_id, seed))
            checkpoint_replay("collecting")
            print(
                f"[replay] state={state_index}/{len(states)} seed={seed_index + 1}/"
                f"{args.replay_seeds_per_state} rollouts={len(records)} "
                f"successes={sum(item['success'] for item in records)}",
                flush=True,
            )
    summary = checkpoint_replay("complete")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def augment_persisted_demo_branches(model, env, args):
    """Append audited demo-guided positives to persisted, unbranchable states."""
    writer = FailureRecoveryWriter(args.recovery_output_dir, resume=True)
    guidance = load_demo_guidance(args.dataset_subdir)
    task_cfg = OmegaConf.load(
        Path(CALVIN_ROOT) / "calvin_models" / "conf" / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    bullet = _bullet_env(env)
    allowlist = {item.strip() for item in args.recovery_task_allowlist.split(",") if item.strip()}
    selected_state_ids = {item.strip() for item in args.augment_state_ids.split(",") if item.strip()}
    already_branchable = writer.branchable_state_ids()
    augmented = []
    for state in writer.states:
        state_id = state["failure_state_id"]
        task = state["task"]
        if state_id in already_branchable or (allowlist and task not in allowlist):
            continue
        if selected_state_ids and state_id not in selected_state_ids:
            continue
        if _demo_guidance_key(task) is None:
            continue
        oracle_path = writer.states_dir / f"{state_id}_oracle_start.pt"
        if not oracle_path.is_file():
            if selected_state_ids and state_id in selected_state_ids:
                raise FileNotFoundError(
                    f"Cannot augment selected legacy state without original oracle baseline: {state_id}"
                )
            continue
        obs = restore_persisted_failure_state(
            env, bullet, writer.states_dir, state_id
        )
        runtime = torch.load(
            writer.states_dir / f"{state_id}_model.pt", map_location="cpu", weights_only=False
        )
        restore_persistent_model_state(model, runtime)
        model.set_current_task(task)
        start_info, oracle_source = load_persisted_oracle_start(
            writer.states_dir,
            state_id,
            env,
        )
        in_memory_state = bullet.p.saveState(physicsClientId=bullet.cid)
        candidate = {
            "bullet_state_id": in_memory_state,
            "model_state": capture_recovery_model_state(model),
            "step": int(state["step"]),
        }
        successes = 0
        try:
            for branch_index in range(args.branches_per_strategy):
                seed = args.seed + int(state["sequence_i"]) * 100003 + int(state["step"]) * 31 + branch_index
                result = run_recovery_branch(
                    env, model, bullet, candidate, task_oracle, start_info, task,
                    state["instruction"], "demo_guided", seed, args.branch_horizon,
                    demo_guidance=guidance,
                )
                branch_id = f"{state_id}_demo_guided_persisted_{branch_index:02d}"
                writer.save_branch(
                    {
                        "branch_id": branch_id,
                        "failure_state_id": state_id,
                        "split": state["split"],
                        "strategy": "demo_guided_persisted",
                        "seed": int(seed),
                        "success": bool(result["success"]),
                        "steps": len(result["actions"]),
                        "first_profile": result["first_profile"],
                        "oracle_source": oracle_source,
                        "restore_contract": "bullet_reset_bullet_v2_gripper_v3",
                    },
                    result["actions"],
                    result["final_obs"],
                    result["condition"],
                    trajectory_chunks=result["trajectory_chunks"],
                )
                successes += int(result["success"])
        finally:
            bullet.p.removeState(in_memory_state, physicsClientId=bullet.cid)
        writer.finalize_state_pairs(state_id, args.max_pairs_per_state)
        writer.checkpoint()
        augmented.append({"failure_state_id": state_id, "task": task, "successes": successes})
    result = {"augmented": augmented, "progress": json.loads((writer.root / "collection_progress.json").read_text())}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def rebuild_persisted_positive_trajectories(model, env, args):
    """Replay paired branches and rebuild complete, trainable 8-step windows."""

    writer = FailureRecoveryWriter(args.recovery_output_dir, resume=True)
    positive_ids = {
        item["positive_branch_id"] for item in writer.pairs
    }
    negative_ids = {
        item["negative_branch_id"] for item in writer.pairs
    }
    rebuild_ids = positive_ids | (
        negative_ids if args.rebuild_include_negatives else set()
    )
    if not positive_ids:
        raise ValueError("Trajectory rebuild requires paired positive branches")
    selected_branches = [
        item for item in writer.branches if item["branch_id"] in rebuild_ids
    ]
    missing = rebuild_ids - {item["branch_id"] for item in selected_branches}
    if missing:
        raise ValueError(f"Pairs reference missing positive branches: {sorted(missing)}")

    task_cfg = OmegaConf.load(
        Path(CALVIN_ROOT)
        / "calvin_models"
        / "conf"
        / "callbacks"
        / "rollout"
        / "tasks"
        / "new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    bullet = _bullet_env(env)
    states = {item["failure_state_id"]: item for item in writer.states}
    old_chunks = [
        item for item in writer.trajectory_chunks
        if item["branch_id"] in rebuild_ids
    ]
    writer.branches = [
        item for item in writer.branches if item["branch_id"] not in rebuild_ids
    ]
    writer.trajectory_chunks = [
        item for item in writer.trajectory_chunks
        if item["branch_id"] not in rebuild_ids
    ]
    for chunk in old_chunks:
        for path in (
            writer.trajectory_chunks_dir / f"{chunk['chunk_id']}.npz",
            writer.trajectory_conditions_dir / f"{chunk['chunk_id']}.pt",
        ):
            if path.is_file():
                path.unlink()

    rebuilt = []
    for branch_index, branch in enumerate(selected_branches, start=1):
        state_id = branch["failure_state_id"]
        state = states[state_id]
        obs = restore_persisted_failure_state(
            env, bullet, writer.states_dir, state_id
        )
        runtime = torch.load(
            writer.states_dir / f"{state_id}_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        restore_persistent_model_state(model, runtime)
        model.set_current_task(state["task"])
        start_info, oracle_source = load_persisted_oracle_start(
            writer.states_dir, state_id, env
        )
        with np.load(
            writer.branches_dir / f"{branch['branch_id']}.npz",
            allow_pickle=False,
        ) as payload:
            original_actions = np.asarray(payload["actions"], dtype=np.float32)
        if branch["success"]:
            actions, padding_steps = _terminal_padded_actions(original_actions)
        else:
            actions, padding_steps = original_actions.copy(), 0
            if len(actions) % 8:
                raise ValueError(
                    f"Negative branch is not chunk aligned: {branch['branch_id']}"
                )
        chunks = []
        active_chunk = None
        first_success_step = None
        for offset, action in enumerate(actions):
            if offset % 8 == 0:
                active_chunk = {
                    "start_offset": offset,
                    "obs": copy.deepcopy(obs),
                    "previous_rgb": (
                        np.asarray(model.obs_buffer, dtype=np.uint8).copy()
                        if model.obs_buffer is not None
                        else np.asarray(
                            obs["rgb_obs"]["rgb_static"], dtype=np.uint8
                        ).copy()
                    ),
                    "hist_action": _history_array(model),
                    "condition": None,
                    "actions": [],
                }
                chunks.append(active_chunk)
                model.step(
                    obs,
                    state["instruction"],
                    int(state["step"]) + offset,
                )
                active_chunk["condition"] = {
                    "slow_action": model.action.detach().to(torch.float32).cpu(),
                    "slow_hidden": model.hidden_states.detach().to(torch.float16).cpu(),
                    "slow_age": int(model.last_step_profile["slow_age_after"]),
                    "strategy": branch["strategy"],
                }
            action = np.asarray(action, dtype=np.float32).copy()
            active_chunk["actions"].append(action)
            action_tensor = torch.from_numpy(action.copy()).to(torch.float32)
            if offset % 8 == 0:
                model.hist_action[-1] = action_tensor
            else:
                model.hist_action.append(action_tensor)
            model.prev_action = action.copy()
            model.obs_buffer = np.asarray(
                obs["rgb_obs"]["rgb_static"], dtype=np.uint8
            ).copy()
            obs, _, _, current_info = env.step(action.copy())
            if (
                first_success_step is None
                and task_oracle.get_task_info_for_set(
                    start_info, current_info, {state["task"]}
                )
            ):
                first_success_step = offset + 1
        if bool(first_success_step is not None) != bool(branch["success"]):
            raise RuntimeError(
                f"Branch outcome changed during trajectory rebuild: "
                f"{branch['branch_id']} expected={branch['success']} "
                f"actual={first_success_step is not None}"
            )
        if any(len(item["actions"]) != 8 for item in chunks):
            raise RuntimeError(
                f"Rebuilt trajectory has an incomplete chunk: {branch['branch_id']}"
            )
        rebuilt_record = {
            **branch,
            "task": state["task"],
            "steps": len(actions),
            "original_steps": int(branch.get("original_steps", len(original_actions))),
            "terminal_padding_steps": int(padding_steps),
            "first_success_step": (
                None if first_success_step is None else int(first_success_step)
            ),
            "trajectory_rebuilt": True,
            "oracle_source": oracle_source,
            "restore_contract": "bullet_reset_bullet_v2_gripper_v3",
        }
        writer.save_branch(
            rebuilt_record,
            actions,
            obs,
            chunks[0]["condition"],
            trajectory_chunks=chunks,
        )
        rebuilt.append({
            "branch_id": branch["branch_id"],
            "state_id": state_id,
            "task": state["task"],
            "original_steps": len(original_actions),
            "rebuilt_steps": len(actions),
            "terminal_padding_steps": padding_steps,
            "first_success_step": first_success_step,
            "chunks": len(chunks),
        })
        print(
            f"[trajectory-rebuild] {branch_index}/{len(selected_branches)} "
            f"{branch['branch_id']} steps={len(original_actions)}->{len(actions)} "
            f"chunks={len(chunks)}",
            flush=True,
        )
    writer.checkpoint()
    result = {
        "format": "robodual_recovery_trajectory_rebuild_v1",
        "status": "audit_required",
        "restore_contract": "bullet_reset_bullet_v2_gripper_v3",
        "positive_branches": sum(
            item["branch_id"] in positive_ids for item in rebuilt
        ),
        "negative_branches": sum(
            item["branch_id"] in negative_ids for item in rebuilt
        ),
        "included_negative_branches": bool(args.rebuild_include_negatives),
        "rebuilt": rebuilt,
    }
    (writer.root / "trajectory_rebuild_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    for stale_name in (
        "persistent_replay_audit.json",
        "dataset_assessment.json",
        "dataset_assessment.md",
    ):
        stale_path = writer.root / stale_name
        if stale_path.is_file():
            stale_path.unlink()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main(args):
    # Set seed #42
    profiler = InitProfiler(args.profile_init)
    profiler.mark("main_start", {"pid": os.getpid()})
    seed_everything(42)
    profiler.mark("seed_everything_done")

    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=12))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    profiler.mark(
        "accelerator_initialized",
        {
            "device": str(device),
            "num_processes": acc.num_processes,
            "process_index": acc.process_index,
        },
    )


    # Load generalist policy
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    profiler.mark("transformers_imported")
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
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    model_kwargs = dict(
        torch_dtype=model_dtype,
        quantization_config=quantization_config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=True,
    )
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation
    profiler.mark(
        "generalist_config_ready",
        {
            "dtype": str(model_dtype),
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "device_map": args.device_map,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
        },
    )
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    profiler.mark("processor_loaded", {"generalist_path": args.generalist_path})
    model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **model_kwargs)
    model.eval()
    profiler.mark("generalist_loaded")

    # Load specialist policy
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers import DPMSolverMultistepScheduler
    profiler.mark("specialist_modules_imported")

    scheduler = DDIMScheduler( num_train_timesteps = 100, beta_schedule = 'squaredcos_cap_v2', prediction_type="epsilon" )
    shape_meta = {'action' : {'shape': [7]}}
    diffusion_policy = DiffusionDiTImagePolicy( shape_meta = shape_meta,
                                                noise_scheduler = scheduler,
                                                n_action_steps=8, 
                                                num_inference_steps=args.fast_num_inference_steps,
                                                vision_encoder='DINO',
                                                with_depth=args.with_depth,
                                                progressive_noise=False,
                                                with_gripper=args.with_gripper,
                                                with_tactile=args.with_tactile,
                                                cond_drop_chance=0.1 if args.with_cfg else 0.,  
                                                # set cond_drop_chance > 0 to activate CFG
                                              ).eval().to(device)
    profiler.mark("specialist_model_initialized", {"fast_num_inference_steps": args.fast_num_inference_steps})
   

    from prismatic.vla.action_tokenizer import ActionTokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    profiler.mark("action_tokenizer_ready")

    from train_spacialist_calvin import DualSystem
    dual_sys = DualSystem(model, diffusion_policy, action_tokenizer)
    profiler.mark("dual_system_constructed")
    specialist_state = torch.load(args.specialist_path)
    profiler.mark("specialist_checkpoint_loaded", {"specialist_path": args.specialist_path})
    dual_sys.ema_fast_system.load_state_dict(specialist_state, strict=False)
    profiler.mark("specialist_state_dict_applied")

    dual_sys = acc.prepare(dual_sys, device_placement=[True])
    profiler.mark("accelerate_prepare_done")

    save_path = REPO_ROOT / 'evaluation_results'
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper', ],  # rgb_tactile
        'depth_obs': ['depth_static', 'depth_gripper'], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}
    eval_dir = save_path / f'eval{torch.cuda.current_device()}'
    os.makedirs(eval_dir, exist_ok=True)
    profiler.mark("eval_dirs_ready", {"eval_dir": eval_dir.as_posix()})
    env = make_env(os.path.join(CALVIN_ROOT, f"dataset/{args.dataset_subdir}"), observation_space, device, args.use_egl)
    profiler.mark("environment_created", {"dataset_subdir": args.dataset_subdir, "use_egl": args.use_egl})
    profile_output = None
    if args.profile_steps:
        profile_output = save_path / f"specialist_profile_rank{acc.process_index}.jsonl"
        with open(profile_output, "w") as file:
            file.write("")
        profiler.mark("profile_output_ready", {"profile_output": profile_output.as_posix()})
    eval_sr_path = save_path / f"success_rate_rank{acc.process_index}.txt"
    eval_result_path = save_path / f"result_rank{acc.process_index}.json"
    with open(eval_sr_path, "w") as file:
        file.write("")
    task_age_config = build_task_age_config(args)
    eva = TaskAgeDualSystemEvaluation(
        dual_sys,
        processor,
        action_tokenizer,
        task_age_config=task_age_config,
        profile_steps=args.profile_steps,
        profile_sample_var_k=args.profile_sample_var_k,
        profile_sample_var_interval=args.profile_sample_var_interval,
        profile_sample_var_ages=args.profile_sample_var_ages,
        slow_trigger_policy=args.slow_trigger_policy,
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
    if args.profile_steps:
        emit_profile_record(
            profile_output,
            {
                "event": "run_config",
                "rank": int(acc.process_index),
                "entrypoint": "evaluate_calvin_failure_recovery_0718.py",
                "dataset_subdir": args.dataset_subdir,
                "num_sequences": int(args.num_sequences),
                "ep_len": int(args.ep_len),
                "max_subtasks": None if args.max_subtasks is None else int(args.max_subtasks),
                "slow_call_strategy": args.slow_call_strategy,
                "slow_trigger_policy_arg": args.slow_trigger_policy,
                "effective_slow_trigger_policy": eva.slow_trigger_policy,
                "max_slow_age": int(args.max_slow_age),
                "empty_ref_after_age": int(args.empty_ref_after_age),
                "min_slow_age": int(args.min_slow_age),
                "risk_start_age": int(args.risk_start_age),
                "risk_score_threshold": int(args.risk_score_threshold),
                "risk_late_age": int(args.risk_late_age),
                "risk_late_score_threshold": int(args.risk_late_score_threshold),
                "aggregation_delta_ee6_threshold": float(args.aggregation_delta_ee6_threshold),
                "aggregation_delta_ee6_medium_threshold": float(args.aggregation_delta_ee6_medium_threshold),
                "jerk_l2_ee6_threshold": float(args.jerk_l2_ee6_threshold),
                "gripper_flip_count_threshold": int(args.gripper_flip_count_threshold),
                "sample_var_ee6_threshold": float(args.sample_var_ee6_threshold),
                "sample_var_gripper_threshold": float(args.sample_var_gripper_threshold),
                "profile_sample_var_k": int(args.profile_sample_var_k),
                "profile_sample_var_interval": int(args.profile_sample_var_interval),
                "profile_sample_var_ages": args.profile_sample_var_ages,
                "task_age_config": task_age_config,
            },
        )
    profiler.mark("evaluation_wrapper_ready")
    dual_sys.eval()
    if args.collect_failure_recovery:
        if acc.num_processes != 1:
            raise ValueError("Failure-recovery collection currently requires one process")
        profiler.mark("before_collect_failure_recovery")
        collect_failure_recovery_dataset(eva, env, args)
        return
    if args.augment_recovery_demo:
        if acc.num_processes != 1:
            raise ValueError("Persisted recovery augmentation requires one process")
        augment_persisted_demo_branches(eva, env, args)
        return
    if args.rebuild_recovery_trajectories:
        if acc.num_processes != 1:
            raise ValueError("Recovery trajectory rebuild requires one process")
        rebuild_persisted_positive_trajectories(eva, env, args)
        return
    if args.replay_recovery_dir:
        if acc.num_processes != 1:
            raise ValueError("Failure-recovery replay currently requires one process")
        profiler.mark("before_replay_failure_recovery")
        replay_failure_recovery_dataset(eva, env, args)
        return
    profiler.mark("before_evaluate_policy")
    avg_reward = torch.tensor(
        evaluate_policy(
            eva,
            env,
            eval_sr_path,
            eval_result_path,
            acc.num_processes,
            acc.process_index,
            eval_dir=eval_dir,
            ep_len=args.ep_len,
            num_sequences=args.num_sequences,
            task_name=args.log_dir,
            enrich_lang=args.enrich_lang,
            debug=args.debug,
            max_subtasks=args.max_subtasks,
            profile_steps=args.profile_steps,
            profile_output=profile_output,
            profile_rank=acc.process_index,
        )
    ).float().mean().to(device)

    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean() 
    if acc.is_main_process:
        print('average success rate ', avg_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generalist_path", default=DEFAULT_GENERALIST_PATH.as_posix(), type=str)
    parser.add_argument("--specialist_path", default=DEFAULT_SPECIALIST_PATH.as_posix(), type=str)
    parser.add_argument("--calvin_path", default="./calvin", type=str)
    parser.add_argument("--log_dir", default="CALVIN_ABC-D", type=str)
    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--with_cfg", default=False, action="store_true")
    parser.add_argument("--enrich_lang", default=False, action="store_true")
    parser.add_argument("--dataset_subdir", default="task_ABC_D", type=str)
    parser.add_argument("--num_sequences", default=BENCHMARK_NUM_SEQUENCES, type=int)
    parser.add_argument("--ep_len", default=360, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--device_map", default="none", type=str)
    parser.add_argument("--attn_implementation", default="none", type=str)
    parser.add_argument("--fast_num_inference_steps", default=10, type=int)
    parser.add_argument("--max_subtasks", default=None, type=int)
    parser.add_argument("--profile_steps", dest="profile_steps", default=True, action="store_true")
    parser.add_argument("--no_profile_steps", dest="profile_steps", action="store_false")
    parser.add_argument("--profile_sample_var_k", default=3, type=int)
    parser.add_argument("--profile_sample_var_interval", default=8, type=int)
    parser.add_argument("--profile_sample_var_ages", default="", type=str)
    parser.add_argument(
        "--slow_call_strategy",
        default="task_age",
        choices=["fixed_mod8", "age_empty", "task_age", "risk_balanced", "risk_score", "risk_conservative", "risk_aggressive"],
        type=str,
    )
    parser.add_argument(
        "--slow_trigger_policy",
        default="age_empty",
        choices=["fixed_mod8", "age_empty"],
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
    parser.add_argument("--profile_init", action="store_true")
    parser.add_argument("--collect_failure_recovery", action="store_true")
    parser.add_argument("--augment_recovery_demo", action="store_true")
    parser.add_argument(
        "--rebuild_recovery_trajectories",
        action="store_true",
        help=(
            "Replay every paired positive in recovery_output_dir and replace its "
            "trajectory samples with complete 8-step windows."
        ),
    )
    parser.add_argument(
        "--rebuild_include_negatives",
        action="store_true",
        help="Also rebuild every paired negative branch for preference training.",
    )
    parser.add_argument("--augment_state_ids", default="", type=str)
    parser.add_argument(
        "--resume_recovery_collection",
        action="store_true",
        help="Resume a non-finalized dataset after validating all committed payloads.",
    )
    parser.add_argument(
        "--recovery_task_allowlist",
        default="",
        type=str,
        help="Optional comma-separated task set selected for recovery branching.",
    )
    parser.add_argument(
        "--recovery_stop_after_tasks",
        default="",
        type=str,
        help="End a targeted sequence after these tasks succeed; defaults to the allowlist.",
    )
    parser.add_argument(
        "--recovery_sequence_indices",
        default="",
        type=str,
        help="Optional comma-separated canonical sequence indices, evaluated in the given order.",
    )
    parser.add_argument(
        "--defer_recovery_finalize",
        action="store_true",
        help="Checkpoint an explicit acquisition phase without writing collection_summary.json.",
    )
    parser.add_argument(
        "--recovery_output_dir",
        default=(REPO_ROOT / "failure_recovery_0718" / "collected_failure_recovery_v2_scale").as_posix(),
        type=str,
    )
    parser.add_argument("--target_failure_states", default=60, type=int)
    parser.add_argument("--min_branchable_failure_states", default=24, type=int)
    parser.add_argument("--min_branchable_states_per_split", default=4, type=int)
    parser.add_argument(
        "--recovery_sequence_start",
        default=200,
        type=int,
        help="Start after canonical benchmark sequences so training states cannot leak into evaluation.",
    )
    parser.add_argument(
        "--recovery_sequence_catalog_size",
        default=1000,
        type=int,
        help="Fixed get_sequences() universe; must not change when resuming a dataset.",
    )
    parser.add_argument(
        "--recovery_sequence_catalog_id",
        default="",
        type=str,
        help="Stable namespace for sequence IDs when one dataset spans multiple fixed catalogs.",
    )
    parser.add_argument("--exclude_benchmark_sequences", default=100, type=int)
    parser.add_argument("--max_failure_states_scanned", default=120, type=int)
    parser.add_argument("--states_per_failed_subtask", default=1, type=int)
    parser.add_argument("--failure_state_min_step", default=32, type=int)
    parser.add_argument("--failure_state_min_age", default=8, type=int)
    parser.add_argument("--failure_state_stride", default=8, type=int)
    parser.add_argument("--failure_state_spacing", default=64, type=int)
    parser.add_argument("--branches_per_strategy", default=6, type=int)
    parser.add_argument(
        "--require_in_process_stable_positive",
        action="store_true",
        help="Admit a positive label only if fixed-action replay preserves outcome and length.",
    )
    parser.add_argument(
        "--stop_after_stable_positive_per_state",
        action="store_true",
        help="Stop branching a state after the first in-process stable positive.",
    )
    parser.add_argument(
        "--branch_strategies",
        default="base_seed",
        type=str,
    )
    parser.add_argument("--branch_horizon", default=80, type=int)
    parser.add_argument(
        "--recovery_success_streak",
        default=1,
        type=int,
        help="Consecutive oracle-positive steps required before a branch is labeled successful.",
    )
    parser.add_argument("--max_pairs_per_state", default=16, type=int)
    parser.add_argument("--restore_audit_states", default=3, type=int)
    parser.add_argument("--restore_audit_repeats", default=3, type=int)
    parser.add_argument("--exact_branch_audit_states", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--replay_recovery_dir", default="", type=str)
    parser.add_argument("--replay_split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--replay_horizon", default=80, type=int)
    parser.add_argument("--replay_seeds_per_state", default=3, type=int)
    parser.add_argument("--replay_max_states", default=None, type=int)
    parser.add_argument(
        "--replay_allow_missing_oracle_start",
        action="store_true",
        help=(
            "Explicitly allow legacy failure-state oracle fallback. This is not "
            "comparable to collection labels and is disabled by default."
        ),
    )
    parser.add_argument(
        "--resume_recovery_replay",
        action="store_true",
        help="Resume an atomically checkpointed replay after exact configuration validation.",
    )
    parser.add_argument(
        "--replay_output",
        default=(REPO_ROOT / "failure_recovery_0718" / "replay_result.json").as_posix(),
        type=str,
    )
    args = parser.parse_args()
    if args.recovery_sequence_catalog_id and not args.recovery_sequence_catalog_id.replace("_", "").isalnum():
        parser.error("--recovery_sequence_catalog_id must contain only letters, digits, and underscores")

    positive_collection_args = (
        "target_failure_states",
        "min_branchable_failure_states",
        "min_branchable_states_per_split",
        "recovery_sequence_catalog_size",
        "max_failure_states_scanned",
        "states_per_failed_subtask",
        "failure_state_stride",
        "failure_state_spacing",
        "branches_per_strategy",
        "branch_horizon",
        "recovery_success_streak",
        "max_pairs_per_state",
        "restore_audit_repeats",
        "replay_horizon",
        "replay_seeds_per_state",
    )
    if any(getattr(args, name) <= 0 for name in positive_collection_args):
        parser.error("Failure-recovery collection counts and horizons must be positive")
    if (
        args.failure_state_min_step < 0
        or args.failure_state_min_age < 0
        or args.restore_audit_states < 0
        or args.exact_branch_audit_states < 0
        or args.recovery_sequence_start < 0
        or args.exclude_benchmark_sequences < 0
    ):
        parser.error("Failure-state thresholds and audit state count must be non-negative")
    if args.max_failure_states_scanned < args.target_failure_states:
        parser.error("--max_failure_states_scanned must be >= --target_failure_states")
    if args.min_branchable_failure_states > args.target_failure_states:
        parser.error("--min_branchable_failure_states must be <= --target_failure_states")
    if args.collect_failure_recovery and args.recovery_sequence_start < args.exclude_benchmark_sequences:
        parser.error("Recovery training collection must exclude canonical benchmark sequences")
    if args.replay_max_states is not None and args.replay_max_states <= 0:
        parser.error("--replay_max_states must be positive when provided")
    selected_modes = sum(bool(item) for item in (
        args.collect_failure_recovery,
        args.augment_recovery_demo,
        args.rebuild_recovery_trajectories,
        args.replay_recovery_dir,
    ))
    if selected_modes > 1:
        parser.error("collection, augmentation, trajectory rebuild, and replay modes are mutually exclusive")
    if args.resume_recovery_collection and not args.collect_failure_recovery:
        parser.error("--resume_recovery_collection requires --collect_failure_recovery")
    if args.defer_recovery_finalize and not args.collect_failure_recovery:
        parser.error("--defer_recovery_finalize requires --collect_failure_recovery")
    if args.rebuild_include_negatives and not args.rebuild_recovery_trajectories:
        parser.error("--rebuild_include_negatives requires --rebuild_recovery_trajectories")
    try:
        parse_branch_strategies(args.branch_strategies)
    except ValueError as error:
        parser.error(str(error))

    main(args)
