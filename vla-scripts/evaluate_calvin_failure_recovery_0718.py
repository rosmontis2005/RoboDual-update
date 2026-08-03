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

"""Code to evaluate Calvin."""
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
    allowed = {"base_seed", "forced_refresh", "slow_override"}
    result = [item.strip() for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or not set(result).issubset(allowed):
        raise ValueError(f"branch strategies must be unique values from {sorted(allowed)}")
    return result


class FailureRecoveryWriter:
    """State-grouped branch dataset inspired by Sirius intervention segments."""

    def __init__(self, output_dir):
        self.root = Path(output_dir).expanduser().resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Recovery output is not empty: {self.root}")
        self.states_dir = self.root / "states"
        self.branches_dir = self.root / "branches"
        self.conditions_dir = self.root / "conditions"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.branches_dir.mkdir(parents=True, exist_ok=True)
        self.conditions_dir.mkdir(parents=True, exist_ok=True)
        self.states = []
        self.branches = []
        self.pairs = []

    def save_state(self, record, obs, previous_rgb, hist_action):
        state_id = record["failure_state_id"]
        np.savez_compressed(
            self.states_dir / f"{state_id}.npz",
            **_portable_obs(obs),
            previous_rgb=np.asarray(previous_rgb, dtype=np.uint8),
            hist_action=np.asarray(hist_action, dtype=np.float32),
        )
        self.states.append(record)

    def save_persistent_runtime(self, state_id, bullet, model_state):
        bullet_path = self.states_dir / f"{state_id}.bullet"
        bullet.p.saveBullet(bullet_path.as_posix(), physicsClientId=bullet.cid)
        torch.save(
            {name: _cpu_runtime_value(value) for name, value in model_state.items()},
            self.states_dir / f"{state_id}_model.pt",
        )
        torch.save(bullet.serialize(), self.states_dir / f"{state_id}_simulator.pt")
        return bullet_path

    def save_branch(self, record, actions, final_obs, condition):
        branch_id = record["branch_id"]
        np.savez_compressed(
            self.branches_dir / f"{branch_id}.npz",
            actions=np.asarray(actions, dtype=np.float32),
            final_robot_obs=np.asarray(final_obs["robot_obs"], dtype=np.float32),
            final_scene_obs=np.asarray(final_obs["scene_obs"], dtype=np.float32),
        )
        torch.save(condition, self.conditions_dir / f"{branch_id}.pt")
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
    for offset in range(horizon):
        absolute_step = candidate["step"] + offset
        action = np.asarray(model.step(obs, instruction, absolute_step), dtype=np.float32).reshape(7)
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
        obs, _, _, current_info = env.step(action.copy())
        if task_oracle.get_task_info_for_set(start_info, current_info, {task}):
            success = True
            break
    return {
        "success": success,
        "actions": actions,
        "final_obs": obs,
        "first_profile": first_profile,
        "condition": condition,
    }


def replay_recorded_action_branch(
    env, bullet, candidate, task_oracle, start_info, task, actions
):
    """Replay fixed actions to audit physics restore without resampling a policy."""

    bullet.p.restoreState(stateId=candidate["bullet_state_id"], physicsClientId=bullet.cid)
    obs = env.get_obs()
    success = False
    steps = 0
    for action in actions:
        obs, _, _, current_info = env.step(np.asarray(action, dtype=np.float32).copy())
        steps += 1
        if task_oracle.get_task_info_for_set(start_info, current_info, {task}):
            success = True
            break
    return {"success": success, "steps": steps, "final_obs": obs}


def collect_failure_recovery_dataset(model, env, args):
    conf_dir = Path(CALVIN_ROOT) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    bullet = _bullet_env(env)
    writer = FailureRecoveryWriter(args.recovery_output_dir)
    sequence_end = args.recovery_sequence_start + args.num_sequences
    eval_sequences = list(get_sequences(sequence_end))[args.recovery_sequence_start : sequence_end]
    restore_audits = []
    exact_branch_audits = []
    branch_strategies = parse_branch_strategies(args.branch_strategies)
    failure_count = 0

    for local_sequence_i, (initial_state, sequence) in enumerate(eval_sequences):
        sequence_i = args.recovery_sequence_start + local_sequence_i
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
                eligible = (
                    step >= args.failure_state_min_step
                    and slow_age is not None
                    and int(slow_age) >= args.failure_state_min_age
                    and step % args.failure_state_stride == 0
                    and len(candidates) < args.states_per_failed_subtask
                    and (
                        not candidates
                        or step - candidates[-1]["step"] >= args.failure_state_spacing
                    )
                )
                if eligible:
                    state_id = f"s{sequence_i:04d}_t{subtask_i}_k{step:03d}"
                    saved = {
                        "failure_state_id": state_id,
                        "sequence_i": sequence_i,
                        "subtask_i": subtask_i,
                        "task": task,
                        "instruction": instruction,
                        "step": step,
                        "slow_age": int(slow_age),
                        "bullet_state_id": bullet.p.saveState(physicsClientId=bullet.cid),
                        "model_state": capture_recovery_model_state(model),
                        "obs": copy.deepcopy(obs),
                        "previous_rgb": (
                            np.asarray(model.obs_buffer, dtype=np.uint8).copy()
                            if model.obs_buffer is not None
                            else np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8).copy()
                        ),
                        "hist_action": _history_array(model),
                    }
                    candidates.append(saved)

                action = np.asarray(model.step(obs, instruction, step), dtype=np.float32).reshape(7)
                obs, _, _, current_info = env.step(action.copy())
                if task_oracle.get_task_info_for_set(start_info, current_info, {task}):
                    succeeded = True
                    break

            if succeeded:
                while candidates:
                    dropped = candidates.popleft()
                    bullet.p.removeState(dropped["bullet_state_id"], physicsClientId=bullet.cid)
                continue

            failure_count += 1
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
                    "subtask_i": int(candidate["subtask_i"]),
                    "task": task,
                    "instruction": instruction,
                    "step": int(candidate["step"]),
                    "slow_age": int(candidate["slow_age"]),
                    "baseline_subtask_failed": True,
                }
                bullet.p.restoreState(stateId=exact_state_id, physicsClientId=bullet.cid)
                writer.save_persistent_runtime(state_id, bullet, candidate["model_state"])
                writer.save_state(
                    state_record,
                    candidate["obs"],
                    candidate["previous_rgb"],
                    candidate["hist_action"],
                )
                for branch_index in range(args.branches_per_strategy):
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
                        )
                        branch_id = f"{state_id}_{strategy}_{branch_index:02d}"
                        writer.save_branch(
                            {
                                "branch_id": branch_id,
                                "failure_state_id": state_id,
                                "split": split,
                                "strategy": strategy,
                                "seed": int(branch_seed),
                                "success": bool(result["success"]),
                                "steps": len(result["actions"]),
                                "first_profile": result["first_profile"],
                            },
                            result["actions"],
                            result["final_obs"],
                            result["condition"],
                        )
                        if (
                            len(exact_branch_audits) < args.exact_branch_audit_states
                            and branch_index == 0
                            and strategy == "base_seed"
                        ):
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
    return writer.finalize(args, restore_audits, exact_branch_audits)


def replay_failure_recovery_dataset(model, env, args):
    root = Path(args.replay_recovery_dir).expanduser().resolve()
    with (root / "failure_states.jsonl").open() as file:
        states = [json.loads(line) for line in file if line.strip()]
    states = [item for item in states if item["split"] == args.replay_split]
    if args.replay_max_states is not None:
        states = states[: args.replay_max_states]
    if not states:
        raise ValueError(f"No recovery states for split={args.replay_split!r}")
    conf_dir = Path(CALVIN_ROOT) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    bullet = _bullet_env(env)
    records = []
    for state in states:
        state_id = state["failure_state_id"]
        for seed_index in range(args.replay_seeds_per_state):
            bullet.p.restoreState(
                fileName=(root / "states" / f"{state_id}.bullet").as_posix(),
                physicsClientId=bullet.cid,
            )
            simulator = torch.load(
                root / "states" / f"{state_id}_simulator.pt",
                map_location="cpu",
                weights_only=False,
            )
            bullet.robot.reset_from_storage(simulator["robot"])
            bullet.scene.reset_from_storage(simulator["scene"])
            runtime = torch.load(
                root / "states" / f"{state_id}_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            restore_persistent_model_state(model, runtime)
            model.set_current_task(state["task"])
            seed = args.seed + int(state["sequence_i"]) * 100003 + int(state["step"]) * 31 + seed_index
            _set_branch_seed(seed)
            obs = env.get_obs()
            start_info = env.get_info()
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
            })
    summary = {
        "format": "robodual_failure_recovery_replay_v1",
        "data_dir": root.as_posix(),
        "specialist_path": str(Path(args.specialist_path).expanduser().resolve()),
        "split": args.replay_split,
        "states": len(states),
        "rollouts": len(records),
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "horizon": args.replay_horizon,
        "records": records,
    }
    output = Path(args.replay_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


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
    parser.add_argument(
        "--recovery_output_dir",
        default=(REPO_ROOT / "failure_recovery_0718" / "collected_failure_recovery_v1").as_posix(),
        type=str,
    )
    parser.add_argument("--target_failure_states", default=20, type=int)
    parser.add_argument("--min_branchable_failure_states", default=5, type=int)
    parser.add_argument("--min_branchable_states_per_split", default=1, type=int)
    parser.add_argument(
        "--recovery_sequence_start",
        default=100,
        type=int,
        help="Start after canonical benchmark sequences so training states cannot leak into evaluation.",
    )
    parser.add_argument("--exclude_benchmark_sequences", default=100, type=int)
    parser.add_argument("--max_failure_states_scanned", default=60, type=int)
    parser.add_argument("--states_per_failed_subtask", default=2, type=int)
    parser.add_argument("--failure_state_min_step", default=24, type=int)
    parser.add_argument("--failure_state_min_age", default=8, type=int)
    parser.add_argument("--failure_state_stride", default=8, type=int)
    parser.add_argument("--failure_state_spacing", default=64, type=int)
    parser.add_argument("--branches_per_strategy", default=4, type=int)
    parser.add_argument(
        "--branch_strategies",
        default="base_seed,forced_refresh,slow_override",
        type=str,
    )
    parser.add_argument("--branch_horizon", default=80, type=int)
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
        "--replay_output",
        default=(REPO_ROOT / "failure_recovery_0718" / "replay_result.json").as_posix(),
        type=str,
    )
    args = parser.parse_args()

    positive_collection_args = (
        "target_failure_states",
        "min_branchable_failure_states",
        "min_branchable_states_per_split",
        "max_failure_states_scanned",
        "states_per_failed_subtask",
        "failure_state_stride",
        "failure_state_spacing",
        "branches_per_strategy",
        "branch_horizon",
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
    if args.collect_failure_recovery and args.replay_recovery_dir:
        parser.error("collection and replay modes are mutually exclusive")
    try:
        parse_branch_strategies(args.branch_strategies)
    except ValueError as error:
        parser.error(str(error))

    main(args)
