#!/usr/bin/env python3
"""CALVIN evaluation entrypoint for task-age V1 plus RTC-only ref handover.

This trial keeps the task-conditioned age scheduler from task_age_v1_0706.py
and adds only an RTC-like ref_action handover wrapper before the specialist
call. Hidden states still hard-switch on slow refresh.
"""

import argparse
import json
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path
import sys
import time

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_CALVIN_ROOT = PROJECT_ROOT / "calvin"
CALVIN_ROOT_PATH = Path(os.environ.get("CALVIN_ROOT", DEFAULT_CALVIN_ROOT)).expanduser().resolve()
os.environ.setdefault("CALVIN_ROOT", CALVIN_ROOT_PATH.as_posix())
CALVIN_ROOT = os.environ["CALVIN_ROOT"]

for dependency_path in (
    REPO_ROOT,
    PROJECT_ROOT,
    CALVIN_ROOT_PATH / "calvin_models",
    CALVIN_ROOT_PATH / "calvin_env",
    CALVIN_ROOT_PATH / "calvin_env" / "tacto",
):
    path_str = dependency_path.as_posix()
    if dependency_path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: E402
from calvin_agent.evaluation.utils import (  # noqa: E402
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
from dual_sys_evaluation_0424test import (  # noqa: E402
    DualSystemCalvinEvaluation as BaseDualSystemCalvinEvaluation,
)


DEFAULT_GENERALIST_PATH = PROJECT_ROOT / "models" / "generalist"
DEFAULT_SPECIALIST_PATH = PROJECT_ROOT / "models" / "specialist" / "Specialist+Depth+Gripper.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation_results" / "trial_smooth_RTConly_v1_0708"
BENCHMARK_NUM_SEQUENCES = 100


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


def emit_profile_record(profile_output, record):
    line = json.dumps(record, sort_keys=True)
    print(f"[specialist-profile] {line}", flush=True)
    if profile_output is None:
        return
    with open(profile_output, "a") as file:
        file.write(line + "\n")


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


class TaskAgeV1DualSystemEvaluation(BaseDualSystemCalvinEvaluation):
    """Task-conditioned age scheduler for the V1 ABCD grouping."""

    _NON_V1_PROFILE_FIELDS = (
        "slow_handover_steps",
        "slow_handover_blend_hidden",
        "slow_handover_active",
        "slow_handover_alpha",
        "slow_handover_reason",
        "action_delta_limit_ee6",
        "action_jerk_limit_ee6",
        "action_slew_applied",
        "action_slew_delta_ee6",
    )

    def __init__(self, *args, task_age_config=None, **kwargs):
        super().__init__(*args, slow_trigger_policy="age_empty", **kwargs)
        self.task_age_config = task_age_config or {
            "default_max_slow_age": int(self.max_slow_age),
            "groups": {},
            "task_age_map": {},
            "task_group_map": {},
        }
        self.global_max_slow_age = int(self.max_slow_age)
        self.current_task = None
        self._active_task_age_info = self._resolve_task_age_info(None)
        self._last_task_age_decision = {}

    def set_current_task(self, task):
        self.current_task = None if task is None else str(task)
        self._active_task_age_info = self._resolve_task_age_info(self.current_task)

    def _resolve_task_age_info(self, task):
        task_age_map = self.task_age_config.get("task_age_map", {})
        task_group_map = self.task_age_config.get("task_group_map", {})
        default_max_slow_age = int(self.task_age_config.get("default_max_slow_age", self.global_max_slow_age))
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
        info = self._active_task_age_info or self._resolve_task_age_info(self.current_task)
        task_max_slow_age = int(info["task_max_slow_age"])
        self._last_task_age_decision = {
            "slow_call_strategy": "task_age_v1",
            "task": self.current_task,
            "task_age_group": info["task_age_group"],
            "task_max_slow_age": task_max_slow_age,
            "task_age_default_max_slow_age": info["task_age_default_max_slow_age"],
        }

        if self.last_slow_step is None:
            return True, "initial"

        slow_age_before = int(step - self.last_slow_step)
        self._last_task_age_decision["slow_age_before_decision"] = slow_age_before
        if slow_age_before >= task_max_slow_age:
            return True, "task_max_slow_age"
        return False, "task_age_skip"

    def step(self, obs, instruction, step):
        self._active_task_age_info = self._resolve_task_age_info(self.current_task)
        previous_max_slow_age = self.max_slow_age
        self.max_slow_age = int(self._active_task_age_info["task_max_slow_age"])
        try:
            action = super().step(obs, instruction, step)
        finally:
            self.max_slow_age = previous_max_slow_age

        if self.last_step_profile is not None:
            self.last_step_profile.update(self._active_task_age_info)
            self.last_step_profile.update(self._last_task_age_decision)
            for field in self._NON_V1_PROFILE_FIELDS:
                self.last_step_profile.pop(field, None)
        return action


class RTCRefHandoverTaskAgeV1DualSystemEvaluation(TaskAgeV1DualSystemEvaluation):
    """Task-age V1 scheduler with RTC-like handover only in ref_action space."""

    _EXPIRED_OLD_MODES = {"zero", "hold_prev_action", "old_last_ref", "none"}
    _BLEND_DIMS = {"all7", "ee6"}

    def __init__(
        self,
        *args,
        rtc_ref_handover_steps=4,
        rtc_ref_expired_old_mode="old_last_ref",
        rtc_ref_blend_dims="all7",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rtc_ref_handover_steps = max(0, int(rtc_ref_handover_steps))
        self.rtc_ref_expired_old_mode = str(rtc_ref_expired_old_mode)
        self.rtc_ref_blend_dims = str(rtc_ref_blend_dims)
        if self.rtc_ref_expired_old_mode not in self._EXPIRED_OLD_MODES:
            raise ValueError(
                "rtc_ref_expired_old_mode must be one of "
                f"{sorted(self._EXPIRED_OLD_MODES)}, got {self.rtc_ref_expired_old_mode!r}"
            )
        if self.rtc_ref_blend_dims not in self._BLEND_DIMS:
            raise ValueError(
                f"rtc_ref_blend_dims must be one of {sorted(self._BLEND_DIMS)}, "
                f"got {self.rtc_ref_blend_dims!r}"
            )
        self._rtc_ref_handover = None
        self._last_rtc_ref_profile = {}

    def reset(self):
        super().reset()
        self._rtc_ref_handover = None
        self._last_rtc_ref_profile = {}

    @staticmethod
    def _l2_tensor(tensor):
        return torch.linalg.vector_norm(tensor.detach().to(torch.float32).reshape(-1), ord=2).item()

    @staticmethod
    def _l2_array(array):
        return float(np.linalg.norm(np.asarray(array, dtype=np.float32).reshape(-1), ord=2))

    def _start_slow_handover(self, old_action, old_hidden_states, old_age_before, step, reason):
        # Keep the parent hidden/ref handover disabled. This trial owns only the
        # RTC-like ref_action handover and never blends hidden states.
        self._slow_handover = None
        if self.rtc_ref_handover_steps <= 0 or old_action is None or old_age_before is None:
            self._rtc_ref_handover = None
            return
        self._rtc_ref_handover = {
            "old_action": old_action,
            "old_age_before": int(old_age_before),
            "start_step": int(step),
            "reason": str(reason),
        }

    def _rtc_ref_alpha(self, slow_age_after):
        if self._rtc_ref_handover is None or self.rtc_ref_handover_steps <= 0:
            return 1.0
        if slow_age_after is None:
            return 1.0
        slow_age_after = max(0, int(slow_age_after))
        if slow_age_after >= self.rtc_ref_handover_steps:
            return 1.0
        return float(slow_age_after + 1) / float(self.rtc_ref_handover_steps)

    def _expired_old_ref_actions(self, new_ref_actions, new_num_cond_actions, old_action=None):
        if new_num_cond_actions <= 0 or self.rtc_ref_expired_old_mode == "none":
            return None, 0, "none"
        if self.rtc_ref_expired_old_mode == "old_last_ref":
            if old_action is None or not torch.is_tensor(old_action) or old_action.shape[1] <= 0:
                return None, 0, "old_last_ref_unavailable"
            fallback_ref_actions = torch.zeros_like(new_ref_actions)
            old_last_ref = old_action.to(device=new_ref_actions.device, dtype=new_ref_actions.dtype)[:, -1:, :]
            if old_last_ref.shape[0] != new_ref_actions.shape[0]:
                if old_last_ref.shape[0] != 1:
                    return None, 0, "old_last_ref_batch_mismatch"
                old_last_ref = old_last_ref.expand(new_ref_actions.shape[0], -1, -1)
            fallback_ref_actions[:, :new_num_cond_actions] = old_last_ref.expand(
                -1,
                int(new_num_cond_actions),
                -1,
            )
            return fallback_ref_actions, int(new_num_cond_actions), "old_last_ref"
        if self.rtc_ref_expired_old_mode == "hold_prev_action":
            if self.prev_action is None:
                return None, 0, "hold_prev_action_unavailable"
            fallback_ref_actions = torch.zeros_like(new_ref_actions)
            prev_action = torch.as_tensor(
                self.prev_action,
                device=new_ref_actions.device,
                dtype=new_ref_actions.dtype,
            ).view(1, 1, -1)
            fallback_ref_actions[:, :new_num_cond_actions] = prev_action
            return fallback_ref_actions, int(new_num_cond_actions), "hold_prev_action"

        return torch.zeros_like(new_ref_actions), int(new_num_cond_actions), "zero"

    def _build_ref_actions(self, slow_age_after):
        new_ref_actions, new_num_cond_actions = self._build_ref_actions_from(self.action, slow_age_after)
        alpha = self._rtc_ref_alpha(slow_age_after)
        rtc_profile = {
            "rtc_ref_handover_steps": int(self.rtc_ref_handover_steps),
            "rtc_ref_alpha": self._round_float(alpha),
            "rtc_ref_blend_dims": self.rtc_ref_blend_dims,
            "rtc_ref_in_window": bool(self._rtc_ref_handover is not None and alpha < 1.0),
            "rtc_ref_active": False,
            "rtc_ref_reason": None if self._rtc_ref_handover is None else self._rtc_ref_handover.get("reason"),
            "rtc_old_age_before": None
            if self._rtc_ref_handover is None
            else int(self._rtc_ref_handover.get("old_age_before")),
            "rtc_old_slow_age": None,
            "rtc_new_num_cond_actions": int(new_num_cond_actions),
            "rtc_old_num_cond_actions": 0,
            "rtc_ref_blend_len": 0,
            "rtc_old_ref_valid": False,
            "rtc_old_ref_source": None,
            "rtc_skipped_old_expired": False,
            "old_new_ref_l2_ee6": None,
            "old_new_ref_first_l2_ee6": None,
            "old_new_ref_gripper_abs_mean": None,
            "old_new_ref_first_gripper_abs": None,
            "rtc_ref_gripper_source": "new",
            "new_ref_first_vs_prev_action_l2_ee6": None,
            "old_ref_first_vs_prev_action_l2_ee6": None,
        }

        if new_num_cond_actions > 0 and self.prev_action is not None:
            new_first = new_ref_actions[0, 0, :6].detach().to(torch.float32).cpu().numpy()
            rtc_profile["new_ref_first_vs_prev_action_l2_ee6"] = self._round_float(
                self._l2_array(new_first - self.prev_action[:6])
            )

        if self._rtc_ref_handover is None:
            self._last_rtc_ref_profile = rtc_profile
            return new_ref_actions, new_num_cond_actions

        if alpha >= 1.0:
            self._rtc_ref_handover = None
            rtc_profile["rtc_ref_in_window"] = False
            rtc_profile["rtc_ref_reason"] = None
            rtc_profile["rtc_old_age_before"] = None
            self._last_rtc_ref_profile = rtc_profile
            return new_ref_actions, new_num_cond_actions

        old_action = self._rtc_ref_handover.get("old_action")
        old_age_before = self._rtc_ref_handover.get("old_age_before")
        if old_action is None or old_age_before is None:
            self._last_rtc_ref_profile = rtc_profile
            return new_ref_actions, new_num_cond_actions

        old_slow_age = int(old_age_before) + max(0, int(slow_age_after or 0))
        old_ref_actions, old_num_cond_actions = self._build_ref_actions_from(
            old_action.to(self.action.device), old_slow_age
        )
        old_ref_actions = old_ref_actions.to(device=new_ref_actions.device, dtype=new_ref_actions.dtype)
        blend_len = int(min(new_num_cond_actions, old_num_cond_actions))
        old_ref_source = "old_overlap" if blend_len > 0 else None

        if blend_len == 0:
            fallback_ref_actions, fallback_len, fallback_source = self._expired_old_ref_actions(
                new_ref_actions,
                new_num_cond_actions,
                old_action=old_action,
            )
            old_ref_source = fallback_source
            if fallback_ref_actions is not None and fallback_len > 0:
                old_ref_actions = fallback_ref_actions
                blend_len = int(fallback_len)

        rtc_profile["rtc_old_slow_age"] = int(old_slow_age)
        rtc_profile["rtc_old_num_cond_actions"] = int(old_num_cond_actions)
        rtc_profile["rtc_ref_blend_len"] = int(blend_len)
        rtc_profile["rtc_old_ref_valid"] = bool(old_num_cond_actions > 0)
        rtc_profile["rtc_old_ref_source"] = old_ref_source
        rtc_profile["rtc_skipped_old_expired"] = bool(alpha < 1.0 and old_num_cond_actions == 0 and blend_len == 0)

        if blend_len > 0:
            overlap_delta = new_ref_actions[:, :blend_len, :6] - old_ref_actions[:, :blend_len, :6]
            first_delta = new_ref_actions[0, 0, :6] - old_ref_actions[0, 0, :6]
            gripper_delta = torch.abs(new_ref_actions[:, :blend_len, -1] - old_ref_actions[:, :blend_len, -1])
            rtc_profile["old_new_ref_l2_ee6"] = self._round_float(self._l2_tensor(overlap_delta))
            rtc_profile["old_new_ref_first_l2_ee6"] = self._round_float(self._l2_tensor(first_delta))
            rtc_profile["old_new_ref_gripper_abs_mean"] = self._round_float(gripper_delta.mean().item())
            rtc_profile["old_new_ref_first_gripper_abs"] = self._round_float(gripper_delta[0, 0].item())
            if self.prev_action is not None:
                old_first = old_ref_actions[0, 0, :6].detach().to(torch.float32).cpu().numpy()
                rtc_profile["old_ref_first_vs_prev_action_l2_ee6"] = self._round_float(
                    self._l2_array(old_first - self.prev_action[:6])
                )

        if blend_len == 0:
            self._last_rtc_ref_profile = rtc_profile
            return new_ref_actions, new_num_cond_actions

        # RTC source keeps the previous chunk aligned to the current frame and
        # constrains the still-valid prefix while generating the next chunk.
        # At wrapper level we approximate that by constructing the effective
        # ref chunk before the specialist call. If the old ref is expired, the
        # fallback represents the conditioner that the specialist was seeing
        # before refresh, so new guidance is introduced gradually.
        effective_ref_actions = new_ref_actions.clone()
        if self.rtc_ref_blend_dims == "all7":
            effective_ref_actions[:, :blend_len] = (
                alpha * new_ref_actions[:, :blend_len] + (1.0 - alpha) * old_ref_actions[:, :blend_len]
            )
            rtc_profile["rtc_ref_gripper_source"] = "blend"
        else:
            effective_ref_actions[:, :blend_len, :6] = (
                alpha * new_ref_actions[:, :blend_len, :6]
                + (1.0 - alpha) * old_ref_actions[:, :blend_len, :6]
            )
            effective_ref_actions[:, :blend_len, -1] = new_ref_actions[:, :blend_len, -1]
            rtc_profile["rtc_ref_gripper_source"] = "new"
        rtc_profile["rtc_ref_active"] = True
        self._last_rtc_ref_profile = rtc_profile
        return effective_ref_actions, new_num_cond_actions

    def step(self, obs, instruction, step):
        self._last_rtc_ref_profile = {}
        action = super().step(obs, instruction, step)
        if self.last_step_profile is not None:
            self.last_step_profile.update(self._last_rtc_ref_profile)
        return action


def current_device_index():
    if torch.cuda.is_available():
        return int(torch.cuda.current_device())
    return 0


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

    split_dir = Path(eval_result_path).parent / str(task_name)
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f"split_{current_device_index()}.json", "w") as file:
        json.dump(chain_sr, file)

    json_data = {**current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def make_env(dataset_path, observation_space, device, use_egl):
    val_folder = Path(dataset_path) / "validation"
    from calvin_env_wrapper import CalvinEnvWrapperRaw

    return CalvinEnvWrapperRaw(val_folder, observation_space, device, use_egl=use_egl)


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
    task_name="test",
    enrich_lang=False,
    debug=False,
    max_subtasks=None,
    profile_steps=False,
    profile_output=None,
    profile_rank=0,
):
    conf_dir = Path(CALVIN_ROOT) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    if enrich_lang:
        enrich_path = REPO_ROOT / "vla-scripts" / "enrich_lang_annotations.json"
        with open(enrich_path, "r") as file:
            val_annotations = json.load(file)
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
            with open(eval_sr_path, "a") as file:
                line = f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                file.write(line + "\n")
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
        print(f"{subtask} ", end="", flush=True)
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    model.set_current_task(subtask)
    start_info = env.get_info()

    if profile_steps:
        print(
            f"[profile] rank={profile_rank} sequence={sequence_i} subtask={subtask_i} "
            f"name={subtask} ep_len={ep_len}",
            flush=True,
        )

    for step in range(ep_len):
        model_start = time.perf_counter()
        action = model.step(obs, lang_annotation, step)
        model_step_s = time.perf_counter() - model_start

        env_start = time.perf_counter()
        obs, _, _, current_info = env.step(action)
        env_step_s = time.perf_counter() - env_start

        oracle_start = time.perf_counter()
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        oracle_step_s = time.perf_counter() - oracle_start
        step_success = len(current_task_info) > 0

        if profile_steps:
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
                    "step_success": bool(step_success),
                    "terminal_step": bool(step_success),
                    "profile": getattr(model, "last_step_profile", {}),
                },
            )

        if step_success:
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
                print("success", end=" ", flush=True)
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
        print("fail", end=" ", flush=True)
    return False


def load_generalist(args):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

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
    model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **model_kwargs)
    model.eval()
    return model, processor


def build_specialist_policy(args, device):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy

    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    return DiffusionDiTImagePolicy(
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
    ).eval().to(device)


def build_dual_system(args, device):
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from train_spacialist_calvin import DualSystem

    generalist, processor = load_generalist(args)
    specialist = build_specialist_policy(args, device)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dual_system = DualSystem(generalist, specialist, action_tokenizer)
    specialist_state = torch.load(Path(args.specialist_path).expanduser().as_posix(), map_location="cpu")
    dual_system.ema_fast_system.load_state_dict(specialist_state, strict=False)
    dual_system.eval()
    return dual_system, processor, action_tokenizer


def main(args):
    seed_everything(args.seed)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=12))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device

    dual_system, processor, action_tokenizer = build_dual_system(args, device)
    dual_system = acc.prepare(dual_system, device_placement=[True])

    save_path = Path(args.output_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    eval_dir = save_path / f"eval{acc.process_index}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(CALVIN_ROOT) / "dataset" / args.dataset_subdir
    env = make_env(dataset_path.as_posix(), observation_space, device, args.use_egl)

    profile_output = None
    if args.profile_steps:
        profile_output = save_path / f"specialist_profile_rank{acc.process_index}.jsonl"
        profile_output.write_text("")

    eval_sr_path = save_path / f"success_rate_rank{acc.process_index}.txt"
    eval_result_path = save_path / f"result_rank{acc.process_index}.json"
    eval_sr_path.write_text("")

    task_age_config = build_task_age_config(args)
    evaluator = RTCRefHandoverTaskAgeV1DualSystemEvaluation(
        dual_system,
        processor,
        action_tokenizer,
        task_age_config=task_age_config,
        profile_steps=args.profile_steps,
        profile_sample_var_k=args.profile_sample_var_k,
        profile_sample_var_interval=args.profile_sample_var_interval,
        profile_sample_var_ages=args.profile_sample_var_ages,
        max_slow_age=args.max_slow_age,
        empty_ref_after_age=args.empty_ref_after_age,
        rtc_ref_handover_steps=args.rtc_ref_handover_steps,
        rtc_ref_expired_old_mode=args.rtc_ref_expired_old_mode,
        rtc_ref_blend_dims=args.rtc_ref_blend_dims,
    )

    if args.profile_steps:
        emit_profile_record(
            profile_output,
            {
                "event": "run_config",
                "rank": int(acc.process_index),
                "entrypoint": "trial_smooth_RTConly_v1_0708.py",
                "strategy": "task_age_v1_rtc_ref_handover",
                "seed": int(args.seed),
                "dataset_subdir": args.dataset_subdir,
                "num_sequences": int(args.num_sequences),
                "ep_len": int(args.ep_len),
                "max_subtasks": None if args.max_subtasks is None else int(args.max_subtasks),
                "max_slow_age": int(args.max_slow_age),
                "empty_ref_after_age": int(args.empty_ref_after_age),
                "rtc_ref_handover_steps": int(args.rtc_ref_handover_steps),
                "rtc_ref_expired_old_mode": args.rtc_ref_expired_old_mode,
                "rtc_ref_blend_dims": args.rtc_ref_blend_dims,
                "profile_sample_var_k": int(args.profile_sample_var_k),
                "profile_sample_var_interval": int(args.profile_sample_var_interval),
                "profile_sample_var_ages": args.profile_sample_var_ages,
                "task_age_config": task_age_config,
            },
        )

    avg_reward = torch.tensor(
        evaluate_policy(
            evaluator,
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
        print("average success rate ", avg_reward)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist_path", default=DEFAULT_GENERALIST_PATH.as_posix(), type=str)
    parser.add_argument("--specialist_path", default=DEFAULT_SPECIALIST_PATH.as_posix(), type=str)
    parser.add_argument("--dataset_subdir", default="calvin_debug_dataset", type=str)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR.as_posix(), type=str)
    parser.add_argument("--log_dir", default="CALVIN_ABC-D", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_sequences", default=BENCHMARK_NUM_SEQUENCES, type=int)
    parser.add_argument("--ep_len", default=360, type=int)
    parser.add_argument("--max_subtasks", default=None, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--enrich_lang", action="store_true")

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

    parser.add_argument("--profile_steps", dest="profile_steps", default=True, action="store_true")
    parser.add_argument("--no_profile_steps", dest="profile_steps", action="store_false")
    parser.add_argument("--profile_sample_var_k", default=3, type=int)
    parser.add_argument("--profile_sample_var_interval", default=8, type=int)
    parser.add_argument("--profile_sample_var_ages", default="8,9,10,11,12", type=str)

    parser.add_argument("--max_slow_age", default=12, type=int)
    parser.add_argument("--empty_ref_after_age", default=8, type=int)
    parser.add_argument("--rtc_ref_handover_steps", default=4, type=int)
    parser.add_argument(
        "--rtc_ref_expired_old_mode",
        default="old_last_ref",
        choices=sorted(RTCRefHandoverTaskAgeV1DualSystemEvaluation._EXPIRED_OLD_MODES),
        help=(
            "Fallback old ref when the previous slow chunk has no valid overlap: "
            "'old_last_ref' repeats the last value from the previous slow chunk, "
            "'zero' ramps in from empty guidance, 'hold_prev_action' ramps in from "
            "the last executed action, and 'none' disables expired-old blending."
        ),
    )
    parser.add_argument(
        "--rtc_ref_blend_dims",
        default="all7",
        choices=sorted(RTCRefHandoverTaskAgeV1DualSystemEvaluation._BLEND_DIMS),
        help=(
            "Dimensions blended during RTC-like ref handover. 'all7' preserves the "
            "old behavior and blends end-effector plus gripper. 'ee6' blends only "
            "the first six end-effector dimensions and takes gripper from the new ref."
        ),
    )
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
    main(parse_args())
