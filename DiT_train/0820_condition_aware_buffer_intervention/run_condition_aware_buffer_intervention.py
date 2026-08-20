#!/usr/bin/env python3
"""M2 preflight: condition-aware temporal-buffer intervention diagnosis."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# Must be set before CUDA/cuBLAS is initialized. This is local to this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
BASELINE_SCRIPT = (
    REPO_ROOT / "DiT_train" / "0820_drift_condition_intervention"
    / "run_drift_condition_intervention.py"
)
CANONICAL_SOURCE_RUN = (
    REPO_ROOT / "DiT_train" / "0818_expert_specialist_trajectory"
    / "runs" / "trajectory_validation_s42_n50"
)
IMMEDIATE_BASELINE_RUN = (
    REPO_ROOT / "DiT_train" / "0820_drift_condition_intervention"
    / "runs" / "drift_intervention_validation50_s42"
)
SCHEMA_VERSION = "condition_aware_buffer_intervention_v1"
ACTION_DIM = 7
ACTION_HORIZON = 8
CONDITIONS = ("old", "ref", "full")
BUFFER_POLICIES = ("keep", "flush")
BRANCHES = tuple(f"{condition}_{buffer}" for condition in CONDITIONS for buffer in BUFFER_POLICIES)
FLUSH_FIELDS = ("action_buffer", "action_buffer_mask")
RAW_EQUALITY_ATOL = 1e-6
RAW_EQUALITY_RTOL = 1e-6
FLUSH_AGGREGATION_ATOL = 1e-6
STATE_EQUALITY_ATOL = 0.0
BOOTSTRAP_SAMPLES = 10_000


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("drift_intervention_0820", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_baseline_module()
T = B.T
SNAPSHOT_FIELDS = B.SNAPSHOT_FIELDS


def validate_immediate_baseline(source_audit: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = IMMEDIATE_BASELINE_RUN / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "status_complete": manifest.get("status") == "complete",
        "seed_42": manifest.get("seed") == 42,
        "ages_8_11": manifest.get("intervention_ages") == [8, 11],
        "anchor_count_50": manifest.get("selected_anchor_count") == 50,
        "same_source_manifest": manifest.get("source_manifest_sha256") == source_audit["source_manifest_sha256"],
        "same_specialist_sha256": (
            (manifest.get("source_current_specialist") or {}).get("current_sha256")
            == source_audit["current_specialist_sha256"]
        ),
        "same_generalist_fingerprint": (
            (manifest.get("source_current_generalist") or {}).get("current_fingerprint")
            == source_audit["current_generalist_fingerprint"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Immediate completed drift baseline contract failed: {checks}")
    return {
        "run_dir": str(IMMEDIATE_BASELINE_RUN),
        "manifest_sha256": B.sha256_file(manifest_path),
        "checks": checks,
    }


def prepare_run_dir(run_name: str) -> Path:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run_name must be one non-empty path component")
    run_dir = (RUNS_ROOT / run_name).resolve()
    if run_dir.parent != RUNS_ROOT.resolve():
        raise ValueError(f"Run path escapes runs root: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def clone_runtime(value: Any) -> Any:
    return B.clone_runtime(value)


def configure_torch_determinism() -> dict[str, Any]:
    import torch

    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "profile_list_digits": 9,
    }


def diagnostic_wrapper_class():
    base = T.frozen_wrapper_class()

    class HighPrecisionDiagnosticEvaluation(base):
        """Experiment-local profile precision; deployment behavior is unchanged."""

        @classmethod
        def _tensor_list(cls, tensor, digits=9):
            return super()._tensor_list(tensor, digits=digits)

        @classmethod
        def _array_list(cls, array, digits=9):
            return super()._array_list(array, digits=digits)

    return HighPrecisionDiagnosticEvaluation


def exact_value_equal(left: Any, right: Any) -> bool:
    try:
        import torch
        if torch.is_tensor(left) or torch.is_tensor(right):
            return bool(torch.equal(torch.as_tensor(left).cpu(), torch.as_tensor(right).cpu()))
    except ImportError:
        pass
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, deque) and isinstance(right, deque):
        return left.maxlen == right.maxlen and len(left) == len(right) and all(
            exact_value_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(exact_value_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(exact_value_equal(a, b) for a, b in zip(left, right))
    return left == right


def value_digest(value: Any) -> str:
    """Stable audit digest for controller values used by isolation assertions."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        try:
            import torch
            if torch.is_tensor(item):
                array = item.detach().cpu().contiguous().numpy()
                digest.update(b"torch")
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
                return
        except ImportError:
            pass
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy")
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, deque):
            digest.update(f"deque:{item.maxlen}:{len(item)}".encode())
            for child in item:
                update(child)
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def split_branch(branch: str) -> tuple[str, str]:
    condition, buffer_policy = branch.rsplit("_", 1)
    if condition not in CONDITIONS or buffer_policy not in BUFFER_POLICIES:
        raise ValueError(branch)
    return condition, buffer_policy


def apply_condition(wrapper, condition: str, old: Mapping[str, Any], fresh: Mapping[str, Any], age: int) -> None:
    import torch

    device = wrapper._runtime_device()
    old_action = torch.as_tensor(old["slow_action"]).detach().clone().to(device, dtype=torch.float32)
    old_hidden = torch.as_tensor(old["slow_hidden"]).detach().clone().to(device)
    fresh_action = torch.as_tensor(fresh["slow_action"]).detach().clone().to(device, dtype=torch.float32)
    fresh_hidden = torch.as_tensor(fresh["slow_hidden"]).detach().clone().to(device)
    wrapper._slow_handover = None
    if condition == "old":
        wrapper.action, wrapper.hidden_states, wrapper.last_slow_step = old_action, old_hidden, 0
    elif condition == "ref":
        wrapper.action, wrapper.hidden_states, wrapper.last_slow_step = fresh_action, old_hidden, age
    elif condition == "full":
        wrapper.action, wrapper.hidden_states, wrapper.last_slow_step = fresh_action, fresh_hidden, age
    else:
        raise ValueError(condition)


def flush_temporal_buffer(wrapper) -> None:
    """The sole diagnostic mutation: invalidate temporal votes, without reset()."""
    wrapper.action_buffer[...] = 0
    wrapper.action_buffer_mask[...] = False


def condition_audit(wrapper, condition: str, old: Mapping[str, Any], fresh: Mapping[str, Any], age: int) -> dict[str, Any]:
    import torch

    expected_action = old["slow_action"] if condition == "old" else fresh["slow_action"]
    expected_hidden = fresh["slow_hidden"] if condition == "full" else old["slow_hidden"]
    expected_last = 0 if condition == "old" else age
    expected_slow_age = age if condition == "old" else 0
    ref_actions, count = wrapper._build_ref_actions(expected_slow_age)
    checks = {
        "condition": condition,
        "last_slow_step": int(wrapper.last_slow_step),
        "expected_last_slow_step": expected_last,
        "action_matches_expected": bool(torch.equal(
            wrapper.action.detach().cpu(), torch.as_tensor(expected_action).to(wrapper.action.dtype)
        )),
        "hidden_matches_expected": bool(torch.equal(
            wrapper.hidden_states.detach().cpu(), torch.as_tensor(expected_hidden).to(wrapper.hidden_states.dtype)
        )),
        "precomputed_num_cond_actions": int(count),
        "precomputed_ref_nonzero": int(torch.count_nonzero(ref_actions).item()),
        "slow_handover_disabled": wrapper._slow_handover is None,
    }
    expected_count = 0 if condition == "old" else 8
    checks["passed"] = bool(
        checks["last_slow_step"] == expected_last
        and checks["action_matches_expected"]
        and checks["hidden_matches_expected"]
        and checks["precomputed_num_cond_actions"] == expected_count
        and (expected_count > 0 or checks["precomputed_ref_nonzero"] == 0)
        and checks["slow_handover_disabled"]
    )
    return checks


def flush_isolation_audit(keep_wrapper, flush_wrapper, restored_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for field in SNAPSHOT_FIELDS:
        keep_value = getattr(keep_wrapper, field)
        flush_value = getattr(flush_wrapper, field)
        fields[field] = {
            "equal": exact_value_equal(keep_value, flush_value),
            "keep_sha256": value_digest(keep_value),
            "flush_sha256": value_digest(flush_value),
            "allowed_to_differ": field in FLUSH_FIELDS,
        }
    non_buffer_equal = all(fields[field]["equal"] for field in SNAPSHOT_FIELDS if field not in FLUSH_FIELDS)
    buffer_cleared = bool(
        np.count_nonzero(flush_wrapper.action_buffer) == 0
        and not np.any(flush_wrapper.action_buffer_mask)
    )
    keep_retained = bool(
        exact_value_equal(keep_wrapper.action_buffer, restored_snapshot["action_buffer"])
        and exact_value_equal(keep_wrapper.action_buffer_mask, restored_snapshot["action_buffer_mask"])
    )
    return {
        "passed": non_buffer_equal and buffer_cleared and keep_retained,
        "non_buffer_fields_value_equivalent": non_buffer_equal,
        "flush_action_buffer_zero": int(np.count_nonzero(flush_wrapper.action_buffer)) == 0,
        "flush_action_buffer_mask_all_false": not bool(np.any(flush_wrapper.action_buffer_mask)),
        "keep_action_buffer_exactly_retained_from_branchpoint": keep_retained,
        "fields": fields,
    }


def profile_json(profile: Mapping[str, Any]) -> dict[str, Any]:
    wanted = (
        "step", "slow_system", "slow_trigger_reason", "slow_age_before", "slow_age_after",
        "num_cond_actions", "ref_action_expired", "ref_action_first", "dp_action_first",
        "dp_action_chunk_mean", "dp_action_chunk_l2_mean", "dp_action_chunk_l2_max",
        "raw_action_prediction", "action_prediction", "aggregation_delta_ee6",
        "raw_aggregation_delta_ee6", "action_slew_applied",
    )
    return {key: profile.get(key) for key in wanted}


def active_voter_info(mask: np.ndarray) -> dict[str, Any]:
    array = np.asarray(mask, dtype=np.bool_)
    return {
        "action_buffer_mask": array.astype(np.uint8).tolist(),
        "active_voter_count_current_action": int(np.count_nonzero(array[:, 0])),
        "active_voter_count_total": int(np.count_nonzero(array)),
    }


def paired_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left)[:6] - np.asarray(right)[:6]))


def same_index_expert_metrics(index, anchor: Mapping[str, Any], source_index: int, state: Mapping[str, Any]) -> dict[str, Any] | None:
    valid = {int(value) for value in anchor["source_frame_indices"]}
    if source_index not in valid:
        return None
    expert = T.dataset_state(index.load_frame(source_index))
    return {
        "metric_name": "same_index_expert_proximity",
        "descriptive_only": True,
        "source_frame_index": source_index,
        **T.state_metrics(state, expert),
    }


def deterministic_bootstrap_ci(values: Iterable[float], seed_text: str) -> tuple[float | None, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None, None
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(BOOTSTRAP_SAMPLES, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_continuous(
    values: list[float], seed_text: str, higher_is_better: bool | None,
    tie_atol: float = 1e-12,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    low, high = deterministic_bootstrap_ci(values, seed_text)
    benefit = -array if higher_is_better is False else array
    win_definition = (
        "left is better (lower outcome)" if higher_is_better is False
        else "left is better (higher outcome)" if higher_is_better is True
        else "left-minus-right is positive; descriptive outcome has no benefit direction"
    )
    return {
        "paired_n": int(array.size),
        "mean_paired_delta": float(np.mean(array)) if array.size else None,
        "median_paired_delta": float(np.median(array)) if array.size else None,
        "std_paired_delta": float(np.std(array)) if array.size else None,
        "wins": int(np.sum(benefit > tie_atol)),
        "ties": int(np.sum(np.abs(benefit) <= tie_atol)),
        "losses": int(np.sum(benefit < -tie_atol)),
        "win_definition": win_definition,
        "paired_bootstrap_95_ci_low": low,
        "paired_bootstrap_95_ci_high": high,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


CONTRASTS = (
    ("ref_keep_minus_old_keep", "ref_keep", "old_keep", "condition effect under preserved buffer"),
    ("full_keep_minus_old_keep", "full_keep", "old_keep", "condition effect under preserved buffer"),
    ("ref_flush_minus_old_flush", "ref_flush", "old_flush", "condition effect under flushed buffer"),
    ("full_flush_minus_old_flush", "full_flush", "old_flush", "condition effect under flushed buffer"),
    ("old_flush_minus_old_keep", "old_flush", "old_keep", "pure buffer effect under old condition"),
    ("ref_flush_minus_ref_keep", "ref_flush", "ref_keep", "pure buffer effect under fresh reference"),
    ("full_flush_minus_full_keep", "full_flush", "full_keep", "pure buffer effect under full refresh"),
    ("full_flush_minus_ref_flush", "full_flush", "ref_flush", "fresh hidden contribution under flush"),
)


CONTINUOUS_OUTCOMES = (
    "success_within_8_numeric", "success_within_16_numeric", "first_step_action_ee6_norm",
    "cumulative_action_ee6_norm", "mean_same_index_robot_ee6_l2",
    "final_same_index_robot_ee6_l2", "mean_same_index_robot_full_l2",
    "mean_same_index_scene_l2", "final_robot_ee6_l2_from_branchpoint",
)
BINARY_OUTCOMES = ("success_within_8", "success_within_16")
OUTCOME_HIGHER_IS_BETTER = {
    "success_within_8_numeric": True,
    "success_within_16_numeric": True,
    "mean_same_index_robot_ee6_l2": False,
    "final_same_index_robot_ee6_l2": False,
    "mean_same_index_robot_full_l2": False,
    "mean_same_index_scene_l2": False,
}


def make_paired_contrasts(branch_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in branch_rows:
        by_pair[(row["condition_id"], int(row["intervention_age"]))][row["branch"]] = row
    output = []
    ages = sorted({age for _, age in by_pair})
    for age in ages:
        pairs = [branches for (condition_id, pair_age), branches in by_pair.items() if pair_age == age]
        for contrast_name, left_name, right_name, family in CONTRASTS:
            complete = [pair for pair in pairs if left_name in pair and right_name in pair]
            for outcome in CONTINUOUS_OUTCOMES:
                deltas = [
                    float(pair[left_name][outcome]) - float(pair[right_name][outcome])
                    for pair in complete
                    if pair[left_name].get(outcome) is not None and pair[right_name].get(outcome) is not None
                ]
                output.append({
                    "intervention_age": age, "contrast": contrast_name, "contrast_family": family,
                    "left_branch": left_name, "right_branch": right_name,
                    "outcome": outcome, "outcome_type": "continuous",
                    **summarize_continuous(
                        deltas, f"{seed}:{age}:{contrast_name}:{outcome}",
                        OUTCOME_HIGHER_IS_BETTER.get(outcome),
                    ),
                })
            for outcome in BINARY_OUTCOMES:
                outcomes = [(bool(pair[left_name][outcome]), bool(pair[right_name][outcome])) for pair in complete]
                output.append({
                    "intervention_age": age, "contrast": contrast_name, "contrast_family": family,
                    "left_branch": left_name, "right_branch": right_name,
                    "outcome": outcome, "outcome_type": "binary",
                    "paired_n": len(outcomes),
                    "left_success_right_fail": sum(left and not right for left, right in outcomes),
                    "left_fail_right_success": sum(not left and right for left, right in outcomes),
                    "both_success": sum(left and right for left, right in outcomes),
                    "both_fail": sum(not left and not right for left, right in outcomes),
                })
    return output


def aggregate_branch_rows(branch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in branch_rows:
        if not row["task_success_before_intervention"]:
            grouped[(int(row["intervention_age"]), row["branch"])].append(row)
    keys = (
        "success_within_8_numeric", "success_within_16_numeric", "first_success_post_step",
        "aggregation_delta_ee6_first", "first_step_action_ee6_norm",
        "cumulative_action_ee6_norm", "mean_same_index_robot_ee6_l2",
        "final_same_index_robot_ee6_l2", "mean_same_index_scene_l2",
        "condition_transmission_ratio",
    )
    rows = []
    for (age, branch), values in sorted(grouped.items()):
        row: dict[str, Any] = {"intervention_age": age, "branch": branch, "n": len(values)}
        for key in keys:
            numeric = [float(value[key]) for value in values if value.get(key) is not None]
            row[f"{key}_n"] = len(numeric)
            row[f"{key}_mean"] = float(np.mean(numeric)) if numeric else None
            row[f"{key}_median"] = float(np.median(numeric)) if numeric else None
            row[f"{key}_std"] = float(np.std(numeric)) if numeric else None
        rows.append(row)
    return rows


def aggregate_task_rows(branch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in branch_rows:
        if not row["task_success_before_intervention"]:
            grouped[(int(row["intervention_age"]), row["branch"], row["task"])].append(row)
    rows = []
    for (age, branch, task), values in sorted(grouped.items()):
        rows.append({
            "intervention_age": age, "branch": branch, "task": task,
            "descriptive_only": True, "n": len(values),
            "success_within_8_count": sum(bool(value["success_within_8"]) for value in values),
            "success_within_8_rate": float(np.mean([value["success_within_8_numeric"] for value in values])),
            "success_within_16_count": sum(bool(value["success_within_16"]) for value in values),
            "success_within_16_rate": float(np.mean([value["success_within_16_numeric"] for value in values])),
        })
    return rows


def intervention_seed(seed: int, condition_id: str, age: int) -> int:
    return int(hashlib.sha256(f"{seed}:{condition_id}:{age}".encode()).hexdigest()[:16], 16)


def run_intervention(
    args, run_dir, env, task_oracle, wrapper_type, dual_system, processor, tokenizer,
    model_dtype, anchor, old_condition, age, generalist_fingerprint,
):
    import torch

    condition_id = str(anchor["condition_id"])
    start_frame = int(anchor["anchor_frame"])
    start = args._index.load_frame(start_frame)
    obs = env.reset(robot_obs=start["robot_obs"].copy(), scene_obs=start["scene_obs"].copy())
    canonical_start_info = copy.deepcopy(env.get_info())
    canonical_start_provenance = {
        "captured_before_common_prefix": True,
        "anchor_frame": start_frame,
        "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32).tolist(),
        "scene_obs": np.asarray(obs["scene_obs"], dtype=np.float32).tolist(),
    }
    common = T.build_wrapper(wrapper_type, dual_system, processor, tokenizer)
    common.inject_frozen_condition(old_condition)
    success_before = False
    first_success_prefix_step = None
    prefix_actions = []
    for k in range(age):
        action = np.asarray(common.step(obs, str(anchor["instruction"]), k), dtype=np.float32).reshape(ACTION_DIM)
        prefix_actions.append(action.copy())
        obs, _, _, info = env.step(action.copy())
        if B.task_succeeded(task_oracle, canonical_start_info, info, str(anchor["task"])):
            success_before = True
            if first_success_prefix_step is None:
                first_success_prefix_step = k
    branchpoint = T.capture_env_state(obs)
    branchpoint_info = copy.deepcopy(env.get_info())
    controller_snapshot = B.snapshot_controller_state(common)
    if common.last_slow_step != 0:
        raise AssertionError("Common prefix unexpectedly refreshed slow condition")
    ref, prefix_count = common._build_ref_actions(age)
    if prefix_count != 0 or torch.count_nonzero(ref).item() != 0:
        raise AssertionError(f"Age {age} branchpoint is not an exact zero-reference state")

    fresh = B.fresh_condition(
        dual_system, processor, model_dtype, anchor, branchpoint, old_condition,
        age, args.generalist_path, generalist_fingerprint,
    )
    fresh["provenance"] = {
        "instruction": str(anchor["instruction"]),
        "rgb_source": "current policy-induced branchpoint rgb_static",
        "condition_fingerprint_sha256": hashlib.sha256(
            torch.as_tensor(fresh["slow_action"]).numpy().tobytes()
            + torch.as_tensor(fresh["slow_hidden"]).numpy().tobytes()
        ).hexdigest(),
    }
    fresh_path = run_dir / "fresh_conditions" / f"{condition_id}_age_{age}.pt"
    torch.save(fresh, fresh_path)
    paired_rng = B.snapshot_rng()
    paired_rng_digest = B.rng_digest(paired_rng)
    canonical_first_policy_input_sha256 = value_digest(branchpoint)
    first_policy_observation_source = "captured_common_prefix_branchpoint"

    wrappers = {}
    condition_audits = {}
    for branch in BRANCHES:
        condition, buffer_policy = split_branch(branch)
        wrapper = T.build_wrapper(wrapper_type, dual_system, processor, tokenizer)
        B.restore_controller_state(wrapper, controller_snapshot)
        apply_condition(wrapper, condition, old_condition, fresh, age)
        if buffer_policy == "flush":
            flush_temporal_buffer(wrapper)
        wrappers[branch] = wrapper
        condition_audits[branch] = condition_audit(wrapper, condition, old_condition, fresh, age)
    independence = B.runtime_independence(wrappers)
    isolation = {
        condition: flush_isolation_audit(
            wrappers[f"{condition}_keep"], wrappers[f"{condition}_flush"], controller_snapshot
        )
        for condition in CONDITIONS
    }
    if not independence["passed"]:
        raise AssertionError("Branch controller runtimes share mutable storage")
    if not all(audit["passed"] for audit in condition_audits.values()):
        raise AssertionError(f"Branch condition invariant failed: {condition_audits}")
    if not all(audit["passed"] for audit in isolation.values()):
        raise AssertionError(f"Flush isolation invariant failed: {isolation}")

    branch_order = list(BRANCHES)
    random.Random(intervention_seed(args.seed, condition_id, age)).shuffle(branch_order)
    reset_states = {}
    reset_records = []
    branch_results = {}
    step_rows_by_key = {}
    for branch in branch_order:
        branch_obs = env.reset(
            robot_obs=branchpoint["robot_obs"].copy(), scene_obs=branchpoint["scene_obs"].copy()
        )
        reset_state = T.capture_env_state(branch_obs)
        reset_states[branch] = reset_state
        reset_records.append({
            "condition_id": condition_id, "intervention_age": age, "branch": branch,
            "restore_method": "env.reset(robot_obs=captured_robot_obs, scene_obs=captured_scene_obs)",
            **T.state_metrics(reset_state, branchpoint),
        })
        B.restore_rng(paired_rng)
        if B.rng_digest(B.snapshot_rng()) != paired_rng_digest:
            raise AssertionError(f"Paired RNG restore failed for {branch}")
        wrapper = wrappers[branch]
        states = [reset_state]
        actions, profiles, successes = [], [], []
        first_success = None
        for j in range(args.post_steps):
            mask_before = active_voter_info(wrapper.action_buffer_mask)
            buffer_nonzero_before = int(np.count_nonzero(wrapper.action_buffer))
            global_step = age + j
            policy_obs = T.dataset_observation(branchpoint) if j == 0 else branch_obs
            action = np.asarray(
                wrapper.step(policy_obs, str(anchor["instruction"]), global_step), dtype=np.float32
            ).reshape(ACTION_DIM)
            profile = profile_json(dict(wrapper.last_step_profile))
            if profile["slow_system"]:
                raise AssertionError(f"Forbidden post-intervention generalist call in {branch}")
            mask_after = active_voter_info(wrapper.action_buffer_mask)
            branch_obs, _, _, info = env.step(action.copy())
            success = B.task_succeeded(task_oracle, canonical_start_info, info, str(anchor["task"]))
            if success and first_success is None:
                first_success = j
            state = T.capture_env_state(branch_obs)
            expert_index = start_frame + age + j + 1
            expert_metrics = same_index_expert_metrics(args._index, anchor, expert_index, state)
            actions.append(action.copy())
            profiles.append(profile)
            successes.append(success)
            states.append(state)
            step_rows_by_key[(branch, j)] = {
                "condition_id": condition_id, "task": anchor["task"],
                "intervention_age": age, "branch": branch, "post_step": j,
                "global_step": global_step, "dp_action_first": profile["dp_action_first"],
                "policy_observation_source": (
                    first_policy_observation_source if j == 0 else "previous_env_step_return"
                ),
                "policy_observation_sha256": value_digest(T.capture_env_state(policy_obs)),
                "raw_action_prediction": profile["raw_action_prediction"],
                "executed_action": action.tolist(),
                "aggregation_delta_ee6": profile["aggregation_delta_ee6"],
                "gripper_sign": int(np.sign(action[-1])), "task_success": success,
                "task_success_within_8_so_far": bool(any(successes[:8])),
                "task_success_within_16_so_far": bool(any(successes[:16])),
                "buffer_nonzero_before": buffer_nonzero_before,
                "buffer_before": mask_before, "buffer_after": mask_after,
                "same_index_expert_proximity": expert_metrics,
                "profile": profile,
            }
        expected_refs = 0 if branch.startswith("old_") else 8
        if int(profiles[0]["num_cond_actions"]) != expected_refs:
            raise AssertionError(f"{branch} first-step explicit references violated")
        branch_results[branch] = {
            "states": states, "actions": np.stack(actions), "profiles": profiles,
            "successes": successes, "first_success": first_success,
        }

    first_policy_input_hashes = {
        branch: step_rows_by_key[(branch, 0)]["policy_observation_sha256"]
        for branch in BRANCHES
    }
    if set(first_policy_input_hashes.values()) != {canonical_first_policy_input_sha256}:
        raise AssertionError(
            f"Canonical first policy observation mismatch: {first_policy_input_hashes}"
        )

    raw_equality = {}
    flush_aggregation = {}
    for condition in CONDITIONS:
        keep_profile = branch_results[f"{condition}_keep"]["profiles"][0]
        flush_profile = branch_results[f"{condition}_flush"]["profiles"][0]
        left = np.asarray(keep_profile["dp_action_first"], dtype=np.float64)
        right = np.asarray(flush_profile["dp_action_first"], dtype=np.float64)
        delta = np.abs(left - right)
        passed = bool(np.allclose(left, right, atol=RAW_EQUALITY_ATOL, rtol=RAW_EQUALITY_RTOL))
        raw_equality[condition] = {
            "passed": passed, "max_abs_delta": float(np.max(delta)),
            "ee6_l2_delta": float(np.linalg.norm(delta[:6])),
            "gripper_abs_delta": float(delta[-1]),
            "atol": RAW_EQUALITY_ATOL, "rtol": RAW_EQUALITY_RTOL,
        }
        raw_prediction = np.asarray(flush_profile["raw_action_prediction"], dtype=np.float64)
        action_prediction = np.asarray(flush_profile["action_prediction"], dtype=np.float64)
        ee6_delta = float(np.linalg.norm(raw_prediction[:6] - action_prediction[:6]))
        aggregation_delta = float(flush_profile["aggregation_delta_ee6"])
        first_step_buffer = step_rows_by_key[(f"{condition}_flush", 0)]
        first_voters = first_step_buffer["buffer_after"]["active_voter_count_current_action"]
        buffer_empty_before = first_step_buffer["buffer_before"]["active_voter_count_total"] == 0
        flush_passed = bool(
            ee6_delta <= FLUSH_AGGREGATION_ATOL
            and abs(aggregation_delta) <= FLUSH_AGGREGATION_ATOL
            and first_voters == 1 and buffer_empty_before
        )
        flush_aggregation[condition] = {
            "passed": flush_passed, "raw_vs_executed_ee6_l2": ee6_delta,
            "aggregation_delta_ee6": aggregation_delta, "atol": FLUSH_AGGREGATION_ATOL,
            "first_step_active_voters": first_voters, "buffer_empty_before_step": buffer_empty_before,
        }

    raw_ok = all(value["passed"] for value in raw_equality.values())
    flush_ok = all(value["passed"] for value in flush_aggregation.values())
    if not raw_ok or not flush_ok:
        failure_path = run_dir / f"preflight_failure_{condition_id}_age_{age}.json"
        failure = {
            "condition_id": condition_id, "intervention_age": age,
            "failed_invariants": {
                "raw_action_equality": not raw_ok,
                "flush_aggregation": not flush_ok,
            "first_policy_observation_source": first_policy_observation_source,
            "canonical_first_policy_input_sha256": canonical_first_policy_input_sha256,
            "first_policy_input_hashes": first_policy_input_hashes,
            },
            "raw_action_equality_audits": raw_equality,
            "flush_aggregation_audits": flush_aggregation,
            "first_dp_action_by_branch": {
                branch: branch_results[branch]["profiles"][0]["dp_action_first"]
                for branch in BRANCHES
            },
            "branch_order": branch_order, "paired_rng": paired_rng_digest,
            "deterministic_execution": getattr(args, "_determinism", None),
            "reset_records": reset_records,
            "reset_pairwise": B.reset_pairwise(reset_states),
            "condition_audits": condition_audits,
            "flush_isolation_audits": isolation,
        }
        B.write_json(failure_path, failure)
        raise AssertionError(
            f"Hard first-step preflight failed for {condition_id} age={age}; "
            f"diagnostics={failure_path}; raw={raw_equality}; flush={flush_aggregation}"
        )

    mechanism = {}
    for condition in ("ref", "full"):
        raw_effect = paired_delta(
            branch_results[f"{condition}_keep"]["profiles"][0]["dp_action_first"],
            branch_results["old_keep"]["profiles"][0]["dp_action_first"],
        )
        keep_effect = paired_delta(
            branch_results[f"{condition}_keep"]["actions"][0], branch_results["old_keep"]["actions"][0]
        )
        flush_effect = paired_delta(
            branch_results[f"{condition}_flush"]["actions"][0], branch_results["old_flush"]["actions"][0]
        )
        mechanism[condition] = {
            "raw_effect_ee6_l2": raw_effect,
            "executed_effect_keep_ee6_l2": keep_effect,
            "executed_effect_flush_ee6_l2": flush_effect,
            "transmission_keep": keep_effect / max(raw_effect, 1e-12),
            "transmission_flush": flush_effect / max(raw_effect, 1e-12),
            "matched_keep_control": "old_keep", "matched_flush_control": "old_flush",
        }

    branch_rows = []
    for branch in BRANCHES:
        result = branch_results[branch]
        condition, buffer_policy = split_branch(branch)
        cumulative_action = float(np.sum(np.linalg.norm(result["actions"][:, :6], axis=1)))
        expert = [step_rows_by_key[(branch, j)]["same_index_expert_proximity"] for j in range(args.post_steps)]
        expert = [value for value in expert if value is not None]
        branchpoint_delta = result["states"][-1]["robot_obs"][:6] - branchpoint["robot_obs"][:6]
        row = {
            "condition_id": condition_id, "task": anchor["task"], "intervention_age": age,
            "branch": branch, "condition_factor": condition, "buffer_factor": buffer_policy,
            "task_success_before_intervention": success_before,
            "excluded_from_primary_outcome": success_before,
            "success_within_8": bool(any(result["successes"][:8])),
            "success_within_16": bool(any(result["successes"][:16])),
            "success_within_8_numeric": float(any(result["successes"][:8])),
            "success_within_16_numeric": float(any(result["successes"][:16])),
            "first_success_post_step": result["first_success"],
            "first_step_action_ee6_norm": float(np.linalg.norm(result["actions"][0, :6])),
            "cumulative_action_ee6_norm": cumulative_action,
            "aggregation_delta_ee6_first": float(result["profiles"][0]["aggregation_delta_ee6"]),
            "mean_same_index_robot_ee6_l2": float(np.mean([x["robot_ee6_l2"] for x in expert])) if expert else None,
            "final_same_index_robot_ee6_l2": float(expert[-1]["robot_ee6_l2"]) if expert else None,
            "mean_same_index_robot_full_l2": float(np.mean([x["robot_full_l2"] for x in expert])) if expert else None,
            "mean_same_index_scene_l2": float(np.mean([x["scene_l2"] for x in expert])) if expert else None,
            "same_index_expert_proximity_steps": len(expert),
            "final_robot_ee6_l2_from_branchpoint": float(np.linalg.norm(branchpoint_delta)),
            "condition_transmission_ratio": mechanism.get(condition, {}).get(f"transmission_{buffer_policy}"),
            "final_robot_obs": result["states"][-1]["robot_obs"].tolist(),
            "final_scene_obs": result["states"][-1]["scene_obs"].tolist(),
        }
        branch_rows.append(row)

    relevant_step_contrasts = {
        "keep_condition": (("ref_keep", "old_keep"), ("full_keep", "old_keep")),
        "flush_condition": (("ref_flush", "old_flush"), ("full_flush", "old_flush")),
        "buffer": (("old_flush", "old_keep"), ("ref_flush", "ref_keep"), ("full_flush", "full_keep")),
    }
    for pairs in relevant_step_contrasts.values():
        for left, right in pairs:
            cumulative = 0.0
            for j in range(args.post_steps):
                action_delta = paired_delta(branch_results[left]["actions"][j], branch_results[right]["actions"][j])
                cumulative += action_delta
                state_delta = T.state_metrics(branch_results[left]["states"][j + 1], branch_results[right]["states"][j + 1])
                step_rows_by_key[(left, j)].setdefault("paired_contrasts", {})[f"{left}_minus_{right}"] = {
                    "executed_action_ee6_l2": action_delta,
                    "cumulative_executed_action_ee6_difference": cumulative,
                    "left_minus_right_state_distance": state_delta,
                }

    pairwise_resets = B.reset_pairwise(reset_states)
    artifact: dict[str, Any] = {
        "branch_order": np.asarray(branch_order), "prefix_actions": np.asarray(prefix_actions),
        "branchpoint_robot_obs": branchpoint["robot_obs"], "branchpoint_scene_obs": branchpoint["scene_obs"],
        "branchpoint_rgb_static": branchpoint["rgb_static"], "branchpoint_rgb_gripper": branchpoint["rgb_gripper"],
        "branchpoint_depth_static": branchpoint["depth_static"], "branchpoint_depth_gripper": branchpoint["depth_gripper"],
    }
    for branch, result in branch_results.items():
        for key in ("robot_obs", "scene_obs", "rgb_static", "rgb_gripper", "depth_static", "depth_gripper"):
            artifact[f"{branch}_{key}"] = np.stack([state[key] for state in result["states"]])
        artifact[f"{branch}_executed_actions"] = result["actions"]
        artifact[f"{branch}_dp_action_first"] = np.asarray([p["dp_action_first"] for p in result["profiles"]])
        artifact[f"{branch}_raw_action_prediction"] = np.asarray([p["raw_action_prediction"] for p in result["profiles"]])
        artifact[f"{branch}_task_success"] = np.asarray(result["successes"], dtype=np.bool_)
    trajectory_path = run_dir / "trajectories" / f"{condition_id}_age_{age}.npz"
    np.savez_compressed(trajectory_path, **artifact)

    preflight = {
        "source_condition_loaded": True, "common_prefix_reached": True,
        "branchpoint_zero_reference": prefix_count == 0 and torch.count_nonzero(ref).item() == 0,
        "fresh_call_exactly_once": fresh["generalist_calls_for_anchor_age"] == 1,
        "fresh_action_hidden_same_call": fresh["same_inference_call"],
        "six_branch_condition_contracts": all(x["passed"] for x in condition_audits.values()),
        "flush_isolation_all_conditions": all(x["passed"] for x in isolation.values()),
        "runtime_independence": independence["passed"],
        "raw_action_equality_all_conditions": all(x["passed"] for x in raw_equality.values()),
        "flush_aggregation_all_conditions": all(x["passed"] for x in flush_aggregation.values()),
        "uniform_branch_reset": len({row["restore_method"] for row in reset_records}) == 1,
        "reset_fidelity_recorded": len(reset_records) == 6 and len(pairwise_resets) == 15,
        "paired_rng_restored": True,
        "canonical_first_policy_observation_shared": len(set(first_policy_input_hashes.values())) == 1,
        "no_post_intervention_generalist_call": all(
            not profile["slow_system"] for result in branch_results.values() for profile in result["profiles"]
        ),
        "env_step_action_copy_contract": True,
        "matched_buffer_transmission_controls": all(
            value["matched_keep_control"] == "old_keep" and value["matched_flush_control"] == "old_flush"
            for value in mechanism.values()
        ),
    }
    if not all(preflight.values()):
        raise AssertionError(f"Hard preflight invariant failed: {preflight}")
    intervention_record = {
        "condition_id": condition_id, "task": anchor["task"], "instruction": anchor["instruction"],
        "intervention_age": age,
        "common_prefix_contract": f"executed real M1 actions k=0..{age - 1}; intervened before action {age}",
        "canonical_task_start_info": canonical_start_info,
        "canonical_task_start_provenance": canonical_start_provenance,
        "branchpoint_info": branchpoint_info,
        "task_success_before_intervention": success_before,
        "first_success_prefix_step": first_success_prefix_step,
        "fresh_condition_file": fresh_path.relative_to(run_dir).as_posix(),
        "fresh_inference_call_id": fresh["fresh_inference_call_id"], "fresh_calls": 1,
        "fresh_condition_fingerprint_sha256": fresh["provenance"]["condition_fingerprint_sha256"],
        "first_policy_observation_source": first_policy_observation_source,
        "canonical_first_policy_input_sha256": canonical_first_policy_input_sha256,
        "first_policy_input_hashes": first_policy_input_hashes,
        "branch_order": branch_order, "paired_rng": paired_rng_digest,
        "condition_difference": B.condition_difference(old_condition, fresh),
        "condition_effect_transmission": mechanism,
        "condition_audits": condition_audits, "flush_isolation_audits": isolation,
        "raw_action_equality_audits": raw_equality, "flush_aggregation_audits": flush_aggregation,
        "runtime_independence": independence,
        "reset_records": reset_records, "reset_pairwise": pairwise_resets,
        "branch_outcomes": branch_rows, "preflight_checks": preflight,
        "trajectory_file": trajectory_path.relative_to(run_dir).as_posix(),
    }
    return intervention_record, branch_rows, list(step_rows_by_key.values()), reset_records, pairwise_resets


def initial_manifest(args, source_manifest, source_audit, anchors, status, checkpoint_audit=None, calls=0, preflight=None):
    m1_config_path = REPO_ROOT / "DiT_train/runs/ageext_m1_long1500_b97f005/config.json"
    m1_latest_path = REPO_ROOT / "DiT_train/runs/ageext_m1_long1500_b97f005/latest_checkpoint.json"
    return {
        "schema_version": SCHEMA_VERSION, "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnosis_only": True, "not_m2_training": True, "not_a_deployment_scheduler": True,
        "git_commit": T.git_commit(), "source_git_commit": source_manifest.get("code_git_commit", "unknown"),
        "source_run_dir": str(args.source_run_dir),
        "source_manifest_sha256": source_audit["source_manifest_sha256"], "source_validation": source_audit,
        "specialist_path": source_audit["current_specialist_path"],
        "specialist_sha256": source_audit["current_specialist_sha256"],
        "generalist_path": source_audit["current_generalist_path"],
        "generalist_fingerprint": source_audit["current_generalist_fingerprint"],
        "m1_training_provenance": {
            "config_path": str(m1_config_path), "config_sha256": B.sha256_file(m1_config_path),
            "latest_checkpoint_path": str(m1_latest_path), "latest_checkpoint_sha256": B.sha256_file(m1_latest_path),
        },
        "seed": args.seed, "intervention_ages": args.intervention_ages, "post_steps": args.post_steps,
        "selected_anchor_count": len(anchors),
        "expected_fresh_generalist_calls": len(anchors) * len(args.intervention_ages),
        "actual_fresh_generalist_calls": calls,
        "branch_definitions": {
            "old_keep": "old action + old hidden + last_slow_step=0; preserve temporal buffer",
            "old_flush": "old action + old hidden + last_slow_step=0; flush temporal buffer",
            "ref_keep": "fresh action + old hidden + last_slow_step=age; preserve temporal buffer",
            "ref_flush": "fresh action + old hidden + last_slow_step=age; flush temporal buffer",
            "full_keep": "fresh action + fresh hidden + last_slow_step=age; preserve temporal buffer",
            "full_flush": "fresh action + fresh hidden + last_slow_step=age; flush temporal buffer",
        },
        "flush_semantics": "after exact snapshot restore and condition intervention, assign action_buffer[...] = 0 and action_buffer_mask[...] = False; never call reset()",
        "fields_intentionally_cleared": list(FLUSH_FIELDS),
        "fields_intentionally_preserved": [field for field in SNAPSHOT_FIELDS if field not in FLUSH_FIELDS],
        "controller_snapshot_fields": list(SNAPSHOT_FIELDS),
        "paired_rng_contract": "snapshot Torch CPU and every CUDA RNG after the one fresh call; restore immediately before every branch rollout",
        "branch_order_contract": "deterministically shuffled per (seed, condition_id, intervention_age)",
        "reset_contract": "all six branches restore with env.reset(robot_obs=captured_robot_obs, scene_obs=captured_scene_obs); because CALVIN reset rendering is not bit-exact, the first specialist call uses an independent copy of the captured canonical branchpoint observation in every branch; later calls use each branch's env.step observation",
        "generalist_call_contract": "exactly one do_sample=False call per anchor-age; same-call action and hidden shared by all branches; no later call",
        "model_evaluator_settings": {
            "fast_num_inference_steps": 10, "with_depth": True, "with_gripper": True,
            "with_tactile": False, "with_cfg": False, "load_in_4bit": args.load_in_4bit,
            "slow_handover_steps": 0, "action_delta_limit_ee6": 0.0,
            "action_jerk_limit_ee6": 0.0, "temporal_aggregation_otherwise_unchanged": True,
        },
        "assertion_tolerances": {
            "raw_action_equality_atol": RAW_EQUALITY_ATOL,
            "raw_action_equality_rtol": RAW_EQUALITY_RTOL,
            "flush_aggregation_ee6_atol": FLUSH_AGGREGATION_ATOL,
            "flush_isolation_non_buffer_exact_atol": STATE_EQUALITY_ATOL,
        },
        "checkpoint_loading": checkpoint_audit, "preflight_results": preflight,
        "same_index_expert_proximity_interpretation": "descriptive only; the persisted expert trajectory is not a unique optimal recovery trajectory after policy-induced drift",
        "deterministic_execution": getattr(args, "_determinism", {"configured": False}),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run and args.preflight_only:
        raise ValueError("--dry_run and --preflight_only are mutually exclusive")
    if args.seed != 42:
        raise ValueError("This scientific population contract requires --seed 42")
    if args.post_steps != 16 and not args.dry_run:
        raise ValueError("GPU preflight/formal runs require the formal --post_steps 16 contract")
    args.intervention_ages = B.parse_ages(args.intervention_ages)
    if args.intervention_ages != [8, 11]:
        raise ValueError("This experiment requires exact intervention ages 8,11")
    loader_contract = {
        "load_in_4bit": args.load_in_4bit is True,
        "low_cpu_mem_usage": args.low_cpu_mem_usage is True,
        "device_map_none": args.device_map == "none",
        "attn_implementation_none": args.attn_implementation == "none",
    }
    if not all(loader_contract.values()):
        raise ValueError(f"Generalist loader differs from the completed source contract: {loader_contract}")
    args._determinism = (
        {"configured": False, "reason": "dry_run"}
        if args.dry_run else {"configured": True, **configure_torch_determinism()}
    )
    source_manifest, source_anchors, source_audit = B.validate_source(args)
    if args.source_run_dir != CANONICAL_SOURCE_RUN.resolve():
        raise ValueError(
            f"This experiment requires the exact completed source run {CANONICAL_SOURCE_RUN.resolve()}"
        )
    source_audit["immediate_drift_baseline"] = validate_immediate_baseline(source_audit)
    selected = source_anchors[:args.preflight_anchors] if args.preflight_only else source_anchors
    run_dir = prepare_run_dir(args.run_name)
    B.write_jsonl(run_dir / "anchors.jsonl", selected)
    eligibility = {
        "source_anchor_count": len(source_anchors), "selected_anchor_count": len(selected),
        "task_counts": dict(sorted(Counter(anchor["task"] for anchor in selected).items())),
        "intervention_ages": args.intervention_ages, "post_steps": args.post_steps,
        "anchor_age_pairs": len(selected) * len(args.intervention_ages),
        "expected_fresh_generalist_calls": len(selected) * len(args.intervention_ages),
    }
    if args.dry_run:
        for filename in ("interventions.jsonl", "branch_steps.jsonl"):
            B.write_jsonl(run_dir / filename, [])
        for filename in ("branch_summary.csv", "paired_contrasts.csv", "task_summary.csv"):
            B.write_csv(run_dir / filename, [])
        B.write_json(run_dir / "summary.json", {"status": "dry_run_complete", "eligibility": eligibility})
        B.write_json(run_dir / "reset_fidelity.json", {"status": "not_run", "reason": "dry_run"})
        B.write_json(run_dir / "manifest.json", initial_manifest(
            args, source_manifest, source_audit, selected, "dry_run_complete"
        ))
        result = {"run_dir": str(run_dir), "status": "dry_run_complete", **eligibility}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    import torch

    B.seed_everything(args.seed)
    (run_dir / "fresh_conditions").mkdir()
    (run_dir / "trajectories").mkdir()
    B.write_json(run_dir / "manifest.json", initial_manifest(
        args, source_manifest, source_audit, selected, "initializing"
    ))
    args.dataset_root = Path(source_manifest["dataset_root"]).expanduser().resolve()
    args.fast_num_inference_steps = 10
    args._index = T.CalvinLanguageIndex(args.dataset_root)
    dual_system, processor, tokenizer, model_dtype, device, checkpoint_audit = T.load_models(args)
    if checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]:
        raise RuntimeError(f"M1 checkpoint incompatibility: {checkpoint_audit}")
    if checkpoint_audit["sha256"] != source_audit["source_specialist_sha256"]:
        raise RuntimeError("Loaded specialist checkpoint hash differs from source")
    if checkpoint_audit["generalist_checkpoint_fingerprint"] != source_audit["source_generalist_fingerprint"]:
        raise RuntimeError("Loaded generalist fingerprint differs from source")
    task_oracle, task_oracle_path = B.make_task_oracle(args.dataset_root)
    env = T.make_env(args._index, args.use_egl)
    wrapper_type = diagnostic_wrapper_class()
    interventions, branch_rows, branch_steps = [], [], []
    reset_records, reset_pairwise_rows, preflight_results = [], [], []
    fresh_calls = 0
    try:
        for anchor_i, anchor in enumerate(selected):
            old = B.source_condition(
                args.source_run_dir / "conditions" / f"{anchor['condition_id']}.pt", anchor
            )
            for age in args.intervention_ages:
                record, rows, steps, resets, pairwise = run_intervention(
                    args, run_dir, env, task_oracle, wrapper_type, dual_system, processor,
                    tokenizer, model_dtype, anchor, old, age,
                    source_audit["current_generalist_fingerprint"],
                )
                fresh_calls += 1
                interventions.append(record)
                branch_rows.extend(rows)
                branch_steps.extend(steps)
                reset_records.extend(resets)
                reset_pairwise_rows.extend({
                    "condition_id": anchor["condition_id"], "intervention_age": age, **row
                } for row in pairwise)
                preflight_results.append(record["preflight_checks"])
            print(f"[{anchor_i + 1}/{len(selected)}] {anchor['condition_id']} complete", flush=True)
    finally:
        try:
            env.close()
        finally:
            underlying_env = getattr(env, "unwrapped", None)
            if underlying_env is not None and hasattr(underlying_env, "cid"):
                underlying_env.cid = -1
    expected_calls = len(selected) * len(args.intervention_ages)
    if fresh_calls != expected_calls:
        raise AssertionError(f"fresh generalist calls {fresh_calls} != expected {expected_calls}")

    branch_summary = aggregate_branch_rows(branch_rows)
    contrasts = make_paired_contrasts(
        [row for row in branch_rows if not row["task_success_before_intervention"]], args.seed
    )
    task_summary = aggregate_task_rows(branch_rows)
    transmission = {
        channel: {
            buffer_policy: {
                "n": len(values), "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
            for buffer_policy in BUFFER_POLICIES
            for values in [[
                float(record["condition_effect_transmission"][channel][f"transmission_{buffer_policy}"])
                for record in interventions
                if not record["task_success_before_intervention"]
            ]]
        }
        for channel in ("ref", "full")
    }
    status = "preflight_complete" if args.preflight_only else "complete"
    summary = {
        "status": status, "anchors": len(selected), "anchor_age_pairs": len(interventions),
        "fresh_generalist_calls": fresh_calls,
        "mechanism_endpoint": {"condition_effect_transmission": transmission},
        "outcome_endpoint": {
            "branch_summary": branch_summary,
            "paired_contrasts_file": "paired_contrasts.csv",
            "interpretation": "Recovery must be assessed against matched buffer controls; transmission alone is not evidence of benefit.",
        },
        "same_index_expert_proximity_limitation": "Descriptive only: the persisted expert trajectory is not a unique optimal recovery trajectory after policy-induced drift.",
        "interpretation_cases": {
            "A": "Fresh condition appears corrective and stale buffer suppresses it only if full_flush improves over both full_keep and old_flush while old_flush is near old_keep.",
            "B": "If old_flush improves strongly and fresh flush adds little, aggregation itself is implicated.",
            "C": "If transmission rises but full_flush does not improve over old_flush, fresh condition is not an effective recovery oracle for M1.",
            "D": "If ref_flush approximately matches full_flush, fresh explicit reference is the dominant channel.",
            "E": "If full_flush consistently outperforms ref_flush, fresh hidden state contributes under unsuppressed control.",
        },
        "automatic_proof_label": None,
    }
    reset_fidelity = {
        "status": "measured",
        "restore_method": "env.reset(robot_obs=captured_robot_obs, scene_obs=captured_scene_obs)",
        "branch_vs_captured": reset_records, "branch_pairwise": reset_pairwise_rows,
        "warning": "All reset deltas are persisted without threshold filtering; inspect before causal interpretation.",
    }
    B.write_jsonl(run_dir / "interventions.jsonl", interventions)
    B.write_jsonl(run_dir / "branch_steps.jsonl", branch_steps)
    B.write_csv(run_dir / "branch_summary.csv", branch_summary)
    B.write_csv(run_dir / "paired_contrasts.csv", contrasts)
    B.write_csv(run_dir / "task_summary.csv", task_summary)
    B.write_json(run_dir / "summary.json", summary)
    B.write_json(run_dir / "reset_fidelity.json", reset_fidelity)
    manifest = initial_manifest(
        args, source_manifest, source_audit, selected, status,
        checkpoint_audit, fresh_calls, preflight_results,
    )
    manifest["task_oracle_path"] = str(task_oracle_path)
    B.write_json(run_dir / "manifest.json", manifest)

    json.loads((run_dir / "manifest.json").read_text())
    B.read_jsonl(run_dir / "interventions.jsonl")
    B.read_jsonl(run_dir / "branch_steps.jsonl")
    with np.load(run_dir / interventions[0]["trajectory_file"], allow_pickle=False) as archive:
        if not archive.files:
            raise AssertionError("Saved trajectory artifact is empty")
    torch.load(run_dir / interventions[0]["fresh_condition_file"], map_location="cpu", weights_only=False)
    result = {
        "run_dir": str(run_dir), "status": status, "anchors": len(selected),
        "anchor_age_pairs": len(interventions), "fresh_generalist_calls": fresh_calls,
        "device": str(device), "checkpoint_missing_keys": checkpoint_audit["missing_keys"],
        "checkpoint_unexpected_keys": checkpoint_audit["unexpected_keys"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run_dir", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--intervention_ages", default="8,11")
    parser.add_argument("--post_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--preflight_anchors", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--specialist_path", type=Path, default=None)
    parser.add_argument("--generalist_path", type=Path, default=None)
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--low_cpu_mem_usage", dest="low_cpu_mem_usage", action="store_true")
    parser.add_argument("--no_low_cpu_mem_usage", dest="low_cpu_mem_usage", action="store_false")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="none")
    parser.set_defaults(load_in_4bit=True, low_cpu_mem_usage=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
