#!/usr/bin/env python3
"""Diagnosis-only causal interventions on M1 slow conditioning at drifted states."""

from __future__ import annotations

import argparse
import copy
import csv
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


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
TRAJECTORY_SCRIPT = (
    REPO_ROOT / "DiT_train" / "0818_expert_specialist_trajectory"
    / "run_expert_specialist_trajectory.py"
)
SCHEMA_VERSION = "drift_condition_intervention_v1"
ACTION_DIM = 7
ACTION_HORIZON = 8
BRANCHES = (
    "frozen_baseline", "fresh_hidden_only", "fresh_ref_only", "full_refresh",
)
SNAPSHOT_FIELDS = (
    "action_buffer", "action_buffer_mask", "obs_buffer", "hist_action",
    "gripper_window", "action", "hidden_states", "last_slow_step",
    "prev_action", "prev_prev_action", "prev_proprio", "prev_obs_tensor",
    "last_step_profile", "_slow_handover", "_fast_device", "frozen_condition_id",
    "forbidden_slow_call_count", "_condition_injected",
)


def load_trajectory_module():
    spec = importlib.util.spec_from_file_location("trajectory_0818", TRAJECTORY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAJECTORY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_trajectory_module()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, default=json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def parse_ages(value: str) -> list[int]:
    ages = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ages or len(ages) != len(set(ages)):
        raise ValueError("--intervention_ages must contain unique comma-separated ages")
    if any(age < 8 or age > 11 for age in ages):
        raise ValueError("This zero-reference diagnostic restricts intervention ages to 8..11")
    return ages


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


def current_generalist_fingerprint(path: Path) -> dict[str, Any]:
    from DiT_train.data_collection.collect_age_extended_expert import artifact_fingerprint
    return artifact_fingerprint(path, hash_model_files=False)


def validate_source(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source = args.source_run_dir.expanduser().resolve()
    manifest_path = source / "manifest.json"
    anchors_path = source / "anchors.jsonl"
    conditions_dir = source / "conditions"
    for path in (manifest_path, anchors_path, conditions_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchors = read_jsonl(anchors_path)
    errors: list[str] = []
    if manifest.get("stable_split") != "validation":
        errors.append(f"stable_split={manifest.get('stable_split')!r}, expected 'validation'")
    if len(anchors) != 50:
        errors.append(f"source anchors={len(anchors)}, formal source contract requires 50")
    if any(anchor.get("split") != "validation" for anchor in anchors):
        errors.append("one or more source anchors are not in stable split validation")
    ids = [str(anchor.get("condition_id")) for anchor in anchors]
    if len(ids) != len(set(ids)) or any(value == "None" for value in ids):
        errors.append("source condition_id values are missing or duplicated")
    missing_conditions = [value for value in ids if not (conditions_dir / f"{value}.pt").is_file()]
    if missing_conditions:
        errors.append(f"missing source condition files: {missing_conditions[:5]}")

    specialist_path = Path(args.specialist_path or manifest.get("specialist_path", "")).expanduser().resolve()
    generalist_path = Path(args.generalist_path or manifest.get("generalist_path", "")).expanduser().resolve()
    if not specialist_path.is_file():
        errors.append(f"current specialist checkpoint missing: {specialist_path}")
    if not generalist_path.is_dir():
        errors.append(f"current generalist path missing: {generalist_path}")
    source_checkpoint = manifest.get("checkpoint_loading") or {}
    source_specialist_hash = source_checkpoint.get("sha256")
    current_specialist_hash = sha256_file(specialist_path) if specialist_path.is_file() else None
    if source_specialist_hash != current_specialist_hash:
        errors.append(
            f"specialist SHA256 mismatch: source={source_specialist_hash}, current={current_specialist_hash}"
        )
    source_specialist_path = Path(manifest.get("specialist_path", "")).expanduser().resolve()
    if source_specialist_path != specialist_path:
        errors.append(f"specialist path mismatch: source={source_specialist_path}, current={specialist_path}")

    source_fingerprint = (manifest.get("generalist") or {}).get("checkpoint_fingerprint")
    current_fingerprint = (
        current_generalist_fingerprint(generalist_path) if generalist_path.is_dir() else None
    )
    if source_fingerprint != current_fingerprint:
        errors.append("generalist checkpoint fingerprint differs from source run")
    source_generalist_path = Path(manifest.get("generalist_path", "")).expanduser().resolve()
    if source_generalist_path != generalist_path:
        errors.append(f"generalist path mismatch: source={source_generalist_path}, current={generalist_path}")
    specialist_contract = manifest.get("specialist") or {}
    condition_contract = manifest.get("condition_contract") or {}
    expected_counts = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0]
    contract_checks = {
        "schema": manifest.get("schema_version") == "expert_specialist_trajectory_v1",
        "source_complete": manifest.get("status") == "complete",
        "formal_source": manifest.get("mode") == "formal",
        "fast_num_inference_steps_10": specialist_contract.get("fast_num_inference_steps") == 10,
        "with_depth": specialist_contract.get("with_depth") is True,
        "with_gripper": specialist_contract.get("with_gripper") is True,
        "without_cfg": specialist_contract.get("with_cfg") is False,
        "temporal_aggregation_enabled": specialist_contract.get("temporal_aggregation_bypassed") is False,
        "slow_handover_disabled": specialist_contract.get("slow_handover_steps") == 0,
        "age_empty_counts": condition_contract.get("age_empty_num_cond_actions") == expected_counts,
        "last_slow_step_zero": condition_contract.get("last_slow_step_at_injection") == 0,
        "source_same_call": condition_contract.get("same_call_slow_action_and_hidden") is True,
    }
    failed_contracts = [key for key, passed in contract_checks.items() if not passed]
    if failed_contracts:
        errors.append(f"source M1/condition contract failures: {failed_contracts}")
    if errors:
        raise ValueError("Source run validation failed:\n- " + "\n- ".join(errors))
    args.source_run_dir = source
    args.specialist_path = specialist_path
    args.generalist_path = generalist_path
    audit = {
        "passed": True,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_anchor_count": len(anchors),
        "source_specialist_path": str(source_specialist_path),
        "current_specialist_path": str(specialist_path),
        "source_specialist_sha256": source_specialist_hash,
        "current_specialist_sha256": current_specialist_hash,
        "source_generalist_path": str(source_generalist_path),
        "current_generalist_path": str(generalist_path),
        "source_generalist_fingerprint": source_fingerprint,
        "current_generalist_fingerprint": current_fingerprint,
        "condition_contract_checks": contract_checks,
        "missing_conditions": missing_conditions,
    }
    return manifest, anchors, audit


def clone_runtime(value: Any) -> Any:
    try:
        import torch
        if torch.is_tensor(value):
            return value.detach().clone()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, deque):
        return deque((clone_runtime(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: clone_runtime(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_runtime(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_runtime(item) for item in value)
    return copy.deepcopy(value)


def snapshot_controller_state(wrapper) -> dict[str, Any]:
    missing = [field for field in SNAPSHOT_FIELDS if not hasattr(wrapper, field)]
    if missing:
        raise AttributeError(f"Controller lacks required runtime fields: {missing}")
    return {field: clone_runtime(getattr(wrapper, field)) for field in SNAPSHOT_FIELDS}


def restore_controller_state(wrapper, snapshot: Mapping[str, Any]) -> None:
    if set(snapshot) != set(SNAPSHOT_FIELDS):
        raise ValueError("Controller snapshot field set changed")
    for field in SNAPSHOT_FIELDS:
        setattr(wrapper, field, clone_runtime(snapshot[field]))


def objects_share_storage(left: Any, right: Any) -> bool:
    import torch
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.data_ptr() == right.data_ptr()
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.shares_memory(left, right)
    if isinstance(left, deque) and isinstance(right, deque):
        if left is right:
            return True
        return any(objects_share_storage(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left is right or any(
            key in right and objects_share_storage(value, right[key]) for key, value in left.items()
        )
    return False


def runtime_independence(wrappers: Mapping[str, Any]) -> dict[str, Any]:
    pairs = []
    names = list(wrappers)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left, right = wrappers[left_name], wrappers[right_name]
            shared = [
                field for field in SNAPSHOT_FIELDS
                if objects_share_storage(getattr(left, field), getattr(right, field))
            ]
            pairs.append({"left": left_name, "right": right_name, "shared_runtime_fields": shared})
    passed = all(not row["shared_runtime_fields"] for row in pairs)
    return {"passed": passed, "pairs": pairs, "fields_checked": list(SNAPSHOT_FIELDS)}


def snapshot_rng() -> dict[str, Any]:
    import torch
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    import torch
    torch.random.set_rng_state(state["cpu"].clone())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([item.clone() for item in state["cuda"]])


def rng_digest(state: Mapping[str, Any]) -> dict[str, Any]:
    def digest(tensor) -> str:
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
    return {
        "cpu_sha256": digest(state["cpu"]),
        "cuda_sha256": None if state["cuda"] is None else [digest(item) for item in state["cuda"]],
    }


def condition_difference(old: Mapping[str, Any], fresh: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    old_action = torch.as_tensor(old["slow_action"]).to(torch.float32)
    fresh_action = torch.as_tensor(fresh["slow_action"]).to(torch.float32)
    action_delta = fresh_action - old_action
    old_gripper = torch.sign(old_action[..., -1])
    fresh_gripper = torch.sign(fresh_action[..., -1])
    old_hidden = torch.as_tensor(old["slow_hidden"]).to(torch.float32)
    fresh_hidden = torch.as_tensor(fresh["slow_hidden"]).to(torch.float32)
    result = {
        "slow_action_first_ee6_l2": float(torch.linalg.vector_norm(action_delta[0, 0, :6]).item()),
        "slow_action_chunk_ee6_rms": float(torch.sqrt(torch.mean(action_delta[..., :6] ** 2)).item()),
        "slow_action_chunk_ee6_l2": float(torch.linalg.vector_norm(action_delta[..., :6]).item()),
        "slow_action_gripper_agreement_fraction": float((old_gripper == fresh_gripper).float().mean().item()),
        "old_hidden_shape": list(old_hidden.shape),
        "fresh_hidden_shape": list(fresh_hidden.shape),
        "hidden_norm_old": float(torch.linalg.vector_norm(old_hidden).item()),
        "hidden_norm_fresh": float(torch.linalg.vector_norm(fresh_hidden).item()),
    }
    if old_hidden.shape == fresh_hidden.shape:
        flat_old, flat_fresh = old_hidden.flatten(), fresh_hidden.flatten()
        denominator = torch.linalg.vector_norm(flat_old) * torch.linalg.vector_norm(flat_fresh)
        result.update({
            "hidden_rms_delta": float(torch.sqrt(torch.mean((flat_fresh - flat_old) ** 2)).item()),
            "hidden_flattened_cosine_similarity": (
                float(torch.dot(flat_old, flat_fresh).div(denominator).item())
                if denominator.item() > 0 else None
            ),
            "hidden_shape_compatible": True,
        })
    else:
        result.update({
            "hidden_rms_delta": None,
            "hidden_flattened_cosine_similarity": None,
            "hidden_shape_compatible": False,
        })
    return result


def profile_json(profile: Mapping[str, Any]) -> dict[str, Any]:
    wanted = (
        "step", "slow_system", "slow_trigger_reason", "slow_age_before",
        "slow_age_after", "num_cond_actions", "ref_action_expired", "ref_action_first",
        "dp_action_first", "dp_action_chunk_mean", "dp_action_chunk_l2_mean",
        "dp_action_chunk_l2_max", "raw_action_prediction", "action_prediction",
        "aggregation_delta_ee6", "raw_aggregation_delta_ee6", "action_slew_applied",
    )
    return {key: profile.get(key) for key in wanted}


def task_succeeded(task_oracle, canonical_start_info: Mapping[str, Any], info: Mapping[str, Any], task: str) -> bool:
    return bool(task_oracle.get_task_info_for_set(canonical_start_info, info, {task}))


def branch_intervention(wrapper, branch: str, old: Mapping[str, Any], fresh: Mapping[str, Any], age: int) -> None:
    import torch
    device = wrapper._runtime_device()
    old_action = torch.as_tensor(old["slow_action"]).detach().clone().to(device, dtype=torch.float32)
    old_hidden = torch.as_tensor(old["slow_hidden"]).detach().clone().to(device)
    fresh_action = torch.as_tensor(fresh["slow_action"]).detach().clone().to(device, dtype=torch.float32)
    fresh_hidden = torch.as_tensor(fresh["slow_hidden"]).detach().clone().to(device)
    wrapper._slow_handover = None
    if branch == "frozen_baseline":
        wrapper.action, wrapper.hidden_states, wrapper.last_slow_step = old_action, old_hidden, 0
    elif branch == "fresh_hidden_only":
        wrapper.action, wrapper.hidden_states, wrapper.last_slow_step = old_action, fresh_hidden, 0
    elif branch == "fresh_ref_only":
        wrapper.action, wrapper.hidden_states, wrapper.last_slow_step = fresh_action, old_hidden, age
    elif branch == "full_refresh":
        wrapper.action, wrapper.hidden_states, wrapper.last_slow_step = fresh_action, fresh_hidden, age
    else:
        raise ValueError(branch)


def branch_contract_audit(wrapper, branch: str, old: Mapping[str, Any], fresh: Mapping[str, Any], age: int) -> dict[str, Any]:
    import torch
    expected_age = age if branch in {"frozen_baseline", "fresh_hidden_only"} else 0
    ref, count = wrapper._build_ref_actions(expected_age)
    action_expected = fresh["slow_action"] if branch in {"fresh_ref_only", "full_refresh"} else old["slow_action"]
    hidden_expected = fresh["slow_hidden"] if branch in {"fresh_hidden_only", "full_refresh"} else old["slow_hidden"]
    checks = {
        "last_slow_step": int(wrapper.last_slow_step),
        "expected_last_slow_step": age if branch in {"fresh_ref_only", "full_refresh"} else 0,
        "action_matches_expected": bool(torch.equal(wrapper.action.cpu(), torch.as_tensor(action_expected).to(wrapper.action.dtype))),
        "hidden_matches_expected": bool(torch.equal(wrapper.hidden_states.cpu(), torch.as_tensor(hidden_expected).to(wrapper.hidden_states.dtype))),
        "expected_slow_age_first": expected_age,
        "precomputed_num_cond_actions": int(count),
        "precomputed_ref_nonzero": int(torch.count_nonzero(ref).item()),
        "slow_handover_disabled": wrapper._slow_handover is None,
    }
    expected_count = 0 if branch in {"frozen_baseline", "fresh_hidden_only"} else 8
    checks["passed"] = bool(
        checks["last_slow_step"] == checks["expected_last_slow_step"]
        and checks["action_matches_expected"] and checks["hidden_matches_expected"]
        and checks["precomputed_num_cond_actions"] == expected_count
        and (expected_count != 0 or checks["precomputed_ref_nonzero"] == 0)
        and checks["slow_handover_disabled"]
    )
    return checks


def reset_pairwise(states: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    names = list(states)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            rows.append({"left": left, "right": right, **T.state_metrics(states[left], states[right])})
    return rows


def intervention_seed(seed: int, condition_id: str, age: int) -> int:
    value = hashlib.sha256(f"{seed}:{condition_id}:{age}".encode()).hexdigest()[:16]
    return int(value, 16)


def aggregate_numeric(rows: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[f"{key}_n"] = len(values)
        result[f"{key}_mean"] = float(np.mean(values)) if values else None
        result[f"{key}_median"] = float(np.median(values)) if values else None
        result[f"{key}_std"] = float(np.std(values)) if values else None
    return result


PRIMARY_KEYS = (
    "dp_action_first_ee6_l2_vs_baseline", "dp_action_first_gripper_sign_diff",
    "executed_action_ee6_l2_vs_baseline", "executed_action_gripper_sign_diff",
    "aggregation_delta_ee6", "temporal_transmission_ratio",
    "task_success_within_post_window", "first_success_post_step",
)


def summarize(intervention_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in intervention_rows if not row["task_success_before_intervention"]]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    task_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[(row["intervention_age"], row["branch"])].append(row)
        task_grouped[(row["intervention_age"], row["branch"], row["task"])].append(row)
    summary_rows = []
    for (age, branch), rows in sorted(grouped.items()):
        summary_rows.append({"intervention_age": age, "branch": branch, **aggregate_numeric(rows, PRIMARY_KEYS)})
    task_rows = []
    for (age, branch, task), rows in sorted(task_grouped.items()):
        task_rows.append({
            "intervention_age": age, "branch": branch, "task": task,
            "descriptive_only": True, **aggregate_numeric(rows, PRIMARY_KEYS),
        })
    highlight = {
        f"age{age}": {
            branch: next((row for row in summary_rows if row["intervention_age"] == age and row["branch"] == branch), None)
            for branch in ("fresh_hidden_only", "fresh_ref_only", "full_refresh")
        }
        for age in sorted({row["intervention_age"] for row in intervention_rows})
    }
    summary = {
        "primary_population": "anchors not successful before intervention",
        "excluded_pre_intervention_success_rows": len(intervention_rows) - len(eligible),
        "by_intervention_age_and_branch": summary_rows,
        "by_task": task_rows,
        "highlighted_paired_comparisons": highlight,
        "interpretation": "Descriptive diagnostic output; no hypothesis is automatically proven.",
    }
    return summary_rows, task_rows, summary


def make_task_oracle(dataset_root: Path):
    import hydra
    from omegaconf import OmegaConf
    calvin_root = T.configure_calvin_imports(dataset_root)
    path = calvin_root / "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    return hydra.utils.instantiate(OmegaConf.load(path)), path


def source_condition(path: Path, anchor: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    condition = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("slow_action", "slow_hidden", "inference_call_id"):
        if key not in condition:
            raise KeyError(f"{path} lacks {key}")
    if str(condition.get("condition_id")) != str(anchor["condition_id"]):
        raise ValueError(f"Condition id mismatch in {path}")
    if tuple(torch.as_tensor(condition["slow_action"]).shape) != (1, 8, 7):
        raise ValueError(f"Invalid source slow_action shape in {path}")
    hidden = torch.as_tensor(condition["slow_hidden"])
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"Invalid source slow_hidden shape in {path}")
    if not condition.get("same_inference_call") or not (
        condition.get("slow_action_inference_call_id")
        == condition.get("slow_hidden_inference_call_id")
        == condition.get("inference_call_id")
    ):
        raise ValueError(f"Source condition is not a same-call action/hidden pair: {path}")
    return condition


def fresh_condition(dual_system, processor, model_dtype, anchor, branchpoint, old, age, generalist_path, fingerprint):
    import torch
    from DiT_train.data_collection.collect_age_extended_expert import infer_condition
    action, hidden, call_id, normalization = infer_condition(
        dual_system.slow_system, processor, model_dtype,
        branchpoint["rgb_static"], str(anchor["instruction"]),
    )
    return {
        "condition_id": f"{anchor['condition_id']}_age_{age:02d}",
        "old_source_condition_id": old["condition_id"],
        "fresh_inference_call_id": call_id,
        "slow_action_inference_call_id": call_id,
        "slow_hidden_inference_call_id": call_id,
        "same_inference_call": True,
        "do_sample": False,
        "slow_action": torch.as_tensor(action).detach().cpu().to(torch.float32),
        "slow_hidden": torch.as_tensor(hidden).detach().cpu().to(torch.float16),
        "slow_action_normalization": normalization,
        "branchpoint_robot_obs": np.asarray(branchpoint["robot_obs"]).copy(),
        "branchpoint_scene_obs": np.asarray(branchpoint["scene_obs"]).copy(),
        "branchpoint_rgb_static_sha256": hashlib.sha256(np.asarray(branchpoint["rgb_static"]).tobytes()).hexdigest(),
        "branchpoint_source": "policy-induced state after executing prefix actions k=0..a-1",
        "generalist_path": str(generalist_path),
        "generalist_fingerprint": fingerprint,
        "generalist_calls_for_anchor_age": 1,
    }


def run_intervention(
    args, run_dir, env, task_oracle, wrapper_type, dual_system, processor, tokenizer,
    model_dtype, anchor, old_condition, age, generalist_fingerprint,
):
    import torch

    condition_id = str(anchor["condition_id"])
    index = args._index
    start_frame = int(anchor["anchor_frame"])
    start = index.load_frame(start_frame)
    obs = env.reset(robot_obs=start["robot_obs"].copy(), scene_obs=start["scene_obs"].copy())
    canonical_task_start_info = copy.deepcopy(env.get_info())
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
    for k in range(age):
        action = np.asarray(common.step(obs, str(anchor["instruction"]), k), dtype=np.float32).reshape(ACTION_DIM)
        obs, _, _, info = env.step(action.copy())
        if task_succeeded(task_oracle, canonical_task_start_info, info, str(anchor["task"])):
            success_before = True
            if first_success_prefix_step is None:
                first_success_prefix_step = k
    branchpoint = T.capture_env_state(obs)
    branchpoint_info = copy.deepcopy(env.get_info())
    controller_snapshot = snapshot_controller_state(common)
    if common.last_slow_step != 0:
        raise AssertionError("Common prefix unexpectedly refreshed slow condition")
    expected_prefix_count = 0 if age >= 8 else 8 - age
    ref, prefix_count = common._build_ref_actions(age)
    if prefix_count != expected_prefix_count or (age >= 8 and torch.count_nonzero(ref).item() != 0):
        raise AssertionError("Intervention branchpoint is not zero-ref under age_empty")

    fresh = fresh_condition(
        dual_system, processor, model_dtype, anchor, branchpoint, old_condition,
        age, args.generalist_path, generalist_fingerprint,
    )
    fresh_path = run_dir / "fresh_conditions" / f"{condition_id}_age_{age}.pt"
    torch.save(fresh, fresh_path)
    condition_metrics = condition_difference(old_condition, fresh)
    paired_rng = snapshot_rng()
    rng_before = rng_digest(paired_rng)

    wrappers = {}
    intervention_audits = {}
    for branch in BRANCHES:
        wrapper = T.build_wrapper(wrapper_type, dual_system, processor, tokenizer)
        restore_controller_state(wrapper, controller_snapshot)
        branch_intervention(wrapper, branch, old_condition, fresh, age)
        wrappers[branch] = wrapper
        intervention_audits[branch] = branch_contract_audit(wrapper, branch, old_condition, fresh, age)
    independence = runtime_independence(wrappers)
    if not independence["passed"] or not all(audit["passed"] for audit in intervention_audits.values()):
        raise AssertionError("Controller runtime independence or branch intervention contract failed")

    branch_order = list(BRANCHES)
    random.Random(intervention_seed(args.seed, condition_id, age)).shuffle(branch_order)
    reset_states = {}
    reset_records = []
    branch_results = {}
    branch_step_rows = []
    for branch in branch_order:
        branch_obs = env.reset(
            robot_obs=branchpoint["robot_obs"].copy(),
            scene_obs=branchpoint["scene_obs"].copy(),
        )
        reset_state = T.capture_env_state(branch_obs)
        reset_states[branch] = reset_state
        reset_records.append({
            "condition_id": condition_id, "intervention_age": age, "branch": branch,
            "restore_method": "env.reset(robot_obs=captured_robot_obs, scene_obs=captured_scene_obs)",
            **T.state_metrics(reset_state, branchpoint),
        })
        restore_rng(paired_rng)
        if rng_digest(snapshot_rng()) != rng_before:
            raise AssertionError(f"RNG restore failed for {branch}")
        wrapper = wrappers[branch]
        states = [reset_state]
        actions = []
        profiles = []
        successes = []
        first_success = None
        for j in range(args.post_steps):
            global_step = age + j
            action = np.asarray(
                wrapper.step(branch_obs, str(anchor["instruction"]), global_step),
                dtype=np.float32,
            ).reshape(ACTION_DIM)
            profile = profile_json(dict(wrapper.last_step_profile))
            if profile["slow_system"]:
                raise AssertionError(f"Forbidden post-intervention generalist call in {branch}")
            branch_obs, _, _, info = env.step(action.copy())
            success = task_succeeded(task_oracle, canonical_task_start_info, info, str(anchor["task"]))
            if success and first_success is None:
                first_success = j
            state = T.capture_env_state(branch_obs)
            actions.append(action.copy())
            profiles.append(profile)
            successes.append(success)
            states.append(state)
            branch_step_rows.append({
                "condition_id": condition_id, "task": anchor["task"],
                "intervention_age": age, "branch": branch, "post_step": j,
                "global_step": global_step, "executed_action": action.tolist(),
                "task_success": success, "profile": profile,
            })
        expected_first_count = 0 if branch in {"frozen_baseline", "fresh_hidden_only"} else 8
        if int(profiles[0]["num_cond_actions"]) != expected_first_count:
            raise AssertionError(
                f"{branch} first num_cond_actions={profiles[0]['num_cond_actions']}, expected {expected_first_count}"
            )
        branch_results[branch] = {
            "states": states, "actions": np.stack(actions), "profiles": profiles,
            "successes": successes, "first_success": first_success,
        }

    baseline = branch_results["frozen_baseline"]
    intervention_rows = []
    cumulative = {branch: 0.0 for branch in BRANCHES}
    for branch in BRANCHES:
        result = branch_results[branch]
        raw = np.asarray(result["profiles"][0]["dp_action_first"], dtype=np.float32)
        base_raw = np.asarray(baseline["profiles"][0]["dp_action_first"], dtype=np.float32)
        executed = result["actions"][0]
        base_executed = baseline["actions"][0]
        raw_delta = float(np.linalg.norm(raw[:6] - base_raw[:6]))
        executed_delta = float(np.linalg.norm(executed[:6] - base_executed[:6]))
        for j in range(args.post_steps):
            step_delta = float(np.linalg.norm(result["actions"][j, :6] - baseline["actions"][j, :6]))
            cumulative[branch] += step_delta
            metrics = T.state_metrics(result["states"][j + 1], baseline["states"][j + 1])
            for row in branch_step_rows:
                if (row["condition_id"] == condition_id and row["intervention_age"] == age
                        and row["branch"] == branch and row["post_step"] == j):
                    row.update({
                        **{f"state_vs_baseline_{key}": value for key, value in metrics.items()},
                        "executed_action_ee6_l2_vs_baseline": step_delta,
                        "cumulative_executed_action_ee6_difference": cumulative[branch],
                    })
                    break
        intervention_rows.append({
            "condition_id": condition_id, "task": anchor["task"],
            "intervention_age": age, "branch": branch,
            "task_success_before_intervention": success_before,
            "excluded_from_primary_task_success": success_before,
            "task_success_within_post_window": bool(any(result["successes"])),
            "first_success_post_step": result["first_success"],
            "dp_action_first_ee6_l2_vs_baseline": raw_delta,
            "dp_action_first_gripper_sign_diff": bool(np.sign(raw[-1]) != np.sign(base_raw[-1])),
            "dp_action_chunk_rms_vs_baseline": None,
            "dp_action_chunk_unavailable_reason": "core evaluator profile exposes first/mean/norm summaries, not the full chunk",
            "executed_action_ee6_l2_vs_baseline": executed_delta,
            "executed_action_gripper_sign_diff": bool(np.sign(executed[-1]) != np.sign(base_executed[-1])),
            "aggregation_delta_ee6": result["profiles"][0]["aggregation_delta_ee6"],
            "baseline_aggregation_delta_ee6": baseline["profiles"][0]["aggregation_delta_ee6"],
            "temporal_transmission_ratio": executed_delta / max(raw_delta, 1e-12),
            "final_robot_obs": result["states"][-1]["robot_obs"].tolist(),
            "final_scene_obs": result["states"][-1]["scene_obs"].tolist(),
            **condition_metrics,
        })

    pairwise = reset_pairwise(reset_states)
    artifact: dict[str, Any] = {
        "branch_order": np.asarray(branch_order),
        "branchpoint_robot_obs": branchpoint["robot_obs"],
        "branchpoint_scene_obs": branchpoint["scene_obs"],
        "branchpoint_rgb_static": branchpoint["rgb_static"],
        "branchpoint_rgb_gripper": branchpoint["rgb_gripper"],
        "branchpoint_depth_static": branchpoint["depth_static"],
        "branchpoint_depth_gripper": branchpoint["depth_gripper"],
    }
    for branch, result in branch_results.items():
        for key in ("robot_obs", "scene_obs", "rgb_static", "rgb_gripper", "depth_static", "depth_gripper"):
            artifact[f"{branch}_{key}"] = np.stack([state[key] for state in result["states"]])
        artifact[f"{branch}_executed_actions"] = result["actions"]
        artifact[f"{branch}_task_success"] = np.asarray(result["successes"], dtype=np.bool_)
    trajectory_path = run_dir / "trajectories" / f"{condition_id}_age_{age}.npz"
    np.savez_compressed(trajectory_path, **artifact)

    intervention_record = {
        "condition_id": condition_id, "task": anchor["task"], "instruction": anchor["instruction"],
        "intervention_age": age,
        "off_by_one_contract": f"executed common-prefix actions k=0..{age - 1}; intervention before action age={age}",
        "task_success_before_intervention": success_before,
        "first_success_prefix_step": first_success_prefix_step,
        "canonical_task_start_provenance": canonical_start_provenance,
        "canonical_task_start_info": canonical_task_start_info,
        "branchpoint_info": branchpoint_info,
        "fresh_condition_file": fresh_path.relative_to(run_dir).as_posix(),
        "fresh_inference_call_id": fresh["fresh_inference_call_id"],
        "fresh_calls": 1, "same_call_audit": fresh["same_inference_call"],
        "condition_difference": condition_metrics,
        "branch_order": branch_order, "paired_rng": rng_before,
        "runtime_independence": independence,
        "branch_contract_audits": intervention_audits,
        "branch_outcomes_and_primary_metrics": intervention_rows,
        "reset_records": reset_records, "reset_pairwise": pairwise,
        "trajectory_file": trajectory_path.relative_to(run_dir).as_posix(),
    }
    preflight_checks = {
        "source_condition_loaded": True,
        "common_prefix_reached": True,
        "intervention_ref_empty": prefix_count == 0 and torch.count_nonzero(ref).item() == 0,
        "fresh_call_exactly_once": fresh["generalist_calls_for_anchor_age"] == 1,
        "fresh_same_call": fresh["same_inference_call"],
        "controller_runtime_independent": independence["passed"],
        "uniform_branch_reset": len({row["restore_method"] for row in reset_records}) == 1,
        "reset_fidelity_recorded": len(reset_records) == 4 and len(pairwise) == 6,
        "rng_paired": True,
        "b0_zero_ref": intervention_audits["frozen_baseline"]["precomputed_num_cond_actions"] == 0,
        "b1_zero_ref": intervention_audits["fresh_hidden_only"]["precomputed_num_cond_actions"] == 0,
        "b2_eight_refs": intervention_audits["fresh_ref_only"]["precomputed_num_cond_actions"] == 8,
        "b3_eight_refs": intervention_audits["full_refresh"]["precomputed_num_cond_actions"] == 8,
        "b1_hidden_fresh_action_old": intervention_audits["fresh_hidden_only"]["passed"],
        "b2_hidden_old_action_fresh": intervention_audits["fresh_ref_only"]["passed"],
        "no_post_generalist_call": all(
            not profile["slow_system"] for result in branch_results.values() for profile in result["profiles"]
        ),
        "canonical_oracle_start_from_task_start": canonical_start_provenance["captured_before_common_prefix"],
        "env_step_action_copy_contract": True,
        "outputs_saved": fresh_path.is_file() and trajectory_path.is_file(),
    }
    if not all(preflight_checks.values()):
        raise AssertionError(f"Preflight/contract check failed: {preflight_checks}")
    intervention_record["preflight_checks"] = preflight_checks
    return intervention_record, intervention_rows, branch_step_rows, reset_records, pairwise


def initial_manifest(args, source_manifest, source_audit, anchors, status, checkpoint_audit=None, calls=0, preflight=None):
    return {
        "schema_version": SCHEMA_VERSION, "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnosis_only": True,
        "not_a_deployment_policy": True,
        "source_run_dir": str(args.source_run_dir),
        "source_manifest_sha256": source_audit["source_manifest_sha256"],
        "source_git_commit": source_manifest.get("code_git_commit", "unknown"),
        "current_git_commit": T.git_commit(),
        "source_current_specialist": {
            "source_path": source_audit["source_specialist_path"],
            "current_path": source_audit["current_specialist_path"],
            "source_sha256": source_audit["source_specialist_sha256"],
            "current_sha256": source_audit["current_specialist_sha256"],
        },
        "source_current_generalist": {
            "source_path": source_audit["source_generalist_path"],
            "current_path": source_audit["current_generalist_path"],
            "source_fingerprint": source_audit["source_generalist_fingerprint"],
            "current_fingerprint": source_audit["current_generalist_fingerprint"],
        },
        "source_validation": source_audit,
        "intervention_ages": args.intervention_ages,
        "post_steps": args.post_steps, "seed": args.seed,
        "selected_anchor_count": len(anchors),
        "expected_fresh_generalist_calls": len(anchors) * len(args.intervention_ages),
        "actual_fresh_generalist_calls": calls,
        "invalid_cases_skipped": [],
        "branch_definitions": {
            "frozen_baseline": "old action + old hidden; last_slow_step=0",
            "fresh_hidden_only": "old action + fresh hidden; last_slow_step=0; zero explicit refs",
            "fresh_ref_only": "fresh action + old hidden; last_slow_step=intervention_age",
            "full_refresh": "fresh action + fresh hidden; last_slow_step=intervention_age",
        },
        "rng_pairing_contract": "snapshot torch CPU + all CUDA RNG after the one fresh call; restore after each uniform env reset before branch rollout",
        "branch_order": "deterministically shuffled per (seed, condition_id, intervention_age)",
        "reset_restore_method": "all four branches use env.reset(robot_obs=captured_robot_obs, scene_obs=captured_scene_obs)",
        "controller_snapshot_fields": list(SNAPSHOT_FIELDS),
        "checkpoint_missing_keys": None if checkpoint_audit is None else checkpoint_audit["missing_keys"],
        "checkpoint_unexpected_keys": None if checkpoint_audit is None else checkpoint_audit["unexpected_keys"],
        "checkpoint_loading": checkpoint_audit,
        "task_oracle": "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml via OmegaConf + hydra.utils.instantiate",
        "generalist_call_contract": "exactly one do_sample=False call per valid anchor-age; action and hidden shared by all branches",
        "post_intervention_generalist_calls_forbidden": True,
        "fast_num_inference_steps": 10, "temporal_aggregation_changed": False,
        "slow_handover_steps": 0, "action_slew_limits": 0,
        "preflight_contracts": preflight,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run and args.preflight_only:
        raise ValueError("--dry_run and --preflight_only are mutually exclusive")
    if args.post_steps <= 0:
        raise ValueError("--post_steps must be positive")
    args.intervention_ages = parse_ages(args.intervention_ages)
    source_manifest, source_anchors, source_audit = validate_source(args)
    selected = source_anchors[:args.preflight_anchors] if args.preflight_only else source_anchors
    run_dir = prepare_run_dir(args.run_name)
    task_counts = dict(sorted(Counter(anchor["task"] for anchor in selected).items()))
    eligibility = {
        "source_anchor_count": len(source_anchors), "selected_anchor_count": len(selected),
        "intervention_ages": args.intervention_ages,
        "eligible_anchor_age_pairs": len(selected) * len(args.intervention_ages),
        "expected_fresh_generalist_calls": len(selected) * len(args.intervention_ages),
        "task_counts": task_counts,
    }
    write_jsonl(run_dir / "anchors.jsonl", selected)
    if args.dry_run:
        write_jsonl(run_dir / "interventions.jsonl", [])
        write_jsonl(run_dir / "branch_steps.jsonl", [])
        write_csv(run_dir / "intervention_summary.csv", [])
        write_csv(run_dir / "task_summary.csv", [])
        write_json(run_dir / "summary.json", {"status": "dry_run_complete", "eligibility": eligibility})
        write_json(run_dir / "reset_fidelity.json", {"status": "not_run", "reason": "dry_run"})
        write_json(run_dir / "manifest.json", initial_manifest(
            args, source_manifest, source_audit, selected, "dry_run_complete", calls=0,
        ))
        result = {"run_dir": str(run_dir), "status": "dry_run_complete", **eligibility}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    import torch
    seed_everything(args.seed)
    (run_dir / "fresh_conditions").mkdir()
    (run_dir / "trajectories").mkdir()
    write_json(run_dir / "manifest.json", initial_manifest(
        args, source_manifest, source_audit, selected, "initializing",
    ))
    dataset_root = Path(source_manifest["dataset_root"]).expanduser().resolve()
    args.dataset_root = dataset_root
    args.fast_num_inference_steps = 10
    args._index = T.CalvinLanguageIndex(dataset_root)
    dual_system, processor, tokenizer, model_dtype, device, checkpoint_audit = T.load_models(args)
    if checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]:
        raise RuntimeError(
            f"M1 checkpoint incompatibility: missing={checkpoint_audit['missing_keys']}, "
            f"unexpected={checkpoint_audit['unexpected_keys']}"
        )
    if checkpoint_audit["sha256"] != source_audit["source_specialist_sha256"]:
        raise RuntimeError("Loaded specialist checkpoint hash differs from source")
    if checkpoint_audit["generalist_checkpoint_fingerprint"] != source_audit["source_generalist_fingerprint"]:
        raise RuntimeError("Loaded generalist fingerprint differs from source")
    task_oracle, task_oracle_path = make_task_oracle(dataset_root)
    env = T.make_env(args._index, args.use_egl)
    wrapper_type = T.frozen_wrapper_class()
    interventions = []
    intervention_rows = []
    branch_steps = []
    reset_records = []
    reset_pairwise_rows = []
    fresh_calls = 0
    preflight_contracts = []
    try:
        for anchor_i, anchor in enumerate(selected):
            old_path = args.source_run_dir / "conditions" / f"{anchor['condition_id']}.pt"
            old = source_condition(old_path, anchor)
            for age in args.intervention_ages:
                record, rows, steps, resets, pairwise = run_intervention(
                    args, run_dir, env, task_oracle, wrapper_type, dual_system,
                    processor, tokenizer, model_dtype, anchor, old, age,
                    source_audit["current_generalist_fingerprint"],
                )
                fresh_calls += 1
                interventions.append(record)
                intervention_rows.extend(rows)
                branch_steps.extend(steps)
                reset_records.extend(resets)
                reset_pairwise_rows.extend([
                    {"condition_id": anchor["condition_id"], "intervention_age": age, **row}
                    for row in pairwise
                ])
                preflight_contracts.append(record["preflight_checks"])
            print(f"[{anchor_i + 1}/{len(selected)}] {anchor['condition_id']} complete", flush=True)
    finally:
        env.close()
    expected_calls = len(selected) * len(args.intervention_ages)
    if fresh_calls != expected_calls:
        raise AssertionError(f"fresh generalist calls {fresh_calls} != expected {expected_calls}")
    summary_rows, task_rows, summary = summarize(intervention_rows)
    summary.update({
        "status": "preflight_complete" if args.preflight_only else "complete",
        "anchors": len(selected), "intervention_age_pairs": len(interventions),
        "fresh_generalist_calls": fresh_calls,
    })
    reset_fidelity = {
        "status": "measured",
        "restore_method": "env.reset(robot_obs=captured_robot_obs, scene_obs=captured_scene_obs)",
        "branch_vs_captured": reset_records,
        "branch_pairwise": reset_pairwise_rows,
        "warning": "Inspect fidelity before causal interpretation; no reset error is hidden or threshold-filtered.",
    }
    write_jsonl(run_dir / "interventions.jsonl", interventions)
    write_jsonl(run_dir / "branch_steps.jsonl", branch_steps)
    write_csv(run_dir / "intervention_summary.csv", summary_rows)
    write_csv(run_dir / "task_summary.csv", task_rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "reset_fidelity.json", reset_fidelity)
    status = "preflight_complete" if args.preflight_only else "complete"
    manifest = initial_manifest(
        args, source_manifest, source_audit, selected, status,
        checkpoint_audit, fresh_calls, preflight_contracts,
    )
    manifest["task_oracle_path"] = str(task_oracle_path)
    write_json(run_dir / "manifest.json", manifest)

    # Preflight requirement 20: every output family must be readable again.
    json.loads((run_dir / "manifest.json").read_text())
    read_jsonl(run_dir / "interventions.jsonl")
    read_jsonl(run_dir / "branch_steps.jsonl")
    with np.load(run_dir / interventions[0]["trajectory_file"], allow_pickle=False) as archive:
        if not archive.files:
            raise AssertionError("Saved trajectory artifact is empty")
    torch.load(run_dir / interventions[0]["fresh_condition_file"], map_location="cpu", weights_only=False)
    result = {
        "run_dir": str(run_dir), "status": status, "anchors": len(selected),
        "intervention_age_pairs": len(interventions), "fresh_generalist_calls": fresh_calls,
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
    parser.add_argument("--post_steps", type=int, default=8)
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
