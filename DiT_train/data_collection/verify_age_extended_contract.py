#!/usr/bin/env python3
"""Independent contract verifier for Age-Extended Expert Dataset runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
COLLECTOR_PATH = HERE / "collect_age_extended_expert.py"


def load_collector():
    spec = importlib.util.spec_from_file_location("age_extended_collector", COLLECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(COLLECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def torch_load_cpu(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_source_frame(dataset_root: Path, reference: dict[str, Any]) -> np.ndarray:
    path = dataset_root / reference["relative_path"]
    with np.load(path) as archive:
        return np.asarray(archive[reference["key"]]).copy()


def reference_at(reference: dict[str, Any], frame: int, key: str) -> dict[str, Any]:
    result = dict(reference)
    source = Path(result["relative_path"])
    match = re.match(r"^(.*?)(\d+)(\.npz)$", source.name)
    if not match:
        raise ValueError(f"Cannot rewrite CALVIN frame reference {source}")
    prefix, digits, suffix = match.groups()
    result.update(
        frame_index=frame,
        relative_path=str(source.with_name(f"{prefix}{frame:0{len(digits)}d}{suffix}")),
        key=key,
    )
    return result


def verify_run(run_dir: Path) -> dict[str, Any]:
    collector = load_collector()
    run_dir = run_dir.expanduser().resolve()
    required = ["manifest.json", "conditions", "samples.jsonl", "anchors.jsonl", "audit_summary.json"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stored_audit = json.loads((run_dir / "audit_summary.json").read_text())
    anchors = read_jsonl(run_dir / "anchors.jsonl")
    samples = read_jsonl(run_dir / "samples.jsonl")
    dataset_root = Path(manifest["dataset_root"])
    source_index = collector.CalvinExpertIndex(dataset_root, "training")
    failures: list[str] = []
    checked = Counter()

    def check(predicate: bool, message: str, category: str) -> None:
        checked[category] += 1
        if not predicate:
            failures.append(message)

    check(manifest.get("schema_version") == collector.SCHEMA_VERSION, "schema version mismatch", "manifest")
    check(manifest.get("status") == "complete", "manifest status is not complete", "manifest")
    check(manifest.get("dataset_source_split") == "training", "source is not CALVIN training", "source")
    check(manifest.get("with_tactile") is False, "with_tactile must be false", "modalities")
    check(manifest.get("target_source") == collector.TARGET_SOURCE, "target source mismatch", "source")
    check(manifest.get("condition_source") == collector.CONDITION_SOURCE, "condition source mismatch", "source")
    check(manifest.get("generalist_inference", "").endswith("predict_action(**inputs, do_sample=False)"), "non-runtime inference declaration", "generalist")
    check(manifest.get("generalist_loader_source") == collector.GENERALIST_LOADER_SOURCE,
          "generalist loader source mismatch", "generalist")
    check(manifest.get("generalist_accelerator_prepare") is False,
          "generalist unexpectedly uses Accelerator.prepare", "generalist")
    check(manifest.get("generalist_action_normalization_source") == collector.GENERALIST_ACTION_NORMALIZATION_SOURCE,
          "generalist action normalization source mismatch", "generalist")
    check(manifest.get("unique_anchors") == len(anchors), "manifest anchor count mismatch", "counts")
    check(manifest.get("unique_conditions") == len(anchors), "manifest condition count mismatch", "counts")
    check(manifest.get("total_samples") == len(samples), "manifest sample count mismatch", "counts")
    check(manifest.get("expected_samples") == len(anchors) * 12, "manifest expected sample count mismatch", "counts")
    check(len(samples) == len(anchors) * 12, "samples != anchors * 12", "counts")
    check(bool(manifest.get("fingerprints", {}).get("generalist_checkpoint")), "missing generalist fingerprint", "manifest")
    check(bool(manifest.get("fingerprints", {}).get("processor_tokenizer")), "missing processor fingerprint", "manifest")
    check(bool(manifest.get("code_git_commit")), "missing git commit", "manifest")

    condition_files = sorted((run_dir / "conditions").glob("condition_*.pt"))
    conditions = {item.stem: torch_load_cpu(item) for item in condition_files}
    check(len(conditions) == len(anchors), "condition file count mismatch", "counts")
    anchor_by_condition = {row["condition_id"]: row for row in anchors}
    check(len(anchor_by_condition) == len(anchors), "duplicate condition_id in anchors", "counts")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_splits: dict[str, set[str]] = defaultdict(set)
    inference_ids: set[str] = set()

    for condition_id, condition in conditions.items():
        check(condition_id in anchor_by_condition, f"orphan condition {condition_id}", "condition")
        check(condition.get("condition_id") == condition_id, f"{condition_id}: embedded ID mismatch", "condition")
        check(condition.get("source") == collector.CONDITION_SOURCE, f"{condition_id}: wrong source", "condition")
        check(condition.get("same_inference_call") is True, f"{condition_id}: same-call flag false", "condition")
        call_ids = {
            condition.get("inference_call_id"), condition.get("slow_action_inference_call_id"),
            condition.get("slow_hidden_inference_call_id"),
        }
        check(len(call_ids) == 1 and None not in call_ids, f"{condition_id}: action/hidden call IDs differ", "condition")
        inference_ids.update(call_ids)
        action, hidden = condition.get("slow_action"), condition.get("slow_hidden")
        check(torch.is_tensor(action) and tuple(action.shape) == (1, 8, 7), f"{condition_id}: slow_action shape", "shape")
        check(torch.is_tensor(hidden) and hidden.ndim == 3 and hidden.shape[0] == 1 and hidden.shape[1] > 0,
              f"{condition_id}: slow_hidden shape", "shape")
        if torch.is_tensor(hidden) and hidden.ndim == 3:
            check(condition.get("hidden_seq_len") == hidden.shape[1], f"{condition_id}: hidden_seq_len", "shape")
        check(condition.get("do_sample") is False, f"{condition_id}: do_sample", "generalist")
        check(condition.get("teacher_forcing") is False, f"{condition_id}: teacher forcing", "generalist")
        check(condition.get("future_expert_actions_supplied_to_generalist") is False,
              f"{condition_id}: future actions supplied", "generalist")
        for field in (
            "generalist_path", "generalist_checkpoint_fingerprint", "processor_tokenizer_fingerprint",
            "generalist_dtype", "generalist_quantization", "code_git_commit", "collector_schema_version",
            "generalist_loader_source", "generalist_action_normalization_source",
        ):
            check(bool(condition.get(field)), f"{condition_id}: missing {field}", "condition")
        check(condition.get("collector_schema_version") == collector.SCHEMA_VERSION,
              f"{condition_id}: condition schema mismatch", "condition")
        check(condition.get("generalist_loader_source") == collector.GENERALIST_LOADER_SOURCE,
              f"{condition_id}: loader source mismatch", "generalist")
        check(condition.get("generalist_accelerator_prepare") is False,
              f"{condition_id}: Accelerator.prepare must be false", "generalist")
        check(condition.get("generalist_action_normalization_source") == collector.GENERALIST_ACTION_NORMALIZATION_SOURCE,
              f"{condition_id}: action normalization source mismatch", "generalist")
        normalization = condition.get("slow_action_normalization", {})
        channels = normalization.get("raw_channels_per_step")
        check(normalization.get("normalized_shape") == [1, 8, 7],
              f"{condition_id}: normalized action shape metadata", "generalist")
        check(isinstance(channels, int) and channels >= 7,
              f"{condition_id}: raw action channels metadata", "generalist")
        if isinstance(channels, int):
            check(normalization.get("raw_numel") == 8 * channels,
                  f"{condition_id}: raw action numel metadata", "generalist")
            check(normalization.get("extra_channels_dropped_per_step") == channels - 7,
                  f"{condition_id}: dropped action channels metadata", "generalist")
        if condition_id in anchor_by_condition:
            anchor = anchor_by_condition[condition_id]
            expected_frames = collector.required_source_frames(
                anchor["task_start_frame"], anchor["anchor_frame"]
            )
            check(condition.get("required_source_frames") == expected_frames,
                  f"{condition_id}: required source frame range mismatch", "source")
            check(all(source_index.frame_path(frame).is_file() for frame in expected_frames),
                  f"{condition_id}: required source frame missing", "source")

    check(len(inference_ids) == len(conditions), "inference call IDs are not unique per anchor", "generalist")

    for sample in samples:
        sid = sample.get("sample_id", "<missing>")
        cid = sample.get("condition_id")
        check(cid in conditions, f"{sid}: missing condition", "condition")
        if cid not in conditions:
            continue
        by_condition[cid].append(sample)
        trajectory_splits[sample["trajectory_id"]].add(sample["split"])
        anchor = anchor_by_condition[cid]
        condition = conditions[cid]
        age = int(sample["slow_age"])
        current, anchor_frame = int(sample["current_frame"]), int(sample["anchor_frame"])
        start, end = int(sample["task_start_frame"]), int(sample["task_end_frame_inclusive"])
        check(age in collector.AGES, f"{sid}: illegal age {age}", "age")
        check(current - anchor_frame == age, f"{sid}: age != current-anchor", "age")
        check(sample["task_local_step"] == current - start, f"{sid}: task local step", "age")
        check(anchor_frame == anchor["anchor_frame"], f"{sid}: anchor record mismatch", "condition")
        check(sample.get("dataset_source_split") == "training", f"{sid}: non-training source", "source")
        check(sample["trajectory_id"] == anchor["trajectory_id"] and sample["split"] == anchor["split"],
              f"{sid}: anchor trajectory/split mismatch", "split")
        check(sample["task"] == anchor["task"] and sample["instruction"] == anchor["instruction"],
              f"{sid}: task/instruction mismatch", "condition")
        expected_previous = current if current == start else current - 1
        check(sample["previous_frame"] == expected_previous, f"{sid}: previous frame crossed boundary", "previous")
        expected_history_indices = list(range(max(start, current - 4), current))
        check(sample["history_source_frames"] == expected_history_indices, f"{sid}: history indices", "history")
        check(all(frame < current for frame in sample["history_source_frames"]), f"{sid}: action leakage", "history")
        check(sample["history_zero_pad_count"] == 4 - len(expected_history_indices), f"{sid}: padding count", "history")
        history = np.asarray(sample["hist_action_before"], dtype=np.float32)
        check(history.shape == (4, 7), f"{sid}: history shape", "shape")
        expected_history = np.zeros((4, 7), dtype=np.float32)
        for offset, frame in enumerate(expected_history_indices, start=4 - len(expected_history_indices)):
            ref = reference_at(sample["current_rgb_static"], frame, "rel_actions")
            expected_history[offset] = load_source_frame(dataset_root, ref).astype(np.float32)
        check(np.array_equal(history, expected_history), f"{sid}: history values/source mismatch", "history")

        expected_target_indices = list(range(current, current + 8))
        check(sample["target_source_frames"] == expected_target_indices, f"{sid}: target indices", "target")
        check(expected_target_indices[-1] <= end, f"{sid}: target crosses task", "target")
        target = np.asarray(sample["target_rel_actions"], dtype=np.float32)
        check(target.shape == (8, 7), f"{sid}: target shape", "shape")
        expected_target = []
        for frame in expected_target_indices:
            ref = reference_at(sample["current_rgb_static"], frame, "rel_actions")
            expected_target.append(load_source_frame(dataset_root, ref).astype(np.float32))
        check(np.array_equal(target, np.stack(expected_target)), f"{sid}: target differs from CALVIN expert", "target")
        check(sample.get("target_source") == collector.TARGET_SOURCE, f"{sid}: target source", "source")
        expected_sources = sorted(set([anchor_frame, current, expected_previous] + expected_history_indices + expected_target_indices))
        check(sample.get("source_frame_indices") == expected_sources, f"{sid}: incomplete source frame indices", "source")

        expected_ref, count, mask = collector.build_reference(condition["slow_action"], age)
        actual_ref = torch.tensor(sample["ref_action"], dtype=torch.float32)
        check(tuple(actual_ref.shape) == (1, 8, 7), f"{sid}: ref shape", "shape")
        check(sample["ref_valid_count"] == count, f"{sid}: ref valid count", "reference")
        check(sample["ref_valid_mask"] == mask.tolist(), f"{sid}: ref mask", "reference")
        check(sample["ref_expired"] == (age >= 8), f"{sid}: ref expired", "reference")
        check(torch.equal(actual_ref, expected_ref.to(torch.float32)), f"{sid}: ref suffix mismatch", "reference")
        if age >= 8:
            check(torch.count_nonzero(actual_ref).item() == 0, f"{sid}: expired ref nonzero", "reference")

        for field, key, frame in (
            ("current_rgb_static", "rgb_static", current),
            ("previous_rgb_static", "rgb_static", expected_previous),
            ("current_depth_static", "depth_static", current),
            ("current_rgb_gripper", "rgb_gripper", current),
            ("current_depth_gripper", "depth_gripper", current),
        ):
            reference = sample[field]
            check(reference["key"] == key and reference["frame_index"] == frame,
                  f"{sid}: bad {field} reference", "observation")
            check((dataset_root / reference["relative_path"]).is_file(), f"{sid}: missing source frame", "observation")
        if sample.get("materialized_observation"):
            check((run_dir / sample["materialized_observation"]).is_file(), f"{sid}: missing materialized obs", "observation")

        anchor_prop = np.asarray(sample["anchor_proprio"], dtype=np.float32)
        current_prop = np.asarray(sample["current_proprio"], dtype=np.float32)
        source_anchor_robot = load_source_frame(
            dataset_root, reference_at(sample["current_rgb_static"], anchor_frame, "robot_obs")
        ).astype(np.float32)
        source_current_robot = load_source_frame(
            dataset_root, reference_at(sample["current_rgb_static"], current, "robot_obs")
        ).astype(np.float32)
        check(np.array_equal(np.asarray(sample["anchor_robot_obs"], dtype=np.float32), source_anchor_robot),
              f"{sid}: anchor robot_obs source", "freshness")
        check(np.array_equal(np.asarray(sample["current_robot_obs"], dtype=np.float32), source_current_robot),
              f"{sid}: current robot_obs source", "freshness")
        check(np.array_equal(anchor_prop, collector.selected_proprio(source_anchor_robot)), f"{sid}: anchor proprio source", "freshness")
        check(np.array_equal(current_prop, collector.selected_proprio(source_current_robot)), f"{sid}: current proprio source", "freshness")
        prop_delta = current_prop - anchor_prop
        check(np.allclose(prop_delta, sample["anchor_to_current_proprio_delta"]["vector"]), f"{sid}: proprio delta", "freshness")
        check(np.isclose(np.linalg.norm(prop_delta), sample["anchor_to_current_proprio_delta"]["l2_norm"]), f"{sid}: proprio norm", "freshness")
        anchor_scene = np.asarray(sample["anchor_scene_state"], dtype=np.float32)
        current_scene = np.asarray(sample["current_scene_state"], dtype=np.float32)
        source_anchor_scene = load_source_frame(
            dataset_root, reference_at(sample["current_rgb_static"], anchor_frame, "scene_obs")
        ).astype(np.float32)
        source_current_scene = load_source_frame(
            dataset_root, reference_at(sample["current_rgb_static"], current, "scene_obs")
        ).astype(np.float32)
        check(np.array_equal(anchor_scene, source_anchor_scene), f"{sid}: anchor scene source", "freshness")
        check(np.array_equal(current_scene, source_current_scene), f"{sid}: current scene source", "freshness")
        scene_delta = current_scene - anchor_scene
        check(np.allclose(scene_delta, sample["anchor_to_current_scene_delta"]["vector"]), f"{sid}: scene delta", "freshness")
        check(np.isclose(np.linalg.norm(scene_delta), sample["anchor_to_current_scene_delta"]["l2_norm"]), f"{sid}: scene norm", "freshness")

    for cid in conditions:
        rows = by_condition[cid]
        check(sorted(row["slow_age"] for row in rows) == list(range(12)), f"{cid}: incomplete ages", "age")
        check(len({row["condition_path"] for row in rows}) == 1, f"{cid}: condition reference differs", "condition")
        age7 = [row for row in rows if row["slow_age"] == 7]
        age8 = [row for row in rows if row["slow_age"] == 8]
        check(len(age7) == 1 and age7[0]["ref_valid_count"] == 1, f"{cid}: age7 != one ref", "reference")
        check(len(age8) == 1 and age8[0]["ref_valid_count"] == 0, f"{cid}: age8 != zero ref", "reference")
    check(all(len(splits) == 1 for splits in trajectory_splits.values()), "trajectory split leakage", "split")
    for trajectory_id, splits in trajectory_splits.items():
        check(splits == {collector.stable_split(trajectory_id)}, f"{trajectory_id}: unstable split", "split")
    check(stored_audit.get("status") == "passed", "stored audit was not passed", "audit")

    result = {
        "status": "passed" if not failures else "failed",
        "run_dir": str(run_dir), "failures": failures,
        "checks_executed": int(sum(checked.values())), "checks_by_category": dict(sorted(checked.items())),
        "unique_anchors": len(anchors), "unique_conditions": len(conditions), "total_samples": len(samples),
        "trajectory_split_leakage": any(len(splits) != 1 for splits in trajectory_splits.values()),
        "inspection": stored_audit.get("inspection"),
    }
    if failures:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--cpu_contract_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = load_collector()
    if args.cpu_contract_test or args.run_dir is None:
        print(json.dumps(collector.cpu_contract_test(), indent=2, sort_keys=True))
    if args.run_dir is not None:
        print(json.dumps(verify_run(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
