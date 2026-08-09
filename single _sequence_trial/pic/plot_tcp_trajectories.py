#!/usr/bin/env python3
"""Visualize paired TCP trajectories for sequence 60.

The script reads simulator-truth TCP link-frame positions from every collected
step. It never reconstructs position by integrating the policy action. One PNG
is written per subtask, plus a compact five-subtask 3-D overview and a JSON
summary of the plotted arrays.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
TRIAL_ROOT = HERE.parent
DEFAULT_RUNS = {
    "Original fixed-8": TRIAL_ROOT
    / "original_8_steps/runs/seq060_seed42_original_fixed8",
    "Fixed age-12": TRIAL_ROOT
    / "fixed_12_steps/runs/seq060_seed42_fixed_age12",
}
COLORS = {"Original fixed-8": "#2166ac", "Fixed age-12": "#d95f02"}
TARGET_OBJECTS = {
    "push_blue_block_left": "block_blue",
    "lift_pink_block_slider": "block_pink",
}


def load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def subtask_directories(run_dir: Path) -> list[Path]:
    directories = sorted((run_dir / "tensors/seq_060").glob("subtask_*"))
    if len(directories) != 5:
        raise RuntimeError(f"Expected 5 subtask directories in {run_dir}, found {len(directories)}")
    return directories


def extract_subtask(directory: Path) -> dict[str, Any]:
    step_files = sorted(directory.glob("step_*.pt"))
    if not step_files:
        raise RuntimeError(f"No step payloads found in {directory}")

    tcp_positions: list[list[float]] = []
    tcp_quaternions: list[list[float]] = []
    gripper_commands: list[float] = []
    gripper_opening: list[float] = []
    slow_steps: list[int] = []
    condition_counts: list[int] = []
    object_positions: list[list[float]] = []
    task = ""
    success = False

    for expected_step, step_file in enumerate(step_files):
        payload = load_torch(step_file)
        meta = payload["meta"]
        step = int(meta["step"])
        if step != expected_step:
            raise AssertionError(
                f"Non-contiguous steps in {directory}: expected {expected_step}, found {step}"
            )
        task = str(meta["subtask"])
        success = bool(meta["task_success"])
        environment = payload["environment"]
        profile = payload["evaluator_profile"]

        if step == 0:
            pre_tcp = environment["pre_physics"]["tcp"]
            tcp_positions.append(list(pre_tcp["world_link_frame_position"]))
            tcp_quaternions.append(list(pre_tcp["world_link_frame_orientation_quaternion"]))

        post_physics = environment["post_physics"]
        post_tcp = post_physics["tcp"]
        tcp_positions.append(list(post_tcp["world_link_frame_position"]))
        tcp_quaternions.append(list(post_tcp["world_link_frame_orientation_quaternion"]))

        action = np.asarray(environment["executed_action"], dtype=np.float64).reshape(-1)
        gripper_commands.append(float(action[-1]))
        joints = post_physics["gripper_joints"]
        gripper_opening.append(float(sum(abs(float(joint["position"])) for joint in joints)))
        condition_counts.append(int(profile.get("num_cond_actions") or 0))
        if bool(profile.get("slow_system")):
            slow_steps.append(step)

        target_name = TARGET_OBJECTS.get(task)
        if target_name is not None:
            object_positions.append(
                list(post_physics["scene_info"]["movable_objects"][target_name]["current_pos"])
            )

    return {
        "subtask_index": int(directory.name.split("_")[1]),
        "task": task,
        "success": success,
        "steps": len(step_files),
        "tcp": np.asarray(tcp_positions, dtype=np.float64),
        "tcp_quaternion": np.asarray(tcp_quaternions, dtype=np.float64),
        "gripper_command": np.asarray(gripper_commands, dtype=np.float64),
        "gripper_opening": np.asarray(gripper_opening, dtype=np.float64),
        "slow_steps": np.asarray(slow_steps, dtype=np.int64),
        "condition_counts": np.asarray(condition_counts, dtype=np.int64),
        "object": None if not object_positions else np.asarray(object_positions, dtype=np.float64),
    }


def load_runs(run_dirs: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    for label, run_dir in run_dirs.items():
        if not (run_dir / "summary.json").is_file():
            raise FileNotFoundError(f"Incomplete or missing run: {run_dir}")
        loaded[label] = [extract_subtask(path) for path in subtask_directories(run_dir)]
    reference_tasks = [item["task"] for item in next(iter(loaded.values()))]
    for label, subtasks in loaded.items():
        if [item["task"] for item in subtasks] != reference_tasks:
            raise AssertionError(f"Subtask order mismatch for {label}")
    return loaded


def padded_limits(arrays: list[np.ndarray], dimensions: tuple[int, ...]) -> list[tuple[float, float]]:
    limits = []
    for dim in dimensions:
        values = np.concatenate([array[:, dim] for array in arrays])
        low, high = float(values.min()), float(values.max())
        pad = max((high - low) * 0.08, 0.005)
        limits.append((low - pad, high + pad))
    return limits


def shade_empty_reference(ax: plt.Axes, condition_counts: np.ndarray) -> None:
    empty = condition_counts == 0
    start = None
    for index, is_empty in enumerate(np.r_[empty, False]):
        if is_empty and start is None:
            start = index
        elif not is_empty and start is not None:
            ax.axvspan(start - 0.5, index - 0.5, color="#d95f02", alpha=0.07, linewidth=0)
            start = None


def plot_subtask(
    output_dir: Path,
    subtask_index: int,
    paired: dict[str, dict[str, Any]],
) -> Path:
    task = next(iter(paired.values()))["task"]
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.25, 1, 1))
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_z = fig.add_subplot(grid[0, 2])
    ax_xy_time = fig.add_subplot(grid[1, 1])
    ax_gripper = fig.add_subplot(grid[1, 2])
    ax_gripper_opening = ax_gripper.twinx()

    tcp_arrays = [item["tcp"] for item in paired.values()]
    xlim, ylim, zlim = padded_limits(tcp_arrays, (0, 1, 2))
    for label, item in paired.items():
        color = COLORS[label]
        tcp = item["tcp"]
        steps = np.arange(item["steps"] + 1)
        ax3d.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=color, lw=2.0, label=label)
        ax3d.scatter(*tcp[0], color=color, marker="o", s=55, edgecolor="white", linewidth=0.8)
        ax3d.scatter(*tcp[-1], color=color, marker="s", s=55, edgecolor="white", linewidth=0.8)
        slow_indices = item["slow_steps"]
        ax3d.scatter(
            tcp[slow_indices, 0], tcp[slow_indices, 1], tcp[slow_indices, 2],
            color=color, marker="x", s=24, alpha=0.7,
        )

        ax_xy.plot(tcp[:, 0], tcp[:, 1], color=color, lw=1.8, label=label)
        ax_xy.scatter(tcp[0, 0], tcp[0, 1], color=color, marker="o", s=38)
        ax_xy.scatter(tcp[-1, 0], tcp[-1, 1], color=color, marker="s", s=38)

        ax_z.plot(steps, tcp[:, 2], color=color, lw=1.8, label=label)
        ax_z.scatter(slow_indices, tcp[slow_indices, 2], color=color, marker="x", s=20)
        if label == "Fixed age-12":
            shade_empty_reference(ax_z, item["condition_counts"])

        action_steps = np.arange(item["steps"])
        ax_xy_time.plot(action_steps, tcp[1:, 0], color=color, lw=1.5, label=f"{label}: X")
        ax_xy_time.plot(action_steps, tcp[1:, 1], color=color, lw=1.5, ls="--", label=f"{label}: Y")
        if label == "Fixed age-12":
            shade_empty_reference(ax_xy_time, item["condition_counts"])

        ax_gripper.step(
            action_steps, item["gripper_command"], where="post", color=color, lw=1.4, label=label
        )
        ax_gripper_opening.plot(
            action_steps,
            item["gripper_opening"] * 1000.0,
            color=color,
            lw=1.0,
            ls=":",
            alpha=0.8,
            label=f"{label}: joint opening",
        )
        if label == "Fixed age-12":
            shade_empty_reference(ax_gripper, item["condition_counts"])

        object_path = item["object"]
        if object_path is not None:
            object_label = TARGET_OBJECTS[task].replace("block_", "") + " block"
            ax3d.plot(
                object_path[:, 0], object_path[:, 1], object_path[:, 2],
                color=color, lw=1.2, ls=":", alpha=0.85, label=f"{label}: {object_label}",
            )
            ax_xy.plot(
                object_path[:, 0], object_path[:, 1], color=color, lw=1.2, ls=":", alpha=0.85
            )
            ax_z.plot(
                np.arange(item["steps"]), object_path[:, 2], color=color, lw=1.2, ls=":", alpha=0.85
            )

    ax3d.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)")
    ax3d.set_xlim(*xlim)
    ax3d.set_ylim(*ylim)
    ax3d.set_zlim(*zlim)
    ax3d.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax3d.view_init(elev=25, azim=-55)
    ax3d.set_title("TCP world trajectory\n○ start   □ end   × slow call")
    ax3d.legend(loc="best", fontsize=8)

    ax_xy.set(xlabel="World X (m)", ylabel="World Y (m)", title="Top view (XY)")
    ax_xy.set_xlim(*xlim)
    ax_xy.set_ylim(*ylim)
    ax_xy.set_aspect("equal", adjustable="box")

    ax_z.set(xlabel="Environment step", ylabel="World Z (m)", title="TCP height")
    ax_xy_time.set(xlabel="Environment step", ylabel="Position (m)", title="TCP X / Y over time")
    ax_xy_time.legend(fontsize=7, ncol=2)
    ax_gripper.set(
        xlabel="Environment step",
        ylabel="Executed gripper command",
        title="Gripper command and measured joint opening",
        ylim=(-1.15, 1.15),
    )
    ax_gripper_opening.set_ylabel("Sum |finger joint position| (mm)")
    command_lines, command_labels = ax_gripper.get_legend_handles_labels()
    opening_lines, opening_labels = ax_gripper_opening.get_legend_handles_labels()
    ax_gripper.legend(
        command_lines + opening_lines,
        command_labels + opening_labels,
        fontsize=7,
        loc="best",
    )

    for ax in (ax_xy, ax_z, ax_xy_time, ax_gripper):
        ax.grid(True, alpha=0.25)
    display_task = task.replace("_", " ")
    fig.suptitle(
        f"Sequence 60 · subtask {subtask_index + 1}/5 · {display_task}\n"
        "fixed-8 vs age-12 · pale orange = age-12 empty reference",
        fontsize=14,
        linespacing=1.35,
    )
    output = output_dir / f"subtask_{subtask_index:02d}_{task}_tcp_trajectory.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_overview(output_dir: Path, loaded: dict[str, list[dict[str, Any]]]) -> Path:
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    for index in range(5):
        ax = fig.add_subplot(2, 3, index + 1, projection="3d")
        arrays = [loaded[label][index]["tcp"] for label in loaded]
        xlim, ylim, zlim = padded_limits(arrays, (0, 1, 2))
        for label, subtasks in loaded.items():
            tcp = subtasks[index]["tcp"]
            ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=COLORS[label], lw=1.7)
            ax.scatter(*tcp[0], color=COLORS[label], marker="o", s=28)
            ax.scatter(*tcp[-1], color=COLORS[label], marker="s", s=28)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
        ax.view_init(elev=25, azim=-55)
        display_task = loaded["Original fixed-8"][index]["task"].replace("_", " ")
        ax.set_title(f"{index + 1}. {display_task}", fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    legend_ax = fig.add_subplot(2, 3, 6)
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=COLORS[label], lw=3, label=label) for label in loaded
    ] + [
        Line2D([0], [0], marker="o", color="gray", lw=0, label="start"),
        Line2D([0], [0], marker="s", color="gray", lw=0, label="end"),
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=13)
    legend_ax.text(
        0.5, 0.25,
        "Each panel uses shared axis limits\nfor the two strategies.",
        ha="center", va="center", fontsize=11, transform=legend_ax.transAxes,
    )
    fig.suptitle("Sequence 60: paired TCP world trajectories", fontsize=17)
    output = output_dir / "sequence_060_all_subtasks_tcp_overview.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def serializable_summary(loaded: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, subtasks in loaded.items():
        rows = []
        for item in subtasks:
            tcp = item["tcp"]
            increments = np.diff(tcp, axis=0)
            rows.append(
                {
                    "subtask_index": item["subtask_index"],
                    "task": item["task"],
                    "steps": item["steps"],
                    "success": item["success"],
                    "slow_calls": int(len(item["slow_steps"])),
                    "empty_reference_steps": int(np.sum(item["condition_counts"] == 0)),
                    "tcp_start_xyz_m": tcp[0].tolist(),
                    "tcp_end_xyz_m": tcp[-1].tolist(),
                    "tcp_path_length_m": float(np.linalg.norm(increments, axis=1).sum()),
                    "tcp_net_displacement_m": float(np.linalg.norm(tcp[-1] - tcp[0])),
                    "tcp_xyz_min_m": tcp.min(axis=0).tolist(),
                    "tcp_xyz_max_m": tcp.max(axis=0).tolist(),
                }
            )
        output[label] = rows
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed8_run", type=Path, default=DEFAULT_RUNS["Original fixed-8"])
    parser.add_argument("--age12_run", type=Path, default=DEFAULT_RUNS["Fixed age-12"])
    parser.add_argument("--output_dir", type=Path, default=HERE)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "Original fixed-8": args.fixed8_run.expanduser().resolve(),
        "Fixed age-12": args.age12_run.expanduser().resolve(),
    }
    loaded = load_runs(run_dirs)
    outputs = []
    for index in range(5):
        outputs.append(
            plot_subtask(
                output_dir,
                index,
                {label: subtasks[index] for label, subtasks in loaded.items()},
            )
        )
    outputs.append(plot_overview(output_dir, loaded))

    summary = {
        "source_runs": {label: str(path) for label, path in run_dirs.items()},
        "coordinate_source": "environment.{pre,post}_physics.tcp.world_link_frame_position",
        "trajectory_convention": "initial pre-step TCP followed by every post-step TCP",
        "figures": [path.name for path in outputs],
        "subtasks": serializable_summary(loaded),
    }
    summary_path = output_dir / "tcp_trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"figures": [str(path) for path in outputs], "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
