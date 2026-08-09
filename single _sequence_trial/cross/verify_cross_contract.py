#!/usr/bin/env python3
"""CPU/static verification for cross-state restoration and factorial contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_cross_experiment as cross


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_bundle", type=Path, default=HERE / "boundary_states.pt")
    args = parser.parse_args()
    bundle_path = args.state_bundle.resolve()
    bundle = cross.load_bundle(bundle_path)
    sidecar = json.loads(bundle_path.with_suffix(".json").read_text())
    assert sha256_file(bundle_path) == sidecar["bundle_sha256"]
    for label, state in bundle["states"].items():
        source = Path(state["source_step_file"])
        assert source.is_file()
        assert sha256_file(source) == state["source_step_sha256"]

    assert [step for step in range(40) if cross.expected_contract("P8", step)[0]] == [0, 7, 15, 23, 31, 39]
    assert [step for step in range(40) if cross.expected_contract("P12", step)[0]] == [0, 12, 24, 36]
    assert [cross.expected_contract("P8", step)[1] for step in range(16)] == [
        8, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 8
    ]
    assert [cross.expected_contract("P12", step)[1] for step in range(12)] == [
        8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0
    ]
    orders = [cross.cell_order(index) for index in range(4)]
    assert all(set(order) == set(cross.CELL_BASE_ORDER) for order in orders)
    for position in range(4):
        assert {orders[replicate][position] for replicate in range(4)} == set(cross.CELL_BASE_ORDER)

    cross.seed_trial(12345)
    first = torch.randn(8)
    cross.seed_trial(12345)
    second = torch.randn(8)
    assert torch.equal(first, second)

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    env = cross.fixed8_collector.original.make_env(
        str(cross.CALVIN_ROOT / "dataset/calvin_debug_dataset"),
        observation_space,
        torch.device("cpu"),
        False,
    )
    restore_args = SimpleNamespace(
        restore_observation_atol=2e-5,
        restore_position_atol=2e-5,
        restore_velocity_atol=2e-5,
    )
    audits = {}
    repeatability = {}
    try:
        for label in ("S8", "S12"):
            obs_a, record_a = cross.restore_boundary(env, bundle["states"][label], restore_args)
            obs_b, record_b = cross.restore_boundary(env, bundle["states"][label], restore_args)
            audits[label] = record_b["audit"]
            repeatability[label] = {
                "robot_obs_max_abs": cross.max_abs(obs_a["robot_obs"], obs_b["robot_obs"]),
                "scene_obs_max_abs": cross.max_abs(obs_a["scene_obs"], obs_b["scene_obs"]),
            }
            assert repeatability[label]["robot_obs_max_abs"] <= 2e-5
            assert repeatability[label]["scene_obs_max_abs"] <= 2e-5
    finally:
        cross.close_env_once(env)

    s8_tcp = np.asarray(bundle["states"]["S8"]["pre_physics"]["tcp"]["world_link_frame_position"])
    s12_tcp = np.asarray(bundle["states"]["S12"]["pre_physics"]["tcp"]["world_link_frame_position"])
    state_difference = {
        "tcp_distance_m": float(np.linalg.norm(s12_tcp - s8_tcp)),
        "robot_obs_max_abs": cross.max_abs(
            bundle["states"]["S8"]["robot_obs"], bundle["states"]["S12"]["robot_obs"]
        ),
        "scene_obs_max_abs": cross.max_abs(
            bundle["states"]["S8"]["scene_obs"], bundle["states"]["S12"]["scene_obs"]
        ),
    }
    assert state_difference["tcp_distance_m"] > 0.01
    controller_target_lag = {}
    for label, state in bundle["states"].items():
        controller = state["controller_state"]
        tcp_position = np.asarray(state["pre_physics"]["tcp"]["world_com_position"])
        lag = float(np.linalg.norm(np.asarray(controller["target_pos"]) - tcp_position))
        controller_target_lag[label] = {
            "target_minus_tcp_l2_m": lag,
            "prior_action_count": int(controller["prior_action_count"]),
            "saved_action_units": controller["saved_action_units"],
        }
        # A target-pose controller may lag the physical TCP, but a second
        # max_rel_pos scaling error produces a ~0.36 m discrepancy here.
        assert lag < 0.03
    print(
        json.dumps(
            {
                "bundle_provenance_passed": True,
                "schedule_contract_passed": True,
                "latin_order_contract_passed": True,
                "rng_reset_contract_passed": True,
                "restore_audits": audits,
                "restore_repeatability": repeatability,
                "S8_vs_S12_difference": state_difference,
                "controller_target_lag": controller_target_lag,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
