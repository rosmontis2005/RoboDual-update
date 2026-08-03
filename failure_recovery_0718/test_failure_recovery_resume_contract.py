"""Contract tests for safe recovery-collection resume."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "vla-scripts" / "evaluate_calvin_failure_recovery_scale_0718.py"
sys.path.insert(0, SCRIPT.parent.as_posix())
SPEC = importlib.util.spec_from_file_location("recovery_scale_eval_0718", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FailureRecoveryResumeContractTest(unittest.TestCase):
    def test_stack_release_seed_sweep_is_distinct_and_bounded(self):
        offsets = [MODULE._stack_release_offset(seed) for seed in range(9)]
        self.assertEqual(len({tuple(item) for item in offsets}), 9)
        self.assertTrue(all(np.max(np.abs(item[:2])) <= 0.004 for item in offsets))
        self.assertTrue(all(abs(float(item[2])) <= 0.006 for item in offsets))

    def test_persisted_restore_reloads_bullet_after_controller_reset(self):
        events = []

        class Physics:
            def restoreState(self, **kwargs):
                events.append(("bullet", kwargs["fileName"]))

        class Storage:
            def __init__(self, name):
                self.name = name
                self.gripper_action = None

            def reset_from_storage(self, payload):
                events.append((self.name, payload))

            def control_gripper(self, action):
                events.append(("gripper", action))

        class Bullet:
            p = Physics()
            cid = 7
            robot = Storage("robot")
            scene = Storage("scene")

        Bullet.robot.target_pos = None
        Bullet.robot.target_orn = None

        class Env:
            @staticmethod
            def get_obs():
                return {
                    "robot_obs": np.asarray(
                        [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        dtype=np.float32,
                    )
                }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE.torch,
            "load",
            return_value={
                "robot": {"gripper_action": -1},
                "scene": "scene_state",
            },
        ):
            MODULE.restore_persisted_failure_state(
                Env(), Bullet(), Path(directory), "state"
            )
        self.assertEqual(
            [item[0] for item in events],
            ["bullet", "robot", "scene", "bullet", "gripper"],
        )
        np.testing.assert_array_equal(Bullet.robot.target_pos, [1, 2, 3])
        np.testing.assert_array_equal(Bullet.robot.target_orn, [4, 5, 6])
        self.assertEqual(Bullet.robot.gripper_action, -1)

    def test_terminal_padding_preserves_gripper_and_completes_chunk(self):
        actions = np.zeros((41, 7), dtype=np.float32)
        actions[:, -1] = -1
        actions[-1, -1] = 1
        padded, count = MODULE._terminal_padded_actions(actions)
        self.assertEqual(count, 7)
        self.assertEqual(padded.shape, (48, 7))
        np.testing.assert_array_equal(padded[41:, :6], 0)
        np.testing.assert_array_equal(padded[41:, -1], 1)

    def test_oracle_start_is_required_unless_fallback_is_explicit(self):
        class Env:
            @staticmethod
            def get_info():
                return {"fallback": True}

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                MODULE.load_persisted_oracle_start(
                    Path(directory), "state", Env()
                )
            payload, source = MODULE.load_persisted_oracle_start(
                Path(directory), "state", Env(), allow_missing=True
            )
            self.assertEqual(payload, {"fallback": True})
            self.assertEqual(source, "failure_state_fallback")

    def test_resume_loads_only_complete_committed_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            writer = MODULE.FailureRecoveryWriter(root)
            state_id = "s0200_t0_k040"
            branch_id = f"{state_id}_base_seed_00"
            writer.states = [{"failure_state_id": state_id, "sequence_i": 200, "split": "train"}]
            writer.branches = [{"branch_id": branch_id, "failure_state_id": state_id}]
            writer.checkpoint()
            for suffix in (".npz", ".bullet", "_model.pt", "_simulator.pt"):
                (writer.states_dir / f"{state_id}{suffix}").write_bytes(b"payload")
            (writer.branches_dir / f"{branch_id}.npz").write_bytes(b"payload")
            (writer.conditions_dir / f"{branch_id}.pt").write_bytes(b"payload")

            resumed = MODULE.FailureRecoveryWriter(root, resume=True)
            self.assertEqual(resumed.states, writer.states)
            self.assertEqual(resumed.branches, writer.branches)

    def test_resume_refuses_finalized_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            writer = MODULE.FailureRecoveryWriter(root)
            writer.checkpoint()
            (root / "collection_summary.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                MODULE.FailureRecoveryWriter(root, resume=True)


if __name__ == "__main__":
    unittest.main()
