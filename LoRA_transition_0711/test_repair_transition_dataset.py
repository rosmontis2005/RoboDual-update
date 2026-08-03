"""Tests for committed-action recovery from persisted history."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from collect_transition_rollouts import Candidate, capture_frame
from finalize_interrupted_collection import reconstruct_candidates
from repair_transition_dataset import fill_split_shortfalls, recover_actions


class RecoverActionsTest(unittest.TestCase):
    def _frame(self, root, step, history, action):
        path = Path(root) / f"step_{step:04d}.npz"
        np.savez(path, hist_action_before=history, rel_actions=action)
        return path

    def test_recovers_previous_commands_from_next_history(self):
        commands = np.arange(21, dtype=np.float32).reshape(3, 7) / 21
        histories = [np.zeros((4, 7), dtype=np.float32)]
        for command in commands:
            history = histories[-1].copy()
            history[:-1] = history[1:]
            history[-1] = command
            histories.append(history)
        with tempfile.TemporaryDirectory() as directory:
            files = [
                self._frame(directory, index, history, np.full(7, 0.001, dtype=np.float32))
                for index, history in enumerate(histories)
            ]
            recovered, stats = recover_actions(files)
        np.testing.assert_allclose(recovered, commands)
        self.assertEqual(stats["recoverable_actions"], 3)
        self.assertEqual(stats["max_history_shift_error"], 0.0)

    def test_rejects_discontinuous_history(self):
        with tempfile.TemporaryDirectory() as directory:
            files = [
                self._frame(directory, 0, np.zeros((4, 7), dtype=np.float32), np.zeros(7)),
                self._frame(directory, 1, np.ones((4, 7), dtype=np.float32), np.zeros(7)),
            ]
            with self.assertRaisesRegex(ValueError, "not temporally continuous"):
                recover_actions(files)

    def test_capture_frame_owns_action_storage(self):
        action = np.arange(7, dtype=np.float32)
        obs = {"robot_obs": np.zeros(15, dtype=np.float32)}
        frame = capture_frame(obs, action, np.zeros((4, 7), dtype=np.float32))
        action[:] = -1
        np.testing.assert_array_equal(frame["rel_actions"], np.arange(7, dtype=np.float32))

    def test_repaired_terminal_boundary_keeps_only_complete_chunks(self):
        actions = [np.zeros(7, dtype=np.float32) for _ in range(12)]
        conditions = [{
            "condition_id": 0,
            "step": 0,
            "refresh_age": None,
            "slow_action": np.zeros((1, 8, 7), dtype=np.float32),
            "old_condition_id": None,
        }]
        args = Namespace(
            history_steps=4,
            action_chunk_size=8,
            empty_ref_after_age=8,
            high_conflict_prev_threshold=0.18,
            high_conflict_old_new_threshold=0.18,
            high_conflict_jerk_threshold=0.24,
        )
        candidates = reconstruct_candidates("seq00000_sub0_close_drawer", actions, conditions, args)
        self.assertEqual([candidate.step for candidate in candidates], [4])
        self.assertEqual(candidates[0].step + args.action_chunk_size, len(actions))

    def test_shortfall_fill_preserves_split_and_unique_windows(self):
        def candidate(step):
            return Candidate("trajectory", "validation", step, "normal", 0, None, 0, None,
                             None, None, None, None, False)

        selected, added = fill_split_shortfalls(
            [candidate(0), candidate(1), candidate(2)], [candidate(0)], {"validation": 3}, 42
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(added), 2)
        self.assertEqual(len({(item.trajectory_id, item.step) for item in selected}), 3)
        self.assertTrue(all(item.split == "validation" for item in selected))


if __name__ == "__main__":
    unittest.main()
