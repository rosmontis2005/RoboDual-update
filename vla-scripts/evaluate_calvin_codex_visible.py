"""Visible-GUI wrapper for CALVIN evaluation.

This entrypoint preserves the evaluation logic from
`evaluate_calvin_codex_test.py` and only overrides environment creation to
keep the simulator window visible.
"""

import argparse
import os
from pathlib import Path

DEFAULT_CALVIN_ROOT = Path(__file__).resolve().parents[2] / "calvin"
DEFAULT_GENERALIST_PATH = Path(__file__).resolve().parents[2] / "models" / "generalist"
DEFAULT_SPECIALIST_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "specialist" / "Specialist+Depth+Gripper.pt"
)
os.environ.setdefault("CALVIN_ROOT", DEFAULT_CALVIN_ROOT.as_posix())

import evaluate_calvin_codex_test as base


def make_env(dataset_path, observation_space, device, use_egl):
    val_folder = Path(dataset_path) / "validation"
    from calvin_env_wrapper import CalvinEnvWrapperRaw

    return CalvinEnvWrapperRaw(
        val_folder,
        observation_space,
        device,
        show_gui=True,
        use_egl=use_egl,
    )


base.make_env = make_env


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
    parser.add_argument("--num_sequences", default=1000, type=int)
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
    parser.add_argument("--profile_steps", action="store_true")
    parser.add_argument("--profile_init", action="store_true")
    parser.set_defaults(
        dataset_subdir="calvin_debug_dataset",
        num_sequences=5,
        ep_len=120,
        max_subtasks=1,
        load_in_4bit=True,
        low_cpu_mem_usage=True,
    )
    args = parser.parse_args()

    base.main(args)
