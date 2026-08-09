"""Non-invasive tensor/state tracing for the original RoboDual evaluator.

The collector wraps Python methods and installs forward hooks only.  It never
sets an RNG state, samples a random value, or changes a model input/output.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def cpu_clone(value: Any) -> Any:
    """Make a serialization-safe, exact CPU snapshot without consuming RNG."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): cpu_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_clone(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    # PyBullet contact records and similar tuple-like values normally hit the
    # list/tuple branch.  Unknown diagnostic objects are represented explicitly.
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def tensor_descriptor(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"type": "torch.Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"type": "numpy.ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: tensor_descriptor(item) for key, item in value.items()}
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "item_schema": None if not value else tensor_descriptor(value[0]),
        }
    return type(value).__name__


def rng_snapshot() -> dict[str, Any]:
    result = {"torch_cpu": torch.random.get_rng_state().clone()}
    if torch.cuda.is_available():
        result["torch_cuda_all"] = [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physics_state(env: Any) -> dict[str, Any]:
    """Collect simulator truth, including velocities absent from robot_obs."""
    raw_env = env.unwrapped
    robot = raw_env.robot
    physics = raw_env.p
    joint_records = []
    for joint_id in robot.arm_joint_ids:
        state = physics.getJointState(robot.robot_uid, joint_id, physicsClientId=robot.cid)
        joint_records.append(
            {
                "joint_id": int(joint_id),
                "position": float(state[0]),
                "velocity": float(state[1]),
                "reaction_forces": list(state[2]),
                "applied_motor_torque": float(state[3]),
            }
        )
    gripper_records = []
    for joint_id in robot.gripper_joint_ids:
        state = physics.getJointState(robot.robot_uid, joint_id, physicsClientId=robot.cid)
        gripper_records.append(
            {
                "joint_id": int(joint_id),
                "position": float(state[0]),
                "velocity": float(state[1]),
                "reaction_forces": list(state[2]),
                "applied_motor_torque": float(state[3]),
            }
        )
    link_state = physics.getLinkState(
        robot.robot_uid,
        robot.tcp_link_id,
        computeLinkVelocity=1,
        computeForwardKinematics=1,
        physicsClientId=robot.cid,
    )
    return {
        "arm_joints": joint_records,
        "gripper_joints": gripper_records,
        "tcp": {
            "world_com_position": list(link_state[0]),
            "world_com_orientation_quaternion": list(link_state[1]),
            "local_inertial_position": list(link_state[2]),
            "local_inertial_orientation_quaternion": list(link_state[3]),
            "world_link_frame_position": list(link_state[4]),
            "world_link_frame_orientation_quaternion": list(link_state[5]),
            "world_linear_velocity": list(link_state[6]),
            "world_angular_velocity": list(link_state[7]),
        },
        "robot_contacts": cpu_clone(
            physics.getContactPoints(bodyA=robot.robot_uid, physicsClientId=robot.cid)
        ),
        "scene_info": cpu_clone(raw_env.scene.get_info()),
    }


class TraceWriter:
    """Writes one atomic torch payload per executed environment step."""

    SCHEMA_VERSION = 1

    def __init__(self, run_dir: Path, manifest: dict[str, Any]):
        self.run_dir = run_dir.resolve()
        self.tensor_root = self.run_dir / "tensors"
        self.tensor_root.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.events_path.touch()
        self.manifest = {"schema_version": self.SCHEMA_VERSION, **manifest}
        self.manifest["started_at_unix"] = time.time()
        self.manifest["files"] = {
            "events": "events.jsonl",
            "tensor_root": "tensors",
            "summary": "summary.json",
        }
        self._write_json(self.run_dir / "manifest.json", self.manifest)
        self.total_tensor_bytes = 0
        self.steps_written = 0

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def write_step(self, payload: dict[str, Any]) -> Path:
        meta = payload["meta"]
        relative = Path(
            f"seq_{int(meta['sequence_index']):03d}/"
            f"subtask_{int(meta['subtask_index']):02d}_{meta['subtask']}/"
            f"step_{int(meta['step']):04d}.pt"
        )
        target = self.tensor_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".pt.incomplete")
        torch.save(cpu_clone(payload), temporary)
        os.replace(temporary, target)
        size = target.stat().st_size
        digest = sha256_file(target)
        event = {
            **meta,
            "event": "step",
            "tensor_file": str(Path("tensors") / relative),
            "bytes": size,
            "sha256": digest,
            "schema": tensor_descriptor(payload),
        }
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.total_tensor_bytes += size
        self.steps_written += 1
        return target

    def finalize(self, summary: dict[str, Any]) -> None:
        summary = {
            **summary,
            "steps_written": self.steps_written,
            "tensor_bytes": self.total_tensor_bytes,
            "tensor_gib": self.total_tensor_bytes / 1024**3,
            "finished_at_unix": time.time(),
        }
        self._write_json(self.run_dir / "summary.json", summary)


class OnlineTraceCapture:
    """Capture generalist, specialist and DiT tensors without changing inference."""

    def __init__(
        self,
        evaluator: Any,
        writer: TraceWriter,
        expected_slow_call: Callable[[int], bool] | None = None,
        schedule_label: str = "fixed_mod8",
    ):
        self.evaluator = evaluator
        self.writer = writer
        self.expected_slow_call = expected_slow_call or (
            lambda step: step == 0 or (step + 1) % 8 == 0
        )
        self.schedule_label = str(schedule_label)
        self.current: dict[str, Any] | None = None
        self._handles: list[Any] = []
        self._patched: list[tuple[Any, str, Any]] = []
        self.fast = evaluator._fast_system()
        self.slow = evaluator._slow_system()
        self.dit = self.fast.model
        self._install()

    def begin_step(
        self,
        *,
        sequence_index: int,
        subtask_index: int,
        subtask: str,
        instruction: str,
        step: int,
        pre_obs: dict[str, Any],
        pre_info: dict[str, Any],
        pre_physics: dict[str, Any],
    ) -> None:
        if self.current is not None:
            raise RuntimeError("Previous trace step was not finalized")
        self.current = {
            "meta": {
                "sequence_index": int(sequence_index),
                "subtask_index": int(subtask_index),
                "subtask": str(subtask),
                "instruction": str(instruction),
                "step": int(step),
            },
            "environment": {
                "pre_observation": cpu_clone(pre_obs),
                "pre_info": cpu_clone(pre_info),
                "pre_physics": cpu_clone(pre_physics),
            },
            "generalist": {"called": False},
            "specialist": {
                "encoder_features": {},
                "adapted_conditions": {},
                "dit_calls": [],
            },
        }

    def finalize_step(
        self,
        *,
        executed_action: Any,
        post_obs: dict[str, Any],
        post_info: dict[str, Any],
        post_physics: dict[str, Any],
        task_success: bool,
        profile: dict[str, Any],
    ) -> Path:
        if self.current is None:
            raise RuntimeError("No active trace step")
        expected_slow = bool(self.expected_slow_call(int(self.current["meta"]["step"])))
        actual_slow = bool(profile.get("slow_system"))
        if actual_slow != expected_slow:
            raise AssertionError(
                f"{self.schedule_label} mismatch at step {self.current['meta']['step']}: "
                f"expected slow={expected_slow}, observed slow={actual_slow}"
            )
        specialist = self.current["specialist"]
        required_specialist = (
            "inputs",
            "condition_data",
            "initial_noise",
            "dit_common_inputs",
            "output_action_chunk",
            "rng_pre",
            "rng_post",
        )
        missing_specialist = [key for key in required_specialist if key not in specialist]
        if missing_specialist:
            raise AssertionError(f"Missing required specialist trace fields: {missing_specialist}")
        inference_steps = int(self.fast.num_inference_steps)
        expected_dit_calls = inference_steps * (2 if self.fast.with_cfg else 1)
        if len(specialist["dit_calls"]) != expected_dit_calls:
            raise AssertionError(
                f"Expected {expected_dit_calls} DiT calls, captured {len(specialist['dit_calls'])}"
            )
        if len(specialist.get("scheduler_steps", [])) != inference_steps:
            raise AssertionError(
                f"Expected {inference_steps} scheduler steps, captured "
                f"{len(specialist.get('scheduler_steps', []))}"
            )
        vision_features = specialist["encoder_features"].get("vision_encoder.forward_features", [])
        expected_vision_calls = 2 + int(bool(self.fast.with_gripper))
        if len(vision_features) != expected_vision_calls:
            raise AssertionError(
                f"Expected {expected_vision_calls} DINO forward_features calls, captured {len(vision_features)}"
            )
        if self.fast.with_depth:
            expected_depth_calls = 1 + int(bool(self.fast.with_gripper))
            depth_features = specialist["encoder_features"].get("depth_encoder", [])
            if len(depth_features) != expected_depth_calls:
                raise AssertionError(
                    f"Expected {expected_depth_calls} depth encoder calls, captured {len(depth_features)}"
                )
        if actual_slow:
            # These are the exact post-reshape tensors cached by the historical
            # evaluator and subsequently consumed by the specialist path.
            self.current["generalist"]["cached_action_chunk_8x7"] = cpu_clone(self.evaluator.action)
            self.current["generalist"]["cached_generalist_condition"] = cpu_clone(
                self.evaluator.hidden_states
            )
            generalist = self.current["generalist"]
            required_generalist = ("inputs", "generation", "feature_embeddings", "output")
            missing_generalist = [key for key in required_generalist if key not in generalist]
            if missing_generalist:
                raise AssertionError(f"Missing required generalist trace fields: {missing_generalist}")
            missing_features = [
                key for key in ("vision_backbone", "projector")
                if key not in generalist["feature_embeddings"]
            ]
            if missing_features:
                raise AssertionError(f"Missing required generalist feature hooks: {missing_features}")
        self.current["meta"]["task_success"] = bool(task_success)
        self.current["environment"].update(
            {
                "executed_action": cpu_clone(executed_action),
                "post_robot_obs": cpu_clone(post_obs.get("robot_obs")),
                "post_scene_obs": cpu_clone(post_obs.get("scene_obs")),
                "post_info": cpu_clone(post_info),
                "post_physics": cpu_clone(post_physics),
            }
        )
        self.current["evaluator_profile"] = cpu_clone(profile)
        target = self.writer.write_step(self.current)
        self.current = None
        return target

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched.clear()

    def _patch_method(self, owner: Any, name: str, function: Any) -> None:
        original = getattr(owner, name)
        self._patched.append((owner, name, original))
        setattr(owner, name, types.MethodType(function, owner))

    def _capture_once_hook(self, destination: str, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if self.current is None:
                return
            store = self.current["specialist"][destination]
            if name not in store:
                store[name] = cpu_clone(output)

        return hook

    def _capture_list_hook(self, destination: str, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if self.current is None:
                return
            store = self.current["specialist"][destination]
            store.setdefault(name, []).append(cpu_clone(output))

        return hook

    def _install(self) -> None:
        collector = self
        original_fast_predict = self.fast.predict_action

        def fast_predict(_owner: Any, *args: Any, **kwargs: Any):
            if collector.current is not None:
                collector.current["specialist"]["rng_pre"] = rng_snapshot()
                collector.current["specialist"]["inputs"] = cpu_clone(kwargs)
            output = original_fast_predict(*args, **kwargs)
            if collector.current is not None:
                collector.current["specialist"]["output_action_chunk"] = cpu_clone(output)
                collector.current["specialist"]["rng_post"] = rng_snapshot()
            return output

        self._patch_method(self.fast, "predict_action", fast_predict)

        original_conditional_sample = self.fast.conditional_sample

        def conditional_sample(_owner: Any, condition_data: Any, *args: Any, **kwargs: Any):
            if collector.current is not None:
                collector.current["specialist"]["condition_data"] = cpu_clone(condition_data)
            return original_conditional_sample(condition_data, *args, **kwargs)

        self._patch_method(self.fast, "conditional_sample", conditional_sample)

        original_scheduler_step = self.fast.noise_scheduler.step

        def scheduler_step(_owner: Any, model_output: Any, timestep: Any, sample: Any, *args: Any, **kwargs: Any):
            output = original_scheduler_step(model_output, timestep, sample, *args, **kwargs)
            if collector.current is not None:
                collector.current["specialist"].setdefault("scheduler_steps", []).append(
                    {
                        "model_output": cpu_clone(model_output),
                        "timestep": cpu_clone(timestep),
                        "sample_in": cpu_clone(sample),
                        "prev_sample": cpu_clone(output.prev_sample),
                        "pred_original_sample": cpu_clone(getattr(output, "pred_original_sample", None)),
                    }
                )
            return output

        self._patch_method(self.fast.noise_scheduler, "step", scheduler_step)

        original_slow_predict = self.slow.predict_action

        def slow_predict(_owner: Any, *args: Any, **kwargs: Any):
            if collector.current is not None:
                collector.current["generalist"]["called"] = True
                collector.current["generalist"]["rng_pre"] = rng_snapshot()
                collector.current["generalist"]["inputs"] = cpu_clone(kwargs)
            output = original_slow_predict(*args, **kwargs)
            if collector.current is not None:
                action, transferred_hidden = output
                collector.current["generalist"]["output"] = {
                    "action_chunk_pre_reshape": cpu_clone(action),
                    "transferred_hidden_states": cpu_clone(transferred_hidden),
                }
                collector.current["generalist"]["rng_post"] = rng_snapshot()
            return output

        self._patch_method(self.slow, "predict_action", slow_predict)

        if hasattr(self.slow, "generate"):
            original_generate = self.slow.generate

            def generate(_owner: Any, *args: Any, **kwargs: Any):
                output = original_generate(*args, **kwargs)
                if collector.current is not None:
                    generated = {"sequences": cpu_clone(output.sequences)}
                    hidden_steps = getattr(output, "hidden_states", None)
                    if hidden_steps:
                        last_layer = torch.cat([layers[-1] for layers in hidden_steps], dim=1)
                        generated["last_layer_all_generation_steps"] = cpu_clone(last_layer)
                        generated["goal_embed_first_256_tokens"] = cpu_clone(last_layer[:, :256])
                        generated["latent_after_goal_tokens"] = cpu_clone(last_layer[:, 256:])
                    collector.current["generalist"]["generation"] = generated
                return output

            self._patch_method(self.slow, "generate", generate)

        # Generalist image features.  These modules are present in the local
        # trust_remote_code checkpoint; guard them for clearer portability.
        for name in ("vision_backbone", "projector"):
            module = getattr(self.slow, name, None)
            if module is not None:
                def slow_hook(_module: Any, _inputs: Any, output: Any, hook_name: str = name) -> None:
                    if collector.current is not None and collector.current["generalist"].get("called"):
                        collector.current["generalist"].setdefault("feature_embeddings", {})[hook_name] = cpu_clone(output)

                self._handles.append(module.register_forward_hook(slow_hook))

        # timm DINO is invoked through forward_features() directly, which
        # bypasses ordinary nn.Module forward hooks. Wrap that method exactly.
        vision_encoder = getattr(self.fast, "vision_encoder", None)
        if vision_encoder is not None:
            for method_name in ("forward_features", "forward_feature"):
                if not hasattr(vision_encoder, method_name):
                    continue
                original_feature = getattr(vision_encoder, method_name)

                def feature_method(
                    _owner: Any,
                    *args: Any,
                    _original: Any = original_feature,
                    _name: str = method_name,
                    **kwargs: Any,
                ):
                    output = _original(*args, **kwargs)
                    if collector.current is not None:
                        collector.current["specialist"]["encoder_features"].setdefault(
                            f"vision_encoder.{_name}", []
                        ).append(cpu_clone(output))
                    return output

                self._patch_method(vision_encoder, method_name, feature_method)

        # Other specialist encoders are called through nn.Module.__call__.
        for name in ("depth_encoder", "tactile_encoder"):
            module = getattr(self.fast, name, None)
            if module is not None:
                self._handles.append(module.register_forward_hook(self._capture_list_hook("encoder_features", name)))

        # Adapted conditions repeat during denoising; the first value is enough
        # because the source condition is unchanged within one specialist call.
        for name in (
            "context_adapter",
            "proprio_embedder",
            "visual_adapter",
            "depth_adapter",
            "gripper_visual_adapter",
            "gripper_depth_adapter",
            "tactile_adapter",
        ):
            module = getattr(self.dit, name, None)
            if module is not None:
                self._handles.append(module.register_forward_hook(self._capture_once_hook("adapted_conditions", name)))

        def dit_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if collector.current is None:
                return
            specialist = collector.current["specialist"]
            if "dit_common_inputs" not in specialist:
                specialist["initial_noise"] = cpu_clone(args[0])
                specialist["dit_common_inputs"] = cpu_clone(
                    {
                        "ref_action_cond": kwargs.get("cond"),
                        "generalist_context": kwargs.get("context"),
                        "visual_embedding": kwargs.get("visual_embedding"),
                        "depth_embedding": kwargs.get("depth_embedding"),
                        "gripper_embedding": kwargs.get("gripper_embedding"),
                        "tactile_embedding": kwargs.get("tactile_embedding"),
                        "instruction_passthrough": kwargs.get("lang"),
                        "action_history_condition": kwargs.get("hist_action"),
                        "robot_state_condition": kwargs.get("proprio"),
                    }
                )
            call = {
                "trajectory_in": cpu_clone(args[0]),
                "timestep": cpu_clone(args[1]),
                "cond_mask": cpu_clone(kwargs.get("cond_mask")),
            }
            specialist["dit_calls"].append(call)

        def dit_post_hook(_module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            if collector.current is not None:
                collector.current["specialist"]["dit_calls"][-1]["predicted_noise"] = cpu_clone(output)

        self._handles.append(self.dit.register_forward_pre_hook(dit_pre_hook, with_kwargs=True))
        self._handles.append(self.dit.register_forward_hook(dit_post_hook, with_kwargs=True))

        first_block = self.dit.blocks[0]

        def block_pre_hook(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
            if collector.current is None:
                return
            call = collector.current["specialist"]["dit_calls"][-1]
            call["global_condition_with_timestep"] = cpu_clone(kwargs.get("c"))
            # context_embed is useful but large and invariant within the step;
            # store it once and reference it from every denoising call.
            if "dit_context_embed" not in collector.current["specialist"]["adapted_conditions"]:
                collector.current["specialist"]["adapted_conditions"]["dit_context_embed"] = cpu_clone(
                    kwargs.get("context_embed")
                )

        self._handles.append(first_block.register_forward_pre_hook(block_pre_hook, with_kwargs=True))
