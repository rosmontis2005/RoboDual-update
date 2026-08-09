#!/usr/bin/env python3
"""Static and synthetic checks for schedule identity and trace non-interference."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "vla-scripts"))

import dual_sys_evaluation as legacy
from collect_original_8_steps import OriginalFixedMod8Evaluation
from trace_capture import OnlineTraceCapture, TraceWriter


class FakeVision(torch.nn.Module):
    def forward_features(self, value):
        return value + 2


class FakeBlock(torch.nn.Module):
    def forward(self, x, c, n_batches=1, context_embed=None, attn_mask=None):
        return x + c[:, None, : x.shape[-1]] * 0


class FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context_adapter = torch.nn.Linear(4, 4, bias=False)
        self.proprio_embedder = torch.nn.Linear(2, 4, bias=False)
        self.visual_adapter = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([FakeBlock()])

    def forward(
        self,
        x,
        timesteps,
        cond=None,
        visual_embedding=None,
        context=None,
        proprio=None,
        cond_mask=None,
        **_kwargs,
    ):
        adapted_context = self.context_adapter(context)
        adapted_proprio = self.proprio_embedder(proprio)
        adapted_visual = self.visual_adapter(visual_embedding)
        global_condition = adapted_context.mean(dim=1) + adapted_proprio
        context_embed = torch.cat([adapted_visual, adapted_context], dim=1)
        hidden = self.blocks[0](
            x=x,
            c=global_condition,
            n_batches=x.shape[0],
            context_embed=context_embed,
            attn_mask=None,
        )
        return hidden + cond * 0


class FakeSchedulerOutput:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
        self.pred_original_sample = prev_sample


class FakeScheduler:
    def step(self, model_output, timestep, sample, **_kwargs):
        return FakeSchedulerOutput(sample - model_output * 0.1)


class FakeFast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeDiT()
        self.vision_encoder = FakeVision()
        self.noise_scheduler = FakeScheduler()
        self.num_inference_steps = 1
        self.with_cfg = False
        self.with_gripper = False
        self.with_depth = False

    def conditional_sample(self, condition_data, local_cond, global_cond, proprio, **kwargs):
        trajectory = torch.randn_like(condition_data)
        model_output = self.model(
            trajectory,
            torch.tensor(9),
            cond=local_cond,
            visual_embedding=global_cond[1],
            context=global_cond[0],
            proprio=proprio,
            cond_mask=None,
            **kwargs,
        )
        return self.noise_scheduler.step(model_output, torch.tensor(9), trajectory).prev_sample

    def predict_action(self, ref_action, action_cond, obs, proprio, **kwargs):
        if isinstance(obs, tuple):
            visual_outputs = [self.vision_encoder.forward_features(item) for item in obs]
            visual = visual_outputs[0]
        else:
            visual = self.vision_encoder.forward_features(obs)
        return self.conditional_sample(
            torch.zeros_like(ref_action),
            local_cond=ref_action,
            global_cond=(action_cond, visual),
            proprio=proprio,
            **kwargs,
        )


class FakeGenerateOutput:
    def __init__(self, input_ids):
        self.sequences = input_ids + 1
        first = torch.ones((1, 256, 4))
        second = torch.ones((1, 2, 4)) * 2
        self.hidden_states = ((first,), (second,))


class FakeSlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = torch.nn.Identity()
        self.projector = torch.nn.Identity()

    def generate(self, input_ids, **_kwargs):
        return FakeGenerateOutput(input_ids)

    def predict_action(self, input_ids, **kwargs):
        image = kwargs["pixel_values"]
        _ = self.projector(self.vision_backbone(image))
        generated = self.generate(input_ids=input_ids)
        hidden = torch.cat([step[-1] for step in generated.hidden_states], dim=1)[:, 256:]
        return torch.arange(56, dtype=torch.float32), hidden


class FakeEvaluator:
    def __init__(self, slow, fast):
        self.slow = slow
        self.fast = fast
        self.action = None
        self.hidden_states = None

    def _slow_system(self):
        return self.slow

    def _fast_system(self):
        return self.fast


def run_models(evaluator):
    slow_output = evaluator.slow.predict_action(
        input_ids=torch.tensor([[1, 2]]),
        pixel_values=torch.ones((1, 3, 2, 2)),
    )
    evaluator.action = slow_output[0].reshape(1, 8, 7)
    evaluator.hidden_states = slow_output[1]
    fast_output = evaluator.fast.predict_action(
        ref_action=torch.zeros((1, 2, 2)),
        action_cond=torch.ones((1, 3, 4)),
        obs=(torch.ones((1, 2, 4)), torch.ones((1, 2, 4)) * 2),
        proprio=torch.ones((1, 2)),
    )
    return slow_output, fast_output, torch.random.get_rng_state().clone()


def main() -> None:
    # The collector subclass must not override the historical action path.
    assert OriginalFixedMod8Evaluation.step is legacy.DualSystemCalvinEvaluation.step
    expected_slow_steps = [0] + list(range(7, 240, 8))
    derived_slow_steps = [step for step in range(240) if step == 0 or (step + 1) % 8 == 0]
    assert derived_slow_steps == expected_slow_steps
    expected_cond = [8 if step == 0 else 8 - ((step + 1) % 8) for step in range(16)]
    assert expected_cond == [8, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 8]

    torch.manual_seed(1234)
    control = FakeEvaluator(FakeSlow(), FakeFast())
    control_result = run_models(control)

    torch.manual_seed(1234)
    traced = FakeEvaluator(FakeSlow(), FakeFast())
    with tempfile.TemporaryDirectory(prefix="robodual_trace_verify_") as temp:
        writer = TraceWriter(Path(temp) / "run", {"test": True})
        capture = OnlineTraceCapture(traced, writer)
        capture.begin_step(
            sequence_index=60,
            subtask_index=0,
            subtask="test",
            instruction="test",
            step=0,
            pre_obs={},
            pre_info={},
            pre_physics={},
        )
        traced_result = run_models(traced)
        captured = capture.current
        step_path = capture.finalize_step(
            executed_action=torch.zeros(2),
            post_obs={"robot_obs": torch.zeros(1), "scene_obs": torch.zeros(1)},
            post_info={},
            post_physics={},
            task_success=False,
            profile={"slow_system": True},
        )
        try:
            saved_payload = torch.load(step_path, map_location="cpu", weights_only=False)
        except TypeError:
            saved_payload = torch.load(step_path, map_location="cpu")
        capture.close()

    assert torch.equal(control_result[0][0], traced_result[0][0])
    assert torch.equal(control_result[0][1], traced_result[0][1])
    assert torch.equal(control_result[1], traced_result[1])
    assert torch.equal(control_result[2], traced_result[2])
    assert captured is not None
    assert captured["generalist"]["called"]
    assert "transferred_hidden_states" in captured["generalist"]["output"]
    assert "goal_embed_first_256_tokens" in captured["generalist"]["generation"]
    assert "vision_encoder.forward_features" in captured["specialist"]["encoder_features"]
    assert "initial_noise" in captured["specialist"]
    assert "condition_data" in captured["specialist"]
    assert "dit_common_inputs" in captured["specialist"]
    assert "robot_state_condition" in captured["specialist"]["dit_common_inputs"]
    assert "action_history_condition" in captured["specialist"]["dit_common_inputs"]
    assert len(captured["specialist"]["dit_calls"]) == 1
    assert len(captured["specialist"]["scheduler_steps"]) == 1
    assert "prev_sample" in captured["specialist"]["scheduler_steps"][0]
    assert "trajectory_in" in captured["specialist"]["dit_calls"][0]
    assert "predicted_noise" in captured["specialist"]["dit_calls"][0]
    assert saved_payload["generalist"]["called"]
    assert saved_payload["specialist"]["output_action_chunk"].shape == (1, 2, 2)
    print(
        json.dumps(
            {
                "legacy_step_identity": True,
                "fixed_mod8_schedule_steps_0_239": expected_slow_steps,
                "trace_output_equal": True,
                "trace_rng_state_equal": True,
                "synthetic_capture_complete": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
