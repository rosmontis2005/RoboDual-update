from collections import deque
import logging
import os
import time

import numpy as np
from PIL import Image
import torch
from einops import rearrange

from calvin_agent.models.calvin_base_model import CalvinBaseModel

logger = logging.getLogger(__name__)


def get_openvla_prompt(instruction: str, tokenized_action: str = None) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class DualSystemCalvinEvaluation(CalvinBaseModel):
    def __init__(
        self,
        model,
        processor,
        action_tokenizer,
        profile_steps=False,
        profile_sample_var_k=3,
        profile_sample_var_interval=8,
    ):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = processor
        self.dual_sys = model
        self.action_tokenizer = action_tokenizer
        self.profile_steps = profile_steps
        self.profile_sample_var_k = max(1, int(profile_sample_var_k))
        self.profile_sample_var_interval = max(1, int(profile_sample_var_interval))

        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        self.action = None
        self.hidden_states = None
        self.obs_buffer = None

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]

        # Dataset statics (rougnly computed with 10k samples in CALVIN)
        self.depth_max = 6.2
        self.depth_min = 3.5
        self.gripper_depth_max = 2.0
        self.gripper_depth_min = 0

        self.hist_action = deque(maxlen=4)
        self.gripper_window = deque(maxlen=8)
        self.last_slow_step = None
        self.prev_action = None
        self.prev_prev_action = None
        self.prev_proprio = None
        self.prev_obs_tensor = None
        self.last_step_profile = {}
        self._fast_device = None

    def _slow_system(self):
        return self.dual_sys.module.slow_system if hasattr(self.dual_sys, "module") else self.dual_sys.slow_system

    def _dual_system(self):
        return self.dual_sys.module if hasattr(self.dual_sys, "module") else self.dual_sys

    def _runtime_device(self):
        slow_system = self._slow_system()
        if hasattr(slow_system, "device"):
            return slow_system.device
        return next(slow_system.parameters()).device

    def _runtime_dtype(self):
        slow_system = self._slow_system()
        if hasattr(slow_system, "dtype") and slow_system.dtype is not None:
            return slow_system.dtype
        return next(slow_system.parameters()).dtype

    def _fast_system(self):
        return self._dual_system().ema_fast_system.ema_model

    def _ensure_fast_system_device(self, runtime_device):
        if self._fast_device == runtime_device:
            return
        fast_system = self._fast_system()
        fast_system.to(runtime_device)
        fast_system.eval()
        self._fast_device = runtime_device
        logger.info("Moved fast system to %s for evaluation", runtime_device)

        
    def reset(self,):
        """
        This is called
        """

        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.obs_buffer = None
        self.hist_action.clear()
        self.gripper_window.clear()
        self.last_slow_step = None
        self.prev_action = None
        self.prev_prev_action = None
        self.prev_proprio = None
        self.prev_obs_tensor = None
        self.last_step_profile = {}


    @staticmethod
    def _round_float(value, digits=6):
        if value is None:
            return None
        return round(float(value), digits)

    @classmethod
    def _tensor_list(cls, tensor, digits=6):
        return [cls._round_float(value, digits) for value in tensor.detach().to(torch.float32).cpu().flatten().tolist()]

    @classmethod
    def _array_list(cls, array, digits=6):
        return [cls._round_float(value, digits) for value in np.asarray(array).reshape(-1).tolist()]

    @staticmethod
    def _rms_tensor(tensor):
        return torch.sqrt(torch.mean(torch.square(tensor.to(torch.float32)))).item()

    @staticmethod
    def _rms_array(array):
        array = np.asarray(array, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(array))))

    def _sample_var_due(self, step):
        return (
            self.profile_steps
            and self.profile_sample_var_k > 1
            and step % self.profile_sample_var_interval == 0
        )


    def step(self, obs, instruction, step):
        """
        Args:
            obs: environment observations
            instruction: embedded language goal
        Returns:
            action: predicted action
        """
        profile = {
            "step": int(step),
            "slow_system": False,
            "step_since_slow_before": None if self.last_slow_step is None else int(step - self.last_slow_step),
        }
        total_start = time.perf_counter()
        live_profile = os.environ.get("ROBODUAL_PROFILE_LIVE") == "1"

        with torch.inference_mode():
            preprocess_start = time.perf_counter()
            image = obs["rgb_obs"]['rgb_static']
            gripper_image = obs["rgb_obs"]['rgb_gripper']
            runtime_device = self._runtime_device()
            runtime_dtype = self._runtime_dtype()
            self._ensure_fast_system_device(runtime_device)
            gripper_image = self.processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3].unsqueeze(0).to(runtime_device)

            tactile_image = None
            # tactile_image = torch.from_numpy(obs["rgb_obs"]['rgb_tactile']).permute(2,0,1).unsqueeze(0).to(runtime_device, dtype=torch.float) / 255
            depth_image = (torch.as_tensor(obs["depth_obs"]['depth_static'], device=runtime_device).unsqueeze(0) - self.depth_min) / (self.depth_max - self.depth_min)
            depth_gripper = (torch.as_tensor(obs["depth_obs"]['depth_gripper'], device=runtime_device).unsqueeze(0) - self.gripper_depth_min) / (self.gripper_depth_max - self.gripper_depth_min)

            prompt = get_openvla_prompt(instruction)
            inputs = self.processor(prompt, Image.fromarray(image)).to(runtime_device, dtype=runtime_dtype)
            profile["preprocess_s"] = round(time.perf_counter() - preprocess_start, 4)
            profile["runtime_dtype"] = str(runtime_dtype)
            current_obs_tensor = inputs["pixel_values"][:, :3].detach().to(torch.float32).cpu()
            profile["obs_delta"] = (
                None
                if self.prev_obs_tensor is None
                else self._round_float(self._rms_tensor(current_obs_tensor - self.prev_obs_tensor))
            )
            if live_profile:
                print(f"[live-profile] step={step} preprocess_done {profile}", flush=True)

            if (step + 1) % 8 == 0 or step == 0:
                slow_start = time.perf_counter()
                action, hidden_states = self._slow_system().predict_action(**inputs, do_sample=False)
                action = torch.as_tensor(action, device=hidden_states.device).unsqueeze(0)
                action = rearrange(action, 'b (f d) -> b f d', f=8)
                self.action = action[:, :, :7]
                self.hidden_states = hidden_states
                self.last_slow_step = step
                profile["slow_system"] = True
                profile["slow_system_s"] = round(time.perf_counter() - slow_start, 4)
                if live_profile:
                    print(f"[live-profile] step={step} slow_done {profile}", flush=True)
            profile["step_since_slow"] = None if self.last_slow_step is None else int(step - self.last_slow_step)

            num_cond_actions = 8 - (step + 1) % 8
            if step == 0:
                num_cond_actions = 8

            zero_actions = torch.zeros((1, self.temporal_size, 7), device=self.action.device)
            zero_actions[:, :num_cond_actions] = self.action[:, -num_cond_actions:]
            ref_actions = zero_actions
            profile["num_cond_actions"] = int(num_cond_actions)
            profile["ref_action_first"] = self._tensor_list(ref_actions[0, 0])
            profile["ref_action_valid_part"] = [
                self._tensor_list(ref_actions[0, idx]) for idx in range(int(num_cond_actions))
            ]

            state = torch.as_tensor(obs['robot_obs'], device=runtime_device, dtype=torch.float32)
            state = torch.cat([state[:6], state[[-1]]], dim=-1).unsqueeze(0)
            state_cpu = state.detach().to(torch.float32).cpu()
            profile["proprio_delta"] = (
                None
                if self.prev_proprio is None
                else self._round_float(self._rms_tensor(state_cpu - self.prev_proprio))
            )

            if step == 0:
                self.obs_buffer = image

            prev_img = self.processor.image_processor.apply_transform(Image.fromarray(self.obs_buffer))[:3].unsqueeze(0).to(runtime_device)
            obs = (inputs["pixel_values"][:, :3].to(torch.float32), prev_img)

            hist_action = torch.zeros((1, 4, 7), device=runtime_device)
            if self.hist_action:
                hist_stack = torch.stack(list(self.hist_action), dim=0).unsqueeze(0).to(runtime_device)
                hist_action[:, -hist_stack.shape[1]:] = hist_stack

            fast_start = time.perf_counter()
            fast_system = self._fast_system()
            ref_actions = ref_actions.to(torch.float32)
            action_cond = self.hidden_states.to(torch.float32)
            predict_kwargs = dict(
                ref_action=ref_actions,
                action_cond=action_cond,
                obs=obs,
                depth_obs=depth_image,
                gripper_obs=(gripper_image, depth_gripper),
                tactile_obs=tactile_image,
                lang=instruction,
                proprio=state,
                hist_action=hist_action,
            )
            dp_action = fast_system.predict_action(**predict_kwargs)
            sample_var_start = time.perf_counter()
            if self._sample_var_due(step):
                sample_actions = [dp_action.detach().to(torch.float32)]
                cpu_rng_state = torch.random.get_rng_state()
                cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                try:
                    for _ in range(self.profile_sample_var_k - 1):
                        sample_actions.append(fast_system.predict_action(**predict_kwargs).detach().to(torch.float32))
                finally:
                    torch.random.set_rng_state(cpu_rng_state)
                    if cuda_rng_states is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_states)
                sample_actions = torch.stack(sample_actions, dim=0)
                sample_var = torch.var(sample_actions, dim=0, unbiased=False)
                profile["sample_k"] = int(self.profile_sample_var_k)
                profile["sample_var"] = self._round_float(sample_var.mean().item())
                profile["sample_var_ee6"] = self._round_float(sample_var[..., :6].mean().item())
                profile["sample_var_gripper"] = self._round_float(sample_var[..., -1].mean().item())
                profile["sample_var_first"] = self._tensor_list(sample_var[0, 0])
                profile["sample_var_s"] = round(time.perf_counter() - sample_var_start, 4)
            else:
                profile["sample_k"] = 1
                profile["sample_var"] = None
                profile["sample_var_ee6"] = None
                profile["sample_var_gripper"] = None
                profile["sample_var_first"] = None
                profile["sample_var_s"] = 0.0
            profile["dp_action_first"] = self._tensor_list(dp_action[0, 0])
            dp_action_l2 = torch.linalg.vector_norm(dp_action.detach().to(torch.float32)[0], ord=2, dim=-1)
            profile["dp_action_chunk_mean"] = self._tensor_list(dp_action[0].mean(dim=0))
            profile["dp_action_chunk_l2_mean"] = self._round_float(dp_action_l2.mean().item())
            profile["dp_action_chunk_l2_max"] = self._round_float(dp_action_l2.max().item())
            valid_dp = dp_action[:, :num_cond_actions].detach().to(torch.float32)
            valid_ref = ref_actions[:, :num_cond_actions].detach().to(torch.float32)
            dp_ref_delta = valid_dp - valid_ref
            profile["dp_ref_l2"] = self._round_float(self._rms_tensor(dp_ref_delta))
            profile["dp_ref_l2_ee6"] = self._round_float(self._rms_tensor(dp_ref_delta[..., :6]))
            profile["dp_ref_l2_gripper"] = self._round_float(self._rms_tensor(dp_ref_delta[..., -1]))
            profile["fast_system_s"] = round(time.perf_counter() - fast_start, 4)
            if live_profile:
                print(f"[live-profile] step={step} fast_done {profile}", flush=True)
        self.obs_buffer = image
        self.prev_obs_tensor = current_obs_tensor
        self.prev_proprio = state_cpu
        to_numpy_start = time.perf_counter()
        action = dp_action.detach().to(torch.float32).cpu().numpy()
        profile["to_numpy_s"] = round(time.perf_counter() - to_numpy_start, 4)


        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)
        raw_action_prediction = action_prediction.copy()


        if action_prediction[-1] < -0.5:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        aggregation_delta = action_prediction - action[0, 0]
        raw_aggregation_delta = raw_action_prediction - action[0, 0]
        profile["action_prediction"] = self._array_list(action_prediction)
        profile["raw_action_prediction"] = self._array_list(raw_action_prediction)
        profile["aggregation_delta"] = self._round_float(self._rms_array(aggregation_delta))
        profile["aggregation_delta_ee6"] = self._round_float(self._rms_array(aggregation_delta[:6]))
        profile["raw_aggregation_delta"] = self._round_float(self._rms_array(raw_aggregation_delta))
        profile["raw_aggregation_delta_ee6"] = self._round_float(self._rms_array(raw_aggregation_delta[:6]))

        if self.prev_action is not None and self.prev_prev_action is not None:
            jerk = action_prediction[:6] - 2 * self.prev_action[:6] + self.prev_prev_action[:6]
            profile["jerk_l2_ee6"] = self._round_float(float(np.linalg.norm(jerk)))
            profile["gripper_jerk"] = self._round_float(
                action_prediction[-1] - 2 * self.prev_action[-1] + self.prev_prev_action[-1]
            )
        else:
            profile["jerk_l2_ee6"] = None
            profile["gripper_jerk"] = None

        self.gripper_window.append(float(np.sign(action_prediction[-1])))
        profile["gripper_flip_count"] = int(
            sum(
                1
                for prev, curr in zip(list(self.gripper_window)[:-1], list(self.gripper_window)[1:])
                if prev != curr
            )
        )

        self.prev_prev_action = None if self.prev_action is None else self.prev_action.copy()
        self.prev_action = action_prediction.copy()

        self.hist_action.append(torch.from_numpy(action_prediction).to(torch.float32))
        profile["total_s"] = round(time.perf_counter() - total_start, 4)
        self.last_step_profile = profile
        if live_profile:
            print(f"[live-profile] step={step} total_done {profile}", flush=True)

        return action_prediction
