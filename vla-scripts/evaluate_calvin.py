# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from dual_sys_evaluation import DualSystemCalvinEvaluation

from ema_pytorch import EMA
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ['CALVIN_ROOT']


from collections import Counter
import json
import numpy as np


def print_and_save(results, sequences, eval_result_path, task_name=None, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    # model_name = 'vla-test'
    if not os.path.isdir(f'./{task_name}'):
        os.mkdir( f'./{task_name}')
    with open(f'./{task_name}/split_{torch.cuda.current_device()}.json', "w") as file:
        json.dump(chain_sr, file)

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )



def make_env(dataset_path, observation_space, device, use_egl):
    val_folder = Path(dataset_path) / "validation"
    from calvin_env_wrapper import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device, use_egl=use_egl)
    return env


def evaluate_policy(
    model,
    env,
    eval_sr_path,
    eval_result_path,
    num_procs,
    procs_id,
    eval_dir,
    ep_len,
    num_sequences,
    task_name='test',
    enrich_lang=False,
    debug=False,
    max_subtasks=None,
    profile_steps=False,
):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    if enrich_lang:
        with open('vla-scripts/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)
    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]
    eval_sequences_for_report = list(eval_sequences)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_dir,
            sequence_i,
            ep_len,
            max_subtasks=max_subtasks,
            profile_steps=profile_steps,
        )
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences_for_report, eval_result_path, task_name, None)
    return results


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug,
    eval_dir,
    sequence_i,
    ep_len,
    max_subtasks=None,
    profile_steps=False,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if max_subtasks is not None:
        eval_sequence = eval_sequence[:max_subtasks]
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(
            env,
            model,
            task_checker,
            subtask,
            val_annotations,
            debug,
            eval_dir,
            subtask_i,
            sequence_i,
            ep_len,
            profile_steps=profile_steps,
        )
        if success:
            # print('success: ', subtask_i)
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len, profile_steps=False):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if profile_steps:
        print(
            f"[profile] sequence={sequence_i} subtask={subtask_i} name={subtask} ep_len={ep_len}",
            flush=True,
        )
    if debug:
        img_dict = {
            'static': [],
            'gripper': [],
        }

    for step in range(ep_len):
        model_start = time.perf_counter()
        action = model.step(obs, lang_annotation, step)
        model_step_s = time.perf_counter() - model_start
        env_start = time.perf_counter()
        obs, _, _, current_info = env.step(action)
        env_step_s = time.perf_counter() - env_start

        if debug:
            img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
            img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        # check if current step solves a task
        oracle_start = time.perf_counter()
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        oracle_step_s = time.perf_counter() - oracle_start
        if profile_steps:
            step_profile = getattr(model, "last_step_profile", {})
            print(
                "[profile] "
                f"sequence={sequence_i} subtask={subtask_i} step={step} "
                f"model_s={model_step_s:.4f} env_s={env_step_s:.4f} oracle_s={oracle_step_s:.4f} "
                f"details={json.dumps(step_profile, sort_keys=True)}",
                flush=True,
            )
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                for key in img_dict.keys():
                    clip = ImageSequenceClip(img_dict[key], fps=30)
                    clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.gif'), fps=30)
            return True

    if debug:
        print(colored("fail", "red"), end=" ")
        for key in img_dict.keys():
            clip = ImageSequenceClip(img_dict[key], fps=30)
            clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-fail.gif'), fps=30)
    return False


def main(args):
    # Set seed #42
    seed_everything(42)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device


    # Load generalist policy
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    quantization_config = None
    model_dtype = torch.bfloat16
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model_dtype = torch.float16
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_dtype = torch.float16
    processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
    model_kwargs = dict(
        torch_dtype=model_dtype,
        quantization_config=quantization_config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=True,
    )
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForVision2Seq.from_pretrained(args.generalist_path, **model_kwargs)
    model.eval()

    # Load specialist policy
    from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers import DPMSolverMultistepScheduler

    scheduler = DDIMScheduler( num_train_timesteps = 100, beta_schedule = 'squaredcos_cap_v2', prediction_type="epsilon" )
    shape_meta = {'action' : {'shape': [7]}}
    diffusion_policy = DiffusionDiTImagePolicy( shape_meta = shape_meta,
                                                noise_scheduler = scheduler,
                                                n_action_steps=8, 
                                                num_inference_steps=args.fast_num_inference_steps,
                                                vision_encoder='DINO',
                                                with_depth=args.with_depth,
                                                progressive_noise=False,
                                                with_gripper=args.with_gripper,
                                                with_tactile=args.with_tactile,
                                                cond_drop_chance=0.1 if args.with_cfg else 0.,  
                                                # set cond_drop_chance > 0 to activate CFG
                                              ).eval().to(device)
   

    from prismatic.vla.action_tokenizer import ActionTokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    from train_spacialist_calvin import DualSystem
    dual_sys = DualSystem(model, diffusion_policy, action_tokenizer)
    dual_sys.ema_fast_system.load_state_dict(torch.load(args.specialist_path), strict=False)

    dual_sys = acc.prepare(dual_sys, device_placement=[True])

    save_path = Path('../evaluation_results')
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper', ],  # rgb_tactile
        'depth_obs': ['depth_static', 'depth_gripper'], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}
    eval_dir = save_path / f'eval{torch.cuda.current_device()}'
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env(os.path.join(CALVIN_ROOT, f"dataset/{args.dataset_subdir}"), observation_space, device, args.use_egl)
    eva = DualSystemCalvinEvaluation(dual_sys, processor, action_tokenizer)
    dual_sys.eval()
    avg_reward = torch.tensor(
        evaluate_policy(
            eva,
            env,
            save_path / 'success_rate.txt',
            save_path / 'result.txt',
            acc.num_processes,
            acc.process_index,
            eval_dir=eval_dir,
            ep_len=args.ep_len,
            num_sequences=args.num_sequences,
            task_name=args.log_dir,
            enrich_lang=args.enrich_lang,
            debug=args.debug,
            max_subtasks=args.max_subtasks,
            profile_steps=args.profile_steps,
        )
    ).float().mean().to(device)

    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean() 
    if acc.is_main_process:
        print('average success rate ', avg_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generalist_path", default="openvla7b", type=str)
    parser.add_argument("--specialist_path", default="specialist_policy.pt", type=str)
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
    args = parser.parse_args()

    main(args)
