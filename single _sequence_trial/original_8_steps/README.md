# Original 8-step online trace collector

本目录用于采集 canonical `sequence_index=60` 在 RoboDual 原始 fixed-mod-8 调度下的完整在线轨迹。

## 一致性边界

动作推理直接继承 `vla-scripts/dual_sys_evaluation.py::DualSystemCalvinEvaluation.step`，采集器没有复制或重写以下逻辑：

- slow call：`step == 0 or (step + 1) % 8 == 0`，即 `0, 7, 15, 23, ...`；
- generalist action reshape 和 `[1, 8, 7]` 裁剪；
- 每个 age 的 reference-action 尾部对齐；
- specialist 输入预处理；
- diffusion sampling 和 scheduler；
- temporal aggregation、夹爪二值化及最终执行动作。

默认使用与历史 0413 profile 一致的 4-bit generalist / FP16 compute、10 个 specialist inference steps、无 CFG、`ep_len=240`。本地可用环境配置来自 `calvin_debug_dataset`。

采集 hook 不设置 RNG、不额外采样。合成验证确认开启采集前后模型输出和 Torch RNG state bitwise 相同。

重要限制：脚本直接从 seq60 开始，因此不会消耗历史 0413 完整评测中 seq0–59 的 Torch/CUDA 随机数。它保持原始算法和全局 RNG 机制，但形成的是一个新的 standalone seed-42 基线，不是历史 seq60 diffusion noise 的 bitwise replay。历史 RNG state 未被保存，除非重新执行前 60 条，否则无法恢复那一段噪声流。当前运行会保存实际 RNG state 和实际 initial noise，之后可以严格复现本次采集。

## 运行

```bash
env CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
  MPLCONFIGDIR=/tmp/robodual_mpl \
  /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/original_8_steps/collect_original_8_steps.py'
```

默认不会覆盖已有 run，而是在 `runs/seq060_seed42_YYYYMMDD_HHMMSS/` 新建目录。

验证采集非干扰性：

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/robodual_mpl \
  /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/original_8_steps/verify_collection_contract.py'
```

查看某一步的 tensor 结构：

```bash
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/original_8_steps/inspect_trace.py' STEP_FILE.pt
```

## Run 目录结构

```text
seq060_seed42_YYYYMMDD_HHMMSS/
├── manifest.json
├── sequence.json
├── events.jsonl
├── summary.json
└── tensors/
    └── seq_060/
        ├── subtask_00_open_drawer/
        │   ├── step_0000.pt
        │   └── ...
        ├── subtask_01_push_blue_block_left/
        ├── subtask_02_move_slider_left/
        ├── subtask_03_turn_on_lightbulb/
        └── subtask_04_lift_pink_block_slider/
```

- `manifest.json`：模型/checkpoint/source SHA256、调度契约、精度、scheduler、seed 和数据说明。
- `sequence.json`：symbolic initial state、实际 `robot_obs/scene_obs`、任务链和语言指令。
- `events.jsonl`：每步文件路径、任务位置、成功标记、字节数、SHA256 和 tensor schema。
- `summary.json`：逐任务结果、实际步数和最终精确磁盘占用。
- `step_XXXX.pt`：该环境步的完整 payload。

## 单步 payload

```text
meta
├── sequence/subtask/step
├── canonical subtask id
├── language instruction
└── task_success

environment
├── pre_observation                 # raw RGB/depth/robot_obs/scene_obs
├── pre_info
├── pre_physics
│   ├── TCP position/quaternion/linear velocity/angular velocity
│   ├── 7 arm joints: position/velocity/reaction/torque
│   ├── 2 gripper joints: position/velocity/reaction/torque
│   ├── contacts
│   └── named scene/object state
├── executed_action
├── post_robot_obs/post_scene_obs
├── post_info
└── post_physics

generalist
├── called
├── rng_pre/rng_post
├── inputs                          # input_ids, attention mask, pixels...
├── generation
│   ├── sequences                   # generated token ids
│   ├── last_layer_all_generation_steps
│   ├── goal_embed_first_256_tokens
│   └── latent_after_goal_tokens
├── feature_embeddings
│   ├── vision_backbone
│   └── projector
├── output
│   ├── action_chunk_pre_reshape
│   └── transferred_hidden_states
├── cached_action_chunk_8x7
└── cached_generalist_condition

specialist
├── rng_pre/rng_post
├── inputs
│   ├── ref_action
│   ├── action_cond/generalist_condition
│   ├── current/previous RGB tensors
│   ├── static/gripper depth
│   ├── gripper RGB
│   ├── instruction passthrough
│   ├── robot_state_condition
│   └── action_history_condition
├── encoder_features                # DINO/depth/gripper features
├── adapted_conditions              # context/proprio/visual adapters
├── condition_data                  # conditional_sample 的实际全零输入
├── initial_noise
├── dit_common_inputs
├── dit_calls[]
│   ├── trajectory_in
│   ├── timestep
│   ├── cond_mask
│   ├── global_condition_with_timestep
│   └── predicted_noise
├── scheduler_steps[]
│   ├── model_output/timestep/sample_in
│   ├── prev_sample
│   └── pred_original_sample
└── output_action_chunk             # 完整 [1,8,7]

evaluator_profile                   # 原始 profile 摘要和最终 aggregation 指标
```

当前模型不存在独立的 `subtask_token` tensor；因此保存 canonical subtask id、原始 instruction、processor `input_ids` 和 generalist generated token ids。基础 DiT 的 `lang` 和 `hist_action` 虽被传入但不一定实际消费，数据中仍保存二者，避免将“传入”误报为“参与计算”。

## 容量估算

历史 fixed-mod-8 seq60 共约 380 个环境步、约 50 次 slow call。按当前无压缩 FP32/FP16 tensor、两帧 224×224 specialist RGB、DINO/depth features、逐步 generalist condition 和完整 pre/post simulator state 估算：

- 预期完整 seq60：约 **2–4 GiB**；
- 若轨迹明显变长：约每 100 个环境步增加 **0.6–1.0 GiB**；
- `ep_len=240`、五个任务均接近上限的保守极端情况：约 **7–12 GiB**。

`summary.json` 会在采集结束后给出实际精确字节数。估算没有保存 generalist 的每一层、每一个 autoregressive step 的全部 layer hidden state；保存的是实际传给 specialist 的 hidden tensor、最后层生成轨迹、goal/latent、vision/projector features。若扩展为所有 generalist 层，容量可能上升到数十至上百 GiB。
