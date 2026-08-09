# Sequence 60 subtask-5 cross experiment

本目录实现从两次已采集轨迹的第四个 subtask 结束边界恢复环境，并对第五个任务 `lift_pink_block_slider` 做 (2\times2) 初始状态 × slow-call 策略交叉实验。

## 实验设计

定义：

- `S8`：original fixed-8 轨迹进入第五项前的状态；
- `S12`：fixed age-12 轨迹进入第五项前的状态；
- `P8`：original fixed-mod-8 策略；
- `P12`：fixed age-12 / age 8–11 empty-reference 策略。

每个 replicate 运行四格：

| 初始状态 | P8 | P12 |
|---|---|---|
| S8 | `S8_P8` | `S8_P12` |
| S12 | `S12_P8` | `S12_P12` |

默认先做 5 个 replicate。四格在同一 replicate 使用同一个 trial seed，并在恢复状态后重新设置 Python、NumPy、Torch CPU 和全部 CUDA RNG。这样四格的 specialist initial noise 按环境步配对；脚本保存每一步实际 initial-noise float32 数值及 SHA256，并轮换四格执行顺序。

主要结果：任务成功、成功步数；辅助连续结果：pink block 最大抬升、最终位移、TCP 到 pink block 最小距离。分析脚本输出：

- 在 S8、S12 内分别计算 `P12-P8`，估计策略效应；
- 在 P8、P12 内分别计算 `S12-S8`，估计状态效应；
- 计算跨初始状态平均的 policy 主效应，以及跨策略平均的 state 主效应；
- difference-in-differences，估计策略和初始状态的交互；
- 每个 paired replicate 的原始差值、均值和标准差。

五次只能作为 pilot。若同一 cell 内部结果有波动，建议扩展到至少 10 次；不要只依据渐近 p-value，应同时报告逐 replicate 配对结果。

## 状态恢复边界

`prepare_boundary_states.py` 从两份 `subtask_04/.../step_0000.pt` 的 `pre_observation/pre_physics` 冻结边界。这一时刻是第四项成功后、第五项第一次策略调用前。

恢复分两层：

1. CALVIN 官方 `env.reset(robot_obs, scene_obs)`；
2. 从 `pre_physics` 补回精确的 arm/gripper 单关节位置和速度、`gripper_action`、三个 movable block 的位姿与线/角速度、door/button/switch joint state 和 scene 逻辑状态；
3. 因为 CALVIN 使用 `use_target_pose=true`，从 sequence 初始 TCP target 和前四项全部 executed action 重建累计的 `Robot.target_pos/target_orn`，并校验 action 数与重建哈希。采集器在 `env.step()` 后保存 action，而 CALVIN 会通过 NumPy view 原地把前六维变成米/弧度物理增量，因此重建时直接累加，禁止再次乘 `max_rel_pos/max_rel_orn`。

每个 cell 开始前都进行 fail-fast 审计。默认容差为 `2e-5`；robot/scene observation、TCP、关节、夹爪和物体任一项超差就停止，不生成结果。

策略内部状态不从第四项恢复。原始 evaluator 在每个 subtask 开始前本来就调用 `reset()`，所以四格全部从第五项新指令、空 action/history/generalist cache 开始。

重要限制：先前数据没有 PyBullet `saveState/saveBullet` checkpoint，因此无法恢复 contact solver warm-start、约束求解缓存等未暴露内部量。当前脚本精确恢复所有已采集且 PyBullet 允许设置的物理量，并在首个新 action 前核验；这是现有数据可实现的最强重建，但不是 Bullet 内部状态的 bitwise checkpoint。若未来要求位级连续，应重新跑前四项并在同一 physics client 内保留原生 checkpoint。

## 文件

- `boundary_states.pt/.json`：冻结的 S8/S12 边界及来源 SHA256；
- `prepare_boundary_states.py`：可重复生成边界 bundle；
- `run_cross_experiment.py`：恢复、2×2 rollout、逐步采集和可选完整 tensor trace；
- `analyze_cross_results.py`：完整性、配对噪声、cell 统计、主效应和交互分析；
- `verify_cross_contract.py`：CPU 状态恢复、schedule、RNG 和执行顺序验证；
- `INDEPENDENT_AUDIT.md`：独立 agent 审查结论。

## 运行前验证

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual

env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/robodual_mpl \
  /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/cross/verify_cross_contract.py'
```

当前 CPU 实测中，两状态所有受审计量均在容差内；重复恢复完全一致。S8 与 S12 的 TCP 起点距离约 2.34 cm，确认交叉实验不是重复同一状态。

模型初始化是完全离线的。specialist checkpoint 已包含 online/EMA 两份 DINO encoder 权重，加载器因此使用 `vision_encoder_pretrained=False` 创建网络，再要求 checkpoint 完整覆盖推理模型；不会访问 Hugging Face。若模型加载在写入 manifest 前失败而只留下空的 `restore_audits/`，可用同一输出目录直接重跑；只要目录中已有任何真实结果，脚本仍会拒绝覆盖。

## 推荐的 5-repeat 启动命令

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual

CUDA_VISIBLE_DEVICES=0 \
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
MPLCONFIGDIR=/tmp/robodual_mpl \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/cross/run_cross_experiment.py' \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --dataset_subdir calvin_debug_dataset \
  --sequence_index 60 \
  --catalog_size 100 \
  --ep_len 240 \
  --seed 42 \
  --replicates 5 \
  --base_trial_seed 42000 \
  --fast_num_inference_steps 10 \
  --load_in_4bit \
  --low_cpu_mem_usage \
  --device_map none \
  --attn_implementation none \
  --output_dir '/home/rosmontis/Projects/dualsys/RoboDual/single _sequence_trial/cross/runs/cross_seq060_r5'
```

输出目录必须不存在或为空。默认保存紧凑但充分的逐步记录，预计几十到数百 MiB。若要对某个 replicate 保存与原实验相同的完整 tensor trace，可增加：

```text
--full_trace_replicates 0
```

完整 trace 预计每个 replicate 的四格额外占用约 8–12 GiB；不建议默认对五次全部开启。

## 分析

```bash
/home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/cross/analyze_cross_results.py' \
  '/home/rosmontis/Projects/dualsys/RoboDual/single _sequence_trial/cross/runs/cross_seq060_r5'
```

run 目录会包含：

```text
manifest.json
cross_events.jsonl
cell_summaries.jsonl
summary.json
restore_audits/
cross_analysis.json
cross_cell_statistics.csv
cross_paired_contrasts.csv
full_traces/                 # 仅在显式要求时存在
```

## 可视化

完成分析后可生成2×2成功矩阵、配对主效应/交互效应、逐步TCP距离与方块抬升轨迹，以及失败阶段分解：

```bash
MPLCONFIGDIR=/tmp/robodual_cross_mpl \
  /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/cross/visualize_cross_results.py' \
  'single _sequence_trial/cross/runs/cross_seq060_r5'
```

默认输出到 `cross_seq060_r5/figures/`，每张图同时保存PNG和可编辑SVG。绘图前会再次检查四格完整性、状态恢复审计和common-random-number审计。失败阶段按方块最大运动5 mm、最大抬升1 cm两个显式阈值分类，阈值和计数写入 `visualization_summary.json`。
