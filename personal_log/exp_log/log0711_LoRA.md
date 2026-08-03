# 0711 Fast Specialist LoRA / Legato-like Continuation 方案说明

## 1. 先前一次 LoRA 效果不理想的可能原因

### 1.1 先前实验实际完成的事情

先前 LoRA 代码和数据位于：

```text
RoboDual/LoRA_trial
```

数据采集入口：

```text
RoboDual/LoRA_trial/collect_lora_rollouts.py
```

训练入口：

```text
RoboDual/LoRA_trial/train_lora_specialist.py
```

训练结果：

```text
RoboDual/LoRA_trial/lora_runs/specialist_empty_ref_lora_v1
```

该实验从 task-age policy 的成功 rollout 中采集五类困难任务：

```text
place_in_slider
lift_blue_block_slider
stack_block
rotate_red_block_right
push_pink_block_right
```

最终保存 33 条成功 rollout、2996 帧，构造出 528 个训练样本。训练时使用 stale age 8、9、10、11，此时 `ref_action` 已经过期并被置空；generalist hidden 则来自较早的 `slow_idx`。

因此先前训练的实际目标是：

```text
当前 observation
+ stale old hidden
+ empty ref_action
-> 成功 rollout 中当前时刻之后的 8 步动作
```

它主要训练 specialist 在 stale / empty guidance 下继续完成困难任务，而不是训练 slow guidance 刷新时的轨迹 continuation。

### 1.2 评测结果中的副作用

targeted evaluation 的结果为：

| 模型 | target SR（all cases） | target SR（attempted） | prefix failed |
|---|---:|---:|---:|
| base specialist | 45.0% | 56.25% | 8 |
| stale-ref LoRA | 40.0% | 66.67% | 16 |

LoRA 在真正到达目标任务后表现有所改善，但到达目标任务之前的失败数从 8 增加到 16。也就是说，它提高了局部 attempted 指标，却破坏了前序任务和整体策略能力。

这个结果不能证明 fast specialist LoRA 无效，但说明先前的训练分布、参数范围和评测口径都存在问题。

### 1.3 数据规模小且任务分布过窄

先前只有：

```text
33 rollouts
528 samples
5 target tasks
```

但 LoRA 被注入 86 个线性模块，可训练参数约 307312。数据覆盖范围远小于被修改的模型能力范围，很容易出现：

- 记住少量困难任务轨迹。
- targeted task attempted SR 提高。
- 简单任务、前序任务和未训练任务退化。
- 训练 loss 降低但 full benchmark 成功率下降。

### 1.4 只保留成功困难任务造成选择偏差

成功 rollout 可以作为动作监督目标，但先前数据几乎全部来自少量困难任务的成功案例，缺少原 specialist 的正常分布 replay。LoRA 因而没有约束去保持：

- 简单任务能力。
- 普通 full-ref 阶段能力。
- 不同任务组的动作分布。
- 前序任务和长链任务能力。

这与 targeted evaluation 中 prefix failure 增加的现象一致。

### 1.5 训练条件与需要解决的问题不一致

当前需要解决的是：

```text
fast specialist 已经执行了一段旧轨迹
-> new slow hidden/ref 突然接入
-> specialist 应当在接受新指导的同时保持动作连续和任务进展
```

先前数据模拟的是：

```text
old hidden 持续 stale
+ ref 过期为空
-> specialist 独立撑过 age 8-11
```

它没有给模型提供 `old committed history + new condition` 的组合，因此没有直接训练 condition-switch continuation。

### 1.6 `hist_action` 当前实际上没有进入模型

虽然评测 wrapper 和 LoRA trainer 都传入了 `hist_action`，但 `DiffusionPolicy` 构造 DiT 时固定使用：

```python
with_hist_action_num = 0
```

对应代码：

```text
RoboDual/prismatic/models/policy/diffusion_policy.py
```

`DiffusionTransformer` 只有在 `with_hist_action_num > 0` 时才会执行：

```python
self.hist_act_embed(hist_action)
```

所以现有 specialist 实际看不到已经执行的动作 prefix。即使数据集返回了 `hist_action`，现有纯 LoRA 也无法学习基于历史动作的 continuation。

### 1.7 LoRA 插入范围过大

先前 LoRA 同时修改了：

- action/ref embedding。
- generalist context adapter。
- proprio embedding。
- visual/depth/gripper adapters。
- 6 层 temporal attention。
- 6 层 MLP。
- cross attention。
- final action head。

这远超过“适配 stale/transition condition”所需的范围。尤其视觉、深度、proprio 和 final head 并不是新旧 slow condition 接入问题的主要位置，修改它们会增加 catastrophic forgetting 风险。

### 1.8 其它需要补充控制的风险

- 如果训练数据来自评测序列，会产生 train-test leakage；正式训练应优先使用 CALVIN training split。
- 训练窗口必须按 trajectory 划分 train/validation/test，不能让同一 rollout 的相邻帧跨 split。
- new generalist condition 可能与示范 target 属于不同动作模式，需要过滤明显语义冲突的 counterfactual 样本。
- 需要确认 LoRA merged checkpoint 的 EMA 权重确实被评测入口加载。
- diffusion policy 使用随机采样，正式对照应为每条 sequence 设置独立固定 seed，避免前序 rollout 长度改变后续随机数流。

## 2. 近似 Legato 所需要的要件

### 2.1 目标定义

当前条件下无法完整复现 Legato 或 training-time RTC，但可以实现一个 Legato-inspired fast specialist adaptation：

> 让 fast specialist 学会在新 slow guidance 接入时，根据当前 observation、robot state 和已经执行的动作历史，生成与成功轨迹一致的后续 action chunk。

目标不是强制降低所有动作的 delta/jerk，而是：

```text
保留正常动作响应速度
+ 减少 condition switch 造成的无意义轨迹突变
+ 保持成功任务的 future continuation
```

### 2.2 必须存在 committed action history 输入

Legato-like continuation 的核心条件之一是模型知道已经执行了什么。至少需要输入：

```text
hist_action = a[t-4:t]
```

由于当前 `with_hist_action_num=0`，必须在 fast specialist 中建立一个真实的 history 通路。最低成本方案是新增小型 history adapter：

```text
hist_action [B,4,7]
-> flatten / temporal encode
-> history feature
-> residual add into global condition
```

建议使用零初始化输出层或零门控：

```python
global_cond = global_cond + history_gate * history_adapter(hist_action)
```

初始化时 `history_gate=0`，使新增结构在训练前严格接近原 specialist 行为，降低直接破坏 checkpoint 的风险。

如果严格限制为完全不增加 history adapter、只对现有层做 LoRA，则只能训练 transition-conditioned robustness，不能称为真正的 prefix-conditioned continuation。

### 2.3 必须使用真实的新 slow condition

每个 refresh transition 样本应在当前时刻 `t` 的 observation 上重新运行 generalist，得到：

```text
new_ref[t]
new_hidden[t]
```

不能继续只使用 `t-age` 时刻生成的 stale old hidden。训练输入应复现部署时的 hard refresh：

```text
old committed history
+ current observation / proprio
+ current new ref
+ current new hidden
```

### 2.4 必须有成功 continuation target

监督目标应为同一成功轨迹在当前时刻之后的真实动作：

```text
target_action = a[t:t+8]
```

不应使用手工线性插值产生 target，也不应把失败 rollout 中实际执行的失败动作作为监督标签。

如果任务在该时刻确实需要快速反向、抓取或纠偏，成功 target 应保留这种行为。此前实验已经证明，动作数值更平滑不等于成功率更高。

### 2.5 必须保留正常分布 replay

为了防止再次出现 prefix failure 增加，训练集不能全部由 transition 或困难任务组成。建议数据比例：

| 样本类型 | 比例 | 目的 |
|---|---:|---|
| 正常原分布 replay | 50% | 保留 base specialist 能力 |
| hard-refresh transition | 30% | 学习 old history 到 new condition 的接入 |
| 高冲突但成功 transition | 10% | 针对真正危险的边界 |
| stale / empty-ref | 10% | 保留 age 8-13 鲁棒性 |

### 2.6 必须覆盖广泛任务并严格划分数据

正式数据应覆盖 ABCD 全部任务组，而不是只覆盖五个 targeted task。建议最低规模：

```text
100-200 条独立成功 trajectory
5000-10000 个训练窗口
```

按完整 trajectory 划分：

```text
train / validation / test = 70% / 15% / 15%
```

同一 trajectory 的不同窗口不得跨 split。

### 2.7 训练目标以成功动作学习为主

第一版继续使用原 diffusion action loss：

```text
L = diffusion_loss(predicted_chunk, successful_target_chunk)
```

暂时不要增加：

- 全局 jerk penalty。
- 全局 delta penalty。
- old/new action interpolation loss。
- hidden linear consistency loss。

第二阶段可以为同一成功 target 构造 stale condition 和 refreshed condition 两个视图，使用相同 diffusion noise/timestep，让两种条件都恢复到同一成功 future target。但这仍然是 target consistency，不是 hidden interpolation。

## 3. 如何构造训练和插入层

### 3.1 Transition 样本构造

从 CALVIN training split 的成功 trajectory 中选择候选刷新点 `t`。每个样本保存：

```text
task
task_group
instruction
refresh_age
obs[t-1]
obs[t]
proprio[t]
hist_action = a[t-4:t]
new_ref = generalist(obs[t], instruction).action
new_hidden = generalist(obs[t], instruction).hidden
target_action = a[t:t+8]
```

refresh age 应匹配当前 task-age 部署分布：

```text
group A: 13
group B: 12
group C: 10
group D: 8
```

也可以在对应 age 附近加入少量 `age-1`、`age+1` augmentation，但不建议无条件覆盖大量与部署无关的 age。

### 3.2 高冲突成功样本筛选

在候选刷新点离线计算：

```text
new_ref_first_vs_prev_action_l2_ee6
old_new_ref_first_l2_ee6
base specialist refresh delta
base specialist refresh jerk
gripper intent change
```

优先保留 new condition 接入后存在较大动作冲突、但示范 future target 最终成功的样本。它们才是需要模型学习的困难 continuation，而不是任意困难任务帧。

对于 new generalist ref 与成功 target 明显语义冲突的样本，应过滤。例如 new ref 要求打开 gripper，而 target 正在稳定抓取时，强制拟合可能让模型忽略 generalist condition。

### 3.3 Normal replay 样本构造

从同一 training split 中随机采样普通窗口，按照原 specialist 的标准 condition 方式构造：

```text
正常 ref_action
正常 generalist hidden
正常 observation / proprio
成功 target action chunk
```

normal replay 应覆盖简单任务、困难任务和不同任务阶段。其作用不是涨点，而是防止 LoRA 为 transition 样本牺牲原有能力。

### 3.4 Stale / empty-ref 样本构造

可以保留先前 trainer 的一小部分逻辑：

```text
stale ages = 8,9,10,11,12,13
old hidden
empty ref_action
successful future target
```

但占比应控制在约 10%，且 age 按不同任务组的实际调度频率采样，避免再次让模型主要学习 empty-ref targeted behavior。

### 3.5 History adapter 插入位置

建议新增：

```text
model.history_adapter
```

输入为最近 4 步、每步 7 维动作。第一版可以使用：

```text
Flatten(4x7)
-> Linear(28, hidden_size)
-> SiLU
-> Linear(hidden_size, hidden_size), zero-init output
```

输出加入 `global_cond`，使所有 DiT block 都能访问历史信息。只修改 fast specialist，不训练 generalist、vision encoder 或 depth encoder。

### 3.6 第一版 LoRA 插入层

建议只在条件融合和时间建模路径插入 LoRA：

| 模块 | 训练方式 | 作用 |
|---|---|---|
| `model.history_adapter` | 完整训练 | 建立 committed action history 通路 |
| `model.x_embedder` | LoRA rank 4 | 适配 noisy action 与 new ref 的融合 |
| `model.context_adapter` | LoRA rank 4 | 适配 new generalist hidden |
| `blocks.*.attn_temporal.qkv` | LoRA rank 2-4 | 学习历史与未来 action token 的关系 |
| `blocks.*.attn_temporal.proj` | LoRA rank 2-4 | 调整 temporal feature 输出 |

暂时冻结：

```text
visual_adapter
depth_adapter
gripper visual/depth adapter
proprio encoder
block MLP
cross attention
final action head
generalist
```

如果第一版 transition validation 明显欠拟合，再逐步加入后 2-3 层 cross-attention projection，不应一次恢复先前 86 个 LoRA target。

### 3.7 推荐训练参数

第一轮建议：

```text
LoRA rank = 4
LoRA alpha = 8
LoRA dropout = 0.05
learning rate = 1e-5 或 3e-5
weight decay = 0
batch size = 2
gradient clipping = 1.0
```

使用 validation early stopping，不把固定 300 steps 或持续下降的 training loss 当作成功标准。分别记录：

```text
normal validation loss
transition validation loss
stale validation loss
history adapter output norm
history gate
LoRA parameter norm
```

### 3.8 最低必要消融

| 实验 | 目的 |
|---|---|
| base specialist | 全局基线 |
| 先前 stale-ref LoRA | 复现旧方法副作用 |
| transition 数据 + LoRA，无 history adapter | 判断新数据本身的作用 |
| transition 数据 + history adapter + LoRA | 完整 Legato-like 近似 |

如果算力有限，先在 held-out transition 数据和少量 CALVIN 序列上筛选 checkpoint，再将唯一候选模型跑完整 100 sequence benchmark。

正式评测必须同时报告：

```text
avg_seq_len
SR1-SR5
prefix_failed
逐任务 success / total
ABCD group success
slow-call rate
refresh delta / jerk（仅作辅助指标）
```

预期的好结果应表现为：transition validation loss 下降、normal validation loss 退化不超过 5%、prefix failure 不增加，并最终使 full benchmark 的 avg sequence length 和 SR4/SR5 提升。

预期的坏结果包括：target attempted SR 提高但 all-case 和 prefix 继续下降、只有训练任务提升、normal loss 上升、history 置零后性能不变、动作更平滑但成功率不升。这些情况分别对应遗忘、过拟合、history 通路未被使用或再次把数值平滑误当作任务目标。

## 4. LoRA 数据采集脚本实现

### 4.1 文件位置

本轮为 transition-conditioned LoRA 建立了独立目录，不修改先前的 `LoRA_trial`：

```text
RoboDual/LoRA_transition_0711
```

其中包含：

```text
collect_transition_rollouts.py   在线成功 rollout 与训练窗口采集
history_adapter.py               committed action history adapter 及安装接口
test_data_contract.py            数据比例、adapter 梯度和安装逻辑测试
README.md                        数据格式和启动说明
```

此外，对 specialist 的 DiT 增加了一个默认关闭的 history adapter 接口：

```text
RoboDual/prismatic/models/policy/diffusion_transformer.py
```

当 `history_adapter is None` 时，原 specialist、旧 checkpoint 和原评测路径不发生变化。只有训练或加载新模型时显式安装 adapter，history feature 才会以 residual 的方式加入 `global_cond`。

### 4.2 相比先前采集脚本的主要修改

先前 `collect_lora_rollouts.py` 的数据主要用于学习：

```text
current observation + stale old hidden + empty ref -> successful future
```

新脚本改为复现真实部署中的 condition refresh：

```text
已经执行的 4 步 action history
+ 当前 observation / proprio
+ 当前 observation 上重新运行 generalist 得到的 new ref / new hidden
-> 同一成功 rollout 的未来 8 步动作
```

主要修改包括：

1. 直接复用 `evaluate_calvin_task_age_0525.py` 的 task-age 分组和调度逻辑。
2. 在真实在线 rollout 中截获 slow call 输出，不离线近似构造 new condition。
3. 在调用当前 step 之前读取 history，避免把当前动作泄漏到输入。
4. 只有完整 subtask 成功后才保存轨迹，失败 rollout 不提供动作监督。
5. 默认使用 CALVIN `training` split，并通过指纹排除官方 100 条 benchmark sequence。
6. 按完整 trajectory 划分 train / validation / test，同一轨迹的相邻窗口不会跨 split。
7. 对 A-C 组设置单任务上限，避免少量简单任务占满整个组的采集配额。
8. 对样本数量、ABCD 配额和各 split 完成度执行强制检查，避免长时间运行后静默生成不可训练的数据集。

### 4.3 slow condition 与 history 的核心逻辑

每个 subtask 开始时调用：

```text
model.reset()
model.set_current_task(task)
```

因此 task-age scheduler 会按照 0525 实验中的配置运行：

| 任务组 | slow refresh age | 默认成功轨迹配额 |
|---|---:|---:|
| A | 13 | 60 |
| B | 12 | 60 |
| C | 10 | 30 |
| D | 8 | 20 |

在每一步调用 `model.step()` 之前，脚本从 evaluator 的 `hist_action` deque 中读取最多 4 个已经送入环境的动作，左侧不足部分补零：

```text
hist_action_before: [4, 7]
```

随后执行当前 step。如果 `last_step_profile["slow_system"] == True`，则保存本次真实 slow call 产生的：

```text
slow_action / new_ref: [1, 8, 7]
slow_hidden / new_hidden
refresh step
refresh age
old_condition_id
```

这些 condition 来自当前 step 的 observation，而不是从较早帧重用 old hidden，也不是在 rollout 结束后重新运行 generalist。因此它们与 fast specialist 当时实际接收到的 condition 一致。

condition 不会在每个 frame 中重复保存。每次 slow call 只在下面的位置保存一次：

```text
conditions/<trajectory_id>/condition_XXX.pt
```

普通样本通过 `condition_id + slow_age` 引用最近一次 condition；refresh 样本额外保存 `old_condition_id`，后续可用于 stale/refreshed 双视图一致性训练。

### 4.4 采集内容与监督目标

每个成功 trajectory 的逐步数据保存在：

```text
trajectories/<trajectory_id>/step_XXXX.npz
```

每帧主要包含：

```text
rgb_static
rgb_gripper
depth_static
depth_gripper
robot_obs
scene_obs
rel_actions
hist_action_before
```

训练样本索引保存在：

```text
samples.jsonl
```

每条样本记录：

```text
trajectory_id
split
step
category
condition_id
old_condition_id（仅 refresh）
slow_age / refresh_age
动作冲突和 jerk 指标
history_steps = 4
action_chunk_size = 8
```

监督 target 由同一成功在线 rollout 中的动作构成：

```text
target_action = action[t:t+8]
```

成功检测后会再次检查 future-window 边界，不保存不足 8 步 target 的尾部窗口。

### 4.5 四类样本与筛选条件

最终样本在 train、validation、test 三个 split 内分别保持：

| 类别 | 比例 | 定义 |
|---|---:|---|
| `normal` | 50% | ref 尚未过期且当前没有发生 refresh 的普通成功窗口 |
| `refresh` | 30% | task-age scheduler 真实触发 slow call 的普通刷新窗口 |
| `high_conflict` | 10% | refresh 的困难子集：new ref 与已执行动作/old ref 冲突较大，或执行 jerk 较大 |
| `stale` | 10% | `slow_age >= 8`、ref 已为空且当前没有 refresh 的成功窗口 |

默认 high-conflict 判定满足以下任意条件：

```text
||new_ref[0, :6] - previous_executed_action[:6]||_2 >= 0.18
||new_ref[0, :6] - old_ref_last[:6]||_2 >= 0.18
refresh step jerk_l2_ee6 >= 0.24
```

`high_conflict` 在在线计数时同时计入 `refresh_total`。最终抽样先选出 10% high-conflict，再从尚未选中的全部 refresh 中选择 30% 普通 refresh；剩余 high-conflict 可以作为普通 refresh 使用，但同一个窗口不会被重复选择。这样既符合“high-conflict 是 refresh 子集”的语义，也避免高冲突样本很多时反而造成普通 refresh 池不足。

数据集默认目标为 8000 个窗口，split 为：

```text
train      5600
validation 1200
test       1200
```

默认退出要求为：

```text
ABCD 四组轨迹配额全部达到
train / validation / test 内的 normal 目标全部达到
三个 split 内的 refresh_total 目标全部达到
三个 split 内的 high_conflict 最低目标全部达到
三个 split 内的 stale 目标全部达到
```

以 8000 样本为例，脚本会维护 12 个在线分项目标。即使候选总数已经超过 8000，只要其中任何一项仍有缺口，采集就会继续。ABCD 达标后，`max_trajectories_per_task` 只是不再让过量任务计入组配额，不会阻止其成功轨迹用于补充稀缺样本。

若未满足要求，`collection_summary.json` 会写入：

```text
status: incomplete
missing_groups
category_deficits
```

当 ABCD 和所有分项都达到时脚本立即退出；若扫描完 `num_sequences` 后仍有缺口，命令返回非零。`--allow_incomplete` 只应在小规模诊断采集时使用。

### 4.6 History adapter 实现

history adapter 的输入和结构为：

```text
hist_action [B, 4, 7]
-> Flatten [B, 28]
-> Linear(28, hidden_size)
-> SiLU
-> Linear(hidden_size, hidden_size)
-> scalar gate
-> residual add to global_cond
```

最后一个 Linear 使用零初始化，gate 初始化为 1。因此在安装后、训练前 adapter 输出严格为零，不改变 base specialist；同时最后一层在第一个 optimizer step 就能获得非零梯度，不会出现“输出层和 gate 同时为零”造成的梯度死锁。

推荐在构造 `DualSystem` 之前安装 adapter，使 EMA deepcopy 自动包含该结构。如果已经构造 `DualSystem`，应使用：

```python
install_dual_system_history_adapters(dual_system)
```

该接口会同时为：

```text
dual_system.fast_system
dual_system.ema_fast_system.ema_model
```

安装匹配的 adapter。加载带 adapter 的 checkpoint 时也必须先安装结构，否则 `strict=False` 可能忽略 adapter 权重。

### 4.7 推荐启动命令

在 `RoboDual` 目录、`dualsys_env` 环境中运行：

```bash
CALVIN_ROOT=/home/rosmontis/Projects/dualsys/calvin \
python LoRA_transition_0711/collect_transition_rollouts.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --dataset_subdir calvin_debug_dataset \
  --dataset_split training \
  --output_dir LoRA_transition_0711/collected_transition_v1 \
  --num_sequences 1000 \
  --sequence_start 100 \
  --exclude_benchmark_sequences 100 \
  --target_samples 8000 \
  --group_trajectory_quotas A:60,B:60,C:30,D:20 \
  --max_trajectories_per_task 8 \
  --high_conflict_prev_threshold 0.18 \
  --high_conflict_old_new_threshold 0.18 \
  --high_conflict_jerk_threshold 0.24 \
  --load_in_4bit \
  --low_cpu_mem_usage
```

首次正式采集不加 `--overwrite`。如果输出目录已经存在，脚本会拒绝覆盖，避免不同模型或参数产生的 condition 被混合。只有确认需要完全删除旧采集结果并重新开始时才使用 `--overwrite`。

### 4.8 主要参数说明

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--dataset_split` | `training` | 明确选择训练环境，正式采集不要改为 validation |
| `--num_sequences` | 1000 | 最多扫描的 CALVIN sequence 数量 |
| `--sequence_start` | 100 | 从候选生成结果的第 100 条之后开始扫描 |
| `--exclude_benchmark_sequences` | 100 | 额外按指纹排除官方 100 条评测 sequence |
| `--ep_len` | 360 | 单个 subtask 最大执行步数 |
| `--target_samples` | 8000 | 最终窗口目标，必须是 200 的正整数倍 |
| `--group_trajectory_quotas` | `A:60,B:60,C:30,D:20` | 四个 task-age 组的成功轨迹目标 |
| `--max_trajectories_per_task` | 8 | A-C 组单任务上限；D 组只有 stack，因此豁免 |
| `--history_steps` | 4 | committed action history 长度，当前固定为 4 |
| `--action_chunk_size` | 8 | future target 长度，当前固定为 8 |
| `--empty_ref_after_age` | 8 | stale / empty-ref 样本的起始 age |
| `--high_conflict_prev_threshold` | 0.18 | new ref 与上一执行动作的 ee6 L2 阈值 |
| `--high_conflict_old_new_threshold` | 0.18 | old/new slow ref 的 ee6 L2 阈值 |
| `--high_conflict_jerk_threshold` | 0.24 | refresh step 的 ee6 jerk 阈值 |
| `--load_in_4bit` | false | 以 4-bit 加载 generalist，降低显存占用 |
| `--fast_num_inference_steps` | 10 | specialist diffusion 推理步数，应与基线保持一致 |
| `--allow_incomplete` | false | 允许不完整诊断数据正常退出，正式采集不应启用 |

脚本已经通过静态编译、CLI 加载和 10 个 CPU 数据契约测试，并经过独立代码审阅。审阅重点确认了有限循环退出、成功后保存、history 无当前动作泄漏、future target 边界、condition 引用、training split、EMA adapter，以及“总量足够但分项不足时继续补采”的退出逻辑。完整 GPU 数据采集尚未在脚本完成阶段自动启动。

### 4.9 分项补采与退出机制修正

最初版本只把 ABCD 成功轨迹配额作为在线提前退出条件，50/30/10/10 和 split 完整度要到采集结束后才检查。这会导致脚本在约 170 条成功轨迹后停止，然后因为 refresh/high-conflict 不足而返回 `incomplete`，无法利用剩余 sequence 自动补充稀缺样本。

修正后，脚本在线维护：

```text
(train, normal / refresh_total / high_conflict / stale)
(validation, normal / refresh_total / high_conflict / stale)
(test, normal / refresh_total / high_conflict / stale)
```

默认 8000 样本对应：

| split | normal | refresh total | 其中 high-conflict 至少 | stale | 最终样本数 |
|---|---:|---:|---:|---:|---:|
| train | 2800 | 2240 | 560 | 560 | 5600 |
| validation | 600 | 480 | 120 | 120 | 1200 |
| test | 600 | 480 | 120 | 120 | 1200 |

其中 `refresh total` 最终拆成 30% 普通 refresh 和 10% high-conflict。真正的提前退出条件变为：

```text
ABCD group quotas complete
AND all 12 split/category requirements complete
```

如果 ABCD 已达标但任一分项不足，后续成功轨迹仍会写盘并更新候选计数。只有所有目标同时完成，或达到 `--num_sequences` 硬上限，循环才停止。后者仍会产生带 `category_deficits` 的 `incomplete` summary 并返回非零。

## 5. Transition History LoRA 训练脚本实现

### 5.1 文件与数据输入

训练入口为：

```text
RoboDual/LoRA_transition_0711/train_transition_lora.py
```

训练脚本直接读取已经 finalize 的：

```text
samples.jsonl
trajectories.jsonl
trajectories/<trajectory>/step_XXXX.npz
conditions/<trajectory>/condition_XXX.pt
```

它不会重新运行 generalist。`generalist_path` 仅用于加载与采集/评测一致的 image processor。训练 condition 直接使用在线 rollout 当时保存的 `slow_action` 和 `slow_hidden`，从而避免重新推理造成 condition 漂移，也显著降低训练显存。

每个训练样本恢复：

```text
current RGB/depth/gripper observation
previous RGB observation
robot proprio
hist_action_before [4,7]
slow action chunk [8,7]
slow hidden [tokens,4096]
target future action [8,7]
```

不同 instruction 的 slow hidden token 数可能为 87、88 等。collator 使用末 token padding 补到 batch 内最大长度，与旧 LoRA trainer 的处理一致。

### 5.2 ref action 重建

训练严格复现 `DualSystemCalvinEvaluation._build_ref_actions_from()`：

```text
age = 0: 完整 8 步 ref
age = 1..7: 使用 slow chunk 最后的 8-age 步，写入 ref 前部
age >= 8: empty ref
```

因此 normal、refresh/high-conflict 和 stale 样本分别获得部署时真实对应的 condition 形式，而不是统一 full-ref 或统一 empty-ref。

### 5.3 实际训练层

脚本启动时强制校验以下 14 个 LoRA target 必须全部存在且没有额外 target：

```text
model.x_embedder
model.context_adapter
model.blocks.0-5.attn_temporal.qkv
model.blocks.0-5.attn_temporal.proj
```

参数为：

```text
LoRA rank = 4
LoRA alpha = 8
LoRA dropout = 0.05
```

另外完整训练：

```text
model.history_adapter
```

默认总可训练参数为 128569。脚本会检查 optimizer 参数名，以下模块一旦意外变为 trainable 就立即报错：

```text
visual/depth/gripper adapters
proprio embedder
block MLP
cross attention
final layer
```

generalist 不加载、不训练。DINO 构造时使用 `vision_encoder_pretrained=False`，随后从 specialist checkpoint 恢复 vision encoder 权重，避免训练启动时访问 Hugging Face；原评测入口默认行为保持不变。

### 5.4 Validation 与 early stopping

训练前先在固定 validation subset 上计算冻结 base specialist loss。subset 从四类样本中各取 64 条：

```text
normal 64
refresh 64
high_conflict 64
stale 64
```

验证使用固定 diffusion seed，并保证每个 validation batch 只包含一种 category，从而分别记录：

```text
loss_overall
loss_normal
loss_refresh
loss_high_conflict
loss_stale
normal_vs_base_ratio
```

每 100 optimizer steps 验证一次，patience 默认为 5。只有满足：

```text
normal validation loss / base normal loss <= 1.05
```

的候选才会写入 `adapter_best.pt`。同时保留 `adapter_best_unconstrained.pt` 作为没有候选满足 normal 约束时的显式 fallback。最终 merged checkpoint 来自选中的 best adapter，不直接使用最后一步参数。

训练还记录：

```text
train loss / loss_ma100
gradient norm
history adapter output norm
history gate
```

best 模型选定后，在独立 test subset 上报告同样的四类 loss，但 test 不参与 early stopping。

### 5.5 Checkpoint 输出

输出目录主要包含：

```text
training_config.json
validation_baseline.json
metrics.jsonl
adapter_best.pt
adapter_best_unconstrained.pt
adapter_final.pt
specialist_transition_lora_merged_policy.pt
specialist_transition_lora_merged_ema.pt
training_summary.json
```

adapter 文件同时保存 LoRA state 与 history adapter state。EMA-compatible checkpoint 为 online/EMA 两套模型补充 history adapter 权重；后续评测加载该 checkpoint 前仍必须先安装 history adapter 结构。

当前数据 summary 中 `D=3/20`，训练脚本会把：

```text
data_status = usable_with_group_undercoverage
data_missing_groups = {D: 17}
```

写入 `training_config.json`，不会掩盖 stack_block 覆盖不足。

### 5.6 推荐启动命令

在 `RoboDual` 目录和 `dualsys_env` 环境中执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python LoRA_transition_0711/train_transition_lora.py \
  --generalist_path /home/rosmontis/Projects/dualsys/models/generalist \
  --specialist_path /home/rosmontis/Projects/dualsys/models/specialist/Specialist+Depth+Gripper.pt \
  --data_dir LoRA_transition_0711/collected_transition_v1 \
  --output_dir LoRA_transition_0711/lora_runs/transition_history_lora_v1 \
  --batch_size 2 \
  --max_steps 3000 \
  --learning_rate 3e-5 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --validation_interval 100 \
  --validation_samples_per_category 64 \
  --validation_batch_size 2 \
  --early_stopping_patience 5 \
  --max_normal_loss_ratio 1.05 \
  --save_adapter_steps 500 \
  --bf16
```

`max_steps=3000` 约等于 batch size 2 下遍历 5600 个 train samples 一轮。首次训练不使用 `--overwrite_output`；只有明确放弃同名旧 run 时才允许该参数清空输出目录。

### 5.7 已完成检查

当前已完成：

```text
py_compile
CLI --help
8000 样本 manifest 与 split/category 读取
87/88 token mixed-hidden collate
age 0/7/8 ref reconstruction tests
14 个 LoRA target 实际模型装配
specialist checkpoint 权重加载
禁止层 trainable 检查
batch=2 compute_loss + backward
LoRA 与 history adapter 非零梯度检查
15 个 CPU contract tests
```

单 batch 烟雾测试得到有限 loss，LoRA 和 history adapter 均存在非零梯度。当前执行环境没有可用 CUDA，因此尚未由此脚本启动完整 GPU 训练。
codex resume 019f3773-e13e-7ba3-b8b4-7eb2f9772be0
