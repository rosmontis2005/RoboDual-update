# Sequence 60 TCP trajectory figures

这些图片对比本次采集的 `Original fixed-8` 与 `Fixed age-12` 两种策略。轨迹直接取自 PyBullet 仿真真值：

`environment.{pre,post}_physics.tcp.world_link_frame_position`

每条轨迹由 subtask 开始前的 TCP 位置和每个环境步结束后的 TCP 位置组成，不通过 action 积分重建。

## 图例

- 蓝线：original fixed-8；
- 橙线：fixed age-12；
- 圆点/方块：轨迹起点/终点；
- `×`：执行 slow call 时的 TCP 位置；
- 淡橙色竖带：age-12 的空 reference 环境步；
- 点线：目标 block 的仿真位置，应用于 blue-block 和 pink-block subtask；
- 夹爪图同时展示执行命令和两个夹爪关节绝对位置之和。

每张 subtask 图中的两种策略共享相同坐标范围。总览图的每个子图也在策略间共享范围，但不同 subtask 之间会独立缩放。

## 重新生成

```bash
cd /home/rosmontis/Projects/dualsys/RoboDual

env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/robodual_mpl \
  /home/rosmontis/miniconda3/envs/dualsys_env/bin/python \
  'single _sequence_trial/pic/plot_tcp_trajectories.py'
```

`tcp_trajectory_summary.json` 保存每张图对应的来源、步数、slow-call 数、空 reference 数、TCP 起止点、路径长度和坐标范围。
