# SAC 算法训练测试

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

## 概述

`test_sac_training.py` 是基于 rsl_rl 框架的 **SAC (Soft Actor-Critic)** 算法训练测试，用于**连续动作空间**的 off-policy 强化学习。

## 文件说明

### `test_sac_training.py`
完整的 SAC 算法训练测试，包含：
- Pendulum-v1 连续控制环境
- Actor-Critic + 自动熵调节
- ReplayBuffer 经验回放
- 双 Q 网络减少过估计

## 主要特性

### 1. SAC 核心机制
- **最大熵强化学习**: 策略最大化回报与熵
- **自动 α 调节**: 维持目标熵
- **双 Q 网络**: Q1、Q2 取最小值降低过估计
- **Off-policy**: 使用 ReplayBuffer，样本可重复利用

### 2. 与 on-policy 算法的区别
| 项目 | PPO/TRPO | SAC |
|------|----------|-----|
| 数据使用 | 单次 | 多次（replay） |
| 存储 | RolloutStorage | ReplayBuffer |
| 更新时机 | 收集满 rollout | 每 step 可更新 |
| 典型环境 | 离散/连续 | 通常连续 |

### 3. 训练流程
- 前期随机探索：`learning_starts` 步内使用随机动作
- 之后按 `update_frequency` 和 `num_updates_per_step` 更新

## 环境

### Pendulum-v1
- **观察空间**: 3 维
- **动作空间**: 1 维连续 `[-2.0, 2.0]`
- **合理目标**: 平均回报 > -200

## 训练流程

### 1. 初始化
```python
sac_networks = SACNetworks(
    num_obs=3, num_actions=1,
    actor_hidden_dims=[256, 256],
    critic_hidden_dims=[256, 256],
)
sac_networks.set_action_bounds(action_space.low, action_space.high)

policy = SAC(
    sac_networks=sac_networks,
    gamma=0.99,
    buffer_size=1e6,
    batch_size=256,
    learning_starts=5000,
    update_frequency=256,
    num_updates_per_step=2,
    ...
)
policy.init_storage(num_envs=num_envs, obs_shape=(3,), action_shape=(1,))
```

### 2. 训练循环
```python
while step < max_steps:
    actions = policy.act(obs)  # learning_starts 前可能返回 None
    if actions is None:
        actions = env.single_action_space.sample()
    next_obs, rewards, dones, truncateds, infos = env.step(actions.cpu().numpy())
    policy.process_env_step(rewards, dones, infos, next_obs=next_obs, obs=obs, actions=actions)
    if step >= learning_starts:
        qf1_loss, qf2_loss, actor_loss, alpha_loss = policy.update()
```

### 3. 成功标准
- `final_avg_reward > -200`

## 运行方式

### 直接运行
```bash
cd /path/to/rsl_rl
python rsl_rl/tests/test_sac_training.py
```

### 主程序默认配置
```python
num_episodes = 10000
num_envs = 8
buffer_size = 1e5      # 测试用较小 buffer
batch_size = 128
learning_starts = 1000
update_frequency = 256
num_updates_per_step = 3
```

## 参数配置

### SAC 关键参数
```python
buffer_size = 1e6           # Replay buffer 大小
batch_size = 256           # 每次更新 batch 大小
learning_starts = 5000     # 开始学习前的随机步数
update_frequency = 256     # 每 N 个 transition 更新
num_updates_per_step = 2   # 每次触发时的更新次数
policy_lr = 1e-4
q_lr = 1e-3
alpha_lr = 1e-5
```

### 梯度裁剪
```python
critic_grad_clip = True
critic_max_grad_norm = 0.5
actor_grad_clip = True
actor_max_grad_norm = 0.5
```

## 输出示例

```
=== SAC 训练测试 ===
开始SAC训练测试...
Replay Buffer大小: 100000
...
SAC更新 - 步数: 1024, Q损失: 0.1234/0.1156, 策略损失: -0.234, Alpha损失: 0.056, ...
训练完成!
最终平均奖励: -178.234
训练成功: True
```

## 依赖

- PyTorch
- gymnasium
- rsl_rl (SAC, SACNetworks, ReplayBuffer)
- tqdm
- numpy

## 注意事项

1. `learning_starts` 过小可能导致早期不稳定
2. `buffer_size` 影响内存，测试时可适当缩小
3. SAC 对超参数相对不敏感，但 `batch_size` 不宜过小
