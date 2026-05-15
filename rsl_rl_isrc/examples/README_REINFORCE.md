# REINFORCE算法使用指南

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

本指南介绍如何在rsl_rl框架中使用新添加的REINFORCE (Policy Gradient) 算法。

## 概述

REINFORCE是一种经典的policy gradient方法，通过最大化期望累积奖励来直接优化策略。与PPO不同，REINFORCE是off-policy算法，可以重用过去的经验进行学习。

## 主要特性

- **Off-Policy学习**: 利用RolloutStorage中的episodes进行离线学习
- **完整的Episode存储**: 支持完整的trajectory存储和discounted returns计算
- **分布式训练兼容**: 支持多GPU分布式训练
- **与现有代码集成**: 无缝集成到rsl_rl框架中

## 文件结构

新增的文件：
```
rsl_rl_isrc/
├── algorithms/
│   ├── reinforce.py              # REINFORCE算法实现
│   └── __init__.py               # 添加REINFORCE导入
├── runners/
│   ├── reinforce_runner.py       # REINFORCE训练runner
│   └── __init__.py               # 添加REINFORCERunner导入
├── storage/
│   └── rollout_storage.py        # 添加Episode类和off-policy功能
└── examples/
    ├── reinforce_example.py      # 使用示例
    └── README_REINFORCE.md       # 本文档
```

## 基本使用

### 1. 配置设置

```python
config = {
    "algorithm": {
        "algorithm_class_name": "REINFORCE",    # 指定使用REINFORCE算法
        "num_learning_epochs": 1,               # 每次更新的epoch数
        "learning_rate": 1e-3,                  # 学习率
        "gamma": 0.99                          # 折扣因子
    },
    "policy": {
        "policy_class_name": "ActorCritic",    # 使用现有的ActorCritic
        "actor_hidden_dims": [64, 64],         # actor网络隐藏层
        "critic_hidden_dims": [64, 64],        # 不用于REINFORCE，但为兼容性保留
        "activation": "elu",
        "init_noise_std": 1.0
    },
    "runner": {
        "experiment_name": "reinforce_experiment",
        "num_steps_per_env": 100,              # 每个环境收集的步数
        "save_interval": 50                    # 保存间隔
    }
}
```

### 2. 初始化

```python
from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.algorithms import REINFORCE
from rsl_rl_isrc.runners import REINFORCERunner

# 创建策略网络
actor_critic = ActorCritic(
    num_obs=env.num_obs,
    num_critic_obs=env.num_obs,  # REINFORCE不使用critic
    num_actions=env.num_actions,
    **config["policy"]
).to(device)

# 创建REINFORCE runner
runner = REINFORCERunner(
    env=env,
    train_cfg=config,
    log_dir=log_dir,
    device=device
)
```

### 3. 训练

```python
# 开始训练
runner.learn(num_learning_iterations=1000)
```

## 核心API

### REINFORCE类

```python
class REINFORCE:
    def __init__(self, actor_critic, learning_rate=1e-3, gamma=0.99, device='cpu', **kwargs)
    def act(self, observations)  # 采样动作
    def update(self, storage)    # 更新策略
    def save(self, path)         # 保存模型
    def load(self, path)         # 加载模型
```

### RolloutStorage Off-Policy功能

```python
class RolloutStorage:
    def add_off_policy_transition(self, transition, env_idx)  # 添加transition到episode
    def finish_episode(self, env_idx, gamma=0.99)            # 完成episode并计算returns
    def get_off_policy_episodes(self, batch_size=None)        # 获取episodes用于训练
    def clear_off_policy_episodes(self)                       # 清理旧episodes
```

### Episode类

```python
class Episode:
    def add_transition(self, transition)  # 添加单步transition
    def length(self)                      # 获取episode长度
    def to_tensors(self, device)          # 转换为tensor格式
```

## 训练流程详解

1. **数据收集**: 在每个时间步，执行动作并观察结果
2. **Episode管理**: 将transitions添加到对应环境的episode中
3. **Episode完成**: 当检测到done时，计算discounted returns
4. **策略更新**: 使用收集的episodes进行REINFORCE更新

```python
# 在训练循环中
for step in range(num_steps):
    # 采样动作
    actions, log_probs = policy.act(observations)

    # 环境交互
    next_obs, rewards, dones, infos = env.step(actions)

    # 为每个环境添加transition
    for env_idx in range(num_envs):
        transition = create_transition(obs[env_idx], actions[env_idx], ...)
        storage.add_off_policy_transition(transition, env_idx)

        # 检查episode结束
        if dones[env_idx]:
            storage.finish_episode(env_idx, gamma=0.99)

    # 策略更新
    loss = reinforce_alg.update(storage)
```

## 超参数调优

- **learning_rate**: 1e-3到1e-4之间，通常从1e-3开始
- **gamma**: 折扣因子，0.95-0.99之间
- **num_learning_epochs**: 每次更新的epoch数，通常设为1
- **episode存储数量**: 根据内存情况调整，建议100-1000个episodes

## 与PPO对比

| 特性 | REINFORCE | PPO |
|------|-----------|-----|
| On/Off Policy | Off-policy | On-policy |
| 样本效率 | 低 | 高 |
| 稳定性 | 低 | 高 |
| 方差 | 高 | 低 |
| 收敛速度 | 慢 | 快 |

## 注意事项

1. **方差问题**: REINFORCE具有较高的梯度方差，建议使用较大的batch size
2. **探索**: 需要适当的探索策略，可以通过entropy bonus或epsilon-greedy实现
3. **内存使用**: Off-policy存储会占用额外内存，需要定期清理旧episodes
4. **收敛**: 可能需要更多的训练时间，比PPO更难收敛

## 示例代码

查看 `reinforce_example.py` 获取完整的使用示例。

## 故障排除

1. **训练不稳定**: 降低学习率，增加episode数量
2. **内存不足**: 减少存储的episodes数量，定期清理
3. **收敛慢**: 检查reward函数，确保有适当的探索

## 扩展

REINFORCE可以扩展为：
- **Actor-Critic**: 结合critic进行baseline减去，降低方差
- **Advantage**: 使用优势函数替代returns
- **Trust Region**: 添加trust region约束，提高稳定性

这些扩展可以参考现有的PPO实现。
