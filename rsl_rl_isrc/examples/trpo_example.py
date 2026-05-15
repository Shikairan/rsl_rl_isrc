#!/usr/bin/env python3
"""
TRPO算法使用示例
基于rsl_rl框架的Trust Region Policy Optimization实现
"""

import torch
import numpy as np
from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.algorithms import TRPO
from rsl_rl_isrc.storage import RolloutStorage

def create_trpo_config():
    """创建TRPO算法配置"""
    config = {
        "algorithm": {
            "algorithm_class_name": "TRPO",          # 指定使用TRPO算法
            "num_learning_epochs": 1,                # 每次更新的epoch数
            "learning_rate": 1e-2,                   # 价值函数学习率
            "gamma": 0.99,                           # 折扣因子
            "lam": 0.95,                             # GAE参数
            "kl_constraint": 0.0005,                 # KL散度约束
            "alpha": 0.5                             # 线性搜索参数
        },
        "policy": {
            "policy_class_name": "ActorCritic",     # 使用现有的ActorCritic
            "actor_hidden_dims": [64, 64],          # actor网络隐藏层
            "critic_hidden_dims": [64, 64],         # critic网络隐藏层
            "activation": "elu",
            "init_noise_std": 1.0
        },
        "runner": {
            "experiment_name": "trpo_cartpole",
            "num_steps_per_env": 100,               # 每个环境收集的步数
            "save_interval": 50
        }
    }
    return config

def trpo_example():
    """
    TRPO算法完整示例
    展示如何使用TRPO进行稳定的policy optimization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建配置
    config = create_trpo_config()

    # 模拟环境参数 (CartPole示例)
    num_envs = 4
    num_obs = 4  # CartPole observation space
    num_actions = 2  # CartPole action space
    num_privileged_obs = None

    # 创建策略网络
    actor_critic = ActorCritic(
        num_obs=num_obs,
        num_critic_obs=num_obs,  # TRPO需要critic
        num_actions=num_actions,
        **config["policy"]
    ).to(device)

    # 创建TRPO算法
    trpo_alg = TRPO(
        actor_critic=actor_critic,
        device=device,
        **config["algorithm"]
    )

    # 创建存储器 (on-policy)
    storage = RolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=config["runner"]["num_steps_per_env"],
        obs_shape=[num_obs],
        privileged_obs_shape=[num_privileged_obs] if num_privileged_obs else [None],
        actions_shape=[num_actions],
        device=device
    )

    print("开始TRPO训练...")
    print(f"设备: {device}")
    print(f"环境数量: {num_envs}")
    print(f"每次迭代步数: {config['runner']['num_steps_per_env']}")
    print(f"KL约束: {config['algorithm']['kl_constraint']}")

    # 训练循环示例
    for iteration in range(10):
        print(f"\n=== 迭代 {iteration + 1} ===")

        # 1. 数据收集阶段 (这里用随机数据模拟)
        collect_transitions(storage, num_envs, num_obs, num_actions, device, trpo_alg)

        # 2. TRPO更新
        critic_loss = trpo_alg.update(storage)
        print(".4f")

        # 3. 清理存储 (为下一轮准备)
        # TRPO是on-policy算法，所以每轮都要重新收集数据

    print("\nTRPO训练完成!")

def collect_transitions(storage, num_envs, num_obs, num_actions, device, trpo_alg):
    """模拟数据收集过程"""
    # 这里用随机数据模拟实际的环境交互
    for env_idx in range(num_envs):
        # 模拟一个episode的数据收集
        episode_length = np.random.randint(10, 50)  # 随机episode长度

        for step in range(episode_length):
            # 创建随机观测
            obs = torch.randn(1, num_obs).to(device)

            # 使用TRPO采样动作
            actions, action_log_probs = trpo_alg.act(obs.unsqueeze(0))

            # 创建随机奖励和done信号
            reward = torch.randn(1, 1).to(device)
            done = torch.tensor([[step == episode_length - 1]], dtype=torch.bool).to(device)

            # 创建transition
            transition = RolloutStorage.Transition()
            transition.observations = obs
            transition.critic_observations = obs  # TRPO使用critic
            transition.actions = actions.squeeze(0)
            transition.rewards = reward
            transition.dones = done
            transition.actions_log_prob = action_log_probs.squeeze(0)
            transition.values = torch.zeros_like(reward)  # 将由critic计算
            transition.action_mean = torch.zeros_like(actions.squeeze(0))
            transition.action_sigma = torch.zeros_like(actions.squeeze(0))
            transition.hidden_states = None

            # 添加到存储器
            storage.add_transitions(transition)

    print(f"收集了 {num_envs * episode_length} 步数据")

if __name__ == "__main__":
    trpo_example()
