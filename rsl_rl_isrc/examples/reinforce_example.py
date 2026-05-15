#!/usr/bin/env python3
"""
REINFORCE算法使用示例
基于rsl_rl框架的非监督强化学习实现
"""

import torch
import numpy as np
from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.algorithms import REINFORCE
from rsl_rl_isrc.storage import RolloutStorage

def create_reinforce_config():
    """创建REINFORCE算法配置"""
    config = {
        "algorithm": {
            "algorithm_class_name": "REINFORCE",
            "num_learning_epochs": 1,
            "learning_rate": 1e-3,
            "gamma": 0.99
        },
        "policy": {
            "policy_class_name": "ActorCritic",
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],  # Not used in REINFORCE
            "activation": "elu",
            "init_noise_std": 1.0
        },
        "runner": {
            "experiment_name": "reinforce_cartpole",
            "num_steps_per_env": 100,  # 每个环境收集的步数
            "save_interval": 50
        }
    }
    return config

def reinforce_example():
    """
    REINFORCE算法完整示例
    展示如何使用off-policy REINFORCE进行训练
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建配置
    config = create_reinforce_config()

    # 模拟环境参数 (CartPole示例)
    num_envs = 4
    num_obs = 4  # CartPole observation space
    num_actions = 2  # CartPole action space
    num_privileged_obs = None

    # 创建策略网络
    actor_critic = ActorCritic(
        num_obs=num_obs,
        num_critic_obs=num_obs,  # REINFORCE不使用critic
        num_actions=num_actions,
        **config["policy"]
    ).to(device)

    # 创建REINFORCE算法
    reinforce_alg = REINFORCE(
        actor_critic=actor_critic,
        device=device,
        **config["algorithm"]
    )

    # 创建存储器 (包含off-policy episodes功能)
    storage = RolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=config["runner"]["num_steps_per_env"],
        obs_shape=[num_obs],
        privileged_obs_shape=[num_privileged_obs] if num_privileged_obs else [None],
        actions_shape=[num_actions],
        device=device
    )

    print("开始REINFORCE训练...")
    print(f"设备: {device}")
    print(f"环境数量: {num_envs}")
    print(f"每次迭代步数: {config['runner']['num_steps_per_env']}")

    # 训练循环示例
    for iteration in range(10):
        print(f"\n=== 迭代 {iteration + 1} ===")

        # 1. 数据收集阶段 (这里用随机数据模拟)
        collect_transitions(storage, num_envs, num_obs, num_actions, device)

        # 2. REINFORCE更新
        loss = reinforce_alg.update(storage)
        print(".4f")

        # 3. 清理旧的episodes (可选)
        if len(storage.off_policy_episodes) > 100:
            storage.clear_off_policy_episodes()

    print("\nREINFORCE训练完成!")

def collect_transitions(storage, num_envs, num_obs, num_actions, device):
    """模拟数据收集过程"""
    # 这里用随机数据模拟实际的环境交互
    for env_idx in range(num_envs):
        # 模拟一个episode
        episode_length = np.random.randint(10, 50)  # 随机episode长度

        for step in range(episode_length):
            # 创建随机观测和动作
            obs = torch.randn(1, num_obs).to(device)
            action = torch.randint(0, num_actions, (1, 1)).to(device)
            reward = torch.randn(1, 1).to(device)
            done = torch.tensor([[step == episode_length - 1]], dtype=torch.bool).to(device)

            # 创建transition
            transition = RolloutStorage.Transition()
            transition.observations = obs
            transition.critic_observations = obs  # REINFORCE不使用critic
            transition.actions = action.float()
            transition.rewards = reward
            transition.dones = done
            transition.actions_log_prob = torch.randn(1, 1).to(device)  # 模拟log prob
            transition.values = torch.zeros_like(reward)  # REINFORCE不使用values
            transition.action_mean = torch.zeros_like(action.float())
            transition.action_sigma = torch.zeros_like(action.float())
            transition.hidden_states = None

            # 添加到episode
            storage.add_off_policy_transition(transition, env_idx)

        # 完成episode
        storage.finish_episode(env_idx, gamma=0.99)

    print(f"收集了 {len(storage.off_policy_episodes)} 个episodes")

if __name__ == "__main__":
    reinforce_example()
