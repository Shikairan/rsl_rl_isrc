#!/usr/bin/env python3
"""
TRPO RNN 示例
演示如何使用带有RNN选项的TRPOPolicy
"""

import torch
import gymnasium as gym
from rsl_rl_isrc.algorithms.trpo_policy import TRPOPolicy

def main():
    # 创建RNN版本的TRPO策略
    trpo_policy = TRPOPolicy(
        num_obs=3,           # Pendulum环境的状态维度
        num_actions=1,       # Pendulum环境的动作维度
        rnn_hidden_size=64,  # 启用RNN，隐藏层大小64
        rnn_type='lstm',     # 使用LSTM
        rnn_num_layers=1,    # RNN层数
        action_bounds=(-2.0, 2.0),  # Pendulum动作边界
        device='cpu'
    )

    # 初始化存储
    trpo_policy.init_storage(
        num_envs=1,
        num_transitions_per_env=200,
        actor_obs_shape=(3,),
        critic_obs_shape=(3,),
        action_shape=(1,)
    )

    # 创建环境
    env = gym.make('Pendulum-v1')
    obs, _ = env.reset()

    print("开始训练...")
    total_reward = 0

    for step in range(100):
        # 转换为tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # 执行动作
        action = trpo_policy.act(obs_tensor)

        # 与环境交互
        action_np = action.squeeze(0).detach().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        # 处理环境步骤
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.bool)
        trpo_policy.process_env_step(reward_tensor, done_tensor, {})

        total_reward += reward
        obs = next_obs

        if done:
            # 计算回报
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            trpo_policy.compute_returns(next_obs_tensor)

            # 执行更新
            trpo_policy.update()

            # 重置环境
            obs, _ = env.reset()
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            total_reward = 0

            # 重置RNN隐藏状态
            trpo_policy.reset()

    env.close()
    print("训练完成!")

if __name__ == "__main__":
    main()