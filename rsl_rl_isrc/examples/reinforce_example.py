#!/usr/bin/env python3
"""
本脚本：REINFORCE 示例配置与训练演示（依赖 ``rsl_rl_isrc`` 包）。

展示如何使用 REINFORCEPolicy 训练一个简单的离散动作任务（CartPole 模拟）。

运行方式::

    python -m rsl_rl_isrc.examples.reinforce_example
"""

import torch
import numpy as np
from rsl_rl_isrc.algorithms import REINFORCEPolicy
from rsl_rl_isrc.storage import RolloutStorage


def reinforce_example():
    """REINFORCE 算法完整示例（模拟 CartPole 离散动作）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 环境参数 ────────────────────────────────────────────────────────────
    num_envs    = 4
    num_obs     = 4   # CartPole 观测维度 [cart_pos, cart_vel, pole_angle, pole_vel]
    num_actions = 2   # 离散动作数量（0=向左, 1=向右）

    # ── 创建 REINFORCEPolicy（内部自动构建 SingleActor 网络）─────────────────
    policy = REINFORCEPolicy(
        num_obs=num_obs,
        num_actions=num_actions,
        num_learning_epochs=1,
        learning_rate=1e-3,
        gamma=0.99,
        hidden_dims=[64, 64],
        activation="tanh",
        action_space_type="discrete",
        device=device,
    )

    # ── 创建存储器（支持 off-policy episode 回合管理）──────────────────────
    storage = RolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=500,
        obs_shape=[num_obs],
        privileged_obs_shape=[None],
        actions_shape=[num_actions],
        device=device,
    )

    print("开始 REINFORCE 训练...")
    print(f"设备: {device}")
    print(f"环境数量: {num_envs}")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    for iteration in range(10):
        # 1. 数据收集：模拟多个 episode
        _collect_episodes(storage, policy, num_envs, num_obs, num_actions, device)

        # 2. 策略更新
        loss = policy.update(storage)
        print(f"迭代 {iteration + 1:2d}  |  episodes={len(storage.off_policy_episodes):3d}"
              f"  |  loss={loss:.4f}")

        # 3. 控制 episode 缓冲区大小
        if len(storage.off_policy_episodes) > 100:
            storage.clear_off_policy_episodes()

    print("\nREINFORCE 训练完成!")


def _collect_episodes(storage, policy, num_envs, num_obs, num_actions, device):
    """模拟 num_envs 个环境各完成一段随机 episode。"""
    for env_idx in range(num_envs):
        episode_length = np.random.randint(10, 50)
        obs = torch.randn(1, num_obs, device=device)

        for step in range(episode_length):
            # 采样动作（返回 one-hot 编码和 log_prob）
            with torch.no_grad():
                actions, log_probs = policy.act(obs)

            reward = torch.tensor([[1.0]], device=device)  # 存活奖励
            done   = torch.tensor([[step == episode_length - 1]], dtype=torch.bool, device=device)

            transition = RolloutStorage.Transition()
            transition.observations     = obs
            transition.critic_observations = obs
            transition.actions          = actions.detach()
            transition.rewards          = reward
            transition.dones            = done
            transition.actions_log_prob = log_probs.detach()
            transition.values           = torch.zeros(1, 1, device=device)
            transition.action_mean      = actions.detach()
            transition.action_sigma     = torch.ones_like(actions)
            transition.hidden_states    = None

            storage.add_off_policy_transition(transition, env_idx)
            obs = torch.randn(1, num_obs, device=device)  # 下一观测（随机模拟）

        storage.finish_episode(env_idx, gamma=policy.gamma)


if __name__ == "__main__":
    reinforce_example()
