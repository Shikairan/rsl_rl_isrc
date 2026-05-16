#!/usr/bin/env python3
"""
本脚本：TRPO 示例配置与训练入口演示（依赖 ``rsl_rl_isrc`` 包）。

展示如何使用 TRPOPolicy 训练一个连续动作任务（Pendulum 模拟）。

运行方式::

    python -m rsl_rl_isrc.examples.trpo_example
"""

import torch
import numpy as np
from rsl_rl_isrc.algorithms import TRPOPolicy
from rsl_rl_isrc.storage import RolloutStorage


def trpo_example():
    """TRPO 算法完整示例（模拟 Pendulum 连续动作任务）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 环境参数 ────────────────────────────────────────────────────────────
    num_envs    = 4
    num_obs     = 3   # Pendulum: [cos θ, sin θ, θ̇]
    num_actions = 1   # 连续力矩控制
    rollout_len = 128  # 每次收集的步数

    # ── 创建 TRPOPolicy（内部构建策略网络与价值网络）──────────────────────
    policy = TRPOPolicy(
        num_obs=num_obs,
        num_actions=num_actions,
        num_learning_epochs=1,
        num_mini_batches=1,
        gamma=0.99,
        tau=0.97,
        max_kl=0.05,       # KL 约束
        damping=0.1,       # Fisher 信息矩阵阻尼
        l2_reg=1e-4,
        vf_lr=1e-2,        # 价值函数学习率
        vf_iters=20,       # 价值函数迭代次数
        action_bounds=(-2.0, 2.0),  # Pendulum 动作范围
        device=device,
    )

    # ── 初始化存储器 ─────────────────────────────────────────────────────────
    policy.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=rollout_len,
        actor_obs_shape=(num_obs,),
        critic_obs_shape=(num_obs,),
        action_shape=(num_actions,),
    )
    policy.train_mode()

    print("开始 TRPO 训练...")
    print(f"设备: {device}")
    print(f"环境数量: {num_envs}")
    print(f"每次收集步数: {rollout_len}")
    print(f"KL 约束: {policy.max_kl}")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    obs = torch.randn(num_envs, num_obs, device=device)

    for iteration in range(5):
        # 1. 收集 rollout（on-policy）
        for _ in range(rollout_len):
            actions = policy.act(obs)
            next_obs = torch.randn(num_envs, num_obs, device=device)  # 模拟下一状态
            rewards  = -torch.sum(obs ** 2, dim=-1)  # 模拟 Pendulum 奖励（负二阶矩）
            dones    = torch.zeros(num_envs, dtype=torch.bool, device=device)
            policy.process_env_step(rewards, dones, {})
            obs = next_obs

        # 2. 计算 GAE 回报
        policy.compute_returns(obs)

        # 3. TRPO 更新（价值函数 + 策略网络）
        value_loss, policy_loss = policy.update()
        print(f"迭代 {iteration + 1:2d}  |  value_loss={value_loss:.4f}"
              f"  |  policy_loss={policy_loss:.4f}")

        # storage 在 update() 内部已清空，下一轮直接收集

    print("\nTRPO 训练完成!")


if __name__ == "__main__":
    trpo_example()
