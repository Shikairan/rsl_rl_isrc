#!/usr/bin/env python3
"""
使用 SACRunner 的 SAC 训练测试
基于 rsl_rl 框架，使用 SAC 算法和正确的 Runner 接口
"""

import torch
import numpy as np
import unittest
import tempfile
import os
import sys

# 确保使用本地 rsl_rl_isrc 版本而不是系统安装的版本
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # 项目根目录 (含 rsl_rl_isrc 的目录)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 rsl_rl_isrc 模块
from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.runners import SACRunner


class DummyVecEnv(VecEnv):
    """简化的连续控制向量环境实现，用于 SAC 测试（Pendulum-like）"""

    def __init__(self, num_envs=4, max_episode_length=200):
        self.num_envs = num_envs
        self.num_obs = 3  # Pendulum 观测维度 [cosθ, sinθ, θ̇]
        self.num_privileged_obs = None
        self.num_actions = 1  # Pendulum 连续动作维度
        self.max_episode_length = max_episode_length

        # 状态缓冲区
        self.obs_buf = torch.zeros(num_envs, self.num_obs)
        self.rew_buf = torch.zeros(num_envs)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32)

        self.device = torch.device('cpu')

        # 初始化环境
        self.reset(torch.arange(num_envs))

    def step(self, actions: torch.Tensor):
        """简化的 Pendulum 环境步骤"""
        batch_size = actions.shape[0]

        # 获取当前状态
        cos_theta = self.obs_buf[:, 0]
        sin_theta = self.obs_buf[:, 1]
        theta_dot = self.obs_buf[:, 2]

        # 重建角度
        theta = torch.atan2(sin_theta, cos_theta)

        # 动作裁剪到 [-2, 2]
        actions = torch.clamp(actions.squeeze(-1), -2.0, 2.0)

        # 简化的 Pendulum 物理
        g = 9.8
        m = 1.0
        l = 1.0
        dt = 0.05

        # 力矩 = 动作
        torque = actions

        # 角度加速度
        alpha = (3.0 * g / (2.0 * l)) * torch.sin(theta) + (3.0 / (m * l**2)) * torque

        # 更新状态
        theta_dot += alpha * dt
        theta_dot *= 0.99  # 阻尼
        theta += theta_dot * dt

        # 更新观测
        self.obs_buf[:, 0] = torch.cos(theta)  # cosθ
        self.obs_buf[:, 1] = torch.sin(theta)  # sinθ
        self.obs_buf[:, 2] = theta_dot         # θ̇

        # 计算奖励 (Pendulum 奖励)
        rewards = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * actions**2)

        # 终止条件（Pendulum 通常不终止）
        terminated = self.episode_length_buf >= self.max_episode_length
        truncated = torch.zeros(batch_size, dtype=torch.bool)

        # 更新 episode 长度
        self.episode_length_buf += 1

        # 处理终止的 episodes
        terminated_env_ids = torch.where(terminated)[0]
        if len(terminated_env_ids) > 0:
            self.reset(terminated_env_ids)

        return self.obs_buf.clone(), None, rewards, terminated, {}

    def reset(self, env_ids):
        """重置环境"""
        for env_id in env_ids:
            # 随机初始角度 [-π, π]
            theta = torch.tensor(np.random.uniform(-np.pi, np.pi))
            theta_dot = torch.tensor(np.random.uniform(-1, 1))

            self.obs_buf[env_id] = torch.tensor([
                torch.cos(theta),  # cosθ
                torch.sin(theta),  # sinθ
                theta_dot          # θ̇
            ])
            self.episode_length_buf[env_id] = 0

        return self.obs_buf.clone()

    def get_observations(self):
        """获取观测"""
        return self.obs_buf.clone()

    def get_privileged_observations(self):
        """获取 privileged 观测"""
        return None


def run_sac_with_runner(num_episodes=50, num_envs=4, device='cpu'):
    """
    使用 SACRunner 运行 SAC 训练
    """
    print("开始 SAC Runner 训练测试...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_episodes}")
    print(f"环境数量: {num_envs}")

    # 创建虚拟环境
    env = DummyVecEnv(num_envs=num_envs, max_episode_length=200)

    # 配置
    train_cfg = {
        "runner": {
            "experiment_name": "sac_pendulum_runner_test",
            "num_steps_per_env": 1,  # SAC 是 off-policy，每步更新
            "save_interval": 50
        },
        "algorithm": {
            "gamma": 0.99,
            "buffer_size": 10000,     # 小 buffer 用于测试
            "batch_size": 256,
            "learning_starts": 1000,  # 较早开始学习
            "update_frequency": 64,   # 每 64 步更新一次
            "num_updates_per_step": 1,
            "policy_lr": 3e-4,
            "q_lr": 1e-3,
            "alpha_lr": 1e-4,
            "action_bounds": [-2.0, 2.0]  # Pendulum 动作边界
        },
        "policy": {
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "relu"
        }
    }

    # 创建临时日志目录
    with tempfile.TemporaryDirectory() as log_dir:
        # 创建 runner
        runner = SACRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device
        )

        # 运行训练
        runner.learn(num_learning_iterations=num_episodes)

        # 获取最终结果
        final_reward = 0.0
        final_length = 0.0

        # 简单评估
        obs = env.reset(torch.arange(num_envs))
        total_reward = 0.0
        total_length = 0.0

        for _ in range(10):  # 10 个评估步骤
            with torch.no_grad():
                actions = runner.get_inference_policy()(obs)
                obs, _, rewards, dones, _ = env.step(actions)
                total_reward += rewards.mean().item()
                total_length += 1

        final_reward = total_reward / 10
        final_length = total_length / 10

    print(f"训练完成 - 平均奖励: {final_reward:.3f}, 平均长度: {final_length:.1f}")

    return {
        'final_avg_reward': final_reward,
        'final_avg_length': final_length,
        'success': final_reward > -500  # Pendulum 的合理性能阈值
    }


class TestSACRunner(unittest.TestCase):
    """SAC Runner 测试"""

    def setUp(self):
        """测试设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        np.random.seed(42)

    def test_dummy_env(self):
        """测试虚拟环境"""
        env = DummyVecEnv(num_envs=2)

        # 测试重置
        obs = env.reset(torch.tensor([0, 1]))
        self.assertEqual(obs.shape, (2, 3))

        # 测试步骤
        actions = torch.randn(2, 1)  # 连续动作
        next_obs, priv_obs, rewards, dones, infos = env.step(actions)
        self.assertEqual(next_obs.shape, (2, 3))
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertEqual(len(dones), 2)

    def test_basic_runner_training(self):
        """测试基本的 Runner 训练功能"""
        results = run_sac_with_runner(
            num_episodes=5,  # 少量 episode 用于快速测试
            num_envs=2,
            device=self.device
        )

        # 验证结果
        self.assertIsInstance(results['final_avg_reward'], (int, float))
        self.assertIsInstance(results['final_avg_length'], (int, float))
        self.assertIsInstance(results['success'], bool)

        # 验证数据合理性
        self.assertFalse(np.isnan(results['final_avg_reward']))
        self.assertFalse(np.isnan(results['final_avg_length']))


if __name__ == '__main__':
    # 运行训练测试
    print("=" * 50)
    print("SAC Runner 完整训练测试")
    print("=" * 50)

    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行训练
    results = run_sac_with_runner(
        num_episodes=200,  # SAC 训练轮数
        num_envs=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n" + "=" * 50)
    print("训练结果摘要:")
    print(f"- 最终平均奖励: {results['final_avg_reward']:.3f}")
    print(f"- 最终平均长度: {results['final_avg_length']:.1f}")
    print(f"- 训练状态: {'成功' if results['success'] else '需要调整'}")

    # 运行单元测试
    print("\n" + "=" * 50)
    print("运行单元测试...")
    unittest.main(verbosity=2, exit=False)