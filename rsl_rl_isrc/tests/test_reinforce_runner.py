#!/usr/bin/env python3
"""
使用 REINFORCERunner 的 REINFORCE 训练测试
基于 rsl_rl 框架，使用 REINFORCEPolicy 和正确的 Runner 接口
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
from rsl_rl_isrc.runners import REINFORCERunner


class DummyVecEnv(VecEnv):
    """简化的向量环境实现，用于 REINFORCE 测试（CartPole-like）"""

    def __init__(self, num_envs=4, max_episode_length=200):
        self.num_envs = num_envs
        self.num_obs = 4  # CartPole 观测维度
        self.num_privileged_obs = None
        self.num_actions = 2  # CartPole 动作维度
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32)

        # 状态缓冲区
        self.obs_buf = torch.zeros(num_envs, self.num_obs)
        self.rew_buf = torch.zeros(num_envs)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32)

        self.device = torch.device('cpu')

        # 初始化环境
        self.reset(torch.arange(num_envs))

    def step(self, actions: torch.Tensor):
        """简化的环境步骤"""
        # 简化的 CartPole 物理模拟
        batch_size = actions.shape[0]

        # 获取当前状态
        cart_pos = self.obs_buf[:, 0]
        cart_vel = self.obs_buf[:, 1]
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]

        # 动作转换 (0=left, 1=right -> -1, +1)
        force = torch.where(actions == 0, -1.0, 1.0)

        # 简化的物理更新
        cart_vel += 0.1 * force
        cart_vel *= 0.99  # 摩擦
        cart_pos += cart_vel

        pole_acc = -9.8 * torch.sin(pole_angle) - 0.1 * cart_vel * torch.cos(pole_angle) + 0.1 * force * torch.cos(pole_angle)
        pole_vel += pole_acc
        pole_vel *= 0.99
        pole_angle += pole_vel

        # 更新观测
        self.obs_buf[:, 0] = cart_pos
        self.obs_buf[:, 1] = cart_vel
        self.obs_buf[:, 2] = pole_angle
        self.obs_buf[:, 3] = pole_vel

        # 计算奖励和终止条件
        rewards = torch.ones(batch_size)  # 存活奖励

        # 终止条件
        terminated = (
            (cart_pos < -2.4) | (cart_pos > 2.4) |
            (pole_angle < -0.2094) | (pole_angle > 0.2094) |
            (self.episode_length_buf >= self.max_episode_length)
        )
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
            # 随机初始状态
            self.obs_buf[env_id] = torch.tensor([
                np.random.uniform(-0.05, 0.05),  # cart_pos
                np.random.uniform(-0.05, 0.05),  # cart_vel
                np.random.uniform(-0.05, 0.05),  # pole_angle
                np.random.uniform(-0.05, 0.05),  # pole_vel
            ])
            self.episode_length_buf[env_id] = 0

        return self.obs_buf.clone()

    def get_observations(self):
        """获取观测"""
        return self.obs_buf.clone()

    def get_privileged_observations(self):
        """获取 privileged 观测"""
        return None


def run_reinforce_with_runner(num_episodes=50, num_envs=4, device='cpu'):
    """
    使用 REINFORCERunner 运行 REINFORCE 训练
    """
    print("开始 REINFORCE Runner 训练测试...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_episodes}")
    print(f"环境数量: {num_envs}")

    # 创建虚拟环境
    env = DummyVecEnv(num_envs=num_envs, max_episode_length=200)

    # 配置
    train_cfg = {
        "runner": {
            "experiment_name": "reinforce_cartpole_runner_test",
            "num_steps_per_env": 100,  # 每个环境的步数
            "save_interval": 50
        },
        "algorithm": {
            "learning_rate": 1e-3,
            "gamma": 0.98,
            "num_learning_epochs": 1
        },
        "policy": {
            "action_space_type": "discrete",
            "hidden_dims": [64, 64],
            "activation": "tanh"
        }
    }

    # 创建临时日志目录
    with tempfile.TemporaryDirectory() as log_dir:
        # 创建 runner
        runner = REINFORCERunner(
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
                actions, _ = runner.get_inference_policy()(obs)
                # 转换动作格式（REINFORCE 返回的 actions 是 one-hot，需要转换为索引）
                if actions.dim() > 1 and actions.shape[-1] == env.num_actions:
                    actions = actions.argmax(dim=-1)
                obs, _, rewards, dones, _ = env.step(actions)
                total_reward += rewards.mean().item()
                total_length += 1

        final_reward = total_reward / 10
        final_length = total_length / 10

    print(f"训练完成 - 平均奖励: {final_reward:.3f}, 平均长度: {final_length:.1f}")

    return {
        'final_avg_reward': final_reward,
        'final_avg_length': final_length,
        'success': final_reward > 30  # CartPole 的合理性能阈值
    }


class TestREINFORCERunner(unittest.TestCase):
    """REINFORCE Runner 测试"""

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
        self.assertEqual(obs.shape, (2, 4))

        # 测试步骤
        actions = torch.tensor([0, 1])  # 离散动作
        next_obs, priv_obs, rewards, dones, infos = env.step(actions)
        self.assertEqual(next_obs.shape, (2, 4))
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertEqual(len(dones), 2)

    def test_basic_runner_training(self):
        """测试基本的 Runner 训练功能"""
        results = run_reinforce_with_runner(
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
    print("REINFORCE Runner 完整训练测试")
    print("=" * 50)

    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行训练
    results = run_reinforce_with_runner(
        num_episodes=200,  # REINFORCE 训练轮数
        num_envs=8,
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