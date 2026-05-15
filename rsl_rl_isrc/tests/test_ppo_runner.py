#!/usr/bin/env python3
"""本测试模块：自动化/冒烟验证 ``rsl_rl_isrc`` 组件（请结合 tests 下 README 运行）。


使用 OnPolicyRunner 的 PPO 训练测试
基于 rsl_rl 框架，使用正确的 PPO 类和 Runner 接口
"""

import torch
import numpy as np
import unittest
import tempfile
import os
import sys
from tqdm import tqdm

# 确保使用本地 rsl_rl_isrc 版本而不是系统安装的版本
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # 项目根目录 (含 rsl_rl_isrc 的目录)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 rsl_rl_isrc 模块
from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.runners import OnPolicyRunner


class DummyVecEnv(VecEnv):
    """简化的向量环境实现，用于测试"""

    def __init__(self, num_envs=4, max_episode_length=200):
        self.num_envs = num_envs
        self.num_obs = 4  # CartPole 观测维度
        self.num_privileged_obs = None
        self.num_actions = 2  # CartPole 动作维度
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 状态缓冲区
        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

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

        return self.obs_buf.clone(), None, rewards, terminated | truncated, {}

    def reset(self, env_ids=None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)

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


def run_ppo_with_runner(num_episodes=50, num_envs=4, device='cpu'):
    """
    使用 OnPolicyRunner 运行 PPO 训练
    """
    print("开始 PPO Runner 训练测试...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_episodes}")
    print(f"环境数量: {num_envs}")

    # 创建虚拟环境
    env = DummyVecEnv(num_envs=num_envs, max_episode_length=200)

    # 配置
    train_cfg = {
        "runner": {
            "experiment_name": "ppo_cartpole_runner_test",
            "num_steps_per_env": 100,  # 每个环境的步数
            "save_interval": 50
        },
        "algorithm": {
            "algorithm_class_name": "PPO",
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True
        },
        "policy": {
            "policy_class_name": "ActorCritic",
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "elu",
            "init_noise_std": 1.0
        }
    }

    # 创建临时日志目录
    with tempfile.TemporaryDirectory() as log_dir:
        # 创建 runner
        runner = OnPolicyRunner(
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
        'success': final_reward > 50  # 合理的性能阈值
    }


class TestPPORunner(unittest.TestCase):
    """PPO Runner 测试"""

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
        actions = torch.tensor([0, 1])  # 左、右动作
        next_obs, priv_obs, rewards, dones, infos = env.step(actions)
        self.assertEqual(next_obs.shape, (2, 4))
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertEqual(len(dones), 2)

    def test_basic_runner_training(self):
        """测试基本的 Runner 初始化和动作采样功能"""
        from rsl_rl_isrc.runners import OnPolicyRunner

        # 创建虚拟环境
        env = DummyVecEnv(num_envs=2, max_episode_length=10)

        # 简化的配置
        train_cfg = {
            "runner": {
                "experiment_name": "ppo_cartpole_test",
                "num_steps_per_env": 10,  # 很小的步数用于测试
                "save_interval": 50
            },
            "algorithm": {
                "algorithm_class_name": "PPO",
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
                "clip_param": 0.2,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 1e-3
            },
            "policy": {
                "policy_class_name": "ActorCritic",
                "actor_hidden_dims": [32, 32],
                "critic_hidden_dims": [32, 32],
                "activation": "elu",
                "init_noise_std": 1.0
            }
        }

        import tempfile
        with tempfile.TemporaryDirectory() as log_dir:
            # 创建 runner
            runner = OnPolicyRunner(
                env=env,
                train_cfg=train_cfg,
                log_dir=log_dir,
                device=self.device
            )

            # 测试基本功能
            obs = env.get_observations()
            actions = runner.get_inference_policy()(obs)

            # 验证动作形状
            self.assertEqual(actions.shape, (2, 2))  # 2个环境，2个动作

            # 测试保存/加载
            runner.save(os.path.join(log_dir, 'test_model.pt'))
            runner.load(os.path.join(log_dir, 'test_model.pt'))


if __name__ == '__main__':
    # 运行训练测试
    print("=" * 50)
    print("PPO Runner 完整训练测试")
    print("=" * 50)

    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行训练
    results = run_ppo_with_runner(
        num_episodes=100,  # 适当的训练轮数
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