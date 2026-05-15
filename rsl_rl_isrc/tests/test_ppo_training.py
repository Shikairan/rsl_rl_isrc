#!/usr/bin/env python3
"""本测试模块：自动化/冒烟验证 ``rsl_rl_isrc`` 组件（请结合 tests 下 README 运行）。


真正的PPO (Proximal Policy Optimization) 训练测试
基于rsl_rl框架，支持离散动作空间
使用clip loss而不是TRPO的KL约束
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

import rsl_rl_isrc.isrcgym as gym

# 导入 rsl_rl_isrc 模块
from rsl_rl_isrc.modules import TrpoPolicy, TrpoValueFunction
from rsl_rl_isrc.algorithms import PPOPolicy
from rsl_rl_isrc.storage import RolloutStorage


def compute_moving_average(data, window_size):
    """计算移动平均"""
    if len(data) < window_size:
        return data
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(data[:window_size-1])[::2] / r
    end = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def run_ppo_training(num_episodes=200, num_envs=4, hidden_dim=32, learning_rate=3e-4, gamma=0.99, device='cuda'):
    """
    运行真正的PPO训练
    使用clip loss + GAE，支持离散动作空间
    """
    print("开始PPO训练测试...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_episodes}")
    print(f"环境数量: {num_envs}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"学习率: {learning_rate}")
    print(f"折扣因子: {gamma}")

    # 创建多环境 (使用gym的CartPole-v1)
    env = gym.vector.AsyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(num_envs)])

    # 创建真正的PPO算法
    ppo = PPOPolicy(
        num_obs=env.single_observation_space.shape[0],
        num_actions=env.single_action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        hidden_dims=[hidden_dim, hidden_dim],
        device=device
    )

    # 初始化存储器
    num_transitions_per_env = 2048  # PPO标准值
    ppo.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_transitions_per_env,
        actor_obs_shape=[env.single_observation_space.shape[0]],
        critic_obs_shape=[env.single_observation_space.shape[0]],
        action_shape=[env.single_action_space.n]
    )

    return_list = []
    episode_lengths_list = []

    # 多环境并行训练
    total_episodes_collected = 0
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)

    # 重置所有环境
    states, infos = env.reset()

    with tqdm(total=num_episodes, desc='训练进度') as pbar:
        while total_episodes_collected < num_episodes:
            # 为所有环境采样动作
            state_tensor = torch.tensor(states, dtype=torch.float).to(device)

            # PPO act方法返回: (actions_one_hot, log_probs, values)
            actions_batch, action_log_probs_batch, values_batch = ppo.act(state_tensor)

            # 转换为gym环境期望的离散动作索引
            action_indices = actions_batch.squeeze(-1).argmax(dim=-1).cpu().numpy()

            # 执行动作
            next_states, rewards, terminated, truncated, infos = env.step(action_indices)

            # 计算完成标志
            dones = terminated | truncated

            # 为所有环境创建transition并添加到存储器
            transition = RolloutStorage.Transition()
            transition.observations = state_tensor.clone()
            transition.critic_observations = state_tensor.clone()
            transition.actions = actions_batch.clone().detach()
            transition.rewards = torch.tensor(rewards, dtype=torch.float).to(device).unsqueeze(1)
            transition.dones = torch.tensor(dones, dtype=torch.bool).to(device).unsqueeze(1)
            transition.actions_log_prob = action_log_probs_batch.clone().detach().unsqueeze(1)
            transition.values = values_batch.clone().detach()
            transition.action_mean = actions_batch.clone().detach()  # one-hot动作
            transition.action_sigma = torch.ones_like(actions_batch)  # 离散动作没有std
            transition.hidden_states = None

            # 添加到存储器
            ppo.storage.add_transitions(transition)

            # 跟踪episode统计
            episode_rewards += rewards
            episode_lengths += 1

            # 检查哪些环境完成了episode
            for env_idx in range(num_envs):
                if dones[env_idx]:
                    # 记录结果
                    return_list.append(episode_rewards[env_idx])
                    episode_lengths_list.append(episode_lengths[env_idx])
                    total_episodes_collected += 1

                    pbar.set_postfix({
                        'episode': total_episodes_collected,
                        'return': f'{episode_rewards[env_idx]:.3f}',
                        'length': f'{episode_lengths[env_idx]:.1f}',
                        'avg_return': f'{np.mean(return_list[-10:]):.3f}' if len(return_list) >= 10 else f'{np.mean(return_list):.3f}'
                    })
                    pbar.update(1)

                    # 重置episode统计
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0

                    if total_episodes_collected >= num_episodes:
                        break

            # 更新状态
            states = next_states.copy()

            # 检查是否收集了足够的transitions进行策略更新
            if ppo.storage.step >= num_transitions_per_env:
                # 计算returns和advantages (GAE)
                with torch.no_grad():
                    last_values = ppo.value_function(state_tensor)
                ppo.storage.compute_returns(last_values, gamma, ppo.lam)

                # 进行PPO策略更新
                losses = ppo.update()

                # 计算当前统计信息
                if len(episode_lengths_list) > 0:
                    current_avg_length = np.mean(episode_lengths_list[-10:])
                else:
                    current_avg_length = 0.0

                if len(return_list) > 0:
                    current_avg_return = np.mean(return_list[-10:])
                else:
                    current_avg_return = 0.0

                print(f"策略更新 - 值损失: {losses['value_loss']:.4f}, 代理损失: {losses['surrogate_loss']:.4f}, "
                      f"熵损失: {losses['entropy_loss']:.4f}, 环境数: {num_envs}, "
                      f"本轮平均长度: {current_avg_length:.1f}, 本轮平均回报: {current_avg_return:.3f}")

    # 计算最终统计信息
    final_avg_return = np.mean(return_list[-100:]) if len(return_list) > 100 else np.mean(return_list)
    final_avg_length = np.mean(episode_lengths_list[-100:]) if len(episode_lengths_list) > 100 else np.mean(episode_lengths_list)

    # 计算移动平均
    mv_return = compute_moving_average(return_list, 9)

    print("\n训练完成!")
    print(f"总轮数: {len(return_list)}")
    print(f"最终平均奖励: {final_avg_return:.3f}")
    print(f"最终平均长度: {final_avg_length:.1f}")

    # 验证训练效果 (CartPole的解决标准是195分)
    success = bool(final_avg_return > 195)
    print(f"训练成功: {success}")

    # 保存模型
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "ppo_model.pt")
        ppo.save(model_path)
        print(f"模型已保存到: {model_path}")

    # 清理环境
    env.close()

    return {
        'return_list': return_list,
        'episode_lengths': episode_lengths_list,
        'moving_average': mv_return,
        'final_avg_return': final_avg_return,
        'final_avg_length': final_avg_length,
        'success': success
    }


class TestPPOTraining(unittest.TestCase):
    """PPO训练测试"""

    def setUp(self):
        """测试设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        np.random.seed(42)

    def test_environment(self):
        """测试gym CartPole环境"""
        env = gym.make('CartPole-v1')

        # 测试重置
        state, info = env.reset()
        self.assertEqual(len(state), 4)
        self.assertEqual(env.action_space.n, 2)  # 离散动作: 0或1

        # 测试步骤
        next_state, reward, terminated, truncated, info = env.step(1)  # 向右推
        self.assertEqual(len(next_state), 4)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

        env.close()

    def test_basic_training(self):
        """测试基本训练功能"""
        results = run_ppo_training(
            num_episodes=20,  # 少量episode用于快速测试
            num_envs=2,
            hidden_dim=32,
            learning_rate=3e-4,
            gamma=0.95,
            device=self.device
        )

        # 验证结果
        self.assertIsInstance(results['return_list'], list)
        self.assertIsInstance(results['episode_lengths'], list)
        self.assertIsInstance(results['moving_average'], np.ndarray)
        self.assertIsInstance(results['final_avg_return'], (int, float))
        self.assertIsInstance(results['final_avg_length'], (int, float))
        self.assertIsInstance(results['success'], bool)

        # 验证数据合理性
        self.assertGreater(len(results['return_list']), 0)
        self.assertGreater(len(results['episode_lengths']), 0)
        self.assertFalse(np.isnan(results['final_avg_return']))
        self.assertFalse(np.isnan(results['final_avg_length']))

    def test_training_convergence(self):
        """测试训练收敛"""
        results = run_ppo_training(
            num_episodes=50,
            num_envs=4,
            hidden_dim=32,
            learning_rate=3e-4,
            gamma=0.98,
            device=self.device
        )

        # 检查是否有学习迹象
        early_returns = results['return_list'][:10]
        late_returns = results['return_list'][-10:]

        early_avg = np.mean(early_returns)
        late_avg = np.mean(late_returns)

        print(f"早期平均奖励: {early_avg:.3f}")
        print(f"晚期平均奖励: {late_avg:.3f}")

        # 验证奖励不为0
        self.assertGreater(early_avg, 0)
        self.assertGreater(late_avg, 0)


if __name__ == '__main__':
    # 运行完整训练测试
    print("=" * 50)
    print("PPO完整训练测试")
    print("=" * 50)

    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行完整训练
    results = run_ppo_training(
        num_episodes=2000,  # 合理的训练轮数
        num_envs=16,        # 16个并行环境
        hidden_dim=64,
        learning_rate=3e-4,
        gamma=0.99,
        device='cuda'
    )

    print("\n" + "=" * 50)
    print("训练结果摘要:")
    print(f"- 总训练轮数: {len(results['return_list'])}")
    print(f"- 最终平均奖励: {results['final_avg_return']:.3f}")
    print(f"- 最终平均长度: {results['final_avg_length']:.1f}")
    print(f"- 训练状态: {'成功' if results['success'] else '需要调整'}")

    # 运行单元测试
    print("\n" + "=" * 50)
    print("运行单元测试...")
    unittest.main(verbosity=2, exit=False)
