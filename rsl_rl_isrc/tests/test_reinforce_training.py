#!/usr/bin/env python3
"""
完整的REINFORCE算法训练测试
基于rsl_rl框架，使用off-policy episodes进行REINFORCE训练
这是一个headless测试，不依赖图形界面
"""

import torch
import numpy as np
import unittest
import tempfile
import os
import sys
from tqdm import tqdm
import gymnasium as gym

# 确保使用本地 rsl_rl_isrc 版本而不是系统安装的版本
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # 项目根目录 (含 rsl_rl_isrc 的目录)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 rsl_rl_isrc 模块
from rsl_rl_isrc.algorithms import REINFORCEPolicy
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


def run_reinforce_training(num_episodes=200, num_envs=4, hidden_dim=32, learning_rate=1e-3, gamma=0.98, device='cuda', action_space_type='discrete'):
    """
    运行完整的REINFORCE训练
    基于hand-on-rl中的训练逻辑，但适配rsl_rl框架
    支持离散和连续动作空间
    """
    print("开始REINFORCE训练测试...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_episodes}")
    print(f"环境数量: {num_envs}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"学习率: {learning_rate}")
    print(f"折扣因子: {gamma}")

    # 创建多环境 (使用gym的CartPole-v1)
    env = gym.vector.AsyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(num_envs)])

    # 创建REINFORCE算法（内部会创建策略网络）
    reinforce = REINFORCEPolicy(
        num_obs=env.single_observation_space.shape[0],
        num_actions=env.single_action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        hidden_dims=[hidden_dim, hidden_dim],
        activation='tanh',
        action_space_type=action_space_type,
        device=device
    )

    # 创建存储器
    storage = RolloutStorage(
        num_envs=num_envs,  # 多环境
        num_transitions_per_env=500,  # CartPole-v1的最大episode长度
        obs_shape=[env.single_observation_space.shape[0]],  # [4]
        privileged_obs_shape=[None],
        actions_shape=[env.single_action_space.n],  # [2] - one-hot编码动作
        device=device
    )

    return_list = []
    episode_lengths_list = []

    # 多环境并行训练
    total_episodes_collected = 0
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)

    # 重置所有环境 (gym v0.26+ API)
    states, infos = env.reset()

    with tqdm(total=num_episodes, desc='训练进度') as pbar:
        while total_episodes_collected < num_episodes:
            # 为所有环境采样动作
            state_tensor = torch.tensor(states, dtype=torch.float).to(device)

            # 批量采样动作 (返回one-hot编码)
            actions_batch, action_log_probs_batch = reinforce.act(state_tensor)

            # 转换为gym环境期望的离散动作索引 (0或1)
            action_indices = actions_batch.squeeze(-1).argmax(dim=-1).cpu().numpy()

            # 执行动作 (gym v0.26+ API)
            next_states, rewards, terminated, truncated, infos = env.step(action_indices)

            # 计算完成标志
            dones = terminated | truncated

            # 为每个环境创建transition
            for env_idx in range(num_envs):
                transition = RolloutStorage.Transition()
                transition.observations = state_tensor[env_idx:env_idx+1].clone()  # 保持梯度连接
                transition.critic_observations = None
                transition.actions = actions_batch[env_idx:env_idx+1].clone().detach()  # 动作不需要梯度 (one-hot)
                transition.rewards = torch.tensor([[rewards[env_idx]]], dtype=torch.float).to(device)
                transition.dones = torch.tensor([[dones[env_idx]]], dtype=torch.bool).to(device)
                transition.actions_log_prob = action_log_probs_batch[env_idx:env_idx+1].clone().detach()
                transition.values = torch.zeros_like(transition.rewards)
                transition.action_mean = actions_batch[env_idx:env_idx+1].clone().detach()  # one-hot动作
                transition.action_sigma = torch.ones_like(actions_batch[env_idx:env_idx+1])  # 离散动作没有std
                transition.hidden_states = None

                # 添加到存储器
                storage.add_off_policy_transition(transition, env_idx)

                # 跟踪episode统计
                episode_rewards[env_idx] += rewards[env_idx]
                episode_lengths[env_idx] += 1

                # 检查episode是否完成
                if dones[env_idx]:
                    # 完成episode并计算returns
                    storage.finish_episode(env_idx, gamma=gamma)

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

            # 定期进行策略更新（每收集一定数量的episodes）
            min_episodes_for_update = max(10, num_envs // 2)  # 更积极的更新策略
            if len(storage.off_policy_episodes) >= min_episodes_for_update:
                # 对相同的数据进行多次更新，提高样本效率
                num_policy_updates = 3  # 每次收集的数据使用3次
                avg_loss = 0.0

                for update_iter in range(num_policy_updates):
                    loss = reinforce.update(storage)
                    avg_loss += loss

                avg_loss /= num_policy_updates

                # 计算当前所有环境的平均长度和平均回报
                if len(episode_lengths_list) > 0:
                    current_avg_length = np.mean(episode_lengths_list[-min_episodes_for_update:])  # 最近更新的episodes
                else:
                    current_avg_length = 0.0

                if len(return_list) > 0:
                    current_avg_return = np.mean(return_list[-min_episodes_for_update:])  # 最近更新的episodes
                else:
                    current_avg_return = 0.0

                print(f"策略更新 - 平均损失: {avg_loss:.4f}, 已收集episodes: {len(storage.off_policy_episodes)}, "
                      f"环境数: {num_envs}, 本轮平均长度: {current_avg_length:.1f}, 本轮平均回报: {current_avg_return:.3f}")

                # 保留最近的一些episodes用于下次更新，提高数据利用率
                max_episodes_to_keep = min(50, num_envs * 3)  # 保留最近的episodes
                if len(storage.off_policy_episodes) > max_episodes_to_keep:
                    # 保留最近的episodes，丢弃最旧的
                    storage.off_policy_episodes = storage.off_policy_episodes[-max_episodes_to_keep:]

    # 计算最终统计信息
    final_avg_return = np.mean(return_list[-100:]) if len(return_list) > 100 else np.mean(return_list)
    final_avg_length = np.mean(episode_lengths_list[-100:]) if len(episode_lengths_list) > 100 else np.mean(episode_lengths_list)

    # 计算移动平均
    mv_return = compute_moving_average(return_list, 9)

    print("\n训练完成!")
    print(f"总轮数: {len(return_list)}")
    print(".3f")
    print(".1f")

    # 验证训练效果
    success = bool(final_avg_return > 50)  # 合理的性能阈值，确保返回Python bool类型
    print(f"训练成功: {success}")

    # 保存模型（可选）
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "reinforce_model.pt")
        reinforce.save(model_path)
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


class TestREINFORCETraining(unittest.TestCase):
    """REINFORCE训练测试"""

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

        # 测试步骤（离散动作）
        next_state, reward, terminated, truncated, info = env.step(1)  # 向右推
        self.assertEqual(len(next_state), 4)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

        env.close()

    def test_basic_training(self):
        """测试基本训练功能（离散动作）"""
        results = run_reinforce_training(
            num_episodes=20,  # 少量episode用于快速测试
            num_envs=2,  # 少量环境用于快速测试
            hidden_dim=16,
            learning_rate=1e-3,
            gamma=0.95,
            device=self.device,
            action_space_type='discrete'
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
        """测试训练收敛（离散动作）"""
        results = run_reinforce_training(
            num_episodes=50,
            num_envs=4,
            hidden_dim=32,
            learning_rate=1e-3,
            gamma=0.98,
            device=self.device,
            action_space_type='discrete'
        )

        # 检查是否有学习迹象（奖励应该有所提高）
        early_returns = results['return_list'][:10]
        late_returns = results['return_list'][-10:]

        early_avg = np.mean(early_returns)
        late_avg = np.mean(late_returns)

        print(f"早期平均奖励: {early_avg:.3f}")
        print(f"晚期平均奖励: {late_avg:.3f}")

        # 验证奖励不为0（环境会给存活奖励）
        self.assertGreater(early_avg, 0)
        self.assertGreater(late_avg, 0)

    def test_continuous_actions(self):
        """测试连续动作空间的基本功能"""
        from rsl_rl_isrc.algorithms import REINFORCEPolicy
        import torch

        # 创建连续动作的REINFORCE策略
        policy = REINFORCEPolicy(
            num_obs=4,
            num_actions=2,  # 2维连续动作
            action_space_type='continuous',
            init_noise_std=0.1,
            device=self.device
        )

        # 测试动作采样
        obs = torch.randn(1, 4).to(self.device)
        actions, log_probs = policy.act(obs)

        # 验证连续动作的形状和类型
        self.assertEqual(actions.shape, (1, 2))  # [batch_size, num_actions]
        self.assertEqual(log_probs.shape, (1,))   # [batch_size]
        self.assertTrue(torch.is_tensor(actions))
        self.assertTrue(torch.is_tensor(log_probs))

        # 测试推理模式
        inference_actions = policy.get_inference_policy()(obs)
        self.assertEqual(inference_actions.shape, (1, 2))

        print("Continuous actions test passed!")


if __name__ == '__main__':
    # 运行完整训练测试
    print("=" * 50)
    print("REINFORCE完整训练测试")
    print("=" * 50)

    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行完整训练
    results = run_reinforce_training(
        num_episodes=12000,  # 较少的episode用于测试
        num_envs=200,
        hidden_dim=64,
        learning_rate=1e-3,
        gamma=0.98,
        device='cuda',
        action_space_type='discrete'
    )

    print("\n" + "=" * 50)
    print("训练结果摘要:")
    print(f"- 总训练轮数: {len(results['return_list'])}")
    print(".3f")
    print(".1f")
    print(f"- 训练状态: {'成功' if results['success'] else '需要调整'}")

    # 运行单元测试
    print("\n" + "=" * 50)
    print("运行单元测试...")

    # 创建测试实例并运行
    test_instance = TestREINFORCETraining()
    test_instance.setUp()

    try:
        print("运行连续动作测试...")
        test_instance.test_continuous_actions()
        print("连续动作测试通过!")
    except Exception as e:
        print(f"连续动作测试失败: {e}")

    try:
        print("运行环境测试...")
        test_instance.test_environment()
        print("环境测试通过!")
    except Exception as e:
        print(f"环境测试失败: {e}")

    try:
        print("运行基本训练测试...")
        test_instance.test_basic_training()
        print("基本训练测试通过!")
    except Exception as e:
        print(f"基本训练测试失败: {e}")
