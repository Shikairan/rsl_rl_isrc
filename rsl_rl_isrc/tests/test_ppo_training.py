#!/usr/bin/env python3
"""本测试模块：自动化/冒烟验证 ``rsl_rl_isrc`` 组件（请结合 tests 下 README 运行）。


PPO (Proximal Policy Optimization) 训练测试
基于 rsl_rl 框架，使用 PPO + ActorCritic + RolloutStorage 进行 on-policy 训练
"""

import torch
import numpy as np
import unittest
import tempfile
import os
import sys
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import rsl_rl_isrc.isrcgym as gym

from rsl_rl_isrc.algorithms import PPO
from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.storage import RolloutStorage


def compute_moving_average(data, window_size):
    """计算移动平均"""
    if len(data) < window_size:
        return np.array(data)
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def run_ppo_training(num_episodes=200, num_envs=4, hidden_dim=32,
                     learning_rate=3e-4, gamma=0.99, device='cpu'):
    """
    运行完整的 PPO 训练（CartPole-v1 离散动作空间）。

    PPO 是 on-policy 算法：收集固定长度 rollout → 计算 GAE 回报 → 多 epoch 小批量更新。
    """
    print("开始 PPO 训练测试...")
    print(f"设备: {device}  环境数: {num_envs}  隐藏层: {hidden_dim}  学习率: {learning_rate}")

    env = gym.vector.AsyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(num_envs)])

    num_obs     = env.single_observation_space.shape[0]   # 4
    num_actions = env.single_action_space.n               # 2

    # ── 构建 ActorCritic 网络 ─────────────────────────────────────────────
    actor_critic = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[hidden_dim, hidden_dim],
        critic_hidden_dims=[hidden_dim, hidden_dim],
        activation='tanh',
        init_noise_std=1.0,
    ).to(device)

    # ── 创建 PPO 算法 ─────────────────────────────────────────────────────
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=4,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=gamma,
        lam=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        learning_rate=learning_rate,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
        device=device,
    )

    # ── 初始化存储器 ──────────────────────────────────────────────────────
    num_steps_per_env = 128
    ppo.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_steps_per_env,
        actor_obs_shape=[num_obs],
        critic_obs_shape=[num_obs],
        action_shape=[num_actions],
    )

    return_list = []
    episode_lengths_list = []
    total_episodes_collected = 0
    env_ep_rewards  = np.zeros(num_envs)
    env_ep_lengths  = np.zeros(num_envs)

    states, _ = env.reset()
    actor_critic.train()

    with tqdm(total=num_episodes, desc='PPO训练') as pbar:
        while total_episodes_collected < num_episodes:
            # ── Rollout 收集阶段 ───────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(num_steps_per_env):
                    obs_t = torch.tensor(states, dtype=torch.float32, device=device)

                    # PPO.act 内部存储 transition（obs/actions/values/log_probs）
                    actions = ppo.act(obs_t, obs_t)
                    # ActorCritic 输出连续动作，argmax 取 0 或 1 作为离散索引
                    action_indices = actions.argmax(dim=-1).cpu().numpy().astype(int)

                    next_states, rewards, terminated, truncated, infos = env.step(action_indices)
                    dones = terminated | truncated

                    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                    dones_t   = torch.tensor(dones,   dtype=torch.bool,    device=device)
                    ppo.process_env_step(rewards_t, dones_t, infos)

                    env_ep_rewards += rewards
                    env_ep_lengths += 1
                    for eidx in range(num_envs):
                        if dones[eidx]:
                            return_list.append(env_ep_rewards[eidx])
                            episode_lengths_list.append(env_ep_lengths[eidx])
                            total_episodes_collected += 1
                            pbar.set_postfix({
                                'ep': total_episodes_collected,
                                'ret': f'{env_ep_rewards[eidx]:.1f}',
                                'avg': f'{np.mean(return_list[-10:]):.1f}' if len(return_list) >= 10 else '--'
                            })
                            pbar.update(1)
                            env_ep_rewards[eidx] = 0
                            env_ep_lengths[eidx] = 0
                    states = next_states

                    if total_episodes_collected >= num_episodes:
                        break

                # GAE 回报计算
                last_obs = torch.tensor(states, dtype=torch.float32, device=device)
                ppo.compute_returns(last_obs)

            # ── PPO 更新阶段 ───────────────────────────────────────────────
            value_loss, surrogate_loss = ppo.update()
            if len(return_list) > 0:
                print(f"  更新 | value={value_loss:.4f} surrogate={surrogate_loss:.4f}"
                      f" | 最近10回合均值={np.mean(return_list[-10:]):.1f}")

    final_avg_return = np.mean(return_list[-100:]) if len(return_list) > 100 else np.mean(return_list)
    final_avg_length = np.mean(episode_lengths_list[-100:]) if len(episode_lengths_list) > 100 else np.mean(episode_lengths_list)
    mv_return = compute_moving_average(return_list, min(9, len(return_list)))

    print(f"\n训练完成! 总回合={len(return_list)}"
          f" | 最终均值奖励={final_avg_return:.2f}")

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "ppo_model.pt")
        torch.save({'actor_critic': actor_critic.state_dict()}, model_path)

    env.close()
    return {
        'return_list':     return_list,
        'episode_lengths': episode_lengths_list,
        'moving_average':  mv_return,
        'final_avg_return': final_avg_return,
        'final_avg_length': final_avg_length,
        'success': bool(final_avg_return > 50),
    }


class TestPPOTraining(unittest.TestCase):
    """PPO 训练测试"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        np.random.seed(42)

    def test_environment(self):
        """测试 gym CartPole 环境"""
        env = gym.make('CartPole-v1')
        state, info = env.reset()
        self.assertEqual(len(state), 4)
        self.assertEqual(env.action_space.n, 2)
        next_state, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(len(next_state), 4)
        env.close()

    def test_basic_training(self):
        """测试基本 PPO 训练不崩溃"""
        results = run_ppo_training(
            num_episodes=20,
            num_envs=2,
            hidden_dim=16,
            learning_rate=3e-4,
            gamma=0.95,
            device=self.device,
        )
        self.assertIsInstance(results['return_list'], list)
        self.assertGreater(len(results['return_list']), 0)
        self.assertFalse(np.isnan(results['final_avg_return']))

    def test_training_convergence(self):
        """测试 PPO 有学习迹象"""
        results = run_ppo_training(
            num_episodes=50,
            num_envs=4,
            hidden_dim=32,
            learning_rate=3e-4,
            gamma=0.98,
            device=self.device,
        )
        early_avg = np.mean(results['return_list'][:10])
        late_avg  = np.mean(results['return_list'][-10:])
        print(f"早期均值: {early_avg:.2f}  晚期均值: {late_avg:.2f}")
        self.assertGreater(early_avg, 0)
        self.assertGreater(late_avg,  0)


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    results = run_ppo_training(
        num_episodes=2000, num_envs=16,
        hidden_dim=64, learning_rate=3e-4,
        gamma=0.99, device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"最终均值奖励: {results['final_avg_return']:.2f}")
    print(f"训练成功: {results['success']}")
    unittest.main(verbosity=2, exit=False)
