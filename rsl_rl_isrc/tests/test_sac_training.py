#!/usr/bin/env python3
"""本测试模块：自动化/冒烟验证 ``rsl_rl_isrc`` 组件（请结合 tests 下 README 运行）。


SAC训练测试 - 只有主程序部分
"""

import torch
import numpy as np
from tqdm import tqdm

# 确保使用本地 rsl_rl_isrc 版本而不是系统安装的版本
script_dir = '/home/data/rl/ppo/pymotrisim_3_mujoco/rsl_rl_withALL/rsl_rl_isrc/tests'
project_root = '/home/data/rl/ppo/pymotrisim_3_mujoco/rsl_rl_withALL'
if project_root not in ['.', '']:
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import rsl_rl_isrc.isrcgym as gym

# 导入 rsl_rl_isrc 模块
from rsl_rl_isrc.algorithms.sac_policy import SAC
from rsl_rl_isrc.modules import SACNetworks
from rsl_rl_isrc.storage import ReplayBuffer


def compute_moving_average(data, window_size):
    """计算移动平均"""
    if len(data) == 0:
        return np.array([0.0])  # 返回默认值避免空数组错误
    if len(data) < window_size:
        return np.array(data)
    # 使用numpy的convolve来计算移动平均
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def run_sac_training(num_episodes=100, num_envs=4, hidden_dim=256, learning_rate=1e-4, gamma=0.99, device='cpu', buffer_size=int(1e6), batch_size=256, learning_starts=5000, update_frequency=256, num_updates_per_step=2, activation='relu'):
    """
    运行真正的SAC训练
    使用Soft Actor-Critic算法，支持连续动作空间
    """
    print("开始SAC训练测试...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_episodes}")
    print(f"环境数量: {num_envs}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"学习率: {learning_rate}")
    print(f"折扣因子: {gamma}")
    print(f"Replay Buffer大小: {buffer_size}")
    print(f"批次大小: {batch_size}")
    print(f"开始学习步数: {learning_starts}")
    print(f"更新频率: {update_frequency}")
    print(f"每次更新次数: {num_updates_per_step}")
    print(f"激活函数: {activation}")

    # 创建环境
    env = gym.vector.AsyncVectorEnv([lambda: gym.make('Pendulum-v1') for _ in range(num_envs)])

    # 获取环境信息
    obs_space = env.single_observation_space
    action_space = env.single_action_space
    num_obs = obs_space.shape[0]
    num_actions = action_space.shape[0]

    print(f"观察空间维度: {num_obs}")
    print(f"动作空间维度: {num_actions}")
    print(f"动作空间范围: {action_space.low} 到 {action_space.high}")

    # 创建SAC网络
    sac_networks = SACNetworks(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[hidden_dim, hidden_dim],
        critic_hidden_dims=[hidden_dim, hidden_dim],
        activation=activation
    )

    # 设置动作边界
    sac_networks.set_action_bounds(action_space.low, action_space.high)

    # 创建SAC策略
    policy = SAC(
        sac_networks=sac_networks,
        gamma=gamma,
        policy_lr=learning_rate,
        q_lr=1e-3,
        alpha_lr=1e-5,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        update_frequency=update_frequency,
        num_updates_per_step=num_updates_per_step,
        device=device,
        critic_grad_clip=True,
        critic_max_grad_norm=0.5,
        actor_grad_clip=True,
        actor_max_grad_norm=0.5
    )

    # 初始化存储 - SAC配置：使用ReplayBuffer
    policy.init_storage(
        num_envs=num_envs,
        obs_shape=(num_obs,),
        action_shape=(num_actions,)
    )

    # 训练循环 - SAC：持续收集transitions并更新
    obs, _ = env.reset()
    policy.train_mode()

    # 计算需要多少次更新来达到目标episodes
    # 假设平均每个episode 200步，那么需要的总transitions
    total_transitions_needed = num_episodes * 200
    max_steps = total_transitions_needed

    episode_rewards = []
    episode_lengths = []
    total_transitions_collected = 0

    # 为每个环境跟踪episode状态
    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs)

    with tqdm(total=max_steps, desc="SAC训练进度") as pbar:
        step = 0
        while step < max_steps:
            # 获取动作
            actions = policy.act(obs)

            # 如果还没开始学习，使用随机动作
            if actions is None:
                actions = torch.tensor(np.array([env.single_action_space.sample() for _ in range(num_envs)]), dtype=torch.float32, device=device)
            else:
                actions = actions.detach()

            # 环境步骤
            next_obs, rewards, dones, truncateds, infos = env.step(actions.cpu().numpy())

            # 处理环境步骤 - SAC方式：直接写入replay buffer
            policy.process_env_step(
                rewards=rewards,
                dones=dones,
                infos=infos,
                next_obs=next_obs,
                obs=obs,
                actions=actions
            )

            # 更新episode统计
            for env_idx in range(num_envs):
                env_episode_rewards[env_idx] += rewards[env_idx]
                env_episode_lengths[env_idx] += 1

                # 检查episode结束
                if dones[env_idx] or truncateds[env_idx]:
                    episode_rewards.append(env_episode_rewards[env_idx])
                    episode_lengths.append(env_episode_lengths[env_idx])

                    # 重置episode统计
                    env_episode_rewards[env_idx] = 0
                    env_episode_lengths[env_idx] = 0

            obs = next_obs
            step += num_envs
            total_transitions_collected += num_envs

            # SAC更新 - 每次都尝试更新（内部会检查是否满足学习条件）
            qf1_loss, qf2_loss, actor_loss, alpha_loss = 0.0, 0.0, 0.0, 0.0
            update_performed = False
            if step >= learning_starts:
                qf1_loss, qf2_loss, actor_loss, alpha_loss = policy.update()
                update_performed = qf1_loss != 0.0 or qf2_loss != 0.0

            # 计算当前统计信息
            recent_episodes = episode_rewards[-min(10, len(episode_rewards)):]
            current_avg_reward = np.mean(recent_episodes) if recent_episodes else 0
            recent_lengths = episode_lengths[-min(10, len(episode_lengths)):]
            current_avg_length = np.mean(recent_lengths) if recent_lengths else 0

            # 更新进度条
            pbar.set_postfix({
                'step': step,
                'episodes': len(episode_rewards),
                'avg_return': f'{current_avg_reward:.3f}',
                'avg_length': f'{current_avg_length:.1f}'
            })
            pbar.update(num_envs)

            # 输出详细更新信息 - 只有当实际执行了更新时才打印
            if update_performed:
                print(f"SAC更新 - 步数: {step}, Q损失: {qf1_loss:.4f}/{qf2_loss:.4f}, "
                      f"策略损失: {actor_loss:.4f}, Alpha损失: {alpha_loss:.4f}, "
                      f"已收集episodes: {len(episode_rewards)}, 已收集transitions: {total_transitions_collected}, "
                      f"环境数: {num_envs}, 本轮平均长度: {current_avg_length:.1f}, 本轮平均回报: {current_avg_reward:.3f}")

            # 定期重置环境（防止episode过长）
            if step % 10000 == 0:
                obs, _ = env.reset()
                env_episode_rewards = np.zeros(num_envs)
                env_episode_lengths = np.zeros(num_envs)

    env.close()

    # 计算最终统计信息
    final_avg_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
    final_avg_length = np.mean(episode_lengths[-min(100, len(episode_lengths)):])
    reward_moving_avg = compute_moving_average(episode_rewards, min(20, len(episode_rewards)))

    print("\n训练完成!")
    print(f"总episode数: {len(episode_rewards)}")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_avg_reward': final_avg_reward,
        'final_avg_length': final_avg_length,
        'reward_moving_avg': reward_moving_avg,
        'success': final_avg_reward > -200  # Pendulum-v1 的合理目标
    }


if __name__ == '__main__':
    # 运行完整训练测试
    print("=== SAC 训练测试 ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    try:
        results = run_sac_training(
            num_episodes=10000,  # 减少训练轮数用于快速测试
            num_envs=8,  # 减少环境数量以节省内存
            device=device,
            buffer_size=int(1e5),  # 较小的buffer用于测试
            batch_size=128,  # 较小的batch size
            learning_starts=1000,  # 较早开始学习
            update_frequency=256,  # 每256个transitions更新一次
            num_updates_per_step=3  # 每次更新执行2次梯度更新
        )

        print("\n=== 训练结果总结 ===")
        print(f"最终平均奖励: {results['final_avg_reward']:.3f}")
        print(f"最终平均长度: {results['final_avg_length']:.1f}")
        print(f"训练成功: {results['success']}")

        if results['success']:
            print("SAC 成功学习了Pendulum控制任务!")
        else:
            print("SAC 达到了基本性能，但可能需要更多训练")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
