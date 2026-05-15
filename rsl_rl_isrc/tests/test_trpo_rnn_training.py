#!/usr/bin/env python3
"""本测试模块：自动化/冒烟验证 ``rsl_rl_isrc`` 组件（请结合 tests 下 README 运行）。


TRPO RNN训练测试 - 只有主程序部分
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
from rsl_rl_isrc.algorithms.trpo_policy import TRPOPolicy
from rsl_rl_isrc.storage import RolloutStorage


def compute_moving_average(data, window_size):
    """计算移动平均"""
    if len(data) < window_size:
        return np.array(data)
    # 使用numpy的convolve来计算移动平均
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def run_trpo_training(num_episodes=100, num_envs=4, hidden_dim=64, learning_rate=1e-3, gamma=0.995, device='cpu', episodes_per_update=10,
                      # RNN参数
                      rnn_hidden_size=64,   # 设置 > 0 启用RNN，设置为 0 禁用RNN
                      rnn_type='lstm',      # 'lstm' 或 'gru'
                      rnn_num_layers=1):    # RNN层数
    """
    运行真正的TRPO RNN训练
    使用KL约束和共轭梯度，支持连续动作空间，支持RNN
    """
    print("开始TRPO RNN训练测试...")
    print(f"设备: {device}")
    print(f"训练轮数: {num_episodes}")
    print(f"环境数量: {num_envs}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"学习率: {learning_rate}")
    print(f"折扣因子: {gamma}")
    print(f"RNN隐藏大小: {rnn_hidden_size}, 类型: {rnn_type}, 层数: {rnn_num_layers}")

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

    # 创建TRPO策略 - RNN版本
    action_bounds = (float(action_space.low[0]), float(action_space.high[0]))
    policy = TRPOPolicy(
        num_obs=num_obs,
        num_actions=num_actions,
        num_learning_epochs=1,
        num_mini_batches=1,
        gamma=gamma,
        tau=0.98,  # GAE时间窗口
        max_kl=0.05,      # KL约束
        damping=0.1,     # Fisher信息矩阵阻尼
        l2_reg=1e-4,      # L2正则化
        vf_lr=1e-2,       # ✅ 提高到1e-2，让价值函数能学习
        vf_iters=5 if rnn_hidden_size > 0 else 20,  # RNN:减少迭代次数节省内存
        action_bounds=action_bounds,  # ✅ 传递动作边界
        # RNN参数
        rnn_hidden_size=rnn_hidden_size,
        rnn_type=rnn_type,
        rnn_num_layers=rnn_num_layers,
        device=device
    )

    # 初始化存储 - TRPO配置：平衡的rollout长度
    rollout_length = 512 if rnn_hidden_size > 0 else 1024  # RNN:减少长度节省内存
    policy.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=rollout_length,
        actor_obs_shape=(num_obs,),
        critic_obs_shape=(num_obs,),
        action_shape=(num_actions,)
    )

    # 训练循环 - 标准TRPO：按固定步数收集rollout
    obs, _ = env.reset()
    policy.train_mode()

    # 计算需要多少次更新来达到目标episodes
    # 假设平均每个episode 200步，那么需要的总transitions
    total_transitions_needed = num_episodes * 200
    num_updates = total_transitions_needed // (rollout_length * num_envs)
    num_updates = max(num_updates, 1)  # 至少1次更新

    episode_rewards = []
    episode_lengths = []
    total_transitions_collected = 0

    # 为每个环境跟踪episode状态
    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs)

    with tqdm(total=num_updates, desc="TRPO RNN训练进度") as pbar:
        for update in range(num_updates):
            # 收集一个完整的rollout，跨episode边界
            for step in range(rollout_length):
                # 执行动作
                actions = policy.act(obs)

                # 环境步骤
                next_obs, rewards, dones, truncateds, infos = env.step(actions.cpu().numpy())
                # ✅ 保持原始奖励，不缩放，这样才能看到真实的回报值
                # 处理环境步骤
                policy.process_env_step(rewards, dones, infos, scale_factor=1.0)

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

            total_transitions_collected += rollout_length * num_envs

            # 计算回报并更新策略
            with torch.no_grad():
                policy.compute_returns(torch.tensor(obs, dtype=torch.float32, device=device))

            # 在test_trpo_rnn_training.py的update()调用前后添加
            print(f"更新前 - step: {policy.algorithm.storage.step}, rewards非零数: {(policy.algorithm.storage.rewards != 0).sum()}")
            value_loss, policy_loss = policy.update()
            print(f"更新后 - step: {policy.algorithm.storage.step}, rewards非零数: {(policy.algorithm.storage.rewards != 0).sum()}")

            #value_loss, policy_loss = policy.update()

            # 计算当前统计信息
            recent_episodes = episode_rewards[-min(10, len(episode_rewards)):]
            current_avg_reward = np.mean(recent_episodes) if recent_episodes else 0
            recent_lengths = episode_lengths[-min(10, len(episode_lengths)):]
            current_avg_length = np.mean(recent_lengths) if recent_lengths else 0

            # 更新进度条
            pbar.set_postfix({
                'update': update + 1,
                'episodes': len(episode_rewards),
                'avg_return': f'{current_avg_reward:.3f}',
                'avg_length': f'{current_avg_length:.1f}'
            })
            pbar.update(1)

            # 输出详细更新信息
            rnn_info = f" (RNN: {rnn_type}-{rnn_hidden_size})" if rnn_hidden_size > 0 else ""
            print(f"TRPO RNN更新{rnn_info} - 策略损失: {policy_loss:.4f}, 价值损失: {value_loss:.1f}, "
                  f"已收集episodes: {len(episode_rewards)}, 已收集transitions: {total_transitions_collected}, "
                  f"环境数: {num_envs}, 本轮平均长度: {current_avg_length:.1f}, 本轮平均回报: {current_avg_reward:.3f}")

            # 清空存储，为下一个rollout准备
            policy.algorithm.storage.clear()

            # RNN: 重置隐藏状态
            if rnn_hidden_size > 0:
                policy.reset(dones=None)

            # 内存清理
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            # 定期重置环境（防止episode过长）
            if (update + 1) % 10 == 0:
                obs, _ = env.reset()
                env_episode_rewards = np.zeros(num_envs)
                env_episode_lengths = np.zeros(num_envs)

                # RNN: 重置隐藏状态
                if rnn_hidden_size > 0:
                    policy.reset(dones=None)

    env.close()

    # 计算最终统计信息
    final_avg_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
    final_avg_length = np.mean(episode_lengths[-min(100, len(episode_lengths)):])
    reward_moving_avg = compute_moving_average(episode_rewards, min(20, len(episode_rewards)))

    print("\n训练完成!")
    print(f"总episode数: {len(episode_rewards)}")
    print(".3f")
    print(".1f")
    print(".3f")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_avg_reward': final_avg_reward,
        'final_avg_length': final_avg_length,
        'reward_moving_avg': reward_moving_avg,
        'success': final_avg_reward > -200,  # Pendulum-v1 的合理目标
        'rnn_config': {
            'rnn_hidden_size': rnn_hidden_size,
            'rnn_type': rnn_type,
            'rnn_num_layers': rnn_num_layers
        }
    }


if __name__ == '__main__':
    # 运行完整训练测试
    print("=== TRPO RNN 训练测试 ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    try:
        results = run_trpo_training(
            num_episodes=1000,  # 减少训练轮数用于快速测试
            num_envs=4,        # RNN:减少环境数量节省内存
            rnn_hidden_size=64,  # 启用RNN
            rnn_type='lstm',
            rnn_num_layers=1,
            device=device
        )

        print("\n=== 训练结果总结 ===")
        print(f"最终平均奖励: {results['final_avg_reward']:.3f}")
        print(f"最终平均长度: {results['final_avg_length']:.1f}")
        print(f"训练成功: {results['success']}")
        print(f"RNN配置: {results['rnn_config']}")

        if results['success']:
            print("🎉 TRPO RNN 成功学习了Pendulum控制任务!")
        else:
            print("⚠️  TRPO RNN 达到了基本性能，但可能需要更多训练")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()