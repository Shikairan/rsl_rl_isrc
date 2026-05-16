# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""REINFORCE（策略梯度）：与 ``RolloutStorage`` 协作的 on-policy episode 学习。"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from rsl_rl_isrc.modules import SingleActor, SingleActorRecurrent
from rsl_rl_isrc.storage import RolloutStorage
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class REINFORCEPolicy:
    """
    专门为REINFORCE算法实现的策略梯度方法
    使用SingleActor网络，支持离散和连续动作空间
    仿照PPO的架构，使用独立的策略网络模块
    """

    def __init__(self,
                 num_obs,
                 num_actions,
                 num_learning_epochs=1,
                 learning_rate=1e-4,  # 降低学习率
                 gamma=0.99,
                 hidden_dims=[64, 64],
                 activation='elu',
                 action_space_type='discrete',  # 'discrete' or 'continuous'
                 init_noise_std=1.0,  # for continuous actions
                 rnn_hidden_size=0,  # > 0 to enable RNN
                 rnn_type='lstm',    # 'lstm' or 'gru'
                 rnn_num_layers=1,   # Number of RNN layers
                 device='cpu',
                 **kwargs):
        if kwargs:
            print("REINFORCEPolicy.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        self.device = device
        self.num_learning_epochs = num_learning_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.action_space_type = action_space_type

        # 检查是否使用RNN（通过检查rnn_hidden_size是否大于0）
        use_rnn = rnn_hidden_size > 0

        if use_rnn:
            # 创建SingleActorRecurrent网络（带有RNN）
            self.actor = SingleActorRecurrent(
                num_obs=num_obs,
                num_actions=num_actions,
                hidden_dims=hidden_dims,
                activation=activation,
                action_space_type=action_space_type,
                init_noise_std=init_noise_std,
                rnn_type=rnn_type,
                rnn_hidden_size=rnn_hidden_size,
                rnn_num_layers=rnn_num_layers
            ).to(device)
        else:
            # 创建SingleActor网络（只包含策略网络）
            self.actor = SingleActor(
                num_obs=num_obs,
                num_actions=num_actions,
                hidden_dims=hidden_dims,
                activation=activation,
                action_space_type=action_space_type,
                init_noise_std=init_noise_std
            ).to(device)

        self.actor.train()  # set to train mode

        # 初始化优化器
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Learning rate scheduler (optional)
        self.scheduler = None

        # For distributed training
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.world_size > 1:
                # device_ids must be a list of ints (GPU indices) for CUDA; None for CPU
                _dev = torch.device(self.device)
                _device_ids = [_dev.index if _dev.index is not None else 0] if _dev.type == 'cuda' else None
                self.actor = DDP(self.actor, device_ids=_device_ids)
        else:
            self.world_size = 1
            self.rank = 0

    def act(self, observations, masks=None, hidden_states=None):
        """Sample action from policy"""
        with torch.no_grad():
            if hasattr(self.actor, 'is_recurrent') and self.actor.is_recurrent:
                actions, log_probs = self.actor.act(observations, masks, hidden_states)
            else:
                actions, log_probs = self.actor.act(observations)
            return actions, log_probs

    def update(self, storage: RolloutStorage):
        """Update policy using REINFORCE algorithm"""
        # Get episodes for training
        episodes = storage.get_off_policy_episodes(batch_size=None)

        if len(episodes) == 0:
            return 0.0  # No episodes to train on

        mean_loss = 0.0
        num_episodes = len(episodes)

        for _ in range(self.num_learning_epochs):
            epoch_loss = 0.0

            # Collect all episode data for batch processing
            all_observations = []
            all_actions = []
            all_returns = []

            for episode in episodes:
                # Convert episode data to tensors
                episode_data = episode.to_tensors(self.device)

                # Get observations and actions for this episode
                observations = episode_data['observations']
                if self.action_space_type == 'continuous':
                    actions = episode_data['actions']  # [seq_len, num_actions] - continuous actions
                else:
                    actions = episode_data['actions'].squeeze(-1)  # Remove extra dimension: [seq_len, num_actions] -> [seq_len, num_actions] for one-hot
                returns = episode_data['returns'].squeeze(-1).detach()  # [seq_len] - detach to avoid double gradients

                all_observations.append(observations)
                all_actions.append(actions)
                all_returns.append(returns)

            # Concatenate all episode data into a single batch
            batch_observations = torch.cat(all_observations, dim=0)
            batch_actions = torch.cat(all_actions, dim=0)
            batch_returns = torch.cat(all_returns, dim=0)

            # Clear gradients once for the entire batch
            self.optimizer.zero_grad()

            # Re-compute log probabilities through the current policy for the entire batch
            batch_log_probs = self.actor.evaluate_actions(batch_observations, batch_actions)

            # REINFORCE update: mean(-log_prob * discounted_return) over all steps in all episodes
            # This gives proper batch gradients instead of episode-wise updates
            step_losses = -(batch_log_probs * batch_returns)
            loss = step_losses.mean()  # Average over all steps in the batch
            epoch_loss = loss.item()

            # Backward pass and optimization for the entire batch
            loss.backward()
            self.optimizer.step()

            mean_loss += epoch_loss

        mean_loss /= self.num_learning_epochs

        # Step learning rate scheduler if available
        if self.scheduler is not None:
            self.scheduler.step()

        return mean_loss

    def update_lr(self, lr):
        """Update learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.learning_rate = lr

    def save(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'action_space_type': self.action_space_type,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.action_space_type = checkpoint.get('action_space_type', self.action_space_type)

        # Update optimizer learning rate
        self.update_lr(self.learning_rate)

    def reset(self, dones=None):
        """Reset RNN hidden states"""
        if hasattr(self.actor, 'reset'):
            self.actor.reset(dones)

    def get_hidden_states(self):
        """Get RNN hidden states"""
        if hasattr(self.actor, 'get_hidden_states'):
            return self.actor.get_hidden_states()
        return None

    def get_inference_policy(self, device=None):
        """Get inference policy"""
        self.actor.eval()  # switch to evaluation mode
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference