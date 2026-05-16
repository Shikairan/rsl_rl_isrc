# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""SAC 的 Actor 与双 Critic 结构及动作缩放。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACNetworks(nn.Module):
    """SAC 连续动作网络组：含 Actor（高斯策略）与双 Q 网络；动作经 ``tanh``  squash 并按边界缩放。"""

    is_recurrent = False

    def __init__(self,
                 num_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256],
                 critic_hidden_dims=[256, 256],
                 activation='relu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("SACNetworks.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(SACNetworks, self).__init__()

        self.is_recurrent = False
        activation_fn = get_activation(activation)

        # Actor network (Policy)
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))  # mean output
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Log standard deviation layer for Actor
        self.actor_logstd = nn.Linear(actor_hidden_dims[-1], num_actions)

        # Q-networks (Critics)
        self.qf1 = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation_fn)
        self.qf2 = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation_fn)

        # Target Q-networks
        self.qf1_target = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation_fn)
        self.qf2_target = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation_fn)

        # Initialize target networks with same weights
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        print(f"Actor MLP: {self.actor}")
        print(f"Q-Network 1: {self.qf1}")
        print(f"Q-Network 2: {self.qf2}")

        # Action scaling (to be set by environment bounds)
        self.register_buffer("action_scale", torch.ones(num_actions, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(num_actions, dtype=torch.float32))

        # Distribution for sampling
        self.distribution = None
        Normal.set_default_validate_args = False

    def _create_q_network(self, num_obs, num_actions, hidden_dims, activation):
        """Create a Q-network that takes state and action as input."""
        layers = []
        input_dim = num_obs + num_actions  # concatenate state and action
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(get_activation(activation))
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], 1))  # Q-value output
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(get_activation(activation))
        return nn.Sequential(*layers)

    def set_action_bounds(self, action_low, action_high):
        """Set action bounds for scaling"""
        if isinstance(action_low, torch.Tensor):
            scale = ((action_high - action_low) / 2.0).float()
            bias = ((action_high + action_low) / 2.0).float()
        else:
            scale = torch.tensor(
                (action_high - action_low) / 2.0, dtype=torch.float32
            )
            bias = torch.tensor(
                (action_high + action_low) / 2.0, dtype=torch.float32
            )
        self.action_scale.data.copy_(scale)
        self.action_bias.data.copy_(bias)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """Update the policy distribution given observations"""
        x = self.actor[:-1](observations)  # Get hidden features before final layer
        mean = self.actor[-1](x)  # Final linear layer gives mean
        log_std = self.actor_logstd(x)

        # Clamp log_std
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        std = log_std.exp()
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """Sample action from policy"""
        self.update_distribution(observations)
        # Reparameterization trick
        x_t = self.distribution.rsample()  # Sample from normal distribution
        y_t = torch.tanh(x_t)  # Squash to [-1, 1]
        action = y_t * self.action_scale + self.action_bias  # Scale to action bounds
        return action

    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current policy"""
        # Convert actions back to squashed space
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + 1e-6, 1 - 1e-6)  # Clamp to avoid numerical issues

        x_t = torch.atanh(y_t)  # Inverse tanh

        log_prob = self.distribution.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return log_prob

    def act_inference(self, observations):
        """Get deterministic action for inference"""
        x = self.actor[:-1](observations)
        mean = self.actor[-1](x)
        # Scale mean to action bounds
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def evaluate(self, observations, actions):
        """Evaluate Q-values for state-action pairs"""
        # Q-networks take concatenated [obs, action] as input
        q1 = self.qf1(torch.cat([observations, actions], dim=-1))
        q2 = self.qf2(torch.cat([observations, actions], dim=-1))
        return q1, q2

    def evaluate_target(self, observations, actions):
        """Evaluate target Q-values for state-action pairs"""
        q1_target = self.qf1_target(torch.cat([observations, actions], dim=-1))
        q2_target = self.qf2_target(torch.cat([observations, actions], dim=-1))
        return q1_target, q2_target

    def update_target_networks(self, tau):
        """Update target networks using Polyak averaging"""
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def get_activation(act_name):
    """Get activation function by name"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return nn.ReLU()