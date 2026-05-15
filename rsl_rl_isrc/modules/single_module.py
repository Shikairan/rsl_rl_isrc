# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""单策略分支 MLP（及 RNN 变体），用于 REINFORCE 等仅策略场景。"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.nn.modules import rnn
from .actor_critic import get_activation
from rsl_rl_isrc.utils import unpad_trajectories


class SingleActor(nn.Module):
    """
    单神经网络模块，只包含策略网络
    支持离散和连续动作空间
    专门为REINFORCE算法设计，不包含价值函数
    """
    is_recurrent = False

    def __init__(self,
                 num_obs,
                 num_actions,
                 hidden_dims=[64, 64],
                 activation='elu',
                 action_space_type='discrete',
                 init_noise_std=1.0,
                 **kwargs):
        super(SingleActor, self).__init__()

        if kwargs:
            print("SingleActor.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.action_space_type = action_space_type

        activation_fn = get_activation(activation)

        # Policy network - 基于ActorCritic的设计，但只包含策略部分
        policy_layers = []
        policy_layers.append(nn.Linear(num_obs, hidden_dims[0]))
        policy_layers.append(activation_fn)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                policy_layers.append(nn.Linear(hidden_dims[l], num_actions))
            else:
                policy_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                policy_layers.append(activation_fn)
        self.actor = nn.Sequential(*policy_layers)

        # For continuous actions, add action noise parameter
        if action_space_type == 'continuous':
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            Normal.set_default_validate_args = False
            print(f"SingleActor (Continuous): {self.actor}, std: {self.std}")
        else:
            print(f"SingleActor (Discrete): {self.actor}")

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, observations):
        """Forward pass"""
        return self.actor(observations)

    def get_action_logits(self, observations):
        """Get action logits from observations"""
        return self.actor(observations)

    def get_action_distribution(self, observations):
        """Get action distribution"""
        if self.action_space_type == 'continuous':
            mean = self.actor(observations)
            return Normal(mean, mean*0. + self.std)
        else:
            logits = self.get_action_logits(observations)
            return Categorical(logits=logits)

    def act(self, observations):
        """
        Sample actions from policy
        For discrete: Returns: (actions_one_hot, log_probs)
        For continuous: Returns: (actions, log_probs)
        """
        distribution = self.get_action_distribution(observations)

        if self.action_space_type == 'continuous':
            # Sample continuous actions
            actions = distribution.sample()
            # Get log probabilities (sum over action dimensions)
            log_probs = distribution.log_prob(actions).sum(dim=-1)
            return actions, log_probs
        else:
            # Sample action indices
            action_indices = distribution.sample()

            # Convert to one-hot encoding
            actions_one_hot = torch.zeros_like(distribution.logits)
            actions_one_hot.scatter_(-1, action_indices.unsqueeze(-1), 1.0)

            # Get log probabilities
            log_probs = distribution.log_prob(action_indices)

            return actions_one_hot, log_probs

    def get_actions_log_prob(self, actions):
        """
        Get log probabilities for given actions
        Note: This method needs current observations, so it's context-dependent
        """
        raise NotImplementedError("get_actions_log_prob needs current observations")

    def evaluate_actions(self, observations, actions):
        """
        Evaluate log probabilities for given actions under current policy
        observations: current observations
        actions: one-hot encoded actions (discrete) or continuous actions (continuous)
        """
        distribution = self.get_action_distribution(observations)

        if self.action_space_type == 'continuous':
            # For continuous actions, actions are already the continuous values
            log_probs = distribution.log_prob(actions).sum(dim=-1)
        else:
            # For discrete actions, convert one-hot to indices
            action_indices = actions.argmax(dim=-1)
            log_probs = distribution.log_prob(action_indices)

        return log_probs

    def act_inference(self, observations):
        """
        Get deterministic actions for inference
        For discrete: Returns one-hot encoded actions
        For continuous: Returns mean actions
        """
        if self.action_space_type == 'continuous':
            # For continuous actions, return the mean
            return self.actor(observations)
        else:
            # For discrete actions, return argmax as one-hot
            logits = self.get_action_logits(observations)
            action_indices = logits.argmax(dim=-1)

            actions_one_hot = torch.zeros_like(logits)
            actions_one_hot.scatter_(-1, action_indices.unsqueeze(-1), 1.0)

            return actions_one_hot

    @property
    def entropy(self):
        """Get entropy of current distribution"""
        # This needs current observations, so it's context-dependent
        raise NotImplementedError("entropy needs current observations")

    def get_entropy(self, observations):
        """Get entropy for given observations"""
        distribution = self.get_action_distribution(observations)
        return distribution.entropy()


class SingleActorRecurrent(SingleActor):
    """
    带有RNN的单神经网络模块，只包含策略网络
    支持离散和连续动作空间
    专门为REINFORCE算法设计，不包含价值函数
    """
    is_recurrent = True

    def __init__(self,
                 num_obs,
                 num_actions,
                 hidden_dims=[64, 64],
                 activation='elu',
                 action_space_type='discrete',
                 init_noise_std=1.0,
                 rnn_type='lstm',
                 rnn_hidden_size=256,
                 rnn_num_layers=1,
                 **kwargs):

        if kwargs:
            print("SingleActorRecurrent.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        super().__init__(num_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         hidden_dims=hidden_dims,
                         activation=activation,
                         action_space_type=action_space_type,
                         init_noise_std=init_noise_std)

        self.memory = Memory(num_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory}")

    def reset(self, dones=None):
        self.memory.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate_actions(self, observations, actions):
        # For RNN, process the entire sequence at once
        # Reset hidden states for this evaluation
        self.memory.hidden_states = None

        # observations shape: [seq_len, obs_dim] or [seq_len, 1, obs_dim]
        # Squeeze any extra dimensions
        if observations.dim() == 3:
            observations = observations.squeeze(1)  # [seq_len, obs_dim]

        # Process the entire sequence through RNN
        # RNN expects [seq_len, batch_size, input_size] but we have [seq_len, input_size]
        # So we need to add batch dimension: [seq_len, 1, input_size]
        input_seq = observations.unsqueeze(1)  # [seq_len, 1, obs_dim]
        output_seq, _ = self.memory.rnn(input_seq)
        # output_seq shape: [seq_len, 1, rnn_hidden_size]
        input_a = output_seq.squeeze(1)  # [seq_len, rnn_hidden_size]

        # Evaluate log probs for the entire sequence
        return super().evaluate_actions(input_a, actions)

    def get_hidden_states(self):
        return self.memory.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)

            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states is a list with hidden_state and cell_state
        if dones is not None and self.hidden_states is not None:
            for hidden_state in self.hidden_states:
                hidden_state[..., dones, :] = 0.0
        else:
            # Initialize hidden states for inference
            self.hidden_states = None