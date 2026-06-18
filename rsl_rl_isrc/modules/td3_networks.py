# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""TD3 的确定性 Actor 与双 Q 网络及目标网络。"""

import torch
import torch.nn as nn


class TD3Networks(nn.Module):
    """TD3 连续动作网络组：确定性 Actor μ(s) 与双 Q(s,a)，各含 target 副本。"""

    is_recurrent = False

    def __init__(
        self,
        num_obs,
        num_actions,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="relu",
        **kwargs,
    ):
        if kwargs:
            print(
                "TD3Networks.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256]

        self.is_recurrent = False
        self.actor = self._create_actor(num_obs, num_actions, actor_hidden_dims, activation)
        self.actor_target = self._create_actor(num_obs, num_actions, actor_hidden_dims, activation)
        self.qf1 = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation)
        self.qf2 = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation)
        self.qf1_target = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation)
        self.qf2_target = self._create_q_network(num_obs, num_actions, critic_hidden_dims, activation)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.register_buffer("action_scale", torch.ones(num_actions, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(num_actions, dtype=torch.float32))

        print(f"TD3 Actor MLP: {self.actor}")
        print(f"TD3 Q-Network 1: {self.qf1}")
        print(f"TD3 Q-Network 2: {self.qf2}")

    def _create_actor(self, num_obs, num_actions, hidden_dims, activation):
        layers = []
        layers.append(nn.Linear(num_obs, hidden_dims[0]))
        layers.append(get_activation(activation))
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], num_actions))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(get_activation(activation))
        return nn.Sequential(*layers)

    def _create_q_network(self, num_obs, num_actions, hidden_dims, activation):
        layers = []
        input_dim = num_obs + num_actions
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(get_activation(activation))
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], 1))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(get_activation(activation))
        return nn.Sequential(*layers)

    def set_action_bounds(self, action_low, action_high):
        if isinstance(action_low, torch.Tensor):
            scale = ((action_high - action_low) / 2.0).float()
            bias = ((action_high + action_low) / 2.0).float()
        else:
            scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
            bias = torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        self.action_scale.data.copy_(scale.to(self.action_scale.device))
        self.action_bias.data.copy_(bias.to(self.action_bias.device))

    def action_low(self):
        return self.action_bias - self.action_scale

    def action_high(self):
        return self.action_bias + self.action_scale

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _scale_action(self, raw_action):
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def act(self, observations, **kwargs):
        raw_action = self.actor(observations)
        return self._scale_action(raw_action)

    def act_target(self, observations):
        raw_action = self.actor_target(observations)
        return self._scale_action(raw_action)

    def act_inference(self, observations):
        return self.act(observations)

    def evaluate(self, observations, actions):
        state_action = torch.cat([observations, actions], dim=-1)
        return self.qf1(state_action), self.qf2(state_action)

    def evaluate_q1(self, observations, actions):
        return self.qf1(torch.cat([observations, actions], dim=-1))

    def evaluate_target(self, observations, actions):
        state_action = torch.cat([observations, actions], dim=-1)
        return self.qf1_target(state_action), self.qf2_target(state_action)

    def update_target_networks(self, tau):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def get_activation(act_name):
    _map = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "crelu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if act_name not in _map:
        raise ValueError(f"未知激活函数 '{act_name}'，支持: {list(_map.keys())}")
    return _map[act_name]
