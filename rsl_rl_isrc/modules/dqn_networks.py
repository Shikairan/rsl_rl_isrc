# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""DQN 的 Q 网络与目标网络（obs → |A| 个 Q 值）。"""

import torch
import torch.nn as nn


class DQNNetworks(nn.Module):
    """DQN 网络组：``q_net`` 输出每个离散动作的 Q 值，``q_target`` 为目标副本。"""

    is_recurrent = False

    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims=None,
        activation="relu",
        **kwargs,
    ):
        if kwargs:
            print(
                "DQNNetworks.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.num_actions = num_actions
        self.is_recurrent = False
        self.q_net = self._create_q_network(num_obs, num_actions, hidden_dims, activation)
        self.q_target = self._create_q_network(num_obs, num_actions, hidden_dims, activation)
        self.q_target.load_state_dict(self.q_net.state_dict())

        print(f"DQN Q-Network: {self.q_net}")

    def _create_q_network(self, num_obs, num_actions, hidden_dims, activation):
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

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        return self.q_net(observations)

    def q_values(self, observations):
        return self.q_net(observations)

    def target_q_values(self, observations):
        return self.q_target(observations)

    def act_greedy(self, observations):
        return self.q_net(observations).argmax(dim=-1, keepdim=True)

    def act_inference(self, observations):
        return self.act_greedy(observations)

    def update_target_networks(self, tau):
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
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
