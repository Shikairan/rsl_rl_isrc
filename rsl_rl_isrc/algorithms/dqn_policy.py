# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""DQN（Deep Q-Network）：离散动作、经验回放、目标网络与 ε-greedy 探索。"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl_isrc.modules import DQNNetworks
from rsl_rl_isrc.storage.discrete_replay_buffer import DiscreteReplayBuffer


class DQN:
    """DQN 算法内核：ε-greedy 采集、TD 目标与目标网络软更新。"""

    dqn_networks: DQNNetworks

    def __init__(
        self,
        dqn_networks=None,
        num_obs=None,
        num_actions=None,
        hidden_dims=None,
        activation="relu",
        gamma=0.99,
        tau=0.005,
        learning_rate=1e-3,
        buffer_size=int(1e6),
        batch_size=64,
        learning_starts=1000,
        update_frequency=1,
        num_updates_per_step=1,
        target_network_frequency=1,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10000,
        double_dqn=False,
        max_grad_norm=10.0,
        device="cpu",
        **kwargs,
    ):
        if kwargs:
            print(
                "DQN.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        self.device = device

        if dqn_networks is None:
            if num_obs is None or num_actions is None:
                raise ValueError("num_obs and num_actions must be provided when dqn_networks is None")
            dqn_networks = DQNNetworks(
                num_obs=num_obs,
                num_actions=num_actions,
                hidden_dims=hidden_dims or [256, 256],
                activation=activation,
            )

        self.dqn_networks = dqn_networks
        self.dqn_networks.to(self.device)
        self.storage = None
        self.num_actions = dqn_networks.num_actions

        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_frequency = update_frequency
        self.num_updates_per_step = num_updates_per_step
        self.target_network_frequency = target_network_frequency
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(int(epsilon_decay), 1)
        self.double_dqn = double_dqn
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.dqn_networks.q_net.parameters(), lr=self.learning_rate)
        self.global_step = 0

    def init_storage(self, num_envs, obs_shape):
        self.storage = DiscreteReplayBuffer(
            buffer_size=self.buffer_size,
            obs_shape=obs_shape,
            device=self.device,
            n_envs=num_envs,
        )

    def current_epsilon(self):
        if self.global_step >= self.epsilon_decay:
            return self.epsilon_end
        progress = self.global_step / self.epsilon_decay
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def test_mode(self):
        self.dqn_networks.eval()

    def train_mode(self):
        self.dqn_networks.train()

    def act(self, obs, explore=False, **kwargs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if explore:
            if self.global_step < self.learning_starts:
                return torch.randint(0, self.num_actions, (obs.shape[0], 1), device=self.device)
            if random.random() < self.current_epsilon():
                return torch.randint(0, self.num_actions, (obs.shape[0], 1), device=self.device)

        with torch.no_grad():
            return self.dqn_networks.act_greedy(obs)

    def process_env_step(self, rewards, dones, infos, next_obs, obs=None, actions=None):
        if obs is None or actions is None:
            raise ValueError("process_env_step requires obs and actions.")
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        if not torch.is_tensor(next_obs):
            next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(dones):
            dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        self.storage.add(obs, next_obs, actions, rewards, dones)
        self.global_step += obs.shape[0]

    def update(self):
        if self.global_step < self.learning_starts:
            return 0.0
        if len(self.storage) < self.batch_size:
            return 0.0
        if self.global_step % self.update_frequency != 0:
            return 0.0

        final_loss = 0.0
        for _ in range(self.num_updates_per_step):
            data = self.storage.sample(self.batch_size)
            if data is None:
                continue

            obs, next_obs, actions, rewards, dones = data

            with torch.no_grad():
                if self.double_dqn:
                    next_actions = self.dqn_networks.q_values(next_obs).argmax(dim=1, keepdim=True)
                    next_q = self.dqn_networks.target_q_values(next_obs).gather(1, next_actions).squeeze(1)
                else:
                    next_q = self.dqn_networks.target_q_values(next_obs).max(dim=1).values
                target_q = rewards.squeeze(1) + (1.0 - dones.squeeze(1)) * self.gamma * next_q

            current_q = self.dqn_networks.q_values(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.dqn_networks.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.global_step % self.target_network_frequency == 0:
                self.dqn_networks.update_target_networks(self.tau)

            final_loss += loss.item()

        return final_loss / self.num_updates_per_step
