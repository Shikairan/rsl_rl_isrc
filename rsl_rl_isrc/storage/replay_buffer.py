# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""定容量 FIFO 经验回放，支撑 SAC 等 off-policy 采样。"""

import torch
import numpy as np


class ReplayBuffer:
    """
    定容量 FIFO 经验回放（off-policy，例如 SAC）。

    与 ``RolloutStorage`` 接口部分对齐，仅存储 ``(obs, next_obs, actions, rewards, dones)``；
    ``add`` 支持一次写入多个并行环境 transition，``sample`` 随机小批量供 critic/actor 更新。
    """

    class Transition:
        """Single transition (s, a, r, s', done)."""
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(self, buffer_size, obs_shape, action_shape, device='cpu', n_envs=1):
        """
        Args:
            buffer_size: Total number of transitions to store (capacity).
            obs_shape: Shape of a single observation, e.g. (obs_dim,) or (3, 84, 84).
            action_shape: Shape of a single action, e.g. (action_dim,).
            device: torch device for tensor storage.
            n_envs: Number of parallel envs; each add() call adds n_envs transitions.
        """
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.n_envs = n_envs

        self.observations = torch.zeros(buffer_size, *obs_shape, dtype=torch.float32, device=device)
        self.next_observations = torch.zeros(buffer_size, *obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, *action_shape, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)

        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, actions, rewards, dones):
        """
        Add transition(s) to the buffer. Supports vectorized (n_envs,) or single transition.

        Args:
            obs: (n_envs, *obs_shape) or (1, *obs_shape).
            next_obs: (n_envs, *obs_shape) or (1, *obs_shape).
            actions: (n_envs, *action_shape) or (1, *action_shape).
            rewards: (n_envs,) or (n_envs, 1) or scalar.
            dones: (n_envs,) or (n_envs, 1) or scalar.
        """
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(next_obs):
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(dones):
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        n = obs.shape[0]
        rewards = rewards.reshape(n, -1)
        dones = dones.reshape(n, -1)

        for i in range(n):
            idx = (self.pos + i) % self.buffer_size
            self.observations[idx].copy_(obs[i])
            self.next_observations[idx].copy_(next_obs[i])
            self.actions[idx].copy_(actions[i])
            self.rewards[idx].copy_(rewards[i].view(1))
            self.dones[idx].copy_(dones[i].view(1))

        old_pos = self.pos
        self.pos = (self.pos + n) % self.buffer_size
        if old_pos + n >= self.buffer_size or self.full:
            self.full = True

    def add_transition(self, transition: Transition):
        """Add a single transition (RolloutStorage-style interface)."""
        self.add(
            transition.observations,
            transition.next_observations,
            transition.actions,
            transition.rewards,
            transition.dones,
        )

    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly from the buffer.

        Returns:
            observations: (batch_size, *obs_shape)
            next_observations: (batch_size, *obs_shape)
            actions: (batch_size, *action_shape)
            rewards: (batch_size, 1)
            dones: (batch_size, 1)
        """
        size = self.buffer_size if self.full else self.pos
        if size == 0:
            return None
        batch_size = min(batch_size, size)
        indices = torch.randint(0, size, (batch_size,), device=self.device)

        observations = self.observations[indices]
        next_observations = self.next_observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        return observations, next_observations, actions, rewards, dones

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def clear(self):
        """Reset buffer (pos=0, full=False). Storage tensors unchanged."""
        self.pos = 0
        self.full = False
