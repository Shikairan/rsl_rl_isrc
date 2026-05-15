# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY EXPRESS OR INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity replay buffer for off-policy RL (e.g. SAC).
    Interface aligned with RolloutStorage where applicable; only (obs, next_obs, actions, rewards, dones).
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
