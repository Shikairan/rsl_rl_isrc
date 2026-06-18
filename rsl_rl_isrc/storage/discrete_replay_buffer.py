# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""离散动作经验回放，供 DQN 等 value-based off-policy 算法使用。"""

import torch


class DiscreteReplayBuffer:
    """定容量 FIFO 经验回放，动作为离散索引 ``long``。"""

    def __init__(self, buffer_size, obs_shape, device="cpu", n_envs=1):
        self.device = device
        self.obs_shape = obs_shape
        self.buffer_size = buffer_size
        self.n_envs = n_envs

        self.observations = torch.zeros(buffer_size, *obs_shape, dtype=torch.float32, device=device)
        self.next_observations = torch.zeros(buffer_size, *obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)

        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, actions, rewards, dones):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(next_obs):
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(dones):
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        actions = actions.reshape(-1).long()
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        n = obs.shape[0]

        for i in range(n):
            idx = (self.pos + i) % self.buffer_size
            self.observations[idx].copy_(obs[i])
            self.next_observations[idx].copy_(next_obs[i])
            self.actions[idx] = actions[i]
            self.rewards[idx].copy_(rewards[i])
            self.dones[idx].copy_(dones[i])

        old_pos = self.pos
        self.pos = (self.pos + n) % self.buffer_size
        if old_pos + n >= self.buffer_size or self.full:
            self.full = True

    def sample(self, batch_size):
        size = self.buffer_size if self.full else self.pos
        if size == 0:
            return None
        batch_size = min(batch_size, size)
        indices = torch.randint(0, size, (batch_size,), device=self.device)
        return (
            self.observations[indices],
            self.next_observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def clear(self):
        self.pos = 0
        self.full = False
