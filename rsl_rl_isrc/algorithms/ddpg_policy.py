# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""DDPG（深度确定性策略梯度）：ReplayBuffer、确定性 Actor 与 Q 网络软更新。"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl_isrc.modules import DDPGNetworks
from rsl_rl_isrc.storage import ReplayBuffer


class DDPG:
    """DDPG 算法内核：经验回放、确定性策略梯度与目标网络 Polyak 更新。"""

    ddpg_networks: DDPGNetworks

    def __init__(
        self,
        ddpg_networks=None,
        num_obs=None,
        num_actions=None,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="relu",
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        buffer_size=int(1e6),
        batch_size=256,
        learning_starts=1000,
        update_frequency=1,
        num_updates_per_step=1,
        policy_frequency=1,
        target_network_frequency=1,
        noise_std=0.1,
        noise_clip=None,
        critic_grad_clip=True,
        critic_max_grad_norm=1.0,
        actor_grad_clip=True,
        actor_max_grad_norm=1.0,
        device="cpu",
        **kwargs,
    ):
        if kwargs:
            print(
                "DDPG.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        self.device = device

        if ddpg_networks is None:
            if num_obs is None or num_actions is None:
                raise ValueError("num_obs and num_actions must be provided when ddpg_networks is None")
            ddpg_networks = DDPGNetworks(
                num_obs=num_obs,
                num_actions=num_actions,
                actor_hidden_dims=actor_hidden_dims or [256, 256],
                critic_hidden_dims=critic_hidden_dims or [256, 256],
                activation=activation,
            )

        self.ddpg_networks = ddpg_networks
        self.ddpg_networks.to(self.device)
        self.storage = None

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_frequency = update_frequency
        self.num_updates_per_step = num_updates_per_step
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.critic_grad_clip = critic_grad_clip
        self.critic_max_grad_norm = critic_max_grad_norm
        self.actor_grad_clip = actor_grad_clip
        self.actor_max_grad_norm = actor_max_grad_norm

        self.actor_optimizer = optim.Adam(self.ddpg_networks.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.ddpg_networks.qf.parameters(), lr=self.critic_lr)

        self.global_step = 0

    def init_storage(self, num_envs, obs_shape, action_shape):
        self.storage = ReplayBuffer(
            buffer_size=self.buffer_size,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.device,
            n_envs=num_envs,
        )

    def test_mode(self):
        self.ddpg_networks.eval()

    def train_mode(self):
        self.ddpg_networks.train()

    def act(self, obs, explore=False, **kwargs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if self.global_step < self.learning_starts:
            return None

        actions = self.ddpg_networks.act(obs)
        if explore:
            actions = self._add_exploration_noise(actions)
        return actions

    def _add_exploration_noise(self, actions):
        noise = torch.randn_like(actions) * self.noise_std
        if self.noise_clip is not None:
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        noisy_actions = actions + noise
        low = self.ddpg_networks.action_bias - self.ddpg_networks.action_scale
        high = self.ddpg_networks.action_bias + self.ddpg_networks.action_scale
        return torch.max(torch.min(noisy_actions, high), low)

    def process_env_step(self, rewards, dones, infos, next_obs, obs=None, actions=None):
        if obs is None or actions is None:
            raise ValueError("process_env_step requires obs and actions.")
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(next_obs):
            next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(dones):
            dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        self.storage.add(obs, next_obs, actions.detach(), rewards, dones)
        self.global_step += obs.shape[0]

    def update(self):
        if self.global_step < self.learning_starts:
            return 0.0, 0.0
        if len(self.storage) < self.batch_size:
            return 0.0, 0.0
        if self.global_step % self.update_frequency != 0:
            return 0.0, 0.0

        final_critic_loss = 0.0
        final_actor_loss = 0.0

        for _ in range(self.num_updates_per_step):
            data = self.storage.sample(self.batch_size)
            if data is None:
                continue

            obs, next_obs, actions, rewards, dones = data

            with torch.no_grad():
                next_actions = self.ddpg_networks.act_target(next_obs)
                target_q = self.ddpg_networks.evaluate_target(next_obs, next_actions)
                target_value = rewards.flatten() + (1.0 - dones.float().flatten()) * self.gamma * target_q.flatten()

            current_q = self.ddpg_networks.evaluate(obs, actions).flatten()
            critic_loss = F.mse_loss(current_q, target_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.critic_grad_clip:
                nn.utils.clip_grad_norm_(self.ddpg_networks.qf.parameters(), self.critic_max_grad_norm)
            self.critic_optimizer.step()

            actor_loss = None
            if self.global_step % self.policy_frequency == 0:
                actor_actions = self.ddpg_networks.act(obs)
                actor_loss = -self.ddpg_networks.evaluate(obs, actor_actions).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.actor_grad_clip:
                    nn.utils.clip_grad_norm_(self.ddpg_networks.actor.parameters(), self.actor_max_grad_norm)
                self.actor_optimizer.step()

            if self.global_step % self.target_network_frequency == 0:
                self.ddpg_networks.update_target_networks(self.tau)

            final_critic_loss += critic_loss.item()
            if actor_loss is not None:
                final_actor_loss += actor_loss.item()

        num_updates = self.num_updates_per_step
        return final_critic_loss / num_updates, final_actor_loss / num_updates
