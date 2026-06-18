# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""TD3（Twin Delayed DDPG）：双 Q、延迟 Actor 更新与目标策略平滑。"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl_isrc.modules import TD3Networks
from rsl_rl_isrc.storage import ReplayBuffer


class TD3:
    """TD3 算法内核：Clipped Double Q、延迟策略更新与目标策略平滑。"""

    td3_networks: TD3Networks

    def __init__(
        self,
        td3_networks=None,
        num_obs=None,
        num_actions=None,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="relu",
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        buffer_size=int(1e6),
        batch_size=256,
        learning_starts=1000,
        update_frequency=1,
        num_updates_per_step=1,
        policy_frequency=2,
        target_network_frequency=1,
        noise_std=0.1,
        noise_clip=None,
        target_noise_std=0.2,
        target_noise_clip=0.5,
        critic_grad_clip=True,
        critic_max_grad_norm=1.0,
        actor_grad_clip=True,
        actor_max_grad_norm=1.0,
        device="cpu",
        **kwargs,
    ):
        if kwargs:
            print(
                "TD3.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        self.device = device

        if td3_networks is None:
            if num_obs is None or num_actions is None:
                raise ValueError("num_obs and num_actions must be provided when td3_networks is None")
            td3_networks = TD3Networks(
                num_obs=num_obs,
                num_actions=num_actions,
                actor_hidden_dims=actor_hidden_dims or [256, 256],
                critic_hidden_dims=critic_hidden_dims or [256, 256],
                activation=activation,
            )

        self.td3_networks = td3_networks
        self.td3_networks.to(self.device)
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
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.critic_grad_clip = critic_grad_clip
        self.critic_max_grad_norm = critic_max_grad_norm
        self.actor_grad_clip = actor_grad_clip
        self.actor_max_grad_norm = actor_max_grad_norm

        critic_params = list(self.td3_networks.qf1.parameters()) + list(self.td3_networks.qf2.parameters())
        self.actor_optimizer = optim.Adam(self.td3_networks.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=self.critic_lr)

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
        self.td3_networks.eval()

    def train_mode(self):
        self.td3_networks.train()

    def act(self, obs, explore=False, **kwargs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if self.global_step < self.learning_starts:
            return None

        actions = self.td3_networks.act(obs)
        if explore:
            actions = self._add_exploration_noise(actions)
        return actions

    def _add_exploration_noise(self, actions):
        noise = torch.randn_like(actions) * self.noise_std
        if self.noise_clip is not None:
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        low = self.td3_networks.action_low()
        high = self.td3_networks.action_high()
        return torch.max(torch.min(actions + noise, high), low)

    def _smooth_target_actions(self, next_actions):
        noise = torch.randn_like(next_actions) * self.target_noise_std
        noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
        low = self.td3_networks.action_low()
        high = self.td3_networks.action_high()
        return torch.max(torch.min(next_actions + noise, high), low)

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
                next_actions = self.td3_networks.act_target(next_obs)
                next_actions = self._smooth_target_actions(next_actions)
                q1_target, q2_target = self.td3_networks.evaluate_target(next_obs, next_actions)
                target_q = torch.min(q1_target, q2_target).flatten()
                target_value = rewards.flatten() + (1.0 - dones.float().flatten()) * self.gamma * target_q

            q1, q2 = self.td3_networks.evaluate(obs, actions)
            critic_loss = F.mse_loss(q1.flatten(), target_value) + F.mse_loss(q2.flatten(), target_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.critic_grad_clip:
                critic_params = list(self.td3_networks.qf1.parameters()) + list(self.td3_networks.qf2.parameters())
                nn.utils.clip_grad_norm_(critic_params, self.critic_max_grad_norm)
            self.critic_optimizer.step()

            actor_loss = None
            if self.global_step % self.policy_frequency == 0:
                actor_actions = self.td3_networks.act(obs)
                actor_loss = -self.td3_networks.evaluate_q1(obs, actor_actions).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.actor_grad_clip:
                    nn.utils.clip_grad_norm_(self.td3_networks.actor.parameters(), self.actor_max_grad_norm)
                self.actor_optimizer.step()

            if self.global_step % self.target_network_frequency == 0:
                self.td3_networks.update_target_networks(self.tau)

            final_critic_loss += critic_loss.item()
            if actor_loss is not None:
                final_actor_loss += actor_loss.item()

        num_updates = self.num_updates_per_step
        return final_critic_loss / num_updates, final_actor_loss / num_updates
