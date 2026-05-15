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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from rsl_rl_isrc.modules import SACNetworks
from rsl_rl_isrc.storage import ReplayBuffer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class SAC:
    """
    Soft Actor-Critic (SAC) algorithm implementation adapted from CleanRL.
    Uses ReplayBuffer for off-policy storage (add + sample).
    """
    sac_networks: SACNetworks

    def __init__(self,
                 sac_networks=None,
                 num_obs=None,
                 num_actions=None,
                 actor_hidden_dims=[256, 256],
                 critic_hidden_dims=[256, 256],
                 activation='relu',
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 autotune=True,
                 target_entropy=None,
                 policy_lr=3e-4,
                 q_lr=1e-3,
                 alpha_lr=1e-3,
                 buffer_size=int(1e6),
                 batch_size=256,
                 learning_starts=5000,
                 policy_frequency=2,
                 target_network_frequency=1,
                 max_grad_norm=1.0,
                 critic_grad_clip=True,
                 critic_max_grad_norm=1.0,
                 actor_grad_clip=True,
                 actor_max_grad_norm=1.0,
                 update_frequency=128,
                 num_updates_per_step=1,
                 device='cpu',
                 **kwargs):
        """
        默认可调参数（与 CleanRL 对齐）:
        - sac_networks: SAC网络对象，如果为None则在内部创建
        - num_obs / num_actions: 观察和动作维度（当sac_networks为None时必需）
        - actor_hidden_dims / critic_hidden_dims: Actor/Critic网络隐藏层维度
        - activation: 激活函数 ('relu', 'elu', 'selu', 'lrelu', 'tanh', 'sigmoid')
        - learning_starts: 开始学习前的 transition 数（与 CleanRL 一致）
        - policy_frequency / target_network_frequency: 策略与目标网络更新频率
        - critic_grad_clip / actor_grad_clip: 是否对Critic/Actor梯度进行裁剪
        - critic_max_grad_norm / actor_max_grad_norm: Critic/Actor梯度裁剪的最大范数
        - update_frequency: 每多少个transitions执行一次更新
        - num_updates_per_step: 每次更新执行多少次梯度更新
        global_step 在 process_env_step 内按每条 transition 递增，不在外部修改。
        """
        if kwargs:
            print("SAC.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        self.device = device

        # Create SAC networks if not provided
        if sac_networks is None:
            if num_obs is None or num_actions is None:
                raise ValueError("num_obs and num_actions must be provided when sac_networks is None")
            from rsl_rl_isrc.modules import SACNetworks
            sac_networks = SACNetworks(
                num_obs=num_obs,
                num_actions=num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation
            )

        self.sac_networks = sac_networks
        self.sac_networks.to(self.device)

        self.storage = None  # initialized later (ReplayBuffer)

        # SAC parameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.autotune = autotune
        self.target_entropy = target_entropy
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.max_grad_norm = max_grad_norm
        self.critic_grad_clip = critic_grad_clip
        self.critic_max_grad_norm = critic_max_grad_norm
        self.actor_grad_clip = actor_grad_clip
        self.actor_max_grad_norm = actor_max_grad_norm
        self.update_frequency = update_frequency
        self.num_updates_per_step = num_updates_per_step

        # Training parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.sac_networks.qf1.parameters()) + list(self.sac_networks.qf2.parameters()),
            lr=self.q_lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.sac_networks.actor.parameters()) + list(self.sac_networks.actor_logstd.parameters()),
            lr=self.policy_lr
        )

        # Automatic entropy tuning
        if self.autotune:
            if self.target_entropy is None:
                # Default target entropy = -dim(actions)
                self.target_entropy = -torch.prod(torch.tensor(self.sac_networks.action_scale.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.a_optimizer = None

        # Training step counter
        self.global_step = 0

    def init_storage(self, num_envs, obs_shape, action_shape):
        """Initialize ReplayBuffer for off-policy learning."""
        self.storage = ReplayBuffer(
            buffer_size=self.buffer_size,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.device,
            n_envs=num_envs,
        )

    def test_mode(self):
        self.sac_networks.eval()

    def train_mode(self):
        self.sac_networks.train()

    def act(self, obs, **kwargs):
        """Sample action from policy"""
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.sac_networks.update_distribution(obs)

        if self.global_step < self.learning_starts:
            return None
        actions = self.sac_networks.act(obs)
        return actions

    def process_env_step(self, rewards, dones, infos, next_obs, obs=None, actions=None):
        """
        处理批量 transitions：写入 ReplayBuffer 并递增 global_step（与 CleanRL 一致）。
        每次调用都会写入 buffer，global_step 按 transitions 数量递增。
        """
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
        self.global_step += obs.shape[0]  # 批量大小

    def update(self):
        """Perform SAC training update (sample batch from ReplayBuffer)."""
        if self.global_step < self.learning_starts:
            return 0.0, 0.0, 0.0, 0.0
        if len(self.storage) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0
        # Check if it's time to update based on update_frequency
        if self.global_step % self.update_frequency != 0:
            return 0.0, 0.0, 0.0, 0.0

        # Perform multiple updates per step
        final_qf1_loss = 0.0
        final_qf2_loss = 0.0
        final_actor_loss = 0.0
        final_alpha_loss = 0.0

        for update_idx in range(self.num_updates_per_step):
            data = self.storage.sample(self.batch_size)
            if data is None:
                continue

            obs, next_obs, actions, rewards, dones = data

            # SAC update logic (adapted from CleanRL)
            with torch.no_grad():
                # Sample next actions and compute target Q
                next_actions = self.sac_networks.act(next_obs)
                next_log_probs = self.sac_networks.get_actions_log_prob(next_actions)

                qf1_next_target, qf2_next_target = self.sac_networks.evaluate_target(next_obs, next_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_probs
                next_q_value = rewards.flatten() + (1 - dones.float().flatten()) * self.gamma * min_qf_next_target.flatten()

            # Q network update
            qf1_a_values, qf2_a_values = self.sac_networks.evaluate(obs, actions)
            qf1_loss = F.mse_loss(qf1_a_values.flatten(), next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values.flatten(), next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.q_optimizer.zero_grad()
            qf_loss.backward()
            if self.critic_grad_clip:
                nn.utils.clip_grad_norm_(
                    list(self.sac_networks.qf1.parameters()) + list(self.sac_networks.qf2.parameters()),
                    self.critic_max_grad_norm
                )
            self.q_optimizer.step()

            actor_loss = None
            alpha_loss = None

            # Delayed policy update
            if self.global_step % self.policy_frequency == 0:
                # Policy (Actor) update
                for _ in range(self.policy_frequency):
                    actions_pi = self.sac_networks.act(obs)
                    log_probs_pi = self.sac_networks.get_actions_log_prob(actions_pi)
                    qf1_pi, qf2_pi = self.sac_networks.evaluate(obs, actions_pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    actor_loss = (self.alpha * log_probs_pi - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if self.actor_grad_clip:
                        nn.utils.clip_grad_norm_(
                            list(self.sac_networks.actor.parameters()) + list(self.sac_networks.actor_logstd.parameters()),
                            self.actor_max_grad_norm
                        )
                    self.actor_optimizer.step()

                    # Entropy temperature update
                    if self.autotune:
                        with torch.no_grad():
                            actions_temp = self.sac_networks.act(obs)
                            log_probs_temp = self.sac_networks.get_actions_log_prob(actions_temp)

                        alpha_loss = (-self.log_alpha.exp() * (log_probs_temp + self.target_entropy)).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            # Target network update
            if self.global_step % self.target_network_frequency == 0:
                self.sac_networks.update_target_networks(self.tau)

            # Accumulate losses for averaging
            final_qf1_loss += qf1_loss.item()
            final_qf2_loss += qf2_loss.item()
            if actor_loss is not None:
                final_actor_loss += actor_loss.item()
            if alpha_loss is not None:
                final_alpha_loss += alpha_loss.item()

        # Return average losses
        num_updates = self.num_updates_per_step
        return (final_qf1_loss / num_updates, final_qf2_loss / num_updates,
                final_actor_loss / num_updates if final_actor_loss > 0 else 0.0,
                final_alpha_loss / num_updates if final_alpha_loss > 0 else 0.0)
