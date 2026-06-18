# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""A2C（Advantage Actor-Critic）：GAE 优势、单轮策略/价值梯度更新（无 PPO 裁剪）。"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.storage import RolloutStorage


class A2C:
    """A2C 算法内核：维护 ``RolloutStorage``、计算 GAE 回报与优势、执行单轮策略梯度更新。"""

    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        learning_rate=7e-4,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.actor_critic = actor_critic
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.transition = RolloutStorage.Transition()

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        step = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            _target_values_batch,
            advantages_batch,
            returns_batch,
            _old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            step += 1
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            entropy_batch = self.actor_critic.entropy

            surrogate_loss = -(torch.squeeze(advantages_batch) * actions_log_prob_batch).mean()
            value_loss = (returns_batch - value_batch).pow(2).mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = max(self.num_learning_epochs * self.num_mini_batches, 1)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss
