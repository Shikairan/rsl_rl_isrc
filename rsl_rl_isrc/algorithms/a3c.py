# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""A3C（异步 Advantage Actor-Critic）：n-step 回报，供多进程 worker Hogwild 更新。"""

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.storage import RolloutStorage


class A3C:
    """A3C 算法内核：单 worker 本地 rollout + n-step 优势 + 策略梯度更新。

    多个 worker 共享同一 ``ActorCritic``（``share_memory``）并异步调用 ``update``。
  """

    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        t_max=20,
        n_steps=5,
        gamma=0.99,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=40.0,
        optimizer_type="sgd",
        device="cpu",
    ):
        self.device = device
        self.actor_critic = actor_critic
        self.t_max = t_max
        self.n_steps = n_steps
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        if optimizer_type == "adam":
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.actor_critic.parameters(), lr=learning_rate)

        self.storage = None
        self.transition = RolloutStorage.Transition()

    def init_storage(self, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            1,
            self.t_max,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

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
        self.storage.compute_n_step_returns(
            last_values, self.gamma, self.n_steps, num_steps=self.storage.step
        )

    def update(self, num_steps=None):
        if num_steps is None:
            num_steps = self.storage.step
        if num_steps <= 0:
            return 0.0, 0.0

        mean_value_loss = 0.0
        mean_policy_loss = 0.0

        obs = self.storage.observations[:num_steps, 0]
        critic_obs = (
            self.storage.privileged_observations[:num_steps, 0]
            if self.storage.privileged_observations is not None
            else obs
        )
        actions = self.storage.actions[:num_steps, 0]
        advantages = self.storage.advantages[:num_steps, 0]
        returns = self.storage.returns[:num_steps, 0]

        self.actor_critic.act(obs)
        actions_log_prob = self.actor_critic.get_actions_log_prob(actions)
        values = self.actor_critic.evaluate(critic_obs)
        entropy = self.actor_critic.entropy

        policy_loss = -(advantages.squeeze(-1) * actions_log_prob).mean()
        value_loss = (returns - values).pow(2).mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_value_loss = value_loss.item()
        mean_policy_loss = policy_loss.item()
        self.storage.clear()
        return mean_value_loss, mean_policy_loss
