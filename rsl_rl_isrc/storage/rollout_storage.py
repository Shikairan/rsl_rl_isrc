# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""按 ``(时间步, 并行环境, …)`` 组织的回合数据与张量广播辅助。"""

import torch
import numpy as np
import os
import torch.distributed as dist
from rsl_rl_isrc.utils import split_and_pad_trajectories


class RolloutStorage:
    """按 ``(时间步, 并行环境, …)`` 存储 on-policy 轨迹，并支持分布式 ``broadcast``/清空缓存。"""

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    class Episode:
        def __init__(self):
            self.observations = []
            self.critic_observations = []
            self.actions = []
            self.rewards = []
            self.dones = []
            self.actions_log_prob = []
            self.values = []
            self.returns = []  # For REINFORCE discounted returns
            self.hidden_states = []

        def add_transition(self, transition: 'RolloutStorage.Transition'):
            self.observations.append(transition.observations)
            self.critic_observations.append(transition.critic_observations)
            self.actions.append(transition.actions)
            self.rewards.append(transition.rewards)
            self.dones.append(transition.dones)
            self.actions_log_prob.append(transition.actions_log_prob)
            self.values.append(transition.values)
            self.hidden_states.append(transition.hidden_states)

        def length(self):
            return len(self.observations)

        def to_tensors(self, device='cpu'):
            """Convert episode data to tensors"""
            # For REINFORCE, we need observations to maintain gradient connection for policy updates
            # But actions, rewards, etc. should be detached as they don't need gradients
            obs = torch.stack([x.clone() for x in self.observations]).to(device)
            critic_obs = torch.stack([x.clone() for x in self.critic_observations]).to(device) if self.critic_observations[0] is not None else None
            actions = torch.stack([x.clone().detach() for x in self.actions]).to(device)
            rewards = torch.stack([x.clone().detach() for x in self.rewards]).to(device)
            dones = torch.stack([x.clone().detach() for x in self.dones]).to(device)
            log_probs = torch.stack([x.clone().detach() for x in self.actions_log_prob]).to(device)
            values = torch.stack([x.clone().detach() for x in self.values]).to(device) if self.values[0] is not None else None

            return {
                'observations': obs,
                'critic_observations': critic_obs,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'actions_log_prob': log_probs,
                'values': values,
                'returns': torch.stack([x.clone().detach() for x in self.returns]).to(device) if self.returns else None
            }

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

        self.off_policy_episodes = []
        self.current_episodes = [self.Episode() for _ in range(num_envs)]

    def init_tensor(self):
        self.observations = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.obs_shape, device=self.device)
        if self.privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.actions = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=self.device)
        self.dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device).byte()

        self.actions_log_prob = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.advantages = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.mu = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=self.device)
        self.sigma = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=self.device)
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None


    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow step:{}, num_transitions_per_env:{},rank:{}".format(self.step,  self.num_transitions_per_env))
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def gather_variable_length(self, tensor, dst=0):
        shape = torch.tensor(tensor.shape)
        timeFrame = shape[0]
        dataFrame = shape[1:]
        dataFrame[-2] = dataFrame[-2]*dist.get_world_size()
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        world_size = dist.get_world_size()
        
        gather_list_2 = [torch.zeros_like(tensor) for _ in range(world_size)] if rank == dst else None

        dist.gather(tensor, gather_list_2, dst=dst)
        dist.barrier()
        if rank == dst:
            if len(tensor.shape) == 3:
                reshape_tensor = torch.cat(gather_list_2, dim=1)
                return reshape_tensor
            else:
                reshape_tensor = torch.cat(gather_list_2, dim=2)
                return reshape_tensor
        return tensor


    def broadcast(self):
        #print("self.observations:", self.observations.shape, self.observations.device)
        #print("rank:{} self.dones:{}".format(dist.get_rank, self.dones.shape))

        self.observations = self.gather_variable_length(self.observations)
        if self.privileged_obs_shape[0] is not None:
            self.privileged_observations = self.gather_variable_length(self.privileged_observations)
        else:
            self.privileged_observations = None
        self.rewards = self.gather_variable_length(self.rewards)
        self.actions = self.gather_variable_length(self.actions)
        self.dones = self.gather_variable_length(self.dones)
        
        self.actions_log_prob = self.gather_variable_length(self.actions_log_prob)
        self.values = self.gather_variable_length(self.values)
        self.returns = self.gather_variable_length(self.returns)
        self.advantages = self.gather_variable_length(self.advantages)
        
        self.mu = self.gather_variable_length(self.mu)
        self.sigma = self.gather_variable_length(self.sigma)
        saved_hidden_states_a_0 = self.gather_variable_length(self.saved_hidden_states_a[0])
        saved_hidden_states_c_0 = self.gather_variable_length(self.saved_hidden_states_c[0])
        saved_hidden_states_a_1 = self.gather_variable_length(self.saved_hidden_states_a[1])
        saved_hidden_states_c_1 = self.gather_variable_length(self.saved_hidden_states_c[1])
        self.saved_hidden_states_a = (saved_hidden_states_a_0, saved_hidden_states_a_1)
        self.saved_hidden_states_c = (saved_hidden_states_c_0, saved_hidden_states_c_1)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            self.dones =  self.dones.byte()
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def clear(self):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            self.init_tensor()
        else:
            self.init_tensor()
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values

        # ✅ 标准化优势函数（TRPO需要）
        # 注意：在标准化之前检查是否有足够的方差
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        value_error = (self.returns - self.values).abs().mean()
        if value_error > 100:  # 如果价值函数误差很大
            print(f"  ⚠️ 价值函数误差过大({value_error:.3f})，优势函数可能不准确") 
        if adv_std > 1e-3:  # 只有当标准差足够大时才标准化
            self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # 如果优势函数方差太小，说明价值函数可能不准确，使用原始值
            print(f"  ⚠️ 优势函数方差太小({adv_std:.6f})，可能价值函数不准确")
            self.advantages = self.advantages - adv_mean  # 只中心化，不缩放
        

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None

    def add_off_policy_transition(self, transition: Transition, env_idx: int):
        """Add a transition to the current episode for off-policy learning"""
        self.current_episodes[env_idx].add_transition(transition)

    def finish_episode(self, env_idx: int, gamma: float = 0.99):
        """Finish an episode and compute discounted returns for REINFORCE"""
        episode = self.current_episodes[env_idx]
        if episode.length() > 0:
            # Compute discounted returns
            returns = []
            G = 0
            for reward in reversed(episode.rewards):
                G = reward + gamma * G
                returns.insert(0, G)
            episode.returns = [
                r.detach().clone().to(self.device) if isinstance(r, torch.Tensor)
                else torch.tensor(r, dtype=torch.float32, device=self.device)
                for r in returns
            ]

            # Store completed episode
            self.off_policy_episodes.append(episode)

            # Reset current episode
            self.current_episodes[env_idx] = self.Episode()

    def get_off_policy_episodes(self, batch_size: int = None):
        """Get episodes for off-policy REINFORCE training"""
        if batch_size is None:
            return self.off_policy_episodes
        return self.off_policy_episodes[-batch_size:] if len(self.off_policy_episodes) >= batch_size else self.off_policy_episodes

    def clear_off_policy_episodes(self):
        """Clear stored off-policy episodes"""
        self.off_policy_episodes.clear()

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        print("reccurent:",self.observations.shape, self.dones.shape)
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        num_mini_batches = 4
        mini_batch_size =  dist.get_world_size() * self.num_envs  // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ] 
                hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_c ]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                

                first_traj = last_traj
