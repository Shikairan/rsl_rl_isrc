# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""
rsl_rl_isrc 运行器：REINFORCE（蒙特卡洛策略梯度）训练主循环。

REINFORCE (Policy Gradient) Runner
仿照 on_policy_runner.py，专为 REINFORCE 算法设计的独立训练 runner
使用 REINFORCEPolicy + RolloutStorage off-policy episodes
"""

import time
import os
from collections import deque
import statistics
import torch
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool
import torch.distributed as dist

from rsl_rl_isrc.algorithms import REINFORCEPolicy
from rsl_rl_isrc.storage import RolloutStorage
from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.sockets import send_post_request, StepObsPublisher


class REINFORCERunner:
    """封装 REINFORCE：``REINFORCEPolicy``、按 env 写入 ``RolloutStorage`` 及回合结束折扣回报。"""

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.cfg = train_cfg["runner"]
        self.task = self.cfg["experiment_name"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = device
        self.env = env

        self.pool = Pool(processes=60) if dist.is_initialized() else None
        self.state_tag = torch.tensor([0, 0, 0, 64], device=self.device)

        action_space_type = self.policy_cfg.get("action_space_type", "continuous")
        num_privileged = self.env.num_privileged_obs
        privileged_shape = [num_privileged] if num_privileged is not None else [None]

        self.alg = REINFORCEPolicy(
            num_obs=self.env.num_obs,
            num_actions=self.env.num_actions,
            action_space_type=action_space_type,
            device=self.device,
            **self.alg_cfg,
            **{k: v for k, v in self.policy_cfg.items() if k != 'action_space_type'}
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.storage = RolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            obs_shape=[self.env.num_obs],
            privileged_obs_shape=privileged_shape,
            actions_shape=[self.env.num_actions],
            device=self.device
        )

        self.retstate_list = []
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.action_space_type = action_space_type
        self.step_obs = StepObsPublisher(self.rank, self.task, self.env.num_envs)

        self.env.reset(torch.arange(self.env.num_envs))

    def socket_send(self):
        """发送状态到远端（若 env 支持 base_pos 等）"""
        if not dist.is_initialized() or self.pool is None:
            return
        if self.state_tag[0] != self.rank:
            return
        if self.state_tag[1] == 1.0:
            return
        if not hasattr(self.env, 'base_pos') or not hasattr(self.env, 'base_quat') or not hasattr(self.env, 'dof_pos'):
            return
        try:
            location = self.env.base_pos[self.state_tag[2]:self.state_tag[3], :].clone()
            location[:, 1] = location[:, 1] - torch.arange(
                int(self.state_tag[2].item()), int(self.state_tag[3].item()), device=location.device
            ).float() * 3
            tt = torch.cat((
                location,
                self.env.base_quat[self.state_tag[2]:self.state_tag[3], :],
                self.env.dof_pos[self.state_tag[2]:self.state_tag[3], :]
            ), dim=1).cpu().tolist()
            ret = self.pool.apply_async(send_post_request, args=(tt, self.rank, self.task))
            self.retstate_list.append(ret)
        except Exception:
            pass

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """REINFORCE 主循环：``act`` 与环境交互、写入 ``RolloutStorage``、分布式同步后更新策略。"""
        if self.log_dir is not None and self.writer is None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        obs = obs.to(self.device)
        privileged_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs

        self.alg.actor.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations

        if dist.is_initialized():
            for param in self.alg.actor.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions, action_log_probs = self.alg.act(obs)

                    if self.action_space_type == 'discrete':
                        env_actions = actions.argmax(dim=-1)
                    else:
                        env_actions = actions

                    obs, privileged_obs, rewards, dones, infos = self.env.step(env_actions)
                    self.step_obs.push(obs)
                    self.socket_send()

                    privileged_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    privileged_obs = privileged_obs.to(self.device)
                    rewards = rewards.to(self.device) if torch.is_tensor(rewards) else torch.tensor(rewards, device=self.device)
                    dones = dones.to(self.device) if torch.is_tensor(dones) else torch.tensor(dones, dtype=torch.bool, device=self.device)

                    for env_idx in range(self.env.num_envs):
                        transition = RolloutStorage.Transition()
                        transition.observations = obs[env_idx:env_idx + 1]
                        transition.critic_observations = privileged_obs[env_idx:env_idx + 1]
                        transition.actions = actions[env_idx:env_idx + 1]
                        transition.rewards = rewards[env_idx:env_idx + 1]
                        transition.dones = dones[env_idx:env_idx + 1]
                        transition.actions_log_prob = action_log_probs[env_idx:env_idx + 1]
                        transition.values = torch.zeros_like(transition.rewards)
                        transition.action_mean = torch.zeros_like(transition.actions)
                        transition.action_sigma = torch.ones_like(transition.actions)
                        transition.hidden_states = self.alg.get_hidden_states()

                        self.storage.add_off_policy_transition(transition, env_idx)

                        if dones[env_idx]:
                            self.storage.finish_episode(env_idx, gamma=self.alg.gamma)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            start = stop

            if dist.is_initialized():
                dist.barrier()

            if not dist.is_initialized() or dist.get_rank() == 0:
                mean_loss = self.alg.update(self.storage)
            if dist.is_initialized():
                dist.barrier()

            if dist.is_initialized():
                for param in self.alg.actor.parameters():
                    dist.broadcast(param.data, src=0)
                dist.barrier()

            min_episodes_to_keep = max(10, self.env.num_envs // 2)
            if len(self.storage.off_policy_episodes) > min_episodes_to_keep * 2:
                self.storage.off_policy_episodes = self.storage.off_policy_episodes[-min_episodes_to_keep:]

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                locs = locals()
                self.log(locs)
            if it % self.save_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        ep_string = ''
        if locs.get('ep_infos'):
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        actor = self.alg.actor.module if hasattr(self.alg.actor, 'module') else self.alg.actor
        mean_std = actor.std.mean() if self.action_space_type == 'continuous' and hasattr(actor, 'std') else torch.tensor(0.0, device=self.device)
        self.writer.add_scalar('Loss/reinforce_loss', locs.get('mean_loss', 0), locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs.get('rewbuffer', [])) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str_ = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        if len(locs.get('rewbuffer', [])) > 0:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'REINFORCE Loss:':>{pad}} {locs.get('mean_loss', 0):.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'REINFORCE Loss:':>{pad}} {locs.get('mean_loss', 0):.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            )
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'actor_state_dict': self.alg.actor.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path)
        self.alg.actor.load_state_dict(loaded['actor_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        self.current_learning_iteration = loaded['iter']
        return loaded.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.actor.eval()
        if device is not None:
            self.alg.actor.to(device)
        return self.alg.get_inference_policy(device)
