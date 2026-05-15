# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""
rsl_rl_isrc 运行器：TRPO（信赖域策略优化）训练主循环。

TRPO (Trust Region Policy Optimization) Runner
仿照 on_policy_runner.py，专为 TRPO 算法设计的独立训练 runner
"""

import time
import os
from collections import deque
import statistics
import torch
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool
import torch.distributed as dist

from rsl_rl_isrc.algorithms.trpo_policy import TRPOPolicy
from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.sockets import send_post_request, StepObsPublisher


class TRPORunner:
    """封装 TRPO 训练：``TRPOPolicy``、on-policy 存储、KL 信赖域更新与分布式参数同步。"""

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

        num_critic_obs = self.env.num_privileged_obs if self.env.num_privileged_obs is not None else self.env.num_obs
        action_bounds = self.alg_cfg.get("action_bounds", (-1.0, 1.0))
        critic_obs_shape = [num_critic_obs] if num_critic_obs is not None else [None]

        self.policy = TRPOPolicy(
            num_obs=self.env.num_obs,
            num_actions=self.env.num_actions,
            action_bounds=action_bounds,
            device=self.device,
            **{k: v for k, v in self.alg_cfg.items() if k != "action_bounds"},
            **self.policy_cfg
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.policy.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[self.env.num_obs],
            critic_obs_shape=critic_obs_shape,
            action_shape=[self.env.num_actions]
        )

        self.retstate_list = []
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.step_obs = StepObsPublisher(self.rank, self.task, self.env.num_envs)

        _, _ = self.env.reset()

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
        """TRPO 主循环：on-policy 采集、``state_tag`` 遥测、信赖域策略/价值更新与参数广播。"""
        if self.log_dir is not None and self.writer is None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.policy.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations

        if dist.is_initialized():
            for param in self.policy.policy_net.parameters():
                dist.broadcast(param.data, src=0)
            for param in self.policy.value_net.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.policy.act(obs, critic_obs)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    self.step_obs.push(obs)
                    self.socket_send()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    rewards = rewards.to(self.device) if torch.is_tensor(rewards) else torch.tensor(rewards, device=self.device)
                    dones = dones.to(self.device) if torch.is_tensor(dones) else torch.tensor(dones, dtype=torch.bool, device=self.device)

                    self.policy.process_env_step(rewards, dones, infos, scale_factor=1.0)

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
                rank_val = int(self.state_tag[0].item()) if torch.is_tensor(self.state_tag[0]) else int(self.state_tag[0])
                if dist.get_rank() == rank_val:
                    try:
                        tmp = self.retstate_list[-1].get() if self.retstate_list else {}
                        if 'error' not in tmp:
                            self.state_tag = torch.tensor(tmp.get('state', self.state_tag.cpu().tolist()), device=self.device)
                        dist.broadcast(self.state_tag, src=dist.get_rank())
                        self.retstate_list = []
                    except Exception:
                        pass
                dist.barrier()

                if hasattr(self.policy.algorithm.storage, 'broadcast'):
                    self.policy.algorithm.storage.broadcast()

            with torch.no_grad():
                self.policy.compute_returns(critic_obs)

            if dist.is_initialized():
                if dist.get_rank() == 0:
                    value_loss, policy_loss = self.policy.update()
                    dist.barrier()
                else:
                    self.policy.algorithm.storage.clear()
                    dist.barrier()
            else:
                value_loss, policy_loss = self.policy.update()

            self.policy.algorithm.storage.clear()
            if hasattr(self.policy, 'reset'):
                self.policy.reset(dones=None)

            if dist.is_initialized():
                for param in self.policy.policy_net.parameters():
                    dist.broadcast(param.data, src=0)
                for param in self.policy.value_net.parameters():
                    dist.broadcast(param.data, src=0)
                dist.barrier()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                self.log(locals())
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
        if locs['ep_infos']:
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

        self.writer.add_scalar('Loss/value_function', locs['value_loss'], locs['it'])
        self.writer.add_scalar('Loss/policy', locs['policy_loss'], locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str_ = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['value_loss']:.4f}\n"
                f"{'Policy loss:':>{pad}} {locs['policy_loss']:.4f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['value_loss']:.4f}\n"
                f"{'Policy loss:':>{pad}} {locs['policy_loss']:.4f}\n"
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
            'policy_net': self.policy.policy_net.state_dict(),
            'value_net': self.policy.value_net.state_dict(),
            'obs_rms': getattr(self.policy, 'obs_rms', None),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path):
        loaded = torch.load(path)
        self.policy.policy_net.load_state_dict(loaded['policy_net'])
        self.policy.value_net.load_state_dict(loaded['value_net'])
        if loaded.get('obs_rms') is not None and hasattr(self.policy, 'obs_rms'):
            self.policy.obs_rms = loaded['obs_rms']
        self.current_learning_iteration = loaded['iter']
        return loaded.get('infos')

    def get_inference_policy(self, device=None):
        self.policy.test_mode()
        if device is not None:
            self.policy.policy_net.to(device)
            self.policy.value_net.to(device)
        return lambda obs, critic_obs=None: self.policy.act(obs, critic_obs or obs)
