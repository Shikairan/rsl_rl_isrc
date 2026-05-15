# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""
rsl_rl_isrc 运行器：SAC（最大熵软 Actor-Critic）训练主循环。

SAC (Soft Actor-Critic) Runner
仿照 on_policy_runner.py，专为 SAC 算法设计的独立训练 runner
SAC 为 off-policy 算法，使用 ReplayBuffer，每步收集后即可更新
"""

import time
import os
from collections import deque
import statistics
import torch
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool
import torch.distributed as dist

from rsl_rl_isrc.algorithms.sac_policy import SAC
from rsl_rl_isrc.modules import SACNetworks
from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.sockets import send_post_request, StepObsPublisher


class SACRunner:
    """封装 SAC 训练：构建 ``SACNetworks`` 与 ``SAC`` 算法、管理 ``ReplayBuffer``、日志与检查点。"""

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

        # SAC 使用 SACNetworks，需要动作边界（从 policy_cfg 或 algorithm 传入）
        action_bounds = self.alg_cfg.get("action_bounds", (-1.0, 1.0))
        sac_networks = SACNetworks(
            num_obs=self.env.num_obs,
            num_actions=self.env.num_actions,
            actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [256, 256]),
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256, 256]),
            activation=self.policy_cfg.get("activation", "relu")
        )
        low = torch.tensor([action_bounds[0]] * self.env.num_actions, device=self.device)
        high = torch.tensor([action_bounds[1]] * self.env.num_actions, device=self.device)
        sac_networks.set_action_bounds(low, high)

        self.alg = SAC(
            sac_networks=sac_networks,
            device=self.device,
            **{k: v for k, v in self.alg_cfg.items() if k != "action_bounds"}
        )

        self.num_steps_per_env = self.cfg.get("num_steps_per_env", 1)
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            obs_shape=(self.env.num_obs,),
            action_shape=(self.env.num_actions,)
        )

        self.retstate_list = []
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.step_obs = StepObsPublisher(self.rank, self.task, self.env.num_envs)

        self.env.reset()

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
        """SAC 主循环：每步 ``process_env_step``、软更新 Q/Actor/温度，并记录 TensorBoard 指标。"""
        if self.log_dir is not None and self.writer is None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        obs = obs.to(self.device)
        self.alg.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations

        if dist.is_initialized():
            for param in self.alg.sac_networks.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            for i in range(self.num_steps_per_env):
                actions = self.alg.act(obs)
                if actions is None:
                    actions = torch.randn(self.env.num_envs, self.env.num_actions, device=self.device)
                    actions = torch.clamp(actions, -1.0, 1.0)
                else:
                    actions = actions.detach()

                obs_next, _, rewards, dones, infos = self.env.step(actions)
                self.step_obs.push(obs_next)
                self.socket_send()

                obs_next = obs_next.to(self.device)
                rewards = rewards.to(self.device) if torch.is_tensor(rewards) else torch.tensor(rewards, device=self.device)
                dones = dones.to(self.device) if torch.is_tensor(dones) else torch.tensor(dones, dtype=torch.float, device=self.device)

                self.alg.process_env_step(
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                    next_obs=obs_next,
                    obs=obs,
                    actions=actions
                )

                qf1_loss, qf2_loss, actor_loss, alpha_loss = self.alg.update()

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

                obs = obs_next

            stop = time.time()
            collection_time = stop - start
            learn_time = 0.0

            if dist.is_initialized():
                dist.barrier()
                for param in self.alg.sac_networks.parameters():
                    dist.broadcast(param.data, src=0)
                dist.barrier()

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
        self.tot_time += locs['collection_time'] + locs.get('learn_time', 0)
        iteration_time = locs['collection_time'] + locs.get('learn_time', 0)
        fps = int(self.num_steps_per_env * self.env.num_envs / max(1e-6, locs['collection_time'] + locs.get('learn_time', 0)))

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

        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        if len(locs.get('rewbuffer', [])) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str_ = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        if len(locs.get('rewbuffer', [])) > 0:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s)\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s)\n"
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
            'sac_networks': self.alg.sac_networks.state_dict(),
            'log_alpha': getattr(self.alg, 'log_alpha', None),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path):
        loaded = torch.load(path)
        self.alg.sac_networks.load_state_dict(loaded['sac_networks'])
        if loaded.get('log_alpha') is not None and hasattr(self.alg, 'log_alpha'):
            self.alg.log_alpha = loaded['log_alpha']
        self.current_learning_iteration = loaded['iter']
        return loaded.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.test_mode()
        if device is not None:
            self.alg.sac_networks.to(device)
        return self.alg.sac_networks.act
