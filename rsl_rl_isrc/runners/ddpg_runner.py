# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""DDPG 训练运行器：off-policy 逐步采集、ReplayBuffer 更新与分布式广播。"""

import os
import time
from collections import deque
import statistics

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from rsl_rl_isrc.algorithms.ddpg_policy import DDPG
from rsl_rl_isrc.modules import DDPGNetworks
from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.sockets import StepObsPublisher


class DDPGRunner:
    """封装 DDPG 训练：构建 ``DDPGNetworks`` 与 ``DDPG`` 算法、管理 ``ReplayBuffer``、日志与检查点。"""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.task = self.cfg["experiment_name"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = device
        self.env = env

        action_bounds = self.alg_cfg.get("action_bounds", (-1.0, 1.0))
        ddpg_networks = DDPGNetworks(
            num_obs=self.env.num_obs,
            num_actions=self.env.num_actions,
            actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [256, 256]),
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256, 256]),
            activation=self.policy_cfg.get("activation", "relu"),
        )
        low = torch.tensor([action_bounds[0]] * self.env.num_actions, device=self.device)
        high = torch.tensor([action_bounds[1]] * self.env.num_actions, device=self.device)
        ddpg_networks.set_action_bounds(low, high)

        self.alg = DDPG(
            ddpg_networks=ddpg_networks,
            device=self.device,
            **{k: v for k, v in self.alg_cfg.items() if k != "action_bounds"},
        )

        self.num_steps_per_env = self.cfg.get("num_steps_per_env", 1)
        self.save_interval = self.cfg["save_interval"]
        self.action_bounds = action_bounds

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            obs_shape=(self.env.num_obs,),
            action_shape=(self.env.num_actions,),
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.step_obs = StepObsPublisher(self.rank, self.task, self.env.num_envs)
        self.step_obs.set_env(self.env)

        self.env.reset(torch.arange(self.env.num_envs))

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations

        if dist.is_initialized():
            for param in self.alg.ddpg_networks.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            critic_loss = 0.0
            actor_loss = 0.0

            for _ in range(self.num_steps_per_env):
                actions = self.alg.act(obs, explore=True)
                if actions is None:
                    low, high = self.action_bounds
                    actions = torch.empty(self.env.num_envs, self.env.num_actions, device=self.device).uniform_(low, high)
                else:
                    actions = actions.detach()

                obs_next, _, rewards, dones, infos = self.env.step(actions)
                self.step_obs.push(obs_next)

                obs_next = obs_next.to(self.device)
                rewards = rewards.to(self.device) if torch.is_tensor(rewards) else torch.tensor(rewards, device=self.device)
                dones = dones.to(self.device) if torch.is_tensor(dones) else torch.tensor(dones, dtype=torch.float, device=self.device)

                self.alg.process_env_step(
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                    next_obs=obs_next,
                    obs=obs,
                    actions=actions,
                )

                if not dist.is_initialized() or dist.get_rank() == 0:
                    critic_loss, actor_loss = self.alg.update()

                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
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

            if dist.is_initialized():
                dist.barrier()
                for param in self.alg.ddpg_networks.parameters():
                    dist.broadcast(param.data, src=0)
                gs_tensor = torch.tensor([self.alg.global_step], dtype=torch.long, device=self.device)
                dist.broadcast(gs_tensor, src=0)
                self.alg.global_step = int(gs_tensor.item())
                dist.barrier()

            if self.log_dir is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                self.log(locals())
            if it % self.save_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"]
        iteration_time = locs["collection_time"]
        fps = int(self.num_steps_per_env * self.env.num_envs / max(1e-6, locs["collection_time"]))

        ep_string = ""
        if locs.get("ep_infos"):
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        self.writer.add_scalar("Loss/critic", locs.get("critic_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/actor", locs.get("actor_loss", 0.0), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        if len(locs.get("rewbuffer", [])) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        str_ = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        if len(locs.get("rewbuffer", [])) > 0:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s)\n"
                f"{'Critic loss:':>{pad}} {locs.get('critic_loss', 0.0):.4f}\n"
                f"{'Actor loss:':>{pad}} {locs.get('actor_loss', 0.0):.4f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n{str_.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s)\n"
                f"{'Critic loss:':>{pad}} {locs.get('critic_loss', 0.0):.4f}\n"
                f"{'Actor loss:':>{pad}} {locs.get('actor_loss', 0.0):.4f}\n"
            )
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "ddpg_networks": self.alg.ddpg_networks.state_dict(),
                "actor_optimizer": self.alg.actor_optimizer.state_dict(),
                "critic_optimizer": self.alg.critic_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "global_step": self.alg.global_step,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        self.alg.ddpg_networks.load_state_dict(loaded["ddpg_networks"])
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
            self.alg.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
        self.current_learning_iteration = loaded["iter"]
        self.alg.global_step = loaded.get("global_step", 0)
        return loaded.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.test_mode()
        if device is not None:
            self.alg.ddpg_networks.to(device)
        return self.alg.ddpg_networks.act_inference
