# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""A3C 训练运行器：多进程异步 worker + 共享 Actor-Critic Hogwild 更新。"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.utils.a3c_worker import run_a3c_worker


class A3CRunner:
    """A3C 异步训练：每个 worker 独立环境实例，共享全局 ``ActorCritic``。

    与 ``OnPolicyRunner`` 的同步 ``VecEnv`` 不同，本 Runner **不使用** 向量环境；
    通过 ``num_workers`` 个进程各自持有 ``gym.make(env_id)`` 实例并异步更新。
    """

    def __init__(
        self,
        train_cfg,
        log_dir=None,
        device="cpu",
        env_id: str = "Pendulum-v1",
        env_factory: Optional[Callable[[], Any]] = None,
        num_obs: Optional[int] = None,
        num_actions: Optional[int] = None,
    ):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env_id = env_id
        self.env_factory = env_factory

        self.num_workers = self.cfg.get("num_workers", 4)
        self.save_interval = self.cfg.get("save_interval", 100)
        self.use_hogwild_lock = self.alg_cfg.get("use_hogwild_lock", True)

        if num_obs is None or num_actions is None:
            num_obs, num_actions = self._probe_env_dims()

        actor_critic = ActorCritic(
            num_obs,
            num_obs,
            num_actions,
            actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [64, 64]),
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [64, 64]),
            activation=self.policy_cfg.get("activation", "tanh"),
            init_noise_std=self.policy_cfg.get("init_noise_std", 1.0),
        ).to(self.device)
        actor_critic.share_memory()
        self.actor_critic = actor_critic

        self.alg_cfg_runtime = {
            **{k: v for k, v in self.alg_cfg.items() if k not in ("use_hogwild_lock",)},
            "device": str(self.device),
        }

        self.log_dir = log_dir
        self.writer = None
        self.current_learning_iteration = 0
        self.tot_time = 0.0

    def _probe_env_dims(self):
        if self.env_factory is not None:
            env = self.env_factory()
        else:
            import rsl_rl_isrc.isrcgym as gym

            env = gym.make(self.env_id)
        try:
            num_obs = int(env.observation_space.shape[0])
            num_actions = int(env.action_space.shape[0]) if len(env.action_space.shape) > 0 else 1
            return num_obs, num_actions
        finally:
            env.close()

    def learn(self, num_learning_iterations, seed: int = 42):
        """启动 ``num_workers`` 个异步 worker，每个执行 ``num_learning_iterations`` 次 rollout。"""
        if self.log_dir is not None and self.writer is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        ctx = mp.get_context("spawn")
        global_counter = ctx.Value("i", 0)
        update_lock = ctx.Lock() if self.use_hogwild_lock else None
        result_queue = ctx.Queue()

        start = time.time()
        processes = []
        for worker_id in range(self.num_workers):
            proc = ctx.Process(
                target=run_a3c_worker,
                args=(
                    worker_id,
                    self.actor_critic,
                    self.alg_cfg_runtime,
                    num_learning_iterations,
                    global_counter,
                ),
                kwargs={
                    "env_id": self.env_id,
                    "env_factory": self.env_factory,
                    "update_lock": update_lock,
                    "result_queue": result_queue,
                    "seed": seed,
                },
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

        self.tot_time = time.time() - start
        self.current_learning_iteration += num_learning_iterations

        worker_results = []
        while not result_queue.empty():
            worker_results.append(result_queue.get())

        total_rollouts = global_counter.value
        if self.writer is not None:
            self.writer.add_scalar("Train/total_rollouts", total_rollouts, self.current_learning_iteration)
            self.writer.add_scalar("Perf/total_time", self.tot_time, self.current_learning_iteration)
            if worker_results:
                rewards = [item[2] for item in worker_results if item[2] is not None]
                if rewards:
                    self.writer.add_scalar("Train/mean_last_episode_reward", sum(rewards) / len(rewards), self.current_learning_iteration)

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        print(
            f"A3C 异步训练完成: workers={self.num_workers}, "
            f"rollouts/worker={num_learning_iterations}, total_rollouts={total_rollouts}, "
            f"time={self.tot_time:.2f}s, hogwild_lock={self.use_hogwild_lock}"
        )
        return {
            "total_rollouts": total_rollouts,
            "worker_results": worker_results,
            "elapsed_time": self.tot_time,
        }

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.actor_critic.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path):
        loaded = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(loaded["model_state_dict"])
        self.current_learning_iteration = loaded.get("iter", 0)
        return loaded.get("infos")

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
