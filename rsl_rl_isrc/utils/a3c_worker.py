# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""A3C 异步 worker 进程：独立环境实例 + Hogwild 共享模型更新。"""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from rsl_rl_isrc.algorithms.a3c import A3C


def _to_tensor(obs, device):
    if torch.is_tensor(obs):
        return obs.to(device=device, dtype=torch.float32)
    return torch.tensor(obs, dtype=torch.float32, device=device)


def run_a3c_worker(
    worker_id: int,
    shared_actor_critic: torch.nn.Module,
    alg_cfg: Dict[str, Any],
    num_rollouts: int,
    global_counter,
    env_id: str = "Pendulum-v1",
    env_factory: Optional[Callable[[], Any]] = None,
    update_lock=None,
    result_queue=None,
    seed: Optional[int] = None,
) -> None:
    """单个 A3C worker：本地 rollout，异步写入共享 ``ActorCritic``。

    参数:
        worker_id: worker 编号（用于日志与随机种子）。
        shared_actor_critic: 已 ``share_memory()`` 的全局 Actor-Critic。
        alg_cfg: 算法超参字典。
        num_rollouts: 本 worker 执行的 rollout 次数。
        global_counter: ``multiprocessing.Value``，统计全局完成 rollout 数。
        env_id: Gymnasium 环境 ID（``env_factory`` 未提供时使用）。
        env_factory: 可选，返回 **单个** 环境实例的可调用对象。
        update_lock: 可选互斥锁；为 None 时使用 Hogwild 无锁更新。
        result_queue: 可选，worker 结束时上报统计。
        seed: 可选随机种子基数。
    """
    torch.set_num_threads(1)
    if seed is not None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    device = alg_cfg.get("device", "cpu")
    if env_factory is not None:
        env = env_factory()
    else:
        import rsl_rl_isrc.isrcgym as gym

        env = gym.make(env_id)
    obs, _ = env.reset(seed=(seed + worker_id) if seed is not None else None)

    num_obs = int(np.prod(env.observation_space.shape))
    num_actions = int(np.prod(env.action_space.shape)) if len(env.action_space.shape) > 0 else 1

    alg = A3C(
        actor_critic=shared_actor_critic,
        t_max=alg_cfg.get("t_max", 20),
        n_steps=alg_cfg.get("n_steps", 5),
        gamma=alg_cfg.get("gamma", 0.99),
        value_loss_coef=alg_cfg.get("value_loss_coef", 0.5),
        entropy_coef=alg_cfg.get("entropy_coef", 0.01),
        learning_rate=alg_cfg.get("learning_rate", 1e-3),
        max_grad_norm=alg_cfg.get("max_grad_norm", 40.0),
        optimizer_type=alg_cfg.get("optimizer_type", "sgd"),
        device=device,
    )
    alg.init_storage(
        actor_obs_shape=[num_obs],
        critic_obs_shape=[num_obs],
        action_shape=[num_actions],
    )
    alg.train_mode()

    rollouts_done = 0
    episode_reward = 0.0
    last_episode_reward = 0.0

    while rollouts_done < num_rollouts:
        alg.storage.clear()
        done_flag = False

        for _ in range(alg.t_max):
            obs_t = _to_tensor(obs, device).unsqueeze(0)
            actions = alg.act(obs_t, obs_t)
            action_np = actions.detach().cpu().numpy().reshape(-1)

            next_obs, reward, terminated, truncated, infos = env.step(action_np)
            done = bool(terminated or truncated)
            episode_reward += float(reward)

            rewards_t = torch.tensor([reward], dtype=torch.float32, device=device)
            dones_t = torch.tensor([done], dtype=torch.bool, device=device)
            alg.process_env_step(rewards_t, dones_t, infos if isinstance(infos, dict) else {})

            if done:
                last_episode_reward = episode_reward
                episode_reward = 0.0
                next_obs, _ = env.reset()
                done_flag = True
                break
            obs = next_obs

        obs_t = _to_tensor(obs, device).unsqueeze(0)
        alg.compute_returns(obs_t)

        lock_ctx = update_lock if update_lock is not None else nullcontext()
        with lock_ctx:
            alg.update(num_steps=alg.storage.step)

        rollouts_done += 1
        with global_counter.get_lock():
            global_counter.value += 1

    if result_queue is not None:
        result_queue.put((worker_id, rollouts_done, last_episode_reward, done_flag))
