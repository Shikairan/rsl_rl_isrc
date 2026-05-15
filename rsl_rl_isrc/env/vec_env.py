# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""向量环境抽象接口：定义 ``step``/``reset``/观测形状，与具体仿真解耦。"""

from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union


class VecEnv(ABC):
    """并行向量环境的最小协议：``num_envs`` 路环境同步 ``step``/``reset``。

    实现类需提供张量形状的观测/奖励/终止标志；Runner 仅依赖本接口与具体物理引擎解耦。
    """

    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor  # 当前回合步数计数（按并行 env）
    extras: dict
    device: torch.device

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        """前进一步。

        参数:
            actions: 形状 ``(num_envs, num_actions)`` 的动作张量。

        返回:
            (obs, privileged_obs, rewards, dones, infos)。privileged_obs 若无则 ``None``；
            ``dones`` 通常为 ``float``/``bool`` 掩码；``infos`` 为每 env 附带的字典（可含 ``episode`` 统计）。
        """
        pass

    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        """重置指定 ``env_ids`` 子集（或全部）环境，并刷新内部缓冲区。"""
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        """返回策略网络使用的观测，形状 ``(num_envs, num_obs)``。"""
        pass

    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        """返回价值/评论家用的特权观测；若与策略观测相同可实现为 ``None``。"""
        pass
