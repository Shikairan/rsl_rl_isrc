# rsl_rl_isrc — VecEnv 包装：显式暴露机器人仿真状态，供遥测与 StepObsPublisher 使用。

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Tuple, Union

import torch

from rsl_rl_isrc.env.vec_env import VecEnv

_ROBOT_STATE_NAMES = ("base_pos", "base_quat", "dof_pos")


def env_has_robot_state(env: Any) -> bool:
    """若 ``env`` 上 ``base_pos`` / ``base_quat`` / ``dof_pos`` 均为 ``torch.Tensor`` 则返回 True。"""
    for name in _ROBOT_STATE_NAMES:
        val = getattr(env, name, None)
        if not torch.is_tensor(val):
            return False
    return True


class StateExportVecEnv(VecEnv):
    """包装内层仿真 env，实现 ``VecEnv`` 并显式绑定机器人状态张量。

    - ``obs`` 维数/布局由用户 ``compute_observations`` 决定，本类不解析。
    - ``base_pos`` (N,3)、``base_quat`` (N,4) xyzw、``dof_pos`` (N, num_dof) 与内层**共享内存**；
      内层 ``step`` 刷新后 wrapper 上属性同步更新。
    - 内层若无上述字段，对应属性为 ``None``，``has_robot_state`` 为 False。

    子类需实现 ``reset``（内层 reset API 不统一）。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.num_envs = int(inner.num_envs)
        self.num_obs = int(inner.num_obs)
        self.num_privileged_obs = inner.num_privileged_obs
        self.num_actions = int(inner.num_actions)
        self.max_episode_length = int(inner.max_episode_length)
        self.device = inner.device

        self.obs_buf = inner.obs_buf
        self.privileged_obs_buf = getattr(inner, "privileged_obs_buf", None)
        self.rew_buf = inner.rew_buf
        self.reset_buf = inner.reset_buf
        self.episode_length_buf = inner.episode_length_buf
        self.extras = getattr(inner, "extras", {})

        self.base_pos: Optional[torch.Tensor] = None
        self.base_quat: Optional[torch.Tensor] = None
        self.dof_pos: Optional[torch.Tensor] = None
        self._bind_robot_state()

    @property
    def has_robot_state(self) -> bool:
        return env_has_robot_state(self)

    def _bind_robot_state(self) -> None:
        for name in _ROBOT_STATE_NAMES:
            val = getattr(self._inner, name, None)
            if torch.is_tensor(val):
                setattr(self, name, val)
            else:
                setattr(self, name, None)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        return self._inner.step(actions)

    def get_observations(self) -> torch.Tensor:
        return self._inner.get_observations()

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        return self._inner.get_privileged_observations()

    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]) -> torch.Tensor:
        """子类按内层 env 的 reset API 实现。"""
