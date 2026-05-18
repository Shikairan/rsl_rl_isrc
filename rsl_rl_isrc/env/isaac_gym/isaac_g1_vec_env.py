# 将 rsl_rl_isrc 内置 Isaac G1 环境适配为 VecEnv。

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from rsl_rl_isrc.env import VecEnv


class IsaacG1VecEnv(VecEnv):
    """包装 ``G1Robot``（``rsl_rl_isrc.env.isaac_gym.legged``），满足 ``OnPolicyRunner`` / ``VecEnv`` 协议。

    主要修复 ``BaseTask.reset()`` 无 ``env_ids`` 参数与 Runner 调用方式不一致的问题。
    """

    def __init__(self, env) -> None:
        self._env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_privileged_obs = env.num_privileged_obs
        self.num_actions = env.num_actions
        self.max_episode_length = int(env.max_episode_length)
        self.device = env.device

        self.obs_buf = env.obs_buf
        self.privileged_obs_buf = env.privileged_obs_buf
        self.rew_buf = env.rew_buf
        self.reset_buf = env.reset_buf
        self.episode_length_buf = env.episode_length_buf
        self.extras = env.extras

    def step(self, actions: torch.Tensor) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        return self._env.step(actions)

    def reset(self, env_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return self.get_observations()
        if len(env_ids) == self.num_envs:
            obs, _ = self._env.reset()
            return obs
        self._env.reset_idx(env_ids)
        self._env.episode_length_buf[env_ids] = 0
        self._env.compute_observations()
        return self.get_observations()

    def get_observations(self) -> torch.Tensor:
        return self._env.get_observations()

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        return self._env.get_privileged_observations()

    def __getattr__(self, name: str):
        """透传 ``base_pos`` / ``dof_pos`` 等，供 Runner 遥测与调试。"""
        return getattr(self._env, name)
