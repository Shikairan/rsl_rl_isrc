"""MuJoCo G1 VecEnv 包装。"""

from __future__ import annotations

from typing import List, Union

import torch

from rsl_rl_isrc.env.state_export_vec_env import StateExportVecEnv


class MujocoG1VecEnv(StateExportVecEnv):
    """包装 :class:`G1MujocoEnv`，满足 Runner / StateExport 协议。"""

    def reset(self, env_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return self.get_observations()
        inner = self._inner
        if len(env_ids) == self.num_envs:
            inner.reset(torch.arange(self.num_envs, device=self.device))
            return self.get_observations()
        inner.reset_idx(env_ids)
        inner.episode_length_buf[env_ids] = 0
        inner.compute_observations()
        return self.get_observations()
