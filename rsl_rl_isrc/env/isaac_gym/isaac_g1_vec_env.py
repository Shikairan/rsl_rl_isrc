# 将 rsl_rl_isrc 内置 Isaac G1 环境适配为 VecEnv，并显式导出 base_pos / base_quat / dof_pos。

from __future__ import annotations

from typing import List, Union

import torch

from rsl_rl_isrc.env.state_export_vec_env import StateExportVecEnv


class IsaacG1VecEnv(StateExportVecEnv):
    """包装 ``G1Robot``，满足 ``OnPolicyRunner`` / ``VecEnv`` 协议。

    主要修复 ``BaseTask.reset()`` 无 ``env_ids`` 参数与 Runner 调用方式不一致的问题。
  机器人状态经 :class:`StateExportVecEnv` 显式暴露，供 ``StepObsPublisher`` 遥测。
    """

    def reset(self, env_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return self.get_observations()
        if len(env_ids) == self.num_envs:
            obs, _ = self._inner.reset()
            return obs
        self._inner.reset_idx(env_ids)
        self._inner.episode_length_buf[env_ids] = 0
        self._inner.compute_observations()
        return self.get_observations()
