# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""导出并行向量环境抽象 ``VecEnv``、MARL 环境入口与状态导出包装。"""

from .marl import MarlEnv, SUPPORTED_MARL_ENV_NAMES, make_marl_env, make_marl_env_from_name
from .state_export_vec_env import StateExportVecEnv, env_has_robot_state
from .vec_env import VecEnv

__all__ = [
    "VecEnv",
    "MarlEnv",
    "make_marl_env",
    "make_marl_env_from_name",
    "SUPPORTED_MARL_ENV_NAMES",
    "StateExportVecEnv",
    "env_has_robot_state",
]