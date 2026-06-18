# rsl_rl_isrc — MARL 环境入口（车道 2）：与单智能体 ``VecEnv`` 并列，不经统一协议。
#
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""多智能体环境：``MarlEnv`` 句柄与 ``make_marl_env`` 工厂（底层委托官方 onpolicy 环境）。"""

from .env_factory import SUPPORTED_MARL_ENV_NAMES, make_marl_env, make_marl_env_from_name
from .marl_env import MarlEnv

__all__ = [
    "MarlEnv",
    "make_marl_env",
    "make_marl_env_from_name",
    "SUPPORTED_MARL_ENV_NAMES",
]
