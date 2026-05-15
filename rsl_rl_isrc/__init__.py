# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""顶层包：算法、运行器、网络与环境抽象均位于子模块，请按需 ``import rsl_rl_isrc.xxx``。"""


def __getattr__(name: str):
    if name == "isrcgym":
        from . import isrcgym as _isrcgym

        return _isrcgym
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
