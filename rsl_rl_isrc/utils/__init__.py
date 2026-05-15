# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""轨迹处理、TRPO 数值工具与 ``RunningMeanStd`` 观测归一化导出。"""

from .utils import (
    split_and_pad_trajectories,
    unpad_trajectories,
    normal_entropy,
    normal_log_density,
    get_flat_params_from,
    set_flat_params_to,
    get_flat_grad_from,
    conjugate_gradients,
    linesearch,
    RunningMeanStd
)