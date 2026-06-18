# rsl_rl_isrc — 官方 onpolicy 集成（仅 MARL 车道，与单智能体算法代码隔离）。
#
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""``onpolicy`` 依赖桥接：配置转换等。"""

from .config_bridge import resolve_run_dir, to_namespace

__all__ = ["to_namespace", "resolve_run_dir"]
