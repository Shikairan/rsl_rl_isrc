# rsl_rl_isrc.isrcgym — 标准 Gym 风格 API 的薄封装（底层为 gymnasium）。
# License: BSD-3-Clause
#
"""测试与示例可 ``import rsl_rl_isrc.isrcgym as gym``，避免在业务代码中散落第三方包名。"""
import gymnasium as _g

make = _g.make
vector = _g.vector
spaces = _g.spaces


def __getattr__(name: str):
    return getattr(_g, name)
