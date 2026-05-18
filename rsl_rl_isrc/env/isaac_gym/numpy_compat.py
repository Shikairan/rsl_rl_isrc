# NumPy 1.24+ 移除了 np.float 等别名；Isaac Gym torch_utils 仍依赖它们。

from __future__ import annotations

# Isaac Gym torch_utils 默认参数使用 np.float（NumPy 1.24+ 已移除）
_ALIASES = (("float", float),)


def apply_numpy_isaac_compat() -> None:
    """在 ``import isaacgym`` 之前调用，恢复 Isaac Gym 所需的 numpy 别名。"""
    import numpy as np

    for name, fallback in _ALIASES:
        if not hasattr(np, name):
            setattr(np, name, fallback)
