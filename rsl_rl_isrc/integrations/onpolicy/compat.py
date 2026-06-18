# rsl_rl_isrc — 官方 onpolicy 在较新 Python 上的兼容垫片（不修改 onpolicy 源码）。
#
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""Python 3.12+ 等环境下为 onpolicy MPE 等提供 ``imp`` 模块垫片。"""

from __future__ import annotations

import importlib.util
import sys
import types


def ensure_onpolicy_compat() -> None:
    """在导入 onpolicy MPE 场景脚本前调用。

    onpolicy 的 ``scenarios/__init__.py`` 使用已移除的 ``imp.load_source``；
    此处注入最小 ``imp`` 垫片，避免修改第三方源码。
    """
    if "imp" in sys.modules:
        return

    imp = types.ModuleType("imp")

    def load_source(name: str, pathname: str):
        module_name = name or "_onpolicy_scenario"
        spec = importlib.util.spec_from_file_location(module_name, pathname)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载场景模块: {pathname}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    imp.load_source = load_source  # type: ignore[attr-defined]
    sys.modules["imp"] = imp
