"""Isaac Gym 导入引导：补全 sys.path、校验 Python 版本。"""

from __future__ import annotations

import os
import sys

_MAX_SUPPORTED_PY = (3, 8)
_ISAACGYM_DIRNAME = os.path.join("isaacgym", "python")


def _candidate_isaacgym_python_dirs() -> list[str]:
    candidates: list[str] = []
    env_dir = os.environ.get("ISAACGYM_PYTHON_DIR", "").strip()
    if env_dir:
        candidates.append(os.path.abspath(env_dir))

    seen: set[str] = set()

    def _add(path: str) -> None:
        abspath = os.path.abspath(path)
        if abspath not in seen and os.path.isdir(abspath):
            seen.add(abspath)
            candidates.append(abspath)

    for anchor in (os.getcwd(), *sys.path):
        if not anchor:
            continue
        cur = os.path.abspath(anchor)
        for _ in range(8):
            _add(os.path.join(cur, _ISAACGYM_DIRNAME))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return candidates


def _prepend_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_isaacgym_importable(*, apply_numpy_compat) -> None:
    """在 ``import torch`` 之前调用，确保 ``import isaacgym`` 可用。"""
    py = sys.version_info

    if py[:2] > _MAX_SUPPORTED_PY:
        raise RuntimeError(
            f"当前 Python {py.major}.{py.minor} 不受 Isaac Gym 支持（最高 3.8）。\n"
            f"请切换环境后重试，例如：conda activate rsl_isrc\n"
            f"当前解释器: {sys.executable}"
        )

    apply_numpy_compat()

    candidates = _candidate_isaacgym_python_dirs()
    isaac_pkg = os.path.join("isaacgym", "__init__.py")
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, isaac_pkg)):
            _prepend_sys_path(candidate)
            break

    try:
        import isaacgym  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "无法加载 isaacgym（No module named 'isaacgym'）。\n"
            "请确认已安装 Isaac Gym，或设置：\n"
            "  export ISAACGYM_PYTHON_DIR=/path/to/isaacgym/python\n"
            "推荐环境：conda activate rsl_isrc"
        ) from exc
    except RuntimeError as exc:
        if "PyTorch was imported before isaacgym" in str(exc):
            raise RuntimeError(
                f"{exc}\n"
                "请确保在 import torch 之前完成 isaacgym 引导（脚本入口应最先调用 bootstrap）。"
            ) from exc
        raise RuntimeError(
            f"{exc}\n"
            "Isaac Gym 预编译绑定仅支持 Python <= 3.8，请使用：conda activate rsl_isrc"
        ) from exc
    except ImportError as exc:
        if "PyTorch was imported before isaacgym" in str(exc):
            raise RuntimeError(
                f"{exc}\n"
                "请确保在 import torch 之前完成 isaacgym 引导。"
            ) from exc
        raise
