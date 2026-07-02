"""训练日志与 checkpoint 默认路径（可通过环境变量覆盖）。"""

from __future__ import annotations

import os
from datetime import datetime


def log_root_default() -> str:
    return os.environ.get("RSL_RL_ISRC_LOG_ROOT", "/var/log/rsl_rl_isrc")


def checkpoint_root_default() -> str:
    return os.environ.get("RSL_RL_ISRC_CHECKPOINT_ROOT", "/tmp/rsl_rl_isrc_checkpoints")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_suffix(train_cfg: dict) -> str:
    run_name = train_cfg["runner"].get("run_name", "")
    stamp = datetime.now().strftime("%b%d_%H-%M-%S")
    return f"{stamp}_{run_name}" if run_name else stamp


def build_run_log_dir(
    train_cfg: dict,
    log_root: str | None = None,
    run_suffix_value: str | None = None,
) -> str:
    """TensorBoard 事件目录：``<log_root>/<experiment>/<stamp>_<run_name>``。"""
    experiment = train_cfg["runner"]["experiment_name"]
    if log_root is None:
        root = ensure_dir(os.path.join(log_root_default(), experiment))
    else:
        root = ensure_dir(os.path.join(log_root, experiment))
    suffix = run_suffix_value if run_suffix_value is not None else run_suffix(train_cfg)
    return ensure_dir(os.path.join(root, suffix))


def build_run_checkpoint_dir(
    train_cfg: dict,
    checkpoint_root: str | None = None,
    run_suffix_value: str | None = None,
) -> str:
    """模型 checkpoint 目录：``<checkpoint_root>/<experiment>/<stamp>_<run_name>``。"""
    experiment = train_cfg["runner"]["experiment_name"]
    if checkpoint_root is None:
        root = ensure_dir(os.path.join(checkpoint_root_default(), experiment))
    else:
        root = ensure_dir(os.path.join(checkpoint_root, experiment))
    suffix = run_suffix_value if run_suffix_value is not None else run_suffix(train_cfg)
    return ensure_dir(os.path.join(root, suffix))


def build_run_dirs(
    train_cfg: dict,
    log_root: str | None = None,
    checkpoint_root: str | None = None,
) -> tuple[str, str]:
    """返回同一 run 的 ``(log_dir, checkpoint_dir)``（共享时间戳后缀）。"""
    suffix = run_suffix(train_cfg)
    log_dir = build_run_log_dir(train_cfg, log_root=log_root, run_suffix_value=suffix)
    checkpoint_dir = build_run_checkpoint_dir(
        train_cfg,
        checkpoint_root=checkpoint_root,
        run_suffix_value=suffix,
    )
    return log_dir, checkpoint_dir
