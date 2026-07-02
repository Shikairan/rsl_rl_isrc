"""检查点路径解析（无 Isaac Gym 依赖）。"""

from __future__ import annotations

import os


def get_load_path(root: str, load_run: str | int = -1, checkpoint: int = -1) -> str:
    """解析训练日志目录下的模型 ``.pt`` 路径。

    Args:
        root: 实验日志根目录（含多个 run 子目录）。
        load_run: run 子目录名；``-1`` 表示取排序后最后一个。
        checkpoint: 检查点编号；``-1`` 表示取最新 ``model_*.pt``。
    """
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except OSError as exc:
        raise ValueError(f"No runs in this directory: {root}") from exc

    if load_run == -1:
        load_run_path = last_run
    else:
        load_run_path = os.path.join(root, str(load_run))

    print("load_run:", load_run_path)
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run_path) if "model" in file]
        models.sort(key=lambda m: f"{m:0>15}")
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    return os.path.join(load_run_path, model)
