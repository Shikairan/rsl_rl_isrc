"""G1 MuJoCo 环境工厂。"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl_isrc.env.mujoco.mujoco_g1_vec_env import MujocoG1VecEnv

_MUJOCO_IMPORT_ERROR: str | None = None


def has_mujoco() -> bool:
    global _MUJOCO_IMPORT_ERROR
    if "mujoco" in sys.modules:
        return True
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError as exc:
        _MUJOCO_IMPORT_ERROR = str(exc)
        return False


def mujoco_import_error() -> str:
    return _MUJOCO_IMPORT_ERROR or ""


def build_g1_ppo_train_cfg() -> dict:
    from rsl_rl_isrc.env.mujoco.g1_mujoco_config import build_g1_ppo_train_cfg as _build
    return _build()


def default_g1_mujoco_log_dir(train_cfg: dict, log_root: str | None = None) -> str:
    if log_root is None:
        log_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs",
            train_cfg["runner"]["experiment_name"],
        )
    run_name = train_cfg["runner"].get("run_name", "")
    stamp = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join(log_root, f"{stamp}_{run_name}")


def make_g1_mujoco_env(
    num_envs: int = 64,
    sim_device: str = "cpu",
    seed: int | None = None,
) -> Tuple["MujocoG1VecEnv", object, dict]:
    """创建 G1 MuJoCo 向量环境（CPU 仿真）。"""
    if not has_mujoco():
        raise ImportError(
            f"未安装 mujoco: {mujoco_import_error()}\n"
            "请执行: pip install 'mujoco>=3.1.0'"
        )

    from rsl_rl_isrc.env.mujoco.g1_mujoco_config import G1MujocoCfg, g1_scene_xml_path
    from rsl_rl_isrc.env.mujoco.g1_mujoco_env import G1MujocoEnv
    from rsl_rl_isrc.env.mujoco.mujoco_g1_vec_env import MujocoG1VecEnv

    cfg = G1MujocoCfg()
    cfg.num_envs = int(num_envs)
    cfg.AssetCfg.file = g1_scene_xml_path()
    cfg.seed = int(seed) if seed is not None else int(getattr(cfg, "seed", 1))

    torch_device = sim_device if sim_device != "cpu" else "cpu"
    import torch
    if cfg.seed >= 0:
        import random
        import numpy as np
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    robot = G1MujocoEnv(cfg=cfg, device=torch_device)
    env = MujocoG1VecEnv(robot)
    train_cfg = build_g1_ppo_train_cfg()
    return env, cfg, train_cfg
