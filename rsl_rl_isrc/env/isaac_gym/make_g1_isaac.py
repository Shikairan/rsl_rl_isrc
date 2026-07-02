# rsl_rl_isrc：Isaac Gym G1 环境工厂与 PPO 训练配置（仅依赖 rsl_rl_isrc + isaacgym）。

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl_isrc.env.isaac_gym.isaac_g1_vec_env import IsaacG1VecEnv


def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _apply_numpy_isaac_compat() -> None:
    compat_path = os.path.join(os.path.dirname(__file__), "numpy_compat.py")
    mod = _load_module_from_file("_isaac_numpy_compat", compat_path)
    mod.apply_numpy_isaac_compat()


def _import_isaac():
    _apply_numpy_isaac_compat()
    import isaacgym  # noqa: F401
    from isaacgym import gymapi
    return gymapi


def _make_sim_args(sim_device: str, headless: bool, gymapi) -> SimpleNamespace:
    use_gpu = sim_device.startswith("cuda")
    device_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
    return SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=use_gpu,
        use_gpu_pipeline=use_gpu,
        subscenes=0,
        num_threads=0,
        device="cuda" if use_gpu else "cpu",
        sim_device=sim_device,
        sim_device_type="cuda" if use_gpu else "cpu",
        sim_device_id=device_id,
        compute_device_id=device_id,
        graphics_device_id=device_id,
        headless=headless,
        num_envs=None,
    )


def _class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if isinstance(val, list):
            result[key] = [_class_to_dict(item) for item in val]
        else:
            result[key] = _class_to_dict(val)
    return result


def normalize_train_cfg(raw: dict) -> dict:
    train_cfg = {
        "seed": raw.get("seed", 1),
        "runner": dict(raw["runner"]),
        "algorithm": dict(raw["algorithm"]),
        "policy": dict(raw["policy"]),
    }
    runner = train_cfg["runner"]
    if "policy_class_name" in runner:
        train_cfg["policy"]["policy_class_name"] = runner["policy_class_name"]
    if "algorithm_class_name" in runner:
        train_cfg["algorithm"]["algorithm_class_name"] = runner["algorithm_class_name"]
    return train_cfg


def build_g1_ppo_train_cfg() -> dict:
    from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_config import G1RoughCfgPPO

    return normalize_train_cfg(_class_to_dict(G1RoughCfgPPO()))


def default_g1_log_dir(train_cfg: dict, log_root: str | None = None) -> str:
    from rsl_rl_isrc.utils.paths import build_run_log_dir

    return build_run_log_dir(train_cfg, log_root=log_root)


def default_g1_checkpoint_dir(
    train_cfg: dict,
    checkpoint_root: str | None = None,
    run_suffix_value: str | None = None,
) -> str:
    from rsl_rl_isrc.utils.paths import build_run_checkpoint_dir

    return build_run_checkpoint_dir(
        train_cfg,
        checkpoint_root=checkpoint_root,
        run_suffix_value=run_suffix_value,
    )


def default_g1_run_dirs(
    train_cfg: dict,
    log_root: str | None = None,
    checkpoint_root: str | None = None,
) -> tuple[str, str]:
    from rsl_rl_isrc.utils.paths import build_run_dirs

    return build_run_dirs(train_cfg, log_root=log_root, checkpoint_root=checkpoint_root)


def make_g1_isaac_env(
    num_envs: int = 64,
    headless: bool = True,
    sim_device: str = "cuda:0",
) -> Tuple["IsaacG1VecEnv", object, dict]:
    """创建 G1 Isaac Gym 向量环境（rsl_rl_isrc 内置 ``legged`` 实现，单进程）。"""
    gymapi = _import_isaac()

    from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_config import G1RoughCfg
    from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_env import G1Robot
    from rsl_rl_isrc.env.isaac_gym.legged.utils.helpers import parse_sim_params, set_seed
    from rsl_rl_isrc.env.isaac_gym.isaac_g1_vec_env import IsaacG1VecEnv

    from rsl_rl_isrc.env.isaac_gym.legged import ensure_g1_urdf

    cfg = G1RoughCfg()
    cfg.env.num_envs = int(num_envs)
    cfg.asset.file = ensure_g1_urdf()

    train_cfg = build_g1_ppo_train_cfg()
    set_seed(train_cfg.get("seed", 1))

    args = _make_sim_args(sim_device, headless, gymapi)
    sim_params = parse_sim_params(args, {"sim": _class_to_dict(cfg.sim)})

    robot = G1Robot(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=sim_device,
        headless=headless,
    )
    env = IsaacG1VecEnv(robot)
    return env, cfg, train_cfg


_ISAACGYM_IMPORT_ERROR: str | None = None


def has_isaac_gym() -> bool:
    if "isaacgym" in sys.modules:
        return True
    try:
        _apply_numpy_isaac_compat()
        import isaacgym  # noqa: F401
        return True
    except ImportError as exc:
        global _ISAACGYM_IMPORT_ERROR
        _ISAACGYM_IMPORT_ERROR = str(exc)
        return False


def isaac_gym_import_error() -> str | None:
    return _ISAACGYM_IMPORT_ERROR
