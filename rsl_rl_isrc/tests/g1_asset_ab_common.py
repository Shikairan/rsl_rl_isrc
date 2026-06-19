# 独立 A/B 工具：复用 rsl_rl_isrc 的 G1 Isaac 环境，不修改 make_g1_isaac 等现有模块。
"""G1 URDF vs XML 验证共用逻辑（仅供 tests/g1_asset_ab_*.py 引用）。"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Literal, Tuple

AssetKind = Literal["urdf", "xml", "current"]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

# 与 legged/__init__.py 一致的路径常量（不 import rsl_rl，避免先于 isaacgym 拉入 torch）
_RSL_RL_ISRC_ROOT = _PACKAGE_ROOT
G1_DESCRIPTION_DIR = os.path.join(_RSL_RL_ISRC_ROOT, "robotmodel", "g1_description")
G1_XML_PATH = os.path.join(G1_DESCRIPTION_DIR, "g1_29dof.xml")
G1_URDF_PATH = os.path.join(G1_DESCRIPTION_DIR, "g1_29dof.urdf")
G1_MESH_DIR = os.path.join(G1_DESCRIPTION_DIR, "meshes")


def ensure_project_on_path() -> None:
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def apply_numpy_isaac_compat() -> None:
    """与 test_ppo_g1_isaac 相同：先打 numpy 补丁再 import isaacgym。"""
    compat_path = os.path.join(
        _PACKAGE_ROOT, "env", "isaac_gym", "numpy_compat.py"
    )
    spec = importlib.util.spec_from_file_location("_isaac_numpy_compat_ab", compat_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {compat_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_numpy_isaac_compat()


def check_isaac_cuda() -> None:
    apply_numpy_isaac_compat()
    try:
        import isaacgym  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"无法加载 isaacgym: {exc}") from exc
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（cuda:0）")


def _ensure_g1_meshes() -> None:
    if not os.path.isdir(G1_MESH_DIR):
        raise FileNotFoundError(
            f"缺少 mesh 目录: {G1_MESH_DIR}\n"
            "请将 g1_description 的 meshes（*.STL）放入该目录。"
        )


def get_g1_asset_paths(*, setup_meshes: bool = False) -> dict[str, str]:
    """返回各资产绝对路径（不创建环境、不 import torch）。"""
    if setup_meshes:
        _ensure_g1_meshes()
    return {
        "xml_robotmodel": G1_XML_PATH,
        "xml_current": G1_XML_PATH,
        "urdf_robotmodel": G1_URDF_PATH,
    }


def resolve_asset_file(kind: AssetKind, *, setup_meshes: bool = True) -> str:
    """解析训练用 asset.file（绝对路径）。"""
    if setup_meshes:
        _ensure_g1_meshes()
    paths = get_g1_asset_paths()
    if kind in ("xml", "current"):
        if not os.path.isfile(paths["xml_current"]):
            raise FileNotFoundError(paths["xml_current"])
        return paths["xml_current"]
    if kind == "urdf":
        p = paths["urdf_robotmodel"]
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return p
    raise ValueError(f"未知 asset kind: {kind}")


def make_g1_env_with_asset(
    asset_kind: AssetKind,
    *,
    num_envs: int = 64,
    headless: bool = True,
    sim_device: str = "cuda:0",
    seed: int | None = None,
):
    """创建 G1 环境；asset_kind 指定 robotmodel 下 urdf 或 xml。"""
    from types import SimpleNamespace

    from rsl_rl_isrc.env.isaac_gym.make_g1_isaac import (
        _class_to_dict,
        _import_isaac,
        _make_sim_args,
        build_g1_ppo_train_cfg,
    )
    from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_config import G1RoughCfg
    from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_env import G1Robot
    from rsl_rl_isrc.env.isaac_gym.legged.utils.helpers import parse_sim_params, set_seed
    from rsl_rl_isrc.env.isaac_gym.isaac_g1_vec_env import IsaacG1VecEnv

    gymapi = _import_isaac()
    cfg = G1RoughCfg()
    cfg.env.num_envs = int(num_envs)
    cfg.asset.file = resolve_asset_file(asset_kind)

    train_cfg = build_g1_ppo_train_cfg()
    if seed is not None:
        train_cfg["seed"] = int(seed)
        set_seed(int(seed))

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
    return env, cfg, train_cfg, robot


def format_robot_summary(robot, asset_path: str) -> str:
    """打印 Isaac 加载后的 DOF / body 摘要。"""
    lines = [
        f"asset.file = {asset_path}",
        f"num_envs   = {robot.num_envs}",
        f"num_dof    = {robot.num_dof}",
        f"num_bodies = {robot.num_bodies}",
        f"dof_names  = {list(robot.dof_names)}",
        f"num_actions = {getattr(robot, 'num_actions', '?')}",
    ]
    if hasattr(robot, "dof_pos_limits"):
        lim = robot.dof_pos_limits[0].cpu().tolist()
        lines.append(f"dof_pos_limits[0] ({robot.dof_names[0]}) = {lim}")
        ri = None
        for i, n in enumerate(robot.dof_names):
            if "right_hip_roll" in n:
                ri = i
                break
        if ri is not None:
            lines.append(
                f"dof_pos_limits[{ri}] ({robot.dof_names[ri]}) = "
                f"{robot.dof_pos_limits[ri].cpu().tolist()}"
            )
        import math

        lims = robot.dof_pos_limits.cpu().tolist()
        n_inf = sum(
            1
            for lo, hi in lims
            if math.isinf(lo) or math.isinf(hi)
        )
        if n_inf:
            lines.append(
                f"WARNING: {n_inf}/{robot.num_dof} 关节限位为 inf "
                f"（MJCF 常见；URDF 通常有有限限位）"
            )
    return "\n".join(lines)


def read_tensorboard_scalars(
    log_dir: str,
    tags: tuple[str, ...] = ("Train/mean_reward", "Train/mean_episode_length"),
) -> dict[str, list[tuple[int, float]]]:
    """从 log_dir 读取标量曲线；无事件文件时返回空列表。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {t: [] for t in tags}

    if not os.path.isdir(log_dir):
        return {t: [] for t in tags}

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in tags:
        if tag not in ea.Tags().get("scalars", []):
            out[tag] = []
            continue
        events = ea.Scalars(tag)
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def scalar_at_step(series: list[tuple[int, float]], step: int) -> float | None:
    """取最接近 step 的标量值（用于对比 iter=100 时的 mean_reward）。"""
    if not series:
        return None
    best = min(series, key=lambda p: abs(p[0] - step))
    return best[1]


def compare_ab_results(
    results: dict[str, dict],
    *,
    checkpoint_iters: tuple[int, ...] = (50, 100, 150, 200),
) -> str:
    """根据各 run 的 tensorboard 曲线生成文本对比表。"""
    lines = ["", "=" * 72, "G1 资产 A/B 对比（Train/mean_reward）", "=" * 72]
    header = f"{'iter':>6}"
    for name in results:
        header += f"  {name:>12}"
    lines.append(header)
    lines.append("-" * 72)

    for it in checkpoint_iters:
        row = f"{it:>6}"
        for name, data in results.items():
            curves = data.get("scalars", {})
            rew = curves.get("Train/mean_reward", [])
            v = scalar_at_step(rew, it)
            row += f"  {v if v is not None else 'n/a':>12}"
        lines.append(row)

    lines.append("-" * 72)
    for name, data in results.items():
        curves = data.get("scalars", {})
        rew = curves.get("Train/mean_reward", [])
        if rew:
            last = rew[-1]
            lines.append(
                f"  [{name}] last logged iter={last[0]}, mean_reward={last[1]:.4f}, "
                f"log_dir={data.get('log_dir', '?')}"
            )
        else:
            lines.append(f"  [{name}] 无 TensorBoard 标量（可能步数过少）")
    lines.append("=" * 72)
    return "\n".join(lines)
