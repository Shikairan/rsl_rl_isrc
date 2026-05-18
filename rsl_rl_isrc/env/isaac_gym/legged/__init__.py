# rsl_rl_isrc 内置 Isaac Gym 足式环境（参考 unitree_rl_gym，不 import 该包）。

from __future__ import annotations

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_DIR = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
_RSL_RL_ISRC_ROOT = os.path.abspath(os.path.join(_ENV_DIR, ".."))

LEGGED_GYM_ROOT_DIR = os.path.join(_ENV_DIR, "unitree_rl_gym")
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs")

G1_DESCRIPTION_DIR = os.path.join(_RSL_RL_ISRC_ROOT, "robotmodel", "g1_description")
G1_XML_PATH = os.path.join(G1_DESCRIPTION_DIR, "g1_12dof.xml")


def ensure_g1_robot_assets() -> str:
    """确保 ``robotmodel/g1_description`` 下模型与 mesh 可用，返回 ``g1_12dof.xml`` 路径。"""
    mesh_dir = os.path.join(G1_DESCRIPTION_DIR, "meshes")
    if not os.path.isdir(mesh_dir):
        fallback = os.path.join(
            LEGGED_GYM_ROOT_DIR, "resources", "robots", "g1_description", "meshes"
        )
        if os.path.isdir(fallback):
            os.symlink(fallback, mesh_dir)
    if not os.path.isfile(G1_XML_PATH):
        raise FileNotFoundError(f"缺少 G1 模型文件: {G1_XML_PATH}")
    return G1_XML_PATH
