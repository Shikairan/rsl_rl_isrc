# rsl_rl_isrc 内置 Isaac Gym 足式环境（G1 物理模型在 robotmodel/g1_description）。

from __future__ import annotations

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_DIR = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
_RSL_RL_ISRC_ROOT = os.path.abspath(os.path.join(_ENV_DIR, ".."))

G1_DESCRIPTION_DIR = os.path.join(_RSL_RL_ISRC_ROOT, "robotmodel", "g1_description")
G1_URDF_PATH = os.path.join(G1_DESCRIPTION_DIR, "g1_29dof.urdf")
G1_XML_PATH = os.path.join(G1_DESCRIPTION_DIR, "g1_29dof.xml")
G1_MESH_DIR = os.path.join(G1_DESCRIPTION_DIR, "meshes")


def ensure_g1_meshes() -> str:
    """要求 ``robotmodel/g1_description/meshes`` 存在（含 URDF/XML 引用的 STL）。"""
    if not os.path.isdir(G1_MESH_DIR):
        raise FileNotFoundError(
            f"缺少 mesh 目录: {G1_MESH_DIR}\n"
            "请将 g1_description 的 meshes（*.STL）放入该目录。"
        )
    return G1_MESH_DIR


def ensure_g1_urdf() -> str:
    """校验 mesh 与 ``g1_29dof.urdf``，返回绝对路径（Isaac Gym 训练默认）。"""
    ensure_g1_meshes()
    if not os.path.isfile(G1_URDF_PATH):
        raise FileNotFoundError(f"缺少 G1 URDF: {G1_URDF_PATH}")
    return os.path.abspath(G1_URDF_PATH)


def ensure_g1_xml() -> str:
    """校验 mesh 与 ``g1_29dof.xml``，返回绝对路径（Mujoco 等）。"""
    ensure_g1_meshes()
    if not os.path.isfile(G1_XML_PATH):
        raise FileNotFoundError(f"缺少 G1 MJCF: {G1_XML_PATH}")
    return os.path.abspath(G1_XML_PATH)


def ensure_g1_robot_assets() -> str:
    """兼容旧名：Isaac 训练默认返回 ``g1_29dof.urdf`` 路径。"""
    return ensure_g1_urdf()
