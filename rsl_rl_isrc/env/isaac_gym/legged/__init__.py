# rsl_rl_isrc 内置 Isaac Gym 足式环境（G1/H1 物理模型在 robotmodel/*_description）。

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_DIR = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
_RSL_RL_ISRC_ROOT = os.path.abspath(os.path.join(_ENV_DIR, ".."))
ROBOTMODEL_DIR = os.path.join(_RSL_RL_ISRC_ROOT, "robotmodel")

G1_DESCRIPTION_DIR = os.path.join(ROBOTMODEL_DIR, "g1_description")
G1_URDF_PATH = os.path.join(G1_DESCRIPTION_DIR, "g1_12dof.urdf")
G1_XML_PATH = os.path.join(G1_DESCRIPTION_DIR, "g1_12dof.xml")
G1_MESH_DIR = os.path.join(G1_DESCRIPTION_DIR, "meshes")

H1_DESCRIPTION_DIR = os.path.join(ROBOTMODEL_DIR, "h1_description")
H1_12DOF_URDF_PATH = os.path.join(H1_DESCRIPTION_DIR, "h1_2_12dof.urdf")
H1_10DOF_URDF_PATH = os.path.join(H1_DESCRIPTION_DIR, "h1_10dof.urdf")
H1_MESH_DIR = os.path.join(H1_DESCRIPTION_DIR, "meshes")


def ensure_g1_meshes() -> str:
    """要求 ``robotmodel/g1_description/meshes`` 存在（含 URDF/XML 引用的 STL）。"""
    if not os.path.isdir(G1_MESH_DIR):
        raise FileNotFoundError(
            f"缺少 mesh 目录: {G1_MESH_DIR}\n"
            "请将 g1_description 的 meshes（*.STL）放入该目录。"
        )
    return G1_MESH_DIR


def ensure_g1_urdf() -> str:
    """校验 mesh 与 ``g1_12dof.urdf``，返回绝对路径（Isaac Gym 训练默认）。"""
    ensure_g1_meshes()
    if not os.path.isfile(G1_URDF_PATH):
        raise FileNotFoundError(f"缺少 G1 URDF: {G1_URDF_PATH}")
    return os.path.abspath(G1_URDF_PATH)


def ensure_g1_xml() -> str:
    """校验 mesh 与 ``g1_12dof.xml``，返回绝对路径（Mujoco 等）。"""
    ensure_g1_meshes()
    if not os.path.isfile(G1_XML_PATH):
        raise FileNotFoundError(f"缺少 G1 MJCF: {G1_XML_PATH}")
    return os.path.abspath(G1_XML_PATH)


def ensure_g1_robot_assets() -> str:
    """兼容旧名：Isaac 训练默认返回 ``g1_12dof.urdf`` 路径。"""
    return ensure_g1_urdf()


def ensure_h1_meshes() -> str:
    """要求 ``robotmodel/h1_description/meshes`` 存在。"""
    if not os.path.isdir(H1_MESH_DIR):
        raise FileNotFoundError(
            f"缺少 mesh 目录: {H1_MESH_DIR}\n"
            "请将 h1_description 的 meshes（*.STL）放入该目录。"
        )
    return H1_MESH_DIR


def _generate_h1_10dof_urdf() -> None:
    if not os.path.isfile(H1_12DOF_URDF_PATH):
        raise FileNotFoundError(f"缺少 H1 12DOF 源 URDF: {H1_12DOF_URDF_PATH}")

    tree = ET.parse(H1_12DOF_URDF_PATH)
    root = tree.getroot()
    ankle_roll_joints = {"left_ankle_roll_joint", "right_ankle_roll_joint"}
    for joint in root.findall("joint"):
        if joint.attrib.get("name") not in ankle_roll_joints:
            continue
        joint.set("type", "fixed")
        for child_name in ("axis", "limit", "dynamics"):
            child = joint.find(child_name)
            if child is not None:
                joint.remove(child)
    tree.write(H1_10DOF_URDF_PATH, encoding="utf-8", xml_declaration=True)


def ensure_h1_10dof_urdf() -> str:
    """生成并校验 ``h1_10dof.urdf``，返回绝对路径。"""
    ensure_h1_meshes()
    if not os.path.isfile(H1_10DOF_URDF_PATH):
        _generate_h1_10dof_urdf()
    return os.path.abspath(H1_10DOF_URDF_PATH)
