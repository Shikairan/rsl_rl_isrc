"""MuJoCo 多环境并行仿真底层（CPU）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import mujoco
import numpy as np

# 与 MJCF actuator 顺序一致（scene.xml / g1_12dof.xml）
G1_ACTUATED_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


@dataclass
class MujocoModelInfo:
    """缓存模型索引，避免每步重复查询。"""

    model: mujoco.MjModel
    joint_qpos_adr: np.ndarray  # (nu,)
    joint_qvel_adr: np.ndarray  # (nu,)
    joint_ids: np.ndarray  # (nu,)
    free_qpos_adr: int
    free_qvel_adr: int
    pelvis_body_id: int
    body_ids: Dict[str, int]
    actuator_forcerange: np.ndarray  # (nu, 2)
    joint_range: np.ndarray  # (nu, 2)


def resolve_g1_scene_xml() -> str:
    """返回 g1 scene.xml 绝对路径。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "robotmodel", "g1_description", "scene.xml")
    )


def load_model_info(xml_path: str | None = None) -> MujocoModelInfo:
    xml_path = xml_path or resolve_g1_scene_xml()
    xml_dir = os.path.dirname(xml_path)
    os.chdir(xml_dir)
    model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
    model.opt.timestep = 0.005

    joint_qpos_adr = np.zeros(len(G1_ACTUATED_JOINT_NAMES), dtype=np.int32)
    joint_qvel_adr = np.zeros(len(G1_ACTUATED_JOINT_NAMES), dtype=np.int32)
    joint_ids = np.zeros(len(G1_ACTUATED_JOINT_NAMES), dtype=np.int32)
    joint_range = np.zeros((len(G1_ACTUATED_JOINT_NAMES), 2), dtype=np.float64)
    actuator_forcerange = np.zeros((len(G1_ACTUATED_JOINT_NAMES), 2), dtype=np.float64)

    for i, jname in enumerate(G1_ACTUATED_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"关节未找到: {jname}")
        joint_ids[i] = jid
        joint_qpos_adr[i] = model.jnt_qposadr[jid]
        joint_qvel_adr[i] = model.jnt_dofadr[jid]
        joint_range[i] = model.jnt_range[jid]
        actuator_forcerange[i] = model.actuator_forcerange[i]

    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if free_jid < 0:
        raise RuntimeError("未找到 floating_base_joint")
    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_bid < 0:
        raise RuntimeError("未找到 pelvis body")

    body_ids: Dict[str, int] = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        body_ids[name] = i

    return MujocoModelInfo(
        model=model,
        joint_qpos_adr=joint_qpos_adr,
        joint_qvel_adr=joint_qvel_adr,
        joint_ids=joint_ids,
        free_qpos_adr=int(model.jnt_qposadr[free_jid]),
        free_qvel_adr=int(model.jnt_dofadr[free_jid]),
        pelvis_body_id=pelvis_bid,
        body_ids=body_ids,
        actuator_forcerange=actuator_forcerange,
        joint_range=joint_range,
    )


class MujocoSimPool:
    """N 个独立 MjData 的并行仿真池。"""

    def __init__(self, num_envs: int, xml_path: str | None = None) -> None:
        self.num_envs = int(num_envs)
        self.info = load_model_info(xml_path)
        self.model = self.info.model
        self.data_list: List[mujoco.MjData] = [
            mujoco.MjData(self.model) for _ in range(self.num_envs)
        ]
        self._contact_buf = np.zeros(6, dtype=np.float64)

    def step_env(self, env_id: int, ctrl: np.ndarray) -> None:
        data = self.data_list[env_id]
        data.ctrl[:] = np.clip(
            ctrl,
            self.info.actuator_forcerange[:, 0],
            self.info.actuator_forcerange[:, 1],
        )
        mujoco.mj_step(self.model, data)

    def step_all(self, ctrl_batch: np.ndarray) -> None:
        """ctrl_batch: (num_envs, nu)"""
        for i in range(self.num_envs):
            self.step_env(i, ctrl_batch[i])

    def reset_env(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetData(self.model, data)

    def read_root_state(self, env_id: int) -> np.ndarray:
        """返回 (13,) = pos3 quat4 lin_vel3 ang_vel3（quat wxyz MuJoCo 原生）。"""
        data = self.data_list[env_id]
        adr = self.info.free_qpos_adr
        vadr = self.info.free_qvel_adr
        qpos = data.qpos[adr : adr + 7]
        qvel = data.qvel[vadr : vadr + 6]
        # MuJoCo quat (w,x,y,z) -> 内部统一 (x,y,z,w)
        quat_wxyz = qpos[3:7]
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )
        out = np.zeros(13, dtype=np.float64)
        out[0:3] = qpos[0:3]
        out[3:7] = quat_xyzw
        out[7:10] = qvel[0:3]
        out[10:13] = qvel[3:6]
        return out

    def write_root_state(self, env_id: int, root_state: np.ndarray) -> None:
        data = self.data_list[env_id]
        adr = self.info.free_qpos_adr
        vadr = self.info.free_qvel_adr
        data.qpos[adr : adr + 3] = root_state[0:3]
        quat_xyzw = root_state[3:7]
        data.qpos[adr + 3 : adr + 7] = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
        data.qvel[vadr : vadr + 3] = root_state[7:10]
        data.qvel[vadr + 3 : vadr + 6] = root_state[10:13]
        mujoco.mj_forward(self.model, data)

    def read_dof_pos_vel(self, env_id: int) -> tuple[np.ndarray, np.ndarray]:
        data = self.data_list[env_id]
        pos = np.zeros(len(G1_ACTUATED_JOINT_NAMES), dtype=np.float64)
        vel = np.zeros(len(G1_ACTUATED_JOINT_NAMES), dtype=np.float64)
        for i, (padr, vadr) in enumerate(
            zip(self.info.joint_qpos_adr, self.info.joint_qvel_adr)
        ):
            pos[i] = data.qpos[padr]
            vel[i] = data.qvel[vadr]
        return pos, vel

    def write_dof_pos_vel(
        self, env_id: int, pos: np.ndarray, vel: np.ndarray
    ) -> None:
        data = self.data_list[env_id]
        for i, (padr, vadr) in enumerate(
            zip(self.info.joint_qpos_adr, self.info.joint_qvel_adr)
        ):
            data.qpos[padr] = pos[i]
            data.qvel[vadr] = vel[i]
        mujoco.mj_forward(self.model, data)

    def read_body_pos_vel(self, env_id: int, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self.info.body_ids[body_name]
        data = self.data_list[env_id]
        pos = np.array(data.xpos[bid], dtype=np.float64)
        vel = np.array(data.cvel[bid, 3:6], dtype=np.float64)
        return pos, vel

    def contact_forces_on_bodies(
        self, env_id: int, body_ids: Sequence[int]
    ) -> np.ndarray:
        """返回 (len(body_ids), 3) 接触力近似（世界系）。"""
        data = self.data_list[env_id]
        forces = np.zeros((len(body_ids), 3), dtype=np.float64)
        bid_to_idx = {bid: i for i, bid in enumerate(body_ids)}
        for ci in range(data.ncon):
            mujoco.mj_contactForce(self.model, data, ci, self._contact_buf)
            f = self._contact_buf[0:3]
            g1, g2 = int(data.contact[ci].geom1), int(data.contact[ci].geom2)
            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])
            if b1 in bid_to_idx:
                forces[bid_to_idx[b1]] += f
            if b2 in bid_to_idx:
                forces[bid_to_idx[b2]] -= f
        return forces
