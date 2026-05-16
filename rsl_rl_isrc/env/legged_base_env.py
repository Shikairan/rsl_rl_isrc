# 本文件与 rsl_rl_isrc 内其它业务子包完全独立，无交叉导入。
# 用途：为足式机器人仿真环境提供功能丰富的基类，继承自 BaseSimEnv。
# 参考：legged_gym LeggedRobot 风格，去除对特定物理引擎的直接依赖。
"""足式机器人并行仿真环境父类 ``LeggedBaseEnv``。

在 :class:`BaseSimEnv` 基础上补充了足式机器人所需的大量通用功能：

**运动控制**

* PD 控制器（位置/速度目标 → 力矩）
* 动作缩放、关节软限位惩罚

**状态观测**

* 基座线速度、角速度（世界/局部系）
* 重力投影方向（IMU-like）
* 关节位置、速度
* 脚部接触状态、脚部高度扫描
* 速度指令缩放向量

**奖励函数库**（``_reward_*`` 方法，自动注册）

- 速度跟踪：``tracking_lin_vel``、``tracking_ang_vel``
- 稳定性：``lin_vel_z``、``ang_vel_xy``、``orientation``
- 效率：``torques``、``dof_vel``、``dof_acc``、``action_rate``
- 接触：``feet_air_time``、``collision``、``feet_stumble``
- 姿态：``base_height``、``stand_still``

**指令采样**

* 均匀随机采样 (vx, vy, yaw_rate)
* 课程学习（根据跟踪成功率动态调整指令范围）

**域随机化**（可选）

* 地面摩擦系数随机化
* 根质量扰动
* 随机推力

快速示例::

    from rsl_rl_isrc.env.legged_base_env import LeggedBaseEnv
    from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg

    class AnymalCCfg(BaseEnvCfg):
        class asset(BaseEnvCfg.AssetCfg):
            file = "/path/to/anymal_c.urdf"
            foot_name = "FOOT"

    class AnymalCEnv(LeggedBaseEnv):
        # 只需实现物理引擎相关的 4 个抽象方法
        def _create_sim(self): ...
        def _create_ground(self): ...
        def _create_envs(self): ...
        def _init_buffers(self):
            super()._init_buffers_legged()  # 初始化所有足式机器人缓冲区
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg
from rsl_rl_isrc.env.base_sim_env import BaseSimEnv


class LeggedBaseEnv(BaseSimEnv):
    """足式机器人通用仿真环境父类。

    继承 :class:`BaseSimEnv`，新增足式机器人专用的：
    - 状态缓冲区（关节、基座、接触力等）
    - 观测构建与指令处理
    - PD 控制器
    - 完整奖励函数库
    - 课程学习与域随机化接口
    """

    def __init__(self, cfg: BaseEnvCfg, device: str = "cuda"):
        # 调用父类 __init__（触发完整构建流程）
        super().__init__(cfg, device)

    # ──────────────────────────────────────────────────────────────────────────
    # 子类应调用的缓冲区初始化（在 _init_buffers 内调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _init_buffers_legged(self) -> None:
        """初始化足式机器人专用的所有 PyTorch 张量缓冲区。

        子类在 ``_init_buffers`` 中调用此方法::

            def _init_buffers(self):
                # 先通过引擎 API 获取 self.num_dof, self.num_bodies 等
                # 再调用
                self._init_buffers_legged()
        """
        n, d = self.num_envs, self.device

        # ── 动作缓冲区 ────────────────────────────────────────────────────────
        self.actions      = torch.zeros(n, self.num_actions, device=d)
        self.last_actions = torch.zeros(n, self.num_actions, device=d)

        # ── 观测 ──────────────────────────────────────────────────────────────
        self.obs_buf = torch.zeros(n, self.num_obs, device=d)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(n, self.num_privileged_obs, device=d)

        # ── 奖励/终止 ─────────────────────────────────────────────────────────
        self.rew_buf   = torch.zeros(n, device=d)
        self.reset_buf = torch.ones(n, dtype=torch.bool, device=d)

        # ── 回合长度 ──────────────────────────────────────────────────────────
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=d)
        self.time_out_buf       = torch.zeros(n, dtype=torch.bool, device=d)

        # ── 根状态 (n, 13): [pos(3), quat(4), lin_vel(3), ang_vel(3)] ────────
        self.base_pos       = torch.zeros(n, 3, device=d)   # x y z
        self.base_quat      = torch.zeros(n, 4, device=d)   # qx qy qz qw
        self.base_lin_vel   = torch.zeros(n, 3, device=d)   # world frame
        self.base_ang_vel   = torch.zeros(n, 3, device=d)   # world frame
        self.base_lin_vel_body = torch.zeros(n, 3, device=d)  # body frame
        self.base_ang_vel_body = torch.zeros(n, 3, device=d)  # body frame
        self.projected_gravity = torch.zeros(n, 3, device=d)
        self.gravity_vec       = torch.tensor(
            [0.0, 0.0, -1.0], device=d).repeat(n, 1)

        # ── 关节状态 ──────────────────────────────────────────────────────────
        self.dof_pos       = torch.zeros(n, self.num_actions, device=d)
        self.dof_vel       = torch.zeros(n, self.num_actions, device=d)
        self.last_dof_vel  = torch.zeros(n, self.num_actions, device=d)
        self.torques       = torch.zeros(n, self.num_actions, device=d)
        self.default_dof_pos = self._get_default_dof_pos()

        # ── 关节限位缓冲区 ────────────────────────────────────────────────────
        self.dof_pos_limits = torch.zeros(self.num_actions, 2, device=d)  # [:,0]=low [:,1]=high
        self.dof_vel_limits = torch.zeros(self.num_actions, device=d)
        self.torque_limits  = torch.zeros(self.num_actions, device=d)
        # 子类需在 _create_envs 后调用 _init_dof_limits() 填充上述三个缓冲区

        # ── 接触力 (n, num_bodies, 3) ─────────────────────────────────────────
        num_bodies = getattr(self, "num_bodies", 1)
        self.contact_forces = torch.zeros(n, num_bodies, 3, device=d)

        # ── 脚部相关 ──────────────────────────────────────────────────────────
        # 子类设置 self.feet_indices (List[int]) 后自动分配
        feet_n = len(getattr(self, "feet_indices", [1]))
        self.feet_air_time      = torch.zeros(n, feet_n, device=d)
        self.last_contacts      = torch.zeros(n, feet_n, dtype=torch.bool, device=d)
        self.feet_contact_state = torch.zeros(n, feet_n, dtype=torch.bool, device=d)

        # ── 指令 (n, 4): [vx, vy, yaw_rate, heading] ─────────────────────────
        self.commands       = torch.zeros(n, self.cfg.CommandsCfg.num_commands, device=d)
        self.commands_scale = torch.ones(self.cfg.CommandsCfg.num_commands, device=d)

        # ── 高度扫描（可选）──────────────────────────────────────────────────
        self.measured_heights = torch.zeros(n, 1, device=d)

        # ── 域随机化缓冲区 ────────────────────────────────────────────────────
        self.friction_coefficients = torch.ones(n, device=d)
        self.added_mass = torch.zeros(n, device=d)

        # ── 观测缩放系数（可在子类中覆盖）──────────────────────────────────────
        self.lin_vel_scale  = 2.0
        self.ang_vel_scale  = 0.25
        self.dof_pos_scale  = 1.0
        self.dof_vel_scale  = 0.05

        # ── 初始化指令缩放向量 ────────────────────────────────────────────────
        self._update_command_curriculum_scale()

    def _get_default_dof_pos(self) -> torch.Tensor:
        """从配置中构建默认关节角度向量，子类可覆盖。"""
        return torch.zeros(self.num_actions, device=self.device)

    def _init_dof_limits(
        self,
        pos_limits: torch.Tensor,
        vel_limits: torch.Tensor,
        torque_limits: torch.Tensor,
    ) -> None:
        """从物理引擎 API 获取的关节限位张量写入缓冲区。

        参数
        ----
        pos_limits : (num_dof, 2) — 关节位置限位 [[lo, hi], ...]
        vel_limits : (num_dof,)   — 关节速度限位
        torque_limits : (num_dof,) — 关节力矩限位
        """
        self.dof_pos_limits[:] = pos_limits.to(self.device)
        self.dof_vel_limits[:] = vel_limits.to(self.device)
        self.torque_limits[:]  = torque_limits.to(self.device)

        # 应用软限位系数
        soft = self.cfg.RewardCfg.soft_dof_pos_limit
        m = (self.dof_pos_limits[:, 1] + self.dof_pos_limits[:, 0]) * 0.5
        h = (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) * 0.5 * soft
        self.dof_pos_limits_soft = torch.stack([m - h, m + h], dim=1)

    # ──────────────────────────────────────────────────────────────────────────
    # PD 控制器（子类调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """将策略动作（[-1,1] 归一化位置偏移量）转换为关节力矩。

        参数
        ----
        actions : (num_envs, num_actions) — 策略输出，将被缩放后加到默认关节角度

        返回
        ----
        torques : (num_envs, num_actions)
        """
        scale = self.cfg.ControlCfg.action_scale
        pos_target = actions * scale + self.default_dof_pos

        # 从配置构建 kp / kd 张量（按关节名称子串匹配）
        kp = self._build_gain_tensor("stiffness")
        kd = self._build_gain_tensor("damping")

        torques = kp * (pos_target - self.dof_pos) - kd * self.dof_vel
        torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
        return torques

    def _build_gain_tensor(self, gain_type: str) -> torch.Tensor:
        """从配置的 stiffness/damping 字典构建关节增益张量。
        仅在首次调用时构建并缓存，之后直接复用。
        """
        cache_attr = f"_cached_{gain_type}"
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        gains_cfg = (self.cfg.ControlCfg.stiffness
                     if gain_type == "stiffness"
                     else self.cfg.ControlCfg.damping)
        dof_names = getattr(self, "dof_names", [f"dof_{i}" for i in range(self.num_actions)])
        gains = torch.zeros(self.num_actions, device=self.device)
        for i, name in enumerate(dof_names):
            for key, val in gains_cfg.items():
                if key in name:
                    gains[i] = val
                    break
        setattr(self, cache_attr, gains)
        return gains

    # ──────────────────────────────────────────────────────────────────────────
    # 默认观测构建（子类可覆盖 compute_observations）
    # ──────────────────────────────────────────────────────────────────────────

    def _build_default_obs(self) -> torch.Tensor:
        """构建标准足式机器人观测向量（48 维 legged_gym 默认格式）。

        维度组成（共 48）::

            base_ang_vel_body (3) + projected_gravity (3) + commands_scaled (3)
            + dof_pos_rel (num_dof) + dof_vel_scaled (num_dof) + actions (num_dof)

        子类可调用此方法，也可完全覆盖 ``compute_observations``。
        """
        return torch.cat([
            self.base_ang_vel_body * self.ang_vel_scale,   # 3
            self.projected_gravity,                        # 3
            self.commands * self.commands_scale,           # 3 或 4
            (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale,  # num_dof
            self.dof_vel * self.dof_vel_scale,             # num_dof
            self.actions,                                  # num_dof
        ], dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # 指令采样与课程学习
    # ──────────────────────────────────────────────────────────────────────────

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样随机速度指令。"""
        cmd_cfg = self.cfg.CommandsCfg
        n = len(env_ids)

        vx_lo, vx_hi = cmd_cfg.lin_vel_x
        vy_lo, vy_hi = cmd_cfg.lin_vel_y
        yw_lo, yw_hi = cmd_cfg.ang_vel_yaw

        self.commands[env_ids, 0] = (
            torch.rand(n, device=self.device) * (vx_hi - vx_lo) + vx_lo
        )
        self.commands[env_ids, 1] = (
            torch.rand(n, device=self.device) * (vy_hi - vy_lo) + vy_lo
        )
        self.commands[env_ids, 2] = (
            torch.rand(n, device=self.device) * (yw_hi - yw_lo) + yw_lo
        )

        if cmd_cfg.num_commands >= 4 and cmd_cfg.heading_command:
            hd_lo, hd_hi = cmd_cfg.heading
            self.commands[env_ids, 3] = (
                torch.rand(n, device=self.device) * (hd_hi - hd_lo) + hd_lo
            )

        # 将速度量级小于阈值的指令置零（站立模式）
        threshold = 0.2
        small_cmd = (
            (torch.abs(self.commands[env_ids, 0]) < threshold)
            & (torch.abs(self.commands[env_ids, 1]) < threshold)
            & (torch.abs(self.commands[env_ids, 2]) < threshold)
        )
        self.commands[env_ids[small_cmd], :3] = 0.0

    def _update_command_curriculum_scale(self) -> None:
        """更新指令缩放向量（给观测用）。
        目前仅做简单缩放，子类可覆盖实现复杂的课程逻辑。
        """
        cmd_cfg = self.cfg.CommandsCfg
        lin_vel_max = max(abs(cmd_cfg.lin_vel_x[0]), abs(cmd_cfg.lin_vel_x[1]), 0.01)
        ang_vel_max = max(abs(cmd_cfg.ang_vel_yaw[0]), abs(cmd_cfg.ang_vel_yaw[1]), 0.01)
        self.commands_scale[:3] = torch.tensor(
            [1.0 / lin_vel_max, 1.0 / lin_vel_max, 1.0 / ang_vel_max],
            device=self.device,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 基座状态更新（子类在 _refresh_sim_tensors 后调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _update_base_state(
        self,
        root_states: torch.Tensor,
    ) -> None:
        """从仿真引擎刷新的根状态张量中提取并转换各基座状态量。

        参数
        ----
        root_states : (num_envs, 13) — [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        """
        self.base_pos[:]      = root_states[:, :3]
        self.base_quat[:]     = root_states[:, 3:7]
        self.base_lin_vel[:]  = root_states[:, 7:10]
        self.base_ang_vel[:]  = root_states[:, 10:13]

        # 转换到机体坐标系
        self.base_lin_vel_body[:] = self.quat_rotate_inverse(
            self.base_quat, self.base_lin_vel
        )
        self.base_ang_vel_body[:] = self.quat_rotate_inverse(
            self.base_quat, self.base_ang_vel
        )
        # IMU 重力方向（表示基座的倾斜程度）
        self.projected_gravity[:] = self.quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 脚部接触状态更新（子类在 _post_physics_step 中调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _update_feet_state(self, contact_forces: torch.Tensor) -> None:
        """更新脚部接触状态与空中时间计数器。

        参数
        ----
        contact_forces : (num_envs, num_bodies, 3) — 各刚体接触力
        """
        feet_idx = getattr(self, "feet_indices", [])
        if not feet_idx:
            return
        contact = torch.norm(contact_forces[:, feet_idx, :], dim=-1) > 1.0
        self.feet_contact_state[:] = contact
        # 空中时间：接触时清零，非接触时累计 dt
        dt = self.cfg.SimParamsCfg.dt * self.cfg.ControlCfg.decimation
        self.feet_air_time += dt
        self.feet_air_time[contact] = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # 域随机化
    # ──────────────────────────────────────────────────────────────────────────

    def _randomize_friction(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """随机化地面摩擦系数（子类根据引擎 API 实际写入）。"""
        if not self.cfg.DomainRandCfg.randomize_friction:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        lo, hi = self.cfg.DomainRandCfg.friction_range
        self.friction_coefficients[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        )
        # 子类继续调用引擎 API 将 friction_coefficients 写入物理材质

    def _randomize_base_mass(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """随机化根连杆附加质量。"""
        if not self.cfg.DomainRandCfg.randomize_base_mass:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        lo, hi = self.cfg.DomainRandCfg.added_mass_range
        self.added_mass[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        )

    def _push_robots(self) -> None:
        """定期对机器人根连杆施加随机速度扰动。
        子类调用时需在此之后将修改后的 root_states 写回引擎。
        """
        if not self.cfg.DomainRandCfg.push_robots:
            return
        push_interval = int(
            self.cfg.DomainRandCfg.push_interval_s
            / (self.cfg.SimParamsCfg.dt * self.cfg.ControlCfg.decimation)
        )
        # 每隔 push_interval 步推一次
        push_envs = (self.episode_length_buf % push_interval == 0).nonzero(as_tuple=False).flatten()
        if len(push_envs) == 0:
            return
        max_vel = self.cfg.DomainRandCfg.max_push_vel_xy
        push_vel = torch.zeros(len(push_envs), 3, device=self.device)
        push_vel[:, :2] = (
            torch.rand(len(push_envs), 2, device=self.device) * 2.0 - 1.0
        ) * max_vel
        self.base_lin_vel[push_envs] += push_vel
        # 子类还需将修改写回 root_states 并调用引擎 API

    # ──────────────────────────────────────────────────────────────────────────
    # 奖励函数库（自动注册到父类框架）
    # ──────────────────────────────────────────────────────────────────────────

    # -- 速度跟踪 ---------------------------------------------------------------

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """线速度跟踪奖励：机体前向和侧向速度与指令之差的高斯函数。"""
        sigma = self.cfg.RewardCfg.tracking_sigma
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel_body[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / sigma)

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        """偏航角速度跟踪奖励。"""
        sigma = self.cfg.RewardCfg.tracking_sigma
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel_body[:, 2]
        )
        return torch.exp(-ang_vel_error / sigma)

    # -- 姿态稳定性 -------------------------------------------------------------

    def _reward_lin_vel_z(self) -> torch.Tensor:
        """惩罚基座垂直线速度（防止跳跃/颠簸）。"""
        return torch.square(self.base_lin_vel_body[:, 2])

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        """惩罚基座横滚和俯仰角速度（保持稳定）。"""
        return torch.sum(torch.square(self.base_ang_vel_body[:, :2]), dim=1)

    def _reward_orientation(self) -> torch.Tensor:
        """惩罚基座偏离水平（重力投影 xy 分量越大越差）。"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self) -> torch.Tensor:
        """惩罚基座高度偏离目标值。"""
        target = self.cfg.RewardCfg.base_height_target
        height = self.base_pos[:, 2] - torch.mean(
            self.measured_heights, dim=1
        )
        return torch.square(height - target)

    # -- 效率 ------------------------------------------------------------------

    def _reward_torques(self) -> torch.Tensor:
        """惩罚关节力矩（节能）。"""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self) -> torch.Tensor:
        """惩罚关节速度（平滑运动）。"""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self) -> torch.Tensor:
        """惩罚关节加速度（减少冲击）。"""
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel)
                         / (self.cfg.SimParamsCfg.dt * self.cfg.ControlCfg.decimation)),
            dim=1,
        )

    def _reward_action_rate(self) -> torch.Tensor:
        """惩罚动作变化率（输出平滑）。"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # -- 接触 ------------------------------------------------------------------

    def _reward_feet_air_time(self) -> torch.Tensor:
        """奖励脚部在摆动相的空中时间（鼓励稳定步态）。"""
        feet_idx = getattr(self, "feet_indices", [])
        if not feet_idx:
            return torch.zeros(self.num_envs, device=self.device)
        # 只在有运动指令时计算
        cmd_norm = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
        contact = self.feet_contact_state  # (n, num_feet)
        # 首次接触时奖励空中时间（目标约 0.5s），超过 0 部分才有奖励
        first_contact = contact & ~self.last_contacts
        self.last_contacts = contact.clone()
        rew = torch.sum(
            (self.feet_air_time - 0.5) * first_contact.float(), dim=1
        )
        rew *= (cmd_norm.squeeze(-1) > 0.1).float()  # 有移动指令才奖励
        return rew

    def _reward_collision(self) -> torch.Tensor:
        """惩罚非脚部刚体（如大腿、机体）与地面或障碍的碰撞。"""
        penalize_idx = getattr(self, "penalize_contact_indices", [])
        if not penalize_idx:
            return torch.zeros(self.num_envs, device=self.device)
        max_f = self.cfg.RewardCfg.max_contact_force
        return torch.sum(
            (torch.norm(self.contact_forces[:, penalize_idx, :], dim=-1) > max_f).float(),
            dim=1,
        )

    def _reward_feet_stumble(self) -> torch.Tensor:
        """惩罚脚部在接触时的水平剪切力（防止绊脚）。"""
        feet_idx = getattr(self, "feet_indices", [])
        if not feet_idx:
            return torch.zeros(self.num_envs, device=self.device)
        feet_forces = self.contact_forces[:, feet_idx, :]
        # 水平力 >> 垂直力 意味着绊脚/滑脚
        horizontal = torch.norm(feet_forces[:, :, :2], dim=-1)
        vertical   = torch.abs(feet_forces[:, :, 2])
        stumble = (horizontal > 5.0 * vertical) & self.feet_contact_state
        return torch.sum(stumble.float(), dim=1)

    def _reward_stand_still(self) -> torch.Tensor:
        """当指令为 0 时惩罚关节运动（站立模式）。"""
        cmd_zero = torch.norm(self.commands[:, :3], dim=1) < 0.1
        return torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        ) * cmd_zero.float()

    def _reward_termination(self) -> torch.Tensor:
        """终止惩罚（在非超时终止的回合给予负奖励）。"""
        return (self.reset_buf & ~self.time_out_buf).float()

    # -- 关节软限位 -------------------------------------------------------------

    def _reward_dof_pos_limits(self) -> Optional[torch.Tensor]:
        """惩罚关节位置超出软限位范围。"""
        if not hasattr(self, "dof_pos_limits_soft"):
            return None
        out_of_limits = -(
            self.dof_pos - self.dof_pos_limits_soft[:, 0]
        ).clamp(max=0.0)
        out_of_limits += (
            self.dof_pos - self.dof_pos_limits_soft[:, 1]
        ).clamp(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    # ──────────────────────────────────────────────────────────────────────────
    # 默认实现（子类不需要引擎 API 时可直接使用）
    # ──────────────────────────────────────────────────────────────────────────

    def compute_observations(self) -> None:
        """默认观测：角速度 + 重力 + 指令 + 关节偏差 + 关节速度 + 动作。
        子类可覆盖以添加高度扫描、特权观测等。
        """
        self.obs_buf[:] = self._build_default_obs()

    def compute_reward(self) -> None:
        """调用父类奖励函数注册框架，遍历所有 ``_reward_*``。"""
        self._compute_reward_all()

    def check_termination(self) -> None:
        """默认终止：回合超时。子类覆盖以添加接触终止等条件。"""
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf[:] = self.time_out_buf

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """默认重置：重采样指令、关节回默认位置。
        子类必须覆盖此方法以调用引擎 API 重置物理状态。

        建议模式::

            def reset_idx(self, env_ids):
                # 1. 随机化根位姿
                # 2. 重置关节状态
                # 3. 调用物理引擎 API 写回
                self.gym.set_actor_root_state_tensor_indexed(...)
                self.gym.set_dof_state_tensor_indexed(...)
                # 4. 调用父类逻辑（重采样指令等）
                super().reset_idx(env_ids)
        """
        self._resample_commands(env_ids)
        if self.cfg.DomainRandCfg.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.DomainRandCfg.randomize_base_mass:
            self._randomize_base_mass(env_ids)

    # ──────────────────────────────────────────────────────────────────────────
    # 仍需子类实现的物理引擎绑定（与 BaseSimEnv 一致的抽象方法）
    # ──────────────────────────────────────────────────────────────────────────

    # _create_sim / _create_ground / _create_envs / _init_buffers
    # 均从 BaseSimEnv 继承为抽象方法，子类必须实现。
    #
    # 建议在 _create_envs 中：
    #   1. 加载 asset
    #   2. 循环创建 env + actor
    #   3. 设置 self.num_actions, self.num_bodies, self.dof_names
    #   4. 填充 self.feet_indices, self.penalize_contact_indices 等索引列表
    #
    # 建议在 _init_buffers 中：
    #   1. 调用 self._init_buffers_legged()
    #   2. 调用 self._init_dof_limits(pos, vel, torque)
    #   3. 从引擎 API 获取初始 root_states, dof_state 并赋值给缓冲区
