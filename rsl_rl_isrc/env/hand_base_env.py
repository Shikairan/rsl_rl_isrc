# 本文件与 rsl_rl_isrc 内其它业务子包完全独立，无交叉导入。
# 用途：为单手（灵巧手）仿真环境提供功能丰富的基类，继承自 BaseSimEnv。
# 参考：legged_base_env LeggedBaseEnv 风格，去除足式专用逻辑。
"""单手并行仿真环境父类 ``HandBaseEnv``。

在 :class:`BaseSimEnv` 基础上补充单手操作所需的通用功能：

**运动控制**

* PD 控制器（位置目标 → 力矩）
* 动作缩放、关节软限位惩罚

**状态观测**

* 重力投影方向（掌根/腕部 IMU-like）
* 关节目标角指令、关节位置、速度
* 指尖接触状态与位置（子类刷新后可用）

**奖励函数库**（``_reward_*`` 方法，自动注册）

- 任务：``tracking_joint_pos``、``fingertip_contact``、``target_hold``
- 效率：``torques``、``dof_vel``、``dof_acc``、``action_rate``
- 安全：``collision``、``dof_pos_limits``、``termination``

**指令采样**

* 相对默认关节角的随机目标偏移（维度 = ``num_actions``）
* 小指令阈值内置“保持默认抓型”

**域随机化**（可选）

* 地面摩擦系数随机化
* 腕部/掌根附加质量扰动（无推机器人）

**G1 左手 7-DOF 配置示例**::

    from rsl_rl_isrc.env.hand_base_env import HandBaseEnv
    from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg

    NUM_HAND_DOF = 7  # left_hand_*_joint in g1_29dof_with_hand_rev_1_0.xml

    class G1LeftHandCfg(BaseEnvCfg):
        num_envs = 4096
        class asset(BaseEnvCfg.AssetCfg):
            file = ".../g1_29dof_with_hand_rev_1_0.xml"
            fingertip_name = "_1_link"   # 匹配 index_1 / middle_1 / thumb_2 等末端连杆
            palm_name = "palm"
            fix_base_link = True
        class init_state(BaseEnvCfg.InitStateCfg):
            pos = (0.0, 0.0, 0.5)
            default_joint_angles = {
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_index_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
            }
        class hand_commands(BaseEnvCfg.HandCommandsCfg):
            num_commands = NUM_HAND_DOF
            joint_target_range = (-0.3, 0.3)
        class obs(BaseEnvCfg.ObsCfg):
            num_obs = 3 + NUM_HAND_DOF + 3 * NUM_HAND_DOF  # = 31
        class control(BaseEnvCfg.ControlCfg):
            stiffness = {"hand": 2.0}
            damping = {"hand": 0.1}
            action_scale = 0.25

    class G1LeftHandEnv(HandBaseEnv):
        def _create_sim(self): ...
        def _create_ground(self): ...
        def _create_envs(self):
            # 设置 self.num_actions, self.dof_names, self.fingertip_indices
            ...
        def _init_buffers(self):
            super()._init_buffers_hand()
            self._init_dof_limits(pos_limits, vel_limits, torque_limits)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg
from rsl_rl_isrc.env.base_sim_env import BaseSimEnv


class HandBaseEnv(BaseSimEnv):
    """单手通用仿真环境父类。

    继承 :class:`BaseSimEnv`，新增手部专用的：
    - 状态缓冲区（关节、掌根、指尖接触等）
    - 观测构建与关节目标指令处理
    - PD 控制器
    - 手部奖励函数库
    - 域随机化接口（无推机器人）

    子类须在 ``_create_envs`` 中设置 ``num_actions``、``dof_names``、
    ``fingertip_indices`` 等；在 ``_init_buffers`` 中调用 ``_init_buffers_hand()``。
    """

    @staticmethod
    def compute_num_obs(num_actions: int, num_commands: Optional[int] = None) -> int:
        """计算默认观测维度：重力(3) + 指令 + 关节偏差 + 关节速度 + 动作。"""
        nc = num_commands if num_commands is not None else num_actions
        return 3 + nc + 3 * num_actions

    def __init__(self, cfg: BaseEnvCfg, device: str = "cuda"):
        super().__init__(cfg, device)

    # ──────────────────────────────────────────────────────────────────────────
    # 子类应调用的缓冲区初始化（在 _init_buffers 内调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _init_buffers_hand(self) -> None:
        """初始化手部专用的所有 PyTorch 张量缓冲区。

        子类在 ``_init_buffers`` 中调用::

            def _init_buffers(self):
                self._init_buffers_hand()
                self._init_dof_limits(pos_limits, vel_limits, torque_limits)
        """
        n, d = self.num_envs, self.device
        cmd_cfg = self.cfg.HandCommandsCfg
        num_cmd = cmd_cfg.num_commands if cmd_cfg.num_commands > 0 else self.num_actions

        # ── 动作缓冲区 ────────────────────────────────────────────────────────
        self.actions = torch.zeros(n, self.num_actions, device=d)
        self.last_actions = torch.zeros(n, self.num_actions, device=d)

        # ── 观测 ──────────────────────────────────────────────────────────────
        self.obs_buf = torch.zeros(n, self.num_obs, device=d)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(n, self.num_privileged_obs, device=d)

        # ── 奖励/终止 ─────────────────────────────────────────────────────────
        self.rew_buf = torch.zeros(n, device=d)
        self.reset_buf = torch.ones(n, dtype=torch.bool, device=d)

        # ── 回合长度 ──────────────────────────────────────────────────────────
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=d)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=d)

        # ── 掌根/腕部根状态 (n, 13): [pos(3), quat(4), lin_vel(3), ang_vel(3)] ─
        self.base_pos = torch.zeros(n, 3, device=d)
        self.base_quat = torch.zeros(n, 4, device=d)
        self.base_lin_vel = torch.zeros(n, 3, device=d)
        self.base_ang_vel = torch.zeros(n, 3, device=d)
        self.projected_gravity = torch.zeros(n, 3, device=d)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=d).repeat(n, 1)

        # ── 关节状态 ──────────────────────────────────────────────────────────
        self.dof_pos = torch.zeros(n, self.num_actions, device=d)
        self.dof_vel = torch.zeros(n, self.num_actions, device=d)
        self.last_dof_vel = torch.zeros(n, self.num_actions, device=d)
        self.torques = torch.zeros(n, self.num_actions, device=d)
        self.default_dof_pos = self._get_default_dof_pos()

        # ── 关节限位 ──────────────────────────────────────────────────────────
        self.dof_pos_limits = torch.zeros(self.num_actions, 2, device=d)
        self.dof_vel_limits = torch.zeros(self.num_actions, device=d)
        self.torque_limits = torch.zeros(self.num_actions, device=d)

        # ── 接触力 (n, num_bodies, 3) ─────────────────────────────────────────
        num_bodies = getattr(self, "num_bodies", 1)
        self.contact_forces = torch.zeros(n, num_bodies, 3, device=d)

        # ── 指尖（子类在 _create_envs 中设置 fingertip_indices）──────────────
        self.fingertip_indices: List[int] = getattr(self, "fingertip_indices", [])
        tips_n = max(len(self.fingertip_indices), 1)
        self.fingertip_pos = torch.zeros(n, tips_n, 3, device=d)
        self.fingertip_contact = torch.zeros(n, tips_n, dtype=torch.bool, device=d)

        # ── 关节目标指令 (n, num_commands) ────────────────────────────────────
        self.commands = torch.zeros(n, num_cmd, device=d)
        self.commands_scale = torch.ones(num_cmd, device=d)

        # ── 域随机化 ──────────────────────────────────────────────────────────
        self.friction_coefficients = torch.ones(n, device=d)
        self.added_mass = torch.zeros(n, device=d)

        # ── 观测缩放系数 ──────────────────────────────────────────────────────
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05

        self._update_command_scale()

    def _get_default_dof_pos(self) -> torch.Tensor:
        """从配置的 ``default_joint_angles`` 按关节名称子串匹配构建默认角向量。"""
        default = torch.zeros(self.num_actions, device=self.device)
        angle_map = self.cfg.InitStateCfg.default_joint_angles
        if not angle_map:
            return default

        dof_names = getattr(
            self, "dof_names", [f"dof_{i}" for i in range(self.num_actions)]
        )
        for i, name in enumerate(dof_names):
            for key, val in angle_map.items():
                if key in name or name in key:
                    default[i] = val
                    break
        return default

    def _init_dof_limits(
        self,
        pos_limits: torch.Tensor,
        vel_limits: torch.Tensor,
        torque_limits: torch.Tensor,
    ) -> None:
        """从物理引擎 API 获取的关节限位写入缓冲区。"""
        self.dof_pos_limits[:] = pos_limits.to(self.device)
        self.dof_vel_limits[:] = vel_limits.to(self.device)
        self.torque_limits[:] = torque_limits.to(self.device)

        soft = self.cfg.RewardCfg.soft_dof_pos_limit
        m = (self.dof_pos_limits[:, 1] + self.dof_pos_limits[:, 0]) * 0.5
        h = (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) * 0.5 * soft
        self.dof_pos_limits_soft = torch.stack([m - h, m + h], dim=1)

    # ──────────────────────────────────────────────────────────────────────────
    # PD 控制器
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """将策略动作（[-1,1] 归一化偏移）转换为关节力矩。"""
        scale = self.cfg.ControlCfg.action_scale
        pos_target = actions * scale + self.default_dof_pos

        kp = self._build_gain_tensor("stiffness")
        kd = self._build_gain_tensor("damping")

        torques = kp * (pos_target - self.dof_pos) - kd * self.dof_vel
        torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
        return torques

    def _build_gain_tensor(self, gain_type: str) -> torch.Tensor:
        """从 stiffness/damping 配置按关节名子串匹配构建增益张量（带缓存）。"""
        cache_attr = f"_cached_{gain_type}"
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        gains_cfg = (
            self.cfg.ControlCfg.stiffness
            if gain_type == "stiffness"
            else self.cfg.ControlCfg.damping
        )
        dof_names = getattr(
            self, "dof_names", [f"dof_{i}" for i in range(self.num_actions)]
        )
        gains = torch.zeros(self.num_actions, device=self.device)
        for i, name in enumerate(dof_names):
            for key, val in gains_cfg.items():
                if key in name:
                    gains[i] = val
                    break
        setattr(self, cache_attr, gains)
        return gains

    # ──────────────────────────────────────────────────────────────────────────
    # 默认观测
    # ──────────────────────────────────────────────────────────────────────────

    def _build_default_obs(self) -> torch.Tensor:
        """构建默认手部观测：重力 + 关节指令 + 关节偏差 + 速度 + 动作。"""
        cmd_dim = self.commands.shape[1]
        cmd_scaled = self.commands[:, :cmd_dim] * self.commands_scale[:cmd_dim]
        return torch.cat([
            self.projected_gravity,
            cmd_scaled,
            (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale,
            self.dof_vel * self.dof_vel_scale,
            self.actions,
        ], dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # 关节目标指令采样
    # ──────────────────────────────────────────────────────────────────────────

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样关节目标角偏移（相对 default_dof_pos）。"""
        cmd_cfg = self.cfg.HandCommandsCfg
        n = len(env_ids)
        lo, hi = cmd_cfg.joint_target_range
        num_cmd = self.commands.shape[1]

        self.commands[env_ids, :num_cmd] = (
            torch.rand(n, num_cmd, device=self.device) * (hi - lo) + lo
        )

        # 各维均很小时置零 → 保持默认抓型
        threshold = cmd_cfg.zero_threshold
        small = torch.all(torch.abs(self.commands[env_ids, :num_cmd]) < threshold, dim=1)
        self.commands[env_ids[small], :num_cmd] = 0.0

    def _update_command_scale(self) -> None:
        """根据 joint_target_range 与 command_scale 更新观测用指令缩放。"""
        cmd_cfg = self.cfg.HandCommandsCfg
        lo, hi = cmd_cfg.joint_target_range
        span = max(abs(lo), abs(hi), 1e-3)
        scale = cmd_cfg.command_scale / span
        self.commands_scale[:] = scale

    def _command_target_pos(self) -> torch.Tensor:
        """当前指令对应的绝对关节目标角 (num_envs, num_actions)。"""
        num_cmd = min(self.commands.shape[1], self.num_actions)
        target = self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        target[:, :num_cmd] = self.default_dof_pos[:num_cmd] + self.commands[:, :num_cmd]
        return target

    # ──────────────────────────────────────────────────────────────────────────
    # 状态更新（子类在 _refresh_sim_tensors 后调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _update_base_state(self, root_states: torch.Tensor) -> None:
        """从根状态张量更新掌根/腕部位姿与重力投影。"""
        self.base_pos[:] = root_states[:, :3]
        self.base_quat[:] = root_states[:, 3:7]
        self.base_lin_vel[:] = root_states[:, 7:10]
        self.base_ang_vel[:] = root_states[:, 10:13]
        self.projected_gravity[:] = self.quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

    def _update_fingertips_state(
        self,
        contact_forces: torch.Tensor,
        body_positions: Optional[torch.Tensor] = None,
        contact_threshold: Optional[float] = None,
    ) -> None:
        """更新指尖接触标志与世界系位置。

        参数
        ----
        contact_forces : (num_envs, num_bodies, 3)
        body_positions : (num_envs, num_bodies, 3)，可选
        contact_threshold : 接触力模长阈值（N），默认读 RewardCfg
        """
        tips = getattr(self, "fingertip_indices", [])
        if not tips:
            return
        thresh = contact_threshold or self.cfg.RewardCfg.fingertip_contact_threshold
        forces = contact_forces[:, tips, :]
        self.fingertip_contact[:] = torch.norm(forces, dim=-1) > thresh
        if body_positions is not None:
            self.fingertip_pos[:] = body_positions[:, tips, :]

    # ──────────────────────────────────────────────────────────────────────────
    # 域随机化（无推机器人）
    # ──────────────────────────────────────────────────────────────────────────

    def _randomize_friction(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """随机化摩擦系数（子类负责写入物理引擎）。"""
        if not self.cfg.DomainRandCfg.randomize_friction:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        lo, hi = self.cfg.DomainRandCfg.friction_range
        self.friction_coefficients[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        )

    def _randomize_base_mass(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """随机化掌根/腕部附加质量。"""
        if not self.cfg.DomainRandCfg.randomize_base_mass:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        lo, hi = self.cfg.DomainRandCfg.added_mass_range
        self.added_mass[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 奖励函数库
    # ──────────────────────────────────────────────────────────────────────────

    def _reward_tracking_joint_pos(self) -> torch.Tensor:
        """关节角跟踪奖励：当前偏差与指令目标之差的高斯核。"""
        sigma = self.cfg.RewardCfg.tracking_sigma
        num_cmd = min(self.commands.shape[1], self.num_actions)
        err = self.commands[:, :num_cmd] - (
            self.dof_pos[:, :num_cmd] - self.default_dof_pos[:num_cmd]
        )
        return torch.exp(-torch.sum(torch.square(err), dim=1) / sigma)

    def _reward_torques(self) -> torch.Tensor:
        """惩罚关节力矩（节能）。"""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self) -> torch.Tensor:
        """惩罚关节速度（平滑）。"""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self) -> torch.Tensor:
        """惩罚关节加速度。"""
        dt = self.cfg.SimParamsCfg.dt * self.cfg.ControlCfg.decimation
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / dt), dim=1)

    def _reward_action_rate(self) -> torch.Tensor:
        """惩罚动作变化率。"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_fingertip_contact(self) -> torch.Tensor:
        """奖励有效指尖接触（任一指尖法向力超过阈值）。"""
        if not getattr(self, "fingertip_indices", []):
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(self.fingertip_contact.float(), dim=1)

    def _reward_collision(self) -> torch.Tensor:
        """惩罚非指尖部件的过大接触力。"""
        penalize_idx = getattr(self, "penalize_contact_indices", [])
        if not penalize_idx:
            return torch.zeros(self.num_envs, device=self.device)
        max_f = self.cfg.RewardCfg.max_contact_force
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, penalize_idx, :], dim=-1) > max_f
            ).float(),
            dim=1,
        )

    def _reward_target_hold(self) -> torch.Tensor:
        """指令接近零时惩罚关节偏离默认姿态（保持默认抓型）。"""
        num_cmd = self.commands.shape[1]
        cmd_zero = torch.norm(self.commands[:, :num_cmd], dim=1) < 0.1
        return torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        ) * cmd_zero.float()

    def _reward_termination(self) -> torch.Tensor:
        """非超时终止惩罚。"""
        return (self.reset_buf & ~self.time_out_buf).float()

    def _reward_dof_pos_limits(self) -> Optional[torch.Tensor]:
        """惩罚关节超出软限位。"""
        if not hasattr(self, "dof_pos_limits_soft"):
            return None
        out_of_limits = -(self.dof_pos - self.dof_pos_limits_soft[:, 0]).clamp(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits_soft[:, 1]).clamp(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    # ──────────────────────────────────────────────────────────────────────────
    # 默认生命周期方法
    # ──────────────────────────────────────────────────────────────────────────

    def compute_observations(self) -> None:
        """默认观测构建。"""
        self.obs_buf[:] = self._build_default_obs()

    def compute_reward(self) -> None:
        """遍历所有已注册 ``_reward_*`` 并累加。"""
        self._compute_reward_all()

    def check_termination(self) -> None:
        """默认终止：仅回合超时。子类可覆盖以加入掌根碰撞终止等。"""
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf[:] = self.time_out_buf

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """默认重置逻辑：重采样关节目标指令与域随机化。

        子类须先通过引擎 API 重置物理状态，再调用 ``super().reset_idx(env_ids)``。
        """
        self._resample_commands(env_ids)
        if self.cfg.DomainRandCfg.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.DomainRandCfg.randomize_base_mass:
            self._randomize_base_mass(env_ids)
