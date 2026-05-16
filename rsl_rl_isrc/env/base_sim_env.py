# 本文件与 rsl_rl_isrc 内其它业务子包完全独立，无交叉导入。
# 用途：为用户自定义物理仿真环境提供通用父类框架。
# 参考：legged_gym BaseTask / Isaac Lab BaseEnv 设计风格。
"""通用物理仿真并行环境父类 ``BaseSimEnv``。

特点
----
* **引擎无关**：物理引擎（Isaac Gym、Isaac Lab、MuJoCo 等）相关调用均为
  抽象方法 ``_create_sim``、``_init_physics`` 等，子类实现具体绑定。
* **配置驱动**：所有超参数通过 :class:`BaseEnvCfg` 数据类注入，不硬编码。
* **自包含**：无需导入 rsl_rl_isrc 其它子包即可独立使用。

快速上手（以 Isaac Gym 为例）::

    from rsl_rl_isrc.env.base_sim_env import BaseSimEnv
    from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg

    class MyEnv(BaseSimEnv):
        def _create_sim(self):
            self.sim = self.gym.create_sim(...)

        def _create_ground(self):
            self.gym.add_ground(self.sim, ...)

        def _create_envs(self):
            asset = self.gym.load_asset(...)
            for i in range(self.num_envs):
                env = self.gym.create_env(...)
                self.gym.create_actor(env, asset, ...)

        def _init_buffers(self):
            self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
            ...

        def compute_observations(self):
            self.obs_buf[:] = ...

        def compute_reward(self):
            self.rew_buf[:] = ...

        def check_termination(self):
            self.reset_buf[:] = ...

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# 仅从同目录导入配置类（同包、不跨包）
from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg


class BaseSimEnv(ABC):
    """物理仿真并行环境通用父类。

    子类至少需要实现以下抽象方法：

    **物理引擎桥接（引擎相关）**

    - :meth:`_create_sim` — 创建物理引擎仿真实例
    - :meth:`_create_ground` — 创建地面/地形
    - :meth:`_create_envs` — 批量创建 actor 并填充 actor handle 列表

    **数据缓冲区**

    - :meth:`_init_buffers` — 初始化 PyTorch 观测/奖励/状态 Tensor

    **逻辑核心**

    - :meth:`compute_observations` — 将仿真状态写入 ``obs_buf``
    - :meth:`compute_reward` — 将奖励写入 ``rew_buf``
    - :meth:`check_termination` — 将终止标志写入 ``reset_buf``

    **可选覆盖（有默认实现）**

    - :meth:`_pre_physics_step` — 物理步之前（动作转矩计算等）
    - :meth:`_post_physics_step` — 物理步之后（刷新状态、计算 rewards/obs）
    - :meth:`reset_idx` — 按索引重置指定环境
    - :meth:`_apply_action` — 将策略动作写入仿真引擎
    - :meth:`_simulate` — 执行若干 substep 的物理推进
    - :meth:`_refresh_sim_tensors` — 刷新引擎到 PyTorch 的张量映射
    """

    def __init__(self, cfg: BaseEnvCfg, device: str = "cuda"):
        """初始化环境。

        参数
        ----
        cfg:
            完整的环境配置实例（或其子类实例）。
        device:
            PyTorch 设备字符串，如 ``'cuda:0'``、``'cpu'``。
        """
        self.cfg = cfg
        self.device = torch.device(device)

        # ── 基本属性（从配置中提取）───────────────────────────────────────────
        self.num_envs: int = cfg.num_envs
        self.episode_length_s: float = cfg.episode_length_s
        self.env_spacing: float = cfg.env_spacing
        self.max_episode_length: int = int(cfg.episode_length_s
                                           / (cfg.SimParamsCfg.dt * cfg.ControlCfg.decimation))

        # ── 观测/动作维度（子类可在 _init_buffers 之前修改）────────────────────
        self.num_obs: int = cfg.ObsCfg.num_obs
        self.num_privileged_obs: Optional[int] = cfg.ObsCfg.num_privileged_obs
        self.num_actions: int = 0  # 须在 _setup_dimensions() 中设定

        # ── 主缓冲区（在 _init_buffers 中分配）──────────────────────────────────
        self.obs_buf: torch.Tensor           # (num_envs, num_obs)
        self.privileged_obs_buf: Optional[torch.Tensor] = None
        self.rew_buf: torch.Tensor           # (num_envs,)
        self.reset_buf: torch.Tensor         # (num_envs,) bool
        self.episode_length_buf: torch.Tensor  # (num_envs,) int
        self.actions: torch.Tensor           # (num_envs, num_actions)
        self.last_actions: torch.Tensor      # (num_envs, num_actions)
        self.extras: Dict = {}               # 额外信息字典（传给 Runner）

        # ── 奖励追踪缓冲区（每个 episode 独立累计）──────────────────────────────
        self.episode_sums: Dict[str, torch.Tensor] = {}

        # ── 初始化流程 ─────────────────────────────────────────────────────────
        self._set_seed(cfg.seed)
        self._setup_dimensions()
        self._create_sim()
        self._create_ground()
        self._create_envs()
        self._init_buffers()
        self._prepare_reward_functions()

        # 环境构建完成，刷新一次初始观测
        self.reset(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口（VecEnv 协议兼容）
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """执行一个策略步（包含 ``decimation`` 个物理步）。

        参数
        ----
        actions : (num_envs, num_actions)
            策略网络输出的归一化动作（通常在 [-1, 1]）。

        返回
        ----
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras
        """
        self.last_actions[:] = self.actions[:]
        self.actions = actions.clone().to(self.device)

        # ── 执行 decimation 次物理步 ──────────────────────────────────────────
        for _ in range(self.cfg.ControlCfg.decimation):
            self._apply_action()
            self._simulate()
            self._refresh_sim_tensors()

        self.episode_length_buf += 1

        # ── 后物理步处理 ──────────────────────────────────────────────────────
        self._post_physics_step()

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset(self, env_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """重置指定环境并返回重置后观测。

        参数
        ----
        env_ids :
            需要重置的环境索引列表或张量。
        """
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return self.obs_buf

        self.reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0
        return self.obs_buf

    def get_observations(self) -> torch.Tensor:
        """返回当前策略观测张量 (num_envs, num_obs)。"""
        return self.obs_buf

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """返回特权观测（Critic 专用），若无则返回 None。"""
        return self.privileged_obs_buf

    # ──────────────────────────────────────────────────────────────────────────
    # 抽象方法：子类必须实现
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _create_sim(self) -> None:
        """创建物理引擎仿真实例。

        实现示例（Isaac Gym）::

            self.gym = gymapi.acquire_gym()
            sim_params = gymapi.SimParams()
            # 填充 sim_params ...
            self.sim = self.gym.create_sim(
                self.cfg.sim.device_id, self.cfg.sim.graphics_device_id,
                gymapi.SIM_PHYSX, sim_params)
        """

    @abstractmethod
    def _create_ground(self) -> None:
        """创建地面或地形平面。

        实现示例（平地）::

            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.add_ground(self.sim, plane_params)
        """

    @abstractmethod
    def _create_envs(self) -> None:
        """批量创建并行环境与 actor。

        实现时需设置 ``self.num_actions``，并保存 actor handle 列表::

            self.envs = []
            self.actor_handles = []
            for i in range(self.num_envs):
                env = self.gym.create_env(...)
                handle = self.gym.create_actor(...)
                self.envs.append(env)
                self.actor_handles.append(handle)
            self.num_actions = self.gym.get_asset_dof_count(asset)
        """

    @abstractmethod
    def _init_buffers(self) -> None:
        """分配所有 PyTorch 观测/状态/奖励 Tensor。

        必须分配的基础缓冲区::

            self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
            self.rew_buf = torch.zeros(self.num_envs, device=self.device)
            self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
            self.last_actions = torch.zeros_like(self.actions)
        """

    @abstractmethod
    def compute_observations(self) -> None:
        """将当前仿真状态组装为观测向量，写入 ``self.obs_buf``。

        示例（简单机器人）::

            self.obs_buf = torch.cat([
                self.base_lin_vel * self.lin_vel_scale,
                self.base_ang_vel * self.ang_vel_scale,
                self.projected_gravity,
                self.commands * self.commands_scale,
                self.dof_pos - self.default_dof_pos,
                self.dof_vel * self.dof_vel_scale,
                self.actions,
            ], dim=-1)
        """

    @abstractmethod
    def compute_reward(self) -> None:
        """计算奖励，写入 ``self.rew_buf``。

        通常调用 ``_reward_*`` 系列辅助函数::

            self.rew_buf[:] = 0.0
            for name, fn in self.reward_functions.items():
                rew = fn() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
        """

    @abstractmethod
    def check_termination(self) -> None:
        """判断各环境是否应当终止，写入 ``self.reset_buf``。

        示例::

            # 超时终止
            self.time_out_buf = self.episode_length_buf >= self.max_episode_length
            # 接触终止
            contact_term = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
            self.reset_buf = contact_term | self.time_out_buf
        """

    @abstractmethod
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """重置指定 ``env_ids`` 环境的根状态、关节状态等。

        实现时通常需要::

            # 设置根位姿（带随机扰动）
            self.root_states[env_ids] = self.base_init_state.clone()
            # 设置关节角度
            self.dof_pos[env_ids] = self.default_dof_pos
            self.dof_vel[env_ids] = 0.0
            # 写回仿真引擎
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(...)
            self.gym.set_dof_state_tensor_indexed(...)
        """

    # ──────────────────────────────────────────────────────────────────────────
    # 可选覆盖方法（有默认实现）
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_dimensions(self) -> None:
        """在 ``_create_envs`` 之前根据配置计算维度。
        子类可覆盖以从 URDF 动态读取 num_actions 等信息。
        默认为空操作（num_actions 由 _create_envs 设置）。
        """
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步之前的钩子函数（可选）。
        默认存储动作并调用 ``_apply_action``。子类如需在此做力矩裁剪、安全检查等可覆盖。
        """
        pass

    def _post_physics_step(self) -> None:
        """物理步完成后的通用处理：终止检测、奖励计算、观测更新、重置终止环境。

        通常不需要覆盖；若需要添加自定义后处理逻辑，
        请在调用 ``super()._post_physics_step()`` 前后插入。
        """
        # 检查终止
        self.check_termination()

        # 计算奖励
        self.compute_reward()

        # 时限截断超时标志（用于 PPO bootstrap 修正）
        if self.cfg.send_timeouts:
            self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length

        # 重置需要重置的环境
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._update_episode_stats(env_ids)
            self.reset_idx(env_ids)
            self.episode_length_buf[env_ids] = 0

        # 更新观测
        self.compute_observations()

        # 添加噪声
        if self.cfg.NoiseCfg.add_noise:
            self._add_obs_noise()

    def _apply_action(self) -> None:
        """将策略动作转换为力矩/位置目标，写入仿真引擎。
        默认为空实现，子类根据控制模式（PD/Torque）覆盖。
        """
        pass

    def _simulate(self) -> None:
        """执行一个物理仿真子步。
        默认调用 ``gym.simulate`` 并刷新仿真视图。子类根据引擎覆盖。
        """
        pass

    def _refresh_sim_tensors(self) -> None:
        """将引擎内部状态刷新到 GPU PyTorch 张量。
        例如 Isaac Gym 需要调用 ``gym.refresh_actor_root_state_tensor(...)``。
        """
        pass

    def _add_obs_noise(self) -> None:
        """向观测缓冲区添加高斯噪声（默认实现：整体加一个统一噪声）。
        子类可覆盖为分维度噪声。
        """
        noise_level = self.cfg.NoiseCfg.noise_level
        self.obs_buf += noise_level * torch.randn_like(self.obs_buf) * 0.1

    # ──────────────────────────────────────────────────────────────────────────
    # 奖励函数注册与计算框架
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_reward_functions(self) -> None:
        """自动注册所有 ``_reward_*`` 方法，并从配置中读取对应缩放系数。

        如果 ``cfg.rewards.scales`` 中有字段 ``foo``，则自动在子类中寻找
        ``self._reward_foo()`` 方法并注册。用户只需定义同名方法即可自动被调用。
        """
        self.reward_functions: Dict = {}
        self.reward_scales: Dict[str, float] = {}
        self.episode_sums = {}

        scales_obj = getattr(self.cfg.RewardCfg, "scales", None) or self.cfg.RewardCfg.scales
        for name in dir(scales_obj):
            if name.startswith("_"):
                continue
            scale = getattr(scales_obj, name, None)
            if not isinstance(scale, (int, float)):
                continue
            method_name = f"_reward_{name}"
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                self.reward_functions[name] = getattr(self, method_name)
                self.reward_scales[name] = float(scale)
                # 初始化 episode 累计
                self.episode_sums[name] = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.float
                )

    def _compute_reward_all(self) -> None:
        """循环调用所有已注册奖励函数并累加。
        子类在 ``compute_reward`` 中调用此方法，或自行实现完整逻辑。
        """
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn()
            if rew is None:
                continue
            rew = rew * self.reward_scales[name]
            self.rew_buf += rew
            if name in self.episode_sums:
                self.episode_sums[name] += rew

        if self.cfg.RewardCfg.only_positive_rewards:
            self.rew_buf[:] = torch.clamp(self.rew_buf, min=0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # episode 统计
    # ──────────────────────────────────────────────────────────────────────────

    def _update_episode_stats(self, env_ids: torch.Tensor) -> None:
        """在环境重置前将本回合的奖励总计写入 extras['episode']。
        Runner 会在每步检查 infos['episode'] 并记录到 TensorBoard。
        """
        episode_info = {}
        for key, values in self.episode_sums.items():
            episode_info[f"rew_{key}"] = torch.mean(values[env_ids]) / self.max_episode_length
            values[env_ids] = 0.0

        episode_info["episode_length"] = torch.mean(
            self.episode_length_buf[env_ids].float()
        )
        self.extras["episode"] = episode_info

    # ──────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _set_seed(seed: int) -> None:
        """设置 Python / NumPy / PyTorch 的随机种子。"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def euler_to_quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
        """欧拉角（内旋 ZYX，弧度）转四元数 (qx, qy, qz, qw)。"""
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return torch.tensor([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ], dtype=torch.float32)

    @staticmethod
    def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """将向量 ``v`` 从世界系转到以四元数 ``q`` 表示的局部系。

        参数
        ----
        q : (..., 4) — 四元数 (qx, qy, qz, qw)
        v : (..., 3) — 世界系向量

        返回
        ----
        (..., 3) — 局部系向量
        """
        q_w = q[..., 3:4]
        q_vec = q[..., :3]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
        c = q_vec * (torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0)
        return a - b + c

    @staticmethod
    def get_euler_xyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """四元数 (qx, qy, qz, qw) 转欧拉角 (roll, pitch, yaw)，弧度。

        返回
        ----
        (roll, pitch, yaw) 各形状与 q[..., 0] 相同。
        """
        qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
        """将角度包裹到 [-π, π] 区间。"""
        return (angles + math.pi) % (2 * math.pi) - math.pi
