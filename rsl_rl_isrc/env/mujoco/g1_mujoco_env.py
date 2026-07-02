"""G1 人形机器人 MuJoCo 环境（CPU 仿真，对齐 Isaac G1 obs/奖励）。"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch

from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg
from rsl_rl_isrc.env.legged_base_env import LeggedBaseEnv
from rsl_rl_isrc.env.mujoco.g1_mujoco_config import G1MujocoCfg
from rsl_rl_isrc.env.mujoco.mujoco_backend import (
    G1_ACTUATED_JOINT_NAMES,
    MujocoSimPool,
)


class G1MujocoEnv(LeggedBaseEnv):
    """MuJoCo 版 G1 足式环境。"""

    def __init__(self, cfg: BaseEnvCfg | type = G1MujocoCfg, device: str = "cpu") -> None:
        self.sim_pool: Optional[MujocoSimPool] = None
        self.root_states = torch.zeros(1, 13)
        self.rpy = torch.zeros(1, 3)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0])
        self.location_O = torch.zeros(1, 2)
        self.phase = torch.zeros(1)
        self.phase_left = torch.zeros(1)
        self.phase_right = torch.zeros(1)
        self.leg_phase = torch.zeros(1, 2)
        self.feet_pos = torch.zeros(1, 2, 3)
        self.feet_vel = torch.zeros(1, 2, 3)
        self.feet_num = 2
        self.last_root_vel = torch.zeros(1, 6)
        self.hip_reward_indices = torch.zeros(0, dtype=torch.long)
        self.penalised_contact_indices: List[int] = []
        self.termination_contact_indices: List[int] = []
        self._body_ids_ordered: List[int] = []
        self._noise_scale_vec: Optional[torch.Tensor] = None
        self.dt = 0.02
        super().__init__(cfg, device=device)

    # ── 引擎桥接 ─────────────────────────────────────────────────────────────

    def _create_sim(self) -> None:
        xml_path = self.cfg.AssetCfg.file
        self.sim_pool = MujocoSimPool(self.num_envs, xml_path=xml_path)
        self.num_bodies = len(self.sim_pool.info.body_ids)

    def _create_ground(self) -> None:
        pass

    def _create_envs(self) -> None:
        assert self.sim_pool is not None
        self.num_actions = len(G1_ACTUATED_JOINT_NAMES)
        self.dof_names = list(G1_ACTUATED_JOINT_NAMES)
        self.num_bodies = len(self.sim_pool.info.body_ids)
        self._body_ids_ordered = list(range(self.num_bodies))
        names = self._body_name_list()

        foot_name = self.cfg.AssetCfg.foot_name
        self.feet_indices = [
            i for i, name in enumerate(names) if foot_name in name
        ]
        if len(self.feet_indices) < 2:
            raise RuntimeError(f"未找到足部 body（foot_name={foot_name}）")

        penalize = self.cfg.AssetCfg.penalize_contacts_on
        self.penalised_contact_indices = [
            i for i, name in enumerate(names) if any(p in name for p in penalize)
        ]
        terminate = self.cfg.AssetCfg.terminate_after_contacts_on
        self.termination_contact_indices = [
            i for i, name in enumerate(names) if any(t in name for t in terminate)
        ]

        info = self.sim_pool.info
        self._pending_pos_limits = torch.tensor(
            info.joint_range, dtype=torch.float32, device=self.device
        )
        self._pending_vel_limits = torch.ones(
            self.num_actions, device=self.device
        ) * 100.0
        self._pending_torque_limits = torch.tensor(
            np.minimum(
                np.abs(info.actuator_forcerange[:, 0]),
                np.abs(info.actuator_forcerange[:, 1]),
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def _body_name_list(self) -> List[str]:
        assert self.sim_pool is not None
        names = [""] * self.num_bodies
        for name, bid in self.sim_pool.info.body_ids.items():
            if bid < self.num_bodies:
                names[bid] = name
        return names

    def _init_buffers(self) -> None:
        assert self.sim_pool is not None
        self._body_ids_ordered = list(range(self.num_bodies))
        n = self.num_envs
        d = self.device

        self.root_states = torch.zeros(n, 13, device=d)
        self.rpy = torch.zeros(n, 3, device=d)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=d)
        self.location_O = torch.zeros(n, 2, device=d)
        self.phase = torch.zeros(n, device=d)
        self.phase_left = torch.zeros(n, device=d)
        self.phase_right = torch.zeros(n, device=d)
        self.leg_phase = torch.zeros(n, 2, device=d)
        self.feet_pos = torch.zeros(n, max(2, len(self.feet_indices)), 3, device=d)
        self.feet_vel = torch.zeros(n, max(2, len(self.feet_indices)), 3, device=d)
        self.last_root_vel = torch.zeros(n, 6, device=d)

        hip_names = ("hip_yaw", "hip_roll")
        hip_indices = [
            i for i, name in enumerate(self.dof_names)
            if any(part in name for part in hip_names)
        ]
        self.hip_reward_indices = torch.tensor(hip_indices, dtype=torch.long, device=d)

        self._init_buffers_legged()
        self._init_dof_limits(
            self._pending_pos_limits,
            self._pending_vel_limits,
            self._pending_torque_limits,
        )
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=d)
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05

        self.base_init_state = torch.zeros(13, device=d)
        self.base_init_state[0:3] = torch.tensor(self.cfg.InitStateCfg.pos, device=d)
        self.base_init_state[3:7] = torch.tensor(self.cfg.InitStateCfg.rot, device=d)
        self.env_origins = torch.zeros(n, 3, device=d)

        self._noise_scale_vec = self._build_noise_scale_vec()
        self._resample_commands(torch.arange(n, device=d))

    def _get_default_dof_pos(self) -> torch.Tensor:
        defaults = torch.zeros(self.num_actions, device=self.device)
        angle_map = self.cfg.InitStateCfg.default_joint_angles
        for i, name in enumerate(self.dof_names):
            if name in angle_map:
                defaults[i] = float(angle_map[name])
        return defaults

    def _apply_action(self) -> None:
        self.torques[:] = self._compute_torques(self.actions)
        ctrl_np = self.torques.detach().cpu().numpy()
        for i in range(self.num_envs):
            self.sim_pool.step_env(i, ctrl_np[i])

    def _simulate(self) -> None:
        pass

    def _refresh_sim_tensors(self) -> None:
        assert self.sim_pool is not None
        for i in range(self.num_envs):
            self.root_states[i] = torch.tensor(
                self.sim_pool.read_root_state(i),
                dtype=torch.float32,
                device=self.device,
            )
            pos, vel = self.sim_pool.read_dof_pos_vel(i)
            self.dof_pos[i] = torch.tensor(pos, dtype=torch.float32, device=self.device)
            self.dof_vel[i] = torch.tensor(vel, dtype=torch.float32, device=self.device)

            body_ids = self._body_ids_ordered
            forces = self.sim_pool.contact_forces_on_bodies(i, body_ids)
            self.contact_forces[i] = torch.tensor(
                forces, dtype=torch.float32, device=self.device
            )

            for fi, bidx in enumerate(self.feet_indices[:2]):
                bname = self._body_name_list()[bidx]
                fpos, fvel = self.sim_pool.read_body_pos_vel(i, bname)
                self.feet_pos[i, fi] = torch.tensor(fpos, dtype=torch.float32, device=self.device)
                self.feet_vel[i, fi] = torch.tensor(fvel, dtype=torch.float32, device=self.device)

        self._update_base_state(self.root_states)
        roll, pitch, yaw = self.get_euler_xyz(self.base_quat)
        self.rpy[:, 0] = roll
        self.rpy[:, 1] = pitch
        self.rpy[:, 2] = yaw

    def _post_physics_step(self) -> None:
        self._g1_callback_before_reward()
        super()._post_physics_step()
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def _g1_callback_before_reward(self) -> None:
        resample_every = max(
            1,
            int(self.cfg.CommandsCfg.resampling_time / self.dt),
        )
        env_ids = (
            (self.episode_length_buf % resample_every == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) > 0:
            self._resample_commands_g1(env_ids)

        if self.cfg.CommandsCfg.heading_command:
            forward = self._quat_apply(self.base_quat, self.forward_vec.expand(self.num_envs, 3))
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clamp(
                0.5 * self.wrap_to_pi(self.commands[:, 3] - heading),
                -1.0,
                1.0,
            )

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf.float() * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1.0
        self.leg_phase = torch.stack([self.phase_left, self.phase_right], dim=-1)

        if self.cfg.DomainRandCfg.push_robots:
            self._push_robots_g1()

    def _resample_commands_g1(self, env_ids: torch.Tensor) -> None:
        cmd = self.cfg.CommandsCfg
        n = len(env_ids)
        self.commands[env_ids, 0] = (
            torch.rand(n, device=self.device) * (cmd.lin_vel_x[1] - cmd.lin_vel_x[0])
            + cmd.lin_vel_x[0]
        )
        self.commands[env_ids, 1] = (
            torch.rand(n, device=self.device) * (cmd.lin_vel_y[1] - cmd.lin_vel_y[0])
            + cmd.lin_vel_y[0]
        )
        if cmd.heading_command:
            self.commands[env_ids, 3] = (
                torch.rand(n, device=self.device) * (cmd.heading[1] - cmd.heading[0])
                + cmd.heading[0]
            )
        small = (
            torch.norm(self.commands[env_ids, :2], dim=1) <= 0.2
        )
        self.commands[env_ids[small], :2] = 0.0

    def _push_robots_g1(self) -> None:
        push_interval = max(
            1,
            int(self.cfg.DomainRandCfg.push_interval_s / self.dt),
        )
        push_envs = (
            (self.episode_length_buf > 0)
            & (self.episode_length_buf % push_interval == 0)
        ).nonzero(as_tuple=False).flatten()
        if len(push_envs) == 0:
            return
        max_vel = self.cfg.DomainRandCfg.max_push_vel_xy
        self.root_states[push_envs, 7:9] += (
            torch.rand(len(push_envs), 2, device=self.device) * 2.0 - 1.0
        ) * max_vel
        for i in push_envs.tolist():
            self.sim_pool.write_root_state(i, self.root_states[i].cpu().numpy())

  # ── 观测 / 奖励 / 终止 ───────────────────────────────────────────────────

    def compute_observations(self) -> None:
        sin_phase = torch.sin(2 * math.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * math.pi * self.phase).unsqueeze(1)
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel_body * self.ang_vel_scale,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                self.dof_vel * self.dof_vel_scale,
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel_body * 2.0,
                    self.base_ang_vel_body * self.ang_vel_scale,
                    self.projected_gravity,
                    self.commands[:, :3] * self.commands_scale,
                    (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    self.dof_vel * self.dof_vel_scale,
                    self.actions,
                ),
                dim=-1,
            )
        if self.cfg.NoiseCfg.add_noise and self._noise_scale_vec is not None:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self._noise_scale_vec

    def compute_reward(self) -> None:
        self._compute_reward_all()

    def check_termination(self) -> None:
        self.reset_buf[:] = False
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length

        if self.termination_contact_indices:
            term_forces = self.contact_forces[:, self.termination_contact_indices, :]
            self.reset_buf |= torch.any(torch.norm(term_forces, dim=-1) > 1.0, dim=1)

        max_pitch = 1.0
        max_roll = 0.8
        self.reset_buf |= torch.logical_or(
            torch.abs(self.rpy[:, 1]) > max_pitch,
            torch.abs(self.rpy[:, 0]) > max_roll,
        )
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        assert self.sim_pool is not None
        for i in env_ids.tolist():
            data = self.sim_pool.data_list[i]
            self.sim_pool.reset_env(data)
            root = self.base_init_state.clone().cpu().numpy()
            root[0:2] += np.random.uniform(-1.0, 1.0, size=2)
            self.sim_pool.write_root_state(i, root)
            dof_pos = (
                self.default_dof_pos.cpu().numpy()
                * np.random.uniform(0.5, 1.5, size=self.num_actions)
            )
            self.sim_pool.write_dof_pos_vel(i, dof_pos, np.zeros(self.num_actions))

        self._resample_commands_g1(env_ids)
        self.location_O[env_ids] = self.env_origins[env_ids, :2]
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0
        self._refresh_sim_tensors()

    def _terrain_height_at_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def _build_noise_scale_vec(self) -> torch.Tensor:
        noise = self.cfg.NoiseCfg
        vec = torch.zeros(self.num_obs, device=self.device)
        vec[:3] = noise.ang_vel * noise.noise_level * self.ang_vel_scale
        vec[3:6] = noise.gravity * noise.noise_level
        vec[6:9] = 0.0
        na = self.num_actions
        vec[9 : 9 + na] = noise.dof_pos * noise.noise_level * self.dof_pos_scale
        vec[9 + na : 9 + 2 * na] = noise.dof_vel * noise.noise_level * self.dof_vel_scale
        return vec

    @staticmethod
    def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_w = q[:, 3:4]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
        c = q_vec * (torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0)
        return a + b + c

    # ── G1 特有奖励（移植自 g1_env.py）────────────────────────────────────

    def _reward_contact(self) -> torch.Tensor:
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(min(2, len(self.feet_indices))):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self) -> torch.Tensor:
        contact = (
            torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        )
        terrain_height = self._terrain_height_at_xy(
            self.feet_pos[:, :, 0],
            self.feet_pos[:, :, 1],
        )
        feet_height = self.feet_pos[:, :, 2] - terrain_height
        pos_error = torch.square(feet_height - 0.1) * (~contact)
        return torch.sum(pos_error, dim=1)

    def _reward_alive(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)

    def _reward_contact_no_vel(self) -> torch.Tensor:
        contact = (
            torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        )
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self) -> torch.Tensor:
        if self.hip_reward_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            torch.square(self.dof_pos[:, self.hip_reward_indices]), dim=1
        )

    def _reward_velx(self) -> torch.Tensor:
        return torch.tanh(self.base_lin_vel_body[:, 0]) * 3.0

    def _reward_vely(self) -> torch.Tensor:
        return torch.abs(self.base_lin_vel_body[:, 1]) * -3.0

    def _reward_posy(self) -> torch.Tensor:
        current_pos = self.base_pos[:, :2] - self.location_O
        return torch.abs(current_pos[:, 1]) * -3.0

    def _reward_posx(self) -> torch.Tensor:
        current_pos = self.base_pos[:, :2] - self.location_O
        return torch.tanh(current_pos[:, 0]) * 3.0

    def _reward_collision(self) -> torch.Tensor:
        if not self.penalised_contact_indices:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :],
                    dim=-1,
                )
                > 0.1
            ).float(),
            dim=1,
        )

    def _reward_dof_acc(self) -> torch.Tensor:
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )
