from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Tuple, TYPE_CHECKING

from rsl_rl_isrc.env.isaac_gym.legged import ensure_h1_10dof_urdf
from rsl_rl_isrc.env.isaac_gym.make_g1_isaac import (
    _class_to_dict,
    _import_isaac,
    _make_sim_args,
    normalize_train_cfg,
)

if TYPE_CHECKING:
    from rsl_rl_isrc.env.isaac_gym.isaac_g1_vec_env import IsaacG1VecEnv


SUPPORTED_ROBOTS = ("g1_12dof", "h1_10dof")
SUPPORTED_SCENES = ("flat", "slope", "collision")


def _robot_short_name(robot: str) -> str:
    return robot.split("_", 1)[0]


def _set_zero_command_cfg(cfg) -> None:
    cfg.commands.heading_command = False
    cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.ranges.heading = [0.0, 0.0]


def _set_common_stability_cfg(cfg) -> None:
    cfg.env.deterministic_reset = True
    cfg.env.init_base_velocity_range = 0.0
    cfg.env.max_pitch = 1.0
    cfg.env.max_roll = 0.8
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    _set_zero_command_cfg(cfg)

    scales = cfg.rewards.scales
    scales.velx = 0.0
    scales.vely = 0.0
    scales.posx = 0.0
    scales.posy = 0.0
    scales.tracking_lin_vel = 1.0
    scales.tracking_ang_vel = 0.5
    scales.orientation = -2.0
    scales.base_height = -15.0
    scales.action_rate = -0.02
    scales.alive = 1.5
    scales.stability = 2.0
    scales.termination = -2.0


def _apply_scene_cfg(cfg, scene: str) -> None:
    cfg.terrain.scene_type = scene
    cfg.terrain.ramp_angle_deg = 0.0
    if scene == "slope":
        cfg.terrain.ramp_angle_deg = 8.0
        cfg.terrain.static_friction = 1.2
        cfg.terrain.dynamic_friction = 1.0
    elif scene == "collision":
        cfg.domain_rand.push_robots = True
        cfg.domain_rand.push_interval_s = 15.0
        cfg.domain_rand.max_push_vel_xy = 0.05
        cfg.rewards.scales.collision = -1.0
        cfg.rewards.scales.termination = -3.0


def build_humanoid_cfg(
    robot: str,
    scene: str,
    num_envs: int,
    max_iterations: int,
) -> Tuple[object, dict]:
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unsupported robot {robot!r}; expected one of {SUPPORTED_ROBOTS}")
    if scene not in SUPPORTED_SCENES:
        raise ValueError(f"Unsupported scene {scene!r}; expected one of {SUPPORTED_SCENES}")

    if robot == "g1_12dof":
        from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_config import (
            G1RoughCfg,
            G1RoughCfgPPO,
        )

        cfg = G1RoughCfg()
        train_cfg = normalize_train_cfg(_class_to_dict(G1RoughCfgPPO()))
    else:
        ensure_h1_10dof_urdf()
        from rsl_rl_isrc.env.isaac_gym.legged.envs.h1.h1_config import (
            H1RoughCfg,
            H1RoughCfgPPO,
        )

        cfg = H1RoughCfg()
        train_cfg = normalize_train_cfg(_class_to_dict(H1RoughCfgPPO()))

    cfg.env.num_envs = int(num_envs)
    _set_common_stability_cfg(cfg)
    _apply_scene_cfg(cfg, scene)

    short = _robot_short_name(robot)
    train_cfg["runner"]["experiment_name"] = f"{short}_{scene}"
    train_cfg["runner"]["run_name"] = robot
    train_cfg["runner"]["max_iterations"] = int(max_iterations)
    train_cfg["runner"]["save_interval"] = 100
    train_cfg["policy"]["init_noise_std"] = 0.2
    train_cfg["algorithm"]["entropy_coef"] = 0.001
    return cfg, train_cfg


def default_humanoid_log_dir(train_cfg: dict, log_root: str | None = None) -> str:
    from rsl_rl_isrc.utils.paths import build_run_log_dir

    return build_run_log_dir(train_cfg, log_root=log_root)


def default_humanoid_checkpoint_dir(
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


def default_humanoid_run_dirs(
    train_cfg: dict,
    log_root: str | None = None,
    checkpoint_root: str | None = None,
) -> tuple[str, str]:
    from rsl_rl_isrc.utils.paths import build_run_dirs

    return build_run_dirs(train_cfg, log_root=log_root, checkpoint_root=checkpoint_root)


def make_humanoid_isaac_env(
    robot: str = "g1_12dof",
    scene: str = "flat",
    num_envs: int = 64,
    max_iterations: int = 10000,
    headless: bool = True,
    sim_device: str = "cuda:0",
) -> Tuple["IsaacG1VecEnv", object, dict]:
    gymapi = _import_isaac()

    from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_env import G1Robot
    from rsl_rl_isrc.env.isaac_gym.legged.utils.helpers import (
        parse_sim_params,
        set_seed,
    )
    from rsl_rl_isrc.env.isaac_gym.isaac_g1_vec_env import IsaacG1VecEnv

    cfg, train_cfg = build_humanoid_cfg(
        robot=robot,
        scene=scene,
        num_envs=num_envs,
        max_iterations=max_iterations,
    )
    set_seed(train_cfg.get("seed", 1))

    args = _make_sim_args(sim_device, headless, gymapi)
    sim_params = parse_sim_params(args, {"sim": _class_to_dict(cfg.sim)})

    robot_env = G1Robot(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=sim_device,
        headless=headless,
    )
    env = IsaacG1VecEnv(robot_env)
    return env, cfg, train_cfg
