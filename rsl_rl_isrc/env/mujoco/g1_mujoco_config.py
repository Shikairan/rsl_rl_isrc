"""G1 MuJoCo 环境配置（对齐 Isaac G1RoughCfg）。"""

from __future__ import annotations

import os

from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg


def g1_scene_xml_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "robotmodel", "g1_description", "scene.xml")
    )


class G1MujocoCfg(BaseEnvCfg):
    """MuJoCo G1 训练配置（嵌套类风格，与 HandBaseEnv 一致）。"""

    num_envs: int = 64
    episode_length_s: float = 20.0
    env_spacing: float = 3.0
    send_timeouts: bool = True
    seed: int = 1

    class SimParamsCfg(BaseEnvCfg.SimParamsCfg):
        dt = 0.005
        substeps = 1
        gravity = (0.0, 0.0, -9.81)
        use_gpu_pipeline = False

    class ObsCfg(BaseEnvCfg.ObsCfg):
        num_obs = 47
        num_privileged_obs = 48
        send_timeouts = True

    class ControlCfg(BaseEnvCfg.ControlCfg):
        control_type = "P"
        stiffness = {
            "hip_yaw": 100.0,
            "hip_roll": 100.0,
            "hip_pitch": 100.0,
            "knee": 150.0,
            "ankle": 40.0,
        }
        damping = {
            "hip_yaw": 2.0,
            "hip_roll": 2.0,
            "hip_pitch": 2.0,
            "knee": 4.0,
            "ankle": 2.0,
        }
        action_scale = 0.25
        decimation = 4

    class CommandsCfg(BaseEnvCfg.CommandsCfg):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        lin_vel_x = (0.7, 0.8)
        lin_vel_y = (-0.1, 0.1)
        ang_vel_yaw = (-1.0, 1.0)
        heading = (-0.11, 0.1)

    class InitStateCfg(BaseEnvCfg.InitStateCfg):
        pos = (0.0, 0.0, 0.8)
        rot = (0.0, 0.0, 0.0, 1.0)
        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
        }

    class AssetCfg(BaseEnvCfg.AssetCfg):
        file = g1_scene_xml_path()
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]

    class DomainRandCfg(BaseEnvCfg.DomainRandCfg):
        randomize_friction = True
        friction_range = (0.1, 1.25)
        randomize_base_mass = True
        added_mass_range = (-1.0, 3.0)
        push_robots = True
        push_interval_s = 5.0
        max_push_vel_xy = 1.5

    class NoiseCfg(BaseEnvCfg.NoiseCfg):
        add_noise = True
        noise_level = 1.0
        ang_vel = 0.2
        gravity = 0.05
        dof_pos = 0.01
        dof_vel = 1.5

    class RewardCfg(BaseEnvCfg.RewardCfg):
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales(BaseEnvCfg.RewardCfg.scales):
            velx = 2.0
            vely = 2.0
            posx = 2.0
            posy = 2.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.45
            hip_pos = -1.0
            contact_no_vel = -0.45
            feet_swing_height = -25.0
            contact = 0.18
            torques = -0.00001


def build_g1_ppo_train_cfg() -> dict:
    """从 G1RoughCfgPPO 映射训练超参（无 Isaac 依赖）。"""
    return {
        "seed": 1,
        "runner": {
            "policy_class_name": "ActorCriticRecurrent",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": 10000,
            "save_interval": 500,
            "experiment_name": "g1_mujoco",
            "run_name": "robotmodel_mujoco",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        },
        "policy": {
            "policy_class_name": "ActorCriticRecurrent",
            "init_noise_std": 0.8,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "rnn_type": "lstm",
            "rnn_hidden_size": 128,
            "rnn_num_layers": 1,
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 5.0e-4,
            "schedule": "adaptive",
            "gamma": 0.95,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
