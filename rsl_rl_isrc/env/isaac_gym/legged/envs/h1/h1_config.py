from rsl_rl_isrc.env.isaac_gym.legged.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class H1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.95]
        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 41
        num_privileged_obs = 44
        num_actions = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.6, 1.25]
        randomize_base_mass = False
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 0.8

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_yaw": 100,
            "hip_roll": 100,
            "hip_pitch": 100,
            "knee": 150,
            "ankle": 40,
        }
        damping = {
            "hip_yaw": 2,
            "hip_roll": 2,
            "hip_pitch": 2,
            "knee": 4,
            "ankle": 2,
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{H1_DESCRIPTION_DIR}/h1_10dof.urdf"
        name = "h1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.95

        class scales(LeggedRobotCfg.rewards.scales):
            velx = 0.0
            vely = 0.0
            posx = 0.0
            posy = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -2.0
            base_height = -15.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.02
            dof_pos_limits = -5.0
            alive = 1.5
            hip_pos = -0.5
            contact_no_vel = -0.25
            feet_swing_height = -10.0
            contact = 0.1
            stability = 2.0


class H1RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.6
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 128
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ""
        experiment_name = "h1"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
