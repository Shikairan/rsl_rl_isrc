from rsl_rl_isrc.env.isaac_gym.legged.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # ── 腿部（12 DOF，与 12dof 模型完全一致）──────────────────────
            'left_hip_yaw_joint'   : 0.,
            'left_hip_roll_joint'  : 0.,
            'left_hip_pitch_joint' : -0.1,
            'left_knee_joint'      : 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint' : 0.,
            'right_hip_yaw_joint'  : 0.,
            'right_hip_roll_joint' : 0.,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint'     : 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.,
            # ── 腰部（3 DOF，29dof 新增）──────────────────────────────────
            'waist_yaw_joint'  : 0.,
            'waist_roll_joint' : 0.,
            'waist_pitch_joint': 0.,
            # ── 左臂（7 DOF，29dof 新增）──────────────────────────────────
            'left_shoulder_pitch_joint': 0.,
            'left_shoulder_roll_joint' : 0.,
            'left_shoulder_yaw_joint'  : 0.,
            'left_elbow_joint'         : 0.,
            'left_wrist_roll_joint'    : 0.,
            'left_wrist_pitch_joint'   : 0.,
            'left_wrist_yaw_joint'     : 0.,
            # ── 右臂（7 DOF，29dof 新增）──────────────────────────────────
            'right_shoulder_pitch_joint': 0.,
            'right_shoulder_roll_joint' : 0.,
            'right_shoulder_yaw_joint'  : 0.,
            'right_elbow_joint'         : 0.,
            'right_wrist_roll_joint'    : 0.,
            'right_wrist_pitch_joint'   : 0.,
            'right_wrist_yaw_joint'     : 0.,
        }

    class env(LeggedRobotCfg.env):
        num_actions = 29
        # obs = ang_vel(3) + gravity(3) + commands(3) + dof_pos(29) + dof_vel(29) + actions(29) + sin/cos(2)
        num_observations = 98
        # privileged = lin_vel(3) + ang_vel(3) + gravity(3) + commands(3) + dof_pos(29) + dof_vel(29) + actions(29) + sin/cos(2)
        num_privileged_obs = 101

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        stiffness = {
            # ── 腿部 ──────────────────────────────────────────────────────
            'hip_yaw'  : 100,
            'hip_roll' : 100,
            'hip_pitch': 100,
            'knee'     : 150,
            'ankle'    : 40,
            # ── 腰部 ──────────────────────────────────────────────────────
            'waist'    : 100,
            # ── 手臂 ──────────────────────────────────────────────────────
            'shoulder' : 40,
            'elbow'    : 40,
            'wrist'    : 20,
        }  # [N*m/rad]
        damping = {
            # ── 腿部 ──────────────────────────────────────────────────────
            'hip_yaw'  : 2,
            'hip_roll' : 2,
            'hip_pitch': 2,
            'knee'     : 4,
            'ankle'    : 2,
            # ── 腰部 ──────────────────────────────────────────────────────
            'waist'    : 3,
            # ── 手臂 ──────────────────────────────────────────────────────
            'shoulder' : 2,
            'elbow'    : 2,
            'wrist'    : 1,
        }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{G1_DESCRIPTION_DIR}/g1_29dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales( LeggedRobotCfg.rewards.scales ):
            velx = 2
            vely = 2
            posx = 2
            posy = 2
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

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims =[512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 128
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        num_steps_per_env = 60
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
