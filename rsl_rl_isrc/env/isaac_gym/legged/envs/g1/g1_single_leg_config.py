"""单腿站立任务配置（G1SingleLegCfg / G1SingleLegCfgPPO）。

该配置与行走任务（G1RoughCfg）完全独立，互不干涉：
- 关闭所有行走相关奖励（velx / vely / posx / posy / tracking_lin_vel / tracking_ang_vel / contact）
- 将速度指令范围置零（机器人不需要向任何方向移动）
- 开启 single_leg_contact 单腿触地一致性奖励
- 保留姿态稳定、存活、关节限制等通用奖励
"""

from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1SingleLegCfg(G1RoughCfg):
    """单腿站立（任意一脚支撑）环境配置。"""

    class commands(G1RoughCfg.commands):
        """速度指令全部置零，机器人只需保持原地单腿站立。"""

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(G1RoughCfg.rewards):
        """奖励设计：禁用行走相关项，启用单腿触地奖励。"""

        # 单腿站立的目标站立高度略低于行走（重心更稳）
        base_height_target = 0.72

        class scales(G1RoughCfg.rewards.scales):
            # ── 禁用行走奖励 ──────────────────────────────────────────────
            velx = 0.0
            vely = 0.0
            posx = 0.0
            posy = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

            # 步态相位 contact / 摆腿高度 与单腿站立不相关，关闭
            contact = 0.0
            feet_swing_height = 0.0

            # ── 单腿触地奖励（核心新增项） ────────────────────────────────
            # 当且仅当恰好一只脚接触地面时给予正向奖励
            single_leg_contact = 1.0

            # ── 保留通用稳定性奖励（与行走任务共享实现，数值独立可调） ──
            alive = 0.45
            orientation = -1.0
            base_height = -8.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            action_rate = -0.01
            dof_pos_limits = -5.0
            hip_pos = -1.0
            contact_no_vel = -0.45


class G1SingleLegCfgPPO(G1RoughCfgPPO):
    """单腿站立任务的 PPO 训练配置（与行走实验完全独立）。"""

    class runner(G1RoughCfgPPO.runner):
        experiment_name = "g1_single_leg"
        run_name = ""
        max_iterations = 10000
