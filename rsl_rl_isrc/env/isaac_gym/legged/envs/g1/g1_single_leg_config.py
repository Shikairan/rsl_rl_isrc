"""单腿站立任务配置（G1SingleLegCfg / G1SingleLegCfgPPO）。

该配置与行走任务（G1RoughCfg）完全独立，互不干涉：
- 关闭所有行走相关奖励（velx / vely / posx / posy / tracking_lin_vel / tracking_ang_vel / contact）
- 将速度指令范围置零（机器人不需要向任何方向移动）
- 开启 single_leg_contact 单腿触地奖励（大权重，作为主信号）
- 保留姿态稳定、存活等通用奖励（已针对 29 DOF 重新标定量级）

奖励设计说明（针对训练 iter~240 观测到的问题修正）
-------------------------------------------------
问题根因：
  1. 负向奖励（dof_vel / ang_vel_xy 等）scale 沿用 12dof 设计，
     29dof 关节数扩大 2.4 倍，累积惩罚远超正向信号，
     导致每步总奖励接近零被 only_positive_rewards 裁掉，
     优势函数方差趋近 0，价值网络无法学习。
  2. single_leg_contact 权重过小，正向信号被完全淹没。

修正原则：
  - 正向信号（alive + single_leg_contact）必须在每一步的总奖励中占主导
  - 负向惩罚项按 DOF 数量折算（12→29，约 0.4x 缩放）并进一步减小
  - 单腿平衡需要角动量调节，ang_vel_xy 惩罚大幅降低
  - 保留足够的正则化防止关节乱抖，但不能淹没任务信号
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
        """奖励设计：正向信号主导，负向惩罚适配 29 DOF 量级。"""

        # 单腿站立目标重心高度（略低于行走 0.78，重心更稳）
        base_height_target = 0.72

        class scales(G1RoughCfg.rewards.scales):
            # ── 禁用行走奖励 ──────────────────────────────────────────────
            velx = 0.0
            vely = 0.0
            posx = 0.0
            posy = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

            # 步态相位 contact / 摆腿高度与单腿站立无关，关闭
            contact = 0.0
            feet_swing_height = 0.0

            # ── 正向信号：必须在每步总奖励中占主导 ──────────────────────

            # 核心任务奖励：恰好一脚触地即得正向激励
            # 从训练日志可知，约 45% 的时间已达成单腿接触，但奖励被淹没；
            # 大幅提权，使正向信号主导每步总奖励。
            single_leg_contact = 8.0

            # 摆动腿高度奖励（阻止双脚交替刷分）
            # 机器人若仅做左右重心交替，摆动脚几乎不离地，相对高度 ≈ 0，奖励为 0。
            # 真正将摆动腿抬起才能持续获益，迫使机器人学习真实单腿站立。
            # 计算：摆动脚踝相对支撑脚踝高度差（0~0.15m，见 _reward_swing_leg_height）。
            swing_leg_height = 5.0

            # 存活奖励：提高基础正向底线，保证每步不被裁到 0
            alive = 3.0

            # ── 姿态惩罚（减小，适配 29 DOF 与单腿平衡需求）─────────────

            # 单腿平衡需要躯干适度倾斜和角动量调节，放宽姿态限制
            orientation = -0.2       # 原 -1.0，减小 5x

            # 重心高度维持（scale 减小，避免过度惩罚过渡动作）
            base_height = -3.0       # 原 -8.0，减小

            # z 方向速度（防止跳动，但不需太强）
            lin_vel_z = -0.5         # 原 -2.0，减小 4x

            # 单腿平衡依赖 roll/pitch 角速度调节，大幅放宽
            ang_vel_xy = -0.005      # 原 -0.05，减小 10x（俯仰/侧滚放宽）

            # 偏航角速度（yaw）惩罚：防止转圈保持平衡
            # 基类 ang_vel_xy 不覆盖 z 轴，需单独设置。
            # 量级分析（假设 single_leg_contact 每步约 +0.112）：
            #   0.3 rad/s 转圈：-0.3 × 0.09 × 0.02 = -0.00054/步  可接受小幅修正
            #   1.0 rad/s 转圈：-0.3 × 1.00 × 0.02 = -0.006/步   开始有明显惩罚
            #   2.0 rad/s 转圈：-0.3 × 4.00 × 0.02 = -0.024/步   约单腿奖励 21%，显著
            #   3.0 rad/s 转圈：-0.5 × 9.00 × 0.02 = -0.090/步   约单腿奖励 80%，强烈抑制
            # iter649 日志显示 ang_vel_z=-0.1624，仍有显著旋转，从 -0.3 加强至 -0.5
            ang_vel_z = -0.5

            # 关节加速度：29 DOF 总量放大，需按 DOF 比例缩放
            dof_acc = -2.5e-8        # 原 -2.5e-7，减小 10x（29/12 ≈ 2.4，额外留余量）

            # 关节速度：同上，29 DOF 总量放大
            dof_vel = -1e-4          # 原 -1e-3，减小 10x

            # 动作变化率：适度平滑，不过度限制探索
            action_rate = -0.002     # 原 -0.01，减小 5x

            # 关节位置限制：保留但减小
            dof_pos_limits = -1.0    # 原 -5.0，减小 5x

            # 髋关节偏移：适当放宽，单腿需要髋部参与平衡
            hip_pos = -0.2           # 原 -1.0，减小 5x

            # 触地时脚部速度：放宽，单腿站立时支撑脚可能有微动
            contact_no_vel = -0.05   # 原 -0.45，减小 9x

            # 力矩：29 DOF 总量放大，略微减小
            torques = -5e-6          # 原 -1e-5，减小 2x


class G1SingleLegCfgPPO(G1RoughCfgPPO):
    """单腿站立任务的 PPO 训练配置（与行走实验完全独立）。"""

    class runner(G1RoughCfgPPO.runner):
        experiment_name = "g1_single_leg"
        run_name = ""
        max_iterations = 10000
