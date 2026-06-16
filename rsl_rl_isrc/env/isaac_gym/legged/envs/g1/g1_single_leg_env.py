"""单腿站立（任意一脚支撑）G1 环境。

该 env 继承自 G1Robot，新增以下两个专用奖励函数，
其余行为（观测构造、物理回调、重置逻辑）完全复用父类实现，
与行走任务互不干涉。

奖励逻辑
--------
``_reward_single_leg_contact``：
    恰好一只脚接触地面，**且**机器人处于站立姿态（非仰卧/侧卧），
    才给予正向奖励 1.0；以下任意情况得 0：

    - 双脚同时触地（双脚站立）
    - 完全腾空（跳起或跌倒腾空）
    - 仰卧 / 侧卧：躯干倾斜超过 60°（projected_gravity z < -0.5 不满足）
    - 躯干高度低于 0.5 m（跌倒躺地后骨盆高度通常 < 0.4 m）

    两道门控共同防止"仰卧刷分"：机器人躺倒后一脚踩地、一脚抬起
    虽能满足 XOR 条件，但倾斜角和高度会使门控拒绝该状态。

    接触判断阈值：法向接触力 z 分量 > 1 N（与基类 _reward_contact 一致）。
    倾斜门控阈值：projected_gravity[:, 2] < -0.5（躯干倾斜不超过 60°）。
    高度门控阈值：base_height > 0.5 m。

``_reward_ang_vel_z``：
    惩罚躯干绕竖直轴（偏航/yaw）的角速度，防止机器人通过持续转圈
    来维持平衡。基类 ``_reward_ang_vel_xy`` 只覆盖俯仰/侧滚，
    本函数专门补充 yaw 轴约束。
    计算：``base_ang_vel[:, 2]²``（与 ang_vel_xy 形式一致）。

``_reward_swing_leg_height``：
    奖励摆动腿（非支撑腿）的脚踝高度高于支撑腿，阻止机器人通过
    "双脚交替站立"来刷分。若仅做左右重心交替，摆动腿几乎不抬起，
    相对高度接近 0，无法获得此奖励；真正单腿站立并将摆动腿抬高
    才能持续获益。计算：摆动脚踝相对支撑脚踝的高度差，上限 0.15 m。
"""

import torch

from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_env import G1Robot

# 单腿站立奖励生效的最低骨盆高度（m）。
# 正常站立：~0.72-0.78m；仰卧倒地：~0.2-0.4m。
_MIN_STAND_HEIGHT = 0.5

# projected_gravity z 分量门控阈值。
# projected_gravity = quat_rotate_inverse(base_quat, [0,0,-1])
#   直立时 ≈ -1.0；侧卧/仰卧时 ≈ 0 或 +1；
#   < -0.5 ⟺ 躯干倾斜 < 60°（cos60° = 0.5）。
_MIN_UPRIGHT_GRAVITY_Z = -0.5


class G1SingleLegRobot(G1Robot):
    """单腿站立（任意一脚支撑）G1 机器人环境。

    在 ``G1Robot`` 基础上扩展 ``_reward_single_leg_contact``，
    配合 ``G1SingleLegCfg`` 中的奖励权重使用。
    """

    def _reward_single_leg_contact(self) -> torch.Tensor:
        """奖励恰好一只脚触地且躯干保持站立姿态。

        三重条件（全部满足才给分）：

        1. **单脚 XOR**：左右脚接触力 z 分量，恰好一个 > 1 N。
        2. **倾斜门控**：``projected_gravity[:, 2] < -0.5``，
           即躯干相对竖直方向倾斜不超过 60°。
           仰卧时重力在体轴 z 方向投影接近 0 或为正，无法通过此门控。
        3. **高度门控**：骨盆（base）高度 > 0.5 m。
           机器人躺倒时骨盆高度通常 < 0.4 m，无法通过此门控。

        Returns:
            shape ``(num_envs,)`` 的 float tensor：三重条件全部满足时为 1.0，否则为 0.0。
        """
        # ── 条件 1：单脚 XOR ──────────────────────────────────────────
        left_contact  = self.contact_forces[:, self.feet_indices[0], 2] > 1.0  # (E,)
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0  # (E,)
        single_leg = left_contact ^ right_contact                               # (E,) bool

        # ── 条件 2：躯干倾斜门控 ──────────────────────────────────────
        # projected_gravity = quat_rotate_inverse(base_quat, [0,0,-1])
        # 直立 → pg_z ≈ -1；侧卧/仰卧 → pg_z ≈ 0 或 +1
        # < -0.5 表示倾斜角 < 60°，允许适度侧倾但拒绝跌倒姿态
        is_upright = self.projected_gravity[:, 2] < _MIN_UPRIGHT_GRAVITY_Z     # (E,) bool

        # ── 条件 3：最低高度门控 ──────────────────────────────────────
        # root_states[:, 2] 为骨盆质心世界坐标 z（近似等于骨盆高度）
        is_high_enough = self.root_states[:, 2] > _MIN_STAND_HEIGHT            # (E,) bool

        return (single_leg & is_upright & is_high_enough).float()

    def _reward_ang_vel_z(self) -> torch.Tensor:
        """惩罚躯干偏航（yaw）角速度，阻止通过转圈来维持平衡。

        基类 ``_reward_ang_vel_xy`` 已惩罚俯仰（pitch）和侧滚（roll），
        但未覆盖偏航轴（z 轴）。机器人发现持续旋转可以借助角动量
        维持单腿平衡，形成投机策略。本奖励以二次方形式惩罚 yaw 角速度：
        慢速小幅修正（< 0.3 rad/s）惩罚极小，不影响正常平衡调整；
        持续转圈（> 1 rad/s）将积累显著惩罚，超过单腿奖励收益。

        Returns:
            shape ``(num_envs,)`` 的 float tensor：``base_ang_vel[:, 2]²``。
        """
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_swing_leg_height(self) -> torch.Tensor:
        """奖励摆动腿脚踝高度高于支撑腿，阻止双脚快速交替刷分。

        机器人若仅做重心左右交替（双脚轮流轻点地），每次切换时
        XOR 条件短暂成立，但摆动脚几乎不离地，相对高度差 ≈ 0，
        本奖励为 0。真正单腿站立并将摆动腿悬空才能持续获益。

        计算逻辑：
        - 支撑腿 = 有地面接触力的脚（z 分量 > 1 N）
        - 摆动腿 = 另一只脚
        - reward = clamp(摆动脚踝 z − 支撑脚踝 z, 0, 0.15)
        - 仅在单脚 XOR 条件成立时有效（与 single_leg_contact 联动）

        Returns:
            shape ``(num_envs,)`` 的 float tensor：
            摆动腿相对支撑腿抬高高度（m），范围 [0, 0.15]。
        """
        left_contact  = self.contact_forces[:, self.feet_indices[0], 2] > 1.0  # (E,)
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0  # (E,)
        single_leg    = left_contact ^ right_contact                            # (E,)

        # 脚踝在世界坐标系中的 z 高度
        left_height  = self.feet_pos[:, 0, 2]   # (E,) 左脚踝高度
        right_height = self.feet_pos[:, 1, 2]   # (E,) 右脚踝高度

        # 当左脚是支撑腿（left_contact=True）时，摆动腿=右脚
        # 奖励 = clamp(right_height - left_height, 0, 0.15)
        right_swing_rel = torch.clamp(right_height - left_height, min=0., max=0.15)

        # 当右脚是支撑腿（right_contact=True）时，摆动腿=左脚
        # 奖励 = clamp(left_height - right_height, 0, 0.15)
        left_swing_rel  = torch.clamp(left_height - right_height, min=0., max=0.15)

        # 合并：只在对应腿为摆动腿时有效，且仅在单脚条件成立时计入
        swing_reward = (
            right_swing_rel * left_contact.float()   # 左为支撑，奖励右脚高度
            + left_swing_rel  * right_contact.float() # 右为支撑，奖励左脚高度
        ) * single_leg.float()

        return swing_reward
