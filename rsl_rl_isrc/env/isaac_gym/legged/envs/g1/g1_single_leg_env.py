"""单腿站立（任意一脚支撑）G1 环境。

该 env 继承自 G1Robot，仅新增 ``_reward_single_leg_contact`` 奖励函数，
其余行为（观测构造、物理回调、重置逻辑）完全复用父类实现，
与行走任务互不干涉。

奖励逻辑
--------
``_reward_single_leg_contact``：
    当且仅当恰好有一只脚（左脚 **或** 右脚，任意一脚）处于接触状态时，
    给予正向奖励 1.0；双脚同时触地或完全腾空时奖励为 0。

    接触判断阈值：法向接触力 z 分量 > 1 N（与基类 _reward_contact 一致）。
"""

import torch

from rsl_rl_isrc.env.isaac_gym.legged.envs.g1.g1_env import G1Robot


class G1SingleLegRobot(G1Robot):
    """单腿站立（任意一脚支撑）G1 机器人环境。

    在 ``G1Robot`` 基础上扩展 ``_reward_single_leg_contact``，
    配合 ``G1SingleLegCfg`` 中的奖励权重使用。
    """

    def _reward_single_leg_contact(self) -> torch.Tensor:
        """奖励恰好一只脚触地（任意一脚支撑）。

        Returns:
            shape ``(num_envs,)`` 的 float tensor：
            恰好单脚触地时为 1.0，否则为 0.0。
        """
        # 取两只脚的法向接触力（z 方向）判断是否接触
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0   # (E,)
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0  # (E,)

        # XOR：左右脚恰好一个接触 = 单腿站立成立
        single_leg = left_contact ^ right_contact   # bool (E,)
        return single_leg.float()
