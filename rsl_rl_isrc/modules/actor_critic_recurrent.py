# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""带 LSTM/GRU 的 Actor-Critic，用于需序列记忆的决策。"""

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl_isrc.utils import unpad_trajectories

class ActorCriticRecurrent(ActorCritic):
    """在 MLP 策略/价值头之前插入 Actor/Critic 各自的 RNN，将原始观测编码为固定维隐状态。

    ``act``/``evaluate`` 在训练期需传入 ``masks`` 与上一步 ``hidden_states``；采集期由内部记忆递推。
    """

    #is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        #self.is_recurrent = True
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)
        self.is_recurrent = True
        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        """在回合结束掩码 ``dones`` 处将 RNN 隐状态清零（并行 env 维）。"""
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        """经 Actor-RNN 编码后调用基类 ``act`` 采样动作；``masks``/``hidden_states`` 用于反传阶段。"""
        input_a = self.memory_a(observations, masks, hidden_states)
        #print(observations.shape)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        """推理模式：仅 Actor-RNN + 确定性动作均值（无显式采样噪声路径）。"""
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """经 Critic-RNN 编码后计算状态价值标量。"""
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        """返回 (actor_hidden, critic_hidden)，供存储或分布式同步使用。"""
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    """单路 LSTM/GRU：支持 ``forward`` 在「填充轨迹 batch」与「单步推理」两种形状约定下切换。"""

    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        """若提供 ``masks`` 则按批训练路径解包轨迹；否则按单步 ``inference`` 递推并写回 ``hidden_states``。"""
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            #rint("memory:", input.shape, out.shape, masks.shape)

            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            #print("self.hidden_states:", input.shape, self.hidden_states[0].shape)
        return out

    def reset(self, dones=None):
        """将 ``dones`` 为 True 的并行环境对应的隐状态切片清零。"""
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
