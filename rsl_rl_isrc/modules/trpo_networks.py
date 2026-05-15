# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl_isrc.utils import unpad_trajectories


class TrpoPolicy(nn.Module):
    """TRPO策略网络"""
    def __init__(self, num_inputs, num_outputs):
        super(TrpoPolicy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.01)  # 进一步减小动作均值的权重初始化
        self.action_mean.bias.data.mul_(0.0)

        # 初始化动作对数标准差为-1（标准差约为0.37），提供适度的探索
        #self.action_log_std = nn.Parameter(torch.full((1, num_outputs), -1.0))
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        # 正确扩展action_log_std以匹配action_mean的batch维度
        batch_size = action_mean.shape[0]
        action_log_std = self.action_log_std.expand(batch_size, -1)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class TrpoValueFunction(nn.Module):
    """TRPO价值函数网络"""
    def __init__(self, num_inputs):
        super(TrpoValueFunction, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 256)  # 增加到256维
        self.affine2 = nn.Linear(256, 256)
        self.value_head = nn.Linear(256, 1)

        # ✅ 正确的初始化：避免冲突，直接设置合理值
        nn.init.xavier_normal_(self.affine1.weight, gain=1.0)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_normal_(self.affine2.weight, gain=1.0)
        nn.init.zeros_(self.affine2.bias)
        nn.init.xavier_normal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, -200.0)  # 直接设置为合理值

        # 添加网络检查
        self.check_network(num_inputs)
    def check_network(self, num_inputs):
        """检查网络是否能产生有意义的变化"""
        test_input = torch.randn(10, num_inputs)
        with torch.no_grad():
            output = self.forward(test_input)
            output_range = output.max() - output.min()
            print(f"✅ 价值函数初始化检查 - 输出范围: {output_range:.3f}")
            if output_range < 1.0:
                print("⚠️ 网络输出变化太小，可能初始化有问题")
            else:
                print("✅ 网络初始化正常")

    def forward(self, x):
        x = torch.relu(self.affine1(x))  # 改用ReLU避免梯度消失
        x = torch.relu(self.affine2(x))
        state_values = self.value_head(x)
        return state_values


class TrpoPolicyRecurrent(nn.Module):
    """带RNN的TRPO策略网络"""
    is_recurrent = True

    def __init__(self, num_inputs, num_outputs, rnn_type='lstm', rnn_hidden_size=256, rnn_num_layers=1):
        super(TrpoPolicyRecurrent, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # RNN层
        self.memory = Memory(num_inputs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # 策略网络（输入是RNN的输出）
        self.affine1 = nn.Linear(rnn_hidden_size, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.01)  # 进一步减小动作均值的权重初始化
        self.action_mean.bias.data.mul_(0.0)

        # 初始化动作对数标准差为-1（标准差约为0.37），提供适度的探索
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        print(f"TrpoPolicyRecurrent: RNN({rnn_type}, {rnn_hidden_size}) -> Policy")

    def forward(self, x, masks=None, hidden_states=None):
        # RNN处理
        input_a = self.memory(x, masks, hidden_states)

        # 如果是序列数据，需要压缩维度
        if input_a.dim() == 3:  # [batch_size, seq_len, rnn_hidden_size]
            input_a = input_a.squeeze(1)  # [batch_size, rnn_hidden_size]

        # MLP处理
        x = torch.tanh(self.affine1(input_a))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        # 正确扩展action_log_std以匹配action_mean的batch维度
        batch_size = action_mean.shape[0]
        action_log_std = self.action_log_std.expand(batch_size, -1)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def act(self, observations, masks=None, hidden_states=None):
        action_mean, action_log_std, action_std = self(observations, masks, hidden_states)
        return torch.normal(action_mean, action_std)

    def evaluate_actions(self, observations, actions, masks=None, hidden_states=None):
        action_mean, action_log_std, action_std = self(observations, masks, hidden_states)

        # 计算对数概率
        var = action_std.pow(2)
        log_prob = -((actions - action_mean).pow(2)) / (2 * var) - action_log_std - np.log(np.sqrt(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)

        return log_prob

    def get_entropy(self, observations, masks=None, hidden_states=None):
        _, action_log_std, action_std = self(observations, masks, hidden_states)
        # 熵的计算
        entropy = action_log_std + 0.5 * np.log(2 * np.pi * np.e)
        return entropy.sum(dim=-1)

    def reset(self, dones=None):
        self.memory.reset(dones)

    def get_hidden_states(self):
        return self.memory.hidden_states


class TrpoValueFunctionRecurrent(nn.Module):
    """带RNN的TRPO价值函数网络"""
    is_recurrent = True

    def __init__(self, num_inputs, rnn_type='lstm', rnn_hidden_size=256, rnn_num_layers=1):
        super(TrpoValueFunctionRecurrent, self).__init__()

        self.num_inputs = num_inputs

        # RNN层
        self.memory = Memory(num_inputs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # 价值函数网络（输入是RNN的输出）
        self.affine1 = nn.Linear(rnn_hidden_size, 256)  # 增加到256维
        self.affine2 = nn.Linear(256, 256)
        self.value_head = nn.Linear(256, 1)

        # ✅ 正确的初始化：避免冲突，直接设置合理值
        nn.init.xavier_normal_(self.affine1.weight, gain=1.0)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_normal_(self.affine2.weight, gain=1.0)
        nn.init.zeros_(self.affine2.bias)
        nn.init.xavier_normal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, -200.0)  # 直接设置为合理值

        # 添加网络检查
        self.check_network(self.num_inputs)

        print(f"TrpoValueFunctionRecurrent: RNN({rnn_type}, {rnn_hidden_size}) -> Value")

    def check_network(self, input_size):
        """检查网络是否能产生有意义的变化"""
        # 对于RNN网络，我们需要创建正确的输入形状
        test_input = torch.randn(1, input_size)  # [batch_size, input_size] - 使用batch_size=1
        with torch.no_grad():
            # 重置hidden states
            self.memory.hidden_states = None
            output = self.forward(test_input)
            output_range = output.max() - output.min()
            print(f"✅ RNN价值函数初始化检查 - 输出范围: {output_range:.3f}")
            if output_range < 1.0:
                print("⚠️ RNN网络输出变化太小，可能初始化有问题")
            else:
                print("✅ RNN网络初始化正常")

    def forward(self, x, masks=None, hidden_states=None):
        # RNN处理
        input_c = self.memory(x, masks, hidden_states)

        # 如果是序列数据，需要压缩维度
        if input_c.dim() == 3:  # [batch_size, seq_len, rnn_hidden_size]
            input_c = input_c.squeeze(1)  # [batch_size, rnn_hidden_size]

        # MLP处理
        x = torch.relu(self.affine1(input_c))  # 改用ReLU避免梯度消失
        x = torch.relu(self.affine2(x))
        state_values = self.value_head(x)
        return state_values

    def reset(self, dones=None):
        self.memory.reset(dones)

    def get_hidden_states(self):
        return self.memory.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)

            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states is a list with hidden_state and cell_state
        if dones is None:
            # Complete reset - set hidden states to None for fresh start
            self.hidden_states = None
        elif self.hidden_states is not None:
            for hidden_state in self.hidden_states:
                hidden_state[..., dones, :] = 0.0
