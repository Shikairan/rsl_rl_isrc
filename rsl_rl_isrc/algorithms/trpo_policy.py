#!/usr/bin/env python3
"""
真正的TRPO (Trust Region Policy Optimization) 实现
基于rsl_rl框架，支持连续动作空间
"""

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl_isrc.modules import TrpoPolicy, TrpoValueFunction, TrpoPolicyRecurrent, TrpoValueFunctionRecurrent
from rsl_rl_isrc.algorithms import TRPO
from rsl_rl_isrc.utils import RunningMeanStd


class TRPOPolicy:
    """
    真正的TRPO实现，使用KL约束和共轭梯度
    支持连续动作空间（Pendulum等）
    """

    def __init__(self,
                 num_obs,
                 num_actions,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 gamma=0.995,
                 tau=0.97,
                 max_kl=1e-2,
                 damping=1e-1,
                 l2_reg=1e-3,
                 vf_lr=1e-3,       # ✅ 价值函数学习率
                 vf_iters=20,      # ✅ 价值函数迭代次数
                 action_bounds=None,  # ✅ 新增：动作边界
                 rnn_hidden_size=0,  # > 0 to enable RNN
                 rnn_type='lstm',    # 'lstm' or 'gru'
                 rnn_num_layers=1,   # Number of RNN layers
                 device='cpu',
                 **kwargs):
        if kwargs:
            print("TRPOPolicy.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        self.device = device
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.tau = tau
        self.max_kl = max_kl
        self.damping = damping
        self.l2_reg = l2_reg

        # 检查是否使用RNN（通过检查rnn_hidden_size是否大于0）
        use_rnn = rnn_hidden_size > 0

        if use_rnn:
            # 创建RNN版本的策略网络和价值网络
            self.policy_net = TrpoPolicyRecurrent(
                num_inputs=num_obs,
                num_outputs=num_actions,
                rnn_type=rnn_type,
                rnn_hidden_size=rnn_hidden_size,
                rnn_num_layers=rnn_num_layers
            )
            self.value_net = TrpoValueFunctionRecurrent(
                num_inputs=num_obs,
                rnn_type=rnn_type,
                rnn_hidden_size=rnn_hidden_size,
                rnn_num_layers=rnn_num_layers
            )
            print(f"TRPOPolicy: Using RNN networks (hidden_size={rnn_hidden_size}, type={rnn_type})")
        else:
            # 创建标准版本的策略网络和价值网络
            self.policy_net = TrpoPolicy(num_inputs=num_obs, num_outputs=num_actions)
            self.value_net = TrpoValueFunction(num_inputs=num_obs)
            print("TRPOPolicy: Using standard networks")

        # 移动到指定设备
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        # 创建观察状态标准化器
        self.obs_rms = RunningMeanStd(shape=(num_obs,), clip=1.0)

        # 训练/测试模式标志
        self.training = True

        # 创建TRPO算法实例
        self.algorithm = TRPO(
            policy_net=self.policy_net,
            value_net=self.value_net,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            gamma=gamma,
            tau=tau,
            max_kl=max_kl,
            damping=damping,
            l2_reg=l2_reg,
            vf_lr=vf_lr,       # ✅ 传递价值函数学习率
            vf_iters=vf_iters, # ✅ 传递价值函数迭代次数
            action_bounds=action_bounds,  # ✅ 传递动作边界
            device=device
        )

        # 存储初始化标志
        self.storage_initialized = False

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        """初始化存储"""
        self.algorithm.init_storage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape)
        self.storage_initialized = True

    def test_mode(self):
        """切换到测试模式"""
        self.algorithm.test_mode()
        self.training = False

    def train_mode(self):
        """切换到训练模式"""
        self.algorithm.train_mode()
        self.training = True

    def act(self, obs, critic_obs=None):
        """
        执行动作
        obs: 观察状态
        critic_obs: 批评家观察状态（如果为None，则使用obs）
        """
        if critic_obs is None:
            critic_obs = obs
        # 确保输入是tensor并且在正确的设备上
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) if not isinstance(obs, torch.Tensor) else obs.to(self.device)
        critic_obs = torch.tensor(critic_obs, dtype=torch.float32, device=self.device) if not isinstance(critic_obs, torch.Tensor) else critic_obs.to(self.device)
        #self.algorithm.transition.observations = obs
        #self.algorithm.transition.critic_observations = critic_obs
        # 标准化观察状态（训练时更新统计）
        obs = self.obs_rms(obs, update=self.training)
        critic_obs = self.obs_rms(critic_obs, update=self.training)

        return self.algorithm.act(obs, critic_obs)

    def act_with_hidden_states(self, obs, critic_obs=None, policy_hidden_states=None, value_hidden_states=None):
        """
        执行动作（支持RNN隐藏状态）
        obs: 观察状态
        critic_obs: 批评家观察状态（如果为None，则使用obs）
        policy_hidden_states: 策略网络的RNN隐藏状态
        value_hidden_states: 价值网络的RNN隐藏状态
        """
        if critic_obs is None:
            critic_obs = obs
        # 确保输入是tensor并且在正确的设备上
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) if not isinstance(obs, torch.Tensor) else obs.to(self.device)
        critic_obs = torch.tensor(critic_obs, dtype=torch.float32, device=self.device) if not isinstance(critic_obs, torch.Tensor) else critic_obs.to(self.device)

        # 标准化观察状态（训练时更新统计）
        obs = self.obs_rms(obs, update=self.training)
        critic_obs = self.obs_rms(critic_obs, update=self.training)

        return self.algorithm.act(obs, critic_obs)

    def process_env_step(self, rewards, dones, infos, scale_factor=1.0):
        """处理环境步骤"""
        # 确保输入是tensor并且在正确的设备上
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device) if not isinstance(rewards, torch.Tensor) else rewards.to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device) if not isinstance(dones, torch.Tensor) else dones.to(self.device)

        # ✅ 保持原始奖励，不进行缩放，这样才能看到真实的回报值
        # scale_factor参数保留以保持接口兼容性，但默认值为1.0（不缩放）
        if scale_factor != 1.0:
            rewards = rewards * scale_factor

        self.algorithm.process_env_step(rewards, dones, infos)

    def compute_returns(self, last_critic_obs):
        """计算回报"""
        last_critic_obs = torch.tensor(last_critic_obs, dtype=torch.float32, device=self.device) if not isinstance(last_critic_obs, torch.Tensor) else last_critic_obs.to(self.device)
        # 标准化观察状态（这里不更新统计，因为是最后一步）
        last_critic_obs = self.obs_rms(last_critic_obs, update=False)
        self.algorithm.compute_returns(last_critic_obs)

    def update(self):
        """执行更新"""
        return self.algorithm.update()

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])

    def reset(self, dones=None):
        """Reset RNN hidden states"""
        if hasattr(self.policy_net, 'reset'):
            self.policy_net.reset(dones)
        if hasattr(self.value_net, 'reset'):
            self.value_net.reset(dones)

    def get_hidden_states(self):
        """Get RNN hidden states"""
        policy_hidden = None
        value_hidden = None

        if hasattr(self.policy_net, 'get_hidden_states'):
            policy_hidden = self.policy_net.get_hidden_states()
        if hasattr(self.value_net, 'get_hidden_states'):
            value_hidden = self.value_net.get_hidden_states()

        return policy_hidden, value_hidden

    @property
    def actor(self):
        """返回策略网络"""
        return self.policy_net

    @property
    def critic(self):
        """返回价值网络"""
        return self.value_net
