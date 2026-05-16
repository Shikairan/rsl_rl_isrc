#!/usr/bin/env python3
"""PPO OnPolicyRunner 冒烟测试（非分布式单进程）。"""

import torch
import numpy as np
import unittest
import tempfile
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.runners import OnPolicyRunner


class DummyVecEnv(VecEnv):
    """CartPole-like 连续观测、连续动作虚拟环境。"""

    def __init__(self, num_envs=4, max_episode_length=50):
        self.num_envs = num_envs
        self.num_obs = 4
        self.num_privileged_obs = None
        self.num_actions = 2
        self.max_episode_length = max_episode_length
        self.obs_buf = torch.zeros(num_envs, self.num_obs)
        self.rew_buf = torch.zeros(num_envs)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32)
        self.device = torch.device("cpu")
        self.reset(torch.arange(num_envs))

    def step(self, actions):
        noise = torch.randn_like(self.obs_buf) * 0.1
        self.obs_buf = self.obs_buf + noise
        self.obs_buf = torch.clamp(self.obs_buf, -5.0, 5.0)
        rewards = torch.ones(self.num_envs)
        self.episode_length_buf += 1
        terminated = self.episode_length_buf >= self.max_episode_length
        terminated_ids = torch.where(terminated)[0]
        if len(terminated_ids) > 0:
            self.reset(terminated_ids)
        return self.obs_buf.clone(), None, rewards, terminated, {}

    def reset(self, env_ids):
        for env_id in env_ids:
            self.obs_buf[env_id] = torch.randn(self.num_obs) * 0.05
            self.episode_length_buf[env_id] = 0
        return self.obs_buf.clone()

    def get_observations(self):
        return self.obs_buf.clone()

    def get_privileged_observations(self):
        return None


class TestOnPolicyRunner(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        np.random.seed(0)

    def test_single_process_no_crash(self):
        """验证非分布式单进程 PPO Runner 不崩溃。"""
        env = DummyVecEnv(num_envs=2, max_episode_length=20)
        train_cfg = {
            "runner": {
                "experiment_name": "ppo_smoke_test",
                "num_steps_per_env": 20,
                "save_interval": 100,
            },
            "algorithm": {
                "algorithm_class_name": "PPO",
                "num_learning_epochs": 2,
                "num_mini_batches": 2,
                "clip_param": 0.2,
                "gamma": 0.99,
                "lam": 0.95,
                "value_loss_coef": 1.0,
                "entropy_coef": 0.01,
                "learning_rate": 1e-3,
                "max_grad_norm": 1.0,
                "use_clipped_value_loss": True,
                "desired_kl": None,
                "schedule": "fixed",
            },
            "policy": {
                "policy_class_name": "ActorCritic",
                "activation": "elu",
                "actor_hidden_dims": [32, 32],
                "critic_hidden_dims": [32, 32],
                "init_noise_std": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as log_dir:
            runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device=self.device)
            runner.learn(num_learning_iterations=3)

        print("PPO OnPolicyRunner 单进程冒烟测试通过!")

    def test_mini_batch_generator(self):
        """验证 mini_batch_generator 使用实际维度（而非固定 num_envs）。"""
        from rsl_rl_isrc.storage import RolloutStorage
        T, E, obs_dim, act_dim = 10, 4, 4, 2
        storage = RolloutStorage(E, T, (obs_dim,), (None,), (act_dim,), device="cpu")
        # 写满 storage
        from rsl_rl_isrc.storage.rollout_storage import RolloutStorage as RS
        t = RS.Transition()
        for _ in range(T):
            t.observations = torch.randn(E, obs_dim)
            t.critic_observations = torch.randn(E, obs_dim)
            t.actions = torch.randn(E, act_dim)
            t.rewards = torch.ones(E)
            t.dones = torch.zeros(E, dtype=torch.bool)
            t.values = torch.zeros(E, 1)
            t.actions_log_prob = torch.zeros(E)
            t.action_mean = torch.zeros(E, act_dim)
            t.action_sigma = torch.ones(E, act_dim)
            t.hidden_states = None
            storage.add_transitions(t)
        # 计算 returns
        storage.compute_returns(torch.zeros(E, 1), gamma=0.99, lam=0.95)
        # 生成 mini batch
        batches = list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=1))
        self.assertEqual(len(batches), 2)
        # 每个 batch 应该包含 T*E // 2 = 20 个样本
        obs_batch, *_ = batches[0]
        self.assertEqual(obs_batch.shape[0], T * E // 2)
        print("mini_batch_generator 测试通过!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
