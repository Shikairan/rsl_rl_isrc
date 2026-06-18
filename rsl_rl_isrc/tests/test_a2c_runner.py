#!/usr/bin/env python3
"""A2C OnPolicyRunner 测试：Gymnasium Pendulum-v1 短程训练冒烟验证。"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import rsl_rl_isrc.isrcgym as gym

from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.runners import OnPolicyRunner


class GymnasiumVecEnv(VecEnv):
    """将 Gymnasium 向量环境适配为 ``VecEnv`` 接口（用于 OnPolicyRunner 测试）。"""

    def __init__(self, env_id: str, num_envs: int = 4, device: str = "cpu"):
        self._gym_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)])
        obs_space = self._gym_env.single_observation_space
        action_space = self._gym_env.single_action_space

        self.num_envs = num_envs
        self.num_obs = obs_space.shape[0]
        self.num_privileged_obs = None
        self.num_actions = action_space.shape[0]
        self.max_episode_length = self._gym_env.envs[0].spec.max_episode_steps or 200
        self.device = torch.device(device)

        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.extras = {}

        self.reset(torch.arange(self.num_envs))

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        obs, rewards, terminated, truncated, infos = self._gym_env.step(actions_np)
        dones = terminated | truncated

        self.obs_buf = torch.tensor(obs, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        self.rew_buf = rewards_t
        self.reset_buf = dones_t
        self.episode_length_buf += 1
        self.episode_length_buf[dones_t] = 0

        episode_infos = {}
        if "episode" in infos:
            for key, value in infos["episode"].items():
                if value is not None:
                    episode_infos[key] = torch.tensor(value, dtype=torch.float32, device=self.device)

        return self.obs_buf.clone(), None, rewards_t, dones_t, {"episode": episode_infos} if episode_infos else {}

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        if len(env_ids) == self.num_envs:
            obs, _ = self._gym_env.reset()
            self.obs_buf = torch.tensor(obs, dtype=torch.float32, device=self.device)
            self.episode_length_buf[:] = 0
        return self.obs_buf.clone()

    def get_observations(self):
        return self.obs_buf.clone()

    def get_privileged_observations(self):
        return None


def build_a2c_pendulum_train_cfg(num_steps_per_env: int = 64) -> dict:
    """Pendulum-v1 连续控制任务的 A2C 短程训练配置。"""
    return {
        "runner": {
            "experiment_name": "a2c_pendulum_test",
            "num_steps_per_env": num_steps_per_env,
            "save_interval": 100,
        },
        "algorithm": {
            "algorithm_class_name": "A2C",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
        },
        "policy": {
            "policy_class_name": "ActorCritic",
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "tanh",
            "init_noise_std": 1.0,
        },
    }


class TestA2CRunner(unittest.TestCase):
    """A2C + OnPolicyRunner + Gymnasium Pendulum 短程测试。"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        np.random.seed(42)

    def test_gymnasium_pendulum_env(self):
        """验证 Gymnasium Pendulum 向量环境适配器基本 step/reset。"""
        env = GymnasiumVecEnv("Pendulum-v1", num_envs=2, device="cpu")
        self.assertEqual(env.num_obs, 3)
        self.assertEqual(env.num_actions, 1)

        obs = env.get_observations()
        self.assertEqual(obs.shape, (2, 3))

        actions = torch.zeros(2, 1)
        next_obs, priv_obs, rewards, dones, infos = env.step(actions)
        self.assertEqual(next_obs.shape, (2, 3))
        self.assertIsNone(priv_obs)
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(dones.shape, (2,))

    def test_a2c_short_training_pendulum(self):
        """短程 A2C 训练：验证 OnPolicyRunner 在 Pendulum 上不崩溃并完成更新。"""
        num_envs = 4
        num_iterations = 5
        num_steps_per_env = 32

        env = GymnasiumVecEnv("Pendulum-v1", num_envs=num_envs, device=str(self.device))
        train_cfg = build_a2c_pendulum_train_cfg(num_steps_per_env=num_steps_per_env)

        with tempfile.TemporaryDirectory() as log_dir:
            runner = OnPolicyRunner(
                env=env,
                train_cfg=train_cfg,
                log_dir=log_dir,
                device=self.device,
            )
            runner.learn(num_learning_iterations=num_iterations)

            obs = env.get_observations()
            with torch.no_grad():
                actions = runner.get_inference_policy()(obs)
            self.assertEqual(actions.shape, (num_envs, env.num_actions))

            ckpt_path = os.path.join(log_dir, "model_0.pt")
            self.assertTrue(os.path.isfile(ckpt_path))
            runner.save(os.path.join(log_dir, "test_reload.pt"))
            runner.load(os.path.join(log_dir, "test_reload.pt"))

        print(
            f"A2C Pendulum 短程训练通过: "
            f"{num_iterations} iters × {num_steps_per_env} steps × {num_envs} envs"
        )

    def test_a2c_algorithm_registered(self):
        """验证 A2C 已注册到 OnPolicyRunner 算法白名单。"""
        from rsl_rl_isrc.algorithms import A2C

        env = GymnasiumVecEnv("Pendulum-v1", num_envs=2, device="cpu")
        train_cfg = build_a2c_pendulum_train_cfg(num_steps_per_env=8)

        with tempfile.TemporaryDirectory() as log_dir:
            runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device="cpu")
            self.assertIsInstance(runner.alg, A2C)


if __name__ == "__main__":
    unittest.main(verbosity=2)
