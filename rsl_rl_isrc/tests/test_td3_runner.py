#!/usr/bin/env python3
"""TD3 Runner 测试：Gymnasium Pendulum-v1 短程训练冒烟验证。"""

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
from rsl_rl_isrc.runners import TD3Runner


class GymnasiumVecEnv(VecEnv):
    """将 Gymnasium 向量环境适配为 ``VecEnv`` 接口（用于 TD3Runner 测试）。"""

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


def build_td3_pendulum_train_cfg() -> dict:
    """Pendulum-v1 连续控制任务的 TD3 短程训练配置。"""
    return {
        "runner": {
            "experiment_name": "td3_pendulum_test",
            "num_steps_per_env": 32,
            "save_interval": 100,
        },
        "algorithm": {
            "gamma": 0.99,
            "tau": 0.005,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "buffer_size": 10000,
            "batch_size": 64,
            "learning_starts": 128,
            "update_frequency": 1,
            "num_updates_per_step": 1,
            "policy_frequency": 2,
            "target_noise_std": 0.2,
            "target_noise_clip": 0.5,
            "noise_std": 0.1,
            "action_bounds": [-2.0, 2.0],
        },
        "policy": {
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "relu",
        },
    }


class TestTD3Runner(unittest.TestCase):
    """TD3 + TD3Runner + Gymnasium Pendulum 短程测试。"""

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
        next_obs, priv_obs, rewards, dones, _ = env.step(actions)
        self.assertEqual(next_obs.shape, (2, 3))
        self.assertIsNone(priv_obs)
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(dones.shape, (2,))

    def test_td3_short_training_pendulum(self):
        """短程 TD3 训练：验证 TD3Runner 在 Pendulum 上不崩溃并完成更新。"""
        num_envs = 4
        num_iterations = 5
        steps_per_iter = 32

        env = GymnasiumVecEnv("Pendulum-v1", num_envs=num_envs, device=str(self.device))
        train_cfg = build_td3_pendulum_train_cfg()
        train_cfg["runner"]["num_steps_per_env"] = steps_per_iter

        with tempfile.TemporaryDirectory() as log_dir:
            runner = TD3Runner(
                env=env,
                train_cfg=train_cfg,
                log_dir=log_dir,
                device=self.device,
            )
            runner.learn(num_learning_iterations=num_iterations)

            self.assertGreater(runner.alg.global_step, 0)

            obs = env.get_observations()
            with torch.no_grad():
                actions = runner.get_inference_policy()(obs)
            self.assertEqual(actions.shape, (num_envs, env.num_actions))

            runner.save(os.path.join(log_dir, "test_reload.pt"))
            runner.load(os.path.join(log_dir, "test_reload.pt"))

        print(
            f"TD3 Pendulum 短程训练通过: "
            f"{num_iterations} iters × {steps_per_iter} steps × {num_envs} envs"
        )

    def test_td3_algorithm_type(self):
        """验证 TD3Runner 构建的是 TD3 算法实例。"""
        from rsl_rl_isrc.algorithms.td3_policy import TD3

        env = GymnasiumVecEnv("Pendulum-v1", num_envs=2, device="cpu")
        train_cfg = build_td3_pendulum_train_cfg()
        train_cfg["runner"]["num_steps_per_env"] = 8
        train_cfg["algorithm"]["learning_starts"] = 16

        with tempfile.TemporaryDirectory() as log_dir:
            runner = TD3Runner(env=env, train_cfg=train_cfg, log_dir=log_dir, device="cpu")
            self.assertIsInstance(runner.alg, TD3)
            self.assertEqual(runner.alg.policy_frequency, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
