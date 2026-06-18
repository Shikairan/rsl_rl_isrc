#!/usr/bin/env python3
"""DQN Runner 测试：离散化 Gymnasium Pendulum-v1 短程训练冒烟验证。

DQN 仅支持离散动作。Pendulum-v1 原生为连续力矩控制，本测试通过
``DiscretizedPendulumVecEnv`` 将 ``[-2, 2]`` 力矩均匀离散为 ``num_actions`` 档，
从而在倒立摆任务上验证 DQN 训练链路。

若环境为原生离散动作（如 CartPole-v1），可直接使用 ``DiscreteGymnasiumVecEnv``。
"""

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
from rsl_rl_isrc.runners import DQNRunner


class DiscretizedPendulumVecEnv(VecEnv):
    """将 Pendulum-v1 连续力矩离散化，供 DQN 使用的 ``VecEnv`` 适配器。"""

    def __init__(self, num_envs: int = 4, num_actions: int = 11, device: str = "cpu"):
        self._gym_env = gym.vector.SyncVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(num_envs)])
        obs_space = self._gym_env.single_observation_space

        self.num_envs = num_envs
        self.num_obs = obs_space.shape[0]
        self.num_privileged_obs = None
        self.num_actions = num_actions
        self.max_episode_length = self._gym_env.envs[0].spec.max_episode_steps or 200
        self.device = torch.device(device)

        low = float(self._gym_env.single_action_space.low[0])
        high = float(self._gym_env.single_action_space.high[0])
        self.torque_levels = np.linspace(low, high, num_actions, dtype=np.float32)

        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.extras = {}

        self.reset(torch.arange(self.num_envs))

    def _indices_to_torque(self, actions: torch.Tensor) -> np.ndarray:
        indices = actions.detach().cpu().numpy().reshape(-1).astype(np.int64)
        indices = np.clip(indices, 0, self.num_actions - 1)
        return self.torque_levels[indices].reshape(-1, 1)

    def step(self, actions: torch.Tensor):
        torque = self._indices_to_torque(actions)
        obs, rewards, terminated, truncated, infos = self._gym_env.step(torque)
        dones = terminated | truncated

        self.obs_buf = torch.tensor(obs, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        self.rew_buf = rewards_t
        self.reset_buf = dones_t
        self.episode_length_buf += 1
        self.episode_length_buf[dones_t] = 0

        return self.obs_buf.clone(), None, rewards_t, dones_t, {}

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


def build_dqn_discretized_pendulum_train_cfg(num_actions: int = 11) -> dict:
    """离散化 Pendulum 的 DQN 短程训练配置。"""
    return {
        "runner": {
            "experiment_name": "dqn_discretized_pendulum_test",
            "num_steps_per_env": 16,
            "save_interval": 100,
        },
        "algorithm": {
            "gamma": 0.99,
            "tau": 0.005,
            "learning_rate": 1e-3,
            "buffer_size": 5000,
            "batch_size": 64,
            "learning_starts": 128,
            "update_frequency": 1,
            "num_updates_per_step": 1,
            "target_network_frequency": 1,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 500,
            "double_dqn": False,
        },
        "policy": {
            "hidden_dims": [64, 64],
            "activation": "relu",
        },
        "discrete_action_bins": num_actions,
    }


class TestDQNRunner(unittest.TestCase):
    """DQN + DQNRunner + 离散化 Pendulum 短程测试。"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        np.random.seed(42)

    def test_discretized_pendulum_env(self):
        """验证离散化倒立摆适配器：离散索引 → 连续力矩。"""
        num_actions = 11
        env = DiscretizedPendulumVecEnv(num_envs=2, num_actions=num_actions, device="cpu")
        self.assertEqual(env.num_obs, 3)
        self.assertEqual(env.num_actions, num_actions)

        actions = torch.tensor([[0], [num_actions - 1]])
        next_obs, _, rewards, dones, _ = env.step(actions)
        self.assertEqual(next_obs.shape, (2, 3))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(dones.shape, (2,))

    def test_dqn_short_training_discretized_pendulum(self):
        """短程 DQN 训练：离散化 Pendulum 上不崩溃并完成 replay 更新。"""
        num_envs = 4
        num_actions = 11
        num_iterations = 5
        steps_per_iter = 16

        env = DiscretizedPendulumVecEnv(
            num_envs=num_envs, num_actions=num_actions, device=str(self.device)
        )
        train_cfg = build_dqn_discretized_pendulum_train_cfg(num_actions=num_actions)
        train_cfg["runner"]["num_steps_per_env"] = steps_per_iter

        with tempfile.TemporaryDirectory() as log_dir:
            runner = DQNRunner(
                env=env,
                train_cfg=train_cfg,
                log_dir=log_dir,
                device=self.device,
            )
            initial_state = {k: v.clone() for k, v in runner.alg.dqn_networks.state_dict().items()}
            runner.learn(num_learning_iterations=num_iterations)

            self.assertGreater(runner.alg.global_step, 0)

            obs = env.get_observations()
            with torch.no_grad():
                actions = runner.get_inference_policy()(obs)
            self.assertEqual(actions.shape, (num_envs, 1))

            final_state = runner.alg.dqn_networks.state_dict()
            changed = any(not torch.equal(initial_state[k], final_state[k]) for k in initial_state)
            self.assertTrue(changed, "Q 网络权重应在训练后发生变化")

            ckpt_path = os.path.join(log_dir, f"model_{num_iterations}.pt")
            self.assertTrue(os.path.isfile(ckpt_path))

        print(
            f"DQN 离散化 Pendulum 短程训练通过: "
            f"{num_iterations} iters × {steps_per_iter} steps × {num_envs} envs, "
            f"action_bins={num_actions}"
        )

    def test_dqn_algorithm_type(self):
        """验证 DQNRunner 构建的是 DQN 算法实例。"""
        from rsl_rl_isrc.algorithms.dqn_policy import DQN

        env = DiscretizedPendulumVecEnv(num_envs=2, num_actions=7, device="cpu")
        train_cfg = build_dqn_discretized_pendulum_train_cfg(num_actions=7)
        train_cfg["runner"]["num_steps_per_env"] = 8
        train_cfg["algorithm"]["learning_starts"] = 32

        with tempfile.TemporaryDirectory() as log_dir:
            runner = DQNRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device="cpu")
            self.assertIsInstance(runner.alg, DQN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
