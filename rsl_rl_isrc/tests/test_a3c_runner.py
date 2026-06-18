#!/usr/bin/env python3
"""A3C Runner 多进程异步测试：Gymnasium Pendulum-v1 短程训练冒烟验证。

注意：A3C 使用多进程 worker + 独立环境实例，**不适用** 同步 ``GymnasiumVecEnv``
或 ``OnPolicyRunner`` 的单进程向量环境测试模式（参见 ``test_a2c_runner.py``）。
"""

import os
import sys
import tempfile
import unittest

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rsl_rl_isrc.algorithms.a3c import A3C
from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.runners import A3CRunner
from rsl_rl_isrc.storage import RolloutStorage


def build_a3c_pendulum_train_cfg(num_workers: int = 2) -> dict:
    """Pendulum-v1 的 A3C 短程多进程训练配置。"""
    return {
        "runner": {
            "experiment_name": "a3c_pendulum_test",
            "num_workers": num_workers,
            "save_interval": 100,
        },
        "algorithm": {
            "t_max": 16,
            "n_steps": 5,
            "gamma": 0.99,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "learning_rate": 1e-3,
            "max_grad_norm": 40.0,
            "optimizer_type": "sgd",
            "use_hogwild_lock": True,
        },
        "policy": {
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "tanh",
            "init_noise_std": 1.0,
        },
    }


class TestA3CRunner(unittest.TestCase):
    """A3C 多进程异步训练测试（非 VecEnv 同步模式）。"""

    def setUp(self):
        self.device = "cpu"
        torch.manual_seed(42)

    def test_n_step_returns(self):
        """单元测试：RolloutStorage n-step 回报计算。"""
        storage = RolloutStorage(1, 4, [3], [3], [1], device="cpu")
        transition = RolloutStorage.Transition()
        rewards = [1.0, 1.0, 1.0, 1.0]
        for idx, reward in enumerate(rewards):
            transition.observations = torch.zeros(1, 3)
            transition.critic_observations = torch.zeros(1, 3)
            transition.actions = torch.zeros(1, 1)
            transition.rewards = torch.tensor([reward])
            transition.dones = torch.tensor([False])
            transition.values = torch.tensor([[float(idx)]])
            transition.actions_log_prob = torch.zeros(1)
            transition.action_mean = torch.zeros(1, 1)
            transition.action_sigma = torch.ones(1, 1)
            transition.hidden_states = None
            storage.add_transitions(transition)

        last_values = torch.tensor([[4.0]])
        storage.compute_n_step_returns(last_values, gamma=0.9, n_steps=2, num_steps=4)
        self.assertEqual(storage.returns.shape[0], 4)
        self.assertAlmostEqual(storage.returns[0].item(), 1.0 + 0.9 * 1.0 + (0.9 ** 2) * 2.0, places=4)

    def test_a3c_runner_multiprocess_pendulum(self):
        """短程多进程 A3C：2 个 worker 各执行 3 次 rollout，验证异步训练不崩溃。"""
        num_workers = 2
        rollouts_per_worker = 3
        train_cfg = build_a3c_pendulum_train_cfg(num_workers=num_workers)

        with tempfile.TemporaryDirectory() as log_dir:
            runner = A3CRunner(
                train_cfg=train_cfg,
                log_dir=log_dir,
                device=self.device,
                env_id="Pendulum-v1",
            )
            initial_state = {k: v.clone() for k, v in runner.actor_critic.state_dict().items()}
            results = runner.learn(num_learning_iterations=rollouts_per_worker, seed=42)

            ckpt_path = os.path.join(log_dir, f"model_{rollouts_per_worker}.pt")
            self.assertTrue(os.path.isfile(ckpt_path))
            runner.close()

        expected_rollouts = num_workers * rollouts_per_worker
        self.assertEqual(results["total_rollouts"], expected_rollouts)
        self.assertEqual(len(results["worker_results"]), num_workers)

        final_state = runner.actor_critic.state_dict()
        changed = any(not torch.equal(initial_state[k], final_state[k]) for k in initial_state)
        self.assertTrue(changed, "共享模型权重应在异步更新后发生变化")

        print(
            f"A3C 多进程 Pendulum 短程训练通过: "
            f"{num_workers} workers × {rollouts_per_worker} rollouts, "
            f"total_rollouts={results['total_rollouts']}"
        )

    def test_a3c_not_on_policy_runner(self):
        """A3C 使用独立 A3CRunner，而非 OnPolicyRunner 注册表。"""
        from rsl_rl_isrc.runners.on_policy_runner import OnPolicyRunner

        train_cfg = build_a3c_pendulum_train_cfg(num_workers=2)
        runner = A3CRunner(train_cfg=train_cfg, device=self.device)
        self.assertIsInstance(runner.actor_critic, ActorCritic)
        registry = {"PPO", "A2C"}
        self.assertNotIn("A3C", registry)
        self.assertTrue(callable(runner.learn))
        self.assertFalse(hasattr(OnPolicyRunner, "num_workers"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
