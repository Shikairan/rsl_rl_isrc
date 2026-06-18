#!/usr/bin/env python3
"""MAPPORunner 冒烟测试：官方 MPE simple_spread 短程训练。"""

import os
import sys
import tempfile
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

pytest = __import__("pytest")
onpolicy = pytest.importorskip("onpolicy")

from rsl_rl_isrc.env.marl import MarlEnv, make_marl_env
from rsl_rl_isrc.runners import MAPPORunner


def build_mappo_mpe_train_cfg(n_rollout_threads: int = 1) -> dict:
    return {
        "runner": {
            "experiment_name": "mappo_mpe_test",
            "n_rollout_threads": n_rollout_threads,
            "num_steps_per_env": 10,
            "save_interval": 100,
            "log_interval": 1,
            "use_wandb": False,
            "share_policy": True,
        },
        "algorithm": {
            "algorithm_class_name": "MAPPO",
            "clip_param": 0.2,
            "ppo_epoch": 1,
            "num_mini_batch": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "learning_rate": 5e-4,
            "use_centralized_V": True,
        },
        "policy": {
            "hidden_size": 32,
            "layer_N": 1,
        },
        "env": {
            "env_name": "MPE",
            "scenario_name": "simple_spread",
            "num_agents": 2,
            "num_landmarks": 2,
        },
    }


class TestMAPPORunner(unittest.TestCase):
    """MAPPO 外观 Runner + MarlEnv 工厂测试。"""

    def test_marl_env_factory(self):
        train_cfg = build_mappo_mpe_train_cfg()
        marl_env = make_marl_env(train_cfg, device="cpu", seed=42)
        self.assertIsInstance(marl_env, MarlEnv)
        self.assertEqual(marl_env.env_name, "MPE")
        self.assertEqual(marl_env.num_agents, 2)
        self.assertIsNotNone(marl_env.unwrap_native())
        marl_env.close()

    def test_mappo_not_on_policy_runner(self):
        from rsl_rl_isrc.runners.on_policy_runner import OnPolicyRunner

        self.assertTrue(callable(MAPPORunner))
        self.assertFalse(hasattr(OnPolicyRunner, "marl_env"))

    def test_mappo_short_training_mpe(self):
        train_cfg = build_mappo_mpe_train_cfg(n_rollout_threads=1)
        num_iters = 2

        with tempfile.TemporaryDirectory() as log_dir:
            runner = MAPPORunner(train_cfg, env_name="MPE", log_dir=log_dir, device="cpu")
            results = runner.learn(num_learning_iterations=num_iters)
            runner.save()
            actor_path = os.path.join(str(runner.run_dir), "models", "actor.pt")
            self.assertTrue(os.path.isfile(actor_path), f"缺少 checkpoint: {actor_path}")
            runner.close()

        self.assertEqual(results["num_learning_iterations"], num_iters)
        self.assertGreater(results["elapsed_time"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
