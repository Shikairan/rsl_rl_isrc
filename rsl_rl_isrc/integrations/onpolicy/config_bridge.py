# rsl_rl_isrc — 将 ``train_cfg`` 转为官方 onpolicy ``argparse.Namespace``（仅 MARL 车道使用）。
#
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""``train_cfg`` 与官方 ``onpolicy`` 超参命名空间的桥接。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from onpolicy.config import get_config


def _normalize_algorithm_name(name: str) -> str:
    key = (name or "MAPPO").strip().lower()
    mapping = {
        "mappo": "mappo",
        "rmappo": "rmappo",
        "ippo": "ippo",
        "happo": "happo",
        "hatrpo": "hatrpo",
    }
    if key not in mapping:
        raise ValueError(f"不支持的 MARL 算法 '{name}'，支持: {sorted(set(mapping.values()))}")
    return mapping[key]


def _apply_algorithm_defaults(all_args: argparse.Namespace) -> None:
    """与官方 ``train_mpe.py`` 等脚本保持一致的算法相关开关。"""
    algo = all_args.algorithm_name
    if algo == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif algo == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif algo == "ippo":
        all_args.use_centralized_V = False


def to_namespace(
    train_cfg: Dict[str, Any],
    *,
    log_dir: Optional[str] = None,
    device: str = "cpu",
    seed: int = 1,
    num_learning_iterations: Optional[int] = None,
) -> argparse.Namespace:
    """把 ISRC 风格 ``train_cfg`` 转为官方 ``all_args``。

    字段映射（主要）:
        ``runner.experiment_name`` → ``experiment_name``
        ``runner.n_rollout_threads`` / ``runner.num_envs`` → ``n_rollout_threads``
        ``runner.num_steps_per_env`` → ``episode_length``
        ``algorithm.*`` → 同名或官方名（``clip_param``、``gamma`` 等）
        ``policy.hidden_size`` → ``hidden_size``
        ``env.*`` → MPE/SMAC 等环境专用字段
    """
    runner_cfg = train_cfg.get("runner", {})
    alg_cfg = train_cfg.get("algorithm", {})
    policy_cfg = train_cfg.get("policy", {})
    env_cfg = train_cfg.get("env", {})

    parser = get_config()
    all_args = parser.parse_args([])

    env_name = env_cfg.get("env_name", "MPE")
    algo_name = _normalize_algorithm_name(alg_cfg.get("algorithm_class_name", "MAPPO"))

    all_args.algorithm_name = algo_name
    all_args.env_name = env_name
    all_args.experiment_name = runner_cfg.get("experiment_name", "mappo_experiment")
    all_args.seed = int(runner_cfg.get("seed", seed))

    use_cuda = device.startswith("cuda")
    all_args.cuda = use_cuda
    all_args.n_training_threads = int(runner_cfg.get("n_training_threads", 1))
    all_args.n_rollout_threads = int(
        runner_cfg.get("n_rollout_threads", runner_cfg.get("num_envs", 2))
    )
    all_args.n_eval_rollout_threads = int(runner_cfg.get("n_eval_rollout_threads", 1))
    all_args.episode_length = int(runner_cfg.get("num_steps_per_env", 25))
    all_args.save_interval = int(runner_cfg.get("save_interval", 100))
    all_args.log_interval = int(runner_cfg.get("log_interval", 1))
    all_args.use_wandb = bool(runner_cfg.get("use_wandb", False))
    all_args.use_eval = bool(runner_cfg.get("use_eval", False))
    all_args.use_linear_lr_decay = bool(runner_cfg.get("use_linear_lr_decay", False))
    all_args.share_policy = bool(runner_cfg.get("share_policy", True))

    # PPO / MAPPO 超参
    all_args.clip_param = float(alg_cfg.get("clip_param", 0.2))
    all_args.ppo_epoch = int(alg_cfg.get("ppo_epoch", alg_cfg.get("num_learning_epochs", 2)))
    all_args.num_mini_batch = int(alg_cfg.get("num_mini_batch", alg_cfg.get("num_mini_batches", 1)))
    all_args.gamma = float(alg_cfg.get("gamma", 0.99))
    all_args.gae_lambda = float(alg_cfg.get("gae_lambda", alg_cfg.get("lam", 0.95)))
    all_args.entropy_coef = float(alg_cfg.get("entropy_coef", 0.01))
    all_args.value_loss_coef = float(alg_cfg.get("value_loss_coef", 1.0))
    all_args.max_grad_norm = float(alg_cfg.get("max_grad_norm", 10.0))
    all_args.lr = float(alg_cfg.get("learning_rate", alg_cfg.get("lr", 5e-4)))
    all_args.critic_lr = float(alg_cfg.get("critic_lr", all_args.lr))
    all_args.use_centralized_V = bool(alg_cfg.get("use_centralized_V", True))

    # 网络
    all_args.hidden_size = int(policy_cfg.get("hidden_size", 64))
    all_args.layer_N = int(policy_cfg.get("layer_N", 1))
    all_args.use_ReLU = policy_cfg.get("activation", "relu") != "tanh"
    all_args.recurrent_N = int(policy_cfg.get("recurrent_N", 1))

    # MPE 等环境字段
    if env_name == "MPE":
        all_args.scenario_name = env_cfg.get("scenario_name", "simple_spread")
        all_args.num_agents = int(env_cfg.get("num_agents", 2))
        all_args.num_landmarks = int(env_cfg.get("num_landmarks", 3))

    _apply_algorithm_defaults(all_args)

    n_rollout = all_args.n_rollout_threads
    episode_length = all_args.episode_length
    if num_learning_iterations is not None:
        all_args.num_env_steps = int(num_learning_iterations) * episode_length * n_rollout
    else:
        all_args.num_env_steps = int(
            runner_cfg.get("num_env_steps", episode_length * n_rollout * 10)
        )

    if log_dir is not None:
        all_args._isrc_log_dir = str(Path(log_dir))
    else:
        all_args._isrc_log_dir = None

    return all_args


def resolve_run_dir(all_args: argparse.Namespace, default_root: Optional[Path] = None) -> Path:
    """解析 TensorBoard / checkpoint 目录（优先 ``train_cfg`` 传入的 ``log_dir``）。"""
    if getattr(all_args, "_isrc_log_dir", None):
        run_dir = Path(all_args._isrc_log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if default_root is not None:
        default_root.mkdir(parents=True, exist_ok=True)
        return default_root

    return Path.cwd() / "mappo_logs"
