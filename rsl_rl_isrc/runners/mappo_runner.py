# rsl_rl_isrc — MAPPO 外观 Runner：委托官方 onpolicy，与 ``OnPolicyRunner`` 并列（车道 2）。
#
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""MAPPO 训练运行器（多智能体车道）：底层不修改、不混入 ``rsl_rl_isrc`` 单智能体算法。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import torch

from rsl_rl_isrc.env.marl import MarlEnv, make_marl_env, make_marl_env_from_name
from rsl_rl_isrc.integrations.onpolicy.config_bridge import resolve_run_dir, to_namespace


def _import_onpolicy():
    try:
        import onpolicy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MAPPORunner 需要可选依赖 onpolicy。请执行: pip install -e \".[marl]\""
        ) from exc


def _resolve_official_runner_class(env_name: str, share_policy: bool) -> Type[Any]:
    """按环境与是否共享策略选择官方 Runner 类。"""
    if env_name == "MPE":
        if share_policy:
            from onpolicy.runner.shared.mpe_runner import MPERunner

            return MPERunner
        from onpolicy.runner.separated.mpe_runner import MPERunner

        return MPERunner
    raise NotImplementedError(
        f"环境 '{env_name}' 的官方 Runner 尚未接入外观层；"
        f"当前可直接使用 MPE，或在 mappo_runner 中扩展 dispatch。"
    )


class MAPPORunner:
    """MAPPO 外观 Runner（车道 2）。

    - **不**接收 ``VecEnv``；环境由 ``MarlEnv`` / ``make_marl_env`` 提供。
    - 训练循环委托官方 ``onpolicy.runner.*``，算法与 buffer 均在官方包内。
    - API 形态与 ``OnPolicyRunner`` 对齐：``train_cfg`` + ``learn()`` + ``save``/``load``。

    示例::

        marl_env = make_marl_env(train_cfg, log_dir=log_dir, device=device)
        runner = MAPPORunner(train_cfg, marl_env=marl_env, log_dir=log_dir, device=device)
        runner.learn(num_learning_iterations=5)
    """

    def __init__(
        self,
        train_cfg: Dict[str, Any],
        marl_env: Optional[MarlEnv] = None,
        env_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        _import_onpolicy()
        self.train_cfg = train_cfg
        self.cfg = train_cfg.get("runner", {})
        self.alg_cfg = train_cfg.get("algorithm", {})
        self.device_str = device
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.log_dir = log_dir
        self.current_learning_iteration = 0
        self.tot_time = 0.0
        self._owns_marl_env = marl_env is None

        if marl_env is None:
            if env_name is not None:
                marl_env = make_marl_env_from_name(
                    env_name, train_cfg, log_dir=log_dir, device=device
                )
            else:
                marl_env = make_marl_env(train_cfg, log_dir=log_dir, device=device)

        self.marl_env: MarlEnv = marl_env
        self.all_args = to_namespace(train_cfg, log_dir=log_dir, device=device)
        self.all_args.cuda = self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.set_num_threads(int(self.cfg.get("n_training_threads", 1)))

        self.run_dir = resolve_run_dir(self.all_args)
        self._official_runner = self._build_official_runner()

    def _build_official_runner(self):
        runner_cls = _resolve_official_runner_class(
            self.marl_env.env_name,
            share_policy=bool(self.all_args.share_policy),
        )
        config = {
            "all_args": self.all_args,
            "envs": self.marl_env.unwrap_native(),
            "eval_envs": None,
            "num_agents": self.marl_env.num_agents,
            "device": self.device,
            "run_dir": self.run_dir,
        }
        return runner_cls(config)

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
        pre_iter_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """执行 MAPPO 训练。

        ``num_learning_iterations`` 映射为官方 ``episodes``::

            num_env_steps = num_learning_iterations × episode_length × n_rollout_threads

        ``init_at_random_ep_len`` / ``pre_iter_callback`` 为与 ``OnPolicyRunner`` 签名兼容而保留，
        官方 Runner 暂不使用（调用安全忽略）。
        """
        del init_at_random_ep_len
        if pre_iter_callback is not None:
            pre_iter_callback(self.current_learning_iteration)

        episode_length = self.all_args.episode_length
        n_rollout = self.all_args.n_rollout_threads
        total_steps = int(num_learning_iterations) * episode_length * n_rollout
        self.all_args.num_env_steps = total_steps
        self._official_runner.num_env_steps = total_steps
        self._official_runner.all_args.num_env_steps = total_steps

        import time

        start = time.time()
        self._official_runner.run()
        self.tot_time = time.time() - start
        self.current_learning_iteration += int(num_learning_iterations)

        if hasattr(self._official_runner, "writter") and self._official_runner.writter is not None:
            try:
                self._official_runner.writter.flush()
            except Exception:
                pass

        return {
            "num_learning_iterations": self.current_learning_iteration,
            "elapsed_time": self.tot_time,
            "num_env_steps": self.all_args.num_env_steps,
        }

    def save(self, path: Optional[str] = None, infos=None):
        """保存官方 actor/critic 权重（目录或前缀由 ``path`` 指定）。"""
        del infos
        if path is None:
            path = str(self.run_dir / "models")
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._official_runner.save_dir = str(save_dir)
        self._official_runner.save(episode=self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """从官方格式 checkpoint 目录加载（需含 ``actor.pt`` / ``critic.pt``）。"""
        del load_optimizer
        self.all_args.model_dir = str(path)
        self._official_runner.model_dir = str(path)
        self._official_runner.restore(str(path))
        return {}

    def close(self):
        """关闭 MARL 环境与 TensorBoard writer。"""
        if hasattr(self._official_runner, "writter") and self._official_runner.writter is not None:
            try:
                self._official_runner.writter.close()
            except Exception:
                pass
        if self._owns_marl_env:
            self.marl_env.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
