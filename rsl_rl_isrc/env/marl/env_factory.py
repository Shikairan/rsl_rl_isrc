# rsl_rl_isrc — MARL 环境工厂：委托官方 onpolicy ``make_train_env``，返回 ``MarlEnv`` 句柄。
#
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""多智能体环境工厂（车道 2）。支持 MPE；SMAC / Football 预留扩展点。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rsl_rl_isrc.env.marl.marl_env import MarlEnv
from rsl_rl_isrc.integrations.onpolicy.compat import ensure_onpolicy_compat
from rsl_rl_isrc.integrations.onpolicy.config_bridge import to_namespace

SUPPORTED_MARL_ENV_NAMES = ("MPE", "StarCraft2", "Football", "Hanabi")


def _import_onpolicy():
    try:
        import onpolicy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MAPPO / MARL 环境需要可选依赖 onpolicy。请执行: "
            'pip install -e ".[marl]"'
        ) from exc


def _make_mpe_env(all_args, for_eval: bool = False):
    ensure_onpolicy_compat()
    from onpolicy.envs.mpe.MPE_env import MPEEnv
    from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv

    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    n_threads = (
        all_args.n_eval_rollout_threads if for_eval else all_args.n_rollout_threads
    )
    if n_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def _make_native_env(all_args, for_eval: bool = False):
    env_name = all_args.env_name
    if env_name == "MPE":
        return _make_mpe_env(all_args, for_eval=for_eval)
    if env_name == "StarCraft2":
        raise NotImplementedError(
            "StarCraft2 环境需安装 SMAC 并配置 SC2PATH；请使用 env_name='MPE' 或自行扩展 env_factory。"
        )
    if env_name == "Football":
        raise NotImplementedError(
            "Football 环境需安装 gfootball；请使用 env_name='MPE' 或自行扩展 env_factory。"
        )
    if env_name == "Hanabi":
        raise NotImplementedError(
            "Hanabi 环境需额外依赖；请使用 env_name='MPE' 或自行扩展 env_factory。"
        )
    raise ValueError(
        f"未知 MARL 环境 '{env_name}'，支持或预留: {SUPPORTED_MARL_ENV_NAMES}"
    )


def make_marl_env(
    train_cfg: Dict[str, Any],
    *,
    for_eval: bool = False,
    log_dir: Optional[str] = None,
    device: str = "cpu",
    seed: int = 1,
) -> MarlEnv:
    """根据 ``train_cfg`` 创建 ``MarlEnv``（内部调用官方并行环境构造）。

    ``train_cfg['env']`` 示例::

        {
            "env_name": "MPE",
            "scenario_name": "simple_spread",
            "num_agents": 2,
        }
    """
    _import_onpolicy()
    ensure_onpolicy_compat()
    all_args = to_namespace(train_cfg, log_dir=log_dir, device=device, seed=seed)
    native = _make_native_env(all_args, for_eval=for_eval)

    num_agents = int(getattr(all_args, "num_agents", 1))
    scenario_name = getattr(all_args, "scenario_name", None)

    return MarlEnv(
        native_env=native,
        env_name=all_args.env_name,
        num_agents=num_agents,
        n_rollout_threads=(
            all_args.n_eval_rollout_threads if for_eval else all_args.n_rollout_threads
        ),
        scenario_name=scenario_name,
        all_args=all_args,
    )


def make_marl_env_from_name(
    env_name: str,
    train_cfg: Dict[str, Any],
    **kwargs: Any,
) -> MarlEnv:
    """便捷入口：设置 ``train_cfg['env']['env_name']`` 后调用 ``make_marl_env``。"""
    cfg = {**train_cfg, "env": {**train_cfg.get("env", {}), "env_name": env_name}}
    return make_marl_env(cfg, **kwargs)
