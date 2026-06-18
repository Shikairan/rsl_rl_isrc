# rsl_rl_isrc — MARL 环境句柄：持有官方 ShareVecEnv，供 MAPPORunner 解包使用。
#
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""多智能体环境句柄（车道 2），与 ``VecEnv`` 并列、不继承。"""

from __future__ import annotations

from typing import Any, Optional


class MarlEnv:
    """MARL 环境句柄：包装官方 ``ShareVecEnv``，对外暴露元数据与 ``unwrap_native()``。

    设计约定：
    - **不**实现 ``VecEnv`` 协议；单智能体训练继续使用 ``rsl_rl_isrc.env.VecEnv``。
    - ``MAPPORunner`` 训练时通过 ``unwrap_native()`` 将原生环境交给官方 Runner。
    - 后续 IPPO / HAPPO 等 MARL 算法可复用本句柄与 ``make_marl_env`` 工厂。
    """

    def __init__(
        self,
        native_env: Any,
        env_name: str,
        num_agents: int,
        n_rollout_threads: int,
        scenario_name: Optional[str] = None,
        **meta: Any,
    ):
        self._native = native_env
        self.env_name = env_name
        self.num_agents = int(num_agents)
        self.n_rollout_threads = int(n_rollout_threads)
        self.scenario_name = scenario_name
        self.meta = meta

    @property
    def native(self) -> Any:
        """官方 ``ShareVecEnv`` / ``SubprocVecEnv`` 实例（与 ``unwrap_native()`` 相同）。"""
        return self._native

    def unwrap_native(self) -> Any:
        """供 ``MAPPORunner`` 交给 ``onpolicy`` Runner 的原生并行环境。"""
        return self._native

    def close(self) -> None:
        if self._native is not None and hasattr(self._native, "close"):
            self._native.close()

    def __repr__(self) -> str:
        parts = [
            f"env_name={self.env_name!r}",
            f"num_agents={self.num_agents}",
            f"n_rollout_threads={self.n_rollout_threads}",
        ]
        if self.scenario_name is not None:
            parts.append(f"scenario_name={self.scenario_name!r}")
        return f"MarlEnv({', '.join(parts)})"
