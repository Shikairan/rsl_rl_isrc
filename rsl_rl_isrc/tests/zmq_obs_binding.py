"""已废弃：请使用 ``G1OnPolicyTestRunner`` + ``ObsInstrServer.bind_publisher``。"""

from __future__ import annotations

import warnings

import torch

from rsl_rl_isrc.sockets.http_post import StepObsPublisher


class _InstrStub:
    """满足 ``StepObsPublisher._bind_to_server`` 对 ``server._instr`` 的最小桩。"""

    def __init__(self, num_envs: int) -> None:
        from rsl_rl_isrc.sockets.obs_server import default_obs_env_hi

        hi = default_obs_env_hi(num_envs)
        self._instr = torch.tensor([0, 0, 0, hi], dtype=torch.int64)


def bind_publisher_to_pull(
    publisher: StepObsPublisher,
    host: str = "localhost",
    port: int = 15555,
) -> None:
    """已废弃：与 ObsInstrServer 端口冲突，勿再使用。"""
    warnings.warn(
        "bind_publisher_to_pull 已废弃，请使用 G1OnPolicyTestRunner + ObsInstrServer",
        DeprecationWarning,
        stacklevel=2,
    )
    stub = _InstrStub(publisher._num_envs)
    publisher._bind_to_server(stub, host=host, port=port)
