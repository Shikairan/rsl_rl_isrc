# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""HTTP POST：``send_post_request``（仿真张量）与 ``StepObsPublisher``（按指令切片上报观测）。

``StepObsPublisher`` 当前**仅负责发送**：

- 绑定到 ``ObsInstrServer`` 后，通过 ZMQ PUSH 将 obs 数据推送到服务端（fire-and-forget）。
- ``self._instr`` 与 ``ObsInstrServer._instr`` 是**同一张量对象**，
  ``ObsInstrServer.sync_instr()`` 广播后 publisher 自动获得最新指令。
- **不做任何数据接收，不调用 dist.broadcast**（该职责由 ObsInstrServer 承担）。

若未绑定服务端，则 ``push()`` 为空操作。
"""

from __future__ import annotations

import json
import os
from typing import Optional, TYPE_CHECKING

import requests
import torch
import torch.distributed as dist
import zmq

if TYPE_CHECKING:
    from rsl_rl_isrc.sockets.obs_server import ObsInstrServer

_DEFAULT_POST_URL = "http://172.17.0.16:18888/post"

# 超时时间通过环境变量配置（秒），避免 pool worker 或训练循环永久阻塞
_POST_TIMEOUT = float(os.environ.get("RSL_RL_ISRC_POST_TIMEOUT", "10"))
_OBS_TIMEOUT  = float(os.environ.get("RSL_RL_ISRC_OBS_TIMEOUT",  "2"))


def send_post_request(data, rank, task):
    """向 ``RSL_RL_ISRC_POST_URL``（或默认 URL）POST 仿真/自定义张量数据。

    请求体含 ``type/data``、``rank``、``task``、``tensor``。成功返回服务端 JSON；异常时返回 ``{"error": ...}``。
    超时由 ``RSL_RL_ISRC_POST_TIMEOUT``（默认 10s）控制，防止 pool worker 永久挂起。
    """
    header = {"Content-Type": "application/json"}
    url    = os.environ.get("RSL_RL_ISRC_POST_URL", _DEFAULT_POST_URL)
    body   = {"type": "data", "rank": rank, "task": task, "tensor": data}
    try:
        response = requests.post(url, json=body, headers=header, timeout=_POST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


class StepObsPublisher:
    """``env.step`` 后按 ``_instr`` 指令切片 obs 并通过 ZMQ PUSH 上报到 ``ObsInstrServer``。

    **设计约定**

    - ``push()`` 只**发送**，不做任何数据接收，不修改 ``_instr``，不调用 ``dist.broadcast``。
    - ``_instr`` 张量由 ``ObsInstrServer`` 统一管理；通过 ``bind_to_server()`` 绑定后，
      两者共享同一张量对象，``ObsInstrServer.sync_instr()`` 广播后 publisher 自动获得最新指令。
    - 未绑定服务端时，``push()`` 为空操作（disabled）。

    **指令格式**：``[sender_rank, aux, env_start, env_end)``（int64，CPU 张量）
    """

    def __init__(self, rank: int, task: str, num_envs: int):
        """
        参数
        ----
        rank : int
            当前进程 rank。
        task : str
            任务名称（透传给消息体）。
        num_envs : int
            并行环境数，用于默认指令上界。
        """
        self._init_rank = int(rank)
        self._task      = task
        self._num_envs  = max(1, int(num_envs))

        # 指令张量：[sender_rank, aux, env_start, env_end]
        # bind_to_server 后会被替换为与 ObsInstrServer 共享的对象
        self._instr: torch.Tensor = torch.tensor(
            [0, 0, 0, self._num_envs], dtype=torch.int64
        )

        # ZMQ PUSH socket（bind_to_server 后初始化）
        self._zmq_ctx:   Optional[zmq.Context] = None
        self._push_sock: Optional[zmq.Socket]  = None

        # 关联的 ObsInstrServer 实例（用于获取共享 _instr）
        self._server: Optional[ObsInstrServer] = None

    # ──────────────────────────────────────────────────────────────────────────
    # 绑定到服务端（由 ObsInstrServer.bind_publisher() 间接调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _bind_to_server(
        self,
        server: ObsInstrServer,
        host: str = "localhost",
        port: int = 15555,
    ) -> None:
        """内部方法：由 ``ObsInstrServer.bind_publisher()`` 调用。

        执行两件事：

        1. 将 ``self._instr`` 替换为与 ``server._instr`` 相同的张量对象，
           使 ``ObsInstrServer.sync_instr()`` 的 ``dist.broadcast`` 原地更新后
           publisher 无需额外操作即可获取最新指令。
        2. 建立 ZMQ PUSH socket，连接到 Server 的 PULL 端口。
        """
        # 共享张量（关键：同一内存对象）
        self._server = server
        self._instr  = server._instr   # noqa: SLF001 — 有意访问，两者共享

        # 建立 ZMQ PUSH 连接
        if self._zmq_ctx is not None:
            self.close()
        self._zmq_ctx   = zmq.Context()
        self._push_sock = self._zmq_ctx.socket(zmq.PUSH)
        self._push_sock.connect(f"tcp://{host}:{port}")
        # 非阻塞发送，发送失败静默丢弃（避免阻塞训练循环）
        self._push_sock.setsockopt(zmq.SNDTIMEO, 0)
        self._push_sock.setsockopt(zmq.LINGER,   0)
        self._push_sock.setsockopt(zmq.SNDHWM,   100)  # 最多排队 100 条消息

    # ──────────────────────────────────────────────────────────────────────────
    # 核心方法
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """是否已绑定到服务端并可发送数据。"""
        return self._push_sock is not None

    def push(self, obs) -> None:
        """按当前 ``_instr`` 指令切片 obs，通过 ZMQ PUSH 发送到 ``ObsInstrServer``。

        特性
        ----
        - **Fire-and-forget**：发送后不等待响应，不阻塞训练循环。
        - **只发送**：不修改 ``_instr``，不调用 ``dist.broadcast``。
        - 仅当 ``_instr[0]``（leader rank）与当前进程 rank 一致时发送。
        - 未绑定服务端时直接返回。

        参数
        ----
        obs : Tensor 或 list
            当前步的观测，形状 ``(num_envs, obs_dim)``。
        """
        if not self.enabled:
            return

        my_rank    = self._my_rank()
        leader     = int(self._instr[0].item())
        world_size = (dist.get_world_size()
                      if dist.is_available() and dist.is_initialized()
                      else 1)
        leader = leader % world_size

        if my_rank != leader:
            return  # 非 leader rank，静默跳过

        try:
            env_lo = max(0, int(self._instr[2].item()))
            env_hi = int(self._instr[3].item())

            if torch.is_tensor(obs):
                n      = int(obs.shape[0])
                env_hi = min(env_hi, self._num_envs, n)
                env_hi = max(env_hi, env_lo)
                obs_slice = obs[env_lo:env_hi].detach().cpu().tolist()
            else:
                n      = len(obs)
                env_hi = min(env_hi, self._num_envs, n)
                env_hi = max(env_hi, env_lo)
                obs_slice = list(obs[env_lo:env_hi])

            msg = {
                "type":        "obs_step",
                "rank":        my_rank,
                "task":        self._task,
                "instruction": self._instr.tolist(),
                "obs":         obs_slice,
            }
            # NOBLOCK：队列满或服务端未就绪时直接丢弃，不抛异常
            self._push_sock.send(json.dumps(msg).encode(), zmq.NOBLOCK)
        except zmq.Again:
            pass   # 队列满，静默丢弃
        except Exception:
            pass   # 其它异常静默处理，不影响训练主循环

    def close(self) -> None:
        """释放 ZMQ 资源。"""
        if self._push_sock is not None:
            self._push_sock.close()
            self._push_sock = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────────────

    def _my_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return self._init_rank
