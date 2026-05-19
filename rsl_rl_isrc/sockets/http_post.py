# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""``StepObsPublisher``：按指令切片上报 obs 与机器人状态（ZMQ → ObsInstrServer）。

对外唯一训练遥测通路（经 ``ObsInstrServer`` 可选 HTTP 中继）::

    Runner.env.step → StepObsPublisher.push(obs)
        → ObsInstrServer (PULL) → RSL_RL_ISRC_OBS_RELAY_URL

``obs_step`` JSON 字段：

- ``obs``：策略观测，维数任意
- ``base_pos`` / ``base_quat`` / ``dof_pos``：若 VecEnv 提供（见 ``StateExportVecEnv``），
  与 ``obs`` 相同 env 切片；四元数 **xyzw**

独立工具 ``send_post_request`` 仍可供非训练脚本使用，Runner 不调用。
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, TYPE_CHECKING, Tuple

import requests
import torch
import torch.distributed as dist
import zmq

if TYPE_CHECKING:
    from rsl_rl_isrc.sockets.obs_server import ObsInstrServer

_DEFAULT_POST_URL = "http://172.17.0.16:18888/post"

_POST_TIMEOUT = float(os.environ.get("RSL_RL_ISRC_POST_TIMEOUT", "10"))
_OBS_TIMEOUT = float(os.environ.get("RSL_RL_ISRC_OBS_TIMEOUT", "2"))

_ROBOT_STATE_KEYS = ("base_pos", "base_quat", "dof_pos")


def send_post_request(data, rank, task):
    """向 ``RSL_RL_ISRC_POST_URL`` POST 自定义张量（``type=data``）。供独立脚本使用，非 Runner 遥测。"""
    header = {"Content-Type": "application/json"}
    url = os.environ.get("RSL_RL_ISRC_POST_URL", _DEFAULT_POST_URL)
    body = {"type": "data", "rank": rank, "task": task, "tensor": data}
    try:
        response = requests.post(url, json=body, headers=header, timeout=_POST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def _env_slice_bounds(env_lo: int, env_hi: int, num_envs: int, n_rows: int) -> Tuple[int, int]:
    env_hi = min(env_hi, num_envs, n_rows)
    env_hi = max(env_hi, env_lo)
    return env_lo, env_hi


def _slice_env_tensor(
    tensor: torch.Tensor,
    env_lo: int,
    env_hi: int,
    num_envs: int,
) -> List[List[float]]:
    env_lo, env_hi = _env_slice_bounds(env_lo, env_hi, num_envs, int(tensor.shape[0]))
    return tensor[env_lo:env_hi].detach().cpu().tolist()


class StepObsPublisher:
    """``env.step`` 后按 ``_instr`` 切片 obs / 机器人状态，ZMQ PUSH 到 ``ObsInstrServer``。"""

    def __init__(self, rank: int, task: str, num_envs: int):
        self._init_rank = int(rank)
        self._task = task
        self._num_envs = max(1, int(num_envs))

        self._instr: torch.Tensor = torch.tensor(
            [0, 0, 0, self._num_envs], dtype=torch.int64
        )

        self._zmq_ctx: Optional[zmq.Context] = None
        self._push_sock: Optional[zmq.Socket] = None
        self._server: Optional[ObsInstrServer] = None
        self._env: Any = None

    def set_env(self, env: Any) -> None:
        """绑定 VecEnv（如 ``StateExportVecEnv``），``push`` 时读取 ``base_pos`` 等。"""
        self._env = env

    def _bind_to_server(
        self,
        server: ObsInstrServer,
        host: str = "localhost",
        port: int = 15555,
    ) -> None:
        self._server = server
        self._instr = server._instr  # noqa: SLF001

        if self._zmq_ctx is not None:
            self.close()
        self._zmq_ctx = zmq.Context()
        self._push_sock = self._zmq_ctx.socket(zmq.PUSH)
        self._push_sock.connect(f"tcp://{host}:{port}")
        self._push_sock.setsockopt(zmq.SNDTIMEO, 0)
        self._push_sock.setsockopt(zmq.LINGER, 0)
        self._push_sock.setsockopt(zmq.SNDHWM, 100)

    @property
    def enabled(self) -> bool:
        return self._push_sock is not None

    def _append_robot_state(self, msg: dict, env_lo: int, env_hi: int) -> None:
        env = self._env
        if env is None:
            return
        if getattr(env, "has_robot_state", False) is False:
            from rsl_rl_isrc.env.state_export_vec_env import env_has_robot_state

            if not env_has_robot_state(env):
                return
        for key in _ROBOT_STATE_KEYS:
            tensor = getattr(env, key, None)
            if torch.is_tensor(tensor):
                msg[key] = _slice_env_tensor(tensor, env_lo, env_hi, self._num_envs)

    def push(self, obs) -> None:
        """按 ``_instr`` 切片 obs（及可选机器人状态），发送到 ``ObsInstrServer``。"""
        if not self.enabled:
            return

        my_rank = self._my_rank()
        leader = int(self._instr[0].item())
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        leader = leader % world_size

        if my_rank != leader:
            return

        try:
            env_lo = max(0, int(self._instr[2].item()))
            env_hi = int(self._instr[3].item())

            if torch.is_tensor(obs):
                n = int(obs.shape[0])
                env_lo, env_hi = _env_slice_bounds(env_lo, env_hi, self._num_envs, n)
                obs_slice = obs[env_lo:env_hi].detach().cpu().tolist()
            else:
                n = len(obs)
                env_lo, env_hi = _env_slice_bounds(env_lo, env_hi, self._num_envs, n)
                obs_slice = list(obs[env_lo:env_hi])

            msg = {
                "type": "obs_step",
                "rank": my_rank,
                "task": self._task,
                "instruction": self._instr.tolist(),
                "obs": obs_slice,
            }
            self._append_robot_state(msg, env_lo, env_hi)

            self._push_sock.send(json.dumps(msg).encode(), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception:
            pass

    def close(self) -> None:
        if self._push_sock is not None:
            self._push_sock.close()
            self._push_sock = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None

    def _my_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return self._init_rank
