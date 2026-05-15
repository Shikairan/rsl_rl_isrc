# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
# RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。
#
# 致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 rsl_rl_isrc 下维护与扩展。
# License: BSD-3-Clause（见仓库根目录及 setup.py）。
#
"""HTTP POST：``send_post_request``（仿真张量）与 ``StepObsPublisher``（按服务端指令切片上报观测）。

不依赖 rsl_rl_isrc 内其它业务子包；分布式下由 ``torch.distributed.broadcast`` 同步路由指令。
"""

import os
from typing import Optional

import requests
import torch
import torch.distributed as dist

_DEFAULT_POST_URL = "http://172.17.0.16:18888/post"


def send_post_request(data, rank, task):
    """向 ``RSL_RL_ISRC_POST_URL``（或默认 URL）POST 仿真/自定义张量数据。

    请求体含 ``type/data``、``rank``、``task``、``tensor``。成功返回服务端 JSON；异常时返回 ``{"error": ...}``。
    """
    header = {"Content-Type": "application/json"}
    url = os.environ.get("RSL_RL_ISRC_POST_URL", _DEFAULT_POST_URL)
    data_package = {"type": "data", "rank": rank, "task": task, "tensor": data}
    try:
        response = requests.post(url, json=data_package, headers=header)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


class StepObsPublisher:
    """在 ``env.step`` 之后按指令将观测切片 POST 到 ``RSL_RL_ISRC_OBS_POST_URL``（与 ``send_post_request`` 独立）。

    指令为 CPU ``int64`` 四元组 ``[sender_rank, aux, env_start, env_end)``；默认 ``[0,0,0,num_envs]``。
    仅 ``sender_rank`` 对应进程发起 HTTP；多卡时先 ``broadcast(changed)``，指令变更时再 ``broadcast(_instr)``。
    """

    def __init__(self, rank: int, task: str, num_envs: int):
        """rank/task 来自训练进程；``num_envs`` 用于默认指令上界与环境切片 clamp。"""
        self._init_rank = int(rank)
        self._task = task
        self._num_envs = max(1, int(num_envs))
        url = os.environ.get("RSL_RL_ISRC_OBS_POST_URL", "") or ""
        self._url = url.strip()
        self._enabled = bool(self._url)
        self._instr = torch.tensor([0, 0, 0, self._num_envs], dtype=torch.int64)

    def _my_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return self._init_rank

    def _wrap_leader(self, leader: int) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return int(leader)
        ws = dist.get_world_size()
        r = int(leader) % ws
        return r if r >= 0 else r + ws

    @staticmethod
    def _parse_state_from_response(data) -> Optional[torch.Tensor]:
        if not isinstance(data, dict) or "state" not in data:
            return None
        s = data["state"]
        if not isinstance(s, (list, tuple)) or len(s) < 4:
            return None
        try:
            vals = [int(s[i]) for i in range(4)]
            return torch.tensor(vals, dtype=torch.int64)
        except (TypeError, ValueError, IndexError):
            return None

    def _normalize_instr(self, t: torch.Tensor) -> torch.Tensor:
        out = t.clone().to(dtype=torch.int64)
        lo = max(0, int(out[2].item()))
        hi = int(out[3].item())
        hi = min(max(hi, lo), self._num_envs)
        out[2] = lo
        out[3] = hi
        out[0] = self._wrap_leader(int(out[0].item()))
        return out

    def push(self, obs) -> None:
        """若未配置 URL 则直接返回。否则由当前 leader rank 切片 ``obs`` 并 POST；全 rank 参与广播同步。"""
        if not self._enabled:
            return

        leader_src = self._wrap_leader(int(self._instr[0].item()))
        my_rank = self._my_rank()
        changed = torch.zeros(1, dtype=torch.int64)

        if my_rank == leader_src:
            old_instr = self._instr.clone()
            try:
                env_lo = max(0, int(self._instr[2].item()))
                env_hi = int(self._instr[3].item())
                if torch.is_tensor(obs):
                    n = int(obs.shape[0])
                    env_hi = min(env_hi, self._num_envs, n)
                    env_hi = max(env_hi, env_lo)
                    obs_slice = obs[env_lo:env_hi].detach().cpu().tolist()
                else:
                    n = len(obs)
                    env_hi = min(env_hi, self._num_envs, n)
                    env_hi = max(env_hi, env_lo)
                    obs_slice = obs[env_lo:env_hi]

                body = {
                    "type": "obs_step",
                    "rank": my_rank,
                    "task": self._task,
                    "instruction": old_instr.tolist(),
                    "obs": obs_slice,
                }
                resp = requests.post(
                    self._url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                new_instr = self._parse_state_from_response(data)
                if new_instr is not None:
                    new_instr = self._normalize_instr(new_instr)
                    if not torch.equal(new_instr, old_instr):
                        self._instr.copy_(new_instr)
                        changed[0] = 1
            except Exception:
                pass

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(changed, src=leader_src)
            if int(changed.item()) != 0:
                dist.broadcast(self._instr, src=leader_src)

    def close(self) -> None:
        """预留接口：当前无额外资源释放逻辑。"""
        pass
