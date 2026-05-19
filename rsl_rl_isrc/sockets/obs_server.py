# rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件。
# License: BSD-3-Clause
"""本地 ZMQ 观测服务端 ``ObsInstrServer``：与 Runner 同生命周期，替代 ``StepObsPublisher`` 的双向通信功能。

架构概览
---------
::

    外部控制器 ────(ZMQ REQ)────┐
                                ▼
    Runner (rank 0)            ObsInstrServer (后台线程)
    ├── ObsInstrServer             ├── PULL socket (obs_pull_port)
    │   └── sync_instr()           │       ← 接收各 rank 推来的 obs 数据
    │       (dist.broadcast)       ├── REP  socket (ctrl_rep_port)
    │                              │       ← 接收外部控制器的指令更新
    └── StepObsPublisher           └── 可选 HTTP 中继（仅 base_pos/quat/dof 行）
        └── push()                         → ZMQ 完整 obs_step；HTTP 为位姿 list
            └── ZMQ PUSH ──────────────────────────▲

指令同步流程
-----------
1. 外部控制器向 ``ctrl_rep_port`` 发送新指令（ZMQ REQ/REP）
2. ``ObsInstrServer`` 在 rank 0 更新内部 ``_instr`` 张量并置 **changed** 标志
3. Runner 在合适的同步点（如每轮 rollout 开始前）调用 ``server.sync_instr()``
4. ``sync_instr()`` 执行 ``dist.broadcast(_instr, src=0)`` → 原地更新各 rank 的张量
5. 因 ``StepObsPublisher._instr`` 与 ``server._instr`` 是**同一张量对象**，
   各 rank 的 publisher 自动获得最新指令

使用示例
--------
::

    # Runner.__init__ 中：
    server = ObsInstrServer(rank=self.rank, task=self.task, num_envs=env.num_envs)
    server.start()
    server.bind_publisher(self.step_obs)   # 关联 StepObsPublisher

    # learn() 每轮 rollout 开始前：
    server.sync_instr()                    # 广播最新指令到所有 rank

    # learn() 结束后：
    server.stop()
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, List, Optional, TYPE_CHECKING

import requests
import torch
import torch.distributed as dist
import zmq

if TYPE_CHECKING:
    from rsl_rl_isrc.sockets.http_post import StepObsPublisher

# ── 默认端口（可通过环境变量覆盖）────────────────────────────────────────────
_DEFAULT_OBS_PULL_PORT  = int(os.environ.get("RSL_RL_ISRC_OBS_PULL_PORT",  "15555"))
_DEFAULT_CTRL_REP_PORT  = int(os.environ.get("RSL_RL_ISRC_CTRL_REP_PORT",  "15556"))
_DEFAULT_RELAY_URL      = os.environ.get("RSL_RL_ISRC_OBS_RELAY_URL", "").strip()
_DEFAULT_RELAY_TIMEOUT  = float(os.environ.get("RSL_RL_ISRC_OBS_RELAY_TIMEOUT", "2"))


def default_obs_env_hi(num_envs: int) -> int:
    """默认 obs 切片上界：``min(64, num_envs)``。"""
    return min(64, max(1, int(num_envs)))


def _list_row_at(field: Any, index: int) -> list:
    """取某字段第 ``index`` 个 env 的行；字段缺失、非 list 或下标越界则返回 ``[]``。"""
    if not isinstance(field, list) or index < 0 or index >= len(field):
        return []
    row = field[index]
    return row if isinstance(row, list) else []


def robot_pose_rows_from_msg(msg: dict) -> List[List[list]]:
    """将 ``obs_step`` 消息转为 HTTP 对外格式：每 env 为 ``[base_pos, base_quat, dof_pos]``。

    结构示例::

        [
          [[x, y, z], [qx, qy, qz, qw], [dof...]],
          ...
        ]

    某一字段整条缺失或不是 list 时，该 env 对应槽位为 ``[]``。
    """
    pos = msg.get("base_pos")
    quat = msg.get("base_quat")
    dof = msg.get("dof_pos")
    n_envs = 0
    for field in (pos, quat, dof, msg.get("obs")):
        if isinstance(field, list):
            n_envs = max(n_envs, len(field))
    return [
        [_list_row_at(pos, i), _list_row_at(quat, i), _list_row_at(dof, i)]
        for i in range(n_envs)
    ]


class ObsInstrServer:
    """与 Runner 同生命周期的本地 ZMQ obs 服务端与指令管理器。

    **职责（替代 StepObsPublisher 的双向通信）**

    - 在 rank 0 上以后台线程运行两个 ZMQ socket：

      - ``PULL`` (``obs_pull_port``)：接收各 rank ``StepObsPublisher`` 推来的 obs 数据；
        可选将其 HTTP 中继到远端可视化服务。
      - ``REP``  (``ctrl_rep_port``)：接收外部控制器发来的指令更新请求；
        更新内部 ``_instr`` 并置 **changed** 标志。

    - ``sync_instr()``：**集体操作**，由 Runner 在各 rank 同步点调用；
      当指令已变更时执行 ``dist.broadcast(_instr, src=0)``，原地更新所有 rank。

    **与 StepObsPublisher 的关系**

    调用 ``bind_publisher(pub)`` 后：

    - ``pub._instr`` 指向与本 Server 相同的张量对象；
    - 每次 ``sync_instr()`` 广播后，``pub._instr`` 自动持有最新值，无需额外复制；
    - ``pub.push(obs)`` 只向本 Server 的 PULL 端口推送数据，不做任何接收。
    """

    def __init__(
        self,
        rank: int,
        task: str,
        num_envs: int,
        obs_pull_port: Optional[int] = None,
        ctrl_rep_port: Optional[int] = None,
        relay_url: str = _DEFAULT_RELAY_URL,
        relay_timeout: float = _DEFAULT_RELAY_TIMEOUT,
        print_obs: Optional[bool] = None,
    ):
        """
        参数
        ----
        rank : int
            当前进程 rank（用于单进程模式下的 leader 判断）。
        task : str
            任务名称（透传给中继消息）。
        num_envs : int
            并行环境数，用于指令合法性校验。
        obs_pull_port : int, optional
            接收 obs 的 PULL 端口。默认读取 ``RSL_RL_ISRC_OBS_PULL_PORT``（15555）。
        ctrl_rep_port : int, optional
            接收控制指令的 REP 端口。默认读取 ``RSL_RL_ISRC_CTRL_REP_PORT``（15556）。
        relay_url : str
            obs HTTP 中继目标 URL。为空则不中继。
        relay_timeout : float
            HTTP 中继超时（秒）。
        print_obs : bool, optional
            是否在收到 obs 时打印摘要；默认读 ``RSL_RL_ISRC_OBS_PRINT``（0/1）。
        """
        self._init_rank     = int(rank)
        self._task          = task
        self._num_envs      = max(1, int(num_envs))
        self._obs_pull_port = obs_pull_port or _DEFAULT_OBS_PULL_PORT
        self._ctrl_rep_port = ctrl_rep_port or _DEFAULT_CTRL_REP_PORT
        self._relay_url     = relay_url.strip()
        self._relay_timeout = relay_timeout
        if print_obs is None:
            print_obs = os.environ.get("RSL_RL_ISRC_OBS_PRINT", "0").strip() in (
                "1",
                "true",
                "yes",
            )
        self._print_obs = bool(print_obs)
        self._obs_print_count = 0

        # 指令张量：[sender_rank, aux, env_start, env_end]
        # 与绑定的 StepObsPublisher 共享同一对象（广播后自动更新）
        _env_hi = default_obs_env_hi(self._num_envs)
        self._instr: torch.Tensor = torch.tensor(
            [0, 0, 0, _env_hi], dtype=torch.int64
        )
        self._instr_lock    = threading.Lock()
        self._instr_changed = threading.Event()   # rank 0 专用标志

        self._running       = False
        self._thread: Optional[threading.Thread]  = None
        self._zmq_ctx: Optional[zmq.Context]      = None

    # ──────────────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """启动后台服务线程（**仅在 rank 0 上实际工作**，其它 rank 调用为空操作）。"""
        if self._running or not self._is_rank0():
            return
        self._zmq_ctx = zmq.Context()
        self._running = True
        self._thread  = threading.Thread(
            target=self._server_loop, daemon=True, name="ObsInstrServer"
        )
        self._thread.start()

    def stop(self) -> None:
        """停止后台线程并清理 ZMQ 上下文。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None

    # ──────────────────────────────────────────────────────────────────────────
    # 指令管理（Runner 调用）
    # ──────────────────────────────────────────────────────────────────────────

    def sync_instr(self) -> bool:
        """将 rank 0 的最新 ``_instr`` 广播到所有 rank（**集体操作**，须所有 rank 同时调用）。

        返回值
        ------
        bool
            True 表示本次同步发生了指令变更。
        """
        is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        if not is_dist:
            # 单进程：无需 broadcast，直接返回变更标志
            changed = self._instr_changed.is_set()
            if changed:
                self._instr_changed.clear()
            return changed

        # 分布式：rank 0 设置 changed_flag，broadcast 给所有 rank
        changed_flag = torch.zeros(1, dtype=torch.int64)
        if self._is_rank0() and self._instr_changed.is_set():
            changed_flag[0] = 1

        dist.broadcast(changed_flag, src=0)

        if int(changed_flag.item()):
            with self._instr_lock:
                # 原地广播 _instr（与所有绑定的 StepObsPublisher 共享同一张量对象，
                # 广播后无需额外赋值，各 rank 的 publisher._instr 自动持有最新值）
                dist.broadcast(self._instr, src=0)
            if self._is_rank0():
                self._instr_changed.clear()
            return True

        return False

    def get_instr(self) -> torch.Tensor:
        """获取当前指令张量的副本（线程安全）。"""
        with self._instr_lock:
            return self._instr.clone()

    # ──────────────────────────────────────────────────────────────────────────
    # 与 StepObsPublisher 的绑定
    # ──────────────────────────────────────────────────────────────────────────

    def bind_publisher(self, publisher: StepObsPublisher, env=None) -> None:
        """将 ``StepObsPublisher`` 与本 Server 关联。

        绑定后：

        - ``publisher._instr`` 与 ``server._instr`` 指向同一张量对象，
          ``sync_instr()`` 广播后 publisher 自动获得最新指令；
        - ``publisher.push()`` 通过 ZMQ PUSH 向本 Server 推送 obs 数据；
        - publisher 不再做任何 HTTP 接收或 ``dist.broadcast``。
        """
        publisher._bind_to_server(
            server=self,
            host="localhost",
            port=self._obs_pull_port,
        )
        if env is not None:
            publisher.set_env(env)

    # ──────────────────────────────────────────────────────────────────────────
    # 服务端主循环（后台线程，仅 rank 0 运行）
    # ──────────────────────────────────────────────────────────────────────────

    def _server_loop(self) -> None:
        """PULL + REP 双 socket 的事件循环。"""
        ctx = self._zmq_ctx

        # ── PULL：接收来自 StepObsPublisher 的 obs ────────────────────────────
        pull_sock: zmq.Socket = ctx.socket(zmq.PULL)
        pull_sock.bind(f"tcp://*:{self._obs_pull_port}")
        pull_sock.setsockopt(zmq.RCVTIMEO, 50)   # 50 ms 超时

        # ── REP：接收来自外部控制器的指令更新 ─────────────────────────────────
        rep_sock: zmq.Socket = ctx.socket(zmq.REP)
        rep_sock.bind(f"tcp://*:{self._ctrl_rep_port}")
        rep_sock.setsockopt(zmq.RCVTIMEO, 50)    # 50 ms 超时

        poller = zmq.Poller()
        poller.register(pull_sock, zmq.POLLIN)
        poller.register(rep_sock,  zmq.POLLIN)

        try:
            while self._running:
                try:
                    ready = dict(poller.poll(timeout=100))  # 100 ms
                except zmq.ZMQError:
                    break

                # ── 处理 obs 数据 ───────────────────────────────────────────
                if pull_sock in ready:
                    self._handle_obs(pull_sock)

                # ── 处理指令更新 ────────────────────────────────────────────
                if rep_sock in ready:
                    self._handle_ctrl(rep_sock)
        finally:
            pull_sock.close()
            rep_sock.close()

    def _handle_obs(self, sock: zmq.Socket) -> None:
        """从 PULL socket 读取 obs 消息，可选中继到 HTTP 端点。"""
        try:
            raw  = sock.recv(zmq.NOBLOCK)
            msg  = json.loads(raw.decode("utf-8"))
            if self._print_obs:
                self._print_obs_summary(msg)
            if self._relay_url:
                self._http_relay(msg)
        except zmq.Again:
            pass
        except (json.JSONDecodeError, Exception):
            pass

    def _print_obs_summary(self, msg: dict) -> None:
        """打印 obs 消息摘要（不输出完整向量）。"""
        self._obs_print_count += 1
        obs = msg.get("obs", [])
        n_envs = len(obs) if isinstance(obs, list) else 0
        obs_dim = len(obs[0]) if n_envs and isinstance(obs[0], list) else 0
        extra = ""
        for key in ("base_pos", "base_quat", "dof_pos"):
            data = msg.get(key)
            if isinstance(data, list) and data and isinstance(data[0], list):
                extra += f" {key}_shape=({len(data)}, {len(data[0])})"
        print(
            f"[obs #{self._obs_print_count}] type={msg.get('type')} "
            f"rank={msg.get('rank')} task={msg.get('task')} "
            f"instruction={msg.get('instruction')} obs_shape=({n_envs}, {obs_dim}){extra}"
        )

    def _handle_ctrl(self, sock: zmq.Socket) -> None:
        """从 REP socket 读取指令更新请求，更新 ``_instr`` 并回复 ok/error。"""
        try:
            raw = sock.recv(zmq.NOBLOCK)
            msg = json.loads(raw.decode("utf-8"))
            ok  = self._apply_instr_update(msg)
            sock.send(json.dumps({"ok": ok}).encode())
        except zmq.Again:
            pass
        except (json.JSONDecodeError, Exception):
            # REP socket 必须回复，否则进入错误状态
            try:
                sock.send(json.dumps({"ok": False, "error": "parse error"}).encode())
            except Exception:
                pass

    def _apply_instr_update(self, msg: dict) -> bool:
        """解析并应用来自控制器的指令更新消息。

        消息格式::

            {"state": [sender_rank, aux, env_start, env_end]}

        返回 True 表示指令发生了变更。
        """
        if "state" not in msg:
            return False
        s = msg["state"]
        if not isinstance(s, (list, tuple)) or len(s) < 4:
            return False
        try:
            new = torch.tensor([int(s[i]) for i in range(4)], dtype=torch.int64)
            # 合法性修正
            world_size = (dist.get_world_size()
                          if dist.is_available() and dist.is_initialized()
                          else 1)
            new[0] = int(new[0].item()) % world_size
            new[2] = max(0, int(new[2].item()))
            new[3] = min(max(int(new[3].item()), int(new[2].item())), self._num_envs)
        except (TypeError, ValueError, IndexError):
            return False

        with self._instr_lock:
            if torch.equal(new, self._instr):
                return False
            self._instr.copy_(new)   # 原地修改，与绑定 publisher 共享的张量同步更新
            self._instr_changed.set()
        return True

    def _http_relay(self, msg: dict) -> None:
        """将机器人位姿/关节行中继到 HTTP（仅 ``[base_pos, base_quat, dof_pos]`` 三层 list）。"""
        try:
            payload = robot_pose_rows_from_msg(msg)
            requests.post(
                self._relay_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self._relay_timeout,
            )
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────────────

    def _is_rank0(self) -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return self._init_rank == 0
