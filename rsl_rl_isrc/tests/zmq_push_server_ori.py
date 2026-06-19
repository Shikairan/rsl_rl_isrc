#!/usr/bin/env python3
"""HTTP POST 接收 + ZMQ PUSH 转发（``http_post_server.py`` 的 ZMQ 中继版）。

训练侧仍通过 HTTP 中继推送遥测；本脚本收到后与 ``http_post_server`` 相同地解析/预览，
并额外将 body 经 ZMQ PUSH 转发给下游 PULL 消费者。

ZMQ 对外格式：每 env 将 ``[base_pos, base_quat, dof_pos]`` 合并为单向量
``[x, y, z, qx, qy, qz, qw, dof...]``（HTTP 终端预览仍为三段结构）。

用法::

    # 终端 1：下游 PULL 消费者
    python -c "
import zmq, json
ctx = zmq.Context()
pull = ctx.socket(zmq.PULL)
pull.bind('tcp://127.0.0.1:18889')
while True:
    print(json.loads(pull.recv().decode()))
"

    # 终端 2：中继服务
    python rsl_rl_isrc/tests/zmq_push_server.py

    # 终端 3：训练
    export RSL_RL_ISRC_OBS_RELAY_URL=http://127.0.0.1:18888/post
    torchrun ... rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py ...
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

import zmq


def _list_shape(data: Any) -> str:
    if not isinstance(data, list):
        return type(data).__name__
    if not data:
        return "(0,)"
    if isinstance(data[0], list):
        return f"({len(data)}, {len(data[0])})"
    return f"({len(data)},)"


def _format_row(row: Any, max_elems: int) -> str:
    """将一行数值格式化为 JSON；max_elems<=0 表示不截断。"""
    if not isinstance(row, list):
        return json.dumps(row, ensure_ascii=False)
    n = len(row)
    if max_elems <= 0 or n <= max_elems:
        return json.dumps(row, ensure_ascii=False)
    head = row[:max_elems]
    return json.dumps(head, ensure_ascii=False) + f" ... (+{n - max_elems})"


def _is_robot_pose_payload(body: Any) -> bool:
    """HTTP relay 体：``[[base_pos, base_quat, dof_pos], ...]``（每 env 一行长度为 3）。"""
    if not isinstance(body, list):
        return False
    if not body:
        return True
    row = body[0]
    return isinstance(row, list) and len(row) == 3


def _merge_robot_pose_row(row: Any) -> list:
    """``[base_pos, base_quat, dof_pos]`` → ``[x, y, z, qx, qy, qz, qw, ...]``。"""
    if not isinstance(row, list) or len(row) != 3:
        return []
    vec: list = []
    for part in row:
        if isinstance(part, list):
            vec.extend(part)
    return vec


def _body_for_zmq_push(body: Any) -> Any:
    """ZMQ 转发前扁平化；HTTP 打印仍使用原始 body。"""
    if _is_robot_pose_payload(body):
        return [_merge_robot_pose_row(row) for row in body]

    if isinstance(body, dict) and body.get("type") == "obs_step":
        pos = body.get("base_pos") or []
        quat = body.get("base_quat") or []
        dof = body.get("dof_pos") or []
        n = max(
            len(pos) if isinstance(pos, list) else 0,
            len(quat) if isinstance(quat, list) else 0,
            len(dof) if isinstance(dof, list) else 0,
        )
        merged = []
        for i in range(n):
            vec: list = []
            for field in (pos, quat, dof):
                if (
                    isinstance(field, list)
                    and i < len(field)
                    and isinstance(field[i], list)
                ):
                    vec.extend(field[i])
            merged.append(vec)
        return merged

    return body


def _summarize_robot_pose_rows(rows: Any) -> str:
    if not isinstance(rows, list):
        return f"robot_pose type={type(rows).__name__}"
    n = len(rows)
    if n == 0:
        return "robot_pose envs=0"
    sample = rows[0]
    if isinstance(sample, list) and len(sample) == 3:
        shapes = (
            _list_shape(sample[0]),
            _list_shape(sample[1]),
            _list_shape(sample[2]),
        )
        return f"robot_pose envs={n} sample=[pos{shapes[0]}, quat{shapes[1]}, dof{shapes[2]}]"
    return f"robot_pose envs={n} (unexpected row layout)"


def _print_robot_pose_preview(
    rows: list,
    *,
    max_elems: int,
    max_env_rows: int,
) -> None:
    if not isinstance(rows, list) or not rows:
        print("    (empty robot_pose list)")
        return
    labels = ("base_pos", "base_quat", "dof_pos")
    n_show = min(len(rows), max(1, max_env_rows))
    for i in range(n_show):
        row = rows[i]
        if not isinstance(row, list) or len(row) != 3:
            print(f"    env[{i}] = (invalid row)")
            continue
        for label, part in zip(labels, row):
            print(f"    env[{i}].{label} = {_format_row(part, max_elems)}")
    if len(rows) > n_show:
        print(f"    ... ({len(rows) - n_show} more env rows omitted)")


def _summarize_body(body: Any) -> str:
    if _is_robot_pose_payload(body):
        return _summarize_robot_pose_rows(body)
    if not isinstance(body, dict):
        return f"body type={type(body).__name__}"

    msg_type = body.get("type", "?")
    rank = body.get("rank")
    task = body.get("task")

    if msg_type == "obs_step":
        obs = body.get("obs", [])
        extra = ""
        for key in ("base_pos", "base_quat", "dof_pos"):
            if key in body:
                extra += f" {key}={_list_shape(body.get(key))}"
        return (
            f"type={msg_type} rank={rank} task={task} "
            f"instruction={body.get('instruction')} obs_shape={_list_shape(obs)}{extra}"
        )

    if msg_type == "data":
        tensor = body.get("tensor")
        return (
            f"type={msg_type} rank={rank} task={task} "
            f"tensor_shape={_list_shape(tensor)}"
        )

    keys = list(body.keys())[:8]
    return f"type={msg_type} rank={rank} task={task} keys={keys}"


def _print_payload_preview(
    body: Any,
    *,
    max_elems: int,
    max_env_rows: int,
) -> None:
    """在摘要行之后打印真实数值（JSON 已为 list，非 Tensor）。"""
    if _is_robot_pose_payload(body):
        _print_robot_pose_preview(
            body,
            max_elems=max_elems,
            max_env_rows=max_env_rows,
        )
        return
    if not isinstance(body, dict):
        print(f"    body = {json.dumps(body, ensure_ascii=False)[:500]}")
        return

    msg_type = body.get("type")
    if msg_type == "obs_step":
        keys = ["obs", "base_pos", "base_quat", "dof_pos"]
    elif msg_type == "data":
        keys = ["tensor"]
    else:
        snippet = json.dumps(body, ensure_ascii=False)
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        print(f"    body = {snippet}")
        return

    for key in keys:
        if key not in body:
            continue
        data = body.get(key)
        if not isinstance(data, list) or not data:
            print(f"    {key} = (empty or not a list)")
            continue
        n_show = min(len(data), max(1, max_env_rows))
        for i in range(n_show):
            row = data[i]
            print(f"    {key}[{i}] = {_format_row(row, max_elems)}")
        if len(data) > n_show:
            print(f"    ... ({key}: {len(data) - n_show} more env rows omitted)")


class ZmqForwarder:
    """ZMQ PUSH 客户端：connect 到下游 PULL bind 地址。"""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)
        self._ctx: Optional[zmq.Context] = None
        self._push_sock: Optional[zmq.Socket] = None
        self._enabled = False

    @property
    def endpoint(self) -> str:
        return f"tcp://{self._host}:{self._port}"

    def start(self) -> None:
        self._ctx = zmq.Context()
        self._push_sock = self._ctx.socket(zmq.PUSH)
        self._push_sock.connect(self.endpoint)
        self._push_sock.setsockopt(zmq.SNDTIMEO, 0)
        self._push_sock.setsockopt(zmq.LINGER, 0)
        self._push_sock.setsockopt(zmq.SNDHWM, 100)
        self._enabled = True

    def forward(self, body: Any) -> bool:
        if not self._enabled or self._push_sock is None:
            return False
        try:
            zmq_body = _body_for_zmq_push(body)
            payload = json.dumps(zmq_body, ensure_ascii=False).encode("utf-8")
            self._push_sock.send(payload, zmq.NOBLOCK)
            return True
        except zmq.Again:
            print("[zmq_push_server] ZMQ PUSH 队列满，丢弃本条消息")
            return False
        except Exception as exc:
            print(f"[zmq_push_server] ZMQ PUSH 失败: {exc!r}")
            return False

    def close(self) -> None:
        if self._push_sock is not None:
            self._push_sock.close()
            self._push_sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
        self._enabled = False


class PostHandler(BaseHTTPRequestHandler):
    """只处理 ``POST /post``（路径可通过 ``--path`` 配置）。"""

    server_count = 0
    path = "/post"
    show_preview = True
    preview_elems = 32
    preview_envs = 1
    zmq_forwarder: Optional[ZmqForwarder] = None

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != PostHandler.path:
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            PostHandler.server_count += 1
            n = PostHandler.server_count
            print(f"[#{n}] JSON 解析失败: {exc!r} len={len(raw)}")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode())
            return

        PostHandler.server_count += 1
        n = PostHandler.server_count
        print(f"[#{n}] {_summarize_body(body)}")
        if PostHandler.show_preview:
            _print_payload_preview(
                body,
                max_elems=PostHandler.preview_elems,
                max_env_rows=PostHandler.preview_envs,
            )

        zmq_ok = False
        if PostHandler.zmq_forwarder is not None:
            zmq_ok = PostHandler.zmq_forwarder.forward(body)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"ok": True, "received": n, "zmq_forwarded": zmq_ok}).encode()
        )


def run_server(
    host: str,
    port: int,
    path: str,
    *,
    zmq_push_host: str,
    zmq_push_port: int,
    enable_zmq_push: bool,
    show_preview: bool,
    preview_elems: int,
    preview_envs: int,
) -> None:
    PostHandler.path = path if path.startswith("/") else f"/{path}"
    PostHandler.show_preview = show_preview
    PostHandler.preview_elems = preview_elems
    PostHandler.preview_envs = preview_envs

    forwarder: Optional[ZmqForwarder] = None
    if enable_zmq_push:
        forwarder = ZmqForwarder(zmq_push_host, zmq_push_port)
        forwarder.start()
        PostHandler.zmq_forwarder = forwarder
    else:
        PostHandler.zmq_forwarder = None

    httpd = HTTPServer((host, port), PostHandler)
    url = f"http://{host}:{port}{PostHandler.path}"
    if host == "0.0.0.0":
        url_local = f"http://127.0.0.1:{port}{PostHandler.path}"
        print(f"[zmq_push_server] HTTP 监听 {url} （本机可用 {url_local}）")
        relay_url = url_local
    else:
        print(f"[zmq_push_server] HTTP 监听 {url}")
        relay_url = url

    print("  训练遥测:  export RSL_RL_ISRC_OBS_RELAY_URL=" + relay_url)
    if enable_zmq_push and forwarder is not None:
        print(f"[zmq_push_server] ZMQ PUSH → {forwarder.endpoint}")
        print("  ZMQ 格式: 每 env 为 [x,y,z,qx,qy,qz,qw,dof...] 合并向量")
        print("  下游请在上述地址 PULL bind")
    else:
        print("[zmq_push_server] ZMQ 转发已关闭（--no-zmq-push）")
    if show_preview:
        print(
            f"  数值预览: 每消息前 {preview_envs} 个 env，"
            f"每行最多 {preview_elems if preview_elems > 0 else '全部'} 维"
        )
    else:
        print("  数值预览: 已关闭（--quiet）")
    print("（Ctrl+C 退出）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[zmq_push_server] 共接收 {PostHandler.server_count} 条 POST")
    finally:
        httpd.server_close()
        if forwarder is not None:
            forwarder.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP POST 接收并 ZMQ PUSH 转发（训练遥测中继）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("RSL_RL_ISRC_POST_HOST", "0.0.0.0"),
        help="HTTP 绑定地址（默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_POST_PORT", "18888")),
    )
    parser.add_argument("--path", type=str, default="/post")
    parser.add_argument(
        "--zmq-push-host",
        type=str,
        default=os.environ.get("RSL_RL_ISRC_ZMQ_PUSH_HOST", "127.0.0.1"),
        help="ZMQ PUSH connect 目标主机（下游 PULL bind 地址）",
    )
    parser.add_argument(
        "--zmq-push-port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_ZMQ_PUSH_PORT", "18889")),
        help="ZMQ PUSH connect 目标端口",
    )
    parser.add_argument(
        "--no-zmq-push",
        action="store_true",
        help="仅 HTTP 接收+打印，不 ZMQ 转发",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只打印摘要，不打印 obs/tensor 数值",
    )
    parser.add_argument(
        "--preview-envs",
        type=int,
        default=1,
        metavar="N",
        help="每条消息预览前 N 个并行 env 的行（默认 1）",
    )
    parser.add_argument(
        "--preview-elems",
        type=int,
        default=32,
        metavar="K",
        help="每行最多打印 K 个标量；0 表示打印整行（默认 32）",
    )
    args = parser.parse_args()
    run_server(
        args.host,
        args.port,
        args.path,
        zmq_push_host=args.zmq_push_host,
        zmq_push_port=args.zmq_push_port,
        enable_zmq_push=not args.no_zmq_push,
        show_preview=not args.quiet,
        preview_elems=args.preview_elems,
        preview_envs=args.preview_envs,
    )


if __name__ == "__main__":
    main()
