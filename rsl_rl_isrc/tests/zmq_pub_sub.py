#!/usr/bin/env python3
"""HTTP POST 接收 + ZMQ PUB 转发（AlphaRay training mode 格式）。

训练侧仍通过 HTTP 中继推送遥测；本脚本收到后与 ``http_post_server`` 相同地解析/预览，
并额外按 ``training_mode_zmq_format.md`` 经 ZMQ PUB 发布：

``<标签> <JSON 一维数字数组>``

多 env 时先将每 env 的 ``[base_pos, base_quat, dof_pos]`` 合并为单向量，再首尾拼平。

用法::

    # 终端 1：AlphaRay 或 SUB 测试
    ./AlphaRayDemoView --mode=training --endpoint=tcp://127.0.0.1:6006 \\
      --num-envs=64 --frame-width=26

    # 终端 2：中继
    python rsl_rl_isrc/tests/zmq_pub_sub.py

    # 终端 3：训练
    export RSL_RL_ISRC_OBS_RELAY_URL=http://127.0.0.1:18888/post
    torchrun ... rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
    """合并为每 env 单向量；HTTP 打印仍使用原始 body。"""
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


def _flat_training_frames(merged_rows: list[list]) -> list[float]:
    """二维合并向量 → 一维拼平数组（跳过空 env）。"""
    flat: list[float] = []
    for vec in merged_rows:
        if vec:
            flat.extend(vec)
    return flat


def _format_training_mode_message(body: Any, label: str) -> str:
    """``label + " " + json_array``（training mode 文本格式）。"""
    merged = _body_for_zmq_push(body)
    if not isinstance(merged, list):
        merged = [merged] if merged else []
    flat = _flat_training_frames(merged)
    payload = json.dumps(flat, separators=(",", ":"))
    return f"{label} {payload}"


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


class ZmqPubForwarder:
    """ZMQ PUB：bind 端点，发送 training mode 文本消息。"""

    def __init__(self, endpoint: str, label: str) -> None:
        self._endpoint = endpoint.strip()
        self._label = label.strip()
        self._ctx: Optional[zmq.Context] = None
        self._pub_sock: Optional[zmq.Socket] = None
        self._enabled = False

    def start(self) -> None:
        self._ctx = zmq.Context()
        self._pub_sock = self._ctx.socket(zmq.PUB)
        self._pub_sock.setsockopt(zmq.SNDHWM, 100)
        self._pub_sock.bind(self._endpoint)
        # PUB/SUB 连接就绪（见 training_mode_zmq_format.md）
        time.sleep(0.5)
        self._enabled = True

    def forward(self, body: Any) -> bool:
        if not self._enabled or self._pub_sock is None:
            return False
        try:
            msg = _format_training_mode_message(body, self._label)
            self._pub_sock.send_string(msg, zmq.NOBLOCK)
            return True
        except zmq.Again:
            print("[zmq_pub_sub] ZMQ PUB 队列满，丢弃本条消息")
            return False
        except Exception as exc:
            print(f"[zmq_pub_sub] ZMQ PUB 失败: {exc!r}")
            return False

    def close(self) -> None:
        if self._pub_sock is not None:
            self._pub_sock.close()
            self._pub_sock = None
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
    zmq_forwarder: Optional[ZmqPubForwarder] = None

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
            json.dumps({"ok": True, "received": n, "zmq_pub_forwarded": zmq_ok}).encode()
        )


def run_server(
    host: str,
    port: int,
    path: str,
    *,
    zmq_endpoint: str,
    zmq_label: str,
    enable_zmq_pub: bool,
    show_preview: bool,
    preview_elems: int,
    preview_envs: int,
) -> None:
    PostHandler.path = path if path.startswith("/") else f"/{path}"
    PostHandler.show_preview = show_preview
    PostHandler.preview_elems = preview_elems
    PostHandler.preview_envs = preview_envs

    forwarder: Optional[ZmqPubForwarder] = None
    if enable_zmq_pub:
        forwarder = ZmqPubForwarder(zmq_endpoint, zmq_label)
        forwarder.start()
        PostHandler.zmq_forwarder = forwarder
    else:
        PostHandler.zmq_forwarder = None

    httpd = HTTPServer((host, port), PostHandler)
    url = f"http://{host}:{port}{PostHandler.path}"
    if host == "0.0.0.0":
        url_local = f"http://127.0.0.1:{port}{PostHandler.path}"
        print(f"[zmq_pub_sub] HTTP 监听 {url} （本机可用 {url_local}）")
        relay_url = url_local
    else:
        print(f"[zmq_pub_sub] HTTP 监听 {url}")
        relay_url = url

    print("  训练遥测:  export RSL_RL_ISRC_OBS_RELAY_URL=" + relay_url)
    if enable_zmq_pub and forwarder is not None:
        print(f"[zmq_pub_sub] ZMQ PUB bind {zmq_endpoint}")
        print(f"  ZMQ 格式: {zmq_label} [x,y,z,qx,qy,qz,qw,dof... 拼平多 env]")
        print(
            f"  AlphaRay: --mode=training --endpoint={zmq_endpoint} "
            f"--frame-width=<每env维数>"
        )
    else:
        print("[zmq_pub_sub] ZMQ 转发已关闭（--no-zmq-pub）")
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
        print(f"\n[zmq_pub_sub] 共接收 {PostHandler.server_count} 条 POST")
    finally:
        httpd.server_close()
        if forwarder is not None:
            forwarder.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP POST 接收并 ZMQ PUB 转发（AlphaRay training mode）"
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
        "--zmq-endpoint",
        type=str,
        default=os.environ.get("RSL_RL_ISRC_ZMQ_PUB_ENDPOINT", "tcp://127.0.0.1:6006"),
        help="ZMQ PUB bind 地址（渲染端 SUB connect）",
    )
    parser.add_argument(
        "--zmq-label",
        type=str,
        default=os.environ.get("RSL_RL_ISRC_ZMQ_PUB_LABEL", "c1:0"),
        help="training mode 消息标签（从 env 0 连续填）",
    )
    parser.add_argument(
        "--no-zmq-pub",
        action="store_true",
        help="仅 HTTP 接收+打印，不 ZMQ PUB",
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
        zmq_endpoint=args.zmq_endpoint,
        zmq_label=args.zmq_label,
        enable_zmq_pub=not args.no_zmq_pub,
        show_preview=not args.quiet,
        preview_elems=args.preview_elems,
        preview_envs=args.preview_envs,
    )


if __name__ == "__main__":
    main()
