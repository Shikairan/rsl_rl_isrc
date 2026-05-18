#!/usr/bin/env python3
"""测试用 HTTP POST 接收服务，默认 ``http://0.0.0.0:18888/post``。

用于接收：

- ``ObsInstrServer`` 中继：设置 ``RSL_RL_ISRC_OBS_RELAY_URL=http://<host>:18888/post``
- ``send_post_request``：设置 ``RSL_RL_ISRC_POST_URL=http://<host>:18888/post``

用法::

    python rsl_rl_isrc/tests/http_post_server.py

    # 训练侧
    export RSL_RL_ISRC_OBS_RELAY_URL=http://127.0.0.1:18888/post
    python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --num-envs 64 --max-iterations 5 --print-obs
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse


def _summarize_body(body: dict) -> str:
    msg_type = body.get("type", "?")
    rank = body.get("rank")
    task = body.get("task")

    if msg_type == "obs_step":
        obs = body.get("obs", [])
        n_envs = len(obs) if isinstance(obs, list) else 0
        obs_dim = len(obs[0]) if n_envs and isinstance(obs[0], list) else 0
        return (
            f"type={msg_type} rank={rank} task={task} "
            f"instruction={body.get('instruction')} obs_shape=({n_envs}, {obs_dim})"
        )

    if msg_type == "data":
        tensor = body.get("tensor")
        if isinstance(tensor, list):
            if tensor and isinstance(tensor[0], list):
                shape = (len(tensor), len(tensor[0]))
            else:
                shape = (len(tensor),)
        else:
            shape = type(tensor).__name__
        return f"type={msg_type} rank={rank} task={task} tensor_shape={shape}"

    keys = list(body.keys())[:8]
    return f"type={msg_type} rank={rank} task={task} keys={keys}"


class PostHandler(BaseHTTPRequestHandler):
    """只处理 ``POST /post``（路径可通过 ``--path`` 配置）。"""

    server_count = 0
    path = "/post"

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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "received": n}).encode())


def run_server(host: str, port: int, path: str) -> None:
    PostHandler.path = path if path.startswith("/") else f"/{path}"
    httpd = HTTPServer((host, port), PostHandler)
    url = f"http://{host}:{port}{PostHandler.path}"
    if host == "0.0.0.0":
        url_local = f"http://127.0.0.1:{port}{PostHandler.path}"
        print(f"[http_post_server] 监听 {url} （本机可用 {url_local}）")
    else:
        print(f"[http_post_server] 监听 {url}")
    print("  obs relay:  export RSL_RL_ISRC_OBS_RELAY_URL=" + url)
    print("  仿真 POST:  export RSL_RL_ISRC_POST_URL=" + url)
    print("（Ctrl+C 退出）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[http_post_server] 共接收 {PostHandler.server_count} 条 POST")
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="测试用 HTTP POST 接收并打印")
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("RSL_RL_ISRC_POST_HOST", "0.0.0.0"),
        help="绑定地址（默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_POST_PORT", "18888")),
    )
    parser.add_argument("--path", type=str, default="/post")
    args = parser.parse_args()
    run_server(args.host, args.port, args.path)


if __name__ == "__main__":
    main()
