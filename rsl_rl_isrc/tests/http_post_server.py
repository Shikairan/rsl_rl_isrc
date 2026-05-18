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

    # 仅摘要、不打印数值
    python rsl_rl_isrc/tests/http_post_server.py --quiet

    # 打印前 3 个 env 的完整行（观测维较大时注意终端刷屏）
    python rsl_rl_isrc/tests/http_post_server.py --preview-envs 3 --preview-elems 0
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse


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


def _summarize_body(body: dict) -> str:
    msg_type = body.get("type", "?")
    rank = body.get("rank")
    task = body.get("task")

    if msg_type == "obs_step":
        obs = body.get("obs", [])
        return (
            f"type={msg_type} rank={rank} task={task} "
            f"instruction={body.get('instruction')} obs_shape={_list_shape(obs)}"
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
    body: dict,
    *,
    max_elems: int,
    max_env_rows: int,
) -> None:
    """在摘要行之后打印真实数值（JSON 已为 list，非 Tensor）。"""
    msg_type = body.get("type")
    if msg_type == "obs_step":
        key = "obs"
    elif msg_type == "data":
        key = "tensor"
    else:
        snippet = json.dumps(body, ensure_ascii=False)
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        print(f"    body = {snippet}")
        return

    data = body.get(key)
    if not isinstance(data, list) or not data:
        print(f"    {key} = (empty or not a list)")
        return

    n_show = min(len(data), max(1, max_env_rows))
    for i in range(n_show):
        row = data[i]
        print(f"    {key}[{i}] = {_format_row(row, max_elems)}")

    if len(data) > n_show:
        print(f"    ... ({len(data) - n_show} more env rows omitted)")


class PostHandler(BaseHTTPRequestHandler):
    """只处理 ``POST /post``（路径可通过 ``--path`` 配置）。"""

    server_count = 0
    path = "/post"
    show_preview = True
    preview_elems = 32
    preview_envs = 1

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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "received": n}).encode())


def run_server(
    host: str,
    port: int,
    path: str,
    *,
    show_preview: bool,
    preview_elems: int,
    preview_envs: int,
) -> None:
    PostHandler.path = path if path.startswith("/") else f"/{path}"
    PostHandler.show_preview = show_preview
    PostHandler.preview_elems = preview_elems
    PostHandler.preview_envs = preview_envs

    httpd = HTTPServer((host, port), PostHandler)
    url = f"http://{host}:{port}{PostHandler.path}"
    if host == "0.0.0.0":
        url_local = f"http://127.0.0.1:{port}{PostHandler.path}"
        print(f"[http_post_server] 监听 {url} （本机可用 {url_local}）")
    else:
        print(f"[http_post_server] 监听 {url}")
    print("  obs relay:  export RSL_RL_ISRC_OBS_RELAY_URL=" + url)
    print("  仿真 POST:  export RSL_RL_ISRC_POST_URL=" + url)
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
        show_preview=not args.quiet,
        preview_elems=args.preview_elems,
        preview_envs=args.preview_envs,
    )


if __name__ == "__main__":
    main()
