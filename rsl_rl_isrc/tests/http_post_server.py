#!/usr/bin/env python3
"""测试用 HTTP POST 接收服务，默认 ``http://0.0.0.0:18888/post``。

用于接收训练遥测（唯一推荐通路）：

- ``ObsInstrServer`` HTTP 中继：设置 ``RSL_RL_ISRC_OBS_RELAY_URL=http://<host>:18888/post``
  （body 为 ``[[base_pos, base_quat, dof_pos], ...]`` 每 env 一行；ZMQ 仍为完整 ``obs_step``）

独立脚本仍可使用 ``send_post_request`` + ``RSL_RL_ISRC_POST_URL``（``type=data``），Runner 不调用。

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


def _is_robot_pose_payload(body: Any) -> bool:
    """HTTP relay 体：``[[base_pos, base_quat, dof_pos], ...]``（每 env 一行长度为 3）。"""
    if not isinstance(body, list):
        return False
    if not body:
        return True
    row = body[0]
    return isinstance(row, list) and len(row) == 3


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
    print("  训练遥测:  export RSL_RL_ISRC_OBS_RELAY_URL=" + url)
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
