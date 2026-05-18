#!/usr/bin/env python3
"""向 ObsInstrServer REP 端口发送 obs 切片指令。

训练进程（``test_ppo_g1_isaac.py``，默认未加 ``--no-zmq-obs``）会 bind：

- PULL ``15555``：接收 rollout obs
- REP  ``15556``：接收本脚本的 ``{"state": [rank, aux, env_lo, env_hi)}``

示例::

    python rsl_rl_isrc/tests/zmq_obs_ctrl_client.py --state 0 0 10 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def send_instruction(
    state: list[int],
    host: str = "localhost",
    port: int = 15556,
) -> dict:
    import zmq

    ctx = zmq.Context()
    req = ctx.socket(zmq.REQ)
    req.connect(f"tcp://{host}:{port}")
    req.send(json.dumps({"state": state}).encode())
    raw = req.recv()
    req.close()
    ctx.term()
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ObsInstrServer 指令客户端（ZMQ REQ）")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="训练进程所在主机",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_CTRL_REP_PORT", "15556")),
    )
    parser.add_argument(
        "--state",
        type=int,
        nargs=4,
        metavar=("RANK", "AUX", "ENV_LO", "ENV_HI"),
        required=True,
        help="指令 [sender_rank, aux, env_start, env_end)",
    )
    args = parser.parse_args()
    try:
        resp = send_instruction(args.state, host=args.host, port=args.port)
    except Exception as exc:
        print(f"错误: 无法连接 REP {args.host}:{args.port} — {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"已发送 state={args.state} → {resp}")


if __name__ == "__main__":
    main()
