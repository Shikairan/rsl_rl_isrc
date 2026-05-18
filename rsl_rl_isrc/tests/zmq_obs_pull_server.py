#!/usr/bin/env python3
"""已废弃：独立 bind PULL 15555 与训练内 ObsInstrServer 冲突。

请使用 ``test_ppo_g1_isaac.py``（内置 ObsInstrServer）：

- obs 接收：训练进程 bind PULL 15555
- 改 env 切片：``zmq_obs_ctrl_client.py`` → REP 15556
- 打印 obs：训练加 ``--print-obs`` 或 ``RSL_RL_ISRC_OBS_PRINT=1``

本脚本仅保留作历史参考；若仍运行会在端口已被占用时失败。
"""

from __future__ import annotations

import sys

print(__doc__, file=sys.stderr)
sys.exit(1)
