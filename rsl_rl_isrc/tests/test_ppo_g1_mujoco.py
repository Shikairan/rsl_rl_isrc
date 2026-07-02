#!/usr/bin/env python3
"""G1 MuJoCo + PPO 训练（CPU 仿真 + GPU/CUDA 策略）。

依赖：``pip install 'mujoco>=3.1.0'``；模型 ``robotmodel/g1_description/scene.xml``。

运行::

    python rsl_rl_isrc/tests/test_ppo_g1_mujoco.py --num-envs 64 --max-iterations 5 --no-zmq-obs

CPU 策略回退::

    python rsl_rl_isrc/tests/test_ppo_g1_mujoco.py --device cpu --num-envs 16 --max-iterations 5
"""

from __future__ import annotations

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(script_dir)
project_root = os.path.dirname(package_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _resolve_policy_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("警告: CUDA 不可用，策略设备回退为 cpu。", file=sys.stderr)
        return "cpu"
    return requested


def run_g1_mujoco_training(
    num_envs: int,
    max_iterations: int,
    log_dir: str | None,
    sim_device: str = "cpu",
    device: str = "cuda:0",
    init_at_random_ep_len: bool = True,
    log_root: str | None = None,
    enable_obs_server: bool = True,
    obs_pull_port: int = 15555,
    ctrl_rep_port: int = 15556,
    print_obs: bool = False,
) -> None:
    from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner
    from rsl_rl_isrc.env.mujoco.make_g1_mujoco import (
        default_g1_mujoco_log_dir,
        has_mujoco,
        make_g1_mujoco_env,
        mujoco_import_error,
    )

    if not has_mujoco():
        print(f"错误: {mujoco_import_error()}", file=sys.stderr)
        sys.exit(1)

    policy_device = _resolve_policy_device(device)
    env, _, train_cfg = make_g1_mujoco_env(
        num_envs=num_envs,
        sim_device=sim_device,
    )
    train_cfg["runner"]["max_iterations"] = max_iterations

    if log_dir is None:
        log_dir = default_g1_mujoco_log_dir(train_cfg, log_root=log_root)

    runner = G1OnPolicyTestRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=policy_device,
        enable_obs_server=enable_obs_server,
        obs_pull_port=obs_pull_port,
        ctrl_rep_port=ctrl_rep_port,
        print_obs=print_obs,
    )

    if enable_obs_server and runner.obs_server is not None:
        instr = runner.get_instruction()
        print(
            f"ObsInstrServer: PULL tcp://*:{obs_pull_port}, "
            f"REP tcp://*:{ctrl_rep_port}, instruction={instr}"
        )

    print(
        f"开始训练: num_envs={num_envs}, max_iterations={max_iterations}, "
        f"sim_device={sim_device}, policy_device={policy_device}, log_dir={log_dir}"
    )
    runner.learn(max_iterations, init_at_random_ep_len=init_at_random_ep_len)


def _parse_args() -> argparse.Namespace:
    default_num_envs = int(os.environ.get("G1_NUM_ENVS", "64"))
    parser = argparse.ArgumentParser(description="G1 MuJoCo PPO 训练")
    parser.add_argument("--num-envs", type=int, default=default_num_envs)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-root", type=str, default=None)
    parser.add_argument(
        "--sim-device",
        type=str,
        default="cpu",
        help="MuJoCo 仿真设备（默认 cpu）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PPO 策略网络设备（默认 cuda:0）",
    )
    parser.add_argument(
        "--no-random-init-ep-len",
        action="store_true",
        help="禁用首轮随机 episode 长度",
    )
    parser.add_argument(
        "--no-zmq-obs",
        action="store_true",
        help="不启动 ObsInstrServer",
    )
    parser.add_argument("--print-obs", action="store_true")
    parser.add_argument(
        "--obs-pull-port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_OBS_PULL_PORT", "15555")),
    )
    parser.add_argument(
        "--ctrl-rep-port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_CTRL_REP_PORT", "15556")),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_g1_mujoco_training(
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        log_dir=args.log_dir,
        sim_device=args.sim_device,
        device=args.device,
        init_at_random_ep_len=not args.no_random_init_ep_len,
        log_root=args.log_root,
        enable_obs_server=not args.no_zmq_obs,
        obs_pull_port=args.obs_pull_port,
        ctrl_rep_port=args.ctrl_rep_port,
        print_obs=args.print_obs,
    )


if __name__ == "__main__":
    main()
