#!/usr/bin/env python3
"""G1 单腿站立任务 PPO 训练入口（任意一脚支撑）。

本脚本与行走训练（``test_ppo_g1.py``）完全独立、互不干涉：
- 使用 ``G1SingleLegCfg`` + ``G1SingleLegRobot``，关闭行走奖励与速度指令
- 通过 ``make_g1_isaac_env(task="single_leg")`` 接入工厂
- 日志默认写入 ``logs/g1_single_leg/<timestamp>``

运行示例::

    # GPU 正式训练
    python rsl_rl_isrc/tests/test_ppo_g1_single_leg.py --num-envs 4096 --max-iterations 10000

    # 快速冒烟验证（无 ZMQ）
    python rsl_rl_isrc/tests/test_ppo_g1_single_leg.py \\
        --num-envs 64 --max-iterations 5 --no-zmq-obs

    # 从 checkpoint 恢复训练
    python rsl_rl_isrc/tests/test_ppo_g1_single_leg.py --resume --load-run -1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(script_dir)
project_root = os.path.dirname(package_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _apply_numpy_isaac_compat() -> None:
    compat_path = os.path.join(
        package_root, "env", "isaac_gym", "numpy_compat.py"
    )
    spec = importlib.util.spec_from_file_location("_isaac_numpy_compat", compat_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {compat_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_numpy_isaac_compat()


def _check_isaac_cuda() -> None:
    _apply_numpy_isaac_compat()
    try:
        import isaacgym  # noqa: F401
    except ImportError as exc:
        print(f"错误: 无法加载 isaacgym（{exc}）", file=sys.stderr)
        sys.exit(1)

    import torch

    if not torch.cuda.is_available():
        print("错误: 需要 CUDA GPU（cuda:0）。", file=sys.stderr)
        sys.exit(1)


def run_g1_single_leg_training(
    num_envs: int,
    max_iterations: int,
    log_dir: str | None,
    device: str = "cuda:0",
    headless: bool = True,
    init_at_random_ep_len: bool = True,
    resume: bool = False,
    load_run: str | int = -1,
    checkpoint: int = -1,
    log_root: str | None = None,
    enable_obs_server: bool = True,
    obs_pull_port: int = 15557,
    ctrl_rep_port: int = 15558,
    print_obs: bool = False,
) -> None:
    """使用 G1 单腿站立环境与 G1OnPolicyTestRunner 进行 PPO 训练。

    Args:
        num_envs: 并行环境数量。
        max_iterations: PPO 迭代次数。
        log_dir: TensorBoard / checkpoint 存储目录；为 None 时自动生成。
        device: 仿真与策略运行设备（如 ``"cuda:0"``）。
        headless: 是否无头模式。
        init_at_random_ep_len: 是否在首轮随机化 episode 长度（有助于数据多样性）。
        resume: 是否从已有 checkpoint 恢复训练。
        load_run: 恢复时的运行目录名或 ``-1``（最近一次）。
        checkpoint: 恢复时的 checkpoint 编号或 ``-1``（最新）。
        log_root: 日志根目录（仅当 ``log_dir`` 为 None 时有效）。
        enable_obs_server: 是否启动 ZMQ ObsInstrServer。
        obs_pull_port: ObsInstrServer PULL 端口（默认 15557，避免与行走任务冲突）。
        ctrl_rep_port: ObsInstrServer REP 端口（默认 15558）。
        print_obs: 是否在收到 obs 时打印摘要。
    """
    _check_isaac_cuda()

    from rsl_rl_isrc.env.isaac_gym.make_g1_isaac import (
        default_g1_log_dir,
        make_g1_isaac_env,
    )
    from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner

    env, _, train_cfg = make_g1_isaac_env(
        num_envs=num_envs,
        headless=headless,
        sim_device=device,
        task="single_leg",
    )

    train_cfg["runner"]["max_iterations"] = max_iterations
    train_cfg["runner"]["resume"] = resume
    train_cfg["runner"]["load_run"] = load_run
    train_cfg["runner"]["checkpoint"] = checkpoint

    if log_dir is None:
        log_dir = default_g1_log_dir(train_cfg, log_root=log_root)

    runner = G1OnPolicyTestRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=device,
        enable_obs_server=enable_obs_server,
        obs_pull_port=obs_pull_port,
        ctrl_rep_port=ctrl_rep_port,
        print_obs=print_obs,
    )

    if enable_obs_server and runner.obs_server is not None:
        instr = runner.get_instruction()
        print(
            f"ObsInstrServer（单腿站立）: PULL tcp://*:{obs_pull_port}, "
            f"REP tcp://*:{ctrl_rep_port}, 默认 instruction={instr}"
        )

    if resume:
        import rsl_rl_isrc.env.isaac_gym.make_g1_isaac as _mig
        from rsl_rl_isrc.env.isaac_gym.legged.utils.helpers import get_load_path

        if log_root is not None:
            root = log_root
        elif log_dir is not None:
            root = os.path.dirname(log_dir)
        else:
            root = os.path.join(
                os.path.dirname(_mig.__file__),
                "logs",
                train_cfg["runner"]["experiment_name"],
            )
        resume_path = get_load_path(
            root,
            load_run=load_run,
            checkpoint=checkpoint,
        )
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path)

    print(
        f"开始单腿站立训练: num_envs={num_envs}, max_iterations={max_iterations}, "
        f"log_dir={log_dir}, device={device}"
    )
    runner.learn(max_iterations, init_at_random_ep_len=init_at_random_ep_len)


def _parse_args() -> argparse.Namespace:
    default_num_envs = int(os.environ.get("G1_NUM_ENVS", "4096"))
    parser = argparse.ArgumentParser(
        description="G1 单腿站立 PPO 训练（任意一脚支撑，G1OnPolicyTestRunner）"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=default_num_envs,
        help=f"并行环境数（默认 {default_num_envs}，或环境变量 G1_NUM_ENVS）",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="PPO 学习迭代次数",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard / checkpoint 目录；默认 logs/g1_single_leg/<timestamp>",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default=None,
        help="日志根目录（仅当未指定 --log-dir 时使用）",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="无头模式（默认开启）",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="关闭无头模式（显示 Isaac 窗口）",
    )
    parser.add_argument(
        "--no-random-init-ep-len",
        action="store_true",
        help="禁用首轮随机 episode 长度",
    )
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复")
    parser.add_argument(
        "--load-run",
        type=str,
        default="-1",
        help="恢复的运行目录名或绝对路径；-1 表示最近一次",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="checkpoint 编号；-1 表示最新 model_*.pt",
    )
    parser.add_argument(
        "--no-zmq-obs",
        action="store_true",
        help="不启动 ObsInstrServer（纯训练，无 ZMQ）",
    )
    parser.add_argument(
        "--print-obs",
        action="store_true",
        help="在 ObsInstrServer 收到 obs 时打印摘要",
    )
    parser.add_argument(
        "--obs-pull-port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_SINGLE_LEG_OBS_PULL_PORT", "15557")),
        help="ObsInstrServer PULL 端口（默认 15557，避免与行走任务 15555 冲突）",
    )
    parser.add_argument(
        "--ctrl-rep-port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_SINGLE_LEG_CTRL_REP_PORT", "15558")),
        help="ObsInstrServer REP 端口（默认 15558）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_run: str | int = args.load_run
    if load_run != "-1":
        try:
            load_run = int(load_run)
        except ValueError:
            pass

    run_g1_single_leg_training(
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        log_dir=args.log_dir,
        device=args.device,
        headless=args.headless,
        init_at_random_ep_len=not args.no_random_init_ep_len,
        resume=args.resume,
        load_run=load_run,
        checkpoint=args.checkpoint,
        log_root=args.log_root,
        enable_obs_server=not args.no_zmq_obs,
        obs_pull_port=args.obs_pull_port,
        ctrl_rep_port=args.ctrl_rep_port,
        print_obs=args.print_obs,
    )


if __name__ == "__main__":
    main()
