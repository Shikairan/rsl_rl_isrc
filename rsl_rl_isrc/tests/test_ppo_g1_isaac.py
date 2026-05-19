#!/usr/bin/env python3
"""Isaac Gym G1 + G1OnPolicyTestRunner 正式 PPO 训练（单进程，无 torch.distributed）。

``G1OnPolicyTestRunner`` 使用单次 ``learn(N)``（每 iter 前 ``sync_instr``），与 LSTM PPO 兼容。

export RSL_RL_ISRC_OBS_RELAY_URL='http://localhost:18888/post'

依赖：已安装 Isaac Gym、CUDA；物理模型为 ``robotmodel/g1_description/g1_12dof.urdf``；
仿真在 ``rsl_rl_isrc.env.isaac_gym.legged``。

运行::

    python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --num-envs 4096 --max-iterations 10000

短训验证::

    python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --num-envs 64 --max-iterations 5 --log-dir /tmp/g1_train_smoke

ObsInstrServer（默认开启，PULL 15555 / REP 15556）::

    python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --num-envs 128 --max-iterations 5 --print-obs

    # 另开终端：修改监控 env 切片
    python rsl_rl_isrc/tests/zmq_obs_ctrl_client.py --state 0 0 10 20

纯训练（不启动 ZMQ）::

    python rsl_rl_isrc/tests/test_ppo_g1_isaac.py --no-zmq-obs --num-envs 64 --max-iterations 5
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
    """不经过 ``rsl_rl_isrc.env`` 加载补丁（``env`` 包会先 ``import torch``）。"""
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
    # Isaac Gym 要求先 import isaacgym，再 import torch。
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


def run_g1_isaac_training(
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
    obs_pull_port: int = 15555,
    ctrl_rep_port: int = 15556,
    print_obs: bool = False,
) -> None:
    """使用 Isaac G1 环境与 G1OnPolicyTestRunner 进行 PPO 训练。"""
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
            f"ObsInstrServer: PULL tcp://*:{obs_pull_port}（obs 入）, "
            f"REP tcp://*:{ctrl_rep_port}（改 index）, "
            f"默认 instruction={instr}"
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
        f"开始训练: num_envs={num_envs}, max_iterations={max_iterations}, "
        f"log_dir={log_dir}, device={device}"
    )
    runner.learn(max_iterations, init_at_random_ep_len=init_at_random_ep_len)


def _parse_args() -> argparse.Namespace:
    default_num_envs = int(os.environ.get("G1_NUM_ENVS", "4096"))
    parser = argparse.ArgumentParser(
        description="G1 Isaac Gym PPO 训练（G1OnPolicyTestRunner + ObsInstrServer）"
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
        help="TensorBoard / checkpoint 目录；默认 logs/g1/<timestamp>_<run_name>",
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
        default=int(os.environ.get("RSL_RL_ISRC_OBS_PULL_PORT", "15555")),
        help="ObsInstrServer PULL 端口（训练进程 bind）",
    )
    parser.add_argument(
        "--ctrl-rep-port",
        type=int,
        default=int(os.environ.get("RSL_RL_ISRC_CTRL_REP_PORT", "15556")),
        help="ObsInstrServer REP 端口（外部发 instruction）",
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

    run_g1_isaac_training(
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
