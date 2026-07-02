#!/usr/bin/env python3
"""Isaac Gym PPO training for G1/H1 scene validation."""

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
    compat_path = os.path.join(package_root, "env", "isaac_gym", "numpy_compat.py")
    spec = importlib.util.spec_from_file_location("_isaac_numpy_compat", compat_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load: {compat_path}")
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
        print("错误: 需要 CUDA GPU。", file=sys.stderr)
        sys.exit(1)


def run_training(
    robot: str,
    scene: str,
    num_envs: int,
    max_iterations: int,
    log_dir: str | None,
    log_root: str | None,
    device: str,
    headless: bool,
    enable_obs_server: bool,
    obs_pull_port: int,
    ctrl_rep_port: int,
    print_obs: bool,
    load_path: str | None,
    load_optimizer: bool,
) -> None:
    _check_isaac_cuda()

    from rsl_rl_isrc.env.isaac_gym.make_humanoid_isaac import (
        default_humanoid_log_dir,
        make_humanoid_isaac_env,
    )
    from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner

    env, _, train_cfg = make_humanoid_isaac_env(
        robot=robot,
        scene=scene,
        num_envs=num_envs,
        max_iterations=max_iterations,
        headless=headless,
        sim_device=device,
    )
    train_cfg["runner"]["max_iterations"] = max_iterations

    if log_dir is None:
        log_dir = default_humanoid_log_dir(train_cfg, log_root=log_root)

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
    if load_path is not None:
        runner.load(load_path, load_optimizer=load_optimizer)
        print(f"Loaded checkpoint: {load_path}")

    print(
        f"开始训练: robot={robot}, scene={scene}, num_envs={num_envs}, "
        f"max_iterations={max_iterations}, log_dir={log_dir}, device={device}"
    )
    runner.learn(max_iterations, init_at_random_ep_len=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1/H1 Isaac Gym PPO 场景训练验证"
    )
    parser.add_argument("--robot", choices=("g1_12dof", "h1_10dof"), required=True)
    parser.add_argument("--scene", choices=("flat", "slope", "collision"), required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--no-zmq-obs", action="store_true")
    parser.add_argument("--print-obs", action="store_true")
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="可选：先加载已有 checkpoint，再继续训练当前场景。",
    )
    parser.add_argument(
        "--load-optimizer",
        action="store_true",
        help="配合 --load-path 使用；默认只加载策略权重，不加载优化器状态。",
    )
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
    run_training(
        robot=args.robot,
        scene=args.scene,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        log_dir=args.log_dir,
        log_root=args.log_root,
        device=args.device,
        headless=args.headless,
        enable_obs_server=not args.no_zmq_obs,
        obs_pull_port=args.obs_pull_port,
        ctrl_rep_port=args.ctrl_rep_port,
        print_obs=args.print_obs,
        load_path=args.load_path,
        load_optimizer=args.load_optimizer,
    )


if __name__ == "__main__":
    main()
