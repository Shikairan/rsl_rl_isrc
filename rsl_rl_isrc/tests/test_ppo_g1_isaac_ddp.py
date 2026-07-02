#!/usr/bin/env python3
"""Isaac Gym G1 PPO 分布式训练（多机多卡，rank0 反向更新 + 参数广播）。

训练流程：

- 各 rank 本地 rollout / 推理
- ``RolloutStorage.broadcast()`` 汇聚到 rank0
- 仅 rank0 执行 ``PPO.update()``
- 权重 ``dist.broadcast`` 回所有 rank

本脚本默认同时启用 ObsInstrServer，可通过 ZMQ 观测 obs。

单机 2 卡 smoke::

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py \
      --num-envs 64 --max-iterations 5 --print-obs

多机 2x2（rank0 节点 IP 假设 10.0.0.1）::

    # 节点0
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
      --master_addr=10.0.0.1 --master_port=29500 \
      rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py \
      --num-envs 64 --max-iterations 5 --obs-server-host 10.0.0.1 --print-obs

    # 节点1（仅 node_rank 不同）
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
      --master_addr=10.0.0.1 --master_port=29500 \
      rsl_rl_isrc/tests/test_ppo_g1_isaac_ddp.py \
      --num-envs 64 --max-iterations 5 --obs-server-host 10.0.0.1 --print-obs
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(script_dir)
project_root = os.path.dirname(package_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _apply_numpy_isaac_compat() -> None:
    compat_path = os.path.join(package_root, "env", "isaac_gym", "numpy_compat.py")
    spec = importlib.util.spec_from_file_location(
        "_isaac_numpy_compat_g1_ddp", compat_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {compat_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_numpy_isaac_compat()


def _check_isaac_cuda(local_rank: int) -> None:
    import torch

    if not torch.cuda.is_available():
        print("错误: 需要 CUDA GPU。", file=sys.stderr)
        sys.exit(1)
    if local_rank >= torch.cuda.device_count():
        print(
            f"错误: LOCAL_RANK={local_rank} 超出 CUDA 设备数量 {torch.cuda.device_count()}",
            file=sys.stderr,
        )
        sys.exit(1)


def _bootstrap_isaac_before_torch() -> None:
    bootstrap_path = os.path.join(
        package_root, "env", "isaac_gym", "isaacgym_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_isaacgym_bootstrap_g1_ddp", bootstrap_path
    )
    if spec is None or spec.loader is None:
        print(f"错误: 无法加载: {bootstrap_path}", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.ensure_isaacgym_importable(apply_numpy_compat=_apply_numpy_isaac_compat)
    except (RuntimeError, ImportError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


def _resolve_obs_server_host(cli_host: str | None, rank: int, world_size: int) -> str:
    if cli_host:
        return cli_host
    env_host = os.environ.get("RSL_RL_ISRC_OBS_SERVER_HOST", "").strip()
    if env_host:
        return env_host
    if world_size > 1 and rank == 0:
        try:
            ip = socket.gethostbyname(socket.gethostname())
            print(f"[rank0] 未显式指定 --obs-server-host，自动推断为 {ip}")
            return ip
        except OSError:
            pass
    return "localhost"


def run_g1_isaac_training(
    num_envs: int,
    max_iterations: int,
    log_dir: str | None,
    device: str,
    headless: bool,
    init_at_random_ep_len: bool,
    resume: bool,
    load_run: str | int,
    checkpoint: int,
    log_root: str | None,
    checkpoint_root: str | None,
    checkpoint_dir: str | None,
    enable_obs_server: bool,
    obs_pull_port: int,
    ctrl_rep_port: int,
    print_obs: bool,
    obs_server_host: str,
    rank: int,
    world_size: int,
    seed: int,
) -> None:
    _check_isaac_cuda(local_rank=int(device.split(":")[-1]))

    from rsl_rl_isrc.env.isaac_gym.make_g1_isaac import (
        default_g1_run_dirs,
        make_g1_isaac_env,
    )
    from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner
    from rsl_rl_isrc.env.isaac_gym.legged.utils.helpers import set_seed
    from rsl_rl_isrc.utils.checkpoint import get_load_path
    from rsl_rl_isrc.utils.distributed import broadcast_log_dir
    from rsl_rl_isrc.utils.paths import checkpoint_root_default

    env, cfg, train_cfg = make_g1_isaac_env(
        num_envs=num_envs,
        headless=headless,
        sim_device=device,
    )
    train_cfg["runner"]["run_name"] = "robotmodel_urdf_ddp"
    train_cfg["runner"]["max_iterations"] = max_iterations
    train_cfg["runner"]["resume"] = resume
    train_cfg["runner"]["load_run"] = load_run
    train_cfg["runner"]["checkpoint"] = checkpoint
    set_seed(int(seed) + int(rank))

    if log_dir is None and rank == 0:
        log_dir, checkpoint_dir = default_g1_run_dirs(
            train_cfg,
            log_root=log_root,
            checkpoint_root=checkpoint_root,
        )
    elif rank == 0 and checkpoint_dir is None and log_dir is not None:
        from rsl_rl_isrc.env.isaac_gym.make_g1_isaac import default_g1_checkpoint_dir

        checkpoint_dir = default_g1_checkpoint_dir(
            train_cfg,
            checkpoint_root=checkpoint_root,
            run_suffix_value=os.path.basename(log_dir),
        )
    log_dir = broadcast_log_dir(log_dir, rank=rank)
    checkpoint_dir = broadcast_log_dir(checkpoint_dir, rank=rank)

    runner = G1OnPolicyTestRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
        enable_obs_server=enable_obs_server,
        obs_pull_port=obs_pull_port,
        ctrl_rep_port=ctrl_rep_port,
        obs_server_host=obs_server_host,
        print_obs=print_obs,
    )

    if rank == 0:
        print(
            f"[rank0] asset.file={cfg.asset.file} world_size={world_size} "
            f"total_envs={num_envs * world_size} obs_server_host={obs_server_host}"
        )
        if enable_obs_server and runner.obs_server is not None:
            instr = runner.get_instruction()
            print(
                f"ObsInstrServer: PULL tcp://*:{obs_pull_port}, "
                f"REP tcp://*:{ctrl_rep_port}, instruction={instr}"
            )

    if resume and rank == 0:
        if checkpoint_root is not None:
            root = os.path.join(checkpoint_root, train_cfg["runner"]["experiment_name"])
        elif checkpoint_dir is not None:
            root = os.path.dirname(checkpoint_dir)
        else:
            root = os.path.join(
                checkpoint_root_default(),
                train_cfg["runner"]["experiment_name"],
            )
        resume_path = get_load_path(root, load_run=load_run, checkpoint=checkpoint)
        print(f"[rank0] Loading model from: {resume_path}")
        runner.load(resume_path)

    if rank == 0:
        print(
            f"[rank0] 开始训练: num_envs/rank={num_envs}, max_iterations={max_iterations}, "
            f"log_dir={log_dir}, checkpoint_dir={checkpoint_dir}, device={device}"
        )
    runner.learn(max_iterations, init_at_random_ep_len=init_at_random_ep_len)


def _parse_args() -> argparse.Namespace:
    default_num_envs = int(os.environ.get("G1_NUM_ENVS", "4096"))
    parser = argparse.ArgumentParser(description="G1 Isaac Gym PPO DDP 训练入口")
    parser.add_argument("--num-envs", type=int, default=default_num_envs)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-root", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--no-random-init-ep-len", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-run", type=str, default="-1")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--no-zmq-obs", action="store_true")
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
    parser.add_argument("--obs-server-host", type=str, default=None)
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _bootstrap_isaac_before_torch()

    from rsl_rl_isrc.utils.distributed import (
        cleanup_distributed,
        resolve_sim_device,
        setup_distributed,
    )

    rank, world_size, local_rank = setup_distributed(backend=args.dist_backend)
    device = resolve_sim_device(local_rank)
    obs_server_host = _resolve_obs_server_host(args.obs_server_host, rank, world_size)

    load_run: str | int = args.load_run
    if load_run != "-1":
        try:
            load_run = int(load_run)
        except ValueError:
            pass

    try:
        run_g1_isaac_training(
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            log_dir=args.log_dir,
            device=device,
            headless=args.headless,
            init_at_random_ep_len=not args.no_random_init_ep_len,
            resume=args.resume,
            load_run=load_run,
            checkpoint=args.checkpoint,
            log_root=args.log_root,
            checkpoint_root=args.checkpoint_root,
            checkpoint_dir=args.checkpoint_dir,
            enable_obs_server=not args.no_zmq_obs,
            obs_pull_port=args.obs_pull_port,
            ctrl_rep_port=args.ctrl_rep_port,
            print_obs=args.print_obs,
            obs_server_host=obs_server_host,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
