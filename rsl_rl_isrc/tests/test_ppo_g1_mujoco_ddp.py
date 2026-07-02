#!/usr/bin/env python3
"""G1 MuJoCo PPO 分布式训练（CPU 仿真 + GPU/CUDA 策略，多机多卡）。

训练流程（与 Isaac DDP 一致）：

- 各 rank 本地 MuJoCo rollout / 策略推理
- ``RolloutStorage.broadcast()`` 汇聚到 rank0
- 仅 rank0 执行 ``PPO.update()``
- 权重 ``dist.broadcast`` 回所有 rank

依赖：``pip install 'mujoco>=3.1.0'``；模型 ``robotmodel/g1_description/scene.xml``。

单机 2 卡 smoke::

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      rsl_rl_isrc/tests/test_ppo_g1_mujoco_ddp.py \
      --num-envs 16 --max-iterations 2 --no-zmq-obs

多机 2x2（rank0 节点 IP 假设 10.0.0.1）::

    # 节点0
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
      --master_addr=10.0.0.1 --master_port=29500 \
      rsl_rl_isrc/tests/test_ppo_g1_mujoco_ddp.py \
      --num-envs 16 --max-iterations 5 --obs-server-host 10.0.0.1 --no-zmq-obs

    # 节点1
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
      --master_addr=10.0.0.1 --master_port=29500 \
      rsl_rl_isrc/tests/test_ppo_g1_mujoco_ddp.py \
      --num-envs 16 --max-iterations 5 --obs-server-host 10.0.0.1 --no-zmq-obs

> ``--num-envs`` 为每个 rank 的本地并行环境数，全局吞吐约为 ``num_envs * world_size``。
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(script_dir)
project_root = os.path.dirname(package_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _check_policy_cuda(local_rank: int) -> None:
    import torch

    if not torch.cuda.is_available():
        print("错误: MuJoCo DDP 策略训练需要 CUDA GPU。", file=sys.stderr)
        sys.exit(1)
    if local_rank >= torch.cuda.device_count():
        print(
            f"错误: LOCAL_RANK={local_rank} 超出 CUDA 设备数量 "
            f"{torch.cuda.device_count()}",
            file=sys.stderr,
        )
        sys.exit(1)


def _resolve_policy_device(local_rank: int) -> str:
    _check_policy_cuda(local_rank)
    return f"cuda:{int(local_rank)}"


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


def run_g1_mujoco_ddp_training(
    num_envs: int,
    max_iterations: int,
    log_dir: str | None,
    sim_device: str,
    policy_device: str,
    init_at_random_ep_len: bool,
    resume: bool,
    load_run: str | int,
    checkpoint: int,
    log_root: str | None,
    enable_obs_server: bool,
    obs_pull_port: int,
    ctrl_rep_port: int,
    print_obs: bool,
    obs_server_host: str,
    rank: int,
    world_size: int,
    seed: int,
) -> None:
    from rsl_rl_isrc.utils.checkpoint import get_load_path
    from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner
    from rsl_rl_isrc.env.mujoco.g1_mujoco_config import g1_scene_xml_path
    from rsl_rl_isrc.env.mujoco.make_g1_mujoco import (
        default_g1_mujoco_log_dir,
        has_mujoco,
        make_g1_mujoco_env,
        mujoco_import_error,
    )
    from rsl_rl_isrc.utils.distributed import broadcast_log_dir

    if not has_mujoco():
        print(f"错误: {mujoco_import_error()}", file=sys.stderr)
        sys.exit(1)

    env, _, train_cfg = make_g1_mujoco_env(
        num_envs=num_envs,
        sim_device=sim_device,
        seed=int(seed) + int(rank),
    )
    train_cfg["runner"]["run_name"] = "robotmodel_mujoco_ddp"
    train_cfg["runner"]["max_iterations"] = max_iterations
    train_cfg["runner"]["resume"] = resume
    train_cfg["runner"]["load_run"] = load_run
    train_cfg["runner"]["checkpoint"] = checkpoint

    if log_dir is None and rank == 0:
        log_dir = default_g1_mujoco_log_dir(train_cfg, log_root=log_root)
    log_dir = broadcast_log_dir(log_dir, rank=rank)

    runner = G1OnPolicyTestRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=policy_device,
        enable_obs_server=enable_obs_server,
        obs_pull_port=obs_pull_port,
        ctrl_rep_port=ctrl_rep_port,
        obs_server_host=obs_server_host,
        print_obs=print_obs,
    )

    if rank == 0:
        print(
            f"[rank0] scene.xml={g1_scene_xml_path()} world_size={world_size} "
            f"total_envs={num_envs * world_size} sim_device={sim_device} "
            f"policy_device={policy_device} obs_server_host={obs_server_host}"
        )
        if enable_obs_server and runner.obs_server is not None:
            instr = runner.get_instruction()
            print(
                f"ObsInstrServer: PULL tcp://*:{obs_pull_port}, "
                f"REP tcp://*:{ctrl_rep_port}, instruction={instr}"
            )

    if resume and rank == 0:
        import rsl_rl_isrc.env.mujoco.make_g1_mujoco as _mm

        if log_root is not None:
            root = log_root
        elif log_dir is not None:
            root = os.path.dirname(log_dir)
        else:
            root = os.path.join(
                os.path.dirname(_mm.__file__),
                "logs",
                train_cfg["runner"]["experiment_name"],
            )
        resume_path = get_load_path(root, load_run=load_run, checkpoint=checkpoint)
        print(f"[rank0] Loading model from: {resume_path}")
        runner.load(resume_path)

    if rank == 0:
        print(
            f"[rank0] 开始训练: num_envs/rank={num_envs}, max_iterations={max_iterations}, "
            f"log_dir={log_dir}, sim_device={sim_device}, policy_device={policy_device}"
        )
    runner.learn(max_iterations, init_at_random_ep_len=init_at_random_ep_len)


def _parse_args() -> argparse.Namespace:
    default_num_envs = int(os.environ.get("G1_NUM_ENVS", "64"))
    parser = argparse.ArgumentParser(description="G1 MuJoCo PPO DDP 训练入口")
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
        "--no-random-init-ep-len",
        action="store_true",
        help="禁用首轮随机 episode 长度",
    )
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

    from rsl_rl_isrc.utils.distributed import cleanup_distributed, setup_distributed

    rank, world_size, local_rank = setup_distributed(backend=args.dist_backend)
    policy_device = _resolve_policy_device(local_rank)
    obs_server_host = _resolve_obs_server_host(args.obs_server_host, rank, world_size)

    load_run: str | int = args.load_run
    if load_run != "-1":
        try:
            load_run = int(load_run)
        except ValueError:
            pass

    try:
        run_g1_mujoco_ddp_training(
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            log_dir=args.log_dir,
            sim_device=args.sim_device,
            policy_device=policy_device,
            init_at_random_ep_len=not args.no_random_init_ep_len,
            resume=args.resume,
            load_run=load_run,
            checkpoint=args.checkpoint,
            log_root=args.log_root,
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
