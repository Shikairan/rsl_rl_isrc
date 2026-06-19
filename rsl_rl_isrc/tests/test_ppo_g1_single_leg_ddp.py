#!/usr/bin/env python3
"""G1 单腿站立任务 PPO 单机多卡（DDP）训练脚本（任意一脚支撑）。

训练流程
--------
- 各 rank 在独立 GPU 上本地 rollout / 推理（``G1SingleLegRobot``）
- ``RolloutStorage.broadcast()`` 汇聚到 rank0
- 仅 rank0 执行 ``PPO.update()``
- 权重 ``dist.broadcast`` 回所有 rank

与行走任务 DDP（``test_ppo_g1_isaac_ddp.py``）完全独立，互不干涉：
- 任务：``task="single_leg"``（``G1SingleLegCfg + G1SingleLegRobot``）
- 日志：``logs/g1_single_leg/<timestamp>_single_leg_ddp``
- ZMQ 端口：PULL **15559** / REP **15560**（与行走 DDP 15555/15556 隔离）

指定显卡（--gpu-ids）
---------------------
通过 ``--gpu-ids`` 控制参与训练的 GPU 编号（物理卡 ID），
``torchrun --nproc_per_node`` 须与 GPU 数量一致。

    # 使用第 0、2 号 GPU（2 卡）
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \\
      rsl_rl_isrc/tests/test_ppo_g1_single_leg_ddp.py \\
      --gpu-ids 0,2 --num-envs 1024 --max-iterations 10000

    # 使用第 1、2、3 号 GPU（3 卡）
    torchrun --standalone --nnodes=1 --nproc_per_node=3 \\
      rsl_rl_isrc/tests/test_ppo_g1_single_leg_ddp.py \\
      --gpu-ids 1,2,3 --num-envs 2048 --max-iterations 10000

    # 不指定 --gpu-ids：使用系统默认前 N 张卡（与 CUDA_VISIBLE_DEVICES 一致）
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
      rsl_rl_isrc/tests/test_ppo_g1_single_leg_ddp.py \\
      --num-envs 2048 --max-iterations 10000

单机 2 卡快速验证::

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \\
      rsl_rl_isrc/tests/test_ppo_g1_single_leg_ddp.py \\
      --gpu-ids 0,1 --num-envs 64 --max-iterations 5 --no-zmq-obs

从 checkpoint 恢复::

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \\
      rsl_rl_isrc/tests/test_ppo_g1_single_leg_ddp.py \\
      --gpu-ids 0,1 --num-envs 64 --max-iterations 10000 --resume

带 obs 监控（另开终端 ``zmq_obs_ctrl_client.py`` 连 15559/15560）::

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \\
      rsl_rl_isrc/tests/test_ppo_g1_single_leg_ddp.py \\
      --gpu-ids 0,1 --num-envs 128 --max-iterations 5 --print-obs
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


def _configure_visible_gpus() -> str | None:
    """从 sys.argv 预提取 ``--gpu-ids`` 并提前设置 ``CUDA_VISIBLE_DEVICES``。

    此函数必须在 ``_bootstrap_isaac_before_torch()`` **之前**调用，确保 CUDA 驱动
    在初始化时已看到正确的设备集合。

    优先级：
        ``--gpu-ids`` CLI 参数 > ``CUDA_VISIBLE_DEVICES`` 环境变量 > 不限制（使用所有卡）

    Returns:
        实际生效的 GPU ID 字符串（如 ``"0,2,3"``），若未指定则返回 None。
    """
    gpu_ids: str | None = None
    argv = sys.argv[1:]
    for i, arg in enumerate(argv[:-1]):
        if arg in ("--gpu-ids", "--gpu_ids"):
            gpu_ids = argv[i + 1].strip()
            break

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        ids = [g.strip() for g in gpu_ids.split(",") if g.strip()]
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(
                f"[rank0] CUDA_VISIBLE_DEVICES={gpu_ids}，"
                f"可用 {len(ids)} 张 GPU（物理 ID: {ids}）"
            )
        if world_size > 1 and len(ids) != world_size:
            if rank == 0:
                print(
                    f"[rank0] 警告: --gpu-ids 指定了 {len(ids)} 张卡，"
                    f"但 torchrun WORLD_SIZE={world_size}，"
                    f"建议 --nproc_per_node 与 GPU 数量一致。",
                    file=sys.stderr,
                )
        return gpu_ids

    # 未传 --gpu-ids，沿用环境变量（若有）
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def _apply_numpy_isaac_compat() -> None:
    compat_path = os.path.join(package_root, "env", "isaac_gym", "numpy_compat.py")
    spec = importlib.util.spec_from_file_location(
        "_isaac_numpy_compat_g1_single_leg_ddp", compat_path
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
    """在 torch.distributed 初始化之前确保 isaacgym 已正确导入。"""
    bootstrap_path = os.path.join(
        package_root, "env", "isaac_gym", "isaacgym_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_isaacgym_bootstrap_g1_single_leg_ddp", bootstrap_path
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
    """推断 ObsInstrServer 监听地址（rank0）。

    优先级：CLI --obs-server-host > 环境变量 > 自动推断 IP（多卡时）> localhost
    """
    if cli_host:
        return cli_host
    env_host = os.environ.get("RSL_RL_ISRC_SINGLE_LEG_OBS_SERVER_HOST", "").strip()
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


def run_g1_single_leg_ddp_training(
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
    enable_obs_server: bool,
    obs_pull_port: int,
    ctrl_rep_port: int,
    print_obs: bool,
    obs_server_host: str,
    rank: int,
    world_size: int,
    seed: int,
    gpu_ids: str | None = None,
) -> None:
    """在单机多 GPU 上训练 G1 单腿站立任务（DDP）。

    Args:
        num_envs: 每张卡的并行环境数量（总环境数 = num_envs × world_size）。
        max_iterations: PPO 迭代次数。
        log_dir: 日志目录；None 时由 rank0 自动生成并广播。
        device: 当前进程使用的仿真设备（如 ``"cuda:0"``）。
        headless: 是否无头模式。
        init_at_random_ep_len: 首轮是否随机化 episode 长度。
        resume: 是否从 checkpoint 恢复。
        load_run: 恢复时的运行目录名或 ``-1``。
        checkpoint: 恢复时的 checkpoint 编号或 ``-1``。
        log_root: 日志根目录（仅 log_dir=None 时有效）。
        enable_obs_server: 是否启动 ZMQ ObsInstrServer（仅 rank0）。
        obs_pull_port: ObsInstrServer PULL 端口（默认 15559）。
        ctrl_rep_port: ObsInstrServer REP 端口（默认 15560）。
        print_obs: 是否打印 obs 摘要。
        obs_server_host: obs server 监听地址。
        rank: 当前进程 global rank。
        world_size: 总进程数。
        seed: 基础随机种子（各 rank 实际 seed = seed + rank）。
        gpu_ids: 生效的 CUDA_VISIBLE_DEVICES 字符串（如 ``"0,2,3"``），仅用于日志打印。
    """
    _check_isaac_cuda(local_rank=int(device.split(":")[-1]))

    from rsl_rl_isrc.env.isaac_gym.make_g1_isaac import (
        default_g1_log_dir,
        make_g1_isaac_env,
    )
    from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner
    from rsl_rl_isrc.env.isaac_gym.legged.utils.helpers import get_load_path, set_seed
    from rsl_rl_isrc.utils.distributed import broadcast_log_dir

    # 使用 task="single_leg" 接入单腿站立工厂
    env, cfg, train_cfg = make_g1_isaac_env(
        num_envs=num_envs,
        headless=headless,
        sim_device=device,
        task="single_leg",
    )

    train_cfg["runner"]["run_name"] = "single_leg_ddp"
    train_cfg["runner"]["max_iterations"] = max_iterations
    train_cfg["runner"]["resume"] = resume
    train_cfg["runner"]["load_run"] = load_run
    train_cfg["runner"]["checkpoint"] = checkpoint
    # 各 rank 使用不同随机种子以保证数据多样性
    set_seed(int(seed) + int(rank))

    if log_dir is None and rank == 0:
        log_dir = default_g1_log_dir(train_cfg, log_root=log_root)
    # rank0 生成目录后广播给所有 rank，确保日志路径一致
    log_dir = broadcast_log_dir(log_dir, rank=rank)

    runner = G1OnPolicyTestRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=device,
        enable_obs_server=enable_obs_server,
        obs_pull_port=obs_pull_port,
        ctrl_rep_port=ctrl_rep_port,
        obs_server_host=obs_server_host,
        print_obs=print_obs,
    )

    if rank == 0:
        import torch
        gpu_info = gpu_ids if gpu_ids else os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        phys_ids = [g.strip() for g in gpu_ids.split(",")] if gpu_ids else [
            str(i) for i in range(torch.cuda.device_count())
        ]
        print(
            f"[rank0] 单腿站立 DDP 训练:\n"
            f"  task=single_leg  asset={cfg.asset.file}\n"
            f"  CUDA_VISIBLE_DEVICES={gpu_info}\n"
            f"  GPU 分配: " + ", ".join(
                f"rank{i}→物理GPU{phys_ids[i] if i < len(phys_ids) else '?'}"
                f"(cuda:{i})"
                for i in range(world_size)
            ) + f"\n"
            f"  world_size={world_size}  每卡 envs={num_envs}  总 envs={num_envs * world_size}"
        )
        if enable_obs_server and runner.obs_server is not None:
            instr = runner.get_instruction()
            print(
                f"[rank0] ObsInstrServer（单腿站立）: "
                f"PULL tcp://*:{obs_pull_port}, "
                f"REP tcp://*:{ctrl_rep_port}, "
                f"instruction={instr}"
            )

    if resume and rank == 0:
        import rsl_rl_isrc.env.isaac_gym.make_g1_isaac as _mig

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
        resume_path = get_load_path(root, load_run=load_run, checkpoint=checkpoint)
        print(f"[rank0] Loading model from: {resume_path}")
        runner.load(resume_path)

    if rank == 0:
        print(
            f"[rank0] 开始训练: 每卡 num_envs={num_envs}, "
            f"max_iterations={max_iterations}, log_dir={log_dir}, device={device}"
        )
    runner.learn(max_iterations, init_at_random_ep_len=init_at_random_ep_len)


def _parse_args() -> argparse.Namespace:
    default_num_envs = int(os.environ.get("G1_NUM_ENVS", "2048"))
    parser = argparse.ArgumentParser(
        description="G1 单腿站立 PPO DDP 训练（单机多卡，任意一脚支撑）"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=default_num_envs,
        help=f"每张卡的并行环境数（默认 {default_num_envs}，或环境变量 G1_NUM_ENVS）",
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
        help="TensorBoard / checkpoint 目录；默认 logs/g1_single_leg/<timestamp>_single_leg_ddp",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default=None,
        help="日志根目录（仅当未指定 --log-dir 时使用）",
    )
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
        help="关闭无头模式（显示 Isaac 窗口，仅调试用）",
    )
    parser.add_argument(
        "--no-random-init-ep-len",
        action="store_true",
        help="禁用首轮随机 episode 长度",
    )
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复训练")
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
        default=int(
            os.environ.get("RSL_RL_ISRC_SINGLE_LEG_DDP_OBS_PULL_PORT", "15559")
        ),
        help="ObsInstrServer PULL 端口（默认 15559，单腿 DDP 专用，避免与其他任务冲突）",
    )
    parser.add_argument(
        "--ctrl-rep-port",
        type=int,
        default=int(
            os.environ.get("RSL_RL_ISRC_SINGLE_LEG_DDP_CTRL_REP_PORT", "15560")
        ),
        help="ObsInstrServer REP 端口（默认 15560）",
    )
    parser.add_argument(
        "--obs-server-host",
        type=str,
        default=None,
        help="ObsInstrServer 监听地址（rank0）；不指定则自动推断",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="torch.distributed 后端（默认 nccl；CPU 调试可改 gloo）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="基础随机种子（各 rank 实际 seed = seed + rank）",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        metavar="ID[,ID,...]",
        help=(
            "指定参与训练的 GPU 物理编号，逗号分隔（如 '0,2,3'）。"
            "脚本内部将其设为 CUDA_VISIBLE_DEVICES，"
            "LOCAL_RANK 0→第一个 ID，LOCAL_RANK 1→第二个 ID，以此类推。"
            "此参数须在任何 CUDA 初始化前生效，已在 main() 最开头预处理。"
            "torchrun --nproc_per_node 须与 GPU 数量一致。"
            "不指定时沿用系统 CUDA_VISIBLE_DEVICES 或使用所有可用 GPU。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    # ── 第一步：必须在任何 CUDA 初始化前设置可见 GPU ────────────────────────
    # _configure_visible_gpus() 从 sys.argv 中预提取 --gpu-ids 并写入
    # CUDA_VISIBLE_DEVICES，之后 isaacgym bootstrap 和 torch.distributed
    # 才能感知到正确的设备集合。
    gpu_ids = _configure_visible_gpus()

    # ── 第二步：isaacgym 必须在 torch.distributed 初始化之前导入 ────────────
    _bootstrap_isaac_before_torch()

    # ── 第三步：解析完整参数（--gpu-ids 已在上面预处理，此处正常解析） ─────
    args = _parse_args()

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
        run_g1_single_leg_ddp_training(
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
            enable_obs_server=not args.no_zmq_obs,
            obs_pull_port=args.obs_pull_port,
            ctrl_rep_port=args.ctrl_rep_port,
            print_obs=args.print_obs,
            obs_server_host=obs_server_host,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            gpu_ids=gpu_ids,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
