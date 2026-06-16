"""分布式训练辅助：进程组初始化、设备解析与 rank0 广播工具。"""

from __future__ import annotations

import os
from datetime import timedelta


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """初始化 torch.distributed 并绑定本地 CUDA 设备。"""
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """销毁进程组。"""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def resolve_sim_device(local_rank: int) -> str:
    """返回当前进程应使用的仿真设备字符串。"""
    import torch

    if torch.cuda.is_available():
        return f"cuda:{int(local_rank)}"
    return "cpu"


def is_rank0(rank: int | None = None) -> bool:
    """判断是否为 rank0。"""
    import torch.distributed as dist

    if rank is not None:
        return int(rank) == 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def broadcast_log_dir(log_dir: str | None, rank: int) -> str | None:
    """将 rank0 的 log_dir 广播到所有 rank。"""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return log_dir
    payload = [log_dir if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]
