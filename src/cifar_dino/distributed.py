"""Minimal torchrun/DDP helpers. Everything degrades to single-process when not launched by torchrun."""

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank); initializes NCCL/Gloo only under torchrun."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return rank, world, local


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t)
    return t


def barrier() -> None:
    if is_dist():
        dist.barrier()
