"""Minimal eight-GPU NCCL health check for the Sophia B300 node."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    value = torch.tensor(float(rank), device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)

    expected = world_size * (world_size - 1) / 2
    observed = value.item()
    if observed != expected:
        raise RuntimeError(
            f"NCCL all-reduce mismatch on rank {rank}: "
            f"observed={observed}, expected={expected}"
        )

    print(
        f"NCCL PASS host={socket.gethostname()} rank={rank}/{world_size} "
        f"gpu={local_rank} all_reduce={observed}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
