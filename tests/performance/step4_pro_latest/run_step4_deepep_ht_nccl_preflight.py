#!/usr/bin/env python3
"""Validate the NCCL transport required by Step4-Pro DeepEP HT."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class PreflightIdentity:
    """Validated multi-node NCCL process identity."""

    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    nccl_ib_hca: str


def validate_environment(environ: Mapping[str, str]) -> PreflightIdentity:
    """Require the exact EP16/EP32 topology and a platform-provided RDMA HCA."""
    required = (
        "NCCL_IB_HCA",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
    )
    missing = [name for name in required if name not in environ]
    if missing:
        raise RuntimeError(f"NCCL preflight environment is incomplete: {missing}")

    nccl_ib_hca = environ["NCCL_IB_HCA"].strip()
    if not nccl_ib_hca:
        raise RuntimeError("Platform NCCL_IB_HCA is empty")

    world_size = int(environ["WORLD_SIZE"])
    rank = int(environ["RANK"])
    local_rank = int(environ["LOCAL_RANK"])
    local_world_size = int(environ["LOCAL_WORLD_SIZE"])
    if world_size not in (16, 32):
        raise RuntimeError(f"NCCL preflight world size must be 16 or 32: {world_size}")
    if local_world_size != 8:
        raise RuntimeError(f"NCCL preflight local world size must be 8: {local_world_size}")
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            "NCCL preflight rank is outside the launcher topology: "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )

    return PreflightIdentity(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        local_world_size=local_world_size,
        nccl_ib_hca=nccl_ib_hca,
    )


def expected_rank_sum(world_size: int) -> int:
    """Return the sum of one-based ranks for an exact all-reduce assertion."""
    return world_size * (world_size + 1) // 2


def main() -> None:
    """Run one real rank-wide NCCL all-reduce and validate its numeric result."""
    identity = validate_environment(os.environ)

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(identity.local_rank)
    device = torch.device(f"cuda:{identity.local_rank}")
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 3):
        raise RuntimeError(f"NCCL preflight requires B300 SM103, got {capability}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=180),
    )
    try:
        values = torch.tensor(
            [identity.rank + 1, 1],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        rank_sum, participant_sum = (int(value) for value in values.tolist())
        expected_sum = expected_rank_sum(identity.world_size)
        if rank_sum != expected_sum or participant_sum != identity.world_size:
            raise RuntimeError(
                "NCCL preflight all-reduce returned the wrong values: "
                f"rank_sum={rank_sum}, expected_rank_sum={expected_sum}, "
                f"participant_sum={participant_sum}, "
                f"expected_participants={identity.world_size}"
            )
        dist.barrier()
        print(
            "STEP4_NCCL_PREFLIGHT_RANK=PASS",
            f"rank={identity.rank}",
            f"local_rank={identity.local_rank}",
            f"world_size={identity.world_size}",
            f"rank_sum={rank_sum}",
            f"participant_sum={participant_sum}",
            "capability=10.3",
            "hca_present=1",
            flush=True,
        )
        if identity.rank == 0:
            print(
                "STEP4_NCCL_PREFLIGHT_DISTRIBUTED=PASS",
                f"world_size={identity.world_size}",
                f"rank_sum={rank_sum}",
                f"participant_sum={participant_sum}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
