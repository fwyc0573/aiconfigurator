from __future__ import annotations

import importlib.metadata
import os

import torch
import torch.distributed as dist

EXPECTED_WORLD_SIZE = 8
EXPECTED_CAPABILITY = (10, 3)
EXPECTED_VLLM_VERSION = "0.19.0.post20.dev26+gc820e5ae1"
NUM_NVL_BYTES = 1024 * 1024 * 1024


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"expected one-node world size {EXPECTED_WORLD_SIZE}, got {world_size}")

    os.environ["VLLM_ALL2ALL_BACKEND"] = "deepep_high_throughput"
    import deep_ep
    import vllm
    from vllm.v1.worker.worker_base import (
        _harmonize_nvshmem_env_for_deepep_optimus,
    )

    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise RuntimeError(f"unexpected vLLM version: {vllm.__version__}; expected {EXPECTED_VLLM_VERSION}")

    _harmonize_nvshmem_env_for_deepep_optimus()
    torch.accelerator.set_device_index(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    capability = torch.cuda.get_device_capability(device)
    if capability != EXPECTED_CAPABILITY:
        raise RuntimeError(f"expected B300 capability {EXPECTED_CAPABILITY}, got {capability}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
    )
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    buffer = None
    try:
        buffer = deep_ep.Buffer(
            group=cpu_group,
            num_nvl_bytes=NUM_NVL_BYTES,
            num_rdma_bytes=0,
            low_latency_mode=False,
            num_qps_per_rank=1,
            explicitly_destroy=True,
        )
        dist.barrier()
        print(
            "STEP4_DEEPEP_LOCAL_IPC=PASS",
            f"rank={rank}",
            f"local_rank={local_rank}",
            f"world_size={world_size}",
            f"capability={capability[0]}.{capability[1]}",
            f"vllm={vllm.__version__}",
            f"deep_ep={importlib.metadata.version('deep_ep')}",
            flush=True,
        )
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
