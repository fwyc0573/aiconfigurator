"""Collect exact-rank NCCL message-size sweeps on rlaunch replicas."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def _build_message_element_sweep(*, world_size: int, min_message_bytes: int, max_message_bytes: int) -> list[int]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if min_message_bytes <= 0 or max_message_bytes <= 0:
        raise ValueError("message-byte bounds must be positive")
    if min_message_bytes > max_message_bytes:
        raise ValueError("min_message_bytes must not exceed max_message_bytes")

    sizes = {world_size}
    target_bytes = min_message_bytes
    while target_bytes <= max_message_bytes:
        target_elements = (target_bytes + 1) // 2
        aligned_elements = -(-target_elements // world_size) * world_size
        sizes.add(aligned_elements)
        target_bytes *= 2
    if target_bytes // 2 < max_message_bytes:
        target_elements = (max_message_bytes + 1) // 2
        sizes.add(-(-target_elements // world_size) * world_size)
    return sorted(sizes)


def _parse_ops(raw: str) -> list[str]:
    ops = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"all_reduce", "all_gather", "reduce_scatter"}
    if not ops or any(op not in allowed for op in ops):
        raise ValueError(f"ops must be a non-empty subset of {sorted(allowed)}")
    return ops


def _allocate_collective_buffers(
    *, op: str, message_elements: int, world_size: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if message_elements <= 0 or world_size <= 0:
        raise ValueError("message_elements and world_size must be positive")
    if op == "all_reduce":
        return torch.ones(message_elements, dtype=torch.float16, device=device), None
    if op not in {"all_gather", "reduce_scatter"}:
        raise ValueError(f"unsupported collective: {op}")
    if message_elements % world_size:
        raise ValueError("message_elements must be divisible by world_size")
    per_rank_elements = message_elements // world_size
    if op == "all_gather":
        return (
            torch.ones(per_rank_elements, dtype=torch.float16, device=device),
            torch.empty(message_elements, dtype=torch.float16, device=device),
        )
    return (
        torch.ones(message_elements, dtype=torch.float16, device=device),
        torch.empty(per_rank_elements, dtype=torch.float16, device=device),
    )


def _build_perf_row(
    *,
    op: str,
    world_size: int,
    message_elements: int,
    latency_ms: float,
    nccl_version: str,
    device_name: str,
) -> dict[str, object]:
    return {
        "framework": "NCCL",
        "version": nccl_version,
        "device": device_name,
        "op_name": op,
        "kernel_source": "NCCL",
        "nccl_dtype": "half",
        "num_gpus": world_size,
        "message_size": message_elements,
        "message_size_bytes": message_elements * 2,
        "latency": latency_ms,
    }


def _run_collective(op: str, input_tensor: torch.Tensor, output_tensor: torch.Tensor | None) -> None:
    if op == "all_reduce":
        dist.all_reduce(input_tensor)
        return
    if op == "all_gather":
        if output_tensor is None:
            raise ValueError("all_gather requires an output tensor")
        dist.all_gather_into_tensor(output_tensor, input_tensor)
        return
    if op == "reduce_scatter":
        if output_tensor is None:
            raise ValueError("reduce_scatter requires an output tensor")
        dist.reduce_scatter_tensor(output_tensor, input_tensor)
        return
    raise ValueError(f"unsupported collective: {op}")


def run_sweep(
    *,
    ops: list[str],
    min_message_bytes: int,
    max_message_bytes: int,
    warmup: int,
    repeats: int,
    samples: int,
    expected_world_size: int,
    expected_vllm_version: str,
    image_reference: str,
    image_manifest_digest: str,
) -> dict[str, object]:
    if warmup < 1 or repeats < 1 or samples < 1:
        raise ValueError("warmup, repeats, and samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NCCL sweep")
    import vllm

    if vllm.__version__ != expected_vllm_version:
        raise RuntimeError(f"expected vLLM {expected_vllm_version}, observed {vllm.__version__}")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != expected_world_size:
        raise RuntimeError(f"expected world_size={expected_world_size}, observed {world_size}")

    nccl_version_tuple = tuple(int(value) for value in torch.cuda.nccl.version())
    nccl_version = ".".join(str(value) for value in nccl_version_tuple)
    device_name = torch.cuda.get_device_name()
    message_elements = _build_message_element_sweep(
        world_size=world_size,
        min_message_bytes=min_message_bytes,
        max_message_bytes=max_message_bytes,
    )
    hostnames = [None] * world_size if rank == 0 else None
    dist.gather_object(socket.gethostname(), hostnames, dst=0)
    measurements: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for op in ops:
        for elements in message_elements:
            input_tensor, output_tensor = _allocate_collective_buffers(
                op=op,
                message_elements=elements,
                world_size=world_size,
                device=torch.device("cuda", local_rank),
            )
            for _ in range(warmup):
                dist.barrier()
                _run_collective(op, input_tensor, output_tensor)
            torch.cuda.synchronize()

            rank_stats: list[dict[str, float]] = []
            max_rank_samples: list[float] = []
            for _ in range(samples):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    _run_collective(op, input_tensor, output_tensor)
                end.record()
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end) / repeats)
                gathered = [None] * world_size if rank == 0 else None
                dist.gather_object(elapsed_ms, gathered, dst=0)
                if rank == 0:
                    rank_latencies = [float(value) for value in gathered]
                    sample_max = max(rank_latencies)
                    max_rank_samples.append(sample_max)
                    rank_stats.append(
                        {
                            "min": min(rank_latencies),
                            "median": statistics.median(rank_latencies),
                            "max": sample_max,
                        }
                    )

            if rank == 0:
                latency_ms = float(statistics.median(max_rank_samples))
                row = _build_perf_row(
                    op=op,
                    world_size=world_size,
                    message_elements=elements,
                    latency_ms=latency_ms,
                    nccl_version=nccl_version,
                    device_name=device_name,
                )
                rows.append(row)
                measurements.append(
                    {
                        "op": op,
                        "world_size": world_size,
                        "message_elements": elements,
                        "message_bytes": elements * 2,
                        "latency_ms_median_of_max_rank": latency_ms,
                        "max_rank_samples_ms": max_rank_samples,
                        "rank_latency_stats_ms": rank_stats,
                    }
                )
            del input_tensor, output_tensor
            torch.cuda.empty_cache()

    dist.barrier()
    payload = {
        "artifact_type": "p6_exact_rank_nccl_message_sweep",
        "backend": "nccl",
        "framework_backend": "vllm",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "nccl_version": nccl_version,
        "device": device_name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "world_size": world_size,
        "rank": rank,
        "hostnames": sorted(hostnames) if rank == 0 else [],
        "ops": ops,
        "dtype": "half",
        "min_message_bytes_requested": min_message_bytes,
        "max_message_bytes_requested": max_message_bytes,
        "message_element_count": len(message_elements),
        "warmup": warmup,
        "repeats": repeats,
        "samples": samples,
        "image_reference": image_reference,
        "image_manifest_digest": image_manifest_digest,
        "platform_nccl_environment_keys": sorted(key for key in os.environ if key.startswith("NCCL_")),
        "canonical_collection": False,
        "diagnostic_only": False,
        "rows": rows if rank == 0 else [],
        "measurements": measurements if rank == 0 else [],
    }
    dist.destroy_process_group()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", default="all_gather,reduce_scatter")
    parser.add_argument("--min-message-bytes", type=int, default=128)
    parser.add_argument("--max-message-bytes", type=int, default=1_073_741_824)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-vllm-version", default="0.19.0")
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--image-manifest-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_sweep(
        ops=_parse_ops(args.ops),
        min_message_bytes=args.min_message_bytes,
        max_message_bytes=args.max_message_bytes,
        warmup=args.warmup,
        repeats=args.repeats,
        samples=args.samples,
        expected_world_size=args.expected_world_size,
        expected_vllm_version=args.expected_vllm_version,
        image_reference=args.image_reference,
        image_manifest_digest=args.image_manifest_digest,
    )
    if payload["rank"] == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "world_size": payload["world_size"],
                    "rows": len(payload["rows"]),
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
