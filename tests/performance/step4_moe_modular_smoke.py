#!/usr/bin/env python3
"""Run one exact-runtime Step4 modular FP8 MoE smoke case."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import vllm

from collector.vllm.collect_moe import run_moe_torch

MODEL_SHAPES = {
    "stepfun-ai/Step4-Pro-V3": {
        "hidden_size": 6144,
        "inter_size": 2048,
        "topk": 16,
        "num_experts": 1024,
    },
    "stepfun-ai/Step4-Pro-V4": {
        "hidden_size": 9216,
        "inter_size": 3584,
        "topk": 8,
        "num_experts": 384,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", choices=sorted(MODEL_SHAPES), required=True)
    parser.add_argument("--expert-parallel-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4096])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if vllm.__version__ != "0.19.0":
        raise RuntimeError(f"Expected exact vLLM 0.19.0, got {vllm.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Step4 MoE smoke")

    shape = MODEL_SHAPES[args.model_path]
    if shape["num_experts"] % args.expert_parallel_size != 0:
        raise ValueError(
            "num_experts must be divisible by expert_parallel_size: "
            f"{shape['num_experts']} % {args.expert_parallel_size} != 0"
        )
    if args.output.exists():
        raise FileExistsError(f"Refusing to append to existing smoke output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        "RUNTIME",
        vllm.__version__,
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_capability(0),
        torch.version.cuda,
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    run_moe_torch(
        "fp8",
        args.tokens,
        shape["hidden_size"],
        shape["inter_size"],
        shape["topk"],
        shape["num_experts"],
        1,
        args.expert_parallel_size,
        args.model_path,
        "power_law",
        1.2,
        perf_filename=str(args.output),
    )
    torch.cuda.synchronize()
    print(
        "PEAK",
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
        flush=True,
    )
    print("OUTPUT", args.output, args.output.exists(), args.output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
