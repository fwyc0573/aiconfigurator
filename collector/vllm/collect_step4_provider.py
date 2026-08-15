# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Step4-Pro-Latest provider operation collectors.

The grouped ``wo_a`` benchmark mirrors
``vllm/model_executor/models/step4pro.py`` at pinned commit
``607d1641ee3fec43653fca510d717725828890c2``:

``torch.einsum("ngi,gri->ngr", grouped, weight)``.
"""

from __future__ import annotations

__compat__ = "vllm>=0.19.0,<0.20.0"

LATEST_MODEL = "stepfun-ai/Step4-Pro-Latest"
GROUPED_WO_A_PROVIDER = "vllm_step4pro_torch_einsum"


def _walk_operations(operations):
    stack = list(reversed(operations))
    while stack:
        operation = stack.pop()
        yield operation
        for group_name in ("_group_b", "_group_a"):
            stack.extend(reversed(getattr(operation, group_name, ()) or ()))


def _grouped_gemm_structural_keys() -> list[tuple[str, int, int, int, str]]:
    from aiconfigurator.sdk import common, config, models
    from aiconfigurator.sdk.operations import GroupedGEMM

    model = models.get_model(
        LATEST_MODEL,
        config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.fp8_block,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_tp_size=1,
            moe_ep_size=1,
        ),
        backend_name="vllm",
    )

    keys = {
        (
            operation._provider,
            operation._groups,
            operation._n,
            operation._k,
            operation._quant_mode.name,
        )
        for operation in _walk_operations([*model.context_ops, *model.generation_ops])
        if isinstance(operation, GroupedGEMM)
    }
    if not keys:
        raise RuntimeError(f"Built {LATEST_MODEL!r} graph emitted no GroupedGEMM operations")
    if any(provider != GROUPED_WO_A_PROVIDER for provider, *_ in keys):
        raise RuntimeError(f"Built {LATEST_MODEL!r} graph emitted an unexpected grouped-GEMM provider: {sorted(keys)}")
    return sorted(keys)


def get_step4_grouped_gemm_test_cases() -> list[list[object]]:
    """Return token sweeps for grouped identities extracted from the AIC graph."""
    from collector.case_generator import get_step4_model_gemm_case_specs

    token_counts = sorted(
        {case.x for case in get_step4_model_gemm_case_specs(LATEST_MODEL, backend="vllm")},
        reverse=True,
    )
    return [
        [provider, groups, num_tokens, n, k, quant_mode]
        for provider, groups, n, k, quant_mode in _grouped_gemm_structural_keys()
        for num_tokens in token_counts
    ]


def run_step4_grouped_gemm(
    provider: str,
    groups: int,
    num_tokens: int,
    n: int,
    k: int,
    quant_mode: str,
    *,
    perf_filename: str,
    device="cuda:0",
) -> None:
    """Measure the exact grouped ``wo_a`` einsum used by pinned Step4-Pro."""
    import torch
    from vllm.version import __version__ as vllm_version

    from collector.helper import benchmark_with_power, log_perf

    if provider != GROUPED_WO_A_PROVIDER:
        raise ValueError(f"Unsupported Step4 grouped-GEMM provider: {provider!r}")
    if quant_mode != "bfloat16":
        raise ValueError(f"Unsupported Step4 grouped-GEMM quant mode: {quant_mode!r}")
    if min(groups, num_tokens, n, k) < 1:
        raise ValueError("Step4 grouped-GEMM dimensions and num_tokens must be positive")

    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    grouped = torch.randn((num_tokens, groups, k), dtype=torch.bfloat16, device=torch_device)
    weight = torch.randn((groups, n, k), dtype=torch.bfloat16, device=torch_device)

    def kernel_func():
        return torch.einsum("ngi,gri->ngr", grouped, weight)

    with benchmark_with_power(
        device=torch_device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "provider": provider,
                "groups": groups,
                "num_tokens": num_tokens,
                "n": n,
                "k": k,
                "quant_mode": quant_mode,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(torch_device),
        op_name="step4_grouped_gemm",
        kernel_source=provider,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
