# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Step4-Pro-Latest provider operation collectors.

The grouped ``wo_a`` benchmark mirrors
``vllm/model_executor/models/step4pro.py`` at pinned commit
``607d1641ee3fec43653fca510d717725828890c2``:

``torch.einsum("ngi,gri->ngr", grouped, weight)``.

The QKV benchmark keeps the two pinned paths separate:

- Full MFA: ``Step4ProAttention._tail_rope`` for Q and
  ``OptimusRMSNorm`` followed by the same tail-RoPE method for K.
- Sliding GQA: ``fused_qknorm_rope_forward_impl`` for Q/K followed by
  ``Step4ProSlidingAttention._prepare_value_for_attention`` for V.

Both attention benchmarks invoke the pinned
``FlashAttentionImpl.do_kv_cache_update`` and ``FlashAttentionImpl.forward``
methods. The pinned FA utility dispatcher routes hd512 to Optimus FA4 and hd128
to the image's vLLM FlashAttention implementation.
"""

from __future__ import annotations

__compat__ = "vllm>=0.19.0,<0.20.0"

LATEST_MODEL = "stepfun-ai/Step4-Pro-Latest"
GROUPED_WO_A_PROVIDER = "vllm_step4pro_torch_einsum"
FP32_ROUTER_PROVIDER = "vllm.optimus_matmul_fp32"
FULL_K_NORM_ROPE_PROVIDER = "vllm_step4pro_k_norm_rope"
SWA_QKV_NORM_ROPE_PROVIDER = "vllm_step4pro_qkv_norm_rope"
FULL_ATTENTION_PROVIDER = "optimus_fa4"
SWA_ATTENTION_PROVIDER = "vllm_native_sliding_gqa"
OPTIMUS_MOE_PROVIDER = "optimus_fp8_moe"
OPTIMUS_MOE_ACTIVATION = "situ_glu"
STEP4_ATTENTION_MAX_KV_CACHE_BYTES = 128 * 1024**3


def _walk_operations(operations):
    stack = list(reversed(operations))
    while stack:
        operation = stack.pop()
        yield operation
        for group_name in ("_group_b", "_group_a"):
            stack.extend(reversed(getattr(operation, group_name, ()) or ()))


def _latest_model_operations():
    from aiconfigurator.sdk import common, config, models

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
    return [*model.context_ops, *model.generation_ops]


def _grouped_gemm_structural_keys() -> list[tuple[str, int, int, int, str]]:
    from aiconfigurator.sdk.operations import GroupedGEMM

    keys = {
        (
            operation._provider,
            operation._groups,
            operation._n,
            operation._k,
            operation._quant_mode.name,
        )
        for operation in _walk_operations(_latest_model_operations())
        if isinstance(operation, GroupedGEMM)
    }
    if not keys:
        raise RuntimeError(f"Built {LATEST_MODEL!r} graph emitted no GroupedGEMM operations")
    if any(provider != GROUPED_WO_A_PROVIDER for provider, *_ in keys):
        raise RuntimeError(f"Built {LATEST_MODEL!r} graph emitted an unexpected grouped-GEMM provider: {sorted(keys)}")
    return sorted(keys)


def _fp32_output_gemm_structural_keys() -> list[tuple[str, int, int, str, str]]:
    from aiconfigurator.sdk.operations import FP32OutputGEMM

    keys = {
        (
            operation._provider,
            operation._n,
            operation._k,
            operation._weight_dtype,
            operation._output_dtype,
        )
        for operation in _walk_operations(_latest_model_operations())
        if isinstance(operation, FP32OutputGEMM)
    }
    if not keys:
        raise RuntimeError(f"Built {LATEST_MODEL!r} graph emitted no FP32OutputGEMM operations")
    if any(provider != FP32_ROUTER_PROVIDER for provider, *_ in keys):
        raise RuntimeError(f"Built {LATEST_MODEL!r} graph emitted an unexpected FP32 router provider: {sorted(keys)}")
    return sorted(keys)


def _qkv_norm_rope_structural_keys() -> list[tuple[str, str, int, int, int]]:
    from aiconfigurator.sdk.operations import QKVNormRoPE

    keys = {
        (
            operation._provider,
            "+".join(operation._normalized_tensors),
            operation._q_heads,
            operation._kv_heads,
            operation._head_dim,
        )
        for operation in _walk_operations(_latest_model_operations())
        if isinstance(operation, QKVNormRoPE)
    }
    expected_keys = {
        (FULL_K_NORM_ROPE_PROVIDER, "k", 64, 1, 512),
        (SWA_QKV_NORM_ROPE_PROVIDER, "q+k+v", 128, 8, 128),
    }
    if keys != expected_keys:
        raise RuntimeError(
            f"Built {LATEST_MODEL!r} graph emitted unexpected QKV norm/RoPE structures: "
            f"expected={sorted(expected_keys)}, actual={sorted(keys)}"
        )
    return sorted(keys)


def _step4_optimus_moe_structural_keys() -> list[tuple[str, int, int, int, int, int, str, str, str]]:
    """Return the pinned expert-compute identity emitted by the AIC graph."""
    from aiconfigurator.sdk.operations import MoE

    keys = {
        (
            operation._provider,
            operation._hidden_size,
            operation._inter_size,
            operation._topk,
            operation._num_experts,
            operation._moe_tp_size,
            operation._quant_mode.name,
            operation._workload_distribution,
            operation._activation,
        )
        for operation in _walk_operations(_latest_model_operations())
        if isinstance(operation, MoE) and operation._provider is not None
    }
    expected_keys = {
        (
            OPTIMUS_MOE_PROVIDER,
            3584,
            3584,
            16,
            896,
            1,
            "fp8_block",
            "power_law_1.2",
            OPTIMUS_MOE_ACTIVATION,
        )
    }
    if keys != expected_keys:
        raise RuntimeError(
            f"Built {LATEST_MODEL!r} graph emitted unexpected Optimus MoE "
            f"structures: expected={sorted(expected_keys)}, actual={sorted(keys)}"
        )
    return sorted(keys)


def _step4_optimus_moe_scale_shapes(
    *,
    num_experts: int,
    moe_ep_size: int,
    hidden_size: int,
    inter_size: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return rank-local 128x128 block-scale shapes for w13 and w2."""
    if min(num_experts, moe_ep_size, hidden_size, inter_size) < 1:
        raise ValueError("Step4 Optimus MoE dimensions must be positive")
    if num_experts % moe_ep_size:
        raise ValueError("Step4 Optimus MoE requires num_experts divisible by moe_ep_size")
    if hidden_size % 128 or inter_size % 128:
        raise ValueError("Step4 Optimus MoE requires hidden/inter dimensions divisible by 128")

    local_num_experts = num_experts // moe_ep_size
    hidden_blocks = hidden_size // 128
    inter_blocks = inter_size // 128
    return (
        (local_num_experts, 2 * inter_blocks, hidden_blocks),
        (local_num_experts, hidden_blocks, inter_blocks),
    )


def _step4_optimus_rank0_workload(
    topk_weights,
    topk_ids,
    *,
    local_num_experts: int,
    num_experts: int,
):
    """Keep rank-0 work while preserving Optimus global expert IDs."""
    import torch

    if topk_weights.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("Step4 Optimus routing tensors must be rank 2")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("Step4 Optimus routing weights and IDs must have equal shapes")
    if not 0 < local_num_experts < num_experts:
        raise ValueError("Step4 Optimus rank-0 workload requires a strict local expert shard")

    local_mask = (topk_ids >= 0) & (topk_ids < local_num_experts)
    rank0_token_mask = local_mask.any(dim=1)
    if not torch.any(rank0_token_mask):
        raise RuntimeError("Step4 Optimus routing produced no rank-0 expert work")

    local_mask = local_mask[rank0_token_mask]
    local_weights = topk_weights[rank0_token_mask].to(torch.float32).clone()
    global_ids = topk_ids[rank0_token_mask].to(torch.int64).clone()
    local_weights[~local_mask] = 0.0
    global_ids[~local_mask] = num_experts - 1
    return local_weights.contiguous(), global_ids.to(torch.int32).contiguous()


def _step4_optimus_provider_path(
    *,
    local_num_tokens: int,
    contiguous_threshold: int,
) -> tuple[str, bool]:
    """Match the pinned Fp8MoEMethod branch and its supported execution mode."""
    if local_num_tokens < 1 or contiguous_threshold < 1:
        raise ValueError("Step4 Optimus provider-path dimensions must be positive")
    if local_num_tokens < contiguous_threshold:
        return "deepgemm_optimus_moe_masked_fp8", True
    return "deepgemm_optimus_moe_fp8", False


def _step4_optimus_hidden_states(
    shape: tuple[int, int],
    *,
    device,
):
    """Create deterministic BF16 inputs without using the default RNG."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    return torch.randn(
        shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )


def _step4_optimus_reserve_pinned_workspaces(
    optimus_module,
    *,
    torch_module,
    device,
    max_num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    max_local_num_experts: int,
) -> None:
    """Initialize and validate the pinned provider's process-global workspaces."""
    if min(max_num_tokens, hidden_size, inter_size, topk, max_local_num_experts) < 1:
        raise ValueError("Step4 Optimus workspace reservation dimensions must be positive")

    hidden_base = torch_module.empty((1,), dtype=torch_module.bfloat16, device=device)
    w1_base = torch_module.empty((1,), dtype=torch_module.float8_e4m3fn, device=device)
    topk_base = torch_module.empty((1,), dtype=torch_module.int32, device=device)
    reservation_hidden = torch_module.as_strided(
        hidden_base,
        (max_num_tokens, hidden_size),
        (0, 0),
    )
    reservation_w1 = torch_module.as_strided(
        w1_base,
        (max_local_num_experts, 2 * inter_size, hidden_size),
        (0, 0, 0),
    )
    reservation_topk = torch_module.as_strided(
        topk_base,
        (max_num_tokens, topk),
        (0, 0),
    )

    expected_sizes = optimus_module._get_ws_size(
        reservation_hidden,
        reservation_w1,
        reservation_topk,
    )
    workspaces = optimus_module._get_workspaces(
        reservation_hidden,
        reservation_w1,
        reservation_topk,
    )
    actual_sizes = tuple(workspace.buffer.numel() * workspace.buffer.element_size() for workspace in workspaces)
    if any(actual < expected for actual, expected in zip(actual_sizes, expected_sizes, strict=True)):
        raise RuntimeError(
            "Pinned Step4 Optimus workspaces were initialized below the full "
            f"collection requirement: expected={expected_sizes}, actual={actual_sizes}"
        )


def _step4_attention_structural_keys(
    phase: str,
) -> list[tuple[str, int, int, int, int, str, str, bool, int, int, int, str]]:
    from aiconfigurator.sdk.operations import ContextAttention, GenerationAttention

    if phase == "context":
        operation_type = ContextAttention
    elif phase == "generation":
        operation_type = GenerationAttention
    else:
        raise ValueError(f"Unsupported Step4 attention phase: {phase!r}")

    keys = set()
    for operation in _walk_operations(_latest_model_operations()):
        if not isinstance(operation, operation_type) or operation._provider is None:
            continue
        kv_cache_dtype = (
            operation._kvcache_quant_mode.name
            if isinstance(operation, ContextAttention)
            else operation._kv_cache_dtype.name
        )
        attn_dtype = operation._fmha_quant_mode.name if isinstance(operation, ContextAttention) else "bfloat16"
        keys.add(
            (
                operation._provider,
                operation._n,
                operation._n_kv,
                operation._head_size,
                operation._window_size,
                kv_cache_dtype,
                attn_dtype,
                operation._kv_storage_alias,
                operation._page_size,
                operation._physical_page_bytes,
                operation._kv_block_stride_bytes,
                operation._kv_cache_layout,
            )
        )

    expected_keys = {
        (
            FULL_ATTENTION_PROVIDER,
            64,
            1,
            512,
            0,
            "bfloat16",
            "bfloat16",
            True,
            128,
            524288,
            524288,
            "NHD",
        ),
        (
            SWA_ATTENTION_PROVIDER,
            128,
            8,
            128,
            512,
            "bfloat16",
            "bfloat16",
            False,
            128,
            524288,
            262144,
            "NHD",
        ),
    }
    if keys != expected_keys:
        raise RuntimeError(
            f"Built {LATEST_MODEL!r} graph emitted unexpected {phase} attention "
            f"structures: expected={sorted(expected_keys)}, actual={sorted(keys)}"
        )
    return sorted(keys)


def _step4_attention_materialized_block_range(
    *,
    query_tokens: int,
    total_context_tokens: int,
    window_size: int,
    page_size: int,
) -> tuple[int, int]:
    """Return the live logical block range used by one pinned invocation."""
    if min(query_tokens, total_context_tokens, page_size) < 1:
        raise ValueError("Step4 attention cache dimensions must be positive")
    if query_tokens > total_context_tokens:
        raise ValueError("Step4 query tokens cannot exceed total context")
    if window_size < 0:
        raise ValueError("Step4 attention window size cannot be negative")

    end_block = (total_context_tokens + page_size - 1) // page_size
    if window_size == 0:
        return 0, end_block

    computed_prefix = total_context_tokens - query_tokens
    skipped_tokens = max(0, computed_prefix - window_size + 1)
    first_materialized_block = min(end_block, skipped_tokens // page_size)
    return first_materialized_block, end_block


def _step4_attention_physical_cache_bytes(
    *,
    batch_size: int,
    query_tokens: int,
    total_context_tokens: int,
    window_size: int,
    page_size: int,
    physical_page_bytes: int,
) -> int:
    """Return physical cache bytes for the live block range and batch."""
    if min(batch_size, physical_page_bytes) < 1:
        raise ValueError("Step4 attention physical cache dimensions must be positive")
    first_block, end_block = _step4_attention_materialized_block_range(
        query_tokens=query_tokens,
        total_context_tokens=total_context_tokens,
        window_size=window_size,
        page_size=page_size,
    )
    return batch_size * (end_block - first_block) * physical_page_bytes


def _step4_attention_expected_cache_strides_bytes(
    *,
    num_blocks: int,
    kv_storage_alias: bool,
    physical_page_bytes: int,
) -> tuple[int, int]:
    """Return pinned block and K/V-plane strides in bytes."""
    if min(num_blocks, physical_page_bytes) < 1:
        raise ValueError("Step4 attention stride dimensions must be positive")
    if kv_storage_alias:
        return physical_page_bytes, 0
    if physical_page_bytes % 2:
        raise ValueError("Non-aliased Step4 physical pages must split evenly into K/V planes")
    block_stride_bytes = physical_page_bytes // 2
    return block_stride_bytes, num_blocks * block_stride_bytes


def _qkv_norm_rope_expected_output_shapes(
    provider: str,
    *,
    num_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[tuple[int, ...], ...]:
    if provider == FULL_K_NORM_ROPE_PROVIDER:
        return (
            (num_tokens, q_heads, head_dim),
            (num_tokens, kv_heads, head_dim),
        )
    if provider == SWA_QKV_NORM_ROPE_PROVIDER:
        return (
            (num_tokens, q_heads * head_dim),
            (num_tokens, kv_heads * head_dim),
            (num_tokens, kv_heads * head_dim),
        )
    raise ValueError(f"Unsupported Step4 QKV norm/RoPE provider: {provider!r}")


def _validate_qkv_norm_rope_probe(
    probe,
    *,
    expected_shapes: tuple[tuple[int, ...], ...],
    expected_dtype,
):
    """Fail fast when the pinned QKV provider returns invalid outputs."""
    probe_outputs = probe if isinstance(probe, tuple) else (probe,)
    actual_shapes = tuple(tuple(output.shape) for output in probe_outputs)
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            "Pinned vLLM QKV norm/RoPE path returned unexpected shapes: "
            f"expected={expected_shapes}, actual={actual_shapes}"
        )
    if any(output.dtype != expected_dtype for output in probe_outputs):
        raise RuntimeError(
            "Pinned vLLM QKV norm/RoPE path returned an unexpected dtype: "
            f"expected={expected_dtype}, "
            f"actual={tuple(output.dtype for output in probe_outputs)}"
        )
    if any(not bool(output.isfinite().all()) for output in probe_outputs):
        raise RuntimeError("Pinned vLLM QKV norm/RoPE path returned non-finite values")
    return probe_outputs


def get_step4_grouped_gemm_test_cases() -> list[list[object]]:
    """Return token sweeps for grouped identities extracted from the AIC graph."""
    from collector.case_generator import get_step4_provider_token_counts

    token_counts = get_step4_provider_token_counts(
        LATEST_MODEL,
        "step4_grouped_gemm",
        backend="vllm",
    )
    return [
        [provider, groups, num_tokens, n, k, quant_mode]
        for provider, groups, n, k, quant_mode in _grouped_gemm_structural_keys()
        for num_tokens in token_counts
    ]


def get_step4_fp32_output_gemm_test_cases() -> list[list[object]]:
    """Return token sweeps for FP32 router identities from the AIC graph."""
    from collector.case_generator import get_step4_provider_token_counts

    token_counts = get_step4_provider_token_counts(
        LATEST_MODEL,
        "step4_fp32_output_gemm",
        backend="vllm",
    )
    return [
        [provider, num_tokens, n, k, weight_dtype, output_dtype]
        for provider, n, k, weight_dtype, output_dtype in _fp32_output_gemm_structural_keys()
        for num_tokens in token_counts
    ]


def get_step4_qkv_norm_rope_test_cases() -> list[list[object]]:
    """Return token sweeps for both pinned QKV preprocessing paths."""
    from collector.case_generator import get_step4_provider_token_counts

    token_counts = get_step4_provider_token_counts(
        LATEST_MODEL,
        "step4_qkv_norm_rope",
        backend="vllm",
    )
    return [
        [provider, num_tokens, normalized_tensors, q_heads, kv_heads, head_dim]
        for provider, normalized_tensors, q_heads, kv_heads, head_dim in (_qkv_norm_rope_structural_keys())
        for num_tokens in token_counts
    ]


def get_step4_optimus_moe_test_cases() -> list[list[object]]:
    """Return exact pinned Optimus expert-compute workloads."""
    from collector.case_generator import get_step4_optimus_moe_workload_config

    config = get_step4_optimus_moe_workload_config(LATEST_MODEL)
    test_cases = []
    for (
        provider,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_dtype,
        _graph_distribution,
        activation,
    ) in _step4_optimus_moe_structural_keys():
        for moe_ep_size in config["expert_parallel_sizes"]:
            _step4_optimus_moe_scale_shapes(
                num_experts=num_experts,
                moe_ep_size=moe_ep_size,
                hidden_size=hidden_size,
                inter_size=inter_size,
            )
            for distribution_name, power_law_alpha in config["routing_distributions"]:
                distribution = distribution_name if distribution_name == "balanced" else f"power_law_{power_law_alpha}"
                for num_tokens in config["token_counts"]:
                    test_cases.append(
                        [
                            provider,
                            num_tokens,
                            hidden_size,
                            inter_size,
                            topk,
                            num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            moe_dtype,
                            distribution,
                            activation,
                        ]
                    )
    return test_cases


def get_step4_context_attention_test_cases() -> list[list[object]]:
    """Return exact requirements prefill/chunk workloads for both providers."""
    from collector.case_generator import get_step4_attention_workload_config

    config = get_step4_attention_workload_config(LATEST_MODEL)
    shared_workloads = config["context_workloads"]
    workloads_by_provider = config["context_workloads_by_provider"]
    structural_keys = _step4_attention_structural_keys("context")
    known_providers = {key[0] for key in structural_keys}
    unknown_providers = set(workloads_by_provider) - known_providers
    if unknown_providers:
        raise ValueError(
            f"Step4 provider-specific context workloads reference unknown providers: {sorted(unknown_providers)}"
        )

    test_cases: list[list[object]] = []
    for (
        provider,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size,
        kv_cache_dtype,
        attn_dtype,
        kv_storage_alias,
        page_size,
        physical_page_bytes,
        kv_block_stride_bytes,
        kv_cache_layout,
    ) in structural_keys:
        workloads = (
            *shared_workloads,
            *workloads_by_provider.get(provider, ()),
        )
        for batch_size, query_tokens, total_context_tokens in workloads:
            test_cases.append(
                [
                    provider,
                    batch_size,
                    query_tokens,
                    total_context_tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    window_size,
                    kv_cache_dtype,
                    attn_dtype,
                    kv_storage_alias,
                    page_size,
                    physical_page_bytes,
                    kv_block_stride_bytes,
                    kv_cache_layout,
                ]
            )
    return test_cases


def get_step4_generation_attention_test_cases() -> list[list[object]]:
    """Return decode workload grids bounded by one B300 cache allocation."""
    from collector.case_generator import get_step4_attention_workload_config

    config = get_step4_attention_workload_config(LATEST_MODEL)
    max_kv_cache_bytes = int(config["max_kv_cache_bytes"])
    if max_kv_cache_bytes != STEP4_ATTENTION_MAX_KV_CACHE_BYTES:
        raise RuntimeError(
            "Step4 attention cache cap drifted from the pinned Collector contract: "
            f"yaml={max_kv_cache_bytes}, expected={STEP4_ATTENTION_MAX_KV_CACHE_BYTES}"
        )

    test_cases = []
    for (
        provider,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size,
        kv_cache_dtype,
        attn_dtype,
        kv_storage_alias,
        page_size,
        physical_page_bytes,
        kv_block_stride_bytes,
        kv_cache_layout,
    ) in _step4_attention_structural_keys("generation"):
        for total_context_tokens in config["generation_context_tokens"]:
            for batch_size in config["generation_batch_sizes"]:
                cache_bytes = _step4_attention_physical_cache_bytes(
                    batch_size=batch_size,
                    query_tokens=1,
                    total_context_tokens=total_context_tokens,
                    window_size=window_size,
                    page_size=page_size,
                    physical_page_bytes=physical_page_bytes,
                )
                if cache_bytes > max_kv_cache_bytes:
                    continue
                test_cases.append(
                    [
                        provider,
                        batch_size,
                        total_context_tokens,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        window_size,
                        kv_cache_dtype,
                        attn_dtype,
                        kv_storage_alias,
                        page_size,
                        physical_page_bytes,
                        kv_block_stride_bytes,
                        kv_cache_layout,
                    ]
                )
    return test_cases


def _step4_attention_expected_output_shape(
    *,
    batch_size: int,
    query_tokens: int,
    num_heads: int,
    head_dim: int,
) -> tuple[int, int, int]:
    if min(batch_size, query_tokens, num_heads, head_dim) < 1:
        raise ValueError("Step4 attention output dimensions must be positive")
    return (batch_size * query_tokens, num_heads, head_dim)


def _run_step4_attention(
    *,
    phase: str,
    provider: str,
    batch_size: int,
    query_tokens: int,
    total_context_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_size: int,
    kv_cache_dtype: str,
    attn_dtype: str,
    kv_storage_alias: bool,
    page_size: int,
    physical_page_bytes: int,
    kv_block_stride_bytes: int,
    kv_cache_layout: str,
    perf_filename: str,
    device: str,
) -> None:
    """Run the pinned vLLM cache-update plus FlashAttention provider path."""
    import os
    from contextlib import ExitStack

    import torch
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.utils import get_kv_cache_layout
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
    from vllm.version import __version__ as vllm_version

    from collector.helper import benchmark_with_power, log_perf
    from collector.vllm.utils import (
        BatchSpec,
        MockAttentionLayer,
        create_common_attn_metadata,
        create_vllm_config,
    )

    if phase not in {"context", "generation"}:
        raise ValueError(f"Unsupported Step4 attention phase: {phase!r}")
    if (
        min(
            batch_size,
            query_tokens,
            total_context_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
        )
        < 1
    ):
        raise ValueError("Step4 attention dimensions must be positive")
    if query_tokens > total_context_tokens:
        raise ValueError("Step4 query_tokens cannot exceed total_context_tokens")

    structural_key = (
        provider,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size,
        kv_cache_dtype,
        attn_dtype,
        kv_storage_alias,
        page_size,
        physical_page_bytes,
        kv_block_stride_bytes,
        kv_cache_layout,
    )
    if structural_key not in _step4_attention_structural_keys(phase):
        raise ValueError(f"Unexpected Step4 {phase}-attention structure: {structural_key!r}")
    if kv_cache_dtype != "bfloat16" or attn_dtype != "bfloat16":
        raise ValueError("Pinned Step4 attention requires BF16 query and KV cache")
    if page_size != 128:
        raise ValueError("Pinned Step4 attention requires page_size=128")
    if physical_page_bytes != 524288:
        raise ValueError("Pinned Step4 hybrid allocator requires physical_page_bytes=524288")
    if kv_cache_layout != "NHD":
        raise ValueError("Pinned Step4 attention requires kv_cache_layout='NHD'")

    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    capability = torch.cuda.get_device_capability(torch_device)
    if capability != (10, 3):
        raise RuntimeError(f"Pinned Step4 attention collection requires B300 SM103, got {capability}")
    torch.manual_seed(42)

    batch_spec = BatchSpec(
        seq_lens=[total_context_tokens] * batch_size,
        query_lens=[query_tokens] * batch_size,
        name=f"step4_{phase}_{provider}",
    )
    first_materialized_block, end_block = _step4_attention_materialized_block_range(
        query_tokens=query_tokens,
        total_context_tokens=total_context_tokens,
        window_size=window_size,
        page_size=page_size,
    )
    blocks_per_sequence = end_block - first_materialized_block
    num_blocks = batch_size * blocks_per_sequence
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=total_context_tokens,
        block_size=page_size,
        num_gpu_blocks=num_blocks,
        max_num_seqs=batch_size,
        max_num_batched_tokens=batch_size * query_tokens,
        use_fp8_kv_cache=False,
        sliding_window=window_size if window_size > 0 else None,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    with ExitStack() as exit_stack:
        exit_stack.enter_context(set_current_vllm_config(vllm_config))
        common_attn_metadata = create_common_attn_metadata(
            batch_spec,
            page_size,
            torch_device,
            arange_block_indices=False,
        )
        common_attn_metadata.block_table_tensor.zero_()
        for sequence_index in range(batch_size):
            physical_block_start = sequence_index * blocks_per_sequence
            common_attn_metadata.block_table_tensor[
                sequence_index,
                first_materialized_block:end_block,
            ] = torch.arange(
                physical_block_start,
                physical_block_start + blocks_per_sequence,
                dtype=torch.int32,
                device=torch_device,
            )
            token_offsets = (
                torch.arange(
                    query_tokens,
                    dtype=torch.int64,
                    device=torch_device,
                )
                + total_context_tokens
                - query_tokens
            )
            block_indices = token_offsets // page_size
            intra_block_offsets = token_offsets % page_size
            start = int(common_attn_metadata.query_start_loc_cpu[sequence_index])
            end = int(common_attn_metadata.query_start_loc_cpu[sequence_index + 1])
            common_attn_metadata.slot_mapping[start:end] = (
                common_attn_metadata.block_table_tensor[sequence_index, block_indices] * page_size + intra_block_offsets
            )

        spec_kwargs = {
            "block_size": page_size,
            "num_kv_heads": num_kv_heads,
            "head_size": head_dim,
            "dtype": torch.bfloat16,
            "indexes_kv_by_block_stride": True,
        }
        if window_size == 0:
            kv_cache_spec = FullAttentionSpec(
                **spec_kwargs,
                page_size_padded=physical_page_bytes,
            )
        else:
            kv_cache_spec = SlidingWindowSpec(
                **spec_kwargs,
                sliding_window=window_size,
            )
        if kv_cache_spec.page_size_bytes != physical_page_bytes:
            raise RuntimeError(
                "Pinned Step4 KV specification produced an unexpected physical page: "
                f"expected={physical_page_bytes}, actual={kv_cache_spec.page_size_bytes}"
            )
        actual_kv_cache_layout = get_kv_cache_layout()
        if actual_kv_cache_layout != kv_cache_layout:
            raise RuntimeError(
                "Pinned Step4 KV cache layout mismatch: "
                f"expected={kv_cache_layout!r}, actual={actual_kv_cache_layout!r}"
            )
        builder_cls = FlashAttentionBackend.get_builder_cls()
        impl_cls = FlashAttentionBackend.get_impl_cls()
        builder = builder_cls(
            kv_cache_spec,
            ["step4_provider_attention"],
            vllm_config,
            torch_device,
        )
        if window_size > 0 and hasattr(builder, "aot_sliding_window"):
            builder.aot_sliding_window = (window_size - 1, 0)
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        sliding_window = vllm_config.model_config.get_sliding_window()
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=sliding_window,
            kv_cache_dtype="auto",
        )
        if impl.__class__.__module__ != "vllm.v1.attention.backends.flash_attn":
            raise RuntimeError(
                "Pinned Step4 attention did not instantiate FlashAttentionImpl: "
                f"{impl.__class__.__module__}.{impl.__class__.__name__}"
            )
        if impl.vllm_flash_attn_version != 4:
            raise RuntimeError(
                "Pinned B300 Step4 attention requires vLLM FlashAttention "
                f"version 4, got {impl.vllm_flash_attn_version}"
            )

        raw_cache_storage = torch.zeros(
            num_blocks * physical_page_bytes,
            dtype=torch.int8,
            device=torch_device,
        )
        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks,
            page_size,
            num_kv_heads,
            head_dim,
        )
        kv_cache = _reshape_attention_kv_cache(
            raw_cache_storage,
            kv_cache_spec,
            kv_cache_shape,
            FlashAttentionBackend.get_kv_cache_stride_order(False),
            block_dim=1,
        )
        expected_block_stride_bytes, expected_plane_stride_bytes = _step4_attention_expected_cache_strides_bytes(
            num_blocks=num_blocks,
            kv_storage_alias=kv_storage_alias,
            physical_page_bytes=physical_page_bytes,
        )
        actual_block_stride_bytes = kv_cache.stride(1) * kv_cache.element_size()
        actual_plane_stride_bytes = kv_cache.stride(0) * kv_cache.element_size()
        if kv_block_stride_bytes != expected_block_stride_bytes:
            raise RuntimeError(
                "Step4 case block stride disagrees with the pinned layout: "
                f"case={kv_block_stride_bytes}, expected={expected_block_stride_bytes}"
            )
        if (
            actual_block_stride_bytes != expected_block_stride_bytes
            or actual_plane_stride_bytes != expected_plane_stride_bytes
        ):
            raise RuntimeError(
                "Pinned vLLM Step4 cache view has unexpected strides: "
                f"expected={(expected_block_stride_bytes, expected_plane_stride_bytes)}, "
                f"actual={(actual_block_stride_bytes, actual_plane_stride_bytes)}"
            )
        actual_alias = kv_cache[0].data_ptr() == kv_cache[1].data_ptr()
        if actual_alias != kv_storage_alias:
            raise RuntimeError(
                f"Pinned vLLM Step4 K/V alias mismatch: expected={kv_storage_alias}, actual={actual_alias}"
            )

        num_query_tokens = batch_size * query_tokens
        query = torch.randn(
            (num_query_tokens, num_heads, head_dim),
            dtype=torch.bfloat16,
            device=torch_device,
        )
        key = torch.randn(
            (num_query_tokens, num_kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=torch_device,
        )
        value = (
            key
            if kv_storage_alias
            else torch.randn(
                (num_query_tokens, num_kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=torch_device,
            )
        )
        output = torch.empty_like(query)
        mock_layer = MockAttentionLayer(torch_device)

        def kernel_func():
            impl.do_kv_cache_update(
                mock_layer,
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
            )
            return impl.forward(
                mock_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
            )

        probe = kernel_func()
        expected_shape = _step4_attention_expected_output_shape(
            batch_size=batch_size,
            query_tokens=query_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        if tuple(probe.shape) != expected_shape:
            raise RuntimeError(
                "Pinned vLLM Step4 attention returned an unexpected shape: "
                f"expected={expected_shape}, actual={tuple(probe.shape)}"
            )
        if probe.dtype != torch.bfloat16:
            raise RuntimeError(
                "Pinned vLLM Step4 attention returned an unexpected dtype: "
                f"expected={torch.bfloat16}, actual={probe.dtype}"
            )
        del probe

        with benchmark_with_power(
            device=torch_device,
            kernel_func=kernel_func,
            num_warmups=3,
            num_runs=6,
            repeat_n=1,
        ) as results:
            pass

    row = {
        "provider": provider,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "window_size": window_size,
        "attn_dtype": attn_dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "kv_storage_alias": kv_storage_alias,
        "page_size": page_size,
        "physical_page_bytes": physical_page_bytes,
        "kv_block_stride_bytes": kv_block_stride_bytes,
        "kv_cache_layout": kv_cache_layout,
        "physical_blocks_per_sequence": blocks_per_sequence,
        "allocated_kv_cache_bytes": num_blocks * physical_page_bytes,
        "latency": results["latency_ms"],
    }
    if phase == "context":
        row["query_tokens"] = query_tokens
        row["total_context_tokens"] = total_context_tokens
    else:
        row["context_tokens"] = total_context_tokens

    log_perf(
        item_list=[row],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(torch_device),
        op_name=f"step4_{phase}_attention",
        kernel_source=provider,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def run_step4_context_attention(
    provider: str,
    batch_size: int,
    query_tokens: int,
    total_context_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_size: int,
    kv_cache_dtype: str,
    attn_dtype: str,
    kv_storage_alias: bool,
    page_size: int,
    physical_page_bytes: int,
    kv_block_stride_bytes: int,
    kv_cache_layout: str,
    *,
    perf_filename: str,
    device="cuda:0",
) -> None:
    _run_step4_attention(
        phase="context",
        provider=provider,
        batch_size=batch_size,
        query_tokens=query_tokens,
        total_context_tokens=total_context_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        window_size=window_size,
        kv_cache_dtype=kv_cache_dtype,
        attn_dtype=attn_dtype,
        kv_storage_alias=kv_storage_alias,
        page_size=page_size,
        physical_page_bytes=physical_page_bytes,
        kv_block_stride_bytes=kv_block_stride_bytes,
        kv_cache_layout=kv_cache_layout,
        perf_filename=perf_filename,
        device=device,
    )


def run_step4_generation_attention(
    provider: str,
    batch_size: int,
    context_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_size: int,
    kv_cache_dtype: str,
    attn_dtype: str,
    kv_storage_alias: bool,
    page_size: int,
    physical_page_bytes: int,
    kv_block_stride_bytes: int,
    kv_cache_layout: str,
    *,
    perf_filename: str,
    device="cuda:0",
) -> None:
    _run_step4_attention(
        phase="generation",
        provider=provider,
        batch_size=batch_size,
        query_tokens=1,
        total_context_tokens=context_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        window_size=window_size,
        kv_cache_dtype=kv_cache_dtype,
        attn_dtype=attn_dtype,
        kv_storage_alias=kv_storage_alias,
        page_size=page_size,
        physical_page_bytes=physical_page_bytes,
        kv_block_stride_bytes=kv_block_stride_bytes,
        kv_cache_layout=kv_cache_layout,
        perf_filename=perf_filename,
        device=device,
    )


def run_step4_optimus_moe(
    provider: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    moe_dtype: str,
    distribution: str,
    activation: str,
    *,
    perf_filename: str,
    device="cuda:0",
) -> None:
    """Measure the exact pinned vLLM Optimus custom-op implementation."""
    import torch
    import torch.nn.functional as F
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe import optimus_fp8_moe as optimus_module
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
    from vllm.version import __version__ as vllm_version

    from collector.case_generator import get_step4_optimus_moe_workload_config
    from collector.helper import (
        balanced_logits,
        benchmark_with_power,
        log_perf,
        power_law_logits_v3,
    )

    structural_key = (
        provider,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_dtype,
        "power_law_1.2",
        activation,
    )
    if structural_key not in _step4_optimus_moe_structural_keys():
        raise ValueError(f"Unexpected Step4 Optimus MoE structure: {structural_key!r}")
    workload_config = get_step4_optimus_moe_workload_config(LATEST_MODEL)
    if num_tokens not in workload_config["token_counts"]:
        raise ValueError(f"Unexpected Step4 Optimus MoE token count: {num_tokens}")
    if moe_ep_size not in workload_config["expert_parallel_sizes"]:
        raise ValueError(f"Unexpected Step4 Optimus MoE EP size: {moe_ep_size}")
    allowed_distributions = {
        name if name == "balanced" else f"power_law_{alpha}" for name, alpha in workload_config["routing_distributions"]
    }
    if distribution not in allowed_distributions:
        raise ValueError(f"Unexpected Step4 Optimus MoE distribution: {distribution!r}")
    if provider != OPTIMUS_MOE_PROVIDER:
        raise ValueError(f"Unsupported Step4 Optimus MoE provider: {provider!r}")
    if activation != OPTIMUS_MOE_ACTIVATION:
        raise ValueError(f"Unsupported Step4 Optimus MoE activation: {activation!r}")
    if moe_dtype != "fp8_block":
        raise ValueError(f"Unsupported Step4 Optimus MoE dtype: {moe_dtype!r}")

    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    capability = torch.cuda.get_device_capability(torch_device)
    if capability != (10, 3):
        raise RuntimeError(f"Pinned Step4 Optimus MoE collection requires B300 SM103, got {capability}")
    if not envs.VLLM_USE_OPTIMUS_MOE:
        raise RuntimeError("Pinned Step4 Optimus MoE requires VLLM_USE_OPTIMUS_MOE=1")
    contiguous_threshold = int(envs.VLLM_OPTIMUS_MOE_MIN_CONTG_SIZE)
    if contiguous_threshold != 6144:
        raise RuntimeError(
            f"Pinned Step4 Optimus MoE requires VLLM_OPTIMUS_MOE_MIN_CONTG_SIZE=6144, got {contiguous_threshold}"
        )

    local_num_experts, expert_map, _ = determine_expert_map(
        moe_ep_size,
        0,
        num_experts,
    )
    if expert_map is None:
        raise RuntimeError("Step4 Optimus MoE requires an EP expert map")
    expert_map = expert_map.to(device=torch_device)

    w13_scale_shape, w2_scale_shape = _step4_optimus_moe_scale_shapes(
        num_experts=num_experts,
        moe_ep_size=moe_ep_size,
        hidden_size=hidden_size,
        inter_size=inter_size,
    )
    fp8_dtype = torch.float8_e4m3fn
    w1 = torch.full(
        (local_num_experts, 2 * inter_size, hidden_size),
        0.03125,
        dtype=fp8_dtype,
        device=torch_device,
    )
    w2 = torch.full(
        (local_num_experts, hidden_size, inter_size),
        0.03125,
        dtype=fp8_dtype,
        device=torch_device,
    )
    w1_scale = torch.ones(
        w13_scale_shape,
        dtype=torch.float32,
        device=torch_device,
    )
    w2_scale = torch.ones(
        w2_scale_shape,
        dtype=torch.float32,
        device=torch_device,
    )

    max_ep_size = min(workload_config["expert_parallel_sizes"])
    if num_experts % max_ep_size:
        raise RuntimeError(
            "Step4 Optimus full-workspace reservation requires an integral "
            f"expert shard: num_experts={num_experts}, ep_size={max_ep_size}"
        )
    _step4_optimus_reserve_pinned_workspaces(
        optimus_module,
        torch_module=torch,
        device=torch_device,
        max_num_tokens=max(workload_config["token_counts"]),
        hidden_size=hidden_size,
        inter_size=inter_size,
        topk=topk,
        max_local_num_experts=num_experts // max_ep_size,
    )

    if distribution == "balanced":
        router_logits = balanced_logits(num_tokens, num_experts, topk).to(device=torch_device)
    else:
        power_law_alpha = float(distribution.removeprefix("power_law_"))
        router_logits = power_law_logits_v3(
            num_tokens,
            num_experts,
            topk,
            moe_ep_size,
            power_law_alpha,
        ).to(device=torch_device)
    routed_weights, routed_ids = torch.topk(router_logits, topk, dim=-1)
    routed_weights = F.softmax(routed_weights.float(), dim=-1)
    topk_weights, topk_ids = _step4_optimus_rank0_workload(
        routed_weights,
        routed_ids,
        local_num_experts=local_num_experts,
        num_experts=num_experts,
    )
    local_num_tokens = int(topk_ids.shape[0])
    hidden_states = _step4_optimus_hidden_states(
        (local_num_tokens, hidden_size),
        device=torch_device,
    )
    output_shape = (local_num_tokens, hidden_size)
    kernel_variant, use_cuda_graph = _step4_optimus_provider_path(
        local_num_tokens=local_num_tokens,
        contiguous_threshold=contiguous_threshold,
    )
    if not hasattr(torch.ops.vllm, kernel_variant):
        raise RuntimeError(f"Pinned vLLM Optimus op is not registered: torch.ops.vllm.{kernel_variant}")
    optimus_op = getattr(torch.ops.vllm, kernel_variant)

    def kernel_func():
        return optimus_op(
            a1=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SITU_GLU.value,
            local_num_experts=local_num_experts,
            global_num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            swigluoai_step_limit=0.0,
            inplace=False,
        )

    probe = kernel_func()
    if tuple(probe.shape) != output_shape or probe.dtype != torch.bfloat16:
        raise RuntimeError(
            "Pinned Step4 Optimus MoE returned an unexpected output: "
            f"expected_shape={output_shape}, actual_shape={tuple(probe.shape)}, "
            f"expected_dtype={torch.bfloat16}, actual_dtype={probe.dtype}"
        )
    if not torch.isfinite(probe).all():
        raise RuntimeError("Pinned Step4 Optimus MoE returned non-finite values")

    with benchmark_with_power(
        device=torch_device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
        allow_graph_fail=False,
        use_cuda_graph=use_cuda_graph,
    ) as results:
        pass
    if bool(results["used_cuda_graph"]) is not use_cuda_graph:
        raise RuntimeError(
            "Pinned Step4 Optimus execution mode mismatch: "
            f"kernel_variant={kernel_variant}, expected_cuda_graph={use_cuda_graph}, "
            f"actual_cuda_graph={results['used_cuda_graph']}"
        )
    execution_mode = "cuda_graph" if use_cuda_graph else "eager"

    log_perf(
        item_list=[
            {
                "provider": provider,
                "num_tokens": num_tokens,
                "local_num_tokens": local_num_tokens,
                "hidden_size": hidden_size,
                "inter_size": inter_size,
                "topk": topk,
                "num_experts": num_experts,
                "moe_tp_size": moe_tp_size,
                "moe_ep_size": moe_ep_size,
                "moe_dtype": moe_dtype,
                "distribution": distribution,
                "activation": activation,
                "kernel_variant": kernel_variant,
                "execution_mode": execution_mode,
                "used_cuda_graph": results["used_cuda_graph"],
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(torch_device),
        op_name="step4_optimus_moe",
        kernel_source=provider,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


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


def run_step4_qkv_norm_rope(
    provider: str,
    num_tokens: int,
    normalized_tensors: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    perf_filename: str,
    device="cuda:0",
) -> None:
    """Measure one exact pinned Step4-Pro QKV preprocessing path."""
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        _run_step4_qkv_norm_rope_in_context(
            provider,
            num_tokens,
            normalized_tensors,
            q_heads,
            kv_heads,
            head_dim,
            perf_filename=perf_filename,
            device=device,
        )


def _run_step4_qkv_norm_rope_in_context(
    provider: str,
    num_tokens: int,
    normalized_tensors: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    perf_filename: str,
    device="cuda:0",
) -> None:
    """Execute the QKV benchmark while the pinned vLLM config is active."""
    import torch
    from vllm.model_executor.layers.layernorm import OptimusRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.step3p5_util import (
        fused_qknorm_rope_forward_impl,
    )
    from vllm.model_executor.models.step4pro import (
        Step4ProAttention,
        Step4ProSlidingAttention,
    )
    from vllm.version import __version__ as vllm_version

    from collector.helper import benchmark_with_power, log_perf

    if min(num_tokens, q_heads, kv_heads, head_dim) < 1:
        raise ValueError("Step4 QKV norm/RoPE dimensions and num_tokens must be positive")

    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=torch_device)
    eps = 1e-5
    rope_theta = 10000.0

    if provider == FULL_K_NORM_ROPE_PROVIDER:
        expected = ("k", 64, 1, 512)
        actual = (normalized_tensors, q_heads, kv_heads, head_dim)
        if actual != expected:
            raise ValueError(f"Unexpected Full-MFA K norm/RoPE structure: expected={expected}, actual={actual}")

        attention = Step4ProAttention.__new__(Step4ProAttention)
        torch.nn.Module.__init__(attention)
        attention.rope_dim = 64
        attention.rotary_emb = get_rope(
            head_size=attention.rope_dim,
            max_position=num_tokens,
            is_neox_style=True,
            rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
            dtype=torch.bfloat16,
        ).to(torch_device)
        attention.k_norm = OptimusRMSNorm(
            head_dim,
            eps,
            zero_centered=False,
            dtype=torch.bfloat16,
        ).to(torch_device)
        key = torch.randn(
            (num_tokens, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=torch_device,
        )
        query = torch.randn(
            (num_tokens, q_heads, head_dim),
            dtype=torch.bfloat16,
            device=torch_device,
        )

        def kernel_func():
            normalized_key = attention.k_norm(key)
            return (
                attention._tail_rope(positions, query),
                attention._tail_rope(positions, normalized_key),
            )

    elif provider == SWA_QKV_NORM_ROPE_PROVIDER:
        expected = ("q+k+v", 128, 8, 128)
        actual = (normalized_tensors, q_heads, kv_heads, head_dim)
        if actual != expected:
            raise ValueError(f"Unexpected SWA QKV norm/RoPE structure: expected={expected}, actual={actual}")

        sliding_attention = Step4ProSlidingAttention.__new__(Step4ProSlidingAttention)
        torch.nn.Module.__init__(sliding_attention)
        sliding_attention.gqa_v_norm = True
        sliding_attention.num_kv_heads = kv_heads
        sliding_attention.head_dim = head_dim
        sliding_attention.v_norm = OptimusRMSNorm(
            head_dim,
            eps,
            zero_centered=False,
            dtype=torch.bfloat16,
        ).to(torch_device)

        rotary_emb = get_rope(
            head_size=head_dim,
            max_position=num_tokens,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": 1.0,
            },
            dtype=torch.bfloat16,
        ).to(torch_device)
        rope_cos, rope_sin = rotary_emb.cos_sin_cache.chunk(2, dim=-1)
        qkv_width = (q_heads + 2 * kv_heads) * head_dim
        qkv = torch.randn(
            (num_tokens, qkv_width),
            dtype=torch.bfloat16,
            device=torch_device,
        )
        qnorm_weight = torch.ones(
            head_dim,
            dtype=torch.bfloat16,
            device=torch_device,
        )
        knorm_weight = torch.ones(
            head_dim,
            dtype=torch.bfloat16,
            device=torch_device,
        )

        def kernel_func():
            query, key, value = fused_qknorm_rope_forward_impl(
                qkv,
                qnorm_weight,
                knorm_weight,
                rope_cos,
                rope_sin,
                positions,
                head_dim,
                q_heads,
                kv_heads,
                head_dim // 2,
                eps,
                norm_weight_bias=0.0,
            )
            value = sliding_attention._prepare_value_for_attention(key, value)
            return query, key, value

    else:
        raise ValueError(f"Unsupported Step4 QKV norm/RoPE provider: {provider!r}")

    expected_shapes = _qkv_norm_rope_expected_output_shapes(
        provider,
        num_tokens=num_tokens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    probe = kernel_func()
    probe_outputs = _validate_qkv_norm_rope_probe(
        probe,
        expected_shapes=expected_shapes,
        expected_dtype=torch.bfloat16,
    )
    del probe, probe_outputs

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
                "num_tokens": num_tokens,
                "normalized_tensors": normalized_tensors,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(torch_device),
        op_name="step4_qkv_norm_rope",
        kernel_source=provider,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def run_step4_fp32_output_gemm(
    provider: str,
    num_tokens: int,
    n: int,
    k: int,
    weight_dtype: str,
    output_dtype: str,
    *,
    perf_filename: str,
    device="cuda:0",
) -> None:
    """Measure the pinned vLLM Optimus BF16-input, FP32-output router op."""
    import torch
    from vllm.model_executor.models import step3p5_util as _step3p5_util  # noqa: F401
    from vllm.version import __version__ as vllm_version

    from collector.helper import benchmark_with_power, log_perf

    if provider != FP32_ROUTER_PROVIDER:
        raise ValueError(f"Unsupported Step4 FP32 router provider: {provider!r}")
    if weight_dtype != "bfloat16":
        raise ValueError(f"Unsupported Step4 FP32 router weight dtype: {weight_dtype!r}")
    if output_dtype != "float32":
        raise ValueError(f"Unsupported Step4 FP32 router output dtype: {output_dtype!r}")
    if min(num_tokens, n, k) < 1:
        raise ValueError("Step4 FP32 router dimensions and num_tokens must be positive")

    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    hidden_states = torch.randn((num_tokens, k), dtype=torch.bfloat16, device=torch_device)
    weight = torch.randn((n, k), dtype=torch.bfloat16, device=torch_device)

    def kernel_func():
        return torch.ops.vllm.optimus_matmul_fp32(hidden_states, weight)

    probe = kernel_func()
    if probe.shape != (num_tokens, n):
        raise RuntimeError(
            "Pinned vLLM FP32 router returned an unexpected shape: "
            f"expected={(num_tokens, n)}, actual={tuple(probe.shape)}"
        )
    if probe.dtype != torch.float32:
        raise RuntimeError(
            f"Pinned vLLM FP32 router returned an unexpected dtype: expected={torch.float32}, actual={probe.dtype}"
        )
    del probe

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
                "num_tokens": num_tokens,
                "n": n,
                "k": k,
                "weight_dtype": weight_dtype,
                "output_dtype": output_dtype,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(torch_device),
        op_name="step4_fp32_output_gemm",
        kernel_source=provider,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
