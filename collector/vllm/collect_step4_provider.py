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
"""

from __future__ import annotations

__compat__ = "vllm>=0.19.0,<0.20.0"

LATEST_MODEL = "stepfun-ai/Step4-Pro-Latest"
GROUPED_WO_A_PROVIDER = "vllm_step4pro_torch_einsum"
FP32_ROUTER_PROVIDER = "vllm.optimus_matmul_fp32"
FULL_K_NORM_ROPE_PROVIDER = "vllm_step4pro_k_norm_rope"
SWA_QKV_NORM_ROPE_PROVIDER = "vllm_step4pro_qkv_norm_rope"


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


def get_step4_fp32_output_gemm_test_cases() -> list[list[object]]:
    """Return token sweeps for FP32 router identities from the AIC graph."""
    from collector.case_generator import get_step4_model_gemm_case_specs

    token_counts = sorted(
        {case.x for case in get_step4_model_gemm_case_specs(LATEST_MODEL, backend="vllm")},
        reverse=True,
    )
    return [
        [provider, num_tokens, n, k, weight_dtype, output_dtype]
        for provider, n, k, weight_dtype, output_dtype in _fp32_output_gemm_structural_keys()
        for num_tokens in token_counts
    ]


def get_step4_qkv_norm_rope_test_cases() -> list[list[object]]:
    """Return token sweeps for both pinned QKV preprocessing paths."""
    from collector.case_generator import get_step4_model_gemm_case_specs

    token_counts = sorted(
        {case.x for case in get_step4_model_gemm_case_specs(LATEST_MODEL, backend="vllm")},
        reverse=True,
    )
    return [
        [provider, num_tokens, normalized_tensors, q_heads, kv_heads, head_dim]
        for provider, normalized_tensors, q_heads, kv_heads, head_dim in (_qkv_norm_rope_structural_keys())
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
    probe_outputs = probe if isinstance(probe, tuple) else (probe,)
    actual_shapes = tuple(tuple(output.shape) for output in probe_outputs)
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            "Pinned vLLM QKV norm/RoPE path returned unexpected shapes: "
            f"expected={expected_shapes}, actual={actual_shapes}"
        )
    if any(output.dtype != torch.bfloat16 for output in probe_outputs):
        raise RuntimeError(
            "Pinned vLLM QKV norm/RoPE path returned a non-BF16 output: "
            f"{tuple(output.dtype for output in probe_outputs)}"
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
