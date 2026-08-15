"""Focused identity tests for Step4-Pro-Latest operation metadata."""

from __future__ import annotations

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


def test_grouped_gemm_preserves_grouped_identity_and_weight_memory():
    operation = ops.GroupedGEMM(
        "wo_a",
        1.0,
        1024,
        4096,
        common.GEMMQuantMode.bfloat16,
        groups=8,
        provider="vllm_step4pro_torch_einsum",
    )

    assert operation._groups == 8
    assert operation._n == 1024
    assert operation._k == 4096
    assert operation._provider == "vllm_step4pro_torch_einsum"
    assert operation.get_weights() == 8 * 1024 * 4096 * 2
    assert operation._persisted_key() == (
        "vllm_step4pro_torch_einsum",
        8,
        1024,
        4096,
        common.GEMMQuantMode.bfloat16,
    )


def test_fp32_output_gemm_preserves_weight_and_output_dtypes():
    operation = ops.FP32OutputGEMM("router", 1.0, 896, 7168)

    assert operation._n == 896
    assert operation._k == 7168
    assert operation._weight_dtype == "bfloat16"
    assert operation._output_dtype == "float32"
    assert operation._provider == "vllm.optimus_matmul_fp32"
    assert operation._persisted_key() == (
        "vllm.optimus_matmul_fp32",
        896,
        7168,
        "bfloat16",
        "float32",
    )


def test_attention_accepts_provider_kv_alias_and_page_size_metadata():
    context = ops.ContextAttention(
        "context_attention",
        1.0,
        64,
        1,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
        provider="optimus_fa4",
        kv_storage_alias=True,
        page_size=128,
    )
    generation = ops.GenerationAttention(
        "generation_attention",
        1.0,
        64,
        1,
        common.KVCacheQuantMode.bfloat16,
        provider="optimus_fa4",
        kv_storage_alias=True,
        page_size=128,
    )

    for operation in (context, generation):
        assert operation._provider == "optimus_fa4"
        assert operation._kv_storage_alias is True
        assert operation._page_size == 128
        assert operation._persisted_key()[0] == "optimus_fa4"


def test_qkv_norm_rope_exposes_normalized_tensors_and_provider():
    operation = ops.QKVNormRoPE(
        "qkv_norm_rope",
        1.0,
        normalized_tensors=("q", "k", "v"),
        provider="vllm_step4pro_qkv_norm_rope",
        q_heads=128,
        kv_heads=8,
        head_dim=128,
    )

    assert operation._normalized_tensors == ("q", "k", "v")
    assert operation._provider == "vllm_step4pro_qkv_norm_rope"
    assert operation._persisted_key() == (
        "vllm_step4pro_qkv_norm_rope",
        ("q", "k", "v"),
        128,
        8,
        128,
    )


def test_deepep_dispatch_and_combine_have_distinct_persisted_keys():
    common_args = (
        "deepep",
        1.0,
        3584,
        16,
        896,
        1,
        16,
        16,
        True,
    )
    dispatch = ops.MoEDispatch(
        *common_args,
        provider="vllm_deepep_high_throughput",
        operation="dispatch",
    )
    combine = ops.MoEDispatch(
        *common_args,
        provider="vllm_deepep_high_throughput",
        operation="combine",
    )

    assert dispatch._provider == "vllm_deepep_high_throughput"
    assert combine._provider == "vllm_deepep_high_throughput"
    assert dispatch._operation == "dispatch"
    assert combine._operation == "combine"
    assert dispatch._persisted_key() != combine._persisted_key()


def test_provider_specific_attention_rejects_generic_perf_database():
    """Optimus FA4 metadata must affect consumer selection."""
    operation = ops.GenerationAttention(
        "generation_attention",
        1.0,
        64,
        1,
        common.KVCacheQuantMode.bfloat16,
        head_size=512,
        provider="optimus_fa4",
        kv_storage_alias=True,
        page_size=128,
    )
    with pytest.raises(NotImplementedError, match="provider-specific"):
        operation.query(object(), batch_size=1, s=1)


def test_provider_specific_moe_and_deepep_reject_generic_perf_database():
    """Optimus MoE and vLLM DeepEP require their own measured consumers."""
    moe = ops.MoE(
        "experts",
        1.0,
        3584,
        3584,
        16,
        896,
        1,
        16,
        common.MoEQuantMode.fp8_block,
        "power_law_1.2",
        16,
        provider="optimus_fp8_moe",
    )
    dispatch = ops.MoEDispatch(
        "dispatch",
        1.0,
        3584,
        16,
        896,
        1,
        16,
        16,
        True,
        provider="vllm_deepep_high_throughput",
        operation="dispatch",
    )

    assert moe._provider == "optimus_fp8_moe"
    with pytest.raises(NotImplementedError, match="provider-specific"):
        moe.query(object(), x=1)
    with pytest.raises(NotImplementedError, match="provider-specific"):
        dispatch.query(object(), x=1)
