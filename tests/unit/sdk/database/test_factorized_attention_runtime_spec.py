# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import MISSING, FrozenInstanceError, fields, is_dataclass
from typing import Any

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import dsv4
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    GenerationDeepSeekV4AttentionModule,
)
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit

RUNTIME_SPEC_FIELDS = (
    "retention_mode",
    "compressed_history_selection",
    "projection_head_dim",
    "cache_projection_width",
    "cache_entry_width",
    "cache_projection_matrix_count",
    "cache_auxiliary_fp32_elements",
    "cache_auxiliary_ops_per_token",
    "window_size",
    "compression_ratio",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
)

DIRECT_INPUT = {
    "b": 2,
    "s": 256,
    "prefix": 0,
    "num_heads": 16,
    "native_heads": 128,
    "tp_size": 8,
    "hidden_size": 7168,
    "q_lora_rank": 1536,
    "o_lora_rank": 1024,
    "rope_head_dim": 64,
    "o_groups": 2,
    "kvcache_quant_mode": common.KVCacheQuantMode.fp8,
    "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
    "gemm_quant_mode": common.GEMMQuantMode.fp8_block,
}

EXPECTED_SOL_FULL = {
    "context": {
        0: (31.679578112, 31.679578112, 0.094961664),
        4: (48.788799488, 48.788799488, 0.131473408),
        128: (35.44645632, 35.44645632, 0.102434816),
    },
    "generation": {
        0: (0.12582912, 0.12582912, 0.058894336),
        4: (0.194693632, 0.194693632, 0.087147392),
        128: (0.140575744, 0.140575744, 0.066235392),
    },
}

EXPECTED_WEIGHTS = {
    0: 58_720_320.0,
    4: 90_591_296.0,
    128: 66_322_496.0,
}


def _runtime_spec(**overrides):
    spec_class = getattr(dsv4, "FactorizedAttentionRuntimeSpec", None)
    assert spec_class is not None, "dsv4.FactorizedAttentionRuntimeSpec must exist"
    values = {
        "retention_mode": "swa",
        "compressed_history_selection": "topk",
        "projection_head_dim": 512,
        "cache_projection_width": 512,
        "cache_entry_width": 512,
        "cache_projection_matrix_count": 5,
        "cache_auxiliary_fp32_elements": 4096,
        "cache_auxiliary_ops_per_token": 2048,
        "window_size": 128,
        "compression_ratio": 4,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 1024,
    }
    values.update(overrides)
    return spec_class(**values)


def _deepseek_runtime_spec(compression_ratio: int):
    if compression_ratio == 0:
        return _runtime_spec(
            compressed_history_selection="none",
            cache_projection_matrix_count=1,
            cache_auxiliary_fp32_elements=0,
            cache_auxiliary_ops_per_token=0,
            compression_ratio=0,
            index_n_heads=0,
            index_head_dim=0,
            index_topk=0,
        )
    if compression_ratio == 4:
        return _runtime_spec()
    if compression_ratio == 128:
        return _runtime_spec(
            compressed_history_selection="all",
            cache_projection_matrix_count=3,
            cache_auxiliary_fp32_elements=65_536,
            cache_auxiliary_ops_per_token=1024,
            compression_ratio=128,
            index_n_heads=0,
            index_head_dim=0,
            index_topk=0,
        )
    raise ValueError(f"Unsupported test compression ratio: {compression_ratio}")


def _direct_query_kwargs(runtime_spec) -> dict[str, Any]:
    return {**DIRECT_INPUT, "runtime_spec": runtime_spec}


def _attention_operation(operation_type, runtime_spec, *, cp_size: int = 1, scale_factor: float = 1.0):
    return operation_type(
        name="runtime_spec_attention",
        scale_factor=scale_factor,
        num_heads=DIRECT_INPUT["num_heads"],
        native_heads=DIRECT_INPUT["native_heads"],
        tp_size=DIRECT_INPUT["tp_size"],
        hidden_size=DIRECT_INPUT["hidden_size"],
        q_lora_rank=DIRECT_INPUT["q_lora_rank"],
        o_lora_rank=DIRECT_INPUT["o_lora_rank"],
        runtime_spec=runtime_spec,
        rope_head_dim=DIRECT_INPUT["rope_head_dim"],
        o_groups=DIRECT_INPUT["o_groups"],
        kvcache_quant_mode=DIRECT_INPUT["kvcache_quant_mode"],
        fmha_quant_mode=DIRECT_INPUT["fmha_quant_mode"],
        gemm_quant_mode=DIRECT_INPUT["gemm_quant_mode"],
        cp_size=cp_size,
    )


def _small_context_sol(database, runtime_spec, *, b: int = 1, s: int = 4, prefix: int = 0):
    return database.query_context_deepseek_v4_attention_module(
        b=b,
        s=s,
        prefix=prefix,
        num_heads=2,
        native_heads=2,
        tp_size=1,
        hidden_size=8,
        q_lora_rank=4,
        o_lora_rank=4,
        runtime_spec=runtime_spec,
        rope_head_dim=2,
        o_groups=1,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        database_mode=common.DatabaseMode.SOL_FULL,
    )


def _pair_test_spec(*, retention_mode: str, selection: str, ratio: int):
    return _runtime_spec(
        retention_mode=retention_mode,
        compressed_history_selection=selection,
        projection_head_dim=4,
        cache_projection_width=3,
        cache_entry_width=5,
        cache_projection_matrix_count=2,
        cache_auxiliary_fp32_elements=0,
        cache_auxiliary_ops_per_token=0,
        window_size=0 if retention_mode == "full" else 2,
        compression_ratio=ratio,
        index_n_heads=0,
        index_head_dim=0,
        index_topk=0,
    )


def _attention_math_ms(database, pair_count: int, *, num_heads: int = 2, projection_head_dim: int = 4) -> float:
    tc_flops = GEMM._get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.bfloat16)
    return 4 * num_heads * projection_head_dim * pair_count / tc_flops * 1000


def test_factorized_attention_runtime_spec_is_frozen_explicit_and_default_free():
    spec_class = getattr(dsv4, "FactorizedAttentionRuntimeSpec", None)
    assert spec_class is not None, "dsv4.FactorizedAttentionRuntimeSpec must exist"
    assert is_dataclass(spec_class)

    dataclass_fields = fields(spec_class)
    assert tuple(field.name for field in dataclass_fields) == RUNTIME_SPEC_FIELDS
    for dataclass_field in dataclass_fields:
        assert dataclass_field.default is MISSING
        assert dataclass_field.default_factory is MISSING

    spec = _deepseek_runtime_spec(4)
    with pytest.raises(FrozenInstanceError):
        spec.window_size = 256


@pytest.mark.parametrize(
    "callable_object",
    [
        ContextDeepSeekV4AttentionModule.__init__,
        GenerationDeepSeekV4AttentionModule.__init__,
        PerfDatabase.query_context_deepseek_v4_attention_module,
        PerfDatabase.query_generation_deepseek_v4_attention_module,
    ],
)
def test_runtime_spec_replaces_the_old_conflated_attention_arguments(callable_object):
    parameters = inspect.signature(callable_object).parameters

    assert "runtime_spec" in parameters
    assert parameters["runtime_spec"].default is inspect.Parameter.empty
    assert {
        "head_dim",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "window_size",
        "compress_ratio",
    }.isdisjoint(parameters)


@pytest.mark.parametrize(
    "field_name",
    [
        "projection_head_dim",
        "cache_projection_width",
        "cache_entry_width",
        "cache_projection_matrix_count",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1, True, 1.0])
def test_runtime_spec_requires_positive_integer_dimensions_and_matrix_count(field_name, invalid_value):
    with pytest.raises(ValueError, match=field_name):
        _runtime_spec(**{field_name: invalid_value})


@pytest.mark.parametrize(
    "field_name",
    ["cache_auxiliary_fp32_elements", "cache_auxiliary_ops_per_token"],
)
@pytest.mark.parametrize("invalid_value", [-1, True, 1.0])
def test_runtime_spec_requires_non_negative_integer_auxiliary_counts(field_name, invalid_value):
    with pytest.raises(ValueError, match=field_name):
        _runtime_spec(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        pytest.param("window_size", -1, id="negative-window"),
        pytest.param("window_size", True, id="boolean-window"),
        pytest.param("window_size", 1.0, id="float-window"),
        pytest.param("compression_ratio", -1, id="negative-ratio"),
        pytest.param("compression_ratio", True, id="boolean-ratio"),
        pytest.param("compression_ratio", 4.0, id="float-ratio"),
        pytest.param("index_n_heads", True, id="boolean-index-heads"),
        pytest.param("index_head_dim", 128.0, id="float-index-dimension"),
        pytest.param("index_topk", -1, id="negative-index-topk"),
    ],
)
def test_runtime_spec_rejects_malformed_retention_and_index_integers(field_name, invalid_value):
    with pytest.raises(ValueError, match=field_name):
        _runtime_spec(**{field_name: invalid_value})


@pytest.mark.parametrize("matrix_count", [1, 3, 5])
def test_runtime_spec_accepts_generic_positive_matrix_counts(matrix_count):
    spec = _runtime_spec(
        compressed_history_selection="all",
        cache_projection_matrix_count=matrix_count,
        compression_ratio=4,
        index_n_heads=0,
        index_head_dim=0,
        index_topk=0,
    )

    assert spec.cache_projection_matrix_count == matrix_count


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {
                "retention_mode": "full",
                "compressed_history_selection": "none",
                "window_size": 0,
                "compression_ratio": 0,
                "index_n_heads": 0,
                "index_head_dim": 0,
                "index_topk": 0,
            },
            id="full-none",
        ),
        pytest.param(
            {
                "retention_mode": "full",
                "compressed_history_selection": "all",
                "window_size": 0,
                "compression_ratio": 4,
                "index_n_heads": 0,
                "index_head_dim": 0,
                "index_topk": 0,
            },
            id="full-all",
        ),
        pytest.param({}, id="swa-topk"),
        pytest.param(
            {
                "compressed_history_selection": "none",
                "compression_ratio": 0,
                "index_n_heads": 0,
                "index_head_dim": 0,
                "index_topk": 0,
            },
            id="swa-none",
        ),
        pytest.param(
            {
                "compressed_history_selection": "all",
                "compression_ratio": 4,
                "index_n_heads": 0,
                "index_head_dim": 0,
                "index_topk": 0,
            },
            id="swa-all",
        ),
    ],
)
def test_runtime_spec_accepts_each_explicit_retention_and_selection_pair(overrides):
    spec = _runtime_spec(**overrides)

    assert spec.retention_mode in {"full", "swa"}
    assert spec.compressed_history_selection in {"none", "all", "topk"}


@pytest.mark.parametrize(
    ("overrides", "error_field"),
    [
        pytest.param({"retention_mode": "dense"}, "retention_mode", id="unknown-retention"),
        pytest.param(
            {"compressed_history_selection": "sample"},
            "compressed_history_selection",
            id="unknown-selection",
        ),
        pytest.param(
            {"compressed_history_selection": "none"},
            "compression_ratio",
            id="none-with-positive-ratio",
        ),
        pytest.param(
            {"compressed_history_selection": "all", "compression_ratio": 0},
            "compression_ratio",
            id="all-with-zero-ratio",
        ),
        pytest.param(
            {"compressed_history_selection": "topk", "compression_ratio": 0},
            "compression_ratio",
            id="topk-with-zero-ratio",
        ),
        pytest.param(
            {"retention_mode": "full"},
            "window_size",
            id="full-with-window",
        ),
        pytest.param(
            {"retention_mode": "swa", "window_size": 0},
            "window_size",
            id="swa-without-window",
        ),
        pytest.param({"index_n_heads": 0}, "index_n_heads", id="topk-without-index-heads"),
        pytest.param({"index_head_dim": 0}, "index_head_dim", id="topk-without-index-dimension"),
        pytest.param({"index_topk": 0}, "index_topk", id="topk-without-limit"),
        pytest.param(
            {
                "compressed_history_selection": "all",
                "index_n_heads": 1,
                "index_head_dim": 0,
                "index_topk": 0,
            },
            "index_n_heads",
            id="all-with-indexer",
        ),
        pytest.param(
            {
                "compressed_history_selection": "none",
                "compression_ratio": 0,
                "index_n_heads": 0,
                "index_head_dim": 128,
                "index_topk": 0,
            },
            "index_head_dim",
            id="none-with-indexer",
        ),
    ],
)
def test_runtime_spec_rejects_inconsistent_retention_selection_and_indexer_state(overrides, error_field):
    with pytest.raises(ValueError, match=error_field):
        _runtime_spec(**overrides)


def test_runtime_spec_constructor_rejects_every_missing_field():
    spec_class = getattr(dsv4, "FactorizedAttentionRuntimeSpec", None)
    assert spec_class is not None, "dsv4.FactorizedAttentionRuntimeSpec must exist"
    complete = {field_name: getattr(_deepseek_runtime_spec(4), field_name) for field_name in RUNTIME_SPEC_FIELDS}

    for missing_field in RUNTIME_SPEC_FIELDS:
        incomplete = dict(complete)
        incomplete.pop(missing_field)
        with pytest.raises(TypeError, match=missing_field):
            spec_class(**incomplete)


def test_attention_pair_semantics_cover_full_and_swa_with_and_without_compressed_history(comprehensive_perf_db):
    full_none = _small_context_sol(
        comprehensive_perf_db,
        _pair_test_spec(retention_mode="full", selection="none", ratio=0),
    )
    full_all = _small_context_sol(
        comprehensive_perf_db,
        _pair_test_spec(retention_mode="full", selection="all", ratio=2),
    )
    swa_none = _small_context_sol(
        comprehensive_perf_db,
        _pair_test_spec(retention_mode="swa", selection="none", ratio=0),
    )
    swa_all = _small_context_sol(
        comprehensive_perf_db,
        _pair_test_spec(retention_mode="swa", selection="all", ratio=2),
    )

    # For four causal queries: full=10 pairs, W=2 gives 7 pairs, and R=2 gives 4 compressed pairs.
    assert full_none[1] - full_all[1] == pytest.approx(_attention_math_ms(comprehensive_perf_db, 6))
    assert full_none[1] - swa_none[1] == pytest.approx(_attention_math_ms(comprehensive_perf_db, 3))
    assert swa_all[1] - swa_none[1] == pytest.approx(_attention_math_ms(comprehensive_perf_db, 4))
    assert swa_all[1] - full_all[1] == pytest.approx(_attention_math_ms(comprehensive_perf_db, 7))


def test_attention_sol_assigns_each_width_and_auxiliary_term_to_its_owned_formula(comprehensive_perf_db):
    spec = _runtime_spec(
        retention_mode="full",
        compressed_history_selection="none",
        projection_head_dim=13,
        cache_projection_width=17,
        cache_entry_width=19,
        cache_projection_matrix_count=4,
        cache_auxiliary_fp32_elements=23,
        cache_auxiliary_ops_per_token=29,
        window_size=0,
        compression_ratio=0,
        index_n_heads=0,
        index_head_dim=0,
        index_topk=0,
    )
    b, s, prefix = 2, 3, 0
    num_heads = 3
    hidden_size = 11
    q_lora_rank = 5
    o_lora_rank = 7
    rope_head_dim = 2
    o_groups = 2
    tokens = b * s
    attention_pairs = b * s * (s + 1) // 2

    actual = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
        b=b,
        s=s,
        prefix=prefix,
        num_heads=num_heads,
        native_heads=num_heads,
        tp_size=1,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        runtime_spec=spec,
        rope_head_dim=rope_head_dim,
        o_groups=o_groups,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        database_mode=common.DatabaseMode.SOL_FULL,
    )

    gemm_weight_elements = (
        hidden_size * q_lora_rank
        + q_lora_rank * num_heads * spec.projection_head_dim
        + spec.cache_projection_matrix_count * hidden_size * spec.cache_projection_width
        + o_groups * o_lora_rank * hidden_size
    )
    bfloat16_weight_elements = num_heads * spec.projection_head_dim * o_lora_rank
    fp32_weight_elements = num_heads + spec.cache_auxiliary_fp32_elements
    expected_weight_bytes = (
        gemm_weight_elements * common.GEMMQuantMode.fp8_block.value.memory
        + bfloat16_weight_elements * common.GEMMQuantMode.bfloat16.value.memory
        + fp32_weight_elements * 4
    )

    gemm_ops = (
        2 * tokens * hidden_size * q_lora_rank
        + 2 * tokens * q_lora_rank * num_heads * spec.projection_head_dim
        + 2 * tokens * hidden_size * spec.cache_projection_matrix_count * spec.cache_projection_width
        + 2 * tokens * o_groups * o_lora_rank * hidden_size
        + tokens * spec.cache_auxiliary_ops_per_token
    )
    bfloat16_ops = 2 * tokens * num_heads * spec.projection_head_dim * o_lora_rank
    attention_ops = 4 * num_heads * spec.projection_head_dim * attention_pairs
    expected_math = (
        gemm_ops / GEMM._get_quant_tc_flops(comprehensive_perf_db.system_spec, common.GEMMQuantMode.fp8_block)
        + bfloat16_ops / GEMM._get_quant_tc_flops(comprehensive_perf_db.system_spec, common.GEMMQuantMode.bfloat16)
        + attention_ops / GEMM._get_quant_tc_flops(comprehensive_perf_db.system_spec, common.FMHAQuantMode.bfloat16)
    ) * 1000

    activation_bytes = (
        tokens
        * (
            hidden_size
            + q_lora_rank
            + num_heads * spec.projection_head_dim
            + spec.cache_projection_width
            + o_groups * o_lora_rank
        )
        * common.GEMMQuantMode.fp8_block.value.memory
    )
    kv_cache_bytes = attention_pairs * spec.cache_entry_width * common.KVCacheQuantMode.fp8.value.memory
    rope_bytes = tokens * num_heads * rope_head_dim * common.FMHAQuantMode.bfloat16.value.memory
    expected_memory = (
        (expected_weight_bytes + activation_bytes + kv_cache_bytes + rope_bytes)
        / comprehensive_perf_db.system_spec["gpu"]["mem_bw"]
        * 1000
    )

    operation = ContextDeepSeekV4AttentionModule(
        name="width_ownership",
        scale_factor=1.0,
        num_heads=num_heads,
        native_heads=num_heads,
        tp_size=1,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        runtime_spec=spec,
        rope_head_dim=rope_head_dim,
        o_groups=o_groups,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        cp_size=1,
    )

    assert operation.get_weights() == pytest.approx(expected_weight_bytes)
    assert actual[1] == pytest.approx(expected_math)
    assert actual[2] == pytest.approx(expected_memory)
    assert actual[0] == pytest.approx(max(expected_math, expected_memory))


def test_cache_projection_matrix_count_applies_to_weights_and_compute_without_changing_kv_width(comprehensive_perf_db):
    base = _pair_test_spec(retention_mode="full", selection="none", ratio=0)
    expanded = _runtime_spec(
        **{
            **{field_name: getattr(base, field_name) for field_name in RUNTIME_SPEC_FIELDS},
            "cache_projection_matrix_count": 5,
        }
    )
    base_op = _attention_operation(ContextDeepSeekV4AttentionModule, base)
    expanded_op = _attention_operation(ContextDeepSeekV4AttentionModule, expanded)
    base_sol = _small_context_sol(comprehensive_perf_db, base)
    expanded_sol = _small_context_sol(comprehensive_perf_db, expanded)

    matrix_delta = expanded.cache_projection_matrix_count - base.cache_projection_matrix_count
    expected_weight_delta = (
        matrix_delta
        * DIRECT_INPUT["hidden_size"]
        * base.cache_projection_width
        * DIRECT_INPUT["gemm_quant_mode"].value.memory
    )
    expected_math_delta = (
        2
        * 4
        * 8
        * matrix_delta
        * base.cache_projection_width
        / GEMM._get_quant_tc_flops(comprehensive_perf_db.system_spec, common.GEMMQuantMode.fp8_block)
        * 1000
    )

    assert expanded_op.get_weights() - base_op.get_weights() == pytest.approx(expected_weight_delta)
    assert expanded_sol[1] - base_sol[1] == pytest.approx(expected_math_delta)


def test_cache_entry_width_changes_persisted_kv_memory_but_not_math_or_weights(comprehensive_perf_db):
    narrow = _pair_test_spec(retention_mode="full", selection="none", ratio=0)
    wide = _runtime_spec(
        **{
            **{field_name: getattr(narrow, field_name) for field_name in RUNTIME_SPEC_FIELDS},
            "cache_entry_width": narrow.cache_entry_width + 7,
        }
    )
    narrow_sol = _small_context_sol(comprehensive_perf_db, narrow)
    wide_sol = _small_context_sol(comprehensive_perf_db, wide)
    narrow_op = _attention_operation(ContextDeepSeekV4AttentionModule, narrow)
    wide_op = _attention_operation(ContextDeepSeekV4AttentionModule, wide)
    pair_count = 10
    expected_memory_delta = (
        pair_count
        * 7
        * common.KVCacheQuantMode.fp8.value.memory
        / comprehensive_perf_db.system_spec["gpu"]["mem_bw"]
        * 1000
    )

    assert wide_sol[1] == pytest.approx(narrow_sol[1])
    assert wide_sol[2] - narrow_sol[2] == pytest.approx(expected_memory_delta)
    assert wide_op.get_weights() == pytest.approx(narrow_op.get_weights())


def test_auxiliary_ops_are_added_once_and_auxiliary_fp32_elements_are_stored_once(comprehensive_perf_db):
    base = _pair_test_spec(retention_mode="full", selection="none", ratio=0)
    expanded = _runtime_spec(
        **{
            **{field_name: getattr(base, field_name) for field_name in RUNTIME_SPEC_FIELDS},
            "cache_auxiliary_fp32_elements": 31,
            "cache_auxiliary_ops_per_token": 37,
        }
    )
    base_sol = _small_context_sol(comprehensive_perf_db, base)
    expanded_sol = _small_context_sol(comprehensive_perf_db, expanded)
    base_op = _attention_operation(ContextDeepSeekV4AttentionModule, base)
    expanded_op = _attention_operation(ContextDeepSeekV4AttentionModule, expanded)
    tokens = 4
    expected_math_delta = (
        tokens * 37 / GEMM._get_quant_tc_flops(comprehensive_perf_db.system_spec, common.GEMMQuantMode.fp8_block) * 1000
    )

    assert expanded_sol[1] - base_sol[1] == pytest.approx(expected_math_delta)
    assert expanded_op.get_weights() - base_op.get_weights() == pytest.approx(31 * 4)


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("compression_ratio", [0, 4, 128])
def test_deepseek_v4_direct_sol_full_preserves_exact_pre_migration_values(
    comprehensive_perf_db,
    phase,
    compression_ratio,
):
    kwargs = _direct_query_kwargs(_deepseek_runtime_spec(compression_ratio))
    if phase == "context":
        actual = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **kwargs,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
    else:
        kwargs.pop("prefix")
        actual = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **kwargs,
            database_mode=common.DatabaseMode.SOL_FULL,
        )

    assert actual == pytest.approx(EXPECTED_SOL_FULL[phase][compression_ratio])
    assert actual[0] == max(actual[1], actual[2])


@pytest.mark.parametrize("operation_type", [ContextDeepSeekV4AttentionModule, GenerationDeepSeekV4AttentionModule])
@pytest.mark.parametrize("compression_ratio", [0, 4, 128])
def test_deepseek_v4_operation_weights_preserve_exact_pre_migration_values(operation_type, compression_ratio):
    operation = _attention_operation(operation_type, _deepseek_runtime_spec(compression_ratio))

    assert operation.get_weights() == pytest.approx(EXPECTED_WEIGHTS[compression_ratio])


class _RecordingCPDatabase:
    def __init__(self) -> None:
        self.nccl_calls: list[dict[str, Any]] = []

    def query_nccl(
        self,
        dtype: common.CommQuantMode,
        num_gpus: int,
        operation: str,
        message_size: int,
    ) -> float:
        self.nccl_calls.append(
            {
                "dtype": dtype.name,
                "num_gpus": num_gpus,
                "operation": operation,
                "message_size_elements": message_size,
            }
        )
        return message_size / 1_000_000.0


class _DeterministicCPContextAttention(ContextDeepSeekV4AttentionModule):
    def _module_base(self, database: Any, b: int, s: int, prefix: int) -> PerformanceResult:
        del database, b, prefix
        return PerformanceResult(10.0 + s / 1000.0, energy=0.0, source="baseline")

    @classmethod
    def _lookup_sparse_kernel(
        cls,
        database: Any,
        kernel: str,
        bs: int,
        isl: int,
        past_kv: int,
        tp_size: int,
        native_heads: int,
    ) -> float:
        del cls, database, kernel, bs, past_kv, tp_size, native_heads
        return isl**2 / 10_000.0

    @classmethod
    def _csa_topk_top_last(
        cls,
        database: Any,
        isl: int,
        step: int,
        native_heads: int,
        b: int,
    ) -> float:
        del cls, database, step, native_heads, b
        return isl**2 / 20_000.0


def _cp_operation(runtime_spec):
    return _attention_operation(_DeterministicCPContextAttention, runtime_spec, cp_size=8)


@pytest.mark.parametrize(
    ("compression_ratio", "expected_latency", "expected_messages"),
    [
        pytest.param(0, 10.163072, [131_072], id="swa-only"),
        pytest.param(4, 11.238272, [65_536, 65_536], id="topk"),
        pytest.param(128, 10.16512, [131_072, 2_048], id="all"),
    ],
)
def test_deepseek_v4_context_parallel_preserves_exact_message_order_widths_and_latency(
    compression_ratio,
    expected_latency,
    expected_messages,
):
    database = _RecordingCPDatabase()

    result = _cp_operation(_deepseek_runtime_spec(compression_ratio))._query_cp(
        database,
        b=DIRECT_INPUT["b"],
        isl=DIRECT_INPUT["s"],
        prefix=DIRECT_INPUT["prefix"],
    )

    assert float(result) == pytest.approx(expected_latency)
    assert result.source == "estimated"
    assert [call["message_size_elements"] for call in database.nccl_calls] == expected_messages
    assert all(call["dtype"] == "half" for call in database.nccl_calls)
    assert all(call["num_gpus"] == 8 for call in database.nccl_calls)
    assert all(call["operation"] == "all_gather" for call in database.nccl_calls)


def test_ratio_four_all_selection_keeps_full_compressed_history_without_enabling_indexer_collectives():
    spec = _runtime_spec(
        compressed_history_selection="all",
        projection_head_dim=640,
        cache_projection_width=512,
        cache_entry_width=512,
        cache_projection_matrix_count=2,
        cache_auxiliary_fp32_elements=2048,
        cache_auxiliary_ops_per_token=0,
        compression_ratio=4,
        index_n_heads=0,
        index_head_dim=0,
        index_topk=0,
    )
    database = _RecordingCPDatabase()

    _cp_operation(spec)._query_cp(database, b=2, isl=256, prefix=0)

    # SWA contributes its window first; all compressed history contributes floor(256 / 4) entries second.
    assert [call["message_size_elements"] for call in database.nccl_calls] == [131_072, 65_536]


def test_context_parallel_uses_index_width_for_indexer_and_cache_entry_width_for_both_kv_messages():
    spec = _runtime_spec(
        projection_head_dim=11,
        cache_projection_width=13,
        cache_entry_width=17,
        window_size=8,
        index_head_dim=19,
    )
    database = _RecordingCPDatabase()

    _cp_operation(spec)._query_cp(database, b=2, isl=32, prefix=0)

    assert [call["message_size_elements"] for call in database.nccl_calls] == [2 * 32 * 19, 2 * (32 // 4) * 17]
