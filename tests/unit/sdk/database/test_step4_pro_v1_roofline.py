# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Formula-only roofline contracts for the cached Step4-Pro-V1 model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config, models
from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention
from aiconfigurator.sdk.operations.communication import NCCL, CustomAllReduce
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    GenerationDeepSeekV4AttentionModule,
)
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.operations.moe import MoE, MoEDispatch
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

MODEL_ID = "stepfun-ai/Step4-Pro-V1"


def _task_kwargs(serving_mode: str, database_mode: str | None) -> dict:
    """Return one exact aggregate or disaggregate Task request."""
    if serving_mode == "agg":
        return {
            "serving_mode": "agg",
            "model_path": MODEL_ID,
            "system_name": "h200_sxm",
            "backend_name": "vllm",
            "backend_version": "0.22.0",
            "database_mode": database_mode,
        }
    return {
        "serving_mode": "disagg",
        "prefill_model_path": MODEL_ID,
        "prefill_system_name": "h200_sxm",
        "prefill_backend_name": "vllm",
        "prefill_backend_version": "0.22.0",
        "decode_model_path": MODEL_ID,
        "decode_system_name": "h200_sxm",
        "decode_backend_name": "vllm",
        "decode_backend_version": "0.22.0",
        "database_mode": database_mode,
    }


def _build_model(*, tp_size: int, attention_dp_size: int, moe_tp_size: int, moe_ep_size: int):
    """Build the cached model for one valid Step4 parallel pattern."""
    model_config = config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        nextn=0,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )
    return models.get_model(MODEL_ID, model_config, backend_name="vllm")


def _query_operation_tree(operation, database, query_kwargs) -> list[tuple[str, PerformanceResult]]:
    """Query one operation and every nested overlap child."""
    results = []
    if isinstance(operation, ops.OverlapOp):
        for child in operation._group_a:
            results.extend(_query_operation_tree(child, database, query_kwargs))
        for child in operation._group_b:
            results.extend(_query_operation_tree(child, database, query_kwargs))
    results.append((operation._name, operation.query(database, **query_kwargs)))
    return results


def _fail_formula_loader(operation_name: str):
    """Return a loader replacement proving that SOL never reads perf data."""

    def fail_load_data(_cls, _database):
        raise AssertionError(f"Step4-Pro-V1 SOL graph must not load {operation_name} perf data")

    return classmethod(fail_load_data)


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
@pytest.mark.parametrize("database_mode", [None, "SILICON", "HYBRID", "EMPIRICAL", "SOL_FULL"])
def test_step4_pro_v1_rejects_unsupported_task_database_modes(serving_mode, database_mode):
    """The Pro identity must admit scalar SOL only at the Task boundary."""
    with pytest.raises(ValueError, match=r"Step4.*database_mode.*SOL"):
        Task(**_task_kwargs(serving_mode, database_mode))


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
def test_step4_pro_v1_sol_validation_never_loads_perfdb(monkeypatch, serving_mode):
    """Scalar SOL Task validation must finish before any database load."""
    task = Task(**_task_kwargs(serving_mode, "SOL"))
    load_database = MagicMock(side_effect=AssertionError("formula-only validation must not load perfdb"))
    monkeypatch.setattr(task, "_load_database", load_database)

    task.validate()

    load_database.assert_not_called()


def _direct_formula_queries(database):
    """Return representative direct query callables from the Pro operation graph."""
    hca_kwargs = {
        "b": 2,
        "s": 4096,
        "num_heads": 12,
        "native_heads": 96,
        "tp_size": 8,
        "hidden_size": 6144,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "head_dim": 512,
        "rope_head_dim": 64,
        "index_n_heads": 0,
        "index_head_dim": 0,
        "index_topk": 0,
        "window_size": 512,
        "compress_ratio": 128,
        "o_groups": 2,
        "kvcache_quant_mode": common.KVCacheQuantMode.fp8,
        "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
        "gemm_quant_mode": common.GEMMQuantMode.fp8,
    }
    return {
        "gemm": lambda mode: database.query_gemm(
            4096,
            6144,
            6144,
            common.GEMMQuantMode.fp8,
            database_mode=mode,
        ),
        "mem_op": lambda mode: database.query_mem_op(4096 * 6144 * 2, database_mode=mode),
        "context_attention": lambda mode: database.query_context_attention(
            b=2,
            s=4096,
            prefix=0,
            n=8,
            n_kv=8,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            database_mode=mode,
            head_size=96,
        ),
        "generation_attention": lambda mode: database.query_generation_attention(
            b=2,
            s=4096,
            n=8,
            n_kv=8,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            database_mode=mode,
            head_size=96,
        ),
        "context_hca": lambda mode: database.query_context_deepseek_v4_attention_module(
            **hca_kwargs,
            prefix=0,
            database_mode=mode,
        ),
        "generation_hca": lambda mode: database.query_generation_deepseek_v4_attention_module(
            **hca_kwargs,
            database_mode=mode,
        ),
        "moe": lambda mode: database.query_moe(
            num_tokens=4096,
            hidden_size=6144,
            inter_size=2048,
            topk=8,
            num_experts=512,
            moe_tp_size=8,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.fp8,
            workload_distribution="uniform",
            database_mode=mode,
            is_gated=True,
        ),
        "custom_allreduce": lambda mode: database.query_custom_allreduce(
            common.CommQuantMode.half,
            8,
            4096 * 6144,
            database_mode=mode,
        ),
        "p2p": lambda mode: database.query_p2p(4096 * 6144 * 2, database_mode=mode),
        "nccl": lambda mode: database.query_nccl(
            common.CommQuantMode.half,
            16,
            "alltoall",
            4096 * 8 * 6144,
            database_mode=mode,
        ),
    }


@pytest.mark.parametrize(
    "query_name",
    [
        "gemm",
        "mem_op",
        "context_attention",
        "generation_attention",
        "context_hca",
        "generation_hca",
        "moe",
        "custom_allreduce",
        "p2p",
        "nccl",
    ],
)
def test_step4_pro_v1_direct_sol_full_components_close_the_roofline(comprehensive_perf_db, query_name):
    """Every direct tuple must select max(math, memory) and equal scalar SOL."""
    query = _direct_formula_queries(comprehensive_perf_db)[query_name]

    selected, math_time, memory_time = query(common.DatabaseMode.SOL_FULL)
    scalar = query(common.DatabaseMode.SOL)

    assert selected == max(math_time, memory_time)
    assert selected == float(scalar)
    assert selected > 0.0
    assert scalar.source == "sol"


@pytest.mark.parametrize("database_mode", [common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL])
@pytest.mark.parametrize(
    ("query_name", "operation_class", "query_method_name"),
    [
        (
            "context_hca",
            ContextDeepSeekV4AttentionModule,
            "query_context_deepseek_v4_attention_module",
        ),
        (
            "generation_hca",
            GenerationDeepSeekV4AttentionModule,
            "query_generation_deepseek_v4_attention_module",
        ),
    ],
)
def test_step4_pro_v1_hca_formula_queries_never_load_perf_data(
    comprehensive_perf_db,
    monkeypatch,
    database_mode,
    query_name,
    operation_class,
    query_method_name,
):
    """HCA scalar and tuple formulas must not touch collector data."""
    getattr(comprehensive_perf_db, query_method_name).cache_clear()
    monkeypatch.setattr(operation_class, "load_data", _fail_formula_loader(operation_class.__name__))

    result = _direct_formula_queries(comprehensive_perf_db)[query_name](database_mode)

    if database_mode == common.DatabaseMode.SOL:
        assert isinstance(result, PerformanceResult)
        assert result.source == "sol"
    else:
        selected, math_time, memory_time = result
        assert selected == max(math_time, memory_time)


@pytest.mark.parametrize(
    ("tp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size"),
    [
        pytest.param(1, 1, 1, 1, id="tp1-ep1"),
        pytest.param(1, 16, 1, 16, id="attention-dp16-ep16"),
        pytest.param(8, 1, 8, 1, id="moe-tp8"),
    ],
)
def test_step4_pro_v1_graph_is_recursively_formula_only(
    mutable_comprehensive_perf_db,
    monkeypatch,
    tp_size,
    attention_dp_size,
    moe_tp_size,
    moe_ep_size,
):
    """Every context/decode node must return explicit SOL provenance."""
    database = mutable_comprehensive_perf_db
    database.backend = common.BackendName.vllm.value
    database.set_default_database_mode(common.DatabaseMode.SOL)

    loader_classes = (
        GEMM,
        ContextAttention,
        GenerationAttention,
        ContextDeepSeekV4AttentionModule,
        GenerationDeepSeekV4AttentionModule,
        MoE,
        MoEDispatch,
        CustomAllReduce,
        NCCL,
    )
    for operation_class in loader_classes:
        monkeypatch.setattr(operation_class, "load_data", _fail_formula_loader(operation_class.__name__))

    model = _build_model(
        tp_size=tp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
    )
    phase_queries = {
        "context": (
            model.context_ops,
            {
                "x": 2 * 4096,
                "batch_size": 2,
                "beam_width": 1,
                "s": 4096,
                "prefix": 0,
                "seq_imbalance_correction_scale": 1.0,
            },
        ),
        "generation": (
            model.generation_ops,
            {
                "x": 2,
                "batch_size": 2,
                "beam_width": 1,
                "s": 4096,
                "gen_seq_imbalance_correction_scale": 1.0,
            },
        ),
    }

    for phase, (operation_list, query_kwargs) in phase_queries.items():
        recursive_results = []
        top_level_results = []
        for operation in operation_list:
            operation_kwargs = dict(query_kwargs)
            if phase == "context" and "logits_gemm" in operation._name:
                operation_kwargs["x"] = operation_kwargs["batch_size"]
            operation_results = _query_operation_tree(operation, database, operation_kwargs)
            recursive_results.extend(operation_results)
            top_level_results.append(operation_results[-1][1])

        assert recursive_results
        assert all(isinstance(result, PerformanceResult) for _, result in recursive_results)
        assert all(result.source == "sol" for _, result in recursive_results)

        phase_total = sum(top_level_results, PerformanceResult(0.0, energy=0.0, source="sol"))
        assert float(phase_total) > 0.0
        assert phase_total.source == "sol"


def test_step4_pro_v1_sol_full_task_rejection_is_explicit():
    """Diagnostic SOL_FULL tuples must be rejected before Task graph construction."""
    with pytest.raises(
        ValueError,
        match=r"Step4 Task execution does not support database_mode='SOL_FULL'.*direct PerfDatabase diagnostic queries",
    ):
        Task(
            **_task_kwargs("agg", "SOL_FULL"),
            isl=128,
            osl=2,
            prefix=0,
            nextn=0,
        )


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
def test_step4_pro_v1_single_point_rejects_mutated_sol_full_before_database_load(
    monkeypatch,
    serving_mode,
):
    """Every single-point entry must revalidate a mutated diagnostic-only mode before I/O."""
    task = Task(
        **_task_kwargs(serving_mode, "SOL"),
        isl=128,
        osl=2,
        prefix=0,
        nextn=0,
    )
    task.database_mode = "SOL_FULL"
    load_database = MagicMock(side_effect=AssertionError("SOL_FULL rejection must precede database loading"))
    monkeypatch.setattr(task, "_load_database", load_database)

    with pytest.raises(
        ValueError,
        match=r"Step4 Task execution does not support database_mode='SOL_FULL'.*direct PerfDatabase diagnostic queries",
    ):
        if serving_mode == "agg":
            task.run_single_agg(
                tp=8,
                pp=2,
                dp=1,
                moe_tp=8,
                moe_ep=1,
                batch_size=1,
                ctx_tokens=128,
                include_per_ops=True,
            )
        else:
            task.run_single_disagg(
                prefill_tp=8,
                prefill_pp=2,
                prefill_dp=1,
                prefill_moe_tp=8,
                prefill_moe_ep=1,
                prefill_batch_size=1,
                decode_tp=8,
                decode_pp=2,
                decode_dp=1,
                decode_moe_tp=8,
                decode_moe_ep=1,
                decode_batch_size=1,
                include_per_ops=True,
            )

    load_database.assert_not_called()


def test_step4_pro_v1_osl_one_reports_context_attention_and_explicit_generation_noop():
    """OSL=1 must expose both context labels and mark decode as unexecuted."""
    task = Task(
        **_task_kwargs("agg", "SOL"),
        isl=1024,
        osl=1,
        prefix=0,
        nextn=0,
    )

    evaluation = task.run_single_agg(
        tp=8,
        pp=2,
        dp=1,
        moe_tp=8,
        moe_ep=1,
        batch_size=1,
        ctx_tokens=1024,
        include_per_ops=True,
    )

    mix_data = evaluation.per_ops_data["mix_step"]
    mix_sources = evaluation.per_ops_source["mix_step"]
    context_attention_names = {
        name for name in mix_data if name.startswith("context_layer_") and name.endswith(" (scaled)")
    }
    assert len(context_attention_names) == 320
    assert sum("_full_" in name for name in context_attention_names) == 140
    assert sum("_nonfull_" in name for name in context_attention_names) == 180
    assert all(mix_sources[name] == "sol" for name in context_attention_names)
    assert not any("mla_approx" in name for name in mix_data)
    assert mix_data["generation_attention (not executed)"] == 0.0
    assert mix_sources["generation_attention (not executed)"] == "not_executed"


def test_step4_pro_v1_decode_consumes_all_generation_attention_components():
    """Decode must retain every per-layer full and non-full attention result."""
    task = Task(
        **_task_kwargs("agg", "SOL"),
        isl=128,
        osl=2,
        prefix=0,
        nextn=0,
    )

    evaluation = task.run_single_agg(
        tp=8,
        pp=2,
        dp=1,
        moe_tp=8,
        moe_ep=1,
        batch_size=2,
        ctx_tokens=128,
        include_per_ops=True,
    )

    mix_data = evaluation.per_ops_data["mix_step"]
    mix_sources = evaluation.per_ops_source["mix_step"]
    generation_attention_names = {name for name in mix_data if name.startswith("generation_layer_")}
    assert len(generation_attention_names) == 320
    assert sum("_full_" in name for name in generation_attention_names) == 140
    assert sum("_nonfull_" in name for name in generation_attention_names) == 180
    assert all(mix_data[name] >= 0.0 for name in generation_attention_names)
    assert all(mix_sources[name] == "sol" for name in generation_attention_names)
    assert not any("mla_approx" in name for name in mix_data)
    assert "generation_attention (not executed)" not in mix_data
