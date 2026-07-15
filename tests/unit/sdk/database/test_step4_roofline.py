# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict roofline-only tests for the Step4 predefined model."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config, models
from aiconfigurator.sdk.operations.communication import NCCL, P2P, CustomAllReduce
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.operations.mla import ContextMLA, GenerationMLA, MLABmm
from aiconfigurator.sdk.operations.moe import MoE, MoEDispatch
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit


def _formula_only_mock_database() -> MagicMock:
    """Build the minimum vLLM SOL database contract used by no-op operations."""
    database = MagicMock()
    database._default_database_mode = common.DatabaseMode.SOL
    database.backend = common.BackendName.vllm.value
    database.system_spec = {
        "gpu": {"sm_version": 90},
        "node": {"num_gpus_per_node": 8},
    }
    return database


def _build_step4_model(*, tp_size: int, attention_dp_size: int, moe_tp_size: int, moe_ep_size: int):
    """Build Step4 for one explicit accepted vLLM parallel row."""
    model_config = config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        nextn=0,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )
    return models.get_model("stepfun-ai/Step4", model_config, backend_name="vllm")


def _query_operation_tree(operation, database, query_kwargs):
    """Query one operation and every nested overlap child."""
    results = []
    if isinstance(operation, ops.OverlapOp):
        for child in operation._group_a:
            results.extend(_query_operation_tree(child, database, query_kwargs))
        for child in operation._group_b:
            results.extend(_query_operation_tree(child, database, query_kwargs))
    results.append((operation._name, operation.query(database, **query_kwargs)))
    return results


def _fail_formula_loader(operation_name):
    """Build a loader replacement that proves strict SOL never reads perf data."""

    def fail_load_data(_cls, _database):
        raise AssertionError(f"strict Step4 SOL graph must not load {operation_name} perf data")

    return classmethod(fail_load_data)


@pytest.mark.parametrize("database_mode", ["SOL", "SOL_FULL"])
def test_step4_task_validation_never_loads_perfdb_in_formula_only_modes(monkeypatch, database_mode):
    """Formula-only validation must not inspect profiling support tables."""
    task = Task(
        serving_mode="agg",
        model_path="stepfun-ai/Step4",
        system_name="h200_sxm",
        backend_name="vllm",
        backend_version="0.22.0",
        database_mode=database_mode,
    )

    def fail_load_database(*_args, **_kwargs):
        raise AssertionError("formula-only Task validation must not load perfdb")

    monkeypatch.setattr(task, "_load_database", fail_load_database)

    task.validate()


def _step4_task_kwargs(serving_mode, database_mode):
    if serving_mode == "agg":
        return {
            "serving_mode": "agg",
            "model_path": "stepfun-ai/Step4",
            "system_name": "h200_sxm",
            "backend_name": "vllm",
            "backend_version": "0.22.0",
            "database_mode": database_mode,
        }
    return {
        "serving_mode": "disagg",
        "prefill_model_path": "stepfun-ai/Step4",
        "prefill_system_name": "h200_sxm",
        "prefill_backend_name": "vllm",
        "prefill_backend_version": "0.22.0",
        "decode_model_path": "stepfun-ai/Step4",
        "decode_system_name": "h200_sxm",
        "decode_backend_name": "vllm",
        "decode_backend_version": "0.22.0",
        "database_mode": database_mode,
    }


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
@pytest.mark.parametrize("database_mode", [None, "SILICON", "HYBRID", "EMPIRICAL"])
def test_step4_task_constructor_rejects_non_formula_database_modes(serving_mode, database_mode):
    with pytest.raises(ValueError, match=r"Step4.*SOL.*SOL_FULL"):
        Task(**_step4_task_kwargs(serving_mode, database_mode))


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
@pytest.mark.parametrize("database_mode", [None, "SILICON", "HYBRID", "EMPIRICAL"])
def test_step4_task_validation_rejects_mutated_non_formula_mode_before_database_checks(
    monkeypatch,
    serving_mode,
    database_mode,
):
    task = Task(**_step4_task_kwargs(serving_mode, "SOL"))
    task.database_mode = database_mode
    database_check = MagicMock()
    monkeypatch.setattr(task, "_check_role_against_db", database_check)

    with pytest.raises(ValueError, match=r"Step4.*SOL.*SOL_FULL"):
        task.validate()

    database_check.assert_not_called()


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
@pytest.mark.parametrize("database_mode", [None, "SILICON", "HYBRID", "EMPIRICAL"])
def test_step4_task_run_validate_false_rejects_mutated_non_formula_mode_before_perfdb_load(
    monkeypatch,
    serving_mode,
    database_mode,
):
    task = Task(**_step4_task_kwargs(serving_mode, "SOL"))
    task.database_mode = database_mode
    load_database = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(task, "_load_database", load_database)
    monkeypatch.setattr("aiconfigurator.sdk.sweep.sweep_agg", MagicMock(return_value=pd.DataFrame()))
    monkeypatch.setattr("aiconfigurator.sdk.sweep.sweep_disagg", MagicMock(return_value=pd.DataFrame()))

    with pytest.raises(ValueError, match=r"Step4.*SOL.*SOL_FULL"):
        task.run(validate=False)

    load_database.assert_not_called()


@pytest.mark.parametrize("database_mode", [common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL])
def test_direct_moe_formula_query_never_loads_perfdb(comprehensive_perf_db, monkeypatch, database_mode):
    """MoE formula queries must resolve before any profiling-data load."""
    comprehensive_perf_db.query_moe.cache_clear()

    def fail_load_data(_cls, _database):
        raise AssertionError("formula-only MoE query must not load perfdb")

    monkeypatch.setattr(MoE, "load_data", classmethod(fail_load_data))

    result = comprehensive_perf_db.query_moe(
        num_tokens=17,
        hidden_size=4096,
        inter_size=1536,
        topk=8,
        num_experts=352,
        moe_tp_size=1,
        moe_ep_size=1,
        quant_mode=common.MoEQuantMode.fp8,
        workload_distribution="uniform",
        database_mode=database_mode,
        is_gated=True,
    )

    if database_mode == common.DatabaseMode.SOL:
        assert float(result) > 0
        assert result.source == "sol"
    else:
        sol_time, sol_math, sol_mem = result
        assert sol_time == max(sol_math, sol_mem) > 0


def test_custom_allreduce_noop_preserves_formula_source():
    """A TP=1 no-op must retain SOL provenance without querying perf data."""
    database = _formula_only_mock_database()
    operation = CustomAllReduce("step4_noop_allreduce", 1.0, 4096, tp_size=1)

    result = operation.query(database, x=17)

    assert float(result) == 0.0
    assert result.source == "sol"
    database.query_custom_allreduce.assert_not_called()


def test_p2p_noop_preserves_formula_source():
    """A PP=1 no-op must retain SOL provenance without querying perf data."""
    database = _formula_only_mock_database()
    operation = P2P("step4_noop_p2p", 1.0, 4096, pp_size=1)

    result = operation.query(database, x=17)

    assert float(result) == 0.0
    assert result.source == "sol"
    database.query_p2p.assert_not_called()


@pytest.mark.parametrize("pre_dispatch", [True, False])
def test_vllm_moe_dispatch_noop_preserves_formula_source(pre_dispatch):
    """A communication-free vLLM dispatch must retain SOL provenance."""
    database = _formula_only_mock_database()
    operation = MoEDispatch(
        "step4_noop_moe_dispatch",
        1.0,
        hidden_size=4096,
        topk=8,
        num_experts=352,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        pre_dispatch=pre_dispatch,
        quant_mode=common.MoEQuantMode.fp8,
    )

    result = operation.query(database, x=17)

    assert float(result) == 0.0
    assert result.source == "sol"
    database.query_custom_allreduce.assert_not_called()
    database.query_nccl.assert_not_called()


@pytest.mark.parametrize(
    ("tp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size"),
    [
        pytest.param(1, 1, 1, 1, id="pattern-b-baseline"),
        pytest.param(1, 16, 1, 16, id="pure-ep16"),
        pytest.param(8, 1, 8, 1, id="pure-moe-tp8"),
    ],
)
def test_step4_graph_is_recursively_formula_only(
    mutable_comprehensive_perf_db,
    monkeypatch,
    tp_size,
    attention_dp_size,
    moe_tp_size,
    moe_ep_size,
):
    """Every Step4 graph result must remain SOL without loading profiling data."""
    database = mutable_comprehensive_perf_db
    database.backend = common.BackendName.vllm.value
    database.set_default_database_mode(common.DatabaseMode.SOL)

    loader_classes = (GEMM, ContextMLA, GenerationMLA, MLABmm, MoE, MoEDispatch, CustomAllReduce, NCCL)
    for operation_class in loader_classes:
        monkeypatch.setattr(
            operation_class,
            "load_data",
            _fail_formula_loader(operation_class.__name__),
        )

    model = _build_step4_model(
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

        zero_results = [(name, result) for name, result in recursive_results if float(result) == 0.0]
        assert zero_results
        assert all(result.source == "sol" for _, result in zero_results)

        phase_total = sum(top_level_results, PerformanceResult(0.0, energy=0.0, source="sol"))
        assert float(phase_total) > 0.0
        assert phase_total.source == "sol"

        if phase == "generation":
            overlap_results = [result for name, result in recursive_results if name == "generation_moe_overlap"]
            assert len(overlap_results) == 1
            assert float(overlap_results[0]) > 0.0
            assert overlap_results[0].source == "sol"


def test_step4_aggregate_osl_one_reports_granular_attention_and_explicit_noop():
    """Executed Full/SWA ops stay SOL while unexecuted generation is labeled explicitly."""
    task = Task(
        serving_mode="agg",
        model_path="stepfun-ai/Step4",
        system_name="h200_sxm",
        backend_name="vllm",
        backend_version="0.22.0",
        database_mode="SOL",
        isl=4096,
        osl=1,
        prefix=0,
        nextn=0,
    )

    evaluation = task.run_single_agg(
        tp=8,
        pp=1,
        dp=1,
        moe_tp=8,
        moe_ep=1,
        batch_size=1,
        ctx_tokens=4096,
        include_per_ops=True,
    )

    mix_data = evaluation.per_ops_data["mix_step"]
    mix_sources = evaluation.per_ops_source["mix_step"]
    context_attention_names = {
        name
        for name in mix_data
        if name.startswith("context_") and "_mla_approx_" in name and name.endswith(" (scaled)")
    }
    assert len(context_attention_names) == 14
    assert sum(name.startswith("context_full_mla_approx_") for name in context_attention_names) == 7
    assert sum(name.startswith("context_swa_mla_approx_") for name in context_attention_names) == 7
    assert all(mix_sources[name] == "sol" for name in context_attention_names)
    assert mix_data["generation_attention (not executed)"] == 0.0
    assert mix_sources["generation_attention (not executed)"] == "not_executed"
    assert all(source == "sol" for name, source in mix_sources.items() if name != "generation_attention (not executed)")


def test_step4_aggregate_decode_consumes_all_generation_attention_components():
    """A mixed decode step must retain all Full/SWA projection, BMM, MLA, and AR evidence."""
    task = Task(
        serving_mode="agg",
        model_path="stepfun-ai/Step4",
        system_name="h200_sxm",
        backend_name="vllm",
        backend_version="0.22.0",
        database_mode="SOL",
        isl=128,
        osl=2,
        prefix=0,
        nextn=0,
    )

    evaluation = task.run_single_agg(
        tp=8,
        pp=1,
        dp=1,
        moe_tp=8,
        moe_ep=1,
        batch_size=2,
        ctx_tokens=128,
        include_per_ops=True,
    )

    mix_data = evaluation.per_ops_data["mix_step"]
    mix_sources = evaluation.per_ops_source["mix_step"]
    generation_attention_names = {
        name for name in mix_data if name.startswith("generation_") and "_mla_approx_" in name
    }
    assert len(generation_attention_names) == 16
    assert sum(name.startswith("generation_full_mla_approx_") for name in generation_attention_names) == 8
    assert sum(name.startswith("generation_swa_mla_approx_") for name in generation_attention_names) == 8
    assert all(mix_data[name] >= 0.0 for name in generation_attention_names)
    assert all(mix_sources[name] == "sol" for name in generation_attention_names)
    assert "generation_attention (not executed)" not in mix_data
