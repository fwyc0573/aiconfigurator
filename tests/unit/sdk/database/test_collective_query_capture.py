# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common, config, models
from aiconfigurator.sdk import perf_database as perf_database_module
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.operations import CustomAllReduce, MoEDispatch
from aiconfigurator.sdk.perf_database import PerfDatabase, load_system_spec
from aiconfigurator.sdk.system_spec import SystemSpec

pytestmark = pytest.mark.unit


def _capture_api():
    evidence_type = getattr(perf_database_module, "CommunicationQueryEvidence", None)
    capture = getattr(perf_database_module, "capture_collective_queries", None)
    bind_operation = getattr(perf_database_module, "bind_collective_operation", None)
    assert evidence_type is not None, "PerfDatabase must expose CommunicationQueryEvidence"
    assert not hasattr(perf_database_module, "CollectiveQueryEvidence")
    assert capture is not None, "PerfDatabase must expose capture_collective_queries"
    assert bind_operation is not None, "PerfDatabase must expose bind_collective_operation"
    return evidence_type, capture, bind_operation


def _make_sol_database(system: str, *, backend: str = "vllm") -> PerfDatabase:
    database = object.__new__(PerfDatabase)
    database.system = system
    database.backend = backend
    database.version = "test"
    database.systems_root = "test"
    database.system_spec = SystemSpec(load_system_spec(system))
    database._default_database_mode = common.DatabaseMode.SOL
    return database


def _clear_collective_query_caches() -> None:
    for name in (
        "query_custom_allreduce",
        "_query_custom_allreduce_cached",
        "query_nccl",
        "_query_nccl_cached",
        "query_p2p",
        "_query_p2p_cached",
    ):
        method = getattr(PerfDatabase, name, None)
        cache_clear = getattr(method, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


@pytest.fixture(autouse=True)
def clear_collective_query_caches():
    _clear_collective_query_caches()
    yield
    _clear_collective_query_caches()


@pytest.mark.parametrize(
    ("group_size", "expected_tier", "expected_bandwidth"),
    [
        (8, "intra_node_bw", 450_000_000_000),
        (16, "inter_node_bw", 50_000_000_000),
    ],
)
def test_h200_nccl_capture_uses_actual_group_and_topology_tier(
    group_size,
    expected_tier,
    expected_bandwidth,
):
    evidence_type, capture, bind_operation = _capture_api()
    database = _make_sol_database("h200_sxm")
    operation_name = "opaque.attention/collective#17"

    with capture() as collected, bind_operation(operation_name):
        database.query_nccl(
            common.CommQuantMode.half,
            group_size,
            "all_reduce",
            1024,
        )

    assert tuple(collected) == (
        evidence_type(
            operation_name=operation_name,
            operation_kind="nccl",
            collective="all_reduce",
            group_size=group_size,
            tier=expected_tier,
            bandwidth_bytes_per_sec=expected_bandwidth,
            message_size_bytes=2048,
        ),
    )


def test_gb300_actual_operations_preserve_group_two_vs_group_eight_and_order():
    evidence_type, capture, _bind_operation = _capture_api()
    database = _make_sol_database("gb300")
    projection = CustomAllReduce(
        name="projection.reduction[opaque]",
        scale_factor=1,
        h=4096,
        tp_size=2,
    )
    dispatch = MoEDispatch(
        name="router.exchange[opaque]",
        scale_factor=1,
        hidden_size=4096,
        topk=8,
        num_experts=352,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=4,
        pre_dispatch=True,
    )
    model = SimpleNamespace(context_ops=[projection, dispatch])
    runtime_config = SimpleNamespace(seq_imbalance_correction_scale=1.0)

    with capture() as collected:
        BaseBackend()._run_context_phase(
            model,
            database,
            runtime_config,
            batch_size=1,
            isl=4,
            prefix=0,
        )

    assert tuple(collected) == (
        evidence_type(
            operation_name=projection._name,
            operation_kind="custom_allreduce",
            collective="all_reduce",
            group_size=2,
            tier="intra_node_bw",
            bandwidth_bytes_per_sec=900_000_000_000,
            message_size_bytes=32_768,
        ),
        evidence_type(
            operation_name=dispatch._name,
            operation_kind="custom_allreduce",
            collective="all_reduce",
            group_size=8,
            tier="inter_node_bw",
            bandwidth_bytes_per_sec=900_000_000_000,
            message_size_bytes=32_768,
        ),
        evidence_type(
            operation_name=dispatch._name,
            operation_kind="nccl",
            collective="all_gather",
            group_size=8,
            tier="inter_node_bw",
            bandwidth_bytes_per_sec=900_000_000_000,
            message_size_bytes=131_072,
        ),
    )


def test_capture_distinguishes_not_requested_from_requested_empty():
    _evidence_type, capture, _bind_operation = _capture_api()
    collector = getattr(
        perf_database_module,
        "_COLLECTIVE_QUERY_EVIDENCE",
        None,
    )
    assert collector is not None
    assert collector.get() is None

    with capture() as collected:
        assert collected == []
        assert collector.get() is collected

    assert collector.get() is None
    assert tuple(collected) == ()


def test_nested_capture_scopes_are_isolated_and_restore_outer_scope():
    _evidence_type, capture, bind_operation = _capture_api()
    database = _make_sol_database("h200_sxm")

    with capture() as outer:
        with bind_operation("outer.first"):
            database.query_p2p(1024)
        with capture() as inner, bind_operation("inner.only"):
            database.query_p2p(2048)
        with bind_operation("outer.second"):
            database.query_p2p(4096)

    assert [record.operation_name for record in outer] == [
        "outer.first",
        "outer.second",
    ]
    assert [record.operation_name for record in inner] == ["inner.only"]


def test_concurrent_capture_scopes_do_not_cross_contaminate():
    _evidence_type, capture, bind_operation = _capture_api()

    def run_one(operation_name: str, message_bytes: int):
        database = _make_sol_database("h200_sxm")
        with capture() as collected, bind_operation(operation_name):
            database.query_p2p(message_bytes)
        return tuple(collected)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_one, "thread.first", 1024)
        second = executor.submit(run_one, "thread.second", 2048)

    assert [record.operation_name for record in first.result()] == ["thread.first"]
    assert [record.message_size_bytes for record in first.result()] == [1024]
    assert [record.operation_name for record in second.result()] == ["thread.second"]
    assert [record.message_size_bytes for record in second.result()] == [2048]


def test_nccl_cache_hit_still_appends_ordered_duplicate_evidence():
    _evidence_type, capture, bind_operation = _capture_api()
    database = _make_sol_database("h200_sxm")
    cached_query = getattr(PerfDatabase, "_query_nccl_cached", None)
    assert cached_query is not None, "NCCL caching must live inside the capture wrapper"
    cached_query.cache_clear()

    with capture() as collected, bind_operation("cached.nccl"):
        database.query_nccl(
            common.CommQuantMode.half,
            8,
            "all_reduce",
            65_536,
        )
        database.query_nccl(
            common.CommQuantMode.half,
            8,
            "all_reduce",
            65_536,
        )

    cache_info = cached_query.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1
    assert len(collected) == 2
    assert collected[0] == collected[1]


def test_group_one_collectives_do_not_emit_evidence():
    _evidence_type, capture, bind_operation = _capture_api()
    database = _make_sol_database("h200_sxm")

    with capture() as collected, bind_operation("no_op.collectives"):
        database.query_custom_allreduce(
            common.CommQuantMode.half,
            1,
            4096,
        )
        database.query_nccl(
            common.CommQuantMode.half,
            1,
            "all_reduce",
            4096,
        )

    assert tuple(collected) == ()


def test_active_capture_requires_explicit_operation_binding():
    _evidence_type, capture, _bind_operation = _capture_api()
    database = _make_sol_database("h200_sxm")

    with capture(), pytest.raises(RuntimeError, match="operation name"):
        database.query_nccl(
            common.CommQuantMode.half,
            8,
            "all_reduce",
            4096,
        )


def test_step4_overlap_collectives_use_inner_operation_names():
    _evidence_type, capture, _bind_operation = _capture_api()
    model_config = config.ModelConfig(
        tp_size=4,
        pp_size=1,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=8,
        nextn=0,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )
    model = models.get_model("stepfun-ai/Step4", model_config, backend_name="vllm")
    database = _make_sol_database("h200_sxm")
    runtime_config = SimpleNamespace(gen_seq_imbalance_correction_scale=1.0)

    with capture() as collected:
        BaseBackend()._run_generation_phase(
            model,
            database,
            runtime_config,
            batch_size=1,
            beam_width=1,
            isl=4,
            osl=2,
            stride=1,
        )

    operation_names = [record.operation_name for record in collected]
    assert {
        "generation_moe_pre_dispatch",
        "generation_moe_post_dispatch",
        "generation_shared_ffn_ar",
    } <= set(operation_names)
    assert "generation_moe_overlap" not in operation_names


def test_collective_evidence_round_trips_exactly_through_json_and_sqlite():
    evidence_type, capture, bind_operation = _capture_api()
    database = _make_sol_database("h200_sxm")

    with capture() as collected, bind_operation("roundtrip.exact"):
        database.query_nccl(
            common.CommQuantMode.half,
            16,
            "reduce_scatter",
            8192,
        )

    payload = json.dumps([asdict(record) for record in collected], sort_keys=True)
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE evidence (payload TEXT NOT NULL)")
        connection.execute("INSERT INTO evidence(payload) VALUES (?)", (payload,))
        stored_payload = connection.execute("SELECT payload FROM evidence").fetchone()[0]

    reconstructed = tuple(evidence_type(**item) for item in json.loads(stored_payload))
    assert reconstructed == tuple(collected)
    assert json.dumps([asdict(record) for record in reconstructed], sort_keys=True) == payload
