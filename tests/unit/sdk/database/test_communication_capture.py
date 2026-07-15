# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for query-boundary communication evidence capture."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.communication_evidence import (
    CommunicationQueryEvidence,
    bind_collective_operation,
    capture_collective_queries,
    get_active_communication_query_collector,
)


def test_nccl_capture_records_cache_hits_in_call_order(comprehensive_perf_db):
    database = comprehensive_perf_db
    database.clear_runtime_caches()
    group_size = 8
    element_count = 8_192
    selection = database.system_spec.select_p2p_bandwidth(group_size)

    with capture_collective_queries() as evidence, bind_collective_operation("context_moe_pre_dispatch"):
        first = database.query_nccl(
            common.CommQuantMode.half,
            group_size,
            "all_gather",
            element_count,
            database_mode=common.DatabaseMode.SOL,
        )
        second = database.query_nccl(
            common.CommQuantMode.half,
            group_size,
            "all_gather",
            element_count,
            database_mode=common.DatabaseMode.SOL,
        )

    expected = CommunicationQueryEvidence(
        operation_name="context_moe_pre_dispatch",
        operation_kind="nccl",
        collective="all_gather",
        group_size=group_size,
        tier=selection.tier,
        bandwidth_bytes_per_sec=selection.bandwidth_bytes_per_second,
        message_size_bytes=(element_count * common.CommQuantMode.half.value.memory),
    )
    assert float(first) == float(second)
    assert evidence == [expected, expected]


def test_custom_allreduce_capture_records_actual_group_and_tier(
    comprehensive_perf_db,
):
    database = comprehensive_perf_db
    group_size = 2
    element_count = 4_096
    selection = database.system_spec.select_p2p_bandwidth(group_size)

    with capture_collective_queries() as evidence, bind_collective_operation("generation_attention_allreduce"):
        database.query_custom_allreduce(
            common.CommQuantMode.half,
            group_size,
            element_count,
            database_mode=common.DatabaseMode.SOL,
        )

    assert evidence == [
        CommunicationQueryEvidence(
            operation_name="generation_attention_allreduce",
            operation_kind="custom_allreduce",
            collective="all_reduce",
            group_size=group_size,
            tier=selection.tier,
            bandwidth_bytes_per_sec=selection.bandwidth_bytes_per_second,
            message_size_bytes=(element_count * common.CommQuantMode.half.value.memory),
        )
    ]


def test_p2p_capture_records_the_fixed_pairwise_formula_tier(
    comprehensive_perf_db,
):
    database = comprehensive_perf_db
    message_size_bytes = 32_768

    with capture_collective_queries() as evidence, bind_collective_operation("context_p2p"):
        database.query_p2p(
            message_size_bytes,
            database_mode=common.DatabaseMode.SOL,
        )

    assert evidence == [
        CommunicationQueryEvidence(
            operation_name="context_p2p",
            operation_kind="p2p",
            collective=None,
            group_size=2,
            tier="inter_node_bw",
            bandwidth_bytes_per_sec=database.system_spec["node"]["inter_node_bw"],
            message_size_bytes=message_size_bytes,
        )
    ]


@pytest.mark.parametrize("query_kind", ["nccl", "custom_allreduce"])
def test_group_one_query_is_an_explicit_communication_noop(
    comprehensive_perf_db,
    query_kind,
):
    database = comprehensive_perf_db

    with capture_collective_queries() as evidence:
        if query_kind == "nccl":
            database.query_nccl(
                common.CommQuantMode.half,
                1,
                "all_reduce",
                4_096,
                database_mode=common.DatabaseMode.SOL,
            )
        else:
            database.query_custom_allreduce(
                common.CommQuantMode.half,
                1,
                4_096,
                database_mode=common.DatabaseMode.SOL,
            )

    assert evidence == []


def test_active_capture_rejects_collective_without_operation_identity(
    comprehensive_perf_db,
):
    with (
        capture_collective_queries(),
        pytest.raises(
            RuntimeError,
            match="operation name",
        ),
    ):
        comprehensive_perf_db.query_nccl(
            common.CommQuantMode.half,
            8,
            "all_reduce",
            4_096,
            database_mode=common.DatabaseMode.SOL,
        )


def test_query_without_capture_leaves_context_unrequested(
    comprehensive_perf_db,
):
    assert get_active_communication_query_collector() is None

    comprehensive_perf_db.query_nccl(
        common.CommQuantMode.half,
        8,
        "all_reduce",
        4_096,
        database_mode=common.DatabaseMode.SOL,
    )

    assert get_active_communication_query_collector() is None


@pytest.mark.parametrize("query_kind", ["nccl", "custom_allreduce"])
def test_query_without_capture_does_not_repeat_topology_selection(
    comprehensive_perf_db,
    monkeypatch,
    query_kind,
):
    database = comprehensive_perf_db
    database.clear_runtime_caches()
    original_select = database.system_spec.select_p2p_bandwidth
    selections = []

    def record_selection(group_size):
        selections.append(group_size)
        return original_select(group_size)

    monkeypatch.setattr(
        database.system_spec,
        "select_p2p_bandwidth",
        record_selection,
    )

    if query_kind == "nccl":
        database.query_nccl(
            common.CommQuantMode.half,
            8,
            "all_reduce",
            4_096,
            database_mode=common.DatabaseMode.SOL,
        )
    else:
        database.query_custom_allreduce(
            common.CommQuantMode.half,
            8,
            4_096,
            database_mode=common.DatabaseMode.SOL,
        )

    assert selections == [8]
