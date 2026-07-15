# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for context-local communication-query evidence capture."""

import importlib
from contextvars import copy_context
from dataclasses import FrozenInstanceError

import pytest


def _evidence_module():
    try:
        return importlib.import_module("aiconfigurator.sdk.communication_evidence")
    except ModuleNotFoundError as exc:
        pytest.fail("communication_evidence module must define the approved context-capture API")
        raise AssertionError("unreachable") from exc


def test_communication_query_evidence_is_immutable():
    module = _evidence_module()
    evidence = module.CommunicationQueryEvidence(
        operation_name="context_moe_pre_dispatch",
        operation_kind="nccl",
        collective="all_gather",
        group_size=8,
        tier="inter_node_bw",
        bandwidth_bytes_per_sec=900_000_000_000,
        message_size_bytes=65_536,
    )

    with pytest.raises(FrozenInstanceError):
        evidence.group_size = 2


def test_capture_distinguishes_not_requested_from_requested_empty():
    module = _evidence_module()

    assert module.get_active_communication_query_collector() is None
    with module.capture_collective_queries() as collector:
        assert collector == []
        assert module.get_active_communication_query_collector() is collector
    assert module.get_active_communication_query_collector() is None


def test_nested_capture_scopes_do_not_cross_contaminate():
    module = _evidence_module()
    outer_record = module.CommunicationQueryEvidence(
        operation_name="outer",
        operation_kind="nccl",
        collective="all_reduce",
        group_size=2,
        tier="intra_node_bw",
        bandwidth_bytes_per_sec=900_000_000_000,
        message_size_bytes=16_384,
    )
    inner_record = module.CommunicationQueryEvidence(
        operation_name="inner",
        operation_kind="nccl",
        collective="all_gather",
        group_size=8,
        tier="inter_node_bw",
        bandwidth_bytes_per_sec=900_000_000_000,
        message_size_bytes=65_536,
    )

    with module.capture_collective_queries() as outer:
        outer.append(outer_record)
        with module.capture_collective_queries() as inner:
            inner.append(inner_record)
            assert module.get_active_communication_query_collector() is inner
        assert module.get_active_communication_query_collector() is outer

    assert outer == [outer_record]
    assert inner == [inner_record]


def test_copied_context_capture_is_isolated_from_parent_context():
    module = _evidence_module()

    def collect_in_child():
        with module.capture_collective_queries() as collector:
            collector.append("child")
            return collector

    with module.capture_collective_queries() as parent:
        child = copy_context().run(collect_in_child)
        parent.append("parent")

    assert parent == ["parent"]
    assert child == ["child"]


def test_operation_name_scope_restores_outer_identity():
    module = _evidence_module()

    assert module.get_current_communication_operation_name() is None
    with module.capture_collective_queries(), module.bind_collective_operation("outer_operation"):
        assert module.get_current_communication_operation_name() == "outer_operation"
        with module.bind_collective_operation("inner_operation"):
            assert module.get_current_communication_operation_name() == "inner_operation"
        assert module.get_current_communication_operation_name() == "outer_operation"
    assert module.get_current_communication_operation_name() is None
