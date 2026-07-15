# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-local capture state for collective-query provenance."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommunicationQueryEvidence:
    """Immutable evidence for one executed communication query."""

    operation_name: str
    operation_kind: str
    collective: str | None
    group_size: int
    tier: str
    bandwidth_bytes_per_sec: int | float
    message_size_bytes: int | float


_COLLECTIVE_QUERY_EVIDENCE: ContextVar[list[CommunicationQueryEvidence] | None] = ContextVar(
    "aic_collective_query_evidence", default=None
)
_COLLECTIVE_OPERATION_NAME: ContextVar[str | None] = ContextVar(
    "aic_collective_operation_name",
    default=None,
)


def get_active_communication_query_collector() -> list[CommunicationQueryEvidence] | None:
    """Return the active collector, or ``None`` when capture is not requested."""
    return _COLLECTIVE_QUERY_EVIDENCE.get()


def get_current_communication_operation_name() -> str | None:
    """Return the operation identity bound to the current execution context."""
    return _COLLECTIVE_OPERATION_NAME.get()


@contextmanager
def capture_collective_queries() -> Iterator[list[CommunicationQueryEvidence]]:
    """Capture communication evidence in an isolated, token-scoped list."""
    collector: list[CommunicationQueryEvidence] = []
    token = _COLLECTIVE_QUERY_EVIDENCE.set(collector)
    try:
        yield collector
    finally:
        _COLLECTIVE_QUERY_EVIDENCE.reset(token)


@contextmanager
def bind_collective_operation(operation_name: str) -> Iterator[None]:
    """Bind one explicit ``Operation._name`` for nested query wrappers."""
    if _COLLECTIVE_QUERY_EVIDENCE.get() is None:
        yield
        return
    if not isinstance(operation_name, str) or not operation_name:
        raise ValueError("collective operation name must be a non-empty string")
    token = _COLLECTIVE_OPERATION_NAME.set(operation_name)
    try:
        yield
    finally:
        _COLLECTIVE_OPERATION_NAME.reset(token)


def _prepare_collective_query_evidence(
    *,
    operation_kind: str,
    collective: str | None,
    group_size: int,
    tier: str,
    bandwidth_bytes_per_sec: int | float,
    message_size_bytes: int | float,
) -> tuple[
    list[CommunicationQueryEvidence] | None,
    CommunicationQueryEvidence | None,
]:
    collector = _COLLECTIVE_QUERY_EVIDENCE.get()
    if collector is None or group_size <= 1:
        return None, None
    operation_name = _COLLECTIVE_OPERATION_NAME.get()
    if operation_name is None:
        raise RuntimeError("Collective query capture requires an explicit executing operation name.")
    return collector, CommunicationQueryEvidence(
        operation_name=operation_name,
        operation_kind=operation_kind,
        collective=collective,
        group_size=group_size,
        tier=tier,
        bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
        message_size_bytes=message_size_bytes,
    )
