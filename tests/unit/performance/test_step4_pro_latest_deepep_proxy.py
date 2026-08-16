"""Tests for the explicit Step4-Pro-Latest DeepEP simulation proxy."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.performance_result import PerformanceResult
from tests.performance.step4_pro_latest.deepep_proxy import (
    B300_NCCL_ALLTOALL_PROXY,
    query_deepep_proxy,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeDatabase:
    calls: list[tuple[common.CommQuantMode, int, str, int]] = field(default_factory=list)

    def query_nccl(
        self,
        dtype: common.CommQuantMode,
        num_gpus: int,
        operation: str,
        message_size: int,
    ) -> PerformanceResult:
        self.calls.append((dtype, num_gpus, operation, message_size))
        return PerformanceResult(1.25, energy=2.5, source="silicon")


class MoEDispatch:
    def __init__(
        self,
        *,
        operation: str,
        ep_size: int,
        scale_factor: float,
    ) -> None:
        self._provider = "vllm_deepep_high_throughput"
        self._operation = operation
        self._hidden_size = 3584
        self._topk = 16
        self._moe_tp_size = 1
        self._moe_ep_size = ep_size
        self._attention_dp_size = ep_size
        self._scale_factor = scale_factor


@pytest.mark.parametrize(
    ("operation_name", "ep_size", "tokens_per_dp_rank", "scale_factor", "table_dtype"),
    [
        ("dispatch", 16, 33, 2.0, common.CommQuantMode.int8),
        ("combine", 32, 17, 3.5, common.CommQuantMode.half),
    ],
)
def test_proxy_maps_dtype_volume_topology_and_scale(
    operation_name: str,
    ep_size: int,
    tokens_per_dp_rank: int,
    scale_factor: float,
    table_dtype: common.CommQuantMode,
) -> None:
    database = _FakeDatabase()
    operation = MoEDispatch(
        operation=operation_name,
        ep_size=ep_size,
        scale_factor=scale_factor,
    )

    result = query_deepep_proxy(
        operation,
        database,
        tokens_per_dp_rank=tokens_per_dp_rank,
        proxy_name=B300_NCCL_ALLTOALL_PROXY,
    )

    expected_message_elements = (tokens_per_dp_rank * operation._hidden_size * operation._topk + ep_size - 1) // ep_size
    assert database.calls == [(table_dtype, ep_size, "alltoall", expected_message_elements)]
    assert result.latency_ms == pytest.approx(1.25 * scale_factor)
    assert result.energy_wms == pytest.approx(2.5 * scale_factor)
    assert result.source == "proxy_b300_nccl_alltoall"
    assert result.metadata == {
        "proxy_name": "b300_nccl_alltoall",
        "result_fidelity": "PROXY",
        "proxy_source": "proxy_b300_nccl_alltoall",
        "original_provider": "vllm_deepep_high_throughput",
        "operation": operation_name,
        "logical_payload_dtype": ("fp8" if operation_name == "dispatch" else "bfloat16"),
        "nccl_table_dtype": ("int8" if operation_name == "dispatch" else "half"),
        "collective": "alltoall",
        "tokens_per_dp_rank": tokens_per_dp_rank,
        "hidden_size": 3584,
        "topk": 16,
        "requested_ep_size": ep_size,
        "measured_topology_max_gpus": 8,
        "topology_extrapolated": True,
        "topology_mapping": "aic_nccl_rank_correction_from_8_gpu_curve",
        "message_elements": expected_message_elements,
        "scale_factor": scale_factor,
    }


def test_proxy_rejects_unknown_name_without_fallback() -> None:
    with pytest.raises(ValueError, match="unsupported DeepEP proxy"):
        query_deepep_proxy(
            MoEDispatch(operation="dispatch", ep_size=16, scale_factor=1.0),
            _FakeDatabase(),
            tokens_per_dp_rank=1,
            proxy_name="automatic",
        )


def test_proxy_rejects_non_step4_deepep_operation() -> None:
    operation = MoEDispatch(
        operation="dispatch",
        ep_size=16,
        scale_factor=1.0,
    )
    operation._provider = "another_provider"

    with pytest.raises(ValueError, match="vllm_deepep_high_throughput"):
        query_deepep_proxy(
            operation,
            _FakeDatabase(),
            tokens_per_dp_rank=1,
            proxy_name=B300_NCCL_ALLTOALL_PROXY,
        )
