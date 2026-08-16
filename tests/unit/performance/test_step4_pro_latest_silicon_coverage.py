"""Tests for the Step4-Pro-Latest task-local SILICON coverage validator."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.performance_result import PerformanceResult
from tests.performance.step4_pro_latest.deepep_proxy import (
    B300_NCCL_ALLTOALL_PROXY,
)
from tests.performance.step4_pro_latest.validate_aic_silicon_coverage import (
    build_latest_model,
    probe_operation_list,
    summarize_probe_records,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeOperation:
    _name: str
    result: PerformanceResult | Exception
    _provider: str | None = None

    def query(self, database, **kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class MoE(_FakeOperation):
    _attention_dp_size: int

    def __init__(self, *args, attention_dp_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._attention_dp_size = attention_dp_size
        self.query_kwargs = None

    def query(self, database, **kwargs):
        self.query_kwargs = kwargs
        return super().query(database, **kwargs)


class ContextAttention(_FakeOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_kwargs = None

    def query(self, database, **kwargs):
        self.query_kwargs = kwargs
        return super().query(database, **kwargs)


class MoEDispatch(_FakeOperation):
    def __init__(
        self,
        result: PerformanceResult | Exception,
        *,
        operation: str = "dispatch",
        ep_size: int = 16,
    ) -> None:
        super().__init__(
            "context_layer_002_deepep",
            result,
            "vllm_deepep_high_throughput",
        )
        self._operation = operation
        self._hidden_size = 3584
        self._topk = 16
        self._moe_tp_size = 1
        self._moe_ep_size = ep_size
        self._attention_dp_size = ep_size
        self._scale_factor = 2.0


class _FakeProxyDatabase:
    def __init__(self) -> None:
        self.calls = []

    def query_nccl(self, dtype, num_gpus, operation, message_size):
        self.calls.append((dtype.name, num_gpus, operation, message_size))
        return PerformanceResult(0.5, energy=0.25, source="silicon")


def test_probe_continues_after_missing_silicon_data() -> None:
    operations = [
        _FakeOperation(
            "context_layer_000_qkv",
            PerfDataNotAvailableError("missing exact qkv key"),
            "vllm_step4pro_qkv_norm_rope",
        ),
        _FakeOperation(
            "context_layer_000_router",
            PerformanceResult(0.25, source="silicon"),
            "vllm.optimus_matmul_fp32",
        ),
    ]

    records = probe_operation_list(
        operations,
        database=object(),
        phase="context",
        local_batch_size=1,
        sequence_length=512,
        prefix=0,
    )

    assert [record["status"] for record in records] == ["missing", "ok"]
    assert records[0]["error_type"] == "PerfDataNotAvailableError"
    assert records[0]["provider"] == "vllm_step4pro_qkv_norm_rope"
    assert records[1]["latency_ms"] == pytest.approx(0.25)
    assert records[1]["source"] == "silicon"


def test_probe_requires_explicit_proxy_and_keeps_default_missing() -> None:
    operation = MoEDispatch(PerfDataNotAvailableError("missing DeepEP table"))

    records = probe_operation_list(
        [operation],
        database=object(),
        phase="context",
        ep_size=16,
        local_batch_size=2,
        sequence_length=8,
    )

    assert records[0]["status"] == "missing"
    assert records[0]["family"] == "deepep_ht"


def test_probe_records_explicit_deepep_proxy_separately() -> None:
    database = _FakeProxyDatabase()
    operation = MoEDispatch(
        PerfDataNotAvailableError("exact path must not be queried"),
    )

    records = probe_operation_list(
        [operation],
        database=database,
        phase="context",
        ep_size=16,
        local_batch_size=2,
        sequence_length=8,
        deepep_proxy=B300_NCCL_ALLTOALL_PROXY,
    )

    assert database.calls == [("int8", 16, "alltoall", 57_344)]
    assert records[0]["status"] == "proxy"
    assert records[0]["source"] == "proxy_b300_nccl_alltoall"
    assert records[0]["result_fidelity"] == "PROXY"
    assert records[0]["latency_ms"] == pytest.approx(1.0)
    assert records[0]["proxy"]["operation"] == "dispatch"


def test_context_probe_maps_global_tokens_only_for_optimus_moe() -> None:
    attention = ContextAttention(
        "context_layer_000_attention",
        PerformanceResult(0.5, source="silicon"),
        "vllm_native_sliding_gqa",
    )
    moe = MoE(
        "context_layer_000_moe",
        PerformanceResult(0.75, source="silicon"),
        "optimus_fp8_moe",
        attention_dp_size=16,
    )

    records = probe_operation_list(
        [attention, moe],
        database=object(),
        phase="context",
        local_batch_size=1,
        sequence_length=65_536,
        global_scheduled_tokens=65_536,
        prefix=0,
    )

    assert [record["status"] for record in records] == ["ok", "ok"]
    assert attention.query_kwargs["x"] == 65_536
    assert moe.query_kwargs["x"] == 4_096
    assert records[0]["global_scheduled_tokens"] == 65_536
    assert records[1]["global_scheduled_tokens"] == 65_536


def test_probe_records_non_silicon_source_without_hiding_it() -> None:
    records = probe_operation_list(
        [_FakeOperation("context_formula_op", PerformanceResult(0.5, source="empirical"))],
        database=object(),
        phase="context",
        local_batch_size=2,
        sequence_length=2048,
        prefix=0,
    )

    assert records[0]["status"] == "non_silicon"
    assert records[0]["source"] == "empirical"


def test_summary_deduplicates_repeated_layer_failures_by_physical_contract() -> None:
    records = [
        {
            "status": "missing",
            "phase": "context",
            "operation_name": "context_layer_002_dispatch",
            "operation_class": "MoEDispatch",
            "family": "deepep_ht",
            "provider": "vllm_deepep_high_throughput",
            "physical_identity": [
                "deepep_ht",
                "vllm_deepep_high_throughput",
                "dispatch",
                16,
            ],
            "error_type": "PerfDataNotAvailableError",
            "error": "missing DeepEP table",
        },
        {
            "status": "missing",
            "phase": "context",
            "operation_name": "context_layer_003_dispatch",
            "operation_class": "MoEDispatch",
            "family": "deepep_ht",
            "provider": "vllm_deepep_high_throughput",
            "physical_identity": [
                "deepep_ht",
                "vllm_deepep_high_throughput",
                "dispatch",
                16,
            ],
            "error_type": "PerfDataNotAvailableError",
            "error": "missing DeepEP table",
        },
    ]

    summary = summarize_probe_records(records)

    assert summary["status"] == "BLOCKED"
    assert summary["record_count"] == 2
    assert summary["missing_record_count"] == 2
    assert summary["missing_physical_contract_count"] == 1
    assert summary["families"]["deepep_ht"]["missing_records"] == 2
    assert summary["missing_physical_contracts"] == [
        {
            "family": "deepep_ht",
            "provider": "vllm_deepep_high_throughput",
            "physical_identity": [
                "deepep_ht",
                "vllm_deepep_high_throughput",
                "dispatch",
                16,
            ],
            "record_count": 2,
            "error_type": "PerfDataNotAvailableError",
            "error": "missing DeepEP table",
        }
    ]


def test_summary_reports_proxy_without_merging_it_into_silicon() -> None:
    summary = summarize_probe_records(
        [
            {
                "family": "deepep_ht",
                "status": "proxy",
                "latency_ms": 1.5,
                "provider": "vllm_deepep_high_throughput",
                "physical_identity": ["deepep_ht", "dispatch", 16],
            },
            {
                "family": "GEMM",
                "status": "ok",
                "latency_ms": 2.0,
                "provider": None,
                "physical_identity": ["GEMM"],
            },
            {
                "family": "ElementWise",
                "status": "non_silicon",
                "latency_ms": 0.25,
                "provider": None,
                "physical_identity": ["ElementWise"],
            },
        ]
    )

    assert summary["status"] == "PASS_WITH_PROXY"
    assert summary["result_fidelity"] == "PROXY"
    assert summary["proxy_record_count"] == 1
    assert summary["proxy_latency_ms"] == pytest.approx(1.5)
    assert summary["ok_record_count"] == 1
    assert summary["non_silicon_record_count"] == 1
    assert summary["families"]["deepep_ht"]["proxy_records"] == 1


@pytest.mark.parametrize("ep_size", [16, 32])
def test_latest_model_contains_all_provider_sensitive_families(ep_size: int) -> None:
    model = build_latest_model(ep_size)
    operations = [*model.context_ops, *model.generation_ops]
    providers = {getattr(operation, "_provider", None) for operation in operations}

    assert "vllm_step4pro_torch_einsum" in providers
    assert "vllm.optimus_matmul_fp32" in providers
    assert "vllm_deepep_high_throughput" in providers
    assert "optimus_fp8_moe" in providers
    assert "optimus_fa4" in providers
    assert "vllm_native_sliding_gqa" in providers
