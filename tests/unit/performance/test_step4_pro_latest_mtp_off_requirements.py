"""Tests for the task-local Step4-Pro-Latest MTP-off requirements driver."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.performance_result import PerformanceResult
from tests.performance.step4_pro_latest.deepep_proxy import (
    B300_NCCL_ALLTOALL_PROXY,
)
from tests.performance.step4_pro_latest.run_mtp_off_requirements import (
    _is_oom,
    build_component_latency_breakdown,
    build_kv_cache_details,
    build_latency_metrics,
    build_moe_observability,
    build_prefill_schedule,
    build_repeat_audit,
    build_requirements_workloads,
    build_review_rows,
    build_throughput_metrics,
    combine_repeated_payloads,
    merge_component_latency_breakdowns,
    query_scheduled_step,
    summarize_scheduled_records,
    write_review_csv,
)
from tests.performance.step4_pro_latest.validate_aic_silicon_coverage import (
    build_latest_model,
)

pytestmark = pytest.mark.unit


def test_prefill_schedule_uses_one_global_budget_without_dividing_by_batch() -> None:
    steps = build_prefill_schedule(
        prompt_tokens=96,
        batch_size=3,
        max_num_batched_tokens=48,
        ep_size=2,
    )

    assert len(steps) == 6
    assert sum(step["global_scheduled_tokens"] for step in steps) == 288
    assert steps[0] == {
        "step_index": 0,
        "prefix_tokens": 0,
        "chunk_tokens_per_request": 16,
        "global_scheduled_tokens": 48,
        "busiest_rank": 0,
        "busiest_rank_local_batch": 2,
        "busiest_rank_scheduled_tokens": 32,
        "completed_requests": 0,
    }
    assert steps[-1]["prefix_tokens"] == 80
    assert steps[-1]["completed_requests"] == 3


@dataclass
class GEMM:
    _name: str = "context_local_gemm"
    _provider: str | None = None
    query_kwargs: dict | None = None

    def query(self, database, **kwargs):
        self.query_kwargs = kwargs
        return PerformanceResult(1.0, source="silicon")


class MoE(GEMM):
    _provider = "optimus_fp8_moe"

    def __init__(self):
        super().__init__(_name="context_global_moe", _provider=self._provider)
        self.global_tokens = None

    def _query_step4_optimus(self, database, *, num_tokens: int):
        self.global_tokens = num_tokens
        return PerformanceResult(2.0, source="silicon")


class MoEDispatch(GEMM):
    _provider = "vllm_deepep_high_throughput"

    def __init__(self):
        super().__init__(_name="context_local_dispatch", _provider=self._provider)
        self._operation = "dispatch"
        self._hidden_size = 3584
        self._topk = 16
        self._moe_tp_size = 1
        self._moe_ep_size = 16
        self._attention_dp_size = 16
        self._scale_factor = 1.0


class _FakeProxyDatabase:
    def __init__(self) -> None:
        self.calls = []

    def query_nccl(self, dtype, num_gpus, operation, message_size):
        self.calls.append((dtype.name, num_gpus, operation, message_size))
        return PerformanceResult(0.75, energy=0.5, source="silicon")


def test_scheduled_query_uses_global_tokens_for_optimus_and_local_tokens_for_deepep() -> None:
    gemm = GEMM()
    moe = MoE()
    dispatch = MoEDispatch()

    records = query_scheduled_step(
        [gemm, moe, dispatch],
        database=object(),
        phase="context",
        ep_size=16,
        local_batch_size=2,
        chunk_tokens_per_request=8,
        prefix_tokens=24,
        global_scheduled_tokens=40,
    )

    assert [record["status"] for record in records] == ["ok", "ok", "ok"]
    assert gemm.query_kwargs["x"] == 16
    assert moe.global_tokens == 40
    assert dispatch.query_kwargs["x"] == 16


def test_scheduled_query_uses_proxy_only_when_explicitly_selected() -> None:
    database = _FakeProxyDatabase()
    dispatch = MoEDispatch()

    records = query_scheduled_step(
        [dispatch],
        database=database,
        phase="context",
        ep_size=16,
        local_batch_size=2,
        chunk_tokens_per_request=8,
        prefix_tokens=24,
        global_scheduled_tokens=40,
        deepep_proxy=B300_NCCL_ALLTOALL_PROXY,
    )

    assert database.calls == [("int8", 16, "alltoall", 57_344)]
    assert records[0]["status"] == "proxy"
    assert records[0]["result_fidelity"] == "PROXY"
    assert records[0]["latency_ms"] == pytest.approx(0.75)


def test_scheduled_summary_keeps_partial_latency_but_blocks_formal_latency() -> None:
    records = [
        {
            "family": "GEMM",
            "provider": None,
            "physical_identity": ["GEMM", "GEMM", None],
            "status": "ok",
            "source": "silicon",
            "latency_ms": 1.25,
        },
        {
            "family": "qkv_norm_rope",
            "provider": "vllm_step4pro_qkv_norm_rope",
            "physical_identity": ["qkv_norm_rope", "vllm_step4pro_qkv_norm_rope"],
            "status": "missing",
            "error_type": "PerfDataNotAvailableError",
            "error": "missing qkv table",
        },
        {
            "family": "ElementWise",
            "provider": None,
            "physical_identity": ["ElementWise", "ElementWise", None],
            "status": "non_silicon",
            "source": "empirical",
            "latency_ms": 0.25,
        },
    ]

    summary = summarize_scheduled_records(records)

    assert summary["status"] == "BLOCKED"
    assert summary["formal_latency_ms"] is None
    assert summary["known_partial_latency_ms"] == pytest.approx(1.5)
    assert summary["exact_silicon_latency_ms"] == pytest.approx(1.25)
    assert summary["missing_physical_contract_count"] == 1


def test_scheduled_summary_includes_proxy_latency_and_fidelity() -> None:
    records = [
        {
            "family": "GEMM",
            "provider": None,
            "physical_identity": ["GEMM"],
            "status": "ok",
            "source": "silicon",
            "latency_ms": 1.25,
        },
        {
            "family": "deepep_ht",
            "provider": "vllm_deepep_high_throughput",
            "physical_identity": ["deepep_ht", "dispatch", 16],
            "status": "proxy",
            "source": "proxy_b300_nccl_alltoall",
            "latency_ms": 0.75,
        },
        {
            "family": "ElementWise",
            "provider": None,
            "physical_identity": ["ElementWise"],
            "status": "non_silicon",
            "source": "empirical",
            "latency_ms": 0.25,
        },
    ]

    summary = summarize_scheduled_records(records)

    assert summary["status"] == "PASS_WITH_PROXY"
    assert summary["result_fidelity"] == "PROXY"
    assert summary["formal_latency_ms"] == pytest.approx(2.25)
    assert summary["known_partial_latency_ms"] == pytest.approx(2.25)
    assert summary["exact_silicon_latency_ms"] == pytest.approx(1.25)
    assert summary["proxy_latency_ms"] == pytest.approx(0.75)
    assert summary["proxy_record_count"] == 1


def test_component_breakdown_separates_required_step4_groups_without_double_counting() -> None:
    records = [
        {
            "operation_name": "context_layer_003_full_attention",
            "status": "ok",
            "latency_ms": 2.0,
        },
        {
            "operation_name": "context_layer_004_swa_attention",
            "status": "ok",
            "latency_ms": 1.0,
        },
        {
            "operation_name": "context_layer_001_dense_gate_up_gemm",
            "status": "ok",
            "latency_ms": 3.0,
        },
        {
            "operation_name": "context_layer_002_latent_moe_experts",
            "status": "ok",
            "latency_ms": 4.0,
        },
        {
            "operation_name": "context_layer_002_latent_moe_dispatch",
            "status": "proxy",
            "latency_ms": 0.5,
        },
        {
            "operation_name": "context_layer_002_latent_moe_combine",
            "status": "proxy",
            "latency_ms": 0.75,
        },
        {
            "operation_name": "context_embedding",
            "status": "non_silicon",
            "latency_ms": 0.25,
        },
    ]

    breakdown = build_component_latency_breakdown(records)

    assert breakdown["latency_ms"]["full_mfa"] == pytest.approx(2.0)
    assert breakdown["latency_ms"]["swa"] == pytest.approx(1.0)
    assert breakdown["latency_ms"]["dense"] == pytest.approx(3.0)
    assert breakdown["latency_ms"]["latent_moe_compute"] == pytest.approx(4.0)
    assert breakdown["latency_ms"]["dispatch"] == pytest.approx(0.5)
    assert breakdown["latency_ms"]["combine"] == pytest.approx(0.75)
    assert breakdown["latency_ms"]["latent_moe_total"] == pytest.approx(5.25)
    assert breakdown["latency_ms"]["other"] == pytest.approx(0.25)
    assert breakdown["latency_ms"]["accounted_total"] == pytest.approx(11.5)
    assert breakdown["proxy_record_count"]["dispatch"] == 1
    assert breakdown["proxy_record_count"]["combine"] == 1


def test_kv_cache_details_expose_logical_and_page_allocated_bytes() -> None:
    model = build_latest_model(16)

    details = build_kv_cache_details(
        model,
        local_batch_size=2,
        sequence_length=513,
        in_flight_tokens=512,
    )

    assert details["requested_dtype"] == "auto"
    assert details["resolved_dtype"] == "bfloat16"
    assert details["layout"] == "NHD"
    assert details["logical_bytes_per_sequence"] == pytest.approx(model.get_kvcache_bytes_per_sequence(513))
    assert details["modeled_resident_allocated_bytes_per_sequence"] == pytest.approx(
        model.get_kvcache_allocated_bytes_per_sequence(513)
    )
    assert details["modeled_peak_allocated_bytes_per_sequence"] == pytest.approx(
        model.get_kvcache_peak_allocated_bytes_per_sequence(
            513,
            in_flight_tokens=512,
        )
    )
    assert details["logical_bytes_per_gpu"] == pytest.approx(2 * details["logical_bytes_per_sequence"])
    assert details["modeled_peak_allocated_bytes_per_gpu"] == pytest.approx(
        2 * details["modeled_peak_allocated_bytes_per_sequence"]
    )
    assert details["actual_runtime_allocated_bytes_per_gpu"] is None
    assert details["actual_runtime_measurement_status"] == "PENDING_EXTERNAL_VLLM"


def test_latency_metrics_keep_live_runtime_only_values_explicitly_unavailable() -> None:
    prefill = build_latency_metrics(
        phase="prefill",
        formal_prefill_latency_ms=123.5,
    )
    decode = build_latency_metrics(
        phase="decode",
        steady_decode_step_ms=12.5,
    )

    assert prefill["values_ms"]["prefill"] == pytest.approx(123.5)
    assert prefill["values_ms"]["ttft"] is None
    assert prefill["values_ms"]["first_decode_step"] is None
    assert prefill["status"] == "PARTIAL_SIMULATOR_ONLY"
    assert decode["values_ms"]["steady_decode_step_p50"] == pytest.approx(12.5)
    assert decode["values_ms"]["tpot"] == pytest.approx(12.5)
    assert decode["values_ms"]["steady_decode_step_p90"] is None
    assert decode["values_ms"]["steady_decode_step_p99"] is None
    assert decode["values_ms"]["itl_p50"] is None
    assert decode["values_ms"]["decode_generation"] is None
    assert decode["values_ms"]["end_to_end"] is None
    assert "steady_decode_step_p90" in decode["unavailable_reasons"]


def test_throughput_metrics_report_replica_aggregate_and_per_gpu_values() -> None:
    metrics = build_throughput_metrics(
        input_tokens_per_replica=512,
        output_tokens_per_replica=0,
        latency_ms=10.0,
        replica_count=2,
        total_gpus=32,
    )

    assert metrics["input_tok_s_per_replica"] == pytest.approx(51_200.0)
    assert metrics["aggregate_input_tok_s"] == pytest.approx(102_400.0)
    assert metrics["aggregate_output_tok_s"] == pytest.approx(0.0)
    assert metrics["aggregate_total_tok_s"] == pytest.approx(102_400.0)
    assert metrics["input_tok_s_per_gpu"] == pytest.approx(3_200.0)
    assert metrics["total_tok_s_per_gpu"] == pytest.approx(3_200.0)


def test_moe_observability_distinguishes_assumed_distribution_from_live_histogram() -> None:
    observability = build_moe_observability(build_latest_model(16))

    assert observability["performance_workload_distribution"] == "power_law_1.2"
    assert observability["expert_token_histogram"] is None
    assert observability["max_mean_load"] is None
    assert observability["padding_ratio"] is None
    assert observability["live_routing_status"] == "PENDING_EXTERNAL_VLLM"


def test_component_breakdowns_merge_prefill_chunks_without_counting_moe_twice() -> None:
    first = build_component_latency_breakdown(
        [
            {
                "operation_name": "context_layer_003_full_attention",
                "status": "ok",
                "latency_ms": 2.0,
            },
            {
                "operation_name": "context_layer_002_latent_moe_dispatch",
                "status": "proxy",
                "latency_ms": 0.5,
            },
        ]
    )
    second = build_component_latency_breakdown(
        [
            {
                "operation_name": "context_layer_004_swa_attention",
                "status": "ok",
                "latency_ms": 1.0,
            },
            {
                "operation_name": "context_layer_002_latent_moe_combine",
                "status": "proxy",
                "latency_ms": 0.75,
            },
        ]
    )

    merged = merge_component_latency_breakdowns([first, second])

    assert merged["latency_ms"]["full_mfa"] == pytest.approx(2.0)
    assert merged["latency_ms"]["swa"] == pytest.approx(1.0)
    assert merged["latency_ms"]["latent_moe_total"] == pytest.approx(1.25)
    assert merged["latency_ms"]["accounted_total"] == pytest.approx(4.25)
    assert merged["proxy_record_count"]["dispatch"] == 1
    assert merged["proxy_record_count"]["combine"] == 1


def _sample_simulation_payload() -> dict:
    return {
        "status": "PASS_WITH_PROXY",
        "matrix_summary": {
            "prefill_result_count": 1,
            "decode_result_count": 1,
        },
        "prefill_results": [
            {
                "status": "PASS_WITH_PROXY",
                "result_fidelity": "PROXY",
                "topology": {
                    "name": "ep16_r1",
                    "ep_size": 16,
                    "replica_count": 1,
                    "total_gpus": 16,
                },
                "workload": {
                    "prompt_tokens": 512,
                    "batch_size": 1,
                    "max_num_batched_tokens": 32_768,
                },
                "formal_prefill_latency_ms": 10.0,
                "proxy_latency_ms": 1.0,
                "proxy_record_count": 2,
                "latency_metrics": build_latency_metrics(
                    phase="prefill",
                    formal_prefill_latency_ms=10.0,
                ),
                "throughput": {
                    "aggregate_input_tok_s": 51_200.0,
                    "input_tok_s_per_gpu": 3_200.0,
                    "aggregate_output_tok_s": 0.0,
                    "aggregate_total_tok_s": 51_200.0,
                    "output_tok_s_per_gpu": 0.0,
                    "total_tok_s_per_gpu": 3_200.0,
                },
                "peak_memory_gib": {
                    "total": 218.0,
                    "weights": 211.0,
                    "activations": 3.0,
                    "kvcache": 0.5,
                    "nccl": 0.0,
                    "others": 3.5,
                },
                "memory_unavailable_reasons": {
                    "scales_only": "not separately modeled",
                    "workspace_only": "not separately modeled",
                },
                "kv_cache": {
                    "requested_dtype": "auto",
                    "resolved_dtype": "bfloat16",
                    "logical_bytes_per_gpu": 1_024.0,
                    "modeled_resident_allocated_bytes_per_gpu": 1_536.0,
                    "modeled_peak_allocated_bytes_per_gpu": 2_048.0,
                    "actual_runtime_allocated_bytes_per_gpu": None,
                    "actual_runtime_measurement_status": "PENDING_EXTERNAL_VLLM",
                    "actual_runtime_measurement_reason": "requires live CUDA counters",
                },
                "component_breakdown": {
                    "latency_ms": {
                        "full_mfa": 2.0,
                        "swa": 1.0,
                        "dense": 1.0,
                        "latent_moe_total": 5.0,
                        "dispatch": 0.5,
                        "combine": 0.5,
                        "other": 1.0,
                        "accounted_total": 10.0,
                    }
                },
                "moe_observability": {
                    "performance_workload_distribution": "power_law_1.2",
                    "expert_token_histogram": None,
                    "max_mean_load": None,
                    "padding_ratio": None,
                    "live_routing_status": "PENDING_EXTERNAL_VLLM",
                    "live_routing_reason": "requires live router outputs",
                },
                "oom": False,
                "backend_fallback": False,
                "retry_count": 0,
                "error_record_count": 0,
                "missing_record_count": 0,
                "exception_log": [],
            }
        ],
        "decode_results": [
            {
                "status": "PASS_WITH_PROXY",
                "result_fidelity": "PROXY",
                "topology": {
                    "name": "ep16_r1",
                    "ep_size": 16,
                    "replica_count": 1,
                    "total_gpus": 16,
                },
                "workload": {"context_tokens": 2_048, "output_tokens": 256},
                "tpot_budget_ms": 33.33,
                "b_max": 0,
                "aggregate_b_max": 0,
                "first_failed_batch": 1,
                "candidates": [
                    {
                        "global_batch_size_per_replica": 1,
                        "steady_decode_step_ms": 60.0,
                        "proxy_latency_ms": 2.0,
                        "proxy_record_count": 2,
                        "latency_metrics": build_latency_metrics(
                            phase="decode",
                            steady_decode_step_ms=60.0,
                        ),
                        "throughput": {
                            "aggregate_output_tok_s": 16.0,
                            "output_tok_s_per_gpu": 1.0,
                            "aggregate_input_tok_s": 0.0,
                            "aggregate_total_tok_s": 16.0,
                            "input_tok_s_per_gpu": 0.0,
                            "total_tok_s_per_gpu": 1.0,
                        },
                        "memory_gib": {
                            "total": 215.0,
                            "weights": 211.0,
                            "activations": 0.1,
                            "kvcache": 0.4,
                            "nccl": 0.0,
                            "others": 3.5,
                        },
                        "memory_unavailable_reasons": {
                            "scales_only": "not separately modeled",
                            "workspace_only": "not separately modeled",
                        },
                        "kv_cache": {
                            "requested_dtype": "auto",
                            "resolved_dtype": "bfloat16",
                            "logical_bytes_per_gpu": 4_096.0,
                            "modeled_resident_allocated_bytes_per_gpu": 6_144.0,
                            "modeled_peak_allocated_bytes_per_gpu": 8_192.0,
                            "actual_runtime_allocated_bytes_per_gpu": None,
                            "actual_runtime_measurement_status": "PENDING_EXTERNAL_VLLM",
                            "actual_runtime_measurement_reason": "requires live CUDA counters",
                        },
                        "component_breakdown": {
                            "latency_ms": {
                                "full_mfa": 10.0,
                                "swa": 8.0,
                                "dense": 3.0,
                                "latent_moe_total": 35.0,
                                "dispatch": 2.0,
                                "combine": 2.0,
                                "other": 4.0,
                                "accounted_total": 60.0,
                            }
                        },
                        "moe_observability": {
                            "performance_workload_distribution": "power_law_1.2",
                            "expert_token_histogram": None,
                            "max_mean_load": None,
                            "padding_ratio": None,
                            "live_routing_status": "PENDING_EXTERNAL_VLLM",
                            "live_routing_reason": "requires live router outputs",
                        },
                        "oom": False,
                        "backend_fallback": False,
                        "retry_count": 0,
                        "error_record_count": 0,
                        "missing_record_count": 0,
                        "exception_log": [],
                        "active_sequences_per_replica": 1,
                        "batched_tokens_per_replica": 1,
                    }
                ],
            }
        ],
    }


def test_repeat_audit_records_every_case_and_numeric_spread() -> None:
    payload = _sample_simulation_payload()
    third = copy.deepcopy(payload)
    third["prefill_results"][0]["formal_prefill_latency_ms"] = 11.0

    audit = build_repeat_audit([payload, copy.deepcopy(payload), third])

    assert audit["repeat_count"] == 3
    assert audit["case_count_per_repeat"] == 2
    assert audit["total_case_executions"] == 6
    assert audit["identical_full_result_count"] == 1
    assert audit["nonidentical_full_result_count"] == 1
    assert audit["max_abs_spread_by_metric"]["prefill_latency_ms"] == pytest.approx(1.0)


def test_repeated_payload_combiner_preserves_results_and_updates_execution_count() -> None:
    payload = _sample_simulation_payload()

    combined = combine_repeated_payloads([payload, copy.deepcopy(payload), copy.deepcopy(payload)])

    assert combined["repeat_audit"]["repeat_count"] == 3
    assert combined["repeat_audit"]["total_case_executions"] == 6
    assert combined["matrix_summary"]["repeat_count"] == 3
    assert combined["matrix_summary"]["total_case_executions"] == 6
    assert len(combined["prefill_results"]) == 1
    assert len(combined["decode_results"]) == 1


def test_review_rows_normalize_prefill_and_decode_for_manual_inspection() -> None:
    rows = build_review_rows(_sample_simulation_payload())

    assert len(rows) == 2
    assert rows[0]["phase"] == "prefill"
    assert rows[0]["topology"] == "ep16_r1"
    assert rows[0]["prefill_ms"] == pytest.approx(10.0)
    assert rows[0]["input_tok_s_per_gpu"] == pytest.approx(3_200.0)
    assert rows[0]["kv_requested_dtype"] == "auto"
    assert rows[0]["kv_logical_bytes_per_gpu"] == pytest.approx(1_024.0)
    assert rows[0]["full_mfa_ms"] == pytest.approx(2.0)
    assert rows[0]["backend_fallback"] is False
    assert rows[0]["retry_count"] == 0
    assert rows[0]["error_record_count"] == 0
    assert rows[0]["missing_record_count"] == 0
    assert rows[0]["exception_log"] == "[]"
    assert rows[0]["mtp1_iteration_ms"] is None
    assert rows[0]["mtp1_status"] == "DEFERRED_BY_OWNER"
    assert rows[1]["phase"] == "decode"
    assert rows[1]["selected_batch_per_replica"] == 1
    assert rows[1]["active_sequences_per_replica"] == 1
    assert rows[1]["batched_tokens_per_replica"] == 1
    assert rows[1]["steady_decode_step_p50_ms"] == pytest.approx(60.0)
    assert rows[1]["b_max"] == 0
    assert rows[1]["output_tok_s_per_gpu"] == pytest.approx(1.0)
    assert rows[1]["vllm_actual_status"] == "PENDING_EXTERNAL_VLLM"


def test_review_csv_uses_lf_line_endings(tmp_path: Path) -> None:
    rows = build_review_rows(_sample_simulation_payload())
    output = tmp_path / "review.csv"

    write_review_csv(rows, output)

    raw = output.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == len(rows) + 1


def test_requirements_workloads_cover_prefill_scans_decode_rows_and_topologies() -> None:
    workloads = build_requirements_workloads()

    assert [topology["name"] for topology in workloads["topologies"]] == [
        "ep16_r1",
        "ep32_r1",
        "ep16_r2",
    ]
    assert len(workloads["prefill"]) == 24
    assert len(workloads["decode"]) == 7
    assert {
        (case["prompt_tokens"], case["batch_size"], case["max_num_batched_tokens"])
        for case in workloads["prefill"]
        if case["prompt_tokens"] == 1_048_544 and case["batch_size"] == 1
    } == {
        (1_048_544, 1, 8_192),
        (1_048_544, 1, 32_768),
        (1_048_544, 1, 65_536),
    }
    assert workloads["decode"][0] == {
        "context_tokens": 2_048,
        "output_tokens": 256,
    }
    assert workloads["decode"][-1] == {
        "context_tokens": 1_048_544,
        "output_tokens": 32,
    }


def test_scheduled_query_records_missing_provider_without_stopping_later_ops() -> None:
    missing = GEMM(_name="context_qkv", _provider="vllm_step4pro_qkv_norm_rope")

    def raise_missing(database, **kwargs):
        raise PerfDataNotAvailableError("missing qkv table")

    missing.query = raise_missing
    later = GEMM(_name="context_later_gemm")

    records = query_scheduled_step(
        [missing, later],
        database=object(),
        phase="context",
        ep_size=16,
        local_batch_size=1,
        chunk_tokens_per_request=512,
        prefix_tokens=0,
        global_scheduled_tokens=512,
    )

    assert [record["status"] for record in records] == ["missing", "ok"]


def test_scheduled_query_keeps_exact_deepep_missing_by_default() -> None:
    dispatch = MoEDispatch()

    def raise_missing(database, **kwargs):
        raise PerfDataNotAvailableError("missing exact DeepEP table")

    dispatch.query = raise_missing
    records = query_scheduled_step(
        [dispatch],
        database=object(),
        phase="context",
        ep_size=16,
        local_batch_size=1,
        chunk_tokens_per_request=8,
        prefix_tokens=0,
        global_scheduled_tokens=8,
    )

    assert records[0]["status"] == "missing"
    assert records[0]["family"] == "deepep_ht"


def test_oom_uses_requirements_gpu_memory_utilization_limit() -> None:
    database = type(
        "Database",
        (),
        {"system_spec": {"gpu": {"mem_capacity": 100 * (1 << 30)}}},
    )()

    assert not _is_oom({"total": 89.99}, database)
    assert _is_oom({"total": 90.0}, database)
