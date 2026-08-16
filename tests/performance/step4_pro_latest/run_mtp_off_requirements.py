"""Run the Step4-Pro-Latest MTP-off requirements matrix in AIC.

This task-local driver maps instance-level request batches onto attention-DP
ranks, keeps ``max_num_batched_tokens`` as one global scheduler budget, and
queries the unchanged AIC operation consumers. Missing provider data blocks a
formal latency result; known partial latency and memory remain visible.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError
from aiconfigurator.sdk.perf_database import PerfDatabase
from tests.performance.step4_pro_latest.deepep_proxy import (
    B300_NCCL_ALLTOALL_PROXY,
    query_deepep_proxy,
)
from tests.performance.step4_pro_latest.validate_aic_silicon_coverage import (
    MODEL_ID,
    _family,
    _json_value,
    _physical_identity,
    build_latest_model,
)

SYSTEM = "b300_sxm"
BACKEND = "vllm"
FRAMEWORK_VERSION = "0.19.0"
GPU_MEMORY_UTILIZATION = 0.9
TPOT_BUDGETS_MS = (33.33, 20.0, 12.5, 10.0)
_MISSING_EXCEPTIONS = (
    PerfDataNotAvailableError,
    InterpolationDataNotAvailableError,
)
_PRIMARY_COMPONENTS = (
    "full_mfa",
    "swa",
    "dense",
    "latent_moe_compute",
    "dispatch",
    "combine",
    "other",
)


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def build_prefill_schedule(
    *,
    prompt_tokens: int,
    batch_size: int,
    max_num_batched_tokens: int,
    ep_size: int,
) -> list[dict[str, int]]:
    """Build equal-progress prefill steps under one global token budget."""
    prompt_tokens = _positive_int("prompt_tokens", prompt_tokens)
    batch_size = _positive_int("batch_size", batch_size)
    max_num_batched_tokens = _positive_int(
        "max_num_batched_tokens",
        max_num_batched_tokens,
    )
    ep_size = _positive_int("ep_size", ep_size)
    if max_num_batched_tokens < batch_size:
        raise ValueError(
            "The equal-progress requirements driver needs at least one token "
            "per active request: "
            f"max_num_batched_tokens={max_num_batched_tokens}, "
            f"batch_size={batch_size}"
        )

    steps: list[dict[str, int]] = []
    prefix_tokens = 0
    while prefix_tokens < prompt_tokens:
        chunk_tokens = min(
            prompt_tokens - prefix_tokens,
            max_num_batched_tokens // batch_size,
        )
        if chunk_tokens <= 0:
            raise RuntimeError("prefill scheduler made no progress")
        global_scheduled_tokens = batch_size * chunk_tokens
        busiest_rank_local_batch = math.ceil(batch_size / ep_size)
        completed_prefix = prefix_tokens + chunk_tokens
        steps.append(
            {
                "step_index": len(steps),
                "prefix_tokens": prefix_tokens,
                "chunk_tokens_per_request": chunk_tokens,
                "global_scheduled_tokens": global_scheduled_tokens,
                "busiest_rank": 0,
                "busiest_rank_local_batch": busiest_rank_local_batch,
                "busiest_rank_scheduled_tokens": busiest_rank_local_batch * chunk_tokens,
                "completed_requests": (batch_size if completed_prefix == prompt_tokens else 0),
            }
        )
        prefix_tokens = completed_prefix
    return steps


def _scheduled_query_kwargs(
    *,
    phase: str,
    family: str,
    local_batch_size: int,
    chunk_tokens_per_request: int,
    prefix_tokens: int,
) -> dict[str, Any]:
    if phase == "context":
        return {
            "x": (local_batch_size if family == "logits_gemm" else local_batch_size * chunk_tokens_per_request),
            "batch_size": local_batch_size,
            "beam_width": 1,
            "s": chunk_tokens_per_request,
            "prefix": prefix_tokens,
            "seq_imbalance_correction_scale": 1.0,
        }
    if phase == "generation":
        return {
            "x": local_batch_size,
            "batch_size": local_batch_size,
            "beam_width": 1,
            "s": prefix_tokens + chunk_tokens_per_request,
            "gen_seq_imbalance_correction_scale": 1.0,
        }
    raise ValueError(f"unsupported phase: {phase!r}")


def query_scheduled_step(
    operations: Iterable[object],
    *,
    database: object,
    phase: str,
    ep_size: int,
    local_batch_size: int,
    chunk_tokens_per_request: int,
    prefix_tokens: int,
    global_scheduled_tokens: int,
    deepep_proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Query one scheduler step while preserving local/global token meanings."""
    ep_size = _positive_int("ep_size", ep_size)
    local_batch_size = _positive_int("local_batch_size", local_batch_size)
    chunk_tokens_per_request = _positive_int(
        "chunk_tokens_per_request",
        chunk_tokens_per_request,
    )
    global_scheduled_tokens = _positive_int(
        "global_scheduled_tokens",
        global_scheduled_tokens,
    )
    if type(prefix_tokens) is not int or prefix_tokens < 0:
        raise ValueError(f"prefix_tokens must be a non-negative integer, got {prefix_tokens!r}")
    if deepep_proxy not in {None, B300_NCCL_ALLTOALL_PROXY}:
        raise ValueError(f"unsupported DeepEP proxy {deepep_proxy!r}")

    records: list[dict[str, Any]] = []
    for operation in operations:
        family = _family(operation)
        base_record = {
            "phase": phase,
            "ep_size": ep_size,
            "local_batch_size": local_batch_size,
            "chunk_tokens_per_request": chunk_tokens_per_request,
            "prefix_tokens": prefix_tokens,
            "global_scheduled_tokens": global_scheduled_tokens,
            "operation_name": getattr(
                operation,
                "_name",
                operation.__class__.__name__,
            ),
            "operation_class": operation.__class__.__name__,
            "family": family,
            "provider": _json_value(getattr(operation, "_provider", None)),
            "physical_identity": _physical_identity(
                operation,
                database,
                family,
            ),
        }
        try:
            direct_optimus_query = getattr(
                operation,
                "_query_step4_optimus",
                None,
            )
            if family == "optimus_moe" and callable(direct_optimus_query):
                result = direct_optimus_query(
                    database,
                    num_tokens=global_scheduled_tokens,
                )
            else:
                kwargs = _scheduled_query_kwargs(
                    phase=phase,
                    family=family,
                    local_batch_size=local_batch_size,
                    chunk_tokens_per_request=chunk_tokens_per_request,
                    prefix_tokens=prefix_tokens,
                )
                if "logits_gemm" in base_record["operation_name"]:
                    kwargs["x"] = local_batch_size
                if family == "deepep_ht" and deepep_proxy is not None:
                    proxy_result = query_deepep_proxy(
                        operation,
                        database,
                        tokens_per_dp_rank=kwargs["x"],
                        proxy_name=deepep_proxy,
                    )
                    records.append(
                        base_record
                        | {
                            "status": "proxy",
                            "source": proxy_result.source,
                            "result_fidelity": "PROXY",
                            "latency_ms": proxy_result.latency_ms,
                            "energy_wms": proxy_result.energy_wms,
                            "proxy": proxy_result.metadata,
                        }
                    )
                    continue
                result = operation.query(database, **kwargs)
        except _MISSING_EXCEPTIONS as error:
            records.append(
                base_record
                | {
                    "status": "missing",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        except Exception as error:
            records.append(
                base_record
                | {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue

        source = getattr(result, "source", "unknown")
        records.append(
            base_record
            | {
                "status": "ok" if source == "silicon" else "non_silicon",
                "source": source,
                "latency_ms": float(result),
                "energy_wms": float(getattr(result, "energy", 0.0)),
            }
        )
    return records


def _component_for_record(record: dict[str, Any]) -> str:
    operation_name = str(record.get("operation_name", ""))
    if "_latent_moe_dispatch" in operation_name:
        return "dispatch"
    if "_latent_moe_combine" in operation_name:
        return "combine"
    if "_latent_moe_" in operation_name:
        return "latent_moe_compute"
    if "_dense_" in operation_name:
        return "dense"
    if "_full_" in operation_name:
        return "full_mfa"
    if "_swa_" in operation_name:
        return "swa"
    return "other"


def build_component_latency_breakdown(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Group one Step4 scheduler step into auditable model components."""
    latency_ms = dict.fromkeys(_PRIMARY_COMPONENTS, 0.0)
    record_count = dict.fromkeys(_PRIMARY_COMPONENTS, 0)
    proxy_record_count = dict.fromkeys(_PRIMARY_COMPONENTS, 0)
    missing_record_count = dict.fromkeys(_PRIMARY_COMPONENTS, 0)
    error_record_count = dict.fromkeys(_PRIMARY_COMPONENTS, 0)

    for record in records:
        component = _component_for_record(record)
        status = record["status"]
        record_count[component] += 1
        if status in {"ok", "non_silicon", "proxy"}:
            latency_ms[component] += float(record.get("latency_ms", 0.0))
        if status == "proxy":
            proxy_record_count[component] += 1
        elif status == "missing":
            missing_record_count[component] += 1
        elif status == "error":
            error_record_count[component] += 1

    latency_ms["latent_moe_total"] = sum(
        latency_ms[component] for component in ("latent_moe_compute", "dispatch", "combine")
    )
    latency_ms["accounted_total"] = sum(latency_ms[component] for component in _PRIMARY_COMPONENTS)
    return {
        "latency_ms": latency_ms,
        "record_count": record_count,
        "proxy_record_count": proxy_record_count,
        "missing_record_count": missing_record_count,
        "error_record_count": error_record_count,
        "definitions": {
            "full_mfa": "Operations whose pinned graph name contains '_full_'.",
            "swa": "Operations whose pinned graph name contains '_swa_'.",
            "dense": "Operations whose pinned graph name contains '_dense_'.",
            "latent_moe_compute": ("Latent-MoE operations excluding dispatch and combine."),
            "dispatch": "Latent-MoE dispatch transport.",
            "combine": "Latent-MoE combine transport.",
            "latent_moe_total": ("latent_moe_compute + dispatch + combine; not added again to accounted_total."),
            "other": (
                "Embedding, final norm/logits, and common residual operations "
                "that do not encode an attention or FFN type in their name."
            ),
        },
    }


def merge_component_latency_breakdowns(
    breakdowns: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Merge per-step component summaries into one workload summary."""
    merged = build_component_latency_breakdown([])
    for breakdown in breakdowns:
        for component in _PRIMARY_COMPONENTS:
            merged["latency_ms"][component] += float(breakdown["latency_ms"][component])
            for count_name in (
                "record_count",
                "proxy_record_count",
                "missing_record_count",
                "error_record_count",
            ):
                merged[count_name][component] += int(breakdown[count_name][component])
    merged["latency_ms"]["latent_moe_total"] = sum(
        merged["latency_ms"][component] for component in ("latent_moe_compute", "dispatch", "combine")
    )
    merged["latency_ms"]["accounted_total"] = sum(merged["latency_ms"][component] for component in _PRIMARY_COMPONENTS)
    return merged


def summarize_scheduled_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one queried step without manufacturing missing latency."""
    missing_contracts: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["status"] != "missing":
            continue
        identity = json.dumps(record["physical_identity"], sort_keys=True)
        if identity not in missing_contracts:
            missing_contracts[identity] = {
                "family": record["family"],
                "provider": record["provider"],
                "physical_identity": record["physical_identity"],
                "record_count": 0,
                "error_type": record["error_type"],
                "error": record["error"],
            }
        missing_contracts[identity]["record_count"] += 1

    missing_count = sum(record["status"] == "missing" for record in records)
    error_count = sum(record["status"] == "error" for record in records)
    non_silicon_count = sum(record["status"] == "non_silicon" for record in records)
    proxy_count = sum(record["status"] == "proxy" for record in records)
    known_partial_latency_ms = sum(
        record.get("latency_ms", 0.0) for record in records if record["status"] in {"ok", "non_silicon", "proxy"}
    )
    exact_silicon_latency_ms = sum(record.get("latency_ms", 0.0) for record in records if record["status"] == "ok")
    proxy_latency_ms = sum(record.get("latency_ms", 0.0) for record in records if record["status"] == "proxy")
    exception_log = [
        {
            "status": record["status"],
            "operation_name": record.get("operation_name"),
            "operation_class": record.get("operation_class"),
            "family": record.get("family"),
            "provider": record.get("provider"),
            "error_type": record.get("error_type"),
            "error": record.get("error"),
        }
        for record in records
        if record["status"] in {"missing", "error"}
    ]
    if missing_count or error_count:
        status = "BLOCKED"
        formal_latency_ms = None
    elif proxy_count:
        status = "PASS_WITH_PROXY"
        formal_latency_ms = known_partial_latency_ms
    elif non_silicon_count:
        status = "PASS_WITH_NON_SILICON"
        formal_latency_ms = known_partial_latency_ms
    else:
        status = "PASS"
        formal_latency_ms = known_partial_latency_ms
    return {
        "status": status,
        "result_fidelity": ("PROXY" if proxy_count else ("MIXED_NON_SILICON" if non_silicon_count else "SILICON")),
        "formal_latency_ms": formal_latency_ms,
        "known_partial_latency_ms": known_partial_latency_ms,
        "exact_silicon_latency_ms": exact_silicon_latency_ms,
        "proxy_latency_ms": proxy_latency_ms,
        "record_count": len(records),
        "exact_silicon_record_count": sum(record["status"] == "ok" for record in records),
        "non_silicon_record_count": non_silicon_count,
        "proxy_record_count": proxy_count,
        "missing_record_count": missing_count,
        "error_record_count": error_count,
        "exception_log": exception_log,
        "missing_physical_contract_count": len(missing_contracts),
        "missing_physical_contracts": [missing_contracts[key] for key in sorted(missing_contracts)],
        "component_breakdown": build_component_latency_breakdown(records),
    }


def merge_execution_observability(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Merge fail-fast query status without treating an explicit proxy as fallback."""
    return {
        "backend_fallback": False,
        "retry_count": 0,
        "error_record_count": sum(int(summary["error_record_count"]) for summary in summaries),
        "missing_record_count": sum(int(summary["missing_record_count"]) for summary in summaries),
        "exception_log": [exception for summary in summaries for exception in summary["exception_log"]],
    }


def build_requirements_workloads() -> dict[str, Any]:
    """Return the exact MTP-off matrix from the requirements document."""
    topology_rows = [
        {"name": "ep16_r1", "ep_size": 16, "replica_count": 1, "total_gpus": 16},
        {"name": "ep32_r1", "ep_size": 32, "replica_count": 1, "total_gpus": 32},
        {"name": "ep16_r2", "ep_size": 16, "replica_count": 2, "total_gpus": 32},
    ]
    prefill_batches = {
        512: (1, 8, 32),
        2_048: (1, 4, 16),
        8_192: (1, 4, 8),
        32_768: (1, 2, 4),
        131_072: (1, 2),
        262_144: (1, 2),
        524_288: (1,),
        1_048_544: (1,),
    }
    prefill_rows = [
        {
            "prompt_tokens": prompt_tokens,
            "batch_size": batch_size,
            "max_num_batched_tokens": 32_768,
            "workload_source": "requirements_table",
        }
        for prompt_tokens, batch_sizes in prefill_batches.items()
        for batch_size in batch_sizes
    ]
    for prompt_tokens in (32_768, 131_072, 1_048_544):
        for max_num_batched_tokens in (8_192, 65_536):
            prefill_rows.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "batch_size": 1,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "workload_source": "required_chunk_scan",
                }
            )
    prefill_rows.sort(
        key=lambda row: (
            row["prompt_tokens"],
            row["batch_size"],
            row["max_num_batched_tokens"],
        )
    )
    decode_rows = [
        {"context_tokens": 2_048, "output_tokens": 256},
        {"context_tokens": 8_192, "output_tokens": 256},
        {"context_tokens": 32_768, "output_tokens": 256},
        {"context_tokens": 131_072, "output_tokens": 256},
        {"context_tokens": 262_144, "output_tokens": 128},
        {"context_tokens": 524_288, "output_tokens": 64},
        {"context_tokens": 1_048_544, "output_tokens": 32},
    ]
    return {
        "topologies": topology_rows,
        "prefill": prefill_rows,
        "decode": decode_rows,
        "tpot_budgets_ms": list(TPOT_BUDGETS_MS),
    }


def _merge_missing_contracts(
    target: dict[str, dict[str, Any]],
    contracts: Sequence[dict[str, Any]],
) -> None:
    for contract in contracts:
        identity = json.dumps(contract["physical_identity"], sort_keys=True)
        if identity not in target:
            target[identity] = dict(contract)
        else:
            target[identity]["record_count"] += contract["record_count"]


def _is_oom(memory: dict[str, float], database: PerfDatabase) -> bool:
    capacity_gib = database.system_spec["gpu"]["mem_capacity"] / (1 << 30)
    return memory["total"] >= capacity_gib * GPU_MEMORY_UTILIZATION


def build_kv_cache_details(
    model: object,
    *,
    local_batch_size: int,
    sequence_length: int,
    in_flight_tokens: int,
) -> dict[str, Any]:
    """Expose logical and pinned-page KV accounting for one AIC worker."""
    local_batch_size = _positive_int("local_batch_size", local_batch_size)
    if type(sequence_length) is not int or sequence_length < 0:
        raise ValueError(f"sequence_length must be a non-negative integer, got {sequence_length!r}")
    in_flight_tokens = _positive_int("in_flight_tokens", in_flight_tokens)

    requested_dtype = getattr(model, "kv_cache_requested_dtype", None)
    resolved_dtype = getattr(model, "kv_cache_resolved_dtype", None)
    extra_params = getattr(model, "extra_params", None)
    layout = getattr(extra_params, "kv_cache_layout", None)
    if (requested_dtype, resolved_dtype, layout) != (
        "auto",
        "bfloat16",
        "NHD",
    ):
        raise ValueError(
            "Step4-Pro-Latest KV details require the pinned "
            "auto->bfloat16 NHD contract, got "
            f"{requested_dtype!r}->{resolved_dtype!r} {layout!r}"
        )

    logical_per_sequence = float(model.get_kvcache_bytes_per_sequence(sequence_length))
    resident_per_sequence = float(model.get_kvcache_allocated_bytes_per_sequence(sequence_length))
    peak_per_sequence = float(
        model.get_kvcache_peak_allocated_bytes_per_sequence(
            sequence_length,
            in_flight_tokens=in_flight_tokens,
        )
    )
    cp_divisor = float(model._cp_kv_memory_divisor())
    return {
        "requested_dtype": requested_dtype,
        "resolved_dtype": resolved_dtype,
        "layout": layout,
        "page_size_tokens": getattr(model, "kv_cache_page_size", None),
        "sequence_length": sequence_length,
        "in_flight_tokens": in_flight_tokens,
        "local_sequence_count": local_batch_size,
        "cp_memory_divisor": cp_divisor,
        "logical_bytes_per_sequence": logical_per_sequence,
        "modeled_resident_allocated_bytes_per_sequence": resident_per_sequence,
        "modeled_peak_allocated_bytes_per_sequence": peak_per_sequence,
        "logical_bytes_per_gpu": (local_batch_size * logical_per_sequence / cp_divisor),
        "modeled_resident_allocated_bytes_per_gpu": (local_batch_size * resident_per_sequence / cp_divisor),
        "modeled_peak_allocated_bytes_per_gpu": (local_batch_size * peak_per_sequence / cp_divisor),
        "actual_runtime_allocated_bytes_per_gpu": None,
        "actual_runtime_measurement_status": "PENDING_EXTERNAL_VLLM",
        "actual_runtime_measurement_reason": (
            "The AIC-only simulation models the pinned vLLM page allocator "
            "but cannot read live CUDA allocation counters."
        ),
    }


def build_latency_metrics(
    *,
    phase: str,
    formal_prefill_latency_ms: float | None = None,
    steady_decode_step_ms: float | None = None,
) -> dict[str, Any]:
    """Return one stable latency schema without inventing live percentiles."""
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unsupported latency phase {phase!r}")
    if phase == "prefill" and steady_decode_step_ms is not None:
        raise ValueError("prefill latency metrics cannot contain a decode step")
    if phase == "decode" and formal_prefill_latency_ms is not None:
        raise ValueError("decode latency metrics cannot contain prefill latency")

    values_ms: dict[str, float | None] = {
        "ttft": None,
        "prefill": (None if formal_prefill_latency_ms is None else float(formal_prefill_latency_ms)),
        "first_decode_step": None,
        "steady_decode_step_p50": (None if steady_decode_step_ms is None else float(steady_decode_step_ms)),
        "steady_decode_step_p90": None,
        "steady_decode_step_p99": None,
        "itl_p50": None,
        "itl_p90": None,
        "itl_p99": None,
        "tpot": (None if steady_decode_step_ms is None else float(steady_decode_step_ms)),
        "decode_generation": None,
        "end_to_end": None,
    }
    engine_reason = (
        "Requires a live request lifecycle and engine scheduler trace; the "
        "task-local AIC operation-sum simulation does not model it."
    )
    distribution_reason = (
        "Requires repeated live engine-step timing samples; deterministic AIC "
        "operation sums do not provide a latency distribution."
    )
    unavailable_reasons = {
        "ttft": engine_reason,
        "first_decode_step": engine_reason,
        "steady_decode_step_p90": distribution_reason,
        "steady_decode_step_p99": distribution_reason,
        "itl_p50": distribution_reason,
        "itl_p90": distribution_reason,
        "itl_p99": distribution_reason,
        "decode_generation": engine_reason,
        "end_to_end": engine_reason,
    }
    if phase == "prefill":
        unavailable_reasons["steady_decode_step_p50"] = "Not part of a prefill-only simulation result."
        unavailable_reasons["tpot"] = "Not part of a prefill-only simulation result."
    else:
        unavailable_reasons["prefill"] = "Decode candidates intentionally exclude prefill."
    return {
        "status": "PARTIAL_SIMULATOR_ONLY",
        "values_ms": values_ms,
        "unavailable_reasons": unavailable_reasons,
        "steady_decode_step_p50_semantics": (
            "Deterministic AIC point estimate used as the p50 alignment target; not a sampled percentile."
            if phase == "decode" and steady_decode_step_ms is not None
            else None
        ),
    }


def build_throughput_metrics(
    *,
    input_tokens_per_replica: int,
    output_tokens_per_replica: int,
    latency_ms: float | None,
    replica_count: int,
    total_gpus: int,
) -> dict[str, float | None]:
    """Normalize per-replica, aggregate, and per-GPU token rates."""
    for name, value in (
        ("input_tokens_per_replica", input_tokens_per_replica),
        ("output_tokens_per_replica", output_tokens_per_replica),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    replica_count = _positive_int("replica_count", replica_count)
    total_gpus = _positive_int("total_gpus", total_gpus)
    if latency_ms is None:
        rate_scale = None
    else:
        if latency_ms <= 0:
            raise ValueError(f"latency_ms must be positive, got {latency_ms!r}")
        rate_scale = 1000.0 / float(latency_ms)

    input_per_replica = None if rate_scale is None else input_tokens_per_replica * rate_scale
    output_per_replica = None if rate_scale is None else output_tokens_per_replica * rate_scale
    total_per_replica = (
        None if rate_scale is None else (input_tokens_per_replica + output_tokens_per_replica) * rate_scale
    )
    aggregate_input = None if input_per_replica is None else input_per_replica * replica_count
    aggregate_output = None if output_per_replica is None else output_per_replica * replica_count
    aggregate_total = None if total_per_replica is None else total_per_replica * replica_count
    return {
        "input_tok_s_per_replica": input_per_replica,
        "output_tok_s_per_replica": output_per_replica,
        "total_tok_s_per_replica": total_per_replica,
        "aggregate_input_tok_s": aggregate_input,
        "aggregate_output_tok_s": aggregate_output,
        "aggregate_total_tok_s": aggregate_total,
        "input_tok_s_per_gpu": (None if aggregate_input is None else aggregate_input / total_gpus),
        "output_tok_s_per_gpu": (None if aggregate_output is None else aggregate_output / total_gpus),
        "total_tok_s_per_gpu": (None if aggregate_total is None else aggregate_total / total_gpus),
    }


def build_moe_observability(model: object) -> dict[str, Any]:
    """Expose the modeled MoE workload while leaving live routing unset."""
    distributions = {
        operation._workload_distribution
        for operation in getattr(model, "context_ops", ())
        if _family(operation) == "optimus_moe" and getattr(operation, "_workload_distribution", None) is not None
    }
    if len(distributions) != 1:
        raise ValueError(
            f"Step4-Pro-Latest requires one Optimus MoE workload distribution, got {sorted(distributions)!r}"
        )
    return {
        "performance_workload_distribution": next(iter(distributions)),
        "performance_workload_semantics": (
            "Collector/simulator performance-shape assumption; not a live request routing observation."
        ),
        "expert_token_histogram": None,
        "max_mean_load": None,
        "padding_ratio": None,
        "live_routing_status": "PENDING_EXTERNAL_VLLM",
        "live_routing_reason": (
            "Dummy-weight AIC simulation has no request-level router outputs. "
            "Histogram, load ratio, and padding require the external pinned "
            "vLLM runtime trace."
        ),
    }


def _prefill_step_memory(
    *,
    model: object,
    backend: object,
    database: PerfDatabase,
    step: dict[str, int],
) -> dict[str, float]:
    return backend._get_memory_usage(
        model,
        database,
        step["busiest_rank_local_batch"],
        1,
        step["prefix_tokens"] + step["chunk_tokens_per_request"],
        1,
        num_tokens=step["busiest_rank_scheduled_tokens"],
        prefix=step["prefix_tokens"],
        max_seq_len=(step["prefix_tokens"] + step["chunk_tokens_per_request"]),
        kv_in_flight_tokens=step["chunk_tokens_per_request"],
    )


def simulate_prefill_case(
    *,
    model: object,
    backend: object,
    database: PerfDatabase,
    topology: dict[str, Any],
    workload: dict[str, Any],
    deepep_proxy: str | None = None,
) -> dict[str, Any]:
    schedule = build_prefill_schedule(
        prompt_tokens=workload["prompt_tokens"],
        batch_size=workload["batch_size"],
        max_num_batched_tokens=workload["max_num_batched_tokens"],
        ep_size=topology["ep_size"],
    )
    queried_steps: list[dict[str, Any]] = []
    missing_contracts: dict[str, dict[str, Any]] = {}
    cumulative_formal_latency_ms = 0.0
    cumulative_known_partial_latency_ms = 0.0
    cumulative_proxy_latency_ms = 0.0
    proxy_record_count = 0
    peak_memory: dict[str, float] | None = None
    peak_kv_cache: dict[str, Any] | None = None
    blocked = False

    for step in schedule:
        memory = _prefill_step_memory(
            model=model,
            backend=backend,
            database=database,
            step=step,
        )
        kv_cache = build_kv_cache_details(
            model,
            local_batch_size=step["busiest_rank_local_batch"],
            sequence_length=(step["prefix_tokens"] + step["chunk_tokens_per_request"]),
            in_flight_tokens=step["chunk_tokens_per_request"],
        )
        if peak_memory is None or memory["total"] > peak_memory["total"]:
            peak_memory = memory
            peak_kv_cache = kv_cache
        if blocked:
            continue

        records = query_scheduled_step(
            model.context_ops,
            database=database,
            phase="context",
            ep_size=topology["ep_size"],
            local_batch_size=step["busiest_rank_local_batch"],
            chunk_tokens_per_request=step["chunk_tokens_per_request"],
            prefix_tokens=step["prefix_tokens"],
            global_scheduled_tokens=step["global_scheduled_tokens"],
            deepep_proxy=deepep_proxy,
        )
        query_summary = summarize_scheduled_records(records)
        queried_steps.append(
            {
                **step,
                "memory_gib": memory,
                "kv_cache": kv_cache,
                "query_summary": query_summary,
            }
        )
        cumulative_known_partial_latency_ms += query_summary["known_partial_latency_ms"]
        cumulative_proxy_latency_ms += query_summary["proxy_latency_ms"]
        proxy_record_count += query_summary["proxy_record_count"]
        if query_summary["formal_latency_ms"] is None:
            blocked = True
            _merge_missing_contracts(
                missing_contracts,
                query_summary["missing_physical_contracts"],
            )
        else:
            cumulative_formal_latency_ms += query_summary["formal_latency_ms"]

    assert peak_memory is not None
    assert peak_kv_cache is not None
    oom = _is_oom(peak_memory, database)
    formal_latency_ms = None if blocked else cumulative_formal_latency_ms
    input_tokens = workload["prompt_tokens"] * workload["batch_size"]
    throughput = build_throughput_metrics(
        input_tokens_per_replica=input_tokens,
        output_tokens_per_replica=0,
        latency_ms=formal_latency_ms,
        replica_count=topology["replica_count"],
        total_gpus=topology["total_gpus"],
    )
    component_breakdown = merge_component_latency_breakdowns(
        [step_result["query_summary"]["component_breakdown"] for step_result in queried_steps]
    )
    execution_observability = merge_execution_observability(
        [step_result["query_summary"] for step_result in queried_steps]
    )
    result_fidelity = "PROXY" if proxy_record_count else "MIXED_NON_SILICON"
    return {
        "status": ("BLOCKED" if blocked else ("OOM" if oom else ("PASS_WITH_PROXY" if proxy_record_count else "PASS"))),
        "result_fidelity": result_fidelity,
        "topology": topology,
        "workload": workload,
        "chunk_count": len(schedule),
        "queried_chunk_count": len(queried_steps),
        "unqueried_chunk_count_after_blocker": len(schedule) - len(queried_steps),
        "formal_prefill_latency_ms": formal_latency_ms,
        "known_partial_latency_ms": cumulative_known_partial_latency_ms,
        "latency_metrics": build_latency_metrics(
            phase="prefill",
            formal_prefill_latency_ms=formal_latency_ms,
        ),
        "component_breakdown": component_breakdown,
        "proxy_latency_ms": cumulative_proxy_latency_ms,
        "proxy_record_count": proxy_record_count,
        "input_tok_s_per_replica": throughput["input_tok_s_per_replica"],
        "aggregate_input_tok_s": throughput["aggregate_input_tok_s"],
        "throughput": throughput,
        "peak_memory_gib": peak_memory,
        "memory_metrics_gib": {
            "peak_hbm": peak_memory["total"],
            "weights_including_scales": peak_memory["weights"],
            "scales_only": None,
            "activations_including_workspace": peak_memory["activations"],
            "workspace_only": None,
            "kv_peak_allocated": peak_memory["kvcache"],
            "nccl": peak_memory["nccl"],
            "other_runtime": peak_memory["others"],
        },
        "memory_unavailable_reasons": {
            "scales_only": (
                "The current AIC weight accounting includes FP8 scales in the "
                "weight total and does not expose them separately."
            ),
            "workspace_only": (
                "The current AIC activation accounting includes the MoE workspace and does not expose it separately."
            ),
        },
        "kv_cache": peak_kv_cache,
        "moe_observability": build_moe_observability(model),
        **execution_observability,
        "gpu_capacity_gib": database.system_spec["gpu"]["mem_capacity"] / (1 << 30),
        "gpu_memory_utilization_limit_gib": database.system_spec["gpu"]["mem_capacity"]
        / (1 << 30)
        * GPU_MEMORY_UTILIZATION,
        "oom": oom,
        "missing_physical_contract_count": len(missing_contracts),
        "missing_physical_contracts": [missing_contracts[key] for key in sorted(missing_contracts)],
        "queried_steps": queried_steps,
    }


def _decode_candidate(
    *,
    model: object,
    backend: object,
    database: PerfDatabase,
    topology: dict[str, Any],
    workload: dict[str, int],
    global_batch_size: int,
    tpot_budget_ms: float,
    deepep_proxy: str | None = None,
) -> dict[str, Any]:
    local_batch_size = math.ceil(global_batch_size / topology["ep_size"])
    steady_sequence_length = workload["context_tokens"] + max(workload["output_tokens"] // 2, 1)
    records = query_scheduled_step(
        model.generation_ops,
        database=database,
        phase="generation",
        ep_size=topology["ep_size"],
        local_batch_size=local_batch_size,
        chunk_tokens_per_request=1,
        prefix_tokens=steady_sequence_length - 1,
        global_scheduled_tokens=global_batch_size,
        deepep_proxy=deepep_proxy,
    )
    query_summary = summarize_scheduled_records(records)
    memory = backend._get_memory_usage(
        model,
        database,
        local_batch_size,
        1,
        workload["context_tokens"],
        workload["output_tokens"],
        num_tokens=local_batch_size,
        max_seq_len=workload["context_tokens"] + workload["output_tokens"],
        kv_in_flight_tokens=1,
    )
    kv_cache = build_kv_cache_details(
        model,
        local_batch_size=local_batch_size,
        sequence_length=workload["context_tokens"] + workload["output_tokens"],
        in_flight_tokens=1,
    )
    oom = _is_oom(memory, database)
    steady_decode_step_ms = query_summary["formal_latency_ms"]
    meets_tpot = steady_decode_step_ms is not None and not oom and steady_decode_step_ms <= tpot_budget_ms
    throughput = build_throughput_metrics(
        input_tokens_per_replica=0,
        output_tokens_per_replica=global_batch_size,
        latency_ms=steady_decode_step_ms,
        replica_count=topology["replica_count"],
        total_gpus=topology["total_gpus"],
    )
    execution_observability = merge_execution_observability([query_summary])
    return {
        "global_batch_size_per_replica": global_batch_size,
        "aggregate_active_sequences": global_batch_size * topology["replica_count"],
        "active_sequences_per_replica": global_batch_size,
        "batched_tokens_per_replica": global_batch_size,
        "aggregate_batched_tokens": global_batch_size * topology["replica_count"],
        "busiest_rank_local_batch": local_batch_size,
        "steady_sequence_length": steady_sequence_length,
        "steady_decode_step_ms": steady_decode_step_ms,
        "latency_metrics": build_latency_metrics(
            phase="decode",
            steady_decode_step_ms=steady_decode_step_ms,
        ),
        "component_breakdown": query_summary["component_breakdown"],
        "result_fidelity": query_summary["result_fidelity"],
        "known_partial_latency_ms": query_summary["known_partial_latency_ms"],
        "proxy_latency_ms": query_summary["proxy_latency_ms"],
        "proxy_record_count": query_summary["proxy_record_count"],
        "effective_tpot_ms": steady_decode_step_ms,
        "output_tok_s_per_replica": throughput["output_tok_s_per_replica"],
        "throughput": throughput,
        "memory_gib": memory,
        "memory_metrics_gib": {
            "peak_hbm": memory["total"],
            "weights_including_scales": memory["weights"],
            "scales_only": None,
            "activations_including_workspace": memory["activations"],
            "workspace_only": None,
            "kv_peak_allocated": memory["kvcache"],
            "nccl": memory["nccl"],
            "other_runtime": memory["others"],
        },
        "memory_unavailable_reasons": {
            "scales_only": (
                "The current AIC weight accounting includes FP8 scales in the "
                "weight total and does not expose them separately."
            ),
            "workspace_only": (
                "The current AIC activation accounting includes the MoE workspace and does not expose it separately."
            ),
        },
        "kv_cache": kv_cache,
        "moe_observability": build_moe_observability(model),
        **execution_observability,
        "oom": oom,
        "meets_tpot_budget": meets_tpot,
        "query_summary": query_summary,
    }


def simulate_decode_budget(
    *,
    model: object,
    backend: object,
    database: PerfDatabase,
    topology: dict[str, Any],
    workload: dict[str, int],
    tpot_budget_ms: float,
    max_search_batch: int,
    deepep_proxy: str | None = None,
) -> dict[str, Any]:
    if tpot_budget_ms <= 0:
        raise ValueError(f"tpot_budget_ms must be positive, got {tpot_budget_ms!r}")
    max_search_batch = _positive_int("max_search_batch", max_search_batch)
    candidates: list[dict[str, Any]] = []
    batch_size = 1
    last_pass = 0
    first_fail: int | None = None

    while batch_size <= max_search_batch:
        candidate = _decode_candidate(
            model=model,
            backend=backend,
            database=database,
            topology=topology,
            workload=workload,
            global_batch_size=batch_size,
            tpot_budget_ms=tpot_budget_ms,
            deepep_proxy=deepep_proxy,
        )
        candidates.append(candidate)
        if candidate["query_summary"]["status"] == "BLOCKED":
            return {
                "status": "BLOCKED",
                "topology": topology,
                "workload": workload,
                "tpot_budget_ms": tpot_budget_ms,
                "b_max": None,
                "first_failed_batch": None,
                "candidates": candidates,
                "result_fidelity": candidate["result_fidelity"],
                "proxy_record_count": candidate["proxy_record_count"],
                "proxy_latency_ms": candidate["proxy_latency_ms"],
                "missing_physical_contracts": candidate["query_summary"]["missing_physical_contracts"],
            }
        if candidate["meets_tpot_budget"]:
            last_pass = batch_size
            batch_size *= 2
            continue
        first_fail = batch_size
        break

    if first_fail is None:
        proxy_record_count = sum(candidate["proxy_record_count"] for candidate in candidates)
        proxy_latency_ms = sum(candidate["proxy_latency_ms"] for candidate in candidates)
        return {
            "status": "SEARCH_LIMIT",
            "result_fidelity": "PROXY" if proxy_record_count else "MIXED_NON_SILICON",
            "topology": topology,
            "workload": workload,
            "tpot_budget_ms": tpot_budget_ms,
            "b_max": last_pass or None,
            "first_failed_batch": None,
            "candidates": candidates,
            "proxy_record_count": proxy_record_count,
            "proxy_latency_ms": proxy_latency_ms,
            "missing_physical_contracts": [],
        }

    low = last_pass + 1
    high = first_fail - 1
    while low <= high:
        candidate_batch = (low + high) // 2
        candidate = _decode_candidate(
            model=model,
            backend=backend,
            database=database,
            topology=topology,
            workload=workload,
            global_batch_size=candidate_batch,
            tpot_budget_ms=tpot_budget_ms,
            deepep_proxy=deepep_proxy,
        )
        candidates.append(candidate)
        if candidate["meets_tpot_budget"]:
            last_pass = candidate_batch
            low = candidate_batch + 1
        else:
            first_fail = candidate_batch
            high = candidate_batch - 1
    candidates.sort(key=lambda item: item["global_batch_size_per_replica"])
    proxy_record_count = sum(candidate["proxy_record_count"] for candidate in candidates)
    proxy_latency_ms = sum(candidate["proxy_latency_ms"] for candidate in candidates)
    return {
        "status": "PASS_WITH_PROXY" if proxy_record_count else "PASS",
        "result_fidelity": "PROXY" if proxy_record_count else "MIXED_NON_SILICON",
        "topology": topology,
        "workload": workload,
        "tpot_budget_ms": tpot_budget_ms,
        "b_max": last_pass,
        "aggregate_b_max": last_pass * topology["replica_count"],
        "first_failed_batch": first_fail,
        "candidates": candidates,
        "proxy_record_count": proxy_record_count,
        "proxy_latency_ms": proxy_latency_ms,
        "missing_physical_contracts": [],
    }


def _base_review_row(
    *,
    phase: str,
    result: dict[str, Any],
    latency_metrics: dict[str, Any],
    throughput: dict[str, Any],
    memory: dict[str, Any],
    kv_cache: dict[str, Any],
    component_breakdown: dict[str, Any],
    moe_observability: dict[str, Any],
    memory_unavailable_reasons: dict[str, str],
    execution_observability: dict[str, Any],
) -> dict[str, Any]:
    topology = result["topology"]
    latency_values = latency_metrics["values_ms"]
    component_latency = component_breakdown["latency_ms"]
    return {
        "phase": phase,
        "topology": topology["name"],
        "ep_size": topology["ep_size"],
        "replica_count": topology["replica_count"],
        "total_gpus": topology["total_gpus"],
        "variant": "mtp_off",
        "status": result["status"],
        "result_fidelity": result["result_fidelity"],
        "ttft_ms": latency_values["ttft"],
        "prefill_ms": latency_values["prefill"],
        "first_decode_step_ms": latency_values["first_decode_step"],
        "steady_decode_step_p50_ms": latency_values["steady_decode_step_p50"],
        "steady_decode_step_p90_ms": latency_values["steady_decode_step_p90"],
        "steady_decode_step_p99_ms": latency_values["steady_decode_step_p99"],
        "itl_p50_ms": latency_values["itl_p50"],
        "itl_p90_ms": latency_values["itl_p90"],
        "itl_p99_ms": latency_values["itl_p99"],
        "tpot_ms": latency_values["tpot"],
        "decode_generation_ms": latency_values["decode_generation"],
        "end_to_end_ms": latency_values["end_to_end"],
        "latency_unavailable_reasons": json.dumps(
            latency_metrics["unavailable_reasons"],
            sort_keys=True,
        ),
        "mtp1_iteration_ms": None,
        "mtp1_status": "DEFERRED_BY_OWNER",
        "mtp1_reason": (
            "The pinned Step4Pro source has no native MTP1 implementation; "
            "the task owner deferred MTP1 construction, measurement, and simulation."
        ),
        "aggregate_input_tok_s": throughput["aggregate_input_tok_s"],
        "aggregate_output_tok_s": throughput["aggregate_output_tok_s"],
        "aggregate_total_tok_s": throughput["aggregate_total_tok_s"],
        "input_tok_s_per_gpu": throughput["input_tok_s_per_gpu"],
        "output_tok_s_per_gpu": throughput["output_tok_s_per_gpu"],
        "total_tok_s_per_gpu": throughput["total_tok_s_per_gpu"],
        "weights_including_scales_gib": memory["weights"],
        "scales_only_gib": None,
        "activations_including_workspace_gib": memory["activations"],
        "workspace_only_gib": None,
        "memory_unavailable_reasons": json.dumps(
            memory_unavailable_reasons,
            sort_keys=True,
        ),
        "kv_peak_allocated_gib": memory["kvcache"],
        "nccl_gib": memory["nccl"],
        "other_runtime_gib": memory["others"],
        "peak_hbm_gib": memory["total"],
        "kv_requested_dtype": kv_cache["requested_dtype"],
        "kv_resolved_dtype": kv_cache["resolved_dtype"],
        "kv_logical_bytes_per_gpu": kv_cache["logical_bytes_per_gpu"],
        "kv_modeled_resident_allocated_bytes_per_gpu": kv_cache["modeled_resident_allocated_bytes_per_gpu"],
        "kv_modeled_peak_allocated_bytes_per_gpu": kv_cache["modeled_peak_allocated_bytes_per_gpu"],
        "kv_actual_runtime_allocated_bytes_per_gpu": kv_cache["actual_runtime_allocated_bytes_per_gpu"],
        "kv_actual_runtime_measurement_status": kv_cache["actual_runtime_measurement_status"],
        "kv_actual_runtime_measurement_reason": kv_cache["actual_runtime_measurement_reason"],
        "full_mfa_ms": component_latency["full_mfa"],
        "swa_ms": component_latency["swa"],
        "dense_ms": component_latency["dense"],
        "latent_moe_ms": component_latency["latent_moe_total"],
        "dispatch_ms": component_latency["dispatch"],
        "combine_ms": component_latency["combine"],
        "other_component_ms": component_latency["other"],
        "component_accounted_total_ms": component_latency["accounted_total"],
        "moe_performance_workload_distribution": moe_observability["performance_workload_distribution"],
        "expert_token_histogram": moe_observability["expert_token_histogram"],
        "max_mean_load": moe_observability["max_mean_load"],
        "padding_ratio": moe_observability["padding_ratio"],
        "live_routing_status": moe_observability["live_routing_status"],
        "live_routing_reason": moe_observability["live_routing_reason"],
        "backend_fallback": execution_observability["backend_fallback"],
        "backend_fallback_reason": (
            "No automatic backend fallback exists. The DeepEP proxy is an explicitly selected simulation approximation."
        ),
        "retry_count": execution_observability["retry_count"],
        "error_record_count": execution_observability["error_record_count"],
        "missing_record_count": execution_observability["missing_record_count"],
        "exception_log": json.dumps(
            execution_observability["exception_log"],
            sort_keys=True,
        ),
        "vllm_actual_status": "PENDING_EXTERNAL_VLLM",
        "vllm_actual_reason": (
            "Whole-model pinned-vLLM results are owned by the external session "
            "and were not supplied to this AIC-only run."
        ),
        "vllm_actual_latency_ms": None,
        "absolute_error_ms": None,
        "relative_error": None,
    }


def build_review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the simulator matrix into one row per required condition."""
    rows: list[dict[str, Any]] = []
    for result in payload["prefill_results"]:
        workload = result["workload"]
        row = _base_review_row(
            phase="prefill",
            result=result,
            latency_metrics=result["latency_metrics"],
            throughput=result["throughput"],
            memory=result["peak_memory_gib"],
            kv_cache=result["kv_cache"],
            component_breakdown=result["component_breakdown"],
            moe_observability=result["moe_observability"],
            memory_unavailable_reasons=result["memory_unavailable_reasons"],
            execution_observability=result,
        )
        row.update(
            {
                "prompt_tokens": workload["prompt_tokens"],
                "context_tokens": None,
                "output_tokens": 0,
                "selected_batch_per_replica": workload["batch_size"],
                "max_num_batched_tokens": workload["max_num_batched_tokens"],
                "tpot_budget_ms": None,
                "b_max": None,
                "aggregate_b_max": None,
                "first_failed_batch": None,
                "active_sequences_per_replica": workload["batch_size"],
                "batched_tokens_per_replica": None,
                "engine_step_trace_location": ("prefill_results[].queried_steps[]"),
                "oom": result["oom"],
                "proxy_record_count": result["proxy_record_count"],
                "proxy_latency_ms": result["proxy_latency_ms"],
            }
        )
        rows.append(row)

    for result in payload["decode_results"]:
        selected_batch = result["b_max"] if result["b_max"] else 1
        candidate = next(
            (item for item in result["candidates"] if item["global_batch_size_per_replica"] == selected_batch),
            None,
        )
        if candidate is None:
            raise ValueError(
                "Decode review row lacks selected candidate "
                f"topology={result['topology']['name']} "
                f"workload={result['workload']} batch={selected_batch}"
            )
        workload = result["workload"]
        row = _base_review_row(
            phase="decode",
            result=result,
            latency_metrics=candidate["latency_metrics"],
            throughput=candidate["throughput"],
            memory=candidate["memory_gib"],
            kv_cache=candidate["kv_cache"],
            component_breakdown=candidate["component_breakdown"],
            moe_observability=candidate["moe_observability"],
            memory_unavailable_reasons=candidate["memory_unavailable_reasons"],
            execution_observability=candidate,
        )
        row.update(
            {
                "prompt_tokens": None,
                "context_tokens": workload["context_tokens"],
                "output_tokens": workload["output_tokens"],
                "selected_batch_per_replica": selected_batch,
                "max_num_batched_tokens": None,
                "tpot_budget_ms": result["tpot_budget_ms"],
                "b_max": result["b_max"],
                "aggregate_b_max": result.get("aggregate_b_max"),
                "first_failed_batch": result["first_failed_batch"],
                "active_sequences_per_replica": candidate["active_sequences_per_replica"],
                "batched_tokens_per_replica": candidate["batched_tokens_per_replica"],
                "engine_step_trace_location": ("decode_results[].candidates[]"),
                "oom": candidate["oom"],
                "proxy_record_count": candidate["proxy_record_count"],
                "proxy_latency_ms": candidate["proxy_latency_ms"],
            }
        )
        rows.append(row)
    return rows


def write_review_csv(rows: Sequence[dict[str, Any]], output: Path) -> None:
    """Write stable manual-review rows without changing simulation values."""
    if not rows:
        raise ValueError("review CSV requires at least one row")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _repeat_case_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for result in payload["prefill_results"]:
        topology = result["topology"]["name"]
        workload = result["workload"]
        identity = (
            f"prefill|{topology}|prompt={workload['prompt_tokens']}|"
            f"batch={workload['batch_size']}|"
            f"budget={workload['max_num_batched_tokens']}"
        )
        cases[identity] = result
    for result in payload["decode_results"]:
        topology = result["topology"]["name"]
        workload = result["workload"]
        identity = (
            f"decode|{topology}|context={workload['context_tokens']}|"
            f"output={workload['output_tokens']}|"
            f"tpot={result['tpot_budget_ms']}"
        )
        cases[identity] = result
    expected_count = len(payload["prefill_results"]) + len(payload["decode_results"])
    if len(cases) != expected_count:
        raise ValueError(f"Repeat audit case identities are not unique: expected={expected_count}, unique={len(cases)}")
    return cases


def _repeat_case_metrics(
    identity: str,
    result: dict[str, Any],
) -> dict[str, int | float | None | str]:
    if identity.startswith("prefill|"):
        return {
            "status": result["status"],
            "prefill_latency_ms": result["formal_prefill_latency_ms"],
            "aggregate_input_tok_s": result["throughput"]["aggregate_input_tok_s"],
            "input_tok_s_per_gpu": result["throughput"]["input_tok_s_per_gpu"],
            "peak_hbm_gib": result["peak_memory_gib"]["total"],
            "logical_kv_bytes": result["kv_cache"]["logical_bytes_per_gpu"],
            "peak_allocated_kv_bytes": result["kv_cache"]["modeled_peak_allocated_bytes_per_gpu"],
            "component_total_ms": result["component_breakdown"]["latency_ms"]["accounted_total"],
        }

    batch_one = next(
        (candidate for candidate in result["candidates"] if candidate["global_batch_size_per_replica"] == 1),
        None,
    )
    if batch_one is None:
        raise ValueError(f"Decode repeat case lacks batch=1 candidate: {identity}")
    return {
        "status": result["status"],
        "b_max": result["b_max"],
        "aggregate_b_max": result.get("aggregate_b_max"),
        "first_failed_batch": result["first_failed_batch"],
        "decode_tpot_ms": batch_one["steady_decode_step_ms"],
        "aggregate_output_tok_s": batch_one["throughput"]["aggregate_output_tok_s"],
        "output_tok_s_per_gpu": batch_one["throughput"]["output_tok_s_per_gpu"],
        "peak_hbm_gib": batch_one["memory_gib"]["total"],
        "logical_kv_bytes": batch_one["kv_cache"]["logical_bytes_per_gpu"],
        "peak_allocated_kv_bytes": batch_one["kv_cache"]["modeled_peak_allocated_bytes_per_gpu"],
        "component_total_ms": batch_one["component_breakdown"]["latency_ms"]["accounted_total"],
    }


def build_repeat_audit(
    payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Audit deterministic AIC results across repeated full matrix runs."""
    if not payloads:
        raise ValueError("repeat audit requires at least one payload")
    case_maps = [_repeat_case_map(payload) for payload in payloads]
    identities = list(case_maps[0])
    expected_identity_set = set(identities)
    for repeat_index, case_map in enumerate(case_maps[1:], start=2):
        if set(case_map) != expected_identity_set:
            raise ValueError(f"Repeat audit case identities changed at repeat {repeat_index}")

    case_audits: list[dict[str, Any]] = []
    max_abs_spread_by_metric: dict[str, float] = {}
    identical_count = 0
    for identity in identities:
        results = [case_map[identity] for case_map in case_maps]
        identical = all(result == results[0] for result in results[1:])
        identical_count += int(identical)
        metrics_by_repeat = [_repeat_case_metrics(identity, result) for result in results]
        metric_spreads: dict[str, dict[str, float]] = {}
        for metric_name in metrics_by_repeat[0]:
            values = [metrics[metric_name] for metrics in metrics_by_repeat]
            if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
                continue
            numeric_values = [float(value) for value in values]
            spread = max(numeric_values) - min(numeric_values)
            metric_spreads[metric_name] = {
                "minimum": min(numeric_values),
                "maximum": max(numeric_values),
                "absolute_spread": spread,
            }
            max_abs_spread_by_metric[metric_name] = max(
                max_abs_spread_by_metric.get(metric_name, 0.0),
                spread,
            )
        case_audits.append(
            {
                "case_id": identity,
                "full_result_identical": identical,
                "metrics_by_repeat": metrics_by_repeat,
                "metric_spreads": metric_spreads,
            }
        )

    return {
        "repeat_count": len(payloads),
        "case_count_per_repeat": len(identities),
        "total_case_executions": len(payloads) * len(identities),
        "identical_full_result_count": identical_count,
        "nonidentical_full_result_count": len(identities) - identical_count,
        "max_abs_spread_by_metric": dict(sorted(max_abs_spread_by_metric.items())),
        "cases": case_audits,
    }


def run_requirements_matrix(
    *,
    systems_root: Path,
    smoke: bool = False,
    max_search_batch: int = 4096,
    deepep_proxy: str | None = None,
) -> dict[str, Any]:
    if deepep_proxy not in {None, B300_NCCL_ALLTOALL_PROXY}:
        raise ValueError(f"unsupported DeepEP proxy {deepep_proxy!r}")
    workloads = build_requirements_workloads()
    if smoke:
        topologies = workloads["topologies"][:1]
        prefill_workloads = [
            workloads["prefill"][0],
            next(row for row in workloads["prefill"] if row["prompt_tokens"] == 65_536)
            if any(row["prompt_tokens"] == 65_536 for row in workloads["prefill"])
            else workloads["prefill"][-1],
        ]
        decode_workloads = workloads["decode"][:1]
        tpot_budgets = workloads["tpot_budgets_ms"][:1]
    else:
        topologies = workloads["topologies"]
        prefill_workloads = workloads["prefill"]
        decode_workloads = workloads["decode"]
        tpot_budgets = workloads["tpot_budgets_ms"]

    database = PerfDatabase(
        SYSTEM,
        BACKEND,
        FRAMEWORK_VERSION,
        str(systems_root),
        database_mode="SILICON",
    )
    backend = get_backend(BACKEND)
    models_by_ep = {topology["ep_size"]: build_latest_model(topology["ep_size"]) for topology in topologies}

    prefill_results = [
        simulate_prefill_case(
            model=models_by_ep[topology["ep_size"]],
            backend=backend,
            database=database,
            topology=topology,
            workload=workload,
            deepep_proxy=deepep_proxy,
        )
        for topology in topologies
        for workload in prefill_workloads
    ]
    decode_results = [
        simulate_decode_budget(
            model=models_by_ep[topology["ep_size"]],
            backend=backend,
            database=database,
            topology=topology,
            workload=workload,
            tpot_budget_ms=tpot_budget_ms,
            max_search_batch=max_search_batch,
            deepep_proxy=deepep_proxy,
        )
        for topology in topologies
        for workload in decode_workloads
        for tpot_budget_ms in tpot_budgets
    ]
    blocked_prefill = sum(result["status"] == "BLOCKED" for result in prefill_results)
    blocked_decode = sum(result["status"] == "BLOCKED" for result in decode_results)
    proxy_prefill = sum(result["result_fidelity"] == "PROXY" for result in prefill_results)
    proxy_decode = sum(result["result_fidelity"] == "PROXY" for result in decode_results)
    proxy_record_count = sum(result["proxy_record_count"] for result in prefill_results) + sum(
        result["proxy_record_count"] for result in decode_results
    )
    proxy_latency_ms = sum(result["proxy_latency_ms"] for result in prefill_results) + sum(
        result["proxy_latency_ms"] for result in decode_results
    )
    return {
        "status": (
            "BLOCKED"
            if blocked_prefill or blocked_decode
            else ("PASS_WITH_PROXY" if proxy_prefill or proxy_decode else "PASS")
        ),
        "result_fidelity": "PROXY" if proxy_prefill or proxy_decode else "MIXED_NON_SILICON",
        "model": MODEL_ID,
        "variant": "mtp_off",
        "system": SYSTEM,
        "backend": BACKEND,
        "framework_version": FRAMEWORK_VERSION,
        "database_mode": "SILICON",
        "deepep_proxy": deepep_proxy,
        "requirements_scope": {
            "mtp_off_simulation": "COMPLETE" if not blocked_prefill and not blocked_decode else "BLOCKED",
            "real_deepep_measurement": ("DEFERRED_PROXY_ACTIVE" if deepep_proxy is not None else "DEFERRED_MISSING"),
            "mtp1": "DEFERRED_BY_OWNER",
            "live_online_runtime_metrics": "PENDING_EXTERNAL_VLLM",
            "vllm_simulator_error_table": "PENDING_EXTERNAL_VLLM",
        },
        "kv_cache_contract": {
            "requested_dtype": "auto",
            "resolved_dtype": "bfloat16",
            "layout": "NHD",
            "actual_runtime_allocation": "PENDING_EXTERNAL_VLLM",
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "matrix_summary": {
            "topology_count": len(topologies),
            "prefill_workload_count_per_topology": len(prefill_workloads),
            "prefill_result_count": len(prefill_results),
            "blocked_prefill_result_count": blocked_prefill,
            "proxy_prefill_result_count": proxy_prefill,
            "decode_workload_count_per_topology": len(decode_workloads),
            "decode_budget_count": len(tpot_budgets),
            "decode_result_count": len(decode_results),
            "blocked_decode_result_count": blocked_decode,
            "proxy_decode_result_count": proxy_decode,
            "proxy_record_count": proxy_record_count,
            "proxy_latency_ms": proxy_latency_ms,
        },
        "prefill_results": prefill_results,
        "decode_results": decode_results,
    }


def run_repeated_requirements_matrix(
    *,
    systems_root: Path,
    repeat_count: int,
    smoke: bool = False,
    max_search_batch: int = 4096,
    deepep_proxy: str | None = None,
) -> dict[str, Any]:
    """Run the same deterministic requirements matrix repeatedly and audit it."""
    repeat_count = _positive_int("repeat_count", repeat_count)
    payloads = [
        run_requirements_matrix(
            systems_root=systems_root,
            smoke=smoke,
            max_search_batch=max_search_batch,
            deepep_proxy=deepep_proxy,
        )
        for _ in range(repeat_count)
    ]
    return combine_repeated_payloads(payloads)


def combine_repeated_payloads(
    payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Combine separately bounded matrix runs into one repeat audit."""
    if not payloads:
        raise ValueError("repeat combination requires at least one payload")
    for key in (
        "model",
        "variant",
        "system",
        "backend",
        "framework_version",
        "database_mode",
        "deepep_proxy",
    ):
        values = [payload.get(key) for payload in payloads]
        if len(set(map(repr, values))) != 1:
            raise ValueError(f"repeat payloads disagree on {key}: {values!r}")
    canonical = copy.deepcopy(payloads[0])
    canonical["repeat_audit"] = build_repeat_audit(payloads)
    repeat_count = len(payloads)
    canonical["matrix_summary"]["repeat_count"] = repeat_count
    canonical["matrix_summary"]["total_case_executions"] = canonical["repeat_audit"]["total_case_executions"]
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems-root",
        type=Path,
        default=Path("src/aiconfigurator/systems"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-search-batch", type=int, default=4096)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument(
        "--repeat-input",
        type=Path,
        action="append",
        dest="repeat_inputs",
    )
    parser.add_argument(
        "--deepep-proxy",
        choices=(B300_NCCL_ALLTOALL_PROXY,),
        default=None,
    )
    parser.add_argument("--review-output", type=Path)
    args = parser.parse_args()

    if args.repeat_inputs:
        if args.smoke or args.repeat_count != 1:
            parser.error("--repeat-input cannot be combined with --smoke or a non-default --repeat-count")
        payload = combine_repeated_payloads(
            [json.loads(path.read_text(encoding="utf-8")) for path in args.repeat_inputs]
        )
    else:
        payload = run_repeated_requirements_matrix(
            systems_root=args.systems_root,
            repeat_count=args.repeat_count,
            smoke=args.smoke,
            max_search_batch=args.max_search_batch,
            deepep_proxy=args.deepep_proxy,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.review_output is not None:
        write_review_csv(build_review_rows(payload), args.review_output)
    print(json.dumps(payload["matrix_summary"], indent=2))
    print(
        json.dumps(
            {
                key: payload["repeat_audit"][key]
                for key in (
                    "repeat_count",
                    "case_count_per_repeat",
                    "total_case_executions",
                    "identical_full_result_count",
                    "nonidentical_full_result_count",
                    "max_abs_spread_by_metric",
                )
            },
            indent=2,
        )
    )
    print(f"status={payload['status']}")
    if payload["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
