"""Explicit simulation-only proxy for deferred Step4-Pro DeepEP measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiconfigurator.sdk import common

B300_NCCL_ALLTOALL_PROXY = "b300_nccl_alltoall"
PROXY_SOURCE = "proxy_b300_nccl_alltoall"
_MEASURED_TOPOLOGY_MAX_GPUS = 8
_SUPPORTED_EP_SIZES = (16, 32)
_STEP4_HIDDEN_SIZE = 3584
_STEP4_TOPK = 16


@dataclass(frozen=True)
class DeepEPProxyResult:
    """One labeled proxy result without changing the DeepEP data contract."""

    latency_ms: float
    energy_wms: float
    source: str
    metadata: dict[str, Any]


def _positive_int_attribute(operation: object, name: str) -> int:
    value = getattr(operation, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"DeepEP proxy requires a positive integer {name}, got {value!r}.")
    return value


def query_deepep_proxy(
    operation: object,
    database: object,
    *,
    tokens_per_dp_rank: int,
    proxy_name: str,
) -> DeepEPProxyResult:
    """Query the explicit B300 NCCL alltoall proxy for one DeepEP operation."""
    if proxy_name != B300_NCCL_ALLTOALL_PROXY:
        raise ValueError(f"unsupported DeepEP proxy {proxy_name!r}")
    if not isinstance(tokens_per_dp_rank, int) or isinstance(tokens_per_dp_rank, bool) or tokens_per_dp_rank <= 0:
        raise ValueError(f"DeepEP proxy requires a positive tokens_per_dp_rank integer, got {tokens_per_dp_rank!r}.")

    provider = getattr(operation, "_provider", None)
    if provider != "vllm_deepep_high_throughput":
        raise ValueError(f"DeepEP proxy supports only provider 'vllm_deepep_high_throughput', got {provider!r}.")
    operation_name = getattr(operation, "_operation", None)
    if operation_name not in {"dispatch", "combine"}:
        raise ValueError(f"DeepEP proxy operation must be 'dispatch' or 'combine', got {operation_name!r}.")

    hidden_size = _positive_int_attribute(operation, "_hidden_size")
    topk = _positive_int_attribute(operation, "_topk")
    moe_tp_size = _positive_int_attribute(operation, "_moe_tp_size")
    ep_size = _positive_int_attribute(operation, "_moe_ep_size")
    attention_dp_size = _positive_int_attribute(operation, "_attention_dp_size")
    if hidden_size != _STEP4_HIDDEN_SIZE or topk != _STEP4_TOPK:
        raise ValueError(
            "DeepEP proxy is scoped to Step4-Pro-Latest hidden_size=3584 "
            f"and topk=16, got hidden_size={hidden_size}, topk={topk}."
        )
    if moe_tp_size != 1 or attention_dp_size != ep_size:
        raise ValueError("DeepEP proxy requires moe_tp_size=1 and attention_dp_size=moe_ep_size.")
    if ep_size not in _SUPPORTED_EP_SIZES:
        raise ValueError(f"DeepEP proxy supports only EP16/EP32, got ep_size={ep_size}.")

    scale_factor = getattr(operation, "_scale_factor", None)
    if not isinstance(scale_factor, int | float) or isinstance(scale_factor, bool) or scale_factor <= 0:
        raise ValueError(f"DeepEP proxy requires a positive numeric operation scale factor, got {scale_factor!r}.")

    numerator = tokens_per_dp_rank * hidden_size * topk
    message_elements = (numerator + ep_size - 1) // ep_size
    if operation_name == "dispatch":
        logical_payload_dtype = "fp8"
        table_dtype = common.CommQuantMode.int8
    else:
        logical_payload_dtype = "bfloat16"
        table_dtype = common.CommQuantMode.half

    result = database.query_nccl(
        table_dtype,
        ep_size,
        "alltoall",
        message_elements,
    )
    latency_ms = float(result) * float(scale_factor)
    energy_wms = float(getattr(result, "energy", 0.0)) * float(scale_factor)
    metadata = {
        "proxy_name": B300_NCCL_ALLTOALL_PROXY,
        "result_fidelity": "PROXY",
        "proxy_source": PROXY_SOURCE,
        "original_provider": provider,
        "operation": operation_name,
        "logical_payload_dtype": logical_payload_dtype,
        "nccl_table_dtype": table_dtype.name,
        "collective": "alltoall",
        "tokens_per_dp_rank": tokens_per_dp_rank,
        "hidden_size": hidden_size,
        "topk": topk,
        "requested_ep_size": ep_size,
        "measured_topology_max_gpus": _MEASURED_TOPOLOGY_MAX_GPUS,
        "topology_extrapolated": ep_size > _MEASURED_TOPOLOGY_MAX_GPUS,
        "topology_mapping": "aic_nccl_rank_correction_from_8_gpu_curve",
        "message_elements": message_elements,
        "scale_factor": scale_factor,
    }
    return DeepEPProxyResult(
        latency_ms=latency_ms,
        energy_wms=energy_wms,
        source=PROXY_SOURCE,
        metadata=metadata,
    )
