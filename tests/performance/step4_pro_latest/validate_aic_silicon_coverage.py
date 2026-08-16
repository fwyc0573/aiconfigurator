"""Probe Step4-Pro-Latest operation consumers against the B300 SILICON database."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import config, models
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError
from aiconfigurator.sdk.perf_database import PerfDatabase
from tests.performance.step4_pro_latest.deepep_proxy import (
    B300_NCCL_ALLTOALL_PROXY,
    query_deepep_proxy,
)

MODEL_ID = "stepfun-ai/Step4-Pro-Latest"
DEFAULT_CONTEXT_PROBES = ((1, 512), (1, 8_192), (1, 32_768), (1, 65_536))
DEFAULT_GENERATION_PROBES = (
    (1, 2_048),
    (2, 8_192),
    (4, 32_768),
    (8, 131_072),
    (16, 524_288),
    (32, 1_048_544),
)
_MISSING_EXCEPTIONS = (PerfDataNotAvailableError, InterpolationDataNotAvailableError)


def build_latest_model(ep_size: int):
    """Build the pinned MTP-off model for one attention-DP/MoE-EP topology."""
    if type(ep_size) is not int or ep_size <= 0:
        raise ValueError(f"ep_size must be a positive integer, got {ep_size!r}")
    return models.get_model(
        MODEL_ID,
        config.ModelConfig(
            tp_size=1,
            pp_size=1,
            attention_dp_size=ep_size,
            moe_tp_size=1,
            moe_ep_size=ep_size,
            nextn=0,
            nextn_accept_rates=[0.85, 0.0, 0.0, 0.0, 0.0],
        ),
        backend_name="vllm",
    )


def _family(operation: object) -> str:
    class_name = operation.__class__.__name__
    provider = getattr(operation, "_provider", None)
    if class_name == "QKVNormRoPE":
        return "qkv_norm_rope"
    if class_name == "GroupedGEMM":
        return "grouped_gemm"
    if class_name == "FP32OutputGEMM":
        return "fp32_router"
    if class_name in {"ContextAttention", "GenerationAttention"} and provider:
        return "attention"
    if class_name == "MoE" and provider == "optimus_fp8_moe":
        return "optimus_moe"
    if class_name == "MoEDispatch" and provider == "vllm_deepep_high_throughput":
        return "deepep_ht"
    return class_name


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if hasattr(value, "name"):
        return value.name
    return repr(value)


def _physical_identity(operation: object, database: object, family: str) -> list[Any]:
    if family == "deepep_ht":
        key_builder = getattr(operation, "_step4_deepep_ht_persisted_key", None)
        if callable(key_builder):
            return [family, *_json_value(key_builder(database))]
    key_builder = getattr(operation, "_persisted_key", None)
    if callable(key_builder):
        return [family, *_json_value(key_builder())]
    return [
        family,
        operation.__class__.__name__,
        _json_value(getattr(operation, "_provider", None)),
    ]


def _query_kwargs(
    operation: object,
    *,
    family: str,
    phase: str,
    local_batch_size: int,
    sequence_length: int,
    prefix: int,
    global_scheduled_tokens: int | None,
) -> dict[str, Any]:
    if phase == "context":
        effective_sequence_length = sequence_length - prefix
        if effective_sequence_length <= 0:
            raise ValueError(
                "context sequence_length must be greater than prefix, got "
                f"sequence_length={sequence_length}, prefix={prefix}"
            )
        x = (
            local_batch_size
            if "logits_gemm" in getattr(operation, "_name", "")
            else local_batch_size * effective_sequence_length
        )
        if family == "optimus_moe" and global_scheduled_tokens is not None:
            attention_dp_size = getattr(operation, "_attention_dp_size", None)
            if not isinstance(attention_dp_size, int) or isinstance(attention_dp_size, bool) or attention_dp_size <= 0:
                raise ValueError(
                    f"Optimus MoE coverage mapping requires a positive attention_dp_size, got {attention_dp_size!r}"
                )
            if global_scheduled_tokens % attention_dp_size:
                raise ValueError(
                    "The current Optimus MoE query contract requires global "
                    "scheduled tokens divisible by attention_dp_size: "
                    f"global_scheduled_tokens={global_scheduled_tokens}, "
                    f"attention_dp_size={attention_dp_size}"
                )
            x = global_scheduled_tokens // attention_dp_size
        return {
            "x": x,
            "batch_size": local_batch_size,
            "beam_width": 1,
            "s": effective_sequence_length,
            "prefix": prefix,
            "seq_imbalance_correction_scale": 1.0,
        }
    if phase == "generation":
        return {
            "x": local_batch_size,
            "batch_size": local_batch_size,
            "beam_width": 1,
            "s": sequence_length,
            "gen_seq_imbalance_correction_scale": 1.0,
        }
    raise ValueError(f"unsupported phase: {phase!r}")


def probe_operation_list(
    operations: Iterable[object],
    *,
    database: object,
    phase: str,
    local_batch_size: int,
    sequence_length: int,
    prefix: int = 0,
    ep_size: int | None = None,
    global_scheduled_tokens: int | None = None,
    deepep_proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Query every operation independently so one missing family does not hide later gaps."""
    if type(local_batch_size) is not int or local_batch_size <= 0:
        raise ValueError(f"local_batch_size must be a positive integer, got {local_batch_size!r}")
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive integer, got {sequence_length!r}")
    if global_scheduled_tokens is not None and (
        type(global_scheduled_tokens) is not int or global_scheduled_tokens <= 0
    ):
        raise ValueError(f"global_scheduled_tokens must be a positive integer or None, got {global_scheduled_tokens!r}")
    if deepep_proxy not in {None, B300_NCCL_ALLTOALL_PROXY}:
        raise ValueError(f"unsupported DeepEP proxy {deepep_proxy!r}")

    records: list[dict[str, Any]] = []
    for operation in operations:
        family = _family(operation)
        base_record = {
            "phase": phase,
            "ep_size": ep_size,
            "local_batch_size": local_batch_size,
            "sequence_length": sequence_length,
            "prefix": prefix,
            "global_scheduled_tokens": global_scheduled_tokens,
            "operation_name": getattr(operation, "_name", operation.__class__.__name__),
            "operation_class": operation.__class__.__name__,
            "family": family,
            "provider": _json_value(getattr(operation, "_provider", None)),
            "physical_identity": _physical_identity(operation, database, family),
        }
        try:
            query_kwargs = _query_kwargs(
                operation,
                family=family,
                phase=phase,
                local_batch_size=local_batch_size,
                sequence_length=sequence_length,
                prefix=prefix,
                global_scheduled_tokens=global_scheduled_tokens,
            )
            if family == "deepep_ht" and deepep_proxy is not None:
                proxy_result = query_deepep_proxy(
                    operation,
                    database,
                    tokens_per_dp_rank=query_kwargs["x"],
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
            result = operation.query(database, **query_kwargs)
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


def summarize_probe_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate record and unique physical-contract coverage without losing detail."""
    family_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "records": 0,
            "ok_records": 0,
            "non_silicon_records": 0,
            "proxy_records": 0,
            "missing_records": 0,
            "error_records": 0,
        }
    )
    missing_contracts: dict[str, dict[str, Any]] = {}
    for record in records:
        family = record["family"]
        status = record["status"]
        family_counts[family]["records"] += 1
        family_counts[family][f"{status}_records"] += 1
        if status == "missing":
            identity_key = json.dumps(record["physical_identity"], sort_keys=True)
            if identity_key not in missing_contracts:
                missing_contracts[identity_key] = {
                    "family": family,
                    "provider": record["provider"],
                    "physical_identity": record["physical_identity"],
                    "record_count": 0,
                    "error_type": record["error_type"],
                    "error": record["error"],
                }
            missing_contracts[identity_key]["record_count"] += 1

    missing_count = sum(record["status"] == "missing" for record in records)
    error_count = sum(record["status"] == "error" for record in records)
    non_silicon_count = sum(record["status"] == "non_silicon" for record in records)
    proxy_count = sum(record["status"] == "proxy" for record in records)
    if missing_count or error_count:
        status = "BLOCKED"
    elif proxy_count:
        status = "PASS_WITH_PROXY"
    elif non_silicon_count:
        status = "PASS_WITH_NON_SILICON"
    else:
        status = "PASS"
    return {
        "status": status,
        "result_fidelity": ("PROXY" if proxy_count else ("MIXED_NON_SILICON" if non_silicon_count else "SILICON")),
        "record_count": len(records),
        "ok_record_count": sum(record["status"] == "ok" for record in records),
        "non_silicon_record_count": non_silicon_count,
        "proxy_record_count": proxy_count,
        "proxy_latency_ms": sum(record.get("latency_ms", 0.0) for record in records if record["status"] == "proxy"),
        "missing_record_count": missing_count,
        "error_record_count": error_count,
        "missing_physical_contract_count": len(missing_contracts),
        "missing_physical_contracts": [missing_contracts[key] for key in sorted(missing_contracts)],
        "families": dict(sorted(family_counts.items())),
    }


def _parse_probe(raw: str) -> tuple[int, int]:
    try:
        batch_text, sequence_text = raw.split(":", maxsplit=1)
        batch_size = int(batch_text)
        sequence_length = int(sequence_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"probe must be LOCAL_BATCH:SEQUENCE_LENGTH, got {raw!r}") from error
    if batch_size <= 0 or sequence_length <= 0:
        raise argparse.ArgumentTypeError(f"probe values must be positive, got {raw!r}")
    return batch_size, sequence_length


def run_coverage_validation(
    *,
    systems_root: Path,
    ep_sizes: Sequence[int],
    context_probes: Sequence[tuple[int, int]],
    generation_probes: Sequence[tuple[int, int]],
    deepep_proxy: str | None = None,
) -> dict[str, Any]:
    database = PerfDatabase(
        "b300_sxm",
        "vllm",
        "0.19.0",
        str(systems_root),
        database_mode="SILICON",
    )
    records: list[dict[str, Any]] = []
    for ep_size in ep_sizes:
        model = build_latest_model(ep_size)
        for local_batch_size, sequence_length in context_probes:
            global_scheduled_tokens = local_batch_size * sequence_length
            records.extend(
                probe_operation_list(
                    model.context_ops,
                    database=database,
                    phase="context",
                    ep_size=ep_size,
                    local_batch_size=local_batch_size,
                    sequence_length=sequence_length,
                    global_scheduled_tokens=global_scheduled_tokens,
                    deepep_proxy=deepep_proxy,
                )
            )
        for local_batch_size, sequence_length in generation_probes:
            records.extend(
                probe_operation_list(
                    model.generation_ops,
                    database=database,
                    phase="generation",
                    ep_size=ep_size,
                    local_batch_size=local_batch_size,
                    sequence_length=sequence_length,
                    deepep_proxy=deepep_proxy,
                )
            )
    summary = summarize_probe_records(records)
    return {
        "model": MODEL_ID,
        "system": "b300_sxm",
        "backend": "vllm",
        "framework_version": "0.19.0",
        "database_mode": "SILICON",
        "ep_sizes": list(ep_sizes),
        "context_probes": [list(probe) for probe in context_probes],
        "generation_probes": [list(probe) for probe in generation_probes],
        "deepep_proxy": deepep_proxy,
        "result_fidelity": summary["result_fidelity"],
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems-root",
        type=Path,
        default=Path("src/aiconfigurator/systems"),
    )
    parser.add_argument("--ep-size", type=int, action="append", dest="ep_sizes")
    parser.add_argument(
        "--context-probe",
        type=_parse_probe,
        action="append",
        dest="context_probes",
    )
    parser.add_argument(
        "--generation-probe",
        type=_parse_probe,
        action="append",
        dest="generation_probes",
    )
    parser.add_argument(
        "--deepep-proxy",
        choices=(B300_NCCL_ALLTOALL_PROXY,),
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run_coverage_validation(
        systems_root=args.systems_root,
        ep_sizes=args.ep_sizes or (16, 32),
        context_probes=args.context_probes or DEFAULT_CONTEXT_PROBES,
        generation_probes=args.generation_probes or DEFAULT_GENERATION_PROBES,
        deepep_proxy=args.deepep_proxy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    if payload["summary"]["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
