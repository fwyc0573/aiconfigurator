"""Static Step4-Pro profiling coverage auditor.

The auditor is deliberately task-local.  It consumes the same public case
getters used by the Collector and the built Step4 operation graph, then emits
an exact structural-key inventory plus bounded workload envelopes.  It never
claims that a key is measured: an optional measured-key file is required for
that comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aiconfigurator.sdk import common, config, models
from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
    get_common_moe_test_cases,
    get_step4_model_gemm_case_specs,
)

MODELS = ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4")
BACKEND = "vllm"
VERSION = "0.19.0"
DEVICE = "h800_sxm"
REQUIRED_FAMILIES = ("attention", "gemm", "moe", "communication")
TP_SIZES = (1, 2, 4)
AGGREGATE_REPLICA_WORLD_SIZES = (1, 2, 4, 8, 16, 32, 64)
VLLM_CUSTOM_ALLREDUCE_WORLD_SIZES = frozenset({2, 4, 6, 8})


@contextmanager
def _model_filter(model_path: str) -> Iterator[None]:
    previous = os.environ.get("COLLECTOR_MODEL_PATH")
    os.environ["COLLECTOR_MODEL_PATH"] = model_path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)
        else:
            os.environ["COLLECTOR_MODEL_PATH"] = previous


def _identity(model: str, family: str, axes: dict[str, object]) -> str:
    prefix = f"{model}:{family}:{BACKEND}:{VERSION}:{DEVICE}:"
    return prefix + json.dumps(axes, sort_keys=True, separators=(",", ":"))


def _record(model: str, family: str, axes: dict[str, object], envelope: dict[str, object]) -> dict:
    record = {
        "model": model,
        "op_family": family,
        "backend": BACKEND,
        "device": DEVICE,
        "system": DEVICE,
        "version": VERSION,
        "structural": {
            "identity": _identity(model, family, axes),
            "axes": {"backend": BACKEND, "version": VERSION, "device": DEVICE, **axes},
        },
        "workload_envelope": envelope,
    }
    # task_v2 validates the model-shape contract from record-level fields.
    # Keep those fields duplicated deliberately: structural.axes is the
    # canonical identity payload, while these top-level values are the
    # consumer admission contract and make the JSON self-describing.
    if family == "moe":
        record.update(axes)
    return record


def _bounds(values: set[int]) -> dict[str, object]:
    if not values:
        raise ValueError("coverage workload envelope cannot be empty")
    ordered = sorted(values)
    return {"min": ordered[0], "max": ordered[-1], "count": len(ordered), "values": ordered}


def _attention_records(model_path: str) -> list[dict]:
    records: list[dict] = []
    with _model_filter(model_path):
        context_sweep = get_attention_context_shape_sweeps(BACKEND)[0]
        generation_sweep = get_attention_generation_shape_sweeps(BACKEND)[0]
        context_configs = get_attention_head_configs(context_sweep, phase="context")
        generation_configs = get_attention_head_configs(generation_sweep, phase="generation")

    context_batches = {int(value) for value in context_sweep["batch_sizes"]}
    context_lengths = {int(value) for value in context_sweep["sequence_lengths"]}
    generation_batches = {int(value) for value in generation_sweep["batch_sizes"]}
    generation_lengths = {int(value) for value in generation_sweep["sequence_lengths"]}

    for phase, configs, batches, lengths in (
        ("context", context_configs, context_batches, context_lengths),
        ("generation", generation_configs, generation_batches, generation_lengths),
    ):
        for shape in configs:
            axes = {
                "phase": phase,
                "num_heads": shape.num_heads,
                "num_key_value_heads": shape.num_kv_heads,
                "head_dim": shape.head_dim,
                "window_size": shape.window_size,
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
            }
            records.append(
                _record(
                    model_path,
                    "attention",
                    axes,
                    {"batch_size": _bounds(batches), "sequence_or_step": _bounds(lengths)},
                )
            )
    return records


def _gemm_records(model_path: str) -> list[dict]:
    specs = get_step4_model_gemm_case_specs(model_path, backend=BACKEND)
    tokens_by_shape: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    for spec in specs:
        for dtype in spec.gemm_types or ():
            tokens_by_shape[(spec.n, spec.k, dtype)].add(spec.x)
    return [
        _record(
            model_path,
            "gemm",
            {"n": n, "k": k, "gemm_dtype": dtype},
            {"m_or_token_count": _bounds(tokens)},
        )
        for (n, k, dtype), tokens in sorted(tokens_by_shape.items())
    ]


def _moe_records(model_path: str) -> list[dict]:
    with _model_filter(model_path):
        cases = get_common_moe_test_cases()
    grouped: dict[tuple[object, ...], set[int]] = defaultdict(set)
    for case in cases:
        distribution = (
            f"power_law_{case.power_law_alpha}"
            if case.token_expert_distribution == "power_law"
            else case.token_expert_distribution
        )
        key = (
            case.hidden_size,
            case.inter_size,
            case.topk,
            case.num_experts,
            case.tp,
            case.ep,
            "fp8",
            distribution,
        )
        grouped[key].update(int(token) for token in case.num_tokens_list)
    records = []
    for (hidden, inter, topk, experts, tp, ep, quantization, distribution), tokens in sorted(grouped.items()):
        records.append(
            _record(
                model_path,
                "moe",
                {
                    "hidden_size": hidden,
                    "inter_size": inter,
                    "topk": topk,
                    "num_experts": experts,
                    "moe_tp_size": tp,
                    "moe_ep_size": ep,
                    "quantization": quantization,
                    "distribution": distribution,
                },
                {"num_tokens": _bounds(tokens)},
            )
        )
    return records


def _communication_records(model_path: str) -> tuple[list[dict], list[dict]]:
    """Inspect every complete-replica topology and emitted collective path.

    The 64-GPU aggregate matrix can place complete model replicas whose world
    sizes divide 64. Step4 requires ``tp * attention_dp == moe_tp * moe_ep``;
    this task fixes ``moe_tp=1`` and ``moe_ep=world_size``. vLLM reduces MoE
    output over the attention-TP group and independently redistributes tokens
    over the complete DP/EP world.
    """

    topology_audit: list[dict] = []
    topology_worlds: dict[tuple[str, int], set[int]] = defaultdict(set)
    topology_sources: dict[tuple[str, int], set[str]] = defaultdict(set)
    model_experts = 1024 if model_path.endswith("V3") else 384
    for world_size in AGGREGATE_REPLICA_WORLD_SIZES:
        if model_experts % world_size != 0:
            topology_audit.append(
                {
                    "world_size": world_size,
                    "status": "skipped",
                    "reason": f"num_experts={model_experts} is not divisible by world_size={world_size}",
                }
            )
            continue
        for tp_size in TP_SIZES:
            if world_size % tp_size != 0:
                continue
            attention_dp_size = world_size // tp_size
            model_cfg = config.ModelConfig(
                tp_size=tp_size,
                pp_size=1,
                attention_dp_size=attention_dp_size,
                gemm_quant_mode=common.GEMMQuantMode.fp8,
                moe_quant_mode=common.MoEQuantMode.fp8,
                kvcache_quant_mode=common.KVCacheQuantMode.fp8,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                moe_tp_size=1,
                moe_ep_size=world_size,
            )
            model = models.get_model(model_path, model_cfg, BACKEND)
            stack = [*model.context_ops, *model.generation_ops]
            visited: set[int] = set()
            topology_queries: dict[tuple[str, int, str], dict[str, object]] = {}

            def add_query(operation: str, rank: int, producer_path: str) -> None:
                topology_queries[(operation, rank, producer_path)] = {
                    "dtype": "half",
                    "op": operation,
                    "producer_path": producer_path,
                    "rank": rank,
                }

            while stack:
                operation = stack.pop()
                if id(operation) in visited:
                    continue
                visited.add(id(operation))
                for group_name in ("_group_a", "_group_b"):
                    stack.extend(getattr(operation, group_name, ()) or ())
                operation_name = operation.__class__.__name__
                if operation_name == "CustomAllReduce":
                    rank = int(operation._tp_size)
                    if rank in VLLM_CUSTOM_ALLREDUCE_WORLD_SIZES:
                        add_query("custom_allreduce", rank, "built_graph")
                    continue
                if operation_name != "MoEDispatch":
                    continue

                # vLLM reduces the MoE output over the local attention-TP
                # process group, while DP/EP redistribution spans the complete
                # MoE world.
                rank = int(operation.num_gpus)
                if int(operation._attention_tp_size) > 1 and bool(operation._reduce_results):
                    add_query("custom_allreduce", int(operation._attention_tp_size), "MoEDispatch")
                if int(operation._attention_dp_size) > 1:
                    for collective in ("nccl_all_gather", "nccl_reduce_scatter"):
                        add_query(collective, rank, "MoEDispatch")

            queries = sorted(
                topology_queries.values(), key=lambda item: (item["op"], item["rank"], item["producer_path"])
            )
            invalid_queries = [
                query
                for query in queries
                if query["op"] == "custom_allreduce" and query["rank"] not in VLLM_CUSTOM_ALLREDUCE_WORLD_SIZES
            ]
            has_custom_allreduce = any(query["op"] == "custom_allreduce" for query in queries)
            has_nccl = any(str(query["op"]).startswith("nccl_") for query in queries)
            if has_custom_allreduce and has_nccl:
                collective_class = "custom_allreduce_and_nccl"
            elif has_custom_allreduce:
                collective_class = "custom_allreduce_only"
            elif has_nccl:
                collective_class = "nccl_only"
            else:
                collective_class = "none"
            status = "invalid_cross_node_custom_allreduce" if invalid_queries else "runnable"
            topology_audit.append(
                {
                    "world_size": world_size,
                    "attention_tp_size": tp_size,
                    "attention_dp_size": attention_dp_size,
                    "moe_tp_size": 1,
                    "moe_ep_size": world_size,
                    "status": status,
                    "collective_class": collective_class,
                    "queries": queries,
                    "invalid_queries": invalid_queries,
                }
            )
            if status != "runnable":
                continue
            for query in queries:
                operation = str(query["op"])
                rank = int(query["rank"])
                topology_worlds[(operation, rank)].add(world_size)
                topology_sources[(operation, rank)].add(str(query["producer_path"]))

    records = []
    for (operation, rank), worlds in sorted(topology_worlds.items()):
        records.append(
            _record(
                model_path,
                "communication",
                {"op": operation, "rank": rank, "dtype": "half", "strategy": "AUTO"},
                {
                    "message_size_bytes": {"min": 128, "max": 1073741824, "axis": "message_size"},
                    "topology_world_sizes": sorted(worlds),
                    "producer_paths": sorted(topology_sources[(operation, rank)]),
                },
            )
        )
    return records, topology_audit


def _deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique: list[dict] = []
    duplicate_count = 0
    for record in records:
        identity = record["structural"]["identity"]
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        unique.append(record)
    return unique, duplicate_count


def build_step4_coverage_inventory(measured_identities: set[str] | None = None) -> dict:
    """Build a planned/measured structural-key inventory for both models."""

    measured = measured_identities or set()
    coverage_keys: dict[str, list[dict]] = {}
    coverage_summary: dict[str, dict[str, dict[str, int]]] = {}
    communication_topology_audit: dict[str, list[dict]] = {}
    for model_path in MODELS:
        raw = _attention_records(model_path) + _gemm_records(model_path) + _moe_records(model_path)
        communication_records, topology_audit = _communication_records(model_path)
        raw += communication_records
        communication_topology_audit[model_path] = topology_audit
        records, duplicate_count = _deduplicate(raw)
        coverage_keys[model_path] = records
        by_family: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            by_family[record["op_family"]].append(record)
        coverage_summary[model_path] = {}
        for family in REQUIRED_FAMILIES:
            family_records = by_family[family]
            identities = {record["structural"]["identity"] for record in family_records}
            measured_count = len(identities & measured)
            coverage_summary[model_path][family] = {
                "required_count": len(identities),
                "measured_count": measured_count,
                "missing_count": len(identities - measured),
                "duplicate_count": duplicate_count if family_records else 0,
                "unassigned_count": 0,
            }

    return {
        "status": "validated"
        if all(
            values[family]["missing_count"] == 0
            and values[family]["duplicate_count"] == 0
            and values[family]["unassigned_count"] == 0
            for values in coverage_summary.values()
            for family in REQUIRED_FAMILIES
        )
        else "planned",
        "system": DEVICE,
        "device": DEVICE,
        "backend": BACKEND,
        "version": VERSION,
        "models": list(MODELS),
        "distribution": "power_law_1.2",
        "required_op_families": list(REQUIRED_FAMILIES),
        "coverage_keys": coverage_keys,
        "coverage_summary": coverage_summary,
        "communication_topology_audit": communication_topology_audit,
        "measured_identity_count": len(measured),
    }


def _load_measured_identities(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(identity) for identity in payload}
    if isinstance(payload, dict) and isinstance(payload.get("identities"), list):
        return {str(identity) for identity in payload["identities"]}
    raise ValueError('measured-key file must be a JSON list or {"identities": [...]} mapping')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-keys", type=Path)
    args = parser.parse_args()
    inventory = build_step4_coverage_inventory(_load_measured_identities(args.measured_keys))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.measured_keys is not None:
        measured_path = args.measured_keys.resolve()
        output_dir = args.output.resolve().parent
        try:
            relative_measured_path = measured_path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError("--measured-keys must be stored beside the output manifest") from exc
        inventory["provenance"] = {
            "measured_key_inventory": {
                "path": str(relative_measured_path),
                "sha256": hashlib.sha256(measured_path.read_bytes()).hexdigest(),
            }
        }
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": inventory["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
