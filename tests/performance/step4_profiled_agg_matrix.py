"""Run the Step4-Pro-V3/V4 profiled aggregate matrix."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from collections.abc import Callable
from numbers import Real
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.errors import (
    InsufficientMemoryError,
    KVCacheCapacityError,
    NoFeasibleConfigError,
    PerfDataNotAvailableError,
)
from aiconfigurator.sdk.sweep import sweep_agg
from aiconfigurator.sdk.task_v2 import Task

MODELS = ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4")
ISLS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
REPLICA_WORLD_SIZES = (1, 2, 4, 8, 16, 32, 64)
ATTENTION_TP_SIZES = (1, 2, 4)
TOTAL_GPUS = 64
OSL = 1024
TTFT_MS = 10_000
TPOT_MS = 50_000
MAX_BATCH_SIZE = 1024
CUSTOM_ALLREDUCE_MAX_LOCAL_RANK = 8
ParallelConfig = tuple[int, int, int, int, int, int]


def _topologies(*, invalid_cross_node_custom_allreduce: bool) -> list[ParallelConfig]:
    topologies = []
    for world_size in REPLICA_WORLD_SIZES:
        for tp_size in ATTENTION_TP_SIZES:
            if world_size % tp_size:
                continue
            is_invalid = tp_size > CUSTOM_ALLREDUCE_MAX_LOCAL_RANK
            if is_invalid != invalid_cross_node_custom_allreduce:
                continue
            topologies.append((tp_size, 1, world_size // tp_size, 1, world_size, 1))
    return topologies


def build_runnable_topologies() -> list[ParallelConfig]:
    """Return aggregate configs whose attention-TP group remains within one node."""
    return _topologies(invalid_cross_node_custom_allreduce=False)


def build_invalid_topologies() -> list[ParallelConfig]:
    """Return configs whose attention-TP group crosses the node boundary."""
    return _topologies(invalid_cross_node_custom_allreduce=True)


def classify_matrix_exception(error: BaseException) -> str:
    if isinstance(error, (InsufficientMemoryError, KVCacheCapacityError)):
        return "memory_infeasible"
    if isinstance(error, NoFeasibleConfigError):
        return "sla_infeasible"
    if isinstance(error, PerfDataNotAvailableError):
        return "data_unavailable"
    return "error"


def _positive_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def run_matrix_point(
    *,
    model: str,
    isl: int,
    task_factory: Callable[..., Any] = Task,
    sweep_fn: Callable[..., Any] = sweep_agg,
) -> dict[str, object]:
    """Run one model/ISL independently and return one terminal outcome."""
    if model not in MODELS:
        raise ValueError(f"unsupported Step4 model: {model}")
    if isl not in ISLS:
        raise ValueError(f"unsupported ISL: {isl}")

    try:
        task = task_factory(
            serving_mode="agg",
            model_path=model,
            system_name="h800_sxm",
            backend_name="vllm",
            backend_version="0.19.0",
            database_mode="SILICON",
            isl=isl,
            osl=OSL,
            prefix=0,
            ttft=TTFT_MS,
            tpot=TPOT_MS,
            pareto_sweep=False,
            total_gpus=TOTAL_GPUS,
            engine_step_backend="python",
            batch_sweep_step=1,
            nextn=0,
            enable_chunked_prefill=False,
            agg_max_batch_size=MAX_BATCH_SIZE,
        )
        database = task._load_database("h800_sxm", "vllm", "0.19.0")
        sweep_kwargs = task.sweep_agg_kwargs(database=database)
        sweep_kwargs["parallel_config_list"] = build_runnable_topologies()
        results = sweep_fn(**sweep_kwargs)
        if results.empty:
            raise RuntimeError("aggregate sweep returned an empty result without a terminal exception")
        best_index = results["tokens/s/gpu"].astype(float).idxmax()
        row = results.loc[best_index]
        throughput_per_gpu = _positive_finite(row["tokens/s/gpu"], field="tokens/s/gpu")
        deployment_gpus = int(row["num_total_gpus"])
        selected_topology = tuple(int(row[field]) for field in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp"))
        if selected_topology not in build_runnable_topologies():
            raise ValueError(f"sweep selected a topology outside the runnable contract: {selected_topology}")
        if deployment_gpus not in REPLICA_WORLD_SIZES or TOTAL_GPUS % deployment_gpus:
            raise ValueError(f"selected deployment does not tile the 64-GPU cluster: {deployment_gpus}")
        replicas = TOTAL_GPUS // deployment_gpus
        total_gpus_used = replicas * deployment_gpus
        ttft_ms = _positive_finite(row["ttft"], field="ttft")
        tpot_ms = _positive_finite(row["tpot"], field="tpot")
        ctx_tokens_number = _positive_finite(row["ctx_tokens"], field="ctx_tokens")
        if not ctx_tokens_number.is_integer():
            raise ValueError(f"ctx_tokens must be an integer, got {ctx_tokens_number}")
        ctx_tokens = int(ctx_tokens_number)
        batch_size = int(row["bs"])
        single_point = task.run_single_agg(
            tp=selected_topology[0],
            pp=selected_topology[1],
            dp=selected_topology[2],
            moe_tp=selected_topology[3],
            moe_ep=selected_topology[4],
            batch_size=batch_size,
            ctx_tokens=ctx_tokens,
            include_per_ops=True,
        )
        per_ops_data = getattr(single_point, "per_ops_data", None)
        per_ops_source = getattr(single_point, "per_ops_source", None)
        if not isinstance(per_ops_data, dict) or not per_ops_data:
            raise RuntimeError("selected aggregate point has no per-operation latency evidence")
        if not isinstance(per_ops_source, dict) or not per_ops_source:
            raise RuntimeError("selected aggregate point has no per-operation source evidence")
        return {
            "model": model,
            "isl": isl,
            "status": "success",
            "throughput_per_used_gpu": throughput_per_gpu,
            "cluster_tokens_per_second": throughput_per_gpu * total_gpus_used,
            "deployment_gpus": deployment_gpus,
            "replicas": replicas,
            "total_gpus_used": total_gpus_used,
            "unused_gpus": TOTAL_GPUS - total_gpus_used,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "ttft_limit_ms": TTFT_MS,
            "tpot_limit_ms": TPOT_MS,
            "selected_config": {
                "tp": selected_topology[0],
                "pp": selected_topology[1],
                "dp": selected_topology[2],
                "moe_tp": selected_topology[3],
                "moe_ep": selected_topology[4],
                "cp": selected_topology[5],
                "batch_size": batch_size,
                "ctx_tokens": ctx_tokens,
            },
            "per_ops_data": per_ops_data,
            "per_ops_source": per_ops_source,
            "provenance": "measured-kernel-backed mixed AIC prediction",
        }
    except Exception as error:
        return {
            "model": model,
            "isl": isl,
            "status": classify_matrix_exception(error),
            "error_type": type(error).__name__,
            "reason": str(error),
        }


def run_matrix() -> dict[str, object]:
    outcomes = [run_matrix_point(model=model, isl=isl) for model in MODELS for isl in ISLS]
    status_counts = Counter(str(outcome["status"]) for outcome in outcomes)
    hard_error_count = status_counts["data_unavailable"] + status_counts["error"]
    return {
        "schema": "step4-profiled-agg-matrix-v1",
        "status": "completed" if hard_error_count == 0 else "completed_with_errors",
        "contract": {
            "serving_mode": "agg",
            "system": "h800_sxm",
            "backend": "vllm",
            "backend_version": "0.19.0",
            "database_mode": "SILICON",
            "total_gpus": TOTAL_GPUS,
            "osl": OSL,
            "ttft_ms": TTFT_MS,
            "tpot_ms": TPOT_MS,
            "nextn": 0,
            "pareto_sweep": False,
            "batch_sweep_step": 1,
            "max_batch_size": MAX_BATCH_SIZE,
        },
        "models": list(MODELS),
        "isls": list(ISLS),
        "point_count": len(outcomes),
        "status_counts": dict(sorted(status_counts.items())),
        "runnable_topologies": [list(topology) for topology in build_runnable_topologies()],
        "invalid_cross_node_custom_allreduce_topologies": [list(topology) for topology in build_invalid_topologies()],
        "terminal_outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    result = run_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "point_count": result["point_count"],
                "status_counts": result["status_counts"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
