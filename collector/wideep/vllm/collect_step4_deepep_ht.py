# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned vLLM Step4-Pro DeepEP high-throughput collector."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from collector.helper import log_perf

__compat__ = "vllm>=0.19.0,<0.20.0"

LATEST_MODEL = "stepfun-ai/Step4-Pro-Latest"
PROVIDER = "vllm_deepep_high_throughput"
PINNED_VLLM_VERSION = "0.19.0.post20.dev26+gc820e5ae1"


@dataclass
class _DeepEPHTRuntime:
    ep_size: int
    ep_ranks_per_node: int
    hidden_size: int
    num_experts: int
    num_sms: int
    requested_device: str
    torch: Any
    dist: Any
    device: Any
    rank: int
    local_rank: int
    world_size: int
    manager: Any
    buffer: Any
    prepare_finalize: Any
    benchmark: Callable[[Callable[[], Any]], dict[str, Any]]
    fp8_dtype: Any
    quantize_input: Callable[..., Any]
    vllm_version: str
    device_name: str
    config_context: Any


_DEEPEP_HT_RUNTIME: _DeepEPHTRuntime | None = None


def get_step4_deepep_ht_test_cases() -> list[list[object]]:
    """Return one distributed invocation per EP topology and local token count."""
    from collector.case_generator import get_step4_deepep_ht_workload_config

    config = get_step4_deepep_ht_workload_config(LATEST_MODEL)
    cases = [
        [
            PROVIDER,
            ep_size,
            config["ep_ranks_per_node"],
            config["hidden_size"],
            config["num_experts"],
            config["topk"],
            tokens_per_dp_rank,
            config["dispatch_format"],
            config["num_sms"],
            config["max_tokens_per_rank"],
        ]
        for ep_size in config["expert_parallel_sizes"]
        for tokens_per_dp_rank in config["tokens_per_dp_rank"]
    ]
    if len(cases) != len({tuple(case) for case in cases}):
        raise RuntimeError("Step4 DeepEP HT cases contain duplicate invocations")
    return cases


def _wait_for_deepep_event(event: Any) -> None:
    """Wait for an asynchronous DeepEP operation on the current stream."""
    if getattr(event, "event", event) is not None:
        event.current_stream_wait()


def _benchmark_deepep_ht_buffer_legs(
    *,
    buffer: Any,
    prepare_finalize: Any,
    token_data: Any,
    topk_ids: Any,
    topk_weights: Any,
    num_experts: int,
    make_combine_input: Callable[[Any], Any],
    benchmark: Callable[[Callable[[], Any]], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Measure the exact Buffer calls used by pinned vLLM HT prepare/finalize."""
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(
        topk_idx=topk_ids,
        num_experts=num_experts,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    dispatch_config = prepare_finalize._get_dispatch_config()
    combine_config = prepare_finalize._get_combine_config()

    def dispatch_once():
        result = buffer.dispatch(
            x=token_data,
            handle=None,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            expert_alignment=1,
            config=dispatch_config,
            previous_event=None,
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        _wait_for_deepep_event(result[5])
        return result

    dispatch_probe = dispatch_once()
    dispatch_results = benchmark(dispatch_once)
    if dispatch_results["used_cuda_graph"]:
        raise RuntimeError("Step4 DeepEP HT dispatch must use eager execution")

    combine_input = make_combine_input(dispatch_probe[0])
    dispatch_handle = dispatch_probe[4]

    def combine_once():
        result = buffer.combine(
            x=combine_input,
            handle=dispatch_handle,
            topk_weights=None,
            config=combine_config,
            previous_event=None,
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        _wait_for_deepep_event(result[2])
        return result

    combine_results = benchmark(combine_once)
    if combine_results["used_cuda_graph"]:
        raise RuntimeError("Step4 DeepEP HT combine must use eager execution")

    return {
        "dispatch": dispatch_results,
        "combine": combine_results,
    }


def _build_step4_deepep_ht_inputs(
    *,
    runtime: _DeepEPHTRuntime,
    hidden_size: int,
    num_experts: int,
    topk: int,
    tokens_per_dp_rank: int,
    dispatch_format: str,
) -> dict[str, Any]:
    """Build the FP8 block-128 payload used by pinned vLLM HT dispatch."""
    if dispatch_format != "fp8_e4m3_block128":
        raise ValueError(f"Unsupported Step4 DeepEP HT dispatch format: {dispatch_format!r}")
    if hidden_size % 128:
        raise ValueError(f"Step4 DeepEP HT hidden size must be divisible by 128, got {hidden_size}")
    if num_experts % runtime.world_size:
        raise ValueError(
            "Step4 DeepEP HT experts must shard evenly across EP ranks: "
            f"num_experts={num_experts}, ep_size={runtime.world_size}"
        )
    if topk > runtime.world_size:
        raise ValueError(
            "Step4 DeepEP HT deterministic routing requires top-k no larger "
            f"than EP size: topk={topk}, ep_size={runtime.world_size}"
        )

    torch = runtime.torch
    hidden_states = torch.full(
        (tokens_per_dp_rank, hidden_size),
        fill_value=(runtime.rank + 1) / runtime.world_size,
        dtype=torch.bfloat16,
        device=runtime.device,
    )
    token_data = runtime.quantize_input(
        hidden_states,
        A_scale=None,
        quant_dtype=runtime.fp8_dtype,
        per_act_token_quant=False,
        block_shape=[128, 128],
    )
    if not isinstance(token_data, tuple) or len(token_data) != 2:
        raise RuntimeError("Pinned vLLM block-FP8 quantization did not return tokens/scales")

    experts_per_rank = num_experts // runtime.world_size
    token_indices = torch.arange(
        tokens_per_dp_rank,
        dtype=torch.int64,
        device=runtime.device,
    ).unsqueeze(1)
    route_indices = torch.arange(
        topk,
        dtype=torch.int64,
        device=runtime.device,
    ).unsqueeze(0)
    target_ranks = (runtime.rank + token_indices + route_indices) % runtime.world_size
    local_experts = (token_indices * topk + route_indices) % experts_per_rank
    topk_ids = target_ranks * experts_per_rank + local_experts
    topk_weights = torch.full(
        (tokens_per_dp_rank, topk),
        fill_value=1.0 / topk,
        dtype=torch.float32,
        device=runtime.device,
    )

    def make_combine_input(received_token_data):
        received_tokens = received_token_data[0] if isinstance(received_token_data, tuple) else received_token_data
        if received_tokens.ndim != 2 or received_tokens.shape[1] != hidden_size:
            raise RuntimeError(
                f"Pinned DeepEP dispatch returned an unexpected token shape: {tuple(received_tokens.shape)}"
            )
        return torch.full(
            received_tokens.shape,
            fill_value=(runtime.rank + 1) / runtime.world_size,
            dtype=torch.bfloat16,
            device=runtime.device,
        )

    return {
        "token_data": token_data,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "make_combine_input": make_combine_input,
    }


def _global_max_latency(
    runtime: _DeepEPHTRuntime,
    latency_ms: float,
) -> float:
    """Return the distributed critical-path latency for one DeepEP leg."""
    value = runtime.torch.tensor(
        [latency_ms],
        dtype=runtime.torch.float64,
        device=runtime.device,
    )
    runtime.dist.all_reduce(value, op=runtime.dist.ReduceOp.MAX)
    return float(value.item())


def _create_deepep_ht_runtime(
    *,
    ep_size: int,
    ep_ranks_per_node: int,
    hidden_size: int,
    num_experts: int,
    num_sms: int,
    device: str,
) -> _DeepEPHTRuntime:
    """Initialize the pinned vLLM distributed groups and DeepEP HT manager."""
    required_env = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing_env = [name for name in required_env if name not in os.environ]
    if missing_env:
        raise RuntimeError(
            "Step4 DeepEP HT must run under a distributed torchrun launcher; "
            f"missing environment variables: {missing_env}"
        )

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != ep_size:
        raise RuntimeError(
            f"Step4 DeepEP HT launcher world size does not match the case: world_size={world_size}, ep_size={ep_size}"
        )
    if local_world_size != ep_ranks_per_node:
        raise RuntimeError(
            "Step4 DeepEP HT launcher local world size does not match the "
            f"topology: local_world_size={local_world_size}, "
            f"ep_ranks_per_node={ep_ranks_per_node}"
        )
    if ep_size % ep_ranks_per_node:
        raise RuntimeError(
            "Step4 DeepEP HT EP size must be divisible by ranks per node: "
            f"ep_size={ep_size}, ep_ranks_per_node={ep_ranks_per_node}"
        )
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            "Step4 DeepEP HT launcher ranks are invalid: "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )

    import torch
    import torch.distributed as dist
    from vllm.config import (
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        get_ep_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.device_communicators.all2all import (
        DeepEPHTAll2AllManager,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
        DeepEPHTPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.utils import (
        moe_kernel_quantize_input,
    )
    from vllm.platforms import current_platform
    from vllm.v1.worker.worker_base import (
        _harmonize_nvshmem_env_for_deepep_optimus,
    )
    from vllm.version import __version__ as vllm_version

    from collector.helper import benchmark_with_power

    if vllm_version != PINNED_VLLM_VERSION:
        raise RuntimeError(
            f"Step4 DeepEP HT requires the pinned vLLM package version {PINNED_VLLM_VERSION!r}, got {vllm_version!r}"
        )

    requested_device = torch.device(device)
    if requested_device.type != "cuda" or requested_device.index != local_rank:
        raise RuntimeError(f"Step4 DeepEP HT device must match LOCAL_RANK: device={device!r}, local_rank={local_rank}")
    torch.cuda.set_device(local_rank)
    torch_device = torch.device(f"cuda:{local_rank}")
    capability = torch.cuda.get_device_capability(torch_device)
    if capability != (10, 3):
        raise RuntimeError(f"Pinned Step4 DeepEP HT collection requires B300 SM103, got {capability}")

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=ep_size,
        data_parallel_size_local=ep_ranks_per_node,
        data_parallel_rank=rank,
        data_parallel_rank_local=local_rank,
        is_moe_model=True,
        enable_expert_parallel=True,
        all2all_backend="deepep_high_throughput",
        distributed_executor_backend="external_launcher",
        nnodes=ep_size // ep_ranks_per_node,
    )
    vllm_config = VllmConfig(parallel_config=parallel_config)
    config_context = set_current_vllm_config(vllm_config)
    config_context.__enter__()

    _harmonize_nvshmem_env_for_deepep_optimus()
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    ep_group = get_ep_group()
    communicator = ep_group.device_communicator
    if communicator is None:
        raise RuntimeError("Pinned vLLM did not create an EP device communicator")
    manager = communicator.all2all_manager
    if not isinstance(manager, DeepEPHTAll2AllManager):
        raise TypeError(f"Pinned vLLM did not select DeepEPHTAll2AllManager: actual={type(manager)!r}")
    manager.set_num_sms(num_sms)
    buffer = manager.get_handle({})

    rounded_hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
        hidden_size,
        torch.bfloat16,
    )
    if rounded_hidden_size != hidden_size:
        raise RuntimeError(
            "Step4 DeepEP HT persisted hidden size does not match the pinned "
            f"transfer size: persisted={hidden_size}, transfer={rounded_hidden_size}"
        )
    if num_experts % ep_size:
        raise RuntimeError(
            f"Step4 DeepEP HT experts must shard evenly across EP ranks: num_experts={num_experts}, ep_size={ep_size}"
        )
    prepare_finalize = DeepEPHTPrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=ep_size,
        dp_size=ep_size,
        rank_expert_offset=(num_experts // ep_size) * ep_group.rank_in_group,
    )

    def benchmark(call):
        dist.barrier()
        with benchmark_with_power(
            device=torch_device,
            kernel_func=call,
            num_warmups=3,
            num_runs=6,
            repeat_n=1,
            measure_power=False,
            allow_graph_fail=False,
            use_cuda_graph=False,
        ) as results:
            pass
        dist.barrier()
        return results

    return _DeepEPHTRuntime(
        ep_size=ep_size,
        ep_ranks_per_node=ep_ranks_per_node,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_sms=num_sms,
        requested_device=str(device),
        torch=torch,
        dist=dist,
        device=torch_device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        manager=manager,
        buffer=buffer,
        prepare_finalize=prepare_finalize,
        benchmark=benchmark,
        fp8_dtype=current_platform.fp8_dtype(),
        quantize_input=moe_kernel_quantize_input,
        vllm_version=vllm_version,
        device_name=torch.cuda.get_device_name(torch_device),
        config_context=config_context,
    )


def _get_or_create_deepep_ht_runtime(
    *,
    ep_size: int,
    ep_ranks_per_node: int,
    hidden_size: int,
    num_experts: int,
    num_sms: int,
    device: str,
) -> _DeepEPHTRuntime:
    """Create one DeepEP runtime per rank and reject topology changes."""
    global _DEEPEP_HT_RUNTIME

    requested = (
        ep_size,
        ep_ranks_per_node,
        hidden_size,
        num_experts,
        num_sms,
        str(device),
    )
    if _DEEPEP_HT_RUNTIME is None:
        _DEEPEP_HT_RUNTIME = _create_deepep_ht_runtime(
            ep_size=ep_size,
            ep_ranks_per_node=ep_ranks_per_node,
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_sms=num_sms,
            device=device,
        )
    existing_device = (
        _DEEPEP_HT_RUNTIME.requested_device
        if hasattr(_DEEPEP_HT_RUNTIME, "requested_device")
        else _DEEPEP_HT_RUNTIME.device
    )
    existing = (
        _DEEPEP_HT_RUNTIME.ep_size,
        _DEEPEP_HT_RUNTIME.ep_ranks_per_node,
        _DEEPEP_HT_RUNTIME.hidden_size,
        _DEEPEP_HT_RUNTIME.num_experts,
        _DEEPEP_HT_RUNTIME.num_sms,
        str(existing_device),
    )
    if existing != requested:
        raise RuntimeError(
            "Step4 DeepEP HT runtime is already initialized for a different "
            f"distributed identity: existing={existing!r}, requested={requested!r}"
        )
    return _DEEPEP_HT_RUNTIME


def run_step4_deepep_ht(
    provider: str,
    ep_size: int,
    ep_ranks_per_node: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    tokens_per_dp_rank: int,
    dispatch_format: str,
    num_sms: int,
    max_tokens_per_rank: int,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> list[dict[str, Any]]:
    """Run one distributed pinned-vLLM DeepEP HT dispatch/combine case."""
    from collector.case_generator import get_step4_deepep_ht_workload_config

    config = get_step4_deepep_ht_workload_config(LATEST_MODEL)
    if provider != PROVIDER:
        raise ValueError(f"Unexpected Step4 DeepEP HT provider: {provider!r}")
    if ep_size not in config["expert_parallel_sizes"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT EP size: {ep_size}")
    if ep_ranks_per_node != config["ep_ranks_per_node"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT ranks per node: {ep_ranks_per_node}")
    if hidden_size != config["hidden_size"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT hidden size: {hidden_size}")
    if num_experts != config["num_experts"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT expert count: {num_experts}")
    if topk != config["topk"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT top-k: {topk}")
    if tokens_per_dp_rank not in config["tokens_per_dp_rank"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT token count: {tokens_per_dp_rank}")
    if dispatch_format != config["dispatch_format"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT dispatch format: {dispatch_format!r}")
    if num_sms != config["num_sms"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT num_sms: {num_sms}")
    if max_tokens_per_rank != config["max_tokens_per_rank"]:
        raise ValueError(f"Unexpected Step4 DeepEP HT max_tokens_per_rank: {max_tokens_per_rank}")

    runtime = _get_or_create_deepep_ht_runtime(
        ep_size=ep_size,
        ep_ranks_per_node=ep_ranks_per_node,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_sms=num_sms,
        device=device,
    )
    inputs = _build_step4_deepep_ht_inputs(
        runtime=runtime,
        hidden_size=hidden_size,
        num_experts=num_experts,
        topk=topk,
        tokens_per_dp_rank=tokens_per_dp_rank,
        dispatch_format=dispatch_format,
    )
    results = _benchmark_deepep_ht_buffer_legs(
        buffer=runtime.buffer,
        prepare_finalize=runtime.prepare_finalize,
        token_data=inputs["token_data"],
        topk_ids=inputs["topk_ids"],
        topk_weights=inputs["topk_weights"],
        num_experts=num_experts,
        make_combine_input=inputs["make_combine_input"],
        benchmark=runtime.benchmark,
    )

    rows = []
    for operation in ("dispatch", "combine"):
        rows.append(
            {
                "provider": provider,
                "deepep_mode": "ht",
                "operation": operation,
                "ep_size": ep_size,
                "ep_ranks_per_node": ep_ranks_per_node,
                "hidden_size": hidden_size,
                "num_experts": num_experts,
                "topk": topk,
                "tokens_per_dp_rank": tokens_per_dp_rank,
                "dispatch_format": dispatch_format,
                "num_sms": num_sms,
                "max_tokens_per_rank": max_tokens_per_rank,
                "latency": _global_max_latency(
                    runtime,
                    float(results[operation]["latency_ms"]),
                ),
            }
        )

    if runtime.rank == 0:
        log_perf(
            item_list=rows,
            framework="VLLM",
            version=runtime.vllm_version,
            device_name=runtime.device_name,
            op_name="step4_deepep_ht",
            kernel_source="deepep_ht",
            perf_filename=perf_filename,
            power_stats=None,
        )
    return rows
