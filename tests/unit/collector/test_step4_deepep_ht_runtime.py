"""Runtime contracts for the pinned vLLM Step4-Pro DeepEP HT collector."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from collector.wideep.vllm import collect_step4_deepep_ht as collector

pytestmark = pytest.mark.unit


def test_runtime_uses_pinned_worker_nvshmem_setup_before_distributed_init() -> None:
    source = inspect.getsource(collector._create_deepep_ht_runtime)
    harmonize_call = "\n    _harmonize_nvshmem_env_for_deepep_optimus()\n"
    distributed_init_call = "\n    init_distributed_environment(\n"

    assert harmonize_call in source
    assert source.index(harmonize_call) < source.index(distributed_init_call)


class _Event:
    def __init__(self) -> None:
        self.wait_count = 0

    def current_stream_wait(self) -> None:
        self.wait_count += 1


class _Buffer:
    def __init__(self) -> None:
        self.layout_calls: list[dict] = []
        self.dispatch_calls: list[dict] = []
        self.combine_calls: list[dict] = []
        self.dispatch_events: list[_Event] = []
        self.combine_events: list[_Event] = []
        self.handle = object()
        self.received_tokens = object()

    def get_dispatch_layout(self, **kwargs):
        self.layout_calls.append(kwargs)
        return (
            "tokens_per_rank",
            "tokens_per_rdma_rank",
            "tokens_per_expert",
            "is_token_in_rank",
            _Event(),
        )

    def dispatch(self, **kwargs):
        self.dispatch_calls.append(kwargs)
        event = _Event()
        self.dispatch_events.append(event)
        return (
            self.received_tokens,
            "received_topk_ids",
            "received_topk_weights",
            [1, 2],
            self.handle,
            event,
        )

    def combine(self, **kwargs):
        self.combine_calls.append(kwargs)
        event = _Event()
        self.combine_events.append(event)
        return "combined_tokens", None, event


@dataclass
class _PrepareFinalize:
    dispatch_config: object
    combine_config: object

    def _get_dispatch_config(self):
        return self.dispatch_config

    def _get_combine_config(self):
        return self.combine_config


def test_buffer_benchmark_uses_exact_pinned_dispatch_and_combine_calls():
    buffer = _Buffer()
    dispatch_config = object()
    combine_config = object()
    prepare_finalize = _PrepareFinalize(dispatch_config, combine_config)
    token_data = ("fp8_tokens", "block128_scales")
    combine_input = object()
    benchmarked: list[object] = []

    def benchmark(call):
        benchmarked.append(call)
        call()
        call()
        return {
            "latency_ms": 0.25 * len(benchmarked),
            "power_stats": None,
            "used_cuda_graph": False,
        }

    results = collector._benchmark_deepep_ht_buffer_legs(
        buffer=buffer,
        prepare_finalize=prepare_finalize,
        token_data=token_data,
        topk_ids="topk_ids",
        topk_weights="topk_weights",
        num_experts=896,
        make_combine_input=lambda received: (
            combine_input
            if received is buffer.received_tokens
            else pytest.fail("combine input factory received the wrong dispatch output")
        ),
        benchmark=benchmark,
    )

    assert results["dispatch"]["latency_ms"] == pytest.approx(0.25)
    assert results["combine"]["latency_ms"] == pytest.approx(0.5)
    assert len(benchmarked) == 2
    assert buffer.layout_calls == [
        {
            "topk_idx": "topk_ids",
            "num_experts": 896,
            "previous_event": None,
            "async_finish": False,
            "allocate_on_comm_stream": False,
        }
    ]

    assert len(buffer.dispatch_calls) == 3
    assert all(
        call
        == {
            "x": token_data,
            "handle": None,
            "num_tokens_per_rank": "tokens_per_rank",
            "num_tokens_per_rdma_rank": "tokens_per_rdma_rank",
            "is_token_in_rank": "is_token_in_rank",
            "num_tokens_per_expert": "tokens_per_expert",
            "topk_idx": "topk_ids",
            "topk_weights": "topk_weights",
            "expert_alignment": 1,
            "config": dispatch_config,
            "previous_event": None,
            "async_finish": True,
            "allocate_on_comm_stream": False,
        }
        for call in buffer.dispatch_calls
    )
    assert all(event.wait_count == 1 for event in buffer.dispatch_events)

    assert len(buffer.combine_calls) == 2
    assert all(
        call
        == {
            "x": combine_input,
            "handle": buffer.handle,
            "topk_weights": None,
            "config": combine_config,
            "previous_event": None,
            "async_finish": True,
            "allocate_on_comm_stream": False,
        }
        for call in buffer.combine_calls
    )
    assert all(event.wait_count == 1 for event in buffer.combine_events)


def test_buffer_benchmark_rejects_cuda_graph_measurement():
    buffer = _Buffer()
    prepare_finalize = _PrepareFinalize(object(), object())

    def benchmark(call):
        call()
        return {
            "latency_ms": 0.1,
            "power_stats": None,
            "used_cuda_graph": True,
        }

    with pytest.raises(RuntimeError, match="must use eager execution"):
        collector._benchmark_deepep_ht_buffer_legs(
            buffer=buffer,
            prepare_finalize=prepare_finalize,
            token_data=("fp8_tokens", "block128_scales"),
            topk_ids="topk_ids",
            topk_weights="topk_weights",
            num_experts=896,
            make_combine_input=lambda received: object(),
            benchmark=benchmark,
        )


def test_run_logs_separate_dispatch_and_combine_physical_rows(monkeypatch):
    runtime = SimpleNamespace(
        rank=0,
        world_size=16,
        buffer=object(),
        prepare_finalize=object(),
        benchmark=object(),
        vllm_version="0.19.0.post20.dev26+gc820e5ae1",
        device_name="NVIDIA B300 SXM6 AC",
    )
    runtime_calls: list[dict] = []
    logged: list[dict] = []

    def get_runtime(**kwargs):
        runtime_calls.append(kwargs)
        return runtime

    monkeypatch.setattr(
        collector,
        "_get_or_create_deepep_ht_runtime",
        get_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        collector,
        "_build_step4_deepep_ht_inputs",
        lambda **kwargs: {
            "token_data": ("fp8_tokens", "block128_scales"),
            "topk_ids": "topk_ids",
            "topk_weights": "topk_weights",
            "make_combine_input": lambda received: "combine_input",
        },
        raising=False,
    )
    monkeypatch.setattr(
        collector,
        "_benchmark_deepep_ht_buffer_legs",
        lambda **kwargs: {
            "dispatch": {
                "latency_ms": 0.3,
                "power_stats": None,
                "used_cuda_graph": False,
            },
            "combine": {
                "latency_ms": 0.7,
                "power_stats": None,
                "used_cuda_graph": False,
            },
        },
    )
    monkeypatch.setattr(
        collector,
        "_global_max_latency",
        lambda runtime, latency_ms: latency_ms + 0.1,
        raising=False,
    )
    monkeypatch.setattr(
        collector,
        "log_perf",
        lambda **kwargs: logged.append(kwargs),
        raising=False,
    )

    rows = collector.run_step4_deepep_ht(
        "vllm_deepep_high_throughput",
        16,
        8,
        3584,
        896,
        16,
        128,
        "fp8_e4m3_block128",
        20,
        0,
        perf_filename="step4_deepep_ht_perf.txt",
        device="cuda:0",
    )

    assert runtime_calls == [
        {
            "ep_size": 16,
            "ep_ranks_per_node": 8,
            "hidden_size": 3584,
            "num_experts": 896,
            "num_sms": 20,
            "device": "cuda:0",
        }
    ]
    assert rows == [
        {
            "provider": "vllm_deepep_high_throughput",
            "deepep_mode": "ht",
            "operation": "dispatch",
            "ep_size": 16,
            "ep_ranks_per_node": 8,
            "hidden_size": 3584,
            "num_experts": 896,
            "topk": 16,
            "tokens_per_dp_rank": 128,
            "dispatch_format": "fp8_e4m3_block128",
            "num_sms": 20,
            "max_tokens_per_rank": 0,
            "latency": pytest.approx(0.4),
        },
        {
            "provider": "vllm_deepep_high_throughput",
            "deepep_mode": "ht",
            "operation": "combine",
            "ep_size": 16,
            "ep_ranks_per_node": 8,
            "hidden_size": 3584,
            "num_experts": 896,
            "topk": 16,
            "tokens_per_dp_rank": 128,
            "dispatch_format": "fp8_e4m3_block128",
            "num_sms": 20,
            "max_tokens_per_rank": 0,
            "latency": pytest.approx(0.8),
        },
    ]
    assert logged == [
        {
            "item_list": rows,
            "framework": "VLLM",
            "version": runtime.vllm_version,
            "device_name": runtime.device_name,
            "op_name": "step4_deepep_ht",
            "kernel_source": "deepep_ht",
            "perf_filename": "step4_deepep_ht_perf.txt",
            "power_stats": None,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("provider", "sglang_deepep", "provider"),
        ("ep_size", 8, "EP size"),
        ("ep_ranks_per_node", 4, "ranks per node"),
        ("hidden_size", 4096, "hidden size"),
        ("num_experts", 256, "expert count"),
        ("topk", 8, "top-k"),
        ("tokens_per_dp_rank", 0, "token count"),
        ("dispatch_format", "bf16", "dispatch format"),
        ("num_sms", 16, "num_sms"),
        ("max_tokens_per_rank", 128, "max_tokens_per_rank"),
    ],
)
def test_run_rejects_non_pinned_invocation_before_runtime_init(
    monkeypatch,
    field,
    value,
    match,
):
    values = {
        "provider": "vllm_deepep_high_throughput",
        "ep_size": 16,
        "ep_ranks_per_node": 8,
        "hidden_size": 3584,
        "num_experts": 896,
        "topk": 16,
        "tokens_per_dp_rank": 128,
        "dispatch_format": "fp8_e4m3_block128",
        "num_sms": 20,
        "max_tokens_per_rank": 0,
    }
    values[field] = value
    monkeypatch.setattr(
        collector,
        "_get_or_create_deepep_ht_runtime",
        lambda **kwargs: pytest.fail("invalid case initialized distributed runtime"),
        raising=False,
    )

    with pytest.raises(ValueError, match=match):
        collector.run_step4_deepep_ht(
            values["provider"],
            values["ep_size"],
            values["ep_ranks_per_node"],
            values["hidden_size"],
            values["num_experts"],
            values["topk"],
            values["tokens_per_dp_rank"],
            values["dispatch_format"],
            values["num_sms"],
            values["max_tokens_per_rank"],
            perf_filename="step4_deepep_ht_perf.txt",
            device="cuda:0",
        )


def test_input_builder_uses_pinned_fp8_block128_transport():
    torch = pytest.importorskip("torch")
    quantize_calls: list[dict] = []

    def quantize_input(hidden_states, **kwargs):
        quantize_calls.append(
            {
                "hidden_states": hidden_states,
                **kwargs,
            }
        )
        return (
            torch.zeros_like(hidden_states, dtype=torch.uint8),
            torch.ones(
                (hidden_states.shape[0], hidden_states.shape[1] // 128),
                dtype=torch.float32,
            ),
        )

    runtime = SimpleNamespace(
        torch=torch,
        device=torch.device("cpu"),
        rank=3,
        world_size=16,
        fp8_dtype=object(),
        quantize_input=quantize_input,
    )

    inputs = collector._build_step4_deepep_ht_inputs(
        runtime=runtime,
        hidden_size=3584,
        num_experts=896,
        topk=16,
        tokens_per_dp_rank=4,
        dispatch_format="fp8_e4m3_block128",
    )

    assert len(quantize_calls) == 1
    quantize_call = quantize_calls[0]
    assert quantize_call["hidden_states"].shape == (4, 3584)
    assert quantize_call["hidden_states"].dtype == torch.bfloat16
    assert quantize_call["A_scale"] is None
    assert quantize_call["quant_dtype"] is runtime.fp8_dtype
    assert quantize_call["per_act_token_quant"] is False
    assert quantize_call["block_shape"] == [128, 128]

    assert inputs["token_data"][0].shape == (4, 3584)
    assert inputs["token_data"][1].shape == (4, 28)
    assert inputs["topk_ids"].shape == (4, 16)
    assert inputs["topk_ids"].dtype == torch.int64
    assert int(inputs["topk_ids"].min()) >= 0
    assert int(inputs["topk_ids"].max()) < 896
    assert all(len(set(row.tolist())) == 16 for row in inputs["topk_ids"])
    assert inputs["topk_weights"].shape == (4, 16)
    assert inputs["topk_weights"].dtype == torch.float32
    assert torch.allclose(
        inputs["topk_weights"].sum(dim=1),
        torch.ones(4),
    )

    combine_input = inputs["make_combine_input"](
        (
            torch.zeros((7, 3584), dtype=torch.uint8),
            torch.ones((7, 28), dtype=torch.float32),
        )
    )
    assert combine_input.shape == (7, 3584)
    assert combine_input.dtype == torch.bfloat16


def test_global_latency_uses_maximum_across_all_ep_ranks():
    torch = pytest.importorskip("torch")

    class _Dist:
        class ReduceOp:
            MAX = object()

        def __init__(self):
            self.calls = []

        def all_reduce(self, value, op):
            self.calls.append((value.clone(), op))
            value.fill_(0.9)

    dist = _Dist()
    runtime = SimpleNamespace(
        torch=torch,
        dist=dist,
        device=torch.device("cpu"),
    )

    result = collector._global_max_latency(runtime, 0.3)

    assert result == pytest.approx(0.9)
    assert len(dist.calls) == 1
    assert float(dist.calls[0][0].item()) == pytest.approx(0.3)
    assert dist.calls[0][1] is dist.ReduceOp.MAX


def test_runtime_cache_reuses_only_the_same_distributed_identity(monkeypatch):
    created: list[dict] = []

    def create_runtime(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(collector, "_DEEPEP_HT_RUNTIME", None, raising=False)
    monkeypatch.setattr(
        collector,
        "_create_deepep_ht_runtime",
        create_runtime,
        raising=False,
    )
    identity = {
        "ep_size": 16,
        "ep_ranks_per_node": 8,
        "hidden_size": 3584,
        "num_experts": 896,
        "num_sms": 20,
        "device": "cuda:0",
    }

    first = collector._get_or_create_deepep_ht_runtime(**identity)
    second = collector._get_or_create_deepep_ht_runtime(**identity)

    assert first is second
    assert created == [identity]

    with pytest.raises(RuntimeError, match="already initialized"):
        collector._get_or_create_deepep_ht_runtime(**(identity | {"ep_size": 32}))
