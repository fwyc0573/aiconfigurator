"""Unit tests for the exact-rank NCCL probe argument contract."""

import pytest

from tests.performance.step4_p6_distributed_probe import (
    _allocate_collective_buffers,
    _build_message_element_sweep,
    _build_perf_row,
    _parse_ops,
)

pytestmark = pytest.mark.unit


def test_parse_ops_accepts_exact_collective_set():
    assert _parse_ops("all_reduce, all_gather, reduce_scatter") == [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
    ]


@pytest.mark.parametrize("raw", ["", "broadcast", "all_reduce,broadcast"])
def test_parse_ops_rejects_unplanned_collectives(raw):
    with pytest.raises(ValueError, match="ops"):
        _parse_ops(raw)


@pytest.mark.parametrize("world_size", [2, 4, 8, 16, 32, 48, 64])
def test_message_sweep_is_rank_aligned_and_brackets_full_byte_envelope(world_size):
    sizes = _build_message_element_sweep(
        world_size=world_size,
        min_message_bytes=128,
        max_message_bytes=1_073_741_824,
    )
    assert sizes == sorted(set(sizes))
    assert all(size > 0 and size % world_size == 0 for size in sizes)
    assert sizes[0] * 2 <= 128
    assert sizes[-1] * 2 >= 1_073_741_824
    assert len(sizes) >= 24


@pytest.mark.parametrize(
    ("world_size", "min_message_bytes", "max_message_bytes"),
    [(0, 128, 1024), (32, 0, 1024), (32, 128, 0), (32, 1024, 128)],
)
def test_message_sweep_rejects_invalid_bounds(world_size, min_message_bytes, max_message_bytes):
    with pytest.raises(ValueError):
        _build_message_element_sweep(
            world_size=world_size,
            min_message_bytes=min_message_bytes,
            max_message_bytes=max_message_bytes,
        )


@pytest.mark.parametrize(
    ("op", "input_elements", "output_elements"),
    [("all_gather", 32, 1024), ("reduce_scatter", 1024, 32), ("all_reduce", 1024, 0)],
)
def test_collective_buffers_preserve_global_message_elements(op, input_elements, output_elements):
    input_tensor, output_tensor = _allocate_collective_buffers(
        op=op,
        message_elements=1024,
        world_size=32,
        device="cpu",
    )
    assert input_tensor.numel() == input_elements
    assert output_tensor is None if output_elements == 0 else output_tensor.numel() == output_elements


def test_collective_buffers_require_exact_rank_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        _allocate_collective_buffers(
            op="all_gather",
            message_elements=1025,
            world_size=32,
            device="cpu",
        )


def test_perf_row_uses_aic_element_count_and_preserves_byte_count():
    row = _build_perf_row(
        op="all_gather",
        world_size=48,
        message_elements=96,
        latency_ms=0.125,
        nccl_version="2.27.5",
        device_name="NVIDIA H800",
    )
    assert row == {
        "framework": "NCCL",
        "version": "2.27.5",
        "device": "NVIDIA H800",
        "op_name": "all_gather",
        "kernel_source": "NCCL",
        "nccl_dtype": "half",
        "num_gpus": 48,
        "message_size": 96,
        "message_size_bytes": 192,
        "latency": 0.125,
    }
