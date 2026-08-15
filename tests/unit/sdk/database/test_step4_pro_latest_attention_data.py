"""Consumer tests for Step4-Pro-Latest provider attention rows."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations import ContextAttention, GenerationAttention
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

CONTEXT_HEADER = (
    "framework,version,device,op_name,kernel_source,provider,batch_size,"
    "query_tokens,total_context_tokens,num_heads,num_key_value_heads,head_dim,"
    "window_size,attn_dtype,kv_cache_dtype,kv_storage_alias,page_size,"
    "physical_page_bytes,kv_block_stride_bytes,kv_cache_layout,latency\n"
)
GENERATION_HEADER = (
    "framework,version,device,op_name,kernel_source,provider,batch_size,"
    "context_tokens,num_heads,num_key_value_heads,head_dim,window_size,"
    "attn_dtype,kv_cache_dtype,kv_storage_alias,page_size,physical_page_bytes,"
    "kv_block_stride_bytes,kv_cache_layout,latency\n"
)


def _build_database(
    tmp_path: Path,
    *,
    context_rows: list[str] | None = None,
    generation_rows: list[str] | None = None,
) -> PerfDatabase:
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "vllm" / "0.19.0"
    data_dir.mkdir(parents=True)
    system_spec = yaml.safe_load(Path("src/aiconfigurator/systems/b300_sxm.yaml").read_text(encoding="utf-8"))
    system_spec["data_dir"] = "data"
    (systems_root / "b300_sxm.yaml").write_text(
        yaml.safe_dump(system_spec),
        encoding="utf-8",
    )
    if context_rows is not None:
        (data_dir / "step4_context_attention_perf.txt").write_text(
            CONTEXT_HEADER + "".join(context_rows),
            encoding="utf-8",
        )
    if generation_rows is not None:
        (data_dir / "step4_generation_attention_perf.txt").write_text(
            GENERATION_HEADER + "".join(generation_rows),
            encoding="utf-8",
        )
    return PerfDatabase("b300_sxm", "vllm", "0.19.0", str(systems_root))


def _context_row(
    *,
    provider: str = "optimus_fa4",
    batch_size: int = 1,
    query_tokens: int = 2,
    total_context_tokens: int = 4,
    num_heads: int = 64,
    num_kv_heads: int = 1,
    head_dim: int = 512,
    window_size: int = 0,
    kv_storage_alias: bool = True,
    physical_page_bytes: int = 524288,
    kv_block_stride_bytes: int = 524288,
    kv_cache_layout: str = "NHD",
    latency: float = 7.0,
) -> str:
    return (
        f"VLLM,0.19.0,B300,step4_context_attention,{provider},{provider},"
        f"{batch_size},{query_tokens},{total_context_tokens},{num_heads},"
        f"{num_kv_heads},{head_dim},{window_size},bfloat16,bfloat16,"
        f"{str(kv_storage_alias).lower()},128,{physical_page_bytes},"
        f"{kv_block_stride_bytes},{kv_cache_layout},{latency}\n"
    )


def _generation_row(
    *,
    provider: str = "optimus_fa4",
    batch_size: int = 1,
    context_tokens: int = 4,
    num_heads: int = 64,
    num_kv_heads: int = 1,
    head_dim: int = 512,
    window_size: int = 0,
    kv_storage_alias: bool = True,
    physical_page_bytes: int = 524288,
    kv_block_stride_bytes: int = 524288,
    kv_cache_layout: str = "NHD",
    latency: float = 5.0,
) -> str:
    return (
        f"VLLM,0.19.0,B300,step4_generation_attention,{provider},{provider},"
        f"{batch_size},{context_tokens},{num_heads},{num_kv_heads},{head_dim},"
        f"{window_size},bfloat16,bfloat16,{str(kv_storage_alias).lower()},"
        f"128,{physical_page_bytes},{kv_block_stride_bytes},{kv_cache_layout},"
        f"{latency}\n"
    )


def _context_operation(
    *,
    provider: str = "optimus_fa4",
    num_heads: int = 64,
    num_kv_heads: int = 1,
    head_dim: int = 512,
    window_size: int = 0,
    kv_storage_alias: bool = True,
    physical_page_bytes: int = 524288,
    kv_block_stride_bytes: int = 524288,
    kv_cache_layout: str = "NHD",
    scale_factor: float = 1.0,
) -> ContextAttention:
    return ContextAttention(
        "context_attention",
        scale_factor,
        num_heads,
        num_kv_heads,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
        window_size=window_size,
        head_size=head_dim,
        provider=provider,
        kv_storage_alias=kv_storage_alias,
        page_size=128,
        physical_page_bytes=physical_page_bytes,
        kv_block_stride_bytes=kv_block_stride_bytes,
        kv_cache_layout=kv_cache_layout,
    )


def _generation_operation(
    *,
    provider: str = "optimus_fa4",
    num_heads: int = 64,
    num_kv_heads: int = 1,
    head_dim: int = 512,
    window_size: int = 0,
    kv_storage_alias: bool = True,
    physical_page_bytes: int = 524288,
    kv_block_stride_bytes: int = 524288,
    kv_cache_layout: str = "NHD",
    scale_factor: float = 1.0,
) -> GenerationAttention:
    return GenerationAttention(
        "generation_attention",
        scale_factor,
        num_heads,
        num_kv_heads,
        common.KVCacheQuantMode.bfloat16,
        window_size=window_size,
        head_size=head_dim,
        provider=provider,
        kv_storage_alias=kv_storage_alias,
        page_size=128,
        physical_page_bytes=physical_page_bytes,
        kv_block_stride_bytes=kv_block_stride_bytes,
        kv_cache_layout=kv_cache_layout,
    )


@pytest.mark.parametrize(
    ("operation", "row"),
    [
        (_context_operation(), _context_row()),
        (
            _context_operation(
                provider="vllm_native_sliding_gqa",
                num_heads=128,
                num_kv_heads=8,
                head_dim=128,
                window_size=512,
                kv_storage_alias=False,
                kv_block_stride_bytes=262144,
            ),
            _context_row(
                provider="vllm_native_sliding_gqa",
                num_heads=128,
                num_kv_heads=8,
                head_dim=128,
                window_size=512,
                kv_storage_alias=False,
                kv_block_stride_bytes=262144,
            ),
        ),
    ],
)
def test_context_attention_queries_both_exact_provider_structures(tmp_path, operation, row):
    database = _build_database(tmp_path, context_rows=[row])

    result = operation.query(database, batch_size=1, s=2, prefix=2)

    assert float(result) == pytest.approx(7.0)
    assert result.source == "silicon"


def test_context_attention_interpolates_only_workload_axes(tmp_path):
    rows = [
        _context_row(
            batch_size=batch_size,
            query_tokens=query_tokens,
            total_context_tokens=total_context_tokens,
            latency=batch_size + query_tokens + total_context_tokens,
        )
        for batch_size in (1, 4)
        for query_tokens in (2, 4)
        for total_context_tokens in (4, 8)
    ]
    database = _build_database(tmp_path, context_rows=rows)

    result = _context_operation(scale_factor=2.0).query(
        database,
        batch_size=2,
        s=3,
        prefix=3,
    )

    assert float(result) == pytest.approx(22.0)


@pytest.mark.parametrize(
    "operation",
    [
        _context_operation(provider="generic_fa"),
        _context_operation(kv_storage_alias=False),
        _context_operation(head_dim=256),
        _context_operation(physical_page_bytes=262144),
        _context_operation(kv_cache_layout="HND"),
    ],
)
def test_context_attention_requires_exact_structural_key(tmp_path, operation):
    database = _build_database(tmp_path, context_rows=[_context_row()])

    with pytest.raises(PerfDataNotAvailableError):
        operation.query(database, batch_size=1, s=2, prefix=2)


def test_context_attention_rejects_invalid_workload(tmp_path):
    database = _build_database(tmp_path, context_rows=[_context_row()])

    with pytest.raises(ValueError, match="positive batch_size"):
        _context_operation().query(database, batch_size=0, s=2, prefix=2)
    with pytest.raises(ValueError, match="positive query token"):
        _context_operation().query(database, batch_size=1, s=0, prefix=2)
    with pytest.raises(ValueError, match="non-negative prefix"):
        _context_operation().query(database, batch_size=1, s=2, prefix=-1)


def test_context_attention_loader_rejects_conflicting_physical_key(tmp_path):
    database = _build_database(
        tmp_path,
        context_rows=[
            _context_row(latency=7.0),
            _context_row(latency=8.0),
        ],
    )

    with pytest.raises(ValueError, match="conflicting Step4 context-attention row"):
        _context_operation().query(database, batch_size=1, s=2, prefix=2)


def test_generation_attention_interpolates_batch_and_context(tmp_path):
    rows = [
        _generation_row(
            batch_size=batch_size,
            context_tokens=context_tokens,
            latency=batch_size + context_tokens,
        )
        for batch_size in (1, 4)
        for context_tokens in (4, 8)
    ]
    database = _build_database(tmp_path, generation_rows=rows)

    result = _generation_operation(scale_factor=2.0).query(
        database,
        beam_width=1,
        batch_size=2,
        s=6,
    )

    assert float(result) == pytest.approx(16.0)
    assert result.source == "silicon"


@pytest.mark.parametrize(
    "operation",
    [
        _generation_operation(provider="generic_fa"),
        _generation_operation(kv_storage_alias=False),
        _generation_operation(head_dim=256),
        _generation_operation(kv_block_stride_bytes=262144),
        _generation_operation(kv_cache_layout="HND"),
    ],
)
def test_generation_attention_requires_exact_structural_key(tmp_path, operation):
    database = _build_database(tmp_path, generation_rows=[_generation_row()])

    with pytest.raises(PerfDataNotAvailableError):
        operation.query(database, beam_width=1, batch_size=1, s=4)


def test_generation_attention_rejects_invalid_workload(tmp_path):
    database = _build_database(tmp_path, generation_rows=[_generation_row()])

    with pytest.raises(ValueError, match="positive batch_size"):
        _generation_operation().query(
            database,
            beam_width=1,
            batch_size=0,
            s=4,
        )
    with pytest.raises(ValueError, match="positive context token"):
        _generation_operation().query(
            database,
            beam_width=1,
            batch_size=1,
            s=0,
        )


def test_generation_attention_loader_rejects_conflicting_physical_key(tmp_path):
    database = _build_database(
        tmp_path,
        generation_rows=[
            _generation_row(latency=5.0),
            _generation_row(latency=6.0),
        ],
    )

    with pytest.raises(
        ValueError,
        match="conflicting Step4 generation-attention row",
    ):
        _generation_operation().query(
            database,
            beam_width=1,
            batch_size=1,
            s=4,
        )
