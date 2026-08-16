"""Consumer tests for pinned vLLM Step4-Pro DeepEP HT rows."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations import MoEDispatch
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

PROVIDER = "vllm_deepep_high_throughput"
HEADER = (
    "framework,version,device,op_name,kernel_source,provider,deepep_mode,"
    "operation,ep_size,ep_ranks_per_node,hidden_size,num_experts,topk,"
    "tokens_per_dp_rank,dispatch_format,num_sms,max_tokens_per_rank,latency,power\n"
)


def _build_database(tmp_path: Path, rows: list[str]) -> PerfDatabase:
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "vllm" / "0.19.0"
    data_dir.mkdir(parents=True)
    system_spec = yaml.safe_load(Path("src/aiconfigurator/systems/b300_sxm.yaml").read_text(encoding="utf-8"))
    system_spec["data_dir"] = "data"
    (systems_root / "b300_sxm.yaml").write_text(
        yaml.safe_dump(system_spec),
        encoding="utf-8",
    )
    (data_dir / "step4_deepep_ht_perf.txt").write_text(
        HEADER + "".join(rows),
        encoding="utf-8",
    )
    return PerfDatabase("b300_sxm", "vllm", "0.19.0", str(systems_root))


def _row(operation: str, tokens: int, latency: float) -> str:
    return (
        "VLLM,0.19.0,B300,step4_deepep_ht,deepep_ht,"
        f"{PROVIDER},ht,{operation},16,8,3584,896,16,{tokens},"
        f"fp8_e4m3_block128,20,0,{latency},0\n"
    )


def _operation(
    operation: str,
    *,
    ep_size: int = 16,
    ep_ranks_per_node: int = 8,
    dispatch_format: str = "fp8_e4m3_block128",
    sms: int = 20,
    scale_factor: float = 1.0,
) -> MoEDispatch:
    return MoEDispatch(
        operation,
        scale_factor,
        3584,
        16,
        896,
        1,
        ep_size,
        ep_size,
        operation == "dispatch",
        quant_mode=common.MoEQuantMode.fp8_block,
        provider=PROVIDER,
        operation=operation,
        ep_ranks_per_node=ep_ranks_per_node,
        dispatch_format=dispatch_format,
        sms=sms,
        max_tokens_per_rank=0,
    )


def test_deepep_ht_queries_dispatch_and_combine_separately(tmp_path):
    database = _build_database(
        tmp_path,
        [
            _row("dispatch", 1, 0.1),
            _row("dispatch", 4, 0.4),
            _row("combine", 1, 0.2),
            _row("combine", 4, 0.8),
        ],
    )

    dispatch = _operation("dispatch", scale_factor=2.0).query(database, x=2)
    combine = _operation("combine", scale_factor=2.0).query(database, x=2)

    assert float(dispatch) == pytest.approx(0.4)
    assert float(combine) == pytest.approx(0.8)
    assert dispatch.source == "silicon"
    assert combine.source == "silicon"


@pytest.mark.parametrize(
    "operation",
    [
        _operation("dispatch", ep_size=32),
        _operation("dispatch", ep_ranks_per_node=4),
        _operation("dispatch", dispatch_format="bf16"),
        _operation("dispatch", sms=16),
    ],
)
def test_deepep_ht_requires_exact_structure_and_topology(
    tmp_path,
    operation,
):
    database = _build_database(tmp_path, [_row("dispatch", 1, 0.1)])

    with pytest.raises(PerfDataNotAvailableError):
        operation.query(database, x=1)


def test_deepep_ht_rejects_token_extrapolation(tmp_path):
    database = _build_database(
        tmp_path,
        [_row("dispatch", 1, 0.1), _row("dispatch", 4, 0.4)],
    )

    with pytest.raises(PerfDataNotAvailableError, match="does not bracket"):
        _operation("dispatch").query(database, x=8)


def test_deepep_ht_loader_rejects_conflicting_physical_key(tmp_path):
    database = _build_database(
        tmp_path,
        [_row("dispatch", 1, 0.1), _row("dispatch", 1, 0.2)],
    )

    with pytest.raises(ValueError, match="conflicting Step4 DeepEP HT row"):
        _operation("dispatch").query(database, x=1)
