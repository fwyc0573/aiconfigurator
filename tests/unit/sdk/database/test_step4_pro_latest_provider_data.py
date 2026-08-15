"""Consumer tests for Step4-Pro-Latest provider-specific perf rows."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations import FP32OutputGEMM, GroupedGEMM
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

PROVIDER = "vllm_step4pro_torch_einsum"
HEADER = "framework,version,device,op_name,kernel_source,provider,groups,num_tokens,n,k,quant_mode,latency\n"
ROUTER_PROVIDER = "vllm.optimus_matmul_fp32"
ROUTER_HEADER = (
    "framework,version,device,op_name,kernel_source,provider,num_tokens,n,k,weight_dtype,output_dtype,latency\n"
)


def _build_database(tmp_path: Path, rows: list[str]) -> PerfDatabase:
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "vllm" / "0.19.0"
    data_dir.mkdir(parents=True)
    source_spec = Path("src/aiconfigurator/systems/b300_sxm.yaml")
    system_spec = yaml.safe_load(source_spec.read_text(encoding="utf-8"))
    system_spec["data_dir"] = "data"
    (systems_root / "b300_sxm.yaml").write_text(
        yaml.safe_dump(system_spec),
        encoding="utf-8",
    )
    (data_dir / "step4_grouped_gemm_perf.txt").write_text(
        HEADER + "".join(rows),
        encoding="utf-8",
    )
    return PerfDatabase("b300_sxm", "vllm", "0.19.0", str(systems_root))


def _grouped_operation(
    *,
    provider: str = PROVIDER,
    n: int = 1024,
    scale_factor: float = 1.0,
) -> GroupedGEMM:
    return GroupedGEMM(
        "wo_a",
        scale_factor,
        n,
        4096,
        common.GEMMQuantMode.bfloat16,
        groups=8,
        provider=provider,
    )


def _build_router_database(tmp_path: Path, rows: list[str]) -> PerfDatabase:
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "vllm" / "0.19.0"
    data_dir.mkdir(parents=True)
    source_spec = Path("src/aiconfigurator/systems/b300_sxm.yaml")
    system_spec = yaml.safe_load(source_spec.read_text(encoding="utf-8"))
    system_spec["data_dir"] = "data"
    (systems_root / "b300_sxm.yaml").write_text(
        yaml.safe_dump(system_spec),
        encoding="utf-8",
    )
    (data_dir / "step4_fp32_output_gemm_perf.txt").write_text(
        ROUTER_HEADER + "".join(rows),
        encoding="utf-8",
    )
    return PerfDatabase("b300_sxm", "vllm", "0.19.0", str(systems_root))


def _router_operation(
    *,
    provider: str = ROUTER_PROVIDER,
    n: int = 896,
    scale_factor: float = 1.0,
) -> FP32OutputGEMM:
    return FP32OutputGEMM(
        "router",
        scale_factor,
        n,
        7168,
        provider=provider,
    )


def test_grouped_gemm_queries_exact_provider_shape_and_interpolates_tokens(tmp_path):
    database = _build_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_grouped_gemm,{PROVIDER},{PROVIDER},8,1,1024,4096,bfloat16,0.1\n",
            f"VLLM,0.19.0,B300,step4_grouped_gemm,{PROVIDER},{PROVIDER},8,4,1024,4096,bfloat16,0.4\n",
        ],
    )

    result = _grouped_operation(scale_factor=2.0).query(database, x=2)

    assert float(result) == pytest.approx(0.4)
    assert result.source == "silicon"


@pytest.mark.parametrize(
    "operation",
    [
        _grouped_operation(provider="generic_einsum"),
        _grouped_operation(n=2048),
    ],
)
def test_grouped_gemm_requires_exact_provider_and_structural_key(tmp_path, operation):
    database = _build_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_grouped_gemm,{PROVIDER},{PROVIDER},8,1,1024,4096,bfloat16,0.1\n",
        ],
    )

    with pytest.raises(PerfDataNotAvailableError):
        operation.query(database, x=1)


def test_grouped_gemm_rejects_non_positive_token_count(tmp_path):
    database = _build_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_grouped_gemm,{PROVIDER},{PROVIDER},8,1,1024,4096,bfloat16,0.1\n",
        ],
    )

    with pytest.raises(ValueError, match="positive num_tokens"):
        _grouped_operation().query(database, x=0)


def test_grouped_gemm_loader_rejects_conflicting_physical_key(tmp_path):
    database = _build_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_grouped_gemm,{PROVIDER},{PROVIDER},8,1,1024,4096,bfloat16,0.1\n",
            f"VLLM,0.19.0,B300,step4_grouped_gemm,{PROVIDER},{PROVIDER},8,1,1024,4096,bfloat16,0.2\n",
        ],
    )

    with pytest.raises(ValueError, match="conflicting grouped-GEMM row"):
        _grouped_operation().query(database, x=1)


def test_fp32_router_queries_exact_provider_shape_and_interpolates_tokens(tmp_path):
    database = _build_router_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_fp32_output_gemm,{ROUTER_PROVIDER},{ROUTER_PROVIDER},"
            "1,896,7168,bfloat16,float32,0.2\n",
            f"VLLM,0.19.0,B300,step4_fp32_output_gemm,{ROUTER_PROVIDER},{ROUTER_PROVIDER},"
            "4,896,7168,bfloat16,float32,0.8\n",
        ],
    )

    result = _router_operation(scale_factor=2.0).query(database, x=2)

    assert float(result) == pytest.approx(0.8)
    assert result.source == "silicon"


@pytest.mark.parametrize(
    "operation",
    [
        _router_operation(provider="generic_fp32_gemm"),
        _router_operation(n=1024),
    ],
)
def test_fp32_router_requires_exact_provider_and_structural_key(tmp_path, operation):
    database = _build_router_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_fp32_output_gemm,{ROUTER_PROVIDER},{ROUTER_PROVIDER},"
            "1,896,7168,bfloat16,float32,0.2\n",
        ],
    )

    with pytest.raises(PerfDataNotAvailableError):
        operation.query(database, x=1)


def test_fp32_router_rejects_non_positive_token_count(tmp_path):
    database = _build_router_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_fp32_output_gemm,{ROUTER_PROVIDER},{ROUTER_PROVIDER},"
            "1,896,7168,bfloat16,float32,0.2\n",
        ],
    )

    with pytest.raises(ValueError, match="positive num_tokens"):
        _router_operation().query(database, x=0)


def test_fp32_router_loader_rejects_conflicting_physical_key(tmp_path):
    database = _build_router_database(
        tmp_path,
        [
            f"VLLM,0.19.0,B300,step4_fp32_output_gemm,{ROUTER_PROVIDER},{ROUTER_PROVIDER},"
            "1,896,7168,bfloat16,float32,0.2\n",
            f"VLLM,0.19.0,B300,step4_fp32_output_gemm,{ROUTER_PROVIDER},{ROUTER_PROVIDER},"
            "1,896,7168,bfloat16,float32,0.3\n",
        ],
    )

    with pytest.raises(ValueError, match="conflicting FP32-output GEMM row"):
        _router_operation().query(database, x=1)
