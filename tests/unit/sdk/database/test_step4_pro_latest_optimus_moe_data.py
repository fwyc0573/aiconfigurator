"""Consumer tests for pinned Step4-Pro-Latest Optimus FP8 MoE rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import MoE
from aiconfigurator.sdk.operations.moe import load_step4_optimus_moe_data

pytestmark = pytest.mark.unit


def _write_rows(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    "provider,num_tokens,hidden_size,inter_size,topk,num_experts,"
                    "moe_tp_size,moe_ep_size,moe_dtype,distribution,activation,"
                    "latency,framework,version,device_name,op_name,kernel_source"
                ),
                (
                    "optimus_fp8_moe,128,3584,3584,16,896,1,16,fp8_block,"
                    "power_law_1.2,situ_glu,0.5,VLLM,0.19.0,"
                    "NVIDIA B300 SXM6 AC,step4_optimus_moe,optimus_fp8_moe"
                ),
                (
                    "optimus_fp8_moe,256,3584,3584,16,896,1,16,fp8_block,"
                    "power_law_1.2,situ_glu,0.9,VLLM,0.19.0,"
                    "NVIDIA B300 SXM6 AC,step4_optimus_moe,optimus_fp8_moe"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_step4_optimus_moe_data_keys_structure_then_tokens(tmp_path):
    path = tmp_path / "step4_optimus_moe_perf.txt"
    _write_rows(path)

    data = load_step4_optimus_moe_data(path)
    key = (
        "optimus_fp8_moe",
        3584,
        3584,
        16,
        896,
        1,
        16,
        "fp8_block",
        "power_law_1.2",
        "situ_glu",
    )
    assert data[key] == {
        128: {"latency": 0.5, "energy": 0.0},
        256: {"latency": 0.9, "energy": 0.0},
    }


def test_step4_optimus_moe_query_interpolates_only_tokens(tmp_path):
    path = tmp_path / "step4_optimus_moe_perf.txt"
    _write_rows(path)
    data = load_step4_optimus_moe_data(path)

    class Database:
        _step4_optimus_moe_data = data

        @staticmethod
        def _interp_pr(latency, *, energy=0.0):
            from aiconfigurator.sdk.operations.base import PerformanceResult

            return PerformanceResult(latency, energy=energy, source="silicon")

    operation = MoE(
        "experts",
        1.0,
        3584,
        3584,
        16,
        896,
        1,
        16,
        common.MoEQuantMode.fp8_block,
        "power_law_1.2",
        1,
        provider="optimus_fp8_moe",
        activation="situ_glu",
    )
    operation.load_data = lambda database: None

    result = operation.query(Database(), x=192)
    assert float(result) == pytest.approx(0.7)
    assert result.source == "silicon"
