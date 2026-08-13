"""Unit contracts for exact-runtime Step4 modular MoE manifests."""

import pytest

from tests.performance.step4_moe_manifest import (
    _parse_invocation,
    _validate_rows_for_invocation,
)

pytestmark = pytest.mark.unit


INVOCATION = (
    "vllm.moe:run_moe_torch:['fp8', [1, 4096], 6144, 2048, 16, 1024, "
    "1, 64, 'stepfun-ai/Step4-Pro-V3', 'power_law', 1.2]"
)


def _row(num_tokens: int, *, kernel_source: str = "vllm_flashinfer_cutlass_moe") -> dict[str, object]:
    return {
        "framework": "VLLM",
        "version": "0.19.0",
        "device": "NVIDIA H800",
        "op_name": "moe",
        "kernel_source": kernel_source,
        "moe_dtype": "fp8",
        "num_tokens": num_tokens,
        "hidden_size": 6144,
        "inter_size": 2048,
        "topk": 16,
        "num_experts": 1024,
        "moe_tp_size": 1,
        "moe_ep_size": 64,
        "distribution": "power_law_1.2",
        "latency": 1.25,
    }


def test_moe_manifest_parses_exact_modular_invocation_contract():
    parsed = _parse_invocation(INVOCATION)

    assert parsed["model"] == "stepfun-ai/Step4-Pro-V3"
    assert parsed["tokens"] == (1, 4096)
    assert parsed["structural_key"] == (6144, 2048, 16, 1024, 1, 64, "fp8", "power_law_1.2")


def test_moe_manifest_requires_exact_kernel_and_complete_token_bijection():
    parsed = _parse_invocation(INVOCATION)

    with pytest.raises(ValueError, match="kernel_source"):
        _validate_rows_for_invocation([_row(1, kernel_source="vllm_fused_moe"), _row(4096)], parsed)
    with pytest.raises(ValueError, match="token coverage"):
        _validate_rows_for_invocation([_row(1)], parsed)

    _validate_rows_for_invocation([_row(1), _row(4096)], parsed)
