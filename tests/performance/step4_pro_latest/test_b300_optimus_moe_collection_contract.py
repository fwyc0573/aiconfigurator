"""Static contracts for the Step4-Pro-Latest B300 Optimus MoE collector."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent
SHARED_HOST_SCRIPT = ROOT / "run_b300_attention_collection.sh"
HOST_SCRIPT = ROOT / "run_b300_optimus_moe_collection.sh"
REMOTE_SCRIPT = ROOT / "remote_b300_optimus_moe_collection.sh"


def test_shared_host_wrapper_accepts_an_exact_remote_collection_script():
    source = SHARED_HOST_SCRIPT.read_text(encoding="utf-8")

    assert "REMOTE_COLLECTION_SCRIPT_HOST" in source
    assert "RESULT_MARKER" in source
    assert "HOST_RESULT_MARKER" in source
    assert "WORKLOAD_LABEL" in source


def test_optimus_host_wrapper_selects_b300_and_moe_markers():
    source = HOST_SCRIPT.read_text(encoding="utf-8")

    assert "remote_b300_optimus_moe_collection.sh" in source
    assert "B300_OPTIMUS_MOE_COLLECTION" in source
    assert "B300_OPTIMUS_MOE_HOST" in source
    assert "b300_train_infra" in source
    assert "MODE" in source
    assert 'exec bash "${REPO_ROOT}/tests/performance/step4_pro_latest/run_b300_attention_collection.sh"' in source


def test_remote_collection_applies_only_approved_optimus_runtime_overlays():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert "OPTIMUS_WHEEL_URL:?OPTIMUS_WHEEL_URL is required" in source
    assert "OPTIMUS_WHEEL_SHA256:?OPTIMUS_WHEEL_SHA256 is required" in source
    assert "step_optimus-3.23.24.dist-info/" in source
    assert "optimus_cutedsl.group_quant_fp8" in source
    assert "VLLM_USE_DEEP_GEMM_E8M0=1" in source
    assert "VLLM_USE_OPTIMUS_MOE=1" in source
    assert "VLLM_OPTIMUS_MOE_MIN_CONTG_SIZE=6144" in source
    assert "OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE=1" in source
    assert "deep_gemm_ep_gather_masked.py" in source
    assert "hidden_size & -hidden_size" in source
    assert "OptimusFp8Experts" in source
    assert "expected step-optimus==3.23.24, got {step_optimus_version}" in source


def test_remote_collection_covers_masked_contiguous_ep16_and_ep32():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert "num_tokens=128" in source
    assert "num_tokens=6144" in source
    assert "moe_ep_size=16" in source
    assert "moe_ep_size=32" in source
    assert 'distribution="balanced"' in source
    assert 'distribution="power_law_1.01"' in source
    assert 'distribution="power_law_1.2"' in source
    assert "--ops step4_optimus_moe" in source
    assert "step4_optimus_moe_perf.txt" in source
    assert "deepgemm_optimus_moe_masked_fp8" in source
    assert "deepgemm_optimus_moe_fp8" in source
    assert "execution_mode" in source
    assert "used_cuda_graph" in source
    assert "remote_result_ready" in source
