"""Collector contracts for pinned Step4-Pro-Latest provider operations."""

from __future__ import annotations

import pytest

from collector.case_generator import get_step4_model_gemm_case_specs
from collector.framework_manifest import get_collector_runtime, validate_collector_runtime
from collector.model_cases import (
    build_collection_case_plan,
    default_architecture_cases_path,
)
from collector.vllm.registry import REGISTRY as VLLM_REGISTRY

pytestmark = pytest.mark.unit

LATEST_MODEL = "stepfun-ai/Step4-Pro-Latest"
LATEST_ARCHITECTURE = "Step4ProForCausalLM"
LATEST_RUNTIME_PROFILE = "step4_pro_latest"
PINNED_IMAGE = (
    "hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled"
)
PINNED_COMMIT = "607d1641ee3fec43653fca510d717725828890c2"


def test_step4_latest_runtime_profile_keeps_stock_vllm_default_unchanged():
    default = get_collector_runtime("vllm")
    latest = get_collector_runtime("vllm", profile=LATEST_RUNTIME_PROFILE)

    assert default.version == "0.19.0"
    assert default.image() == "vllm/vllm-openai:v0.19.0"
    assert latest.profile == LATEST_RUNTIME_PROFILE
    assert latest.version == "0.19.0"
    assert latest.package_version == "0.19.0.post20.dev26.gc820e5ae1"
    assert latest.image() == PINNED_IMAGE
    assert latest.source_commit == PINNED_COMMIT


def test_step4_latest_runtime_profile_rejects_stock_vllm_package():
    runtime = validate_collector_runtime(
        "vllm",
        "0.19.0.post20.dev26.gc820e5ae1",
        profile=LATEST_RUNTIME_PROFILE,
    )
    assert runtime.profile == LATEST_RUNTIME_PROFILE

    with pytest.raises(RuntimeError, match="requires package version"):
        validate_collector_runtime(
            "vllm",
            "0.19.0",
            profile=LATEST_RUNTIME_PROFILE,
        )


def test_step4_latest_sm103_plan_selects_only_required_provider_ops(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", LATEST_MODEL)
    plan = build_collection_case_plan(
        backend="vllm",
        model_path=LATEST_MODEL,
        gpu_type="b300_sxm",
    )

    assert plan.model_architecture == LATEST_ARCHITECTURE
    assert plan.model_cases_paths == [default_architecture_cases_path(LATEST_ARCHITECTURE)]
    assert plan.runtime_profile == LATEST_RUNTIME_PROFILE
    assert set(plan.op_cases) == {
        "gemm",
        "step4_context_attention",
        "step4_deepep_ht",
        "step4_fp32_output_gemm",
        "step4_generation_attention",
        "step4_grouped_gemm",
        "step4_optimus_moe",
        "step4_qkv_norm_rope",
    }


def test_step4_latest_plan_rejects_non_sm103_target():
    with pytest.raises(ValueError, match="requires SM versions \\[103\\]"):
        build_collection_case_plan(
            backend="vllm",
            model_path=LATEST_MODEL,
            gpu_type="h800_sxm",
        )


def test_default_full_plan_excludes_custom_runtime_profile():
    plan = build_collection_case_plan(backend="vllm", gpu_type="b300_sxm", full=True)

    assert plan.runtime_profile is None
    assert "step4_grouped_gemm" not in plan.op_cases


def test_vllm_registry_exposes_distinct_grouped_gemm_provider_file():
    entries = {entry.op: entry for entry in VLLM_REGISTRY}

    grouped = entries["step4_grouped_gemm"]
    assert grouped.module == "collector.vllm.collect_step4_provider"
    assert grouped.get_func == "get_step4_grouped_gemm_test_cases"
    assert grouped.run_func == "run_step4_grouped_gemm"
    assert grouped.perf_filename == "step4_grouped_gemm_perf.txt"


def test_step4_grouped_gemm_cases_preserve_exact_einsum_identity():
    from collector.vllm.collect_step4_provider import (
        get_step4_grouped_gemm_test_cases,
    )

    cases = get_step4_grouped_gemm_test_cases()
    assert cases
    assert {(case[0], case[1], case[3], case[4], case[5]) for case in cases} == {
        ("vllm_step4pro_torch_einsum", 8, 1024, 4096, "bfloat16")
    }
    assert len(cases) == len({case[2] for case in cases})


def test_step4_latest_generic_gemm_excludes_provider_specific_gemms():
    specs = get_step4_model_gemm_case_specs(LATEST_MODEL, backend="vllm")

    assert specs
    assert {dtype for spec in specs for dtype in (spec.gemm_types or ())} == {"bfloat16"}
    assert (896, 7168) not in {(spec.n, spec.k) for spec in specs}
