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


def test_vllm_registry_exposes_distinct_fp32_router_provider_file():
    entries = {entry.op: entry for entry in VLLM_REGISTRY}

    router = entries["step4_fp32_output_gemm"]
    assert router.module == "collector.vllm.collect_step4_provider"
    assert router.get_func == "get_step4_fp32_output_gemm_test_cases"
    assert router.run_func == "run_step4_fp32_output_gemm"
    assert router.perf_filename == "step4_fp32_output_gemm_perf.txt"


def test_vllm_registry_exposes_distinct_qkv_norm_rope_provider_file():
    entries = {entry.op: entry for entry in VLLM_REGISTRY}

    qkv = entries["step4_qkv_norm_rope"]
    assert qkv.module == "collector.vllm.collect_step4_provider"
    assert qkv.get_func == "get_step4_qkv_norm_rope_test_cases"
    assert qkv.run_func == "run_step4_qkv_norm_rope"
    assert qkv.perf_filename == "step4_qkv_norm_rope_perf.txt"


def test_vllm_registry_exposes_distinct_step4_attention_provider_files():
    entries = {entry.op: entry for entry in VLLM_REGISTRY}

    context = entries["step4_context_attention"]
    assert context.module == "collector.vllm.collect_step4_provider"
    assert context.get_func == "get_step4_context_attention_test_cases"
    assert context.run_func == "run_step4_context_attention"
    assert context.perf_filename == "step4_context_attention_perf.txt"

    generation = entries["step4_generation_attention"]
    assert generation.module == "collector.vllm.collect_step4_provider"
    assert generation.get_func == "get_step4_generation_attention_test_cases"
    assert generation.run_func == "run_step4_generation_attention"
    assert generation.perf_filename == "step4_generation_attention_perf.txt"


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


def test_step4_fp32_router_cases_preserve_exact_optimus_identity():
    from collector.vllm.collect_step4_provider import (
        get_step4_fp32_output_gemm_test_cases,
    )

    cases = get_step4_fp32_output_gemm_test_cases()
    assert cases
    assert {(case[0], case[2], case[3], case[4], case[5]) for case in cases} == {
        ("vllm.optimus_matmul_fp32", 896, 7168, "bfloat16", "float32")
    }
    assert len(cases) == len({case[1] for case in cases})


def test_step4_qkv_norm_rope_cases_preserve_full_and_swa_identities():
    from collector.vllm.collect_step4_provider import (
        get_step4_qkv_norm_rope_test_cases,
    )

    cases = get_step4_qkv_norm_rope_test_cases()
    assert cases
    assert {(case[0], case[2], case[3], case[4], case[5]) for case in cases} == {
        ("vllm_step4pro_k_norm_rope", "k", 64, 1, 512),
        ("vllm_step4pro_qkv_norm_rope", "q+k+v", 128, 8, 128),
    }
    assert len(cases) == 2 * len({case[1] for case in cases})


def test_full_qkv_norm_rope_runtime_probe_requires_query_and_key_outputs():
    from collector.vllm.collect_step4_provider import (
        _qkv_norm_rope_expected_output_shapes,
    )

    assert _qkv_norm_rope_expected_output_shapes(
        "vllm_step4pro_k_norm_rope",
        num_tokens=17,
        q_heads=64,
        kv_heads=1,
        head_dim=512,
    ) == (
        (17, 64, 512),
        (17, 1, 512),
    )


def test_step4_context_attention_cases_cover_both_providers_and_required_workloads():
    from collector.vllm.collect_step4_provider import (
        get_step4_context_attention_test_cases,
    )

    cases = get_step4_context_attention_test_cases()
    assert len(cases) == 50
    assert len(cases) == len({tuple(case) for case in cases})
    assert {
        (
            case[0],
            case[4],
            case[5],
            case[6],
            case[7],
            case[8],
            case[9],
            case[10],
            case[11],
            case[12],
            case[13],
            case[14],
        )
        for case in cases
    } == {
        (
            "optimus_fa4",
            64,
            1,
            512,
            0,
            "bfloat16",
            "bfloat16",
            True,
            128,
            524288,
            524288,
            "NHD",
        ),
        (
            "vllm_native_sliding_gqa",
            128,
            8,
            128,
            512,
            "bfloat16",
            "bfloat16",
            False,
            128,
            524288,
            262144,
            "NHD",
        ),
    }

    workloads = {(case[1], case[2], case[3]) for case in cases}
    assert (32, 512, 512) in workloads
    assert (1, 1_048_544, 1_048_544) in workloads
    assert (1, 8_192, 131_072) in workloads
    assert (1, 65_536, 1_048_544) in workloads


def test_step4_generation_attention_cases_cover_decode_search_without_exceeding_cache_cap():
    from collector.vllm.collect_step4_provider import (
        STEP4_ATTENTION_MAX_KV_CACHE_BYTES,
        _step4_attention_physical_cache_bytes,
        get_step4_generation_attention_test_cases,
    )

    cases = get_step4_generation_attention_test_cases()
    assert len(cases) == 149
    assert len(cases) == len({tuple(case) for case in cases})
    assert {
        (
            case[0],
            case[3],
            case[4],
            case[5],
            case[6],
            case[7],
            case[8],
            case[9],
            case[10],
            case[11],
            case[12],
            case[13],
        )
        for case in cases
    } == {
        (
            "optimus_fa4",
            64,
            1,
            512,
            0,
            "bfloat16",
            "bfloat16",
            True,
            128,
            524288,
            524288,
            "NHD",
        ),
        (
            "vllm_native_sliding_gqa",
            128,
            8,
            128,
            512,
            "bfloat16",
            "bfloat16",
            False,
            128,
            524288,
            262144,
            "NHD",
        ),
    }

    assert all(
        _step4_attention_physical_cache_bytes(
            batch_size=case[1],
            query_tokens=1,
            total_context_tokens=case[2],
            window_size=case[6],
            page_size=case[10],
            physical_page_bytes=case[11],
        )
        <= STEP4_ATTENTION_MAX_KV_CACHE_BYTES
        for case in cases
    )
    assert ("optimus_fa4", 32, 1_048_544) in {(case[0], case[1], case[2]) for case in cases}
    assert ("optimus_fa4", 64, 1_048_544) not in {(case[0], case[1], case[2]) for case in cases}
    assert ("vllm_native_sliding_gqa", 2048, 1_048_544) in {(case[0], case[1], case[2]) for case in cases}


def test_step4_attention_materializes_only_live_swa_blocks():
    from collector.vllm.collect_step4_provider import (
        _step4_attention_materialized_block_range,
    )

    assert _step4_attention_materialized_block_range(
        query_tokens=1,
        total_context_tokens=513,
        window_size=512,
        page_size=128,
    ) == (0, 5)
    assert _step4_attention_materialized_block_range(
        query_tokens=1,
        total_context_tokens=640,
        window_size=512,
        page_size=128,
    ) == (1, 5)
    assert _step4_attention_materialized_block_range(
        query_tokens=8192,
        total_context_tokens=1_048_544,
        window_size=512,
        page_size=128,
    ) == (8123, 8192)


def test_step4_attention_cache_stride_contract_matches_pinned_vllm():
    from collector.vllm.collect_step4_provider import (
        _step4_attention_expected_cache_strides_bytes,
    )

    assert _step4_attention_expected_cache_strides_bytes(
        num_blocks=3,
        kv_storage_alias=True,
        physical_page_bytes=524288,
    ) == (524288, 0)
    assert _step4_attention_expected_cache_strides_bytes(
        num_blocks=3,
        kv_storage_alias=False,
        physical_page_bytes=524288,
    ) == (262144, 786432)


def test_step4_attention_runtime_probe_uses_phase_specific_query_shape():
    from collector.vllm.collect_step4_provider import (
        _step4_attention_expected_output_shape,
    )

    assert _step4_attention_expected_output_shape(
        batch_size=2,
        query_tokens=512,
        num_heads=64,
        head_dim=512,
    ) == (1024, 64, 512)
    assert _step4_attention_expected_output_shape(
        batch_size=32,
        query_tokens=1,
        num_heads=128,
        head_dim=128,
    ) == (32, 128, 128)


def test_step4_latest_generic_gemm_excludes_provider_specific_gemms():
    specs = get_step4_model_gemm_case_specs(LATEST_MODEL, backend="vllm")

    assert specs
    assert {dtype for spec in specs for dtype in (spec.gemm_types or ())} == {"bfloat16"}
    assert (896, 7168) not in {(spec.n, spec.k) for spec in specs}
