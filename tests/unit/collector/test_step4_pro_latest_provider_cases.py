"""Collector contracts for pinned Step4-Pro-Latest provider operations."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest

from collector.case_generator import get_gemm_case_specs, get_step4_model_gemm_case_specs
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
    assert latest.package_version == "0.19.0.post20.dev26+gc820e5ae1"
    assert latest.image() == PINNED_IMAGE
    assert latest.source_commit == PINNED_COMMIT


def test_step4_latest_runtime_profile_rejects_stock_vllm_package():
    runtime = validate_collector_runtime(
        "vllm",
        "0.19.0.post20.dev26+gc820e5ae1",
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


def test_vllm_registry_exposes_distinct_optimus_moe_provider_file():
    entries = {entry.op: entry for entry in VLLM_REGISTRY}

    moe = entries["step4_optimus_moe"]
    assert moe.module == "collector.vllm.collect_step4_provider"
    assert moe.get_func == "get_step4_optimus_moe_test_cases"
    assert moe.run_func == "run_step4_optimus_moe"
    assert moe.perf_filename == "step4_optimus_moe_perf.txt"


def test_vllm_wideep_registry_exposes_distinct_step4_deepep_ht_file():
    from collector.wideep.vllm.registry import REGISTRY as WIDEEP_VLLM_REGISTRY

    entries = {entry.op: entry for entry in WIDEEP_VLLM_REGISTRY}
    deepep = entries["step4_deepep_ht"]

    assert deepep.module == "collector.wideep.vllm.collect_step4_deepep_ht"
    assert deepep.get_func == "get_step4_deepep_ht_test_cases"
    assert deepep.run_func == "run_step4_deepep_ht"
    assert deepep.perf_filename == "step4_deepep_ht_perf.txt"


def test_step4_deepep_ht_cases_preserve_exact_topology_and_transport_identity():
    from collector.wideep.vllm.collect_step4_deepep_ht import (
        get_step4_deepep_ht_test_cases,
    )

    cases = get_step4_deepep_ht_test_cases()

    assert len(cases) == 58
    assert len(cases) == len({tuple(case) for case in cases})
    assert {
        (
            case[0],
            case[1],
            case[2],
            case[3],
            case[4],
            case[5],
            case[7],
            case[8],
            case[9],
        )
        for case in cases
    } == {
        (
            "vllm_deepep_high_throughput",
            ep_size,
            8,
            3584,
            896,
            16,
            "fp8_e4m3_block128",
            20,
            0,
        )
        for ep_size in (16, 32)
    }
    assert {case[6] for case in cases} == {
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        32768,
        65536,
    }


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


def test_step4_provider_token_sweep_adds_65k_without_expanding_generic_gemm():
    from collector.vllm.collect_step4_provider import (
        get_step4_fp32_output_gemm_test_cases,
        get_step4_grouped_gemm_test_cases,
        get_step4_qkv_norm_rope_test_cases,
    )

    generic_tokens = {case.x for case in get_gemm_case_specs("vllm")}
    expected_provider_tokens = generic_tokens | {65_536}
    grouped_tokens = {case[2] for case in get_step4_grouped_gemm_test_cases()}
    router_tokens = {case[1] for case in get_step4_fp32_output_gemm_test_cases()}
    qkv_tokens = {case[1] for case in get_step4_qkv_norm_rope_test_cases()}

    assert max(generic_tokens) == 32_768
    assert 65_536 not in generic_tokens
    assert grouped_tokens == expected_provider_tokens
    assert router_tokens == expected_provider_tokens
    assert qkv_tokens == expected_provider_tokens
    assert len(grouped_tokens) == 75
    assert len(router_tokens) == 75
    assert len(get_step4_qkv_norm_rope_test_cases()) == 150


def test_step4_qkv_runtime_keeps_pinned_vllm_config_context_active(monkeypatch):
    from collector.vllm import collect_step4_provider as provider

    assert hasattr(provider, "_run_step4_qkv_norm_rope_in_context")

    state = {"active": False}
    config = object()

    @contextmanager
    def fake_set_current_vllm_config(actual_config):
        assert actual_config is config
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False

    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    config_module = types.ModuleType("vllm.config")
    config_module.VllmConfig = lambda: config
    config_module.set_current_vllm_config = fake_set_current_vllm_config
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)

    captured = {}

    def fake_run_in_context(*args, **kwargs):
        assert state["active"]
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(provider, "_run_step4_qkv_norm_rope_in_context", fake_run_in_context)

    provider.run_step4_qkv_norm_rope(
        "vllm_step4pro_k_norm_rope",
        1,
        "k",
        64,
        1,
        512,
        perf_filename="step4_qkv_norm_rope_perf.txt",
    )

    assert not state["active"]
    assert captured == {
        "args": (
            "vllm_step4pro_k_norm_rope",
            1,
            "k",
            64,
            1,
            512,
        ),
        "kwargs": {
            "perf_filename": "step4_qkv_norm_rope_perf.txt",
            "device": "cuda:0",
        },
    }


def test_step4_optimus_moe_cases_preserve_pinned_ep_and_routing_identities():
    from collector.vllm.collect_step4_provider import (
        get_step4_optimus_moe_test_cases,
    )

    cases = get_step4_optimus_moe_test_cases()
    assert len(cases) == 174
    assert len(cases) == len({tuple(case) for case in cases})
    assert {
        (
            case[0],
            case[2],
            case[3],
            case[4],
            case[5],
            case[6],
            case[7],
            case[8],
            case[9],
            case[10],
        )
        for case in cases
    } == {
        (
            "optimus_fp8_moe",
            3584,
            3584,
            16,
            896,
            1,
            ep_size,
            "fp8_block",
            distribution,
            "situ_glu",
        )
        for ep_size in (16, 32)
        for distribution in ("balanced", "power_law_1.01", "power_law_1.2")
    }
    assert {case[1] for case in cases} == {
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        32768,
        65536,
    }


def test_step4_optimus_moe_runtime_contract_uses_local_block_scales():
    from collector.vllm.collect_step4_provider import (
        _step4_optimus_moe_scale_shapes,
    )

    assert _step4_optimus_moe_scale_shapes(
        num_experts=896,
        moe_ep_size=16,
        hidden_size=3584,
        inter_size=3584,
    ) == ((56, 56, 28), (56, 28, 28))
    assert _step4_optimus_moe_scale_shapes(
        num_experts=896,
        moe_ep_size=32,
        hidden_size=3584,
        inter_size=3584,
    ) == ((28, 56, 28), (28, 28, 28))


def test_step4_optimus_provider_path_matches_pinned_masked_contiguous_boundary():
    from collector.vllm.collect_step4_provider import (
        _step4_optimus_provider_path,
    )

    assert _step4_optimus_provider_path(
        local_num_tokens=6143,
        contiguous_threshold=6144,
    ) == ("deepgemm_optimus_moe_masked_fp8", True)
    assert _step4_optimus_provider_path(
        local_num_tokens=6144,
        contiguous_threshold=6144,
    ) == ("deepgemm_optimus_moe_fp8", False)


def test_step4_optimus_hidden_states_are_deterministic_bfloat16():
    import torch

    from collector.vllm.collect_step4_provider import (
        _step4_optimus_hidden_states,
    )

    first = _step4_optimus_hidden_states((4, 8), device="cpu")
    second = _step4_optimus_hidden_states((4, 8), device="cpu")

    assert first.dtype == torch.bfloat16
    torch.testing.assert_close(first, second)


def test_step4_optimus_hidden_states_do_not_consume_default_rng():
    import torch

    from collector.vllm.collect_step4_provider import (
        _step4_optimus_hidden_states,
    )

    torch.manual_seed(20260815)
    expected_next = torch.randn((4,), dtype=torch.float32)

    torch.manual_seed(20260815)
    _step4_optimus_hidden_states((4, 8), device="cpu")
    actual_next = torch.randn((4,), dtype=torch.float32)

    torch.testing.assert_close(actual_next, expected_next)


def test_step4_optimus_workspace_validation_compares_bytes_after_dtype_view():
    import torch

    from collector.vllm.collect_step4_provider import (
        _step4_optimus_reserve_pinned_workspaces,
    )

    class Workspace:
        def __init__(self):
            self.buffer = torch.empty((4,), dtype=torch.bfloat16)

    class OptimusModule:
        @staticmethod
        def _get_ws_size(hidden_states, w1, topk_ids):
            del hidden_states, w1, topk_ids
            return (8, 8, 8)

        @staticmethod
        def _get_workspaces(hidden_states, w1, topk_ids):
            del hidden_states, w1, topk_ids
            return Workspace(), Workspace(), Workspace()

    _step4_optimus_reserve_pinned_workspaces(
        OptimusModule,
        torch_module=torch,
        device="cpu",
        max_num_tokens=1,
        hidden_size=1,
        inter_size=1,
        topk=1,
        max_local_num_experts=1,
    )


def test_step4_optimus_rank0_workload_keeps_global_ids_without_negative_sentinels():
    import torch

    from collector.vllm.collect_step4_provider import (
        _step4_optimus_rank0_workload,
    )

    topk_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
        ],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [
            [0, 60],
            [60, 70],
            [1, 2],
        ],
        dtype=torch.int64,
    )

    local_weights, global_ids = _step4_optimus_rank0_workload(
        topk_weights,
        topk_ids,
        local_num_experts=56,
        num_experts=896,
    )

    assert global_ids.tolist() == [[0, 895], [1, 2]]
    torch.testing.assert_close(
        local_weights,
        torch.tensor([[0.75, 0.0], [0.55, 0.45]], dtype=torch.float32),
    )
    assert torch.all(global_ids >= 0)


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


def test_qkv_norm_rope_runtime_probe_rejects_non_finite_outputs():
    import torch

    from collector.vllm.collect_step4_provider import (
        _validate_qkv_norm_rope_probe,
    )

    finite = torch.ones((2, 3), dtype=torch.bfloat16)
    outputs = _validate_qkv_norm_rope_probe(
        finite,
        expected_shapes=((2, 3),),
        expected_dtype=torch.bfloat16,
    )
    assert len(outputs) == 1
    assert outputs[0] is finite

    non_finite = finite.clone()
    non_finite[0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _validate_qkv_norm_rope_probe(
            non_finite,
            expected_shapes=((2, 3),),
            expected_dtype=torch.bfloat16,
        )


def test_qkv_norm_rope_runtime_probe_rejects_unexpected_shape():
    import torch

    from collector.vllm.collect_step4_provider import (
        _validate_qkv_norm_rope_probe,
    )

    with pytest.raises(RuntimeError, match="unexpected shapes"):
        _validate_qkv_norm_rope_probe(
            torch.ones((2, 4), dtype=torch.bfloat16),
            expected_shapes=((2, 3),),
            expected_dtype=torch.bfloat16,
        )


def test_qkv_norm_rope_runtime_probe_rejects_unexpected_dtype():
    import torch

    from collector.vllm.collect_step4_provider import (
        _validate_qkv_norm_rope_probe,
    )

    with pytest.raises(RuntimeError, match="unexpected dtype"):
        _validate_qkv_norm_rope_probe(
            torch.ones((2, 3), dtype=torch.float32),
            expected_shapes=((2, 3),),
            expected_dtype=torch.bfloat16,
        )


def test_step4_context_attention_cases_cover_both_providers_and_required_workloads():
    from collector.vllm.collect_step4_provider import (
        get_step4_context_attention_test_cases,
    )

    cases = get_step4_context_attention_test_cases()
    assert len(cases) == 68
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
    assert (2, 512, 512) in workloads
    assert (1, 4_096, 4_096) in workloads
    assert (1, 16_384, 16_384) in workloads
    assert (1, 1_048_544, 1_048_544) in workloads
    assert (1, 8_192, 131_072) in workloads
    assert (1, 65_536, 1_048_544) in workloads
    scheduler_workloads = {
        (1, 4_096, 8_192),
        (1, 16_384, 32_768),
        (1, 16_384, 262_144),
        (1, 8_160, 1_048_544),
        (1, 32_736, 1_048_544),
    }
    assert {
        (case[0], case[1], case[2], case[3]) for case in cases if (case[1], case[2], case[3]) in scheduler_workloads
    } == {
        (provider, *workload)
        for provider in {
            "optimus_fa4",
            "vllm_native_sliding_gqa",
        }
        for workload in scheduler_workloads
    }
    exact_65k = [case for case in cases if (case[1], case[2], case[3]) == (1, 65_536, 65_536)]
    assert {case[0] for case in exact_65k} == {
        "optimus_fa4",
        "vllm_native_sliding_gqa",
    }


def test_step4_generation_attention_cases_cover_decode_search_without_exceeding_cache_cap():
    from collector.vllm.collect_step4_provider import (
        STEP4_ATTENTION_MAX_KV_CACHE_BYTES,
        _step4_attention_physical_cache_bytes,
        get_step4_generation_attention_test_cases,
    )

    cases = get_step4_generation_attention_test_cases()
    assert len(cases) == 167
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
    assert ("optimus_fa4", 32, 1_048_576) in {(case[0], case[1], case[2]) for case in cases}
    assert ("optimus_fa4", 64, 1_048_576) not in {(case[0], case[1], case[2]) for case in cases}
    assert ("vllm_native_sliding_gqa", 2048, 1_048_576) in {(case[0], case[1], case[2]) for case in cases}


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
