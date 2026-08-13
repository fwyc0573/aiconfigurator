"""Collector-v2 population contracts for Step4-Pro V3/V4."""

import pytest

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
    get_attention_kv_cache_dtype_options,
    get_common_moe_test_cases,
    get_gemm_types_for_case,
    get_moe_quantization_modes,
    get_step4_model_gemm_case_specs,
    moe_model_allows_quantization,
)
from collector.model_cases import (
    build_collection_case_plan,
    default_architecture_cases_path,
    expected_failure_for_test_case,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model_path", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_step4_model_plan_selects_targeted_kernel_ops(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    plan = build_collection_case_plan(backend="vllm", model_path=model_path, gpu_type="h800_sxm")
    assert plan.model_architecture == "Step4ForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Step4ForCausalLM")]
    assert {"attention_context", "attention_generation", "gemm", "moe"} <= set(plan.op_cases)


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("model_path", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_step4_attention_profiles_expand_only_native_topologies(monkeypatch, phase, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    shape_sweeps = (
        get_attention_context_shape_sweeps("vllm")
        if phase == "context"
        else get_attention_generation_shape_sweeps("vllm")
    )
    configs = {
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for config in get_attention_head_configs(shape_sweeps[0], phase=phase)
    }
    assert configs == {
        (96, 12, 128, 0),
        (48, 6, 128, 0),
        (24, 3, 128, 0),
        (128, 16, 128, 512),
        (64, 8, 128, 512),
        (32, 4, 128, 512),
    }


@pytest.mark.parametrize(
    ("model_path", "hidden", "inter", "topk", "experts"),
    [
        ("stepfun-ai/Step4-Pro-V3", 6144, 2048, 16, 1024),
        ("stepfun-ai/Step4-Pro-V4", 9216, 3584, 8, 384),
    ],
)
def test_step4_moe_population_is_model_scoped_and_exact(monkeypatch, model_path, hidden, inter, topk, experts):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = get_common_moe_test_cases()
    assert cases
    assert {(case.hidden_size, case.inter_size, case.topk, case.num_experts) for case in cases} == {
        (hidden, inter, topk, experts)
    }
    assert {case.tp for case in cases} == {1}
    assert {case.token_expert_distribution for case in cases} == {"power_law"}
    assert {case.power_law_alpha for case in cases} == {1.2}
    assert all(experts % case.ep == 0 for case in cases)
    assert "fp8" in get_moe_quantization_modes("vllm", sm_version=90)
    assert moe_model_allows_quantization("vllm", model_path, "fp8")
    assert not moe_model_allows_quantization("vllm", model_path, "bfloat16")

    recipe_count = len(cases)
    raw_getter_count = len(cases)
    post_selector_count = sum(1 for case in cases if case.tp == 1 and experts % case.ep == 0)
    token_expanded_count = sum(len(case.num_tokens_list) for case in cases)
    persisted_keys = {
        (
            case.hidden_size,
            case.inter_size,
            case.topk,
            case.num_experts,
            case.tp,
            case.ep,
            case.token_expert_distribution,
            case.power_law_alpha,
            token,
        )
        for case in cases
        for token in case.num_tokens_list
    }
    assert (recipe_count, raw_getter_count, post_selector_count) == (7, 7, 7)
    assert token_expanded_count == 7 * len(cases[0].num_tokens_list)
    assert len(persisted_keys) == token_expanded_count


@pytest.mark.parametrize("model_path", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_step4_vllm_moe_has_no_expected_failures(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    plan = build_collection_case_plan(backend="vllm", model_path=model_path, gpu_type="h800_sxm")
    for common_case in get_common_moe_test_cases():
        # Match the positional case contract emitted by collector.vllm.collect_moe.get_moe_test_cases().
        case = [
            "fp8",
            common_case.num_tokens_list,
            common_case.hidden_size,
            common_case.inter_size,
            common_case.topk,
            common_case.num_experts,
            common_case.tp,
            common_case.ep,
            common_case.model_name,
            common_case.token_expert_distribution,
            common_case.power_law_alpha,
        ]
        assert (
            expected_failure_for_test_case(
                case,
                plan=plan.op_cases["moe"],
                full_module_name="vllm.moe",
                run_func_name="run_moe_torch",
                runtime_version="0.19.0",
            )
            is None
        )


@pytest.mark.parametrize("model_path", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_step4_gemm_population_uses_built_model_structural_shapes(model_path):
    specs = get_step4_model_gemm_case_specs(model_path, backend="vllm")
    assert specs
    structural_shapes = {(spec.n, spec.k) for spec in specs}
    assert len(structural_shapes) > 1
    assert all(spec.x > 0 for spec in specs)
    # Targeted collection must not schedule the shared global Cartesian feature grid.
    assert len(specs) < 5000
    assert {dtype for spec in specs for dtype in (spec.gemm_types or ())} == {"bfloat16", "fp8"}
    assert all("fp8_block" not in (spec.gemm_types or ()) for spec in specs)


@pytest.mark.parametrize("model_path", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_step4_gemm_selector_uses_exact_built_dtypes(model_path):
    specs = get_step4_model_gemm_case_specs(model_path, backend="vllm")
    selected = {dtype for spec in specs for dtype in get_gemm_types_for_case(spec, ["bfloat16", "fp8", "fp8_block"])}
    assert selected == {"bfloat16", "fp8"}
    assert "fp8_block" not in selected


def test_step4_attention_precision_policy_is_bfloat16_compute_with_fp8_kv():
    assert get_attention_kv_cache_dtype_options("stepfun-ai/Step4-Pro-V3", sm_version=90) == [True]
    assert get_attention_kv_cache_dtype_options("stepfun-ai/Step4-Pro-V4", sm_version=90) == [True]
    assert get_attention_kv_cache_dtype_options("unregistered/model", sm_version=90) == [False, True]
