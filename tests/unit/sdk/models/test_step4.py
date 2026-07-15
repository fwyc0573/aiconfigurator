# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Step4 predefined model."""

from copy import deepcopy

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config, models, utils
from aiconfigurator.sdk.models import check_is_moe
from aiconfigurator.sdk.models.base import _MODEL_REGISTRY
from aiconfigurator.sdk.models.helpers import calc_expectation
from aiconfigurator.sdk.models.step4 import Step4Model

pytestmark = pytest.mark.unit

EXPECTED_BLOCK_TYPES = ("dense_swa",) * 4 + ("moe_full",) * 23 + ("moe_swa",) * 65


def _step4_raw_config() -> dict:
    """Return an isolated copy of the package-local Step4 config."""
    return deepcopy(utils._load_pre_downloaded_hf_config("stepfun-ai/Step4"))


def _build_step4_model(
    *,
    tp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
    nextn: int = 0,
) -> Step4Model:
    """Build Step4 with an explicit vLLM parallel and MTP contract."""
    model_config = config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        nextn=nextn,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )
    model = models.get_model("stepfun-ai/Step4", model_config, backend_name="vllm")
    assert isinstance(model, Step4Model)
    return model


def _walk_ops(operation_list):
    """Yield top-level operations and nested overlap-group operations."""
    for operation in operation_list:
        yield operation
        if isinstance(operation, ops.OverlapOp):
            yield from _walk_ops(operation._group_a)
            yield from _walk_ops(operation._group_b)


def _ops_by_name(operation_list) -> dict[str, object]:
    """Index a graph recursively and reject duplicate operation names."""
    indexed = {}
    for operation in _walk_ops(operation_list):
        assert operation._name not in indexed, f"duplicate operation name: {operation._name}"
        indexed[operation._name] = operation
    return indexed


@pytest.fixture(autouse=True)
def _clear_model_config_caches():
    """Keep model-loader cache state local to each test."""
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    yield
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()


def test_step4_family_and_architecture_are_registered():
    """Step4 should have a dedicated family and architecture mapping."""
    assert "STEP4" in common.ModelFamily
    assert common.ARCHITECTURE_TO_MODEL_FAMILY["Step4ForCausalLM"] == "STEP4"


def test_step4_model_id_is_a_cached_default():
    """The local Step4 identity should resolve without a remote lookup."""
    assert "stepfun-ai/Step4" in common.DefaultHFModels


def test_step4_cached_config_never_downloads_from_huggingface(monkeypatch):
    """The predefined Step4 config must be loaded from package data."""

    def fail_download(_model_id):
        raise AssertionError("Step4 cached config must not download from HuggingFace")

    monkeypatch.setattr(utils, "_download_hf_config", fail_download)

    model_info = utils.get_model_config_from_model_path("stepfun-ai/Step4")

    assert model_info["architecture"] == "Step4ForCausalLM"
    assert model_info["layers"] == 92
    assert model_info["hidden_size"] == 4096
    assert model_info["num_experts"] == 352
    assert model_info["topk"] == 8
    assert model_info["raw_config"]["num_nextn_predict_layers"] == 3


def test_step4_model_module_is_auto_discovered():
    """Package import should register the dedicated Step4 model class."""
    assert "STEP4" in _MODEL_REGISTRY
    assert _MODEL_REGISTRY["STEP4"].__name__ == "Step4Model"


def test_step4_model_defines_its_factory():
    """Step4 should construct through the standard model registry contract."""
    assert "create" in Step4Model.__dict__


def test_step4_config_parses_normalized_blocks_and_attention_geometry():
    """The cached config should preserve architecture labels and MLA geometry."""
    model_info = utils._parse_hf_config_json(_step4_raw_config())

    extra = model_info["extra_params"]
    assert isinstance(extra, common.Step4Config)
    assert extra.block_types == EXPECTED_BLOCK_TYPES
    assert extra.block_types.count("dense_swa") == 4
    assert extra.block_types.count("moe_full") == 23
    assert extra.block_types.count("moe_swa") == 65
    assert len(extra.block_types) == 92

    assert extra.full_num_attention_heads == 64
    assert extra.full_num_key_value_heads == 8
    assert extra.sliding_num_attention_heads == 96
    assert extra.sliding_num_key_value_heads == 8
    assert extra.attention_head_dim == 128
    assert extra.sliding_window_size == 512

    assert extra.q_lora_rank == 1536
    assert extra.kv_lora_rank == 512
    assert extra.qk_nope_head_dim == 128
    assert extra.qk_rope_head_dim == 64
    assert extra.v_head_dim == 128
    assert extra.dense_inter_size == 13824
    assert extra.shared_expert_inter_size == 1536


def test_step4_block_count_must_match_trunk_layers():
    """A normalized block sequence must cover every trunk layer exactly once."""
    config = _step4_raw_config()
    config["block_types"] = list(EXPECTED_BLOCK_TYPES[:-1])

    with pytest.raises(ValueError, match="block_types length 91 != num_hidden_layers 92"):
        utils._parse_hf_config_json(config)


def test_step4_rejects_unknown_block_label():
    """Unknown Step4 block labels must fail instead of being silently grouped."""
    config = _step4_raw_config()
    config["block_types"] = list(EXPECTED_BLOCK_TYPES)
    config["block_types"][4] = "moe_unknown"

    with pytest.raises(ValueError, match="unsupported block type 'moe_unknown'"):
        utils._parse_hf_config_json(config)


@pytest.mark.parametrize(
    "field",
    [
        "block_types",
        "full_num_attention_heads",
        "full_num_key_value_heads",
        "sliding_num_attention_heads",
        "sliding_num_key_value_heads",
        "attention_head_dim",
        "sliding_window_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "shared_expert_intermediate_size",
    ],
)
def test_step4_missing_required_structure_fails_fast(field):
    """Required Step4 modeling fields must never default to invented values."""
    config = _step4_raw_config()
    config.pop(field, None)

    with pytest.raises(ValueError, match=f"missing required field '{field}'"):
        utils._parse_hf_config_json(config)


@pytest.mark.parametrize(
    "field",
    [
        "full_num_attention_heads",
        "full_num_key_value_heads",
        "sliding_num_attention_heads",
        "sliding_num_key_value_heads",
        "attention_head_dim",
        "sliding_window_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "intermediate_size",
        "shared_expert_intermediate_size",
    ],
)
def test_step4_structure_dimensions_must_be_positive(field):
    """Invalid dimensions must fail before the model operation graph is built."""
    config = _step4_raw_config()
    config[field] = 0

    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        utils._parse_hf_config_json(config)


def test_step4_topk_cannot_exceed_routed_expert_count():
    """The routed top-k must fit within the declared routed experts."""
    config = _step4_raw_config()
    config["num_experts_per_tok"] = config["n_routed_experts"] + 1

    with pytest.raises(ValueError, match="num_experts_per_tok 353 exceeds n_routed_experts 352"):
        utils._parse_hf_config_json(config)


def test_step4_is_classified_as_moe():
    """Step4 must enter the MoE parallelism path."""
    assert check_is_moe("stepfun-ai/Step4") is True


def test_step4_model_builds_granular_graph_with_fp8_defaults():
    """Step4 should infer FP8 weights/KV while vLLM keeps MLA FMHA in BF16."""
    model = _build_step4_model()

    assert model.config.gemm_quant_mode == common.GEMMQuantMode.fp8
    assert model.config.moe_quant_mode == common.MoEQuantMode.fp8
    assert model.config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    assert model.config.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert model.context_ops
    assert model.generation_ops


def test_step4_declares_complete_full_and_swa_attention_semantics():
    """Aggregate scheduling must classify every granular attention operation."""
    model = _build_step4_model()

    assert model.MIXED_STEP_CONTEXT_ATTENTION_KEYS == (
        "context_full_mla_approx_attn_norm",
        "context_full_mla_approx_downscale_gemm",
        "context_full_mla_approx_q_b_proj_gemm",
        "context_full_mla_approx_kv_b_proj_gemm",
        "context_full_mla_approx_attention",
        "context_full_mla_approx_proj_gemm",
        "context_full_mla_approx_attention_ar",
        "context_swa_mla_approx_attn_norm",
        "context_swa_mla_approx_downscale_gemm",
        "context_swa_mla_approx_q_b_proj_gemm",
        "context_swa_mla_approx_kv_b_proj_gemm",
        "context_swa_mla_approx_attention",
        "context_swa_mla_approx_proj_gemm",
        "context_swa_mla_approx_attention_ar",
    )
    assert model.MIXED_STEP_GENERATION_ATTENTION_KEYS == (
        "generation_full_mla_approx_attn_norm",
        "generation_full_mla_approx_downscale_gemm",
        "generation_full_mla_approx_q_b_proj_gemm",
        "generation_full_mla_approx_bmm_pre",
        "generation_full_mla_approx_attention",
        "generation_full_mla_approx_bmm_post",
        "generation_full_mla_approx_proj_gemm",
        "generation_full_mla_approx_attention_ar",
        "generation_swa_mla_approx_attn_norm",
        "generation_swa_mla_approx_downscale_gemm",
        "generation_swa_mla_approx_q_b_proj_gemm",
        "generation_swa_mla_approx_bmm_pre",
        "generation_swa_mla_approx_attention",
        "generation_swa_mla_approx_bmm_post",
        "generation_swa_mla_approx_proj_gemm",
        "generation_swa_mla_approx_attention_ar",
    )


def test_step4_attention_semantic_contract_matches_attention_builders_exactly():
    """Adding an attention component must require updating the static contract."""
    model = _build_step4_model()
    _dense_count, full_count, swa_count = model._layer_counts()

    context_names = tuple(
        operation._name
        for label, count in (("full", full_count), ("swa", swa_count))
        for operation in model._context_attention_ops(label, count)
    )
    generation_names = tuple(
        operation._name
        for label, count in (("full", full_count), ("swa", swa_count))
        for operation in model._generation_attention_ops(label, count)
    )

    assert context_names == model.MIXED_STEP_CONTEXT_ATTENTION_KEYS
    assert generation_names == model.MIXED_STEP_GENERATION_ATTENTION_KEYS
    assert len(context_names) == len(set(context_names)) == 14
    assert len(generation_names) == len(set(generation_names)) == 16


def test_step4_rejects_unvalidated_non_vllm_backends():
    """The predefined graph must fail fast outside its reviewed vLLM scope."""
    model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=0,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )

    with pytest.raises(NotImplementedError, match="Step4 predefined ops currently support only backend='vllm'"):
        models.get_model("stepfun-ai/Step4", model_config, backend_name="trtllm")


def test_step4_activation_and_mla_kv_cache_contract():
    """Step4 MLA approximation should use H=4096 activations and 576 KV elements per layer."""
    model = _build_step4_model()

    assert model.activation_hidden_size == 4096
    assert model.get_kvcache_elements_per_token() == 92 * (512 + 64) == 52992


def test_step4_attention_approximation_preserves_full_and_swa_audit_groups():
    """Full and SWA labels must remain separately replaceable despite sharing temporary MLA."""
    model = _build_step4_model()
    context = _ops_by_name(model.context_ops)
    generation = _ops_by_name(model.generation_ops)

    assert context["context_full_mla_approx_attention"]._scale_factor == 23
    assert context["context_swa_mla_approx_attention"]._scale_factor == 69
    assert generation["generation_full_mla_approx_attention"]._scale_factor == 23
    assert generation["generation_swa_mla_approx_attention"]._scale_factor == 69
    assert sum(context[name]._scale_factor for name in context if name.endswith("_mla_approx_attention")) == 92


@pytest.mark.parametrize("label,count", [("full", 23), ("swa", 69)])
def test_step4_context_mla_projection_geometry_and_quantization(label, count):
    """Context MLA should expose every accepted DeepSeek-V3 projection dimension."""
    model = _build_step4_model()
    indexed = _ops_by_name(model.context_ops)
    prefix = f"context_{label}_mla_approx"

    downscale = indexed[f"{prefix}_downscale_gemm"]
    q_b_proj = indexed[f"{prefix}_q_b_proj_gemm"]
    kv_b_proj = indexed[f"{prefix}_kv_b_proj_gemm"]
    attention = indexed[f"{prefix}_attention"]
    projection = indexed[f"{prefix}_proj_gemm"]
    reduction = indexed[f"{prefix}_attention_ar"]

    assert (downscale._n, downscale._k) == (2112, 4096)
    assert (q_b_proj._n, q_b_proj._k) == (24576, 1536)
    assert (kv_b_proj._n, kv_b_proj._k) == (32768, 512)
    assert isinstance(attention, ops.ContextMLA)
    assert attention._num_heads == 128
    assert attention._kvcache_quant_mode == common.KVCacheQuantMode.fp8
    assert attention._fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert (projection._n, projection._k) == (4096, 16384)
    assert isinstance(reduction, ops.CustomAllReduce)
    assert reduction._tp_size == 1
    assert all(
        operation._quant_mode == common.GEMMQuantMode.fp8 for operation in (downscale, q_b_proj, kv_b_proj, projection)
    )
    assert all(
        operation._scale_factor == count for operation in (downscale, q_b_proj, kv_b_proj, attention, projection)
    )


@pytest.mark.parametrize("label,count", [("full", 23), ("swa", 69)])
def test_step4_generation_mla_projection_and_bmm_geometry(label, count):
    """Generation MLA should retain explicit projections and both factorized BMMs."""
    model = _build_step4_model()
    indexed = _ops_by_name(model.generation_ops)
    prefix = f"generation_{label}_mla_approx"

    downscale = indexed[f"{prefix}_downscale_gemm"]
    q_b_proj = indexed[f"{prefix}_q_b_proj_gemm"]
    bmm_pre = indexed[f"{prefix}_bmm_pre"]
    attention = indexed[f"{prefix}_attention"]
    bmm_post = indexed[f"{prefix}_bmm_post"]
    projection = indexed[f"{prefix}_proj_gemm"]

    assert (downscale._n, downscale._k) == (2112, 4096)
    assert (q_b_proj._n, q_b_proj._k) == (24576, 1536)
    assert isinstance(bmm_pre, ops.MLABmm) and bmm_pre._if_pre is True
    assert isinstance(attention, ops.GenerationMLA)
    assert isinstance(bmm_post, ops.MLABmm) and bmm_post._if_pre is False
    assert bmm_pre._num_heads == attention._num_heads == bmm_post._num_heads == 128
    assert bmm_pre._quant_mode == bmm_post._quant_mode == common.GEMMQuantMode.fp8
    assert attention._kv_cache_dtype == common.KVCacheQuantMode.fp8
    assert (projection._n, projection._k) == (4096, 16384)
    assert all(
        operation._scale_factor == count
        for operation in (downscale, q_b_proj, bmm_pre, attention, bmm_post, projection)
    )


def test_step4_dense_ffn_uses_gated_swiglu_with_exact_parameter_count():
    """The four-layer dense prefix should match the CSV H*I*3 parameter formula."""
    model = _build_step4_model()
    context = _ops_by_name(model.context_ops)

    norm = context["context_dense_ffn_norm"]
    gate_up = context["context_dense_gate_up_gemm"]
    activation = context["context_dense_swiglu"]
    down = context["context_dense_down_gemm"]
    reduction = context["context_dense_ffn_ar"]

    assert norm._scale_factor == gate_up._scale_factor == activation._scale_factor == down._scale_factor == 4
    assert (norm._dim_in, norm._dim_out) == (8192, 8192)
    assert (gate_up._n, gate_up._k) == (27648, 4096)
    assert (activation._dim_in, activation._dim_out) == (27648, 13824)
    assert (down._n, down._k) == (4096, 13824)
    assert gate_up._quant_mode == down._quant_mode == common.GEMMQuantMode.fp8
    assert isinstance(reduction, ops.CustomAllReduce)

    parameters_per_layer = gate_up._n * gate_up._k + down._n * down._k
    assert parameters_per_layer == 169869312
    assert parameters_per_layer * 4 == 679477248


def test_step4_context_moe_has_routed_shared_and_bf16_merge_paths():
    """Context MoE should be sequential with FP8 routed/shared work and BF16 boundaries."""
    model = _build_step4_model()
    context = _ops_by_name(model.context_ops)

    norm = context["context_moe_ffn_norm"]
    router = context["context_moe_router_gemm"]
    pre_dispatch = context["context_moe_pre_dispatch"]
    routed = context["context_moe"]
    post_dispatch = context["context_moe_post_dispatch"]
    shared_gate_up = context["context_shared_gate_up_gemm"]
    shared_activation = context["context_shared_swiglu"]
    shared_down = context["context_shared_down_gemm"]
    shared_reduction = context["context_shared_ffn_ar"]
    merge = context["context_moe_shared_merge"]

    assert norm._scale_factor == router._scale_factor == routed._scale_factor == merge._scale_factor == 88
    assert (norm._dim_in, norm._dim_out) == (8192, 8192)
    assert (router._n, router._k, router._quant_mode) == (352, 4096, common.GEMMQuantMode.bfloat16)
    assert pre_dispatch._pre_dispatch is True and pre_dispatch._reduce_results is False
    assert post_dispatch._pre_dispatch is False
    assert pre_dispatch._quant_mode == post_dispatch._quant_mode == common.MoEQuantMode.fp8
    assert (routed._hidden_size, routed._inter_size, routed._topk, routed._num_experts) == (4096, 1536, 8, 352)
    assert routed._quant_mode == common.MoEQuantMode.fp8
    assert routed._is_context is True and routed._is_gated is True
    assert (shared_gate_up._n, shared_gate_up._k) == (3072, 4096)
    assert (shared_activation._dim_in, shared_activation._dim_out) == (3072, 1536)
    assert (shared_down._n, shared_down._k) == (4096, 1536)
    assert shared_gate_up._quant_mode == shared_down._quant_mode == common.GEMMQuantMode.fp8
    assert isinstance(shared_reduction, ops.CustomAllReduce)
    assert (merge._dim_in, merge._dim_out) == (8192, 4096)
    assert not any(isinstance(operation, ops.OverlapOp) for operation in model.context_ops)


def test_step4_generation_moe_overlaps_routed_and_shared_paths():
    """Generation should overlap routed/shared work and mark dispatch/compute as decode."""
    model = _build_step4_model()
    generation = _ops_by_name(model.generation_ops)
    overlap = generation["generation_moe_overlap"]

    assert isinstance(overlap, ops.OverlapOp)
    assert [operation._name for operation in overlap._group_a] == [
        "generation_moe_router_gemm",
        "generation_moe_pre_dispatch",
        "generation_moe",
        "generation_moe_post_dispatch",
    ]
    assert [operation._name for operation in overlap._group_b] == [
        "generation_shared_gate_up_gemm",
        "generation_shared_swiglu",
        "generation_shared_down_gemm",
        "generation_shared_ffn_ar",
    ]

    pre_dispatch = generation["generation_moe_pre_dispatch"]
    routed = generation["generation_moe"]
    post_dispatch = generation["generation_moe_post_dispatch"]
    merge = generation["generation_moe_shared_merge"]
    assert pre_dispatch._is_context is False and pre_dispatch._reduce_results is False
    assert routed._is_context is False and routed._is_gated is True
    assert post_dispatch._is_context is False
    assert (merge._dim_in, merge._dim_out) == (8192, 4096)


def test_step4_graph_excludes_profiled_fallback_and_deepseek_v4_modules():
    """Step4 must remain a granular SOL graph without profiled module wrappers."""
    model = _build_step4_model()
    operation_types = {type(operation).__name__ for operation in _walk_ops(model.context_ops + model.generation_ops)}

    assert {"ContextMLA", "GenerationMLA", "MLABmm", "MoE", "MoEDispatch", "OverlapOp"} <= operation_types
    assert operation_types.isdisjoint(
        {
            "FallbackOp",
            "MLAModule",
            "WideEPContextMLA",
            "WideEPGenerationMLA",
            "ContextDeepSeekV4AttentionModule",
            "GenerationDeepSeekV4AttentionModule",
            "DeepSeekV4MegaMoEModule",
        }
    )
    assert operation_types <= {
        "Embedding",
        "ElementWise",
        "GEMM",
        "ContextMLA",
        "GenerationMLA",
        "MLABmm",
        "MoE",
        "MoEDispatch",
        "CustomAllReduce",
        "P2P",
        "OverlapOp",
    }


def test_step4_tp_shards_projection_and_ffn_geometry_with_explicit_reductions():
    """TP=4 should shard only local projection/FFN dimensions and retain explicit reductions."""
    model = _build_step4_model(tp_size=4, moe_tp_size=1, moe_ep_size=4)
    context = _ops_by_name(model.context_ops)
    generation = _ops_by_name(model.generation_ops)

    assert context["context_full_mla_approx_q_b_proj_gemm"]._n == 24576 // 4
    assert context["context_full_mla_approx_kv_b_proj_gemm"]._n == 32768 // 4
    assert context["context_full_mla_approx_attention"]._num_heads == 128 // 4
    assert context["context_full_mla_approx_proj_gemm"]._k == 16384 // 4
    assert context["context_dense_gate_up_gemm"]._n == 27648 // 4
    assert context["context_dense_swiglu"]._dim_out == 13824 // 4
    assert context["context_dense_down_gemm"]._k == 13824 // 4
    assert context["context_shared_gate_up_gemm"]._n == 3072 // 4
    assert context["context_shared_down_gemm"]._k == 1536 // 4
    assert generation["generation_full_mla_approx_bmm_pre"]._num_heads == 128 // 4

    for name in (
        "context_full_mla_approx_attention_ar",
        "context_swa_mla_approx_attention_ar",
        "context_dense_ffn_ar",
        "context_shared_ffn_ar",
    ):
        assert context[name]._tp_size == 4
    assert context["context_moe_pre_dispatch"]._reduce_results is False
    assert generation["generation_moe_pre_dispatch"]._reduce_results is False


def test_step4_mtp_scales_generation_only_including_overlap_inner_ops():
    """nextn=3 should recursively scale every generation leaf and no context leaf."""
    baseline = _build_step4_model(nextn=0)
    mtp = _build_step4_model(nextn=3)
    expected_scale = 1.0 / (1.0 + calc_expectation(3, [0.85, 0.3, 0.0, 0.0, 0.0])) * (92 + 3) / 92

    assert baseline._mtp_scale_factor == 1.0
    assert mtp._mtp_scale_factor == pytest.approx(0.4905504492409377)
    assert mtp._mtp_scale_factor == pytest.approx(expected_scale)

    baseline_context = {operation._name: operation._scale_factor for operation in _walk_ops(baseline.context_ops)}
    mtp_context = {operation._name: operation._scale_factor for operation in _walk_ops(mtp.context_ops)}
    assert mtp_context == baseline_context

    baseline_generation = {
        operation._name: operation._scale_factor
        for operation in _walk_ops(baseline.generation_ops)
        if not isinstance(operation, ops.OverlapOp)
    }
    mtp_generation = {
        operation._name: operation._scale_factor
        for operation in _walk_ops(mtp.generation_ops)
        if not isinstance(operation, ops.OverlapOp)
    }
    assert mtp_generation.keys() == baseline_generation.keys()
    for name, scale_factor in baseline_generation.items():
        assert mtp_generation[name] == pytest.approx(scale_factor * expected_scale), name
