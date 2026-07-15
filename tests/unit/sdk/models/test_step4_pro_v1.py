# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the cached Step4-Pro-V1 architecture contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from aiconfigurator.sdk import common, config, models, utils
from aiconfigurator.sdk.models.step4 import Step4Model

pytestmark = pytest.mark.unit

MODEL_ID = "stepfun-ai/Step4-Pro-V1"
EXPECTED_BLOCK_TYPES = ("dense_swa",) * 4 + ("moe_full",) * 20 + ("moe_swa",) * 56


@pytest.fixture(autouse=True)
def _clear_model_config_caches():
    """Keep cached model discovery isolated between tests."""
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    yield
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()


def _step4_pro_raw_config() -> dict:
    """Return an isolated copy of the package-local Step4-Pro-V1 config."""
    return deepcopy(utils._load_pre_downloaded_hf_config(MODEL_ID))


def _operations_by_name(operation_list) -> dict[str, object]:
    """Index top-level operations by their unique names."""
    return {operation._name: operation for operation in operation_list}


def _build_synthetic_step4_model(
    monkeypatch,
    raw_config: dict,
    *,
    tp_size: int,
    moe_tp_size: int = 1,
    moe_ep_size: int | None = None,
) -> Step4Model:
    """Build one synthetic Step4 through the real registry and config parser."""
    model_info = utils._parse_hf_config_json(deepcopy(raw_config))
    model_info["raw_config"] = deepcopy(raw_config)
    monkeypatch.setattr(models, "_get_model_info", lambda _model_path: model_info)
    model_config = config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=moe_tp_size,
        moe_ep_size=tp_size if moe_ep_size is None else moe_ep_size,
        nextn=0,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )

    model = models.get_model("synthetic/Step4", model_config, backend_name="vllm")

    assert isinstance(model, Step4Model)
    return model


def test_step4_pro_v1_is_a_cached_default():
    """The exact model identity must resolve through package-local data."""
    assert MODEL_ID in common.DefaultHFModels


def test_step4_pro_v1_cached_config_never_downloads_from_huggingface(monkeypatch):
    """Cached discovery must not use a network fallback."""

    def fail_download(_model_id):
        raise AssertionError("Step4-Pro-V1 cached config must not download from HuggingFace")

    monkeypatch.setattr(utils, "_download_hf_config", fail_download)

    model_info = utils.get_model_config_from_model_path(MODEL_ID)

    assert model_info["architecture"] == "Step4ForCausalLM"
    assert model_info["layers"] == 80
    assert model_info["hidden_size"] == 6144
    assert model_info["inter_size"] == 16384
    assert model_info["num_experts"] == 512
    assert model_info["topk"] == 8
    assert model_info["moe_inter_size"] == 2048
    assert model_info["vocab"] == 128896


def test_step4_pro_v1_config_matches_authoritative_topology():
    """The cached config must preserve every CSV-backed structural value."""
    model_info = utils._parse_hf_config_json(_step4_pro_raw_config())
    extra = model_info["extra_params"]

    assert isinstance(extra, common.Step4Config)
    assert extra.block_types == EXPECTED_BLOCK_TYPES
    assert extra.block_types.count("dense_swa") == 4
    assert extra.block_types.count("moe_full") == 20
    assert extra.block_types.count("moe_swa") == 56
    assert len(extra.block_types) == 80
    assert extra.full_num_attention_heads == 64
    assert extra.sliding_num_attention_heads == 96
    assert extra.dense_inter_size == 16384
    assert extra.shared_expert_inter_size == 2048

    assert model_info["layers"] == 80
    assert model_info["hidden_size"] == 6144
    assert model_info["inter_size"] == 16384
    assert model_info["num_experts"] == 512
    assert model_info["topk"] == 8
    assert model_info["moe_inter_size"] == 2048
    assert model_info["vocab"] == 128896


def test_step4_pro_v1_parameter_arithmetic_closes_authoritative_totals():
    """Dense, MoE, attention, RMS, embedding, and activation totals must close exactly."""
    model_info = utils._parse_hf_config_json(_step4_pro_raw_config())
    extra = model_info["extra_params"]
    hidden_size = model_info["hidden_size"]
    dense_inter_size = model_info["inter_size"]
    routed_inter_size = model_info["moe_inter_size"]
    shared_inter_size = extra.shared_expert_inter_size
    num_experts = model_info["num_experts"]
    topk = model_info["topk"]
    layers = model_info["layers"]
    vocab = model_info["vocab"]

    dense_per_layer = 3 * hidden_size * dense_inter_size
    router_per_layer = hidden_size * num_experts
    active_moe_per_layer = router_per_layer + 3 * hidden_size * (topk * routed_inter_size + shared_inter_size)
    all_moe_per_layer = router_per_layer + 3 * hidden_size * (num_experts * routed_inter_size + shared_inter_size)
    attention_total = 20 * 153_095_232 + 60 * 213_911_648
    rms_total = 2 * hidden_size * layers
    total_without_embedding = 4 * dense_per_layer + 76 * all_moe_per_layer + attention_total + rms_total
    embedding_total = 2 * vocab * hidden_size
    total_activation = 4 * dense_per_layer + 76 * active_moe_per_layer + attention_total + rms_total

    assert dense_per_layer == 301_989_888
    assert router_per_layer == 3_145_728
    assert active_moe_per_layer == 342_884_352
    assert all_moe_per_layer == 19_368_247_296
    assert attention_total == 15_896_603_520
    assert rms_total == 983_040
    assert total_without_embedding == 1_489_092_340_608
    assert embedding_total == 1_583_874_048
    assert total_without_embedding + embedding_total == 1_490_676_214_656
    assert total_activation == 43_164_756_864


def test_original_step4_projection_formulas_preserve_existing_widths():
    """Config-derived formulas must remain numerically identical for the original Step4."""
    raw = deepcopy(utils._load_pre_downloaded_hf_config("stepfun-ai/Step4"))

    downscale_output = raw["q_lora_rank"] + raw["kv_lora_rank"] + raw["qk_rope_head_dim"]
    q_b_output = raw["num_attention_heads"] * (raw["qk_nope_head_dim"] + raw["qk_rope_head_dim"])
    kv_b_output = raw["num_attention_heads"] * (raw["qk_nope_head_dim"] + raw["v_head_dim"])

    assert downscale_output == 2112
    assert q_b_output == 24576
    assert kv_b_output == 32768


def test_step4_projection_shapes_are_derived_from_synthetic_geometry(monkeypatch):
    """A second Step4 geometry must not inherit the original model's projection literals."""
    raw = _step4_pro_raw_config()
    raw.update(
        {
            "num_attention_heads": 64,
            "q_lora_rank": 1024,
            "kv_lora_rank": 256,
            "qk_nope_head_dim": 96,
            "qk_rope_head_dim": 32,
            "v_head_dim": 64,
        }
    )
    model = _build_synthetic_step4_model(monkeypatch, raw, tp_size=4)
    context = _operations_by_name(model.context_ops)
    generation = _operations_by_name(model.generation_ops)

    for label in ("full", "swa"):
        context_prefix = f"context_{label}_mla_approx"
        generation_prefix = f"generation_{label}_mla_approx"
        assert (context[f"{context_prefix}_downscale_gemm"]._n, context[f"{context_prefix}_downscale_gemm"]._k) == (
            1312,
            6144,
        )
        assert (context[f"{context_prefix}_q_b_proj_gemm"]._n, context[f"{context_prefix}_q_b_proj_gemm"]._k) == (
            2048,
            1024,
        )
        assert (context[f"{context_prefix}_kv_b_proj_gemm"]._n, context[f"{context_prefix}_kv_b_proj_gemm"]._k) == (
            2560,
            256,
        )
        assert context[f"{context_prefix}_attention"]._num_heads == 16
        assert context[f"{context_prefix}_proj_gemm"]._k == 1024
        assert generation[f"{generation_prefix}_downscale_gemm"]._n == 1312
        assert generation[f"{generation_prefix}_q_b_proj_gemm"]._n == 2048
        assert generation[f"{generation_prefix}_bmm_pre"]._num_heads == 16
        assert generation[f"{generation_prefix}_proj_gemm"]._k == 1024


@pytest.mark.parametrize(
    "field",
    [
        "moe_intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
    ],
)
def test_step4_required_routed_moe_fields_cannot_be_missing(field):
    """Step4 routed-MoE geometry must never inherit generic defaults."""
    raw = _step4_pro_raw_config()
    raw.pop(field)

    with pytest.raises(ValueError, match=f"Step4 config missing required field '{field}'"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("moe_intermediate_size", 0, id="moe-intermediate-zero"),
        pytest.param("moe_intermediate_size", True, id="moe-intermediate-bool"),
        pytest.param("moe_intermediate_size", 2048.0, id="moe-intermediate-float"),
        pytest.param("n_routed_experts", 0, id="routed-experts-zero"),
        pytest.param("n_routed_experts", True, id="routed-experts-bool"),
        pytest.param("n_routed_experts", 512.0, id="routed-experts-float"),
        pytest.param("num_experts_per_tok", 0, id="topk-zero"),
        pytest.param("num_experts_per_tok", True, id="topk-bool"),
        pytest.param("num_experts_per_tok", 8.0, id="topk-float"),
    ],
)
def test_step4_routed_moe_fields_must_be_positive_integers(field, invalid_value):
    """Python truthiness and numeric coercion must not admit malformed geometry."""
    raw = _step4_pro_raw_config()
    raw[field] = invalid_value

    with pytest.raises(ValueError, match=f"Step4 {field} must be a positive integer"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    "field",
    [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    ],
)
def test_step4_core_dimensions_reject_boolean_values(field):
    """Boolean values must not pass as integers in model-wide Step4 geometry."""
    raw = _step4_pro_raw_config()
    raw[field] = True

    with pytest.raises(ValueError, match=f"Step4 {field} must be a positive integer"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("field", "value", "tp_size", "moe_tp_size", "moe_ep_size", "parallel_name"),
    [
        pytest.param("num_attention_heads", 130, 4, 1, 4, "tp_size", id="attention-heads"),
        pytest.param("vocab_size", 128897, 4, 1, 4, "tp_size", id="vocab"),
        pytest.param("intermediate_size", 16385, 4, 1, 4, "tp_size", id="dense-intermediate"),
        pytest.param(
            "shared_expert_intermediate_size",
            2049,
            4,
            1,
            4,
            "tp_size",
            id="shared-intermediate",
        ),
        pytest.param("n_routed_experts", 513, 4, 1, 4, "moe_ep_size", id="routed-experts"),
        pytest.param("moe_intermediate_size", 2050, 4, 4, 1, "moe_tp_size", id="routed-intermediate"),
    ],
)
def test_step4_model_rejects_non_divisible_parallel_geometry(
    monkeypatch,
    field,
    value,
    tp_size,
    moe_tp_size,
    moe_ep_size,
    parallel_name,
):
    """Step4 must reject geometry that integer sharding would truncate."""
    raw = _step4_pro_raw_config()
    raw[field] = value

    with pytest.raises(
        ValueError,
        match=rf"Step4 {field} \({value}\) must be divisible by {parallel_name}",
    ):
        _build_synthetic_step4_model(
            monkeypatch,
            raw,
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )
