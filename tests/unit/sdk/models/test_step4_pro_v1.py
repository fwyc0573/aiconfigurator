# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the cached Step4-Pro-V1 architecture contract."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import MISSING, FrozenInstanceError, fields, replace

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config, models, utils
from aiconfigurator.sdk.models.helpers import calc_expectation
from aiconfigurator.sdk.models.step4 import Step4Model

pytestmark = pytest.mark.unit

MODEL_ID = "stepfun-ai/Step4-Pro-V1"
STEP4_MFA_FIELD_NAMES = (
    "hidden_size",
    "attention_type",
    "num_query_heads",
    "output_groups",
    "q_lora_rank",
    "o_lora_rank",
    "projection_head_dim",
    "cache_projection_width",
    "cache_entry_width",
    "rope_head_dim",
    "retention_mode",
    "window_size",
    "compression_ratio",
    "window_allocation_policy",
    "cache_tp_policy",
    "inference_source",
    "target_parameter_count",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
)
STEP4_MFA_SECTION_FIELD_NAMES = tuple(field for field in STEP4_MFA_FIELD_NAMES if field != "hidden_size")
EXPECTED_BLOCK_TYPES = ("dense_swa",) * 4 + ("moe_full",) * 20 + ("moe_swa",) * 56
EXPECTED_LAYER_RECORDS = (
    *((layer_id, "nonfull", "dense") for layer_id in range(4)),
    *((layer_id, "full", "moe") for layer_id in range(4, 24)),
    *((layer_id, "nonfull", "moe") for layer_id in range(24, 80)),
)
FULL_ATTENTION_SUFFIXES = (
    "attn_norm",
    "q_proj_gemm",
    "k_proj_gemm",
    "v_proj_gemm",
    "attention",
    "o_proj_gemm",
    "attention_ar",
)
NONFULL_ATTENTION_SUFFIXES = (
    "attn_norm",
    "hca_attention",
    "attention_ar",
)
LEGACY_CONTEXT_ATTENTION_NAMES = (
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
LEGACY_GENERATION_ATTENTION_NAMES = (
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


def _candidate_pro_attention_sections() -> dict:
    """Return the reviewed explicit layer and attention schema."""
    return {
        "layers": [
            {"layer_id": layer_id, "attention_type": attention_type, "ffn_type": ffn_type}
            for layer_id, attention_type, ffn_type in EXPECTED_LAYER_RECORDS
        ],
        "full_attention": {
            "num_query_heads": 64,
            "num_kv_heads": 64,
            "q_head_dim": 96,
            "k_head_dim": 96,
            "v_head_dim": 96,
            "q_projection": "linear",
            "k_projection": "linear",
            "v_projection": "linear",
            "output_projection": "linear",
            "rope_dimension": 64,
            "latent_rank": None,
            "target_parameter_count": 153_095_232,
            "unknown_extra_projection_params": 0,
            "unknown_router_params": 0,
            "unknown_compression_params": 0,
        },
        "nonfull_attention": {
            "mechanism": "hca",
            "num_query_heads": 96,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "o_groups": 16,
            "head_dim": 512,
            "rope_dimension": 64,
            "window_size": 512,
            "compression_ratio": 128,
            "index_n_heads": 0,
            "index_head_dim": 0,
            "index_topk": 0,
            "target_parameter_count": 213_911_648,
            "unknown_extra_projection_params": 0,
            "unknown_router_params": 0,
            "unknown_compression_params": 0,
        },
    }


def _as_candidate_pro_schema(raw_config: dict) -> dict:
    """Replace the legacy attention section with the reviewed Pro schema."""
    raw_config = deepcopy(raw_config)
    raw_config.pop("block_types", None)
    raw_config.update(_candidate_pro_attention_sections())
    return raw_config


def _step4_mfa_attention_kwargs(**overrides) -> dict:
    """Return one complete V1 Full MFA config payload for direct construction."""
    kwargs = {
        "hidden_size": 6144,
        "attention_type": "mfa",
        "num_query_heads": 64,
        "output_groups": 8,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "projection_head_dim": 640,
        "cache_projection_width": 512,
        "cache_entry_width": 512,
        "rope_head_dim": 64,
        "retention_mode": "full",
        "window_size": 0,
        "compression_ratio": 0,
        "window_allocation_policy": "none",
        "cache_tp_policy": "replicated",
        "inference_source": "csv_reverse_inference",
        "target_parameter_count": 153_095_232,
        "index_n_heads": 0,
        "index_head_dim": 0,
        "index_topk": 0,
    }
    kwargs.update(overrides)
    return kwargs


def _build_step4_mfa_attention_config(**overrides):
    """Construct the reviewed shared MFA config without hiding a missing class."""
    config_class = getattr(common, "Step4MFAAttentionConfig", None)
    assert config_class is not None, "common.Step4MFAAttentionConfig must exist"
    return config_class(**_step4_mfa_attention_kwargs(**overrides))


def _step4_mfa_attention_sections() -> dict:
    """Return the exact reviewed V1 Full and SWA JSON sections."""
    full = _step4_mfa_attention_kwargs()
    full.pop("hidden_size")
    swa = _step4_mfa_attention_kwargs(
        num_query_heads=96,
        output_groups=12,
        cache_entry_width=128,
        retention_mode="swa",
        window_size=512,
        window_allocation_policy="sequence_capped",
        target_parameter_count=213_911_648,
    )
    swa.pop("hidden_size")
    return {"full_attention": full, "nonfull_attention": swa}


def _as_step4_mfa_pro_schema(raw_config: dict) -> dict:
    """Replace the inherited attention sections with the reviewed shared MFA schema."""
    raw_config = deepcopy(raw_config)
    raw_config.pop("block_types", None)
    raw_config.update(_step4_mfa_attention_sections())
    return raw_config


def _walk_operations(operation_list):
    """Yield top-level operations and nested overlap-group operations."""
    for operation in operation_list:
        yield operation
        if isinstance(operation, ops.OverlapOp):
            yield from _walk_operations(operation._group_a)
            yield from _walk_operations(operation._group_b)


def _operations_by_name(operation_list) -> dict[str, object]:
    """Index a graph recursively and reject duplicate operation names."""
    indexed = {}
    for operation in _walk_operations(operation_list):
        assert operation._name not in indexed, f"duplicate operation name: {operation._name}"
        indexed[operation._name] = operation
    return indexed


def _expected_pro_attention_names(phase: str) -> tuple[str, ...]:
    """Return the complete ordered per-layer attention contract for one phase."""
    names = []
    for layer_id, attention_type, _ in EXPECTED_LAYER_RECORDS:
        suffixes = FULL_ATTENTION_SUFFIXES if attention_type == "full" else NONFULL_ATTENTION_SUFFIXES
        names.extend(f"{phase}_layer_{layer_id:03d}_{attention_type}_{suffix}" for suffix in suffixes)
    return tuple(names)


def _build_cached_step4_pro_model(*, tp_size: int = 1, moe_tp_size: int = 1, moe_ep_size: int = 1, nextn: int = 0):
    """Build the package-local Step4-Pro-V1 through the public registry."""
    model_config = config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        nextn=nextn,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )
    model = models.get_model(MODEL_ID, model_config, backend_name="vllm")
    assert isinstance(model, Step4Model)
    return model


def _build_cached_step4_model() -> Step4Model:
    """Build the original package-local Step4 graph for legacy regression."""
    model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=0,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )
    model = models.get_model("stepfun-ai/Step4", model_config, backend_name="vllm")
    assert isinstance(model, Step4Model)
    return model


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


def test_step4_mfa_attention_config_is_frozen_explicit_and_default_free():
    """The shared schema must expose only explicit required fields and be immutable."""
    config_class = getattr(common, "Step4MFAAttentionConfig", None)
    assert config_class is not None, "common.Step4MFAAttentionConfig must exist"

    dataclass_fields = fields(config_class)
    assert tuple(field.name for field in dataclass_fields) == STEP4_MFA_FIELD_NAMES
    for dataclass_field in dataclass_fields:
        assert dataclass_field.default is MISSING
        assert dataclass_field.default_factory is MISSING

    attention_config = _build_step4_mfa_attention_config()
    with pytest.raises(FrozenInstanceError):
        attention_config.hidden_size = 1


def test_step4_mfa_v1_cached_schema_records_all_three_widths_and_provenance():
    """The package-local V1 JSON must declare the complete reviewed MFA contract."""
    raw = _step4_pro_raw_config()
    expected_sections = _step4_mfa_attention_sections()

    assert raw["full_attention"] == expected_sections["full_attention"]
    assert raw["nonfull_attention"] == expected_sections["nonfull_attention"]

    extra = utils._parse_hf_config_json(raw)["extra_params"]
    assert isinstance(extra.full_attention, common.Step4MFAAttentionConfig)
    assert isinstance(extra.nonfull_attention, common.Step4MFAAttentionConfig)
    assert extra.full_attention.retention_mode == "full"
    assert extra.full_attention.window_allocation_policy == "none"
    assert extra.nonfull_attention.retention_mode == "swa"
    assert extra.nonfull_attention.window_allocation_policy == "sequence_capped"
    assert (
        extra.nonfull_attention.projection_head_dim,
        extra.nonfull_attention.cache_projection_width,
        extra.nonfull_attention.cache_entry_width,
    ) == (640, 512, 128)
    assert (
        len(
            {
                extra.nonfull_attention.projection_head_dim,
                extra.nonfull_attention.cache_projection_width,
                extra.nonfull_attention.cache_entry_width,
            }
        )
        == 3
    )


def test_step4_mfa_parser_builds_the_shared_class_for_both_retention_modes():
    """The real parser must use one shared class without mode-specific fallback schemas."""
    raw = _as_step4_mfa_pro_schema(_step4_pro_raw_config())

    extra = utils._parse_hf_config_json(raw)["extra_params"]

    assert type(extra.full_attention) is common.Step4MFAAttentionConfig
    assert type(extra.nonfull_attention) is common.Step4MFAAttentionConfig
    assert extra.full_attention == _build_step4_mfa_attention_config()
    assert extra.nonfull_attention == _build_step4_mfa_attention_config(
        num_query_heads=96,
        output_groups=12,
        cache_entry_width=128,
        retention_mode="swa",
        window_size=512,
        window_allocation_policy="sequence_capped",
        target_parameter_count=213_911_648,
    )


@pytest.mark.parametrize("section", ["full_attention", "nonfull_attention"])
@pytest.mark.parametrize("field", STEP4_MFA_SECTION_FIELD_NAMES)
def test_step4_mfa_parser_requires_every_explicit_attention_field(section, field):
    """Missing MFA fields must fail instead of borrowing dimensions from another field."""
    raw = _as_step4_mfa_pro_schema(_step4_pro_raw_config())
    raw[section].pop(field)

    with pytest.raises(ValueError, match=rf"Step4-Pro {section} missing required field '{field}'"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("section", ["full_attention", "nonfull_attention"])
def test_step4_mfa_parser_rejects_unexpected_attention_fields(section):
    """Legacy or misspelled fields must not be silently ignored by the shared parser."""
    raw = _as_step4_mfa_pro_schema(_step4_pro_raw_config())
    raw[section]["legacy_head_dim"] = 512

    with pytest.raises(ValueError, match=rf"Step4-Pro {section} has unsupported field 'legacy_head_dim'"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("model", "retention_mode", "hidden_size", "num_query_heads", "output_groups", "target"),
    [
        pytest.param("v1", "full", 6144, 64, 8, 153_095_232, id="v1-full"),
        pytest.param("v1", "swa", 6144, 96, 12, 213_911_648, id="v1-swa"),
        pytest.param("v3", "full", 12288, 96, 12, 314_575_968, id="v3-full"),
        pytest.param("v3", "swa", 12288, 128, 16, 394_266_752, id="v3-swa"),
        pytest.param("v4", "full", 9216, 96, 12, 267_390_048, id="v4-full"),
        pytest.param("v4", "swa", 9216, 128, 16, 337_643_648, id="v4-swa"),
    ],
)
def test_step4_mfa_parameter_formula_closes_all_six_targets_exactly(
    model,
    retention_mode,
    hidden_size,
    num_query_heads,
    output_groups,
    target,
):
    """One shared integer formula must reproduce all CSV parameter targets with zero residual."""
    del model
    is_swa = retention_mode == "swa"
    attention_config = _build_step4_mfa_attention_config(
        hidden_size=hidden_size,
        num_query_heads=num_query_heads,
        output_groups=output_groups,
        cache_entry_width=128 if hidden_size == 6144 and is_swa else 512,
        retention_mode=retention_mode,
        window_size=512 if is_swa else 0,
        compression_ratio=0 if hidden_size == 6144 else 4,
        window_allocation_policy=(
            "sequence_capped" if hidden_size == 6144 and is_swa else "fixed_capacity" if is_swa else "none"
        ),
        target_parameter_count=target,
    )

    actual = attention_config.compute_parameter_count()
    absolute_error = abs(actual - target)
    relative_error = absolute_error / target

    assert actual == target
    assert absolute_error == 0
    assert relative_error == 0


@pytest.mark.parametrize(
    ("seq_len", "expected_total_bytes"),
    [
        pytest.param(0, 0, id="empty"),
        pytest.param(1, 17_920, id="one-token"),
        pytest.param(511, 9_157_120, id="below-window"),
        pytest.param(512, 9_175_040, id="at-window"),
        pytest.param(513, 9_185_280, id="above-window"),
        pytest.param(1_048_576, 10_741_350_400, id="one-million"),
    ],
)
def test_step4_mfa_v1_runtime_kv_closes_sequence_capped_audit(seq_len, expected_total_bytes):
    """V1 must combine uncompressed Full history with a 512-token, 128-wide SWA cache."""
    full = _build_step4_mfa_attention_config()
    swa = _build_step4_mfa_attention_config(
        num_query_heads=96,
        output_groups=12,
        cache_entry_width=128,
        retention_mode="swa",
        window_size=512,
        window_allocation_policy="sequence_capped",
        target_parameter_count=213_911_648,
    )

    full_bytes = full.compute_kv_cache_bytes(seq_len, tp_size=1, bytes_per_element=1)
    swa_bytes = swa.compute_kv_cache_bytes(seq_len, tp_size=1, bytes_per_element=1)
    actual_total_bytes = 20 * full_bytes + 60 * swa_bytes

    assert actual_total_bytes == expected_total_bytes


@pytest.mark.parametrize(
    ("seq_len", "expected_total_bytes"),
    [
        pytest.param(0, 15_728_640, id="empty-fixed-capacity"),
        pytest.param(1, 15_728_640, id="below-first-compressed-entry"),
        pytest.param(3, 15_728_640, id="last-before-first-compressed-entry"),
        pytest.param(4, 15_769_600, id="first-compressed-entry"),
        pytest.param(5, 15_769_600, id="floor-retention"),
        pytest.param(100, 16_752_640, id="locked-short-sequence"),
        pytest.param(1_048_576, 10_753_146_880, id="one-million"),
    ],
)
def test_step4_mfa_v3_v4_runtime_kv_closes_fixed_capacity_audit(seq_len, expected_total_bytes):
    """V3/V4 must reserve the SWA window and add floor(T/4) compressed history."""
    full = _build_step4_mfa_attention_config(
        hidden_size=12288,
        num_query_heads=96,
        output_groups=12,
        compression_ratio=4,
        target_parameter_count=314_575_968,
    )
    swa = _build_step4_mfa_attention_config(
        hidden_size=12288,
        num_query_heads=128,
        output_groups=16,
        retention_mode="swa",
        window_size=512,
        compression_ratio=4,
        window_allocation_policy="fixed_capacity",
        target_parameter_count=394_266_752,
    )

    full_bytes = full.compute_kv_cache_bytes(seq_len, tp_size=1, bytes_per_element=1)
    swa_bytes = swa.compute_kv_cache_bytes(seq_len, tp_size=1, bytes_per_element=1)
    actual_total_bytes = 20 * full_bytes + 60 * swa_bytes

    assert actual_total_bytes == expected_total_bytes


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "attention_config",
    [
        pytest.param({}, id="v1-full"),
        pytest.param(
            {
                "num_query_heads": 96,
                "output_groups": 12,
                "cache_entry_width": 128,
                "retention_mode": "swa",
                "window_size": 512,
                "window_allocation_policy": "sequence_capped",
                "target_parameter_count": 213_911_648,
            },
            id="v1-swa",
        ),
        pytest.param(
            {
                "hidden_size": 12288,
                "num_query_heads": 128,
                "output_groups": 16,
                "retention_mode": "swa",
                "window_size": 512,
                "compression_ratio": 4,
                "window_allocation_policy": "fixed_capacity",
                "target_parameter_count": 394_266_752,
            },
            id="v3-swa",
        ),
    ],
)
def test_step4_mfa_runtime_kv_is_replicated_across_attention_tp(attention_config, tp_size):
    """Shared MFA cache bytes must be identical on every attention-TP rank."""
    config_instance = _build_step4_mfa_attention_config(**attention_config)
    expected = config_instance.compute_kv_cache_bytes(513, tp_size=1, bytes_per_element=1)

    actual = config_instance.compute_kv_cache_bytes(513, tp_size=tp_size, bytes_per_element=1)

    assert actual == expected


def test_step4_mfa_projection_and_cache_widths_are_orthogonal():
    """Projection, cache-projection, and persisted-entry widths must not leak into each other."""
    baseline = _build_step4_mfa_attention_config()
    narrower_entry = replace(baseline, cache_entry_width=256)
    narrower_cache_projection = replace(baseline, cache_projection_width=256)
    narrower_projection = replace(baseline, projection_head_dim=512)

    baseline_params = baseline.compute_parameter_count()
    baseline_kv = baseline.compute_kv_cache_bytes(100, tp_size=1, bytes_per_element=1)

    assert narrower_entry.compute_parameter_count() == baseline_params
    assert narrower_entry.compute_kv_cache_bytes(100, tp_size=1, bytes_per_element=1) == baseline_kv // 2
    assert narrower_cache_projection.compute_parameter_count() != baseline_params
    assert narrower_cache_projection.compute_kv_cache_bytes(100, tp_size=1, bytes_per_element=1) == baseline_kv
    assert narrower_projection.compute_parameter_count() != baseline_params
    assert narrower_projection.compute_kv_cache_bytes(100, tp_size=1, bytes_per_element=1) == baseline_kv


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        pytest.param({"attention_type": "mla"}, "attention_type", id="attention-type"),
        pytest.param({"retention_mode": "global"}, "retention_mode", id="retention-mode"),
        pytest.param({"window_allocation_policy": "adaptive"}, "window_allocation_policy", id="policy"),
        pytest.param({"window_size": 512}, "window_size", id="full-window"),
        pytest.param(
            {"window_allocation_policy": "fixed_capacity"},
            "window_allocation_policy",
            id="full-allocation-policy",
        ),
        pytest.param(
            {
                "retention_mode": "swa",
                "window_size": 0,
                "window_allocation_policy": "sequence_capped",
            },
            "window_size",
            id="swa-zero-window",
        ),
        pytest.param(
            {"retention_mode": "swa", "window_size": 512, "window_allocation_policy": "none"},
            "window_allocation_policy",
            id="swa-none-policy",
        ),
        pytest.param({"compression_ratio": -1}, "compression_ratio", id="negative-compression"),
        pytest.param({"compression_ratio": True}, "compression_ratio", id="boolean-compression"),
        pytest.param({"compression_ratio": 4.0}, "compression_ratio", id="float-compression"),
        pytest.param({"window_size": False}, "window_size", id="boolean-window"),
        pytest.param({"num_query_heads": 65}, "num_query_heads", id="head-group-divisibility"),
        pytest.param({"output_groups": 4}, "output_groups", id="output-groups"),
        pytest.param({"o_lora_rank": 512}, "lora_rank", id="lora-rank-mismatch"),
        pytest.param({"rope_head_dim": 641}, "rope_head_dim", id="rope-too-wide"),
        pytest.param({"cache_tp_policy": "sharded"}, "cache_tp_policy", id="cache-tp-policy"),
        pytest.param({"inference_source": "checkpoint"}, "inference_source", id="inference-source"),
        pytest.param({"index_n_heads": 1}, "index_n_heads", id="index-heads"),
        pytest.param({"index_head_dim": 1}, "index_head_dim", id="index-dimension"),
        pytest.param({"index_topk": 1}, "index_topk", id="index-topk"),
        pytest.param({"index_n_heads": False}, "index_n_heads", id="boolean-index-heads"),
    ],
)
def test_step4_mfa_attention_config_rejects_inconsistent_states(overrides, expected_error):
    """Every unsupported MFA state must fail during direct construction."""
    with pytest.raises(ValueError, match=expected_error):
        _build_step4_mfa_attention_config(**overrides)


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_query_heads",
        "output_groups",
        "q_lora_rank",
        "o_lora_rank",
        "projection_head_dim",
        "cache_projection_width",
        "cache_entry_width",
        "rope_head_dim",
        "target_parameter_count",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1, True, 1.0, float("nan"), float("inf")])
def test_step4_mfa_attention_config_requires_positive_integer_dimensions(field, invalid_value):
    """All integer dimensions must reject booleans, non-positive values, floats, NaN, and infinity."""
    with pytest.raises(ValueError, match=field):
        _build_step4_mfa_attention_config(**{field: invalid_value})


@pytest.mark.parametrize(
    ("seq_len", "tp_size", "bytes_per_element", "expected_error"),
    [
        pytest.param(-1, 1, 1, "seq_len", id="negative-sequence"),
        pytest.param(True, 1, 1, "seq_len", id="boolean-sequence"),
        pytest.param(1.0, 1, 1, "seq_len", id="float-sequence"),
        pytest.param(1, 0, 1, "tp_size", id="zero-tp"),
        pytest.param(1, True, 1, "tp_size", id="boolean-tp"),
        pytest.param(1, 1.0, 1, "tp_size", id="float-tp"),
        pytest.param(1, 1, 0, "bytes_per_element", id="zero-element-width"),
        pytest.param(1, 1, False, "bytes_per_element", id="boolean-element-width"),
        pytest.param(1, 1, float("nan"), "bytes_per_element", id="nan-element-width"),
        pytest.param(1, 1, float("inf"), "bytes_per_element", id="infinite-element-width"),
    ],
)
def test_step4_mfa_runtime_kv_rejects_invalid_inputs(seq_len, tp_size, bytes_per_element, expected_error):
    """KV accounting must fail fast instead of propagating malformed dimensions."""
    attention_config = _build_step4_mfa_attention_config()

    with pytest.raises(ValueError, match=expected_error):
        attention_config.compute_kv_cache_bytes(
            seq_len,
            tp_size=tp_size,
            bytes_per_element=bytes_per_element,
        )


def test_step4_mfa_migration_removes_obsolete_step4_pro_attention_classes():
    """No standard-MHA or HCA class may remain as an alternate Step4-Pro path."""
    assert not hasattr(common, "FullAttentionConfig")
    assert not hasattr(common, "NonFullAttentionConfig")


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

    assert isinstance(extra, common.Step4ProConfig)
    assert tuple((layer.layer_id, layer.attention_type, layer.ffn_type) for layer in extra.layers) == (
        EXPECTED_LAYER_RECORDS
    )
    assert sum(layer.attention_type == "full" for layer in extra.layers) == 20
    assert sum(layer.attention_type == "nonfull" for layer in extra.layers) == 60
    assert sum(layer.ffn_type == "dense" for layer in extra.layers) == 4
    assert sum(layer.ffn_type == "moe" for layer in extra.layers) == 76
    assert len(extra.layers) == 80
    assert extra.full_attention.num_query_heads == 64
    assert extra.nonfull_attention.num_query_heads == 96
    assert extra.dense_inter_size == 16384
    assert extra.shared_expert_inter_size == 2048

    assert model_info["layers"] == 80
    assert model_info["hidden_size"] == 6144
    assert model_info["inter_size"] == 16384
    assert model_info["num_experts"] == 512
    assert model_info["topk"] == 8
    assert model_info["moe_inter_size"] == 2048
    assert model_info["vocab"] == 128896


def test_step4_pro_v1_attention_configs_close_authoritative_targets():
    """Trainable matrix counts must close targets while resident state stays separate."""
    extra = utils._parse_hf_config_json(_step4_pro_raw_config())["extra_params"]

    full = extra.full_attention
    nonfull = extra.nonfull_attention

    assert isinstance(full, common.FullAttentionConfig)
    assert full.compute_parameter_count() == 150_994_944
    assert full.target_parameter_count == 153_095_232
    full_relative_error = (
        abs(full.compute_parameter_count() - full.target_parameter_count) / full.target_parameter_count
    )
    assert full_relative_error == pytest.approx(0.013718833516644071)

    assert isinstance(nonfull, common.NonFullAttentionConfig)
    assert nonfull.compute_parameter_count() == 217_055_232
    assert nonfull.target_parameter_count == 213_911_648
    assert abs(nonfull.compute_parameter_count() - nonfull.target_parameter_count) / nonfull.target_parameter_count == (
        pytest.approx(0.014695712128775709)
    )
    assert nonfull.resident_state_elements == 65_632

    for attention_config in (full, nonfull):
        assert attention_config.unknown_extra_projection_params == 0
        assert attention_config.unknown_router_params == 0
        assert attention_config.unknown_compression_params == 0


def test_step4_parser_rejects_mixed_legacy_and_pro_attention_schemas():
    """A config must never make the parser choose silently between two schemas."""
    raw = _step4_pro_raw_config()
    raw["block_types"] = ["dense_swa"] * raw["num_hidden_layers"]
    raw.update(_candidate_pro_attention_sections())

    with pytest.raises(ValueError, match="cannot define both 'block_types' and the Step4-Pro attention schema"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("missing_section", ["layers", "full_attention", "nonfull_attention"])
def test_step4_pro_parser_requires_every_attention_schema_section(missing_section):
    """Partial Pro schemas must fail rather than falling back to legacy Step4."""
    raw = _as_candidate_pro_schema(_step4_pro_raw_config())
    raw.pop(missing_section)

    with pytest.raises(ValueError, match="requires 'layers', 'full_attention', and 'nonfull_attention' together"):
        utils._parse_hf_config_json(raw)


def test_step4_parser_requires_exactly_one_attention_schema():
    """A Step4 config without either complete schema must fail explicitly."""
    raw = _step4_pro_raw_config()
    for field in ("block_types", "layers", "full_attention", "nonfull_attention"):
        raw.pop(field, None)

    with pytest.raises(ValueError, match="must define exactly one attention schema"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        pytest.param(
            lambda raw: raw.__setitem__("layers", {}),
            "Step4-Pro layers must be a list",
            id="layers-not-list",
        ),
        pytest.param(
            lambda raw: raw.__setitem__("layers", raw["layers"][:-1]),
            "Step4-Pro layers length 79 != num_hidden_layers 80",
            id="layers-wrong-length",
        ),
        pytest.param(
            lambda raw: raw["layers"].__setitem__(0, None),
            "Step4-Pro layer 0 must be a mapping",
            id="layer-not-mapping",
        ),
        pytest.param(
            lambda raw: raw["layers"].__setitem__(0, {**raw["layers"][0], "layer_id": True}),
            "Step4-Pro layer 0 layer_id must be a non-negative integer",
            id="boolean-layer-id",
        ),
    ],
)
def test_step4_pro_parser_rejects_malformed_layer_records(mutator, expected_error):
    """The ordered layer collection must have one valid record per trunk layer."""
    raw = _as_candidate_pro_schema(_step4_pro_raw_config())
    mutator(raw)

    with pytest.raises(ValueError, match=expected_error):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        pytest.param(
            lambda layers: layers.__setitem__(1, {**layers[1], "layer_id": 0}),
            "layer 1 must have contiguous layer_id 1, got 0",
            id="duplicate-layer-id",
        ),
        pytest.param(
            lambda layers: layers.__setitem__(4, {**layers[4], "attention_type": "mla"}),
            "layer 4 has unsupported attention_type 'mla'",
            id="unsupported-attention-type",
        ),
        pytest.param(
            lambda layers: layers.__setitem__(4, {**layers[4], "ffn_type": "hybrid"}),
            "layer 4 has unsupported ffn_type 'hybrid'",
            id="unsupported-ffn-type",
        ),
    ],
)
def test_step4_pro_parser_rejects_malformed_layer_identity(mutator, expected_error):
    """Layer identity, attention type, and FFN type form one fail-fast contract."""
    raw = _as_candidate_pro_schema(_step4_pro_raw_config())
    mutator(raw["layers"])

    with pytest.raises(ValueError, match=expected_error):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("section", "field", "invalid_value", "expected_error"),
    [
        pytest.param(
            "full_attention",
            "num_query_heads",
            True,
            "full_attention.num_query_heads must be a positive integer",
            id="full-bool-heads",
        ),
        pytest.param(
            "full_attention",
            "q_projection",
            "lora",
            "full_attention.q_projection supports only 'linear'",
            id="unsupported-full-projection",
        ),
        pytest.param(
            "nonfull_attention",
            "compression_ratio",
            4,
            "nonfull_attention.compression_ratio supports only 0 or 128",
            id="unsupported-csa-ratio",
        ),
        pytest.param(
            "nonfull_attention",
            "o_groups",
            0,
            "nonfull_attention.o_groups must be a positive integer",
            id="zero-output-groups",
        ),
    ],
)
def test_step4_pro_parser_rejects_unsupported_attention_geometry(section, field, invalid_value, expected_error):
    """Unsupported geometry must not be coerced into a nearby attention recipe."""
    raw = _as_candidate_pro_schema(_step4_pro_raw_config())
    raw[section][field] = invalid_value

    with pytest.raises(ValueError, match=expected_error):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        pytest.param(
            lambda raw: raw.__setitem__("full_attention", []),
            "Step4-Pro full_attention must be a mapping",
            id="full-not-mapping",
        ),
        pytest.param(
            lambda raw: raw["full_attention"].pop("q_head_dim"),
            "Step4-Pro full_attention missing required field 'q_head_dim'",
            id="full-missing-field",
        ),
        pytest.param(
            lambda raw: raw["full_attention"].__setitem__("unknown_router_params", -1),
            "full_attention.unknown_router_params must be a non-negative integer",
            id="full-negative-unknown",
        ),
        pytest.param(
            lambda raw: raw["full_attention"].__setitem__("latent_rank", 512),
            "full_attention.latent_rank must be null",
            id="full-latent-rank",
        ),
        pytest.param(
            lambda raw: raw["full_attention"].__setitem__("k_head_dim", 64),
            "full_attention requires equal q_head_dim, k_head_dim, and v_head_dim",
            id="full-unequal-head-dimensions",
        ),
        pytest.param(
            lambda raw: raw["full_attention"].__setitem__("rope_dimension", 128),
            "full_attention.rope_dimension 128 exceeds q_head_dim 96",
            id="full-rope-too-wide",
        ),
        pytest.param(
            lambda raw: raw["full_attention"].__setitem__("num_kv_heads", 24),
            "full_attention.num_query_heads 64 must be divisible by num_kv_heads 24",
            id="full-incompatible-head-ratio",
        ),
    ],
)
def test_step4_pro_parser_rejects_invalid_full_attention_contract(mutator, expected_error):
    """Full attention must use the exact standard linear projection contract."""
    raw = _as_candidate_pro_schema(_step4_pro_raw_config())
    mutator(raw)

    with pytest.raises(ValueError, match=expected_error):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        pytest.param(
            lambda raw: raw.__setitem__("nonfull_attention", []),
            "Step4-Pro nonfull_attention must be a mapping",
            id="nonfull-not-mapping",
        ),
        pytest.param(
            lambda raw: raw["nonfull_attention"].pop("head_dim"),
            "Step4-Pro nonfull_attention missing required field 'head_dim'",
            id="nonfull-missing-field",
        ),
        pytest.param(
            lambda raw: raw["nonfull_attention"].__setitem__("mechanism", "csa"),
            "nonfull_attention.mechanism supports only 'swa' or 'hca'",
            id="nonfull-unsupported-mechanism",
        ),
        pytest.param(
            lambda raw: raw["nonfull_attention"].update(mechanism="swa", compression_ratio=128),
            "nonfull_attention mechanism 'swa' requires compression_ratio 0",
            id="swa-with-compression",
        ),
        pytest.param(
            lambda raw: raw["nonfull_attention"].update(mechanism="hca", compression_ratio=0),
            "nonfull_attention mechanism 'hca' requires compression_ratio 128",
            id="hca-without-compression",
        ),
        pytest.param(
            lambda raw: raw["nonfull_attention"].__setitem__("index_n_heads", 1),
            "nonfull_attention indexer fields must all be zero",
            id="nonfull-indexer-enabled",
        ),
        pytest.param(
            lambda raw: raw["nonfull_attention"].__setitem__("rope_dimension", 1024),
            "nonfull_attention.rope_dimension 1024 exceeds head_dim 512",
            id="nonfull-rope-too-wide",
        ),
        pytest.param(
            lambda raw: raw["nonfull_attention"].__setitem__("unknown_compression_params", -1),
            "nonfull_attention.unknown_compression_params must be a non-negative integer",
            id="nonfull-negative-unknown",
        ),
    ],
)
def test_step4_pro_parser_rejects_invalid_nonfull_attention_contract(mutator, expected_error):
    """Non-full attention must select one supported SWA or HCA recipe exactly."""
    raw = _as_candidate_pro_schema(_step4_pro_raw_config())
    mutator(raw)

    with pytest.raises(ValueError, match=expected_error):
        utils._parse_hf_config_json(raw)


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
    raw = deepcopy(utils._load_pre_downloaded_hf_config("stepfun-ai/Step4"))
    raw.update(
        {
            "hidden_size": 6144,
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


def test_step4_pro_v1_builds_all_per_layer_attention_operations():
    """All 80 Pro attention layers must be explicit, unique, and free of MLA approximations."""
    model = _build_cached_step4_pro_model()
    context = _operations_by_name(model.context_ops)
    generation = _operations_by_name(model.generation_ops)
    expected_context = _expected_pro_attention_names("context")
    expected_generation = _expected_pro_attention_names("generation")

    assert model._layer_counts() == (4, 20, 60)
    assert len(expected_context) == len(expected_generation) == 20 * 7 + 60 * 3 == 320
    assert len(set(expected_context)) == len(expected_context)
    assert len(set(expected_generation)) == len(expected_generation)
    assert all(name in context for name in expected_context)
    assert all(name in generation for name in expected_generation)
    assert all(context[name]._scale_factor == 1 for name in expected_context)
    assert all(generation[name]._scale_factor == 1 for name in expected_generation)
    assert not any("mla_approx" in name for name in context)
    assert not any("mla_approx" in name for name in generation)

    for name in (
        "context_dense_ffn_norm",
        "context_dense_gate_up_gemm",
        "context_dense_swiglu",
        "context_dense_down_gemm",
        "context_dense_ffn_ar",
    ):
        assert context[name]._scale_factor == 4
    for name in (
        "context_moe_ffn_norm",
        "context_moe_router_gemm",
        "context_moe_pre_dispatch",
        "context_moe",
        "context_moe_post_dispatch",
        "context_shared_gate_up_gemm",
        "context_shared_swiglu",
        "context_shared_down_gemm",
        "context_shared_ffn_ar",
        "context_moe_shared_merge",
    ):
        assert context[name]._scale_factor == 76


def test_step4_pro_v1_builds_explicit_factorized_attention_runtime_specs():
    """Full and SWA builders must derive their complete runtime specs from the shared MFA config."""
    model = _build_cached_step4_pro_model(tp_size=4, moe_ep_size=4)
    operations_by_phase = {
        "context": _operations_by_name(model.context_ops),
        "generation": _operations_by_name(model.generation_ops),
    }
    expected_by_attention_type = {
        "full": {
            "retention_mode": "full",
            "compressed_history_selection": "none",
            "projection_head_dim": 640,
            "cache_projection_width": 512,
            "cache_entry_width": 512,
            "cache_projection_matrix_count": 4,
            "cache_auxiliary_fp32_elements": 3_072,
            "cache_auxiliary_ops_per_token": 0,
            "window_size": 0,
            "compression_ratio": 0,
            "index_n_heads": 0,
            "index_head_dim": 0,
            "index_topk": 0,
        },
        "nonfull": {
            "retention_mode": "swa",
            "compressed_history_selection": "none",
            "projection_head_dim": 640,
            "cache_projection_width": 512,
            "cache_entry_width": 128,
            "cache_projection_matrix_count": 2,
            "cache_auxiliary_fp32_elements": 2_048,
            "cache_auxiliary_ops_per_token": 0,
            "window_size": 512,
            "compression_ratio": 0,
            "index_n_heads": 0,
            "index_head_dim": 0,
            "index_topk": 0,
        },
    }

    for phase, operation_type in (
        ("context", ops.ContextDeepSeekV4AttentionModule),
        ("generation", ops.GenerationDeepSeekV4AttentionModule),
    ):
        indexed = operations_by_phase[phase]
        for layer_id, attention_type in ((4, "full"), (0, "nonfull")):
            prefix = f"{phase}_layer_{layer_id:03d}_{attention_type}"
            attention_operations = [
                operation
                for name, operation in indexed.items()
                if name.startswith(prefix) and isinstance(operation, operation_type)
            ]

            assert len(attention_operations) == 1
            runtime_spec = attention_operations[0]._runtime_spec
            assert {
                field_name: getattr(runtime_spec, field_name)
                for field_name in expected_by_attention_type[attention_type]
            } == expected_by_attention_type[attention_type]


def test_step4_pro_v1_full_attention_geometry_is_independent():
    """Full layers must use sharded standard-MHA Q/K/V/O geometry rather than Step4 MLA ranks."""
    model = _build_cached_step4_pro_model(tp_size=4, moe_ep_size=4)
    context = _operations_by_name(model.context_ops)
    generation = _operations_by_name(model.generation_ops)

    for layer_id in (4, 23):
        for phase, indexed, attention_type in (
            ("context", context, ops.ContextAttention),
            ("generation", generation, ops.GenerationAttention),
        ):
            prefix = f"{phase}_layer_{layer_id:03d}_full"
            norm = indexed[f"{prefix}_attn_norm"]
            q_proj = indexed[f"{prefix}_q_proj_gemm"]
            k_proj = indexed[f"{prefix}_k_proj_gemm"]
            v_proj = indexed[f"{prefix}_v_proj_gemm"]
            attention = indexed[f"{prefix}_attention"]
            o_proj = indexed[f"{prefix}_o_proj_gemm"]
            reduction = indexed[f"{prefix}_attention_ar"]

            assert isinstance(norm, ops.ElementWise)
            assert (norm._dim_in, norm._dim_out) == (12288, 12288)
            assert isinstance(q_proj, ops.GEMM)
            assert isinstance(k_proj, ops.GEMM)
            assert isinstance(v_proj, ops.GEMM)
            assert (q_proj._n, q_proj._k) == (1536, 6144)
            assert (k_proj._n, k_proj._k) == (1536, 6144)
            assert (v_proj._n, v_proj._k) == (1536, 6144)
            assert isinstance(attention, attention_type)
            assert (attention._n, attention._n_kv, attention._head_size, attention._window_size) == (
                16,
                16,
                96,
                0,
            )
            assert isinstance(o_proj, ops.GEMM)
            assert (o_proj._n, o_proj._k, o_proj._low_precision_input) == (6144, 1536, True)
            assert isinstance(reduction, ops.CustomAllReduce)
            assert (reduction._h, reduction._tp_size) == (6144, 4)


def test_step4_pro_v1_hca_geometry_is_independent():
    """Non-full layers must use the reviewed HCA module and an explicit output reduction."""
    model = _build_cached_step4_pro_model(tp_size=4, moe_ep_size=4)
    context = _operations_by_name(model.context_ops)
    generation = _operations_by_name(model.generation_ops)

    for layer_id in (0, 24, 79):
        for phase, indexed, attention_type in (
            ("context", context, ops.ContextDeepSeekV4AttentionModule),
            ("generation", generation, ops.GenerationDeepSeekV4AttentionModule),
        ):
            prefix = f"{phase}_layer_{layer_id:03d}_nonfull"
            norm = indexed[f"{prefix}_attn_norm"]
            attention = indexed[f"{prefix}_hca_attention"]
            reduction = indexed[f"{prefix}_attention_ar"]

            assert isinstance(norm, ops.ElementWise)
            assert (norm._dim_in, norm._dim_out) == (12288, 12288)
            assert isinstance(attention, attention_type)
            assert (
                attention._num_heads,
                attention._native_heads,
                attention._tp_size,
                attention._hidden_size,
                attention._q_lora_rank,
                attention._o_lora_rank,
                attention._head_dim,
                attention._rope_head_dim,
                attention._index_n_heads,
                attention._index_head_dim,
                attention._index_topk,
                attention._window_size,
                attention._compress_ratio,
                attention._o_groups,
            ) == (24, 96, 4, 6144, 1024, 1024, 512, 64, 0, 0, 0, 512, 128, 4)
            assert isinstance(reduction, ops.CustomAllReduce)
            assert (reduction._h, reduction._tp_size) == (6144, 4)


def test_step4_pro_v1_dynamic_mixed_step_keys_cover_attention_graph():
    """Per-instance semantic keys must enumerate every top-level Pro attention operation exactly once."""
    model = _build_cached_step4_pro_model()
    expected_context = _expected_pro_attention_names("context")
    expected_generation = _expected_pro_attention_names("generation")
    context_names = {operation._name for operation in model.context_ops}
    generation_names = {operation._name for operation in model.generation_ops}

    assert expected_context == model.MIXED_STEP_CONTEXT_ATTENTION_KEYS
    assert expected_generation == model.MIXED_STEP_GENERATION_ATTENTION_KEYS
    assert len(model.MIXED_STEP_CONTEXT_ATTENTION_KEYS) == 320
    assert len(model.MIXED_STEP_GENERATION_ATTENTION_KEYS) == 320
    assert set(expected_context) <= context_names
    assert set(expected_generation) <= generation_names


def test_step4_pro_v1_reports_parameter_validation_and_kv_conflict(caplog):
    """Initialization must emit deterministic parameter evidence and the unresolved KV contradiction."""
    with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.models.step4"):
        _build_cached_step4_pro_model()

    report = "\n".join(record.getMessage() for record in caplog.records)
    assert "Step4-Pro-V1 attention parameter validation" in report
    assert "full target=153095232" in report
    assert "estimate=150994944" in report
    assert "absolute_error=2100288" in report
    assert "relative_error=1.3718833517%" in report
    assert "nonfull target=213911648" in report
    assert "estimate=217055232" in report
    assert "absolute_error=3143584" in report
    assert "relative_error=1.4695712129%" in report
    assert report.count("status=PASS") == 2
    assert "resident_state_elements=65632" in report
    assert "257.99688192 GB" in report
    assert "10.7 GB" in report
    assert "24.1118581234x" in report
    assert "unresolved" in report


@pytest.mark.parametrize(
    ("section", "label"),
    [
        pytest.param("full_attention", "full", id="full"),
        pytest.param("nonfull_attention", "nonfull", id="nonfull"),
    ],
)
def test_step4_pro_v1_rejects_parameter_mismatch_before_operation_construction(monkeypatch, section, label):
    """A parameter error above 5% must fail before any graph operation is constructed."""
    raw = _step4_pro_raw_config()
    raw[section]["target_parameter_count"] = 1

    def fail_operation_construction(*_args, **_kwargs):
        raise AssertionError("operation construction must not begin after failed parameter validation")

    monkeypatch.setattr(ops, "GEMM", fail_operation_construction)

    with pytest.raises(ValueError, match=rf"Step4-Pro {label} attention parameter error .* exceeds 5%"):
        _build_synthetic_step4_model(monkeypatch, raw, tp_size=1)


def test_step4_pro_v1_graph_uses_exact_ffn_and_moe_geometry():
    """The existing FFN/MoE graph must continue to consume the audited Pro dimensions and quant modes."""
    model = _build_cached_step4_pro_model()
    context = _operations_by_name(model.context_ops)

    assert model.config.gemm_quant_mode == common.GEMMQuantMode.fp8
    assert model.config.moe_quant_mode == common.MoEQuantMode.fp8
    assert model.config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    assert model.config.fmha_quant_mode == common.FMHAQuantMode.bfloat16

    assert (context["context_dense_gate_up_gemm"]._n, context["context_dense_gate_up_gemm"]._k) == (32768, 6144)
    assert (context["context_dense_swiglu"]._dim_in, context["context_dense_swiglu"]._dim_out) == (32768, 16384)
    assert (context["context_dense_down_gemm"]._n, context["context_dense_down_gemm"]._k) == (6144, 16384)

    assert (context["context_moe_router_gemm"]._n, context["context_moe_router_gemm"]._k) == (512, 6144)
    routed = context["context_moe"]
    assert (routed._hidden_size, routed._inter_size, routed._topk, routed._num_experts) == (6144, 2048, 8, 512)
    assert (context["context_shared_gate_up_gemm"]._n, context["context_shared_gate_up_gemm"]._k) == (4096, 6144)
    assert (context["context_shared_swiglu"]._dim_in, context["context_shared_swiglu"]._dim_out) == (4096, 2048)
    assert (context["context_shared_down_gemm"]._n, context["context_shared_down_gemm"]._k) == (6144, 2048)
    assert (context["context_moe_shared_merge"]._dim_in, context["context_moe_shared_merge"]._dim_out) == (
        12288,
        6144,
    )
    assert (context["context_logits_gemm"]._n, context["context_logits_gemm"]._k) == (128896, 6144)


def test_step4_pro_v1_generation_overlaps_routed_and_shared_moe_paths():
    """Decode must overlap the routed and shared paths without hiding either branch."""
    model = _build_cached_step4_pro_model()
    generation = _operations_by_name(model.generation_ops)
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
    assert generation["generation_moe_pre_dispatch"]._reduce_results is False
    assert generation["generation_moe"]._is_context is False
    assert generation["generation_moe_post_dispatch"]._is_context is False


def test_step4_pro_v1_mtp_scale_is_applied_once_per_attention_layer():
    """Per-layer attention scales must be one in context and exactly one MTP factor in generation."""
    baseline = _build_cached_step4_pro_model(nextn=0)
    mtp = _build_cached_step4_pro_model(nextn=3)
    expected_scale = 1.0 / (1.0 + calc_expectation(3, [0.85, 0.3, 0.0, 0.0, 0.0])) * (80 + 3) / 80

    assert baseline.activation_hidden_size == 6144
    assert baseline._mtp_scale_factor == 1.0
    assert mtp._mtp_scale_factor == pytest.approx(expected_scale)

    baseline_context = _operations_by_name(baseline.context_ops)
    mtp_context = _operations_by_name(mtp.context_ops)
    baseline_generation = _operations_by_name(baseline.generation_ops)
    mtp_generation = _operations_by_name(mtp.generation_ops)
    for name in _expected_pro_attention_names("context"):
        assert baseline_context[name]._scale_factor == 1
        assert mtp_context[name]._scale_factor == 1
    for name in _expected_pro_attention_names("generation"):
        assert baseline_generation[name]._scale_factor == 1
        assert mtp_generation[name]._scale_factor == pytest.approx(expected_scale)


def test_original_step4_attention_graph_remains_legacy_mla():
    """The Pro conversion must not alter the original Step4 semantic names or aggregate MLA graph."""
    model = _build_cached_step4_model()
    context = _operations_by_name(model.context_ops)
    generation = _operations_by_name(model.generation_ops)

    assert Step4Model.MIXED_STEP_CONTEXT_ATTENTION_KEYS == LEGACY_CONTEXT_ATTENTION_NAMES
    assert Step4Model.MIXED_STEP_GENERATION_ATTENTION_KEYS == LEGACY_GENERATION_ATTENTION_NAMES
    assert model.MIXED_STEP_CONTEXT_ATTENTION_KEYS == LEGACY_CONTEXT_ATTENTION_NAMES
    assert model.MIXED_STEP_GENERATION_ATTENTION_KEYS == LEGACY_GENERATION_ATTENTION_NAMES
    assert len(model.MIXED_STEP_CONTEXT_ATTENTION_KEYS) == 14
    assert len(model.MIXED_STEP_GENERATION_ATTENTION_KEYS) == 16
    assert all(name in context for name in LEGACY_CONTEXT_ATTENTION_NAMES)
    assert all(name in generation for name in LEGACY_GENERATION_ATTENTION_NAMES)


@pytest.mark.parametrize(
    ("seq_len", "full_tp1", "full_tp8", "swa", "hca"),
    [
        pytest.param(0, 0, 0, 0, 524_288, id="empty"),
        pytest.param(1, 12_288, 1_536, 512, 524_800, id="one-token"),
        pytest.param(511, 6_279_168, 784_896, 261_632, 787_456, id="below-window"),
        pytest.param(512, 6_291_456, 786_432, 262_144, 788_480, id="at-window"),
        pytest.param(513, 6_303_744, 787_968, 262_144, 788_480, id="above-window"),
        pytest.param(
            1_048_576,
            12_884_901_888,
            1_610_612_736,
            262_144,
            4_980_736,
            id="one-million",
        ),
    ],
)
def test_step4_pro_attention_configs_distinguish_full_swa_and_hca_kv_curves(
    seq_len,
    full_tp1,
    full_tp8,
    swa,
    hca,
):
    """Config-level KV formulas must expose full growth, SWA saturation, and HCA compressed history."""
    extra = utils._parse_hf_config_json(_step4_pro_raw_config())["extra_params"]
    swa_config = replace(extra.nonfull_attention, mechanism="swa", compression_ratio=0)

    assert extra.full_attention.compute_kv_cache_bytes(seq_len, tp_size=1, bytes_per_element=1) == full_tp1
    assert extra.full_attention.compute_kv_cache_bytes(seq_len, tp_size=8, bytes_per_element=1) == full_tp8
    assert swa_config.compute_kv_cache_bytes(seq_len, bytes_per_element=1) == swa
    assert extra.nonfull_attention.compute_kv_cache_bytes(seq_len, bytes_per_element=1) == hca


@pytest.mark.parametrize(
    ("method", "expected_error"),
    [
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(-1, tp_size=1, bytes_per_element=1),
            "seq_len must be a non-negative integer",
            id="full-negative-sequence",
        ),
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(True, tp_size=1, bytes_per_element=1),
            "seq_len must be a non-negative integer",
            id="full-boolean-sequence",
        ),
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(1, tp_size=0, bytes_per_element=1),
            "tp_size must be a positive integer",
            id="full-zero-tp",
        ),
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(1, tp_size=3, bytes_per_element=1),
            "num_kv_heads 64 must be divisible by tp_size 3",
            id="full-nondivisible-tp",
        ),
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(1, tp_size=1, bytes_per_element=0),
            "bytes_per_element must be positive",
            id="full-zero-element-bytes",
        ),
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(
                1,
                tp_size=1,
                bytes_per_element=float("nan"),
            ),
            "bytes_per_element must be positive and finite",
            id="full-nan-element-bytes",
        ),
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(
                1,
                tp_size=1,
                bytes_per_element=float("inf"),
            ),
            "bytes_per_element must be positive and finite",
            id="full-positive-infinite-element-bytes",
        ),
        pytest.param(
            lambda full, _nonfull: full.compute_kv_cache_bytes(
                1,
                tp_size=1,
                bytes_per_element=float("-inf"),
            ),
            "bytes_per_element must be positive and finite",
            id="full-negative-infinite-element-bytes",
        ),
        pytest.param(
            lambda _full, nonfull: nonfull.compute_kv_cache_bytes(-1, bytes_per_element=1),
            "seq_len must be a non-negative integer",
            id="nonfull-negative-sequence",
        ),
        pytest.param(
            lambda _full, nonfull: nonfull.compute_kv_cache_bytes(1, bytes_per_element=False),
            "bytes_per_element must be positive",
            id="nonfull-boolean-element-bytes",
        ),
        pytest.param(
            lambda _full, nonfull: nonfull.compute_kv_cache_bytes(
                1,
                bytes_per_element=float("nan"),
            ),
            "bytes_per_element must be positive and finite",
            id="nonfull-nan-element-bytes",
        ),
        pytest.param(
            lambda _full, nonfull: nonfull.compute_kv_cache_bytes(
                1,
                bytes_per_element=float("inf"),
            ),
            "bytes_per_element must be positive and finite",
            id="nonfull-positive-infinite-element-bytes",
        ),
        pytest.param(
            lambda _full, nonfull: nonfull.compute_kv_cache_bytes(
                1,
                bytes_per_element=float("-inf"),
            ),
            "bytes_per_element must be positive and finite",
            id="nonfull-negative-infinite-element-bytes",
        ),
    ],
)
def test_step4_pro_attention_config_kv_formulas_reject_invalid_inputs(method, expected_error):
    """Direct KV formula calls must fail fast on invalid sequence, TP, and element-width inputs."""
    extra = utils._parse_hf_config_json(_step4_pro_raw_config())["extra_params"]

    with pytest.raises(ValueError, match=expected_error):
        method(extra.full_attention, extra.nonfull_attention)


@pytest.mark.parametrize(
    ("tp_size", "expected_bytes"),
    [
        pytest.param(
            1,
            {
                0: 31_457_280,
                1: 31_733_760,
                511: 172_830_720,
                512: 173_137_920,
                513: 173_383_680,
                1_048_576: 257_996_881_920,
            },
            id="tp1",
        ),
        pytest.param(
            8,
            {
                0: 31_457_280,
                1: 31_518_720,
                511: 62_945_280,
                512: 63_037_440,
                513: 63_068_160,
                1_048_576: 32_511_098_880,
            },
            id="tp8",
        ),
    ],
)
def test_step4_pro_v1_kv_bytes_sum_explicit_layer_types(tp_size, expected_bytes):
    """Model KV bytes must equal the explicit 20-full plus 60-HCA per-layer sum at TP1 and TP8."""
    model = _build_cached_step4_pro_model(tp_size=tp_size, moe_ep_size=tp_size)

    for seq_len, expected in expected_bytes.items():
        assert model.get_kvcache_bytes_per_sequence(seq_len) == expected


def test_step4_pro_v1_rejects_constant_kv_slope_while_legacy_step4_stays_linear():
    """The Pro curve is non-linear, while original Step4 retains its exact latent-MLA slope."""
    pro = _build_cached_step4_pro_model()
    legacy = _build_cached_step4_model()

    with pytest.raises(
        ValueError,
        match="Step4-Pro KV cache is sequence-length dependent; use get_kvcache_bytes_per_sequence",
    ):
        pro.get_kvcache_elements_per_token()
    assert legacy.get_kvcache_elements_per_token() == 92 * (512 + 64) == 52_992


def test_step4_pro_v1_kv_capacity_uses_nonlinear_binary_search():
    """KV capacity must invert the differentiated curve at exact and one-byte-below boundaries."""
    model = _build_cached_step4_pro_model()
    exact_length = 513
    exact_budget = model.get_kvcache_bytes_per_sequence(exact_length)

    assert model.get_kvcache_max_tokens(exact_budget) == exact_length
    assert model.get_kvcache_max_tokens(exact_budget - 1) == exact_length - 1
    assert model.get_kvcache_max_tokens(0) == 0
    assert model.get_kvcache_max_tokens(-1) == 0


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


def test_original_step4_model_rejects_non_divisible_attention_heads(monkeypatch):
    """Legacy Step4 must continue to shard its top-level MLA query heads exactly."""
    raw = deepcopy(utils._load_pre_downloaded_hf_config("stepfun-ai/Step4"))
    raw["num_attention_heads"] = 130

    with pytest.raises(
        ValueError,
        match=r"Step4 num_attention_heads \(130\) must be divisible by tp_size \(4\)",
    ):
        _build_synthetic_step4_model(monkeypatch, raw, tp_size=4)


@pytest.mark.parametrize(
    ("section", "field", "value", "related_values", "tp_size"),
    [
        pytest.param(
            "full_attention",
            "num_query_heads",
            60,
            {"num_kv_heads": 20},
            8,
            id="full-query-heads",
        ),
        pytest.param(
            "full_attention",
            "num_kv_heads",
            4,
            {},
            8,
            id="full-kv-heads",
        ),
        pytest.param(
            "nonfull_attention",
            "num_query_heads",
            98,
            {},
            4,
            id="nonfull-query-heads",
        ),
        pytest.param(
            "nonfull_attention",
            "o_groups",
            18,
            {},
            4,
            id="nonfull-output-groups",
        ),
    ],
)
def test_step4_pro_model_rejects_non_divisible_attention_geometry(
    monkeypatch,
    section,
    field,
    value,
    related_values,
    tp_size,
):
    """Pro attention types must validate their nested TP geometry independently."""
    raw = _step4_pro_raw_config()
    raw[section][field] = value
    raw[section].update(related_values)

    with pytest.raises(
        ValueError,
        match=rf"Step4-Pro {section}.{field} \({value}\) must be divisible by tp_size \({tp_size}\)",
    ):
        _build_synthetic_step4_model(monkeypatch, raw, tp_size=tp_size)
