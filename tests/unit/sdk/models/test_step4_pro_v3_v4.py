"""Unit coverage for the CSV/README-backed Step4-Pro V3/V4 MQA models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aiconfigurator.sdk import common, config, utils
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations import MoE, MoEDispatch
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    GenerationDeepSeekV4AttentionModule,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("model_id", "hidden", "experts", "topk", "moe_inter", "latent"),
    [
        ("stepfun-ai/Step4-Pro-V3", 12_288, 1_024, 16, 2_048, 6_144),
        ("stepfun-ai/Step4-Pro-V4", 9_216, 384, 8, 3_584, 0),
    ],
)
def test_v3_v4_config_matches_series_reference(model_id, hidden, experts, topk, moe_inter, latent):
    """The package-local configs encode the README and CSV values exactly."""
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    info = utils.get_model_config_from_model_path(model_id)
    extra = info["extra_params"]

    assert isinstance(extra, common.Step4ProMQAConfig)
    assert info["hidden_size"] == hidden
    assert info["num_experts"] == experts
    assert info["topk"] == topk
    assert info["moe_inter_size"] == moe_inter
    assert extra.latent_moe_dim == latent
    assert extra.dense_inter_size == moe_inter * topk
    assert len(extra.layers) == 80
    assert sum(layer.attention_type == "full" for layer in extra.layers) == 20
    assert sum(layer.attention_type == "nonfull" for layer in extra.layers) == 60
    assert sum(layer.ffn_type == "dense" for layer in extra.layers) == 4
    assert all(layer.attention_type == ("full" if layer.layer_id % 4 == 3 else "nonfull") for layer in extra.layers)

    assert extra.full_attention.attention_type == "mqa"
    assert extra.full_attention.num_query_heads == 96
    assert extra.full_attention.num_kv_heads == 12
    assert extra.full_attention.output_groups == 12
    assert extra.full_attention.cache_entry_width == 512
    assert extra.full_attention.retention_mode == "full"
    assert extra.nonfull_attention.attention_type == "mqa"
    assert extra.nonfull_attention.num_query_heads == 128
    assert extra.nonfull_attention.num_kv_heads == 16
    assert extra.nonfull_attention.output_groups == 16
    assert extra.nonfull_attention.cache_entry_width == 512
    assert extra.nonfull_attention.window_size == 512
    assert extra.nonfull_attention.retention_mode == "swa"


@pytest.mark.parametrize(
    ("model_id", "full_target", "swa_target", "attention_total", "total_parameters", "active_parameters"),
    [
        (
            "stepfun-ai/Step4-Pro-V3",
            314_575_968,
            394_266_752,
            29_947_524_480,
            2_990_708_684_160,
            98_853_516_672,
        ),
        (
            "stepfun-ai/Step4-Pro-V4",
            267_390_048,
            337_643_648,
            25_606_419_840,
            2_928_433_788_288,
            96_825_603_456,
        ),
    ],
)
def test_v3_v4_parameter_ledger_closes_csv_targets(
    model_id, full_target, swa_target, attention_total, total_parameters, active_parameters
):
    """The explicit MQA projection ledger must reproduce every CSV parameter target."""
    info = utils.get_model_config_from_model_path(model_id)
    extra = info["extra_params"]
    assert isinstance(extra, common.Step4ProMQAConfig)
    assert extra.full_attention.compute_parameter_count() == full_target
    assert extra.nonfull_attention.compute_parameter_count() == swa_target
    assert extra.compute_attention_parameter_count() == attention_total
    assert extra.compute_total_parameter_count(info) == total_parameters
    assert extra.compute_active_parameter_count(info) == active_parameters


@pytest.mark.parametrize("model_id", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_v3_v4_kv_audit_matches_full_history_and_swa_window(model_id):
    """The MQA topology uses full-history layers and 512-token SWA layers."""
    info = utils.get_model_config_from_model_path(model_id)
    extra = info["extra_params"]
    assert isinstance(extra, common.Step4ProMQAConfig)
    assert extra.compute_kv_cache_bytes(1_048_576, bytes_per_element=1) == 10_753_146_880


def test_v3_v4_inferred_mqa_ledger_fails_fast_on_inconsistent_inputs():
    """Inferred geometry is explicit and rejects mismatched targets or runtime mappings."""
    info = utils.get_model_config_from_model_path("stepfun-ai/Step4-Pro-V3")
    full = info["extra_params"].full_attention
    assert full.compute_standard_mqa_parameter_count() == 339_738_624
    assert full.compute_standard_mqa_parameter_count() != full.target_parameter_count
    assert full.compute_kv_cache_bytes(0, bytes_per_element=1) == 0
    with pytest.raises(ValueError, match="output_groups must equal num_kv_heads"):
        replace(full, output_groups=6)
    with pytest.raises(ValueError, match="target_parameter_count does not match"):
        replace(full, target_parameter_count=full.target_parameter_count + 1)
    with pytest.raises(ValueError, match="bytes_per_element must be positive and finite"):
        full.compute_kv_cache_bytes(1, bytes_per_element=float("nan"))


def test_v3_latent_moe_uses_latent_routed_expert_width():
    """V3 routed experts run in latent space while the shared expert stays in hidden space."""
    info = utils.get_model_config_from_model_path("stepfun-ai/Step4-Pro-V3")
    extra = info["extra_params"]
    assert isinstance(extra, common.Step4ProMQAConfig)
    assert (
        extra.compute_moe_parameter_count(
            hidden_size=info["hidden_size"],
            num_experts=info["num_experts"],
            topk=info["topk"],
            moe_inter_size=info["moe_inter_size"],
            shared_expert_inter_size=extra.shared_expert_inter_size,
        )
        == 38_893_780_992
    )


@pytest.mark.parametrize("model_id", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_v3_v4_model_uses_mqa_operations_and_no_dsv4(model_id):
    """V3/V4 attention is standard Context/GenerationAttention, not DSV4 MFA."""
    model = get_model(
        model_id,
        config.ModelConfig(tp_size=4, pp_size=8, moe_tp_size=4, moe_ep_size=1),
        "vllm",
    )
    operations = [*model.context_ops, *model.generation_ops]
    assert not any(
        isinstance(op, (ContextDeepSeekV4AttentionModule, GenerationDeepSeekV4AttentionModule)) for op in operations
    )
    attention_names = [op._name for op in operations if op._name.endswith("_attention")]
    assert len(attention_names) == 160
    assert attention_names[0].startswith("context_layer_000_nonfull_mqa")
    assert attention_names[3].startswith("context_layer_003_full_mqa")
    assert attention_names[80].startswith("generation_layer_000_nonfull_mqa")


def test_v3_v4_kv_cache_is_mqa_kv_and_swa_capped():
    """Full MQA grows with history while SWA stops growing after 512 tokens."""
    for model_id in ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"):
        model = get_model(
            model_id,
            config.ModelConfig(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=1),
            "vllm",
        )
        at_512 = model.get_kvcache_bytes_per_sequence(512)
        at_1024 = model.get_kvcache_bytes_per_sequence(1024)
        assert at_1024 > at_512
        expected_delta = 20 * 512 * (1024 - 512)
        assert at_1024 - at_512 == pytest.approx(expected_delta)


@pytest.mark.parametrize(
    ("model_id", "expected_width"),
    [
        ("stepfun-ai/Step4-Pro-V3", 6_144),
        ("stepfun-ai/Step4-Pro-V4", 9_216),
    ],
)
def test_v3_v4_routed_moe_operations_use_architecture_width(model_id, expected_width):
    """Routed MoE operations use latent width only for the V3 latent-MoE path."""
    model = get_model(
        model_id,
        config.ModelConfig(tp_size=4, pp_size=8, moe_tp_size=4, moe_ep_size=1),
        "vllm",
    )
    for operations, phase in ((model.context_ops, "context"), (model.generation_ops, "generation")):
        expanded_operations = [
            nested
            for operation in operations
            for nested in (operation._group_a if hasattr(operation, "_group_a") else [operation])
        ]
        routed = [operation for operation in expanded_operations if isinstance(operation, (MoE, MoEDispatch))]
        assert routed
        assert all(operation._hidden_size == expected_width for operation in routed)
        names = [operation._name for operation in expanded_operations]
        routed_indexes = [names.index(operation._name) for operation in routed]
        if model_id.endswith("V3"):
            down_index = names.index(f"{phase}_latent_moe_down_proj")
            up_index = names.index(f"{phase}_latent_moe_up_proj")
            assert down_index < min(routed_indexes)
            assert max(routed_indexes) < up_index
        else:
            assert f"{phase}_latent_moe_down_proj" not in names
            assert f"{phase}_latent_moe_up_proj" not in names
