import json
from pathlib import Path

from tests.e2e.step4_pro_latest.generate_step4pro_dummy_configs import (
    build_smoke_config,
    build_target_config,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    ROOT
    / "task_memory"
    / "task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation"
    / "step4_pro_latest_shape_manifest.reconstructed.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_target_config_matches_authoritative_shape_manifest() -> None:
    manifest = _manifest()
    config = build_target_config(manifest)

    assert config["model_type"] == "step4pro"
    assert config["architectures"] == ["Step4ProForCausalLM"]
    assert config["hidden_size"] == 7168
    assert config["num_hidden_layers"] == 78
    assert config["vocab_size"] == 128896
    assert config["moe_num_experts"] == 896
    assert config["moe_top_k"] == 16
    assert config["latent_moe_hidden_size"] == 3584
    assert config["moe_intermediate_size"] == 3584
    assert config["share_expert_dim"] == 3584
    assert config["num_attention_heads"] == 64
    assert config["num_attention_groups"] == 1
    assert config["head_dim"] == 512
    assert config["mqa_q_lora_rank"] == 2048
    assert config["mqa_o_lora_rank"] == 1024
    assert config["mqa_output_heads_per_group"] == 8
    assert config["attention_other_setting"] == {
        "attention_type": "sliding_attention",
        "num_attention_heads": 128,
        "num_attention_groups": 8,
        "head_dim": 128,
        "gqa_v_norm": True,
    }
    assert config["max_position_embeddings"] == 1048576
    assert config["quantization_config"] == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }

    full_ids = [index for index, layer_type in enumerate(config["layer_types"]) if layer_type == "full_attention"]
    assert full_ids == manifest["layers"]["full_attention_ids"]
    assert config["moe_layer_list"] == list(range(2, 78))
    assert config["num_nextn_predict_layers"] == 0


def test_smoke_config_exercises_required_provider_paths_on_one_gpu() -> None:
    config = build_smoke_config()

    assert config["num_hidden_layers"] == 14
    assert config["hidden_size"] == 1792
    assert config["head_dim"] == 512
    assert config["mqa_rope_dim"] == 64
    assert config["moe_num_experts"] == 64
    assert config["moe_top_k"] == 16
    assert config["moe_layer_list"] == list(range(2, 14))
    assert config["layer_types"] == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert config["attention_other_setting"]["head_dim"] == 128
    assert config["attention_other_setting"]["gqa_v_norm"] is True
    assert config["quantization_config"]["weight_block_size"] == [128, 128]
