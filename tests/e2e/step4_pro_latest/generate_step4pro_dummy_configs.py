#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FP8_QUANTIZATION_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}


def _base_config(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    vocab_size: int,
    moe_num_experts: int,
    moe_top_k: int,
    moe_intermediate_size: int,
    share_expert_dim: int,
    latent_moe_hidden_size: int,
    num_attention_heads: int,
    head_dim: int,
    mqa_q_lora_rank: int,
    mqa_o_lora_rank: int,
    mqa_output_heads_per_group: int,
    sliding_num_attention_heads: int,
    sliding_num_attention_groups: int,
    layer_types: list[str],
    max_position_embeddings: int,
) -> dict[str, Any]:
    return {
        "architectures": ["Step4ProForCausalLM"],
        "model_type": "step4pro",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "att_impl_type": "GQA",
        "num_attention_heads": num_attention_heads,
        "num_attention_groups": 1,
        "head_dim": head_dim,
        "layer_types": layer_types,
        "attention_other_setting": {
            "attention_type": "sliding_attention",
            "num_attention_heads": sliding_num_attention_heads,
            "num_attention_groups": sliding_num_attention_groups,
            "head_dim": 128,
            "gqa_v_norm": True,
        },
        "sliding_window": 512,
        "use_head_wise_attn_gate": True,
        "mqa_q_lora_rank": mqa_q_lora_rank,
        "mqa_o_lora_rank": mqa_o_lora_rank,
        "mqa_rope_dim": 64,
        "mqa_rope_position": "tail",
        "mqa_rope_on_v": True,
        "mqa_inverse_rope_output": True,
        "mqa_sliding_attention_heads": sliding_num_attention_heads,
        "mqa_sliding_o_lora_rank": mqa_o_lora_rank,
        "mqa_output_heads_per_group": mqa_output_heads_per_group,
        "mqa_sliding_output_heads_per_group": 8,
        "use_moe": True,
        "latent_moe_enabled": True,
        "latent_moe_hidden_size": latent_moe_hidden_size,
        "latent_moe_shared_expert_scale": 0.25,
        "latent_moe_use_gated_norm_for_post_proj": True,
        "latent_moe_projection_mode": "independent",
        "latent_moe_use_origin_input_for_router": True,
        "moe_layer_list": list(range(2, num_hidden_layers)),
        "moe_num_experts": moe_num_experts,
        "moe_top_k": moe_top_k,
        "moe_intermediate_size": moe_intermediate_size,
        "share_expert_dim": share_expert_dim,
        "moe_router_activation": "sigmoid",
        "moe_router_scaling_factor": 1.0,
        "use_moe_router_bias": True,
        "need_fp32_gate": True,
        "norm_expert_weight": True,
        "ffn_activation": "situ-glu",
        "norm_dtype": "fp32",
        "zero_centered": False,
        "fp32_residual_connection": False,
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": 0,
        "bos_token_id": 1,
        "eos_token_id": [2, 3],
        "quantization_config": dict(FP8_QUANTIZATION_CONFIG),
    }


def build_smoke_config() -> dict[str, Any]:
    full_attention_ids = {3, 7, 11, 13}
    layer_types = [
        "full_attention" if layer_id in full_attention_ids else "sliding_attention" for layer_id in range(14)
    ]
    return _base_config(
        hidden_size=1792,
        intermediate_size=7168,
        num_hidden_layers=14,
        vocab_size=128896,
        moe_num_experts=64,
        moe_top_k=16,
        moe_intermediate_size=896,
        share_expert_dim=896,
        latent_moe_hidden_size=896,
        num_attention_heads=8,
        head_dim=512,
        mqa_q_lora_rank=512,
        mqa_o_lora_rank=256,
        mqa_output_heads_per_group=2,
        sliding_num_attention_heads=16,
        sliding_num_attention_groups=4,
        layer_types=layer_types,
        max_position_embeddings=8192,
    )


def build_target_config(manifest: dict[str, Any]) -> dict[str, Any]:
    model = manifest["model"]
    full = manifest["attention"]["full_mfa"]
    sliding = manifest["attention"]["swa_gqa"]
    dense = manifest["ffn"]["dense"]
    moe = manifest["ffn"]["latent_moe"]
    full_attention_ids = set(manifest["layers"]["full_attention_ids"])
    layer_types = [
        "full_attention" if layer_id in full_attention_ids else "sliding_attention"
        for layer_id in range(model["num_hidden_layers"])
    ]
    return _base_config(
        hidden_size=model["hidden_size"],
        intermediate_size=dense["intermediate_size"],
        num_hidden_layers=model["num_hidden_layers"],
        vocab_size=model["vocab_size"],
        moe_num_experts=moe["num_experts"],
        moe_top_k=moe["topk"],
        moe_intermediate_size=moe["moe_intermediate_size"],
        share_expert_dim=moe["shared_expert_intermediate_size"],
        latent_moe_hidden_size=moe["latent_hidden_size"],
        num_attention_heads=full["num_query_heads"],
        head_dim=full["head_dim"],
        mqa_q_lora_rank=full["q_lora_rank"],
        mqa_o_lora_rank=full["o_lora_rank"],
        mqa_output_heads_per_group=(full["num_query_heads"] // full["output_groups"]),
        sliding_num_attention_heads=sliding["num_query_heads"],
        sliding_num_attention_groups=sliding["num_kv_heads"],
        layer_types=layer_types,
        max_position_embeddings=1048576,
    )


def write_configs(manifest_path: Path, output_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    configs = {
        "step4pro_smoke_14l_dummy": build_smoke_config(),
        "step4pro_v4_78l_dummy": build_target_config(manifest),
    }
    for name, config in configs.items():
        output_dir = output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    write_configs(args.manifest, args.output_root)


if __name__ == "__main__":
    main()
