# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step4 predefined model built from granular roofline-capable operations."""

from __future__ import annotations

from typing import ClassVar

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model
from aiconfigurator.sdk.models.helpers import calc_expectation


@register_model("STEP4")
class Step4Model(BaseModel):
    """Step4 trunk graph with an explicitly labeled temporary MLA approximation."""

    MIXED_STEP_CONTEXT_ATTENTION_KEYS: ClassVar[tuple[str, ...]] = (
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
    MIXED_STEP_GENERATION_ATTENTION_KEYS: ClassVar[tuple[str, ...]] = (
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

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        moe_args = (model_info["topk"], model_info["num_experts"], model_info["moe_inter_size"])
        base_args = (
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
            model_info["extra_params"],
        )
        return cls(*moe_args, *base_args, backend_name=backend_name)

    def __init__(
        self,
        topk: int,
        num_experts: int,
        moe_inter_size: int,
        *args,
        backend_name: str,
    ) -> None:
        super().__init__(*args)
        if backend_name != "vllm":
            raise NotImplementedError(
                f"Step4 predefined ops currently support only backend='vllm'; got backend={backend_name!r}."
            )
        if not isinstance(self.extra_params, common.Step4Config):
            raise TypeError("Step4Model requires Step4Config extra_params.")

        attention_width = self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
        moe_width = self.config.moe_tp_size * self.config.moe_ep_size
        if attention_width != moe_width:
            raise ValueError(f"Step4 attention width {attention_width} must equal MoE width {moe_width}.")
        if num_experts < self.config.moe_ep_size:
            raise ValueError(f"Step4 num_experts ({num_experts}) must be >= moe_ep_size ({self.config.moe_ep_size}).")

        self._backend_name = backend_name
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = (
            1.0
            / (1.0 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._num_layers + self._nextn)
            / self._num_layers
        )

        self._build_context_ops()
        self._build_generation_ops()

    @property
    def activation_hidden_size(self) -> int:
        """Return the residual-stream width rather than the temporary MLA head product."""
        return self._hidden_size

    def get_kvcache_elements_per_token(self) -> int:
        """Return the per-GPU latent MLA KV elements retained for every trunk layer."""
        cfg = self.extra_params
        return self._num_layers * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)

    def _layer_counts(self) -> tuple[int, int, int]:
        cfg = self.extra_params
        dense_count = cfg.block_types.count("dense_swa")
        full_count = cfg.block_types.count("moe_full")
        swa_count = dense_count + cfg.block_types.count("moe_swa")
        return dense_count, full_count, swa_count

    def _context_attention_ops(self, label: str, count: int) -> list:
        cfg = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        prefix = f"context_{label}_mla_approx"
        return [
            ops.ElementWise(f"{prefix}_attn_norm", count, 2 * h, 2 * h, 0.8),
            ops.GEMM(f"{prefix}_downscale_gemm", count, 2112, h, self.config.gemm_quant_mode),
            ops.GEMM(
                f"{prefix}_q_b_proj_gemm",
                count,
                24576 // tp,
                cfg.q_lora_rank,
                self.config.gemm_quant_mode,
            ),
            ops.GEMM(
                f"{prefix}_kv_b_proj_gemm",
                count,
                32768 // tp,
                cfg.kv_lora_rank,
                self.config.gemm_quant_mode,
            ),
            ops.ContextMLA(
                f"{prefix}_attention",
                count,
                self._num_heads // tp,
                self.config.kvcache_quant_mode,
                self.config.fmha_quant_mode,
            ),
            ops.GEMM(
                f"{prefix}_proj_gemm",
                count,
                h,
                self._num_heads * cfg.v_head_dim // tp,
                self.config.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.CustomAllReduce(f"{prefix}_attention_ar", count, h, tp),
        ]

    def _generation_attention_ops(self, label: str, count: int) -> list:
        cfg = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        scale = count * self._mtp_scale_factor
        prefix = f"generation_{label}_mla_approx"
        bmm_quant_mode = (
            common.GEMMQuantMode.bfloat16
            if self.config.gemm_quant_mode == common.GEMMQuantMode.bfloat16
            else common.GEMMQuantMode.fp8
        )
        return [
            ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.GEMM(f"{prefix}_downscale_gemm", scale, 2112, h, self.config.gemm_quant_mode),
            ops.GEMM(
                f"{prefix}_q_b_proj_gemm",
                scale,
                24576 // tp,
                cfg.q_lora_rank,
                self.config.gemm_quant_mode,
            ),
            ops.MLABmm(f"{prefix}_bmm_pre", scale, self._num_heads // tp, bmm_quant_mode, if_pre=True),
            ops.GenerationMLA(
                f"{prefix}_attention",
                scale,
                self._num_heads // tp,
                self.config.kvcache_quant_mode,
            ),
            ops.MLABmm(f"{prefix}_bmm_post", scale, self._num_heads // tp, bmm_quant_mode, if_pre=False),
            ops.GEMM(
                f"{prefix}_proj_gemm",
                scale,
                h,
                self._num_heads * cfg.v_head_dim // tp,
                self.config.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.CustomAllReduce(f"{prefix}_attention_ar", scale, h, tp),
        ]

    def _dense_ffn_ops(self, phase: str, count: int, scale: float) -> list:
        cfg = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        local_inter_size = cfg.dense_inter_size // tp
        return [
            ops.ElementWise(f"{phase}_dense_ffn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.GEMM(
                f"{phase}_dense_gate_up_gemm",
                scale,
                2 * local_inter_size,
                h,
                self.config.gemm_quant_mode,
            ),
            ops.ElementWise(
                f"{phase}_dense_swiglu",
                scale,
                2 * local_inter_size,
                local_inter_size,
                0.8,
            ),
            ops.GEMM(
                f"{phase}_dense_down_gemm",
                scale,
                h,
                local_inter_size,
                self.config.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.CustomAllReduce(f"{phase}_dense_ffn_ar", scale, h, tp),
        ]

    def _shared_ffn_ops(self, phase: str, scale: float) -> list:
        cfg = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        local_inter_size = cfg.shared_expert_inter_size // tp
        return [
            ops.GEMM(
                f"{phase}_shared_gate_up_gemm",
                scale,
                2 * local_inter_size,
                h,
                self.config.gemm_quant_mode,
            ),
            ops.ElementWise(
                f"{phase}_shared_swiglu",
                scale,
                2 * local_inter_size,
                local_inter_size,
                0.8,
            ),
            ops.GEMM(
                f"{phase}_shared_down_gemm",
                scale,
                h,
                local_inter_size,
                self.config.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.CustomAllReduce(f"{phase}_shared_ffn_ar", scale, h, tp),
        ]

    def _context_moe_ops(self, count: int) -> list:
        h = self._hidden_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attention_dp = self.config.attention_dp_size
        result = [
            ops.ElementWise("context_moe_ffn_norm", count, 2 * h, 2 * h, 0.8),
            ops.GEMM(
                "context_moe_router_gemm",
                count,
                self._num_experts,
                h,
                common.GEMMQuantMode.bfloat16,
            ),
            ops.MoEDispatch(
                "context_moe_pre_dispatch",
                count,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attention_dp,
                True,
                quant_mode=self.config.moe_quant_mode,
                reduce_results=False,
            ),
            ops.MoE(
                "context_moe",
                count,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                self.config.moe_quant_mode,
                self.config.workload_distribution,
                attention_dp,
                is_context=True,
                is_gated=True,
            ),
            ops.MoEDispatch(
                "context_moe_post_dispatch",
                count,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attention_dp,
                False,
                quant_mode=self.config.moe_quant_mode,
            ),
        ]
        result.extend(self._shared_ffn_ops("context", count))
        result.append(ops.ElementWise("context_moe_shared_merge", count, 2 * h, h, 0.8))
        return result

    def _generation_moe_ops(self, count: int) -> list:
        h = self._hidden_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attention_dp = self.config.attention_dp_size
        scale = count * self._mtp_scale_factor
        routed_ops = [
            ops.GEMM(
                "generation_moe_router_gemm",
                scale,
                self._num_experts,
                h,
                common.GEMMQuantMode.bfloat16,
            ),
            ops.MoEDispatch(
                "generation_moe_pre_dispatch",
                scale,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attention_dp,
                True,
                quant_mode=self.config.moe_quant_mode,
                reduce_results=False,
                is_context=False,
            ),
            ops.MoE(
                "generation_moe",
                scale,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                self.config.moe_quant_mode,
                self.config.workload_distribution,
                attention_dp,
                is_context=False,
                is_gated=True,
            ),
            ops.MoEDispatch(
                "generation_moe_post_dispatch",
                scale,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attention_dp,
                False,
                quant_mode=self.config.moe_quant_mode,
                is_context=False,
            ),
        ]
        shared_ops = self._shared_ffn_ops("generation", scale)
        return [
            ops.ElementWise("generation_moe_ffn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.OverlapOp("generation_moe_overlap", group_a=routed_ops, group_b=shared_ops),
            ops.ElementWise("generation_moe_shared_merge", scale, 2 * h, h, 0.8),
        ]

    def _build_context_ops(self) -> None:
        dense_count, full_count, swa_count = self._layer_counts()
        moe_count = self._num_layers - dense_count
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size

        self.context_ops = [
            ops.Embedding("context_embedding", 1, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("context_embedding_ar", 1, h, tp),
        ]
        self.context_ops.extend(self._context_attention_ops("full", full_count))
        self.context_ops.extend(self._context_attention_ops("swa", swa_count))
        self.context_ops.extend(self._dense_ffn_ops("context", dense_count, dense_count))
        self.context_ops.extend(self._context_moe_ops(moe_count))
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp,
                    h,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

    def _build_generation_ops(self) -> None:
        dense_count, full_count, swa_count = self._layer_counts()
        moe_count = self._num_layers - dense_count
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        scale = self._mtp_scale_factor

        self.generation_ops = [
            ops.Embedding("generation_embedding", scale, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("generation_embedding_ar", scale, h, tp),
        ]
        self.generation_ops.extend(self._generation_attention_ops("full", full_count))
        self.generation_ops.extend(self._generation_attention_ops("swa", swa_count))
        self.generation_ops.extend(self._dense_ffn_ops("generation", dense_count, dense_count * scale))
        self.generation_ops.extend(self._generation_moe_ops(moe_count))
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    scale,
                    self._vocab_size // tp,
                    h,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.P2P("generation_p2p", (pp - 1) * scale, h, pp),
            ]
        )
