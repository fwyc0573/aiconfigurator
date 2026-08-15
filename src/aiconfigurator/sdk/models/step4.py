# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step4 predefined model built from granular roofline-capable operations."""

from __future__ import annotations

import logging
from typing import ClassVar

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model
from aiconfigurator.sdk.models.helpers import calc_expectation

logger = logging.getLogger(__name__)

_STEP4_PRO_PARAMETER_TOLERANCE = 0.05
_STEP4_PRO_KV_AUDIT_SEQUENCE_LENGTH = 1_048_576
_STEP4_PRO_KV_TARGET_GB = 10.7
_STEP4_PRO_MOE_WORKLOAD_DISTRIBUTION = "power_law_1.2"


@register_model("STEP4")
class Step4Model(BaseModel):
    """Step4 legacy MLA graph and Step4-Pro explicit hybrid-attention graph."""

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
        step4_config = model_info["extra_params"]
        supported_configs = (
            common.Step4Config,
            common.Step4ProConfig,
            common.Step4ProMQAConfig,
            common.Step4ProLatestConfig,
        )
        if not isinstance(step4_config, supported_configs):
            raise TypeError("Step4Model requires a supported Step4 extra_params contract.")

        attention_parallel_geometry = []
        if isinstance(step4_config, common.Step4Config):
            attention_parallel_geometry.append(
                ("Step4", "num_attention_heads", model_info["n"], "tp_size", model_config.tp_size)
            )
        elif isinstance(step4_config, common.Step4ProLatestConfig):
            if model_config.nextn != 0:
                raise ValueError("Step4-Pro-Latest is MTP-off and does not support nextn or multi-token prediction.")
            if model_config.tp_size != 1:
                raise NotImplementedError("Step4-Pro-Latest Full MFA currently requires tensor parallel TP=1.")
        elif isinstance(step4_config, common.Step4ProMQAConfig):
            attention_parallel_geometry.extend(
                (
                    (
                        "Step4-Pro",
                        "full_attention.num_query_heads",
                        step4_config.full_attention.num_query_heads,
                        "tp_size",
                        model_config.tp_size,
                    ),
                    (
                        "Step4-Pro",
                        "full_attention.num_kv_heads",
                        step4_config.full_attention.num_kv_heads,
                        "tp_size",
                        model_config.tp_size,
                    ),
                    (
                        "Step4-Pro",
                        "nonfull_attention.num_query_heads",
                        step4_config.nonfull_attention.num_query_heads,
                        "tp_size",
                        model_config.tp_size,
                    ),
                    (
                        "Step4-Pro",
                        "nonfull_attention.num_kv_heads",
                        step4_config.nonfull_attention.num_kv_heads,
                        "tp_size",
                        model_config.tp_size,
                    ),
                )
            )
        else:
            full_attention = step4_config.full_attention
            nonfull_attention = step4_config.nonfull_attention
            attention_parallel_geometry.append(
                (
                    "Step4-Pro",
                    "full_attention.num_query_heads",
                    full_attention.num_query_heads,
                    "tp_size",
                    model_config.tp_size,
                )
            )
            if isinstance(full_attention, common.FullAttentionConfig):
                attention_parallel_geometry.append(
                    (
                        "Step4-Pro",
                        "full_attention.num_kv_heads",
                        full_attention.num_kv_heads,
                        "tp_size",
                        model_config.tp_size,
                    )
                )
            attention_parallel_geometry.append(
                (
                    "Step4-Pro",
                    "nonfull_attention.num_query_heads",
                    nonfull_attention.num_query_heads,
                    "tp_size",
                    model_config.tp_size,
                )
            )
            if isinstance(nonfull_attention, common.NonFullAttentionConfig):
                attention_parallel_geometry.append(
                    (
                        "Step4-Pro",
                        "nonfull_attention.o_groups",
                        nonfull_attention.o_groups,
                        "tp_size",
                        model_config.tp_size,
                    )
                )

        parallel_geometry = (
            *attention_parallel_geometry,
            ("Step4", "vocab_size", model_info["vocab"], "tp_size", model_config.tp_size),
            (
                "Step4",
                "intermediate_size",
                step4_config.dense_inter_size,
                "tp_size",
                model_config.tp_size,
            ),
            (
                "Step4",
                "shared_expert_intermediate_size",
                step4_config.shared_expert_inter_size,
                "tp_size",
                model_config.tp_size,
            ),
            (
                "Step4",
                "n_routed_experts",
                model_info["num_experts"],
                "moe_ep_size",
                model_config.moe_ep_size,
            ),
            (
                "Step4",
                "moe_intermediate_size",
                model_info["moe_inter_size"],
                "moe_tp_size",
                model_config.moe_tp_size,
            ),
        )
        for model_label, field, value, parallel_name, parallel_size in parallel_geometry:
            if value % parallel_size != 0:
                raise ValueError(
                    f"{model_label} {field} ({value}) must be divisible by {parallel_name} ({parallel_size})."
                )

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
        supported_configs = (
            common.Step4Config,
            common.Step4ProConfig,
            common.Step4ProMQAConfig,
            common.Step4ProLatestConfig,
        )
        if not isinstance(self.extra_params, supported_configs):
            raise TypeError("Step4Model requires a supported Step4 extra_params contract.")

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

        if isinstance(self.extra_params, common.Step4ProLatestConfig):
            self.kv_cache_requested_dtype = self.extra_params.kv_cache_requested_dtype
            self.kv_cache_resolved_dtype = self.extra_params.kv_cache_resolved_dtype
            self.kv_cache_page_size = self.extra_params.full_page_size
        elif isinstance(self.extra_params, common.Step4ProConfig):
            self._validate_and_report_pro_attention()
        self._build_context_ops()
        self._build_generation_ops()

    @property
    def activation_hidden_size(self) -> int:
        """Return the residual-stream width rather than the temporary MLA head product."""
        return self._hidden_size

    @staticmethod
    def _attention_parameter_result(label: str, attention_config) -> tuple[str, int, int, int, float, str]:
        """Return one deterministic Step4-Pro parameter-validation row."""
        estimate = attention_config.compute_parameter_count()
        target = attention_config.target_parameter_count
        absolute_error = abs(estimate - target)
        relative_error = absolute_error / target
        status = "PASS" if relative_error <= _STEP4_PRO_PARAMETER_TOLERANCE else "FAIL"
        return label, target, estimate, absolute_error, relative_error, status

    def _pro_kv_audit_gb(self) -> float:
        """Return the reviewed TP1/FP8 1M-token candidate used in the warning report."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProConfig):
            raise TypeError("Step4-Pro KV audit requires Step4ProConfig")

        sequence_length = _STEP4_PRO_KV_AUDIT_SEQUENCE_LENGTH
        full_count = sum(layer.attention_type == "full" for layer in cfg.layers)
        nonfull_count = sum(layer.attention_type == "nonfull" for layer in cfg.layers)
        full = cfg.full_attention
        nonfull = cfg.nonfull_attention

        full_bytes = full_count * full.compute_kv_cache_bytes(
            sequence_length,
            tp_size=1,
            bytes_per_element=1,
        )
        nonfull_bytes_per_layer = nonfull.compute_kv_cache_bytes(sequence_length, tp_size=1, bytes_per_element=1)
        return (full_bytes + nonfull_count * nonfull_bytes_per_layer) / 1_000_000_000

    def _validate_and_report_pro_attention(self) -> None:
        """Warn with numeric evidence and reject attention parameter errors above 5%."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProConfig):
            raise TypeError("Step4-Pro parameter validation requires Step4ProConfig")

        results = (
            self._attention_parameter_result("full", cfg.full_attention),
            self._attention_parameter_result("nonfull", cfg.nonfull_attention),
        )
        kv_estimate_gb = self._pro_kv_audit_gb()
        kv_ratio = kv_estimate_gb / _STEP4_PRO_KV_TARGET_GB
        report_rows = [
            "Step4-Pro-V1 attention parameter validation",
            *(
                f"{label} target={target} estimate={estimate} absolute_error={absolute_error} "
                f"relative_error={relative_error * 100:.10f}% status={status}"
                for label, target, estimate, absolute_error, relative_error, status in results
            ),
            (
                f"nonfull resident_state_elements={cfg.nonfull_attention.resident_state_elements}"
                if isinstance(cfg.nonfull_attention, common.NonFullAttentionConfig)
                else f"nonfull cache_entry_width={cfg.nonfull_attention.cache_entry_width}"
            ),
            (
                f"Step4-Pro-V1 KV target conflict at {_STEP4_PRO_KV_AUDIT_SEQUENCE_LENGTH} FP8 tokens: "
                f"estimate={kv_estimate_gb:.8f} GB target={_STEP4_PRO_KV_TARGET_GB} GB "
                f"ratio={kv_ratio:.10f}x status=unresolved"
            ),
        ]
        logger.warning("\n".join(report_rows))

        for label, target, estimate, _, relative_error, status in results:
            if status == "FAIL":
                raise ValueError(
                    f"Step4-Pro {label} attention parameter error {relative_error * 100:.10f}% exceeds 5% "
                    f"(estimate={estimate}, target={target})."
                )

    def get_kvcache_elements_per_token(self) -> int:
        """Return the per-GPU latent MLA KV elements retained for every trunk layer."""
        cfg = self.extra_params
        if isinstance(cfg, (common.Step4ProConfig, common.Step4ProMQAConfig, common.Step4ProLatestConfig)):
            raise ValueError(  # noqa: TRY004
                "Step4-Pro KV cache is sequence-length dependent; use get_kvcache_bytes_per_sequence instead."
            )
        return self._num_layers * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """Return per-GPU bytes from each explicit Pro attention layer's KV curve."""
        cfg = self.extra_params
        if isinstance(cfg, common.Step4ProLatestConfig):
            return cfg.compute_kv_cache_bytes(
                seq_len,
                bytes_per_element=common.GEMMQuantMode.bfloat16.value.memory,
            )
        if isinstance(cfg, common.Step4ProMQAConfig):
            bytes_per_element = self.config.kvcache_quant_mode.value.memory
            return sum(
                (
                    cfg.full_attention if layer.attention_type == "full" else cfg.nonfull_attention
                ).compute_kv_cache_bytes(seq_len, bytes_per_element=bytes_per_element)
                for layer in cfg.layers
            )
        if not isinstance(cfg, common.Step4ProConfig):
            return super().get_kvcache_bytes_per_sequence(seq_len)

        bytes_per_element = self.config.kvcache_quant_mode.value.memory
        total_bytes = 0.0
        for layer in cfg.layers:
            if layer.attention_type == "full":
                total_bytes += cfg.full_attention.compute_kv_cache_bytes(
                    seq_len,
                    tp_size=self.config.tp_size,
                    bytes_per_element=bytes_per_element,
                )
            elif layer.attention_type == "nonfull":
                total_bytes += cfg.nonfull_attention.compute_kv_cache_bytes(
                    seq_len,
                    tp_size=self.config.tp_size,
                    bytes_per_element=bytes_per_element,
                )
            else:
                raise ValueError(
                    f"Unsupported Step4-Pro attention_type {layer.attention_type!r} at layer {layer.layer_id}"
                )
        return total_bytes

    def get_kvcache_allocated_bytes_per_sequence(self, seq_len: int) -> float:
        """Return page-allocated KV bytes for the pinned Latest cache geometry."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProLatestConfig):
            return super().get_kvcache_allocated_bytes_per_sequence(seq_len)
        return cfg.compute_allocated_kv_cache_bytes(
            seq_len,
            bytes_per_element=common.GEMMQuantMode.bfloat16.value.memory,
        )

    def get_kvcache_peak_allocated_bytes_per_sequence(self, seq_len: int) -> float:
        """Return peak page allocation while decoding through ``seq_len``."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProLatestConfig):
            return super().get_kvcache_peak_allocated_bytes_per_sequence(seq_len)
        return cfg.compute_peak_allocated_kv_cache_bytes(
            seq_len,
            bytes_per_element=common.GEMMQuantMode.bfloat16.value.memory,
        )

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Invert the Pro peak-allocation curve without assuming a constant slope."""
        if not isinstance(
            self.extra_params,
            (common.Step4ProConfig, common.Step4ProMQAConfig, common.Step4ProLatestConfig),
        ):
            return super().get_kvcache_max_tokens(kv_budget_bytes)
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)

    def _layer_counts(self) -> tuple[int, int, int]:
        cfg = self.extra_params
        if isinstance(cfg, common.Step4ProMQAConfig):
            return (
                sum(layer.ffn_type == "dense" for layer in cfg.layers),
                sum(layer.attention_type == "full" for layer in cfg.layers),
                sum(layer.attention_type == "nonfull" for layer in cfg.layers),
            )
        if isinstance(cfg, common.Step4ProConfig):
            dense_count = sum(layer.ffn_type == "dense" for layer in cfg.layers)
            full_count = sum(layer.attention_type == "full" for layer in cfg.layers)
            nonfull_count = sum(layer.attention_type == "nonfull" for layer in cfg.layers)
            return dense_count, full_count, nonfull_count
        dense_count = cfg.block_types.count("dense_swa")
        full_count = cfg.block_types.count("moe_full")
        swa_count = dense_count + cfg.block_types.count("moe_swa")
        return dense_count, full_count, swa_count

    def _moe_layer_count(self) -> int:
        """Return the MoE count from the active schema's single layer source of truth."""
        cfg = self.extra_params
        if isinstance(cfg, common.Step4ProMQAConfig):
            return sum(layer.ffn_type == "moe" for layer in cfg.layers)
        if isinstance(cfg, common.Step4ProConfig):
            return sum(layer.ffn_type == "moe" for layer in cfg.layers)
        return self._num_layers - cfg.block_types.count("dense_swa")

    def _mqa_attention_ops(self, phase: str) -> list:
        """Build V3/V4 MQA attention in the exact per-layer SSSF order."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProMQAConfig):
            raise TypeError("MQA attention builder requires Step4ProMQAConfig")
        tp = self.config.tp_size
        scale = 1.0 if phase == "context" else self._mtp_scale_factor
        operations = []
        for layer in cfg.layers:
            attention_cfg = cfg.full_attention if layer.attention_type == "full" else cfg.nonfull_attention
            local_q = attention_cfg.num_query_heads // tp
            local_kv = attention_cfg.num_kv_heads // tp
            prefix = f"{phase}_layer_{layer.layer_id:03d}_{layer.attention_type}_mqa"
            if phase == "context":
                attention = ops.ContextAttention(
                    f"{prefix}_attention",
                    scale,
                    local_q,
                    local_kv,
                    self.config.kvcache_quant_mode,
                    self.config.fmha_quant_mode,
                    window_size=attention_cfg.window_size,
                    head_size=attention_cfg.head_dim,
                )
            elif phase == "generation":
                attention = ops.GenerationAttention(
                    f"{prefix}_attention",
                    scale,
                    local_q,
                    local_kv,
                    self.config.kvcache_quant_mode,
                    window_size=attention_cfg.window_size,
                    head_size=attention_cfg.head_dim,
                )
            else:
                raise ValueError(f"Unsupported Step4-Pro attention phase: {phase!r}")
            hidden = self._hidden_size
            q_width = local_q * attention_cfg.head_dim
            kv_width = local_kv * attention_cfg.head_dim
            operations.extend(
                (
                    ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * hidden, 2 * hidden, 0.8),
                    ops.GEMM(f"{prefix}_q_proj_gemm", scale, q_width, hidden, self.config.gemm_quant_mode),
                    ops.GEMM(f"{prefix}_k_proj_gemm", scale, kv_width, hidden, self.config.gemm_quant_mode),
                    ops.GEMM(f"{prefix}_v_proj_gemm", scale, kv_width, hidden, self.config.gemm_quant_mode),
                    attention,
                    ops.GEMM(
                        f"{prefix}_o_proj_gemm",
                        scale,
                        hidden,
                        q_width,
                        self.config.gemm_quant_mode,
                        low_precision_input=True,
                    ),
                    ops.CustomAllReduce(f"{prefix}_attention_ar", scale, hidden, tp),
                )
            )
        return operations

    def _context_attention_ops(self, label: str, count: int) -> list:
        cfg = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        prefix = f"context_{label}_mla_approx"
        downscale_output = cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim
        q_b_output = self._num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        kv_b_output = self._num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
        return [
            ops.ElementWise(f"{prefix}_attn_norm", count, 2 * h, 2 * h, 0.8),
            ops.GEMM(f"{prefix}_downscale_gemm", count, downscale_output, h, self.config.gemm_quant_mode),
            ops.GEMM(
                f"{prefix}_q_b_proj_gemm",
                count,
                q_b_output // tp,
                cfg.q_lora_rank,
                self.config.gemm_quant_mode,
            ),
            ops.GEMM(
                f"{prefix}_kv_b_proj_gemm",
                count,
                kv_b_output // tp,
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
        downscale_output = cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim
        q_b_output = self._num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        bmm_quant_mode = (
            common.GEMMQuantMode.bfloat16
            if self.config.gemm_quant_mode == common.GEMMQuantMode.bfloat16
            else common.GEMMQuantMode.fp8
        )
        return [
            ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.GEMM(f"{prefix}_downscale_gemm", scale, downscale_output, h, self.config.gemm_quant_mode),
            ops.GEMM(
                f"{prefix}_q_b_proj_gemm",
                scale,
                q_b_output // tp,
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

    def _pro_full_attention_ops(self, phase: str, layer_id: int, scale: float) -> list:
        """Build one independently parameterized standard-MHA layer."""
        model_cfg = self.extra_params
        if not isinstance(model_cfg, common.Step4ProConfig):
            raise TypeError("Step4-Pro full-attention builder requires Step4ProConfig")
        cfg = model_cfg.full_attention
        h = self._hidden_size
        tp = self.config.tp_size
        if isinstance(cfg, common.Step4MFAAttentionConfig):
            local_query_heads = cfg.num_query_heads // tp
            local_kv_heads = cfg.output_groups // tp
            prefix = f"{phase}_layer_{layer_id:03d}_full"
            if phase == "context":
                attention = ops.ContextAttention(
                    f"{prefix}_attention",
                    scale,
                    local_query_heads,
                    local_kv_heads,
                    self.config.kvcache_quant_mode,
                    self.config.fmha_quant_mode,
                    window_size=0,
                    head_size=cfg.projection_head_dim,
                )
            elif phase == "generation":
                attention = ops.GenerationAttention(
                    f"{prefix}_attention",
                    scale,
                    local_query_heads,
                    local_kv_heads,
                    self.config.kvcache_quant_mode,
                    window_size=0,
                    head_size=cfg.projection_head_dim,
                )
            else:
                raise ValueError(f"Unsupported Step4-Pro attention phase: {phase!r}")
            q_width = local_query_heads * cfg.projection_head_dim
            kv_width = local_kv_heads * cfg.projection_head_dim
            return [
                ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
                ops.GEMM(f"{prefix}_q_proj_gemm", scale, q_width, h, self.config.gemm_quant_mode),
                ops.GEMM(f"{prefix}_k_proj_gemm", scale, kv_width, h, self.config.gemm_quant_mode),
                ops.GEMM(f"{prefix}_v_proj_gemm", scale, kv_width, h, self.config.gemm_quant_mode),
                attention,
                ops.GEMM(
                    f"{prefix}_o_proj_gemm", scale, h, q_width, self.config.gemm_quant_mode, low_precision_input=True
                ),
                ops.CustomAllReduce(f"{prefix}_attention_ar", scale, h, tp),
            ]
        local_query_heads = cfg.num_query_heads // tp
        local_kv_heads = cfg.num_kv_heads // tp
        prefix = f"{phase}_layer_{layer_id:03d}_full"

        if phase == "context":
            attention = ops.ContextAttention(
                f"{prefix}_attention",
                scale,
                local_query_heads,
                local_kv_heads,
                self.config.kvcache_quant_mode,
                self.config.fmha_quant_mode,
                window_size=0,
                head_size=cfg.q_head_dim,
            )
        elif phase == "generation":
            attention = ops.GenerationAttention(
                f"{prefix}_attention",
                scale,
                local_query_heads,
                local_kv_heads,
                self.config.kvcache_quant_mode,
                window_size=0,
                head_size=cfg.q_head_dim,
            )
        else:
            raise ValueError(f"Unsupported Step4-Pro attention phase: {phase!r}")

        return [
            ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.GEMM(
                f"{prefix}_q_proj_gemm",
                scale,
                local_query_heads * cfg.q_head_dim,
                h,
                self.config.gemm_quant_mode,
            ),
            ops.GEMM(
                f"{prefix}_k_proj_gemm",
                scale,
                local_kv_heads * cfg.k_head_dim,
                h,
                self.config.gemm_quant_mode,
            ),
            ops.GEMM(
                f"{prefix}_v_proj_gemm",
                scale,
                local_kv_heads * cfg.v_head_dim,
                h,
                self.config.gemm_quant_mode,
            ),
            attention,
            ops.GEMM(
                f"{prefix}_o_proj_gemm",
                scale,
                h,
                local_query_heads * cfg.v_head_dim,
                self.config.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.CustomAllReduce(f"{prefix}_attention_ar", scale, h, tp),
        ]

    def _pro_nonfull_attention_ops(self, phase: str, layer_id: int, scale: float) -> list:
        """Build one DeepSeek-V4-style SWA/HCA layer plus its explicit reduction."""
        model_cfg = self.extra_params
        if not isinstance(model_cfg, common.Step4ProConfig):
            raise TypeError("Step4-Pro non-full-attention builder requires Step4ProConfig")
        cfg = model_cfg.nonfull_attention
        h = self._hidden_size
        tp = self.config.tp_size
        prefix = f"{phase}_layer_{layer_id:03d}_nonfull"
        if isinstance(cfg, common.Step4MFAAttentionConfig):
            local_heads = cfg.num_query_heads // tp
            if phase == "context":
                attention = ops.ContextMLA(
                    f"{prefix}_hca_attention",
                    scale,
                    local_heads,
                    self.config.kvcache_quant_mode,
                    self.config.fmha_quant_mode,
                )
            elif phase == "generation":
                attention = ops.GenerationMLA(
                    f"{prefix}_hca_attention",
                    scale,
                    local_heads,
                    self.config.kvcache_quant_mode,
                )
            else:
                raise ValueError(f"Unsupported Step4-Pro attention phase: {phase!r}")
            return [
                ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
                attention,
                ops.CustomAllReduce(f"{prefix}_attention_ar", scale, h, tp),
            ]
        attention_name = f"{prefix}_{cfg.mechanism}_attention"
        attention_args = (
            attention_name,
            scale,
            cfg.num_query_heads // tp,
            cfg.num_query_heads,
            tp,
            h,
            cfg.q_lora_rank,
            cfg.o_lora_rank,
            cfg.head_dim,
            cfg.rope_dimension,
            cfg.index_n_heads,
            cfg.index_head_dim,
            cfg.index_topk,
            cfg.window_size,
            cfg.compression_ratio,
            cfg.o_groups // tp,
            self.config.kvcache_quant_mode,
            self.config.fmha_quant_mode,
            self.config.gemm_quant_mode,
        )
        if phase == "context":
            attention = ops.ContextDeepSeekV4AttentionModule(*attention_args, cp_size=1)
        elif phase == "generation":
            attention = ops.GenerationDeepSeekV4AttentionModule(*attention_args, cp_size=1)
        else:
            raise ValueError(f"Unsupported Step4-Pro attention phase: {phase!r}")

        return [
            ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
            attention,
            ops.CustomAllReduce(f"{prefix}_attention_ar", scale, h, tp),
        ]

    def _pro_attention_ops(self, phase: str) -> tuple[list, tuple[str, ...]]:
        """Build all Pro attention layers in authoritative layer order."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProConfig):
            raise TypeError("Step4-Pro attention builder requires Step4ProConfig")
        scale = 1.0 if phase == "context" else self._mtp_scale_factor
        operations = []
        for layer in cfg.layers:
            if layer.attention_type == "full":
                operations.extend(self._pro_full_attention_ops(phase, layer.layer_id, scale))
            elif layer.attention_type == "nonfull":
                operations.extend(self._pro_nonfull_attention_ops(phase, layer.layer_id, scale))
            else:
                raise ValueError(
                    f"Unsupported Step4-Pro attention_type {layer.attention_type!r} at layer {layer.layer_id}"
                )
        return operations, tuple(operation._name for operation in operations)

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
        result = []
        is_latent_moe = isinstance(self.extra_params, common.Step4ProMQAConfig) and self.extra_params.latent_moe_dim > 0
        routed_hidden = self.extra_params.latent_moe_dim if is_latent_moe else h
        if is_latent_moe:
            latent = self.extra_params.latent_moe_dim
            result.append(
                ops.GEMM(
                    "context_latent_moe_down_proj",
                    count,
                    latent // self.config.tp_size,
                    h,
                    self.config.gemm_quant_mode,
                )
            )
        result.extend(
            [
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
                    routed_hidden,
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
                    routed_hidden,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp,
                    moe_ep,
                    self.config.moe_quant_mode,
                    _STEP4_PRO_MOE_WORKLOAD_DISTRIBUTION,
                    attention_dp,
                    is_context=True,
                    is_gated=True,
                ),
                ops.MoEDispatch(
                    "context_moe_post_dispatch",
                    count,
                    routed_hidden,
                    self._topk,
                    self._num_experts,
                    moe_tp,
                    moe_ep,
                    attention_dp,
                    False,
                    quant_mode=self.config.moe_quant_mode,
                ),
            ]
        )
        if is_latent_moe:
            latent = self.extra_params.latent_moe_dim
            result.append(
                ops.GEMM(
                    "context_latent_moe_up_proj",
                    count,
                    h,
                    latent // self.config.tp_size,
                    self.config.gemm_quant_mode,
                )
            )
        result.extend(self._shared_ffn_ops("context", count))
        result.append(ops.ElementWise("context_moe_shared_merge", count, 2 * h, h, 0.8))
        return result

    def _generation_moe_ops(self, count: int) -> list:
        h = self._hidden_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attention_dp = self.config.attention_dp_size
        scale = count * self._mtp_scale_factor
        routed_ops = []
        is_latent_moe = isinstance(self.extra_params, common.Step4ProMQAConfig) and self.extra_params.latent_moe_dim > 0
        routed_hidden = self.extra_params.latent_moe_dim if is_latent_moe else h
        if is_latent_moe:
            latent = self.extra_params.latent_moe_dim
            routed_ops.append(
                ops.GEMM(
                    "generation_latent_moe_down_proj",
                    scale,
                    latent // self.config.tp_size,
                    h,
                    self.config.gemm_quant_mode,
                )
            )
        routed_ops.extend(
            [
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
                    routed_hidden,
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
                    routed_hidden,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp,
                    moe_ep,
                    self.config.moe_quant_mode,
                    _STEP4_PRO_MOE_WORKLOAD_DISTRIBUTION,
                    attention_dp,
                    is_context=False,
                    is_gated=True,
                ),
                ops.MoEDispatch(
                    "generation_moe_post_dispatch",
                    scale,
                    routed_hidden,
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
        )
        if is_latent_moe:
            latent = self.extra_params.latent_moe_dim
            routed_ops.append(
                ops.GEMM(
                    "generation_latent_moe_up_proj",
                    scale,
                    h,
                    latent // self.config.tp_size,
                    self.config.gemm_quant_mode,
                )
            )
        shared_ops = self._shared_ffn_ops("generation", scale)
        return [
            ops.ElementWise("generation_moe_ffn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.OverlapOp("generation_moe_overlap", group_a=routed_ops, group_b=shared_ops),
            ops.ElementWise("generation_moe_shared_merge", scale, 2 * h, h, 0.8),
        ]

    def _latest_attention_ops(self, phase: str, layer: common.Step4LayerSpec, scale: float) -> list:
        """Build one pinned Latest attention block with provider metadata."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProLatestConfig):
            raise TypeError("Latest attention builder requires Step4ProLatestConfig.")
        h = self._hidden_size
        prefix = f"{phase}_layer_{layer.layer_id:03d}_{layer.attention_type}"

        if layer.attention_type == "full":
            if phase == "context":
                attention = ops.ContextAttention(
                    f"{prefix}_attention",
                    scale,
                    cfg.full_num_query_heads,
                    cfg.full_num_kv_heads,
                    self.config.kvcache_quant_mode,
                    self.config.fmha_quant_mode,
                    window_size=cfg.full_window_size,
                    head_size=cfg.full_head_dim,
                    provider="optimus_fa4",
                    kv_storage_alias=True,
                    page_size=cfg.full_page_size,
                    physical_page_bytes=cfg.full_physical_page_bytes,
                    kv_block_stride_bytes=cfg.full_kv_block_stride_bytes,
                    kv_cache_layout=cfg.kv_cache_layout,
                )
            elif phase == "generation":
                attention = ops.GenerationAttention(
                    f"{prefix}_attention",
                    scale,
                    cfg.full_num_query_heads,
                    cfg.full_num_kv_heads,
                    self.config.kvcache_quant_mode,
                    window_size=cfg.full_window_size,
                    head_size=cfg.full_head_dim,
                    provider="optimus_fa4",
                    kv_storage_alias=True,
                    page_size=cfg.full_page_size,
                    physical_page_bytes=cfg.full_physical_page_bytes,
                    kv_block_stride_bytes=cfg.full_kv_block_stride_bytes,
                    kv_cache_layout=cfg.kv_cache_layout,
                )
            else:
                raise ValueError(f"Unsupported Latest attention phase: {phase!r}")

            return [
                ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
                ops.GEMM(f"{prefix}_wq_a_gemm", scale, cfg.full_q_lora_rank, h, self.config.gemm_quant_mode),
                ops.ElementWise(
                    f"{prefix}_q_lora_norm",
                    scale,
                    cfg.full_q_lora_rank,
                    cfg.full_q_lora_rank,
                    0.8,
                ),
                ops.GEMM(
                    f"{prefix}_wq_b_gemm",
                    scale,
                    cfg.full_num_query_heads * cfg.full_head_dim,
                    cfg.full_q_lora_rank,
                    self.config.gemm_quant_mode,
                ),
                ops.ElementWise(
                    f"{prefix}_q_norm",
                    scale,
                    cfg.full_num_query_heads * cfg.full_head_dim,
                    cfg.full_num_query_heads * cfg.full_head_dim,
                    0.8,
                ),
                ops.GEMM(
                    f"{prefix}_wkv_gemm",
                    scale,
                    cfg.full_num_kv_heads * cfg.full_head_dim,
                    h,
                    self.config.gemm_quant_mode,
                ),
                ops.QKVNormRoPE(
                    f"{prefix}_k_norm_rope",
                    scale,
                    normalized_tensors=("k",),
                    provider="vllm_step4pro_k_norm_rope",
                    q_heads=cfg.full_num_query_heads,
                    kv_heads=cfg.full_num_kv_heads,
                    head_dim=cfg.full_head_dim,
                ),
                attention,
                ops.ElementWise(
                    f"{prefix}_inverse_rope",
                    scale,
                    cfg.full_num_query_heads * cfg.full_head_dim,
                    cfg.full_num_query_heads * cfg.full_head_dim,
                    0.8,
                ),
                ops.GEMM(
                    f"{prefix}_head_gate_gemm",
                    scale,
                    cfg.full_num_query_heads,
                    h,
                    self.config.gemm_quant_mode,
                ),
                ops.ElementWise(
                    f"{prefix}_head_gate_sigmoid_mul",
                    scale,
                    cfg.full_num_query_heads,
                    cfg.full_num_query_heads,
                    0.8,
                ),
                ops.GroupedGEMM(
                    f"{prefix}_wo_a_grouped_gemm",
                    scale,
                    cfg.full_o_lora_rank,
                    cfg.full_num_query_heads * cfg.full_head_dim // cfg.full_output_groups,
                    self.config.gemm_quant_mode,
                    groups=cfg.full_output_groups,
                    provider="vllm_step4pro_torch_einsum",
                ),
                ops.GEMM(
                    f"{prefix}_wo_b_gemm",
                    scale,
                    h,
                    cfg.full_output_groups * cfg.full_o_lora_rank,
                    self.config.gemm_quant_mode,
                ),
                ops.ElementWise(
                    f"{phase}_layer_{layer.layer_id:03d}_attention_residual_add",
                    scale,
                    2 * h,
                    h,
                    0.8,
                ),
            ]

        if phase == "context":
            attention = ops.ContextAttention(
                f"{prefix}_attention",
                scale,
                cfg.swa_num_query_heads,
                cfg.swa_num_kv_heads,
                self.config.kvcache_quant_mode,
                self.config.fmha_quant_mode,
                window_size=cfg.swa_window_size,
                head_size=cfg.swa_head_dim,
                provider="vllm_native_sliding_gqa",
                page_size=cfg.swa_page_size,
                physical_page_bytes=cfg.swa_physical_page_bytes,
                kv_block_stride_bytes=cfg.swa_kv_block_stride_bytes,
                kv_cache_layout=cfg.kv_cache_layout,
            )
        elif phase == "generation":
            attention = ops.GenerationAttention(
                f"{prefix}_attention",
                scale,
                cfg.swa_num_query_heads,
                cfg.swa_num_kv_heads,
                self.config.kvcache_quant_mode,
                window_size=cfg.swa_window_size,
                head_size=cfg.swa_head_dim,
                provider="vllm_native_sliding_gqa",
                page_size=cfg.swa_page_size,
                physical_page_bytes=cfg.swa_physical_page_bytes,
                kv_block_stride_bytes=cfg.swa_kv_block_stride_bytes,
                kv_cache_layout=cfg.kv_cache_layout,
            )
        else:
            raise ValueError(f"Unsupported Latest attention phase: {phase!r}")

        q_width = cfg.swa_num_query_heads * cfg.swa_head_dim
        kv_width = cfg.swa_num_kv_heads * cfg.swa_head_dim
        return [
            ops.ElementWise(f"{prefix}_attn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.GEMM(
                f"{prefix}_qkv_proj_gemm",
                scale,
                q_width + 2 * kv_width,
                h,
                self.config.gemm_quant_mode,
            ),
            ops.QKVNormRoPE(
                f"{prefix}_qkv_norm_rope",
                scale,
                normalized_tensors=("q", "k", "v"),
                provider="vllm_step4pro_qkv_norm_rope",
                q_heads=cfg.swa_num_query_heads,
                kv_heads=cfg.swa_num_kv_heads,
                head_dim=cfg.swa_head_dim,
            ),
            attention,
            ops.GEMM(
                f"{prefix}_head_gate_gemm",
                scale,
                cfg.swa_num_query_heads,
                h,
                self.config.gemm_quant_mode,
            ),
            ops.ElementWise(
                f"{prefix}_head_gate_sigmoid_mul",
                scale,
                cfg.swa_num_query_heads,
                cfg.swa_num_query_heads,
                0.8,
            ),
            ops.GEMM(
                f"{prefix}_o_proj_gemm",
                scale,
                h,
                q_width,
                self.config.gemm_quant_mode,
            ),
            ops.ElementWise(
                f"{phase}_layer_{layer.layer_id:03d}_attention_residual_add",
                scale,
                2 * h,
                h,
                0.8,
            ),
        ]

    def _latest_dense_ffn_ops(self, phase: str, layer_id: int, scale: float) -> list:
        """Build one pinned dense SiTU-GLU block."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProLatestConfig):
            raise TypeError("Latest dense FFN builder requires Step4ProLatestConfig.")
        h = self._hidden_size
        inter = cfg.dense_inter_size
        prefix = f"{phase}_layer_{layer_id:03d}_dense"
        return [
            ops.ElementWise(f"{prefix}_ffn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.GEMM(f"{prefix}_gate_up_gemm", scale, 2 * inter, h, self.config.gemm_quant_mode),
            ops.ElementWise(f"{prefix}_situ_glu", scale, 2 * inter, inter, 0.8),
            ops.GEMM(
                f"{prefix}_down_gemm",
                scale,
                h,
                inter,
                self.config.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.ElementWise(f"{prefix}_ffn_residual_add", scale, 2 * h, h, 0.8),
        ]

    def _latest_latent_moe_ops(self, phase: str, layer_id: int, scale: float) -> list:
        """Build one serial Latest latent-MoE block."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProLatestConfig):
            raise TypeError("Latest latent MoE builder requires Step4ProLatestConfig.")
        h = self._hidden_size
        latent = cfg.latent_moe_dim
        shared = cfg.shared_expert_inter_size
        prefix = f"{phase}_layer_{layer_id:03d}_latent_moe"
        is_context = phase == "context"

        experts = ops.MoE(
            f"{prefix}_experts",
            scale,
            latent,
            self._moe_inter_size,
            self._topk,
            self._num_experts,
            self.config.moe_tp_size,
            self.config.moe_ep_size,
            common.MoEQuantMode.fp8_block,
            _STEP4_PRO_MOE_WORKLOAD_DISTRIBUTION,
            self.config.attention_dp_size,
            is_context=is_context,
            is_gated=True,
            provider="optimus_fp8_moe",
        )

        dispatch_args = (
            latent,
            self._topk,
            self._num_experts,
            self.config.moe_tp_size,
            self.config.moe_ep_size,
            self.config.attention_dp_size,
        )
        return [
            ops.ElementWise(f"{prefix}_ffn_norm", scale, 2 * h, 2 * h, 0.8),
            ops.FP32OutputGEMM(f"{prefix}_router_gemm", scale, self._num_experts, h),
            ops.GEMM(f"{prefix}_pre_proj", scale, latent, h, common.GEMMQuantMode.bfloat16),
            ops.MoEDispatch(
                f"{prefix}_dispatch",
                scale,
                *dispatch_args,
                True,
                quant_mode=common.MoEQuantMode.fp8_block,
                reduce_results=False,
                is_context=is_context,
                provider="vllm_deepep_high_throughput",
                operation="dispatch",
            ),
            experts,
            ops.MoEDispatch(
                f"{prefix}_combine",
                scale,
                *dispatch_args,
                False,
                quant_mode=common.MoEQuantMode.fp8_block,
                is_context=is_context,
                provider="vllm_deepep_high_throughput",
                operation="combine",
            ),
            ops.GEMM(
                f"{prefix}_shared_gate_up_gemm",
                scale,
                2 * shared,
                h,
                self.config.gemm_quant_mode,
            ),
            ops.ElementWise(f"{prefix}_shared_situ_glu", scale, 2 * shared, shared, 0.8),
            ops.GEMM(
                f"{prefix}_shared_down_gemm",
                scale,
                h,
                shared,
                self.config.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.ElementWise(f"{prefix}_post_proj_norm", scale, latent, latent, 0.8),
            ops.GEMM(f"{prefix}_post_proj", scale, h, latent, common.GEMMQuantMode.bfloat16),
            ops.ElementWise(f"{prefix}_shared_scale_mul", scale, h, h, 0.8),
            ops.ElementWise(f"{prefix}_routed_shared_add", scale, 2 * h, h, 0.8),
            ops.ElementWise(f"{prefix}_ffn_residual_add", scale, 2 * h, h, 0.8),
        ]

    def _build_latest_ops(self, phase: str) -> list:
        """Build the complete Latest graph in decoder-layer execution order."""
        cfg = self.extra_params
        if not isinstance(cfg, common.Step4ProLatestConfig):
            raise TypeError("Latest graph builder requires Step4ProLatestConfig.")
        h = self._hidden_size
        scale = 1.0 if phase == "context" else self._mtp_scale_factor
        result = [ops.Embedding(f"{phase}_embedding", scale, self._vocab_size, h, 0.3)]
        for layer in cfg.layers:
            result.extend(self._latest_attention_ops(phase, layer, scale))
            if layer.ffn_type == "dense":
                result.extend(self._latest_dense_ffn_ops(phase, layer.layer_id, scale))
            elif layer.ffn_type == "latent_moe":
                result.extend(self._latest_latent_moe_ops(phase, layer.layer_id, scale))
            else:
                raise ValueError(f"Unsupported Latest ffn_type {layer.ffn_type!r}.")
        result.extend(
            [
                ops.ElementWise(f"{phase}_final_norm", scale, h, h, 0.8),
                ops.GEMM(
                    f"{phase}_logits_gemm",
                    scale,
                    self._vocab_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                ),
            ]
        )
        return result

    def _build_context_ops(self) -> None:
        if isinstance(self.extra_params, common.Step4ProLatestConfig):
            self.context_ops = self._build_latest_ops("context")
            return
        dense_count, full_count, nonfull_count = self._layer_counts()
        moe_count = self._moe_layer_count()
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size

        self.context_ops = [
            ops.Embedding("context_embedding", 1, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("context_embedding_ar", 1, h, tp),
        ]
        if isinstance(self.extra_params, common.Step4ProMQAConfig):
            mqa_attention_ops = self._mqa_attention_ops("context")
            self.MIXED_STEP_CONTEXT_ATTENTION_KEYS = tuple(operation._name for operation in mqa_attention_ops)
            self.context_ops.extend(mqa_attention_ops)
        elif isinstance(self.extra_params, common.Step4ProConfig):
            attention_ops, attention_names = self._pro_attention_ops("context")
            self.MIXED_STEP_CONTEXT_ATTENTION_KEYS = attention_names
            self.context_ops.extend(attention_ops)
        else:
            self.context_ops.extend(self._context_attention_ops("full", full_count))
            self.context_ops.extend(self._context_attention_ops("swa", nonfull_count))
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
        if isinstance(self.extra_params, common.Step4ProLatestConfig):
            self.generation_ops = self._build_latest_ops("generation")
            return
        dense_count, full_count, nonfull_count = self._layer_counts()
        moe_count = self._moe_layer_count()
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        scale = self._mtp_scale_factor

        self.generation_ops = [
            ops.Embedding("generation_embedding", scale, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("generation_embedding_ar", scale, h, tp),
        ]
        if isinstance(self.extra_params, common.Step4ProMQAConfig):
            mqa_attention_ops = self._mqa_attention_ops("generation")
            self.MIXED_STEP_GENERATION_ATTENTION_KEYS = tuple(operation._name for operation in mqa_attention_ops)
            self.generation_ops.extend(mqa_attention_ops)
        elif isinstance(self.extra_params, common.Step4ProConfig):
            attention_ops, attention_names = self._pro_attention_ops("generation")
            self.MIXED_STEP_GENERATION_ATTENTION_KEYS = attention_names
            self.generation_ops.extend(attention_ops)
        else:
            self.generation_ops.extend(self._generation_attention_ops("full", full_count))
            self.generation_ops.extend(self._generation_attention_ops("swa", nonfull_count))
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
