# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-sensitive operation identities used by Step4-Pro-Latest."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


class GroupedGEMM(Operation):
    """A grouped matrix multiplication with one shape shared by each group."""

    _CP_AWARE = True

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        *,
        groups: int,
        provider: str,
    ) -> None:
        if groups < 1:
            raise ValueError("GroupedGEMM requires groups >= 1.")
        if n < 1 or k < 1:
            raise ValueError("GroupedGEMM requires positive per-group dimensions.")
        if not provider:
            raise ValueError("GroupedGEMM requires a provider identity.")

        super().__init__(name, scale_factor)
        self._n = n
        self._k = k
        self._quant_mode = quant_mode
        self._groups = groups
        self._provider = provider
        self._weights = self._groups * self._n * self._k * quant_mode.value.memory

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Reject dense-table queries until grouped data has an explicit consumer."""
        raise NotImplementedError("GroupedGEMM requires a provider-specific grouped-GEMM performance dataset.")

    def _persisted_key(self) -> tuple:
        """Return the physical identity required by the grouped-GEMM dataset."""
        return (self._provider, self._groups, self._n, self._k, self._quant_mode)

    def get_weights(self, **kwargs) -> float:
        return self._weights * self._scale_factor


class FP32OutputGEMM(GEMM):
    """GEMM with BF16 weights and an explicitly FP32 output contract."""

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        k: int,
        *,
        weight_dtype: str = "bfloat16",
        output_dtype: str = "float32",
        provider: str = "vllm.optimus_matmul_fp32",
    ) -> None:
        if not provider:
            raise ValueError("FP32OutputGEMM requires a provider identity.")
        if weight_dtype != "bfloat16":
            raise ValueError("FP32OutputGEMM requires bfloat16 weights.")
        if output_dtype != "float32":
            raise ValueError("FP32OutputGEMM requires float32 output.")

        super().__init__(name, scale_factor, n, k, common.GEMMQuantMode.bfloat16)
        self._weight_dtype = weight_dtype
        self._output_dtype = output_dtype
        self._provider = provider

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Reject generic GEMM rows because they omit FP32 output and provider."""
        raise NotImplementedError("FP32OutputGEMM requires a provider-specific FP32-output GEMM performance dataset.")

    def _persisted_key(self) -> tuple:
        """Return the physical identity required by the FP32-output dataset."""
        return (self._provider, self._n, self._k, self._weight_dtype, self._output_dtype)


class QKVNormRoPE(Operation):
    """Metadata-only identity for Q/K/V normalization followed by RoPE."""

    _CP_AWARE = True

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        normalized_tensors: tuple[str, ...],
        provider: str,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        normalized_tensors = tuple(normalized_tensors)
        if not normalized_tensors:
            raise ValueError("QKVNormRoPE requires at least one normalized tensor.")
        if not provider:
            raise ValueError("QKVNormRoPE requires a provider identity.")
        if min(q_heads, kv_heads, head_dim) < 1:
            raise ValueError("QKVNormRoPE requires positive q_heads, kv_heads, and head_dim.")

        super().__init__(name, scale_factor)
        self._normalized_tensors = normalized_tensors
        self._provider = provider
        self._q_heads = q_heads
        self._kv_heads = kv_heads
        self._head_dim = head_dim

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Reject generic memory-op substitution for this provider-specific identity."""
        raise NotImplementedError("QKVNormRoPE requires a provider-specific normalization/RoPE dataset.")

    def get_weights(self, **kwargs) -> float:
        return 0.0

    def _persisted_key(self) -> tuple:
        """Return the physical identity required by the fused QKV dataset."""
        return (
            self._provider,
            self._normalized_tensors,
            self._q_heads,
            self._kv_heads,
            self._head_dim,
        )
