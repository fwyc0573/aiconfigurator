# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-sensitive operation identities used by Step4-Pro-Latest."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations.base import Operation, _read_filtered_rows
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


class GroupedGEMM(Operation):
    """A grouped matrix multiplication with one shape shared by each group."""

    _data_cache: ClassVar[dict] = {}
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

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return (
            database.systems_root,
            database.system,
            database.backend,
            database.version,
            database.enable_shared_layer,
        )

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.step4_grouped_gemm.value)
            sources = database._build_op_sources(
                PerfDataFilename.step4_grouped_gemm,
                primary_path,
                system_data_root,
            )
            cls._data_cache[key] = LoadedOpData(
                load_step4_grouped_gemm_data(sources),
                PerfDataFilename.step4_grouped_gemm,
                primary_path,
            )
            cls._record_load()

        if "_step4_grouped_gemm_data" not in database.__dict__:
            database._step4_grouped_gemm_data = cls._data_cache[key]

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query only the exact provider/shape slice and interpolate tokens."""
        from aiconfigurator.sdk import interpolation

        num_tokens = kwargs.get("x")
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens <= 0:
            raise ValueError("GroupedGEMM requires a positive num_tokens integer.")
        num_tokens = -(-num_tokens // self._seq_split)

        self.load_data(database)
        data = database._step4_grouped_gemm_data
        data.raise_if_not_loaded()
        structural_key = self._persisted_key()
        if structural_key not in data:
            raise PerfDataNotAvailableError(
                "Grouped-GEMM perf data not available for exact provider/shape key. "
                f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                f"provider='{self._provider}', groups={self._groups}, n={self._n}, k={self._k}, "
                f"quant_mode='{self._quant_mode.name}'."
            )

        token_data = data[structural_key]
        if num_tokens in token_data:
            result = token_data[num_tokens]
        else:
            token_points = sorted(token_data)
            try:
                left, right = interpolation.nearest_1d_point_helper(num_tokens, token_points)
                result = interpolation.interp_1d(
                    [left, right],
                    [token_data[left], token_data[right]],
                    num_tokens,
                )
            except interpolation.InterpolationDataNotAvailableError as exc:
                raise PerfDataNotAvailableError(
                    "Grouped-GEMM perf data does not bracket the requested token count. "
                    f"provider='{self._provider}', groups={self._groups}, n={self._n}, k={self._k}, "
                    f"quant_mode='{self._quant_mode.name}', num_tokens={num_tokens}, "
                    f"available_tokens={token_points}."
                ) from exc

        return PerformanceResult(
            latency=float(result["latency"]) * self._scale_factor,
            energy=float(result.get("energy", 0.0)) * self._scale_factor,
            source="silicon",
        )

    def _persisted_key(self) -> tuple:
        """Return the physical identity required by the grouped-GEMM dataset."""
        return (self._provider, self._groups, self._n, self._k, self._quant_mode)

    def get_weights(self, **kwargs) -> float:
        return self._weights * self._scale_factor


def load_step4_grouped_gemm_data(perf_file):
    """Load exact Step4 grouped-einsum rows keyed by structure then tokens."""
    rows = _read_filtered_rows(perf_file)
    if rows is None:
        return None

    grouped_data: dict[tuple, dict[int, dict[str, float]]] = {}
    for row in rows:
        provider = str(row["provider"])
        kernel_source = str(row["kernel_source"])
        op_name = str(row["op_name"])
        if op_name != "step4_grouped_gemm":
            raise ValueError(f"unexpected op_name in grouped-GEMM data: {op_name!r}")
        if kernel_source != provider:
            raise ValueError(
                "grouped-GEMM provider does not match kernel_source: "
                f"provider={provider!r}, kernel_source={kernel_source!r}"
            )

        structural_key = (
            provider,
            int(row["groups"]),
            int(row["n"]),
            int(row["k"]),
            common.GEMMQuantMode[str(row["quant_mode"])],
        )
        num_tokens = int(row["num_tokens"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0) or 0.0)
        value = {
            "latency": latency,
            "power": power,
            "energy": power * latency,
        }

        token_data = grouped_data.setdefault(structural_key, {})
        existing = token_data.get(num_tokens)
        if existing is not None and existing != value:
            raise ValueError(
                "conflicting grouped-GEMM row for physical key "
                f"{(*structural_key, num_tokens)!r}: {existing!r} != {value!r}"
            )
        token_data[num_tokens] = value

    return grouped_data


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
