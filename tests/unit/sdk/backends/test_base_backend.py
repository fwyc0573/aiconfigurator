# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig

pytestmark = pytest.mark.unit


class _LatencyResult:
    def __init__(self, latency_ms: float, energy_wms: float) -> None:
        self._latency_ms = latency_ms
        self.energy = energy_wms

    def __float__(self) -> float:
        return self._latency_ms


class _StaticOp:
    def __init__(self, name: str, latency_ms: float, energy_wms: float) -> None:
        self._name = name
        self._latency_ms = latency_ms
        self._energy_wms = energy_wms

    def query(self, *args, **kwargs) -> _LatencyResult:
        return _LatencyResult(self._latency_ms, self._energy_wms)


class _TestBackend(BaseBackend):
    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError

    def _get_memory_usage(
        self,
        model,
        database,
        batch_size,
        beam_width,
        isl,
        osl,
        num_tokens=0,
        prefix=0,
        encoder_memory=None,
    ) -> dict[str, float]:
        return {"total": 1.0}


@pytest.fixture
def backend() -> BaseBackend:
    return _TestBackend()


@pytest.fixture
def database():
    return SimpleNamespace(
        backend="test-backend",
        version="test-version",
        system="test-system",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
        get_default_database_mode=lambda: common.DatabaseMode.SOL,
    )


@pytest.fixture
def model():
    model = MagicMock()
    model.model_path = "test-model"
    model.model_name = "test-model"
    model._nextn = 0
    model.encoder_ops = []
    model.context_ops = [
        _StaticOp("context_attention", latency_ms=11.0, energy_wms=110.0),
        _StaticOp("logits_gemm", latency_ms=3.0, energy_wms=30.0),
    ]
    model.generation_ops = [
        _StaticOp("generation_attention", latency_ms=2.0, energy_wms=20.0),
        _StaticOp("generation_mlp", latency_ms=1.0, energy_wms=10.0),
    ]
    model.config = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )
    return model


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)


@pytest.mark.parametrize("mode", ["static", "static_ctx", "static_gen"])
@pytest.mark.parametrize("latency_correction_scale", [1.0, 1.25])
def test_run_static_latency_only_matches_run_static_latency(
    backend: BaseBackend,
    model,
    database,
    runtime_config: RuntimeConfig,
    mode: str,
    latency_correction_scale: float,
) -> None:
    summary = backend.run_static(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )
    latency_only = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )

    summary_latency = sum(summary.get_context_latency_dict().values()) + sum(
        summary.get_generation_latency_dict().values()
    )
    request_latency = float(summary.get_summary_df().iloc[0]["request_latency"])

    assert latency_only == pytest.approx(summary_latency)
    assert latency_only == pytest.approx(request_latency, abs=1e-3)


def test_run_static_capacity_context_uses_peak_physical_kv(
    backend: BaseBackend,
    model,
    database,
    runtime_config: RuntimeConfig,
) -> None:
    model.get_kvcache_peak_allocated_bytes_per_sequence.return_value = 12_345.0
    model.get_kvcache_bytes_per_sequence.side_effect = AssertionError(
        "Logical KV bytes must not drive memory-capacity reporting."
    )
    model._cp_kv_memory_divisor.return_value = 1

    summary = backend.run_static(
        model,
        database,
        runtime_config,
        mode="static",
        stride=2,
    )

    assert summary.get_kv_per_seq() == (12_345.0, runtime_config.isl + runtime_config.osl)


def test_memory_usage_uses_peak_physical_kv_allocation() -> None:
    class _WeightlessOp:
        @staticmethod
        def get_weights() -> float:
            return 0.0

    class _PhysicalKVModel:
        context_ops: ClassVar[list[_WeightlessOp]] = [_WeightlessOp()]
        config = SimpleNamespace(
            pp_size=1,
            tp_size=1,
            attention_dp_size=1,
            nextn=0,
        )
        _num_heads = 1
        _head_size = 1
        _num_experts = 0
        _topk = 0
        model_family = "test"

        @staticmethod
        def get_kvcache_bytes_per_sequence(seq_len: int) -> float:
            raise AssertionError("Logical KV bytes must not drive OOM checks.")

        @staticmethod
        def get_kvcache_allocated_bytes_per_sequence(seq_len: int) -> float:
            raise AssertionError("Point-in-time residency must not drive OOM checks.")

        @staticmethod
        def get_kvcache_peak_allocated_bytes_per_sequence(seq_len: int) -> float:
            assert seq_len == 10
            return 1_024.0

        @staticmethod
        def _cp_kv_memory_divisor() -> int:
            return 1

    database = SimpleNamespace(
        system_spec={
            "misc": {
                "nccl_mem": {1: 0.0},
                "other_mem": 0.0,
            }
        }
    )

    memory = BaseBackend()._get_memory_usage(
        _PhysicalKVModel(),
        database,
        batch_size=2,
        beam_width=1,
        isl=8,
        osl=2,
    )

    assert memory["kvcache"] == pytest.approx(2_048.0 / (1 << 30))


def test_run_static_can_route_to_rust_engine_step_backend(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    calls = []

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        calls.append((model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg))
        return (
            {"rust_engine_step_context": 7.0},
            {"rust_engine_step_generation": 3.0},
            {"rust_engine_step_context": "rust"},
            {"rust_engine_step_generation": "rust"},
        )

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    summary = backend.run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode="static",
        stride=2,
        latency_correction_scale=1.25,
    )

    assert len(calls) == 1
    assert calls[0][3:] == ("static", 2, 1.25)
    assert summary.get_context_latency_dict() == {"rust_engine_step_context": 7.0}
    assert summary.get_generation_latency_dict() == {"rust_engine_step_generation": 3.0}
    assert summary.get_context_energy_wms_dict() == {"rust_engine_step_context": 0.0}
    assert summary.get_generation_energy_wms_dict() == {"rust_engine_step_generation": 0.0}
    assert summary.get_context_source_dict() == {"rust_engine_step_context": "rust"}
    assert summary.get_generation_source_dict() == {"rust_engine_step_generation": "rust"}


def test_run_agg_with_osl_one_does_not_divide_by_zero(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """Regression: osl=1 (no-decode) must not raise and tokens/s/user must be 0.0."""
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *args, **kwargs: (1.0, 1.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *args, **kwargs: (0.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_memory_usage",
        lambda *args, **kwargs: {"total": 1.0},
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=1, prefix=2),
        ctx_tokens=8,
    )

    row = summary.get_summary_df().iloc[0]
    assert row["tpot"] > 0.0
    assert row["tokens/s/user"] == 0.0


def test_mix_step_efficiency_base_default_is_one(backend: BaseBackend) -> None:
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=16) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=0) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=0, gen_tokens=0) == 1.0


def _mixed_step_summary(
    *,
    context_latency=None,
    context_energy=None,
    context_source=None,
    generation_latency=None,
    generation_energy=None,
    generation_source=None,
):
    summary = MagicMock()
    summary.get_context_latency_dict.return_value = context_latency or {}
    summary.get_context_energy_wms_dict.return_value = context_energy or {}
    summary.get_context_source_dict.return_value = context_source or {}
    summary.get_generation_latency_dict.return_value = generation_latency or {}
    summary.get_generation_energy_wms_dict.return_value = generation_energy or {}
    summary.get_generation_source_dict.return_value = generation_source or {}
    return summary


def _semantic_attention_model(
    *,
    context_names,
    generation_names,
    context_graph_names=None,
    generation_graph_names=None,
):
    context_graph_names = context_names if context_graph_names is None else context_graph_names
    generation_graph_names = generation_names if generation_graph_names is None else generation_graph_names
    return SimpleNamespace(
        model_path="semantic-attention-model",
        MIXED_STEP_CONTEXT_ATTENTION_KEYS=context_names,
        MIXED_STEP_GENERATION_ATTENTION_KEYS=generation_names,
        context_ops=tuple(SimpleNamespace(_name=name) for name in context_graph_names),
        generation_ops=tuple(SimpleNamespace(_name=name) for name in generation_graph_names),
    )


def test_mix_step_consumes_every_semantic_attention_operation_exactly_once(backend, database, monkeypatch):
    model = _semantic_attention_model(
        context_names=("context_full_attention", "context_swa_attention"),
        generation_names=("generation_full_attention", "generation_swa_attention"),
    )
    summaries = iter(
        (
            _mixed_step_summary(
                context_latency={
                    "context_full_attention": 10.0,
                    "context_swa_attention": 20.0,
                    "context_mlp": 3.0,
                },
                context_energy={
                    "context_full_attention": 100.0,
                    "context_swa_attention": 200.0,
                    "context_mlp": 30.0,
                },
                context_source={
                    "context_full_attention": "sol",
                    "context_swa_attention": "sol",
                    "context_mlp": "sol",
                },
            ),
            _mixed_step_summary(
                context_latency={"context_full_attention": 12.0, "context_swa_attention": 24.0},
                context_energy={"context_full_attention": 120.0, "context_swa_attention": 240.0},
                context_source={"context_full_attention": "sol", "context_swa_attention": "sol"},
            ),
            _mixed_step_summary(
                generation_latency={"generation_full_attention": 2.0, "generation_swa_attention": 4.0},
                generation_energy={"generation_full_attention": 20.0, "generation_swa_attention": 40.0},
                generation_source={"generation_full_attention": "sol", "generation_swa_attention": "sol"},
            ),
        )
    )
    monkeypatch.setattr(backend, "run_static", lambda *_args, **_kwargs: next(summaries))

    latency, energy, per_ops, sources = backend._get_mix_step_latency(
        model,
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=5),
        ctx_tokens=4,
        gen_tokens=2,
        isl=8,
        osl=5,
        prefix=0,
    )

    assert latency == pytest.approx(27.0)
    assert energy == pytest.approx(270.0)
    assert per_ops == {
        "context_mlp": 3.0,
        "context_full_attention (scaled)": 6.0,
        "context_swa_attention (scaled)": 12.0,
        "generation_full_attention": 2.0,
        "generation_swa_attention": 4.0,
    }
    assert sources == dict.fromkeys(per_ops, "sol")


def test_mix_step_rejects_duplicate_semantic_attention_names(backend, database, monkeypatch):
    model = _semantic_attention_model(
        context_names=("context_attention", "context_attention"),
        generation_names=("generation_attention",),
    )
    monkeypatch.setattr(
        backend,
        "run_static",
        lambda *_args, **_kwargs: pytest.fail("duplicate semantic names must fail before execution"),
    )

    with pytest.raises(ValueError, match="duplicate context attention operation name"):
        backend._get_mix_step_latency(
            model,
            database,
            RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=1),
            ctx_tokens=8,
            gen_tokens=0,
            isl=8,
            osl=1,
            prefix=0,
        )


def test_mix_step_rejects_declared_attention_name_missing_from_graph_before_query(
    backend,
    database,
    monkeypatch,
):
    model = _semantic_attention_model(
        context_names=("context_attention", "context_missing"),
        generation_names=("generation_attention",),
        context_graph_names=("context_attention",),
    )
    monkeypatch.setattr(
        backend,
        "run_static",
        lambda *_args, **_kwargs: pytest.fail("graph contract must fail before execution"),
    )

    with pytest.raises(ValueError, match="context attention contract references missing graph operations"):
        backend._get_mix_step_latency(
            model,
            database,
            RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=1),
            ctx_tokens=8,
            gen_tokens=0,
            isl=8,
            osl=1,
            prefix=0,
        )


def test_mix_step_rejects_missing_executed_context_attention_energy(backend, database, monkeypatch):
    model = _semantic_attention_model(
        context_names=("context_attention",),
        generation_names=("generation_attention",),
    )
    summaries = iter(
        (
            _mixed_step_summary(
                context_latency={"context_attention": 10.0, "context_mlp": 3.0},
                context_energy={"context_attention": 100.0, "context_mlp": 30.0},
                context_source={"context_attention": "sol", "context_mlp": "sol"},
            ),
            _mixed_step_summary(
                context_latency={"context_attention": 12.0},
                context_energy={},
                context_source={"context_attention": "sol"},
            ),
        )
    )
    monkeypatch.setattr(backend, "run_static", lambda *_args, **_kwargs: next(summaries))

    with pytest.raises(ValueError, match="missing energy for executed context attention operation"):
        backend._get_mix_step_latency(
            model,
            database,
            RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=1),
            ctx_tokens=8,
            gen_tokens=0,
            isl=8,
            osl=1,
            prefix=0,
        )


def test_mix_step_rejects_missing_executed_context_attention_source(backend, database, monkeypatch):
    model = _semantic_attention_model(
        context_names=("context_attention",),
        generation_names=("generation_attention",),
    )
    summaries = iter(
        (
            _mixed_step_summary(
                context_latency={"context_attention": 10.0, "context_mlp": 3.0},
                context_energy={"context_attention": 100.0, "context_mlp": 30.0},
                context_source={"context_attention": "sol", "context_mlp": "sol"},
            ),
            _mixed_step_summary(
                context_latency={"context_attention": 12.0},
                context_energy={"context_attention": 120.0},
                context_source={},
            ),
        )
    )
    monkeypatch.setattr(backend, "run_static", lambda *_args, **_kwargs: next(summaries))

    with pytest.raises(ValueError, match="missing source for executed context attention operation"):
        backend._get_mix_step_latency(
            model,
            database,
            RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=1),
            ctx_tokens=8,
            gen_tokens=0,
            isl=8,
            osl=1,
            prefix=0,
        )


def test_mix_step_rejects_missing_executed_generation_attention_source(backend, database, monkeypatch):
    model = _semantic_attention_model(
        context_names=("context_attention",),
        generation_names=("generation_attention",),
    )
    summaries = iter(
        (
            _mixed_step_summary(
                context_latency={"context_attention": 10.0, "context_mlp": 3.0},
                context_energy={"context_attention": 100.0, "context_mlp": 30.0},
                context_source={"context_attention": "sol", "context_mlp": "sol"},
            ),
            _mixed_step_summary(
                context_latency={"context_attention": 12.0},
                context_energy={"context_attention": 120.0},
                context_source={"context_attention": "sol"},
            ),
            _mixed_step_summary(
                generation_latency={"generation_attention": 2.0},
                generation_energy={"generation_attention": 20.0},
                generation_source={},
            ),
        )
    )
    monkeypatch.setattr(backend, "run_static", lambda *_args, **_kwargs: next(summaries))

    with pytest.raises(ValueError, match="missing source for executed generation attention operation"):
        backend._get_mix_step_latency(
            model,
            database,
            RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=5),
            ctx_tokens=8,
            gen_tokens=2,
            isl=8,
            osl=5,
            prefix=0,
        )


def test_mix_step_marks_unexecuted_generation_attention_as_explicit_noop(backend, database, monkeypatch):
    model = _semantic_attention_model(
        context_names=("context_attention",),
        generation_names=("generation_attention",),
    )
    summaries = iter(
        (
            _mixed_step_summary(
                context_latency={"context_attention": 10.0, "context_mlp": 3.0},
                context_energy={"context_attention": 100.0, "context_mlp": 30.0},
                context_source={"context_attention": "sol", "context_mlp": "sol"},
            ),
            _mixed_step_summary(
                context_latency={"context_attention": 12.0},
                context_energy={"context_attention": 120.0},
                context_source={"context_attention": "sol"},
            ),
        )
    )
    monkeypatch.setattr(backend, "run_static", lambda *_args, **_kwargs: next(summaries))

    _latency, _energy, per_ops, sources = backend._get_mix_step_latency(
        model,
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=1),
        ctx_tokens=8,
        gen_tokens=0,
        isl=8,
        osl=1,
        prefix=0,
    )

    assert per_ops["generation_attention (not executed)"] == 0.0
    assert sources["generation_attention (not executed)"] == "not_executed"
