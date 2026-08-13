# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
from collections import Counter

import pytest

import aiconfigurator_core
from aiconfigurator.sdk import common, config, engine
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    GenerationDeepSeekV4AttentionModule,
)
from tests.unit.sdk.database.conftest import _get_comprehensive_db_singleton

pytestmark = pytest.mark.unit

MODEL_CASES = {
    "flash": {
        "model_path": "sgl-project/DeepSeek-V4-Flash-FP8",
        "num_layers": 43,
        "attention_scales": {4: 21.0, 128: 22.0},
        "context_operation_count": 16,
        "generation_operation_count": 10,
        "context_attention_ms": 809.7607188479999,
        "generation_attention_ms": 9.693396479999999,
        "context_total_ms": 1390.3770746879998,
        "generation_total_ms": 16.628882412000003,
        "scaled_weight_bytes": {4: 892_244_640.0, 128: 605_553_344.0},
        "engine_json_bytes": 6393,
        "engine_json_sha256": "6c3547cb37d070d53bf8677d08eefa3ce0b870c37b4b98db55586b03e0c358a2",
        "engine_bincode_bytes": 2686,
        "engine_bincode_sha256": "a3721774a5eef1f9aaaf809244ed2ac0714622e01cfeed7c6720107b6da449e2",
    },
    "pro": {
        "model_path": "sgl-project/DeepSeek-V4-Pro-FP8",
        "num_layers": 61,
        "attention_scales": {4: 30.0, 128: 31.0},
        "context_operation_count": 16,
        "generation_operation_count": 10,
        "context_attention_ms": 2562.50413056,
        "generation_attention_ms": 30.60943872,
        "context_total_ms": 4662.762635263998,
        "generation_total_ms": 53.961132468,
        "scaled_weight_bytes": {4: 2_717_738_880.0, 128: 2_055_997_376.0},
        "engine_json_bytes": 6399,
        "engine_json_sha256": "7e44127359a2aaed0be3f1e20e5d7bfb730415b696c4567c23e3d26999a50eaa",
        "engine_bincode_bytes": 2684,
        "engine_bincode_sha256": "d5e1bfd867dc59294df0ba670bbb24fe7aa9be957f926966bb2e16cea4f9b1d4",
    },
}

EXPECTED_RUNTIME_SPECS = {
    4: {
        "retention_mode": "swa",
        "compressed_history_selection": "topk",
        "projection_head_dim": 512,
        "cache_projection_width": 512,
        "cache_entry_width": 512,
        "cache_projection_matrix_count": 5,
        "cache_auxiliary_fp32_elements": 4096,
        "cache_auxiliary_ops_per_token": 2048,
        "window_size": 128,
        "compression_ratio": 4,
        "index_n_heads": 64,
        "index_head_dim": 128,
    },
    128: {
        "retention_mode": "swa",
        "compressed_history_selection": "all",
        "projection_head_dim": 512,
        "cache_projection_width": 512,
        "cache_entry_width": 512,
        "cache_projection_matrix_count": 3,
        "cache_auxiliary_fp32_elements": 65_536,
        "cache_auxiliary_ops_per_token": 1024,
        "window_size": 128,
        "compression_ratio": 128,
        "index_n_heads": 0,
        "index_head_dim": 0,
        "index_topk": 0,
    },
}


def _model_config() -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=1,
        nextn=0,
    )


def _attention_operations(model, phase: str):
    operations = model.context_ops if phase == "context" else model.generation_ops
    operation_type = ContextDeepSeekV4AttentionModule if phase == "context" else GenerationDeepSeekV4AttentionModule
    return operations, [operation for operation in operations if isinstance(operation, operation_type)]


@pytest.mark.parametrize("case_name", MODEL_CASES)
@pytest.mark.parametrize("phase", ["context", "generation"])
def test_deepseek_v4_model_builds_explicit_runtime_specs_without_changing_graph_or_weights(case_name, phase):
    case = MODEL_CASES[case_name]
    model = get_model(case["model_path"], _model_config(), backend_name="trtllm")
    operations, attention_operations = _attention_operations(model, phase)

    assert model._num_layers == case["num_layers"]
    assert len(operations) == case[f"{phase}_operation_count"]
    assert len(attention_operations) == 2
    assert Counter(
        {
            operation._runtime_spec.compression_ratio: float(operation._scale_factor)
            for operation in attention_operations
        }
    ) == Counter(case["attention_scales"])

    for operation in attention_operations:
        ratio = operation._runtime_spec.compression_ratio
        expected_spec = EXPECTED_RUNTIME_SPECS[ratio]
        for field_name, expected_value in expected_spec.items():
            assert getattr(operation._runtime_spec, field_name) == expected_value
        if ratio == 4:
            assert operation._runtime_spec.index_topk == model.extra_params.index_topk
        assert operation.get_weights() == pytest.approx(case["scaled_weight_bytes"][ratio])


@pytest.mark.parametrize("case_name", MODEL_CASES)
def test_deepseek_v4_static_sol_preserves_exact_pre_migration_totals(case_name):
    case = MODEL_CASES[case_name]
    database = copy.deepcopy(_get_comprehensive_db_singleton())
    database.system_spec["gpu"]["mem_capacity"] = 288_400_343_040
    database.system_spec["misc"]["nccl_mem"] = {1: 0, 2: 0, 4: 0, 8: 0}
    database.system_spec["misc"]["other_mem"] = 0
    database.set_default_database_mode(common.DatabaseMode.SOL)
    model = get_model(case["model_path"], _model_config(), backend_name="trtllm")

    summary = TRTLLMBackend().run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=256, osl=4, prefix=0),
        mode="static",
        stride=1,
    )
    context_latency = summary.get_context_latency_dict()
    generation_latency = summary.get_generation_latency_dict()

    assert context_latency["context_attention"] == pytest.approx(case["context_attention_ms"])
    assert generation_latency["generation_attention"] == pytest.approx(case["generation_attention_ms"])
    assert sum(context_latency.values()) == pytest.approx(case["context_total_ms"])
    assert sum(generation_latency.values()) == pytest.approx(case["generation_total_ms"])


@pytest.mark.parametrize("case_name", MODEL_CASES)
def test_deepseek_v4_engine_json_and_bincode_preserve_exact_pre_migration_payloads(case_name):
    case = MODEL_CASES[case_name]
    model = get_model(case["model_path"], _model_config(), backend_name="trtllm")

    spec_json = engine.build_engine_spec_json(
        model,
        model_path=case["model_path"],
        system="test_system",
        backend="trtllm",
        backend_version="v1",
        kv_block_size=None,
        systems_path=None,
        nextn=0,
        nextn_accept_rates=None,
        database=None,
    )
    spec_bytes = spec_json.encode("utf-8")
    bincode = bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json))

    assert len(spec_bytes) == case["engine_json_bytes"]
    assert hashlib.sha256(spec_bytes).hexdigest() == case["engine_json_sha256"]
    assert len(bincode) == case["engine_bincode_bytes"]
    assert hashlib.sha256(bincode).hexdigest() == case["engine_bincode_sha256"]
