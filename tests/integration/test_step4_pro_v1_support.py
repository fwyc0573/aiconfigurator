# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline SDK and CLI integration coverage for Step4-Pro-V1."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aiconfigurator.sdk import utils
from aiconfigurator.sdk.operations.communication import NCCL, CustomAllReduce
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.operations.mla import ContextMLA, GenerationMLA, MLABmm
from aiconfigurator.sdk.operations.moe import MoE, MoEDispatch
from aiconfigurator.sdk.task_v2 import SinglePointEvaluation, Task

pytestmark = pytest.mark.integration

MODEL_ID = "stepfun-ai/Step4-Pro-V1"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_model_config_caches():
    """Keep package-local model discovery isolated between integration tests."""
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    yield
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()


def _prohibit_external_formula_inputs(monkeypatch) -> None:
    """Fail if a cached SOL run attempts network or profile-data access."""

    def fail_network(*_args, **_kwargs):
        raise AssertionError("Step4-Pro-V1 cached integration must not access HuggingFace")

    monkeypatch.setattr(utils, "_download_hf_config", fail_network)
    monkeypatch.setattr(utils, "_download_hf_json", fail_network)

    def fail_loader(operation_name):
        def fail_load_data(_cls, _database):
            raise AssertionError(f"Step4-Pro-V1 SOL integration must not load {operation_name} profile data")

        return classmethod(fail_load_data)

    for operation_class in (GEMM, ContextMLA, GenerationMLA, MLABmm, MoE, MoEDispatch, CustomAllReduce, NCCL):
        monkeypatch.setattr(operation_class, "load_data", fail_loader(operation_class.__name__))


def _assert_formula_sources(evaluation: SinglePointEvaluation, *, phases: tuple[str, ...]) -> None:
    """Require real per-operation evidence with SOL provenance in every phase."""
    assert set(evaluation.per_ops_source) == set(phases)
    for phase in phases:
        sources = evaluation.per_ops_source[phase]
        data = evaluation.per_ops_data[phase]
        assert sources
        for name, source in sources.items():
            if source == "not_executed":
                assert "(not executed)" in name
                assert data[name] == 0.0
            else:
                assert source == "sol"
        assert any("full_mla_approx" in name for name in sources)
        assert any("swa_mla_approx" in name for name in sources)


def test_step4_pro_v1_aggregate_sol_runs_offline_with_formula_evidence(monkeypatch):
    """The cached Pro ID must run the complete aggregate graph without external data."""
    _prohibit_external_formula_inputs(monkeypatch)
    task = Task(
        serving_mode="agg",
        model_path=MODEL_ID,
        system_name="h200_sxm",
        backend_name="vllm",
        backend_version="0.22.0",
        database_mode="SOL",
        isl=128,
        osl=2,
        prefix=0,
        nextn=0,
    )

    evaluation = task.run_single_agg(
        tp=8,
        pp=2,
        dp=1,
        moe_tp=8,
        moe_ep=1,
        batch_size=1,
        ctx_tokens=128,
        include_per_ops=True,
    )

    assert isinstance(evaluation, SinglePointEvaluation)
    assert task.primary_model_path == MODEL_ID
    assert math.isfinite(evaluation.row["ttft"]) and evaluation.row["ttft"] > 0.0
    assert math.isfinite(evaluation.row["tpot"]) and evaluation.row["tpot"] > 0.0
    _assert_formula_sources(evaluation, phases=("mix_step", "genonly_step"))


def test_step4_pro_v1_disaggregate_sol_runs_offline_with_formula_evidence(monkeypatch):
    """The cached Pro ID must run independent prefill/decode SOL graphs without external data."""
    _prohibit_external_formula_inputs(monkeypatch)
    task = Task(
        serving_mode="disagg",
        prefill_model_path=MODEL_ID,
        prefill_system_name="h200_sxm",
        prefill_backend_name="vllm",
        prefill_backend_version="0.22.0",
        decode_model_path=MODEL_ID,
        decode_system_name="h200_sxm",
        decode_backend_name="vllm",
        decode_backend_version="0.22.0",
        database_mode="SOL",
        isl=128,
        osl=2,
        prefix=0,
        nextn=0,
    )

    evaluation = task.run_single_disagg(
        prefill_tp=8,
        prefill_pp=2,
        prefill_dp=1,
        prefill_moe_tp=8,
        prefill_moe_ep=1,
        prefill_batch_size=1,
        decode_tp=8,
        decode_pp=2,
        decode_dp=1,
        decode_moe_tp=8,
        decode_moe_ep=1,
        decode_batch_size=1,
        include_per_ops=True,
    )

    assert isinstance(evaluation, SinglePointEvaluation)
    assert task.primary_model_path == MODEL_ID
    assert math.isfinite(evaluation.row["ttft"]) and evaluation.row["ttft"] > 0.0
    assert math.isfinite(evaluation.row["tpot"]) and evaluation.row["tpot"] > 0.0
    _assert_formula_sources(evaluation, phases=("prefill", "decode"))


def _offline_subprocess_environment() -> dict[str, str]:
    """Return a subprocess environment that cannot rely on HuggingFace network access."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    return environment


def test_step4_pro_v1_cli_estimate_sol_subprocess_is_offline():
    """The real estimate CLI must report the exact cached model through SOL."""
    command = [
        sys.executable,
        "-m",
        "aiconfigurator.main",
        "cli",
        "estimate",
        "--model-path",
        MODEL_ID,
        "--estimate-mode",
        "agg",
        "--system",
        "h200_sxm",
        "--backend",
        "vllm",
        "--backend-version",
        "0.22.0",
        "--database-mode",
        "SOL",
        "--isl",
        "128",
        "--osl",
        "2",
        "--batch-size",
        "1",
        "--ctx-tokens",
        "128",
        "--tp",
        "8",
        "--pp",
        "2",
        "--dp",
        "1",
        "--etp",
        "8",
        "--ep",
        "1",
        "--nextn",
        "0",
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_offline_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Performance Estimate (agg)" in completed.stdout
    assert f"Model:            {MODEL_ID}" in completed.stdout
    assert "System:           h200_sxm" in completed.stdout
    assert "Backend:          vllm (0.22.0)" in completed.stdout
    assert "TTFT:" in completed.stdout
    assert "TPOT:" in completed.stdout
    assert "huggingface.co" not in (completed.stdout + completed.stderr).lower()


def test_step4_pro_v1_cli_generate_subprocess_uses_cached_identity(tmp_path):
    """The requested naive generate command must resolve the Pro ID without network access."""
    command = [
        sys.executable,
        "-m",
        "aiconfigurator.main",
        "cli",
        "generate",
        "--model-path",
        MODEL_ID,
        "--total-gpus",
        "8",
        "--system",
        "h200_sxm",
        "--save-dir",
        str(tmp_path),
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_offline_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Naive Configuration Generated Successfully" in completed.stdout
    assert f"Model:           {MODEL_ID}" in completed.stdout
    assert "Total GPUs:      8" in completed.stdout
    assert "huggingface.co" not in (completed.stdout + completed.stderr).lower()
    generated_configs = list(tmp_path.rglob("generator_config.yaml"))
    assert len(generated_configs) == 1
