"""Unit tests for the Step4 profiled aggregate matrix runner."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aiconfigurator.sdk.errors import (
    InsufficientMemoryError,
    KVCacheCapacityError,
    NoFeasibleConfigError,
    PerfDataNotAvailableError,
)
from tests.performance.step4_profiled_agg_matrix import (
    _positive_finite,
    build_invalid_topologies,
    build_runnable_topologies,
    classify_matrix_exception,
    run_matrix_point,
)

pytestmark = pytest.mark.unit


def test_positive_finite_accepts_numpy_integer_scalars():
    assert _positive_finite(np.int64(1024), field="ctx_tokens") == 1024.0


def test_topology_contract_keeps_local_attention_tp_groups_runnable():
    assert build_runnable_topologies() == [
        (1, 1, 1, 1, 1, 1),
        (1, 1, 2, 1, 2, 1),
        (2, 1, 1, 1, 2, 1),
        (1, 1, 4, 1, 4, 1),
        (2, 1, 2, 1, 4, 1),
        (4, 1, 1, 1, 4, 1),
        (1, 1, 8, 1, 8, 1),
        (2, 1, 4, 1, 8, 1),
        (4, 1, 2, 1, 8, 1),
        (1, 1, 16, 1, 16, 1),
        (2, 1, 8, 1, 16, 1),
        (4, 1, 4, 1, 16, 1),
        (1, 1, 32, 1, 32, 1),
        (2, 1, 16, 1, 32, 1),
        (4, 1, 8, 1, 32, 1),
        (1, 1, 64, 1, 64, 1),
        (2, 1, 32, 1, 64, 1),
        (4, 1, 16, 1, 64, 1),
    ]
    assert build_invalid_topologies() == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (InsufficientMemoryError("model"), "memory_infeasible"),
        (KVCacheCapacityError("kv"), "memory_infeasible"),
        (NoFeasibleConfigError("sla"), "sla_infeasible"),
        (PerfDataNotAvailableError("data"), "data_unavailable"),
        (RuntimeError("unexpected"), "error"),
    ],
)
def test_exception_classification_is_terminal_and_explicit(error, expected):
    assert classify_matrix_exception(error) == expected


def test_successful_point_selects_best_legal_config_and_complete_replicas():
    captured = {}

    class FakeTask:
        def __init__(self, **kwargs):
            captured["task_kwargs"] = kwargs

        def _load_database(self, system, backend, version):
            captured["database_identity"] = (system, backend, version)
            return object()

        def sweep_agg_kwargs(self, *, database):
            assert database is not None
            return {"parallel_config_list": [(99, 1, 1, 1, 1, 1)]}

        def run_single_agg(self, **kwargs):
            captured["single_point_kwargs"] = kwargs
            return SimpleNamespace(
                per_ops_data={"mix_step": {"context_moe": 1.25}},
                per_ops_source={"mix_step": {"context_moe": "silicon"}},
            )

    def fake_sweep(**kwargs):
        captured["parallel_config_list"] = kwargs["parallel_config_list"]
        return pd.DataFrame(
            [
                {
                    "tokens/s/gpu": 12.5,
                    "tokens/s": 200.0,
                    "num_total_gpus": 16,
                    "tp": 1,
                    "pp": 1,
                    "dp": 16,
                    "moe_tp": 1,
                    "moe_ep": 16,
                    "cp": 1,
                    "bs": 1,
                    "ctx_tokens": 1024,
                    "ttft": 9000.0,
                    "tpot": 25.0,
                },
                {
                    "tokens/s/gpu": 10.0,
                    "tokens/s": 640.0,
                    "num_total_gpus": 64,
                    "tp": 1,
                    "pp": 1,
                    "dp": 64,
                    "moe_tp": 1,
                    "moe_ep": 64,
                    "cp": 1,
                    "bs": 2,
                    "ctx_tokens": 2048,
                    "ttft": 8000.0,
                    "tpot": 20.0,
                },
            ]
        )

    result = run_matrix_point(
        model="stepfun-ai/Step4-Pro-V3",
        isl=1024,
        task_factory=FakeTask,
        sweep_fn=fake_sweep,
    )

    assert result["status"] == "success"
    assert result["throughput_per_used_gpu"] == pytest.approx(12.5)
    assert result["deployment_gpus"] == 16
    assert result["replicas"] == 4
    assert result["total_gpus_used"] == 64
    assert result["unused_gpus"] == 0
    assert result["selected_config"]["moe_ep"] == 16
    assert result["selected_config"]["ctx_tokens"] == 1024
    assert result["per_ops_data"] == {"mix_step": {"context_moe": 1.25}}
    assert result["per_ops_source"] == {"mix_step": {"context_moe": "silicon"}}
    assert captured["single_point_kwargs"] == {
        "tp": 1,
        "pp": 1,
        "dp": 16,
        "moe_tp": 1,
        "moe_ep": 16,
        "batch_size": 1,
        "ctx_tokens": 1024,
        "include_per_ops": True,
    }
    assert captured["parallel_config_list"] == build_runnable_topologies()
    assert captured["task_kwargs"]["database_mode"] == "SILICON"
    assert captured["task_kwargs"]["nextn"] == 0
    assert captured["task_kwargs"]["ttft"] == 10_000
    assert captured["task_kwargs"]["tpot"] == 50_000


def test_point_failure_returns_terminal_outcome_without_fabricated_throughput():
    class FakeTask:
        def __init__(self, **_kwargs):
            pass

        def _load_database(self, *_args):
            return object()

        def sweep_agg_kwargs(self, *, database):
            return {"parallel_config_list": []}

    def failing_sweep(**_kwargs):
        raise InsufficientMemoryError("does not fit")

    result = run_matrix_point(
        model="stepfun-ai/Step4-Pro-V4",
        isl=131072,
        task_factory=FakeTask,
        sweep_fn=failing_sweep,
    )

    assert result == {
        "model": "stepfun-ai/Step4-Pro-V4",
        "isl": 131072,
        "status": "memory_infeasible",
        "error_type": "InsufficientMemoryError",
        "reason": "does not fit",
    }
