# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact fixed-cluster ranking tests for disaggregated sweeps."""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiconfigurator.sdk import config, sweep
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit


def _worker(
    role: str,
    *,
    tp: int,
    dp: int = 1,
    bs: int,
    seq_s: float,
    category: str | None = None,
    ttft: float = 10.0,
    tpot: float = 10.0,
) -> dict[str, Any]:
    width = tp * dp
    parallel = category or f"tp{tp}pp1dp{dp}moetp1moeep{width}cp1"
    return {
        "model": "test-model",
        "isl": 4096,
        "osl": 1,
        "prefix": 0,
        "concurrency": bs,
        "bs": bs,
        "global_bs": bs * dp,
        "tp": tp,
        "pp": 1,
        "dp": dp,
        "moe_tp": 1,
        "moe_ep": width,
        "cp": 1,
        "parallel": parallel,
        "ttft": ttft if role == "prefill" else 0.0,
        "tpot": tpot if role == "decode" else 0.0,
        "seq/s": seq_s,
        "seq/s/gpu": seq_s / width,
        "tokens/s/user": 1000.0 / max(tpot, 1.0),
        "num_total_gpus": width,
        "gemm": "fp8",
        "kvcache": "fp8",
        "fmha": "fp8",
        "moe": "fp8",
        "comm": "half",
        "memory": 1.0,
        "backend": "vllm",
        "version": "0.22.0",
        "system": "h200_sxm",
        "power_w": 500.0,
    }


def _identity(
    prefill: dict[str, Any],
    prefill_workers: int,
    decode: dict[str, Any],
    decode_workers: int,
) -> tuple[int, ...]:
    return (
        prefill["tp"],
        prefill["pp"],
        prefill["dp"],
        prefill["moe_tp"],
        prefill["moe_ep"],
        prefill.get("cp", 1),
        prefill["bs"],
        prefill_workers,
        decode["tp"],
        decode["pp"],
        decode["dp"],
        decode["moe_tp"],
        decode["moe_ep"],
        decode.get("cp", 1),
        decode["bs"],
        decode_workers,
    )


def _brute_force_rank_one(
    *,
    prefill_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    prefill_workers: list[int],
    decode_workers: list[int],
    num_gpu_set: set[int],
    cluster_gpus: int,
    ttft_target: float = 100.0,
    tpot_target: float = 100.0,
    prefill_degradation: float = 1.0,
    decode_degradation: float = 1.0,
    require_same_tp: bool = False,
    max_prefill_gpus: int | None = None,
    max_decode_gpus: int | None = None,
) -> tuple[dict[str, Any], float, tuple[int, ...]]:
    candidates: list[tuple[float, tuple[int, ...], dict[str, Any]]] = []
    for prefill in prefill_rows:
        if prefill["ttft"] >= ttft_target:
            continue
        for decode in decode_rows:
            if not 0 < decode["tpot"] < tpot_target:
                continue
            if require_same_tp and prefill["tp"] != decode["tp"]:
                continue
            for p_workers in prefill_workers:
                for d_workers in decode_workers:
                    p_gpus = prefill["num_total_gpus"] * p_workers
                    d_gpus = decode["num_total_gpus"] * d_workers
                    deployment_gpus = p_gpus + d_gpus
                    if num_gpu_set and deployment_gpus not in num_gpu_set:
                        continue
                    if deployment_gpus > cluster_gpus:
                        continue
                    if max_prefill_gpus is not None and p_gpus > max_prefill_gpus:
                        continue
                    if max_decode_gpus is not None and d_gpus > max_decode_gpus:
                        continue
                    row = sweep._rate_match_dict(
                        prefill,
                        p_workers,
                        decode,
                        d_workers,
                        prefill_degradation=prefill_degradation,
                        decode_degradation=decode_degradation,
                    )
                    score = row["tokens/s/gpu"] * (cluster_gpus // deployment_gpus) * deployment_gpus / cluster_gpus
                    candidates.append((score, _identity(prefill, p_workers, decode, d_workers), row))
    assert candidates
    score, identity, row = min(candidates, key=lambda item: (-item[0], item[1]))
    return row, score, identity


def _solve(
    *,
    prefill_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    prefill_workers: list[int],
    decode_workers: list[int],
    num_gpu_set: set[int],
    cluster_gpus: int,
    **kwargs: Any,
) -> pd.DataFrame:
    result = sweep._find_best_disagg_cluster_under_constraint(
        ttft_target=kwargs.pop("ttft_target", 100.0),
        tpot_target=kwargs.pop("tpot_target", 100.0),
        prefill_summary_df=pd.DataFrame(prefill_rows),
        decode_summary_df=pd.DataFrame(decode_rows),
        ranking_total_gpus=cluster_gpus,
        num_gpu_set=num_gpu_set,
        prefill_num_worker_list=prefill_workers,
        decode_num_worker_list=decode_workers,
        max_prefill_gpus=kwargs.pop("max_prefill_gpus", None),
        max_decode_gpus=kwargs.pop("max_decode_gpus", None),
        require_same_tp=kwargs.pop("require_same_tp", False),
        prefill_degradation=kwargs.pop("prefill_degradation", 1.0),
        decode_degradation=kwargs.pop("decode_degradation", 1.0),
        autoscale_ttft_correction_factor=1.0,
        ranking_metric_kind=kwargs.pop("ranking_metric_kind", "output_token_throughput"),
    )
    assert not kwargs
    assert result is not None
    assert len(result) == 1
    return result


def test_fixed_cluster_solver_raises_sla_error_when_prefill_misses_ttft():
    with pytest.raises(NoFeasibleConfigError, match="TTFT"):
        sweep._find_best_disagg_cluster_under_constraint(
            ttft_target=10.0,
            tpot_target=100.0,
            prefill_summary_df=pd.DataFrame([_worker("prefill", tp=8, bs=1, seq_s=10.0, ttft=10.0)]),
            decode_summary_df=pd.DataFrame([_worker("decode", tp=8, bs=1, seq_s=10.0, tpot=10.0)]),
            ranking_total_gpus=64,
            num_gpu_set=set(),
            prefill_num_worker_list=[1],
            decode_num_worker_list=[1],
            max_prefill_gpus=None,
            max_decode_gpus=None,
            require_same_tp=False,
            prefill_degradation=1.0,
            decode_degradation=1.0,
            autoscale_ttft_correction_factor=1.0,
        )


def test_sweep_disagg_real_solver_fails_fast_when_deployment_exceeds_cluster(monkeypatch):
    prefill = pd.DataFrame([_worker("prefill", tp=8, bs=1, seq_s=10.0, ttft=10.0)])
    decode = pd.DataFrame([_worker("decode", tp=8, bs=1, seq_s=10.0, tpot=10.0)])
    monkeypatch.setattr(
        sweep,
        "_get_disagg_worker_candidates",
        MagicMock(side_effect=[prefill, decode]),
    )

    with pytest.raises(RuntimeError, match="worker allocation"):
        sweep.sweep_disagg(
            model_path="test-model",
            runtime_config=config.RuntimeConfig(isl=4096, osl=1, ttft=100.0, tpot=100.0),
            prefill_database=object(),
            prefill_backend_name="vllm",
            prefill_model_config=config.ModelConfig(),
            prefill_parallel_config_list=[(8, 1, 1, 1, 8, 1)],
            prefill_latency_correction=1.0,
            decode_database=object(),
            decode_backend_name="vllm",
            decode_model_config=config.ModelConfig(),
            decode_parallel_config_list=[(8, 1, 1, 1, 8, 1)],
            decode_latency_correction=1.0,
            prefill_num_worker_list=[1],
            decode_num_worker_list=[1],
            ranking_total_gpus=8,
        )


def _brute_force_prefill_rank_one(
    *,
    prefill_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    prefill_workers: list[int],
    decode_workers: list[int],
    num_gpu_set: set[int],
    cluster_gpus: int,
    ttft_target: float = 100.0,
) -> tuple[dict[str, Any], float, tuple[int, ...]]:
    candidates: list[tuple[float, tuple[int, ...], dict[str, Any]]] = []
    for prefill in prefill_rows:
        if not 0 < prefill["ttft"] < ttft_target:
            continue
        fresh_tokens = prefill["isl"] - prefill["prefix"]
        assert fresh_tokens > 0
        for decode in decode_rows:
            for p_workers in prefill_workers:
                for d_workers in decode_workers:
                    p_gpus = prefill["num_total_gpus"] * p_workers
                    d_gpus = decode["num_total_gpus"] * d_workers
                    deployment_gpus = p_gpus + d_gpus
                    if num_gpu_set and deployment_gpus not in num_gpu_set:
                        continue
                    if deployment_gpus > cluster_gpus:
                        continue
                    row = sweep._rate_match_dict(
                        prefill,
                        p_workers,
                        decode,
                        d_workers,
                        prefill_degradation=1.0,
                        decode_degradation=1.0,
                    )
                    prefill_tokens_per_second = (
                        prefill["global_bs"] * p_workers * fresh_tokens / (prefill["ttft"] / 1000.0)
                    )
                    score = (
                        prefill_tokens_per_second
                        / deployment_gpus
                        * (cluster_gpus // deployment_gpus)
                        * deployment_gpus
                        / cluster_gpus
                    )
                    candidates.append((score, _identity(prefill, p_workers, decode, d_workers), row))
    assert candidates
    score, identity, row = min(candidates, key=lambda item: (-item[0], item[1]))
    return row, score, identity


def test_cluster_solver_recovers_winner_that_raw_worker_matching_would_drop():
    prefill = [_worker("prefill", tp=8, bs=1, seq_s=1600.0)]
    decode = [_worker("decode", tp=8, bs=1, seq_s=1000.0)]

    result = _solve(
        prefill_rows=prefill,
        decode_rows=decode,
        prefill_workers=[1],
        decode_workers=[1, 2],
        num_gpu_set={16, 24},
        cluster_gpus=64,
    ).iloc[0]

    raw_16 = min(1600.0, 1000.0) / 16
    raw_24 = min(1600.0, 2000.0) / 24
    cluster_16 = raw_16
    cluster_24 = raw_24 * 48 / 64
    assert raw_24 > raw_16
    assert cluster_16 > cluster_24
    assert result["num_total_gpus"] == 16
    assert result["(d)workers"] == 1


def test_cluster_solver_uses_full_precision_before_returning_rank_one():
    prefill = [
        _worker("prefill", tp=8, bs=1, seq_s=1600.0004),
        _worker("prefill", tp=8, bs=2, seq_s=1600.0005),
    ]
    decode = [_worker("decode", tp=8, bs=1, seq_s=5000.0)]

    result = _solve(
        prefill_rows=prefill,
        decode_rows=decode,
        prefill_workers=[1],
        decode_workers=[1],
        num_gpu_set={16},
        cluster_gpus=64,
    ).iloc[0]

    assert result["(p)bs"] == 2
    assert result["tokens/s/gpu"] == pytest.approx(1600.0005 / 16)


def test_cluster_identity_orders_numeric_two_before_sixteen():
    decode = _worker("decode", tp=1, bs=1, seq_s=1.0)
    tp2 = _worker("prefill", tp=2, bs=1, seq_s=1.0)
    tp16 = _worker("prefill", tp=16, bs=1, seq_s=1.0)

    assert sweep._disagg_cluster_identity(tp2, 1, decode, 1) < sweep._disagg_cluster_identity(
        tp16,
        1,
        decode,
        1,
    )


def test_cluster_solver_excludes_deployment_wider_than_cluster():
    prefill = [_worker("prefill", tp=8, bs=1, seq_s=1000.0)]
    decode = [_worker("decode", tp=8, bs=1, seq_s=1000.0)]

    result = _solve(
        prefill_rows=prefill,
        decode_rows=decode,
        prefill_workers=[1, 4],
        decode_workers=[1, 4],
        num_gpu_set=set(),
        cluster_gpus=32,
    ).iloc[0]

    assert result["num_total_gpus"] <= 32


def test_cluster_solver_applies_role_gpu_caps_independently():
    prefill = [_worker("prefill", tp=4, bs=1, seq_s=1000.0)]
    decode = [_worker("decode", tp=4, bs=1, seq_s=1000.0)]

    result = _solve(
        prefill_rows=prefill,
        decode_rows=decode,
        prefill_workers=[1, 2, 3],
        decode_workers=[1],
        num_gpu_set=set(),
        cluster_gpus=64,
        max_prefill_gpus=4,
    ).iloc[0]

    assert result["(p)workers"] == 1


def test_cluster_solver_preserves_same_tp_constraint():
    prefill = [
        _worker("prefill", tp=2, bs=1, seq_s=100.0),
        _worker("prefill", tp=4, bs=1, seq_s=1000.0),
    ]
    decode = [_worker("decode", tp=2, bs=1, seq_s=1000.0)]

    result = _solve(
        prefill_rows=prefill,
        decode_rows=decode,
        prefill_workers=[1],
        decode_workers=[1],
        num_gpu_set=set(),
        cluster_gpus=64,
        require_same_tp=True,
    ).iloc[0]

    assert result["(p)tp"] == result["(d)tp"] == 2


def test_cluster_solver_matches_brute_force_oracle_on_random_small_fixtures():
    rng = random.Random(20260713)
    for _ in range(40):
        prefill = [
            _worker(
                "prefill",
                tp=rng.choice((1, 2, 4)),
                bs=index,
                seq_s=rng.uniform(10.0, 500.0),
                ttft=rng.uniform(1.0, 80.0),
            )
            for index in range(1, 5)
        ]
        decode = [
            _worker(
                "decode",
                tp=tp,
                bs=index,
                seq_s=rng.uniform(10.0, 500.0),
                category=f"decode-tp{tp}",
                tpot=rng.uniform(1.0, 80.0),
            )
            for tp in (1, 2)
            for index in range(1, 4)
        ]
        kwargs = {
            "prefill_rows": prefill,
            "decode_rows": decode,
            "prefill_workers": [1, 2, 3],
            "decode_workers": [1, 2, 3],
            "num_gpu_set": set(range(2, 17)),
            "cluster_gpus": 16,
            "prefill_degradation": 0.9,
            "decode_degradation": 0.92,
        }
        expected_row, expected_score, expected_identity = _brute_force_rank_one(**kwargs)
        actual = _solve(**kwargs).iloc[0].to_dict()
        actual_score = (
            actual["tokens/s/gpu"]
            * (kwargs["cluster_gpus"] // actual["num_total_gpus"])
            * actual["num_total_gpus"]
            / kwargs["cluster_gpus"]
        )
        actual_identity = (
            actual["(p)tp"],
            actual["(p)pp"],
            actual["(p)dp"],
            actual["(p)moe_tp"],
            actual["(p)moe_ep"],
            actual["(p)cp"],
            actual["(p)bs"],
            actual["(p)workers"],
            actual["(d)tp"],
            actual["(d)pp"],
            actual["(d)dp"],
            actual["(d)moe_tp"],
            actual["(d)moe_ep"],
            1,
            actual["(d)bs"],
            actual["(d)workers"],
        )
        assert actual_score == pytest.approx(expected_score)
        assert actual_identity == expected_identity
        assert actual["num_total_gpus"] == expected_row["num_total_gpus"]


def test_cluster_solver_default_output_metric_matches_explicit_legacy_mode():
    kwargs = {
        "prefill_rows": [_worker("prefill", tp=4, bs=1, seq_s=1000.0)],
        "decode_rows": [_worker("decode", tp=4, bs=1, seq_s=1000.0)],
        "prefill_workers": [1, 2],
        "decode_workers": [1, 2],
        "num_gpu_set": set(),
        "cluster_gpus": 64,
    }

    implicit = _solve(**kwargs)
    explicit = _solve(**kwargs, ranking_metric_kind="output_token_throughput")

    pd.testing.assert_frame_equal(implicit, explicit)


def test_prefill_metric_solver_accepts_osl_one_zero_tpot_and_uses_smallest_decode_batch_tie():
    prefill = [_worker("prefill", tp=4, bs=2, seq_s=1.0, ttft=50.0)]
    decode = [
        _worker("decode", tp=4, bs=2, seq_s=0.0, tpot=0.0, category="decode-tp4"),
        _worker("decode", tp=4, bs=1, seq_s=0.0, tpot=0.0, category="decode-tp4"),
    ]

    result = _solve(
        prefill_rows=prefill,
        decode_rows=decode,
        prefill_workers=[1],
        decode_workers=[1],
        num_gpu_set={8},
        cluster_gpus=64,
        ranking_metric_kind="prefill_input_throughput",
    ).iloc[0]

    assert result["(p)bs"] == 2
    assert result["(d)bs"] == 1
    assert result["tpot"] == 0.0
    assert result["tokens/s/gpu"] == 0.0


def test_prefill_metric_solver_counts_decode_gpus_in_fixed_cluster_denominator():
    prefill = [_worker("prefill", tp=8, bs=2, seq_s=1.0, ttft=50.0)]
    decode = [
        _worker("decode", tp=8, bs=1, seq_s=0.0, tpot=0.0, category="decode-width8"),
        _worker("decode", tp=16, bs=1, seq_s=0.0, tpot=0.0, category="decode-width16"),
    ]

    result = _solve(
        prefill_rows=prefill,
        decode_rows=decode,
        prefill_workers=[1],
        decode_workers=[1],
        num_gpu_set={16, 24},
        cluster_gpus=64,
        ranking_metric_kind="prefill_input_throughput",
    ).iloc[0]

    assert result["num_total_gpus"] == 16
    assert result["(d)tp"] == 8


def test_prefill_metric_solver_matches_brute_force_oracle_on_random_small_fixtures():
    rng = random.Random(20260714)
    for _ in range(40):
        prefill = [
            _worker(
                "prefill",
                tp=rng.choice((1, 2, 4)),
                bs=index,
                seq_s=rng.uniform(10.0, 500.0),
                ttft=rng.uniform(1.0, 80.0),
            )
            for index in range(1, 5)
        ]
        decode = [
            _worker(
                "decode",
                tp=tp,
                bs=index,
                seq_s=0.0,
                category=f"decode-tp{tp}",
                tpot=0.0,
            )
            for tp in (1, 2)
            for index in range(1, 4)
        ]
        kwargs = {
            "prefill_rows": prefill,
            "decode_rows": decode,
            "prefill_workers": [1, 2, 3],
            "decode_workers": [1, 2, 3],
            "num_gpu_set": set(range(2, 17)),
            "cluster_gpus": 16,
        }
        expected_row, expected_score, expected_identity = _brute_force_prefill_rank_one(**kwargs)
        actual = _solve(**kwargs, ranking_metric_kind="prefill_input_throughput").iloc[0].to_dict()
        deployment_gpus = actual["num_total_gpus"]
        prefill_tokens_per_second = (
            actual["(p)global_bs"]
            * actual["(p)workers"]
            * (actual["isl"] - actual["prefix"])
            / (actual["ttft"] / 1000.0)
        )
        actual_score = (
            prefill_tokens_per_second
            / deployment_gpus
            * (kwargs["cluster_gpus"] // deployment_gpus)
            * deployment_gpus
            / kwargs["cluster_gpus"]
        )
        actual_identity = sweep._disagg_result_cluster_identity(actual)

        assert actual_score == pytest.approx(expected_score)
        assert actual_identity == expected_identity
        assert actual["num_total_gpus"] == expected_row["num_total_gpus"]


@pytest.mark.parametrize(
    ("cluster_gpus", "expected_error", "expected_message"),
    [
        (True, TypeError, "ranking_total_gpus must be a positive integer"),
        (64.0, TypeError, "ranking_total_gpus must be a positive integer"),
        (0, ValueError, "ranking_total_gpus must be positive"),
        (-1, ValueError, "ranking_total_gpus must be positive"),
    ],
)
def test_sweep_disagg_rejects_invalid_ranking_total_gpus(
    cluster_gpus,
    expected_error,
    expected_message,
):
    with pytest.raises(expected_error, match=expected_message):
        sweep.sweep_disagg(
            model_path="x",
            runtime_config=config.RuntimeConfig(),
            prefill_database=None,
            prefill_backend_name="vllm",
            prefill_model_config=config.ModelConfig(),
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="vllm",
            decode_model_config=config.ModelConfig(),
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            prefill_num_worker_list=[1],
            decode_num_worker_list=[1],
            ranking_total_gpus=cluster_gpus,
        )


def test_sweep_disagg_rejects_cluster_ranking_with_autoscale():
    with pytest.raises(ValueError, match=r"ranking_total_gpus.*autoscale"):
        sweep.sweep_disagg(
            model_path="x",
            runtime_config=config.RuntimeConfig(),
            prefill_database=None,
            prefill_backend_name="vllm",
            prefill_model_config=config.ModelConfig(),
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="vllm",
            decode_model_config=config.ModelConfig(),
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            prefill_num_worker_list=[1],
            decode_num_worker_list=[1],
            ranking_total_gpus=64,
            autoscale=True,
        )


def test_task_round_trips_and_propagates_disagg_ranking_total_gpus():
    task = Task(
        serving_mode="disagg",
        prefill_model_path="Qwen/Qwen3-32B",
        prefill_system_name="h200_sxm",
        prefill_backend_name="vllm",
        decode_model_path="Qwen/Qwen3-32B",
        decode_system_name="h200_sxm",
        decode_backend_name="vllm",
        disagg_ranking_total_gpus=64,
        disagg_ranking_metric_kind="prefill_input_throughput",
        osl=1,
    )

    assert task.to_dict()["disagg_ranking_total_gpus"] == 64
    restored = Task.from_yaml(task.to_dict())
    assert restored.disagg_ranking_total_gpus == 64
    assert restored.disagg_ranking_metric_kind == "prefill_input_throughput"
    kwargs = task.sweep_disagg_kwargs(prefill_database=None, decode_database=None)
    assert kwargs["ranking_total_gpus"] == 64
    assert kwargs["ranking_metric_kind"] == "prefill_input_throughput"


@pytest.mark.parametrize(
    ("metric_kind", "osl", "cluster_gpus", "message"),
    [
        ("unknown", 1, 64, "disagg_ranking_metric_kind"),
        ("prefill_input_throughput", 2, 64, "requires osl=1"),
        ("prefill_input_throughput", 1, None, "requires disagg_ranking_total_gpus"),
    ],
)
def test_task_rejects_invalid_disagg_ranking_metric_contract(metric_kind, osl, cluster_gpus, message):
    with pytest.raises(ValueError, match=message):
        Task(
            serving_mode="disagg",
            prefill_model_path="Qwen/Qwen3-32B",
            prefill_system_name="h200_sxm",
            prefill_backend_name="vllm",
            decode_model_path="Qwen/Qwen3-32B",
            decode_system_name="h200_sxm",
            decode_backend_name="vllm",
            disagg_ranking_total_gpus=cluster_gpus,
            disagg_ranking_metric_kind=metric_kind,
            osl=osl,
        )
