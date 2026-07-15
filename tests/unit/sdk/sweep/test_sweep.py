# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sweep.py helpers and sweep_disagg placeholder.

Sweep output correctness is validated by the integration parity test
(``tests/integration/test_task_v1_v2_parity.py``) against the legacy CLI path;
the unit coverage here targets local control flow and terminal classification.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiconfigurator.sdk import config, sweep
from aiconfigurator.sdk.errors import (
    InsufficientMemoryError,
    KVCacheCapacityError,
    NoFeasibleConfigError,
)
from aiconfigurator.sdk.sweep import (
    _DEFAULT_AGG_BATCH_SCHEDULE,
    _agg_ctx_tokens_list,
    sweep_disagg,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _agg_ctx_tokens_list — parity with legacy base_backend._get_ctx_tokens_list_for_agg_sweep
# ---------------------------------------------------------------------------


def _legacy_ctx_tokens_list(isl: int, ctx_stride: int, enable_chunked_prefill: bool) -> list[int]:
    """Wrap the legacy helper on BaseBackend for parity comparison."""
    from aiconfigurator.sdk.backends.factory import get_backend

    legacy = get_backend("trtllm")  # any backend exposes the helper, it's on BaseBackend
    return legacy._get_ctx_tokens_list_for_agg_sweep(
        isl=isl,
        ctx_stride=ctx_stride,
        enable_chunked_prefill=enable_chunked_prefill,
    )


@pytest.mark.parametrize("isl", [1024, 2048, 4000, 8000, 16384])
@pytest.mark.parametrize("ctx_stride", [128, 256, 512, 1024])
@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
def test_agg_ctx_tokens_list_matches_legacy(isl, ctx_stride, enable_chunked_prefill):
    new = _agg_ctx_tokens_list(isl, ctx_stride, enable_chunked_prefill)
    old = _legacy_ctx_tokens_list(isl, ctx_stride, enable_chunked_prefill)
    assert new == old, (
        f"Mismatch for isl={isl}, ctx_stride={ctx_stride}, "
        f"enable_chunked_prefill={enable_chunked_prefill}\nnew={new}\nold={old}"
    )


# ---------------------------------------------------------------------------
# Batch schedule shape
# ---------------------------------------------------------------------------


def test_default_agg_batch_schedule_is_monotonic_and_capped():
    assert sorted(_DEFAULT_AGG_BATCH_SCHEDULE) == _DEFAULT_AGG_BATCH_SCHEDULE
    assert _DEFAULT_AGG_BATCH_SCHEDULE[0] == 1
    assert _DEFAULT_AGG_BATCH_SCHEDULE[-1] == 1024


def test_batch_list_builders_preserve_legacy_schedules_by_default():
    from aiconfigurator.sdk.sweep import (
        _DEFAULT_DECODE_BATCH_SCHEDULE,
        _agg_batch_list,
        _decode_batch_list,
    )

    assert _agg_batch_list(512, None) == [b for b in _DEFAULT_AGG_BATCH_SCHEDULE if b <= 512]
    assert _agg_batch_list(1024, None) == list(_DEFAULT_AGG_BATCH_SCHEDULE)
    assert _decode_batch_list(512, None) == [b for b in _DEFAULT_DECODE_BATCH_SCHEDULE if b <= 512]
    assert _decode_batch_list(2048, None) == _DEFAULT_DECODE_BATCH_SCHEDULE + [2048]


def test_batch_list_builders_support_exact_fixed_steps_and_caps():
    from aiconfigurator.sdk.sweep import _agg_batch_list, _decode_batch_list

    assert _agg_batch_list(5, 1) == [1, 2, 3, 4, 5]
    assert _agg_batch_list(10, 3) == [1, 4, 7, 10]
    assert _decode_batch_list(4, 1) == [1, 2, 3, 4]
    assert _decode_batch_list(9, 4) == [1, 5, 9]


@pytest.mark.parametrize(
    ("step", "expected_error"),
    (
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
        ("1", TypeError),
    ),
)
def test_batch_list_builders_reject_invalid_exact_integer_steps(step, expected_error):
    from aiconfigurator.sdk.sweep import _agg_batch_list, _decode_batch_list

    with pytest.raises(expected_error, match="batch_sweep_step"):
        _agg_batch_list(10, step)
    with pytest.raises(expected_error, match="batch_sweep_step"):
        _decode_batch_list(10, step)


# ---------------------------------------------------------------------------
# sweep_agg no-result classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("memory_states", "expected_error"),
    [
        ([(True, False), (True, False)], InsufficientMemoryError),
        ([(False, True), (True, False)], KVCacheCapacityError),
        ([(False, False), (True, False)], NoFeasibleConfigError),
    ],
)
def test_sweep_agg_classifies_no_result_outcomes(monkeypatch, memory_states, expected_error):
    summaries = []
    for model_oom, kv_cache_oom in memory_states:
        summary = MagicMock()
        summary.check_oom.return_value = model_oom
        summary.check_kv_cache_oom.return_value = kv_cache_oom
        summary.get_result_dict.return_value = {"ttft": 2.0, "tpot": 2.0}
        summaries.append(summary)

    monkeypatch.setattr(sweep, "get_backend", lambda _backend_name: MagicMock())
    monkeypatch.setattr(sweep, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(sweep, "predict_agg_worker", MagicMock(side_effect=summaries))

    with pytest.raises(expected_error):
        sweep.sweep_agg(
            model_path="test-model",
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=1.0, tpot=1.0),
            database=MagicMock(),
            backend_name="trtllm",
            model_config=config.ModelConfig(),
            parallel_config_list=[(1, 1, 1, 1, 1, 1), (2, 1, 1, 2, 1, 1)],
            max_batch_size=1,
            ctx_stride=1024,
        )


@pytest.mark.parametrize("failing_config_index", [0, 1])
def test_sweep_agg_propagates_unknown_exception_instead_of_returning_partial_results(
    monkeypatch,
    failing_config_index,
):
    class SentinelError(RuntimeError):
        pass

    sentinel = SentinelError("unknown aggregate configuration failure")
    call_index = 0

    def evaluate_parallel_config(**_kwargs):
        nonlocal call_index
        current_index = call_index
        call_index += 1
        if current_index == failing_config_index:
            raise sentinel
        return pd.DataFrame({"tokens/s/gpu": [1.0]}), True, True

    monkeypatch.setattr(sweep, "get_backend", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(sweep, "get_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(sweep, "_sweep_one_parallel_agg", evaluate_parallel_config)

    with pytest.raises(SentinelError) as raised:
        sweep.sweep_agg(
            model_path="test-model",
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=1.0, tpot=1.0),
            database=MagicMock(),
            backend_name="trtllm",
            model_config=config.ModelConfig(),
            parallel_config_list=[(1, 1, 1, 1, 1, 1), (2, 1, 1, 2, 1, 1)],
            max_batch_size=1,
            ctx_stride=1024,
        )

    assert raised.value is sentinel


def test_sweep_agg_point_config_preserves_multimodal_fields(monkeypatch):
    """Regression for NVBug 6401839: the agg per-batch RuntimeConfig must carry
    every multimodal field from the base runtime_config. The old field-by-field
    construction dropped image_height/width, num_images_per_request, and
    num_image_tokens, zeroing the image encoder workload in agg while disagg
    (which deep-copies) stayed correct."""
    captured: list[config.RuntimeConfig] = []

    def _record(*, runtime_config, **_kwargs):
        captured.append(runtime_config)
        summary = MagicMock()
        summary.check_oom.return_value = False
        summary.check_kv_cache_oom.return_value = False
        summary.get_result_dict.return_value = {"ttft": 1.0, "tpot": 1.0}
        summary.get_per_ops_source.return_value = {}
        return summary

    monkeypatch.setattr(sweep, "get_backend", lambda _backend_name: MagicMock())
    monkeypatch.setattr(sweep, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(sweep, "predict_agg_worker", _record)

    base_rt = config.RuntimeConfig(
        isl=256,
        osl=256,
        ttft=1e9,
        tpot=1e9,
        image_height=1024,
        image_width=1024,
        num_images_per_request=2,
        num_image_tokens=333,
        seq_imbalance_correction_scale=1.5,
        engine_step_backend="rust",
    )

    sweep.sweep_agg(
        model_path="test-model",
        runtime_config=base_rt,
        database=MagicMock(),
        backend_name="trtllm",
        model_config=config.ModelConfig(),
        parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        max_batch_size=1,
        ctx_stride=1024,
    )

    assert captured, "expected at least one agg point to be evaluated"
    for point_rt in captured:
        assert point_rt.image_height == 1024
        assert point_rt.image_width == 1024
        assert point_rt.num_images_per_request == 2
        assert point_rt.num_image_tokens == 333
        # Non-multimodal fields must survive too (the deep-copy carries them all).
        assert point_rt.seq_imbalance_correction_scale == 1.5
        assert point_rt.engine_step_backend == "rust"
        assert point_rt.batch_size == 1


def test_sweep_agg_step_one_evaluates_every_batch_through_the_cap(monkeypatch):
    captured: list[int] = []

    def _record(*, runtime_config, **_kwargs):
        captured.append(runtime_config.batch_size)
        summary = MagicMock()
        summary.check_oom.return_value = False
        summary.check_kv_cache_oom.return_value = False
        summary.get_result_dict.return_value = {"ttft": 1.0, "tpot": 1.0}
        summary.get_per_ops_source.return_value = {}
        return summary

    monkeypatch.setattr(sweep, "get_backend", lambda _backend_name: MagicMock())
    monkeypatch.setattr(sweep, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(sweep, "predict_agg_worker", _record)

    sweep.sweep_agg(
        model_path="test-model",
        runtime_config=config.RuntimeConfig(isl=1024, osl=8, ttft=1e9, tpot=1e9),
        database=MagicMock(),
        backend_name="trtllm",
        model_config=config.ModelConfig(),
        parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        max_batch_size=5,
        ctx_stride=1024,
        batch_sweep_step=1,
    )

    assert sorted(set(captured)) == [1, 2, 3, 4, 5]


def test_sweep_agg_stops_after_the_first_batch_whose_smallest_context_is_oom(monkeypatch):
    captured: list[int] = []

    def _record(*, runtime_config, **_kwargs):
        batch_size = runtime_config.batch_size
        captured.append(batch_size)
        summary = MagicMock()
        summary.check_oom.return_value = batch_size >= 3
        summary.check_kv_cache_oom.return_value = False
        summary.get_result_dict.return_value = {"ttft": 1.0, "tpot": 1.0}
        summary.get_per_ops_source.return_value = {}
        return summary

    monkeypatch.setattr(sweep, "get_backend", lambda _backend_name: MagicMock())
    monkeypatch.setattr(sweep, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(sweep, "predict_agg_worker", _record)

    sweep.sweep_agg(
        model_path="test-model",
        runtime_config=config.RuntimeConfig(isl=1024, osl=8, ttft=1e9, tpot=1e9),
        database=MagicMock(),
        backend_name="trtllm",
        model_config=config.ModelConfig(),
        parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        max_batch_size=5,
        ctx_stride=1024,
        batch_sweep_step=1,
    )

    assert captured == [1, 2, 3]


# ---------------------------------------------------------------------------
# sweep_disagg validation
# ---------------------------------------------------------------------------


def test_sweep_disagg_rejects_invalid_max_prefill_gpus():
    with pytest.raises(ValueError, match="max_prefill_gpus must be > 0"):
        sweep_disagg(
            model_path="x",
            runtime_config=None,
            prefill_database=None,
            prefill_backend_name="trtllm",
            prefill_model_config=None,
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="trtllm",
            decode_model_config=None,
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            max_prefill_gpus=0,
        )


def test_sweep_disagg_rejects_invalid_max_decode_gpus():
    with pytest.raises(ValueError, match="max_decode_gpus must be > 0"):
        sweep_disagg(
            model_path="x",
            runtime_config=None,
            prefill_database=None,
            prefill_backend_name="trtllm",
            prefill_model_config=None,
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="trtllm",
            decode_model_config=None,
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            max_decode_gpus=-5,
        )


def test_sweep_disagg_rejects_empty_num_worker_lists():
    """Empty worker lists silently skipped the rate-match inner loop in earlier
    versions; now fail loud to avoid surprising zero-result sweeps."""
    with pytest.raises(ValueError, match="non-empty prefill_num_worker_list and decode_num_worker_list"):
        sweep_disagg(
            model_path="x",
            runtime_config=None,
            prefill_database=None,
            prefill_backend_name="trtllm",
            prefill_model_config=None,
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="trtllm",
            decode_model_config=None,
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            prefill_num_worker_list=[],
            decode_num_worker_list=[1, 2, 4],
        )


def _run_disagg_sweep(
    monkeypatch,
    *,
    prefill_candidates,
    decode_candidates,
    runtime_config=None,
    **kwargs,
):
    monkeypatch.setattr(
        sweep,
        "_get_disagg_worker_candidates",
        MagicMock(side_effect=[prefill_candidates, decode_candidates]),
    )
    return sweep_disagg(
        model_path="test-model",
        runtime_config=runtime_config or config.RuntimeConfig(isl=1024, osl=1, ttft=200.0, tpot=10.0),
        prefill_database=MagicMock(),
        prefill_backend_name="vllm",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        prefill_latency_correction=1.0,
        decode_database=MagicMock(),
        decode_backend_name="vllm",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        decode_latency_correction=1.0,
        prefill_num_worker_list=[1],
        decode_num_worker_list=[1],
        **kwargs,
    )


def _run_fixed_cluster_disagg_sweep(monkeypatch, *, prefill_candidates, decode_candidates, **kwargs):
    return _run_disagg_sweep(
        monkeypatch,
        prefill_candidates=prefill_candidates,
        decode_candidates=decode_candidates,
        ranking_total_gpus=64,
        **kwargs,
    )


def test_sweep_disagg_raises_sla_error_when_fixed_cluster_prefill_misses_ttft(monkeypatch):
    prefill_candidates = pd.DataFrame({"ttft": [371.929], "osl": [1]})
    decode_candidates = pd.DataFrame({"tpot": [1.0], "osl": [1]})

    with pytest.raises(NoFeasibleConfigError, match="TTFT/TPOT"):
        _run_fixed_cluster_disagg_sweep(
            monkeypatch,
            prefill_candidates=prefill_candidates,
            decode_candidates=decode_candidates,
            ranking_metric_kind="prefill_input_throughput",
        )


def test_sweep_disagg_raises_sla_error_when_fixed_cluster_decode_misses_tpot(monkeypatch):
    prefill_candidates = pd.DataFrame({"ttft": [100.0], "osl": [1]})
    decode_candidates = pd.DataFrame({"tpot": [100.0], "osl": [1]})

    with pytest.raises(NoFeasibleConfigError, match="TTFT/TPOT"):
        _run_fixed_cluster_disagg_sweep(
            monkeypatch,
            prefill_candidates=prefill_candidates,
            decode_candidates=decode_candidates,
        )


def test_sweep_disagg_does_not_relabel_worker_allocation_failure_as_sla(monkeypatch):
    prefill_candidates = pd.DataFrame({"ttft": [100.0], "osl": [1]})
    decode_candidates = pd.DataFrame({"tpot": [1.0], "osl": [1]})
    monkeypatch.setattr(sweep, "_find_best_disagg_cluster_under_constraint", MagicMock(return_value=None))

    with pytest.raises(RuntimeError, match="worker allocation"):
        _run_fixed_cluster_disagg_sweep(
            monkeypatch,
            prefill_candidates=prefill_candidates,
            decode_candidates=decode_candidates,
            ranking_metric_kind="prefill_input_throughput",
        )


def test_sweep_disagg_prefers_allocation_error_when_any_fixed_cluster_pair_meets_sla(monkeypatch):
    prefill_candidates = pd.DataFrame({"ttft": [100.0], "osl": [1]})
    decode_candidates = pd.DataFrame({"tpot": [1.0], "osl": [1]})
    sla_error = NoFeasibleConfigError("TTFT miss")
    monkeypatch.setattr(
        sweep,
        "_find_best_disagg_cluster_under_constraint",
        MagicMock(side_effect=[sla_error, None]),
    )

    with pytest.raises(RuntimeError, match="worker allocation"):
        _run_fixed_cluster_disagg_sweep(
            monkeypatch,
            prefill_candidates=prefill_candidates,
            decode_candidates=decode_candidates,
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=200.0, tpot=[10.0, 20.0]),
            ranking_metric_kind="prefill_input_throughput",
        )


def test_sweep_disagg_reports_sla_when_all_fixed_cluster_pairs_miss(monkeypatch):
    prefill_candidates = pd.DataFrame({"ttft": [100.0], "osl": [1]})
    decode_candidates = pd.DataFrame({"tpot": [1.0], "osl": [1]})
    monkeypatch.setattr(
        sweep,
        "_find_best_disagg_cluster_under_constraint",
        MagicMock(
            side_effect=[
                NoFeasibleConfigError("TTFT miss at first pair"),
                NoFeasibleConfigError("TPOT miss at second pair"),
            ]
        ),
    )

    with pytest.raises(NoFeasibleConfigError, match="TTFT/TPOT"):
        _run_fixed_cluster_disagg_sweep(
            monkeypatch,
            prefill_candidates=prefill_candidates,
            decode_candidates=decode_candidates,
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=200.0, tpot=[10.0, 20.0]),
            ranking_metric_kind="prefill_input_throughput",
        )


def test_sweep_disagg_legacy_path_preserves_empty_result_semantics(monkeypatch):
    prefill_candidates = pd.DataFrame({"ttft": [300.0], "osl": [1]})
    decode_candidates = pd.DataFrame({"tpot": [1.0], "osl": [1]})

    result = _run_disagg_sweep(
        monkeypatch,
        prefill_candidates=prefill_candidates,
        decode_candidates=decode_candidates,
    )

    assert result.empty
    assert list(result.columns) == list(sweep.common.ColumnsDisagg)


def test_sweep_disagg_autoscale_preserves_empty_result_semantics(monkeypatch):
    prefill_candidates = pd.DataFrame({"ttft": [100.0], "osl": [1]})
    decode_candidates = pd.DataFrame({"tpot": [1.0], "osl": [1]})
    monkeypatch.setattr("aiconfigurator.sdk.picking.pick_autoscale", MagicMock(return_value={"best_config_df": None}))

    result = _run_disagg_sweep(
        monkeypatch,
        prefill_candidates=prefill_candidates,
        decode_candidates=decode_candidates,
        autoscale=True,
    )

    assert result.empty
    assert list(result.columns) == list(sweep.common.ColumnsDisagg)


def test_sweep_disagg_fails_fast_if_worker_candidate_builder_returns_empty(monkeypatch):
    with pytest.raises(RuntimeError, match="worker candidate builder returned an empty result"):
        _run_fixed_cluster_disagg_sweep(
            monkeypatch,
            prefill_candidates=pd.DataFrame(),
            decode_candidates=pd.DataFrame({"tpot": [1.0], "osl": [1]}),
            ranking_metric_kind="prefill_input_throughput",
        )


@pytest.mark.parametrize(
    ("parallel_config_list", "b_list", "message"),
    [
        ([], [1], "parallel_config_list must be non-empty"),
        ([(1, 1, 1, 1, 1, 1)], [], "b_list must be non-empty"),
    ],
)
def test_disagg_worker_candidate_builder_rejects_empty_search_axes(parallel_config_list, b_list, message):
    with pytest.raises(ValueError, match=message):
        sweep._get_disagg_worker_candidates(
            model_path="test-model",
            model_config=config.ModelConfig(),
            parallel_config_list=parallel_config_list,
            b_list=b_list,
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=200.0, tpot=10.0),
            role="prefill",
            database=MagicMock(),
            backend_name="vllm",
            latency_correction=1.0,
        )


def test_disagg_worker_candidate_builder_does_not_relabel_internal_empty_result_as_sla(monkeypatch):
    summary = MagicMock()
    summary.check_oom.return_value = False
    summary.get_summary_df.return_value = pd.DataFrame()
    monkeypatch.setattr(sweep, "get_backend", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(sweep, "get_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(sweep, "predict_disagg_worker", MagicMock(return_value=summary))

    with pytest.raises(RuntimeError, match="produced no rows despite observing a non-OOM candidate"):
        sweep._get_disagg_worker_candidates(
            model_path="test-model",
            model_config=config.ModelConfig(),
            parallel_config_list=[(1, 1, 1, 1, 1, 1)],
            b_list=[1],
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=200.0, tpot=10.0),
            role="prefill",
            database=MagicMock(),
            backend_name="vllm",
            latency_correction=1.0,
        )


@pytest.mark.parametrize("failing_config_index", [0, 1])
def test_disagg_worker_candidate_builder_propagates_unknown_exception_instead_of_partial_results(
    monkeypatch,
    failing_config_index,
):
    class SentinelError(RuntimeError):
        pass

    sentinel = SentinelError("unknown disaggregate configuration failure")
    model_call_index = 0

    def get_model_for_parallel_config(**_kwargs):
        nonlocal model_call_index
        current_index = model_call_index
        model_call_index += 1
        if current_index == failing_config_index:
            raise sentinel
        return MagicMock()

    summary = MagicMock()
    summary.check_oom.return_value = False
    summary.get_summary_df.return_value = pd.DataFrame(
        [[0] * len(sweep.common.ColumnsStatic)],
        columns=sweep.common.ColumnsStatic,
    )
    monkeypatch.setattr(sweep, "get_backend", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(sweep, "get_model", get_model_for_parallel_config)
    monkeypatch.setattr(sweep, "predict_disagg_worker", MagicMock(return_value=summary))

    with pytest.raises(SentinelError) as raised:
        sweep._get_disagg_worker_candidates(
            model_path="test-model",
            model_config=config.ModelConfig(),
            parallel_config_list=[(1, 1, 1, 1, 1, 1), (2, 1, 1, 2, 1, 1)],
            b_list=[1],
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=200.0, tpot=10.0),
            role="prefill",
            database=MagicMock(),
            backend_name="vllm",
            latency_correction=1.0,
        )

    assert raised.value is sentinel
