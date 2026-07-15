import importlib
import re
from dataclasses import replace
from pathlib import Path

import pytest

from aiconfigurator.sdk import common, perf_database
from aiconfigurator.sdk.errors import InsufficientMemoryError, KVCacheCapacityError, NoFeasibleConfigError
from aiconfigurator.sdk.task_v2 import SinglePointEvaluation, Task

SYSTEMS_DIR = Path(__file__).parents[2] / "performance" / "aic_roofline_pareto" / "systems"
H800_SPEC_PATH = SYSTEMS_DIR / "h800_sxm.yaml"
RUNNER_PATH = SYSTEMS_DIR.parent / "run_step4_comparison.py"


def _comparison_runner():
    assert RUNNER_PATH.is_file(), f"Missing Step4 comparison runner: {RUNNER_PATH}"
    return importlib.import_module("tests.performance.aic_roofline_pareto.run_step4_comparison")


def test_simulated_h800_system_spec_matches_roofline_contract():
    assert H800_SPEC_PATH.is_file(), f"Missing simulated H800 system spec: {H800_SPEC_PATH}"

    spec = perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR])

    assert spec["metadata"]["simulation_status"] == "simulated"
    assert spec["gpu"] == {
        "mem_bw": 3_350_000_000_000,
        "mem_bw_empirical_scaling_factor": 0.8,
        "mem_empirical_constant_latency": 0.000003,
        "mem_capacity": 85_899_345_920,
        "bfloat16_tc_flops": 989_500_000_000_000,
        "int8_tc_flops": 1_979_000_000_000_000,
        "fp8_tc_flops": 1_979_000_000_000_000,
        "power": 700,
        "sm_version": 90,
    }
    assert spec["node"] == {
        "num_gpus_per_node": 8,
        "inter_node_bw": 50_000_000_000,
        "intra_node_bw": 200_000_000_000,
        "pcie_bw": 64_000_000_000,
        "p2p_latency": 0.00001,
    }


def test_simulated_h800_is_classified_as_hopper():
    assert H800_SPEC_PATH.is_file(), f"Missing simulated H800 system spec: {H800_SPEC_PATH}"

    previous_paths = perf_database.get_systems_paths()
    try:
        perf_database.set_systems_paths([SYSTEMS_DIR, *previous_paths])
        assert perf_database.is_hopper_system("h800_sxm")
    finally:
        perf_database.set_systems_paths(previous_paths)


def test_dsv4_pro_vllm_native_checkpoint_uses_hopper_moe_quant_on_simulated_h800_sol():
    assert H800_SPEC_PATH.is_file(), f"Missing simulated H800 system spec: {H800_SPEC_PATH}"

    previous_paths = perf_database.get_systems_paths()
    try:
        perf_database.set_systems_paths([SYSTEMS_DIR, *previous_paths])
        task = Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V4-Pro",
            system_name="h800_sxm",
            backend_name="vllm",
            backend_version="0.22.0",
            database_mode="SOL",
        )

        task.validate()
    finally:
        perf_database.set_systems_paths(previous_paths)

    assert task.moe_quant_mode == common.MoEQuantMode.w4a16_mxfp4


def test_common_vllm_parallel_rows_match_authoritative_21_row_space():
    runner = _comparison_runner()

    rows = runner.build_common_vllm_parallel_rows()
    signatures = {
        (row.pattern, row.tp, row.dp, row.pp, row.moe_tp, row.moe_ep, row.cp, row.worker_gpus) for row in rows
    }
    expected = {
        ("B", 1, 1, 1, 1, 1, 1, 1),
        ("B", 2, 1, 1, 2, 1, 1, 2),
        ("B", 4, 1, 1, 4, 1, 1, 4),
        ("B", 8, 1, 1, 8, 1, 1, 8),
        ("A", 1, 2, 1, 1, 2, 1, 2),
        ("A", 1, 4, 1, 1, 4, 1, 4),
        ("A", 1, 8, 1, 1, 8, 1, 8),
        ("A", 1, 16, 1, 1, 16, 1, 16),
        ("A", 1, 32, 1, 1, 32, 1, 32),
        ("A", 2, 1, 1, 1, 2, 1, 2),
        ("A", 2, 2, 1, 1, 4, 1, 4),
        ("A", 2, 4, 1, 1, 8, 1, 8),
        ("A", 2, 8, 1, 1, 16, 1, 16),
        ("A", 2, 16, 1, 1, 32, 1, 32),
        ("A", 4, 1, 1, 1, 4, 1, 4),
        ("A", 4, 2, 1, 1, 8, 1, 8),
        ("A", 4, 4, 1, 1, 16, 1, 16),
        ("A", 4, 8, 1, 1, 32, 1, 32),
        ("A", 8, 1, 1, 1, 8, 1, 8),
        ("A", 8, 2, 1, 1, 16, 1, 16),
        ("A", 8, 4, 1, 1, 32, 1, 32),
    }

    assert signatures == expected
    assert len(rows) == 21
    assert sum(row.pattern == "A" for row in rows) == 17
    assert sum(row.pattern == "B" for row in rows) == 4
    assert sum(row.moe_ep > 8 for row in rows) == 8
    assert sum(row.moe_ep == 16 for row in rows) == 4
    assert sum(row.moe_ep == 32 for row in rows) == 4
    assert max(row.worker_gpus for row in rows) == 32


def test_disagg_parallel_pair_groups_match_authoritative_cartesian_product():
    runner = _comparison_runner()

    pairs = runner.pair_disagg_parallel_rows()

    assert set(pairs) == {"AA", "AB", "BA", "BB"}
    assert {name: len(rows) for name, rows in pairs.items()} == {
        "AA": 289,
        "AB": 68,
        "BA": 68,
        "BB": 16,
    }
    assert sum(len(rows) for rows in pairs.values()) == 441
    for name, group in pairs.items():
        assert all(pair.prefill.pattern == name[0] for pair in group)
        assert all(pair.decode.pattern == name[1] for pair in group)


def test_matrix_points_cover_exact_primary_and_decode_smoke_contract():
    runner = _comparison_runner()

    points = runner.build_matrix_points()
    mode_runs = runner.build_mode_run_specs()

    assert len(points) == 240
    assert len(mode_runs) == 480
    assert {point.model for point in points} == {
        "stepfun-ai/Step4",
        "deepseek-ai/DeepSeek-V4-Pro",
    }
    assert {point.system for point in points} == {
        "gb300",
        "h200_sxm",
        "h100_sxm",
        "h800_sxm",
    }
    assert {point.ttft_sla_ms for point in points} == {200, 500, 1000, 2000, 5000}
    assert {run.serving_mode for run in mode_runs} == {"agg", "disagg"}
    assert all(point.backend == "vllm" for point in points)
    assert all(point.backend_version == "0.22.0" for point in points)
    assert all(point.engine_step_backend == "python" for point in points)
    assert all(point.database_mode == "SOL" for point in points)
    assert all(point.total_gpus == 64 for point in points)
    assert all(point.prefix == 0 for point in points)
    assert all(point.nextn == 0 for point in points)
    assert all(point.tpot_ms == 50_000 for point in points)
    assert all(point.pareto_sweep is False for point in points)
    assert all(point.chunked_prefill is False for point in points)

    primary = [point for point in points if point.workload_kind == "primary"]
    decode_smoke = [point for point in points if point.workload_kind == "decode_smoke"]
    assert len(primary) == 200
    assert len(decode_smoke) == 40
    assert {point.isl for point in primary} == {4096, 16384, 65536, 262144, 1048576}
    assert {point.osl for point in primary} == {1}
    assert {(point.isl, point.osl) for point in decode_smoke} == {(4096, 1024)}


@pytest.mark.parametrize(
    ("experiment", "serving_mode", "prefill_pattern", "decode_pattern"),
    [
        ("agg_patternA", "agg", "A", None),
        ("agg_patternB", "agg", "B", None),
        ("disagg_AA", "disagg", "A", "A"),
        ("disagg_AB", "disagg", "A", "B"),
        ("disagg_BA", "disagg", "B", "A"),
        ("disagg_BB", "disagg", "B", "B"),
    ],
)
def test_build_comparison_task_materializes_exact_experiment_contract(
    experiment,
    serving_mode,
    prefill_pattern,
    decode_pattern,
    monkeypatch,
):
    runner = _comparison_runner()
    monkeypatch.setenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", "rust")
    point = next(
        point
        for point in runner.build_matrix_points()
        if point.model == "stepfun-ai/Step4"
        and point.system == "h200_sxm"
        and point.workload_kind == "primary"
        and point.isl == 4096
        and point.ttft_sla_ms == 500
    )
    caps = runner.BatchCaps(agg=2048, prefill=32, decode=2048)

    task = runner.build_comparison_task(
        runner.ModeRunSpec(point=point, serving_mode=serving_mode),
        experiment=experiment,
        caps=caps,
    )

    assert task.serving_mode == serving_mode
    assert task.isl == 4096
    assert task.osl == 1
    assert task.prefix == 0
    assert task.ttft == 500
    assert task.tpot == 50_000
    assert task.pareto_sweep is False
    assert task.total_gpus == 64
    assert task.database_mode == "SOL"
    assert task.engine_step_backend == "python"
    assert task.build_runtime_config().engine_step_backend == "python"
    assert task.batch_sweep_step == 1
    assert task.nextn == 0
    assert task.prefill_latency_correction == 1.0
    assert task.decode_latency_correction == 1.0
    assert task.rate_match_prefill_degradation == 1.0
    assert task.rate_match_decode_degradation == 1.0
    assert task.autoscale_ttft_correction_factor == 1.0

    if serving_mode == "agg":
        assert task.disagg_ranking_total_gpus is None
        assert task.model_path == point.model
        assert task.system_name == point.system
        assert task.backend_name == "vllm"
        assert task.backend_version == "0.22.0"
        assert task.enable_chunked_prefill is False
        assert task.agg_max_batch_size == 2048
        rows = tuple(task.iter_parallel("agg"))
        assert len(rows) == (17 if prefill_pattern == "A" else 4)
    else:
        assert task.disagg_ranking_total_gpus == 64
        assert task.prefill_model_path == task.decode_model_path == point.model
        assert task.prefill_system_name == task.decode_system_name == point.system
        assert task.prefill_backend_name == task.decode_backend_name == "vllm"
        assert task.prefill_backend_version == task.decode_backend_version == "0.22.0"
        assert task.prefill_enable_chunked_prefill is False
        assert task.prefill_max_batch_size == 32
        assert task.decode_max_batch_size == 2048
        assert task.num_gpu_per_replica == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
        assert task.max_gpu_per_replica == 64
        assert task.max_prefill_workers == 64
        assert task.max_decode_workers == 64
        prefill_rows = tuple(task.iter_parallel("prefill"))
        decode_rows = tuple(task.iter_parallel("decode"))
        assert len(prefill_rows) == (17 if prefill_pattern == "A" else 4)
        assert len(decode_rows) == (17 if decode_pattern == "A" else 4)


def test_build_comparison_task_rejects_unknown_or_mismatched_experiment():
    runner = _comparison_runner()
    point = runner.build_matrix_points()[0]

    with pytest.raises(ValueError, match="Unknown comparison experiment"):
        runner.build_comparison_task(
            runner.ModeRunSpec(point=point, serving_mode="agg"),
            experiment="agg_patternC",
        )

    with pytest.raises(ValueError, match="does not match serving mode"):
        runner.build_comparison_task(
            runner.ModeRunSpec(point=point, serving_mode="agg"),
            experiment="disagg_AA",
        )


@pytest.mark.parametrize(
    ("num_total_gpus", "expected_replicas", "expected_used", "expected_unused", "expected_throughput"),
    [
        (24, 2, 48, 16, 75.0),
        (40, 1, 40, 24, 62.5),
        (64, 1, 64, 0, 100.0),
    ],
)
def test_derive_cluster_allocation_uses_only_complete_replicas(
    num_total_gpus,
    expected_replicas,
    expected_used,
    expected_unused,
    expected_throughput,
):
    runner = _comparison_runner()

    allocation = runner.derive_cluster_allocation(
        tokens_per_second_per_gpu=100.0,
        num_total_gpus=num_total_gpus,
        total_gpus=64,
    )

    assert allocation.replicas == expected_replicas
    assert allocation.total_gpus_used == expected_used
    assert allocation.unused_gpus == expected_unused
    assert allocation.tokens_per_second_per_gpu_cluster == expected_throughput


@pytest.mark.parametrize(
    ("tokens_per_second_per_gpu", "num_total_gpus", "total_gpus", "error_type", "message"),
    [
        (100.0, 0, 64, ValueError, "num_total_gpus must be positive"),
        (100.0, -1, 64, ValueError, "num_total_gpus must be positive"),
        (100.0, 65, 64, ValueError, "exceeds total_gpus"),
        (100.0, 1, 0, ValueError, "total_gpus must be positive"),
        (100.0, "1", 64, TypeError, "num_total_gpus must be a positive integer"),
        (100.0, 1, "64", TypeError, "total_gpus must be a positive integer"),
        (float("nan"), 1, 64, ValueError, "must be finite"),
        ("100", 1, 64, TypeError, "finite number"),
    ],
)
def test_derive_cluster_allocation_rejects_invalid_evidence(
    tokens_per_second_per_gpu,
    num_total_gpus,
    total_gpus,
    error_type,
    message,
):
    runner = _comparison_runner()

    with pytest.raises(error_type, match=message):
        runner.derive_cluster_allocation(
            tokens_per_second_per_gpu=tokens_per_second_per_gpu,
            num_total_gpus=num_total_gpus,
            total_gpus=total_gpus,
        )


def _comparison_run_spec(
    runner,
    *,
    model,
    system,
    serving_mode,
    isl=4096,
    osl=1,
    ttft_sla_ms=500,
):
    point = next(
        point
        for point in runner.build_matrix_points()
        if point.model == model
        and point.system == system
        and point.isl == isl
        and point.osl == osl
        and point.ttft_sla_ms == ttft_sla_ms
    )
    return runner.ModeRunSpec(point=point, serving_mode=serving_mode)


def _successful_cap_result(runner, *, final_caps=None):
    initial = runner.BatchCaps()
    final = final_caps or initial
    history = (initial,) if final == initial else (initial, final)
    return runner.CapSearchResult(
        terminal_status="success",
        final_caps=final,
        cap_history=history,
        cap_rerun_count=len(history) - 1,
        cap_saturated=False,
        ranking_eligible=True,
    )


def _raw_agg_result(*, ttft=500.0):
    return {
        "model": "stepfun-ai/Step4",
        "isl": 4096,
        "osl": 1,
        "prefix": 0,
        "concurrency": 512,
        "request_rate": 200.0,
        "bs": 512,
        "global_bs": 8192,
        "ttft": ttft,
        "tpot": 9.0,
        "request_latency": ttft,
        "seq/s": 1600.0,
        "seq/s/gpu": 100.0,
        "tokens/s": 1600.0,
        "tokens/s/gpu": 100.0,
        "tokens/s/user": 1.0,
        "num_total_gpus": 16,
        "ctx_tokens": 4096,
        "tp": 1,
        "pp": 1,
        "dp": 16,
        "moe_tp": 1,
        "moe_ep": 16,
        "cp": 1,
        "parallel": "tp1pp1dp16moetp1moeep16cp1",
        "gemm": 1.0,
        "kvcache": 2.0,
        "fmha": 3.0,
        "moe": 4.0,
        "comm": 5.0,
        "memory": 60_000_000_000.0,
        "backend": "vllm",
        "version": "0.22.0",
        "system": "h800_sxm",
    }


def _raw_disagg_result(*, ttft=499.0):
    return {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "isl": 4096,
        "osl": 1,
        "prefix": 0,
        "concurrency": 256,
        "request_rate": 2400.0,
        "(p)bs": 8,
        "(p)global_bs": 32,
        "(p)workers": 2,
        "(d)bs": 128,
        "(d)global_bs": 128,
        "(d)workers": 2,
        "ttft": ttft,
        "tpot": 8.0,
        "request_latency": ttft,
        "seq/s": 2400.0,
        "seq/s/gpu": 100.0,
        "tokens/s": 2400.0,
        "tokens/s/gpu": 100.0,
        "tokens/s/user": 1.0,
        "(p)seq/s/worker": 1500.0,
        "(d)seq/s/worker": 1200.0,
        "num_total_gpus": 24,
        "(p)tp": 1,
        "(p)pp": 1,
        "(p)dp": 4,
        "(p)moe_tp": 1,
        "(p)moe_ep": 4,
        "(p)cp": 1,
        "(p)parallel": "tp1pp1dp4moetp1moeep4cp1",
        "(p)gemm": 1.0,
        "(p)kvcache": 2.0,
        "(p)fmha": 3.0,
        "(p)moe": 4.0,
        "(p)comm": 5.0,
        "(p)memory": 60_000_000_000.0,
        "(p)backend": "vllm",
        "(p)version": "0.22.0",
        "(p)system": "h200_sxm",
        "(d)tp": 8,
        "(d)pp": 1,
        "(d)dp": 1,
        "(d)moe_tp": 8,
        "(d)moe_ep": 1,
        "(d)parallel": "tp8pp1dp1moetp8moeep1",
        "(d)gemm": 1.5,
        "(d)kvcache": 2.5,
        "(d)fmha": 3.5,
        "(d)moe": 4.5,
        "(d)comm": 5.5,
        "(d)memory": 70_000_000_000.0,
        "(d)backend": "vllm",
        "(d)version": "0.22.0",
        "(d)system": "h200_sxm",
    }


def test_normalize_success_result_records_complete_aggregate_contract():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = _successful_cap_result(
        runner,
        final_caps=runner.BatchCaps(agg=2048, prefill=16, decode=1024),
    )
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)
    system_spec = perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR])

    normalized = runner.normalize_success_result(
        run_spec,
        experiment="agg_patternA",
        task=task,
        raw_row=_raw_agg_result(),
        cap_result=cap_result,
        system_spec=system_spec,
    )

    assert normalized["canonical_config_id"] == (
        "agg_patternA|tp=1|pp=1|dp=16|moe_tp=1|moe_ep=16|cp=1|bs=512|ctx_tokens=4096"
    )
    assert normalized["canonical_config_sort_key"] == ("agg_patternA", 1, 1, 16, 1, 16, 1, 512, 4096)
    assert normalized["simulation_status"] == "simulated"
    assert normalized["estimate_kind"] == "theoretical_sol_roofline"
    assert normalized["attention_approximation"] == "temporary_mla_substitute"
    assert normalized["attention_approximation_groups"] == {
        "full_mla_approx_layers": 23,
        "swa_mla_approx_layers": 69,
    }
    assert normalized["approximation_dominated"] is False
    assert normalized["tokens/s/gpu_cluster"] == 100.0
    assert normalized["ranking_metric_kind"] == "prefill_input_throughput"
    assert normalized["prefill_tokens/s"] == 67_108_864.0
    assert normalized["prefill_tokens/s/gpu"] == 4_194_304.0
    assert normalized["prefill_tokens/s/gpu_cluster"] == 4_194_304.0
    assert normalized["ranking_metric_value"] == 4_194_304.0
    assert normalized["replicas"] == 4
    assert normalized["total_gpus_used"] == 64
    assert normalized["unused_gpus"] == 0
    assert normalized["worker_gpus"] == 16
    assert normalized["best_batch_size"] == 512
    assert normalized["ttft_pass"] is True
    assert normalized["tpot_observed_only"] is True
    assert normalized["engine_step_backend"] == "python"
    assert normalized["ranking_eligible"] is True
    assert normalized["cap_rerun_count"] == 1
    assert normalized["agg_cap"] == 2048
    assert normalized["cap_history"] == (
        {"agg": 1024, "prefill": 16, "decode": 1024},
        {"agg": 2048, "prefill": 16, "decode": 1024},
    )
    assert "communication_evidence" not in normalized
    assert "effective_bandwidth_tiers" not in normalized
    assert {field: normalized[field] for field in runner.NEUTRAL_CORRECTIONS} == runner.NEUTRAL_CORRECTIONS


def test_normalize_success_result_records_complete_disaggregate_contract():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="deepseek-ai/DeepSeek-V4-Pro",
        system="h200_sxm",
        serving_mode="disagg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="disagg_AB", caps=cap_result.final_caps)
    system_spec = perf_database.load_system_spec("h200_sxm")

    normalized = runner.normalize_success_result(
        run_spec,
        experiment="disagg_AB",
        task=task,
        raw_row=_raw_disagg_result(),
        cap_result=cap_result,
        system_spec=system_spec,
    )

    assert normalized["canonical_config_id"] == (
        "disagg_AB|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=8|p_workers=2|"
        "d_tp=8|d_pp=1|d_dp=1|d_moe_tp=8|d_moe_ep=1|d_cp=1|d_bs=128|d_workers=2"
    )
    assert normalized["canonical_config_sort_key"] == (
        "disagg_AB",
        1,
        1,
        4,
        1,
        4,
        1,
        8,
        2,
        8,
        1,
        1,
        8,
        1,
        1,
        128,
        2,
    )
    assert normalized["simulation_status"] == "not_simulated"
    assert normalized["attention_approximation"] is None
    assert normalized["attention_approximation_groups"] == {}
    assert normalized["approximation_dominated"] is False
    assert normalized["tokens/s/gpu_cluster"] == 75.0
    assert normalized["ranking_metric_kind"] == "prefill_input_throughput"
    expected_prefill_tokens_per_second = 32 * 2 * 4096 / 0.499
    assert normalized["prefill_tokens/s"] == pytest.approx(expected_prefill_tokens_per_second)
    assert normalized["prefill_tokens/s/gpu"] == pytest.approx(expected_prefill_tokens_per_second / 24)
    assert normalized["prefill_tokens/s/gpu_cluster"] == pytest.approx(expected_prefill_tokens_per_second / 32)
    assert normalized["ranking_metric_value"] == pytest.approx(expected_prefill_tokens_per_second / 32)
    assert normalized["replicas"] == 2
    assert normalized["total_gpus_used"] == 48
    assert normalized["unused_gpus"] == 16
    assert normalized["prefill_worker_gpus"] == 4
    assert normalized["decode_worker_gpus"] == 8
    assert normalized["prefill_worker_count"] == 2
    assert normalized["decode_worker_count"] == 2
    assert normalized["prefill_batch_size"] == 8
    assert normalized["decode_batch_size"] == 128
    assert normalized["best_batch_size"] == {"prefill": 8, "decode": 128}
    assert normalized["ttft_pass"] is True
    assert normalized["tpot_observed_only"] is True
    assert normalized["engine_step_backend"] == "python"
    assert normalized["ranking_eligible"] is True
    assert "communication_evidence" not in normalized
    assert "effective_bandwidth_tiers" not in normalized


@pytest.mark.parametrize(
    ("serving_mode", "experiment", "raw_result", "expected_saturated"),
    [
        ("agg", "agg_patternA", lambda: _raw_agg_result() | {"bs": 2048, "global_bs": 32768}, ("agg",)),
        (
            "disagg",
            "disagg_AB",
            lambda: _raw_disagg_result() | {"(p)bs": 16, "(p)global_bs": 64},
            ("prefill",),
        ),
    ],
)
def test_normalize_success_result_marks_each_rows_own_cap_saturation(
    serving_mode,
    experiment,
    raw_result,
    expected_saturated,
):
    runner = _comparison_runner()
    model = "stepfun-ai/Step4" if serving_mode == "agg" else "deepseek-ai/DeepSeek-V4-Pro"
    system = "h800_sxm" if serving_mode == "agg" else "h200_sxm"
    run_spec = _comparison_run_spec(
        runner,
        model=model,
        system=system,
        serving_mode=serving_mode,
    )
    final_caps = runner.BatchCaps(agg=2048, prefill=16, decode=1024)
    cap_result = _successful_cap_result(runner, final_caps=final_caps)
    task = runner.build_comparison_task(run_spec, experiment=experiment, caps=final_caps)
    system_spec = perf_database.load_system_spec(
        system,
        systems_paths=[SYSTEMS_DIR] if system == "h800_sxm" else None,
    )

    normalized = runner.normalize_success_result(
        run_spec,
        experiment=experiment,
        task=task,
        raw_row=raw_result(),
        cap_result=cap_result,
        system_spec=system_spec,
    )

    assert normalized["saturated_cap_names"] == expected_saturated
    assert normalized["cap_saturated"] is True
    assert normalized["ranking_eligible"] is False
    assert "communication_evidence" not in normalized
    assert "effective_bandwidth_tiers" not in normalized


def test_normalize_success_result_rejects_row_batch_above_final_cap():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    final_caps = runner.BatchCaps(agg=1024, prefill=16, decode=1024)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=final_caps)
    raw_row = _raw_agg_result() | {"bs": 1025, "global_bs": 16400}

    with pytest.raises(ValueError, match="aggregate batch 1025 exceeds final agg cap 1024"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=raw_row,
            cap_result=_successful_cap_result(runner, final_caps=final_caps),
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


@pytest.mark.parametrize(
    ("serving_mode", "experiment", "raw_row_factory", "expected_ttft_pass"),
    [
        ("agg", "agg_patternA", _raw_agg_result, True),
        ("disagg", "disagg_AB", lambda: _raw_disagg_result(ttft=500.0), False),
    ],
)
def test_normalize_success_result_preserves_existing_ttft_boundary_semantics(
    serving_mode,
    experiment,
    raw_row_factory,
    expected_ttft_pass,
):
    runner = _comparison_runner()
    model = "stepfun-ai/Step4" if serving_mode == "agg" else "deepseek-ai/DeepSeek-V4-Pro"
    system = "h800_sxm" if serving_mode == "agg" else "h200_sxm"
    run_spec = _comparison_run_spec(
        runner,
        model=model,
        system=system,
        serving_mode=serving_mode,
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment=experiment, caps=cap_result.final_caps)
    system_spec = perf_database.load_system_spec(
        system,
        systems_paths=[SYSTEMS_DIR] if system == "h800_sxm" else None,
    )

    normalized = runner.normalize_success_result(
        run_spec,
        experiment=experiment,
        task=task,
        raw_row=raw_row_factory(),
        cap_result=cap_result,
        system_spec=system_spec,
    )

    assert normalized["ttft"] == 500.0
    assert normalized["ttft_pass"] is expected_ttft_pass
    assert normalized["ranking_eligible"] is expected_ttft_pass


@pytest.mark.parametrize(
    ("serving_mode", "experiment", "raw_row", "field"),
    [
        ("agg", "agg_patternA", _raw_agg_result(), "memory"),
        ("disagg", "disagg_AB", _raw_disagg_result(), "(d)memory"),
    ],
)
def test_normalize_success_result_rejects_non_finite_emitted_numeric_evidence(
    serving_mode,
    experiment,
    raw_row,
    field,
):
    runner = _comparison_runner()
    raw_row[field] = float("nan")
    model = "stepfun-ai/Step4" if serving_mode == "agg" else "deepseek-ai/DeepSeek-V4-Pro"
    system = "h800_sxm" if serving_mode == "agg" else "h200_sxm"
    run_spec = _comparison_run_spec(
        runner,
        model=model,
        system=system,
        serving_mode=serving_mode,
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment=experiment, caps=cap_result.final_caps)
    system_spec = perf_database.load_system_spec(
        system,
        systems_paths=[SYSTEMS_DIR] if system == "h800_sxm" else None,
    )

    with pytest.raises(ValueError, match=rf"{re.escape(field)} must be finite"):
        runner.normalize_success_result(
            run_spec,
            experiment=experiment,
            task=task,
            raw_row=raw_row,
            cap_result=cap_result,
            system_spec=system_spec,
        )


@pytest.mark.parametrize(
    ("serving_mode", "experiment", "raw_row", "field"),
    [
        ("agg", "agg_patternA", _raw_agg_result(), "global_bs"),
        ("disagg", "disagg_AB", _raw_disagg_result(), "(d)global_bs"),
    ],
)
def test_normalize_success_result_rejects_inconsistent_global_batch_arithmetic(
    serving_mode,
    experiment,
    raw_row,
    field,
):
    runner = _comparison_runner()
    raw_row[field] -= 1
    model = "stepfun-ai/Step4" if serving_mode == "agg" else "deepseek-ai/DeepSeek-V4-Pro"
    system = "h800_sxm" if serving_mode == "agg" else "h200_sxm"
    run_spec = _comparison_run_spec(
        runner,
        model=model,
        system=system,
        serving_mode=serving_mode,
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment=experiment, caps=cap_result.final_caps)
    system_spec = perf_database.load_system_spec(
        system,
        systems_paths=[SYSTEMS_DIR] if system == "h800_sxm" else None,
    )

    with pytest.raises(ValueError, match="global batch"):
        runner.normalize_success_result(
            run_spec,
            experiment=experiment,
            task=task,
            raw_row=raw_row,
            cap_result=cap_result,
            system_spec=system_spec,
        )


def test_normalize_success_result_rejects_missing_raw_fields():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)
    raw_row = _raw_agg_result()
    del raw_row["tokens/s"]

    with pytest.raises(ValueError, match=r"missing required fields: \['tokens/s'\]"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=raw_row,
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


def test_normalize_success_result_rejects_raw_model_identity_mismatch():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)
    raw_row = _raw_agg_result()
    raw_row["model"] = "deepseek-ai/DeepSeek-V4-Pro"

    with pytest.raises(ValueError, match="raw model mismatch"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=raw_row,
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


@pytest.mark.parametrize(("field", "value"), [("backend", "sglang"), ("version", "0.21.0")])
def test_normalize_success_result_rejects_raw_backend_contract_mismatch(field, value):
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)
    raw_row = _raw_agg_result()
    raw_row[field] = value

    with pytest.raises(ValueError, match=rf"raw {field} mismatch"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=raw_row,
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


def test_normalize_success_result_rejects_non_neutral_task_correction():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)
    task.prefill_latency_correction = 1.1

    with pytest.raises(ValueError, match=r"Task\.prefill_latency_correction mismatch"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=_raw_agg_result(),
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


def test_normalize_success_result_rejects_invalid_cap_evidence():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = replace(_successful_cap_result(runner), cap_saturated=True)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)

    with pytest.raises(ValueError, match="Successful cap result must be non-saturated"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=_raw_agg_result(),
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


def test_normalize_success_result_rejects_invalid_h800_simulation_metadata():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)
    system_spec = perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR])
    system_spec = {**system_spec, "metadata": {"simulation_status": "not_simulated"}}

    with pytest.raises(ValueError, match="h800_sxm system spec must declare"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=_raw_agg_result(),
            cap_result=cap_result,
            system_spec=system_spec,
        )


def test_normalize_success_result_rejects_incorrect_parallel_pattern():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=cap_result.final_caps)
    raw_row = _raw_agg_result()
    raw_row["moe_tp"] = 16
    raw_row["moe_ep"] = 1

    with pytest.raises(ValueError, match="does not match Pattern A"):
        runner.normalize_success_result(
            run_spec,
            experiment="agg_patternA",
            task=task,
            raw_row=raw_row,
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


def test_normalize_success_result_rejects_invalid_decode_cp_task_evidence():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="deepseek-ai/DeepSeek-V4-Pro",
        system="h200_sxm",
        serving_mode="disagg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="disagg_AB", caps=cap_result.final_caps)
    task.decode_cp_candidates = [1, 2]

    with pytest.raises(ValueError, match=r"Task\.decode_cp_candidates must provide exact decode cp evidence"):
        runner.normalize_success_result(
            run_spec,
            experiment="disagg_AB",
            task=task,
            raw_row=_raw_disagg_result(),
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h200_sxm"),
        )


def test_normalize_success_result_rejects_missing_fixed_cluster_ranking_contract():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="deepseek-ai/DeepSeek-V4-Pro",
        system="h200_sxm",
        serving_mode="disagg",
    )
    cap_result = _successful_cap_result(runner)
    task = runner.build_comparison_task(run_spec, experiment="disagg_AB", caps=cap_result.final_caps)
    task.disagg_ranking_total_gpus = None

    with pytest.raises(ValueError, match=r"Task\.disagg_ranking_total_gpus mismatch"):
        runner.normalize_success_result(
            run_spec,
            experiment="disagg_AB",
            task=task,
            raw_row=_raw_disagg_result(),
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h200_sxm"),
        )


def test_matrix_points_apply_neutral_corrections_and_explicit_mla_labels():
    runner = _comparison_runner()

    points = runner.build_matrix_points()

    assert runner.NEUTRAL_CORRECTIONS == {
        "prefill_latency_correction": 1.0,
        "decode_latency_correction": 1.0,
        "rate_match_prefill_degradation": 1.0,
        "rate_match_decode_degradation": 1.0,
        "autoscale_ttft_correction_factor": 1.0,
    }
    step4 = [point for point in points if point.model == "stepfun-ai/Step4"]
    deepseek = [point for point in points if point.model == "deepseek-ai/DeepSeek-V4-Pro"]
    assert all(point.attention_approximation == "temporary_mla_substitute" for point in step4)
    assert all(point.attention_approximation is None for point in deepseek)
    assert all(point.approximation_dominated == (point.isl >= 65_536) for point in step4)
    assert all(point.approximation_dominated is False for point in deepseek)


def test_aggregate_cap_expansion_repeats_until_rank_one_is_not_saturated():
    runner = _comparison_runner()
    seen_caps = []

    def evaluate(caps):
        seen_caps.append(caps)
        if caps.agg <= 2048:
            return runner.SearchAttempt(rank1_batch_sizes={"agg": caps.agg})
        return runner.SearchAttempt(rank1_batch_sizes={"agg": 3000})

    result = runner.expand_caps_until_terminal("agg", evaluate)

    assert [caps.agg for caps in seen_caps] == [1024, 2048, 4096]
    assert all(caps.prefill == 16 and caps.decode == 1024 for caps in seen_caps)
    assert result.terminal_status == "success"
    assert result.final_caps == runner.BatchCaps(agg=4096, prefill=16, decode=1024)
    assert result.cap_rerun_count == 2
    assert result.cap_saturated is False
    assert result.ranking_eligible is True


def test_cap_search_performs_one_detailed_rerun_after_two_saturated_attempts():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="agg",
        ttft_sla_ms=5000,
    )
    selected_batches = {2: 2, 4: 4, 8: 6}
    detailed_calls = []

    class FakeFrame:
        def __init__(self, row):
            self.row = row

        def to_dict(self, *, orient):
            assert orient == "records"
            return [self.row]

    class FakeTask:
        def __init__(self, row):
            self.row = row

        def run(self):
            return FakeFrame(self.row)

        def run_single_agg(self, **kwargs):
            detailed_calls.append(kwargs)
            return SinglePointEvaluation(
                row=dict(self.row),
                per_ops_data={
                    "mix_step": {"context_mla_attention": 2.0, "context_moe": 3.0},
                    "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
                },
                per_ops_source={
                    "mix_step": {"context_mla_attention": "sol", "context_moe": "sol"},
                },
            )

    def task_factory(_run_spec, *, experiment, caps):
        assert experiment == "agg_patternA"
        selected_batch = selected_batches[caps.agg]
        row = _raw_agg_result(ttft=100.0) | {
            "bs": selected_batch,
            "global_bs": selected_batch * 16,
        }
        return FakeTask(row)

    result = runner.run_experiment_cap_search(
        run_spec,
        experiment="agg_patternA",
        initial_caps=runner.BatchCaps(agg=2, prefill=16, decode=1024),
        task_factory=task_factory,
    )

    assert [evidence.status for evidence in result.attempt_evidence] == [
        "cap_saturated",
        "cap_saturated",
        "success",
    ]
    assert all(
        evidence.search_attempt.selected_evaluation is None and evidence.search_attempt.per_ops_evidence is None
        for evidence in result.attempt_evidence[:2]
    )
    assert result.attempt_evidence[-1].search_attempt.selected_evaluation is not None
    assert result.attempt_evidence[-1].search_attempt.per_ops_evidence is not None
    assert len(detailed_calls) == 1
    assert detailed_calls[0]["include_per_ops"] is True


def test_per_ops_totals_are_exactly_stable_across_canonical_json_key_order():
    runner = _comparison_runner()
    scheduling = {"num_mix_steps": 1.0, "num_genonly_steps": 0.0}
    insertion_order = {"z_moe": 1e16, "a_moe": 1.0, "b_moe": 1.0}
    canonical_order = {"a_moe": 1.0, "b_moe": 1.0, "z_moe": 1e16}

    def validate(phase_data):
        return runner.validate_per_ops_evidence(
            SinglePointEvaluation(
                row={},
                per_ops_data={"mix_step": phase_data, "scheduling": scheduling},
                per_ops_source={"mix_step": dict.fromkeys(phase_data, "sol")},
            ),
            serving_mode="agg",
            osl=1,
        )

    assert validate(insertion_order) == validate(canonical_order)


def test_disagg_cap_expansion_doubles_only_each_saturated_cap():
    runner = _comparison_runner()
    seen_caps = []

    def evaluate(caps):
        seen_caps.append(caps)
        if len(seen_caps) == 1:
            return runner.SearchAttempt(rank1_batch_sizes={"prefill": 16, "decode": 512})
        if len(seen_caps) == 2:
            return runner.SearchAttempt(rank1_batch_sizes={"prefill": 24, "decode": 1024})
        return runner.SearchAttempt(rank1_batch_sizes={"prefill": 24, "decode": 1500})

    result = runner.expand_caps_until_terminal("disagg", evaluate)

    assert seen_caps == [
        runner.BatchCaps(agg=1024, prefill=16, decode=1024),
        runner.BatchCaps(agg=1024, prefill=32, decode=1024),
        runner.BatchCaps(agg=1024, prefill=32, decode=2048),
    ]
    assert result.terminal_status == "success"
    assert result.final_caps == seen_caps[-1]
    assert result.cap_rerun_count == 2
    assert result.ranking_eligible is True


def test_every_experiment_completes_cap_expansion_before_global_ranking():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="disagg",
    )
    calls = []

    def evaluate(experiment, caps):
        calls.append((experiment, caps))
        experiment_calls = sum(1 for seen_experiment, _ in calls if seen_experiment == experiment)
        if experiment_calls == 1:
            return runner.SearchAttempt(rank1_batch_sizes={"prefill": caps.prefill, "decode": 512})
        return runner.SearchAttempt(rank1_batch_sizes={"prefill": 24, "decode": 512})

    results = runner.expand_caps_for_all_experiments(run_spec, evaluate)

    assert tuple(results) == ("disagg_AA", "disagg_AB", "disagg_BA", "disagg_BB")
    assert all(result.terminal_status == "success" for result in results.values())
    assert all(result.final_caps.prefill == 32 for result in results.values())
    assert {experiment: sum(1 for seen, _ in calls if seen == experiment) for experiment in results} == {
        "disagg_AA": 2,
        "disagg_AB": 2,
        "disagg_BA": 2,
        "disagg_BB": 2,
    }


def test_every_aggregate_pattern_completes_cap_expansion_independently():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="agg",
    )
    calls = []

    def evaluate(experiment, caps):
        calls.append((experiment, caps))
        return runner.SearchAttempt(rank1_batch_sizes={"agg": 512})

    results = runner.expand_caps_for_all_experiments(run_spec, evaluate)

    assert tuple(results) == ("agg_patternA", "agg_patternB")
    assert all(result.terminal_status == "success" for result in results.values())
    assert [experiment for experiment, _ in calls] == ["agg_patternA", "agg_patternB"]


def test_typed_canonical_sort_key_orders_numeric_two_before_sixteen():
    runner = _comparison_runner()
    base = {"pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "cp": 1, "bs": 1, "ctx_tokens": 4096}

    assert runner._canonical_agg_sort_key("agg_patternA", base | {"tp": 2}) < runner._canonical_agg_sort_key(
        "agg_patternA",
        base | {"tp": 16},
    )


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (InsufficientMemoryError("model does not fit"), "memory_infeasible"),
        (KVCacheCapacityError("KV cache does not fit"), "memory_infeasible"),
        (NoFeasibleConfigError("TTFT SLA infeasible"), "sla_infeasible"),
    ],
)
def test_cap_expansion_stops_on_explicit_known_terminal_error(error, expected_status):
    runner = _comparison_runner()
    calls = 0

    def evaluate(caps):
        nonlocal calls
        calls += 1
        if calls == 1:
            active = "agg" if expected_status == "memory_infeasible" else "prefill"
            if active == "agg":
                return runner.SearchAttempt(rank1_batch_sizes={"agg": caps.agg})
        raise error

    serving_mode = "agg" if expected_status == "memory_infeasible" else "disagg"
    result = runner.expand_caps_until_terminal(serving_mode, evaluate)

    assert result.terminal_status == expected_status
    assert result.ranking_eligible is False
    assert result.cap_saturated is False
    assert result.cap_rerun_count == (1 if serving_mode == "agg" else 0)


def test_error_classification_follows_explicit_exception_cause_chain():
    runner = _comparison_runner()

    try:
        raise RuntimeError("sweep wrapper") from InsufficientMemoryError("model does not fit")
    except RuntimeError as error:
        assert runner.classify_evaluation_error(error) == "memory_infeasible"


def test_unknown_evaluation_error_fails_fast_without_oom_relabeling():
    runner = _comparison_runner()
    unexpected = RuntimeError("unexpected evaluator failure")

    def evaluate(_caps):
        raise unexpected

    with pytest.raises(RuntimeError, match="unexpected evaluator failure") as raised:
        runner.expand_caps_until_terminal("agg", evaluate)

    assert raised.value is unexpected


@pytest.mark.parametrize(
    ("serving_mode", "rank1_batch_sizes", "message"),
    [
        ("agg", {}, "exactly.*agg"),
        ("agg", {"agg": 1025}, "exceeds active cap"),
        ("agg", {"agg": 0}, "positive integer"),
        ("disagg", {"prefill": 8}, "exactly.*prefill.*decode"),
        ("disagg", {"prefill": 8, "decode": 512, "agg": 1}, "exactly.*prefill.*decode"),
    ],
)
def test_cap_expansion_rejects_invalid_rank_one_batch_evidence(
    serving_mode,
    rank1_batch_sizes,
    message,
):
    runner = _comparison_runner()

    def evaluate(_caps):
        return runner.SearchAttempt(rank1_batch_sizes=rank1_batch_sizes)

    with pytest.raises(ValueError, match=message):
        runner.expand_caps_until_terminal(serving_mode, evaluate)


def test_cap_expansion_rejects_unknown_serving_mode():
    runner = _comparison_runner()

    with pytest.raises(ValueError, match="Unsupported serving mode"):
        runner.expand_caps_until_terminal("hybrid", lambda _caps: None)


def test_real_task_evaluator_uses_prefill_metric_and_defers_exact_rerun_until_terminal_success():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="agg",
        ttft_sla_ms=5000,
    )
    low = _raw_agg_result(ttft=100.0) | {
        "system": "h200_sxm",
        "bs": 16,
        "global_bs": 256,
        "tokens/s/gpu": 80.0,
    }
    high = _raw_agg_result(ttft=200.0) | {
        "system": "h200_sxm",
        "bs": 32,
        "global_bs": 512,
        "ctx_tokens": 8192,
        "tokens/s/gpu": 100.0,
    }
    exact_calls = []

    class FakeFrame:
        def to_dict(self, *, orient):
            assert orient == "records"
            return [low, high]

    class FakeTask:
        def run(self):
            return FakeFrame()

        def run_single_agg(self, **kwargs):
            exact_calls.append(kwargs)
            return SinglePointEvaluation(
                row=dict(low),
                per_ops_data={
                    "mix_step": {"context_mla_attention": 2.0, "context_moe": 3.0},
                    "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
                },
                per_ops_source={
                    "mix_step": {"context_mla_attention": "sol", "context_moe": "sol"},
                },
            )

    attempt = runner.evaluate_experiment_attempt(
        run_spec,
        experiment="agg_patternA",
        caps=runner.BatchCaps(),
        task_factory=lambda *_args, **_kwargs: FakeTask(),
    )

    assert attempt.rank1_batch_sizes == {"agg": 16}
    assert attempt.rank1_row == low
    assert attempt.candidate_rows == (low, high)
    assert attempt.selected_point_identity == ("agg_patternA", 1, 1, 16, 1, 16, 1, 16, 4096)
    assert attempt.selected_evaluation is None
    assert attempt.per_ops_evidence is None
    assert exact_calls == []


def test_derive_ranking_metric_evidence_uses_fresh_tokens_and_cluster_waste():
    runner = _comparison_runner()
    base = _comparison_run_spec(
        runner,
        model="deepseek-ai/DeepSeek-V4-Pro",
        system="h200_sxm",
        serving_mode="disagg",
    )
    run_spec = replace(base, point=replace(base.point, prefix=1024))
    row = _raw_disagg_result(ttft=500.0) | {
        "prefix": 1024,
        "(d)workers": 4,
        "num_total_gpus": 40,
    }

    evidence = runner.derive_ranking_metric_evidence(run_spec, row)

    expected_system = 32 * 2 * (4096 - 1024) / 0.5
    assert evidence == {
        "ranking_metric_kind": "prefill_input_throughput",
        "ranking_metric_value": expected_system / 64,
        "prefill_tokens/s": expected_system,
        "prefill_tokens/s/gpu": expected_system / 40,
        "prefill_tokens/s/gpu_cluster": expected_system / 64,
    }


def test_derive_ranking_metric_evidence_keeps_decode_smoke_on_output_throughput():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="agg",
        isl=4096,
        osl=1024,
        ttft_sla_ms=500,
    )
    row = _raw_agg_result(ttft=100.0) | {
        "system": "h200_sxm",
        "osl": 1024,
        "tokens/s/gpu": 100.0,
    }

    evidence = runner.derive_ranking_metric_evidence(run_spec, row)

    assert evidence == {
        "ranking_metric_kind": "output_token_throughput",
        "ranking_metric_value": 100.0,
    }


@pytest.mark.parametrize(
    ("mutation", "point_mutation", "message"),
    [
        ({"ttft": 0.0}, {}, "ttft_ms must be positive"),
        ({"global_bs": 0}, {}, "global_batch_size must be positive"),
        ({"prefix": 4096}, {"prefix": 4096}, "fresh input tokens must be positive"),
    ],
)
def test_derive_ranking_metric_evidence_rejects_invalid_prefill_inputs(mutation, point_mutation, message):
    runner = _comparison_runner()
    base = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="agg",
    )
    run_spec = replace(base, point=replace(base.point, **point_mutation))
    row = _raw_agg_result() | {"system": "h200_sxm", **mutation}

    with pytest.raises(ValueError, match=message):
        runner.derive_ranking_metric_evidence(run_spec, row)


def test_real_disaggregate_evaluator_defers_exact_selected_role_rerun():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="deepseek-ai/DeepSeek-V4-Pro",
        system="h200_sxm",
        serving_mode="disagg",
        ttft_sla_ms=5000,
    )
    selected = _raw_disagg_result(ttft=200.0)
    exact_calls = []

    class FakeFrame:
        def to_dict(self, *, orient):
            assert orient == "records"
            return [selected]

    class FakeTask:
        def run(self):
            return FakeFrame()

        def run_single_disagg(self, **kwargs):
            exact_calls.append(kwargs)
            return SinglePointEvaluation(
                row=dict(selected),
                per_ops_data={
                    "prefill": {"context_mla_attention": 2.0, "context_moe": 3.0},
                    "decode": {"generation_mla_bmm": 0.5, "generation_moe": 0.75},
                },
                per_ops_source={
                    "prefill": {"context_mla_attention": "sol", "context_moe": "sol"},
                    "decode": {"generation_mla_bmm": "sol", "generation_moe": "sol"},
                },
            )

    attempt = runner.evaluate_experiment_attempt(
        run_spec,
        experiment="disagg_AB",
        caps=runner.BatchCaps(),
        task_factory=lambda *_args, **_kwargs: FakeTask(),
    )

    assert attempt.rank1_batch_sizes == {"prefill": 8, "decode": 128}
    assert attempt.selected_evaluation is None
    assert attempt.per_ops_evidence is None
    assert exact_calls == []


def test_cap_expansion_preserves_every_successful_attempt_for_artifact_audit():
    runner = _comparison_runner()
    attempts = []

    def evaluate(caps):
        attempt = runner.SearchAttempt(
            rank1_batch_sizes={"agg": caps.agg if caps.agg == 1024 else 1536},
            candidate_rows=({"cap": caps.agg},),
            rank1_row={"cap": caps.agg},
        )
        attempts.append(attempt)
        return attempt

    result = runner.expand_caps_until_terminal("agg", evaluate)

    assert result.attempt_history == tuple(attempts)
    assert [attempt.rank1_row for attempt in result.attempt_history] == [
        {"cap": 1024},
        {"cap": 2048},
    ]
    assert [evidence.status for evidence in result.attempt_evidence] == ["cap_saturated", "success"]
    assert [evidence.caps.agg for evidence in result.attempt_evidence] == [1024, 2048]
    assert all(evidence.error_type is None and evidence.error_message is None for evidence in result.attempt_evidence)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (InsufficientMemoryError("model OOM"), "memory_infeasible"),
        (KVCacheCapacityError("KV OOM"), "memory_infeasible"),
        (NoFeasibleConfigError("TTFT SLA"), "sla_infeasible"),
    ],
)
def test_cap_expansion_records_typed_terminal_attempt_evidence(error, expected_status):
    runner = _comparison_runner()

    def evaluate(_caps):
        raise error

    result = runner.expand_caps_until_terminal(
        "agg",
        evaluate,
        experiment="agg_patternA",
    )

    assert result.terminal_status == expected_status
    assert result.attempt_history == ()
    assert len(result.attempt_evidence) == 1
    evidence = result.attempt_evidence[0]
    assert evidence.experiment == "agg_patternA"
    assert evidence.caps == runner.BatchCaps()
    assert evidence.status == expected_status
    assert evidence.search_attempt is None
    assert evidence.error_type == type(error).__name__
    assert evidence.error_message == str(error)


def test_production_cap_search_runs_every_experiment_without_external_callback(monkeypatch):
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="agg",
    )
    calls = []

    def evaluate(run_spec_arg, *, experiment, caps, task_factory):
        assert run_spec_arg == run_spec
        assert task_factory is runner.build_comparison_task
        calls.append((experiment, caps))
        count = sum(seen_experiment == experiment for seen_experiment, _caps in calls)
        selected_batch = caps.agg if count == 1 else caps.agg - 1
        return runner.SearchAttempt(
            rank1_batch_sizes={"agg": selected_batch},
            rank1_row={"experiment": experiment, "batch": selected_batch},
        )

    monkeypatch.setattr(runner, "evaluate_experiment_attempt", evaluate)
    monkeypatch.setattr(
        runner,
        "_attach_final_detailed_evaluation",
        lambda _run_spec, *, experiment, cap_result, task_factory: cap_result,
    )

    results = runner.run_all_experiment_cap_searches(run_spec)

    assert tuple(results) == ("agg_patternA", "agg_patternB")
    assert calls == [
        ("agg_patternA", runner.BatchCaps()),
        ("agg_patternA", runner.BatchCaps(agg=2048, prefill=16, decode=1024)),
        ("agg_patternB", runner.BatchCaps()),
        ("agg_patternB", runner.BatchCaps(agg=2048, prefill=16, decode=1024)),
    ]
    assert all(result.terminal_status == "success" for result in results.values())
    assert all(
        [evidence.status for evidence in result.attempt_evidence] == ["cap_saturated", "success"]
        for result in results.values()
    )


def test_exact_aggregate_rerun_propagates_selected_ctx_tokens_and_requires_same_identity():
    runner = _comparison_runner()
    selected = _raw_agg_result() | {"ctx_tokens": 8192}
    calls = []

    class FakeTask:
        def run_single_agg(self, **kwargs):
            calls.append(kwargs)
            return SinglePointEvaluation(
                row=dict(selected),
                per_ops_data={
                    "mix_step": {"context_mla_attention": 2.0, "context_moe": 3.0},
                    "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
                },
                per_ops_source={
                    "mix_step": {"context_mla_attention": "sol", "context_moe": "sol"},
                },
            )

    evaluation = runner.rerun_selected_point(
        serving_mode="agg",
        experiment="agg_patternA",
        task=FakeTask(),
        selected_row=selected,
    )

    assert isinstance(evaluation, SinglePointEvaluation)
    assert evaluation.communication_evidence == ()
    assert calls == [
        {
            "tp": 1,
            "pp": 1,
            "dp": 16,
            "moe_tp": 1,
            "moe_ep": 16,
            "batch_size": 512,
            "ctx_tokens": 8192,
            "include_per_ops": True,
        }
    ]


def test_aggregate_identity_includes_ctx_tokens_and_rejects_mismatched_rerun():
    runner = _comparison_runner()
    selected = _raw_agg_result() | {"ctx_tokens": 8192}
    rerun = dict(selected) | {"ctx_tokens": 4096}

    assert runner._canonical_agg_sort_key("agg_patternA", selected) != runner._canonical_agg_sort_key(
        "agg_patternA",
        rerun,
    )
    assert "ctx_tokens=8192" in runner._canonical_agg_config_id("agg_patternA", selected)

    class FakeTask:
        def run_single_agg(self, **_kwargs):
            return SinglePointEvaluation(
                row=rerun,
                per_ops_data={"mix_step": {"context_mla_attention": 1.0}, "scheduling": {}},
                per_ops_source={"mix_step": {"context_mla_attention": "sol"}},
            )

    with pytest.raises(ValueError, match="rerun identity mismatch"):
        runner.rerun_selected_point(
            serving_mode="agg",
            experiment="agg_patternA",
            task=FakeTask(),
            selected_row=selected,
        )


def test_exact_rerun_rejects_non_unit_context_parallelism_before_calling_task():
    runner = _comparison_runner()
    selected = _raw_disagg_result() | {"(p)cp": 2}

    class FakeTask:
        def run_single_disagg(self, **_kwargs):
            raise AssertionError("single-point API must not be called for unsupported cp")

    with pytest.raises(ValueError, match="prefill cp mismatch: expected 1"):
        runner.rerun_selected_point(
            serving_mode="disagg",
            experiment="disagg_AB",
            task=FakeTask(),
            selected_row=selected,
        )


def test_empty_task_result_fails_fast_without_guessing_sla_terminal():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h200_sxm",
        serving_mode="agg",
    )

    with pytest.raises(RuntimeError, match="returned no candidate rows") as raised:
        runner._select_experiment_rank_one(
            run_spec,
            experiment="agg_patternA",
            candidate_rows=(),
        )

    assert type(raised.value) is RuntimeError


def test_per_ops_validation_requires_finite_strict_sol_shape_and_reports_attention_totals():
    runner = _comparison_runner()
    evaluation = SinglePointEvaluation(
        row={"osl": 1},
        per_ops_data={
            "mix_step": {"context_mla_attention": 2.0, "context_moe": 3.0},
            "genonly_step": {"generation_mla_bmm_pre": 5.0, "generation_moe": 7.0},
            "scheduling": {
                "num_mix_steps": 2.0,
                "num_genonly_steps": 0.0,
                "mix_step_latency_ms": 5.0,
                "genonly_step_latency_ms": 12.0,
            },
        },
        per_ops_source={
            "mix_step": {"context_mla_attention": "sol", "context_moe": "sol"},
            "genonly_step": {"generation_mla_bmm_pre": "sol", "generation_moe": "sol"},
        },
    )

    evidence = runner.validate_per_ops_evidence(evaluation, serving_mode="agg", osl=1)

    assert evidence["phase_totals_ms"] == {
        "mix_step": {"attention": 2.0, "non_attention": 3.0, "total": 5.0},
        "genonly_step": {"attention": 5.0, "non_attention": 7.0, "total": 12.0},
    }
    assert evidence["weighted_totals_ms"] == {
        "attention": 4.0,
        "non_attention": 6.0,
        "total": 10.0,
    }


@pytest.mark.parametrize(
    ("per_ops_data", "per_ops_source", "message"),
    [
        (
            {"prefill": {"attention": float("nan")}, "decode": {}},
            {"prefill": {"attention": "sol"}, "decode": {}},
            "finite",
        ),
        (
            {"prefill": {"attention": 1.0}, "decode": {}},
            {"prefill": {"attention": "silicon"}, "decode": {}},
            "strict SOL",
        ),
        ({"prefill": {"attention": 1.0}, "decode": {}}, {"prefill": {"moe": "sol"}, "decode": {}}, "keys"),
        ({"prefill": {"attention": 1.0}, "decode": {}}, {"prefill": {"attention": "sol"}}, "phases"),
    ],
)
def test_per_ops_validation_fails_fast_on_malformed_evidence(per_ops_data, per_ops_source, message):
    runner = _comparison_runner()
    evaluation = SinglePointEvaluation(
        row={"osl": 1},
        per_ops_data=per_ops_data,
        per_ops_source=per_ops_source,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        runner.validate_per_ops_evidence(evaluation, serving_mode="disagg", osl=1)


def test_per_ops_validation_accepts_only_explicit_zero_latency_noop_source():
    runner = _comparison_runner()
    evaluation = SinglePointEvaluation(
        row={"osl": 1},
        per_ops_data={
            "mix_step": {
                "context_mla_attention": 1.0,
                "generation_attention (not executed)": 0.0,
            },
            "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
        },
        per_ops_source={
            "mix_step": {
                "context_mla_attention": "sol",
                "generation_attention (not executed)": "not_executed",
            }
        },
    )

    evidence = runner.validate_per_ops_evidence(evaluation, serving_mode="agg", osl=1)

    assert evidence["phase_totals_ms"]["mix_step"] == {
        "attention": 1.0,
        "non_attention": 0.0,
        "total": 1.0,
    }
    assert evidence["weighted_totals_ms"] == {
        "attention": 1.0,
        "non_attention": 0.0,
        "total": 1.0,
    }


@pytest.mark.parametrize(
    ("operation_name", "latency"),
    (
        ("generation_attention (not executed)", 0.5),
        ("generation_attention", 0.0),
    ),
)
def test_per_ops_validation_rejects_misused_noop_source(operation_name, latency):
    runner = _comparison_runner()
    evaluation = SinglePointEvaluation(
        row={"osl": 1},
        per_ops_data={
            "mix_step": {"context_mla_attention": 1.0, operation_name: latency},
            "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
        },
        per_ops_source={"mix_step": {"context_mla_attention": "sol", operation_name: "not_executed"}},
    )

    with pytest.raises(ValueError, match="not_executed"):
        runner.validate_per_ops_evidence(evaluation, serving_mode="agg", osl=1)


def test_completed_experiment_normalization_uses_exact_rerun_row_and_per_op_evidence():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    exact_row = _raw_agg_result() | {"tokens/s/gpu": 123.456789, "tokens/s": 1975.308624}
    communication_evidence = (
        perf_database.CommunicationQueryEvidence(
            operation_name="context_moe_dispatch",
            operation_kind="nccl",
            collective="all_gather",
            group_size=16,
            tier="inter_node_bw",
            bandwidth_bytes_per_sec=50_000_000_000,
            message_size_bytes=131_072,
        ),
    )
    evaluation = SinglePointEvaluation(
        row=exact_row,
        per_ops_data={
            "mix_step": {"context_mla_attention": 2.0, "context_moe": 3.0},
            "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
        },
        per_ops_source={
            "mix_step": {"context_mla_attention": "sol", "context_moe": "sol"},
        },
        communication_evidence=communication_evidence,
    )
    attempt = runner.SearchAttempt(
        rank1_batch_sizes={"agg": 512},
        candidate_rows=(_raw_agg_result(),),
        rank1_row=_raw_agg_result(),
        selected_point_identity=runner._canonical_agg_sort_key("agg_patternA", exact_row),
        selected_evaluation=evaluation,
        per_ops_evidence=runner.validate_per_ops_evidence(evaluation, serving_mode="agg", osl=1),
    )
    caps = runner.BatchCaps()
    cap_result = runner.CapSearchResult(
        terminal_status="success",
        final_caps=caps,
        cap_history=(caps,),
        cap_rerun_count=0,
        cap_saturated=False,
        ranking_eligible=True,
        attempt_evidence=(
            runner.CapAttemptEvidence(
                experiment="agg_patternA",
                caps=caps,
                status="success",
                search_attempt=attempt,
            ),
        ),
    )
    task = runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=caps)

    normalized = runner.normalize_completed_experiment(
        run_spec,
        experiment="agg_patternA",
        task=task,
        cap_result=cap_result,
        system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
    )

    assert normalized["tokens/s/gpu"] == 123.456789
    assert normalized["tokens/s/gpu_cluster"] == 123.456789
    assert normalized["per_ops_data"] == evaluation.per_ops_data
    assert normalized["per_ops_source"] == evaluation.per_ops_source
    assert normalized["communication_evidence"] == (
        {
            "operation_name": "context_moe_dispatch",
            "operation_kind": "nccl",
            "collective": "all_gather",
            "group_size": 16,
            "tier": "inter_node_bw",
            "bandwidth_bytes_per_sec": 50_000_000_000,
            "message_size_bytes": 131_072,
        },
    )
    assert normalized["effective_bandwidth_tiers"] == (
        {
            "operation_name": "context_moe_dispatch",
            "operation_kind": "nccl",
            "collective": "all_gather",
            "group_size": 16,
            "tier": "inter_node_bw",
            "bandwidth_bytes_per_sec": 50_000_000_000,
        },
    )
    assert normalized["per_ops_phase_totals_ms"] == {"mix_step": {"attention": 2.0, "non_attention": 3.0, "total": 5.0}}
    assert normalized["per_ops_weighted_totals_ms"] == {
        "attention": 2.0,
        "non_attention": 3.0,
        "total": 5.0,
    }


def test_completed_experiment_normalization_rejects_missing_exact_evaluation():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    caps = runner.BatchCaps()
    cap_result = runner.CapSearchResult(
        terminal_status="success",
        final_caps=caps,
        cap_history=(caps,),
        cap_rerun_count=0,
        cap_saturated=False,
        ranking_eligible=True,
        attempt_evidence=(
            runner.CapAttemptEvidence(
                experiment="agg_patternA",
                caps=caps,
                status="success",
                search_attempt=runner.SearchAttempt(rank1_batch_sizes={"agg": 512}),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="missing exact selected-point evaluation"):
        runner.normalize_completed_experiment(
            run_spec,
            experiment="agg_patternA",
            task=runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=caps),
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec("h800_sxm", systems_paths=[SYSTEMS_DIR]),
        )


def test_completed_experiment_normalization_rejects_uncaptured_communication_evidence():
    runner = _comparison_runner()
    run_spec = _comparison_run_spec(
        runner,
        model="stepfun-ai/Step4",
        system="h800_sxm",
        serving_mode="agg",
    )
    caps = runner.BatchCaps()
    exact_row = _raw_agg_result()
    evaluation = SinglePointEvaluation(
        row=exact_row,
        per_ops_data={
            "mix_step": {"context_mla_attention": 2.0, "context_moe": 3.0},
            "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
        },
        per_ops_source={
            "mix_step": {"context_mla_attention": "sol", "context_moe": "sol"},
        },
    )
    attempt = runner.SearchAttempt(
        rank1_batch_sizes={"agg": 512},
        candidate_rows=(exact_row,),
        rank1_row=exact_row,
        selected_point_identity=runner._canonical_agg_sort_key("agg_patternA", exact_row),
        selected_evaluation=evaluation,
        per_ops_evidence=runner.validate_per_ops_evidence(
            evaluation,
            serving_mode="agg",
            osl=1,
        ),
    )
    cap_result = runner.CapSearchResult(
        terminal_status="success",
        final_caps=caps,
        cap_history=(caps,),
        cap_rerun_count=0,
        cap_saturated=False,
        ranking_eligible=True,
        attempt_evidence=(
            runner.CapAttemptEvidence(
                experiment="agg_patternA",
                caps=caps,
                status="success",
                search_attempt=attempt,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="communication query evidence was not captured"):
        runner.normalize_completed_experiment(
            run_spec,
            experiment="agg_patternA",
            task=runner.build_comparison_task(run_spec, experiment="agg_patternA", caps=caps),
            cap_result=cap_result,
            system_spec=perf_database.load_system_spec(
                "h800_sxm",
                systems_paths=[SYSTEMS_DIR],
            ),
        )


def test_runner_has_no_dimension_based_bandwidth_reconstruction_helper():
    runner = _comparison_runner()

    assert not hasattr(runner, "_bandwidth_evidence")


def _rank_row(
    *,
    model="stepfun-ai/Step4",
    system="h200_sxm",
    workload_kind="primary",
    isl=4096,
    osl=1,
    ttft_sla_ms=500,
    serving_mode="agg",
    config_id,
    throughput,
    sort_key=None,
    ttft=100.0,
    tpot=10.0,
    request_latency=200.0,
    terminal_status="success",
    cap_saturated=False,
    ttft_pass=True,
    ranking_eligible=True,
):
    return {
        "model": model,
        "system": system,
        "workload_kind": workload_kind,
        "isl": isl,
        "osl": osl,
        "prefix": 0,
        "ttft_sla_ms": ttft_sla_ms,
        "serving_mode": serving_mode,
        "backend": "vllm",
        "database_mode": "SOL",
        "nextn": 0,
        "canonical_config_id": config_id,
        "canonical_config_sort_key": sort_key if sort_key is not None else (config_id,),
        "tokens/s/gpu_cluster": throughput,
        "ranking_metric_kind": (
            "prefill_input_throughput" if workload_kind == "primary" and osl == 1 else "output_token_throughput"
        ),
        "ranking_metric_value": throughput,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": request_latency,
        "terminal_status": terminal_status,
        "cap_saturated": cap_saturated,
        "ttft_pass": ttft_pass,
        "ranking_eligible": ranking_eligible,
    }


def test_rank_final_rows_excludes_saturated_infeasible_and_ttft_failed_rows():
    runner = _comparison_runner()
    rows = [
        _rank_row(config_id="eligible-low", throughput=100.0),
        _rank_row(config_id="eligible-high", throughput=200.0),
        _rank_row(config_id="saturated", throughput=1000.0, cap_saturated=True),
        _rank_row(config_id="ttft-failed", throughput=900.0, ttft=501.0, ttft_pass=False),
        _rank_row(
            config_id="memory-infeasible",
            throughput=800.0,
            terminal_status="memory_infeasible",
            ranking_eligible=False,
        ),
    ]

    ranked = runner.rank_final_rows(rows)

    assert [(row["canonical_config_id"], row["rank"]) for row in ranked] == [
        ("eligible-high", 1),
        ("eligible-low", 2),
    ]


def test_rank_final_rows_uses_only_canonical_identity_for_exact_throughput_ties():
    runner = _comparison_runner()
    rows = [
        _rank_row(config_id="config-z", throughput=200.0, ttft=10.0, tpot=1.0),
        _rank_row(config_id="config-a", throughput=200.0, ttft=499.0, tpot=999.0),
    ]

    ranked = runner.rank_final_rows(rows)

    assert [row["canonical_config_id"] for row in ranked] == ["config-a", "config-z"]
    assert runner.RANKING_CONTRACT == {
        "primary": "prefill_input_throughput per fixed cluster descending",
        "decode_smoke": "output_token_throughput per fixed cluster descending",
        "tie_breaker": "typed canonical configuration identity ascending",
    }


def test_rank_final_rows_uses_numeric_canonical_key_for_exact_throughput_ties():
    runner = _comparison_runner()
    rows = [
        _rank_row(
            config_id="agg_patternA|tp=16",
            sort_key=("agg_patternA", 16, 1, 1, 1, 1, 1, 1),
            throughput=200.0,
        ),
        _rank_row(
            config_id="agg_patternA|tp=2",
            sort_key=("agg_patternA", 2, 1, 1, 1, 1, 1, 1),
            throughput=200.0,
        ),
    ]

    ranked = runner.rank_final_rows(rows)

    assert [row["canonical_config_id"] for row in ranked] == [
        "agg_patternA|tp=2",
        "agg_patternA|tp=16",
    ]


def test_rank_final_rows_restores_json_checkpoint_canonical_key():
    runner = _comparison_runner()
    row = _rank_row(
        config_id="agg_patternA|tp=8",
        sort_key=["agg_patternA", 8, 1, 1, 8, 1, 1, 9, 4096],
        throughput=200.0,
    )

    ranked = runner.rank_final_rows([row])

    assert ranked[0]["canonical_config_sort_key"] == (
        "agg_patternA",
        8,
        1,
        1,
        8,
        1,
        1,
        9,
        4096,
    )


@pytest.mark.parametrize(
    ("serving_mode", "ttft", "ttft_pass"),
    [
        ("agg", 501.0, True),
        ("disagg", 500.0, True),
        ("agg", 500.0, False),
        ("disagg", 499.0, False),
    ],
)
def test_rank_final_rows_rejects_ttft_pass_evidence_that_contradicts_mode_boundary(
    serving_mode,
    ttft,
    ttft_pass,
):
    runner = _comparison_runner()
    row = _rank_row(
        config_id="contradictory-ttft",
        throughput=100.0,
        serving_mode=serving_mode,
        ttft=ttft,
        ttft_pass=ttft_pass,
    )

    with pytest.raises(ValueError, match="ttft_pass contradicts"):
        runner.rank_final_rows([row])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"tokens/s/gpu_cluster": float("nan")}, "finite"),
        ({"tokens/s/gpu_cluster": float("inf")}, "finite"),
        ({"terminal_status": "network_error"}, "Unknown terminal status"),
        ({"canonical_config_id": ""}, "canonical_config_id"),
    ],
)
def test_rank_final_rows_rejects_invalid_evidence(mutation, message):
    runner = _comparison_runner()
    row = _rank_row(config_id="valid", throughput=100.0)
    row.update(mutation)

    with pytest.raises(ValueError, match=message):
        runner.rank_final_rows([row])


def test_model_comparisons_use_deepseek_baseline_and_record_metric_polarity():
    runner = _comparison_runner()
    rows = runner.rank_final_rows(
        [
            _rank_row(config_id="step4-config", throughput=120.0, ttft=150.0, tpot=12.0),
            _rank_row(
                model="deepseek-ai/DeepSeek-V4-Pro",
                config_id="deepseek-config",
                throughput=100.0,
                ttft=200.0,
                tpot=10.0,
            ),
        ]
    )

    comparisons = runner.build_model_comparisons(
        rows,
        metrics=("tokens/s/gpu_cluster", "ttft", "tpot"),
    )

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.status == "paired"
    assert comparison.step4_config_id == "step4-config"
    assert comparison.deepseek_config_id == "deepseek-config"
    assert comparison.metric_deltas["tokens/s/gpu_cluster"] == runner.MetricDelta(
        step4_value=120.0,
        deepseek_value=100.0,
        absolute_delta=20.0,
        relative_delta=0.2,
        polarity="higher_is_better",
        status="computed",
    )
    assert comparison.metric_deltas["ttft"] == runner.MetricDelta(
        step4_value=150.0,
        deepseek_value=200.0,
        absolute_delta=-50.0,
        relative_delta=-0.25,
        polarity="lower_is_better",
        status="computed",
    )
    assert comparison.metric_deltas["tpot"] == runner.MetricDelta(
        step4_value=12.0,
        deepseek_value=10.0,
        absolute_delta=2.0,
        relative_delta=0.2,
        polarity="lower_is_better",
        status="computed",
    )
    assert runner.DELTA_CONTRACT == {
        "absolute": "Step4 - DeepSeek-V4-Pro",
        "relative": "(Step4 - DeepSeek-V4-Pro) / DeepSeek-V4-Pro",
        "baseline": "DeepSeek-V4-Pro",
        "zero_baseline": "error",
        "zero_baseline_both_zero": (
            "tpot only: absolute_delta=0.0, relative_delta=null, status=zero_baseline_both_zero"
        ),
    }


def test_metric_delta_requires_explicit_status():
    runner = _comparison_runner()

    with pytest.raises(TypeError, match="status"):
        runner.MetricDelta(
            step4_value=12.0,
            deepseek_value=10.0,
            absolute_delta=2.0,
            relative_delta=0.2,
            polarity="lower_is_better",
        )


def test_model_comparisons_publish_null_relative_delta_for_both_zero_tpot():
    runner = _comparison_runner()
    rows = runner.rank_final_rows(
        [
            _rank_row(config_id="step4-config", throughput=120.0, serving_mode="disagg", tpot=0.0),
            _rank_row(
                model="deepseek-ai/DeepSeek-V4-Pro",
                config_id="deepseek-config",
                throughput=100.0,
                serving_mode="disagg",
                tpot=0.0,
            ),
        ]
    )

    comparison = runner.build_model_comparisons(rows, metrics=("tpot",))[0]

    assert comparison.metric_deltas["tpot"] == runner.MetricDelta(
        step4_value=0.0,
        deepseek_value=0.0,
        absolute_delta=0.0,
        relative_delta=None,
        polarity="lower_is_better",
        status="zero_baseline_both_zero",
    )


def test_model_comparisons_preserve_nonzero_decode_smoke_tpot_arithmetic():
    runner = _comparison_runner()
    rows = runner.rank_final_rows(
        [
            _rank_row(
                workload_kind="decode_smoke",
                osl=1024,
                config_id="step4-config",
                throughput=120.0,
                tpot=12.0,
            ),
            _rank_row(
                model="deepseek-ai/DeepSeek-V4-Pro",
                workload_kind="decode_smoke",
                osl=1024,
                config_id="deepseek-config",
                throughput=100.0,
                tpot=10.0,
            ),
        ]
    )

    delta = runner.build_model_comparisons(rows, metrics=("tpot",))[0].metric_deltas["tpot"]

    assert delta.relative_delta == pytest.approx(0.2)
    assert delta.status == "computed"


def test_model_comparisons_mark_unpaired_aligned_keys_without_fabricating_deltas():
    runner = _comparison_runner()
    rows = runner.rank_final_rows(
        [
            _rank_row(config_id="step4-only", throughput=120.0),
            _rank_row(
                model="deepseek-ai/DeepSeek-V4-Pro",
                system="gb300",
                config_id="deepseek-only",
                throughput=100.0,
            ),
        ]
    )

    comparisons = runner.build_model_comparisons(rows, metrics=("tokens/s/gpu_cluster",))

    assert len(comparisons) == 2
    assert {comparison.status for comparison in comparisons} == {"unpaired"}
    assert all(comparison.metric_deltas == {} for comparison in comparisons)


def test_model_comparisons_fail_fast_on_zero_deepseek_baseline():
    runner = _comparison_runner()
    rows = runner.rank_final_rows(
        [
            _rank_row(config_id="step4-config", throughput=120.0),
            _rank_row(
                model="deepseek-ai/DeepSeek-V4-Pro",
                config_id="deepseek-config",
                throughput=0.0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="zero DeepSeek-V4-Pro baseline"):
        runner.build_model_comparisons(rows, metrics=("tokens/s/gpu_cluster",))


def test_model_comparisons_fail_fast_on_both_zero_ranking_baseline():
    runner = _comparison_runner()
    rows = runner.rank_final_rows(
        [
            _rank_row(config_id="step4-config", throughput=0.0),
            _rank_row(
                model="deepseek-ai/DeepSeek-V4-Pro",
                config_id="deepseek-config",
                throughput=0.0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="zero DeepSeek-V4-Pro baseline"):
        runner.build_model_comparisons(rows, metrics=("ranking_metric_value",))


def test_model_comparisons_fail_fast_on_nonzero_tpot_with_zero_baseline():
    runner = _comparison_runner()
    rows = runner.rank_final_rows(
        [
            _rank_row(config_id="step4-config", throughput=120.0, tpot=5.0),
            _rank_row(
                model="deepseek-ai/DeepSeek-V4-Pro",
                config_id="deepseek-config",
                throughput=100.0,
                tpot=0.0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="zero DeepSeek-V4-Pro baseline"):
        runner.build_model_comparisons(rows, metrics=("tpot",))


def test_model_comparisons_reject_duplicate_rank_one_evidence():
    runner = _comparison_runner()
    row = _rank_row(config_id="step4-config", throughput=120.0)
    row["rank"] = 1

    with pytest.raises(ValueError, match="Duplicate rank-one evidence"):
        runner.build_model_comparisons([row, dict(row)], metrics=("tokens/s/gpu_cluster",))
