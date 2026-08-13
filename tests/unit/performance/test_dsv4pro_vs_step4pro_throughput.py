"""Unit coverage for the fresh Step4-Pro/DeepSeek throughput-ratio workflow."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from tests.performance.aic_roofline_pareto import plot_step4pro_throughput_ratio as plotting
from tests.performance.aic_roofline_pareto import run_dsv4pro_vs_step4pro_throughput as runner

pytestmark = pytest.mark.unit

BASE_RUNNER_SOURCE = "tests/performance/aic_roofline_pareto/run_step4_comparison.py"
THROUGHPUT_RUNNER_SOURCE = "tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4pro_throughput.py"


def _build_test_execution_contract(tmp_path: Path, run_spec):
    sdk_source = tmp_path / "src/aiconfigurator/sdk/runtime.py"
    sdk_source.parent.mkdir(parents=True, exist_ok=True)
    if not sdk_source.exists():
        sdk_source.write_text("RUNTIME = 1\n", encoding="utf-8")
    model_config = (
        tmp_path / "src/aiconfigurator/model_configs" / f"{run_spec.point.model.replace('/', '--')}_config.json"
    )
    model_config.parent.mkdir(parents=True, exist_ok=True)
    if not model_config.exists():
        model_config.write_text('{"hidden_size":4096}\n', encoding="utf-8")
    for relative_path in (BASE_RUNNER_SOURCE, THROUGHPUT_RUNNER_SOURCE):
        source = tmp_path / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            source.write_text(f"SOURCE = {relative_path!r}\n", encoding="utf-8")

    with runner._base_contract():
        return runner.base.build_execution_contract(
            (run_spec,),
            initial_caps=runner.base.BatchCaps(),
            repo_root=tmp_path,
            system_loader=lambda _name: {"gpu": {"mem_bw": 1}},
        )


def test_matrix_uses_only_disagg_and_uniform_osl_and_sla():
    points = runner.build_matrix_points()
    runs = runner.build_mode_run_specs()

    assert len(points) == 64
    assert len(runs) == 64
    assert {point.model for point in points} == {
        "stepfun-ai/Step4-Pro-V1",
        "deepseek-ai/DeepSeek-V4-Pro",
    }
    assert {point.system for point in points} == {"gb300", "h200_sxm", "h100_sxm", "h800_sxm"}
    assert {point.isl for point in points} == set(runner.ISL_VALUES)
    assert {point.osl for point in points} == {1024}
    assert {point.ttft_sla_ms for point in points} == {5000}
    assert {point.database_mode for point in points} == {"SOL"}
    assert {point.serving_mode for point in runs} == {"disagg"}
    assert all(point.total_gpus == 64 for point in points)
    assert all(point.tpot_ms == 50_000 for point in points)
    assert all(point.prefix == 0 and point.nextn == 0 for point in points)
    assert all(point.chunked_prefill is False and point.pareto_sweep is False for point in points)
    step4pro_points = [point for point in points if point.model == "stepfun-ai/Step4-Pro-V1"]
    assert all(point.attention_approximation is None for point in step4pro_points)
    assert not any(point.approximation_dominated for point in step4pro_points)
    assert runner.base.NEUTRAL_CORRECTIONS == {
        "prefill_latency_correction": 1.0,
        "decode_latency_correction": 1.0,
        "rate_match_prefill_degradation": 1.0,
        "rate_match_decode_degradation": 1.0,
        "autoscale_ttft_correction_factor": 1.0,
    }


def test_execution_contract_hashes_wrapper_and_executing_base_runner(tmp_path: Path):
    run_spec = runner.build_mode_run_specs()[0]
    baseline_contract = _build_test_execution_contract(tmp_path, run_spec)

    assert {BASE_RUNNER_SOURCE, THROUGHPUT_RUNNER_SOURCE} <= set(baseline_contract["source_files"])
    assert baseline_contract["used_models"] == ["stepfun-ai/Step4-Pro-V1"]
    assert baseline_contract["model_configs"]["stepfun-ai/Step4-Pro-V1"]["path"] == (
        "src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json"
    )
    baseline_sha256 = runner.base.execution_contract_sha256(baseline_contract)
    (tmp_path / BASE_RUNNER_SOURCE).write_text("SOURCE = 'changed base semantics'\n", encoding="utf-8")

    changed_contract = _build_test_execution_contract(tmp_path, run_spec)

    assert runner.base.execution_contract_sha256(changed_contract) != baseline_sha256


def test_resume_rejects_checkpoint_from_previous_base_runner_source(tmp_path: Path):
    run_spec = runner.build_mode_run_specs()[0]
    baseline_contract = _build_test_execution_contract(tmp_path, run_spec)
    baseline_sha256 = runner.base.execution_contract_sha256(baseline_contract)
    checkpoint = tmp_path / "mode_runs.sqlite3"
    runner.base.initialize_checkpoint(
        checkpoint,
        runner.base.build_checkpoint_header(
            (run_spec,),
            execution_contract_sha256=baseline_sha256,
            git_head="test-head",
        ),
    )
    (tmp_path / BASE_RUNNER_SOURCE).write_text("SOURCE = 'changed base semantics'\n", encoding="utf-8")
    changed_sha256 = runner.base.execution_contract_sha256(_build_test_execution_contract(tmp_path, run_spec))

    with pytest.raises(ValueError, match="execution_contract_sha256 mismatch"):
        runner.base.execute_matrix_runs(
            (run_spec,),
            checkpoint_path=checkpoint,
            execution_contract_sha256=changed_sha256,
            git_head="test-head",
            resume=True,
            initial_caps=runner.base.BatchCaps(),
            executor=lambda *_args, **_kwargs: pytest.fail("resume must reject before executing a mode run"),
        )


def test_task_contract_uses_output_ranking_for_osl_1024():
    point = runner.build_matrix_points()[0]
    task = runner.base.build_comparison_task(
        runner.ModeRunSpec(point=point, serving_mode="disagg"),
        experiment="disagg_AA",
        caps=runner.base.BatchCaps(prefill=16, decode=32),
    )

    assert task.osl == 1024
    assert task.ttft == 5000
    assert task.disagg_ranking_total_gpus == 64
    assert task.disagg_ranking_metric_kind == "output_token_throughput"
    assert task.prefill_max_batch_size == 16
    assert task.decode_max_batch_size == 32
    assert task.prefill_enable_chunked_prefill is False


def test_ranking_metric_is_output_throughput_and_exposes_both_figure_metrics():
    point = runner.build_matrix_points()[0]
    run_spec = runner.ModeRunSpec(point=point, serving_mode="disagg")
    row = {
        "num_total_gpus": 16,
        "tokens/s/gpu": 25.0,
        "(p)global_bs": 4,
        "(p)workers": 2,
        "ttft": 1000.0,
    }

    evidence = runner.derive_ranking_metric_evidence(run_spec, row)

    assert evidence["ranking_metric_kind"] == "output_token_throughput"
    assert evidence["ranking_metric_value"] == pytest.approx(25.0)
    assert evidence["output_token_throughput"] == pytest.approx(25.0)
    assert evidence["prefill_input_throughput"] > 0
    assert evidence["prefill_input_throughput"] == evidence["prefill_input_throughput/gpu_cluster"]


def test_ranking_metric_rejects_aggregate_mode():
    point = runner.build_matrix_points()[0]
    run_spec = runner.ModeRunSpec(point=point, serving_mode="agg")

    with pytest.raises(ValueError, match="only supports disagg"):
        runner.derive_ranking_metric_evidence(run_spec, {})


def test_throughput_workload_passes_base_rank_pipeline():
    point = runner.build_matrix_points()[0]
    row = {
        "model": point.model,
        "system": point.system,
        "workload_kind": point.workload_kind,
        "isl": point.isl,
        "osl": point.osl,
        "prefix": point.prefix,
        "ttft_sla_ms": point.ttft_sla_ms,
        "serving_mode": "disagg",
        "backend": point.backend,
        "database_mode": point.database_mode,
        "nextn": point.nextn,
        "canonical_config_id": "disagg_AA|synthetic=1",
        "canonical_config_sort_key": ("disagg_AA", 1),
        "tokens/s/gpu_cluster": 10.0,
        "ranking_metric_kind": "output_token_throughput",
        "ranking_metric_value": 10.0,
        "terminal_status": "success",
        "cap_saturated": False,
        "ttft": 4999.0,
        "ttft_pass": True,
        "ranking_eligible": True,
    }

    ranked = runner.base.rank_final_rows([row])

    assert len(ranked) == 1
    assert ranked[0]["rank"] == 1
    assert ranked[0]["workload_kind"] == "throughput"


def _comparison_row(model: str, value: float) -> dict:
    return {
        "model": model,
        "system": "h200_sxm",
        "workload_kind": "throughput",
        "isl": 1024,
        "osl": 1024,
        "prefix": 0,
        "ttft_sla_ms": 5000,
        "serving_mode": "disagg",
        "backend": "vllm",
        "database_mode": "SOL",
        "nextn": 0,
        "canonical_config_id": f"config-{model}",
        "rank": 1,
        "ranking_metric_kind": "output_token_throughput",
        "ranking_metric_value": value,
        "ttft": 1000.0,
    }


def _figure_row(
    model: str,
    *,
    system: str = "h200_sxm",
    isl: int = 1024,
    prefill_per_used_gpu: float,
    decode_per_used_gpu: float,
    num_total_gpus: int,
    replicas: int,
) -> dict:
    total_gpus = 64
    total_gpus_used = num_total_gpus * replicas
    return {
        "rank": 1,
        "model": model,
        "system": system,
        "workload_kind": "throughput",
        "isl": isl,
        "serving_mode": "disagg",
        "num_total_gpus": num_total_gpus,
        "replicas": replicas,
        "total_gpus": total_gpus,
        "total_gpus_used": total_gpus_used,
        "unused_gpus": total_gpus - total_gpus_used,
        "prefill_input_throughput/gpu": prefill_per_used_gpu,
        "tokens/s/gpu": decode_per_used_gpu,
        "prefill_input_throughput": prefill_per_used_gpu * total_gpus_used / total_gpus,
        "output_token_throughput": decode_per_used_gpu * total_gpus_used / total_gpus,
    }


def test_base_contract_finalizes_step4pro_comparisons_and_restores_globals():
    original_builder = runner.base.build_model_comparisons
    original_renderer = runner.base._render_markdown_report
    original_delta_contract = runner.base.DELTA_CONTRACT
    rows = [
        _comparison_row("stepfun-ai/Step4-Pro-V1", 120.0),
        _comparison_row("deepseek-ai/DeepSeek-V4-Pro", 100.0),
    ]

    with runner._base_contract():
        comparisons = runner.base.build_model_comparisons(rows, metrics=("ranking_metric_value",))
        assert len(comparisons) == 1
        assert comparisons[0]["status"] == "paired"
        assert comparisons[0]["candidate_model"] == "stepfun-ai/Step4-Pro-V1"
        assert comparisons[0]["baseline_model"] == "deepseek-ai/DeepSeek-V4-Pro"
        delta = comparisons[0]["metric_deltas"]["ranking_metric_value"]
        assert delta["candidate_value"] == pytest.approx(120.0)
        assert delta["baseline_value"] == pytest.approx(100.0)
        assert delta["absolute_delta"] == pytest.approx(20.0)
        assert delta["relative_delta"] == pytest.approx(0.2)
        assert runner.base.DELTA_CONTRACT["absolute"] == "Step4-Pro-V1 - DeepSeek-V4-Pro"
        report = runner.base._render_markdown_report(
            {
                "summary": {
                    "mode_run_count": 2,
                    "normalized_row_count": 2,
                    "ranked_row_count": 2,
                    "paired_comparison_count": 1,
                    "unpaired_comparison_count": 0,
                },
                "ranked_rows": rows,
                "comparisons": comparisons,
                "comparison_metrics": ("ranking_metric_value",),
            }
        )
        assert "# Step4-Pro-V1 vs DeepSeek-V4-Pro SOL Comparison" in report
        assert "Absolute delta is Step4-Pro-V1 minus DeepSeek-V4-Pro" in report
        assert "Temporary MLA substitution" not in report

    assert runner.base.build_model_comparisons is original_builder
    assert runner.base._render_markdown_report is original_renderer
    assert runner.base.DELTA_CONTRACT is original_delta_contract


def test_step4pro_comparison_builder_preserves_unpaired_and_zero_baseline_rules():
    candidate = _comparison_row("stepfun-ai/Step4-Pro-V1", 0.0)
    baseline = _comparison_row("deepseek-ai/DeepSeek-V4-Pro", 0.0)
    candidate["tpot"] = 0.0
    baseline["tpot"] = 0.0

    with runner._base_contract():
        unpaired = runner.base.build_model_comparisons([candidate], metrics=("tpot",))
        assert len(unpaired) == 1
        assert unpaired[0]["status"] == "unpaired"
        assert unpaired[0]["baseline_config_id"] is None

        both_zero = runner.base.build_model_comparisons([candidate, baseline], metrics=("tpot",))
        assert both_zero[0]["metric_deltas"]["tpot"]["status"] == "zero_baseline_both_zero"
        assert both_zero[0]["metric_deltas"]["tpot"]["relative_delta"] is None

        with pytest.raises(ValueError, match="zero DeepSeek-V4-Pro baseline"):
            runner.base.build_model_comparisons(
                [
                    {**candidate, "ranking_metric_value": 1.0},
                    baseline,
                ],
                metrics=("ranking_metric_value",),
            )


def test_ratio_direction_is_step4pro_over_deepseek():
    rows = [
        _figure_row(
            "stepfun-ai/Step4-Pro-V1",
            prefill_per_used_gpu=120.0,
            decode_per_used_gpu=80.0,
            num_total_gpus=16,
            replicas=4,
        ),
        _figure_row(
            "deepseek-ai/DeepSeek-V4-Pro",
            prefill_per_used_gpu=100.0,
            decode_per_used_gpu=40.0,
            num_total_gpus=40,
            replicas=1,
        ),
    ]

    ratios = plotting.compute_throughput_ratios(rows)

    assert ratios["prefill"]["h200"][1024] == pytest.approx(1.2)
    assert ratios["decode"]["h200"][1024] == pytest.approx(2.0)


def test_ratio_uses_gpus_actually_used_for_gb300_131072():
    rows = [
        _figure_row(
            "stepfun-ai/Step4-Pro-V1",
            system="gb300",
            isl=131_072,
            prefill_per_used_gpu=8025.876397388652,
            decode_per_used_gpu=56.448,
            num_total_gpus=16,
            replicas=4,
        ),
        _figure_row(
            "deepseek-ai/DeepSeek-V4-Pro",
            system="gb300",
            isl=131_072,
            prefill_per_used_gpu=4365.748697656448,
            decode_per_used_gpu=30.689280000000004,
            num_total_gpus=40,
            replicas=1,
        ),
    ]

    ratios = plotting.compute_throughput_ratios(rows)

    assert ratios["prefill"]["gb300"][131_072] == pytest.approx(1.8383734276086428)
    assert ratios["decode"]["gb300"][131_072] == pytest.approx(1.8393393393393391)
    assert ratios["decode"]["gb300"][131_072] != pytest.approx(2.942942942942943)


def test_ratio_rejects_inconsistent_used_gpu_allocation():
    candidate = _figure_row(
        "stepfun-ai/Step4-Pro-V1",
        prefill_per_used_gpu=120.0,
        decode_per_used_gpu=80.0,
        num_total_gpus=24,
        replicas=2,
    )
    candidate["total_gpus_used"] = 64
    baseline = _figure_row(
        "deepseek-ai/DeepSeek-V4-Pro",
        prefill_per_used_gpu=100.0,
        decode_per_used_gpu=40.0,
        num_total_gpus=32,
        replicas=2,
    )

    with pytest.raises(ValueError, match="total_gpus_used mismatch"):
        plotting.compute_throughput_ratios([candidate, baseline])


def test_plot_contract_uses_step4pro_identity_and_baseline_text():
    assert plotting.MODEL_STEP4PRO == "stepfun-ai/Step4-Pro-V1"
    assert plotting.MODEL_DEEPSEEK == "deepseek-ai/DeepSeek-V4-Pro"
    assert plotting.BASELINE_LABEL == "Step4-Pro-V1 baseline"
    assert plotting.RATIO_DIRECTION == "Step4-Pro-V1 / DeepSeek-V4-Pro"
    assert plotting.NORMALIZATION_METHOD == "per_used_gpu"
    assert plotting.FIGURE_FIELDS == {
        "prefill": "prefill_input_throughput/gpu",
        "decode": "tokens/s/gpu",
    }


def test_annotation_positions_enforce_minimum_gap_without_center_drift():
    original = [100.0, 102.0, 107.0, 109.0]

    adjusted = plotting._spread_annotation_positions(original, minimum_gap=18.0)

    assert all(right - left >= 18.0 for left, right in pairwise(adjusted))
    assert sum(adjusted) / len(adjusted) == pytest.approx(sum(original) / len(original))
    assert plotting._spread_annotation_positions([], minimum_gap=18.0) == []
    assert plotting._spread_annotation_positions([5.0], minimum_gap=18.0) == [5.0]

    with pytest.raises(ValueError, match="minimum_gap must be positive"):
        plotting._spread_annotation_positions(original, minimum_gap=0.0)


def test_x_axis_limits_leave_room_for_edge_annotations():
    lower, upper = plotting._x_axis_limits()

    assert lower < min(runner.ISL_VALUES)
    assert upper > max(runner.ISL_VALUES)


def test_ratio_computation_rejects_missing_or_zero_baseline():
    with pytest.raises(ValueError, match="missing model row"):
        plotting.compute_throughput_ratios(
            [
                _figure_row(
                    "stepfun-ai/Step4-Pro-V1",
                    prefill_per_used_gpu=1.0,
                    decode_per_used_gpu=1.0,
                    num_total_gpus=16,
                    replicas=4,
                )
            ]
        )


def test_ratio_computation_rejects_duplicate_rank_one_rows():
    duplicate = _figure_row(
        "stepfun-ai/Step4-Pro-V1",
        prefill_per_used_gpu=1.0,
        decode_per_used_gpu=1.0,
        num_total_gpus=16,
        replicas=4,
    )

    with pytest.raises(ValueError, match="duplicate rank-one row"):
        plotting.compute_throughput_ratios([duplicate, duplicate])

    with pytest.raises(ValueError, match="zero DeepSeek-V4-Pro baseline"):
        plotting.compute_throughput_ratios(
            [
                duplicate,
                _figure_row(
                    "deepseek-ai/DeepSeek-V4-Pro",
                    prefill_per_used_gpu=0.0,
                    decode_per_used_gpu=1.0,
                    num_total_gpus=32,
                    replicas=2,
                ),
            ]
        )


def test_plot_script_accepts_system_shards_and_writes_combined_results(tmp_path: Path):
    sources = []
    for system in ("gb300", "h200_sxm", "h100_sxm", "h800_sxm"):
        payload = {
            "ranked_rows": [
                _figure_row(
                    model,
                    system=system,
                    isl=isl,
                    prefill_per_used_gpu=100.0 if model.startswith("deepseek") else 120.0,
                    decode_per_used_gpu=50.0 if model.startswith("deepseek") else 75.0,
                    num_total_gpus=40 if model.startswith("deepseek") else 16,
                    replicas=1 if model.startswith("deepseek") else 4,
                )
                for model in ("stepfun-ai/Step4-Pro-V1", "deepseek-ai/DeepSeek-V4-Pro")
                for isl in runner.ISL_VALUES
            ]
        }
        source = tmp_path / f"{system}.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        sources.append(source)
    output_dir = tmp_path / "figures"
    combined = output_dir / "combined_results.json"

    paths = plotting.generate_figures(sources, output_dir, combined_output=combined)

    assert paths["prefill"].is_file()
    assert paths["decode"].is_file()
    assert paths["prefill"].stat().st_size > 0
    assert paths["decode"].stat().st_size > 0
    combined_payload = json.loads(combined.read_text(encoding="utf-8"))
    assert combined_payload["summary"]["rank_one_row_count"] == 64
    assert len(combined_payload["ranked_rows"]) == 64
    assert combined_payload["throughput_normalization"] == {
        "cluster_budget_role": "execution provenance only; excluded from the denominator",
        "decode_source_field": "tokens/s/gpu",
        "denominator_field": "total_gpus_used",
        "formula": "total throughput across complete replicas / total_gpus_used",
        "idle_gpus_excluded": True,
        "normalization_method": "per_used_gpu",
        "prefill_source_field": "prefill_input_throughput/gpu",
        "total_gpus_cluster_budget": 64,
    }
    assert combined_payload["ratios"]["decode"]["gb300"]["131072"] == pytest.approx(1.5)
    assert combined_payload["legacy_fixed_cluster_ratios_for_audit"]["decode"]["gb300"]["131072"] == pytest.approx(2.4)


def test_plot_script_records_and_skips_unpaired_infeasible_points(tmp_path: Path):
    rows = [
        _figure_row(
            model,
            system=system,
            isl=isl,
            prefill_per_used_gpu=100.0 if model.startswith("deepseek") else 120.0,
            decode_per_used_gpu=50.0 if model.startswith("deepseek") else 75.0,
            num_total_gpus=40 if model.startswith("deepseek") else 16,
            replicas=1 if model.startswith("deepseek") else 4,
        )
        for system in ("gb300", "h200_sxm", "h100_sxm", "h800_sxm")
        for model in ("stepfun-ai/Step4-Pro-V1", "deepseek-ai/DeepSeek-V4-Pro")
        for isl in runner.ISL_VALUES
        if not (system == "h100_sxm" and model == "deepseek-ai/DeepSeek-V4-Pro" and isl == 131_072)
    ]
    source = tmp_path / "results.json"
    source.write_text(json.dumps({"ranked_rows": rows}), encoding="utf-8")
    output_dir = tmp_path / "figures"
    combined = output_dir / "combined_results.json"

    paths = plotting.generate_figures(source, output_dir, combined_output=combined)

    assert paths["prefill"].is_file()
    assert paths["decode"].is_file()
    combined_payload = json.loads(combined.read_text(encoding="utf-8"))
    assert combined_payload["summary"]["paired_ratio_point_count"] == 31
    assert combined_payload["summary"]["missing_ratio_point_count"] == 1
    assert combined_payload["missing_ratio_points"] == [
        {
            "isl": 131_072,
            "missing_models": ["deepseek-ai/DeepSeek-V4-Pro"],
            "system": "h100",
        }
    ]
    assert "131072" not in combined_payload["ratios"]["prefill"]["h100"]
    assert "131072" not in combined_payload["ratios"]["decode"]["h100"]


def test_plot_script_limits_lines_to_systems_present_in_shards(tmp_path: Path):
    rows = [
        _figure_row(
            model,
            system=system,
            isl=isl,
            prefill_per_used_gpu=100.0 if model.startswith("deepseek") else 120.0,
            decode_per_used_gpu=50.0 if model.startswith("deepseek") else 75.0,
            num_total_gpus=40 if model.startswith("deepseek") else 16,
            replicas=1 if model.startswith("deepseek") else 4,
        )
        for system in ("h200_sxm", "h100_sxm", "h800_sxm")
        for model in ("stepfun-ai/Step4-Pro-V1", "deepseek-ai/DeepSeek-V4-Pro")
        for isl in runner.ISL_VALUES
    ]
    source = tmp_path / "results.json"
    source.write_text(json.dumps({"ranked_rows": rows}), encoding="utf-8")
    combined = tmp_path / "figures" / "combined_results.json"

    plotting.generate_figures(source, combined.parent, combined_output=combined)

    combined_payload = json.loads(combined.read_text(encoding="utf-8"))
    assert combined_payload["summary"]["systems"] == ["h200", "h100", "h800"]
    assert combined_payload["summary"]["missing_ratio_point_count"] == 0
