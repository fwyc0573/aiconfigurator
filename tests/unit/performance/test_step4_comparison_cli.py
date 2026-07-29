"""CLI and final-artifact coverage for the Step4 SOL comparison runner."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import replace

import pytest

from tests.performance.aic_roofline_pareto import run_step4_comparison as runner

EXPERIMENTS_BY_MODE = {
    "agg": ("agg_patternA", "agg_patternB"),
    "disagg": ("disagg_AA", "disagg_AB", "disagg_BA", "disagg_BB"),
}


def _terminal_cap_result(experiment):
    caps = runner.BatchCaps()
    return runner.CapSearchResult(
        terminal_status="memory_infeasible",
        final_caps=caps,
        cap_history=(caps,),
        cap_rerun_count=0,
        cap_saturated=False,
        ranking_eligible=False,
        attempt_evidence=(
            runner.CapAttemptEvidence(
                experiment=experiment,
                caps=caps,
                status="memory_infeasible",
                search_attempt=None,
                error_type="InsufficientMemoryError",
                error_message="model does not fit",
            ),
        ),
    )


def _mode_record(run_spec):
    experiments = EXPERIMENTS_BY_MODE[run_spec.serving_mode]
    return runner.build_mode_run_record(
        run_spec,
        cap_results={experiment: _terminal_cap_result(experiment) for experiment in experiments},
        normalized_rows=(),
    )


def test_execute_matrix_runs_initializes_checkpoint_and_resume_skips_completed(tmp_path):
    run_specs = runner.build_mode_run_specs()[:2]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    executed = []
    loaded_systems = []

    def system_loader(system):
        loaded_systems.append(system)
        return {"name": system}

    def executor(run_spec, *, system_spec, initial_caps):
        executed.append((run_spec, system_spec, initial_caps))
        return _mode_record(run_spec)

    records = runner.execute_matrix_runs(
        run_specs,
        checkpoint_path=checkpoint,
        execution_contract_sha256="a" * 64,
        git_head="0123456789abcdef",
        resume=False,
        initial_caps=runner.BatchCaps(agg=2, prefill=3, decode=4),
        system_loader=system_loader,
        executor=executor,
    )

    assert list(records) == [runner.mode_run_key(spec) for spec in run_specs]
    assert loaded_systems == [spec.point.system for spec in run_specs]
    assert [item[2] for item in executed] == [runner.BatchCaps(agg=2, prefill=3, decode=4)] * 2
    with sqlite3.connect(checkpoint) as connection:
        assert connection.execute("SELECT COUNT(*) FROM mode_runs").fetchone()[0] == 2

    resumed = runner.execute_matrix_runs(
        run_specs,
        checkpoint_path=checkpoint,
        execution_contract_sha256="a" * 64,
        git_head="0123456789abcdef",
        resume=True,
        initial_caps=runner.BatchCaps(),
        system_loader=lambda _system: pytest.fail("completed runs must not reload system specs"),
        executor=lambda *_args, **_kwargs: pytest.fail("completed runs must not execute again"),
    )

    assert resumed == records


def test_execute_matrix_runs_rejects_executor_identity_mismatch(tmp_path):
    run_spec = runner.build_mode_run_specs()[0]
    wrong_spec = replace(run_spec, serving_mode="disagg")

    with pytest.raises(ValueError, match=r"mode-run (key|identity) mismatch"):
        runner.execute_matrix_runs(
            (run_spec,),
            checkpoint_path=tmp_path / "mode_runs.sqlite3",
            execution_contract_sha256="a" * 64,
            git_head="0123456789abcdef",
            resume=False,
            initial_caps=runner.BatchCaps(),
            system_loader=lambda system: {"name": system},
            executor=lambda *_args, **_kwargs: _mode_record(wrong_spec),
        )


def test_merge_completed_checkpoints_builds_full_ordered_checkpoint(tmp_path):
    run_specs = runner.build_mode_run_specs()[:4]
    source_hash = "a" * 64
    git_head = "0123456789abcdef"
    shards = []
    for index, shard_specs in enumerate((run_specs[:2], run_specs[2:])):
        checkpoint = tmp_path / f"shard-{index}.sqlite3"
        runner.initialize_checkpoint(
            checkpoint,
            runner.build_checkpoint_header(
                shard_specs,
                execution_contract_sha256=source_hash,
                git_head=git_head,
            ),
        )
        for run_spec in shard_specs:
            runner.commit_checkpoint_record(checkpoint, run_spec, _mode_record(run_spec))
        shards.append((shard_specs, checkpoint))

    merged = tmp_path / "merged.sqlite3"
    header, records = runner.merge_completed_checkpoints(
        run_specs,
        shards=shards,
        output_checkpoint_path=merged,
        execution_contract_sha256=source_hash,
        git_head=git_head,
    )

    assert header["mode_run_count"] == 4
    assert list(records) == [runner.mode_run_key(run_spec) for run_spec in run_specs]
    loaded_header, loaded_records = runner.load_checkpoint(merged, expected_header=header, run_specs=run_specs)
    assert loaded_header == header
    assert loaded_records == records
    with sqlite3.connect(merged) as connection:
        assert connection.execute("SELECT COUNT(*) FROM mode_runs").fetchone()[0] == 4


def test_merge_completed_checkpoints_rejects_incomplete_or_overlapping_shards(tmp_path):
    run_specs = runner.build_mode_run_specs()[:2]
    source_hash = "a" * 64
    git_head = "0123456789abcdef"
    incomplete = tmp_path / "incomplete.sqlite3"
    header = runner.build_checkpoint_header(
        run_specs,
        execution_contract_sha256=source_hash,
        git_head=git_head,
    )
    runner.initialize_checkpoint(incomplete, header)
    runner.commit_checkpoint_record(incomplete, run_specs[0], _mode_record(run_specs[0]))

    with pytest.raises(ValueError, match="shard checkpoint is incomplete"):
        runner.merge_completed_checkpoints(
            run_specs,
            shards=((run_specs, incomplete),),
            output_checkpoint_path=tmp_path / "incomplete-merged.sqlite3",
            execution_contract_sha256=source_hash,
            git_head=git_head,
        )

    complete = tmp_path / "complete.sqlite3"
    runner.initialize_checkpoint(complete, header)
    for run_spec in run_specs:
        runner.commit_checkpoint_record(complete, run_spec, _mode_record(run_spec))
    with pytest.raises(ValueError, match="overlapping mode runs"):
        runner.merge_completed_checkpoints(
            run_specs,
            shards=((run_specs, complete), (run_specs, complete)),
            output_checkpoint_path=tmp_path / "overlap-merged.sqlite3",
            execution_contract_sha256=source_hash,
            git_head=git_head,
        )


def test_shard_merge_rolls_back_all_records_when_second_insert_fails(tmp_path, monkeypatch):
    run_specs = runner.build_mode_run_specs()[:2]
    contract_hash = "a" * 64
    git_head = "0123456789abcdef"
    shards = []
    for index, run_spec in enumerate(run_specs):
        shard = tmp_path / f"transaction-shard-{index}.sqlite3"
        shard_header = runner.build_checkpoint_header(
            (run_spec,),
            execution_contract_sha256=contract_hash,
            git_head=git_head,
        )
        runner.initialize_checkpoint(shard, shard_header)
        runner.commit_checkpoint_record(shard, run_spec, _mode_record(run_spec))
        shards.append(((run_spec,), shard))

    original_insert = runner._insert_mode_run
    attempted_inserts = 0

    def fail_on_second_insert(connection, record):
        nonlocal attempted_inserts
        attempted_inserts += 1
        if attempted_inserts == 2:
            raise RuntimeError("injected second-record failure")
        return original_insert(connection, record)

    monkeypatch.setattr(runner, "_insert_mode_run", fail_on_second_insert)
    output = tmp_path / "transaction-merged.sqlite3"

    with pytest.raises(RuntimeError, match="injected second-record failure"):
        runner.merge_completed_checkpoints(
            run_specs,
            shards=shards,
            output_checkpoint_path=output,
            execution_contract_sha256=contract_hash,
            git_head=git_head,
        )

    full_header = runner.build_checkpoint_header(
        run_specs,
        execution_contract_sha256=contract_hash,
        git_head=git_head,
    )
    _, records = runner.load_checkpoint(output, expected_header=full_header, run_specs=run_specs)
    assert attempted_inserts == 2
    assert records == {}


def test_resume_rejects_execution_contract_after_step4_source_changes(tmp_path):
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.point.model == "stepfun-ai/Step4")
    sdk_root = tmp_path / "src/aiconfigurator/sdk"
    model_source = sdk_root / "models/step4.py"
    model_source.parent.mkdir(parents=True)
    model_source.write_text("MODEL_VERSION = 1\n")
    (sdk_root / "task_v2.py").write_text("TASK_VERSION = 1\n")
    runner_source = tmp_path / runner.BASE_RUNNER_RELATIVE_PATH
    runner_source.parent.mkdir(parents=True)
    runner_source.write_text("RUNNER_VERSION = 1\n")
    config_root = tmp_path / "src/aiconfigurator/model_configs"
    config_root.mkdir(parents=True)
    (config_root / "stepfun-ai--Step4_config.json").write_text('{"hidden_size":4096}\n')
    system_specs = {run_spec.point.system: {"gpu": {"mem_bw": 1}}}
    caps = runner.BatchCaps()

    def contract_hash():
        contract = runner.build_execution_contract(
            (run_spec,),
            initial_caps=caps,
            repo_root=tmp_path,
            system_loader=lambda name: system_specs[name],
        )
        return runner.execution_contract_sha256(contract)

    stored_hash = contract_hash()
    checkpoint = tmp_path / "resume.sqlite3"
    runner.execute_matrix_runs(
        (run_spec,),
        checkpoint_path=checkpoint,
        execution_contract_sha256=stored_hash,
        git_head="0123456789abcdef",
        resume=False,
        initial_caps=caps,
        system_loader=lambda name: system_specs[name],
        executor=lambda spec, **_kwargs: _mode_record(spec),
    )
    model_source.write_text("MODEL_VERSION = 2\n")
    current_hash = contract_hash()

    assert current_hash != stored_hash
    with pytest.raises(ValueError, match="execution_contract_sha256 mismatch"):
        runner.execute_matrix_runs(
            (run_spec,),
            checkpoint_path=checkpoint,
            execution_contract_sha256=current_hash,
            git_head="0123456789abcdef",
            resume=True,
            initial_caps=caps,
            system_loader=lambda _name: pytest.fail("resume must stop before loading a completed system"),
            executor=lambda *_args, **_kwargs: pytest.fail("resume must stop before re-execution"),
        )


def test_select_mode_run_specs_applies_exact_filters_and_rejects_empty_selection():
    selected = runner.select_mode_run_specs(
        runner.build_mode_run_specs(),
        models=("stepfun-ai/Step4",),
        systems=("h200_sxm",),
        workload_kinds=("primary",),
        isls=(4096,),
        ttft_sla_ms=(5000,),
        serving_modes=("agg",),
    )

    assert len(selected) == 1
    assert selected[0].point.model == "stepfun-ai/Step4"
    assert selected[0].point.osl == 1
    assert selected[0].serving_mode == "agg"
    with pytest.raises(ValueError, match="No mode runs match"):
        runner.select_mode_run_specs(runner.build_mode_run_specs(), systems=("missing",))


def test_finalize_matrix_results_requires_every_selected_mode_run(monkeypatch):
    run_specs = runner.build_mode_run_specs()[:2]
    header = runner.build_checkpoint_header(
        run_specs,
        execution_contract_sha256="a" * 64,
        git_head="0123456789abcdef",
    )
    records = {runner.mode_run_key(run_specs[0]): _mode_record(run_specs[0])}

    with pytest.raises(ValueError, match="incomplete mode-run set"):
        runner.finalize_matrix_results(run_specs, header=header, records=records)

    records[runner.mode_run_key(run_specs[1])] = _mode_record(run_specs[1])

    artifact = runner.finalize_matrix_results(run_specs, header=header, records=records)

    assert artifact["summary"] == {
        "mode_run_count": 2,
        "normalized_row_count": 0,
        "ranked_row_count": 0,
        "paired_comparison_count": 0,
        "unpaired_comparison_count": 0,
        "experiment_terminal_counts": {"memory_infeasible": 6},
    }
    assert artifact["mode_runs"] == [records[runner.mode_run_key(spec)] for spec in run_specs]
    assert artifact["engine_step_backend"] == "python"
    assert artifact["normalized_rows"] == []
    assert artifact["ranked_rows"] == []
    assert artifact["comparison_metrics"] == [
        "ranking_metric_value",
        "ttft",
        "tpot",
        "request_latency",
    ]


def test_finalize_matrix_results_revalidates_mode_run_before_ranking(monkeypatch):
    run_spec = runner.build_mode_run_specs()[0]
    header = runner.build_checkpoint_header(
        (run_spec,),
        execution_contract_sha256="a" * 64,
        git_head="0123456789abcdef",
    )
    record = _mode_record(run_spec)
    record["experiments"] = {}
    monkeypatch.setattr(runner, "rank_final_rows", lambda _rows: pytest.fail("ranking must not run"))

    with pytest.raises(ValueError, match="experiment set mismatch"):
        runner.finalize_matrix_results(
            (run_spec,),
            header=header,
            records={runner.mode_run_key(run_spec): record},
        )


def test_finalize_matrix_results_accepts_primary_disagg_both_zero_tpot(monkeypatch):
    run_specs = tuple(
        spec
        for spec in runner.build_mode_run_specs()
        if spec.point.model in runner.MODELS
        and spec.point.system == "h200_sxm"
        and spec.point.workload_kind == "primary"
        and spec.point.isl == 4096
        and spec.point.ttft_sla_ms == 5000
        and spec.serving_mode == "disagg"
    )
    assert len(run_specs) == 2
    header = runner.build_checkpoint_header(
        run_specs,
        execution_contract_sha256="a" * 64,
        git_head="0123456789abcdef",
    )

    def normalized_row(run_spec):
        point = run_spec.point
        ranking_value = 120.0 if point.model == "stepfun-ai/Step4" else 100.0
        return {
            "model": point.model,
            "system": point.system,
            "workload_kind": point.workload_kind,
            "isl": point.isl,
            "osl": point.osl,
            "prefix": point.prefix,
            "ttft_sla_ms": point.ttft_sla_ms,
            "serving_mode": run_spec.serving_mode,
            "backend": point.backend,
            "database_mode": point.database_mode,
            "nextn": point.nextn,
            "canonical_config_id": f"{point.model}:disagg_AA",
            "canonical_config_sort_key": ["disagg_AA", 1, 1],
            "tokens/s/gpu_cluster": 0.0,
            "ranking_metric_kind": "prefill_input_throughput",
            "ranking_metric_value": ranking_value,
            "terminal_status": "success",
            "cap_saturated": False,
            "ttft": 100.0,
            "ttft_pass": True,
            "ranking_eligible": True,
            "tpot": 0.0,
            "request_latency": 100.0,
        }

    records = {
        runner.mode_run_key(spec): {
            "normalized_rows": [normalized_row(spec)],
            "experiments": {"disagg_AA": {"terminal_status": "success"}},
        }
        for spec in run_specs
    }
    monkeypatch.setattr(runner, "validate_serialized_mode_run", lambda _spec, record: record)

    artifact = runner.finalize_matrix_results(run_specs, header=header, records=records)

    assert artifact["summary"]["paired_comparison_count"] == 1
    assert artifact["comparisons"][0]["metric_deltas"]["tpot"] == {
        "step4_value": 0.0,
        "deepseek_value": 0.0,
        "absolute_delta": 0.0,
        "relative_delta": None,
        "polarity": "lower_is_better",
        "status": "zero_baseline_both_zero",
    }


def test_write_final_artifacts_emits_strict_json_csv_and_markdown(tmp_path):
    artifact = {
        "checkpoint_header": {"mode_run_count": 1},
        "summary": {
            "mode_run_count": 1,
            "normalized_row_count": 1,
            "ranked_row_count": 1,
            "paired_comparison_count": 1,
            "unpaired_comparison_count": 0,
            "experiment_terminal_counts": {"success": 2},
        },
        "ranking_contract": runner.RANKING_CONTRACT,
        "delta_contract": runner.DELTA_CONTRACT,
        "comparison_metrics": ["ranking_metric_value", "tpot"],
        "mode_runs": [],
        "normalized_rows": [{"nested": {"source": "sol"}}],
        "ranked_rows": [
            {
                "model": "stepfun-ai/Step4",
                "system": "h200_sxm",
                "workload_kind": "primary",
                "isl": 4096,
                "osl": 1,
                "ttft_sla_ms": 5000,
                "serving_mode": "agg",
                "ranking_metric_kind": "prefill_input_throughput",
                "ranking_metric_value": 123.5,
                "ttft": 156.256,
                "tpot": 12.0,
                "canonical_config_id": "agg_patternB:8:1",
                "rank": 1,
                "nested": {"source": "sol"},
            }
        ],
        "comparisons": [
            {
                "aligned_key": ["h200_sxm", "primary"],
                "status": "paired",
                "step4_config_id": "step4",
                "deepseek_config_id": "deepseek",
                "metric_deltas": {
                    "ranking_metric_value": {
                        "step4_value": 123.5,
                        "deepseek_value": 100.0,
                        "absolute_delta": 23.5,
                        "relative_delta": 0.235,
                        "polarity": "higher_is_better",
                        "status": "computed",
                    },
                    "tpot": {
                        "step4_value": 0.0,
                        "deepseek_value": 0.0,
                        "absolute_delta": 0.0,
                        "relative_delta": None,
                        "polarity": "lower_is_better",
                        "status": "zero_baseline_both_zero",
                    },
                },
            }
        ],
    }

    paths = runner.write_final_artifacts(tmp_path, artifact)

    assert set(paths) == {"json", "ranked_csv", "comparisons_csv", "markdown"}
    assert json.loads(paths["json"].read_text()) == artifact
    with paths["ranked_csv"].open(newline="") as stream:
        ranked = list(csv.DictReader(stream))
    assert ranked[0]["ranking_metric_value"] == "123.5"
    assert ranked[0]["nested"] == '{"source":"sol"}'
    with paths["comparisons_csv"].open(newline="") as stream:
        comparisons = list(csv.DictReader(stream))
    assert comparisons[0]["status"] == "paired"
    metric_deltas = json.loads(comparisons[0]["metric_deltas"])
    assert metric_deltas["ranking_metric_value"]["relative_delta"] == 0.235
    assert metric_deltas["tpot"] == {
        "absolute_delta": 0.0,
        "deepseek_value": 0.0,
        "polarity": "lower_is_better",
        "relative_delta": None,
        "status": "zero_baseline_both_zero",
        "step4_value": 0.0,
    }
    report = paths["markdown"].read_text()
    assert "Mode runs | 1" in report
    assert "Temporary MLA substitution" in report
    assert "123.5" in report
    assert (
        "| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |"
    ) in report


def test_write_final_artifacts_reports_missing_rows_without_fabricated_values(tmp_path):
    """Empty result sets must be described explicitly rather than encoded as zero-valued rows."""
    artifact = {
        "summary": {
            "mode_run_count": 1,
            "normalized_row_count": 0,
            "ranked_row_count": 0,
            "paired_comparison_count": 0,
            "unpaired_comparison_count": 0,
        },
        "comparison_metrics": ["ranking_metric_value"],
        "ranked_rows": [],
        "comparisons": [],
    }

    report = runner.write_final_artifacts(tmp_path, artifact)["markdown"].read_text()

    assert "No rank-one results were produced." in report
    assert "No paired model deltas were available." in report
    assert "| None |" not in report


def test_parse_args_rejects_nonpositive_caps():
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "results", "--initial-agg-cap", "0"])
