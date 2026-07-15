import copy
import sqlite3
from collections import defaultdict
from dataclasses import replace

import pytest

from aiconfigurator.sdk import perf_database
from aiconfigurator.sdk.task_v2 import SinglePointEvaluation
from tests.performance.aic_roofline_pareto import run_step4_comparison as runner

EXPERIMENTS_BY_MODE = {
    "agg": ("agg_patternA", "agg_patternB"),
    "disagg": ("disagg_AA", "disagg_AB", "disagg_BA", "disagg_BB"),
}


def _header(run_specs):
    return runner.build_checkpoint_header(
        run_specs,
        execution_contract_sha256="a" * 64,
        git_head="0123456789abcdef",
    )


def _successful_cap_result():
    caps = runner.BatchCaps(agg=8, prefill=16, decode=1024)
    evaluation = SinglePointEvaluation(
        row={"exact": 2.5},
        per_ops_data={"context": {"attention": 1.0}},
        per_ops_source={"context": {"attention": "sol"}},
        communication_evidence=(),
    )
    attempt = runner.SearchAttempt(
        rank1_batch_sizes={"agg": 4},
        candidate_rows=({"candidate": 1.0}, {"candidate": 2.0}),
        rank1_row={"candidate": 2.0},
        selected_point_identity=("agg_patternA", 1, 1),
        selected_evaluation=evaluation,
        per_ops_evidence={"phase_totals_ms": {"context": 1.0}},
    )
    return runner.CapSearchResult(
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


def _terminal_cap_result(experiment="agg_patternB"):
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


def _valid_success_cap_result(run_spec, experiment):
    caps = runner.BatchCaps(agg=8, prefill=16, decode=1024)
    _mode, prefill_pattern, decode_pattern = runner.EXPERIMENT_PATTERNS[experiment]
    if run_spec.serving_mode == "agg":
        if prefill_pattern == "A":
            tp, dp, moe_tp, moe_ep = 1, 2, 1, 2
        else:
            tp, dp, moe_tp, moe_ep = 2, 1, 2, 1
        row = {
            "model": run_spec.point.model,
            "isl": run_spec.point.isl,
            "osl": run_spec.point.osl,
            "prefix": run_spec.point.prefix,
            "ttft": run_spec.point.ttft_sla_ms / 2,
            "tpot": 9.0,
            "request_latency": run_spec.point.ttft_sla_ms / 2,
            "seq/s": 100.0,
            "seq/s/gpu": 50.0,
            "tokens/s": 100.0,
            "tokens/s/gpu": 50.0,
            "tokens/s/user": 1.0,
            "num_total_gpus": tp * dp,
            "bs": 4,
            "global_bs": 4 * dp,
            "tp": tp,
            "pp": 1,
            "dp": dp,
            "moe_tp": moe_tp,
            "moe_ep": moe_ep,
            "cp": 1,
            "ctx_tokens": run_spec.point.isl,
            "parallel": f"tp{tp}pp1dp{dp}moetp{moe_tp}moeep{moe_ep}cp1",
            "gemm": 1.0,
            "kvcache": 2.0,
            "fmha": 3.0,
            "moe": 4.0,
            "comm": 5.0,
            "memory": 60_000_000_000.0,
            "backend": run_spec.point.backend,
            "version": run_spec.point.backend_version,
            "system": run_spec.point.system,
        }
        rank1_batch_sizes = {"agg": 4}
        selected_identity = runner._canonical_agg_sort_key(experiment, row)
        evaluation = SinglePointEvaluation(
            row=row,
            per_ops_data={
                "mix_step": {"context_mla_attention": 1.0},
                "scheduling": {"num_mix_steps": 1.0, "num_genonly_steps": 0.0},
            },
            per_ops_source={"mix_step": {"context_mla_attention": "sol"}},
            communication_evidence=(),
        )
    else:
        if prefill_pattern == "A":
            prefill_tp, prefill_dp, prefill_moe_tp, prefill_moe_ep = 1, 2, 1, 2
        else:
            prefill_tp, prefill_dp, prefill_moe_tp, prefill_moe_ep = 2, 1, 2, 1
        if decode_pattern == "A":
            decode_tp, decode_dp, decode_moe_tp, decode_moe_ep = 1, 2, 1, 2
        else:
            decode_tp, decode_dp, decode_moe_tp, decode_moe_ep = 2, 1, 2, 1
        decode_data = {} if run_spec.point.osl == 1 else {"generation_mla_bmm": 0.5}
        decode_source = {} if run_spec.point.osl == 1 else {"generation_mla_bmm": "sol"}
        row = {
            "model": run_spec.point.model,
            "isl": run_spec.point.isl,
            "osl": run_spec.point.osl,
            "prefix": run_spec.point.prefix,
            "ttft": run_spec.point.ttft_sla_ms / 2,
            "tpot": 9.0,
            "request_latency": run_spec.point.ttft_sla_ms / 2,
            "seq/s": 100.0,
            "seq/s/gpu": 25.0,
            "tokens/s": 100.0,
            "tokens/s/gpu": 25.0,
            "tokens/s/user": 1.0,
            "num_total_gpus": prefill_tp * prefill_dp + decode_tp * decode_dp,
            "(p)bs": 8,
            "(p)global_bs": 8 * prefill_dp,
            "(p)workers": 1,
            "(d)bs": 128,
            "(d)global_bs": 128 * decode_dp,
            "(d)workers": 1,
            "(p)tp": prefill_tp,
            "(p)pp": 1,
            "(p)dp": prefill_dp,
            "(p)moe_tp": prefill_moe_tp,
            "(p)moe_ep": prefill_moe_ep,
            "(p)cp": 1,
            "(p)parallel": (f"tp{prefill_tp}pp1dp{prefill_dp}moetp{prefill_moe_tp}moeep{prefill_moe_ep}cp1"),
            "(p)gemm": 1.0,
            "(p)kvcache": 2.0,
            "(p)fmha": 3.0,
            "(p)moe": 4.0,
            "(p)comm": 5.0,
            "(p)memory": 60_000_000_000.0,
            "(p)backend": run_spec.point.backend,
            "(p)version": run_spec.point.backend_version,
            "(p)system": run_spec.point.system,
            "(d)tp": decode_tp,
            "(d)pp": 1,
            "(d)dp": decode_dp,
            "(d)moe_tp": decode_moe_tp,
            "(d)moe_ep": decode_moe_ep,
            "(d)parallel": (f"tp{decode_tp}pp1dp{decode_dp}moetp{decode_moe_tp}moeep{decode_moe_ep}"),
            "(d)gemm": 1.5,
            "(d)kvcache": 2.5,
            "(d)fmha": 3.5,
            "(d)moe": 4.5,
            "(d)comm": 5.5,
            "(d)memory": 70_000_000_000.0,
            "(d)backend": run_spec.point.backend,
            "(d)version": run_spec.point.backend_version,
            "(d)system": run_spec.point.system,
        }
        rank1_batch_sizes = {"prefill": 8, "decode": 128}
        selected_identity = runner._canonical_disagg_sort_key(experiment, row, decode_cp=1)
        evaluation = SinglePointEvaluation(
            row=row,
            per_ops_data={
                "prefill": {"context_mla_attention": 1.0},
                "decode": decode_data,
            },
            per_ops_source={
                "prefill": {"context_mla_attention": "sol"},
                "decode": decode_source,
            },
            communication_evidence=(),
        )
    attempt = runner.SearchAttempt(
        rank1_batch_sizes=rank1_batch_sizes,
        candidate_rows=(dict(row),),
        rank1_row=dict(row),
        selected_point_identity=selected_identity,
        selected_evaluation=evaluation,
        per_ops_evidence=runner.validate_per_ops_evidence(
            evaluation,
            serving_mode=run_spec.serving_mode,
            osl=run_spec.point.osl,
        ),
    )
    return runner.CapSearchResult(
        terminal_status="success",
        final_caps=caps,
        cap_history=(caps,),
        cap_rerun_count=0,
        cap_saturated=False,
        ranking_eligible=True,
        attempt_evidence=(
            runner.CapAttemptEvidence(
                experiment=experiment,
                caps=caps,
                status="success",
                search_attempt=attempt,
            ),
        ),
    )


def _normalized_success_row(run_spec, experiment, cap_result):
    task = runner.build_comparison_task(run_spec, experiment=experiment, caps=cap_result.final_caps)
    return runner.normalize_completed_experiment(
        run_spec,
        experiment=experiment,
        task=task,
        cap_result=cap_result,
        system_spec=perf_database.load_system_spec(run_spec.point.system),
    )


def _valid_mode_record(run_spec, *, successful_experiments=()):
    success_set = set(successful_experiments)
    cap_results = {
        experiment: (
            _valid_success_cap_result(run_spec, experiment)
            if experiment in success_set
            else _terminal_cap_result(experiment)
        )
        for experiment in EXPERIMENTS_BY_MODE[run_spec.serving_mode]
    }
    normalized_rows = tuple(
        _normalized_success_row(run_spec, experiment, cap_results[experiment])
        for experiment in EXPERIMENTS_BY_MODE[run_spec.serving_mode]
        if experiment in success_set
    )
    return runner.build_mode_run_record(
        run_spec,
        cap_results=cap_results,
        normalized_rows=normalized_rows,
    )


def _mutate_normalized_row(record, mutation):
    mutated = copy.deepcopy(record)
    row = mutated["normalized_rows"][0]
    if mutation == "ranking_metric_value":
        row["ranking_metric_value"] += 1.0
    elif mutation == "canonical_config_id":
        row["canonical_config_id"] += "|mutated=1"
    elif mutation == "canonical_config_sort_key":
        row["canonical_config_sort_key"][-1] += 1
    elif mutation == "per_ops_weighted_totals_ms":
        row["per_ops_weighted_totals_ms"]["total"] += 1.0
    elif mutation == "extra":
        row["extra_published_field"] = "unexpected"
    elif mutation == "missing":
        row.pop("ranking_metric_kind")
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    return mutated


def test_candidate_fingerprint_uses_strict_canonical_json():
    first = ({"b": 2, "a": 1.25}, {"nested": {"z": "sol", "x": 0.0}})
    same_values = ({"a": 1.25, "b": 2}, {"nested": {"x": 0.0, "z": "sol"}})
    reversed_rows = tuple(reversed(first))

    fingerprint = runner.candidate_fingerprint(first)

    assert fingerprint == runner.candidate_fingerprint(same_values)
    assert fingerprint["candidate_count"] == 2
    assert len(fingerprint["candidate_sha256"]) == 64
    assert fingerprint != runner.candidate_fingerprint(reversed_rows)
    with pytest.raises(ValueError, match="finite JSON number"):
        runner.candidate_fingerprint(({"bad": float("nan")},))


def test_checkpoint_round_trip_preserves_header_and_completed_mode_run(tmp_path):
    run_specs = runner.build_mode_run_specs()[:2]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header(run_specs)
    record = _valid_mode_record(run_specs[0], successful_experiments=("agg_patternA",))

    runner.initialize_checkpoint(checkpoint, header)
    runner.commit_checkpoint_record(checkpoint, run_specs[0], record)
    loaded_header, records = runner.load_checkpoint(
        checkpoint,
        expected_header=header,
        run_specs=run_specs,
    )

    assert loaded_header == header
    assert records == {record["mode_run_key"]: record}
    with sqlite3.connect(checkpoint) as connection:
        assert connection.execute("SELECT COUNT(*) FROM mode_runs").fetchone()[0] == 1


def test_checkpoint_rejects_schema_matrix_or_source_state_mismatch(tmp_path):
    run_specs = runner.build_mode_run_specs()[:2]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header(run_specs)
    runner.initialize_checkpoint(checkpoint, header)

    for mutation, message in (
        ({"schema_version": 999}, "schema_version mismatch"),
        ({"matrix_spec_hash": "b" * 64}, "matrix_spec_hash mismatch"),
        ({"execution_contract_sha256": "c" * 64}, "execution_contract_sha256 mismatch"),
    ):
        with pytest.raises(ValueError, match=message):
            runner.load_checkpoint(checkpoint, expected_header=header | mutation, run_specs=run_specs)


def test_checkpoint_rejects_duplicate_completed_mode_run_key(tmp_path):
    run_specs = runner.build_mode_run_specs()[:1]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header(run_specs)
    record = _valid_mode_record(run_specs[0])
    runner.initialize_checkpoint(checkpoint, header)
    runner.commit_checkpoint_record(checkpoint, run_specs[0], record)

    with pytest.raises(ValueError, match="Duplicate completed mode_run_key"):
        runner.commit_checkpoint_record(checkpoint, run_specs[0], record)


def test_checkpoint_rejects_malformed_sqlite_payload_without_fallback(tmp_path):
    run_spec = runner.build_mode_run_specs()[0]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header((run_spec,))
    runner.initialize_checkpoint(checkpoint, header)
    with sqlite3.connect(checkpoint) as connection:
        connection.execute(
            "INSERT INTO mode_runs(mode_run_key, payload) VALUES (?, ?)",
            (runner.mode_run_key(run_spec), '{"record_type":"mode_run"'),
        )

    with pytest.raises(ValueError, match="Malformed mode-run JSON"):
        runner.load_checkpoint(checkpoint, expected_header=header, run_specs=(run_spec,))


def test_checkpoint_load_revalidates_complete_mode_run_semantics(tmp_path):
    run_spec = runner.build_mode_run_specs()[0]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header((run_spec,))
    record = _valid_mode_record(run_spec)
    record["experiments"] = {}
    runner.initialize_checkpoint(checkpoint, header)
    with sqlite3.connect(checkpoint) as connection:
        connection.execute(
            "INSERT INTO mode_runs(mode_run_key, payload) VALUES (?, ?)",
            (runner.mode_run_key(run_spec), runner._canonical_json(record)),
        )

    with pytest.raises(ValueError, match="experiment set mismatch"):
        runner.load_checkpoint(checkpoint, expected_header=header, run_specs=(run_spec,))


def test_checkpoint_path_must_not_be_preexisting_when_initializing(tmp_path):
    run_specs = runner.build_mode_run_specs()[:1]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    checkpoint.write_text("unrelated content\n")

    with pytest.raises(FileExistsError, match="already exists"):
        runner.initialize_checkpoint(checkpoint, _header(run_specs))


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_checkpoint_header_schema_is_exact_before_file_creation(tmp_path, mutation):
    run_specs = runner.build_mode_run_specs()[:1]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header(run_specs)
    if mutation == "missing":
        header.pop("git_head")
    else:
        header["unexpected"] = "field"

    with pytest.raises(ValueError, match="checkpoint header fields mismatch"):
        runner.initialize_checkpoint(checkpoint, header)

    assert not checkpoint.exists()


def test_checkpoint_header_insert_failure_does_not_claim_target_path(tmp_path, monkeypatch):
    run_specs = runner.build_mode_run_specs()[:1]
    checkpoint = tmp_path / "mode_runs.sqlite3"

    class InsertFailingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def executescript(self, _script):
            return None

        def execute(self, statement, _parameters=()):
            if statement.lstrip().startswith("INSERT INTO checkpoint_header"):
                raise sqlite3.OperationalError("injected checkpoint-header insert failure")
            return self

        def close(self):
            return None

    monkeypatch.setattr(runner.sqlite3, "connect", lambda *_args, **_kwargs: InsertFailingConnection())

    with pytest.raises(sqlite3.OperationalError, match="injected checkpoint-header insert failure"):
        runner.initialize_checkpoint(checkpoint, _header(run_specs))

    assert not checkpoint.exists()


def test_checkpoint_initialization_creates_exactly_one_header_and_zero_mode_runs(tmp_path):
    run_specs = runner.build_mode_run_specs()[:1]
    checkpoint = tmp_path / "mode_runs.sqlite3"

    runner.initialize_checkpoint(checkpoint, _header(run_specs))

    with sqlite3.connect(checkpoint) as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoint_header").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM mode_runs").fetchone()[0] == 0


def test_checkpoint_load_rejects_extra_stored_header_field(tmp_path):
    run_specs = runner.build_mode_run_specs()[:1]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header(run_specs)
    runner.initialize_checkpoint(checkpoint, header)
    with sqlite3.connect(checkpoint) as connection:
        connection.execute(
            "UPDATE checkpoint_header SET payload = ? WHERE singleton = 1",
            (runner._canonical_json(header | {"unexpected": "field"}),),
        )

    with pytest.raises(ValueError, match="checkpoint header fields mismatch"):
        runner.load_checkpoint(checkpoint, expected_header=header, run_specs=run_specs)


def test_commit_rejects_semantic_mismatch_before_sqlite_transaction(tmp_path):
    run_specs = runner.build_mode_run_specs()[:2]
    checkpoint = tmp_path / "mode_runs.sqlite3"
    header = _header(run_specs)
    runner.initialize_checkpoint(checkpoint, header)
    runner.commit_checkpoint_record(checkpoint, run_specs[0], _valid_mode_record(run_specs[0]))
    invalid = _valid_mode_record(run_specs[1])
    invalid["experiments"] = {}

    with sqlite3.connect(checkpoint) as connection:
        before = connection.execute("SELECT COUNT(*) FROM mode_runs").fetchone()[0]
    with pytest.raises(ValueError, match="experiment set mismatch"):
        runner.commit_checkpoint_record(checkpoint, run_specs[1], invalid)
    with sqlite3.connect(checkpoint) as connection:
        after = connection.execute("SELECT COUNT(*) FROM mode_runs").fetchone()[0]

    assert before == after == 1


def test_mode_run_key_changes_for_every_execution_dimension():
    base = runner.build_mode_run_specs()[0]
    base_identity = runner.mode_run_identity(base)
    keys = {runner.mode_run_key(base)}

    for field, replacement in (
        ("model", "deepseek-ai/DeepSeek-V4-Pro"),
        ("system", "h200_sxm"),
        ("workload_kind", "decode_smoke"),
        ("isl", 16384),
        ("osl", 1024),
        ("ttft_sla_ms", 500),
        ("backend_version", "0.22.1"),
        ("engine_step_backend", "rust"),
        ("total_gpus", 32),
        ("nextn", 3),
    ):
        changed_identity = dict(base_identity)
        changed_identity[field] = replacement
        keys.add(runner.mode_run_key_from_identity(changed_identity))
    changed_mode = dict(base_identity)
    changed_mode["serving_mode"] = "disagg"
    keys.add(runner.mode_run_key_from_identity(changed_mode))

    assert len(keys) == 12


def test_cap_search_serialization_is_compact_and_preserves_exact_evidence():
    serialized = runner.serialize_cap_search_result(_successful_cap_result())
    attempt = serialized["attempt_evidence"][0]["search_attempt"]

    assert attempt["candidate_count"] == 2
    assert len(attempt["candidate_sha256"]) == 64
    assert "candidate_rows" not in attempt
    assert attempt["rank1_row"] == {"candidate": 2.0}
    assert attempt["selected_point_identity"] == ["agg_patternA", 1, 1]
    assert attempt["selected_evaluation"] == {
        "row": {"exact": 2.5},
        "per_ops_data": {"context": {"attention": 1.0}},
        "per_ops_source": {"context": {"attention": "sol"}},
        "communication_evidence": [],
    }
    assert attempt["per_ops_evidence"] == {"phase_totals_ms": {"context": 1.0}}


def test_cap_search_serialization_handles_defaultdict_in_selected_evaluation():
    cap_result = _successful_cap_result()
    attempt = cap_result.attempt_evidence[0].search_attempt
    evaluation = replace(
        attempt.selected_evaluation,
        row=defaultdict(dict, {"exact": 2.5}),
    )
    updated_attempt = replace(attempt, selected_evaluation=evaluation)
    evidence = replace(cap_result.attempt_evidence[0], search_attempt=updated_attempt)

    serialized = runner.serialize_cap_search_result(replace(cap_result, attempt_evidence=(evidence,)))

    assert serialized["attempt_evidence"][0]["search_attempt"]["selected_evaluation"]["row"] == {"exact": 2.5}


def test_terminal_cap_search_serialization_preserves_typed_error():
    serialized = runner.serialize_cap_search_result(_terminal_cap_result())

    assert serialized["terminal_status"] == "memory_infeasible"
    assert serialized["ranking_eligible"] is False
    assert serialized["attempt_evidence"] == [
        {
            "experiment": "agg_patternB",
            "caps": {"agg": 1024, "prefill": 16, "decode": 1024},
            "status": "memory_infeasible",
            "search_attempt": None,
            "error_type": "InsufficientMemoryError",
            "error_message": "model does not fit",
        }
    ]


def test_serialized_mode_run_validator_accepts_aggregate_and_disaggregate_round_trips():
    aggregate = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    disaggregate = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "disagg")
    aggregate_record = _valid_mode_record(aggregate, successful_experiments=("agg_patternA",))
    disaggregate_record = _valid_mode_record(disaggregate, successful_experiments=("disagg_AB",))

    assert runner.validate_serialized_mode_run(aggregate, aggregate_record) == aggregate_record
    assert runner.validate_serialized_mode_run(disaggregate, disaggregate_record) == disaggregate_record
    assert set(aggregate_record["experiments"]) == {"agg_patternA", "agg_patternB"}
    assert set(disaggregate_record["experiments"]) == {
        "disagg_AA",
        "disagg_AB",
        "disagg_BA",
        "disagg_BB",
    }
    assert len(aggregate_record["normalized_rows"]) == 1
    assert len(disaggregate_record["normalized_rows"]) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "ranking_metric_value",
        "canonical_config_id",
        "canonical_config_sort_key",
        "per_ops_weighted_totals_ms",
        "extra",
        "missing",
    ),
)
def test_serialized_mode_run_validator_rejects_normalized_row_mutations(mutation):
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    record = _valid_mode_record(run_spec, successful_experiments=("agg_patternA",))

    with pytest.raises(ValueError, match="normalized row"):
        runner.validate_serialized_mode_run(run_spec, _mutate_normalized_row(record, mutation))


@pytest.mark.parametrize("boundary", ("build", "commit", "load", "merge", "finalize"))
def test_normalized_row_cross_binding_is_enforced_at_every_durable_boundary(tmp_path, boundary):
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    success = _valid_success_cap_result(run_spec, "agg_patternA")
    terminal = _terminal_cap_result()
    cap_results = {"agg_patternA": success, "agg_patternB": terminal}
    normalized = _normalized_success_row(run_spec, "agg_patternA", success)

    if boundary == "build":
        mutated = _mutate_normalized_row({"normalized_rows": [normalized]}, "ranking_metric_value")
        with pytest.raises(ValueError, match="normalized row"):
            runner.build_mode_run_record(
                run_spec,
                cap_results=cap_results,
                normalized_rows=mutated["normalized_rows"],
            )
        return

    valid_record = runner.build_mode_run_record(
        run_spec,
        cap_results=cap_results,
        normalized_rows=(normalized,),
    )
    mutated_record = _mutate_normalized_row(valid_record, "ranking_metric_value")
    header = _header((run_spec,))

    if boundary == "commit":
        checkpoint = tmp_path / "commit.sqlite3"
        runner.initialize_checkpoint(checkpoint, header)
        with pytest.raises(ValueError, match="normalized row"):
            runner.commit_checkpoint_record(checkpoint, run_spec, mutated_record)
        with sqlite3.connect(checkpoint) as connection:
            assert connection.execute("SELECT COUNT(*) FROM mode_runs").fetchone()[0] == 0
        return

    if boundary in {"load", "merge"}:
        checkpoint = tmp_path / f"{boundary}-input.sqlite3"
        runner.initialize_checkpoint(checkpoint, header)
        with sqlite3.connect(checkpoint) as connection:
            connection.execute(
                "INSERT INTO mode_runs(mode_run_key, payload) VALUES (?, ?)",
                (runner.mode_run_key(run_spec), runner._canonical_json(mutated_record)),
            )
        if boundary == "load":
            with pytest.raises(ValueError, match="normalized row"):
                runner.load_checkpoint(checkpoint, expected_header=header, run_specs=(run_spec,))
        else:
            with pytest.raises(ValueError, match="normalized row"):
                runner.merge_completed_checkpoints(
                    (run_spec,),
                    shards=(((run_spec,), checkpoint),),
                    output_checkpoint_path=tmp_path / "merged.sqlite3",
                    execution_contract_sha256="a" * 64,
                    git_head="0123456789abcdef",
                )
        return

    with pytest.raises(ValueError, match="normalized row"):
        runner.finalize_matrix_results(
            (run_spec,),
            header=header,
            records={runner.mode_run_key(run_spec): mutated_record},
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_serialized_mode_run_validator_rejects_wrong_experiment_set(mutation):
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    record = _valid_mode_record(run_spec)
    if mutation == "missing":
        record["experiments"].pop("agg_patternB")
    else:
        record["experiments"]["disagg_AA"] = record["experiments"]["agg_patternA"]

    with pytest.raises(ValueError, match="experiment set mismatch"):
        runner.validate_serialized_mode_run(run_spec, record)


def test_execution_contract_fingerprint_tracks_only_used_runtime_inputs(tmp_path):
    step4 = next(spec for spec in runner.build_mode_run_specs() if spec.point.model == "stepfun-ai/Step4")
    deepseek = next(spec for spec in runner.build_mode_run_specs() if spec.point.model == "deepseek-ai/DeepSeek-V4-Pro")
    run_specs = (step4, deepseek)
    sdk_root = tmp_path / "src/aiconfigurator/sdk"
    sdk_root.mkdir(parents=True)
    step4_source = sdk_root / "models/step4.py"
    step4_source.parent.mkdir(parents=True)
    step4_source.write_text("STEP4 = 1\n")
    (sdk_root / "task_v2.py").write_text("TASK = 1\n")
    runner_source = tmp_path / runner.BASE_RUNNER_RELATIVE_PATH
    runner_source.parent.mkdir(parents=True)
    runner_source.write_text("RUNNER = 1\n")
    configs = tmp_path / "src/aiconfigurator/model_configs"
    configs.mkdir(parents=True)
    for model in {spec.point.model for spec in run_specs}:
        (configs / f"{model.replace('/', '--')}_config.json").write_text('{"hidden_size":4096}\n')
    unrelated = tmp_path / "src/aiconfigurator/generator/unrelated.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("UNRELATED = 1\n")
    system_state = {step4.point.system: {"gpu": {"mem_bw": 1}}}

    def contract(caps=None):
        active_caps = runner.BatchCaps() if caps is None else caps
        return runner.build_execution_contract(
            run_specs,
            initial_caps=active_caps,
            repo_root=tmp_path,
            system_loader=lambda name: system_state[name],
        )

    def fingerprint(caps=None):
        return runner.execution_contract_sha256(contract(caps))

    assert contract()["engine_step_backend"] == "python"
    baseline = fingerprint()
    step4_source.write_text("STEP4 = 2\n")
    assert fingerprint() != baseline
    step4_source.write_text("STEP4 = 1\n")
    runner_source.write_text("RUNNER = 2\n")
    assert fingerprint() != baseline
    runner_source.write_text("RUNNER = 1\n")
    step4_config = configs / "stepfun-ai--Step4_config.json"
    step4_config.write_text('{"hidden_size":8192}\n')
    assert fingerprint() != baseline
    step4_config.write_text('{"hidden_size":4096}\n')
    deepseek_config = configs / "deepseek-ai--DeepSeek-V4-Pro_config.json"
    deepseek_config.write_text('{"hidden_size":8192}\n')
    assert fingerprint() != baseline
    deepseek_config.write_text('{"hidden_size":4096}\n')
    system_state[step4.point.system] = {"gpu": {"mem_bw": 2}}
    assert fingerprint() != baseline
    system_state[step4.point.system] = {"gpu": {"mem_bw": 1}}
    assert fingerprint(runner.BatchCaps(agg=2048)) != baseline
    unrelated.write_text("UNRELATED = 2\n")
    assert fingerprint() == baseline


def test_cap_search_serialization_rejects_inconsistent_success_or_terminal_evidence():
    with pytest.raises(ValueError, match="Successful cap result"):
        runner.serialize_cap_search_result(replace(_successful_cap_result(), ranking_eligible=False))

    terminal = _terminal_cap_result()
    missing_error_type = replace(terminal.attempt_evidence[0], error_type=None)
    with pytest.raises(ValueError, match="error type and message"):
        runner.serialize_cap_search_result(replace(terminal, attempt_evidence=(missing_error_type,)))


def test_mode_run_record_requires_complete_experiment_set_and_exact_success_rows():
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    success = _valid_success_cap_result(run_spec, "agg_patternA")
    terminal = _terminal_cap_result()
    normalized = _normalized_success_row(run_spec, "agg_patternA", success)

    record = runner.build_mode_run_record(
        run_spec,
        cap_results={"agg_patternA": success, "agg_patternB": terminal},
        normalized_rows=(normalized,),
    )

    assert record["record_type"] == "mode_run"
    assert record["mode_run_key"] == runner.mode_run_key(run_spec)
    assert record["mode_run_identity"] == runner.mode_run_identity(run_spec)
    assert record["normalized_rows"] == [runner._jsonable(normalized)]
    assert set(record["experiments"]) == {"agg_patternA", "agg_patternB"}
    with pytest.raises(ValueError, match="experiment set mismatch"):
        runner.build_mode_run_record(
            run_spec,
            cap_results={"agg_patternA": success},
            normalized_rows=(normalized,),
        )
    with pytest.raises(ValueError, match="normalized success rows mismatch"):
        runner.build_mode_run_record(
            run_spec,
            cap_results={"agg_patternA": success, "agg_patternB": terminal},
            normalized_rows=(),
        )


def test_mode_run_record_rejects_attempt_evidence_for_another_experiment():
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    success = _valid_success_cap_result(run_spec, "agg_patternA")
    wrong_evidence = replace(success.attempt_evidence[0], experiment="agg_patternB")

    with pytest.raises(ValueError, match="attempt experiment mismatch"):
        runner.build_mode_run_record(
            run_spec,
            cap_results={
                "agg_patternA": replace(success, attempt_evidence=(wrong_evidence,)),
                "agg_patternB": _terminal_cap_result(),
            },
            normalized_rows=(_normalized_success_row(run_spec, "agg_patternA", success),),
        )


def test_execute_mode_run_normalizes_only_successful_experiments(monkeypatch):
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    success = _valid_success_cap_result(run_spec, "agg_patternA")
    terminal = _terminal_cap_result()
    normalized = _normalized_success_row(run_spec, "agg_patternA", success)
    tasks = []

    monkeypatch.setattr(
        runner,
        "run_all_experiment_cap_searches",
        lambda spec, **_kwargs: {"agg_patternA": success, "agg_patternB": terminal},
    )

    def task_factory(spec, *, experiment, caps):
        tasks.append((spec, experiment, caps))
        return object()

    monkeypatch.setattr(
        runner,
        "normalize_completed_experiment",
        lambda spec, *, experiment, task, cap_result, system_spec: copy.deepcopy(normalized),
    )

    record = runner.execute_mode_run(
        run_spec,
        system_spec={"name": "test-system"},
        task_factory=task_factory,
    )

    assert [item[1] for item in tasks] == ["agg_patternA"]
    assert tasks[0][2] == success.final_caps
    assert [row["experiment"] for row in record["normalized_rows"]] == ["agg_patternA"]
    assert record["experiments"]["agg_patternB"]["terminal_status"] == "memory_infeasible"


def test_execute_mode_run_rejects_incomplete_cap_result_set(monkeypatch):
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.serving_mode == "agg")
    monkeypatch.setattr(
        runner,
        "run_all_experiment_cap_searches",
        lambda spec, **_kwargs: {"agg_patternA": _terminal_cap_result()},
    )

    with pytest.raises(ValueError, match="experiment set mismatch"):
        runner.execute_mode_run(
            run_spec,
            system_spec={"name": "test-system"},
        )
