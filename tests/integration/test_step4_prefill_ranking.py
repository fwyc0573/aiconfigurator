"""Integration coverage for Step4 primary Prefill-throughput ranking."""

from __future__ import annotations

import pytest

from tests.performance.aic_roofline_pareto import run_step4_comparison as runner

pytestmark = pytest.mark.integration


def test_step4_primary_disagg_osl_one_returns_prefill_ranked_candidate():
    point = next(
        point
        for point in runner.build_matrix_points()
        if point.model == "stepfun-ai/Step4"
        and point.system == "h200_sxm"
        and point.workload_kind == "primary"
        and point.isl == 4096
        and point.osl == 1
        and point.ttft_sla_ms == 500
    )
    run_spec = runner.ModeRunSpec(point=point, serving_mode="disagg")
    task = runner.build_comparison_task(
        run_spec,
        experiment="disagg_BB",
        caps=runner.BatchCaps(agg=1, prefill=1, decode=1),
    )

    rows = task.run().to_dict(orient="records")

    assert task.disagg_ranking_metric_kind == "prefill_input_throughput"
    assert len(rows) == 1
    row = rows[0]
    evidence = runner.derive_ranking_metric_evidence(run_spec, row)
    expected_system_throughput = (
        row["(p)global_bs"] * row["(p)workers"] * (point.isl - point.prefix) / (row["ttft"] / 1000.0)
    )
    assert row["tpot"] == 0.0
    assert row["tokens/s/gpu"] == 0.0
    assert row["num_total_gpus"] == 64
    assert evidence["ranking_metric_kind"] == "prefill_input_throughput"
    assert evidence["prefill_tokens/s"] == pytest.approx(expected_system_throughput)
    assert evidence["ranking_metric_value"] == pytest.approx(expected_system_throughput / 64)
    assert evidence["ranking_metric_value"] > 0.0
