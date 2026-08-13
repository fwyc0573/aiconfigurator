"""Contract checks for the matched V3/V4 throughput-search wrappers."""

from __future__ import annotations

import pytest

from tests.performance.aic_roofline_pareto import run_dsv4pro_vs_step4pro_v3_throughput as v3
from tests.performance.aic_roofline_pareto import run_dsv4pro_vs_step4pro_v4_throughput as v4

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("runner,model", [(v3, "stepfun-ai/Step4-Pro-V3"), (v4, "stepfun-ai/Step4-Pro-V4")])
def test_runner_preserves_reference_search_contract(runner, model):
    points = runner.build_matrix_points()
    assert len(points) == 64
    assert {point.model for point in points} == {model, "deepseek-ai/DeepSeek-V4-Pro"}
    assert {point.system for point in points} == {"gb300", "h200_sxm", "h100_sxm", "h800_sxm"}
    assert {point.isl for point in points} == set(runner.ISL_VALUES)
    assert {point.osl for point in points} == {1024}
    assert {point.ttft_sla_ms for point in points} == {5000}
    assert {point.database_mode for point in points} == {"SOL"}
    assert all(point.total_gpus == 64 and point.prefix == 0 and point.nextn == 0 for point in points)
    assert all(point.chunked_prefill is False and point.pareto_sweep is False for point in points)


@pytest.mark.parametrize("runner,model", [(v3, "stepfun-ai/Step4-Pro-V3"), (v4, "stepfun-ai/Step4-Pro-V4")])
def test_step4_pro_search_excludes_structurally_invalid_tp8(runner, model):
    """The model-specific search contract excludes TP=8 because 12 output groups are indivisible."""
    run_spec = next(spec for spec in runner.build_mode_run_specs() if spec.point.model == model)

    task = runner.build_throughput_task(run_spec, experiment="disagg_AA")

    assert task.prefill_tp_candidates == [1, 2, 4]
    assert task.decode_tp_candidates == [1, 2, 4]
