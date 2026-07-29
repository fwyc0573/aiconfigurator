"""Run the fresh disaggregated Step4-Pro-V3/DeepSeek throughput-ratio matrix."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import perf_database
from aiconfigurator.sdk.utils import calculate_prefill_tokens_per_second
from tests.performance.aic_roofline_pareto import run_step4_comparison as base

MODELS = ("stepfun-ai/Step4-Pro-V3", "deepseek-ai/DeepSeek-V4-Pro")
MODEL_STEP4PRO, MODEL_DEEPSEEK = MODELS
SYSTEMS = base.SYSTEMS
MatrixPoint = base.MatrixPoint
ModeRunSpec = base.ModeRunSpec
ISL_VALUES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
OSL = 1024
TTFT_SLA_MS = 5000
WORKLOAD_KIND = "throughput"
THROUGHPUT_RUNNER_RELATIVE_PATH = "tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4pro_v3_throughput.py"
RUNNER_SOURCE_RELATIVE_PATHS = (
    base.BASE_RUNNER_RELATIVE_PATH,
    THROUGHPUT_RUNNER_RELATIVE_PATH,
)
RANKING_CONTRACT = {
    "throughput": "output_token_throughput per fixed cluster descending",
    "ttft_constraint": "disagg corrected TTFT strictly less than 5000 ms",
    "tie_breaker": "typed canonical configuration identity ascending",
}
DELTA_CONTRACT = {
    "absolute": "Step4-Pro-V3 - DeepSeek-V4-Pro",
    "relative": "(Step4-Pro-V3 - DeepSeek-V4-Pro) / DeepSeek-V4-Pro",
    "candidate": MODEL_STEP4PRO,
    "baseline": MODEL_DEEPSEEK,
    "zero_baseline": "error",
    "zero_baseline_both_zero": "tpot only: absolute_delta=0.0, relative_delta=null, status=zero_baseline_both_zero",
}

# Step4-Pro full attention has 12 inferred output groups.  TP=8 cannot shard
# that projection evenly (12 % 8 != 0), so it is excluded explicitly rather
# than being allowed to fail inside Task.run() or silently replaced by TP=4.
STEP4PRO_TP_CANDIDATES = (1, 2, 4)


def build_throughput_task(
    run_spec: base.ModeRunSpec,
    *,
    experiment: str,
    caps: base.BatchCaps | None = None,
) -> base.Task:
    """Build a throughput Task with only structurally valid Step4 TP values."""
    task = base.build_comparison_task(run_spec, experiment=experiment, caps=caps)
    if run_spec.point.model != MODEL_STEP4PRO:
        return task
    roles = ("agg",) if run_spec.serving_mode == "agg" else ("prefill", "decode")
    for role in roles:
        field = f"{role}_tp_candidates"
        candidates = getattr(task, field)
        filtered = [tp for tp in candidates if tp in STEP4PRO_TP_CANDIDATES]
        if not filtered:
            raise ValueError(f"{field} has no valid Step4-Pro TP candidates: {candidates!r}")
        setattr(task, field, filtered)
    return task


def _execute_mode_run(
    run_spec: base.ModeRunSpec,
    *,
    system_spec: Mapping[str, Any],
    initial_caps: base.BatchCaps | None = None,
) -> dict[str, Any]:
    """Execute one mode run using the model-aware candidate contract."""
    return base.execute_mode_run(
        run_spec,
        system_spec=system_spec,
        initial_caps=initial_caps,
        task_factory=build_throughput_task,
    )


def build_matrix_points() -> tuple[base.MatrixPoint, ...]:
    """Build the 64 model/system/ISL points required by this task."""
    points = tuple(
        base.MatrixPoint(
            model=model,
            system=system,
            workload_kind=WORKLOAD_KIND,
            isl=isl,
            osl=OSL,
            ttft_sla_ms=TTFT_SLA_MS,
            backend="vllm",
            backend_version=base.BACKEND_VERSION,
            engine_step_backend=base.ENGINE_STEP_BACKEND,
            database_mode="SOL",
            total_gpus=64,
            prefix=0,
            nextn=0,
            tpot_ms=50_000,
            pareto_sweep=False,
            chunked_prefill=False,
            attention_approximation=None,
            approximation_dominated=False,
        )
        for model in MODELS
        for system in SYSTEMS
        for isl in ISL_VALUES
    )
    if len(points) != 64:
        raise RuntimeError(f"Unexpected throughput matrix-point count: {len(points)}")
    return points


def build_mode_run_specs() -> tuple[base.ModeRunSpec, ...]:
    """Expand every point into exactly one disaggregated mode run."""
    specs = tuple(base.ModeRunSpec(point=point, serving_mode="disagg") for point in build_matrix_points())
    if len(specs) != 64 or {spec.serving_mode for spec in specs} != {"disagg"}:
        raise RuntimeError("Throughput matrix must contain exactly 64 disagg mode runs")
    return specs


def derive_ranking_metric_evidence(
    run_spec: base.ModeRunSpec,
    raw_row: Mapping[str, Any],
) -> dict[str, float | str]:
    """Derive fixed-cluster output and prefill throughput from one disagg row."""
    if run_spec.serving_mode != "disagg":
        raise ValueError(f"Throughput ratio workflow only supports disagg; got {run_spec.serving_mode!r}")
    deployment_gpus = base._positive_integer(raw_row.get("num_total_gpus"), field="num_total_gpus")
    output_allocation = base.derive_cluster_allocation(
        tokens_per_second_per_gpu=raw_row.get("tokens/s/gpu"),
        num_total_gpus=deployment_gpus,
        total_gpus=run_spec.point.total_gpus,
    )
    prefill_tokens_per_second = calculate_prefill_tokens_per_second(
        global_batch_size=raw_row.get("(p)global_bs"),
        num_workers=raw_row.get("(p)workers"),
        isl=run_spec.point.isl,
        prefix=run_spec.point.prefix,
        ttft_ms=raw_row.get("ttft"),
    )
    prefill_per_gpu = prefill_tokens_per_second / deployment_gpus
    prefill_allocation = base.derive_cluster_allocation(
        tokens_per_second_per_gpu=prefill_per_gpu,
        num_total_gpus=deployment_gpus,
        total_gpus=run_spec.point.total_gpus,
    )
    prefill_cluster = prefill_allocation.tokens_per_second_per_gpu_cluster
    output_cluster = output_allocation.tokens_per_second_per_gpu_cluster
    return {
        "ranking_metric_kind": "output_token_throughput",
        "ranking_metric_value": output_cluster,
        "output_token_throughput": output_cluster,
        "prefill_input_throughput": prefill_cluster,
        "prefill_input_throughput/gpu": prefill_per_gpu,
        "prefill_input_throughput/gpu_cluster": prefill_cluster,
    }


def build_model_comparisons(
    ranked_rows: Iterable[Mapping[str, Any]],
    *,
    metrics: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Compare aligned Step4-Pro-V3 rank-one rows against DeepSeek-V4-Pro."""
    if not metrics:
        raise ValueError("At least one comparison metric is required")
    unknown_metrics = [metric for metric in metrics if metric not in base.METRIC_POLARITY]
    if unknown_metrics:
        raise ValueError(f"Missing metric-polarity contract for: {unknown_metrics}")

    rank_one: dict[tuple[tuple[Any, ...], str], Mapping[str, Any]] = {}
    aligned_keys: set[tuple[Any, ...]] = set()
    for row in ranked_rows:
        base._require_row_fields(
            row,
            (*base.RANK_GROUP_FIELDS, "canonical_config_id", "rank", *metrics),
        )
        if row["rank"] != 1:
            continue
        model = row["model"]
        if model not in MODELS:
            raise ValueError(f"Unexpected comparison model: {model!r}")
        aligned_key = tuple(row[field] for field in base.COMPARISON_KEY_FIELDS)
        evidence_key = (aligned_key, model)
        if evidence_key in rank_one:
            raise ValueError(f"Duplicate rank-one evidence for model={model!r}, key={aligned_key!r}")
        rank_one[evidence_key] = row
        aligned_keys.add(aligned_key)

    comparisons = []
    for aligned_key in sorted(aligned_keys, key=repr):
        candidate = rank_one.get((aligned_key, MODEL_STEP4PRO))
        baseline = rank_one.get((aligned_key, MODEL_DEEPSEEK))
        comparison = {
            "aligned_key": aligned_key,
            "candidate_model": MODEL_STEP4PRO,
            "baseline_model": MODEL_DEEPSEEK,
            "candidate_config_id": None if candidate is None else str(candidate["canonical_config_id"]),
            "baseline_config_id": None if baseline is None else str(baseline["canonical_config_id"]),
            "metric_deltas": {},
        }
        if candidate is None or baseline is None:
            comparisons.append({**comparison, "status": "unpaired"})
            continue

        metric_deltas = {}
        for metric in metrics:
            candidate_value = base._finite_number(candidate[metric], field=f"Step4-Pro-V3 {metric}")
            baseline_value = base._finite_number(baseline[metric], field=f"DeepSeek-V4-Pro {metric}")
            if baseline_value == 0:
                if metric != "tpot" or candidate_value != 0:
                    raise ValueError(f"Cannot compute {metric}: zero DeepSeek-V4-Pro baseline")
                relative_delta = None
                status = "zero_baseline_both_zero"
            else:
                relative_delta = (candidate_value - baseline_value) / baseline_value
                status = "computed"
            metric_deltas[metric] = {
                "candidate_value": candidate_value,
                "baseline_value": baseline_value,
                "absolute_delta": candidate_value - baseline_value,
                "relative_delta": relative_delta,
                "polarity": base.METRIC_POLARITY[metric],
                "status": status,
            }
        comparisons.append({**comparison, "status": "paired", "metric_deltas": metric_deltas})
    return tuple(comparisons)


def _render_markdown_report(artifact: Mapping[str, Any]) -> str:
    """Render the reference report schema with explicit Step4-Pro-V3 semantics."""
    summary = artifact["summary"]
    lines = [
        "# Step4-Pro-V3 vs DeepSeek-V4-Pro SOL Comparison",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mode runs | {summary['mode_run_count']} |",
        f"| Normalized rows | {summary['normalized_row_count']} |",
        f"| Ranked rows | {summary['ranked_row_count']} |",
        f"| Paired comparisons | {summary['paired_comparison_count']} |",
        f"| Unpaired comparisons | {summary['unpaired_comparison_count']} |",
        "",
        "## Modeling Boundary",
        "",
        "Step4-Pro-V3 uses 20 explicit MQA full-attention layers and 60 explicit MQA SWA(512) layers. "
        "No legacy Step4 MLA substitute or SWA-to-HCA fallback is used in DatabaseMode.SOL.",
        "",
        "Full-attention runtime KV representation, FP8 tensor details, and "
        "small-token/high-EP workload distribution remain documented model-evidence limitations.",
        "",
        "All operation latency evidence in this comparison uses DatabaseMode.SOL. OSL=1024 disaggregated "
        "rows rank by fixed-cluster output throughput under strict corrected TTFT < 5000 ms.",
        "",
        "## Rank-One Results",
        "",
        "| Model | System | Workload | ISL | OSL | TTFT SLA (ms) | Mode | Metric | Value | TTFT (ms) | Config |",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|---|",
    ]
    rank_one_rows = [row for row in artifact["ranked_rows"] if row["rank"] == 1]
    for row in rank_one_rows:
        lines.append(
            f"| {row['model']} | {row['system']} | {row['workload_kind']} | {row['isl']} | "
            f"{row['osl']} | {row['ttft_sla_ms']} | {row['serving_mode']} | "
            f"{row['ranking_metric_kind']} | {row['ranking_metric_value']} | {row['ttft']} | "
            f"{row['canonical_config_id']} |"
        )
    if not rank_one_rows:
        lines.append("| None | None | None | 0 | 0 | 0 | None | None | 0 | 0 | None |")
    lines.extend(
        [
            "",
            "## Paired Model Deltas",
            "",
            "Absolute delta is Step4-Pro-V3 minus DeepSeek-V4-Pro. "
            "TPOT is observational and does not affect eligibility.",
            "",
            "| Metric | Step4-Pro-V3 | DeepSeek-V4-Pro | Absolute delta | Relative delta | Polarity | Status |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    paired_metric_count = 0
    for comparison in artifact["comparisons"]:
        if comparison["status"] != "paired":
            continue
        metric_deltas = comparison["metric_deltas"]
        for metric in artifact["comparison_metrics"]:
            if metric not in metric_deltas:
                continue
            delta = metric_deltas[metric]
            relative_delta = (
                "N/A (zero baseline, both zero)"
                if delta["status"] == "zero_baseline_both_zero"
                else delta["relative_delta"]
            )
            lines.append(
                f"| {metric} | {delta['candidate_value']} | {delta['baseline_value']} | "
                f"{delta['absolute_delta']} | {relative_delta} | {delta['polarity']} | {delta['status']} |"
            )
            paired_metric_count += 1
    if paired_metric_count == 0:
        lines.append("| None | 0 | 0 | 0 | 0 | None | None |")
    lines.append("")
    return "\n".join(lines)


@contextmanager
def _base_contract() -> Iterator[None]:
    """Apply this workflow's ranking and contract semantics only while executing."""
    previous = (
        base.derive_ranking_metric_evidence,
        base.RUNNER_SOURCE_RELATIVE_PATHS,
        base.RANKING_CONTRACT,
        base.build_model_comparisons,
        base.DELTA_CONTRACT,
        base._render_markdown_report,
    )
    base.derive_ranking_metric_evidence = derive_ranking_metric_evidence
    base.RUNNER_SOURCE_RELATIVE_PATHS = RUNNER_SOURCE_RELATIVE_PATHS
    base.RANKING_CONTRACT = RANKING_CONTRACT
    base.build_model_comparisons = build_model_comparisons
    base.DELTA_CONTRACT = DELTA_CONTRACT
    base._render_markdown_report = _render_markdown_report
    try:
        yield
    finally:
        (
            base.derive_ranking_metric_evidence,
            base.RUNNER_SOURCE_RELATIVE_PATHS,
            base.RANKING_CONTRACT,
            base.build_model_comparisons,
            base.DELTA_CONTRACT,
            base._render_markdown_report,
        ) = previous


def _select_specs(
    specs: tuple[base.ModeRunSpec, ...],
    *,
    models: tuple[str, ...] | None,
    systems: tuple[str, ...] | None,
    isls: tuple[int, ...] | None,
) -> tuple[base.ModeRunSpec, ...]:
    selected = tuple(
        spec
        for spec in specs
        if (models is None or spec.point.model in models)
        and (systems is None or spec.point.system in systems)
        and (isls is None or spec.point.isl in isls)
    )
    if not selected:
        raise ValueError("No throughput mode runs match the requested filters")
    return selected


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer; got {value!r}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--systems-paths",
        default="default,tests/performance/aic_roofline_pareto/systems",
        help="Comma-separated AIC systems search paths.",
    )
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--system", action="append", choices=SYSTEMS)
    parser.add_argument("--isl", action="append", type=_positive_int)
    parser.add_argument("--initial-prefill-cap", type=_positive_int, default=16)
    parser.add_argument("--initial-decode-cap", type=_positive_int, default=1024)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    perf_database.set_systems_paths(args.systems_paths)
    specs = _select_specs(
        build_mode_run_specs(),
        models=None if args.model is None else tuple(args.model),
        systems=None if args.system is None else tuple(args.system),
        isls=None if args.isl is None else tuple(args.isl),
    )
    initial_caps = base.BatchCaps(prefill=args.initial_prefill_cap, decode=args.initial_decode_cap)
    with _base_contract():
        head = base._git_head()
        contract = base.build_execution_contract(specs, initial_caps=initial_caps)
        contract_sha256 = base.execution_contract_sha256(contract)
        checkpoint_path = args.output_dir / "mode_runs.sqlite3"
        records = base.execute_matrix_runs(
            specs,
            checkpoint_path=checkpoint_path,
            execution_contract_sha256=contract_sha256,
            git_head=head,
            resume=args.resume,
            initial_caps=initial_caps,
            executor=_execute_mode_run,
        )
        header, loaded_records = base.load_checkpoint(
            checkpoint_path,
            expected_header=base.build_checkpoint_header(
                specs,
                execution_contract_sha256=contract_sha256,
                git_head=head,
            ),
            run_specs=specs,
        )
        if records != loaded_records:
            raise ValueError("in-memory and durable checkpoint records differ")
        artifact = base.finalize_matrix_results(specs, header=header, records=loaded_records)
        paths = base.write_final_artifacts(args.output_dir, artifact)
    print(f"completed {len(specs)} throughput mode runs; results={paths['json']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
