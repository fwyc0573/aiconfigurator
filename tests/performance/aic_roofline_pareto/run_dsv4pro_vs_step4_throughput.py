"""Run the fresh disaggregated Step4/DeepSeek throughput-ratio matrix."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import perf_database
from aiconfigurator.sdk.utils import calculate_prefill_tokens_per_second
from tests.performance.aic_roofline_pareto import run_step4_comparison as base

MODELS = base.MODELS
SYSTEMS = base.SYSTEMS
MatrixPoint = base.MatrixPoint
ModeRunSpec = base.ModeRunSpec
ISL_VALUES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
OSL = 1024
TTFT_SLA_MS = 5000
WORKLOAD_KIND = "throughput"
THROUGHPUT_RUNNER_RELATIVE_PATH = "tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py"
RUNNER_SOURCE_RELATIVE_PATHS = (
    base.BASE_RUNNER_RELATIVE_PATH,
    THROUGHPUT_RUNNER_RELATIVE_PATH,
)
RANKING_CONTRACT = {
    "throughput": "output_token_throughput per fixed cluster descending",
    "ttft_constraint": "disagg corrected TTFT strictly less than 5000 ms",
    "tie_breaker": "typed canonical configuration identity ascending",
}


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
            attention_approximation=("temporary_mla_substitute" if model == MODELS[0] else None),
            approximation_dominated=model == MODELS[0] and isl >= 65_536,
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


@contextmanager
def _base_contract() -> Iterator[None]:
    """Apply this workflow's ranking and contract semantics only while executing."""
    previous = (
        base.derive_ranking_metric_evidence,
        base.RUNNER_SOURCE_RELATIVE_PATHS,
        base.RANKING_CONTRACT,
    )
    base.derive_ranking_metric_evidence = derive_ranking_metric_evidence
    base.RUNNER_SOURCE_RELATIVE_PATHS = RUNNER_SOURCE_RELATIVE_PATHS
    base.RANKING_CONTRACT = RANKING_CONTRACT
    try:
        yield
    finally:
        (
            base.derive_ranking_metric_evidence,
            base.RUNNER_SOURCE_RELATIVE_PATHS,
            base.RANKING_CONTRACT,
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
