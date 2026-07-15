"""Build the fixed Step4 versus DeepSeek-V4-Pro SOL comparison matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sqlite3
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from itertools import product
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import common, perf_database
from aiconfigurator.sdk.errors import InsufficientMemoryError, KVCacheCapacityError, NoFeasibleConfigError
from aiconfigurator.sdk.task_v2 import SinglePointEvaluation, Task
from aiconfigurator.sdk.utils import (
    calculate_prefill_tokens_per_second,
    cluster_normalized_throughput,
    enumerate_parallel_config,
)

MODELS = ("stepfun-ai/Step4", "deepseek-ai/DeepSeek-V4-Pro")
SYSTEMS = ("gb300", "h200_sxm", "h100_sxm", "h800_sxm")
PRIMARY_ISL = (4096, 16384, 65536, 262144, 1048576)
TTFT_SLA_MS = (200, 500, 1000, 2000, 5000)
SERVING_MODES = ("agg", "disagg")
NEUTRAL_CORRECTIONS = {
    "prefill_latency_correction": 1.0,
    "decode_latency_correction": 1.0,
    "rate_match_prefill_degradation": 1.0,
    "rate_match_decode_degradation": 1.0,
    "autoscale_ttft_correction_factor": 1.0,
}
STEP4_ATTENTION_APPROXIMATION_GROUPS = {
    "full_mla_approx_layers": 23,
    "swa_mla_approx_layers": 69,
}
RANKING_CONTRACT = {
    "primary": "prefill_input_throughput per fixed cluster descending",
    "decode_smoke": "output_token_throughput per fixed cluster descending",
    "tie_breaker": "typed canonical configuration identity ascending",
}
DELTA_CONTRACT = {
    "absolute": "Step4 - DeepSeek-V4-Pro",
    "relative": "(Step4 - DeepSeek-V4-Pro) / DeepSeek-V4-Pro",
    "baseline": "DeepSeek-V4-Pro",
    "zero_baseline": "error",
    "zero_baseline_both_zero": ("tpot only: absolute_delta=0.0, relative_delta=null, status=zero_baseline_both_zero"),
}
METRIC_POLARITY = {
    "tokens/s": "higher_is_better",
    "tokens/s/gpu": "higher_is_better",
    "tokens/s/gpu_cluster": "higher_is_better",
    "ranking_metric_value": "higher_is_better",
    "prefill_tokens/s": "higher_is_better",
    "prefill_tokens/s/gpu": "higher_is_better",
    "prefill_tokens/s/gpu_cluster": "higher_is_better",
    "tokens/s/user": "higher_is_better",
    "seq/s/gpu": "higher_is_better",
    "ttft": "lower_is_better",
    "tpot": "lower_is_better",
    "request_latency": "lower_is_better",
}
RANK_GROUP_FIELDS = (
    "model",
    "system",
    "workload_kind",
    "isl",
    "osl",
    "prefix",
    "ttft_sla_ms",
    "serving_mode",
    "backend",
    "database_mode",
    "nextn",
)
COMPARISON_KEY_FIELDS = tuple(field for field in RANK_GROUP_FIELDS if field != "model")
TERMINAL_STATUSES = {"success", "memory_infeasible", "sla_infeasible"}
BACKEND_VERSION = "0.22.0"
ENGINE_STEP_BACKEND = "python"
NUM_GPU_PER_REPLICA = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)
COMMON_RAW_FIELDS = (
    "model",
    "isl",
    "osl",
    "prefix",
    "ttft",
    "tpot",
    "request_latency",
    "seq/s",
    "seq/s/gpu",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
    "num_total_gpus",
)
AGG_RAW_FIELDS = (
    *COMMON_RAW_FIELDS,
    "bs",
    "global_bs",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    "ctx_tokens",
    "parallel",
    "gemm",
    "kvcache",
    "fmha",
    "moe",
    "comm",
    "memory",
    "backend",
    "version",
    "system",
)
AGG_POINT_IDENTITY_FIELDS = ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp", "bs", "ctx_tokens")
DISAGG_RAW_FIELDS = (
    *COMMON_RAW_FIELDS,
    "(p)bs",
    "(p)global_bs",
    "(p)workers",
    "(d)bs",
    "(d)global_bs",
    "(d)workers",
    "(p)tp",
    "(p)pp",
    "(p)dp",
    "(p)moe_tp",
    "(p)moe_ep",
    "(p)cp",
    "(p)parallel",
    "(p)gemm",
    "(p)kvcache",
    "(p)fmha",
    "(p)moe",
    "(p)comm",
    "(p)memory",
    "(p)backend",
    "(p)version",
    "(p)system",
    "(d)tp",
    "(d)pp",
    "(d)dp",
    "(d)moe_tp",
    "(d)moe_ep",
    "(d)parallel",
    "(d)gemm",
    "(d)kvcache",
    "(d)fmha",
    "(d)moe",
    "(d)comm",
    "(d)memory",
    "(d)backend",
    "(d)version",
    "(d)system",
)
PATTERN_CANDIDATES = {
    "A": {
        "num_gpu_candidates": (2, 4, 8, 16, 32),
        "tp_candidates": (1, 2, 4, 8),
        "pp_candidates": (1,),
        "dp_candidates": (1, 2, 4, 8, 16, 32),
        "moe_tp_candidates": (1,),
        "moe_ep_candidates": (2, 4, 8, 16, 32),
        "cp_candidates": (1,),
    },
    "B": {
        "num_gpu_candidates": (1, 2, 4, 8),
        "tp_candidates": (1, 2, 4, 8),
        "pp_candidates": (1,),
        "dp_candidates": (1,),
        "moe_tp_candidates": (1, 2, 4, 8),
        "moe_ep_candidates": (1,),
        "cp_candidates": (1,),
    },
}
EXPERIMENT_PATTERNS = {
    "agg_patternA": ("agg", "A", None),
    "agg_patternB": ("agg", "B", None),
    "disagg_AA": ("disagg", "A", "A"),
    "disagg_AB": ("disagg", "A", "B"),
    "disagg_BA": ("disagg", "B", "A"),
    "disagg_BB": ("disagg", "B", "B"),
}
CHECKPOINT_SCHEMA_VERSION = 3
EXECUTION_CONTRACT_SCHEMA_VERSION = 1
BASE_RUNNER_RELATIVE_PATH = "tests/performance/aic_roofline_pareto/run_step4_comparison.py"
RUNNER_SOURCE_RELATIVE_PATHS = (BASE_RUNNER_RELATIVE_PATH,)
CHECKPOINT_HEADER_FIELDS = {
    "record_type",
    "schema_version",
    "execution_contract_sha256",
    "git_head",
    "matrix_spec_hash",
    "mode_run_count",
}


@dataclass(frozen=True, slots=True)
class ParallelRow:
    """One exact vLLM worker parallel configuration."""

    pattern: str
    tp: int
    pp: int
    dp: int
    moe_tp: int
    moe_ep: int
    cp: int

    @property
    def worker_gpus(self) -> int:
        return self.tp * self.pp * self.dp * self.cp


@dataclass(frozen=True, slots=True)
class DisaggParallelPair:
    """One exact prefill/decode parallel-row pair."""

    prefill: ParallelRow
    decode: ParallelRow


@dataclass(frozen=True, slots=True)
class MatrixPoint:
    """One model/system/workload/SLA point before serving-mode expansion."""

    model: str
    system: str
    workload_kind: str
    isl: int
    osl: int
    ttft_sla_ms: int
    backend: str
    backend_version: str
    engine_step_backend: str
    database_mode: str
    total_gpus: int
    prefix: int
    nextn: int
    tpot_ms: int
    pareto_sweep: bool
    chunked_prefill: bool
    attention_approximation: str | None
    approximation_dominated: bool


@dataclass(frozen=True, slots=True)
class ModeRunSpec:
    """One matrix point expanded to aggregate or disaggregate serving."""

    point: MatrixPoint
    serving_mode: str


@dataclass(frozen=True, slots=True)
class BatchCaps:
    """Active exhaustive batch-search caps."""

    agg: int = 1024
    prefill: int = 16
    decode: int = 1024

    def __post_init__(self) -> None:
        for name in ("agg", "prefill", "decode"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} cap must be a positive integer; got {value!r}")


@dataclass(frozen=True, slots=True)
class SearchAttempt:
    """Successful evaluator evidence for the rank-one batch sizes."""

    rank1_batch_sizes: Mapping[str, int]
    candidate_rows: tuple[dict[str, Any], ...] = ()
    rank1_row: dict[str, Any] | None = None
    selected_point_identity: tuple[str | int, ...] | None = None
    selected_evaluation: SinglePointEvaluation | None = None
    per_ops_evidence: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CapAttemptEvidence:
    """One cap attempt with either selected-point or typed terminal evidence."""

    experiment: str | None
    caps: BatchCaps
    status: str
    search_attempt: SearchAttempt | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CapSearchResult:
    """Terminal result of repeated selective cap expansion."""

    terminal_status: str
    final_caps: BatchCaps
    cap_history: tuple[BatchCaps, ...]
    cap_rerun_count: int
    cap_saturated: bool
    ranking_eligible: bool
    attempt_evidence: tuple[CapAttemptEvidence, ...] = ()

    @property
    def attempt_history(self) -> tuple[SearchAttempt, ...]:
        return tuple(
            evidence.search_attempt for evidence in self.attempt_evidence if evidence.search_attempt is not None
        )


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """One explicit Step4-minus-DeepSeek metric comparison."""

    step4_value: float
    deepseek_value: float
    absolute_delta: float
    relative_delta: float | None
    polarity: str
    status: str


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """One aligned paired or unpaired model-level rank-one comparison."""

    aligned_key: tuple[Any, ...]
    status: str
    step4_config_id: str | None
    deepseek_config_id: str | None
    metric_deltas: dict[str, MetricDelta]


@dataclass(frozen=True, slots=True)
class ClusterAllocation:
    """Complete-replica allocation and cluster-normalized throughput."""

    replicas: int
    total_gpus_used: int
    unused_gpus: int
    tokens_per_second_per_gpu_cluster: float


def _materialize_parallel_rows(
    *,
    pattern: str,
    num_gpu_list: list[int],
    tp_list: list[int],
    dp_list: list[int],
    moe_tp_list: list[int],
    moe_ep_list: list[int],
) -> tuple[ParallelRow, ...]:
    rows = enumerate_parallel_config(
        num_gpu_list=num_gpu_list,
        tp_list=tp_list,
        pp_list=[1],
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        cp_list=[1],
        is_moe=True,
        backend=common.BackendName.vllm,
    )
    return tuple(
        ParallelRow(
            pattern=pattern,
            tp=tp,
            pp=pp,
            dp=dp,
            moe_tp=moe_tp,
            moe_ep=moe_ep,
            cp=cp,
        )
        for tp, pp, dp, moe_tp, moe_ep, cp in rows
    )


def build_common_vllm_parallel_rows() -> tuple[ParallelRow, ...]:
    """Materialize the approved common 17-row EP plus 4-row MoE-TP space."""
    pattern_a = _materialize_parallel_rows(
        pattern="A",
        num_gpu_list=[2, 4, 8, 16, 32],
        tp_list=[1, 2, 4, 8],
        dp_list=[1, 2, 4, 8, 16, 32],
        moe_tp_list=[1],
        moe_ep_list=[2, 4, 8, 16, 32],
    )
    pattern_b = _materialize_parallel_rows(
        pattern="B",
        num_gpu_list=[1, 2, 4, 8],
        tp_list=[1, 2, 4, 8],
        dp_list=[1],
        moe_tp_list=[1, 2, 4, 8],
        moe_ep_list=[1],
    )
    if len(pattern_a) != 17 or len(pattern_b) != 4:
        raise RuntimeError(
            f"Unexpected vLLM parallel-space materialization: Pattern A={len(pattern_a)}, Pattern B={len(pattern_b)}"
        )
    return (*pattern_b, *pattern_a)


def pair_disagg_parallel_rows() -> dict[str, tuple[DisaggParallelPair, ...]]:
    """Build the exact AA/AB/BA/BB disaggregate Cartesian products."""
    rows = build_common_vllm_parallel_rows()
    by_pattern = {pattern: tuple(row for row in rows if row.pattern == pattern) for pattern in ("A", "B")}
    return {
        f"{prefill_pattern}{decode_pattern}": tuple(
            DisaggParallelPair(prefill=prefill, decode=decode)
            for prefill, decode in product(
                by_pattern[prefill_pattern],
                by_pattern[decode_pattern],
            )
        )
        for prefill_pattern, decode_pattern in product(("A", "B"), repeat=2)
    }


def build_matrix_points() -> tuple[MatrixPoint, ...]:
    """Build the 240 model/system/workload/TTFT comparison points."""
    workloads = (
        *(("primary", isl, 1) for isl in PRIMARY_ISL),
        ("decode_smoke", 4096, 1024),
    )
    points = []
    for model, system, workload, ttft_sla_ms in product(
        MODELS,
        SYSTEMS,
        workloads,
        TTFT_SLA_MS,
    ):
        workload_kind, isl, osl = workload
        uses_step4_approximation = model == "stepfun-ai/Step4"
        points.append(
            MatrixPoint(
                model=model,
                system=system,
                workload_kind=workload_kind,
                isl=isl,
                osl=osl,
                ttft_sla_ms=ttft_sla_ms,
                backend="vllm",
                backend_version=BACKEND_VERSION,
                engine_step_backend=ENGINE_STEP_BACKEND,
                database_mode="SOL",
                total_gpus=64,
                prefix=0,
                nextn=0,
                tpot_ms=50_000,
                pareto_sweep=False,
                chunked_prefill=False,
                attention_approximation=("temporary_mla_substitute" if uses_step4_approximation else None),
                approximation_dominated=uses_step4_approximation and isl >= 65_536,
            )
        )
    if len(points) != 240:
        raise RuntimeError(f"Unexpected matrix-point count: {len(points)}")
    return tuple(points)


def build_mode_run_specs() -> tuple[ModeRunSpec, ...]:
    """Expand every matrix point across aggregate and disaggregate serving."""
    runs = tuple(
        ModeRunSpec(point=point, serving_mode=serving_mode)
        for point, serving_mode in product(build_matrix_points(), SERVING_MODES)
    )
    if len(runs) != 480:
        raise RuntimeError(f"Unexpected mode-run count: {len(runs)}")
    return runs


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Expected finite JSON number; got {value!r}")
        return number
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings; got {key!r}")
            normalized[key] = _jsonable(item)
        return normalized
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    raise TypeError(f"Unsupported checkpoint JSON value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _strict_json_loads(payload: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Expected finite JSON number; got {value}")

    return json.loads(
        payload,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def candidate_fingerprint(candidate_rows: Iterable[Mapping[str, Any]]) -> dict[str, str | int]:
    """Fingerprint an ordered candidate sequence without persisting every row."""
    rows = tuple(candidate_rows)
    return {
        "candidate_count": len(rows),
        "candidate_sha256": _sha256_json(rows),
    }


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Execution-contract input does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_jsonable(value: Any) -> Any:
    """Normalize resolved specs while preserving non-string mapping-key types."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Expected finite execution-contract number; got {value!r}")
        return number
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            if isinstance(key, str):
                typed_key = ["str", key]
            elif type(key) is int:
                typed_key = ["int", key]
            else:
                raise TypeError(f"Resolved system-spec keys must be strings or integers; got {type(key).__name__}")
            entries.append([typed_key, _contract_jsonable(item)])
        entries.sort(key=lambda entry: _canonical_json(entry[0]))
        return {"mapping_entries": entries}
    if isinstance(value, list | tuple):
        return [_contract_jsonable(item) for item in value]
    raise TypeError(f"Unsupported execution-contract value: {type(value).__name__}")


def build_execution_contract(
    run_specs: Iterable[ModeRunSpec],
    *,
    initial_caps: BatchCaps,
    repo_root: str | Path | None = None,
    system_loader: Callable[[str], Mapping[str, Any]] = perf_database.load_system_spec,
) -> dict[str, Any]:
    """Build an auditable content manifest for every execution-relevant input."""
    specs = tuple(run_specs)
    if not specs:
        raise ValueError("Execution contract requires at least one mode run")
    if not isinstance(initial_caps, BatchCaps):
        raise TypeError(f"initial_caps must be BatchCaps; got {type(initial_caps).__name__}")
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    sdk_root = root / "src/aiconfigurator/sdk"
    sdk_sources = tuple(sorted(sdk_root.rglob("*.py")))
    if not sdk_sources:
        raise FileNotFoundError(f"Execution-contract SDK source set is empty: {sdk_root}")

    runner_sources = tuple(root / relative_path for relative_path in RUNNER_SOURCE_RELATIVE_PATHS)
    source_hashes = {path.relative_to(root).as_posix(): _sha256_file(path) for path in (*sdk_sources, *runner_sources)}
    models = tuple(sorted({run_spec.point.model for run_spec in specs}))
    model_configs = {}
    for model in models:
        relative_path = Path("src/aiconfigurator/model_configs") / f"{model.replace('/', '--')}_config.json"
        model_configs[model] = {
            "path": relative_path.as_posix(),
            "sha256": _sha256_file(root / relative_path),
        }

    systems = tuple(sorted({run_spec.point.system for run_spec in specs}))
    engine_step_backends = {run_spec.point.engine_step_backend for run_spec in specs}
    if engine_step_backends != {ENGINE_STEP_BACKEND}:
        raise ValueError(
            f"Execution contract requires engine_step_backend='python'; got {sorted(engine_step_backends)!r}"
        )
    resolved_system_specs = {}
    for system in systems:
        resolved = system_loader(system)
        if not isinstance(resolved, Mapping) or not resolved:
            raise ValueError(f"Resolved system spec for {system!r} must be a non-empty mapping")
        normalized = _contract_jsonable(resolved)
        resolved_system_specs[system] = {
            "sha256": _sha256_json(normalized),
            "content": normalized,
        }

    return {
        "schema_version": EXECUTION_CONTRACT_SCHEMA_VERSION,
        "engine_step_backend": ENGINE_STEP_BACKEND,
        "initial_caps": _jsonable(initial_caps),
        "used_models": list(models),
        "used_systems": list(systems),
        "source_files": dict(sorted(source_hashes.items())),
        "model_configs": model_configs,
        "resolved_system_specs": resolved_system_specs,
    }


def execution_contract_sha256(contract: Mapping[str, Any]) -> str:
    if not isinstance(contract, Mapping):
        raise TypeError(f"contract must be a mapping; got {type(contract).__name__}")
    return _sha256_json(contract)


def mode_run_identity(run_spec: ModeRunSpec) -> dict[str, Any]:
    """Return the complete flat identity for one executable matrix mode run."""
    return {**asdict(run_spec.point), "serving_mode": run_spec.serving_mode}


def mode_run_key_from_identity(identity: Mapping[str, Any]) -> str:
    if not identity:
        raise ValueError("Mode-run identity must not be empty")
    return _sha256_json(identity)


def mode_run_key(run_spec: ModeRunSpec) -> str:
    return mode_run_key_from_identity(mode_run_identity(run_spec))


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a 64-character SHA256 hex digest") from error
    return value


def build_checkpoint_header(
    run_specs: Iterable[ModeRunSpec],
    *,
    execution_contract_sha256: str,
    git_head: str,
) -> dict[str, Any]:
    specs = tuple(run_specs)
    if not specs:
        raise ValueError("Checkpoint matrix must contain at least one mode run")
    _require_sha256(execution_contract_sha256, field="execution_contract_sha256")
    if not isinstance(git_head, str) or not git_head:
        raise ValueError("git_head must be a non-empty string")
    identities = tuple(mode_run_identity(spec) for spec in specs)
    if len({mode_run_key_from_identity(identity) for identity in identities}) != len(identities):
        raise ValueError("Checkpoint matrix contains duplicate mode-run identities")
    return {
        "record_type": "checkpoint_header",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "execution_contract_sha256": execution_contract_sha256,
        "git_head": git_head,
        "matrix_spec_hash": _sha256_json(identities),
        "mode_run_count": len(identities),
    }


def _validated_checkpoint_header(header: Any) -> dict[str, Any]:
    validated = _require_exact_mapping_fields(
        header,
        expected=CHECKPOINT_HEADER_FIELDS,
        context="checkpoint header",
    )
    if validated["record_type"] != "checkpoint_header":
        raise ValueError("checkpoint header must have record_type='checkpoint_header'")
    if validated["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint header schema_version must equal {CHECKPOINT_SCHEMA_VERSION}; "
            f"got {validated['schema_version']!r}"
        )
    _require_sha256(validated["execution_contract_sha256"], field="execution_contract_sha256")
    _require_sha256(validated["matrix_spec_hash"], field="matrix_spec_hash")
    if not isinstance(validated["git_head"], str) or not validated["git_head"]:
        raise ValueError("checkpoint header git_head must be a non-empty string")
    mode_run_count = validated["mode_run_count"]
    if type(mode_run_count) is not int or mode_run_count < 1:
        raise ValueError(f"checkpoint header mode_run_count must be a positive integer; got {mode_run_count!r}")
    return _jsonable(validated)


def _build_initialized_checkpoint_bytes(header: Mapping[str, Any]) -> bytes:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE checkpoint_header (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                payload TEXT NOT NULL CHECK (typeof(payload) = 'text')
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE mode_runs (
                mode_run_key TEXT PRIMARY KEY CHECK (length(mode_run_key) = 64),
                payload TEXT NOT NULL CHECK (typeof(payload) = 'text')
            )
            """
        )
        connection.execute(
            "INSERT INTO checkpoint_header(singleton, payload) VALUES (1, ?)",
            (_canonical_json(header),),
        )
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


def initialize_checkpoint(path: str | Path, header: Mapping[str, Any]) -> None:
    validated_header = _validated_checkpoint_header(header)
    database_bytes = _build_initialized_checkpoint_bytes(validated_header)
    checkpoint = Path(path)
    try:
        descriptor = os.open(checkpoint, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}") from error
    try:
        remaining = memoryview(database_bytes)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise OSError("Checkpoint initialization wrote zero bytes")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_checkpoint(path: str | Path) -> sqlite3.Connection:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    connection = sqlite3.connect(f"{checkpoint.as_uri()}?mode=rw", uri=True)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _insert_mode_run(connection: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    connection.execute(
        "INSERT INTO mode_runs(mode_run_key, payload) VALUES (?, ?)",
        (record["mode_run_key"], _canonical_json(record)),
    )


def commit_checkpoint_record(
    path: str | Path,
    run_spec: ModeRunSpec,
    record: Mapping[str, Any],
) -> None:
    validated = validate_serialized_mode_run(run_spec, record)
    try:
        with _open_checkpoint(path) as connection:
            _insert_mode_run(connection, validated)
    except sqlite3.IntegrityError as error:
        if "UNIQUE constraint failed: mode_runs.mode_run_key" in str(error):
            raise ValueError(f"Duplicate completed mode_run_key: {validated['mode_run_key']}") from error
        raise


def _validate_checkpoint_header(
    header: Any,
    *,
    expected_header: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _validated_checkpoint_header(header)
    expected = _require_exact_mapping_fields(
        expected_header,
        expected=CHECKPOINT_HEADER_FIELDS,
        context="expected checkpoint header",
    )
    for field in (
        "record_type",
        "schema_version",
        "execution_contract_sha256",
        "git_head",
        "matrix_spec_hash",
        "mode_run_count",
    ):
        if validated[field] != expected[field]:
            raise ValueError(f"{field} mismatch: checkpoint={validated[field]!r}, expected={expected[field]!r}")
    return validated


def load_checkpoint(
    path: str | Path,
    *,
    expected_header: Mapping[str, Any],
    run_specs: Iterable[ModeRunSpec],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    checkpoint = Path(path)
    specs = tuple(run_specs)
    expected_specs = {mode_run_key(run_spec): run_spec for run_spec in specs}
    if len(expected_specs) != len(specs):
        raise ValueError("Checkpoint run_specs contain duplicate mode-run identities")
    try:
        with _open_checkpoint(checkpoint) as connection:
            header_rows = connection.execute("SELECT payload FROM checkpoint_header").fetchall()
            stored_rows = connection.execute("SELECT mode_run_key, payload FROM mode_runs ORDER BY rowid").fetchall()
    except sqlite3.DatabaseError as error:
        raise ValueError(f"Invalid SQLite checkpoint: {checkpoint}") from error
    if len(header_rows) != 1:
        raise ValueError(f"Checkpoint must contain exactly one header; got {len(header_rows)}")
    try:
        parsed_header = _strict_json_loads(header_rows[0][0])
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("Malformed checkpoint header JSON") from error
    header = _validate_checkpoint_header(parsed_header, expected_header=expected_header)

    records = {}
    for key, payload in stored_rows:
        try:
            record = _strict_json_loads(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed mode-run JSON for key {key!r}") from error
        if key not in expected_specs:
            raise ValueError(f"Checkpoint contains a mode run outside the selected matrix: {key}")
        validated = validate_serialized_mode_run(expected_specs[key], record)
        if validated["mode_run_key"] != key:
            raise ValueError(f"SQLite mode_run_key column/payload mismatch: {key}")
        if key in records:
            raise ValueError(f"Duplicate completed mode_run_key: {key}")
        records[key] = validated
    return header, records


def _role_candidates(role: str, pattern: str) -> dict[str, list[int]]:
    return {f"{role}_{field}": list(values) for field, values in PATTERN_CANDIDATES[pattern].items()}


def build_comparison_task(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    caps: BatchCaps | None = None,
) -> Task:
    """Build one isolated aggregate pattern or disaggregate pattern pair."""
    if experiment not in EXPERIMENT_PATTERNS:
        raise ValueError(f"Unknown comparison experiment: {experiment!r}")
    expected_mode, prefill_pattern, decode_pattern = EXPERIMENT_PATTERNS[experiment]
    if run_spec.serving_mode != expected_mode:
        raise ValueError(f"Experiment {experiment!r} does not match serving mode {run_spec.serving_mode!r}")

    point = run_spec.point
    active_caps = caps or BatchCaps()
    common_kwargs = {
        "serving_mode": run_spec.serving_mode,
        "isl": point.isl,
        "osl": point.osl,
        "prefix": point.prefix,
        "ttft": point.ttft_sla_ms,
        "tpot": point.tpot_ms,
        "pareto_sweep": point.pareto_sweep,
        "total_gpus": point.total_gpus,
        "database_mode": point.database_mode,
        "engine_step_backend": point.engine_step_backend,
        "batch_sweep_step": 1,
        "nextn": point.nextn,
        **NEUTRAL_CORRECTIONS,
    }
    if run_spec.serving_mode == "agg":
        return Task(
            **common_kwargs,
            model_path=point.model,
            system_name=point.system,
            backend_name=point.backend,
            backend_version=point.backend_version,
            enable_chunked_prefill=point.chunked_prefill,
            agg_max_batch_size=active_caps.agg,
            **_role_candidates("agg", prefill_pattern),
        )

    return Task(
        **common_kwargs,
        prefill_model_path=point.model,
        prefill_system_name=point.system,
        prefill_backend_name=point.backend,
        prefill_backend_version=point.backend_version,
        prefill_enable_chunked_prefill=point.chunked_prefill,
        prefill_max_batch_size=active_caps.prefill,
        decode_model_path=point.model,
        decode_system_name=point.system,
        decode_backend_name=point.backend,
        decode_backend_version=point.backend_version,
        decode_max_batch_size=active_caps.decode,
        num_gpu_per_replica=list(NUM_GPU_PER_REPLICA),
        max_gpu_per_replica=64,
        max_prefill_workers=64,
        max_decode_workers=64,
        disagg_ranking_total_gpus=point.total_gpus,
        disagg_ranking_metric_kind=(
            "prefill_input_throughput"
            if point.workload_kind == "primary" and point.osl == 1
            else "output_token_throughput"
        ),
        **_role_candidates("prefill", prefill_pattern),
        **_role_candidates("decode", decode_pattern),
    )


def _exception_chain(error: BaseException):
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None


def classify_evaluation_error(error: Exception) -> str:
    """Classify only explicit AIC memory or SLA terminal exceptions."""
    for current in _exception_chain(error):
        if isinstance(current, InsufficientMemoryError | KVCacheCapacityError):
            return "memory_infeasible"
        if isinstance(current, NoFeasibleConfigError):
            return "sla_infeasible"
    raise error


def _task_result_records(result: Any) -> tuple[dict[str, Any], ...]:
    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"Task.run() must return a DataFrame-like object; got {type(result).__name__}")
    records = to_dict(orient="records")
    if not isinstance(records, list):
        raise TypeError(f"Task.run().to_dict(orient='records') must return a list; got {type(records).__name__}")
    if not all(isinstance(row, Mapping) for row in records):
        raise TypeError("Every Task.run() record must be a mapping")
    return tuple(dict(row) for row in records)


def _select_experiment_rank_one(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    candidate_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if not candidate_rows:
        raise RuntimeError(f"Task.run() returned no candidate rows for experiment {experiment!r}")

    required_fields = AGG_RAW_FIELDS if run_spec.serving_mode == "agg" else DISAGG_RAW_FIELDS
    ranked = []
    for row in candidate_rows:
        _require_row_fields(row, required_fields)
        metric = derive_ranking_metric_evidence(run_spec, row)
        if run_spec.serving_mode == "agg":
            sort_key = _canonical_agg_sort_key(experiment, row)
        else:
            sort_key = _canonical_disagg_sort_key(experiment, row, decode_cp=1)
        ranked.append(
            (
                -metric["ranking_metric_value"],
                sort_key,
                row,
            )
        )
    return dict(min(ranked, key=lambda item: (item[0], item[1]))[2])


def evaluate_experiment_attempt(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    caps: BatchCaps,
    task_factory: Callable[..., Task] = build_comparison_task,
) -> SearchAttempt:
    """Run one sweep attempt without detailed per-operation rerun work."""
    task = task_factory(run_spec, experiment=experiment, caps=caps)
    candidate_rows = _task_result_records(task.run())
    rank1_row = _select_experiment_rank_one(
        run_spec,
        experiment=experiment,
        candidate_rows=candidate_rows,
    )
    if run_spec.serving_mode == "agg":
        rank1_batch_sizes = {"agg": _positive_integer(rank1_row["bs"], field="aggregate batch")}
    else:
        rank1_batch_sizes = {
            "prefill": _positive_integer(rank1_row["(p)bs"], field="prefill batch"),
            "decode": _positive_integer(rank1_row["(d)bs"], field="decode batch"),
        }
    if run_spec.serving_mode == "agg":
        selected_point_identity = _canonical_agg_sort_key(experiment, rank1_row)
    else:
        selected_point_identity = _canonical_disagg_sort_key(experiment, rank1_row, decode_cp=1)
    return SearchAttempt(
        rank1_batch_sizes=rank1_batch_sizes,
        candidate_rows=candidate_rows,
        rank1_row=rank1_row,
        selected_point_identity=selected_point_identity,
        selected_evaluation=None,
        per_ops_evidence=None,
    )


def _active_cap_names(serving_mode: str) -> tuple[str, ...]:
    if serving_mode == "agg":
        return ("agg",)
    if serving_mode == "disagg":
        return ("prefill", "decode")
    raise ValueError(f"Unsupported serving mode: {serving_mode!r}")


def _validate_rank1_batch_sizes(
    *,
    serving_mode: str,
    caps: BatchCaps,
    rank1_batch_sizes: Mapping[str, int],
) -> tuple[str, ...]:
    active_names = _active_cap_names(serving_mode)
    if set(rank1_batch_sizes) != set(active_names):
        expected = ", ".join(active_names)
        raise ValueError(
            f"Rank-one batch evidence for {serving_mode} must contain exactly {expected}; "
            f"got {sorted(rank1_batch_sizes)}"
        )

    saturated = []
    for name in active_names:
        value = rank1_batch_sizes[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"Rank-one {name} batch must be a positive integer; got {value!r}")
        cap = getattr(caps, name)
        if value > cap:
            raise ValueError(f"Rank-one {name} batch {value} exceeds active cap {cap}")
        if value == cap:
            saturated.append(name)
    return tuple(saturated)


def expand_caps_until_terminal(
    serving_mode: str,
    evaluate: Callable[[BatchCaps], SearchAttempt],
    *,
    initial_caps: BatchCaps | None = None,
    experiment: str | None = None,
) -> CapSearchResult:
    """Repeat selective cap doubling until success, memory, or SLA termination."""
    _active_cap_names(serving_mode)
    caps = initial_caps or BatchCaps()
    history: list[BatchCaps] = []
    evidence_history: list[CapAttemptEvidence] = []
    rerun_count = 0

    while True:
        history.append(caps)
        try:
            attempt = evaluate(caps)
        except Exception as error:
            terminal_status = classify_evaluation_error(error)
            evidence_history.append(
                CapAttemptEvidence(
                    experiment=experiment,
                    caps=caps,
                    status=terminal_status,
                    search_attempt=None,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            return CapSearchResult(
                terminal_status=terminal_status,
                final_caps=caps,
                cap_history=tuple(history),
                cap_rerun_count=rerun_count,
                cap_saturated=False,
                ranking_eligible=False,
                attempt_evidence=tuple(evidence_history),
            )

        if not isinstance(attempt, SearchAttempt):
            raise TypeError(f"Evaluator must return SearchAttempt; got {type(attempt).__name__}")
        saturated = _validate_rank1_batch_sizes(
            serving_mode=serving_mode,
            caps=caps,
            rank1_batch_sizes=attempt.rank1_batch_sizes,
        )
        evidence_history.append(
            CapAttemptEvidence(
                experiment=experiment,
                caps=caps,
                status="cap_saturated" if saturated else "success",
                search_attempt=attempt,
            )
        )
        if not saturated:
            return CapSearchResult(
                terminal_status="success",
                final_caps=caps,
                cap_history=tuple(history),
                cap_rerun_count=rerun_count,
                cap_saturated=False,
                ranking_eligible=True,
                attempt_evidence=tuple(evidence_history),
            )

        caps = replace(
            caps,
            **{name: getattr(caps, name) * 2 for name in saturated},
        )
        rerun_count += 1


def expand_caps_for_all_experiments(
    run_spec: ModeRunSpec,
    evaluate: Callable[[str, BatchCaps], SearchAttempt],
    *,
    initial_caps: BatchCaps | None = None,
) -> dict[str, CapSearchResult]:
    """Complete cap expansion independently for every experiment in one mode run."""
    if not isinstance(run_spec, ModeRunSpec):
        raise TypeError(f"run_spec must be ModeRunSpec; got {type(run_spec).__name__}")
    experiments = tuple(
        experiment
        for experiment, (serving_mode, _prefill_pattern, _decode_pattern) in EXPERIMENT_PATTERNS.items()
        if serving_mode == run_spec.serving_mode
    )
    if not experiments:
        raise ValueError(f"Unsupported serving mode: {run_spec.serving_mode!r}")

    results: dict[str, CapSearchResult] = {}
    for experiment in experiments:
        results[experiment] = expand_caps_until_terminal(
            run_spec.serving_mode,
            lambda caps, experiment=experiment: evaluate(experiment, caps),
            initial_caps=initial_caps,
            experiment=experiment,
        )
    return results


def _attach_final_detailed_evaluation(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    cap_result: CapSearchResult,
    task_factory: Callable[..., Task],
) -> CapSearchResult:
    if cap_result.terminal_status != "success":
        raise ValueError(f"Cannot attach detailed evaluation to {cap_result.terminal_status!r}")
    if not cap_result.attempt_evidence:
        raise RuntimeError("Successful cap search is missing final attempt evidence")
    final_evidence = cap_result.attempt_evidence[-1]
    attempt = final_evidence.search_attempt
    if final_evidence.status != "success" or final_evidence.caps != cap_result.final_caps or attempt is None:
        raise ValueError("Successful cap search has inconsistent final attempt evidence")
    if attempt.rank1_row is None or attempt.selected_point_identity is None:
        raise RuntimeError("Successful cap search is missing its selected sweep point")
    if attempt.selected_evaluation is not None or attempt.per_ops_evidence is not None:
        raise ValueError("Final sweep attempt already contains detailed evaluation evidence")

    task = task_factory(run_spec, experiment=experiment, caps=cap_result.final_caps)
    selected_evaluation = rerun_selected_point(
        serving_mode=run_spec.serving_mode,
        experiment=experiment,
        task=task,
        selected_row=attempt.rank1_row,
    )
    per_ops_evidence = validate_per_ops_evidence(
        selected_evaluation,
        serving_mode=run_spec.serving_mode,
        osl=run_spec.point.osl,
    )
    detailed_attempt = replace(
        attempt,
        selected_evaluation=selected_evaluation,
        per_ops_evidence=per_ops_evidence,
    )
    detailed_evidence = replace(final_evidence, search_attempt=detailed_attempt)
    return replace(
        cap_result,
        attempt_evidence=(*cap_result.attempt_evidence[:-1], detailed_evidence),
    )


def run_experiment_cap_search(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    initial_caps: BatchCaps | None = None,
    task_factory: Callable[..., Task] | None = None,
) -> CapSearchResult:
    """Run real sweep and exact selected-point attempts until one terminal result."""
    if experiment not in EXPERIMENT_PATTERNS:
        raise ValueError(f"Unknown comparison experiment: {experiment!r}")
    expected_mode = EXPERIMENT_PATTERNS[experiment][0]
    if run_spec.serving_mode != expected_mode:
        raise ValueError(f"Experiment {experiment!r} does not match serving mode {run_spec.serving_mode!r}")
    active_factory = task_factory or build_comparison_task
    cap_result = expand_caps_until_terminal(
        run_spec.serving_mode,
        lambda caps: evaluate_experiment_attempt(
            run_spec,
            experiment=experiment,
            caps=caps,
            task_factory=active_factory,
        ),
        initial_caps=initial_caps,
        experiment=experiment,
    )
    if cap_result.terminal_status != "success":
        return cap_result
    return _attach_final_detailed_evaluation(
        run_spec,
        experiment=experiment,
        cap_result=cap_result,
        task_factory=active_factory,
    )


def run_all_experiment_cap_searches(
    run_spec: ModeRunSpec,
    *,
    initial_caps: BatchCaps | None = None,
    task_factory: Callable[..., Task] | None = None,
) -> dict[str, CapSearchResult]:
    """Run all aggregate patterns or disaggregate pairings without an external callback."""
    active_factory = task_factory or build_comparison_task
    experiments = tuple(
        experiment
        for experiment, (serving_mode, _prefill_pattern, _decode_pattern) in EXPERIMENT_PATTERNS.items()
        if serving_mode == run_spec.serving_mode
    )
    if not experiments:
        raise ValueError(f"Unsupported serving mode: {run_spec.serving_mode!r}")
    return {
        experiment: run_experiment_cap_search(
            run_spec,
            experiment=experiment,
            initial_caps=initial_caps,
            task_factory=active_factory,
        )
        for experiment in experiments
    }


def _require_row_fields(row: Mapping[str, Any], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"Result row is missing required fields: {missing}")


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a finite number; got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return numeric


def derive_cluster_allocation(
    *,
    tokens_per_second_per_gpu: float,
    num_total_gpus: int,
    total_gpus: int,
) -> ClusterAllocation:
    """Scale one deployment to the cluster using complete replicas only."""
    if type(total_gpus) is not int:
        raise TypeError(f"total_gpus must be a positive integer; got {total_gpus!r}")
    if total_gpus < 1:
        raise ValueError(f"total_gpus must be positive; got {total_gpus!r}")
    if type(num_total_gpus) is not int:
        raise TypeError(f"num_total_gpus must be a positive integer; got {num_total_gpus!r}")
    if num_total_gpus < 1:
        raise ValueError(f"num_total_gpus must be positive; got {num_total_gpus!r}")
    if num_total_gpus > total_gpus:
        raise ValueError(f"num_total_gpus {num_total_gpus} exceeds total_gpus {total_gpus}")

    throughput = _finite_number(tokens_per_second_per_gpu, field="tokens_per_second_per_gpu")
    replicas = total_gpus // num_total_gpus
    total_gpus_used = replicas * num_total_gpus
    return ClusterAllocation(
        replicas=replicas,
        total_gpus_used=total_gpus_used,
        unused_gpus=total_gpus - total_gpus_used,
        tokens_per_second_per_gpu_cluster=cluster_normalized_throughput(
            tokens_per_second_per_gpu=throughput,
            deployment_gpus=num_total_gpus,
            total_gpus=total_gpus,
        ),
    )


def derive_ranking_metric_evidence(
    run_spec: ModeRunSpec,
    raw_row: Mapping[str, Any],
) -> dict[str, float | str]:
    """Return the workload-specific fixed-cluster ranking metric."""
    if not isinstance(run_spec, ModeRunSpec):
        raise TypeError(f"run_spec must be ModeRunSpec; got {type(run_spec).__name__}")
    if not isinstance(raw_row, Mapping):
        raise TypeError(f"raw_row must be a mapping; got {type(raw_row).__name__}")

    point = run_spec.point
    deployment_gpus = _positive_integer(raw_row.get("num_total_gpus"), field="num_total_gpus")
    if point.workload_kind == "primary" and point.osl == 1:
        if run_spec.serving_mode == "agg":
            global_batch_size = raw_row.get("global_bs")
            num_workers = 1
        elif run_spec.serving_mode == "disagg":
            global_batch_size = raw_row.get("(p)global_bs")
            num_workers = raw_row.get("(p)workers")
        else:
            raise ValueError(f"Unsupported serving mode: {run_spec.serving_mode!r}")
        prefill_tokens_per_second = calculate_prefill_tokens_per_second(
            global_batch_size=global_batch_size,
            num_workers=num_workers,
            isl=point.isl,
            prefix=point.prefix,
            ttft_ms=raw_row.get("ttft"),
        )
        per_gpu = prefill_tokens_per_second / deployment_gpus
        allocation = derive_cluster_allocation(
            tokens_per_second_per_gpu=per_gpu,
            num_total_gpus=deployment_gpus,
            total_gpus=point.total_gpus,
        )
        cluster_value = allocation.tokens_per_second_per_gpu_cluster
        return {
            "ranking_metric_kind": "prefill_input_throughput",
            "ranking_metric_value": cluster_value,
            "prefill_tokens/s": prefill_tokens_per_second,
            "prefill_tokens/s/gpu": per_gpu,
            "prefill_tokens/s/gpu_cluster": cluster_value,
        }
    if point.workload_kind == "decode_smoke" and point.osl > 1:
        allocation = derive_cluster_allocation(
            tokens_per_second_per_gpu=raw_row.get("tokens/s/gpu"),
            num_total_gpus=deployment_gpus,
            total_gpus=point.total_gpus,
        )
        return {
            "ranking_metric_kind": "output_token_throughput",
            "ranking_metric_value": allocation.tokens_per_second_per_gpu_cluster,
        }
    raise ValueError(f"Unsupported ranking workload contract: workload_kind={point.workload_kind!r}, osl={point.osl!r}")


def _positive_integer(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a positive integer; got {value!r}")
    if value < 1:
        raise ValueError(f"{field} must be positive; got {value!r}")
    return value


def _require_equal(*, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


def _validate_success_cap_result(cap_result: CapSearchResult) -> None:
    if not isinstance(cap_result, CapSearchResult):
        raise TypeError(f"cap_result must be CapSearchResult; got {type(cap_result).__name__}")
    if cap_result.terminal_status != "success":
        raise ValueError(f"Cannot normalize non-success cap result: {cap_result.terminal_status!r}")
    if cap_result.cap_saturated is not False or cap_result.ranking_eligible is not True:
        raise ValueError("Successful cap result must be non-saturated and ranking-eligible")
    if not cap_result.cap_history or cap_result.cap_history[-1] != cap_result.final_caps:
        raise ValueError("cap_history must be non-empty and end at final_caps")
    if cap_result.cap_rerun_count != len(cap_result.cap_history) - 1:
        raise ValueError("cap_rerun_count must equal len(cap_history) - 1")
    if cap_result.attempt_evidence:
        evidence_caps = tuple(evidence.caps for evidence in cap_result.attempt_evidence)
        if evidence_caps != cap_result.cap_history:
            raise ValueError("attempt_evidence caps must exactly match cap_history")


def _validate_task_result_contract(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    task: Task,
) -> None:
    if experiment not in EXPERIMENT_PATTERNS:
        raise ValueError(f"Unknown comparison experiment: {experiment!r}")
    expected_mode = EXPERIMENT_PATTERNS[experiment][0]
    _require_equal(field="experiment serving_mode", actual=run_spec.serving_mode, expected=expected_mode)

    point = run_spec.point
    for field, expected in (
        ("serving_mode", run_spec.serving_mode),
        ("isl", point.isl),
        ("osl", point.osl),
        ("prefix", point.prefix),
        ("ttft", point.ttft_sla_ms),
        ("tpot", point.tpot_ms),
        ("pareto_sweep", point.pareto_sweep),
        ("total_gpus", point.total_gpus),
        ("database_mode", point.database_mode),
        ("nextn", point.nextn),
        ("batch_sweep_step", 1),
    ):
        _require_equal(field=f"Task.{field}", actual=getattr(task, field), expected=expected)
    for field, expected in NEUTRAL_CORRECTIONS.items():
        _require_equal(field=f"Task.{field}", actual=getattr(task, field), expected=expected)

    if run_spec.serving_mode == "agg":
        _require_equal(
            field="Task.disagg_ranking_total_gpus",
            actual=task.disagg_ranking_total_gpus,
            expected=None,
        )
        for field, expected in (
            ("model_path", point.model),
            ("system_name", point.system),
            ("backend_name", point.backend),
            ("backend_version", point.backend_version),
            ("enable_chunked_prefill", point.chunked_prefill),
        ):
            _require_equal(field=f"Task.{field}", actual=getattr(task, field), expected=expected)
        return

    _require_equal(
        field="Task.disagg_ranking_total_gpus",
        actual=task.disagg_ranking_total_gpus,
        expected=point.total_gpus,
    )
    _require_equal(
        field="Task.disagg_ranking_metric_kind",
        actual=task.disagg_ranking_metric_kind,
        expected=(
            "prefill_input_throughput"
            if point.workload_kind == "primary" and point.osl == 1
            else "output_token_throughput"
        ),
    )
    for role in ("prefill", "decode"):
        for field, expected in (
            ("model_path", point.model),
            ("system_name", point.system),
            ("backend_name", point.backend),
            ("backend_version", point.backend_version),
        ):
            attr = f"{role}_{field}"
            _require_equal(field=f"Task.{attr}", actual=getattr(task, attr), expected=expected)
    _require_equal(
        field="Task.prefill_enable_chunked_prefill",
        actual=task.prefill_enable_chunked_prefill,
        expected=point.chunked_prefill,
    )


def _validate_parallel_pattern(
    *,
    pattern: str,
    tp: int,
    pp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
    cp: int,
    role: str,
) -> int:
    values = {
        "tp": _positive_integer(tp, field=f"{role}.tp"),
        "pp": _positive_integer(pp, field=f"{role}.pp"),
        "dp": _positive_integer(dp, field=f"{role}.dp"),
        "moe_tp": _positive_integer(moe_tp, field=f"{role}.moe_tp"),
        "moe_ep": _positive_integer(moe_ep, field=f"{role}.moe_ep"),
        "cp": _positive_integer(cp, field=f"{role}.cp"),
    }
    if values["pp"] != 1 or values["cp"] != 1 or values["tp"] > 8:
        raise ValueError(f"{role} violates pp=1, cp=1, tp<=8: {values}")
    if values["dp"] * values["tp"] * values["cp"] != values["moe_tp"] * values["moe_ep"]:
        raise ValueError(f"{role} violates dp*tp*cp == moe_tp*moe_ep: {values}")
    if pattern == "A":
        valid = values["moe_tp"] == 1 and values["moe_ep"] in (2, 4, 8, 16, 32)
    elif pattern == "B":
        valid = values["dp"] == 1 and values["moe_tp"] == values["tp"] and values["moe_ep"] == 1
    else:
        raise ValueError(f"Unknown parallel pattern: {pattern!r}")
    if not valid:
        raise ValueError(f"{role} does not match Pattern {pattern}: {values}")
    return values["tp"] * values["pp"] * values["dp"] * values["cp"]


def _simulation_status(system: str, system_spec: Mapping[str, Any]) -> str:
    metadata = system_spec.get("metadata")
    declared = metadata.get("simulation_status") if isinstance(metadata, Mapping) else None
    if system == "h800_sxm":
        if declared != "simulated":
            raise ValueError("h800_sxm system spec must declare metadata.simulation_status='simulated'")
        return "simulated"
    if declared == "simulated":
        raise ValueError(f"Non-H800 system {system!r} unexpectedly declares simulation_status='simulated'")
    return "not_simulated"


def _canonical_agg_sort_key(experiment: str, row: Mapping[str, Any]) -> tuple[str | int, ...]:
    _require_row_fields(row, AGG_POINT_IDENTITY_FIELDS)
    return (
        experiment,
        *(_positive_integer(row[field], field=f"aggregate {field}") for field in AGG_POINT_IDENTITY_FIELDS),
    )


def _canonical_agg_config_id(experiment: str, row: Mapping[str, Any]) -> str:
    sort_key = _canonical_agg_sort_key(experiment, row)
    return "|".join(
        (
            experiment,
            *(f"{field}={value}" for field, value in zip(AGG_POINT_IDENTITY_FIELDS, sort_key[1:], strict=True)),
        )
    )


def _canonical_disagg_values(
    row: Mapping[str, Any],
    *,
    decode_cp: int,
) -> tuple[tuple[str, int], ...]:
    raw_values = (
        ("p_tp", "(p)tp"),
        ("p_pp", "(p)pp"),
        ("p_dp", "(p)dp"),
        ("p_moe_tp", "(p)moe_tp"),
        ("p_moe_ep", "(p)moe_ep"),
        ("p_cp", "(p)cp"),
        ("p_bs", "(p)bs"),
        ("p_workers", "(p)workers"),
        ("d_tp", "(d)tp"),
        ("d_pp", "(d)pp"),
        ("d_dp", "(d)dp"),
        ("d_moe_tp", "(d)moe_tp"),
        ("d_moe_ep", "(d)moe_ep"),
        ("d_bs", "(d)bs"),
        ("d_workers", "(d)workers"),
    )
    _require_row_fields(row, (raw_field for _label, raw_field in raw_values))
    values = [
        (label, _positive_integer(row[raw_field], field=f"disaggregate {label}")) for label, raw_field in raw_values
    ]
    values.insert(13, ("d_cp", _positive_integer(decode_cp, field="disaggregate d_cp")))
    return tuple(values)


def _canonical_disagg_sort_key(
    experiment: str,
    row: Mapping[str, Any],
    *,
    decode_cp: int,
) -> tuple[str | int, ...]:
    values = _canonical_disagg_values(row, decode_cp=decode_cp)
    return (experiment, *(value for _field, value in values))


def _canonical_disagg_config_id(experiment: str, row: Mapping[str, Any], *, decode_cp: int) -> str:
    values = _canonical_disagg_values(row, decode_cp=decode_cp)
    return "|".join((experiment, *(f"{field}={value}" for field, value in values)))


def rerun_selected_point(
    *,
    serving_mode: str,
    experiment: str,
    task: Any,
    selected_row: Mapping[str, Any],
) -> SinglePointEvaluation:
    """Re-evaluate one selected sweep row with authoritative per-op evidence."""
    if not isinstance(selected_row, Mapping):
        raise TypeError(f"selected_row must be a mapping; got {type(selected_row).__name__}")
    with perf_database.capture_collective_queries() as communication_evidence:
        if serving_mode == "agg":
            _require_row_fields(selected_row, AGG_RAW_FIELDS)
            _require_equal(field="aggregate cp", actual=selected_row["cp"], expected=1)
            evaluation = task.run_single_agg(
                tp=_positive_integer(selected_row["tp"], field="aggregate tp"),
                pp=_positive_integer(selected_row["pp"], field="aggregate pp"),
                dp=_positive_integer(selected_row["dp"], field="aggregate dp"),
                moe_tp=_positive_integer(selected_row["moe_tp"], field="aggregate moe_tp"),
                moe_ep=_positive_integer(selected_row["moe_ep"], field="aggregate moe_ep"),
                batch_size=_positive_integer(selected_row["bs"], field="aggregate batch"),
                ctx_tokens=_positive_integer(selected_row["ctx_tokens"], field="aggregate ctx_tokens"),
                include_per_ops=True,
            )
            selected_key = _canonical_agg_sort_key(experiment, selected_row)
        elif serving_mode == "disagg":
            _require_row_fields(selected_row, DISAGG_RAW_FIELDS)
            _require_equal(field="prefill cp", actual=selected_row["(p)cp"], expected=1)
            evaluation = task.run_single_disagg(
                prefill_tp=_positive_integer(selected_row["(p)tp"], field="prefill tp"),
                prefill_pp=_positive_integer(selected_row["(p)pp"], field="prefill pp"),
                prefill_dp=_positive_integer(selected_row["(p)dp"], field="prefill dp"),
                prefill_moe_tp=_positive_integer(selected_row["(p)moe_tp"], field="prefill moe_tp"),
                prefill_moe_ep=_positive_integer(selected_row["(p)moe_ep"], field="prefill moe_ep"),
                prefill_batch_size=_positive_integer(selected_row["(p)bs"], field="prefill batch"),
                prefill_num_workers=_positive_integer(selected_row["(p)workers"], field="prefill workers"),
                decode_tp=_positive_integer(selected_row["(d)tp"], field="decode tp"),
                decode_pp=_positive_integer(selected_row["(d)pp"], field="decode pp"),
                decode_dp=_positive_integer(selected_row["(d)dp"], field="decode dp"),
                decode_moe_tp=_positive_integer(selected_row["(d)moe_tp"], field="decode moe_tp"),
                decode_moe_ep=_positive_integer(selected_row["(d)moe_ep"], field="decode moe_ep"),
                decode_batch_size=_positive_integer(selected_row["(d)bs"], field="decode batch"),
                decode_num_workers=_positive_integer(selected_row["(d)workers"], field="decode workers"),
                include_per_ops=True,
            )
            selected_key = _canonical_disagg_sort_key(experiment, selected_row, decode_cp=1)
        else:
            raise ValueError(f"Unsupported serving mode: {serving_mode!r}")

    if not isinstance(evaluation, SinglePointEvaluation):
        raise TypeError(
            f"Detailed selected-point evaluation must return SinglePointEvaluation; got {type(evaluation).__name__}"
        )
    evaluation = replace(
        evaluation,
        communication_evidence=tuple(communication_evidence),
    )
    if not isinstance(evaluation.row, Mapping):
        raise TypeError(f"Detailed selected-point row must be a mapping; got {type(evaluation.row).__name__}")
    if serving_mode == "agg":
        rerun_key = _canonical_agg_sort_key(experiment, evaluation.row)
    else:
        rerun_key = _canonical_disagg_sort_key(experiment, evaluation.row, decode_cp=1)
    if rerun_key != selected_key:
        raise ValueError(f"Selected-point rerun identity mismatch: expected {selected_key!r}, got {rerun_key!r}")
    return evaluation


def _is_attention_op(op_name: str) -> bool:
    normalized = op_name.lower()
    return any(marker in normalized for marker in ("attention", "attn", "mla", "bmm", "qkv"))


def validate_per_ops_evidence(
    evaluation: SinglePointEvaluation,
    *,
    serving_mode: str,
    osl: int,
) -> dict[str, Any]:
    """Validate strict-SOL per-op evidence and compute attention splits."""
    if not isinstance(evaluation, SinglePointEvaluation):
        raise TypeError(f"evaluation must be SinglePointEvaluation; got {type(evaluation).__name__}")
    output_length = _positive_integer(osl, field="osl")
    data = evaluation.per_ops_data
    sources = evaluation.per_ops_source
    if not isinstance(data, dict) or not isinstance(sources, dict):
        raise TypeError("per-operation data and source evidence must both be dictionaries")

    if serving_mode == "agg":
        if "scheduling" not in data:
            raise ValueError("aggregate per-operation data must include scheduling evidence")
        if "scheduling" in sources:
            raise ValueError("aggregate scheduling counters must not have operation sources")
        data_phases = set(data) - {"scheduling"}
        source_phases = set(sources)
        if not data_phases or data_phases != source_phases:
            raise ValueError(
                f"aggregate per-operation phases mismatch: data={sorted(data_phases)}, source={sorted(source_phases)}"
            )
    elif serving_mode == "disagg":
        data_phases = set(data)
        source_phases = set(sources)
        expected_phases = {"prefill", "decode"}
        if data_phases != expected_phases or source_phases != expected_phases:
            raise ValueError(
                f"disaggregate per-operation phases must be prefill/decode: "
                f"data={sorted(data_phases)}, source={sorted(source_phases)}"
            )
    else:
        raise ValueError(f"Unsupported serving mode: {serving_mode!r}")

    phase_totals: dict[str, dict[str, float]] = {}
    for phase in sorted(data_phases):
        phase_data = data[phase]
        phase_sources = sources[phase]
        if not isinstance(phase_data, dict) or not isinstance(phase_sources, dict):
            raise TypeError(f"{phase} per-operation data and sources must both be dictionaries")
        allow_empty = serving_mode == "disagg" and phase == "decode" and output_length == 1
        if not phase_data and not allow_empty:
            raise ValueError(f"{phase} per-operation evidence must not be empty")
        if set(phase_data) != set(phase_sources):
            raise ValueError(
                f"{phase} per-operation data/source keys mismatch: "
                f"data={sorted(phase_data)}, source={sorted(phase_sources)}"
            )

        attention_latencies = []
        non_attention_latencies = []
        for op_name in sorted(phase_data):
            latency = phase_data[op_name]
            if not isinstance(op_name, str) or not op_name:
                raise TypeError(f"{phase} operation names must be non-empty strings; got {op_name!r}")
            latency_ms = _finite_number(latency, field=f"{phase}.{op_name}")
            if latency_ms < 0:
                raise ValueError(f"{phase}.{op_name} latency must be non-negative; got {latency_ms}")
            source = phase_sources[op_name]
            if source == "not_executed":
                if op_name != "generation_attention (not executed)" or latency_ms != 0.0:
                    raise ValueError(
                        f"{phase}.{op_name} uses not_executed without the exact zero-latency generation placeholder"
                    )
            elif source != "sol":
                raise ValueError(f"{phase}.{op_name} must use strict SOL source; got {source!r}")
            if _is_attention_op(op_name):
                attention_latencies.append(latency_ms)
            else:
                non_attention_latencies.append(latency_ms)
        attention = math.fsum(attention_latencies)
        non_attention = math.fsum(non_attention_latencies)
        phase_totals[phase] = {
            "attention": attention,
            "non_attention": non_attention,
            "total": attention + non_attention,
        }

    if serving_mode == "agg":
        scheduling = data["scheduling"]
        if not isinstance(scheduling, dict):
            raise TypeError("aggregate scheduling evidence must be a dictionary")
        for name, value in scheduling.items():
            numeric = _finite_number(value, field=f"scheduling.{name}")
            if numeric < 0:
                raise ValueError(f"scheduling.{name} must be non-negative; got {numeric}")
        multipliers = {
            "mix_step": _finite_number(scheduling.get("num_mix_steps"), field="scheduling.num_mix_steps"),
            "genonly_step": _finite_number(
                scheduling.get("num_genonly_steps"),
                field="scheduling.num_genonly_steps",
            ),
        }
        weighted_attention = math.fsum(
            phase_totals[phase]["attention"] * multipliers.get(phase, 1.0) for phase in sorted(phase_totals)
        )
        weighted_non_attention = math.fsum(
            phase_totals[phase]["non_attention"] * multipliers.get(phase, 1.0) for phase in sorted(phase_totals)
        )
    else:
        weighted_attention = math.fsum(phase_totals[phase]["attention"] for phase in sorted(phase_totals))
        weighted_non_attention = math.fsum(phase_totals[phase]["non_attention"] for phase in sorted(phase_totals))
    weighted = {
        "attention": weighted_attention,
        "non_attention": weighted_non_attention,
        "total": math.fsum((weighted_attention, weighted_non_attention)),
    }
    return {
        "per_ops_data": data,
        "per_ops_source": sources,
        "phase_totals_ms": phase_totals,
        "weighted_totals_ms": weighted,
    }


def _communication_evidence_payload(
    evaluation: SinglePointEvaluation,
) -> tuple[dict[str, Any], ...]:
    evidence = evaluation.communication_evidence
    if evidence is None:
        raise RuntimeError("Exact rerun communication query evidence was not captured")
    if not isinstance(evidence, tuple):
        raise TypeError("communication_evidence must be a tuple")

    payload: list[dict[str, Any]] = []
    for index, record in enumerate(evidence):
        if not isinstance(record, perf_database.CommunicationQueryEvidence):
            raise TypeError(
                "communication_evidence entries must be CommunicationQueryEvidence; "
                f"entry {index} is {type(record).__name__}"
            )
        if not isinstance(record.operation_name, str) or not record.operation_name:
            raise ValueError(f"communication_evidence[{index}].operation_name must be non-empty")
        if record.operation_kind not in {"custom_allreduce", "nccl", "p2p"}:
            raise ValueError(f"communication_evidence[{index}].operation_kind is invalid: {record.operation_kind!r}")
        if record.operation_kind == "p2p":
            if record.collective is not None:
                raise ValueError(f"communication_evidence[{index}] p2p collective must be None")
        elif not isinstance(record.collective, str) or not record.collective:
            raise ValueError(f"communication_evidence[{index}].collective must be non-empty")
        _positive_integer(record.group_size, field=f"communication_evidence[{index}].group_size")
        if record.group_size <= 1:
            raise ValueError(f"communication_evidence[{index}].group_size must exceed one")
        if record.tier not in {"intra_node_bw", "inter_node_bw", "inter_rack_bw"}:
            raise ValueError(f"communication_evidence[{index}].tier is invalid: {record.tier!r}")
        bandwidth = _finite_number(
            record.bandwidth_bytes_per_sec,
            field=f"communication_evidence[{index}].bandwidth_bytes_per_sec",
        )
        if bandwidth <= 0:
            raise ValueError(f"communication_evidence[{index}].bandwidth_bytes_per_sec must be positive")
        message_size = _finite_number(
            record.message_size_bytes,
            field=f"communication_evidence[{index}].message_size_bytes",
        )
        if message_size <= 0:
            raise ValueError(f"communication_evidence[{index}].message_size_bytes must be positive")
        payload.append(asdict(record))
    return tuple(payload)


def _deserialize_communication_evidence(
    value: Any,
    *,
    experiment: str,
) -> tuple[perf_database.CommunicationQueryEvidence, ...]:
    if value is None:
        raise ValueError(f"selected evaluation for {experiment} is missing communication_evidence")
    if not isinstance(value, list):
        raise TypeError(f"selected evaluation communication_evidence for {experiment} must be a list")
    expected_fields = {
        "operation_name",
        "operation_kind",
        "collective",
        "group_size",
        "tier",
        "bandwidth_bytes_per_sec",
        "message_size_bytes",
    }
    records = tuple(
        perf_database.CommunicationQueryEvidence(
            **_require_exact_mapping_fields(
                raw_record,
                expected=expected_fields,
                context=f"collective evidence {index} for {experiment}",
            )
        )
        for index, raw_record in enumerate(value)
    )
    _communication_evidence_payload(
        SinglePointEvaluation(
            row={},
            per_ops_data={},
            per_ops_source={},
            communication_evidence=records,
        )
    )
    return records


def _row_saturated_cap_names(
    serving_mode: str,
    row: Mapping[str, Any],
    caps: BatchCaps,
) -> tuple[str, ...]:
    if serving_mode == "agg":
        fields = (("agg", "aggregate", "bs"),)
    elif serving_mode == "disagg":
        fields = (
            ("prefill", "prefill", "(p)bs"),
            ("decode", "decode", "(d)bs"),
        )
    else:
        raise ValueError(f"Unsupported serving mode: {serving_mode!r}")

    saturated = []
    for cap_name, role_name, row_field in fields:
        batch_size = _positive_integer(row[row_field], field=f"{role_name} batch")
        cap = getattr(caps, cap_name)
        if batch_size > cap:
            raise ValueError(f"{role_name} batch {batch_size} exceeds final {cap_name} cap {cap}")
        if batch_size == cap:
            saturated.append(cap_name)
    return tuple(saturated)


def normalize_success_result(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    task: Task,
    raw_row: Mapping[str, Any],
    cap_result: CapSearchResult,
    system_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and enrich one successful single-point AIC result row."""
    _validate_success_cap_result(cap_result)
    _validate_task_result_contract(run_spec, experiment=experiment, task=task)
    if not isinstance(raw_row, Mapping):
        raise TypeError(f"raw_row must be a mapping; got {type(raw_row).__name__}")
    if not isinstance(system_spec, Mapping):
        raise TypeError(f"system_spec must be a mapping; got {type(system_spec).__name__}")

    point = run_spec.point
    required_fields = AGG_RAW_FIELDS if run_spec.serving_mode == "agg" else DISAGG_RAW_FIELDS
    _require_row_fields(raw_row, required_fields)
    for field, expected in (
        ("model", point.model),
        ("isl", point.isl),
        ("osl", point.osl),
        ("prefix", point.prefix),
    ):
        _require_equal(field=f"raw {field}", actual=raw_row[field], expected=expected)

    numeric_evidence_fields = ("memory",) if run_spec.serving_mode == "agg" else ("(p)memory", "(d)memory")
    for metric in (
        "ttft",
        "tpot",
        "request_latency",
        "seq/s",
        "seq/s/gpu",
        "tokens/s",
        "tokens/s/gpu",
        "tokens/s/user",
        *numeric_evidence_fields,
    ):
        _finite_number(raw_row[metric], field=metric)
    num_total_gpus = _positive_integer(raw_row["num_total_gpus"], field="num_total_gpus")
    allocation = derive_cluster_allocation(
        tokens_per_second_per_gpu=raw_row["tokens/s/gpu"],
        num_total_gpus=num_total_gpus,
        total_gpus=point.total_gpus,
    )
    ranking_metric_evidence = derive_ranking_metric_evidence(run_spec, raw_row)

    expected_mode, prefill_pattern, decode_pattern = EXPERIMENT_PATTERNS[experiment]
    _require_equal(field="serving_mode", actual=run_spec.serving_mode, expected=expected_mode)
    normalized = dict(raw_row)
    if run_spec.serving_mode == "agg":
        for field, expected in (
            ("backend", point.backend),
            ("version", point.backend_version),
            ("system", point.system),
        ):
            _require_equal(field=f"raw {field}", actual=raw_row[field], expected=expected)
        worker_gpus = _validate_parallel_pattern(
            pattern=prefill_pattern,
            tp=raw_row["tp"],
            pp=raw_row["pp"],
            dp=raw_row["dp"],
            moe_tp=raw_row["moe_tp"],
            moe_ep=raw_row["moe_ep"],
            cp=raw_row["cp"],
            role="aggregate",
        )
        _require_equal(field="raw num_total_gpus", actual=num_total_gpus, expected=worker_gpus)
        best_batch_size: int | dict[str, int] = _positive_integer(raw_row["bs"], field="bs")
        global_batch_size = _positive_integer(raw_row["global_bs"], field="global_bs")
        _require_equal(
            field="raw aggregate global batch",
            actual=global_batch_size,
            expected=best_batch_size * raw_row["dp"],
        )
        canonical_config_sort_key = _canonical_agg_sort_key(experiment, raw_row)
        canonical_config_id = _canonical_agg_config_id(experiment, raw_row)
        normalized.update(
            worker_gpus=worker_gpus,
            best_batch_size=best_batch_size,
        )
    else:
        for role_prefix in ("(p)", "(d)"):
            for field, expected in (
                ("backend", point.backend),
                ("version", point.backend_version),
                ("system", point.system),
            ):
                raw_field = f"{role_prefix}{field}"
                _require_equal(field=f"raw {raw_field}", actual=raw_row[raw_field], expected=expected)
        decode_cp_candidates = task.decode_cp_candidates
        if decode_cp_candidates != [1]:
            raise ValueError(
                f"Task.decode_cp_candidates must provide exact decode cp evidence [1]; got {decode_cp_candidates!r}"
            )
        decode_cp = 1
        prefill_worker_gpus = _validate_parallel_pattern(
            pattern=prefill_pattern,
            tp=raw_row["(p)tp"],
            pp=raw_row["(p)pp"],
            dp=raw_row["(p)dp"],
            moe_tp=raw_row["(p)moe_tp"],
            moe_ep=raw_row["(p)moe_ep"],
            cp=raw_row["(p)cp"],
            role="prefill",
        )
        decode_worker_gpus = _validate_parallel_pattern(
            pattern=decode_pattern,
            tp=raw_row["(d)tp"],
            pp=raw_row["(d)pp"],
            dp=raw_row["(d)dp"],
            moe_tp=raw_row["(d)moe_tp"],
            moe_ep=raw_row["(d)moe_ep"],
            cp=decode_cp,
            role="decode",
        )
        prefill_worker_count = _positive_integer(raw_row["(p)workers"], field="prefill workers")
        decode_worker_count = _positive_integer(raw_row["(d)workers"], field="decode workers")
        expected_total = prefill_worker_gpus * prefill_worker_count + decode_worker_gpus * decode_worker_count
        _require_equal(field="raw num_total_gpus", actual=num_total_gpus, expected=expected_total)
        prefill_batch_size = _positive_integer(raw_row["(p)bs"], field="prefill batch")
        decode_batch_size = _positive_integer(raw_row["(d)bs"], field="decode batch")
        prefill_global_batch_size = _positive_integer(
            raw_row["(p)global_bs"],
            field="prefill global batch",
        )
        decode_global_batch_size = _positive_integer(
            raw_row["(d)global_bs"],
            field="decode global batch",
        )
        _require_equal(
            field="raw prefill global batch",
            actual=prefill_global_batch_size,
            expected=prefill_batch_size * raw_row["(p)dp"],
        )
        _require_equal(
            field="raw decode global batch",
            actual=decode_global_batch_size,
            expected=decode_batch_size * raw_row["(d)dp"],
        )
        best_batch_size = {"prefill": prefill_batch_size, "decode": decode_batch_size}
        canonical_config_sort_key = _canonical_disagg_sort_key(
            experiment,
            raw_row,
            decode_cp=decode_cp,
        )
        canonical_config_id = _canonical_disagg_config_id(experiment, raw_row, decode_cp=decode_cp)
        normalized.update(
            prefill_worker_gpus=prefill_worker_gpus,
            decode_worker_gpus=decode_worker_gpus,
            prefill_worker_count=prefill_worker_count,
            decode_worker_count=decode_worker_count,
            prefill_batch_size=prefill_batch_size,
            decode_batch_size=decode_batch_size,
            decode_cp=decode_cp,
            best_batch_size=best_batch_size,
        )

    saturated_cap_names = _row_saturated_cap_names(
        run_spec.serving_mode,
        raw_row,
        cap_result.final_caps,
    )
    row_cap_saturated = bool(saturated_cap_names)
    ttft = _finite_number(raw_row["ttft"], field="ttft")
    ttft_pass = ttft <= point.ttft_sla_ms if run_spec.serving_mode == "agg" else ttft < point.ttft_sla_ms
    normalized.update(
        model=point.model,
        system=point.system,
        simulation_status=_simulation_status(point.system, system_spec),
        workload_kind=point.workload_kind,
        isl=point.isl,
        osl=point.osl,
        prefix=point.prefix,
        ttft_sla_ms=point.ttft_sla_ms,
        tpot_constraint_ms=point.tpot_ms,
        serving_mode=run_spec.serving_mode,
        backend=point.backend,
        backend_version=point.backend_version,
        engine_step_backend=point.engine_step_backend,
        database_mode=point.database_mode,
        total_gpus=point.total_gpus,
        nextn=point.nextn,
        pareto_sweep=point.pareto_sweep,
        chunked_prefill=point.chunked_prefill,
        experiment=experiment,
        canonical_config_id=canonical_config_id,
        canonical_config_sort_key=canonical_config_sort_key,
        estimate_kind="theoretical_sol_roofline",
        attention_approximation=point.attention_approximation,
        attention_approximation_groups=(
            dict(STEP4_ATTENTION_APPROXIMATION_GROUPS)
            if point.attention_approximation == "temporary_mla_substitute"
            else {}
        ),
        approximation_dominated=point.approximation_dominated,
        replicas=allocation.replicas,
        total_gpus_used=allocation.total_gpus_used,
        unused_gpus=allocation.unused_gpus,
        **{"tokens/s/gpu_cluster": allocation.tokens_per_second_per_gpu_cluster},
        **ranking_metric_evidence,
        terminal_status=cap_result.terminal_status,
        saturated_cap_names=saturated_cap_names,
        cap_saturated=row_cap_saturated,
        ranking_eligible=cap_result.ranking_eligible and not row_cap_saturated and ttft_pass,
        ttft_pass=ttft_pass,
        tpot_observed_only=True,
        cap_rerun_count=cap_result.cap_rerun_count,
        agg_cap=cap_result.final_caps.agg,
        prefill_cap=cap_result.final_caps.prefill,
        decode_cap=cap_result.final_caps.decode,
        cap_history=tuple(
            {"agg": caps.agg, "prefill": caps.prefill, "decode": caps.decode} for caps in cap_result.cap_history
        ),
        **NEUTRAL_CORRECTIONS,
    )
    return normalized


def normalize_completed_experiment(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    task: Task,
    cap_result: CapSearchResult,
    system_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the final exact rerun and attach its strict-SOL operation evidence."""
    _validate_success_cap_result(cap_result)
    if not cap_result.attempt_evidence:
        raise RuntimeError("Successful cap result is missing attempt evidence")
    final_evidence = cap_result.attempt_evidence[-1]
    if final_evidence.experiment != experiment:
        raise ValueError(
            f"Final attempt experiment mismatch: expected {experiment!r}, got {final_evidence.experiment!r}"
        )
    if final_evidence.caps != cap_result.final_caps or final_evidence.status != "success":
        raise ValueError("Final attempt evidence must be successful and use final_caps")
    attempt = final_evidence.search_attempt
    if attempt is None or attempt.selected_evaluation is None:
        raise RuntimeError("Successful cap result is missing exact selected-point evaluation")

    evaluation = attempt.selected_evaluation
    if run_spec.serving_mode == "agg":
        exact_identity = _canonical_agg_sort_key(experiment, evaluation.row)
    elif run_spec.serving_mode == "disagg":
        exact_identity = _canonical_disagg_sort_key(experiment, evaluation.row, decode_cp=1)
    else:
        raise ValueError(f"Unsupported serving mode: {run_spec.serving_mode!r}")
    if attempt.selected_point_identity != exact_identity:
        raise ValueError(
            "Final selected-point identity mismatch: "
            f"expected {attempt.selected_point_identity!r}, got {exact_identity!r}"
        )

    per_ops_evidence = validate_per_ops_evidence(
        evaluation,
        serving_mode=run_spec.serving_mode,
        osl=run_spec.point.osl,
    )
    communication_evidence = _communication_evidence_payload(evaluation)
    if attempt.per_ops_evidence != per_ops_evidence:
        raise ValueError("Stored per-operation evidence does not match the exact selected-point evaluation")

    normalized = normalize_success_result(
        run_spec,
        experiment=experiment,
        task=task,
        raw_row=evaluation.row,
        cap_result=cap_result,
        system_spec=system_spec,
    )
    normalized.update(
        per_ops_data=copy.deepcopy(per_ops_evidence["per_ops_data"]),
        per_ops_source=copy.deepcopy(per_ops_evidence["per_ops_source"]),
        per_ops_phase_totals_ms=copy.deepcopy(per_ops_evidence["phase_totals_ms"]),
        per_ops_weighted_totals_ms=copy.deepcopy(per_ops_evidence["weighted_totals_ms"]),
        communication_evidence=copy.deepcopy(communication_evidence),
        effective_bandwidth_tiers=tuple(
            {
                key: record[key]
                for key in (
                    "operation_name",
                    "operation_kind",
                    "collective",
                    "group_size",
                    "tier",
                    "bandwidth_bytes_per_sec",
                )
            }
            for record in communication_evidence
        ),
    )
    return normalized


def _serialize_search_attempt(
    attempt: SearchAttempt,
    *,
    require_detailed_evidence: bool,
) -> dict[str, Any]:
    if not isinstance(attempt, SearchAttempt):
        raise TypeError(f"attempt must be SearchAttempt; got {type(attempt).__name__}")
    if not attempt.candidate_rows:
        raise ValueError("Successful search attempt must contain candidate rows")
    if attempt.rank1_row is None:
        raise ValueError("Successful search attempt must contain a rank-one row")
    if attempt.selected_point_identity is None:
        raise ValueError("Successful search attempt must contain a selected-point identity")
    if require_detailed_evidence:
        if attempt.selected_evaluation is None:
            raise ValueError("Final successful search attempt must contain an exact selected-point evaluation")
        if attempt.per_ops_evidence is None:
            raise ValueError("Final successful search attempt must contain per-operation evidence")
    elif attempt.selected_evaluation is not None or attempt.per_ops_evidence is not None:
        raise ValueError("Cap-saturated search attempt must not contain detailed evaluation evidence")
    return {
        **candidate_fingerprint(attempt.candidate_rows),
        "rank1_batch_sizes": _jsonable(attempt.rank1_batch_sizes),
        "rank1_row": _jsonable(attempt.rank1_row),
        "selected_point_identity": _jsonable(attempt.selected_point_identity),
        "selected_evaluation": _jsonable(attempt.selected_evaluation),
        "per_ops_evidence": _jsonable(attempt.per_ops_evidence),
    }


def serialize_cap_search_result(cap_result: CapSearchResult) -> dict[str, Any]:
    """Serialize one cap search without retaining its full candidate rows."""
    if not isinstance(cap_result, CapSearchResult):
        raise TypeError(f"cap_result must be CapSearchResult; got {type(cap_result).__name__}")
    if cap_result.terminal_status not in TERMINAL_STATUSES:
        raise ValueError(f"Unknown terminal status: {cap_result.terminal_status!r}")
    if cap_result.terminal_status == "success":
        _validate_success_cap_result(cap_result)
    elif cap_result.cap_saturated or cap_result.ranking_eligible:
        raise ValueError("Terminal cap result must be non-saturated and ranking-ineligible")
    if not cap_result.cap_history or cap_result.cap_history[-1] != cap_result.final_caps:
        raise ValueError("cap_history must be non-empty and end at final_caps")
    if cap_result.cap_rerun_count != len(cap_result.cap_history) - 1:
        raise ValueError("cap_rerun_count must equal len(cap_history) - 1")
    if tuple(evidence.caps for evidence in cap_result.attempt_evidence) != cap_result.cap_history:
        raise ValueError("attempt_evidence caps must exactly match cap_history")

    serialized_evidence = []
    for evidence in cap_result.attempt_evidence:
        if evidence.search_attempt is None:
            if evidence.status not in TERMINAL_STATUSES - {"success"}:
                raise ValueError(f"Missing search attempt for non-terminal status: {evidence.status!r}")
            if not evidence.error_type or evidence.error_message is None:
                raise ValueError("Terminal attempt evidence must contain an error type and message")
            search_attempt = None
        else:
            if evidence.status not in {"success", "cap_saturated"}:
                raise ValueError(f"Unexpected successful-attempt status: {evidence.status!r}")
            if evidence.error_type is not None or evidence.error_message is not None:
                raise ValueError("Successful attempt evidence must not contain terminal error fields")
            search_attempt = _serialize_search_attempt(
                evidence.search_attempt,
                require_detailed_evidence=evidence.status == "success",
            )
        serialized_evidence.append(
            {
                "experiment": evidence.experiment,
                "caps": _jsonable(evidence.caps),
                "status": evidence.status,
                "search_attempt": search_attempt,
                "error_type": evidence.error_type,
                "error_message": evidence.error_message,
            }
        )

    return {
        "terminal_status": cap_result.terminal_status,
        "final_caps": _jsonable(cap_result.final_caps),
        "cap_history": _jsonable(cap_result.cap_history),
        "cap_rerun_count": cap_result.cap_rerun_count,
        "cap_saturated": cap_result.cap_saturated,
        "ranking_eligible": cap_result.ranking_eligible,
        "attempt_evidence": serialized_evidence,
    }


def _require_exact_mapping_fields(
    value: Any,
    *,
    expected: set[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping; got {type(value).__name__}")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _deserialize_caps(value: Any, *, context: str) -> BatchCaps:
    mapping = _require_exact_mapping_fields(
        value,
        expected={"agg", "prefill", "decode"},
        context=context,
    )
    return BatchCaps(
        agg=mapping["agg"],
        prefill=mapping["prefill"],
        decode=mapping["decode"],
    )


def _validate_serialized_search_attempt(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    caps: BatchCaps,
    status: str,
    serialized: Any,
) -> tuple[str, ...]:
    attempt = _require_exact_mapping_fields(
        serialized,
        expected={
            "candidate_count",
            "candidate_sha256",
            "rank1_batch_sizes",
            "rank1_row",
            "selected_point_identity",
            "selected_evaluation",
            "per_ops_evidence",
        },
        context=f"serialized search attempt for {experiment}",
    )
    candidate_count = attempt["candidate_count"]
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError(f"candidate_count for {experiment} must be a positive integer")
    _require_sha256(attempt["candidate_sha256"], field=f"candidate_sha256 for {experiment}")
    rank1_batch_sizes = attempt["rank1_batch_sizes"]
    if not isinstance(rank1_batch_sizes, Mapping):
        raise TypeError(f"rank1_batch_sizes for {experiment} must be a mapping")
    saturated = _validate_rank1_batch_sizes(
        serving_mode=run_spec.serving_mode,
        caps=caps,
        rank1_batch_sizes=rank1_batch_sizes,
    )
    if status == "cap_saturated" and not saturated:
        raise ValueError(f"cap_saturated evidence for {experiment} has no saturated active cap")
    if status == "success" and saturated:
        raise ValueError(f"success evidence for {experiment} remains cap-saturated: {saturated}")

    rank1_row = attempt["rank1_row"]
    if not isinstance(rank1_row, Mapping):
        raise TypeError(f"rank1_row for {experiment} must be a mapping")
    if run_spec.serving_mode == "agg":
        if rank1_batch_sizes["agg"] != rank1_row.get("bs"):
            raise ValueError(f"aggregate rank-one batch mismatch for {experiment}")
        expected_identity = _canonical_agg_sort_key(experiment, rank1_row)
    else:
        if rank1_batch_sizes["prefill"] != rank1_row.get("(p)bs") or rank1_batch_sizes["decode"] != rank1_row.get(
            "(d)bs"
        ):
            raise ValueError(f"disaggregate rank-one batch mismatch for {experiment}")
        expected_identity = _canonical_disagg_sort_key(experiment, rank1_row, decode_cp=1)
    if attempt["selected_point_identity"] != _jsonable(expected_identity):
        raise ValueError(f"selected-point identity mismatch for {experiment}")

    evaluation = attempt["selected_evaluation"]
    per_ops_evidence = attempt["per_ops_evidence"]
    if status == "cap_saturated":
        if evaluation is not None or per_ops_evidence is not None:
            raise ValueError(f"cap-saturated attempt for {experiment} must not contain detailed evidence")
        return saturated
    if status != "success":
        raise ValueError(f"Unexpected serialized search-attempt status: {status!r}")

    evaluation_mapping = _require_exact_mapping_fields(
        evaluation,
        expected={"row", "per_ops_data", "per_ops_source", "communication_evidence"},
        context=f"selected evaluation for {experiment}",
    )
    if not isinstance(evaluation_mapping["row"], Mapping):
        raise TypeError(f"selected evaluation row for {experiment} must be a mapping")
    if run_spec.serving_mode == "agg":
        exact_identity = _canonical_agg_sort_key(experiment, evaluation_mapping["row"])
    else:
        exact_identity = _canonical_disagg_sort_key(experiment, evaluation_mapping["row"], decode_cp=1)
    if exact_identity != expected_identity:
        raise ValueError(f"selected evaluation identity mismatch for {experiment}")
    detailed = SinglePointEvaluation(
        row=dict(evaluation_mapping["row"]),
        per_ops_data=dict(evaluation_mapping["per_ops_data"]),
        per_ops_source=dict(evaluation_mapping["per_ops_source"]),
        communication_evidence=_deserialize_communication_evidence(
            evaluation_mapping["communication_evidence"],
            experiment=experiment,
        ),
    )
    expected_per_ops = validate_per_ops_evidence(
        detailed,
        serving_mode=run_spec.serving_mode,
        osl=run_spec.point.osl,
    )
    if per_ops_evidence != _jsonable(expected_per_ops):
        raise ValueError(f"per-operation evidence mismatch for {experiment}")
    return saturated


def _validate_serialized_cap_result(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    serialized: Any,
) -> str:
    result = _require_exact_mapping_fields(
        serialized,
        expected={
            "terminal_status",
            "final_caps",
            "cap_history",
            "cap_rerun_count",
            "cap_saturated",
            "ranking_eligible",
            "attempt_evidence",
        },
        context=f"serialized cap result for {experiment}",
    )
    terminal_status = result["terminal_status"]
    if terminal_status not in TERMINAL_STATUSES:
        raise ValueError(f"Unknown terminal status for {experiment}: {terminal_status!r}")
    final_caps = _deserialize_caps(result["final_caps"], context=f"final_caps for {experiment}")
    history_raw = result["cap_history"]
    if not isinstance(history_raw, list) or not history_raw:
        raise ValueError(f"cap_history for {experiment} must be a non-empty list")
    history = tuple(
        _deserialize_caps(value, context=f"cap_history[{index}] for {experiment}")
        for index, value in enumerate(history_raw)
    )
    if history[-1] != final_caps:
        raise ValueError(f"cap_history for {experiment} must end at final_caps")
    rerun_count = result["cap_rerun_count"]
    if type(rerun_count) is not int or rerun_count != len(history) - 1:
        raise ValueError(f"cap_rerun_count for {experiment} must equal len(cap_history) - 1")
    if type(result["cap_saturated"]) is not bool or result["cap_saturated"] is not False:
        raise ValueError(f"terminal cap result for {experiment} must be non-saturated")
    expected_ranking_eligible = terminal_status == "success"
    if result["ranking_eligible"] is not expected_ranking_eligible:
        raise ValueError(f"ranking_eligible contradicts terminal status for {experiment}")

    evidence_raw = result["attempt_evidence"]
    if not isinstance(evidence_raw, list) or len(evidence_raw) != len(history):
        raise ValueError(f"attempt_evidence for {experiment} must exactly match cap_history")
    for index, (serialized_evidence, caps) in enumerate(zip(evidence_raw, history, strict=True)):
        evidence = _require_exact_mapping_fields(
            serialized_evidence,
            expected={"experiment", "caps", "status", "search_attempt", "error_type", "error_message"},
            context=f"attempt_evidence[{index}] for {experiment}",
        )
        if evidence["experiment"] != experiment:
            raise ValueError(f"attempt experiment mismatch for {experiment}: got {evidence['experiment']!r}")
        if _deserialize_caps(evidence["caps"], context=f"attempt caps for {experiment}") != caps:
            raise ValueError(f"attempt caps mismatch for {experiment} at index {index}")
        final_attempt = index == len(history) - 1
        expected_status = terminal_status if final_attempt else "cap_saturated"
        if evidence["status"] != expected_status:
            raise ValueError(
                f"attempt status mismatch for {experiment} at index {index}: "
                f"expected {expected_status!r}, got {evidence['status']!r}"
            )
        if expected_status in {"cap_saturated", "success"}:
            if evidence["error_type"] is not None or evidence["error_message"] is not None:
                raise ValueError(f"non-terminal-error evidence for {experiment} must not contain error fields")
            saturated = _validate_serialized_search_attempt(
                run_spec,
                experiment=experiment,
                caps=caps,
                status=expected_status,
                serialized=evidence["search_attempt"],
            )
            if expected_status == "cap_saturated":
                expected_next = replace(caps, **{name: getattr(caps, name) * 2 for name in saturated})
                if history[index + 1] != expected_next:
                    raise ValueError(f"cap history expansion mismatch for {experiment} at index {index}")
        else:
            if evidence["search_attempt"] is not None:
                raise ValueError(f"terminal error evidence for {experiment} must not contain a search attempt")
            if not isinstance(evidence["error_type"], str) or not evidence["error_type"]:
                raise ValueError(f"terminal error evidence for {experiment} must contain an error type")
            if not isinstance(evidence["error_message"], str):
                raise ValueError(f"terminal error evidence for {experiment} must contain an error message")
    return terminal_status


def _normalized_row_identity(run_spec: ModeRunSpec) -> dict[str, Any]:
    point = run_spec.point
    return {
        "model": point.model,
        "system": point.system,
        "workload_kind": point.workload_kind,
        "isl": point.isl,
        "osl": point.osl,
        "prefix": point.prefix,
        "ttft_sla_ms": point.ttft_sla_ms,
        "tpot_constraint_ms": point.tpot_ms,
        "serving_mode": run_spec.serving_mode,
        "backend": point.backend,
        "backend_version": point.backend_version,
        "engine_step_backend": point.engine_step_backend,
        "database_mode": point.database_mode,
        "total_gpus": point.total_gpus,
        "nextn": point.nextn,
        "pareto_sweep": point.pareto_sweep,
        "chunked_prefill": point.chunked_prefill,
        "attention_approximation": point.attention_approximation,
        "approximation_dominated": point.approximation_dominated,
    }


def _rebuild_normalized_success_row(
    run_spec: ModeRunSpec,
    *,
    experiment: str,
    serialized_cap_result: Mapping[str, Any],
) -> dict[str, Any]:
    final_caps = _deserialize_caps(
        serialized_cap_result["final_caps"],
        context=f"final_caps for {experiment}",
    )
    cap_history = tuple(
        _deserialize_caps(value, context=f"cap_history[{index}] for {experiment}")
        for index, value in enumerate(serialized_cap_result["cap_history"])
    )
    cap_result = CapSearchResult(
        terminal_status="success",
        final_caps=final_caps,
        cap_history=cap_history,
        cap_rerun_count=serialized_cap_result["cap_rerun_count"],
        cap_saturated=False,
        ranking_eligible=True,
    )
    final_attempt = serialized_cap_result["attempt_evidence"][-1]["search_attempt"]
    evaluation = final_attempt["selected_evaluation"]
    per_ops = final_attempt["per_ops_evidence"]
    communication_records = _deserialize_communication_evidence(
        evaluation["communication_evidence"],
        experiment=experiment,
    )
    communication_evidence = _communication_evidence_payload(
        SinglePointEvaluation(
            row={},
            per_ops_data={},
            per_ops_source={},
            communication_evidence=communication_records,
        )
    )
    task = build_comparison_task(run_spec, experiment=experiment, caps=final_caps)
    system_spec = perf_database.load_system_spec(run_spec.point.system)
    if not isinstance(system_spec, Mapping) or not system_spec:
        raise ValueError(f"System spec for {run_spec.point.system!r} must be available during durable validation")
    normalized = normalize_success_result(
        run_spec,
        experiment=experiment,
        task=task,
        raw_row=evaluation["row"],
        cap_result=cap_result,
        system_spec=system_spec,
    )
    normalized.update(
        per_ops_data=copy.deepcopy(per_ops["per_ops_data"]),
        per_ops_source=copy.deepcopy(per_ops["per_ops_source"]),
        per_ops_phase_totals_ms=copy.deepcopy(per_ops["phase_totals_ms"]),
        per_ops_weighted_totals_ms=copy.deepcopy(per_ops["weighted_totals_ms"]),
        communication_evidence=copy.deepcopy(communication_evidence),
        effective_bandwidth_tiers=tuple(
            {
                key: record[key]
                for key in (
                    "operation_name",
                    "operation_kind",
                    "collective",
                    "group_size",
                    "tier",
                    "bandwidth_bytes_per_sec",
                )
            }
            for record in communication_evidence
        ),
    )
    return _jsonable(normalized)


def validate_serialized_mode_run(
    run_spec: ModeRunSpec,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one durable mode-run payload against its external matrix spec."""
    if not isinstance(run_spec, ModeRunSpec):
        raise TypeError(f"run_spec must be ModeRunSpec; got {type(run_spec).__name__}")
    validated = _require_exact_mapping_fields(
        record,
        expected={"record_type", "mode_run_key", "mode_run_identity", "normalized_rows", "experiments"},
        context="serialized mode run",
    )
    if validated["record_type"] != "mode_run":
        raise ValueError("serialized mode run must have record_type='mode_run'")
    expected_key = mode_run_key(run_spec)
    if validated["mode_run_key"] != expected_key:
        raise ValueError("stored mode-run key mismatch")
    if validated["mode_run_identity"] != mode_run_identity(run_spec):
        raise ValueError("stored mode-run identity mismatch")

    expected_experiments = _mode_experiments(run_spec.serving_mode)
    experiments = validated["experiments"]
    if not isinstance(experiments, Mapping):
        raise TypeError("serialized mode-run experiments must be a mapping")
    if set(experiments) != set(expected_experiments):
        raise ValueError(f"experiment set mismatch: expected {sorted(expected_experiments)}, got {sorted(experiments)}")
    terminal_statuses = {
        experiment: _validate_serialized_cap_result(
            run_spec,
            experiment=experiment,
            serialized=experiments[experiment],
        )
        for experiment in expected_experiments
    }
    successful_experiments = tuple(
        experiment for experiment in expected_experiments if terminal_statuses[experiment] == "success"
    )

    rows = validated["normalized_rows"]
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("normalized_rows must be a list of mappings")
    row_experiments = tuple(row.get("experiment") for row in rows)
    if row_experiments != successful_experiments:
        raise ValueError(
            f"normalized success rows mismatch: expected {list(successful_experiments)}, got {list(row_experiments)}"
        )
    expected_identity = _normalized_row_identity(run_spec)
    for row in rows:
        experiment = row["experiment"]
        if row.get("terminal_status") != "success":
            raise ValueError(f"normalized row for {experiment} must have terminal_status='success'")
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"normalized row/run identity mismatch for {experiment}: "
                    f"{field}={row.get(field)!r}, expected={expected!r}"
                )
        expected_row = _rebuild_normalized_success_row(
            run_spec,
            experiment=experiment,
            serialized_cap_result=experiments[experiment],
        )
        actual_row = _jsonable(row)
        actual_fields = set(actual_row)
        expected_fields = set(expected_row)
        if actual_fields != expected_fields:
            raise ValueError(
                f"normalized row fields mismatch for {experiment}: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"extra={sorted(actual_fields - expected_fields)}"
            )
        mismatched_fields = sorted(field for field in expected_fields if actual_row[field] != expected_row[field])
        if mismatched_fields:
            raise ValueError(f"normalized row evidence mismatch for {experiment}: fields={mismatched_fields}")
    return _jsonable(validated)


def _mode_experiments(serving_mode: str) -> tuple[str, ...]:
    experiments = tuple(
        experiment
        for experiment, (mode, _prefill_pattern, _decode_pattern) in EXPERIMENT_PATTERNS.items()
        if mode == serving_mode
    )
    if not experiments:
        raise ValueError(f"Unsupported serving mode: {serving_mode!r}")
    return experiments


def build_mode_run_record(
    run_spec: ModeRunSpec,
    *,
    cap_results: Mapping[str, CapSearchResult],
    normalized_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one complete checkpoint record for an aggregate or disaggregate run."""
    if not isinstance(run_spec, ModeRunSpec):
        raise TypeError(f"run_spec must be ModeRunSpec; got {type(run_spec).__name__}")
    if not isinstance(cap_results, Mapping):
        raise TypeError(f"cap_results must be a mapping; got {type(cap_results).__name__}")
    expected_experiments = _mode_experiments(run_spec.serving_mode)
    if set(cap_results) != set(expected_experiments):
        raise ValueError(f"experiment set mismatch: expected {sorted(expected_experiments)}, got {sorted(cap_results)}")
    for experiment in expected_experiments:
        cap_result = cap_results[experiment]
        if not isinstance(cap_result, CapSearchResult):
            raise TypeError(f"cap result for {experiment!r} must be CapSearchResult; got {type(cap_result).__name__}")
        evidence_experiments = {evidence.experiment for evidence in cap_result.attempt_evidence}
        if evidence_experiments != {experiment}:
            raise ValueError(
                f"attempt experiment mismatch for {experiment!r}: got {sorted(evidence_experiments, key=str)}"
            )

    rows = tuple(normalized_rows)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("Every normalized row must be a mapping")
    successful_experiments = tuple(
        experiment for experiment in expected_experiments if cap_results[experiment].terminal_status == "success"
    )
    normalized_experiments = tuple(row.get("experiment") for row in rows)
    if sorted(normalized_experiments) != sorted(successful_experiments):
        raise ValueError(
            "normalized success rows mismatch: "
            f"expected {sorted(successful_experiments)}, got {sorted(normalized_experiments)}"
        )

    record = {
        "record_type": "mode_run",
        "mode_run_key": mode_run_key(run_spec),
        "mode_run_identity": mode_run_identity(run_spec),
        "normalized_rows": _jsonable(rows),
        "experiments": {
            experiment: serialize_cap_search_result(cap_results[experiment]) for experiment in expected_experiments
        },
    }
    return validate_serialized_mode_run(run_spec, record)


def execute_mode_run(
    run_spec: ModeRunSpec,
    *,
    system_spec: Mapping[str, Any],
    initial_caps: BatchCaps | None = None,
    task_factory: Callable[..., Task] = build_comparison_task,
) -> dict[str, Any]:
    """Execute, exactly normalize, and serialize one resumable mode run."""
    if not isinstance(system_spec, Mapping):
        raise TypeError(f"system_spec must be a mapping; got {type(system_spec).__name__}")
    cap_results = run_all_experiment_cap_searches(
        run_spec,
        initial_caps=initial_caps,
        task_factory=task_factory,
    )
    expected_experiments = _mode_experiments(run_spec.serving_mode)
    if set(cap_results) != set(expected_experiments):
        raise ValueError(f"experiment set mismatch: expected {sorted(expected_experiments)}, got {sorted(cap_results)}")
    normalized_rows = []
    for experiment in expected_experiments:
        cap_result = cap_results[experiment]
        if cap_result.terminal_status != "success":
            continue
        task = task_factory(run_spec, experiment=experiment, caps=cap_result.final_caps)
        normalized_rows.append(
            normalize_completed_experiment(
                run_spec,
                experiment=experiment,
                task=task,
                cap_result=cap_result,
                system_spec=system_spec,
            )
        )
    return build_mode_run_record(
        run_spec,
        cap_results=cap_results,
        normalized_rows=normalized_rows,
    )


def _validated_canonical_sort_key(value: Any) -> tuple[str | int, ...]:
    if not isinstance(value, tuple | list) or not value or not isinstance(value[0], str) or not value[0]:
        raise TypeError(f"canonical_config_sort_key must be a non-empty typed sequence; got {value!r}")
    if any(type(item) is not int or item < 1 for item in value[1:]):
        raise TypeError(
            f"canonical_config_sort_key fields after the experiment name must be positive integers; got {value!r}"
        )
    return tuple(value)


def rank_final_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Filter final evidence and rank by throughput with identity-only ties."""
    required = (
        *RANK_GROUP_FIELDS,
        "canonical_config_id",
        "canonical_config_sort_key",
        "tokens/s/gpu_cluster",
        "ranking_metric_kind",
        "ranking_metric_value",
        "terminal_status",
        "cap_saturated",
        "ttft",
        "ttft_pass",
        "ranking_eligible",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for raw_row in rows:
        _require_row_fields(raw_row, required)
        row = dict(raw_row)
        terminal_status = row["terminal_status"]
        if terminal_status not in TERMINAL_STATUSES:
            raise ValueError(f"Unknown terminal status: {terminal_status!r}")
        config_id = row["canonical_config_id"]
        if not isinstance(config_id, str) or not config_id:
            raise ValueError(f"canonical_config_id must be a non-empty string; got {config_id!r}")
        row["canonical_config_sort_key"] = _validated_canonical_sort_key(row["canonical_config_sort_key"])
        throughput = _finite_number(row["tokens/s/gpu_cluster"], field="tokens/s/gpu_cluster")
        row["tokens/s/gpu_cluster"] = throughput
        ranking_metric_kind = row["ranking_metric_kind"]
        expected_metric_kind = (
            "prefill_input_throughput"
            if row["workload_kind"] == "primary" and row["osl"] == 1
            else "output_token_throughput"
        )
        if ranking_metric_kind != expected_metric_kind:
            raise ValueError(
                f"ranking_metric_kind mismatch: expected {expected_metric_kind!r}, got {ranking_metric_kind!r}"
            )
        ranking_metric_value = _finite_number(row["ranking_metric_value"], field="ranking_metric_value")
        row["ranking_metric_value"] = ranking_metric_value
        ttft = _finite_number(row["ttft"], field="ttft")
        ttft_sla_ms = _finite_number(row["ttft_sla_ms"], field="ttft_sla_ms")
        serving_mode = row["serving_mode"]
        if serving_mode == "agg":
            expected_ttft_pass = ttft <= ttft_sla_ms
        elif serving_mode == "disagg":
            expected_ttft_pass = ttft < ttft_sla_ms
        else:
            raise ValueError(f"Unknown serving mode: {serving_mode!r}")
        if row["ttft_pass"] is not expected_ttft_pass:
            raise ValueError(
                "ttft_pass contradicts "
                f"{serving_mode} boundary: ttft={ttft}, sla={ttft_sla_ms}, "
                f"ttft_pass={row['ttft_pass']!r}"
            )

        if (
            terminal_status != "success"
            or row["cap_saturated"] is not False
            or row["ttft_pass"] is not True
            or row["ranking_eligible"] is not True
        ):
            continue
        group_key = tuple(row[field] for field in RANK_GROUP_FIELDS)
        groups[group_key].append(row)

    ranked: list[dict[str, Any]] = []
    for group_key in sorted(groups, key=repr):
        group = sorted(
            groups[group_key],
            key=lambda row: (-row["ranking_metric_value"], row["canonical_config_sort_key"]),
        )
        for rank, row in enumerate(group, start=1):
            ranked.append({**row, "rank": rank})
    return ranked


def build_model_comparisons(
    ranked_rows: Iterable[Mapping[str, Any]],
    *,
    metrics: tuple[str, ...],
) -> tuple[ModelComparison, ...]:
    """Compare aligned model-level rank-one rows with DeepSeek as baseline."""
    if not metrics:
        raise ValueError("At least one comparison metric is required")
    unknown_metrics = [metric for metric in metrics if metric not in METRIC_POLARITY]
    if unknown_metrics:
        raise ValueError(f"Missing metric-polarity contract for: {unknown_metrics}")

    rank_one: dict[tuple[tuple[Any, ...], str], Mapping[str, Any]] = {}
    aligned_keys: set[tuple[Any, ...]] = set()
    for row in ranked_rows:
        _require_row_fields(
            row,
            (*RANK_GROUP_FIELDS, "canonical_config_id", "rank", *metrics),
        )
        if row["rank"] != 1:
            continue
        model = row["model"]
        if model not in MODELS:
            raise ValueError(f"Unexpected comparison model: {model!r}")
        aligned_key = tuple(row[field] for field in COMPARISON_KEY_FIELDS)
        evidence_key = (aligned_key, model)
        if evidence_key in rank_one:
            raise ValueError(f"Duplicate rank-one evidence for model={model!r}, key={aligned_key!r}")
        rank_one[evidence_key] = row
        aligned_keys.add(aligned_key)

    comparisons = []
    for aligned_key in sorted(aligned_keys, key=repr):
        step4 = rank_one.get((aligned_key, "stepfun-ai/Step4"))
        deepseek = rank_one.get((aligned_key, "deepseek-ai/DeepSeek-V4-Pro"))
        step4_config_id = None if step4 is None else str(step4["canonical_config_id"])
        deepseek_config_id = None if deepseek is None else str(deepseek["canonical_config_id"])
        if step4 is None or deepseek is None:
            comparisons.append(
                ModelComparison(
                    aligned_key=aligned_key,
                    status="unpaired",
                    step4_config_id=step4_config_id,
                    deepseek_config_id=deepseek_config_id,
                    metric_deltas={},
                )
            )
            continue

        metric_deltas = {}
        for metric in metrics:
            step4_value = _finite_number(step4[metric], field=f"Step4 {metric}")
            deepseek_value = _finite_number(deepseek[metric], field=f"DeepSeek-V4-Pro {metric}")
            if deepseek_value == 0:
                if metric != "tpot" or step4_value != 0:
                    raise ValueError(f"Cannot compute {metric}: zero DeepSeek-V4-Pro baseline")
                relative_delta = None
                status = "zero_baseline_both_zero"
            else:
                relative_delta = (step4_value - deepseek_value) / deepseek_value
                status = "computed"
            absolute_delta = step4_value - deepseek_value
            metric_deltas[metric] = MetricDelta(
                step4_value=step4_value,
                deepseek_value=deepseek_value,
                absolute_delta=absolute_delta,
                relative_delta=relative_delta,
                polarity=METRIC_POLARITY[metric],
                status=status,
            )
        comparisons.append(
            ModelComparison(
                aligned_key=aligned_key,
                status="paired",
                step4_config_id=step4_config_id,
                deepseek_config_id=deepseek_config_id,
                metric_deltas=metric_deltas,
            )
        )
    return tuple(comparisons)


def select_mode_run_specs(
    run_specs: Iterable[ModeRunSpec],
    *,
    models: tuple[str, ...] | None = None,
    systems: tuple[str, ...] | None = None,
    workload_kinds: tuple[str, ...] | None = None,
    isls: tuple[int, ...] | None = None,
    ttft_sla_ms: tuple[int, ...] | None = None,
    serving_modes: tuple[str, ...] | None = None,
) -> tuple[ModeRunSpec, ...]:
    """Select an exact deterministic subset for smoke or full-matrix execution."""
    selected = tuple(
        run_spec
        for run_spec in run_specs
        if (models is None or run_spec.point.model in models)
        and (systems is None or run_spec.point.system in systems)
        and (workload_kinds is None or run_spec.point.workload_kind in workload_kinds)
        and (isls is None or run_spec.point.isl in isls)
        and (ttft_sla_ms is None or run_spec.point.ttft_sla_ms in ttft_sla_ms)
        and (serving_modes is None or run_spec.serving_mode in serving_modes)
    )
    if not selected:
        raise ValueError("No mode runs match the requested filters")
    return selected


def execute_matrix_runs(
    run_specs: Iterable[ModeRunSpec],
    *,
    checkpoint_path: str | Path,
    execution_contract_sha256: str,
    git_head: str,
    resume: bool,
    initial_caps: BatchCaps,
    system_loader: Callable[[str], Mapping[str, Any]] = perf_database.load_system_spec,
    executor: Callable[..., dict[str, Any]] = execute_mode_run,
) -> dict[str, dict[str, Any]]:
    """Execute missing mode runs and append each completed record durably."""
    specs = tuple(run_specs)
    header = build_checkpoint_header(
        specs,
        execution_contract_sha256=execution_contract_sha256,
        git_head=git_head,
    )
    checkpoint = Path(checkpoint_path)
    if resume:
        _, records = load_checkpoint(checkpoint, expected_header=header, run_specs=specs)
    else:
        initialize_checkpoint(checkpoint, header)
        records = {}

    expected_keys = {mode_run_key(run_spec) for run_spec in specs}
    unexpected_keys = set(records) - expected_keys
    if unexpected_keys:
        raise ValueError(f"Checkpoint contains mode runs outside the selected matrix: {sorted(unexpected_keys)}")

    for index, run_spec in enumerate(specs, start=1):
        key = mode_run_key(run_spec)
        if key in records:
            continue
        print(
            f"[{index}/{len(specs)}] {run_spec.point.model} {run_spec.point.system} "
            f"{run_spec.point.workload_kind} isl={run_spec.point.isl} "
            f"ttft={run_spec.point.ttft_sla_ms} mode={run_spec.serving_mode}",
            flush=True,
        )
        system_spec = system_loader(run_spec.point.system)
        record = executor(
            run_spec,
            system_spec=system_spec,
            initial_caps=initial_caps,
        )
        if not isinstance(record, dict):
            raise TypeError(f"executor must return a mode-run dictionary; got {type(record).__name__}")
        validated = validate_serialized_mode_run(run_spec, record)
        commit_checkpoint_record(checkpoint, run_spec, validated)
        records[key] = validated
    return records


def merge_completed_checkpoints(
    run_specs: Iterable[ModeRunSpec],
    *,
    shards: Iterable[tuple[Iterable[ModeRunSpec], str | Path]],
    output_checkpoint_path: str | Path,
    execution_contract_sha256: str,
    git_head: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate complete disjoint shards and write one full ordered checkpoint."""
    specs = tuple(run_specs)
    full_header = build_checkpoint_header(
        specs,
        execution_contract_sha256=execution_contract_sha256,
        git_head=git_head,
    )
    expected_keys = tuple(mode_run_key(run_spec) for run_spec in specs)
    expected_key_set = set(expected_keys)
    merged_records: dict[str, dict[str, Any]] = {}
    for shard_specs_iterable, checkpoint_path in shards:
        shard_specs = tuple(shard_specs_iterable)
        shard_header = build_checkpoint_header(
            shard_specs,
            execution_contract_sha256=execution_contract_sha256,
            git_head=git_head,
        )
        _, shard_records = load_checkpoint(
            checkpoint_path,
            expected_header=shard_header,
            run_specs=shard_specs,
        )
        shard_keys = {mode_run_key(run_spec) for run_spec in shard_specs}
        if set(shard_records) != shard_keys:
            missing = sorted(shard_keys - set(shard_records))
            unexpected = sorted(set(shard_records) - shard_keys)
            raise ValueError(f"shard checkpoint is incomplete: missing={missing}, unexpected={unexpected}")
        overlap = set(merged_records) & set(shard_records)
        if overlap:
            raise ValueError(f"shard checkpoints contain overlapping mode runs: {sorted(overlap)}")
        merged_records.update(shard_records)

    if set(merged_records) != expected_key_set:
        missing = sorted(expected_key_set - set(merged_records))
        unexpected = sorted(set(merged_records) - expected_key_set)
        raise ValueError(f"merged shard set does not match full matrix: missing={missing}, unexpected={unexpected}")

    ordered_records = {key: merged_records[key] for key in expected_keys}
    initialize_checkpoint(output_checkpoint_path, full_header)
    validated_records = tuple(
        validate_serialized_mode_run(run_spec, ordered_records[mode_run_key(run_spec)]) for run_spec in specs
    )
    with _open_checkpoint(output_checkpoint_path) as connection:
        for record in validated_records:
            _insert_mode_run(connection, record)
    return full_header, ordered_records


def _count_experiment_terminals(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        experiments = record.get("experiments")
        if not isinstance(experiments, Mapping):
            raise TypeError("mode-run experiments must be a mapping")
        for experiment in experiments.values():
            if not isinstance(experiment, Mapping):
                raise TypeError("serialized experiment evidence must be a mapping")
            status = experiment.get("terminal_status")
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"Unknown terminal status: {status!r}")
            counts[str(status)] += 1
    return dict(sorted(counts.items()))


def finalize_matrix_results(
    run_specs: Iterable[ModeRunSpec],
    *,
    header: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the strict final artifact after every selected mode run completes."""
    specs = tuple(run_specs)
    if not isinstance(header, Mapping):
        raise TypeError(f"header must be a mapping; got {type(header).__name__}")
    expected_header = build_checkpoint_header(
        specs,
        execution_contract_sha256=str(header.get("execution_contract_sha256", "")),
        git_head=str(header.get("git_head", "")),
    )
    _validate_checkpoint_header(dict(header), expected_header=expected_header)

    expected_keys = tuple(mode_run_key(run_spec) for run_spec in specs)
    if set(records) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(records))
        unexpected = sorted(set(records) - set(expected_keys))
        raise ValueError(f"incomplete mode-run set: missing={missing}, unexpected={unexpected}")

    ordered_records = []
    normalized_rows = []
    for run_spec, key in zip(specs, expected_keys, strict=True):
        record = records[key]
        if not isinstance(record, Mapping):
            raise TypeError(f"mode-run record must be a mapping; got {type(record).__name__}")
        validated = validate_serialized_mode_run(run_spec, record)
        rows = validated["normalized_rows"]
        ordered_records.append(validated)
        normalized_rows.extend(dict(row) for row in rows)

    ranked_rows = rank_final_rows(normalized_rows)
    comparison_metrics = (
        "ranking_metric_value",
        "ttft",
        "tpot",
        "request_latency",
    )
    comparisons = build_model_comparisons(ranked_rows, metrics=comparison_metrics)
    serialized_comparisons = _jsonable(comparisons)
    paired_count = sum(comparison["status"] == "paired" for comparison in serialized_comparisons)
    unpaired_count = sum(comparison["status"] == "unpaired" for comparison in serialized_comparisons)
    return _jsonable(
        {
            "checkpoint_header": dict(header),
            "engine_step_backend": ENGINE_STEP_BACKEND,
            "summary": {
                "mode_run_count": len(specs),
                "normalized_row_count": len(normalized_rows),
                "ranked_row_count": len(ranked_rows),
                "paired_comparison_count": paired_count,
                "unpaired_comparison_count": unpaired_count,
                "experiment_terminal_counts": _count_experiment_terminals(ordered_records),
            },
            "ranking_contract": RANKING_CONTRACT,
            "delta_contract": DELTA_CONTRACT,
            "comparison_metrics": comparison_metrics,
            "mode_runs": ordered_records,
            "normalized_rows": normalized_rows,
            "ranked_rows": ranked_rows,
            "comparisons": serialized_comparisons,
        }
    )


def _csv_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _canonical_json(value)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = tuple(rows)
    fieldnames = tuple(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _render_markdown_report(artifact: Mapping[str, Any]) -> str:
    summary = artifact["summary"]
    lines = [
        "# Step4 vs DeepSeek-V4-Pro SOL Comparison",
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
        "Temporary MLA substitution is used for all 92 Step4 attention layers. "
        "The original labels remain 23 Full-MLA-approx and 69 SWA-MLA-approx; "
        "rows at ISL >= 65536 are approximation dominated.",
        "",
        "All operation latency evidence in this comparison is required to use DatabaseMode.SOL. "
        "Primary OSL=1 rows rank by fixed-cluster Prefill input throughput; the 4K/1024 decode smoke "
        "ranks by fixed-cluster output throughput.",
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
            "Absolute delta is Step4 minus DeepSeek-V4-Pro. TPOT is observational and does not affect eligibility.",
            "",
            "| Metric | Step4 | DeepSeek-V4-Pro | Absolute delta | Relative delta | Polarity | Status |",
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
                f"| {metric} | {delta['step4_value']} | {delta['deepseek_value']} | "
                f"{delta['absolute_delta']} | {relative_delta} | {delta['polarity']} | {delta['status']} |"
            )
            paired_metric_count += 1
    if paired_metric_count == 0:
        lines.append("| None | 0 | 0 | 0 | 0 | None | None |")
    lines.append("")
    return "\n".join(lines)


def write_final_artifacts(output_dir: str | Path, artifact: Mapping[str, Any]) -> dict[str, Path]:
    """Write strict JSON plus tabular and human-readable comparison artifacts."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": directory / "results.json",
        "ranked_csv": directory / "ranked_rows.csv",
        "comparisons_csv": directory / "model_comparisons.csv",
        "markdown": directory / "report.md",
    }
    paths["json"].write_text(_canonical_json(artifact) + "\n", encoding="utf-8")
    _write_csv(paths["ranked_csv"], artifact["ranked_rows"])
    _write_csv(paths["comparisons_csv"], artifact["comparisons"])
    paths["markdown"].write_text(_render_markdown_report(artifact), encoding="utf-8")
    return paths


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
    parser.add_argument("--workload-kind", action="append", choices=("primary", "decode_smoke"))
    parser.add_argument("--isl", action="append", type=_positive_int)
    parser.add_argument("--ttft-sla-ms", action="append", type=_positive_int)
    parser.add_argument("--serving-mode", action="append", choices=SERVING_MODES)
    parser.add_argument("--initial-agg-cap", type=_positive_int, default=1024)
    parser.add_argument("--initial-prefill-cap", type=_positive_int, default=16)
    parser.add_argument("--initial-decode-cap", type=_positive_int, default=1024)
    return parser.parse_args(argv)


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    if not head:
        raise RuntimeError("git rev-parse HEAD returned an empty value")
    return head


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    perf_database.set_systems_paths(args.systems_paths)
    run_specs = select_mode_run_specs(
        build_mode_run_specs(),
        models=None if args.model is None else tuple(args.model),
        systems=None if args.system is None else tuple(args.system),
        workload_kinds=None if args.workload_kind is None else tuple(args.workload_kind),
        isls=None if args.isl is None else tuple(args.isl),
        ttft_sla_ms=None if args.ttft_sla_ms is None else tuple(args.ttft_sla_ms),
        serving_modes=None if args.serving_mode is None else tuple(args.serving_mode),
    )
    head = _git_head()
    initial_caps = BatchCaps(
        agg=args.initial_agg_cap,
        prefill=args.initial_prefill_cap,
        decode=args.initial_decode_cap,
    )
    contract = build_execution_contract(run_specs, initial_caps=initial_caps)
    contract_sha256 = execution_contract_sha256(contract)
    checkpoint_path = args.output_dir / "mode_runs.sqlite3"
    records = execute_matrix_runs(
        run_specs,
        checkpoint_path=checkpoint_path,
        execution_contract_sha256=contract_sha256,
        git_head=head,
        resume=args.resume,
        initial_caps=initial_caps,
    )
    header, loaded_records = load_checkpoint(
        checkpoint_path,
        expected_header=build_checkpoint_header(
            run_specs,
            execution_contract_sha256=contract_sha256,
            git_head=head,
        ),
        run_specs=run_specs,
    )
    if records != loaded_records:
        raise ValueError("in-memory and durable checkpoint records differ")
    artifact = finalize_matrix_results(run_specs, header=header, records=loaded_records)
    paths = write_final_artifacts(args.output_dir, artifact)
    print(
        f"completed {len(run_specs)} mode runs; results={paths['json']} report={paths['markdown']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
