"""Build auditable Step4 attention collection manifests from real run artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from tests.performance.step4_collection_manifest import validate_collection_manifest

BACKEND = "vllm"
VERSION = "0.19.0"
SYSTEM = "h800_sxm"
DEVICE = "NVIDIA H800"
KERNEL_SOURCE = "vllm_flash_attn"
ATTN_DTYPE = "bfloat16"
KV_CACHE_DTYPE = "fp8"


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside task_root") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {path}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_fingerprint(row: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def _parse_invocation(invocation_id: str, attention_phase: str) -> tuple[tuple[int, ...], dict[str, int]]:
    prefix = f"vllm.attention_{attention_phase}:run_attention_torch:"
    if not invocation_id.startswith(prefix):
        raise ValueError(f"unexpected attention invocation prefix: {invocation_id}")
    try:
        args = ast.literal_eval(invocation_id[len(prefix) :])
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"unable to parse attention invocation: {invocation_id}") from exc
    if not isinstance(args, list) or len(args) != 8:
        raise ValueError(f"attention invocation must contain eight arguments: {invocation_id}")
    batch_size, sequence_arg, heads, kv_heads, head_dim, kv_fp8, is_context, window_size = args
    if kv_fp8 is not True:
        raise ValueError(f"attention invocation must use FP8 KV cache: {invocation_id}")
    expected_context = attention_phase == "context"
    if is_context is not expected_context:
        raise ValueError(f"attention invocation phase flag mismatch: {invocation_id}")
    numeric = [batch_size, sequence_arg, heads, kv_heads, head_dim, window_size]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in numeric):
        raise ValueError(f"attention invocation shape arguments must be integers: {invocation_id}")
    if any(value <= 0 for value in numeric[:-1]) or window_size < 0:
        raise ValueError(f"attention invocation shape arguments are out of range: {invocation_id}")
    if attention_phase == "context":
        isl, step = sequence_arg, 0
    else:
        isl, step = 1, sequence_arg
    workload = {"batch_size": batch_size, "isl": isl, "step": step}
    key = (batch_size, isl, step, heads, kv_heads, head_dim, window_size)
    return key, workload


def _validate_row(row: Mapping[str, object], attention_phase: str) -> tuple[int, ...]:
    expected_op = f"{attention_phase}_attention"
    expected = {
        "framework": "VLLM",
        "version": VERSION,
        "device": DEVICE,
        "op_name": expected_op,
        "kernel_source": KERNEL_SOURCE,
        "beam_width": 1,
        "attn_dtype": ATTN_DTYPE,
        "kv_cache_dtype": KV_CACHE_DTYPE,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"attention parquet {field} must equal {value!r}; got {row.get(field)!r}")
    fields = ("batch_size", "isl", "step", "num_heads", "num_key_value_heads", "head_dim", "window_size")
    values = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"attention parquet {field} must be an integer")
        values.append(value)
    if attention_phase == "context" and row["step"] != 0:
        raise ValueError("context attention parquet step must equal zero")
    if attention_phase == "generation" and row["isl"] != 1:
        raise ValueError("generation attention parquet isl must equal one")
    latency = row.get("latency")
    if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
        raise ValueError("attention parquet latency must be positive")
    return tuple(values)


def _validate_csv_parquet_rows(csv_path: Path, parquet_rows: list[dict[str, object]]) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(parquet_rows):
        raise ValueError("attention CSV and parquet row counts differ")
    for row_index, (csv_row, parquet_row) in enumerate(zip(csv_rows, parquet_rows, strict=True)):
        if set(csv_row) != set(parquet_row):
            raise ValueError(f"attention CSV/parquet columns differ at row {row_index}")
        for field, parquet_value in parquet_row.items():
            raw_value = csv_row[field]
            if isinstance(parquet_value, bool):
                converted: object = raw_value.lower() == "true"
            elif isinstance(parquet_value, int):
                converted = int(raw_value)
            elif isinstance(parquet_value, float):
                converted = float(raw_value)
            else:
                converted = raw_value
            if converted != parquet_value:
                raise ValueError(f"attention CSV/parquet value mismatch at row {row_index}, field {field}")


def _inventory_lookup(
    inventory: Mapping[str, object], model: str, attention_phase: str
) -> dict[tuple[int, int, int, int], str]:
    coverage = inventory.get("coverage_keys")
    if not isinstance(coverage, Mapping) or not isinstance(coverage.get(model), list):
        raise TypeError(f"coverage inventory has no records for {model}")
    lookup: dict[tuple[int, int, int, int], str] = {}
    for record in coverage[model]:
        if not isinstance(record, Mapping) or record.get("op_family") != "attention":
            continue
        structural = record.get("structural")
        if not isinstance(structural, Mapping) or not isinstance(structural.get("axes"), Mapping):
            raise TypeError("attention coverage record lacks structural axes")
        axes = structural["axes"]
        if axes.get("phase") != attention_phase:
            continue
        if axes.get("attn_dtype") != ATTN_DTYPE or axes.get("kv_cache_dtype") != KV_CACHE_DTYPE:
            raise ValueError("attention coverage precision does not match runtime contract")
        key = (
            int(axes["num_heads"]),
            int(axes["num_key_value_heads"]),
            int(axes["head_dim"]),
            int(axes["window_size"]),
        )
        identity = structural.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("attention coverage record lacks structural identity")
        if key in lookup:
            raise ValueError(f"attention coverage has duplicate structural key: {key}")
        lookup[key] = identity
    if not lookup:
        raise ValueError(f"coverage inventory has no {attention_phase} attention identities")
    return lookup


def build_attention_collection_artifacts(
    *,
    task_root: Path,
    model: str,
    attention_phase: str,
    checkpoint_path: Path,
    parquet_path: Path,
    csv_path: Path,
    raw_summary_path: Path,
    coverage_inventory_path: Path,
    artifact_prefix: str,
) -> Path:
    """Build one phase-specific full manifest from exact runtime artifacts."""

    if attention_phase not in {"context", "generation"}:
        raise ValueError("attention_phase must be context or generation")
    root = task_root.resolve()
    checkpoint_path = _inside(root, checkpoint_path, "checkpoint_path")
    parquet_path = _inside(root, parquet_path, "parquet_path")
    csv_path = _inside(root, csv_path, "csv_path")
    raw_summary_path = _inside(root, raw_summary_path, "raw_summary_path")
    coverage_inventory_path = _inside(root, coverage_inventory_path, "coverage_inventory_path")

    checkpoint = _load_json(checkpoint_path, "checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a JSON object")
    expected_checkpoint = {
        "schema": "collector-resume-v1",
        "backend": BACKEND,
        "module": f"vllm.attention_{attention_phase}",
        "run_func": "run_attention_torch",
    }
    for field, value in expected_checkpoint.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"checkpoint {field} must equal {value!r}")
    done = checkpoint.get("done")
    if not isinstance(done, list) or not done or any(not isinstance(value, str) for value in done):
        raise ValueError("checkpoint done must be a non-empty invocation ID list")
    if len(set(done)) != len(done):
        raise ValueError("checkpoint done contains duplicate invocation IDs")
    if checkpoint.get("failed", []) or checkpoint.get("expected_failed", []):
        raise ValueError("attention full checkpoint must contain no failed or expected_failed invocations")

    raw_summary = _load_json(raw_summary_path, "raw collection summary")
    if not isinstance(raw_summary, Mapping) or not isinstance(raw_summary.get("summary"), Mapping):
        raise TypeError("raw collection summary must contain a summary mapping")
    raw_summary_values = raw_summary["summary"]
    if (
        raw_summary_values.get("backend") != BACKEND
        or raw_summary_values.get("version") != VERSION
        or raw_summary_values.get("total_errors") != 0
    ):
        raise ValueError("raw collection summary backend/version/errors violate exact runtime contract")

    table = pq.read_table(parquet_path)
    parquet_rows = table.to_pylist()
    if len(parquet_rows) != len(done):
        raise ValueError("checkpoint and parquet row counts do not form a bijection")
    _validate_csv_parquet_rows(csv_path, parquet_rows)
    row_index_by_key: dict[tuple[int, ...], int] = {}
    for row_index, row in enumerate(parquet_rows):
        key = _validate_row(row, attention_phase)
        if key in row_index_by_key:
            raise ValueError(f"attention parquet has ambiguous workload key: {key}")
        row_index_by_key[key] = row_index

    invocation_by_key: dict[tuple[int, ...], tuple[str, dict[str, int]]] = {}
    for invocation_id in done:
        key, workload = _parse_invocation(invocation_id, attention_phase)
        if key in invocation_by_key:
            raise ValueError(f"attention checkpoint has duplicate workload key: {key}")
        invocation_by_key[key] = (invocation_id, workload)
    if set(invocation_by_key) != set(row_index_by_key):
        missing_rows = sorted(set(invocation_by_key) - set(row_index_by_key))
        extra_rows = sorted(set(row_index_by_key) - set(invocation_by_key))
        raise ValueError(
            f"checkpoint and parquet workloads do not form a bijection: missing={missing_rows}, extra={extra_rows}"
        )

    inventory = _load_json(coverage_inventory_path, "coverage inventory")
    if not isinstance(inventory, Mapping):
        raise TypeError("coverage inventory must be a JSON object")
    structural_lookup = _inventory_lookup(inventory, model, attention_phase)
    output_rel = _relative(root, parquet_path)
    output_sha256 = _sha256(parquet_path)
    expected_invocations = []
    outcomes = []
    reconciliation_entries = []
    measured_identities: set[str] = set()
    for invocation_id in done:
        key, workload = _parse_invocation(invocation_id, attention_phase)
        row_index = row_index_by_key[key]
        heads, kv_heads, head_dim, window_size = key[3], key[4], key[5], key[6]
        structural_key = (heads, kv_heads, head_dim, window_size)
        if structural_key not in structural_lookup:
            raise ValueError(f"attention workload has no planned structural identity: {structural_key}")
        identity = structural_lookup[structural_key]
        measured_identities.add(identity)
        expected_invocations.append(
            {
                "invocation_id": invocation_id,
                "op_family": "attention",
                "structural_identity": identity,
                "workload": workload,
            }
        )
        outcomes.append(
            {
                "invocation_id": invocation_id,
                "status": "measured",
                "output_path": output_rel,
                "sha256": output_sha256,
            }
        )
        reconciliation_entries.append(
            {
                "invocation_id": invocation_id,
                "structural_identity": identity,
                "workload": workload,
                "output_path": output_rel,
                "row_index": row_index,
                "row_fingerprint": _row_fingerprint(parquet_rows[row_index]),
            }
        )
    if measured_identities != set(structural_lookup.values()):
        raise ValueError("measured attention identities do not cover the phase inventory")

    reconciliation_path = root / f"{artifact_prefix}_reconciliation.json"
    measured_keys_path = root / f"{artifact_prefix}_measured_keys.json"
    summary_path = root / f"{artifact_prefix}_collection_summary_manifest.json"
    manifest_path = root / f"{artifact_prefix}_collection_manifest.json"
    reconciliation = {
        "schema": "step4-invocation-reconciliation-v1",
        "source_csv": {
            "path": _relative(root, csv_path),
            "sha256": _sha256(csv_path),
            "row_count": len(parquet_rows),
        },
        "output_parquet": {
            "path": output_rel,
            "sha256": output_sha256,
            "row_count": len(parquet_rows),
        },
        "invocations": reconciliation_entries,
    }
    _write_json(reconciliation_path, reconciliation)
    _write_json(measured_keys_path, {"identities": sorted(measured_identities)})
    summary = {
        "summary": {
            "backend": BACKEND,
            "version": VERSION,
            "total_errors": 0,
            "model": model,
            "system": SYSTEM,
            "phase": "full",
            "op_family": "attention",
            "attention_phase": attention_phase,
            "expected_invocation_count": len(done),
            "terminal_outcome_count": len(done),
            "structural_identity_count": len(measured_identities),
            "measured_count": len(done),
            "unsupported_count": 0,
            "expected_failed_count": 0,
            "failed_count": 0,
        },
        "errors": [],
        "provenance": {
            "raw_collection_summary_path": _relative(root, raw_summary_path),
            "raw_collection_summary_sha256": _sha256(raw_summary_path),
        },
    }
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": "step4-collection-manifest-v1",
        "backend": BACKEND,
        "version": VERSION,
        "system": SYSTEM,
        "device": SYSTEM,
        "model": model,
        "phase": "full",
        "scope": {"op_family": "attention", "attention_phase": attention_phase},
        "coverage_inventory_path": _relative(root, coverage_inventory_path),
        "coverage_inventory_sha256": _sha256(coverage_inventory_path),
        "measured_key_inventory_path": _relative(root, measured_keys_path),
        "measured_key_inventory_sha256": _sha256(measured_keys_path),
        "checkpoint_path": _relative(root, checkpoint_path),
        "collection_summary_path": _relative(root, summary_path),
        "invocation_reconciliation_path": _relative(root, reconciliation_path),
        "invocation_reconciliation_sha256": _sha256(reconciliation_path),
        "expected_invocations": expected_invocations,
        "outcomes": outcomes,
    }
    _write_json(manifest_path, manifest)
    validate_collection_manifest(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention-phase", choices=("context", "generation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--coverage-inventory", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    output = build_attention_collection_artifacts(
        task_root=args.task_root,
        model=args.model,
        attention_phase=args.attention_phase,
        checkpoint_path=args.checkpoint,
        parquet_path=args.parquet,
        csv_path=args.csv,
        raw_summary_path=args.raw_summary,
        coverage_inventory_path=args.coverage_inventory,
        artifact_prefix=args.artifact_prefix,
    )
    print(json.dumps({"manifest": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
