"""Build auditable Step4 modular FP8 MoE collection manifests."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from tests.performance.step4_collection_manifest import validate_collection_manifest

BACKEND = "vllm"
VERSION = "0.19.0"
SYSTEM = "h800_sxm"
DEVICE = "NVIDIA H800"
KERNEL_SOURCE = "vllm_flashinfer_cutlass_moe"
MODELS = {"stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"}


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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_invocation(invocation_id: str) -> dict[str, object]:
    prefix = "vllm.moe:run_moe_torch:"
    if not invocation_id.startswith(prefix):
        raise ValueError(f"unexpected MoE invocation prefix: {invocation_id}")
    try:
        args = ast.literal_eval(invocation_id[len(prefix) :])
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"unable to parse MoE invocation: {invocation_id}") from exc
    if not isinstance(args, list) or len(args) != 11:
        raise ValueError(f"MoE invocation must contain eleven arguments: {invocation_id}")
    dtype, tokens, hidden, inter, topk, experts, tp, ep, model, distribution, alpha = args
    if dtype != "fp8" or distribution != "power_law" or alpha != 1.2:
        raise ValueError("Step4 modular MoE invocation must use fp8 and power_law_1.2")
    if model not in MODELS:
        raise ValueError(f"unsupported Step4 model: {model!r}")
    numeric = (hidden, inter, topk, experts, tp, ep)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in numeric):
        raise ValueError("MoE invocation topology values must be positive integers")
    if tp != 1 or experts % ep != 0:
        raise ValueError("Step4 modular MoE invocation requires tp=1 and experts divisible by ep")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("MoE invocation tokens must be unique positive integers")
    token_tuple = tuple(tokens)
    if token_tuple != tuple(sorted(token_tuple)):
        raise ValueError("MoE invocation tokens must be sorted")
    distribution_key = f"{distribution}_{alpha}"
    return {
        "invocation_id": invocation_id,
        "model": model,
        "tokens": token_tuple,
        "structural_key": (hidden, inter, topk, experts, tp, ep, dtype, distribution_key),
    }


def _validate_rows_for_invocation(rows: list[Mapping[str, object]], invocation: Mapping[str, object]) -> None:
    structural_key = invocation["structural_key"]
    if not isinstance(structural_key, tuple) or len(structural_key) != 8:
        raise TypeError("parsed invocation structural_key is invalid")
    hidden, inter, topk, experts, tp, ep, dtype, distribution = structural_key
    expected = {
        "framework": "VLLM",
        "version": VERSION,
        "device": DEVICE,
        "op_name": "moe",
        "kernel_source": KERNEL_SOURCE,
        "moe_dtype": dtype,
        "hidden_size": hidden,
        "inter_size": inter,
        "topk": topk,
        "num_experts": experts,
        "moe_tp_size": tp,
        "moe_ep_size": ep,
        "distribution": distribution,
    }
    observed_tokens = []
    for row in rows:
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"MoE parquet {field} must equal {value!r}; got {row.get(field)!r}")
        num_tokens = row.get("num_tokens")
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens <= 0:
            raise ValueError("MoE parquet num_tokens must be a positive integer")
        latency = row.get("latency")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, int | float)
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError("MoE parquet latency must be finite and positive")
        observed_tokens.append(num_tokens)
    if tuple(sorted(observed_tokens)) != invocation["tokens"]:
        raise ValueError("MoE parquet token coverage does not match checkpoint invocation")


def _validate_csv_parquet_rows(csv_path: Path, parquet_rows: list[dict[str, object]]) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(parquet_rows):
        raise ValueError("MoE CSV and parquet row counts differ")
    for row_index, (csv_row, parquet_row) in enumerate(zip(csv_rows, parquet_rows, strict=True)):
        if set(csv_row) != set(parquet_row):
            raise ValueError(f"MoE CSV/parquet columns differ at row {row_index}")
        for field, parquet_value in parquet_row.items():
            raw_value = csv_row[field]
            if isinstance(parquet_value, int):
                converted: object = int(raw_value)
            elif isinstance(parquet_value, float):
                converted = float(raw_value)
            else:
                converted = raw_value
            if converted != parquet_value:
                raise ValueError(f"MoE CSV/parquet value mismatch at row {row_index}, field {field}")


def _inventory_lookup(inventory: Mapping[str, object], model: str) -> dict[tuple[object, ...], str]:
    coverage = inventory.get("coverage_keys")
    if not isinstance(coverage, Mapping) or not isinstance(coverage.get(model), list):
        raise TypeError(f"coverage inventory has no records for {model}")
    lookup: dict[tuple[object, ...], str] = {}
    for record in coverage[model]:
        if not isinstance(record, Mapping) or record.get("op_family") != "moe":
            continue
        structural = record.get("structural")
        if not isinstance(structural, Mapping) or not isinstance(structural.get("axes"), Mapping):
            raise TypeError("MoE coverage record lacks structural axes")
        axes = structural["axes"]
        if axes.get("backend") != BACKEND or axes.get("version") != VERSION or axes.get("device") != SYSTEM:
            raise ValueError("MoE coverage runtime identity violates the exact contract")
        key = (
            axes.get("hidden_size"),
            axes.get("inter_size"),
            axes.get("topk"),
            axes.get("num_experts"),
            axes.get("moe_tp_size"),
            axes.get("moe_ep_size"),
            axes.get("quantization"),
            axes.get("distribution"),
        )
        identity = structural.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("MoE coverage record lacks structural identity")
        if key in lookup:
            raise ValueError(f"MoE coverage has duplicate structural key: {key}")
        lookup[key] = identity
    if not lookup:
        raise ValueError(f"coverage inventory has no MoE identities for {model}")
    return lookup


def build_moe_collection_artifacts(
    *,
    task_root: Path,
    model: str,
    checkpoint_path: Path,
    parquet_path: Path,
    csv_path: Path,
    raw_summary_path: Path,
    coverage_inventory_path: Path,
    artifact_prefix: str,
) -> Path:
    """Build one full MoE manifest from exact modular runtime artifacts."""

    if model not in MODELS:
        raise ValueError(f"unsupported Step4 model: {model!r}")
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
        "module": "vllm.moe",
        "run_func": "run_moe_torch",
    }
    for field, value in expected_checkpoint.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"checkpoint {field} must equal {value!r}")
    done = checkpoint.get("done")
    if not isinstance(done, list) or not done or any(not isinstance(value, str) for value in done):
        raise ValueError("checkpoint done must be a non-empty invocation ID list")
    if len(set(done)) != len(done):
        raise ValueError("checkpoint done contains duplicate invocation IDs")
    if checkpoint.get("failed", []) or checkpoint.get("expected_failed", []) or checkpoint.get("unsupported", []):
        raise ValueError("modular MoE full checkpoint must contain only measured outcomes")

    raw_summary = _load_json(raw_summary_path, "raw collection summary")
    if not isinstance(raw_summary, Mapping) or not isinstance(raw_summary.get("summary"), Mapping):
        raise TypeError("raw collection summary must contain a summary mapping")
    raw_summary_values = raw_summary["summary"]
    if (
        raw_summary_values.get("backend") != BACKEND
        or raw_summary_values.get("version") != VERSION
        or raw_summary_values.get("total_errors") != 0
        or raw_summary.get("errors") != []
    ):
        raise ValueError("raw collection summary violates the exact zero-error runtime contract")

    table = pq.read_table(parquet_path)
    parquet_rows = table.to_pylist()
    _validate_csv_parquet_rows(csv_path, parquet_rows)
    parsed_invocations = [_parse_invocation(invocation_id) for invocation_id in done]
    if any(invocation["model"] != model for invocation in parsed_invocations):
        raise ValueError("checkpoint invocation model does not match manifest model")
    invocation_by_key = {invocation["structural_key"]: invocation for invocation in parsed_invocations}
    if len(invocation_by_key) != len(parsed_invocations):
        raise ValueError("checkpoint contains duplicate MoE structural keys")

    rows_by_key: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in parquet_rows:
        key = (
            row.get("hidden_size"),
            row.get("inter_size"),
            row.get("topk"),
            row.get("num_experts"),
            row.get("moe_tp_size"),
            row.get("moe_ep_size"),
            row.get("moe_dtype"),
            row.get("distribution"),
        )
        rows_by_key[key].append(row)
    if set(rows_by_key) != set(invocation_by_key):
        missing = sorted(set(invocation_by_key) - set(rows_by_key))
        extra = sorted(set(rows_by_key) - set(invocation_by_key))
        raise ValueError(f"checkpoint and MoE parquet structural keys differ: missing={missing}, extra={extra}")
    for key, invocation in invocation_by_key.items():
        _validate_rows_for_invocation(rows_by_key[key], invocation)

    inventory = _load_json(coverage_inventory_path, "coverage inventory")
    if not isinstance(inventory, Mapping):
        raise TypeError("coverage inventory must be a JSON object")
    structural_lookup = _inventory_lookup(inventory, model)
    if set(structural_lookup) != set(invocation_by_key):
        raise ValueError("measured MoE structural keys do not cover the planned inventory")

    output_rel = _relative(root, parquet_path)
    output_sha256 = _sha256(parquet_path)
    expected_invocations = []
    outcomes = []
    measured_identities = set()
    for invocation in parsed_invocations:
        invocation_id = invocation["invocation_id"]
        key = invocation["structural_key"]
        identity = structural_lookup[key]
        measured_identities.add(identity)
        expected_invocations.append(
            {
                "invocation_id": invocation_id,
                "op_family": "moe",
                "structural_identity": identity,
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

    measured_keys_path = root / f"{artifact_prefix}_measured_keys.json"
    summary_path = root / f"{artifact_prefix}_collection_summary_manifest.json"
    manifest_path = root / f"{artifact_prefix}_collection_manifest.json"
    _write_json(measured_keys_path, {"identities": sorted(measured_identities)})
    summary = {
        "summary": {
            "backend": BACKEND,
            "version": VERSION,
            "total_errors": 0,
            "model": model,
            "system": SYSTEM,
            "phase": "full",
            "op_family": "moe",
            "expected_invocation_count": len(parsed_invocations),
            "terminal_outcome_count": len(parsed_invocations),
            "structural_identity_count": len(measured_identities),
            "measured_count": len(parsed_invocations),
            "unsupported_count": 0,
            "expected_failed_count": 0,
            "failed_count": 0,
        },
        "errors": [],
        "provenance": {
            "raw_collection_summary_path": _relative(root, raw_summary_path),
            "raw_collection_summary_sha256": _sha256(raw_summary_path),
            "source_csv_path": _relative(root, csv_path),
            "source_csv_sha256": _sha256(csv_path),
            "source_row_count": len(parquet_rows),
            "kernel_source": KERNEL_SOURCE,
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
        "scope": {"op_family": "moe"},
        "coverage_inventory_path": _relative(root, coverage_inventory_path),
        "coverage_inventory_sha256": _sha256(coverage_inventory_path),
        "measured_key_inventory_path": _relative(root, measured_keys_path),
        "measured_key_inventory_sha256": _sha256(measured_keys_path),
        "checkpoint_path": _relative(root, checkpoint_path),
        "collection_summary_path": _relative(root, summary_path),
        "expected_invocations": expected_invocations,
        "outcomes": outcomes,
    }
    _write_json(manifest_path, manifest)
    validate_collection_manifest(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--coverage-inventory", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    output = build_moe_collection_artifacts(
        task_root=args.task_root,
        model=args.model,
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
