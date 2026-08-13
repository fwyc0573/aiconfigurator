"""Validate and merge exact-rank P6 NCCL sweep artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping
from pathlib import Path

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXPECTED_OPS = ("all_gather", "reduce_scatter")
_REQUIRED_WORLD_SIZES = frozenset({2, 4, 8, 16, 32, 64})
_ALLOWED_EXTRA_WORLD_SIZES = frozenset({48})


def merge_sweep_payloads(payloads: list[Mapping[str, object]]) -> dict[str, object]:
    if not payloads:
        raise ValueError("at least one sweep payload is required")
    allowed_ranks = _REQUIRED_WORLD_SIZES | _ALLOWED_EXTRA_WORLD_SIZES
    seen_ranks: set[int] = set()
    identities: set[tuple[str, int, int]] = set()
    rows: list[Mapping[str, object]] = []
    reference = None
    for payload in payloads:
        world_size = payload.get("world_size")
        if not isinstance(world_size, int):
            raise TypeError("sweep world_size must be an integer")
        validate_sweep_payload(payload, expected_world_size=world_size)
        if world_size not in allowed_ranks:
            raise ValueError(f"sweep world_size is outside the approved required/extra set: world_size={world_size}")
        if world_size in seen_ranks:
            raise ValueError("duplicate sweep world_size")
        seen_ranks.add(world_size)
        runtime = (
            payload.get("vllm_version"),
            payload.get("nccl_version"),
            payload.get("device"),
            payload.get("compute_capability"),
            payload.get("ops"),
            payload.get("image_reference"),
            payload.get("image_manifest_digest"),
        )
        if reference is None:
            reference = runtime
        elif runtime != reference:
            raise ValueError("sweep runtime provenance differs across ranks")
        for row in payload["rows"]:
            identity = (str(row["op_name"]), int(row["num_gpus"]), int(row["message_size"]))
            if identity in identities:
                raise ValueError("duplicate sweep row identity")
            identities.add(identity)
            rows.append(row)
    missing_ranks = _REQUIRED_WORLD_SIZES - seen_ranks
    if missing_ranks:
        raise ValueError(
            "sweep ranks are incomplete: "
            f"required={sorted(_REQUIRED_WORLD_SIZES)} missing={sorted(missing_ranks)} actual={sorted(seen_ranks)}"
        )
    rows.sort(key=lambda row: (int(row["num_gpus"]), str(row["op_name"]), int(row["message_size"])))
    return {
        "row_count": len(rows),
        "world_sizes": sorted(seen_ranks),
        "required_world_sizes": sorted(_REQUIRED_WORLD_SIZES),
        "extra_world_sizes": sorted(seen_ranks - _REQUIRED_WORLD_SIZES),
        "rows": [dict(row) for row in rows],
        "vllm_version": reference[0],
        "nccl_version": reference[1],
        "device": reference[2],
        "compute_capability": reference[3],
        "ops": list(reference[4]),
        "image_reference": reference[5],
        "image_manifest_digest": reference[6],
    }


def build_nccl_manifest(
    *,
    source_paths: list[Path],
    output_parquet: Path,
    output_manifest: Path,
    image_config_digest: str,
    runtime_provenance_path: Path,
) -> dict[str, object]:
    if not _DIGEST_RE.fullmatch(image_config_digest):
        raise ValueError("image config digest is invalid")
    manifest_dir = output_manifest.resolve().parent
    runtime_path = runtime_provenance_path.resolve()
    try:
        relative_runtime_path = runtime_path.relative_to(manifest_dir)
    except ValueError as exc:
        raise ValueError("runtime provenance must be stored beneath the manifest directory") from exc
    if not runtime_path.is_file():
        raise ValueError("runtime provenance file does not exist")
    runtime_provenance_sha256 = hashlib.sha256(runtime_path.read_bytes()).hexdigest()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in source_paths]
    merged = merge_sweep_payloads(payloads)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("P6 manifest packaging requires pyarrow") from exc
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(merged["rows"])
    pq.write_table(table, output_parquet)
    parquet_sha256 = hashlib.sha256(output_parquet.read_bytes()).hexdigest()
    source_artifacts = []
    for path, payload in zip(source_paths, payloads, strict=True):
        source_artifacts.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "world_size": payload["world_size"],
                "row_count": len(payload["rows"]),
            }
        )
    rows = merged["rows"]
    manifest = {
        "schema": "p6-nccl-manifest-v1",
        "artifact_type": "p6_exact_rank_nccl_message_sweep",
        "status": "validated",
        "canonical_collection": False,
        "diagnostic_only": False,
        "backend": "vllm",
        "vllm_version": merged["vllm_version"],
        "nccl_version": merged["nccl_version"],
        "device": merged["device"],
        "compute_capability": merged["compute_capability"],
        "world_sizes": merged["world_sizes"],
        "required_world_sizes": merged["required_world_sizes"],
        "extra_world_sizes": merged["extra_world_sizes"],
        "ops": merged["ops"],
        "dtype": "half",
        "image_reference": merged["image_reference"],
        "image_manifest_digest": merged["image_manifest_digest"],
        "image_config_digest": image_config_digest,
        "runtime_provenance_path": str(relative_runtime_path),
        "runtime_provenance_sha256": runtime_provenance_sha256,
        "source_artifacts": source_artifacts,
        "output_parquet": {
            "path": output_parquet.name,
            "sha256": parquet_sha256,
            "row_count": len(rows),
            "columns": table.column_names,
        },
        "message_size_elements": {
            "min": min(int(row["message_size"]) for row in rows),
            "max": max(int(row["message_size"]) for row in rows),
            "axis": "message_size",
        },
        "message_size_bytes": {
            "min": min(int(row["message_size_bytes"]) for row in rows),
            "max": max(int(row["message_size_bytes"]) for row in rows),
            "axis": "message_size_bytes",
        },
        "samples_per_point": 3,
        "measured_row_count": len(rows),
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--image-config-digest", required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_nccl_manifest(
        source_paths=args.source,
        output_parquet=args.output_parquet,
        output_manifest=args.output_manifest,
        image_config_digest=args.image_config_digest,
        runtime_provenance_path=args.runtime_provenance,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rows": manifest["measured_row_count"],
                "world_sizes": manifest["world_sizes"],
                "output_manifest": str(args.output_manifest),
                "output_parquet": str(args.output_parquet),
            },
            sort_keys=True,
        )
    )


def validate_sweep_payload(payload: Mapping[str, object], *, expected_world_size: int) -> dict[str, object]:
    if payload.get("artifact_type") != "p6_exact_rank_nccl_message_sweep":
        raise ValueError("sweep artifact_type is invalid")
    if payload.get("backend") != "nccl":
        raise ValueError("sweep backend must be nccl")
    if payload.get("framework_backend") != "vllm":
        raise ValueError("sweep framework_backend must be vllm")
    if payload.get("dtype") != "half":
        raise ValueError("sweep dtype must be half")
    if payload.get("canonical_collection") is not False or payload.get("diagnostic_only") is not False:
        raise ValueError("sweep disposition flags are invalid")
    if payload.get("vllm_version") != "0.19.0":
        raise ValueError("sweep vllm version must be exact 0.19.0")
    if payload.get("nccl_version") != "2.27.5":
        raise ValueError("sweep NCCL version must be exact 2.27.5")
    if payload.get("device") != "NVIDIA H800":
        raise ValueError("sweep device must be NVIDIA H800")
    if payload.get("compute_capability") != [9, 0]:
        raise ValueError("sweep compute capability must be SM90")
    if payload.get("world_size") != expected_world_size:
        raise ValueError("sweep world size does not match expected rank")
    if payload.get("ops") != list(_EXPECTED_OPS):
        raise ValueError("sweep ops must be all_gather and reduce_scatter")
    if not isinstance(payload.get("image_reference"), str) or not payload["image_reference"]:
        raise ValueError("sweep image reference is required")
    if not isinstance(payload.get("image_manifest_digest"), str) or not _DIGEST_RE.fullmatch(
        payload["image_manifest_digest"]
    ):
        raise ValueError("sweep image manifest digest is invalid")

    rows = payload.get("rows")
    measurements = payload.get("measurements")
    if not isinstance(rows, list) or not isinstance(measurements, list) or len(rows) != len(measurements):
        raise ValueError("sweep rows and measurements must be equal lists")
    expected_samples = payload.get("samples", 3)
    if not isinstance(expected_samples, int) or expected_samples < 3:
        raise ValueError("sweep samples must be at least three")

    identities: set[tuple[str, int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("sweep rows must be mappings")
        if (
            row.get("framework") != "NCCL"
            or row.get("kernel_source") != "NCCL"
            or row.get("nccl_dtype") != "half"
            or row.get("device") != "NVIDIA H800"
            or row.get("version") != "2.27.5"
        ):
            raise ValueError("sweep row runtime identity is invalid")
        op = row.get("op_name")
        world_size = row.get("num_gpus")
        message_size = row.get("message_size")
        if op not in _EXPECTED_OPS or world_size != expected_world_size:
            raise ValueError("sweep row operation or world size is invalid")
        if not isinstance(message_size, int) or message_size <= 0 or message_size % expected_world_size:
            raise ValueError("sweep row message_size is not rank-aligned")
        if row.get("message_size_bytes") != message_size * 2:
            raise ValueError("sweep row byte count does not match FP16 elements")
        latency = row.get("latency")
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("sweep row latency must be finite and positive")
        identity = (str(op), int(world_size), message_size)
        if identity in identities:
            raise ValueError("sweep rows contain duplicate identity")
        identities.add(identity)

    measurement_identities: set[tuple[str, int, int]] = set()
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise TypeError("sweep measurements must be mappings")
        op = measurement.get("op")
        world_size = measurement.get("world_size")
        message_elements = measurement.get("message_elements")
        if op not in _EXPECTED_OPS or world_size != expected_world_size:
            raise ValueError("sweep measurement operation or world size is invalid")
        if not isinstance(message_elements, int) or message_elements <= 0 or message_elements % expected_world_size:
            raise ValueError("sweep measurement message_elements is not rank-aligned")
        measurement_identity = (str(op), int(world_size), message_elements)
        if measurement_identity in measurement_identities:
            raise ValueError("sweep measurements contain duplicate identity")
        measurement_identities.add(measurement_identity)
        samples = measurement.get("max_rank_samples_ms")
        if not isinstance(samples, list) or len(samples) != expected_samples:
            raise ValueError("sweep measurements have invalid samples")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("sweep measurement latency must be finite and positive")
        median_latency = measurement.get("latency_ms_median_of_max_rank")
        if not isinstance(median_latency, (int, float)) or not math.isfinite(median_latency) or median_latency <= 0:
            raise ValueError("sweep measurement median latency is invalid")
        if not math.isclose(median_latency, statistics.median(samples), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("sweep measurement median latency is invalid")

    if measurement_identities != identities:
        raise ValueError("sweep rows and measurements have different identities")

    return {
        "row_count": len(rows),
        "world_size": expected_world_size,
        "ops": list(_EXPECTED_OPS),
        "message_size_count": len({identity[2] for identity in identities}),
    }


if __name__ == "__main__":
    main()
