"""Validate measured Step4-Pro-Latest core-provider rows through AIC consumers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import FP32OutputGEMM, GroupedGEMM, QKVNormRoPE
from aiconfigurator.sdk.perf_database import PerfDatabase

RUNTIME_VERSION = "0.19.0.post20.dev26+gc820e5ae1"
DEVICE = "NVIDIA B300 SXM6 AC"

FILES = {
    "grouped": "step4_grouped_gemm_perf.txt",
    "router": "step4_fp32_output_gemm_perf.txt",
    "qkv": "step4_qkv_norm_rope_perf.txt",
}
DEFAULT_EXPECTED_ROWS = {
    "grouped": 75,
    "router": 75,
    "qkv": 150,
}
EXPECTED_QKV_PROVIDER_ROWS = {
    "vllm_step4pro_k_norm_rope": 75,
    "vllm_step4pro_qkv_norm_rope": 75,
}

PHYSICAL_KEY_FIELDS = {
    "grouped": ("provider", "groups", "num_tokens", "n", "k", "quant_mode"),
    "router": (
        "provider",
        "num_tokens",
        "n",
        "k",
        "weight_dtype",
        "output_dtype",
    ),
    "qkv": (
        "provider",
        "num_tokens",
        "normalized_tensors",
        "q_heads",
        "kv_heads",
        "head_dim",
    ),
}


def _prepare_database(
    dataset_dir: Path,
    work_dir: Path,
    families: tuple[str, ...],
) -> PerfDatabase:
    systems_root = work_dir / "systems"
    data_dir = systems_root / "data" / "vllm" / "0.19.0"
    data_dir.mkdir(parents=True, exist_ok=True)
    system_spec = yaml.safe_load(Path("src/aiconfigurator/systems/b300_sxm.yaml").read_text(encoding="utf-8"))
    system_spec["data_dir"] = "data"
    (systems_root / "b300_sxm.yaml").write_text(
        yaml.safe_dump(system_spec),
        encoding="utf-8",
    )
    for family in families:
        filename = FILES[family]
        shutil.copyfile(dataset_dir / filename, data_dir / filename)
    return PerfDatabase("b300_sxm", "vllm", "0.19.0", str(systems_root))


def _read_and_validate_rows(
    dataset_dir: Path,
    family: str,
    *,
    expected_rows: int,
    expected_op_name: str,
    validate_structure: Callable[[dict[str, str]], None],
) -> list[dict[str, str]]:
    path = dataset_dir / FILES[family]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    keys: set[tuple[str, ...]] = set()
    for row in rows:
        if row["framework"] != "VLLM":
            raise RuntimeError(f"{family} framework mismatch: {row['framework']!r}")
        if row["version"] != RUNTIME_VERSION:
            raise RuntimeError(f"{family} runtime version mismatch: {row['version']!r}")
        if row["device"] != DEVICE:
            raise RuntimeError(f"{family} device mismatch: {row['device']!r}")
        if row["op_name"] != expected_op_name:
            raise RuntimeError(f"{family} op_name mismatch: {row['op_name']!r}")
        if row["kernel_source"] != row["provider"]:
            raise RuntimeError(f"{family} kernel/provider mismatch: {row['kernel_source']!r} != {row['provider']!r}")
        latency = float(row["latency"])
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError(f"{family} invalid latency: {row['latency']!r}")
        validate_structure(row)
        key = tuple(row[field] for field in PHYSICAL_KEY_FIELDS[family])
        if key in keys:
            raise RuntimeError(f"{family} duplicate physical key: {key!r}")
        keys.add(key)
    if len(rows) != expected_rows:
        raise RuntimeError(f"{family} row count mismatch: expected={expected_rows}, actual={len(rows)}")
    return rows


def _validate_grouped_structure(row: dict[str, str]) -> None:
    actual = (
        row["provider"],
        int(row["groups"]),
        int(row["n"]),
        int(row["k"]),
        row["quant_mode"],
    )
    expected = ("vllm_step4pro_torch_einsum", 8, 1024, 4096, "bfloat16")
    if actual != expected:
        raise RuntimeError(f"grouped structure mismatch: expected={expected}, actual={actual}")


def _validate_router_structure(row: dict[str, str]) -> None:
    actual = (
        row["provider"],
        int(row["n"]),
        int(row["k"]),
        row["weight_dtype"],
        row["output_dtype"],
    )
    expected = ("vllm.optimus_matmul_fp32", 896, 7168, "bfloat16", "float32")
    if actual != expected:
        raise RuntimeError(f"router structure mismatch: expected={expected}, actual={actual}")


def _validate_qkv_structure(row: dict[str, str]) -> None:
    actual = (
        row["provider"],
        row["normalized_tensors"],
        int(row["q_heads"]),
        int(row["kv_heads"]),
        int(row["head_dim"]),
    )
    expected = {
        ("vllm_step4pro_k_norm_rope", "k", 64, 1, 512),
        ("vllm_step4pro_qkv_norm_rope", "q+k+v", 128, 8, 128),
    }
    if actual not in expected:
        raise RuntimeError(f"qkv structure mismatch: actual={actual}")


def _expected_qkv_physical_keys() -> set[tuple[str, ...]]:
    from collector.vllm.collect_step4_provider import (
        get_step4_qkv_norm_rope_test_cases,
    )

    return {tuple(str(value) for value in case) for case in get_step4_qkv_norm_rope_test_cases()}


def _validate_qkv_expected_key_set(
    rows: list[dict[str, str]],
    *,
    expected_rows: int,
) -> dict[str, int]:
    expected_keys = _expected_qkv_physical_keys()
    if expected_rows != len(expected_keys):
        raise RuntimeError(
            f"QKV expected-row contract mismatch: argument={expected_rows}, collector={len(expected_keys)}"
        )
    actual_keys = {tuple(row[field] for field in PHYSICAL_KEY_FIELDS["qkv"]) for row in rows}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            "QKV physical key set mismatch: "
            f"missing={missing[:5]!r} ({len(missing)} total), "
            f"unexpected={unexpected[:5]!r} ({len(unexpected)} total)"
        )
    provider_rows = dict(Counter(row["provider"] for row in rows))
    if provider_rows != EXPECTED_QKV_PROVIDER_ROWS:
        raise RuntimeError(
            f"QKV provider row-count mismatch: expected={EXPECTED_QKV_PROVIDER_ROWS}, actual={provider_rows}"
        )
    return provider_rows


def _query_grouped(database: PerfDatabase, row: dict[str, str]):
    operation = GroupedGEMM(
        "wo_a",
        1.0,
        int(row["n"]),
        int(row["k"]),
        common.GEMMQuantMode[row["quant_mode"]],
        groups=int(row["groups"]),
        provider=row["provider"],
    )
    return operation.query(database, x=int(row["num_tokens"]))


def _query_router(database: PerfDatabase, row: dict[str, str]):
    operation = FP32OutputGEMM(
        "router",
        1.0,
        int(row["n"]),
        int(row["k"]),
        weight_dtype=row["weight_dtype"],
        output_dtype=row["output_dtype"],
        provider=row["provider"],
    )
    return operation.query(database, x=int(row["num_tokens"]))


def _query_qkv(database: PerfDatabase, row: dict[str, str]):
    operation = QKVNormRoPE(
        "qkv_norm_rope",
        1.0,
        normalized_tensors=tuple(row["normalized_tensors"].split("+")),
        provider=row["provider"],
        q_heads=int(row["q_heads"]),
        kv_heads=int(row["kv_heads"]),
        head_dim=int(row["head_dim"]),
    )
    return operation.query(database, x=int(row["num_tokens"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-grouped-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS["grouped"],
    )
    parser.add_argument(
        "--expected-router-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS["router"],
    )
    parser.add_argument(
        "--expected-qkv-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS["qkv"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FILES),
        default=tuple(FILES),
    )
    args = parser.parse_args()
    families = tuple(dict.fromkeys(args.families))

    grouped_rows = (
        _read_and_validate_rows(
            args.dataset_dir,
            "grouped",
            expected_rows=args.expected_grouped_rows,
            expected_op_name="step4_grouped_gemm",
            validate_structure=_validate_grouped_structure,
        )
        if "grouped" in families
        else []
    )
    router_rows = (
        _read_and_validate_rows(
            args.dataset_dir,
            "router",
            expected_rows=args.expected_router_rows,
            expected_op_name="step4_fp32_output_gemm",
            validate_structure=_validate_router_structure,
        )
        if "router" in families
        else []
    )
    qkv_rows = (
        _read_and_validate_rows(
            args.dataset_dir,
            "qkv",
            expected_rows=args.expected_qkv_rows,
            expected_op_name="step4_qkv_norm_rope",
            validate_structure=_validate_qkv_structure,
        )
        if "qkv" in families
        else []
    )
    qkv_provider_rows = (
        _validate_qkv_expected_key_set(
            qkv_rows,
            expected_rows=args.expected_qkv_rows,
        )
        if "qkv" in families
        else {}
    )

    database = _prepare_database(args.dataset_dir, args.work_dir, families)
    queries = [
        *((row, _query_grouped(database, row)) for row in grouped_rows),
        *((row, _query_router(database, row)) for row in router_rows),
        *((row, _query_qkv(database, row)) for row in qkv_rows),
    ]
    for row, result in queries:
        measured = float(row["latency"])
        queried = float(result)
        if queried != measured:
            raise RuntimeError(f"consumer latency mismatch: op={row['op_name']} measured={measured} queried={queried}")
        if result.source != "silicon":
            raise RuntimeError(f"non-silicon consumer result: op={row['op_name']} source={result.source!r}")

    latencies = [float(row["latency"]) for row in (*grouped_rows, *router_rows, *qkv_rows)]
    payload = {
        "status": "PASS",
        "runtime_version": RUNTIME_VERSION,
        "device": DEVICE,
        "grouped_rows": len(grouped_rows),
        "router_rows": len(router_rows),
        "qkv_rows": len(qkv_rows),
        "qkv_provider_rows": qkv_provider_rows,
        "total_rows": len(latencies),
        "exact_consumer_matches": len(queries),
        "silicon_source_rows": sum(result.source == "silicon" for _, result in queries),
        "duplicate_physical_keys": 0,
        "minimum_latency_ms": min(latencies),
        "maximum_latency_ms": max(latencies),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
