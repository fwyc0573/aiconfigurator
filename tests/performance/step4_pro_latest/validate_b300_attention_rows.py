"""Load measured Step4-Pro-Latest Attention rows through the AIC consumer."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import ContextAttention, GenerationAttention
from aiconfigurator.sdk.perf_database import PerfDatabase


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _prepare_database(dataset_dir: Path, work_dir: Path) -> PerfDatabase:
    systems_root = work_dir / "systems"
    data_dir = systems_root / "data" / "vllm" / "0.19.0"
    data_dir.mkdir(parents=True, exist_ok=True)
    system_spec = yaml.safe_load(Path("src/aiconfigurator/systems/b300_sxm.yaml").read_text(encoding="utf-8"))
    system_spec["data_dir"] = "data"
    (systems_root / "b300_sxm.yaml").write_text(
        yaml.safe_dump(system_spec),
        encoding="utf-8",
    )
    for filename in (
        "step4_context_attention_perf.txt",
        "step4_generation_attention_perf.txt",
    ):
        shutil.copyfile(dataset_dir / filename, data_dir / filename)
    return PerfDatabase("b300_sxm", "vllm", "0.19.0", str(systems_root))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"no measured rows in {path}")
    return rows


def _query_context(database: PerfDatabase, row: dict[str, str]) -> dict[str, object]:
    operation = ContextAttention(
        "context_attention",
        1.0,
        int(row["num_heads"]),
        int(row["num_key_value_heads"]),
        common.KVCacheQuantMode[row["kv_cache_dtype"]],
        common.FMHAQuantMode[row["attn_dtype"]],
        window_size=int(row["window_size"]),
        head_size=int(row["head_dim"]),
        provider=row["provider"],
        kv_storage_alias=_parse_bool(row["kv_storage_alias"]),
        page_size=int(row["page_size"]),
        physical_page_bytes=int(row["physical_page_bytes"]),
        kv_block_stride_bytes=int(row["kv_block_stride_bytes"]),
        kv_cache_layout=row["kv_cache_layout"],
    )
    query_tokens = int(row["query_tokens"])
    total_context_tokens = int(row["total_context_tokens"])
    result = operation.query(
        database,
        batch_size=int(row["batch_size"]),
        s=query_tokens,
        prefix=total_context_tokens - query_tokens,
    )
    measured = float(row["latency"])
    queried = float(result)
    if queried != measured:
        raise RuntimeError(
            f"context consumer mismatch: provider={row['provider']} measured={measured} queried={queried}"
        )
    return {
        "phase": "context",
        "provider": row["provider"],
        "measured_latency_ms": measured,
        "queried_latency_ms": queried,
        "source": result.source,
    }


def _query_generation(database: PerfDatabase, row: dict[str, str]) -> dict[str, object]:
    operation = GenerationAttention(
        "generation_attention",
        1.0,
        int(row["num_heads"]),
        int(row["num_key_value_heads"]),
        common.KVCacheQuantMode[row["kv_cache_dtype"]],
        window_size=int(row["window_size"]),
        head_size=int(row["head_dim"]),
        provider=row["provider"],
        kv_storage_alias=_parse_bool(row["kv_storage_alias"]),
        page_size=int(row["page_size"]),
        physical_page_bytes=int(row["physical_page_bytes"]),
        kv_block_stride_bytes=int(row["kv_block_stride_bytes"]),
        kv_cache_layout=row["kv_cache_layout"],
    )
    result = operation.query(
        database,
        beam_width=1,
        batch_size=int(row["batch_size"]),
        s=int(row["context_tokens"]),
    )
    measured = float(row["latency"])
    queried = float(result)
    if queried != measured:
        raise RuntimeError(
            f"generation consumer mismatch: provider={row['provider']} measured={measured} queried={queried}"
        )
    return {
        "phase": "generation",
        "provider": row["provider"],
        "measured_latency_ms": measured,
        "queried_latency_ms": queried,
        "source": result.source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    database = _prepare_database(args.dataset_dir, args.work_dir)
    context_rows = _read_rows(args.dataset_dir / "step4_context_attention_perf.txt")
    generation_rows = _read_rows(args.dataset_dir / "step4_generation_attention_perf.txt")
    results = [
        *(_query_context(database, row) for row in context_rows),
        *(_query_generation(database, row) for row in generation_rows),
    ]
    if any(item["source"] != "silicon" for item in results):
        raise RuntimeError(f"non-silicon consumer result: {results}")
    payload = {
        "status": "PASS",
        "context_rows": len(context_rows),
        "generation_rows": len(generation_rows),
        "queries": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
