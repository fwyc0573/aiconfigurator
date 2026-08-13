"""Tests for real attention checkpoint-to-parquet reconciliation artifacts."""

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.performance.step4_attention_manifest import build_attention_collection_artifacts
from tests.performance.step4_collection_manifest import validate_collection_manifest

pytestmark = pytest.mark.unit

MODEL = "stepfun-ai/Step4-Pro-V3"


def _identity(heads: int, kv_heads: int, window_size: int) -> str:
    axes = {
        "attn_dtype": "bfloat16",
        "head_dim": 128,
        "kv_cache_dtype": "fp8",
        "num_heads": heads,
        "num_key_value_heads": kv_heads,
        "phase": "context",
        "window_size": window_size,
    }
    return f"{MODEL}:attention:vllm:0.19.0:h800_sxm:" + json.dumps(axes, sort_keys=True, separators=(",", ":"))


def _write_input_fixture(tmp_path: Path) -> dict[str, Path]:
    rows = [
        {
            "framework": "VLLM",
            "version": "0.19.0",
            "device": "NVIDIA H800",
            "op_name": "context_attention",
            "kernel_source": "vllm_flash_attn",
            "batch_size": 2,
            "isl": 16,
            "num_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "beam_width": 1,
            "attn_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "step": 0,
            "window_size": 512,
            "latency": 0.2,
        },
        {
            "framework": "VLLM",
            "version": "0.19.0",
            "device": "NVIDIA H800",
            "op_name": "context_attention",
            "kernel_source": "vllm_flash_attn",
            "batch_size": 1,
            "isl": 8,
            "num_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "beam_width": 1,
            "attn_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "step": 0,
            "window_size": 0,
            "latency": 0.1,
        },
    ]
    parquet_path = tmp_path / "context_attention_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    csv_path = tmp_path / "context_attention_perf.txt"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    invocation_second_row = "vllm.attention_context:run_attention_torch:[1, 8, 2, 1, 128, True, True, 0]"
    invocation_first_row = "vllm.attention_context:run_attention_torch:[2, 16, 4, 2, 128, True, True, 512]"
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": "collector-resume-v1",
                "backend": "vllm",
                "module": "vllm.attention_context",
                "run_func": "run_attention_torch",
                "done": [invocation_second_row, invocation_first_row],
                "failed": [],
                "expected_failed": [],
            }
        )
    )
    raw_summary_path = tmp_path / "raw_summary.json"
    raw_summary_path.write_text(
        json.dumps(
            {
                "summary": {"backend": "vllm", "version": "0.19.0", "total_errors": 0},
                "errors": [],
            }
        )
    )
    records = []
    for heads, kv_heads, window in ((2, 1, 0), (4, 2, 512)):
        identity = _identity(heads, kv_heads, window)
        records.append(
            {
                "model": MODEL,
                "op_family": "attention",
                "structural": {
                    "identity": identity,
                    "axes": {
                        "attn_dtype": "bfloat16",
                        "head_dim": 128,
                        "kv_cache_dtype": "fp8",
                        "num_heads": heads,
                        "num_key_value_heads": kv_heads,
                        "phase": "context",
                        "window_size": window,
                    },
                },
            }
        )
    inventory_path = tmp_path / "coverage.json"
    inventory_path.write_text(json.dumps({"coverage_keys": {MODEL: records}}))
    return {
        "checkpoint": checkpoint_path,
        "parquet": parquet_path,
        "csv": csv_path,
        "raw_summary": raw_summary_path,
        "inventory": inventory_path,
    }


def test_attention_manifest_builds_workload_bijection_independent_of_checkpoint_order(tmp_path):
    paths = _write_input_fixture(tmp_path)
    manifest_path = build_attention_collection_artifacts(
        task_root=tmp_path,
        model=MODEL,
        attention_phase="context",
        checkpoint_path=paths["checkpoint"],
        parquet_path=paths["parquet"],
        csv_path=paths["csv"],
        raw_summary_path=paths["raw_summary"],
        coverage_inventory_path=paths["inventory"],
        artifact_prefix="attention_context",
    )

    report = validate_collection_manifest(manifest_path)
    assert report["admissible"] is True
    assert report["expected_invocation_count"] == 2
    reconciliation = json.loads((tmp_path / "attention_context_reconciliation.json").read_text())
    row_index_by_id = {entry["invocation_id"]: entry["row_index"] for entry in reconciliation["invocations"]}
    assert row_index_by_id["vllm.attention_context:run_attention_torch:[1, 8, 2, 1, 128, True, True, 0]"] == 1
    assert row_index_by_id["vllm.attention_context:run_attention_torch:[2, 16, 4, 2, 128, True, True, 512]"] == 0


def test_attention_manifest_rejects_missing_parquet_workload(tmp_path):
    paths = _write_input_fixture(tmp_path)
    checkpoint = json.loads(paths["checkpoint"].read_text())
    checkpoint["done"][0] = "vllm.attention_context:run_attention_torch:[1, 9, 2, 1, 128, True, True, 0]"
    paths["checkpoint"].write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match=r"checkpoint.*parquet|bijection"):
        build_attention_collection_artifacts(
            task_root=tmp_path,
            model=MODEL,
            attention_phase="context",
            checkpoint_path=paths["checkpoint"],
            parquet_path=paths["parquet"],
            csv_path=paths["csv"],
            raw_summary_path=paths["raw_summary"],
            coverage_inventory_path=paths["inventory"],
            artifact_prefix="attention_context",
        )
