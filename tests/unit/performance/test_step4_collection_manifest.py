"""Task-local P5/P7 collection manifest contract tests."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.performance.step4_collection_manifest import validate_collection_manifest

pytestmark = pytest.mark.unit


MODEL = "stepfun-ai/Step4-Pro-V3"


def _identity(window_size):
    axes = json.dumps({"phase": "context", "window_size": window_size}, sort_keys=True, separators=(",", ":"))
    return f"{MODEL}:attention:vllm:0.19.0:h800_sxm:{axes}"


IDENTITY_1 = _identity(0)
IDENTITY_2 = _identity(512)


def _parquet_row_fingerprint(path: Path, row_index: int) -> str:
    row = pq.read_table(path).slice(row_index, 1).to_pylist()[0]
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_fixture(tmp_path: Path, *, status_2: str = "unsupported") -> Path:
    measured_path = tmp_path / "perf_attention.parquet"
    pq.write_table(pa.Table.from_pylist([{"tokens": 1, "latency": 0.1}]), measured_path)
    measured_bytes = measured_path.read_bytes()
    (tmp_path / "skip.json").write_text('{"reason": "runtime kernel unsupported", "source": "worker"}\n')
    measured_inventory_bytes = json.dumps({"identities": [IDENTITY_1]}, sort_keys=True).encode()
    (tmp_path / "measured_keys.json").write_bytes(measured_inventory_bytes)
    status_counts = {"measured": 1, status_2: 1}
    (tmp_path / "collection_summary_vllm.json").write_text(
        json.dumps(
            {
                "summary": {
                    "backend": "vllm",
                    "version": "0.19.0",
                    "total_errors": 0,
                    "model": MODEL,
                    "system": "h800_sxm",
                    "phase": "full",
                    "op_family": "attention",
                    "attention_phase": "context",
                    "expected_invocation_count": 2,
                    "terminal_outcome_count": 2,
                    "structural_identity_count": 2,
                    "measured_count": status_counts.get("measured", 0),
                    "unsupported_count": status_counts.get("unsupported", 0),
                    "expected_failed_count": status_counts.get("expected_failed", 0),
                    "failed_count": status_counts.get("failed", 0),
                },
                "errors": [],
            }
        )
    )
    inventory = {
        "models": [MODEL],
        "coverage_keys": {
            MODEL: [
                {
                    "model": MODEL,
                    "op_family": "attention",
                    "structural": {"identity": IDENTITY_1, "axes": {"phase": "context", "window_size": 0}},
                },
                {
                    "model": MODEL,
                    "op_family": "attention",
                    "structural": {"identity": IDENTITY_2, "axes": {"phase": "context", "window_size": 512}},
                },
            ]
        },
    }
    coverage_bytes = json.dumps(inventory, sort_keys=True).encode()
    (tmp_path / "coverage.json").write_bytes(coverage_bytes)
    manifest = {
        "schema_version": "step4-collection-manifest-v1",
        "backend": "vllm",
        "version": "0.19.0",
        "system": "h800_sxm",
        "device": "h800_sxm",
        "model": MODEL,
        "phase": "full",
        "scope": {"op_family": "attention", "attention_phase": "context"},
        "coverage_inventory_path": "coverage.json",
        "coverage_inventory_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "measured_key_inventory_path": "measured_keys.json",
        "measured_key_inventory_sha256": hashlib.sha256(measured_inventory_bytes).hexdigest(),
        "checkpoint_path": "checkpoint.json",
        "collection_summary_path": "collection_summary_vllm.json",
        "expected_invocations": [
            {"invocation_id": "attention-0", "op_family": "attention", "structural_identity": IDENTITY_1},
            {"invocation_id": "attention-512", "op_family": "attention", "structural_identity": IDENTITY_2},
        ],
        "outcomes": [
            {
                "invocation_id": "attention-0",
                "status": "measured",
                "output_path": "perf_attention.parquet",
                "sha256": hashlib.sha256(measured_bytes).hexdigest(),
            },
            {
                "invocation_id": "attention-512",
                "status": status_2,
                "reason": "runtime kernel unsupported",
                "evidence_path": "skip.json",
            },
        ],
    }
    checkpoint = {
        "schema": "collector-resume-v1",
        "backend": "vllm",
        "module": "vllm.attention_context",
        "run_func": "run_attention_torch",
        "done": ["attention-0"],
        "failed": [],
        "expected_failed": ["attention-512"] if status_2 == "expected_failed" else [],
        "unsupported": ["attention-512"] if status_2 == "unsupported" else [],
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _write_repeated_identity_fixture(tmp_path: Path, *, measured_extra: bool = False) -> Path:
    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["expected_invocations"][0]["workload"] = {"tokens": 1}
    payload["expected_invocations"][1]["workload"] = {"tokens": 512}
    payload["expected_invocations"].append(
        {
            "invocation_id": "attention-0-token-2",
            "op_family": "attention",
            "structural_identity": IDENTITY_1,
            "workload": {"tokens": 2},
        }
    )
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    if measured_extra:
        parquet_path = tmp_path / "perf_attention.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"tokens": 1, "latency": 0.1}, {"tokens": 2, "latency": 0.2}]),
            parquet_path,
        )
        output_sha256 = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        payload["outcomes"][0]["sha256"] = output_sha256
        payload["outcomes"].append(
            {
                "invocation_id": "attention-0-token-2",
                "status": "measured",
                "output_path": "perf_attention.parquet",
                "sha256": output_sha256,
            }
        )
        checkpoint["done"].append("attention-0-token-2")
        measured_count = 2
        unsupported_count = 1
    else:
        payload["outcomes"].append(
            {
                "invocation_id": "attention-0-token-2",
                "status": "unsupported",
                "reason": "runtime allocation is not available in this fixture",
                "evidence_path": "skip.json",
            }
        )
        checkpoint["unsupported"].append("attention-0-token-2")
        measured_count = 1
        unsupported_count = 2
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))

    summary = json.loads((tmp_path / "collection_summary_vllm.json").read_text())
    summary["summary"].update(
        {
            "expected_invocation_count": 3,
            "terminal_outcome_count": 3,
            "structural_identity_count": 2,
            "measured_count": measured_count,
            "unsupported_count": unsupported_count,
        }
    )
    (tmp_path / "collection_summary_vllm.json").write_text(json.dumps(summary))

    reconciliation_invocations = [
        {
            "invocation_id": "attention-0",
            "structural_identity": IDENTITY_1,
            "workload": {"tokens": 1},
            "output_path": "perf_attention.parquet",
            "row_index": 0,
            "row_fingerprint": _parquet_row_fingerprint(tmp_path / "perf_attention.parquet", 0),
        },
        {
            "invocation_id": "attention-512",
            "structural_identity": IDENTITY_2,
            "workload": {"tokens": 512},
        },
        {
            "invocation_id": "attention-0-token-2",
            "structural_identity": IDENTITY_1,
            "workload": {"tokens": 2},
        },
    ]
    if measured_extra:
        reconciliation_invocations[2].update(
            {
                "output_path": "perf_attention.parquet",
                "row_index": 1,
                "row_fingerprint": _parquet_row_fingerprint(tmp_path / "perf_attention.parquet", 1),
            }
        )
    reconciliation = {
        "schema": "step4-invocation-reconciliation-v1",
        "invocations": reconciliation_invocations,
    }
    reconciliation_bytes = json.dumps(reconciliation, sort_keys=True).encode()
    (tmp_path / "reconciliation.json").write_bytes(reconciliation_bytes)
    payload["invocation_reconciliation_path"] = "reconciliation.json"
    payload["invocation_reconciliation_sha256"] = hashlib.sha256(reconciliation_bytes).hexdigest()
    path.write_text(json.dumps(payload))
    return path


def _rewrite_reconciliation(path: Path, mutate) -> None:
    payload = json.loads(path.read_text())
    reconciliation_path = path.parent / payload["invocation_reconciliation_path"]
    reconciliation = json.loads(reconciliation_path.read_text())
    mutate(reconciliation)
    reconciliation_bytes = json.dumps(reconciliation, sort_keys=True).encode()
    reconciliation_path.write_bytes(reconciliation_bytes)
    payload["invocation_reconciliation_sha256"] = hashlib.sha256(reconciliation_bytes).hexdigest()
    path.write_text(json.dumps(payload))


def test_collection_manifest_accepts_measured_and_evidenced_unsupported(tmp_path):
    report = validate_collection_manifest(_write_fixture(tmp_path))
    assert report["admissible"] is True
    assert report["outcome_counts"] == {"measured": 1, "unsupported": 1}
    assert report["structural_identity_count"] == 2


def test_collection_manifest_accepts_expected_failed_with_checkpoint_evidence(tmp_path):
    report = validate_collection_manifest(_write_fixture(tmp_path, status_2="expected_failed"))
    assert report["admissible"] is True
    assert report["measured_count"] == 1
    assert report["expected_failed_count"] == 1


def test_collection_manifest_accepts_real_v3_gemm_full_checkpoint_contract():
    manifest = Path("task_memory/task_2026-07-28_step4_pro_v3_v4_op_profiling/p5_v3_gemm_full_collection_manifest.json")
    report = validate_collection_manifest(manifest)
    assert report["admissible"] is True
    assert report["op_family"] == "gemm"
    assert report["expected_invocation_count"] == 3774
    assert report["measured_count"] == 3774


def test_collection_manifest_rejects_missing_terminal_outcome(tmp_path):
    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["outcomes"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing terminal outcome"):
        validate_collection_manifest(path)


def test_collection_manifest_rejects_coverage_inventory_hash_mismatch(tmp_path):
    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["coverage_inventory_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="coverage inventory sha256"):
        validate_collection_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "other"),
        ("module", "vllm.attention_generation"),
        ("run_func", "other"),
    ],
)
def test_collection_manifest_rejects_checkpoint_identity_mismatch(tmp_path, field, value):
    path = _write_fixture(tmp_path)
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    checkpoint[field] = value
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match=f"checkpoint {field}"):
        validate_collection_manifest(path)


def test_collection_manifest_attention_phase_filters_inventory(tmp_path):
    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    coverage_path = tmp_path / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    generation_identity = _identity(0).replace('"phase":"context"', '"phase":"generation"')
    coverage["coverage_keys"][MODEL].append(
        {
            "model": MODEL,
            "op_family": "attention",
            "structural": {
                "identity": generation_identity,
                "axes": {"phase": "generation", "window_size": 0},
            },
        }
    )
    coverage_bytes = json.dumps(coverage, sort_keys=True).encode()
    coverage_path.write_bytes(coverage_bytes)
    payload["coverage_inventory_sha256"] = hashlib.sha256(coverage_bytes).hexdigest()
    path.write_text(json.dumps(payload))

    report = validate_collection_manifest(path)
    assert report["structural_identity_count"] == 2


def test_collection_manifest_rejects_duplicate_identity_and_path_traversal(tmp_path):
    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["expected_invocations"][1]["structural_identity"] = IDENTITY_1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="structural identities do not match inventory"):
        validate_collection_manifest(path)

    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["coverage_inventory_path"] = "../coverage.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="path must stay"):
        validate_collection_manifest(path)


def test_collection_manifest_accepts_multiple_workloads_per_structural_identity(tmp_path):
    report = validate_collection_manifest(_write_repeated_identity_fixture(tmp_path))
    assert report["admissible"] is True
    assert report["expected_invocation_count"] == 3
    assert report["structural_identity_count"] == 2
    assert report["unsupported_count"] == 2


def test_collection_manifest_requires_reconciliation_for_repeated_identity(tmp_path):
    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["expected_invocations"].append(
        {
            "invocation_id": "attention-0-token-2",
            "op_family": "attention",
            "structural_identity": IDENTITY_1,
            "workload": {"tokens": 2},
        }
    )
    payload["outcomes"].append(
        {
            "invocation_id": "attention-0-token-2",
            "status": "unsupported",
            "reason": "runtime allocation is not available in this fixture",
            "evidence_path": "skip.json",
        }
    )
    path.write_text(json.dumps(payload))
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    checkpoint["unsupported"].append("attention-0-token-2")
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    summary = json.loads((tmp_path / "collection_summary_vllm.json").read_text())
    summary["summary"].update(
        {
            "expected_invocation_count": 3,
            "terminal_outcome_count": 3,
            "structural_identity_count": 2,
            "unsupported_count": 2,
        }
    )
    (tmp_path / "collection_summary_vllm.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="reconciliation"):
        validate_collection_manifest(path)


def test_collection_manifest_rejects_out_of_range_reconciliation_row(tmp_path):
    path = _write_repeated_identity_fixture(tmp_path)
    _rewrite_reconciliation(path, lambda value: value["invocations"][0].update({"row_index": 1}))
    with pytest.raises(ValueError, match="row_index"):
        validate_collection_manifest(path)


def test_collection_manifest_rejects_duplicate_reconciliation_row(tmp_path):
    path = _write_repeated_identity_fixture(tmp_path, measured_extra=True)

    def duplicate_first_row(value):
        value["invocations"][2].update(
            {
                "row_index": 0,
                "row_fingerprint": value["invocations"][0]["row_fingerprint"],
            }
        )

    _rewrite_reconciliation(path, duplicate_first_row)
    with pytest.raises(ValueError, match=r"duplicate.*row_index|cover"):
        validate_collection_manifest(path)


def test_collection_manifest_rejects_forged_reconciliation_fingerprint(tmp_path):
    path = _write_repeated_identity_fixture(tmp_path)
    _rewrite_reconciliation(path, lambda value: value["invocations"][0].update({"row_fingerprint": "0" * 64}))
    with pytest.raises(ValueError, match="fingerprint"):
        validate_collection_manifest(path)


def test_collection_manifest_rejects_reconciliation_workload_mismatch(tmp_path):
    path = _write_repeated_identity_fixture(tmp_path)
    _rewrite_reconciliation(path, lambda value: value["invocations"][0].update({"workload": {"tokens": 99}}))
    with pytest.raises(ValueError, match="workload"):
        validate_collection_manifest(path)


def test_collection_manifest_rejects_measured_checkpoint_or_output_mismatch(tmp_path):
    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["outcomes"][0]["output_path"] = "missing.parquet"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="measured output_path"):
        validate_collection_manifest(path)

    path = _write_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["outcomes"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="output sha256 mismatch"):
        validate_collection_manifest(path)

    path = _write_fixture(tmp_path)
    (tmp_path / "measured_keys.json").write_text(json.dumps({"identities": [IDENTITY_2]}))
    with pytest.raises(ValueError, match="measured-key inventory identities"):
        validate_collection_manifest(path)

    path = _write_fixture(tmp_path)
    (tmp_path / "measured_keys.json").write_text(json.dumps({"identities": [IDENTITY_1, IDENTITY_1]}))
    with pytest.raises(ValueError, match="duplicate identities"):
        validate_collection_manifest(path)

    path = _write_fixture(tmp_path)
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    checkpoint["done"] = []
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="checkpoint done"):
        validate_collection_manifest(path)


def test_collection_manifest_rejects_failed_outcome_or_summary_errors(tmp_path):
    path = _write_fixture(tmp_path, status_2="failed")
    payload = json.loads(path.read_text())
    payload["outcomes"][1]["evidence_path"] = "skip.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="failed outcome"):
        validate_collection_manifest(path)

    path = _write_fixture(tmp_path)
    summary = json.loads((tmp_path / "collection_summary_vllm.json").read_text())
    summary["summary"]["total_errors"] = 1
    summary["errors"] = [{"module": "attention", "error_type": "RuntimeError"}]
    (tmp_path / "collection_summary_vllm.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="collection summary reports errors"):
        validate_collection_manifest(path)

    path = _write_fixture(tmp_path)
    summary = json.loads((tmp_path / "collection_summary_vllm.json").read_text())
    summary["summary"]["model"] = "stepfun-ai/Step4-Pro-V4"
    (tmp_path / "collection_summary_vllm.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="collection summary model"):
        validate_collection_manifest(path)


@pytest.mark.parametrize("model", ["stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"])
def test_collection_manifest_matches_real_step4_planned_inventory(tmp_path, model):
    inventory_source = Path(
        "task_memory/task_2026-07-28_step4_pro_v3_v4_op_profiling/step4_pro_v3_v4_coverage_inventory.json"
    )
    inventory_path = tmp_path / "coverage.json"
    shutil.copyfile(inventory_source, inventory_path)
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_path.read_text())
    records = inventory["coverage_keys"][model]
    measured_bytes = b"one measured row"
    (tmp_path / "measured.parquet").write_bytes(measured_bytes)
    (tmp_path / "unsupported.json").write_text('{"reason": "pending runtime allocation"}\n')
    measured_inventory_bytes = json.dumps(
        {"identities": [records[0]["structural"]["identity"]]}, sort_keys=True
    ).encode()
    (tmp_path / "measured_keys.json").write_bytes(measured_inventory_bytes)
    unsupported_count = len(records) - 1
    (tmp_path / "collection_summary_vllm.json").write_text(
        json.dumps(
            {
                "summary": {
                    "backend": "vllm",
                    "version": "0.19.0",
                    "total_errors": 0,
                    "model": model,
                    "system": "h800_sxm",
                    "phase": "full",
                    "op_family": "all",
                    "expected_invocation_count": len(records),
                    "terminal_outcome_count": len(records),
                    "structural_identity_count": len(records),
                    "measured_count": 1,
                    "unsupported_count": unsupported_count,
                    "expected_failed_count": 0,
                    "failed_count": 0,
                },
                "errors": [],
            }
        )
    )
    expected = [
        {
            "invocation_id": f"case-{index}",
            "op_family": record["op_family"],
            "structural_identity": record["structural"]["identity"],
        }
        for index, record in enumerate(records)
    ]
    outcomes = [
        {
            "invocation_id": "case-0",
            "status": "measured",
            "output_path": "measured.parquet",
            "sha256": hashlib.sha256(measured_bytes).hexdigest(),
        },
        *[
            {
                "invocation_id": f"case-{index}",
                "status": "unsupported",
                "reason": "runtime allocation is not available in this static fixture",
                "evidence_path": "unsupported.json",
            }
            for index in range(1, len(records))
        ],
    ]
    (tmp_path / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "collector-resume-v1",
                "backend": "vllm",
                "done": ["case-0"],
                "failed": [],
                "expected_failed": [],
                "unsupported": [f"case-{index}" for index in range(1, len(records))],
            }
        )
    )
    manifest = {
        "schema_version": "step4-collection-manifest-v1",
        "backend": "vllm",
        "version": "0.19.0",
        "system": "h800_sxm",
        "device": "h800_sxm",
        "model": model,
        "phase": "full",
        "scope": {"op_family": "all"},
        "coverage_inventory_path": "coverage.json",
        "coverage_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "measured_key_inventory_path": "measured_keys.json",
        "measured_key_inventory_sha256": hashlib.sha256(measured_inventory_bytes).hexdigest(),
        "checkpoint_path": "checkpoint.json",
        "collection_summary_path": "collection_summary_vllm.json",
        "expected_invocations": expected,
        "outcomes": outcomes,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report = validate_collection_manifest(manifest_path)
    assert report["structural_identity_count"] == len(records)
    assert report["measured_count"] == 1
    assert report["unsupported_count"] == len(records) - 1
