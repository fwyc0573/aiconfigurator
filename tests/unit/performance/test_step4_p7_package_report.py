"""Unit tests for the Step4 P7 consumer/package report builder."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.performance.step4_p7_package_report import _artifact_binding, build_reports

pytestmark = pytest.mark.unit

MODELS = ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4")
ISLS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
TOPOLOGIES = (
    (1, 1, 1, 1, 1, 1),
    (1, 1, 2, 1, 2, 1),
    (2, 1, 1, 1, 2, 1),
    (1, 1, 4, 1, 4, 1),
    (2, 1, 2, 1, 4, 1),
    (4, 1, 1, 1, 4, 1),
    (1, 1, 8, 1, 8, 1),
    (2, 1, 4, 1, 8, 1),
    (4, 1, 2, 1, 8, 1),
    (1, 1, 16, 1, 16, 1),
    (2, 1, 8, 1, 16, 1),
    (4, 1, 4, 1, 16, 1),
    (1, 1, 32, 1, 32, 1),
    (2, 1, 16, 1, 32, 1),
    (4, 1, 8, 1, 32, 1),
    (1, 1, 64, 1, 64, 1),
    (2, 1, 32, 1, 64, 1),
    (4, 1, 16, 1, 64, 1),
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coverage_counts(required_count: int) -> dict[str, int]:
    return {
        "required_count": required_count,
        "measured_count": required_count,
        "missing_count": 0,
        "duplicate_count": 0,
        "unassigned_count": 0,
    }


def _communication_audit() -> list[dict[str, Any]]:
    audit = []
    for tp, _, dp, moe_tp, moe_ep, cp in TOPOLOGIES:
        world_size = tp * dp * cp
        queries = []
        if tp > 1:
            queries.append({"op": "custom_allreduce", "rank": tp})
        if world_size > 1:
            queries.extend({"op": op, "rank": moe_tp * moe_ep} for op in ("nccl_all_gather", "nccl_reduce_scatter"))
        audit.append({"status": "runnable", "queries": queries})
    return audit


def _success_outcome(model: str, isl: int, index: int) -> dict[str, Any]:
    if model == MODELS[0]:
        config = {
            "tp": 4,
            "pp": 1,
            "dp": 16,
            "moe_tp": 1,
            "moe_ep": 64,
            "cp": 1,
            "batch_size": 495 - index,
            "ctx_tokens": isl,
        }
    else:
        config = {
            "tp": 2,
            "pp": 1,
            "dp": 32,
            "moe_tp": 1,
            "moe_ep": 64,
            "cp": 1,
            "batch_size": 256 - index,
            "ctx_tokens": max(2048, isl),
        }
    throughput = 160.0 / (index + 1)
    return {
        "model": model,
        "isl": isl,
        "status": "success",
        "throughput_per_used_gpu": throughput,
        "cluster_tokens_per_second": throughput * 64,
        "deployment_gpus": 64,
        "ttft_ms": 5000.0 + index,
        "tpot_ms": 1000.0 + index,
        "selected_config": config,
        "per_ops_data": {"step": {"op": 1.0}},
        "per_ops_source": {"step": {"op": "silicon"}},
    }


def _matrix() -> dict[str, Any]:
    outcomes = []
    for model in MODELS:
        for index, isl in enumerate(ISLS):
            if index < 4:
                outcomes.append(_success_outcome(model, isl, index))
                continue
            status = "memory_infeasible"
            if model == MODELS[1] and index in (4, 5):
                status = "sla_infeasible"
            outcomes.append({"model": model, "isl": isl, "status": status, "reason": status})
    return {
        "schema": "step4-profiled-agg-matrix-v1",
        "status": "completed",
        "models": list(MODELS),
        "isls": list(ISLS),
        "point_count": 16,
        "status_counts": {"success": 8, "memory_infeasible": 6, "sla_infeasible": 2},
        "contract": {
            "backend": "vllm",
            "backend_version": "0.19.0",
            "database_mode": "SILICON",
            "nextn": 0,
            "osl": 1024,
            "pareto_sweep": False,
            "serving_mode": "agg",
            "system": "h800_sxm",
            "total_gpus": 64,
            "tpot_ms": 50000,
            "ttft_ms": 10000,
        },
        "runnable_topologies": [list(topology) for topology in TOPOLOGIES],
        "invalid_cross_node_custom_allreduce_topologies": [],
        "terminal_outcomes": outcomes,
    }


@pytest.fixture
def package_fixture(tmp_path: Path) -> dict[str, Path]:
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    identities = [f"{MODELS[0]}:identity-{index:03d}" for index in range(84)]
    identities.extend(f"{MODELS[1]}:identity-{index:03d}" for index in range(96))
    measured_path = task_dir / "step4_pro_v3_v4_measured_keys.json"
    _write_json(measured_path, {"identities": identities})

    coverage = {
        "status": "validated",
        "measured_identity_count": 180,
        "coverage_summary": {
            MODELS[0]: {
                "attention": _coverage_counts(12),
                "gemm": _coverage_counts(51),
                "moe": _coverage_counts(7),
                "communication": _coverage_counts(14),
            },
            MODELS[1]: {
                "attention": _coverage_counts(12),
                "gemm": _coverage_counts(63),
                "moe": _coverage_counts(7),
                "communication": _coverage_counts(14),
            },
        },
        "coverage_keys": {
            MODELS[0]: [{"structural": {"identity": identity}} for identity in identities[:84]],
            MODELS[1]: [{"structural": {"identity": identity}} for identity in identities[84:]],
        },
        "provenance": {
            "measured_key_inventory": {
                "path": measured_path.name,
                "sha256": _sha256(measured_path),
            }
        },
        "communication_topology_audit": {model: _communication_audit() for model in MODELS},
    }
    _write_json(task_dir / "step4_pro_v3_v4_coverage_inventory.json", coverage)
    _write_json(task_dir / "p8_isl_matrix_20260801.json", _matrix())

    extension_path = tmp_path / "rust-extension.so"
    extension_path.write_bytes(b"rust-extension")
    _write_json(
        task_dir / "p7_rust_consumer_evidence.json",
        {
            "schema": "step4-p7-rust-consumer-evidence-v1",
            "status": "validated",
            "purpose": "consumer coverage/execution evidence, not Python/Rust numeric parity",
            "engine_step_backend": "rust",
            "extension_path": extension_path.name,
            "extension_sha256": _sha256(extension_path),
            "models": {
                MODELS[0]: {
                    "coverage_gate": True,
                    "source": "rust",
                    "memory_gib": 79.0,
                    "selected_config": {
                        "tp": 4,
                        "pp": 1,
                        "dp": 16,
                        "moe_tp": 1,
                        "moe_ep": 64,
                        "batch_size": 495,
                        "ctx_tokens": 1024,
                    },
                    "throughput_per_used_gpu": 190.0,
                    "tokens_per_second": 12160.0,
                    "ttft_ms": 1900.0,
                    "tpot_ms": 648.0,
                },
                MODELS[1]: {
                    "coverage_gate": True,
                    "source": "rust",
                    "memory_gib": 72.0,
                    "selected_config": {
                        "tp": 2,
                        "pp": 1,
                        "dp": 32,
                        "moe_tp": 1,
                        "moe_ep": 64,
                        "batch_size": 256,
                        "ctx_tokens": 2048,
                    },
                    "throughput_per_used_gpu": 201.0,
                    "tokens_per_second": 12864.0,
                    "ttft_ms": 4600.0,
                    "tpot_ms": 630.0,
                },
            },
        },
    )

    canonical_path = tmp_path / "canonical.bin"
    canonical_path.write_bytes(b"canonical")
    collection_manifest_path = task_dir / "collection.json"
    _write_json(collection_manifest_path, {"status": "validated"})
    _write_json(
        task_dir / "p7_canonical_package_manifest.json",
        {
            "schema": "step4-p7-canonical-package-v1",
            "status": "blocked",
            "canonical_collection": False,
            "namespace_promotion_staged": False,
            "canonical_artifacts": [{"path": canonical_path.name, "sha256": _sha256(canonical_path), "rows": 1}],
            "collection_manifests": [
                {
                    "path": collection_manifest_path.name,
                    "sha256": _sha256(collection_manifest_path),
                    "status": "validated",
                }
            ],
        },
    )
    return {"repo_root": tmp_path, "task_dir": task_dir}


def test_artifact_binding_resolves_relative_output_against_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    output_path = task_dir / "consumer.json"
    output_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    binding = _artifact_binding(Path("task/consumer.json"), task_dir=task_dir)

    assert binding["path"] == "consumer.json"


def test_cli_rebuilds_validated_consumer_report_and_review_pending_package(package_fixture: dict[str, Path]):
    task_dir = package_fixture["task_dir"]
    consumer_output = task_dir / "consumer.new.json"
    package_output = task_dir / "package.new.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.step4_p7_package_report",
            "--repo-root",
            str(package_fixture["repo_root"]),
            "--task-dir",
            str(task_dir),
            "--consumer-output",
            str(consumer_output),
            "--package-output",
            str(package_output),
        ],
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    consumer = _load_json(consumer_output)
    package = _load_json(package_output)
    assert consumer["status"] == "validated"
    assert consumer["measured_identity_count"] == 180
    assert consumer["communication"]["custom_allreduce"] == {
        "required_ranks": [2, 4],
        "extra_measured_ranks": [8],
    }
    assert consumer["communication"]["nccl"] == {
        "required_ranks": [2, 4, 8, 16, 32, 64],
        "extra_measured_ranks": [48],
    }
    assert consumer["matrix"]["status_counts"] == {
        "success": 8,
        "memory_infeasible": 6,
        "sla_infeasible": 2,
        "data_unavailable": 0,
        "error": 0,
    }
    assert consumer["matrix"]["runnable_topology_count"] == 18
    assert consumer["package_integrity"] == {
        "canonical_artifacts_checked": 1,
        "collection_manifests_checked": 1,
        "total_checked": 2,
        "hash_error_count": 0,
    }
    assert consumer["rust_consumer"]["models"][MODELS[0]]["coverage_gate"] is True
    assert package["status"] == "pending_independent_review"
    assert package["canonical_collection"] is False
    assert package["namespace_promotion_staged"] is True
    assert package["consumer_exit_report"]["sha256"] == _sha256(consumer_output)
    assert package["p8_matrix"]["status_counts"]["success"] == 8
    assert package["communication_disposition"]["custom_allreduce"].startswith("required attention-TP ranks 2/4")


def test_coverage_rejects_duplicate_measured_identity(package_fixture: dict[str, Path]):
    task_dir = package_fixture["task_dir"]
    measured_path = task_dir / "step4_pro_v3_v4_measured_keys.json"
    measured = _load_json(measured_path)
    measured["identities"][-1] = measured["identities"][0]
    _write_json(measured_path, measured)
    coverage_path = task_dir / "step4_pro_v3_v4_coverage_inventory.json"
    coverage = _load_json(coverage_path)
    coverage["provenance"]["measured_key_inventory"]["sha256"] = _sha256(measured_path)
    _write_json(coverage_path, coverage)

    with pytest.raises(ValueError, match="measured identities must be unique"):
        build_reports(**package_fixture)


def test_coverage_rejects_foreign_measured_identity(package_fixture: dict[str, Path]):
    task_dir = package_fixture["task_dir"]
    measured_path = task_dir / "step4_pro_v3_v4_measured_keys.json"
    measured = _load_json(measured_path)
    measured["identities"][-1] = "foreign:model:identity"
    _write_json(measured_path, measured)
    coverage_path = task_dir / "step4_pro_v3_v4_coverage_inventory.json"
    coverage = _load_json(coverage_path)
    coverage["provenance"]["measured_key_inventory"]["sha256"] = _sha256(measured_path)
    _write_json(coverage_path, coverage)

    with pytest.raises(ValueError, match="measured and coverage identity sets differ"):
        build_reports(**package_fixture)


def test_coverage_rejects_stale_measured_key_provenance_hash(package_fixture: dict[str, Path]):
    task_dir = package_fixture["task_dir"]
    coverage_path = task_dir / "step4_pro_v3_v4_coverage_inventory.json"
    coverage = _load_json(coverage_path)
    coverage["provenance"]["measured_key_inventory"]["sha256"] = "0" * 64
    _write_json(coverage_path, coverage)

    with pytest.raises(ValueError, match="measured-key provenance SHA-256 mismatch"):
        build_reports(**package_fixture)


def test_coverage_rejects_duplicate_coverage_identity(package_fixture: dict[str, Path]):
    task_dir = package_fixture["task_dir"]
    coverage_path = task_dir / "step4_pro_v3_v4_coverage_inventory.json"
    coverage = _load_json(coverage_path)
    coverage["coverage_keys"][MODELS[1]][-1]["structural"]["identity"] = coverage["coverage_keys"][MODELS[0]][0][
        "structural"
    ]["identity"]
    _write_json(coverage_path, coverage)

    with pytest.raises(ValueError, match="coverage identities must be unique"):
        build_reports(**package_fixture)


def _rewrite_matrix(package_fixture: dict[str, Path], matrix: dict[str, Any]) -> None:
    _write_json(package_fixture["task_dir"] / "p8_isl_matrix_20260801.json", matrix)


def test_matrix_rejects_duplicate_outcome(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][-1] = matrix["terminal_outcomes"][0]
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="duplicate P8 matrix outcome"):
        build_reports(**package_fixture)


def test_matrix_rejects_missing_outcome(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"].pop()
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="exactly 16 terminal outcomes"):
        build_reports(**package_fixture)


def test_matrix_rejects_gap_throughput(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][4]["throughput_per_used_gpu"] = 1.0
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="terminal gap must not carry throughput"):
        build_reports(**package_fixture)


def test_matrix_rejects_gap_selected_config(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][4]["selected_config"] = {"tp": 1}
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="terminal gap must not carry selected_config"):
        build_reports(**package_fixture)


def test_matrix_rejects_gap_per_op_evidence(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][4]["per_ops_data"] = {"fake": 1.0}
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="terminal gap must not carry per-op evidence"):
        build_reports(**package_fixture)


def test_matrix_rejects_illegal_selected_topology(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][0]["selected_config"]["tp"] = 3
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="selected topology is not runnable"):
        build_reports(**package_fixture)


def test_matrix_rejects_nan_throughput(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][0]["throughput_per_used_gpu"] = float("nan")
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="throughput_per_used_gpu must be positive and finite"):
        build_reports(**package_fixture)


def test_matrix_rejects_sla_violation(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][0]["ttft_ms"] = 10000.001
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="success point violates TTFT SLA"):
        build_reports(**package_fixture)


def test_matrix_rejects_cluster_throughput_mismatch(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["terminal_outcomes"][0]["cluster_tokens_per_second"] += 1.0
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="cluster throughput arithmetic mismatch"):
        build_reports(**package_fixture)


def test_matrix_rejects_declared_status_count_mismatch(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["status_counts"] = {"success": 7, "memory_infeasible": 7, "sla_infeasible": 2}
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="status counts disagree with terminal outcomes"):
        build_reports(**package_fixture)


def test_matrix_rejects_fixed_contract_mismatch(package_fixture: dict[str, Path]):
    matrix = _matrix()
    matrix["contract"]["total_gpus"] = 32
    _rewrite_matrix(package_fixture, matrix)

    with pytest.raises(ValueError, match="P8 contract total_gpus must be 64"):
        build_reports(**package_fixture)


def _rewrite_rust_evidence(package_fixture: dict[str, Path], evidence: dict[str, Any]) -> None:
    _write_json(package_fixture["task_dir"] / "p7_rust_consumer_evidence.json", evidence)


@pytest.mark.parametrize(
    "field",
    ("memory_gib", "throughput_per_used_gpu", "tokens_per_second", "ttft_ms", "tpot_ms"),
)
def test_rust_evidence_rejects_non_finite_numeric_field(package_fixture: dict[str, Path], field: str):
    rust_path = package_fixture["task_dir"] / "p7_rust_consumer_evidence.json"
    evidence = _load_json(rust_path)
    evidence["models"][MODELS[0]][field] = float("nan")
    _rewrite_rust_evidence(package_fixture, evidence)

    with pytest.raises(ValueError, match=rf"Rust {field} must be positive and finite"):
        build_reports(**package_fixture)


def test_rust_evidence_rejects_stale_extension_hash(package_fixture: dict[str, Path]):
    rust_path = package_fixture["task_dir"] / "p7_rust_consumer_evidence.json"
    evidence = _load_json(rust_path)
    evidence["extension_sha256"] = "0" * 64
    _rewrite_rust_evidence(package_fixture, evidence)

    with pytest.raises(ValueError, match="Rust extension SHA-256 mismatch"):
        build_reports(**package_fixture)


def test_rust_evidence_rejects_config_without_matching_p8_success(package_fixture: dict[str, Path]):
    rust_path = package_fixture["task_dir"] / "p7_rust_consumer_evidence.json"
    evidence = _load_json(rust_path)
    evidence["models"][MODELS[0]]["selected_config"]["batch_size"] = 1
    _rewrite_rust_evidence(package_fixture, evidence)

    with pytest.raises(ValueError, match="Rust selected config has no matching P8 success"):
        build_reports(**package_fixture)


def test_rust_evidence_rejects_non_validated_status(package_fixture: dict[str, Path]):
    rust_path = package_fixture["task_dir"] / "p7_rust_consumer_evidence.json"
    evidence = _load_json(rust_path)
    evidence["status"] = "pending"
    _rewrite_rust_evidence(package_fixture, evidence)

    with pytest.raises(ValueError, match="Rust evidence status must be validated"):
        build_reports(**package_fixture)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema", "wrong", "unexpected Rust evidence schema"),
        ("engine_step_backend", "python", "Rust evidence engine_step_backend must be rust"),
        ("purpose", "numeric parity", "unexpected Rust evidence purpose"),
    ),
)
def test_rust_evidence_rejects_wrong_top_level_contract(
    package_fixture: dict[str, Path], field: str, value: str, message: str
):
    rust_path = package_fixture["task_dir"] / "p7_rust_consumer_evidence.json"
    evidence = _load_json(rust_path)
    evidence[field] = value
    _rewrite_rust_evidence(package_fixture, evidence)

    with pytest.raises(ValueError, match=message):
        build_reports(**package_fixture)


def test_rust_evidence_rejects_non_rust_model_source(package_fixture: dict[str, Path]):
    rust_path = package_fixture["task_dir"] / "p7_rust_consumer_evidence.json"
    evidence = _load_json(rust_path)
    evidence["models"][MODELS[0]]["source"] = "python"
    _rewrite_rust_evidence(package_fixture, evidence)

    with pytest.raises(ValueError, match="Rust model source must be rust"):
        build_reports(**package_fixture)


def test_rust_evidence_rejects_sla_violation(package_fixture: dict[str, Path]):
    rust_path = package_fixture["task_dir"] / "p7_rust_consumer_evidence.json"
    evidence = _load_json(rust_path)
    evidence["models"][MODELS[0]]["tpot_ms"] = 50000.001
    _rewrite_rust_evidence(package_fixture, evidence)

    with pytest.raises(ValueError, match="Rust evidence violates TPOT SLA"):
        build_reports(**package_fixture)


MANDATORY_REMEDIATIONS = (
    "coverage_exact_set_and_provenance",
    "matrix_terminal_contract",
    "rust_evidence_contract",
    "documentation_state",
)


def _build_pending_artifacts(package_fixture: dict[str, Path]) -> tuple[Path, Path]:
    task_dir = package_fixture["task_dir"]
    consumer_path = task_dir / "p7_consumer_exit_report.json"
    package_path = task_dir / "p7_canonical_package_manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.step4_p7_package_report",
            "--repo-root",
            str(package_fixture["repo_root"]),
            "--task-dir",
            str(task_dir),
            "--consumer-output",
            str(consumer_path),
            "--package-output",
            str(package_path),
        ],
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return consumer_path, package_path


def _review_record(consumer_path: Path, package_path: Path, *, verdict: str = "WATCH") -> dict[str, Any]:
    return {
        "schema": "step4-p7-independent-review-v1",
        "reviewer": "/root/communication_contract_reviewer",
        "verdict": verdict,
        "critical_findings": 0,
        "required_remediations": list(MANDATORY_REMEDIATIONS),
        "remediations_verified": list(MANDATORY_REMEDIATIONS),
        "reviewed_artifacts": {
            "consumer_exit_report": {"path": consumer_path.name, "sha256": _sha256(consumer_path)},
            "pending_package": {"path": package_path.name, "sha256": _sha256(package_path)},
        },
    }


def _run_finalizer(
    package_fixture: dict[str, Path], consumer_path: Path, package_path: Path, review_path: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.step4_p7_finalize_package",
            "--repo-root",
            str(package_fixture["repo_root"]),
            "--task-dir",
            str(package_fixture["task_dir"]),
            "--consumer-report",
            str(consumer_path),
            "--package",
            str(package_path),
            "--review-record",
            str(review_path),
            "--output",
            str(package_fixture["task_dir"] / "finalized.json"),
        ],
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_finalizer_rejects_missing_reviewer_record(package_fixture: dict[str, Path]):
    consumer_path, package_path = _build_pending_artifacts(package_fixture)
    review_path = package_fixture["task_dir"] / "missing-review.json"

    result = _run_finalizer(package_fixture, consumer_path, package_path, review_path)

    assert result.returncode != 0
    assert "independent review record does not exist" in result.stderr


def test_finalizer_rejects_stale_reviewer_hash(package_fixture: dict[str, Path]):
    consumer_path, package_path = _build_pending_artifacts(package_fixture)
    review_path = package_fixture["task_dir"] / "review.json"
    review = _review_record(consumer_path, package_path)
    review["reviewed_artifacts"]["pending_package"]["sha256"] = "0" * 64
    _write_json(review_path, review)

    result = _run_finalizer(package_fixture, consumer_path, package_path, review_path)

    assert result.returncode != 0
    assert "reviewed pending package SHA-256 mismatch" in result.stderr


def test_finalizer_rejects_block_verdict(package_fixture: dict[str, Path]):
    consumer_path, package_path = _build_pending_artifacts(package_fixture)
    review_path = package_fixture["task_dir"] / "review.json"
    _write_json(review_path, _review_record(consumer_path, package_path, verdict="BLOCK"))

    result = _run_finalizer(package_fixture, consumer_path, package_path, review_path)

    assert result.returncode != 0
    assert "independent review verdict BLOCK prevents canonical admission" in result.stderr


def test_finalizer_rejects_unsupported_verdict(package_fixture: dict[str, Path]):
    consumer_path, package_path = _build_pending_artifacts(package_fixture)
    review_path = package_fixture["task_dir"] / "review.json"
    _write_json(review_path, _review_record(consumer_path, package_path, verdict="UNKNOWN"))

    result = _run_finalizer(package_fixture, consumer_path, package_path, review_path)

    assert result.returncode != 0
    assert "unsupported independent review verdict" in result.stderr


def test_finalizer_rejects_incomplete_remediation_verification(package_fixture: dict[str, Path]):
    consumer_path, package_path = _build_pending_artifacts(package_fixture)
    review_path = package_fixture["task_dir"] / "review.json"
    review = _review_record(consumer_path, package_path)
    review["remediations_verified"].pop()
    _write_json(review_path, review)

    result = _run_finalizer(package_fixture, consumer_path, package_path, review_path)

    assert result.returncode != 0
    assert "review remediation verification is incomplete" in result.stderr


@pytest.mark.parametrize("verdict", ("WATCH", "APPROVE"))
def test_finalizer_admits_fully_verified_review(package_fixture: dict[str, Path], verdict: str):
    consumer_path, package_path = _build_pending_artifacts(package_fixture)
    review_path = package_fixture["task_dir"] / "review.json"
    _write_json(review_path, _review_record(consumer_path, package_path, verdict=verdict))

    result = _run_finalizer(package_fixture, consumer_path, package_path, review_path)

    assert result.returncode == 0, result.stderr
    finalized_path = package_fixture["task_dir"] / "finalized.json"
    finalized = _load_json(finalized_path)
    assert finalized["status"] == "validated"
    assert finalized["canonical_collection"] is True
    assert finalized["namespace_promotion_staged"] is True
    assert finalized["independent_review"]["verdict"] == verdict
    assert finalized["independent_review"]["sha256"] == _sha256(review_path)
    assert finalized["finalization"]["pending_package_sha256"] == _sha256(package_path)
