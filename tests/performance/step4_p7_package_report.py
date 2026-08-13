"""Build the Step4 P7 consumer exit report and review-pending package manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MODELS = ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4")
ISLS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
EXPECTED_MEASURED_IDENTITIES = 180
EXPECTED_CUSTOM_ALLREDUCE_RANKS = (2, 4)
EXTRA_CUSTOM_ALLREDUCE_RANKS = (8,)
EXPECTED_NCCL_RANKS = (2, 4, 8, 16, 32, 64)
EXTRA_NCCL_RANKS = (48,)
RUST_EVIDENCE_PURPOSE = "consumer coverage/execution evidence, not Python/Rust numeric parity"
RUST_CONFIG_FIELDS = ("tp", "pp", "dp", "moe_tp", "moe_ep", "batch_size", "ctx_tokens")
STATUS_NAMES = (
    "success",
    "memory_infeasible",
    "sla_infeasible",
    "data_unavailable",
    "error",
)
P8_CONTRACT = {
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
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_binding(path: Path, *, task_dir: Path) -> dict[str, str]:
    path = path.resolve()
    task_dir = task_dir.resolve()
    try:
        display_path = str(path.relative_to(task_dir))
    except ValueError:
        display_path = str(path)
    return {"path": display_path, "sha256": _sha256(path)}


def _validate_coverage(coverage: dict[str, Any], measured_keys: dict[str, Any], *, measured_keys_path: Path) -> None:
    if coverage.get("status") != "validated":
        raise ValueError("coverage inventory is not validated")
    measured_count = coverage.get("measured_identity_count")
    if measured_count != EXPECTED_MEASURED_IDENTITIES:
        raise ValueError(f"measured identity count must be {EXPECTED_MEASURED_IDENTITIES}, got {measured_count}")
    identities = measured_keys.get("identities")
    if not isinstance(identities, list) or len(identities) != measured_count:
        raise ValueError("measured-key identity count does not match the coverage inventory")
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("measured identities must be non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("measured identities must be unique")

    coverage_keys = coverage.get("coverage_keys")
    if not isinstance(coverage_keys, dict) or set(coverage_keys) != set(MODELS):
        raise ValueError("coverage keys must contain exactly the two Step4 models")
    coverage_identities = []
    for model in MODELS:
        records = coverage_keys[model]
        if not isinstance(records, list):
            raise TypeError(f"coverage keys for {model} must be a list")
        for record in records:
            structural = record.get("structural") if isinstance(record, dict) else None
            identity = structural.get("identity") if isinstance(structural, dict) else None
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"coverage identity for {model} must be a non-empty string")
            coverage_identities.append(identity)
    if len(set(coverage_identities)) != len(coverage_identities):
        raise ValueError("coverage identities must be unique")
    if set(identities) != set(coverage_identities):
        raise ValueError("measured and coverage identity sets differ")

    provenance = coverage.get("provenance")
    inventory = provenance.get("measured_key_inventory") if isinstance(provenance, dict) else None
    if not isinstance(inventory, dict):
        raise TypeError("coverage measured-key provenance must be an object")
    if inventory.get("path") != measured_keys_path.name:
        raise ValueError("measured-key provenance path must be the task-relative measured-key filename")
    if inventory.get("sha256") != _sha256(measured_keys_path):
        raise ValueError("measured-key provenance SHA-256 mismatch")

    coverage_summary = coverage.get("coverage_summary")
    if not isinstance(coverage_summary, dict) or set(coverage_summary) != set(MODELS):
        raise ValueError("coverage summary must contain exactly the two Step4 models")
    total_required = 0
    for model in MODELS:
        families = coverage_summary[model]
        for family in ("attention", "gemm", "moe", "communication"):
            counts = families.get(family)
            if not isinstance(counts, dict):
                raise TypeError(f"missing coverage counts for {model}/{family}")
            required = counts.get("required_count")
            if counts.get("measured_count") != required:
                raise ValueError(f"measured coverage is incomplete for {model}/{family}")
            for field in ("missing_count", "duplicate_count", "unassigned_count"):
                if counts.get(field) != 0:
                    raise ValueError(f"{model}/{family} has non-zero {field}")
            total_required += required
    if total_required != measured_count:
        raise ValueError(f"coverage summary totals {total_required} identities, expected {measured_count}")


def _communication_contract(coverage: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    audits = coverage.get("communication_topology_audit")
    if not isinstance(audits, dict) or set(audits) != set(MODELS):
        raise ValueError("communication topology audit must contain exactly the two Step4 models")

    model_rank_sets: dict[str, dict[str, set[int]]] = {}
    topology_counts = set()
    invalid_count = 0
    for model in MODELS:
        topologies = audits[model]
        if not isinstance(topologies, list):
            raise TypeError(f"communication topology audit for {model} must be a list")
        topology_counts.add(len(topologies))
        ranks = {"custom_allreduce": set(), "nccl": set()}
        for topology in topologies:
            if topology.get("status") != "runnable":
                invalid_count += 1
                continue
            for query in topology.get("queries", []):
                op = query.get("op")
                rank = query.get("rank")
                if not isinstance(rank, int):
                    raise TypeError(f"communication rank must be an integer: {query}")
                if op == "custom_allreduce":
                    ranks["custom_allreduce"].add(rank)
                elif op in ("nccl_all_gather", "nccl_reduce_scatter"):
                    ranks["nccl"].add(rank)
        model_rank_sets[model] = ranks

    if len(topology_counts) != 1:
        raise ValueError("the two models have different communication topology counts")
    runnable_topology_count = topology_counts.pop()
    for model, ranks in model_rank_sets.items():
        if tuple(sorted(ranks["custom_allreduce"])) != EXPECTED_CUSTOM_ALLREDUCE_RANKS:
            raise ValueError(f"unexpected CustomAllReduce ranks for {model}: {ranks['custom_allreduce']}")
        if tuple(sorted(ranks["nccl"])) != EXPECTED_NCCL_RANKS:
            raise ValueError(f"unexpected NCCL ranks for {model}: {ranks['nccl']}")

    return (
        {
            "custom_allreduce": {
                "required_ranks": list(EXPECTED_CUSTOM_ALLREDUCE_RANKS),
                "extra_measured_ranks": list(EXTRA_CUSTOM_ALLREDUCE_RANKS),
            },
            "nccl": {
                "required_ranks": list(EXPECTED_NCCL_RANKS),
                "extra_measured_ranks": list(EXTRA_NCCL_RANKS),
            },
        },
        runnable_topology_count,
        invalid_count,
    )


def _positive_finite(value: Any, *, field: str, identity: tuple[Any, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be positive and finite for {identity}")
    return float(value)


def _topology_tuple(config: dict[str, Any], *, identity: tuple[str, int]) -> tuple[int, ...]:
    fields = ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")
    values = tuple(config.get(field) for field in fields)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"selected topology fields must be positive integers for {identity}")
    return values


def _matrix_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    if matrix.get("schema") != "step4-profiled-agg-matrix-v1":
        raise ValueError("unexpected P8 matrix schema")
    if matrix.get("status") != "completed":
        raise ValueError("P8 matrix is not completed without hard errors")
    if tuple(matrix.get("models", ())) != MODELS:
        raise ValueError("P8 matrix models do not match the fixed Step4 contract")
    if tuple(matrix.get("isls", ())) != ISLS:
        raise ValueError("P8 matrix ISLs do not match the fixed Step4 contract")
    if matrix.get("point_count") != 16:
        raise ValueError(f"P8 matrix must contain 16 terminal points, got {matrix.get('point_count')}")
    contract = matrix.get("contract")
    if not isinstance(contract, dict):
        raise TypeError("P8 contract must be an object")
    for field, expected in P8_CONTRACT.items():
        if contract.get(field) != expected:
            raise ValueError(f"P8 contract {field} must be {expected}")

    raw_counts = matrix.get("status_counts")
    if not isinstance(raw_counts, dict):
        raise TypeError("P8 status_counts must be an object")
    unknown_statuses = set(raw_counts) - set(STATUS_NAMES)
    if unknown_statuses:
        raise ValueError(f"P8 status_counts contains unsupported statuses: {sorted(unknown_statuses)}")
    counts = {}
    for name in STATUS_NAMES:
        value = raw_counts.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"P8 status count must be a non-negative integer: {name}={value}")
        counts[name] = value
    if sum(counts.values()) != 16:
        raise ValueError(f"P8 status counts must sum to 16, got {counts}")

    runnable_topologies = matrix.get("runnable_topologies")
    invalid_topologies = matrix.get("invalid_cross_node_custom_allreduce_topologies")
    if not isinstance(runnable_topologies, list) or not isinstance(invalid_topologies, list):
        raise TypeError("P8 topology fields must be lists")
    runnable_set = set()
    for raw_topology in runnable_topologies:
        if not isinstance(raw_topology, list) or len(raw_topology) != 6:
            raise ValueError(f"runnable topology must contain six integer axes: {raw_topology}")
        topology = tuple(raw_topology)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in topology):
            raise ValueError(f"runnable topology axes must be positive integers: {raw_topology}")
        tp, _, dp, moe_tp, moe_ep, cp = topology
        deployment_gpus = tp * dp * cp
        if deployment_gpus != moe_tp * moe_ep or P8_CONTRACT["total_gpus"] % deployment_gpus:
            raise ValueError(f"runnable topology violates deployment geometry: {raw_topology}")
        if topology in runnable_set:
            raise ValueError(f"duplicate runnable topology: {raw_topology}")
        runnable_set.add(topology)

    outcomes = matrix.get("terminal_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(MODELS) * len(ISLS):
        raise ValueError("P8 matrix must contain exactly 16 terminal outcomes")
    expected_identities = {(model, isl) for model in MODELS for isl in ISLS}
    seen_identities = set()
    observed_counts = dict.fromkeys(STATUS_NAMES, 0)
    successes = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise TypeError("P8 terminal outcome must be an object")
        identity = (outcome.get("model"), outcome.get("isl"))
        if identity not in expected_identities:
            raise ValueError(f"P8 outcome is outside the fixed matrix: {identity}")
        if identity in seen_identities:
            raise ValueError(f"duplicate P8 matrix outcome: {identity}")
        seen_identities.add(identity)
        status = outcome.get("status")
        if status not in STATUS_NAMES:
            raise ValueError(f"unsupported P8 terminal status for {identity}: {status}")
        observed_counts[status] += 1

        if status != "success":
            if (
                outcome.get("throughput_per_used_gpu") is not None
                or outcome.get("cluster_tokens_per_second") is not None
            ):
                raise ValueError(f"terminal gap must not carry throughput for {identity}")
            if outcome.get("selected_config") is not None:
                raise ValueError(f"terminal gap must not carry selected_config for {identity}")
            if outcome.get("per_ops_data") is not None or outcome.get("per_ops_source") is not None:
                raise ValueError(f"terminal gap must not carry per-op evidence for {identity}")
            for field in ("deployment_gpus", "ttft_ms", "tpot_ms"):
                if outcome.get(field) is not None:
                    raise ValueError(f"terminal gap must not carry success metric {field} for {identity}")
            if not isinstance(outcome.get("reason"), str) or not outcome["reason"]:
                raise ValueError(f"terminal gap must carry a non-empty reason for {identity}")
            continue

        throughput = _positive_finite(
            outcome.get("throughput_per_used_gpu"), field="throughput_per_used_gpu", identity=identity
        )
        cluster_throughput = _positive_finite(
            outcome.get("cluster_tokens_per_second"), field="cluster_tokens_per_second", identity=identity
        )
        ttft_ms = _positive_finite(outcome.get("ttft_ms"), field="ttft_ms", identity=identity)
        tpot_ms = _positive_finite(outcome.get("tpot_ms"), field="tpot_ms", identity=identity)
        if ttft_ms > P8_CONTRACT["ttft_ms"]:
            raise ValueError(f"success point violates TTFT SLA for {identity}")
        if tpot_ms > P8_CONTRACT["tpot_ms"]:
            raise ValueError(f"success point violates TPOT SLA for {identity}")
        if not math.isclose(
            cluster_throughput,
            throughput * P8_CONTRACT["total_gpus"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"cluster throughput arithmetic mismatch for {identity}")

        config = outcome.get("selected_config")
        if not isinstance(config, dict):
            raise TypeError(f"success point selected_config must be an object for {identity}")
        topology = _topology_tuple(config, identity=identity)
        if topology not in runnable_set:
            raise ValueError(f"selected topology is not runnable for {identity}: {topology}")
        tp, _, dp, moe_tp, moe_ep, cp = topology
        deployment_gpus = outcome.get("deployment_gpus")
        if isinstance(deployment_gpus, bool) or not isinstance(deployment_gpus, int) or deployment_gpus <= 0:
            raise ValueError(f"deployment_gpus must be a positive integer for {identity}")
        if tp * dp * cp != moe_tp * moe_ep or deployment_gpus != tp * dp * cp:
            raise ValueError(f"selected topology deployment geometry mismatch for {identity}")
        if P8_CONTRACT["total_gpus"] % deployment_gpus:
            raise ValueError(f"selected deployment does not tile total_gpus for {identity}")
        for field in ("batch_size", "ctx_tokens"):
            value = config.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"selected_config {field} must be a positive integer for {identity}")

        successes.append(
            {
                key: outcome[key]
                for key in (
                    "model",
                    "isl",
                    "throughput_per_used_gpu",
                    "cluster_tokens_per_second",
                    "deployment_gpus",
                    "ttft_ms",
                    "tpot_ms",
                    "selected_config",
                )
            }
        )

    if seen_identities != expected_identities:
        raise ValueError(f"P8 matrix outcomes are incomplete: {expected_identities - seen_identities}")
    if observed_counts != counts:
        raise ValueError(f"P8 status counts disagree with terminal outcomes: {counts} != {observed_counts}")
    if counts["data_unavailable"] or counts["error"]:
        raise ValueError(f"P8 contains hard data/runtime errors: {counts}")
    return {
        "status": matrix["status"],
        "point_count": matrix["point_count"],
        "status_counts": counts,
        "runnable_topology_count": len(runnable_topologies),
        "invalid_topology_count": len(invalid_topologies),
        "successful_points": successes,
    }


def _validate_rust_evidence(rust_evidence: dict[str, Any], *, repo_root: Path, matrix_summary: dict[str, Any]) -> None:
    if rust_evidence.get("schema") != "step4-p7-rust-consumer-evidence-v1":
        raise ValueError("unexpected Rust evidence schema")
    if rust_evidence.get("status") != "validated":
        raise ValueError("Rust evidence status must be validated")
    if rust_evidence.get("purpose") != RUST_EVIDENCE_PURPOSE:
        raise ValueError("unexpected Rust evidence purpose")
    if rust_evidence.get("engine_step_backend") != "rust":
        raise ValueError("Rust evidence engine_step_backend must be rust")

    extension_value = rust_evidence.get("extension_path")
    if not isinstance(extension_value, str) or not extension_value:
        raise ValueError("Rust extension path must be a non-empty repository-relative path")
    extension_relative = Path(extension_value)
    if extension_relative.is_absolute():
        raise ValueError("Rust extension path must be a non-empty repository-relative path")
    extension_path = (repo_root / extension_relative).resolve()
    try:
        extension_path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError("Rust extension path escapes the repository root") from error
    if not extension_path.is_file():
        raise ValueError(f"Rust extension does not exist: {extension_value}")
    if rust_evidence.get("extension_sha256") != _sha256(extension_path):
        raise ValueError("Rust extension SHA-256 mismatch")

    models = rust_evidence.get("models")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise ValueError("Rust evidence must contain exactly the two Step4 models")
    for model in MODELS:
        record = models[model]
        if not isinstance(record, dict):
            raise TypeError(f"Rust model evidence must be an object for {model}")
        if record.get("coverage_gate") is not True:
            raise ValueError(f"Rust coverage gate did not pass for {model}")
        if record.get("source") != "rust":
            raise ValueError(f"Rust model source must be rust for {model}")
        for field in ("memory_gib", "throughput_per_used_gpu", "tokens_per_second", "ttft_ms", "tpot_ms"):
            _positive_finite(record.get(field), field=f"Rust {field}", identity=(model, "selected"))
        if record["ttft_ms"] > P8_CONTRACT["ttft_ms"]:
            raise ValueError(f"Rust evidence violates TTFT SLA for {model}")
        if record["tpot_ms"] > P8_CONTRACT["tpot_ms"]:
            raise ValueError(f"Rust evidence violates TPOT SLA for {model}")

        config = record.get("selected_config")
        if not isinstance(config, dict) or set(config) != set(RUST_CONFIG_FIELDS):
            raise ValueError(f"Rust selected_config must contain exactly {RUST_CONFIG_FIELDS} for {model}")
        if any(
            isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] <= 0
            for field in RUST_CONFIG_FIELDS
        ):
            raise ValueError(f"Rust selected_config fields must be positive integers for {model}")
        matching_success = any(
            point["model"] == model
            and all(point["selected_config"].get(field) == config[field] for field in RUST_CONFIG_FIELDS)
            for point in matrix_summary["successful_points"]
        )
        if not matching_success:
            raise ValueError(f"Rust selected config has no matching P8 success for {model}")


def _validate_package_hashes(package: dict[str, Any], *, repo_root: Path, task_dir: Path) -> dict[str, int]:
    canonical_artifacts = package.get("canonical_artifacts")
    collection_manifests = package.get("collection_manifests")
    if not isinstance(canonical_artifacts, list) or not isinstance(collection_manifests, list):
        raise TypeError("package artifact inventories must be lists")

    hash_errors = []
    for record in canonical_artifacts:
        path = repo_root / record["path"]
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            hash_errors.append(str(path))
    for record in collection_manifests:
        path = task_dir / record["path"]
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            hash_errors.append(str(path))
    if hash_errors:
        raise ValueError(f"package artifact hash mismatch: {hash_errors}")

    return {
        "canonical_artifacts_checked": len(canonical_artifacts),
        "collection_manifests_checked": len(collection_manifests),
        "total_checked": len(canonical_artifacts) + len(collection_manifests),
        "hash_error_count": 0,
    }


def build_reports(*, repo_root: Path, task_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build validated P7 consumer evidence and a package awaiting independent review."""
    coverage_path = task_dir / "step4_pro_v3_v4_coverage_inventory.json"
    measured_keys_path = task_dir / "step4_pro_v3_v4_measured_keys.json"
    matrix_path = task_dir / "p8_isl_matrix_20260801.json"
    rust_path = task_dir / "p7_rust_consumer_evidence.json"
    package_path = task_dir / "p7_canonical_package_manifest.json"

    coverage = _load_json(coverage_path)
    measured_keys = _load_json(measured_keys_path)
    matrix = _load_json(matrix_path)
    rust_evidence = _load_json(rust_path)
    package = _load_json(package_path)

    _validate_coverage(coverage, measured_keys, measured_keys_path=measured_keys_path)
    communication, runnable_count, invalid_count = _communication_contract(coverage)
    matrix_summary = _matrix_summary(matrix)
    if matrix_summary["runnable_topology_count"] != runnable_count:
        raise ValueError("coverage and P8 runnable topology counts disagree")
    if matrix_summary["invalid_topology_count"] != invalid_count:
        raise ValueError("coverage and P8 invalid topology counts disagree")
    _validate_rust_evidence(rust_evidence, repo_root=repo_root, matrix_summary=matrix_summary)
    package_integrity = _validate_package_hashes(package, repo_root=repo_root, task_dir=task_dir)

    consumer = {
        "schema": "step4-p7-consumer-exit-v2",
        "status": "validated",
        "measured_identity_count": coverage["measured_identity_count"],
        "coverage_summary": coverage["coverage_summary"],
        "communication": communication,
        "matrix": matrix_summary,
        "rust_consumer": rust_evidence,
        "package_integrity": package_integrity,
        "source_artifacts": {
            "coverage": _artifact_binding(coverage_path, task_dir=task_dir),
            "measured_keys": _artifact_binding(measured_keys_path, task_dir=task_dir),
            "p8_matrix": _artifact_binding(matrix_path, task_dir=task_dir),
            "rust_consumer": _artifact_binding(rust_path, task_dir=task_dir),
        },
        "checks": {
            "coverage": "PASS",
            "communication_process_groups": "PASS",
            "matrix_terminal_outcomes": "PASS",
            "rust_consumer": "PASS",
            "package_hashes": "PASS",
        },
        "no_fallbacks_or_synthetic_rows": True,
    }

    updated_package = copy.deepcopy(package)
    updated_package.update(
        {
            "status": "pending_independent_review",
            "canonical_collection": False,
            "namespace_promotion_staged": True,
            "block_reason": (
                "Consumer coverage, P8 matrix, Rust selected-point execution, and package hashes "
                "validate; canonical admission awaits a fresh independent review."
            ),
            "communication_disposition": {
                "custom_allreduce": (
                    "required attention-TP ranks 2/4 are measured exactly; rank8 is retained as "
                    "extra measured evidence; no complete-world rank substitution or borrowing"
                ),
                "nccl": (
                    "required DP/EP ranks 2/4/8/16/32/64 are measured exactly; rank48 is retained "
                    "as extra measured evidence; no rank borrowing"
                ),
            },
            "p8_matrix": {
                **_artifact_binding(matrix_path, task_dir=task_dir),
                "status": matrix_summary["status"],
                "status_counts": matrix_summary["status_counts"],
                "runnable_topology_count": matrix_summary["runnable_topology_count"],
                "invalid_topology_count": matrix_summary["invalid_topology_count"],
            },
            "package_integrity": package_integrity,
        }
    )
    return consumer, updated_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--consumer-output", type=Path, required=True)
    parser.add_argument("--package-output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    task_dir = args.task_dir.resolve()
    consumer, package = build_reports(repo_root=repo_root, task_dir=task_dir)
    _write_json(args.consumer_output, consumer)
    package["consumer_exit_report"] = {
        **_artifact_binding(args.consumer_output, task_dir=task_dir),
        "status": consumer["status"],
    }
    _write_json(args.package_output, package)


if __name__ == "__main__":
    main()
