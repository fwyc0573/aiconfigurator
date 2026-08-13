"""Finalize the Step4 P7 package after a bound independent review."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from tests.performance.step4_p7_package_report import (
    _artifact_binding,
    _load_json,
    _sha256,
    _write_json,
    build_reports,
)

MANDATORY_REMEDIATIONS = (
    "coverage_exact_set_and_provenance",
    "matrix_terminal_contract",
    "rust_evidence_contract",
    "documentation_state",
)
SUPPORTED_VERDICTS = ("WATCH", "APPROVE")


def _validate_string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


def _validate_review_record(
    review: dict[str, Any],
    *,
    review_path: Path,
    task_dir: Path,
    consumer_path: Path,
    package_path: Path,
) -> None:
    if review.get("schema") != "step4-p7-independent-review-v1":
        raise ValueError("unexpected independent review schema")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer:
        raise ValueError("independent reviewer identity must be a non-empty string")
    verdict = review.get("verdict")
    if verdict == "BLOCK":
        raise ValueError("independent review verdict BLOCK prevents canonical admission")
    if verdict not in SUPPORTED_VERDICTS:
        raise ValueError(f"unsupported independent review verdict: {verdict}")
    critical_findings = review.get("critical_findings")
    if isinstance(critical_findings, bool) or critical_findings != 0:
        raise ValueError("independent review must report zero critical findings")

    required = _validate_string_list(review.get("required_remediations"), field="required_remediations")
    verified = _validate_string_list(review.get("remediations_verified"), field="remediations_verified")
    mandatory = set(MANDATORY_REMEDIATIONS)
    if set(required) != mandatory:
        raise ValueError("independent review does not declare the complete mandatory remediation set")
    if set(verified) != set(required):
        raise ValueError("review remediation verification is incomplete")

    task_root = task_dir.resolve()
    if review_path.resolve().parent != task_root:
        raise ValueError("independent review record must be a task-local file")
    reviewed_artifacts = review.get("reviewed_artifacts")
    if not isinstance(reviewed_artifacts, dict):
        raise TypeError("reviewed_artifacts must be an object")
    bindings = (
        ("consumer_exit_report", consumer_path, "reviewed consumer report"),
        ("pending_package", package_path, "reviewed pending package"),
    )
    for key, path, label in bindings:
        record = reviewed_artifacts.get(key)
        if not isinstance(record, dict):
            raise TypeError(f"{label} binding must be an object")
        if record.get("path") != path.name:
            raise ValueError(f"{label} path mismatch")
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"{label} SHA-256 mismatch")


def finalize_package(
    *,
    repo_root: Path,
    task_dir: Path,
    consumer_path: Path,
    package_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    """Revalidate a pending package and bind its independent review."""
    repo_root = repo_root.resolve()
    task_dir = task_dir.resolve()
    consumer_path = consumer_path.resolve()
    package_path = package_path.resolve()
    review_path = review_path.resolve()
    if not review_path.is_file():
        raise FileNotFoundError(f"independent review record does not exist: {review_path}")
    if consumer_path.parent != task_dir or package_path.parent != task_dir:
        raise ValueError("consumer report and pending package must be task-local files")

    current_consumer = _load_json(consumer_path)
    current_package = _load_json(package_path)
    expected_consumer, expected_package = build_reports(repo_root=repo_root, task_dir=task_dir)
    expected_package["consumer_exit_report"] = {
        **_artifact_binding(consumer_path, task_dir=task_dir),
        "status": expected_consumer["status"],
    }
    if current_consumer != expected_consumer:
        raise ValueError("consumer report differs from the deterministic hardened rebuild")
    if current_package != expected_package:
        raise ValueError("pending package differs from the deterministic hardened rebuild")

    review = _load_json(review_path)
    _validate_review_record(
        review,
        review_path=review_path,
        task_dir=task_dir,
        consumer_path=consumer_path,
        package_path=package_path,
    )

    pending_package_sha256 = _sha256(package_path)
    finalized = copy.deepcopy(current_package)
    finalized.pop("block_reason", None)
    finalized.update(
        {
            "status": "validated",
            "canonical_collection": True,
            "namespace_promotion_staged": True,
            "independent_review": {
                **_artifact_binding(review_path, task_dir=task_dir),
                "reviewer": review["reviewer"],
                "verdict": review["verdict"],
                "critical_findings": review["critical_findings"],
                "required_remediations": list(review["required_remediations"]),
                "remediations_verified": list(review["remediations_verified"]),
            },
            "finalization": {
                "schema": "step4-p7-finalization-v1",
                "status": "validated",
                "consumer_exit_report_sha256": _sha256(consumer_path),
                "pending_package_sha256": pending_package_sha256,
                "review_record_sha256": _sha256(review_path),
                "hardened_revalidation": "PASS",
            },
        }
    )
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--consumer-report", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--review-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    finalized = finalize_package(
        repo_root=args.repo_root,
        task_dir=args.task_dir,
        consumer_path=args.consumer_report,
        package_path=args.package,
        review_path=args.review_record,
    )
    _write_json(args.output, finalized)


if __name__ == "__main__":
    main()
