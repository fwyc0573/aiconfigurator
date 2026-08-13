"""Static validation for resumable Step4 profiling collection runs.

This task-local gate reconciles a declared invocation set with Collector-v2
checkpoint state, collection summaries, structural coverage identities, and
measured output files.  Multiple workload invocations may explicitly map to
one structural identity; workload identity and structural identity remain
separate.  It never invents rows and never changes generic Task admission
behavior.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

MANIFEST_SCHEMA = "step4-collection-manifest-v1"
BACKEND = "vllm"
VERSION = "0.19.0"
SYSTEM = "h800_sxm"
MODELS = {"stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"}
FAMILIES = {"attention", "gemm", "moe", "communication"}
TERMINAL_STATUSES = {"measured", "expected_failed", "unsupported", "failed"}


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {path}") from exc


def _relative_path(manifest_dir: Path, raw_path: object, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field} must be a non-empty relative path")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{field} must be relative")
    resolved = (manifest_dir / candidate).resolve()
    try:
        resolved.relative_to(manifest_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} path must stay inside the manifest directory") from exc
    return resolved


def _require_string(mapping: Mapping[str, object], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_identity(model: str, family: str, identity: object) -> str:
    if not isinstance(identity, str) or not identity:
        raise ValueError("structural_identity must be a non-empty string")
    prefix = f"{model}:{family}:{BACKEND}:{VERSION}:{SYSTEM}:"
    if not identity.startswith(prefix):
        raise ValueError(f"structural_identity has unexpected identity prefix: {identity!r}")
    return identity


def _inventory_identities(
    inventory: Mapping[str, object], model: str, family: str, attention_phase: str | None = None
) -> set[str]:
    coverage_keys = inventory.get("coverage_keys")
    if not isinstance(coverage_keys, Mapping):
        raise TypeError("coverage inventory must contain coverage_keys mapping")
    records = coverage_keys.get(model)
    if not isinstance(records, list):
        raise TypeError(f"coverage inventory has no records for model {model!r}")
    identities: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("coverage inventory records must be mappings")
        if record.get("op_family") != family:
            continue
        structural = record.get("structural")
        if not isinstance(structural, Mapping):
            raise TypeError("coverage inventory record missing structural mapping")
        if attention_phase is not None:
            axes = structural.get("axes")
            if not isinstance(axes, Mapping):
                raise TypeError("phase-filtered coverage inventory record missing structural axes")
            if axes.get("phase") != attention_phase:
                continue
        identity = _validate_identity(model, family, structural.get("identity"))
        if identity in identities:
            raise ValueError(f"coverage inventory has duplicate structural identity: {identity}")
        identities.add(identity)
    if not identities:
        raise ValueError(f"coverage inventory has no {family} identities for model {model!r}")
    return identities


def _validate_checkpoint(
    checkpoint: Mapping[str, object],
    expected_ids: set[str],
    outcome_by_id: Mapping[str, Mapping[str, object]],
    *,
    expected_module: str | None,
    expected_run_func: str | None,
) -> None:
    if checkpoint.get("schema") != "collector-resume-v1":
        raise ValueError("checkpoint schema must equal collector-resume-v1")
    if checkpoint.get("backend") != BACKEND:
        raise ValueError(f"checkpoint backend must equal {BACKEND!r}")
    if expected_module is not None and checkpoint.get("module") != expected_module:
        raise ValueError(f"checkpoint module must equal {expected_module!r}")
    if expected_run_func is not None and checkpoint.get("run_func") != expected_run_func:
        raise ValueError(f"checkpoint run_func must equal {expected_run_func!r}")
    sets: dict[str, set[str]] = {}
    for field in ("done", "failed", "expected_failed", "unsupported"):
        values = checkpoint.get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"checkpoint {field} must be a list of non-empty strings")
        values_set = set(values)
        if len(values_set) != len(values):
            raise ValueError(f"checkpoint {field} contains duplicate invocation IDs")
        if not values_set <= expected_ids:
            raise ValueError(f"checkpoint {field} contains an unknown invocation ID")
        sets[field] = values_set
    if sets["done"] & (sets["failed"] | sets["expected_failed"] | sets["unsupported"]):
        raise ValueError("checkpoint done/failed/unsupported sets overlap")
    if sets["failed"] & (sets["expected_failed"] | sets["unsupported"]):
        raise ValueError("checkpoint failed/unsupported sets overlap")
    if sets["expected_failed"] & sets["unsupported"]:
        raise ValueError("checkpoint expected_failed/unsupported sets overlap")
    expected_sets = {
        "done": {invocation_id for invocation_id, outcome in outcome_by_id.items() if outcome["status"] == "measured"},
        "failed": {invocation_id for invocation_id, outcome in outcome_by_id.items() if outcome["status"] == "failed"},
        "expected_failed": {
            invocation_id for invocation_id, outcome in outcome_by_id.items() if outcome["status"] == "expected_failed"
        },
        "unsupported": {
            invocation_id for invocation_id, outcome in outcome_by_id.items() if outcome["status"] == "unsupported"
        },
    }
    for field, expected in expected_sets.items():
        if sets[field] != expected:
            raise ValueError(f"checkpoint {field} does not reconcile with manifest outcomes")


def _validate_invocation_reconciliation(
    manifest_path: Path,
    payload: Mapping[str, object],
    expected_ids: set[str],
    identity_by_id: Mapping[str, str],
    workload_by_id: Mapping[str, Mapping[str, object] | None],
    outcome_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    """Validate raw workload invocation provenance when identities repeat.

    Collector attention checkpoints identify workload tuples, while the P3
    inventory identifies structural profiles.  A separate reconciliation file
    preserves that one-to-many mapping and prevents the validator from
    silently treating an aggregate output as one structural row.
    """

    repeated_identity = len(expected_ids) != len(set(identity_by_id.values()))
    raw_path = payload.get("invocation_reconciliation_path")
    if repeated_identity and raw_path is None:
        raise ValueError("repeated structural identities require invocation reconciliation")
    if raw_path is None:
        return
    reconciliation_path = _relative_path(manifest_path.parent, raw_path, "invocation_reconciliation_path")
    reconciliation = _load_json(reconciliation_path, "invocation reconciliation")
    if not isinstance(reconciliation, Mapping):
        raise TypeError("invocation reconciliation must be a JSON object")
    if reconciliation.get("schema") != "step4-invocation-reconciliation-v1":
        raise ValueError("invocation reconciliation schema must equal step4-invocation-reconciliation-v1")
    entries = reconciliation.get("invocations")
    if not isinstance(entries, list):
        raise TypeError("invocation reconciliation invocations must be a list")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("invocation reconciliation validation requires pyarrow") from exc
    entry_by_id: dict[str, Mapping[str, object]] = {}
    tables_by_path: dict[Path, object] = {}
    measured_indices_by_path: dict[Path, set[int]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("invocation reconciliation entries must be mappings")
        invocation_id = _require_string(entry, "invocation_id")
        if invocation_id in entry_by_id:
            raise ValueError(f"invocation reconciliation has duplicate invocation_id: {invocation_id}")
        if invocation_id not in expected_ids:
            raise ValueError(f"invocation reconciliation references unknown invocation_id: {invocation_id}")
        if entry.get("structural_identity") != identity_by_id[invocation_id]:
            raise ValueError(f"invocation reconciliation structural identity mismatch: {invocation_id}")
        workload = entry.get("workload")
        if not isinstance(workload, Mapping) or not workload:
            raise ValueError(f"invocation reconciliation workload must be non-empty: {invocation_id}")
        if workload_by_id[invocation_id] != workload:
            raise ValueError(f"invocation reconciliation workload mismatch: {invocation_id}")
        outcome = outcome_by_id[invocation_id]
        if outcome["status"] == "measured":
            row_index = entry.get("row_index")
            if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
                raise ValueError(f"measured reconciliation row_index must be a non-negative integer: {invocation_id}")
            row_fingerprint = entry.get("row_fingerprint")
            if not isinstance(row_fingerprint, str) or len(row_fingerprint) != 64:
                raise ValueError(f"measured reconciliation row_fingerprint must be a SHA-256 digest: {invocation_id}")
            try:
                int(row_fingerprint, 16)
            except ValueError as exc:
                raise ValueError(
                    f"measured reconciliation row_fingerprint must be a SHA-256 digest: {invocation_id}"
                ) from exc
            if entry.get("output_path") != outcome.get("output_path"):
                raise ValueError(f"invocation reconciliation output_path mismatch: {invocation_id}")
            output_path = _relative_path(
                manifest_path.parent, outcome.get("output_path"), "reconciliation measured output_path"
            )
            table = tables_by_path.get(output_path)
            if table is None:
                table = pq.read_table(output_path)
                tables_by_path[output_path] = table
                measured_indices_by_path[output_path] = set()
            if row_index >= table.num_rows:
                raise ValueError(f"measured reconciliation row_index is out of range: {invocation_id}")
            if row_index in measured_indices_by_path[output_path]:
                raise ValueError(f"measured reconciliation has duplicate row_index: {invocation_id}")
            measured_indices_by_path[output_path].add(row_index)
            row = table.slice(row_index, 1).to_pylist()[0]
            try:
                canonical_row = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"measured parquet row is not canonically serializable: {invocation_id}") from exc
            actual_fingerprint = hashlib.sha256(canonical_row).hexdigest()
            if actual_fingerprint != row_fingerprint.lower():
                raise ValueError(f"measured reconciliation row_fingerprint mismatch: {invocation_id}")
        entry_by_id[invocation_id] = entry
    if set(entry_by_id) != expected_ids:
        missing = sorted(expected_ids - set(entry_by_id))
        extra = sorted(set(entry_by_id) - expected_ids)
        raise ValueError(f"invocation reconciliation IDs do not match manifest: missing={missing}, extra={extra}")
    for output_path, table in tables_by_path.items():
        expected_indices = set(range(table.num_rows))
        if measured_indices_by_path[output_path] != expected_indices:
            raise ValueError(f"measured reconciliation row indices do not cover output: {output_path}")
    declared_sha256 = payload.get("invocation_reconciliation_sha256")
    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        raise ValueError("invocation_reconciliation_sha256 must be a 64-character hex digest")
    try:
        int(declared_sha256, 16)
    except ValueError as exc:
        raise ValueError("invocation_reconciliation_sha256 must be a 64-character hex digest") from exc
    actual_sha256 = hashlib.sha256(reconciliation_path.read_bytes()).hexdigest()
    if actual_sha256 != declared_sha256.lower():
        raise ValueError("invocation reconciliation sha256 mismatch")


def validate_collection_manifest(manifest_path: str | Path) -> dict[str, object]:
    """Validate one complete collection-run manifest and return numeric counts."""

    path = Path(manifest_path).resolve()
    payload = _load_json(path, "collection manifest")
    if not isinstance(payload, Mapping):
        raise TypeError("collection manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema_version must equal {MANIFEST_SCHEMA}")
    for field, expected in (("backend", BACKEND), ("version", VERSION), ("system", SYSTEM), ("device", SYSTEM)):
        if payload.get(field) != expected:
            raise ValueError(f"manifest {field} must equal {expected!r}")
    model = _require_string(payload, "model")
    if model not in MODELS:
        raise ValueError(f"unsupported Step4 model: {model!r}")
    phase = _require_string(payload, "phase")
    if phase not in {"smoke", "limited", "full"}:
        raise ValueError(f"manifest phase must be smoke, limited, or full; got {phase!r}")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise TypeError("manifest scope must be a mapping")
    family = _require_string(scope, "op_family")
    if family not in FAMILIES | {"all"}:
        raise ValueError(f"unsupported manifest op_family scope: {family!r}")
    attention_phase = scope.get("attention_phase")
    if attention_phase is not None and (family != "attention" or attention_phase not in {"context", "generation"}):
        raise ValueError("scope attention_phase is valid only for context/generation attention")

    expected_raw = payload.get("expected_invocations")
    if not isinstance(expected_raw, list) or not expected_raw:
        raise ValueError("expected_invocations must be a non-empty list")
    expected_ids: set[str] = set()
    expected_identities: set[str] = set()
    identity_by_id: dict[str, str] = {}
    workload_by_id: dict[str, Mapping[str, object] | None] = {}
    for item in expected_raw:
        if not isinstance(item, Mapping):
            raise TypeError("expected_invocations entries must be mappings")
        invocation_id = _require_string(item, "invocation_id")
        if invocation_id in expected_ids:
            raise ValueError(f"duplicate invocation_id: {invocation_id}")
        item_family = _require_string(item, "op_family")
        if item_family not in FAMILIES or (family != "all" and item_family != family):
            raise ValueError(f"invocation op_family is outside manifest scope: {item_family!r}")
        identity = _validate_identity(model, item_family, item.get("structural_identity"))
        expected_ids.add(invocation_id)
        expected_identities.add(identity)
        identity_by_id[invocation_id] = identity
        workload = item.get("workload")
        if workload is not None and (not isinstance(workload, Mapping) or not workload):
            raise ValueError(f"expected invocation workload must be a non-empty mapping: {invocation_id}")
        workload_by_id[invocation_id] = workload

    inventory_path = _relative_path(path.parent, payload.get("coverage_inventory_path"), "coverage_inventory_path")
    declared_inventory_sha256 = payload.get("coverage_inventory_sha256")
    if not isinstance(declared_inventory_sha256, str) or len(declared_inventory_sha256) != 64:
        raise ValueError("coverage_inventory_sha256 must be a 64-character hex digest")
    try:
        int(declared_inventory_sha256, 16)
    except ValueError as exc:
        raise ValueError("coverage_inventory_sha256 must be a 64-character hex digest") from exc
    actual_inventory_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    if actual_inventory_sha256 != declared_inventory_sha256.lower():
        raise ValueError("coverage inventory sha256 mismatch")
    inventory = _load_json(inventory_path, "coverage inventory")
    if not isinstance(inventory, Mapping):
        raise TypeError("coverage inventory must be a JSON object")
    inventory_identities: set[str] = set()
    for inventory_family in FAMILIES if family == "all" else {family}:
        inventory_identities |= _inventory_identities(
            inventory,
            model,
            inventory_family,
            attention_phase if inventory_family == "attention" else None,
        )
    if expected_identities != inventory_identities:
        missing = sorted(inventory_identities - expected_identities)
        extra = sorted(expected_identities - inventory_identities)
        raise ValueError(f"manifest structural identities do not match inventory: missing={missing}, extra={extra}")

    outcomes_raw = payload.get("outcomes")
    if not isinstance(outcomes_raw, list):
        raise TypeError("outcomes must be a list")
    outcome_by_id: dict[str, Mapping[str, object]] = {}
    for outcome in outcomes_raw:
        if not isinstance(outcome, Mapping):
            raise TypeError("outcomes entries must be mappings")
        invocation_id = _require_string(outcome, "invocation_id")
        if invocation_id not in expected_ids:
            raise ValueError(f"outcome references unknown invocation_id: {invocation_id}")
        if invocation_id in outcome_by_id:
            raise ValueError(f"duplicate outcome for invocation_id: {invocation_id}")
        status = _require_string(outcome, "status")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"outcome is not terminal: {status!r}")
        outcome_by_id[invocation_id] = outcome
    missing_outcomes = sorted(expected_ids - set(outcome_by_id))
    if missing_outcomes:
        raise ValueError(f"missing terminal outcome for invocation IDs: {missing_outcomes}")

    for invocation_id, outcome in outcome_by_id.items():
        status = outcome["status"]
        if status == "measured":
            output_path = _relative_path(path.parent, outcome.get("output_path"), "measured output_path")
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise ValueError(f"measured output_path must be a non-empty file: {output_path}")
            declared_sha256 = outcome.get("sha256")
            if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
                raise ValueError("measured outcome sha256 must be a 64-character hex digest")
            try:
                int(declared_sha256, 16)
            except ValueError as exc:
                raise ValueError("measured outcome sha256 must be a 64-character hex digest") from exc
            actual_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
            if actual_sha256 != declared_sha256.lower():
                raise ValueError(
                    f"measured output sha256 mismatch for {output_path}: "
                    f"reported={declared_sha256}, actual={actual_sha256}"
                )
        else:
            reason = outcome.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{status} outcome requires a non-empty reason")
            evidence_path = _relative_path(path.parent, outcome.get("evidence_path"), f"{status} evidence_path")
            if not evidence_path.is_file() or evidence_path.stat().st_size <= 0:
                raise ValueError(f"{status} evidence_path must be a non-empty file: {evidence_path}")

    _validate_invocation_reconciliation(path, payload, expected_ids, identity_by_id, workload_by_id, outcome_by_id)

    failed_ids = sorted(
        invocation_id for invocation_id, outcome in outcome_by_id.items() if outcome["status"] == "failed"
    )
    if failed_ids:
        raise ValueError(f"failed outcome for invocation IDs: {failed_ids}")

    counts = Counter(outcome["status"] for outcome in outcome_by_id.values())
    measured_count = counts.get("measured", 0)
    if phase == "full" and measured_count == 0:
        raise ValueError("full collection manifest must contain at least one measured outcome")
    if measured_count:
        measured_inventory_path = _relative_path(
            path.parent, payload.get("measured_key_inventory_path"), "measured_key_inventory_path"
        )
        measured_inventory = _load_json(measured_inventory_path, "measured-key inventory")
        if isinstance(measured_inventory, list):
            measured_inventory_ids = measured_inventory
        elif isinstance(measured_inventory, Mapping) and isinstance(measured_inventory.get("identities"), list):
            measured_inventory_ids = measured_inventory["identities"]
        else:
            raise TypeError("measured-key inventory must be a list or an identities mapping")
        if any(not isinstance(identity, str) for identity in measured_inventory_ids):
            raise ValueError("measured-key inventory identities must be strings")
        if len(set(measured_inventory_ids)) != len(measured_inventory_ids):
            raise ValueError("measured-key inventory contains duplicate identities")
        actual_measured_ids = {
            identity_by_id[invocation_id]
            for invocation_id, outcome in outcome_by_id.items()
            if outcome["status"] == "measured"
        }
        if set(measured_inventory_ids) != actual_measured_ids:
            raise ValueError("measured-key inventory identities do not match measured outcomes")
        declared_inventory_sha256 = payload.get("measured_key_inventory_sha256")
        if not isinstance(declared_inventory_sha256, str) or len(declared_inventory_sha256) != 64:
            raise ValueError("measured_key_inventory_sha256 must be a 64-character hex digest")
        actual_inventory_sha256 = hashlib.sha256(measured_inventory_path.read_bytes()).hexdigest()
        if actual_inventory_sha256 != declared_inventory_sha256.lower():
            raise ValueError("measured-key inventory sha256 mismatch")

    summary_path = _relative_path(path.parent, payload.get("collection_summary_path"), "collection_summary_path")
    summary_payload = _load_json(summary_path, "collection summary")
    if not isinstance(summary_payload, Mapping) or not isinstance(summary_payload.get("summary"), Mapping):
        raise TypeError("collection summary must contain a summary mapping")
    summary = summary_payload["summary"]
    if summary.get("backend") != BACKEND or summary.get("version") != VERSION:
        raise ValueError("collection summary backend/version does not match manifest")
    total_errors = summary.get("total_errors")
    if isinstance(total_errors, bool) or not isinstance(total_errors, int) or total_errors < 0:
        raise ValueError("collection summary total_errors must be a non-negative integer")
    if total_errors != 0:
        raise ValueError(f"collection summary reports errors: total_errors={total_errors}")
    errors = summary_payload.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError("collection summary errors must be an empty list")
    expected_summary_fields = {
        "model": model,
        "system": SYSTEM,
        "phase": phase,
        "op_family": family,
        "expected_invocation_count": len(expected_ids),
        "terminal_outcome_count": len(outcome_by_id),
        "structural_identity_count": len(expected_identities),
        "measured_count": counts.get("measured", 0),
        "unsupported_count": counts.get("unsupported", 0),
        "expected_failed_count": counts.get("expected_failed", 0),
        "failed_count": counts.get("failed", 0),
    }
    if attention_phase is not None:
        expected_summary_fields["attention_phase"] = attention_phase
    for field, expected in expected_summary_fields.items():
        if summary.get(field) != expected:
            raise ValueError(f"collection summary {field} does not reconcile with manifest")

    checkpoint_path = _relative_path(path.parent, payload.get("checkpoint_path"), "checkpoint_path")
    checkpoint = _load_json(checkpoint_path, "collector checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("collector checkpoint must be a JSON object")
    if family == "attention" and attention_phase is not None:
        expected_module = f"vllm.attention_{attention_phase}"
    elif family in FAMILIES - {"communication"}:
        expected_module = f"vllm.{family}"
    else:
        expected_module = None
    expected_run_func = {
        "attention": "run_attention_torch",
        "gemm": "run_gemm",
        "moe": "run_moe_torch",
    }.get(family)
    _validate_checkpoint(
        checkpoint,
        expected_ids,
        outcome_by_id,
        expected_module=expected_module,
        expected_run_func=expected_run_func,
    )

    return {
        "admissible": True,
        "model": model,
        "phase": phase,
        "op_family": family,
        "expected_invocation_count": len(expected_ids),
        "structural_identity_count": len(expected_identities),
        "outcome_counts": dict(sorted(counts.items())),
        "measured_count": measured_count,
        "expected_failed_count": counts.get("expected_failed", 0),
        "unsupported_count": counts.get("unsupported", 0),
        "collection_summary_errors": total_errors,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    report = validate_collection_manifest(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
