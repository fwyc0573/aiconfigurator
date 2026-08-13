"""Unit tests for the task-local P6 NCCL sweep manifest gate."""

import copy
import hashlib
import json

import pytest

from tests.performance.step4_p6_manifest import (
    build_nccl_manifest,
    merge_sweep_payloads,
    validate_sweep_payload,
)

pytestmark = pytest.mark.unit


def _payload():
    return {
        "artifact_type": "p6_exact_rank_nccl_message_sweep",
        "backend": "nccl",
        "framework_backend": "vllm",
        "dtype": "half",
        "canonical_collection": False,
        "diagnostic_only": False,
        "vllm_version": "0.19.0",
        "torch_version": "2.10.0+cu129",
        "nccl_version": "2.27.5",
        "device": "NVIDIA H800",
        "compute_capability": [9, 0],
        "world_size": 32,
        "ops": ["all_gather", "reduce_scatter"],
        "image_reference": "hub.stepfun-inc.com/stepcast/stepcast:vllm-openai-v0.19.0",
        "image_manifest_digest": "sha256:" + "a" * 64,
        "rows": [
            {
                "op_name": "all_gather",
                "framework": "NCCL",
                "kernel_source": "NCCL",
                "nccl_dtype": "half",
                "device": "NVIDIA H800",
                "version": "2.27.5",
                "num_gpus": 32,
                "message_size": 64,
                "message_size_bytes": 128,
                "latency": 0.1,
            }
        ],
        "measurements": [
            {
                "op": "all_gather",
                "world_size": 32,
                "message_elements": 64,
                "latency_ms_median_of_max_rank": 0.1,
                "max_rank_samples_ms": [0.1, 0.11, 0.09],
            }
        ],
    }


def _payload_for_rank(rank):
    payload = _payload()
    payload["world_size"] = rank
    payload["rows"][0]["num_gpus"] = rank
    payload["rows"][0]["message_size"] = rank * 2
    payload["rows"][0]["message_size_bytes"] = rank * 4
    payload["measurements"][0]["world_size"] = rank
    payload["measurements"][0]["message_elements"] = rank * 2
    return payload


def test_validate_sweep_payload_returns_row_count_and_runtime_identity():
    result = validate_sweep_payload(_payload(), expected_world_size=32)
    assert result["row_count"] == 1
    assert result["world_size"] == 32
    assert result["ops"] == ["all_gather", "reduce_scatter"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(vllm_version="0.19.0.post15"),
        lambda p: p.update(world_size=48),
        lambda p: p["rows"].append(copy.deepcopy(p["rows"][0])),
        lambda p: p["rows"][0].update(latency=0.0),
        lambda p: p["measurements"][0].update(max_rank_samples_ms=[0.1]),
        lambda p: p["measurements"][0].update(op="reduce_scatter"),
        lambda p: p["measurements"][0].update(latency_ms_median_of_max_rank=0.5),
        lambda p: p.update(backend="forged"),
        lambda p: p.update(framework_backend="forged"),
        lambda p: p.update(dtype="bfloat16"),
        lambda p: p["rows"][0].update(framework="FORGED"),
        lambda p: p["rows"][0].update(kernel_source="FORGED"),
        lambda p: p["rows"][0].update(nccl_dtype="int8"),
        lambda p: p["rows"][0].update(device="NVIDIA A100"),
        lambda p: p["rows"][0].update(version="2.27.3"),
    ],
)
def test_validate_sweep_payload_rejects_runtime_or_row_corruption(mutation):
    payload = _payload()
    mutation(payload)
    with pytest.raises((AssertionError, ValueError), match=r"sweep|version|world|duplicate|latency|samples"):
        validate_sweep_payload(payload, expected_world_size=32)


def test_validate_sweep_payload_does_not_mutate_input():
    payload = _payload()
    original = json.dumps(payload, sort_keys=True)
    validate_sweep_payload(payload, expected_world_size=32)
    assert json.dumps(payload, sort_keys=True) == original


def test_merge_sweep_payloads_reconciles_exact_ranks():
    payloads = [_payload_for_rank(rank) for rank in (2, 4, 8, 16, 32, 48, 64)]
    merged = merge_sweep_payloads(payloads)
    assert merged["row_count"] == 7
    assert merged["world_sizes"] == [2, 4, 8, 16, 32, 48, 64]
    assert merged["required_world_sizes"] == [2, 4, 8, 16, 32, 64]
    assert merged["extra_world_sizes"] == [48]
    assert len(merged["rows"]) == 7


def test_merge_sweep_payloads_rejects_missing_complete_replica_rank():
    payloads = [_payload_for_rank(rank) for rank in (4, 8, 16, 32, 48, 64)]
    with pytest.raises(ValueError, match=r"incomplete.*2"):
        merge_sweep_payloads(payloads)


def test_merge_sweep_payloads_rejects_unapproved_extra_rank():
    payloads = [_payload_for_rank(rank) for rank in (2, 4, 8, 16, 24, 32, 48, 64)]
    with pytest.raises(ValueError, match=r"outside.*24"):
        merge_sweep_payloads(payloads)


def test_merge_sweep_payloads_rejects_duplicate_rank_identity():
    payload = _payload()
    with pytest.raises(ValueError, match="duplicate"):
        merge_sweep_payloads([payload, copy.deepcopy(payload)])


def test_build_manifest_binds_image_config_and_runtime_provenance(tmp_path):
    source_paths = []
    for rank in (2, 4, 8, 16, 32, 48, 64):
        source_path = tmp_path / f"rank{rank}.json"
        source_path.write_text(json.dumps(_payload_for_rank(rank)), encoding="utf-8")
        source_paths.append(source_path)
    runtime_path = tmp_path / "runtime" / "exact_runtime_provenance.json"
    runtime_path.parent.mkdir()
    runtime_path.write_text('{"vllm_version":"0.19.0"}\n', encoding="utf-8")
    output_manifest = tmp_path / "p6_manifest.json"

    manifest = build_nccl_manifest(
        source_paths=source_paths,
        output_parquet=tmp_path / "p6_rows.parquet",
        output_manifest=output_manifest,
        image_config_digest="sha256:" + "b" * 64,
        runtime_provenance_path=runtime_path,
    )

    assert manifest["image_config_digest"] == "sha256:" + "b" * 64
    assert manifest["runtime_provenance_path"] == "runtime/exact_runtime_provenance.json"
    assert manifest["runtime_provenance_sha256"] == hashlib.sha256(runtime_path.read_bytes()).hexdigest()


def test_build_manifest_rejects_invalid_image_config_digest(tmp_path):
    with pytest.raises(ValueError, match="image config digest"):
        build_nccl_manifest(
            source_paths=[],
            output_parquet=tmp_path / "p6_rows.parquet",
            output_manifest=tmp_path / "p6_manifest.json",
            image_config_digest="not-a-digest",
            runtime_provenance_path=tmp_path / "runtime.json",
        )
