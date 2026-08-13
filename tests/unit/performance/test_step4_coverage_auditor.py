"""Tests for the task-local Step4 structural-key coverage auditor."""

import pytest

from tests.performance.step4_profiling_auditor import (
    MODELS,
    REQUIRED_FAMILIES,
    build_step4_coverage_inventory,
)

pytestmark = pytest.mark.unit


def test_auditor_emits_complete_structural_family_inventory():
    inventory = build_step4_coverage_inventory()
    assert inventory["status"] == "planned"
    assert inventory["models"] == list(MODELS)
    assert inventory["required_op_families"] == list(REQUIRED_FAMILIES)
    for model in MODELS:
        records = inventory["coverage_keys"][model]
        assert records
        assert {record["op_family"] for record in records} == set(REQUIRED_FAMILIES)
        identities = [record["structural"]["identity"] for record in records]
        assert len(identities) == len(set(identities))
        assert all(identity.startswith(f"{model}:") for identity in identities)


def test_auditor_preserves_step4_dtype_and_distribution_contract():
    inventory = build_step4_coverage_inventory()
    for model in MODELS:
        gemm_records = [record for record in inventory["coverage_keys"][model] if record["op_family"] == "gemm"]
        moe_records = [record for record in inventory["coverage_keys"][model] if record["op_family"] == "moe"]
        assert {record["structural"]["axes"]["gemm_dtype"] for record in gemm_records} == {"bfloat16", "fp8"}
        assert all("fp8_block" not in record["structural"]["identity"] for record in gemm_records)
        assert moe_records
        assert {record["structural"]["axes"]["distribution"] for record in moe_records} == {"power_law_1.2"}
        assert {record["structural"]["axes"]["quantization"] for record in moe_records} == {"fp8"}


def test_auditor_reports_missing_keys_without_claiming_measurement():
    inventory = build_step4_coverage_inventory()
    for model in MODELS:
        for family in REQUIRED_FAMILIES:
            summary = inventory["coverage_summary"][model][family]
            assert summary["required_count"] > 0
            assert summary["measured_count"] == 0
            assert summary["missing_count"] == summary["required_count"]
            assert summary["duplicate_count"] == 0
            assert summary["unassigned_count"] == 0


def test_auditor_derives_complete_replica_communication_ranks():
    inventory = build_step4_coverage_inventory()
    for model in MODELS:
        communication = [
            record for record in inventory["coverage_keys"][model] if record["op_family"] == "communication"
        ]
        rank_by_op = {record["structural"]["axes"]["op"]: set() for record in communication}
        for record in communication:
            axes = record["structural"]["axes"]
            rank_by_op.setdefault(axes["op"], set()).add(axes["rank"])
        assert rank_by_op["custom_allreduce"] == {2, 4}
        expected_nccl_ranks = {2, 4, 8, 16, 32, 64}
        assert rank_by_op["nccl_all_gather"] == expected_nccl_ranks
        assert rank_by_op["nccl_reduce_scatter"] == expected_nccl_ranks
        assert 48 not in set().union(*rank_by_op.values())


def test_auditor_classifies_each_replica_topology_by_actual_collective_path():
    inventory = build_step4_coverage_inventory()
    for model in MODELS:
        topology_audit = inventory["communication_topology_audit"][model]
        assert len(topology_audit) == 18
        topology_by_shape = {(item["world_size"], item["attention_tp_size"]): item for item in topology_audit}
        assert all(item["status"] == "runnable" for item in topology_audit)

        assert topology_by_shape[(1, 1)]["status"] == "runnable"
        assert topology_by_shape[(1, 1)]["collective_class"] == "none"

        assert topology_by_shape[(4, 1)]["collective_class"] == "nccl_only"
        assert topology_by_shape[(4, 2)]["collective_class"] == "custom_allreduce_and_nccl"
        assert topology_by_shape[(4, 4)]["collective_class"] == "custom_allreduce_only"

        assert topology_by_shape[(8, 2)]["status"] == "runnable"
        assert topology_by_shape[(8, 2)]["collective_class"] == "custom_allreduce_and_nccl"

        assert topology_by_shape[(16, 1)]["status"] == "runnable"
        assert topology_by_shape[(16, 1)]["collective_class"] == "nccl_only"
        assert topology_by_shape[(16, 2)]["status"] == "runnable"
        assert topology_by_shape[(16, 2)]["collective_class"] == "custom_allreduce_and_nccl"
        assert topology_by_shape[(16, 2)]["invalid_queries"] == []
        assert {
            (query["op"], query["rank"], query["producer_path"]) for query in topology_by_shape[(16, 2)]["queries"]
        } == {
            ("custom_allreduce", 2, "MoEDispatch"),
            ("custom_allreduce", 2, "built_graph"),
            ("nccl_all_gather", 16, "MoEDispatch"),
            ("nccl_reduce_scatter", 16, "MoEDispatch"),
        }
