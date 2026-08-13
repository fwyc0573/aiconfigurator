"""P2 contract tests for Step4-Pro profiling readiness fixes."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common, config, models
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.common import PerfDataFilename
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations.attention import ContextAttention
from aiconfigurator.sdk.operations.communication import _require_exact_custom_allreduce_tp
from aiconfigurator.sdk.operations.moe import MoE
from aiconfigurator.sdk.perf_database import LoadedOpData
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.task_v2 import Task


def test_step4_moe_ops_emit_persisted_power_law_distribution():
    for model_id in ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4"):
        model = models.get_model(
            model_id,
            config.ModelConfig(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=1),
            "vllm",
        )
        moe_ops = [op for op in [*model.context_ops, *model.generation_ops] if op.__class__.__name__ == "MoE"]
        assert moe_ops
        assert {op._workload_distribution for op in moe_ops} == {"power_law_1.2"}


def test_step4_is_classified_as_moe_workspace_family():
    assert "STEP4" in BaseBackend.MOE_WORKSPACE_FAMILIES


@pytest.mark.parametrize(
    ("model_id", "expected_width"),
    [("stepfun-ai/Step4-Pro-V3", 6144), ("stepfun-ai/Step4-Pro-V4", 9216)],
)
def test_step4_moe_workspace_uses_routed_width(model_id, expected_width):
    model = models.get_model(
        model_id,
        config.ModelConfig(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=1),
        "vllm",
    )
    backend = BaseBackend()
    assert backend._moe_workspace_width(model, "STEP4", model._hidden_size) == expected_width


def test_context_attention_kv_write_uses_kv_cache_dtype():
    op = ContextAttention(
        "context_attention",
        1.0,
        n=8,
        n_kv=2,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    )
    mem_queries = []

    class FakeDatabase:
        def query_context_attention(self, *args, **kwargs):
            return PerformanceResult(1.0)

        def query_mem_op(self, size):
            mem_queries.append(size)
            return PerformanceResult(float(size))

    op.query(FakeDatabase(), batch_size=1, s=128, prefix=0)
    assert mem_queries[-2:] == [2 * 128 * 1, 2 * 128 * 1]


def test_step4_silicon_requires_coverage_manifest(monkeypatch, tmp_path):
    import aiconfigurator.sdk.task_v2 as task_v2

    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    (systems_root / "h800_sxm.yaml").write_text(
        yaml.safe_dump({"data_dir": "data/h800_sxm"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(task_v2, "get_systems_paths", lambda: [str(systems_root)])
    task = Task.__new__(Task)
    task._model_family = "STEP4"
    task.database_mode = common.DatabaseMode.SILICON.name
    task.system_name = "h800_sxm"
    task.backend_name = "vllm"
    task.backend_version = "0.19.0"
    with pytest.raises(ValueError, match="measured coverage"):
        task._validate_step4_database_mode()


def test_step4_silicon_manifest_requires_distribution_and_structural_keys(monkeypatch, tmp_path):
    import yaml

    import aiconfigurator.sdk.task_v2 as task_v2

    systems_root = tmp_path / "systems"
    data_root = systems_root / "data" / "h800_sxm" / "vllm" / "0.19.0"
    data_root.mkdir(parents=True)
    (systems_root / "h800_sxm.yaml").write_text(yaml.safe_dump({"data_dir": "data/h800_sxm"}), encoding="utf-8")
    manifest_path = data_root / "step4_pro_v3_v4_coverage.json"
    manifest_path.write_text(
        '{"version":"0.19.0","system":"h800_sxm","models":["stepfun-ai/Step4-Pro-V3","stepfun-ai/Step4-Pro-V4"],"status":"validated"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(task_v2, "get_systems_paths", lambda: [str(systems_root)])
    task = Task.__new__(Task)
    task._model_family = "STEP4"
    task.database_mode = common.DatabaseMode.SILICON.name
    task.system_name = "h800_sxm"
    task.backend_name = "vllm"
    task.backend_version = "0.19.0"
    assert not task._step4_silicon_coverage_available()

    def record(model, op_family, **fields):
        return {
            "model": model,
            "op_family": op_family,
            "backend": "vllm",
            "device": "h800_sxm",
            "system": "h800_sxm",
            "version": "0.19.0",
            "structural": {
                "identity": f"{model}:{op_family}:vllm:0.19.0:h800_sxm:shape",
                "axes": {"backend": "vllm", "version": "0.19.0", "device": "h800_sxm"},
            },
            **fields,
        }

    models_and_records = {
        model: [
            record(model, "attention"),
            record(model, "gemm"),
            record(
                model,
                "moe",
                hidden_size=6144 if model.endswith("V3") else 9216,
                inter_size=2048 if model.endswith("V3") else 3584,
                topk=16 if model.endswith("V3") else 8,
                num_experts=1024 if model.endswith("V3") else 384,
                moe_tp_size=1,
                moe_ep_size=1,
                quantization="fp8",
                distribution="power_law_1.2",
            ),
            record(model, "communication"),
        ]
        for model in ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4")
    }
    coverage_summary = {
        model: {
            family: {
                "required_count": 1,
                "measured_count": 1,
                "missing_count": 0,
                "duplicate_count": 0,
                "unassigned_count": 0,
            }
            for family in ("attention", "gemm", "moe", "communication")
        }
        for model in models_and_records
    }
    measured_key_path = data_root / "measured_keys.json"
    measured_key_payload = {
        "identities": sorted(
            record["structural"]["identity"] for records in models_and_records.values() for record in records
        )
    }
    measured_key_path.write_text(json.dumps(measured_key_payload), encoding="utf-8")
    provenance = {
        "measured_key_inventory": {
            "path": measured_key_path.name,
            "sha256": hashlib.sha256(measured_key_path.read_bytes()).hexdigest(),
        }
    }
    manifest_path.write_text(
        json.dumps(
            {
                "version": "0.19.0",
                "system": "h800_sxm",
                "models": list(models_and_records),
                "status": "validated",
                "backend": "vllm",
                "device": "h800_sxm",
                "distribution": "power_law_1.2",
                "required_op_families": ["attention", "gemm", "moe", "communication"],
                "coverage_keys": models_and_records,
                "coverage_summary": coverage_summary,
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    assert task._step4_silicon_coverage_available()
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["structural"]["identity"] = "not-a-real-key-power_law_1.2"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "0.19.0",
                "system": "h800_sxm",
                "models": list(models_and_records),
                "status": "validated",
                "backend": "vllm",
                "device": "h800_sxm",
                "distribution": "power_law_1.2",
                "required_op_families": ["attention", "gemm", "moe", "communication"],
                "coverage_keys": models_and_records,
                "coverage_summary": coverage_summary,
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    assert not task._step4_silicon_coverage_available()
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["backend"] = "sglang"
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["structural"]["identity"] = (
        "stepfun-ai/Step4-Pro-V3:attention:sglang:0.19.0:h800_sxm:shape"
    )
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["structural"]["axes"]["backend"] = "sglang"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "0.19.0",
                "system": "h800_sxm",
                "models": list(models_and_records),
                "status": "validated",
                "backend": "vllm",
                "device": "h800_sxm",
                "distribution": "power_law_1.2",
                "required_op_families": ["attention", "gemm", "moe", "communication"],
                "coverage_keys": models_and_records,
                "coverage_summary": coverage_summary,
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    assert not task._step4_silicon_coverage_available()
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["backend"] = "vllm"
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["structural"]["identity"] = (
        "stepfun-ai/Step4-Pro-V3:attention:vllm:0.19.0:h800_sxm:shape"
    )
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["structural"]["axes"]["backend"] = "vllm"
    models_and_records["stepfun-ai/Step4-Pro-V3"][0]["structural"]["identity"] = (
        "stepfun-ai/Step4-Pro-V3:attention:vllm:0.19.0:h800_sxm:shape"
    )
    models_and_records["stepfun-ai/Step4-Pro-V3"][1]["structural"]["identity"] = (
        "stepfun-ai/Step4-Pro-V3:attention:vllm:0.19.0:h800_sxm:shape"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "version": "0.19.0",
                "system": "h800_sxm",
                "models": list(models_and_records),
                "status": "validated",
                "backend": "vllm",
                "device": "h800_sxm",
                "distribution": "power_law_1.2",
                "required_op_families": ["attention", "gemm", "moe", "communication"],
                "coverage_keys": models_and_records,
                "coverage_summary": coverage_summary,
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    assert not task._step4_silicon_coverage_available()
    models_and_records["stepfun-ai/Step4-Pro-V3"][1]["structural"]["identity"] = (
        "stepfun-ai/Step4-Pro-V3:gemm:vllm:0.19.0:h800_sxm:shape"
    )
    models_and_records["stepfun-ai/Step4-Pro-V3"] = models_and_records["stepfun-ai/Step4-Pro-V3"][:1]
    manifest_path.write_text(
        json.dumps(
            {
                "version": "0.19.0",
                "system": "h800_sxm",
                "models": list(models_and_records),
                "status": "validated",
                "backend": "vllm",
                "device": "h800_sxm",
                "distribution": "power_law_1.2",
                "required_op_families": ["attention", "gemm", "moe", "communication"],
                "coverage_keys": models_and_records,
                "coverage_summary": coverage_summary,
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    assert not task._step4_silicon_coverage_available()


def test_step4_auditor_inventory_can_admit_complete_manifest(monkeypatch, tmp_path):
    import yaml

    import aiconfigurator.sdk.task_v2 as task_v2
    from tests.performance.step4_profiling_auditor import build_step4_coverage_inventory

    inventory = build_step4_coverage_inventory()
    inventory["status"] = "validated"
    for model_summary in inventory["coverage_summary"].values():
        for summary in model_summary.values():
            summary["measured_count"] = summary["required_count"]
            summary["missing_count"] = 0

    systems_root = tmp_path / "systems"
    data_root = systems_root / "data" / "h800_sxm" / "vllm" / "0.19.0"
    data_root.mkdir(parents=True)
    (systems_root / "h800_sxm.yaml").write_text(yaml.safe_dump({"data_dir": "data/h800_sxm"}), encoding="utf-8")
    measured_key_path = data_root / "measured_keys.json"
    measured_key_path.write_text(
        json.dumps(
            {
                "identities": sorted(
                    record["structural"]["identity"]
                    for records in inventory["coverage_keys"].values()
                    for record in records
                )
            }
        ),
        encoding="utf-8",
    )
    inventory["provenance"] = {
        "measured_key_inventory": {
            "path": measured_key_path.name,
            "sha256": hashlib.sha256(measured_key_path.read_bytes()).hexdigest(),
        }
    }
    (data_root / "step4_pro_v3_v4_coverage.json").write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(task_v2, "get_systems_paths", lambda: [str(systems_root)])

    task = Task.__new__(Task)
    task._model_family = "STEP4"
    task.database_mode = common.DatabaseMode.SILICON.name
    task.system_name = "h800_sxm"
    task.backend_name = "vllm"
    task.backend_version = "0.19.0"
    assert task._step4_silicon_coverage_available()
    inventory.pop("provenance")
    (data_root / "step4_pro_v3_v4_coverage.json").write_text(json.dumps(inventory), encoding="utf-8")
    assert not task._step4_silicon_coverage_available()
    inventory["provenance"] = {
        "measured_key_inventory": {
            "path": measured_key_path.name,
            "sha256": "0" * 64,
        }
    }
    (data_root / "step4_pro_v3_v4_coverage.json").write_text(json.dumps(inventory), encoding="utf-8")
    assert not task._step4_silicon_coverage_available()
    inventory.pop("coverage_summary")
    (data_root / "step4_pro_v3_v4_coverage.json").write_text(json.dumps(inventory), encoding="utf-8")
    assert not task._step4_silicon_coverage_available()


def _fake_vllm_moe_database(distribution):
    curve = {1: {"latency": 1.0, "energy": 2.0}}
    shape = {common.MoEQuantMode.fp8: {distribution: {2: {8: {2048: {8192: {1: {1: curve}}}}}}}}

    class FakeDatabase:
        backend = common.BackendName.vllm.value
        system = "h800_sxm"
        version = "0.19.0"
        _moe_data = LoadedOpData(shape, PerfDataFilename.moe, "fake-moe.parquet")

        @staticmethod
        def _interp_pr(latency, energy=0.0):
            return PerformanceResult(latency, energy=energy, source="silicon")

        @staticmethod
        def _query_silicon_or_hybrid(*, get_silicon, get_empirical, database_mode, error_msg):
            return get_silicon()

    return FakeDatabase()


def test_vllm_moe_missing_distribution_fails_fast(monkeypatch):
    monkeypatch.setattr(MoE, "load_data", lambda database: None)
    with pytest.raises(PerfDataNotAvailableError, match=r"power_law_1\.2"):
        MoE._query_moe_table(
            _fake_vllm_moe_database("uniform"),
            num_tokens=1,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.fp8,
            workload_distribution="power_law_1.2",
            database_mode=common.DatabaseMode.SILICON,
        )


def test_vllm_moe_exact_distribution_queries_without_relabeling(monkeypatch):
    monkeypatch.setattr(MoE, "load_data", lambda database: None)
    result = MoE._query_moe_table(
        _fake_vllm_moe_database("power_law_1.2"),
        num_tokens=1,
        hidden_size=2048,
        inter_size=8192,
        topk=2,
        num_experts=8,
        moe_tp_size=1,
        moe_ep_size=1,
        quant_mode=common.MoEQuantMode.fp8,
        workload_distribution="power_law_1.2",
        database_mode=common.DatabaseMode.SILICON,
    )
    assert float(result) == pytest.approx(1.0)


def test_h800_is_registered_with_a_first_class_system_spec():
    systems_root = Path(__file__).parents[3] / "src" / "aiconfigurator" / "systems"
    assert "h800_sxm" in common.SupportedSystems
    spec_path = systems_root / "h800_sxm.yaml"
    assert spec_path.is_file()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assert spec["gpu"]["mem_capacity"] == 85_029_158_912
    assert spec["misc"]["nccl_version"] == "2.27.5"


def test_cross_node_custom_allreduce_fails_fast_instead_of_scaling():
    with pytest.raises(PerfDataNotAvailableError, match="exact CustomAllReduce"):
        _require_exact_custom_allreduce_tp(requested_tp=16, node_gpus=8)
    assert _require_exact_custom_allreduce_tp(requested_tp=8, node_gpus=8) == 8
