"""Contracts for the Step4-Pro-Latest B300 core-provider collection."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent
HOST_SCRIPT = ROOT / "run_b300_provider_core_collection.sh"
SHARED_HOST_SCRIPT = ROOT / "run_b300_attention_collection.sh"
REMOTE_SCRIPT = ROOT / "remote_b300_attention_collection.sh"
VALIDATOR = ROOT / "validate_b300_provider_core_rows.py"

GROUPED_HEADER = "framework,version,device,op_name,kernel_source,provider,groups,num_tokens,n,k,quant_mode,latency\n"
ROUTER_HEADER = (
    "framework,version,device,op_name,kernel_source,provider,num_tokens,n,k,weight_dtype,output_dtype,latency\n"
)
QKV_HEADER = (
    "framework,version,device,op_name,kernel_source,provider,num_tokens,"
    "normalized_tensors,q_heads,kv_heads,head_dim,latency\n"
)
PREFIX = "VLLM,0.19.0.post20.dev26+gc820e5ae1,NVIDIA B300 SXM6 AC"


def _write_fixture(
    dataset_dir: Path,
    *,
    duplicate_grouped: bool = False,
    include_qkv: bool = True,
    replace_qkv_token: bool = False,
) -> None:
    dataset_dir.mkdir(parents=True)
    grouped_row = (
        f"{PREFIX},step4_grouped_gemm,vllm_step4pro_torch_einsum,"
        "vllm_step4pro_torch_einsum,8,1,1024,4096,bfloat16,0.1\n"
    )
    (dataset_dir / "step4_grouped_gemm_perf.txt").write_text(
        GROUPED_HEADER + grouped_row + (grouped_row if duplicate_grouped else ""),
        encoding="utf-8",
    )
    (dataset_dir / "step4_fp32_output_gemm_perf.txt").write_text(
        ROUTER_HEADER
        + (
            f"{PREFIX},step4_fp32_output_gemm,vllm.optimus_matmul_fp32,"
            "vllm.optimus_matmul_fp32,1,896,7168,bfloat16,float32,0.2\n"
        ),
        encoding="utf-8",
    )
    if include_qkv:
        from collector.vllm.collect_step4_provider import (
            get_step4_qkv_norm_rope_test_cases,
        )

        qkv_cases = get_step4_qkv_norm_rope_test_cases()
        if replace_qkv_token:
            qkv_cases[0] = [*qkv_cases[0]]
            qkv_cases[0][1] = 999_999
        qkv_rows = []
        for provider, num_tokens, normalized_tensors, q_heads, kv_heads, head_dim in qkv_cases:
            latency = 0.3 if provider == "vllm_step4pro_k_norm_rope" else 0.4
            qkv_rows.append(
                f"{PREFIX},step4_qkv_norm_rope,{provider},{provider},"
                f"{num_tokens},{normalized_tensors},{q_heads},{kv_heads},"
                f"{head_dim},{latency}\n"
            )
        (dataset_dir / "step4_qkv_norm_rope_perf.txt").write_text(
            QKV_HEADER + "".join(qkv_rows),
            encoding="utf-8",
        )


def _run_validator(
    dataset_dir: Path,
    work_dir: Path,
    output: Path,
    *,
    families: tuple[str, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    command = [
        sys.executable,
        str(VALIDATOR),
        "--dataset-dir",
        str(dataset_dir),
        "--work-dir",
        str(work_dir),
        "--output",
        str(output),
        "--expected-grouped-rows",
        "1",
        "--expected-router-rows",
        "1",
        "--expected-qkv-rows",
        "150",
    ]
    if families is not None:
        command.extend(["--families", *families])
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_host_wrapper_selects_core_provider_suite_and_exact_markers():
    source = HOST_SCRIPT.read_text(encoding="utf-8")

    assert "provider_core" in source
    assert "B300_STEP4_PROVIDER_CORE_COLLECTION" in source
    assert "B300_STEP4_PROVIDER_CORE_HOST" in source
    assert "b300_train_infra" in source
    assert 'exec bash "${REPO_ROOT}/tests/performance/step4_pro_latest/run_b300_attention_collection.sh"' in source


def test_shared_remote_runner_executes_exact_core_provider_ops():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'COLLECTION_SUITE="${COLLECTION_SUITE:-attention}"' in source
    assert "B300_STEP4_PROVIDER_CORE_COLLECTION" in source
    assert "run_step4_grouped_gemm" in source
    assert "run_step4_fp32_output_gemm" in source
    assert "run_step4_qkv_norm_rope" in source
    assert "--ops step4_grouped_gemm step4_fp32_output_gemm" in source
    assert "--ops step4_grouped_gemm step4_fp32_output_gemm step4_qkv_norm_rope" not in source
    assert "step4_grouped_gemm_perf.txt" in source
    assert "step4_fp32_output_gemm_perf.txt" in source
    assert "step4_qkv_norm_rope_perf.txt" in source


def test_provider_core_requires_an_explicit_collection_slice():
    wrapper_source = HOST_SCRIPT.read_text(encoding="utf-8")
    shared_source = SHARED_HOST_SCRIPT.read_text(encoding="utf-8")
    remote_source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'export PROVIDER_CORE_SLICE="${PROVIDER_CORE_SLICE:?' in wrapper_source
    assert 'PROVIDER_CORE_SLICE="${PROVIDER_CORE_SLICE:-all}"' in shared_source
    assert "export PROVIDER_CORE_SLICE='${PROVIDER_CORE_SLICE}'" in shared_source
    for source in (shared_source, remote_source):
        assert '"${COLLECTION_SUITE}" == "provider_core"' in source
        assert '"${PROVIDER_CORE_SLICE}" == "all"' in source
        assert "provider_core requires an explicit non-all PROVIDER_CORE_SLICE" in source


def test_shared_remote_runner_can_collect_grouped_router_without_qkv():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'PROVIDER_CORE_SLICE="${PROVIDER_CORE_SLICE:-all}"' in source
    assert '"${PROVIDER_CORE_SLICE}" != "grouped_router"' in source
    assert 'if [[ "${PROVIDER_CORE_SLICE}" == "grouped_router" ]]' in source
    assert "--ops step4_grouped_gemm step4_fp32_output_gemm" in source
    assert "selected_files+=(step4_grouped_gemm_perf.txt)" in source
    assert "selected_files+=(step4_fp32_output_gemm_perf.txt)" in source


def test_shared_remote_runner_can_collect_full_mfa_qkv_without_swa():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert '"${PROVIDER_CORE_SLICE}" != "qkv_full"' in source
    assert 'if [[ "${PROVIDER_CORE_SLICE}" == "qkv_full" ]]' in source
    assert "get_step4_qkv_norm_rope_test_cases" in source
    assert "FULL_K_NORM_ROPE_PROVIDER" in source
    assert "selected_files+=(step4_qkv_norm_rope_perf.txt)" in source
    assert '1 if mode == "smoke" else 75' in source
    assert '{"vllm_step4pro_k_norm_rope"}' in source
    assert ('expected = {}\nif provider_core_slice not in {"qkv_full", "qkv_swa"}:\n    expected.update(') in source


def test_shared_remote_runner_bounds_swa_qkv_annotation_overlay():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert '"${PROVIDER_CORE_SLICE}" != "qkv_swa"' in source
    assert ('QKNORM_ROPE_SOURCE_SHA256="5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783"') in source
    assert 'cat >"${OPTIMUS_OVERLAY_ROOT}/step4_swa_qkv_runtime_overlay.py"' in source
    assert 'if os.environ.get("PROVIDER_CORE_SLICE") == "qkv_swa":' in source
    assert 'for annotation_name in ("reload_from", "delay_w_load"):' in source
    assert "annotations[annotation_name] = cutlass.Constexpr" in source
    assert source.count("import step4_swa_qkv_runtime_overlay") >= 3
    assert "qknorm_rope_source.write_text" not in source


def test_shared_remote_runner_can_collect_swa_qkv_as_an_exact_slice():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'if [[ "${PROVIDER_CORE_SLICE}" == "qkv_swa" ]]' in source
    assert "SWA_QKV_NORM_ROPE_PROVIDER" in source
    assert "Expected 75 unique SWA QKV cases" in source
    assert "selected_files+=(step4_qkv_norm_rope_perf.txt)" in source
    assert '1 if mode == "smoke" else 75' in source
    assert '{"vllm_step4pro_qkv_norm_rope"}' in source


def test_full_provider_core_requires_matching_smoke_evidence():
    shared_source = SHARED_HOST_SCRIPT.read_text(encoding="utf-8")
    remote_source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    for source in (shared_source, remote_source):
        assert "PROVIDER_CORE_SMOKE_EVIDENCE" in source
        assert "PROVIDER_CORE_SMOKE_EVIDENCE_SHA256" in source
        assert "sha256sum --check" in source
        assert 'payload["mode"] != "smoke"' in source
        assert 'payload["suite"] != "provider_core"' in source
        assert 'payload["slice"] != expected_slice' in source
        assert 'payload["completed_cases"] != expected_completed_cases' in source


def test_full_provider_core_contract_counts_include_65k_cases():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")
    validator_namespace = runpy.run_path(str(VALIDATOR))

    assert source.count('1 if mode == "smoke" else 75') == 4
    assert '2 if mode == "smoke" else 150' not in source
    assert validator_namespace["DEFAULT_EXPECTED_ROWS"] == {
        "grouped": 75,
        "router": 75,
        "qkv": 150,
    }


def test_shared_remote_runner_records_qknorm_jit_runtime_identity():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'importlib.metadata.version("optimus-jit")' in source
    assert 'importlib.metadata.version("nvidia-cutlass-dsl")' in source
    assert '"qknorm_rope_source_sha256"' in source
    assert '"cutlass_dsl_source_sha256"' in source


def test_validator_queries_every_fixture_row_through_exact_consumers(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    output = tmp_path / "result.json"
    _write_fixture(dataset_dir)

    result = _run_validator(dataset_dir, tmp_path / "work", output)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["total_rows"] == 152
    assert payload["exact_consumer_matches"] == 152
    assert payload["silicon_source_rows"] == 152
    assert payload["duplicate_physical_keys"] == 0
    assert payload["minimum_latency_ms"] == pytest.approx(0.1)
    assert payload["maximum_latency_ms"] == pytest.approx(0.4)


def test_validator_rejects_duplicate_physical_keys(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    output = tmp_path / "result.json"
    _write_fixture(dataset_dir, duplicate_grouped=True)

    result = _run_validator(dataset_dir, tmp_path / "work", output)

    assert result.returncode != 0
    assert "duplicate physical key" in result.stderr


def test_validator_rejects_wrong_but_unique_qkv_token_set(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    output = tmp_path / "result.json"
    _write_fixture(dataset_dir, replace_qkv_token=True)

    result = _run_validator(dataset_dir, tmp_path / "work", output)

    assert result.returncode != 0
    assert "QKV physical key set mismatch" in result.stderr


def test_validator_accepts_explicit_grouped_router_slice_without_qkv_file(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    output = tmp_path / "result.json"
    _write_fixture(dataset_dir, include_qkv=False)

    result = _run_validator(
        dataset_dir,
        tmp_path / "work",
        output,
        families=("grouped", "router"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["grouped_rows"] == 1
    assert payload["router_rows"] == 1
    assert payload["qkv_rows"] == 0
    assert payload["total_rows"] == 2
    assert payload["exact_consumer_matches"] == 2
