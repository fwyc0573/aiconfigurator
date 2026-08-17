import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = (ROOT / "run_b300_two_node_smoke.sh").read_text()
REMOTE_SCRIPT = (ROOT / "remote_b300_single_smoke.sh").read_text()


def test_two_node_wrapper_pins_topology_and_dummy_model() -> None:
    for marker in (
        'CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-2G}"',
        "b300_train_infra",
        'POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"',
        "--replica 2",
        "--gpu 8",
        "--cpu 64",
        "--memory 600000",
        "--host-network=true",
        "--custom-resources=rdma/mlnx_shared=8",
        "--custom-resources=mellanox.com/mlnx_rdma=1",
        "--topo-group=yes",
        "DISTRIBUTED_JOB=true",
        "DATA_PARALLEL_SIZE=16",
        "DATA_PARALLEL_SIZE_LOCAL=8",
        "VLLM_ALL2ALL_BACKEND=allgather_reducescatter",
        "VLLM_ENABLE_SEQUENCE_PARALLEL=0",
        "NCCL_PREFLIGHT_EVIDENCE",
        'grep -o "NCCL_PREFLIGHT_RANK=PASS"',
        'grep -o "NCCL_PREFLIGHT_NODE=PASS"',
        "step4pro_smoke_14l_dummy",
        "OPTIMUS_WHEEL_URL",
        "OPTIMUS_WHEEL_SHA256",
        "REQUIRED_B300_GPUS=16",
        "B300_QUOTA_EVIDENCE",
        "require_b300_quota_evidence",
        "b300_quota_evidence.txt",
        "b300_quota_evidence.sha256",
        'EVIDENCE_PULL_TIMEOUT_SECONDS="${EVIDENCE_PULL_TIMEOUT_SECONDS:-300}"',
        'REMOTE_EXEC_TIMEOUT_SECONDS="${REMOTE_EXEC_TIMEOUT_SECONDS:-${LIVE_TIMEOUT_SECONDS}}"',
        "MIN_REMOTE_EXEC_TIMEOUT_SECONDS=",
        "Remote execution timeout is too short for validation and evidence pulls",
        "--predict-only",
        "predict_only.log",
        "predict_candidate_count",
    ):
        assert marker in SCRIPT
    assert "--gang-start" not in SCRIPT
    assert "qy1-pt" not in SCRIPT
    assert "step2-alignment-jfs" not in SCRIPT
    assert "deepep_high_throughput" not in SCRIPT
    assert "NVSHMEM_ENABLE_NIC_PE_MAPPING" not in SCRIPT


def test_two_node_holder_persists_platform_distributed_environment() -> None:
    for marker in (
        'DISTRIBUTED_ENV_FILE="/home/step4pro-distributed-${RJOB_NAME}.env"',
        'DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS="${DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS:-180}"',
        "NODE_RANK NODE_COUNT MASTER_ADDR PROC_PER_NODE",
        "NCCL_*)",
        "printf 'export %s=%q",
        "env_ready=0",
        "env_ready_deadline=",
        'test -s "${DISTRIBUTED_ENV_FILE}"',
        'test "${env_ready}" = "1"',
        "source '${DISTRIBUTED_ENV_FILE}'",
        "distributed_env_ready=PASS",
    ):
        assert marker in SCRIPT
    assert "/proc/1/environ" not in SCRIPT
    assert "NVSHMEM_*" not in SCRIPT


def test_two_node_wrapper_transfers_runtime_contract_to_each_replica() -> None:
    assert 'cp "${CONTRACT_LIB}" "${PAYLOAD_ROOT}/b300_runtime_contract.sh"' in SCRIPT
    assert ("export RUNTIME_CONTRACT_LIB='${REMOTE_BOOTSTRAP_ROOT}/b300_runtime_contract.sh'") in SCRIPT


def test_two_node_wrapper_streams_payload_and_cleans_exact_resources() -> None:
    assert 'RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"' in SCRIPT
    assert '-l "${RJOB_LABEL}"' in SCRIPT
    assert 'tar cf - -C "${PAYLOAD_ROOT}" .' in SCRIPT
    assert '/kubebrain/brainctl -n "${NAMESPACE}" exec -i' in SCRIPT
    assert "brainctl delete rjob" in SCRIPT
    assert SCRIPT.count("brainctl delete rjob") == 1
    assert "cleanup_inventory_is_empty" in SCRIPT
    assert "assert_runtime_log_clean" in SCRIPT
    assert "cleanup_replicas_final.log" in SCRIPT
    assert "TWO_NODE_HOST_WRAPPER=PASS" in SCRIPT
    assert "Using AgRsAll2AllManager all2all manager" in SCRIPT
    assert 'if grep -Eq "Using DeepEP[A-Za-z0-9_]*All2AllManager" "${server_log}"; then' in SCRIPT
    assert "Unexpected Step MoE automatic backend selection" in SCRIPT


def test_two_node_runner_collects_all_evidence_before_single_job_delete() -> None:
    validation_probe = 'test -f "${evidence_root}/remote_validation_ready"'

    assert validation_probe in SCRIPT
    evidence_pull = 'tar cf - -C /home "$(basename "${evidence_root}")"'
    result_validation = "DISTRIBUTED_RUNTIME_VALIDATION=PASS"
    cleanup_call = "\ncleanup\ncleanup_exit=$?"
    assert evidence_pull in SCRIPT
    assert result_validation in SCRIPT
    assert cleanup_call in SCRIPT
    assert SCRIPT.index(validation_probe) < SCRIPT.index(evidence_pull)
    assert SCRIPT.index(evidence_pull) < SCRIPT.index(result_validation)
    assert SCRIPT.index(result_validation) < SCRIPT.index(cleanup_call)

    assert 'REMOTE_VALIDATION_READY_FILE="${EVIDENCE_ROOT}/remote_validation_ready"' in REMOTE_SCRIPT
    validation_paths = re.findall(
        r'touch "\$\{REMOTE_VALIDATION_READY_FILE\}"\n'
        r"\s*hold_distributed_runtime_for_host_cleanup",
        REMOTE_SCRIPT,
    )
    assert len(validation_paths) == 2
    assert "coordinated_shutdown" not in SCRIPT
    assert "COORDINATED_SHUTDOWN" not in REMOTE_SCRIPT
    assert "remote_shutdown_armed" not in SCRIPT
    assert "REMOTE_SHUTDOWN_ARMED" not in REMOTE_SCRIPT
    assert "remote_result_ready" not in SCRIPT
    assert 'stop_server\nSERVER_PID=""\nprintf \'ONE_GPU_REMOTE_EXECUTION=PASS' not in REMOTE_SCRIPT


def test_two_node_wrapper_runs_predict_only_before_live_launch() -> None:
    predict = "/kubebrain/brainctl rjob launch --predict-only"
    live = "/kubebrain/brainctl rjob launch --detach"
    assert predict in SCRIPT
    assert live in SCRIPT
    assert SCRIPT.index(predict) < SCRIPT.index(live)


def test_headless_validation_uses_node_local_backend_evidence() -> None:
    backend_marker = (
        "DeepEP runtime not available; using allgather_reducescatter all2all backend without sequence parallelism"
    )

    assert backend_marker in REMOTE_SCRIPT
    assert "validate_distributed_all2all_runtime 0" in REMOTE_SCRIPT
    assert "validate_distributed_all2all_runtime 1" in REMOTE_SCRIPT
    assert "agrs_manager_replica_count=0" in SCRIPT
    assert 'agrs_manager_replica_count="$((agrs_manager_replica_count + 1))"' in SCRIPT
    assert 'test "${agrs_manager_replica_count}" -ge 1' in SCRIPT
    assert ('    grep -q "Using AgRsAll2AllManager all2all manager" "${server_log}"\n') not in SCRIPT
