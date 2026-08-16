from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent / "run_b300_two_node_smoke.sh").read_text()


def test_two_node_wrapper_pins_topology_and_dummy_model() -> None:
    for marker in (
        'CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"',
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
        "deepep_high_throughput",
        'NVSHMEM_ENABLE_NIC_PE_MAPPING="${NVSHMEM_ENABLE_NIC_PE_MAPPING:-1}"',
        '-e NVSHMEM_ENABLE_NIC_PE_MAPPING="${NVSHMEM_ENABLE_NIC_PE_MAPPING}"',
        "NCCL_PREFLIGHT_EVIDENCE",
        'grep -o "NCCL_PREFLIGHT_RANK=PASS"',
        'grep -o "NCCL_PREFLIGHT_NODE=PASS"',
        "step4pro_smoke_14l_dummy",
        "OPTIMUS_WHEEL_URL",
        "OPTIMUS_WHEEL_SHA256",
    ):
        assert marker in SCRIPT
    assert "--gang-start" not in SCRIPT
    assert "qy1-pt" not in SCRIPT
    assert "step2-alignment-jfs" not in SCRIPT


def test_two_node_holder_persists_platform_distributed_environment() -> None:
    for marker in (
        'DISTRIBUTED_ENV_FILE="/home/step4pro-distributed-${RJOB_NAME}.env"',
        'DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS="${DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS:-180}"',
        "NODE_RANK NODE_COUNT MASTER_ADDR PROC_PER_NODE",
        "NCCL_*|NVSHMEM_*",
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


def test_two_node_wrapper_streams_payload_and_cleans_exact_resources() -> None:
    assert 'RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"' in SCRIPT
    assert '-l "${RJOB_LABEL}"' in SCRIPT
    assert 'tar cf - -C "${PAYLOAD_ROOT}" .' in SCRIPT
    assert '/kubebrain/brainctl -n "${NAMESPACE}" exec -i' in SCRIPT
    assert "brainctl delete rjob" in SCRIPT
    assert "cleanup_replicas_final.log" in SCRIPT
    assert "TWO_NODE_HOST_WRAPPER=PASS" in SCRIPT
    assert "Using DeepEPHTAll2AllManager" in SCRIPT
    assert "backend=HT.*op=dispatch" in SCRIPT
    assert "backend=HT.*op=combine" in SCRIPT
