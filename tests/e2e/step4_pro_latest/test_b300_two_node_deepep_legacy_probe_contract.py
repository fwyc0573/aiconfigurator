from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "run_b300_two_node_deepep_legacy_probe.sh"


def _script() -> str:
    assert SCRIPT_PATH.is_file(), f"missing legacy probe: {SCRIPT_PATH}"
    return SCRIPT_PATH.read_text()


def test_legacy_probe_uses_documented_process_launcher_and_full_rdma() -> None:
    script = _script()
    assert "HISTORICAL/INACTIVE" in script
    assert "not used by the active AgRs runtime path" in script
    for marker in (
        "--i-know-i-am-using-legacy-rlaunch",
        "--replica 2",
        "--host-network=true",
        "--custom-resources=rdma/mlnx_shared=8",
        "--custom-resources=mellanox.com/mlnx_rdma=1",
        "--topo-group=yes",
        "--set-env=DISTRIBUTED_JOB=true",
        "--enable-sshd=false",
        'CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"',
        "b300_train_infra",
        'POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"',
    ):
        assert marker in script
    assert '--name "${RJOB_NAME}"' not in script


def test_legacy_probe_runs_explicit_nvshmem_and_deepep_buffer_contract() -> None:
    script = _script()
    for marker in (
        "NVSHMEM_DISABLE_NCCL",
        "NVSHMEM_ENABLE_NIC_PE_MAPPING",
        "NVSHMEM_MAX_TEAMS",
        "NVSHMEM_IB_TRAFFIC_CLASS",
        "nvshmem.init(",
        "num_rdma_bytes=1024 * 1024 * 1024",
        "DEEPEP_LEGACY_NCCL=PASS",
        "DEEPEP_LEGACY_NVSHMEM=PASS",
        "DEEPEP_LEGACY_BUFFER=PASS",
        "DEEPEP_LEGACY_DESTROY=PASS",
        "findmnt /dev/shm",
        "/proc/self/ns/ipc",
    ):
        assert marker in script
    assert "NCCL_IB_DISABLE=1" not in script
    assert "VLLM_ALL2ALL_BACKEND=allgather_reducescatter" not in script
    assert "/tmp/" not in script.replace("/data/ycfeng/tmp/", "")


def test_legacy_probe_is_bounded_and_preserves_disk_evidence() -> None:
    script = _script()
    for marker in (
        "timeout --signal=TERM --kill-after=30s",
        "LIVE_TIMEOUT_SECONDS",
        "legacy_probe.log",
        "DEEPEP_LEGACY_PROBE_HOST=PASS",
    ):
        assert marker in script
