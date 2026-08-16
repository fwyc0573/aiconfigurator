from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "run_b300_two_node_nccl_preflight.sh"


def _script() -> str:
    assert SCRIPT_PATH.is_file(), f"missing preflight script: {SCRIPT_PATH}"
    return SCRIPT_PATH.read_text()


def test_preflight_requests_full_b300_rdma_launch_contract() -> None:
    script = _script()
    for marker in (
        'CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"',
        "--replica 2",
        "--gpu 8",
        "--host-network=true",
        "--custom-resources=rdma/mlnx_shared=8",
        "--custom-resources=mellanox.com/mlnx_rdma=1",
        "--topo-group=yes",
        "--set-env=DISTRIBUTED_JOB=true",
        "--enable-jobutil-config=false",
        "b300_train_infra",
        'POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"',
    ):
        assert marker in script


def test_preflight_fails_fast_without_platform_hca_and_runs_16_rank_nccl() -> None:
    script = _script()
    for marker in (
        "NCCL_PREFLIGHT_HCA=FAIL",
        'NCCL_IB_HCA="${NCCL_IB_HCA:-}"',
        '[[ -n "${NCCL_IB_HCA}" ]]',
        "export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64",
        'torchrun --nnodes "${NODE_COUNT}"',
        '--nproc_per_node "${PROC_PER_NODE}"',
        'dist.init_process_group("nccl")',
        "expected = world_size * (world_size + 1) / 2",
        "NCCL_PREFLIGHT_RANK=PASS",
        "NCCL_PREFLIGHT_NODE=PASS",
        'grep -o "NCCL_PREFLIGHT_RANK=PASS"',
        'grep -o "NCCL_PREFLIGHT_NODE=PASS"',
        "expected=136",
    ):
        assert marker in script
    assert "NCCL_IB_DISABLE=1" not in script
    assert "export NCCL_IB_HCA=" not in script
    assert "/usr/local/cuda-12.8/compat" not in script
    assert "/tmp/" not in script.replace("/data/ycfeng/tmp/", "")


def test_preflight_uses_exact_queries_and_zero_resource_cleanup() -> None:
    script = _script()
    for marker in (
        'RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"',
        '-l "${RJOB_LABEL}"',
        'brainctl delete rjob "${RJOB_NAME}"',
        "cleanup_rjob_final.log",
        "cleanup_replicas_final.log",
        "cleanup_local_processes.log",
        "NCCL_PREFLIGHT_HOST=PASS",
    ):
        assert marker in script
