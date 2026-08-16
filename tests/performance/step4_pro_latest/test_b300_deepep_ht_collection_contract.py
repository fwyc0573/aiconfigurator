"""Static contracts for the B300 DeepEP HT multi-node collection wrappers."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST_SCRIPT = (ROOT / "run_b300_deepep_ht_collection.sh").read_text()
REMOTE_SCRIPT = (ROOT / "remote_b300_deepep_ht_collection.sh").read_text()
PREFLIGHT_SCRIPT = ROOT / "run_step4_deepep_ht_nccl_preflight.py"


def test_host_wrapper_pins_b300_image_topologies_and_rdma() -> None:
    for marker in (
        'CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"',
        "b300_train_infra",
        'POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"',
        'EP_SIZE="${EP_SIZE:-16}"',
        "EP_SIZE == 16",
        "EP_SIZE == 32",
        "NNODES=2",
        "NNODES=4",
        '--replica "${NNODES}"',
        "--gpu 8",
        "--custom-resources=rdma/mlnx_shared=8",
        "--custom-resources=mellanox.com/mlnx_rdma=1",
        "--host-network",
        "--topo-group=yes",
        "--set-env=DISTRIBUTED_JOB=true",
        "NVSHMEM_ENABLE_NIC_PE_MAPPING",
        '"preflight"',
        "run_step4_deepep_ht_nccl_preflight.py",
    ):
        assert marker in HOST_SCRIPT


def test_host_wrapper_reuses_pinned_source_reconstruction_and_streams_payload() -> None:
    for marker in (
        "image_to_pinned_vllm.patch.gz",
        "pinned_vllm_manifest.sha256.gz",
        "pinned_identity_fulltrees_pack_v2",
        "remote_b300_source_probe.sh",
        "aic_payload.tar",
        'tar cf - -C "${PAYLOAD_ROOT}" .',
        'RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"',
        '-l "${RJOB_LABEL}"',
        "brainctl delete rjob",
        "cleanup_replicas_final.log",
    ):
        assert marker in HOST_SCRIPT
    assert "${TMPDIR:-/tmp}" not in HOST_SCRIPT
    assert 'ARTIFACT_ROOT="${ARTIFACT_ROOT:-/tmp/' not in HOST_SCRIPT


def test_remote_wrapper_runs_one_torchrun_per_node_with_exact_driver() -> None:
    for marker in (
        "remote_b300_source_probe.sh",
        "pinned_manifest_files_verified",
        "run_step4_deepep_ht_distributed.py",
        'torchrun --nproc-per-node="${PROC_PER_NODE}"',
        '--nnodes="${NODE_COUNT}"',
        '--node-rank="${NODE_RANK}"',
        '--master-addr="${MASTER_ADDR}"',
        '--ep-size "${EP_SIZE}"',
        '--mode "${MODE}"',
        "DeepEPHTAll2AllManager",
        "vllm_deepep_high_throughput",
        "B300_DEEPEP_HT_COLLECTION=PASS",
        "B300_DEEPEP_HT_NCCL_PREFLIGHT=PASS",
        'if [[ -z "${NCCL_IB_HCA:-}" ]]',
        "run_step4_deepep_ht_nccl_preflight.py",
        "STEP4_NCCL_PREFLIGHT_RANK=PASS",
    ):
        assert marker in REMOTE_SCRIPT
    assert "NCCL_SOCKET_IFNAME=" not in REMOTE_SCRIPT
    assert "NCCL_IB_HCA=" not in REMOTE_SCRIPT


def test_remote_wrapper_uses_the_live_verified_b300_cuda_library_contract() -> None:
    assert "export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64\n" in REMOTE_SCRIPT
    assert "${LD_LIBRARY_PATH:+" not in REMOTE_SCRIPT
    assert 'print("capability", torch.cuda.get_device_capability())' in REMOTE_SCRIPT


def test_nccl_preflight_script_exists_under_the_performance_test_tree() -> None:
    assert PREFLIGHT_SCRIPT.is_file()


def test_remote_wrapper_counts_concurrent_rank_markers_by_occurrence(
    tmp_path: Path,
) -> None:
    marker = "STEP4_NCCL_PREFLIGHT_RANK=PASS"
    observed_log = tmp_path / "interleaved_nccl_preflight.log"
    observed_log.write_text(marker * 8 + "\n", encoding="utf-8")

    assert 'grep -oF -- "${marker}" "${log_path}"' in REMOTE_SCRIPT
    assert "grep -c '^STEP4_NCCL_PREFLIGHT_RANK=PASS '" not in REMOTE_SCRIPT
    assert "grep -c '^STEP4_DEEPEP_HT_RANK=PASS '" not in REMOTE_SCRIPT
    assert 'count_marker_occurrences "STEP4_NCCL_PREFLIGHT_RANK=PASS"' in REMOTE_SCRIPT
    assert 'count_marker_occurrences "STEP4_DEEPEP_HT_RANK=PASS"' in REMOTE_SCRIPT

    result = subprocess.run(
        [
            "bash",
            "-c",
            ('marker="$1"; log_path="$2"; grep -oF -- "${marker}" "${log_path}" | wc -l | tr -d \'[:space:]\''),
            "_",
            marker,
            str(observed_log),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "8"
