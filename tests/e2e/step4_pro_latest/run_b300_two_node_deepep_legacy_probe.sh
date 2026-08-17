#!/usr/bin/env bash
set -euo pipefail

# HISTORICAL/INACTIVE: not used by the active AgRs runtime path.
# Retained only for DeepEP/NVSHMEM diagnosis.

NAMESPACE="${NAMESPACE:-shai-core}"
RJOB_NAME="${RJOB_NAME:-s4p-legacy-deepep-$(date +%m%d-%H%M%S)}"
IMAGE="${IMAGE:-hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled}"
CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814/legacy_deepep_${RJOB_NAME}}"
CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"
LIVE_TIMEOUT_SECONDS="${LIVE_TIMEOUT_SECONDS:-1200}"
MASTER_PORT="${MASTER_PORT:-29673}"
PROBE_PATH="/home/deepep-legacy-probe-${RJOB_NAME}.py"

if (( ${#RJOB_NAME} > 50 )); then
    echo "RJOB_NAME exceeds platform limit: ${#RJOB_NAME} > 50" >&2
    exit 1
fi
mkdir -p "${ARTIFACT_ROOT}"

PROBE_SOURCE="$(
    cat <<'PY'
from __future__ import annotations

import os
import socket

import deep_ep
import torch
import torch.distributed as dist


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

value = torch.tensor([float(rank + 1)], device="cuda")
dist.all_reduce(value)
torch.cuda.synchronize()
expected = world_size * (world_size + 1) / 2
actual = float(value.item())
if actual != expected:
    raise RuntimeError(f"NCCL mismatch: actual={actual} expected={expected}")
print(
    "DEEPEP_LEGACY_NCCL=PASS",
    f"host={socket.gethostname()}",
    f"rank={rank}",
    f"actual={actual}",
    f"expected={expected}",
    flush=True,
)

import nvshmem.core as nvshmem


def is_nvshmem_initialized() -> bool:
    try:
        return int(nvshmem.init_status()) >= int(
            nvshmem.InitStatus.STATUS_IS_INITIALIZED
        )
    except Exception:
        return False


if not is_nvshmem_initialized():
    try:
        from cuda.core import Device as CudaDevice
    except ImportError:
        from cuda.core.experimental import Device as CudaDevice

    uid = (
        nvshmem.get_unique_id()
        if rank == 0
        else nvshmem.get_unique_id(empty=True)
    )
    uid_objects = [uid]
    dist.broadcast_object_list(uid_objects, src=0)
    dist.barrier()
    nvshmem.init(
        device=CudaDevice(local_rank),
        uid=uid_objects[0],
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )
print(
    "DEEPEP_LEGACY_NVSHMEM=PASS",
    f"rank={rank}",
    f"world_size={world_size}",
    flush=True,
)

buffer = deep_ep.Buffer(
    group=dist.group.WORLD,
    num_nvl_bytes=1024 * 1024 * 1024,
    num_rdma_bytes=1024 * 1024 * 1024,
    low_latency_mode=False,
    num_qps_per_rank=10,
    explicitly_destroy=True,
)
dist.barrier()
torch.cuda.synchronize()
print(
    "DEEPEP_LEGACY_BUFFER=PASS",
    f"host={socket.gethostname()}",
    f"rank={rank}",
    flush=True,
)

buffer.destroy()
print("DEEPEP_LEGACY_DESTROY=PASS", f"rank={rank}", flush=True)
dist.destroy_process_group()
PY
)"
PROBE_B64="$(printf '%s' "${PROBE_SOURCE}" | base64 -w0)"

worker_command="$(
    cat <<'WORKER'
set -euo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-INFO}"
export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
export NVSHMEM_IB_TRAFFIC_CLASS="${NVSHMEM_IB_TRAFFIC_CLASS:-186}"
export NVSHMEM_DISABLE_NCCL="${NVSHMEM_DISABLE_NCCL:-1}"
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_MAX_TEAMS="${NVSHMEM_MAX_TEAMS:-128}"

: "${NODE_RANK:?NODE_RANK is required}"
: "${NODE_COUNT:?NODE_COUNT is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${PROC_PER_NODE:?PROC_PER_NODE is required}"
: "${PROBE_B64:?PROBE_B64 is required}"
: "${PROBE_PATH:?PROBE_PATH is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA is required}"

if [[ -z "${NVSHMEM_HCA_PE_MAPPING:-}" ]]; then
    hca_names="${NCCL_IB_HCA#=}"
    IFS=',' read -r -a hca_array <<<"${hca_names}"
    mapping=()
    for hca in "${hca_array[@]}"; do
        [[ -n "${hca}" ]] && mapping+=("${hca}:1:1")
    done
    if (( ${#mapping[@]} == 0 )); then
        echo "Unable to derive NVSHMEM_HCA_PE_MAPPING" >&2
        exit 1
    fi
    printf -v mapping_joined '%s,' "${mapping[@]}"
    export NVSHMEM_HCA_PE_MAPPING="${mapping_joined%,}"
fi

echo "DEEPEP_LEGACY_ENV node_rank=${NODE_RANK} hca=${NCCL_IB_HCA} gid=${NCCL_IB_GID_INDEX:-} socket=${NCCL_SOCKET_IFNAME:-}"
findmnt /dev/shm
readlink /proc/self/ns/ipc
ulimit -l
ls -1 /sys/class/infiniband
nvidia-smi -L

printf '%s' "${PROBE_B64}" | base64 -d >"${PROBE_PATH}"
torchrun --nnodes "${NODE_COUNT}" \
    --nproc_per_node "${PROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${PROBE_PATH}"
echo "DEEPEP_LEGACY_NODE=PASS node_rank=${NODE_RANK}"
WORKER
)"

set +e
sudo -n systemd-run --scope -p MemoryMax="${CONTROL_MEMORY_MAX}" \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${LIVE_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl launch \
    --i-know-i-am-using-legacy-rlaunch \
    --replica 2 --replica-prefix --replica-restart=never \
    --charged-group="${CHARGED_GROUP}" --private-machine=group \
    --positive-tags="${POSITIVE_TAGS}" \
    --set-env=DISTRIBUTED_JOB=true \
    --host-network=true \
    --custom-resources=rdma/mlnx_shared=8 \
    --custom-resources=mellanox.com/mlnx_rdma=1 \
    --topo-group=yes \
    --gpu=8 --cpu=32 --memory=300000 \
    --enable-sshd=false \
    --image "${IMAGE}" --entrypoint /bin/bash \
    -e PROBE_B64="${PROBE_B64}" \
    -e PROBE_PATH="${PROBE_PATH}" \
    -e MASTER_PORT="${MASTER_PORT}" \
    -- -lc "${worker_command}" \
    >"${ARTIFACT_ROOT}/legacy_probe.log" 2>&1
launch_status=$?
set -e

count_marker() {
    local marker="$1"
    {
        grep -o "${marker}" "${ARTIFACT_ROOT}/legacy_probe.log" || true
    } | wc -l | tr -d ' '
}

nccl_count="$(count_marker "DEEPEP_LEGACY_NCCL=PASS")"
nvshmem_count="$(count_marker "DEEPEP_LEGACY_NVSHMEM=PASS")"
buffer_count="$(count_marker "DEEPEP_LEGACY_BUFFER=PASS")"
destroy_count="$(count_marker "DEEPEP_LEGACY_DESTROY=PASS")"
node_count="$(count_marker "DEEPEP_LEGACY_NODE=PASS")"
printf 'launch_status=%s\nnccl_count=%s\nnvshmem_count=%s\nbuffer_count=%s\ndestroy_count=%s\nnode_count=%s\n' \
    "${launch_status}" "${nccl_count}" "${nvshmem_count}" "${buffer_count}" \
    "${destroy_count}" "${node_count}" |
    tee "${ARTIFACT_ROOT}/metrics.env"

ps -eo pid=,args= |
    awk -v job="${RJOB_NAME}" 'index($0, job) > 0 { print }' \
        >"${ARTIFACT_ROOT}/cleanup_local_processes.log"

test "${launch_status}" = "0"
test "${nccl_count}" -ge 16
test "${nvshmem_count}" -ge 16
test "${buffer_count}" -ge 16
test "${destroy_count}" -ge 16
test "${node_count}" -ge 2
printf 'DEEPEP_LEGACY_PROBE_HOST=PASS\nARTIFACT_ROOT=%s\n' \
    "${ARTIFACT_ROOT}" | tee "${ARTIFACT_ROOT}/result.env"
