#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-shai-core}"
RJOB_NAME="${RJOB_NAME:-s4p-nccl-$(date +%m%d-%H%M%S)}"
IMAGE="${IMAGE:-hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled}"
CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814/nccl_preflight_${RJOB_NAME}}"
CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-2G}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1200}"
RESULT_TIMEOUT_SECONDS="${RESULT_TIMEOUT_SECONDS:-600}"
LIVE_TIMEOUT_SECONDS="${LIVE_TIMEOUT_SECONDS:-1200}"
CLEANUP_TIMEOUT_SECONDS="${CLEANUP_TIMEOUT_SECONDS:-300}"
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-300}"
NCCL_PREFLIGHT_MASTER_PORT="${NCCL_PREFLIGHT_MASTER_PORT:-29671}"
PREFLIGHT_PATH="/home/nccl-preflight-${RJOB_NAME}.py"
RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"
LAUNCH_PID=""
CLEANUP_DONE=0

if (( ${#RJOB_NAME} > 50 )); then
    echo "RJOB_NAME exceeds platform limit: ${#RJOB_NAME} > 50" >&2
    exit 1
fi
mkdir -p "${ARTIFACT_ROOT}"

scoped() {
    local timeout_seconds="$1"
    shift
    sudo -n systemd-run --scope -p MemoryMax="${CONTROL_MEMORY_MAX}" \
        --expand-environment=no \
        timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" "$@"
}

query_rjob() {
    local output="$1"
    scoped 60 /kubebrain/brainctl get rjob "${RJOB_NAME}" \
        -n "${NAMESPACE}" --ignore-not-found >"${output}" 2>&1
}

query_replicas() {
    local output="$1"
    scoped 60 /kubebrain/brainctl get replica -n "${NAMESPACE}" \
        -l "${RJOB_LABEL}" >"${output}" 2>&1
}

cleanup() {
    local status=$?
    if (( CLEANUP_DONE == 1 )); then
        return
    fi
    CLEANUP_DONE=1
    set +e
    scoped 60 /kubebrain/brainctl delete rjob "${RJOB_NAME}" \
        -n "${NAMESPACE}" >"${ARTIFACT_ROOT}/cleanup_delete.log" 2>&1
    local deadline=$(( $(date +%s) + CLEANUP_TIMEOUT_SECONDS ))
    while (( $(date +%s) < deadline )); do
        query_rjob "${ARTIFACT_ROOT}/cleanup_rjob_poll.log"
        query_replicas "${ARTIFACT_ROOT}/cleanup_replicas_poll.log"
        if ! grep -q "${RJOB_NAME}" \
            "${ARTIFACT_ROOT}/cleanup_rjob_poll.log" \
            "${ARTIFACT_ROOT}/cleanup_replicas_poll.log"; then
            break
        fi
        sleep 5
    done
    cp "${ARTIFACT_ROOT}/cleanup_rjob_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_rjob_final.log"
    cp "${ARTIFACT_ROOT}/cleanup_replicas_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_replicas_final.log"
    if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        kill -TERM -- "-${LAUNCH_PID}" 2>/dev/null
        sleep 2
        kill -KILL -- "-${LAUNCH_PID}" 2>/dev/null
    fi
    [[ -n "${LAUNCH_PID}" ]] && wait "${LAUNCH_PID}" 2>/dev/null
    ps -eo pid=,args= |
        awk -v job="${RJOB_NAME}" 'index($0, job) > 0 { print }' \
            >"${ARTIFACT_ROOT}/cleanup_local_processes.log"
    set -e
    return "${status}"
}
trap 'status=$?; cleanup; exit "${status}"' EXIT

PREFLIGHT_SOURCE="$(
    cat <<'PY'
from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
hca = os.environ.get("NCCL_IB_HCA", "")

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
dist.all_reduce(value)
torch.cuda.synchronize()

actual = float(value.item())
expected = world_size * (world_size + 1) / 2
if actual != expected:
    raise RuntimeError(
        f"unexpected all-reduce result: actual={actual} expected={expected}"
    )

print(
    "NCCL_PREFLIGHT_RANK=PASS",
    f"host={socket.gethostname()}",
    f"rank={rank}",
    f"local_rank={local_rank}",
    f"world_size={world_size}",
    f"actual={actual}",
    f"expected={expected}",
    f"hca={hca}",
    flush=True,
)
dist.barrier()
dist.destroy_process_group()
PY
)"
PREFLIGHT_B64="$(printf '%s' "${PREFLIGHT_SOURCE}" | base64 -w0)"

worker_command="$(
    cat <<'WORKER'
set -euo pipefail

# Match the pinned B300 recipe: exclude the image's stale CUDA 12.8 compat
# library so torch cu129 resolves the CUDA 13 compatibility library.
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64

: "${NODE_RANK:?NODE_RANK is required}"
: "${NODE_COUNT:?NODE_COUNT is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${PROC_PER_NODE:?PROC_PER_NODE is required}"
: "${PREFLIGHT_B64:?PREFLIGHT_B64 is required}"
: "${PREFLIGHT_PATH:?PREFLIGHT_PATH is required}"
: "${NCCL_PREFLIGHT_MASTER_PORT:?NCCL_PREFLIGHT_MASTER_PORT is required}"

NCCL_IB_HCA="${NCCL_IB_HCA:-}"
printf 'NCCL_PREFLIGHT_ENV node_rank=%s node_count=%s proc_per_node=%s master_addr=%s hca=%q gid=%q socket_if=%q\n' \
    "${NODE_RANK}" "${NODE_COUNT}" "${PROC_PER_NODE}" "${MASTER_ADDR}" \
    "${NCCL_IB_HCA}" "${NCCL_IB_GID_INDEX:-}" "${NCCL_SOCKET_IFNAME:-}"
if [[ -z "${NCCL_IB_HCA}" ]]; then
    echo "NCCL_PREFLIGHT_HCA=FAIL reason=empty_platform_hca" >&2
    exit 42
fi
[[ -n "${NCCL_IB_HCA}" ]]
test "${NODE_COUNT}" = "2"
test "${PROC_PER_NODE}" = "8"

printf '%s' "${PREFLIGHT_B64}" | base64 -d >"${PREFLIGHT_PATH}"
torchrun --nnodes "${NODE_COUNT}" \
    --nproc_per_node "${PROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${NCCL_PREFLIGHT_MASTER_PORT}" \
    "${PREFLIGHT_PATH}"

printf 'NCCL_PREFLIGHT_NODE=PASS node_rank=%s hca=%q\n' \
    "${NODE_RANK}" "${NCCL_IB_HCA}"
sleep "${EVIDENCE_HOLD_SECONDS}"
WORKER
)"

query_rjob "${ARTIFACT_ROOT}/preflight_rjob.log"
query_replicas "${ARTIFACT_ROOT}/preflight_replicas.log"
if grep -q "${RJOB_NAME}" \
    "${ARTIFACT_ROOT}/preflight_rjob.log" \
    "${ARTIFACT_ROOT}/preflight_replicas.log"; then
    echo "RJob name already exists: ${RJOB_NAME}" >&2
    exit 1
fi

setsid sudo -n systemd-run --scope -p MemoryMax="${CONTROL_MEMORY_MAX}" \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${LIVE_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl rjob launch --detach \
    --auto-delete-duration=60m --max-wait-duration=15m \
    --name "${RJOB_NAME}" --replica 2 \
    --charged-group "${CHARGED_GROUP}" --private-machine group \
    --positive-tags "${POSITIVE_TAGS}" \
    --set-env=DISTRIBUTED_JOB=true \
    --host-network=true \
    --custom-resources=rdma/mlnx_shared=8 \
    --custom-resources=mellanox.com/mlnx_rdma=1 \
    --topo-group=yes \
    --gpu 8 --cpu 64 --memory 600000 --backoff-limit 1 \
    --enable-sshd=false --enable-jobutil-config=false \
    --image "${IMAGE}" --entrypoint /bin/bash \
    -e PREFLIGHT_B64="${PREFLIGHT_B64}" \
    -e PREFLIGHT_PATH="${PREFLIGHT_PATH}" \
    -e NCCL_PREFLIGHT_MASTER_PORT="${NCCL_PREFLIGHT_MASTER_PORT}" \
    -e EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS}" \
    -- -lc "${worker_command}" \
    >"${ARTIFACT_ROOT}/launch.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

start_epoch="$(date +%s)"
ready_deadline=$(( start_epoch + READY_TIMEOUT_SECONDS ))
while (( $(date +%s) < ready_deadline )); do
    query_replicas "${ARTIFACT_ROOT}/replicas_poll.log" || true
    running_replicas="$(
        awk -v prefix="${RJOB_NAME}-" '
            NR > 1 && index($1, prefix) == 1 && $2 == "1/1" && $3 == "Running" {
                count += 1
            }
            END { print count + 0 }
        ' "${ARTIFACT_ROOT}/replicas_poll.log"
    )"
    [[ "${running_replicas}" = "2" ]] && break
    sleep 5
done
test "${running_replicas:-0}" = "2"
printf 'scheduling_seconds=%s\n' "$(( $(date +%s) - start_epoch ))" \
    >"${ARTIFACT_ROOT}/host_metrics.env"

result_deadline=$(( $(date +%s) + RESULT_TIMEOUT_SECONDS ))
rank_pass_count=0
node_pass_count=0
while (( $(date +%s) < result_deadline )); do
    rank_pass_count="$(
        {
            grep -o "NCCL_PREFLIGHT_RANK=PASS" \
                "${ARTIFACT_ROOT}/launch.log" || true
        } | wc -l | tr -d ' '
    )"
    node_pass_count="$(
        {
            grep -o "NCCL_PREFLIGHT_NODE=PASS" \
                "${ARTIFACT_ROOT}/launch.log" || true
        } | wc -l | tr -d ' '
    )"
    if (( rank_pass_count >= 16 && node_pass_count >= 2 )); then
        break
    fi
    if grep -Eq \
        "NCCL_PREFLIGHT_HCA=FAIL|NCCL error|RuntimeError|Traceback" \
        "${ARTIFACT_ROOT}/launch.log"; then
        echo "NCCL preflight failed; see launch.log" >&2
        exit 1
    fi
    sleep 2
done

test "${rank_pass_count}" -ge 16
test "${node_pass_count}" -ge 2
grep -q "expected=136.0" "${ARTIFACT_ROOT}/launch.log"
grep "NCCL_PREFLIGHT_" "${ARTIFACT_ROOT}/launch.log" \
    >"${ARTIFACT_ROOT}/nccl_preflight_markers.log"

cleanup
cleanup_exit=$?
trap - EXIT
test "${cleanup_exit}" = "0"
if grep -q "${RJOB_NAME}" \
    "${ARTIFACT_ROOT}/cleanup_rjob_final.log" \
    "${ARTIFACT_ROOT}/cleanup_replicas_final.log"; then
    echo "Resource cleanup verification failed" >&2
    exit 1
fi
printf 'NCCL_PREFLIGHT_HOST=PASS\nRJOB_NAME=%s\nARTIFACT_ROOT=%s\n' \
    "${RJOB_NAME}" "${ARTIFACT_ROOT}" |
    tee "${ARTIFACT_ROOT}/host_result.env"
