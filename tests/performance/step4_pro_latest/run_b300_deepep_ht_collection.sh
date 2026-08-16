#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-shai-core}"
MODE="${MODE:-smoke}"
EP_SIZE="${EP_SIZE:-16}"
RJOB_NAME="${RJOB_NAME:-s4p-aic-deepep-${EP_SIZE}-$(date +%m%d-%H%M%S)}"
IMAGE="${IMAGE:-hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled}"
CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
PINNED_COMMIT="${PINNED_COMMIT:-607d1641ee3fec43653fca510d717725828890c2}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
AIC_HOST_PYTHON="${AIC_HOST_PYTHON:-/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python}"
PAYLOAD_SOURCE_ROOT="${PAYLOAD_SOURCE_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814}"
IDENTITY_PAYLOAD_ROOT="${IDENTITY_PAYLOAD_ROOT:-${PAYLOAD_SOURCE_ROOT}/pinned_identity_fulltrees_pack_v2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/step4_aic_deepep_ht_b300_20260815/${MODE}_ep${EP_SIZE}_${RJOB_NAME}}"
CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"
DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS="${DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS:-180}"
RESULT_TIMEOUT_SECONDS="${RESULT_TIMEOUT_SECONDS:-3600}"
CLEANUP_TIMEOUT_SECONDS="${CLEANUP_TIMEOUT_SECONDS:-300}"
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-300}"
MASTER_PORT="${MASTER_PORT:-5678}"
NVSHMEM_ENABLE_NIC_PE_MAPPING="${NVSHMEM_ENABLE_NIC_PE_MAPPING:-1}"

if [[ "${MODE}" != "preflight" && "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "MODE must be preflight, smoke, or full: ${MODE}" >&2
    exit 1
fi
if (( EP_SIZE == 16 )); then
    NNODES=2
elif (( EP_SIZE == 32 )); then
    NNODES=4
else
    echo "EP_SIZE must be 16 or 32: ${EP_SIZE}" >&2
    exit 1
fi
if [[ "${MODE}" == "full" && "${RESULT_TIMEOUT_SECONDS}" == "3600" ]]; then
    RESULT_TIMEOUT_SECONDS=28800
fi
WORKER_HOLD_SECONDS="$((RESULT_TIMEOUT_SECONDS + EVIDENCE_HOLD_SECONDS + 900))"
LIVE_TIMEOUT_SECONDS="$((WORKER_HOLD_SECONDS + READY_TIMEOUT_SECONDS + 300))"
REMOTE_EXEC_TIMEOUT_SECONDS="$((RESULT_TIMEOUT_SECONDS + EVIDENCE_HOLD_SECONDS + 300))"

PAYLOAD_ROOT="${ARTIFACT_ROOT}/payload"
REMOTE_BOOTSTRAP_ROOT="/home/s4p-aic-deepep-${RJOB_NAME}"
REMOTE_RUNTIME_REPO="/home/s4p-pinned-vllm-${RJOB_NAME}"
DISTRIBUTED_ENV_FILE="/home/s4p-aic-deepep-${RJOB_NAME}.env"
REMOTE_AIC_PAYLOAD="${REMOTE_BOOTSTRAP_ROOT}/aic_payload.tar"
REMOTE_AIC_METADATA="${REMOTE_BOOTSTRAP_ROOT}/aic_metadata.tar"
RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"
REPLICAS=()
EXEC_PIDS=()
LAUNCH_PID=""
CLEANUP_DONE=0

if (( ${#RJOB_NAME} > 50 )); then
    echo "RJOB_NAME exceeds platform limit: ${#RJOB_NAME} > 50" >&2
    exit 1
fi
mkdir -p "${PAYLOAD_ROOT}"

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
        return "${status}"
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
    local process_id
    for process_id in "${EXEC_PIDS[@]:-}" "${LAUNCH_PID}"; do
        if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
            kill -TERM -- "-${process_id}" 2>/dev/null
            sleep 2
            kill -KILL -- "-${process_id}" 2>/dev/null
        fi
        [[ -n "${process_id}" ]] && wait "${process_id}" 2>/dev/null
    done
    ps -eo pid=,args= |
        awk -v job="${RJOB_NAME}" 'index($0, job) > 0 { print }' \
            >"${ARTIFACT_ROOT}/cleanup_local_processes.log"
    set -e
    return "${status}"
}
trap 'status=$?; cleanup; exit "${status}"' EXIT

test "$(git -C "${REPO_ROOT}" branch --show-current)" = \
    "task/step4-pro-latest-b300"
test "$(git -C "${REPO_ROOT}/vllm-step4-pro" rev-parse HEAD)" = \
    "${PINNED_COMMIT}"
bash -n "${REPO_ROOT}/tests/performance/step4_pro_latest/remote_b300_deepep_ht_collection.sh"

test -f "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz"
identity_pack="$(
    find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.pack' -print
)"
identity_index="$(
    find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.idx' -print
)"
test -n "${identity_pack}"
test -n "${identity_index}"

cp "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz" "${PAYLOAD_ROOT}/"
cp "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz" "${PAYLOAD_ROOT}/"
cp "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz" "${PAYLOAD_ROOT}/"
cp "${identity_pack}" "${identity_index}" "${PAYLOAD_ROOT}/"
cp "${REPO_ROOT}/tests/e2e/step4_pro_latest/remote_b300_source_probe.sh" \
    "${PAYLOAD_ROOT}/remote_b300_source_probe.sh"
cp "${REPO_ROOT}/tests/performance/step4_pro_latest/remote_b300_deepep_ht_collection.sh" \
    "${PAYLOAD_ROOT}/remote_b300_deepep_ht_collection.sh"
cp "${REPO_ROOT}/tests/performance/step4_pro_latest/run_step4_deepep_ht_distributed.py" \
    "${PAYLOAD_ROOT}/run_step4_deepep_ht_distributed.py"
cp "${REPO_ROOT}/tests/performance/step4_pro_latest/run_step4_deepep_ht_nccl_preflight.py" \
    "${PAYLOAD_ROOT}/run_step4_deepep_ht_nccl_preflight.py"

git -C "${REPO_ROOT}" ls-files collector src/aiconfigurator pyproject.toml \
    | grep -v '^src/aiconfigurator/systems/data/' \
    >"${PAYLOAD_ROOT}/aic_payload_files.txt"
find "${REPO_ROOT}/collector/wideep/vllm" -type f -name '*.py' \
    -printf '%P\n' \
    | sed 's#^#collector/wideep/vllm/#' \
    >>"${PAYLOAD_ROOT}/aic_payload_files.txt"
sort -u "${PAYLOAD_ROOT}/aic_payload_files.txt" \
    -o "${PAYLOAD_ROOT}/aic_payload_files.txt"
tar cf "${PAYLOAD_ROOT}/aic_payload.tar" \
    -C "${REPO_ROOT}" -T "${PAYLOAD_ROOT}/aic_payload_files.txt"

aic_dist_info="$(
    "${AIC_HOST_PYTHON}" - <<'PY'
import importlib.metadata

distribution = importlib.metadata.distribution("aiconfigurator")
if distribution.version != "0.10.0":
    raise SystemExit(
        f"unexpected aiconfigurator version: {distribution.version}"
    )
print(distribution._path)
PY
)"
tar cf "${PAYLOAD_ROOT}/aic_metadata.tar" \
    -C "$(dirname "${aic_dist_info}")" "$(basename "${aic_dist_info}")"

sha256sum \
    "${PAYLOAD_ROOT}/aic_payload.tar" \
    "${PAYLOAD_ROOT}/aic_metadata.tar" \
    "${PAYLOAD_ROOT}/remote_b300_source_probe.sh" \
    "${PAYLOAD_ROOT}/remote_b300_deepep_ht_collection.sh" \
    "${PAYLOAD_ROOT}/run_step4_deepep_ht_distributed.py" \
    "${PAYLOAD_ROOT}/run_step4_deepep_ht_nccl_preflight.py" \
    >"${ARTIFACT_ROOT}/payload.sha256"

query_rjob "${ARTIFACT_ROOT}/preflight_rjob.log"
query_replicas "${ARTIFACT_ROOT}/preflight_replicas.log"
if grep -q "${RJOB_NAME}" \
    "${ARTIFACT_ROOT}/preflight_rjob.log" \
    "${ARTIFACT_ROOT}/preflight_replicas.log"; then
    echo "RJob name already exists: ${RJOB_NAME}" >&2
    exit 1
fi

scoped 300 /kubebrain/rlaunch --predict-only \
    --charged-group="${CHARGED_GROUP}" \
    --private-machine=group \
    --positive-tags="${POSITIVE_TAGS}" \
    -P "${NNODES}" \
    --gpu=8 \
    --cpu=64 \
    --memory=600000 \
    --custom-resources=rdma/mlnx_shared=8 \
    --custom-resources=mellanox.com/mlnx_rdma=1 \
    --host-network \
    --topo-group=yes \
    --set-env=DISTRIBUTED_JOB=true \
    --replica-prefix \
    --backoff-limit=1 \
    --predict-node-num="${NNODES}" \
    -- bash -lc 'true' \
    >"${ARTIFACT_ROOT}/predict_only.log" 2>&1

cat >"${ARTIFACT_ROOT}/launch.command.txt" <<EOF
/kubebrain/brainctl rjob launch --detach --name ${RJOB_NAME} --replica ${NNODES} --charged-group ${CHARGED_GROUP} --private-machine group --positive-tags ${POSITIVE_TAGS} --set-env=DISTRIBUTED_JOB=true --custom-resources=rdma/mlnx_shared=8 --custom-resources=mellanox.com/mlnx_rdma=1 --host-network --topo-group=yes --gpu 8 --cpu 64 --memory 600000 --backoff-limit 1 --image ${IMAGE} --entrypoint /bin/bash -- -lc '<bounded Step4 DeepEP HT collector holder>'
EOF

worker_command="set -euo pipefail
{
    for env_key in NODE_RANK NODE_COUNT MASTER_ADDR PROC_PER_NODE; do
        env_value=\${!env_key:?missing required platform variable \${env_key}}
        printf 'export %s=%q\n' \"\${env_key}\" \"\${env_value}\"
    done
    while IFS='=' read -r env_key env_value; do
        case \"\${env_key}\" in
            JOB_ID|SOCKET_IP|NODE_NAME|GPU_TYPE|GPU_VENDOR|\
RDMA_NETWORK_LINK_TYPE|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|\
NCCL_*|NVSHMEM_*)
                printf 'export %s=%q\n' \"\${env_key}\" \"\${env_value}\"
                ;;
        esac
    done < <(env)
} > '${DISTRIBUTED_ENV_FILE}'
chmod 600 '${DISTRIBUTED_ENV_FILE}'
printf 'distributed_env_ready=PASS path=%s\n' '${DISTRIBUTED_ENV_FILE}'
nvidia-smi -L
sleep '${WORKER_HOLD_SECONDS}'"

launch_started_epoch="$(date +%s)"
setsid sudo -n systemd-run --scope -p MemoryMax="${CONTROL_MEMORY_MAX}" \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${LIVE_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl rjob launch \
    --detach \
    --auto-delete-duration=10h \
    --max-wait-duration=20m \
    --name "${RJOB_NAME}" \
    --replica "${NNODES}" \
    --charged-group "${CHARGED_GROUP}" \
    --private-machine group \
    --positive-tags "${POSITIVE_TAGS}" \
    --set-env=DISTRIBUTED_JOB=true \
    -e NVSHMEM_ENABLE_NIC_PE_MAPPING="${NVSHMEM_ENABLE_NIC_PE_MAPPING}" \
    --custom-resources=rdma/mlnx_shared=8 \
    --custom-resources=mellanox.com/mlnx_rdma=1 \
    --host-network \
    --topo-group=yes \
    --gpu 8 \
    --cpu 64 \
    --memory 600000 \
    --backoff-limit 1 \
    --enable-sshd=false \
    --enable-jobutil-config=false \
    --entrypoint /bin/bash \
    --image "${IMAGE}" \
    -- -lc "${worker_command}" \
    >"${ARTIFACT_ROOT}/launch.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

ready_deadline=$((launch_started_epoch + READY_TIMEOUT_SECONDS))
while (( $(date +%s) < ready_deadline )); do
    query_replicas "${ARTIFACT_ROOT}/replicas_poll.log" || true
    mapfile -t REPLICAS < <(
        awk -v prefix="${RJOB_NAME}-" '
            NR > 1 && index($1, prefix) == 1 && $2 == "1/1" && $3 == "Running" {
                print $1
            }
        ' "${ARTIFACT_ROOT}/replicas_poll.log"
    )
    (( ${#REPLICAS[@]} == NNODES )) && break
    sleep 5
done
if (( ${#REPLICAS[@]} != NNODES )); then
    echo "Expected ${NNODES} Running replicas, got ${#REPLICAS[@]}" >&2
    exit 1
fi
printf 'scheduling_seconds=%s\nreplica_count=%s\n' \
    "$(( $(date +%s) - launch_started_epoch ))" "${#REPLICAS[@]}" \
    >"${ARTIFACT_ROOT}/host_metrics.env"

for replica in "${REPLICAS[@]}"; do
    env_ready=0
    env_ready_deadline="$(( $(date +%s) + DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS ))"
    while (( $(date +%s) < env_ready_deadline )); do
        if scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
            "replica/${replica}" -- test -s "${DISTRIBUTED_ENV_FILE}" \
            >"${ARTIFACT_ROOT}/distributed_env_probe_${replica}.log" 2>&1 \
            < /dev/null; then
            env_ready=1
            break
        fi
        sleep 2
    done
    test "${env_ready}" = "1"
    scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${replica}" -- mkdir -p "${REMOTE_BOOTSTRAP_ROOT}" \
        >"${ARTIFACT_ROOT}/bootstrap_${replica}.log" 2>&1 < /dev/null
    tar cf - -C "${PAYLOAD_ROOT}" . |
        scoped 600 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
            "replica/${replica}" -- tar xf - -C "${REMOTE_BOOTSTRAP_ROOT}" \
            >"${ARTIFACT_ROOT}/transport_${replica}.log" 2>&1

    evidence_root="/home/s4p-aic-deepep-evidence-${RJOB_NAME}-${replica}"
    remote_command="set -euo pipefail
source '${DISTRIBUTED_ENV_FILE}'
: \"\${NODE_RANK:?NODE_RANK missing after distributed env restore}\"
: \"\${NODE_COUNT:?NODE_COUNT missing after distributed env restore}\"
: \"\${MASTER_ADDR:?MASTER_ADDR missing after distributed env restore}\"
: \"\${PROC_PER_NODE:?PROC_PER_NODE missing after distributed env restore}\"
export BOOTSTRAP_ROOT='${REMOTE_BOOTSTRAP_ROOT}'
export EVIDENCE_ROOT='${evidence_root}'
export RUNTIME_REPO='${REMOTE_RUNTIME_REPO}'
export AIC_PAYLOAD_PATH='${REMOTE_AIC_PAYLOAD}'
export AIC_METADATA_PATH='${REMOTE_AIC_METADATA}'
export PATCH_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_to_pinned_vllm.patch.gz'
export BASE_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_base_changed_manifest.tsv.gz'
export PINNED_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/pinned_vllm_manifest.sha256.gz'
export IDENTITY_PACK_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_pack}")'
export IDENTITY_INDEX_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_index}")'
export PINNED_COMMIT='${PINNED_COMMIT}'
export EP_SIZE='${EP_SIZE}'
export MODE='${MODE}'
export MASTER_PORT='${MASTER_PORT}'
export EVIDENCE_HOLD_SECONDS='${EVIDENCE_HOLD_SECONDS}'
chmod +x '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_source_probe.sh'
chmod +x '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_deepep_ht_collection.sh'
exec '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_deepep_ht_collection.sh'"
    setsid sudo -n systemd-run --scope -p MemoryMax="${CONTROL_MEMORY_MAX}" \
        --expand-environment=no \
        timeout --signal=TERM --kill-after=30s "${REMOTE_EXEC_TIMEOUT_SECONDS}s" \
        /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${replica}" -- /bin/bash -lc "${remote_command}" \
        >"${ARTIFACT_ROOT}/remote_exec_${replica}.log" 2>&1 < /dev/null &
    EXEC_PIDS+=("$!")
done

for replica in "${REPLICAS[@]}"; do
    evidence_root="/home/s4p-aic-deepep-evidence-${RJOB_NAME}-${replica}"
    result_deadline=$(( $(date +%s) + RESULT_TIMEOUT_SECONDS ))
    result_ready=0
    while (( $(date +%s) < result_deadline )); do
        if scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
            "replica/${replica}" -- \
            test -f "${evidence_root}/remote_result_ready" \
            >"${ARTIFACT_ROOT}/result_probe_${replica}.log" 2>&1 \
            < /dev/null; then
            result_ready=1
            break
        fi
        sleep 10
    done
    test "${result_ready}" = "1"

    evidence_tar="${ARTIFACT_ROOT}/evidence_${replica}.tar"
    scoped 600 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${replica}" -- \
        tar cf - -C /home "$(basename "${evidence_root}")" \
        >"${evidence_tar}" 2>"${ARTIFACT_ROOT}/evidence_pull_${replica}.log" \
        < /dev/null
    evidence_dir="${ARTIFACT_ROOT}/evidence_${replica}"
    mkdir -p "${evidence_dir}"
    tar xf "${evidence_tar}" -C "${evidence_dir}"
    result_file="${evidence_dir}/$(basename "${evidence_root}")/result.env"
    if [[ "${MODE}" == "preflight" ]]; then
        grep -q '^B300_DEEPEP_HT_NCCL_PREFLIGHT=PASS$' "${result_file}"
    else
        grep -q '^B300_DEEPEP_HT_COLLECTION=PASS$' "${result_file}"
    fi
done

dataset_dir=""
if [[ "${MODE}" == "preflight" ]]; then
    mapfile -t preflight_metrics < <(
        find "${ARTIFACT_ROOT}" -name 'nccl_preflight.env' -type f -print
    )
    test "${#preflight_metrics[@]}" = "${NNODES}"
    for metrics_file in "${preflight_metrics[@]}"; do
        grep -q '^rank_pass_count=8$' "${metrics_file}"
        grep -q "^world_size=${EP_SIZE}$" "${metrics_file}"
        grep -q "^participant_sum=${EP_SIZE}$" "${metrics_file}"
    done
else
    mapfile -t summaries < <(
        find "${ARTIFACT_ROOT}" -path '*/dataset/step4_deepep_ht_summary.json' \
            -type f -print
    )
    test "${#summaries[@]}" = "1"
    dataset_dir="$(dirname "${summaries[0]}")"
    expected_rows=58
    if [[ "${MODE}" == "smoke" ]]; then
        expected_rows=6
    fi
    "${AIC_HOST_PYTHON}" - "${dataset_dir}" "${EP_SIZE}" "${MODE}" "${expected_rows}" <<'PY'
from pathlib import Path
import csv
import json
import sys

dataset_dir = Path(sys.argv[1])
expected_ep = int(sys.argv[2])
expected_mode = sys.argv[3]
expected_rows = int(sys.argv[4])
summary = json.loads(
    (dataset_dir / "step4_deepep_ht_summary.json").read_text(encoding="utf-8")
)
with (dataset_dir / "step4_deepep_ht_perf.txt").open(
    newline="", encoding="utf-8"
) as stream:
    rows = list(csv.DictReader(stream))
if summary["status"] != "PASS":
    raise SystemExit(f"unexpected summary status: {summary}")
if summary["ep_size"] != expected_ep or summary["mode"] != expected_mode:
    raise SystemExit(f"unexpected summary identity: {summary}")
if summary["row_count"] != expected_rows or len(rows) != expected_rows:
    raise SystemExit(
        f"unexpected row count: summary={summary['row_count']} csv={len(rows)} "
        f"expected={expected_rows}"
    )
if {row["operation"] for row in rows} != {"dispatch", "combine"}:
    raise SystemExit("dataset does not contain both DeepEP HT operations")
if {int(row["ep_size"]) for row in rows} != {expected_ep}:
    raise SystemExit("dataset contains the wrong EP topology")
if any(float(row["latency"]) <= 0.0 for row in rows):
    raise SystemExit("dataset contains non-positive latency")
print(
    "HOST_DATASET_VALIDATION=PASS",
    f"ep_size={expected_ep}",
    f"mode={expected_mode}",
    f"rows={len(rows)}",
    f"latency_min_ms={min(float(row['latency']) for row in rows)}",
    f"latency_max_ms={max(float(row['latency']) for row in rows)}",
)
PY
fi

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

printf 'B300_DEEPEP_HT_HOST=PASS\nmode=%s\nep_size=%s\nrjob=%s\nartifact_root=%s\ndataset_dir=%s\n' \
    "${MODE}" "${EP_SIZE}" "${RJOB_NAME}" "${ARTIFACT_ROOT}" "${dataset_dir}" \
    | tee "${ARTIFACT_ROOT}/host_result.env"
