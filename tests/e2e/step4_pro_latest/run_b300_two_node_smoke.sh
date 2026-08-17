#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-shai-core}"
RJOB_NAME="${RJOB_NAME:-s4p-2n-$(date +%m%d-%H%M%S)}"
IMAGE="${IMAGE:-hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled}"
CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
PINNED_COMMIT="${PINNED_COMMIT:-607d1641ee3fec43653fca510d717725828890c2}"
IMAGE_SOURCE_COMMIT="${IMAGE_SOURCE_COMMIT:-c820e5ae1e43246b194080cecc772dcd3fa956cb}"
OPTIMUS_WHEEL_URL="${OPTIMUS_WHEEL_URL:-https://artifactory.stepfun-inc.com/artifactory/api/pypi/stepcast-pypi-release/step-optimus/3.23.24/step_optimus-3.23.24-cp310-cp310-manylinux_2_28_x86_64.whl}"
OPTIMUS_WHEEL_SHA256="${OPTIMUS_WHEEL_SHA256:-2eaec8660cd8505486ec06b09b5b508d73483e0729cbe9a2a60afb5cf9a19cfe}"
MODEL_NAME="${MODEL_NAME:-step4pro-dp16-ep16-dummy}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs/step4pro_smoke_14l_dummy}"
SERVING_PORT="${SERVING_PORT:-8000}"
VLLM_REPO="${VLLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../vllm-step4-pro" && pwd)}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/remote_b300_single_smoke.sh}"
CONTRACT_LIB="${CONTRACT_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/b300_runtime_contract.sh}"
PAYLOAD_SOURCE_ROOT="${PAYLOAD_SOURCE_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814}"
IDENTITY_PAYLOAD_ROOT="${IDENTITY_PAYLOAD_ROOT:-${PAYLOAD_SOURCE_ROOT}/pinned_identity_fulltrees_pack_v2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814/two_node_${RJOB_NAME}}"
CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-2G}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1200}"
DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS="${DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS:-180}"
REMOTE_RESULT_TIMEOUT_SECONDS="${REMOTE_RESULT_TIMEOUT_SECONDS:-2400}"
LIVE_TIMEOUT_SECONDS="${LIVE_TIMEOUT_SECONDS:-3600}"
CLEANUP_TIMEOUT_SECONDS="${CLEANUP_TIMEOUT_SECONDS:-300}"
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-300}"
EVIDENCE_PULL_TIMEOUT_SECONDS="${EVIDENCE_PULL_TIMEOUT_SECONDS:-300}"
DATA_PARALLEL_SIZE=16
DATA_PARALLEL_SIZE_LOCAL=8
VLLM_ALL2ALL_BACKEND=allgather_reducescatter
VLLM_ENABLE_SEQUENCE_PARALLEL=0
NCCL_PREFLIGHT_EVIDENCE="${NCCL_PREFLIGHT_EVIDENCE:-}"
B300_QUOTA_EVIDENCE="${B300_QUOTA_EVIDENCE:-}"
REQUIRED_B300_GPUS=16
REMOTE_EXEC_TIMEOUT_SECONDS="${REMOTE_EXEC_TIMEOUT_SECONDS:-${LIVE_TIMEOUT_SECONDS}}"
MIN_REMOTE_EXEC_TIMEOUT_SECONDS="$((REMOTE_RESULT_TIMEOUT_SECONDS + 2 * EVIDENCE_PULL_TIMEOUT_SECONDS + 60))"

PAYLOAD_ROOT="${ARTIFACT_ROOT}/payload"
REMOTE_BOOTSTRAP_ROOT="/home/step4pro-bootstrap-${RJOB_NAME}"
REMOTE_RUNTIME_REPO="/home/pinned-vllm-runtime-${RJOB_NAME}"
REMOTE_MODEL_CONFIG_DIR="${REMOTE_BOOTSTRAP_ROOT}/model_config"
DISTRIBUTED_ENV_FILE="/home/step4pro-distributed-${RJOB_NAME}.env"
RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"
REPLICAS=()
EXEC_PIDS=()
declare -A EXEC_PID_BY_REPLICA=()
CLEANUP_DONE=0

if (( ${#RJOB_NAME} > 50 )); then
    echo "RJOB_NAME exceeds platform limit: ${#RJOB_NAME} > 50" >&2
    exit 1
fi
if (( REMOTE_EXEC_TIMEOUT_SECONDS < MIN_REMOTE_EXEC_TIMEOUT_SECONDS ||
    LIVE_TIMEOUT_SECONDS < MIN_REMOTE_EXEC_TIMEOUT_SECONDS )); then
    echo "Remote execution timeout is too short for validation and evidence pulls: remote=${REMOTE_EXEC_TIMEOUT_SECONDS} live=${LIVE_TIMEOUT_SECONDS} required=${MIN_REMOTE_EXEC_TIMEOUT_SECONDS}" >&2
    exit 1
fi
mkdir -p "${PAYLOAD_ROOT}"
test -f "${CONTRACT_LIB}"
source "${CONTRACT_LIB}"

: "${NCCL_PREFLIGHT_EVIDENCE:?NCCL_PREFLIGHT_EVIDENCE is required}"
test -s "${NCCL_PREFLIGHT_EVIDENCE}"
preflight_rank_pass_count="$(
    {
        grep -o "NCCL_PREFLIGHT_RANK=PASS" \
            "${NCCL_PREFLIGHT_EVIDENCE}" || true
    } | wc -l | tr -d ' '
)"
preflight_node_pass_count="$(
    {
        grep -o "NCCL_PREFLIGHT_NODE=PASS" \
            "${NCCL_PREFLIGHT_EVIDENCE}" || true
    } | wc -l | tr -d ' '
)"
test "${preflight_rank_pass_count}" -ge 16
test "${preflight_node_pass_count}" -ge 2
grep -q "expected=136.0" "${NCCL_PREFLIGHT_EVIDENCE}"
if grep -Eq "NCCL_PREFLIGHT_HCA=FAIL|Error 803|NCCL error" \
    "${NCCL_PREFLIGHT_EVIDENCE}"; then
    echo "NCCL preflight evidence contains a failure" >&2
    exit 1
fi
sha256sum "${NCCL_PREFLIGHT_EVIDENCE}" \
    >"${ARTIFACT_ROOT}/nccl_preflight_evidence.sha256"

: "${B300_QUOTA_EVIDENCE:?B300_QUOTA_EVIDENCE is required}"
available_b300_gpus="$(
    require_b300_quota_evidence \
        "${B300_QUOTA_EVIDENCE}" \
        "${REQUIRED_B300_GPUS}" \
        "${CHARGED_GROUP}"
)"
cp "${B300_QUOTA_EVIDENCE}" \
    "${ARTIFACT_ROOT}/b300_quota_evidence.txt"
sha256sum "${ARTIFACT_ROOT}/b300_quota_evidence.txt" \
    >"${ARTIFACT_ROOT}/b300_quota_evidence.sha256"
printf 'required_b300_gpus=%s\navailable_b300_gpus=%s\n' \
    "${REQUIRED_B300_GPUS}" "${available_b300_gpus}" \
    >"${ARTIFACT_ROOT}/quota_metrics.env"

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
    local cleanup_status=1
    while (( $(date +%s) < deadline )); do
        local rjob_query_status=0
        local replica_query_status=0
        query_rjob "${ARTIFACT_ROOT}/cleanup_rjob_poll.log"
        rjob_query_status=$?
        query_replicas "${ARTIFACT_ROOT}/cleanup_replicas_poll.log"
        replica_query_status=$?
        if cleanup_inventory_is_empty \
            "${rjob_query_status}" \
            "${replica_query_status}" \
            "${ARTIFACT_ROOT}/cleanup_rjob_poll.log" \
            "${ARTIFACT_ROOT}/cleanup_replicas_poll.log" \
            "${RJOB_NAME}"; then
            cleanup_status=0
            break
        fi
        sleep 5
    done
    cp "${ARTIFACT_ROOT}/cleanup_rjob_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_rjob_final.log"
    cp "${ARTIFACT_ROOT}/cleanup_replicas_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_replicas_final.log"
    local process_id
    for process_id in "${EXEC_PIDS[@]:-}" "${LAUNCH_PID:-}"; do
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
    if (( cleanup_status != 0 && status == 0 )); then
        status=1
    fi
    return "${status}"
}
trap 'status=$?; cleanup; exit "${status}"' EXIT

test "$(git -C "${VLLM_REPO}" rev-parse HEAD)" = "${PINNED_COMMIT}"
git -C "${VLLM_REPO}" cat-file -e "${IMAGE_SOURCE_COMMIT}^{commit}"
test -f "${MODEL_CONFIG_DIR}/config.json"
bash -n "${REMOTE_SCRIPT}"

test -f "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz"
identity_pack="$(find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.pack' -print)"
identity_index="$(find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.idx' -print)"
test -n "${identity_pack}"
test -n "${identity_index}"
cp "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz" "${PAYLOAD_ROOT}/"
cp "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz" "${PAYLOAD_ROOT}/"
cp "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz" "${PAYLOAD_ROOT}/"
cp "${identity_pack}" "${identity_index}" "${PAYLOAD_ROOT}/"
cp "${REMOTE_SCRIPT}" "${PAYLOAD_ROOT}/remote_b300_single_smoke.sh"
cp "${CONTRACT_LIB}" "${PAYLOAD_ROOT}/b300_runtime_contract.sh"
mkdir -p "${PAYLOAD_ROOT}/model_config"
cp "${MODEL_CONFIG_DIR}/config.json" "${PAYLOAD_ROOT}/model_config/config.json"

query_rjob "${ARTIFACT_ROOT}/preflight_rjob.log"
query_replicas "${ARTIFACT_ROOT}/preflight_replicas.log"
if grep -q "${RJOB_NAME}" \
    "${ARTIFACT_ROOT}/preflight_rjob.log" \
    "${ARTIFACT_ROOT}/preflight_replicas.log"; then
    echo "RJob name already exists: ${RJOB_NAME}" >&2
    exit 1
fi

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
NCCL_*)
                printf 'export %s=%q\n' \"\${env_key}\" \"\${env_value}\"
                ;;
        esac
    done < <(env)
} > '${DISTRIBUTED_ENV_FILE}'
chmod 600 '${DISTRIBUTED_ENV_FILE}'
printf 'distributed_env_ready=PASS path=%s\n' '${DISTRIBUTED_ENV_FILE}'
cat '${DISTRIBUTED_ENV_FILE}'
nvidia-smi -L
sleep '${LIVE_TIMEOUT_SECONDS}'"

launch_args=(
    --auto-delete-duration=60m
    --max-wait-duration=15m
    --name "${RJOB_NAME}"
    --replica 2
    --charged-group "${CHARGED_GROUP}"
    --private-machine group
    --positive-tags "${POSITIVE_TAGS}"
    --set-env=DISTRIBUTED_JOB=true
    -e MODELNAME="${MODEL_NAME}"
    -e PORT_AUTO0="${SERVING_PORT}"
    --host-network=true
    --custom-resources=rdma/mlnx_shared=8
    --custom-resources=mellanox.com/mlnx_rdma=1
    --topo-group=yes
    --gpu 8
    --cpu 64
    --memory 600000
    --backoff-limit 1
    --enable-sshd=false
    --enable-jobutil-config=false
    --image "${IMAGE}"
    --entrypoint /bin/bash
    -- -lc "${worker_command}"
)

scoped 300 /kubebrain/brainctl rjob launch --predict-only \
    "${launch_args[@]}" >"${ARTIFACT_ROOT}/predict_only.log" 2>&1
sha256sum "${ARTIFACT_ROOT}/predict_only.log" \
    >"${ARTIFACT_ROOT}/predict_only.sha256"
predict_candidate_count="$(
    {
        grep -c '^Node:' "${ARTIFACT_ROOT}/predict_only.log" || true
    }
)"
if (( predict_candidate_count < 1 )); then
    echo "Same-shape predict-only returned no B300 candidates" >&2
    exit 1
fi
printf 'predict_candidate_count=%s\n' "${predict_candidate_count}" \
    >"${ARTIFACT_ROOT}/predict_only_metrics.env"

set +e
query_rjob "${ARTIFACT_ROOT}/predict_only_rjob.log"
predict_rjob_query_status=$?
query_replicas "${ARTIFACT_ROOT}/predict_only_replicas.log"
predict_replica_query_status=$?
set -e
cleanup_inventory_is_empty \
    "${predict_rjob_query_status}" \
    "${predict_replica_query_status}" \
    "${ARTIFACT_ROOT}/predict_only_rjob.log" \
    "${ARTIFACT_ROOT}/predict_only_replicas.log" \
    "${RJOB_NAME}"

setsid sudo -n systemd-run --scope -p MemoryMax="${CONTROL_MEMORY_MAX}" \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${LIVE_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl rjob launch --detach \
    "${launch_args[@]}" \
    >"${ARTIFACT_ROOT}/launch.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

start_epoch="$(date +%s)"
deadline=$(( start_epoch + READY_TIMEOUT_SECONDS ))
while (( $(date +%s) < deadline )); do
    query_replicas "${ARTIFACT_ROOT}/replicas_poll.log" || true
    mapfile -t REPLICAS < <(
        awk -v prefix="${RJOB_NAME}-" '
            NR > 1 && index($1, prefix) == 1 && $2 == "1/1" && $3 == "Running" {
                print $1
            }
        ' "${ARTIFACT_ROOT}/replicas_poll.log"
    )
    (( ${#REPLICAS[@]} == 2 )) && break
    sleep 5
done
if (( ${#REPLICAS[@]} != 2 )); then
    echo "Expected 2 Running replicas, got ${#REPLICAS[@]}" >&2
    exit 1
fi
printf 'scheduling_seconds=%s\nreplica_0=%s\nreplica_1=%s\n' \
    "$(( $(date +%s) - start_epoch ))" "${REPLICAS[0]}" "${REPLICAS[1]}" \
    >"${ARTIFACT_ROOT}/host_metrics.env"

for replica in "${REPLICAS[@]}"; do
    env_ready=0
    env_ready_deadline="$(date +%s)"
    env_ready_deadline="$((env_ready_deadline + DISTRIBUTED_ENV_READY_TIMEOUT_SECONDS))"
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
        scoped 300 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
            "replica/${replica}" -- tar xf - -C "${REMOTE_BOOTSTRAP_ROOT}" \
            >"${ARTIFACT_ROOT}/transport_${replica}.log" 2>&1

    evidence_root="/home/step4pro-evidence-${RJOB_NAME}-${replica}"
    remote_command="set -euo pipefail
source '${DISTRIBUTED_ENV_FILE}'
: \"\${NODE_RANK:?NODE_RANK missing after distributed env restore}\"
: \"\${NODE_COUNT:?NODE_COUNT missing after distributed env restore}\"
: \"\${MASTER_ADDR:?MASTER_ADDR missing after distributed env restore}\"
: \"\${PROC_PER_NODE:?PROC_PER_NODE missing after distributed env restore}\"
printf 'distributed_env_restored=PASS node_rank=%s node_count=%s master_addr=%s proc_per_node=%s\n' \
    \"\${NODE_RANK}\" \"\${NODE_COUNT}\" \"\${MASTER_ADDR}\" \"\${PROC_PER_NODE}\"
export BOOTSTRAP_ROOT='${REMOTE_BOOTSTRAP_ROOT}'
export EVIDENCE_ROOT='${evidence_root}'
export RUNTIME_REPO='${REMOTE_RUNTIME_REPO}'
export MODEL_PATH='${REMOTE_MODEL_CONFIG_DIR}'
export PATCH_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_to_pinned_vllm.patch.gz'
export BASE_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_base_changed_manifest.tsv.gz'
export PINNED_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/pinned_vllm_manifest.sha256.gz'
export IDENTITY_PACK_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_pack}")'
export IDENTITY_INDEX_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_index}")'
export PINNED_COMMIT='${PINNED_COMMIT}'
export MODELNAME='${MODEL_NAME}'
export PORT_AUTO0='${SERVING_PORT}'
export DATA_PARALLEL_SIZE='${DATA_PARALLEL_SIZE}'
export DATA_PARALLEL_SIZE_LOCAL='${DATA_PARALLEL_SIZE_LOCAL}'
export VLLM_ALL2ALL_BACKEND='${VLLM_ALL2ALL_BACKEND}'
export VLLM_ENABLE_SEQUENCE_PARALLEL='${VLLM_ENABLE_SEQUENCE_PARALLEL}'
export OPTIMUS_WHEEL_URL='${OPTIMUS_WHEEL_URL}'
export OPTIMUS_WHEEL_SHA256='${OPTIMUS_WHEEL_SHA256}'
export RUNTIME_CONTRACT_LIB='${REMOTE_BOOTSTRAP_ROOT}/b300_runtime_contract.sh'
export ENABLE_PROFILER=0
export EVIDENCE_HOLD_SECONDS='${EVIDENCE_HOLD_SECONDS}'
chmod +x '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_single_smoke.sh'
exec '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_single_smoke.sh'"
    setsid sudo -n systemd-run --scope -p MemoryMax="${CONTROL_MEMORY_MAX}" \
        --expand-environment=no \
        timeout --signal=TERM --kill-after=30s "${REMOTE_EXEC_TIMEOUT_SECONDS}s" \
        /kubebrain/brainctl -n "${NAMESPACE}" exec -i "replica/${replica}" -- \
        /bin/bash -lc "${remote_command}" \
        >"${ARTIFACT_ROOT}/remote_exec_${replica}.log" 2>&1 < /dev/null &
    remote_exec_pid=$!
    EXEC_PIDS+=("${remote_exec_pid}")
    EXEC_PID_BY_REPLICA["${replica}"]="${remote_exec_pid}"
done

validation_deadline=$(( $(date +%s) + REMOTE_RESULT_TIMEOUT_SECONDS ))
for replica in "${REPLICAS[@]}"; do
    evidence_root="/home/step4pro-evidence-${RJOB_NAME}-${replica}"
    remote_exec_pid="${EXEC_PID_BY_REPLICA[$replica]}"
    validation_ready=0
    while (( $(date +%s) < validation_deadline )); do
        if scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
            "replica/${replica}" -- test -f "${evidence_root}/remote_validation_ready" \
            >/dev/null 2>"${ARTIFACT_ROOT}/validation_probe_${replica}.log" < /dev/null; then
            validation_ready=1
            break
        fi
        if ! kill -0 "${remote_exec_pid}" 2>/dev/null; then
            remote_exec_status=0
            wait "${remote_exec_pid}" || remote_exec_status=$?
            echo "Remote execution exited before validation: replica=${replica} status=${remote_exec_status}" >&2
            tail -n 200 "${ARTIFACT_ROOT}/remote_exec_${replica}.log" >&2
            exit 1
        fi
        sleep 5
    done
    test "${validation_ready}" = "1"
done

agrs_manager_replica_count=0
for replica in "${REPLICAS[@]}"; do
    evidence_root="/home/step4pro-evidence-${RJOB_NAME}-${replica}"
    remote_exec_pid="${EXEC_PID_BY_REPLICA[$replica]}"
    if ! kill -0 "${remote_exec_pid}" 2>/dev/null; then
        echo "Remote execution exited before evidence pull: replica=${replica}" >&2
        tail -n 200 "${ARTIFACT_ROOT}/remote_exec_${replica}.log" >&2
        exit 1
    fi

    evidence_tar="${ARTIFACT_ROOT}/evidence_${replica}.tar"
    scoped "${EVIDENCE_PULL_TIMEOUT_SECONDS}" \
        /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${replica}" -- \
        tar cf - -C /home "$(basename "${evidence_root}")" \
        >"${evidence_tar}" 2>"${ARTIFACT_ROOT}/evidence_pull_${replica}.log" \
        < /dev/null
    evidence_dir="${ARTIFACT_ROOT}/evidence_${replica}"
    mkdir -p "${evidence_dir}"
    tar xf "${evidence_tar}" -C "${evidence_dir}"
    pulled_evidence_root="${evidence_dir}/$(basename "${evidence_root}")"
    remote_execution_file="${pulled_evidence_root}/remote_execution.env"
    server_log="${pulled_evidence_root}/vllm_server.log"
    grep -q '^DISTRIBUTED_RUNTIME_VALIDATION=PASS$' \
        "${remote_execution_file}"
    grep -Fq \
        "DeepEP runtime not available; using allgather_reducescatter all2all backend without sequence parallelism" \
        "${server_log}"
    grep -Eq "MODEL_FORWARD_COMPLETE.*batch=real" "${server_log}"
    assert_runtime_log_clean "${server_log}"
    if grep -q "Using AgRsAll2AllManager all2all manager" "${server_log}"; then
        agrs_manager_replica_count="$((agrs_manager_replica_count + 1))"
    fi
    if grep -Eq "Using DeepEP[A-Za-z0-9_]*All2AllManager" "${server_log}"; then
        echo "Unexpected DeepEP all2all manager selected" >&2
        exit 1
    fi
    if grep -q 'Auto-configured .*VLLM_ALL2ALL_BACKEND=' "${server_log}"; then
        echo "Unexpected Step MoE automatic backend selection" >&2
        exit 1
    fi
    model_forward_complete_count="$(
        grep -Ec "MODEL_FORWARD_COMPLETE.*batch=real" "${server_log}"
    )"
    printf 'replica=%s model_forward_complete_count=%s\n' \
        "${replica}" "${model_forward_complete_count}" \
        >>"${ARTIFACT_ROOT}/host_metrics.env"
done
test "${agrs_manager_replica_count}" -ge 1
printf 'agrs_manager_replica_count=%s\n' "${agrs_manager_replica_count}" \
    >>"${ARTIFACT_ROOT}/host_metrics.env"

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
printf 'TWO_NODE_HOST_WRAPPER=PASS\nRJOB_NAME=%s\nARTIFACT_ROOT=%s\n' \
    "${RJOB_NAME}" "${ARTIFACT_ROOT}" |
    tee "${ARTIFACT_ROOT}/host_result.env"
