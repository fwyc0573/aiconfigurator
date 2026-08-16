#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-shai-core}"
RJOB_NAME="${RJOB_NAME:-step4pro-b300-single-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b}"
CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
PINNED_COMMIT="${PINNED_COMMIT:-607d1641ee3fec43653fca510d717725828890c2}"
IMAGE_SOURCE_COMMIT="${IMAGE_SOURCE_COMMIT:-c820e5ae1e43246b194080cecc772dcd3fa956cb}"
MODEL_NAME="${MODEL_NAME:-step4pro-l14-optimus}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs/step4pro_smoke_14l_dummy}"
SERVING_PORT="${SERVING_PORT:-8000}"
OPTIMUS_WHEEL_URL="${OPTIMUS_WHEEL_URL:-https://artifactory.stepfun-inc.com/artifactory/api/pypi/stepcast-pypi-release/step-optimus/3.23.24/step_optimus-3.23.24-cp310-cp310-manylinux_2_28_x86_64.whl}"
OPTIMUS_WHEEL_SHA256="${OPTIMUS_WHEEL_SHA256:-2eaec8660cd8505486ec06b09b5b508d73483e0729cbe9a2a60afb5cf9a19cfe}"
VLLM_REPO="${VLLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../vllm-step4-pro" && pwd)}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/remote_b300_single_smoke.sh}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814/single_${RJOB_NAME}}"
PAYLOAD_SOURCE_ROOT="${PAYLOAD_SOURCE_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814}"
IDENTITY_PAYLOAD_ROOT="${IDENTITY_PAYLOAD_ROOT:-${PAYLOAD_SOURCE_ROOT}/pinned_identity_fulltrees_pack_v2}"
LIVE_TIMEOUT_SECONDS="${LIVE_TIMEOUT_SECONDS:-1200}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-900}"
REMOTE_RESULT_TIMEOUT_SECONDS="${REMOTE_RESULT_TIMEOUT_SECONDS:-900}"
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-300}"
CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"
CLEANUP_TIMEOUT_SECONDS="${CLEANUP_TIMEOUT_SECONDS:-180}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
ENABLE_PROFILER="${ENABLE_PROFILER:-0}"
REMOTE_EXEC_TIMEOUT_SECONDS="$((REMOTE_RESULT_TIMEOUT_SECONDS + EVIDENCE_HOLD_SECONDS + 300))"

PAYLOAD_ROOT="${ARTIFACT_ROOT}/payload"
REMOTE_BOOTSTRAP_ROOT="/home/step4pro-bootstrap-${RJOB_NAME}"
REMOTE_EVIDENCE_ROOT="/home/step4pro-evidence-${RJOB_NAME}"
REMOTE_RUNTIME_REPO="/home/pinned-vllm-runtime-${RJOB_NAME}"
REMOTE_MODEL_CONFIG_DIR="${REMOTE_BOOTSTRAP_ROOT}/model_config"
RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"
REMOTE_EVIDENCE_TAR="${ARTIFACT_ROOT}/remote_evidence.tar"
REPLICA=""
CLEANUP_DONE=0

if (( ${#RJOB_NAME} > 50 )); then
    echo "RJOB_NAME exceeds platform limit: ${#RJOB_NAME} > 50" >&2
    exit 1
fi
if [[ ! "${SERVING_PORT}" =~ ^[0-9]+$ ]] ||
    (( SERVING_PORT < 1024 || SERVING_PORT > 65535 )); then
    echo "SERVING_PORT must be an integer in [1024, 65535]: ${SERVING_PORT}" >&2
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

discover_replica() {
    local inventory="$1"
    awk -v prefix="${RJOB_NAME}-" '
        NR > 1 && index($1, prefix) == 1 && $2 == "1/1" && $3 == "Running" {
            print $1
            exit
        }
    ' "${inventory}"
}

pull_remote_evidence() {
    query_replicas "${ARTIFACT_ROOT}/replicas_for_evidence.log"
    REPLICA="$(discover_replica "${ARTIFACT_ROOT}/replicas_for_evidence.log")"
    if [[ -z "${REPLICA}" ]]; then
        return 1
    fi
    scoped 300 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${REPLICA}" -- \
        tar cf - -C /home "$(basename "${REMOTE_EVIDENCE_ROOT}")" \
        >"${REMOTE_EVIDENCE_TAR}" 2>"${ARTIFACT_ROOT}/evidence_pull.log" \
        < /dev/null
    tar xf "${REMOTE_EVIDENCE_TAR}" -C "${ARTIFACT_ROOT}"
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
    cleanup_deadline=$(( $(date +%s) + CLEANUP_TIMEOUT_SECONDS ))
    cleanup_status=1
    while (( $(date +%s) < cleanup_deadline )); do
        query_rjob "${ARTIFACT_ROOT}/cleanup_rjob_poll.log"
        query_replicas "${ARTIFACT_ROOT}/cleanup_replicas_poll.log"
        if ! grep -q "${RJOB_NAME}" \
            "${ARTIFACT_ROOT}/cleanup_rjob_poll.log" \
            "${ARTIFACT_ROOT}/cleanup_replicas_poll.log"; then
            cleanup_status=0
            break
        fi
        sleep 5
    done
    cp "${ARTIFACT_ROOT}/cleanup_rjob_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_rjob_final.log"
    cp "${ARTIFACT_ROOT}/cleanup_replicas_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_replicas_final.log"
    for process_id in "${REMOTE_EXEC_PID:-}" "${LAUNCH_PID:-}"; do
        if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
            kill -TERM -- "-${process_id}" 2>/dev/null
            sleep 2
            kill -KILL -- "-${process_id}" 2>/dev/null
        fi
        if [[ -n "${process_id}" ]]; then
            wait "${process_id}" 2>/dev/null
        fi
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

actual_commit="$(git -C "${VLLM_REPO}" rev-parse HEAD)"
test "${actual_commit}" = "${PINNED_COMMIT}"
test -f "${MODEL_CONFIG_DIR}/config.json"
git -C "${VLLM_REPO}" cat-file -e "${IMAGE_SOURCE_COMMIT}^{commit}"
git -C "${VLLM_REPO}" merge-base --is-ancestor \
    "${IMAGE_SOURCE_COMMIT}" "${PINNED_COMMIT}"
bash -n "${REMOTE_SCRIPT}"

sha256sum \
    "${VLLM_REPO}/rjob-step4pro-optimus-single.sh" \
    "${VLLM_REPO}/rjob-step4pro-2node.sh" \
    "${VLLM_REPO}/vllm/model_executor/models/step4pro.py" \
    "${VLLM_REPO}/vllm/v1/attention/backends/optimus_fa4.py" \
    "${VLLM_REPO}/vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py" \
    >"${ARTIFACT_ROOT}/pinned_inputs.sha256"

query_rjob "${ARTIFACT_ROOT}/preflight_rjob.log"
query_replicas "${ARTIFACT_ROOT}/preflight_replicas.log"
if grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/preflight_rjob.log" ||
    grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/preflight_replicas.log"; then
    echo "RJob name already exists: ${RJOB_NAME}" >&2
    exit 1
fi

test -f "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz"
identity_pack="$(find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.pack' -print)"
identity_index="$(find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.idx' -print)"
test -n "${identity_pack}"
test -n "${identity_index}"
cp "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz" \
    "${PAYLOAD_ROOT}/image_to_pinned_vllm.patch.gz"
cp "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz" \
    "${PAYLOAD_ROOT}/image_base_changed_manifest.tsv.gz"
cp "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz" \
    "${PAYLOAD_ROOT}/pinned_vllm_manifest.sha256.gz"
cp "${identity_pack}" "${PAYLOAD_ROOT}/"
cp "${identity_index}" "${PAYLOAD_ROOT}/"
cp "${REMOTE_SCRIPT}" "${PAYLOAD_ROOT}/remote_b300_single_smoke.sh"
mkdir -p "${PAYLOAD_ROOT}/model_config"
cp "${MODEL_CONFIG_DIR}/config.json" "${PAYLOAD_ROOT}/model_config/config.json"

remote_command="set -euo pipefail
mkdir -p '${REMOTE_BOOTSTRAP_ROOT}' '${REMOTE_EVIDENCE_ROOT}'
chmod +x '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_single_smoke.sh'
export BOOTSTRAP_ROOT='${REMOTE_BOOTSTRAP_ROOT}'
export EVIDENCE_ROOT='${REMOTE_EVIDENCE_ROOT}'
export RUNTIME_REPO='${REMOTE_RUNTIME_REPO}'
export MODEL_PATH='${REMOTE_MODEL_CONFIG_DIR}'
export PATCH_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_to_pinned_vllm.patch.gz'
export BASE_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_base_changed_manifest.tsv.gz'
export PINNED_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/pinned_vllm_manifest.sha256.gz'
export IDENTITY_PACK_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_pack}")'
export IDENTITY_INDEX_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_index}")'
export PINNED_COMMIT='${PINNED_COMMIT}'
export OPTIMUS_WHEEL_URL='${OPTIMUS_WHEEL_URL}'
export OPTIMUS_WHEEL_SHA256='${OPTIMUS_WHEEL_SHA256}'
export EVIDENCE_HOLD_SECONDS='${EVIDENCE_HOLD_SECONDS}'
export CUDA_LAUNCH_BLOCKING='${CUDA_LAUNCH_BLOCKING}'
export ENABLE_PROFILER='${ENABLE_PROFILER}'
exec '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_single_smoke.sh'"
remote_command_bytes="$(printf '%s' "${remote_command}" | wc -c)"
recorded_arg_max="$(getconf ARG_MAX)"
printf 'remote_command_bytes=%s\narg_max=%s\n' \
    "${remote_command_bytes}" "${recorded_arg_max}" |
    tee "${ARTIFACT_ROOT}/command_size.env"
if (( remote_command_bytes >= recorded_arg_max / 2 )); then
    echo "Remote command exceeds the bounded argument-size budget" >&2
    exit 1
fi

cat >"${ARTIFACT_ROOT}/launch.command.txt" <<EOF
/kubebrain/brainctl rjob launch --detach --name ${RJOB_NAME} --charged-group ${CHARGED_GROUP} --private-machine group --positive-tags ${POSITIVE_TAGS} --gpu 1 --cpu 32 --memory 300000 --backoff-limit 1 --image ${IMAGE} --entrypoint /bin/bash -- -lc '<streamed pinned-source single-GPU smoke>'
EOF
printf 'image=%s\nimage_digest=%s\npinned_commit=%s\nimage_source_commit=%s\n' \
    "${IMAGE}" "${IMAGE_DIGEST}" "${PINNED_COMMIT}" "${IMAGE_SOURCE_COMMIT}" \
    >"${ARTIFACT_ROOT}/identities.env"

worker_command="set -euo pipefail
echo WORKER_STARTED=\$(date --iso-8601=seconds)
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
df -hT / /home /dev/shm
test -w /home
echo HOME_WRITABLE=PASS
sleep '${LIVE_TIMEOUT_SECONDS}'"

launch_started_epoch="$(date +%s)"
setsid sudo -n systemd-run --scope \
    -p MemoryMax="${CONTROL_MEMORY_MAX}" \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${LIVE_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl rjob launch \
    --detach \
    --auto-delete-duration=45m \
    --max-wait-duration=10m \
    --name "${RJOB_NAME}" \
    --charged-group "${CHARGED_GROUP}" \
    --private-machine group \
    --positive-tags "${POSITIVE_TAGS}" \
    --gpu 1 \
    --cpu 32 \
    --memory 300000 \
    --backoff-limit 1 \
    --enable-sshd=false \
    --enable-jobutil-config=false \
    -e MODELNAME="${MODEL_NAME}" \
    -e PORT_AUTO0="${SERVING_PORT}" \
    --entrypoint /bin/bash \
    --image "${IMAGE}" \
    -- -lc "${worker_command}" \
    >"${ARTIFACT_ROOT}/launch.log" 2>&1 < /dev/null &
LAUNCH_PID=$!
printf 'launch_pid=%s\n' "${LAUNCH_PID}" >"${ARTIFACT_ROOT}/host_metrics.env"

ready_deadline=$((launch_started_epoch + READY_TIMEOUT_SECONDS))
while (( $(date +%s) < ready_deadline )); do
    query_replicas "${ARTIFACT_ROOT}/replicas_poll.log" || true
    REPLICA="$(discover_replica "${ARTIFACT_ROOT}/replicas_poll.log")"
    if [[ -n "${REPLICA}" ]]; then
        break
    fi
    sleep 5
done
if [[ -z "${REPLICA}" ]]; then
    echo "Replica did not become Running before ${READY_TIMEOUT_SECONDS}s" >&2
    exit 1
fi
printf 'replica=%s\nscheduling_seconds=%s\n' \
    "${REPLICA}" "$(( $(date +%s) - launch_started_epoch ))" \
    >>"${ARTIFACT_ROOT}/host_metrics.env"

scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
    "replica/${REPLICA}" -- mkdir -p "${REMOTE_BOOTSTRAP_ROOT}" \
    >"${ARTIFACT_ROOT}/remote_bootstrap.log" 2>&1 < /dev/null
tar cf - -C "${PAYLOAD_ROOT}" . |
    scoped 300 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${REPLICA}" -- \
        tar xf - -C "${REMOTE_BOOTSTRAP_ROOT}" \
        >"${ARTIFACT_ROOT}/payload_transport.log" 2>&1

setsid sudo -n systemd-run --scope \
    -p MemoryMax="${CONTROL_MEMORY_MAX}" \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${REMOTE_EXEC_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl -n "${NAMESPACE}" exec -i "replica/${REPLICA}" -- \
    /bin/bash -lc "${remote_command}" \
    >"${ARTIFACT_ROOT}/remote_exec.log" 2>&1 < /dev/null &
REMOTE_EXEC_PID=$!
printf 'remote_exec_pid=%s\n' "${REMOTE_EXEC_PID}" \
    >>"${ARTIFACT_ROOT}/host_metrics.env"

result_deadline=$(( $(date +%s) + REMOTE_RESULT_TIMEOUT_SECONDS ))
remote_result_ready=0
while (( $(date +%s) < result_deadline )); do
    if scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${REPLICA}" -- \
        test -f "${REMOTE_EVIDENCE_ROOT}/remote_result_ready" \
        >/dev/null 2>"${ARTIFACT_ROOT}/result_probe.log" < /dev/null; then
        remote_result_ready=1
        break
    fi
    sleep 5
done
if (( remote_result_ready != 1 )); then
    echo "Remote smoke did not publish remote_result_ready before ${REMOTE_RESULT_TIMEOUT_SECONDS}s" >&2
    exit 1
fi

pull_remote_evidence
evidence_exit=$?
printf 'evidence_pull_exit=%s\n' "${evidence_exit}" \
    >>"${ARTIFACT_ROOT}/host_metrics.env"
test "${evidence_exit}" = "0"

remote_result="${ARTIFACT_ROOT}/$(basename "${REMOTE_EVIDENCE_ROOT}")/result.env"
test -f "${remote_result}"
grep -q '^ONE_GPU_SMOKE=PASS$' "${remote_result}"

cleanup
cleanup_exit=$?
trap - EXIT
test "${cleanup_exit}" = "0"
if grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/cleanup_rjob_final.log" ||
    grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/cleanup_replicas_final.log"; then
    echo "Resource cleanup verification failed" >&2
    exit 1
fi

find "${ARTIFACT_ROOT}" -type f \
    ! -name raw_artifacts.sha256 -print0 |
    sort -z |
    xargs -0 sha256sum >"${ARTIFACT_ROOT}/raw_artifacts.sha256"

printf 'ONE_GPU_HOST_WRAPPER=PASS\nRJOB_NAME=%s\nARTIFACT_ROOT=%s\n' \
    "${RJOB_NAME}" "${ARTIFACT_ROOT}" |
    tee "${ARTIFACT_ROOT}/host_result.env"
