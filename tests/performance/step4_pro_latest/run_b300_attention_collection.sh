#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-shai-core}"
MODE="${MODE:-smoke}"
RJOB_NAME="${RJOB_NAME:-s4p-aic-attn-$(date +%m%d-%H%M%S)}"
IMAGE="${IMAGE:-hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled}"
CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
PINNED_COMMIT="${PINNED_COMMIT:-607d1641ee3fec43653fca510d717725828890c2}"
OPTIMUS_WHEEL_URL="${OPTIMUS_WHEEL_URL:-https://artifactory.stepfun-inc.com/artifactory/api/pypi/stepcast-pypi-release/step-optimus/3.23.24/step_optimus-3.23.24-cp310-cp310-manylinux_2_28_x86_64.whl}"
OPTIMUS_WHEEL_SHA256="${OPTIMUS_WHEEL_SHA256:-2eaec8660cd8505486ec06b09b5b508d73483e0729cbe9a2a60afb5cf9a19cfe}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REMOTE_COLLECTION_SCRIPT_HOST="${REMOTE_COLLECTION_SCRIPT_HOST:-${REPO_ROOT}/tests/performance/step4_pro_latest/remote_b300_attention_collection.sh}"
REMOTE_COLLECTION_SCRIPT_BASENAME="$(basename "${REMOTE_COLLECTION_SCRIPT_HOST}")"
RESULT_MARKER="${RESULT_MARKER:-B300_ATTENTION_COLLECTION}"
HOST_RESULT_MARKER="${HOST_RESULT_MARKER:-B300_ATTENTION_HOST}"
REMOTE_RESULT_MARKER="${REMOTE_RESULT_MARKER:-${RESULT_MARKER}}"
COLLECTION_SUITE="${COLLECTION_SUITE:-attention}"
PROVIDER_CORE_SLICE="${PROVIDER_CORE_SLICE:-all}"
PROVIDER_CORE_SMOKE_EVIDENCE="${PROVIDER_CORE_SMOKE_EVIDENCE:-}"
PROVIDER_CORE_SMOKE_EVIDENCE_SHA256="${PROVIDER_CORE_SMOKE_EVIDENCE_SHA256:-}"
FULL_CONTEXT_SMOKE_TOKENS="${FULL_CONTEXT_SMOKE_TOKENS:-512}"
FULL_CONTEXT_SMOKE_TOTAL_TOKENS="${FULL_CONTEXT_SMOKE_TOTAL_TOKENS:-${FULL_CONTEXT_SMOKE_TOKENS}}"
SWA_CONTEXT_SMOKE_TOKENS="${SWA_CONTEXT_SMOKE_TOKENS:-512}"
SWA_CONTEXT_SMOKE_TOTAL_TOKENS="${SWA_CONTEXT_SMOKE_TOTAL_TOKENS:-${SWA_CONTEXT_SMOKE_TOKENS}}"
WORKLOAD_LABEL="${WORKLOAD_LABEL:-Attention}"
AIC_HOST_PYTHON="${AIC_HOST_PYTHON:-/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python}"
PAYLOAD_SOURCE_ROOT="${PAYLOAD_SOURCE_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814}"
IDENTITY_PAYLOAD_ROOT="${IDENTITY_PAYLOAD_ROOT:-${PAYLOAD_SOURCE_ROOT}/pinned_identity_fulltrees_pack_v2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/step4_aic_attention_b300_20260815/${MODE}_${RJOB_NAME}}"
CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-3G}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-600}"
RESULT_TIMEOUT_SECONDS="${RESULT_TIMEOUT_SECONDS:-2400}"
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-120}"
WORKER_HOLD_SECONDS="$((RESULT_TIMEOUT_SECONDS + EVIDENCE_HOLD_SECONDS + 600))"
LIVE_TIMEOUT_SECONDS="$((WORKER_HOLD_SECONDS + READY_TIMEOUT_SECONDS + 300))"
REMOTE_EXEC_TIMEOUT_SECONDS="$((RESULT_TIMEOUT_SECONDS + EVIDENCE_HOLD_SECONDS + 300))"

PAYLOAD_ROOT="${ARTIFACT_ROOT}/payload"
REMOTE_BOOTSTRAP_ROOT="/home/s4p-aic-attn-${RJOB_NAME}"
REMOTE_EVIDENCE_ROOT="/home/s4p-aic-attn-evidence-${RJOB_NAME}"
REMOTE_RUNTIME_REPO="/home/s4p-pinned-vllm-${RJOB_NAME}"
REMOTE_AIC_PAYLOAD="${REMOTE_BOOTSTRAP_ROOT}/aic_payload.tar"
REMOTE_AIC_METADATA="${REMOTE_BOOTSTRAP_ROOT}/aic_metadata.tar"
RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"
REMOTE_EVIDENCE_TAR="${ARTIFACT_ROOT}/remote_evidence.tar"
REPLICA=""
LAUNCH_PID=""
REMOTE_EXEC_PID=""
CLEANUP_DONE=0

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "MODE must be smoke or full: ${MODE}" >&2
    exit 1
fi
if [[ "${COLLECTION_SUITE}" == "provider_core" \
    && "${PROVIDER_CORE_SLICE}" == "all" ]]; then
    echo "provider_core requires an explicit non-all PROVIDER_CORE_SLICE" >&2
    exit 1
fi
if [[ ! "${SWA_CONTEXT_SMOKE_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SWA_CONTEXT_SMOKE_TOKENS must be a positive integer: ${SWA_CONTEXT_SMOKE_TOKENS}" >&2
    exit 1
fi
if [[ ! "${FULL_CONTEXT_SMOKE_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FULL_CONTEXT_SMOKE_TOKENS must be a positive integer: ${FULL_CONTEXT_SMOKE_TOKENS}" >&2
    exit 1
fi
if [[ ! "${FULL_CONTEXT_SMOKE_TOTAL_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FULL_CONTEXT_SMOKE_TOTAL_TOKENS must be a positive integer: ${FULL_CONTEXT_SMOKE_TOTAL_TOKENS}" >&2
    exit 1
fi
if (( FULL_CONTEXT_SMOKE_TOTAL_TOKENS < FULL_CONTEXT_SMOKE_TOKENS )); then
    echo "FULL_CONTEXT_SMOKE_TOTAL_TOKENS cannot be smaller than FULL_CONTEXT_SMOKE_TOKENS" >&2
    exit 1
fi
if [[ ! "${SWA_CONTEXT_SMOKE_TOTAL_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SWA_CONTEXT_SMOKE_TOTAL_TOKENS must be a positive integer: ${SWA_CONTEXT_SMOKE_TOTAL_TOKENS}" >&2
    exit 1
fi
if (( SWA_CONTEXT_SMOKE_TOTAL_TOKENS < SWA_CONTEXT_SMOKE_TOKENS )); then
    echo "SWA_CONTEXT_SMOKE_TOTAL_TOKENS cannot be smaller than SWA_CONTEXT_SMOKE_TOKENS" >&2
    exit 1
fi
if [[ ! -f "${REMOTE_COLLECTION_SCRIPT_HOST}" ]]; then
    echo "Remote collection script is missing: ${REMOTE_COLLECTION_SCRIPT_HOST}" >&2
    exit 1
fi
if (( ${#RJOB_NAME} > 50 )); then
    echo "RJOB_NAME exceeds platform limit: ${#RJOB_NAME} > 50" >&2
    exit 1
fi
if [[ "${MODE}" == "full" && "${RESULT_TIMEOUT_SECONDS}" == "2400" ]]; then
    RESULT_TIMEOUT_SECONDS=28800
    WORKER_HOLD_SECONDS="$((RESULT_TIMEOUT_SECONDS + EVIDENCE_HOLD_SECONDS + 600))"
    LIVE_TIMEOUT_SECONDS="$((WORKER_HOLD_SECONDS + READY_TIMEOUT_SECONDS + 300))"
    REMOTE_EXEC_TIMEOUT_SECONDS="$((RESULT_TIMEOUT_SECONDS + EVIDENCE_HOLD_SECONDS + 300))"
fi

validate_provider_core_smoke_evidence() {
    local evidence_path="$1"
    local evidence_sha256="$2"
    local expected_slice="$3"
    if [[ -z "${evidence_path}" || -z "${evidence_sha256}" ]]; then
        echo "Full provider_core collection requires PROVIDER_CORE_SMOKE_EVIDENCE and PROVIDER_CORE_SMOKE_EVIDENCE_SHA256" >&2
        return 1
    fi
    test -f "${evidence_path}"
    printf '%s  %s\n' "${evidence_sha256}" "${evidence_path}" \
        | sha256sum --check --status
    "${AIC_HOST_PYTHON}" - "${evidence_path}" "${expected_slice}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_slice = sys.argv[2]
expected_completed_cases = 2 if expected_slice == "grouped_router" else 1
if payload["mode"] != "smoke":
    raise RuntimeError(f"Expected smoke evidence, got {payload!r}")
if payload["suite"] != "provider_core":
    raise RuntimeError(f"Expected provider_core smoke evidence, got {payload!r}")
if payload["slice"] != expected_slice:
    raise RuntimeError(
        f"Smoke evidence slice mismatch: expected={expected_slice}, "
        f"actual={payload['slice']}"
    )
if payload["completed_cases"] != expected_completed_cases:
    raise RuntimeError(
        "Smoke evidence completed-case mismatch: "
        f"expected={expected_completed_cases}, "
        f"actual={payload['completed_cases']}"
    )
PY
}

if [[ "${COLLECTION_SUITE}" == "provider_core" && "${MODE}" == "full" ]]; then
    validate_provider_core_smoke_evidence \
        "${PROVIDER_CORE_SMOKE_EVIDENCE}" \
        "${PROVIDER_CORE_SMOKE_EVIDENCE_SHA256}" \
        "${PROVIDER_CORE_SLICE}"
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

cleanup() {
    local status=$?
    if (( CLEANUP_DONE == 1 )); then
        return
    fi
    CLEANUP_DONE=1
    set +e
    scoped 60 /kubebrain/brainctl delete rjob "${RJOB_NAME}" \
        -n "${NAMESPACE}" >"${ARTIFACT_ROOT}/cleanup_delete.log" 2>&1
    cleanup_deadline=$(( $(date +%s) + 180 ))
    while (( $(date +%s) < cleanup_deadline )); do
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
    for pid in "${REMOTE_EXEC_PID}" "${LAUNCH_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null
            sleep 2
            kill -KILL -- "-${pid}" 2>/dev/null
        fi
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null
        fi
    done
    set -e
    return "${status}"
}
trap cleanup EXIT

test "$(git -C "${REPO_ROOT}" branch --show-current)" = \
    "task/step4-pro-latest-b300"
test "$(git -C "${REPO_ROOT}/vllm-step4-pro" rev-parse HEAD)" = \
    "${PINNED_COMMIT}"

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

cp "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz" \
    "${PAYLOAD_ROOT}/image_to_pinned_vllm.patch.gz"
cp "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz" \
    "${PAYLOAD_ROOT}/image_base_changed_manifest.tsv.gz"
cp "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz" \
    "${PAYLOAD_ROOT}/pinned_vllm_manifest.sha256.gz"
cp "${identity_pack}" "${PAYLOAD_ROOT}/"
cp "${identity_index}" "${PAYLOAD_ROOT}/"
cp "${REPO_ROOT}/tests/e2e/step4_pro_latest/remote_b300_source_probe.sh" \
    "${PAYLOAD_ROOT}/remote_b300_source_probe.sh"
cp "${REMOTE_COLLECTION_SCRIPT_HOST}" \
    "${PAYLOAD_ROOT}/${REMOTE_COLLECTION_SCRIPT_BASENAME}"
if [[ "${COLLECTION_SUITE}" == "provider_core" && "${MODE}" == "full" ]]; then
    cp "${PROVIDER_CORE_SMOKE_EVIDENCE}" \
        "${PAYLOAD_ROOT}/provider_core_smoke_evidence.json"
fi

git -C "${REPO_ROOT}" ls-files collector src/aiconfigurator pyproject.toml \
    | grep -v '^src/aiconfigurator/systems/data/' \
    >"${PAYLOAD_ROOT}/aic_payload_files.txt"
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
    "${PAYLOAD_ROOT}/${REMOTE_COLLECTION_SCRIPT_BASENAME}" \
    >"${ARTIFACT_ROOT}/payload.sha256"
if [[ "${COLLECTION_SUITE}" == "provider_core" && "${MODE}" == "full" ]]; then
    sha256sum "${PAYLOAD_ROOT}/provider_core_smoke_evidence.json" \
        >>"${ARTIFACT_ROOT}/payload.sha256"
fi

query_rjob "${ARTIFACT_ROOT}/preflight_rjob.log"
query_replicas "${ARTIFACT_ROOT}/preflight_replicas.log"
if grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/preflight_rjob.log" \
    || grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/preflight_replicas.log"; then
    echo "RJob name already exists: ${RJOB_NAME}" >&2
    exit 1
fi

cat >"${ARTIFACT_ROOT}/launch.command.txt" <<EOF
/kubebrain/brainctl rjob launch --detach --name ${RJOB_NAME} --charged-group ${CHARGED_GROUP} --private-machine group --positive-tags ${POSITIVE_TAGS} --gpu 1 --cpu 16 --memory 131072 --backoff-limit 1 --image ${IMAGE} --entrypoint /bin/bash -- -lc '<bounded AIC ${WORKLOAD_LABEL} collector worker>'
EOF

worker_command="set -euo pipefail
echo WORKER_STARTED=\$(date --iso-8601=seconds)
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
df -hT / /home /dev/shm
test -w /home
echo HOME_WRITABLE=PASS
sleep '${WORKER_HOLD_SECONDS}'"

launch_started_epoch="$(date +%s)"
setsid sudo -n systemd-run --scope -p MemoryMax=3G \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${LIVE_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl rjob launch \
    --detach \
    --auto-delete-duration=10h \
    --max-wait-duration=10m \
    --name "${RJOB_NAME}" \
    --charged-group "${CHARGED_GROUP}" \
    --private-machine group \
    --positive-tags "${POSITIVE_TAGS}" \
    --gpu 1 \
    --cpu 16 \
    --memory 131072 \
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
    >"${ARTIFACT_ROOT}/worker.env"

scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
    "replica/${REPLICA}" -- \
    mkdir -p "${REMOTE_BOOTSTRAP_ROOT}" "${REMOTE_EVIDENCE_ROOT}" \
    >"${ARTIFACT_ROOT}/remote_bootstrap.log" 2>&1 < /dev/null
tar cf - -C "${PAYLOAD_ROOT}" . \
    | scoped 600 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${REPLICA}" -- \
        tar xf - -C "${REMOTE_BOOTSTRAP_ROOT}" \
        >"${ARTIFACT_ROOT}/payload_transport.log" 2>&1

remote_command="set -euo pipefail
chmod +x '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_source_probe.sh'
chmod +x '${REMOTE_BOOTSTRAP_ROOT}/${REMOTE_COLLECTION_SCRIPT_BASENAME}'
export BOOTSTRAP_ROOT='${REMOTE_BOOTSTRAP_ROOT}'
export EVIDENCE_ROOT='${REMOTE_EVIDENCE_ROOT}'
export RUNTIME_REPO='${REMOTE_RUNTIME_REPO}'
export AIC_PAYLOAD_PATH='${REMOTE_AIC_PAYLOAD}'
export AIC_METADATA_PATH='${REMOTE_AIC_METADATA}'
export PATCH_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_to_pinned_vllm.patch.gz'
export BASE_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/image_base_changed_manifest.tsv.gz'
export PINNED_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/pinned_vllm_manifest.sha256.gz'
export IDENTITY_PACK_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_pack}")'
export IDENTITY_INDEX_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_index}")'
export PINNED_COMMIT='${PINNED_COMMIT}'
export OPTIMUS_WHEEL_URL='${OPTIMUS_WHEEL_URL}'
export OPTIMUS_WHEEL_SHA256='${OPTIMUS_WHEEL_SHA256}'
export MODE='${MODE}'
export COLLECTION_SUITE='${COLLECTION_SUITE}'
export PROVIDER_CORE_SLICE='${PROVIDER_CORE_SLICE}'
export PROVIDER_CORE_SMOKE_EVIDENCE='${REMOTE_BOOTSTRAP_ROOT}/provider_core_smoke_evidence.json'
export PROVIDER_CORE_SMOKE_EVIDENCE_SHA256='${PROVIDER_CORE_SMOKE_EVIDENCE_SHA256}'
export FULL_CONTEXT_SMOKE_TOKENS='${FULL_CONTEXT_SMOKE_TOKENS}'
export FULL_CONTEXT_SMOKE_TOTAL_TOKENS='${FULL_CONTEXT_SMOKE_TOTAL_TOKENS}'
export SWA_CONTEXT_SMOKE_TOKENS='${SWA_CONTEXT_SMOKE_TOKENS}'
export SWA_CONTEXT_SMOKE_TOTAL_TOKENS='${SWA_CONTEXT_SMOKE_TOTAL_TOKENS}'
export REMOTE_RESULT_MARKER='${REMOTE_RESULT_MARKER}'
export EVIDENCE_HOLD_SECONDS='${EVIDENCE_HOLD_SECONDS}'
exec '${REMOTE_BOOTSTRAP_ROOT}/${REMOTE_COLLECTION_SCRIPT_BASENAME}'"

setsid sudo -n systemd-run --scope -p MemoryMax=3G \
    --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${REMOTE_EXEC_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl -n "${NAMESPACE}" exec -i "replica/${REPLICA}" -- \
    /bin/bash -lc "${remote_command}" \
    >"${ARTIFACT_ROOT}/remote_exec.log" 2>&1 < /dev/null &
REMOTE_EXEC_PID=$!

result_deadline=$(( $(date +%s) + RESULT_TIMEOUT_SECONDS ))
result_ready=0
while (( $(date +%s) < result_deadline )); do
    if scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${REPLICA}" -- \
        test -f "${REMOTE_EVIDENCE_ROOT}/remote_result_ready" \
        >"${ARTIFACT_ROOT}/result_probe.log" 2>&1 < /dev/null; then
        result_ready=1
        break
    fi
    sleep 10
done
if (( result_ready != 1 )); then
    echo "Remote collection did not finish before ${RESULT_TIMEOUT_SECONDS}s" >&2
    exit 1
fi

scoped 600 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
    "replica/${REPLICA}" -- \
    tar cf - -C /home "$(basename "${REMOTE_EVIDENCE_ROOT}")" \
    >"${REMOTE_EVIDENCE_TAR}" 2>"${ARTIFACT_ROOT}/evidence_pull.log" \
    < /dev/null
tar xf "${REMOTE_EVIDENCE_TAR}" -C "${ARTIFACT_ROOT}"

remote_result="${ARTIFACT_ROOT}/$(basename "${REMOTE_EVIDENCE_ROOT}")/result.env"
test -f "${remote_result}"
grep -q "^${RESULT_MARKER}=PASS$" "${remote_result}"

cleanup
cleanup_exit=$?
trap - EXIT
test "${cleanup_exit}" = "0"
if grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/cleanup_rjob_final.log" \
    || grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/cleanup_replicas_final.log"; then
    echo "Resource cleanup verification failed" >&2
    exit 1
fi

printf '%s=PASS\nmode=%s\nrjob=%s\nartifact_root=%s\n' \
    "${HOST_RESULT_MARKER}" "${MODE}" "${RJOB_NAME}" "${ARTIFACT_ROOT}" \
    | tee "${ARTIFACT_ROOT}/host_result.env"
