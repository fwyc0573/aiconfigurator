#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-shai-core}"
RJOB_NAME="${RJOB_NAME:-step4pro-b300-source-probe-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled}"
CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../vllm-step4-pro" && pwd)}"
IMAGE_SOURCE_COMMIT="${IMAGE_SOURCE_COMMIT:-c820e5ae1e43246b194080cecc772dcd3fa956cb}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-607d1641ee3fec43653fca510d717725828890c2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814/source_probe_${RJOB_NAME}}"
REMOTE_PARENT="${REMOTE_PARENT:-/home}"
REMOTE_REPO="${REMOTE_PARENT}/pinned-vllm-source-${RJOB_NAME}"
PAYLOAD_ROOT="${ARTIFACT_ROOT}/payload"
REMOTE_BOOTSTRAP_ROOT="${REMOTE_PARENT}/step4pro-source-probe-${RJOB_NAME}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/remote_b300_source_probe.sh}"
PAYLOAD_SOURCE_ROOT="${PAYLOAD_SOURCE_ROOT:-/data/ycfeng/tmp/b300_step4_smoke_20260814}"
IDENTITY_PAYLOAD_ROOT="${IDENTITY_PAYLOAD_ROOT:-${PAYLOAD_SOURCE_ROOT}/pinned_identity_fulltrees_pack_v2}"
WORKER_HOLD_SECONDS="${WORKER_HOLD_SECONDS:-1800}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-420}"
LIVE_TIMEOUT_SECONDS="$((READY_TIMEOUT_SECONDS + WORKER_HOLD_SECONDS + 300))"
RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"

if (( ${#RJOB_NAME} > 50 )); then
    echo "RJOB_NAME exceeds platform limit: ${#RJOB_NAME} > 50" >&2
    exit 1
fi

mkdir -p "${ARTIFACT_ROOT}" "${PAYLOAD_ROOT}"

scoped() {
    local timeout_seconds="$1"
    shift
    sudo -n systemd-run --scope -p MemoryMax=3G --expand-environment=no \
        timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" "$@"
}

exact_inventory() {
    local kind="$1"
    local name="$2"
    local output="$3"
    scoped 60 /kubebrain/brainctl get "${kind}" "${name}" \
        -n "${NAMESPACE}" --ignore-not-found >"${output}" 2>&1
}

replica_inventory() {
    local output="$1"
    scoped 60 /kubebrain/brainctl get replica -n "${NAMESPACE}" \
        -l "${RJOB_LABEL}" >"${output}" 2>&1
}

cleanup() {
    local status=$?
    set +e
    scoped 60 /kubebrain/brainctl delete rjob "${RJOB_NAME}" -n "${NAMESPACE}" \
        >"${ARTIFACT_ROOT}/cleanup_delete.log" 2>&1
    cleanup_deadline=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
    while (( $(date +%s) < cleanup_deadline )); do
        exact_inventory rjob "${RJOB_NAME}" "${ARTIFACT_ROOT}/cleanup_rjobs_poll.log"
        replica_inventory "${ARTIFACT_ROOT}/cleanup_replicas_poll.log"
        if ! grep -q "${RJOB_NAME}" \
            "${ARTIFACT_ROOT}/cleanup_rjobs_poll.log" \
            "${ARTIFACT_ROOT}/cleanup_replicas_poll.log"; then
            break
        fi
        sleep 5
    done
    cp "${ARTIFACT_ROOT}/cleanup_rjobs_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_rjobs.log"
    cp "${ARTIFACT_ROOT}/cleanup_replicas_poll.log" \
        "${ARTIFACT_ROOT}/cleanup_replicas.log"
    if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        kill -TERM -- "-${LAUNCH_PID}" 2>/dev/null
        sleep 2
        kill -KILL -- "-${LAUNCH_PID}" 2>/dev/null
    fi
    if [[ -n "${LAUNCH_PID:-}" ]]; then
        wait "${LAUNCH_PID}" 2>/dev/null
    fi
    pgrep -af "${RJOB_NAME}" >"${ARTIFACT_ROOT}/cleanup_local_processes.log" 2>&1
    set -e
    exit "${status}"
}
trap cleanup EXIT

actual_commit="$(git -C "${LOCAL_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Pinned repository mismatch: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
    exit 1
fi
git -C "${LOCAL_REPO}" cat-file -e "${IMAGE_SOURCE_COMMIT}^{commit}"
git -C "${LOCAL_REPO}" merge-base --is-ancestor \
    "${IMAGE_SOURCE_COMMIT}" "${EXPECTED_COMMIT}"

sha256sum \
    "${LOCAL_REPO}/vllm/model_executor/models/step4pro.py" \
    "${LOCAL_REPO}/vllm/v1/attention/backends/optimus_fa4.py" \
    "${LOCAL_REPO}/vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py" \
    >"${ARTIFACT_ROOT}/local_source.sha256"

cp "${REMOTE_SCRIPT}" "${PAYLOAD_ROOT}/remote_b300_source_probe.sh"
test -f "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz"
test -f "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz"
identity_pack="$(find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.pack' -print)"
identity_index="$(find "${IDENTITY_PAYLOAD_ROOT}" -maxdepth 1 -type f -name 'pack-*.idx' -print)"
test -n "${identity_pack}"
test -n "${identity_index}"
cp "${PAYLOAD_SOURCE_ROOT}/image_to_pinned_vllm.patch.gz" \
    "${PAYLOAD_ROOT}/pinned_commit.patch.gz"
cp "${PAYLOAD_SOURCE_ROOT}/image_base_changed_manifest.tsv.gz" \
    "${PAYLOAD_ROOT}/base_manifest.tsv.gz"
cp "${PAYLOAD_SOURCE_ROOT}/pinned_vllm_manifest.sha256.gz" \
    "${PAYLOAD_ROOT}/pinned_vllm_manifest.sha256.gz"
cp "${identity_pack}" "${PAYLOAD_ROOT}/"
cp "${identity_index}" "${PAYLOAD_ROOT}/"

exact_inventory rjob "${RJOB_NAME}" "${ARTIFACT_ROOT}/preflight_rjobs.log"
replica_inventory "${ARTIFACT_ROOT}/preflight_replicas.log"
if grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/preflight_rjobs.log" ||
    grep -q "${RJOB_NAME}" "${ARTIFACT_ROOT}/preflight_replicas.log"; then
    echo "RJob name already exists: ${RJOB_NAME}" >&2
    exit 1
fi

launch_started_epoch="$(date +%s)"
launch_started_iso="$(date --iso-8601=seconds)"
cat >"${ARTIFACT_ROOT}/launch.command.txt" <<EOF
/kubebrain/brainctl rjob launch --detach --auto-delete-duration=45m --name ${RJOB_NAME} --charged-group ${CHARGED_GROUP} --private-machine group --positive-tags ${POSITIVE_TAGS} --gpu 1 --cpu 8 --memory 65536 --backoff-limit 1 --enable-sshd=false --enable-jobutil-config=false --entrypoint /bin/bash --image ${IMAGE} -- -lc '<source-probe-worker-command>'
EOF

worker_command="$(
    cat <<EOF
set -euo pipefail
echo WORKER_STARTED=\$(date --iso-8601=seconds)
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
echo ===FILESYSTEMS===
df -hT / /home /dev/shm 2>&1
echo ===WRITE_PROBE===
probe_path=${REMOTE_PARENT}/.step4pro-write-probe-\$\$
printf 'ok\n' > "\${probe_path}"
cat "\${probe_path}"
rm -f "\${probe_path}"
echo WRITABLE_REMOTE_PARENT=${REMOTE_PARENT}
sleep ${WORKER_HOLD_SECONDS}
EOF
)"

setsid sudo -n systemd-run --scope -p MemoryMax=3G --expand-environment=no \
    timeout --signal=TERM --kill-after=30s "${LIVE_TIMEOUT_SECONDS}s" \
    /kubebrain/brainctl rjob launch \
    --detach \
    --auto-delete-duration=45m \
    --name "${RJOB_NAME}" \
    --charged-group "${CHARGED_GROUP}" \
    --private-machine group \
    --positive-tags "${POSITIVE_TAGS}" \
    --gpu 1 \
    --cpu 8 \
    --memory 65536 \
    --backoff-limit 1 \
    --enable-sshd=false \
    --enable-jobutil-config=false \
    --entrypoint /bin/bash \
    --image "${IMAGE}" \
    -- -lc "${worker_command}" \
    >"${ARTIFACT_ROOT}/launch.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

deadline=$((launch_started_epoch + READY_TIMEOUT_SECONDS))
replica=""
while (( $(date +%s) < deadline )); do
    replica_inventory "${ARTIFACT_ROOT}/replicas_poll.log"
    replica="$(
        awk -v prefix="${RJOB_NAME}-" '
            NR > 1 && index($1, prefix) == 1 && $2 == "1/1" && $3 == "Running" {
                print $1
                exit
            }
        ' "${ARTIFACT_ROOT}/replicas_poll.log"
    )"
    if [[ -n "${replica}" ]]; then
        break
    fi
    sleep 5
done
if [[ -z "${replica}" ]]; then
    echo "Replica did not become Running before ${READY_TIMEOUT_SECONDS}s" >&2
    exit 1
fi
printf 'replica=%s\n' "${replica}" >"${ARTIFACT_ROOT}/replica.env"
printf 'scheduling_seconds=%s\n' "$(( $(date +%s) - launch_started_epoch ))" \
    >>"${ARTIFACT_ROOT}/replica.env"
printf 'launch_started=%s\n' "${launch_started_iso}" >>"${ARTIFACT_ROOT}/replica.env"

scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i "replica/${replica}" -- \
    /bin/bash -lc '
        set -euo pipefail
        echo EXEC_STARTED=$(date --iso-8601=seconds)
        nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
        df -hT / /home /dev/shm
        test -w /home
        echo HOME_WRITABLE=PASS
        python3 --version
        python3 - <<'"'"'PY'"'"'
import importlib.metadata
import sys

import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
for package in ("vllm", "step-optimus"):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package, "<not installed>")
PY
    ' >"${ARTIFACT_ROOT}/worker_preflight.log" 2>&1 < /dev/null

scoped 60 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
    "replica/${replica}" -- mkdir -p "${REMOTE_BOOTSTRAP_ROOT}" \
    >"${ARTIFACT_ROOT}/remote_bootstrap.log" 2>&1 < /dev/null
tar cf - -C "${PAYLOAD_ROOT}" . |
    scoped 300 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
        "replica/${replica}" -- \
        tar xf - -C "${REMOTE_BOOTSTRAP_ROOT}" \
        >"${ARTIFACT_ROOT}/source_transport.log" 2>&1

remote_command="$(
    cat <<EOF
set -euo pipefail
export BOOTSTRAP_ROOT='${REMOTE_BOOTSTRAP_ROOT}'
export EVIDENCE_ROOT='${REMOTE_BOOTSTRAP_ROOT}/evidence'
export RUNTIME_REPO='${REMOTE_REPO}'
export PATCH_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/pinned_commit.patch.gz'
export BASE_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/base_manifest.tsv.gz'
export PINNED_MANIFEST_GZ_PATH='${REMOTE_BOOTSTRAP_ROOT}/pinned_vllm_manifest.sha256.gz'
export IDENTITY_PACK_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_pack}")'
export IDENTITY_INDEX_PATH='${REMOTE_BOOTSTRAP_ROOT}/$(basename "${identity_index}")'
export PINNED_COMMIT='${EXPECTED_COMMIT}'
chmod +x '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_source_probe.sh'
exec '${REMOTE_BOOTSTRAP_ROOT}/remote_b300_source_probe.sh'
EOF
)"
scoped 300 /kubebrain/brainctl -n "${NAMESPACE}" exec -i \
    "replica/${replica}" -- /bin/bash -lc "${remote_command}" \
    >"${ARTIFACT_ROOT}/remote_source_identity.log" 2>&1 < /dev/null

grep -F "${EXPECTED_COMMIT}" "${ARTIFACT_ROOT}/remote_source_identity.log" >/dev/null
while read -r expected_hash _; do
    grep -F "${expected_hash}" "${ARTIFACT_ROOT}/remote_source_identity.log" >/dev/null
done <"${ARTIFACT_ROOT}/local_source.sha256"
grep -F "vllm_file ${REMOTE_REPO}/vllm/__init__.py" \
    "${ARTIFACT_ROOT}/remote_source_identity.log" >/dev/null
grep -F "step4pro_file ${REMOTE_REPO}/vllm/model_executor/models/step4pro.py" \
    "${ARTIFACT_ROOT}/remote_source_identity.log" >/dev/null
grep -F "optimus_fa4_file ${REMOTE_REPO}/vllm/v1/attention/backends/optimus_fa4.py" \
    "${ARTIFACT_ROOT}/remote_source_identity.log" >/dev/null
grep -F "SOURCE_PROBE=PASS" \
    "${ARTIFACT_ROOT}/remote_source_identity.log" >/dev/null

printf 'SOURCE_PROBE=PASS\nRJOB_NAME=%s\nREPLICA=%s\nARTIFACT_ROOT=%s\n' \
    "${RJOB_NAME}" "${replica}" "${ARTIFACT_ROOT}" |
    tee "${ARTIFACT_ROOT}/result.txt"
