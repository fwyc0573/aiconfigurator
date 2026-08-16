#!/usr/bin/env bash
set -euo pipefail

: "${BOOTSTRAP_ROOT:?BOOTSTRAP_ROOT is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"
: "${RUNTIME_REPO:?RUNTIME_REPO is required}"
: "${AIC_PAYLOAD_PATH:?AIC_PAYLOAD_PATH is required}"
: "${AIC_METADATA_PATH:?AIC_METADATA_PATH is required}"
: "${PATCH_GZ_PATH:?PATCH_GZ_PATH is required}"
: "${BASE_MANIFEST_GZ_PATH:?BASE_MANIFEST_GZ_PATH is required}"
: "${PINNED_MANIFEST_GZ_PATH:?PINNED_MANIFEST_GZ_PATH is required}"
: "${IDENTITY_PACK_PATH:?IDENTITY_PACK_PATH is required}"
: "${IDENTITY_INDEX_PATH:?IDENTITY_INDEX_PATH is required}"
: "${PINNED_COMMIT:?PINNED_COMMIT is required}"
: "${EP_SIZE:?EP_SIZE is required}"
: "${MODE:?MODE is required}"
: "${NODE_RANK:?NODE_RANK is required}"
: "${NODE_COUNT:?NODE_COUNT is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${PROC_PER_NODE:?PROC_PER_NODE is required}"

EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-300}"
MASTER_PORT="${MASTER_PORT:-5678}"
SOURCE_PROBE_SCRIPT="${BOOTSTRAP_ROOT}/remote_b300_source_probe.sh"
DRIVER_PATH="${BOOTSTRAP_ROOT}/run_step4_deepep_ht_distributed.py"
NCCL_PREFLIGHT_PATH="${BOOTSTRAP_ROOT}/run_step4_deepep_ht_nccl_preflight.py"
AIC_ROOT="${BOOTSTRAP_ROOT}/aic"
AIC_METADATA_ROOT="${AIC_ROOT}/python_metadata"
OUTPUT_ROOT="${EVIDENCE_ROOT}/dataset"

count_marker_occurrences() {
    local marker="$1"
    local log_path="$2"
    grep -oF -- "${marker}" "${log_path}" | wc -l | tr -d '[:space:]'
}

if [[ "${MODE}" != "preflight" && "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "MODE must be preflight, smoke, or full: ${MODE}" >&2
    exit 1
fi
if [[ "${EP_SIZE}" != "16" && "${EP_SIZE}" != "32" ]]; then
    echo "EP_SIZE must be 16 or 32: ${EP_SIZE}" >&2
    exit 1
fi
if (( NODE_COUNT * PROC_PER_NODE != EP_SIZE )); then
    echo "Platform topology does not match EP_SIZE" >&2
    exit 1
fi

mkdir -p \
    "${EVIDENCE_ROOT}" \
    "${AIC_ROOT}" \
    "${AIC_METADATA_ROOT}" \
    "${OUTPUT_ROOT}"
exec > >(tee -a "${EVIDENCE_ROOT}/remote_stdout.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if (( status == 0 )); then
        if [[ "${MODE}" == "preflight" ]]; then
            printf 'B300_DEEPEP_HT_NCCL_PREFLIGHT=PASS\nmode=%s\nep_size=%s\nnode_rank=%s\n' \
                "${MODE}" "${EP_SIZE}" "${NODE_RANK}" \
                | tee "${EVIDENCE_ROOT}/result.env"
        else
            printf 'B300_DEEPEP_HT_COLLECTION=PASS\nmode=%s\nep_size=%s\nnode_rank=%s\n' \
                "${MODE}" "${EP_SIZE}" "${NODE_RANK}" \
                | tee "${EVIDENCE_ROOT}/result.env"
        fi
    else
        if [[ "${MODE}" == "preflight" ]]; then
            printf 'B300_DEEPEP_HT_NCCL_PREFLIGHT=FAIL\nmode=%s\nep_size=%s\nnode_rank=%s\nexit_code=%s\n' \
                "${MODE}" "${EP_SIZE}" "${NODE_RANK}" "${status}" \
                | tee "${EVIDENCE_ROOT}/result.env"
        else
            printf 'B300_DEEPEP_HT_COLLECTION=FAIL\nmode=%s\nep_size=%s\nnode_rank=%s\nexit_code=%s\n' \
                "${MODE}" "${EP_SIZE}" "${NODE_RANK}" "${status}" \
                | tee "${EVIDENCE_ROOT}/result.env"
        fi
    fi
    touch "${EVIDENCE_ROOT}/remote_result_ready"
    sleep "${EVIDENCE_HOLD_SECONDS}"
    exit "${status}"
}
trap finish EXIT

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64
export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ALL2ALL_BACKEND=deepep_high_throughput
export VLLM_DEEPEP_BUFFER_SIZE_MB="${VLLM_DEEPEP_BUFFER_SIZE_MB:-1024}"

nvidia-smi \
    --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap \
    --format=csv,noheader | tee "${EVIDENCE_ROOT}/gpu_identity.csv"

if [[ "${MODE}" == "preflight" ]]; then
    if [[ -z "${NCCL_IB_HCA:-}" ]]; then
        echo "Platform NCCL_IB_HCA is empty; refusing to run NCCL or DeepEP" >&2
        exit 64
    fi
    torchrun --nproc-per-node="${PROC_PER_NODE}" \
        --nnodes="${NODE_COUNT}" \
        --node-rank="${NODE_RANK}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        "${NCCL_PREFLIGHT_PATH}" \
        2>&1 | tee "${EVIDENCE_ROOT}/nccl_preflight.log"

    rank_pass_count="$(
        count_marker_occurrences "STEP4_NCCL_PREFLIGHT_RANK=PASS" \
            "${EVIDENCE_ROOT}/nccl_preflight.log"
    )"
    test "${rank_pass_count}" = "${PROC_PER_NODE}"
    expected_rank_sum="$((EP_SIZE * (EP_SIZE + 1) / 2))"
    grep -q "rank_sum=${expected_rank_sum}" \
        "${EVIDENCE_ROOT}/nccl_preflight.log"
    grep -q "participant_sum=${EP_SIZE}" \
        "${EVIDENCE_ROOT}/nccl_preflight.log"
    if [[ "${NODE_RANK}" == "0" ]]; then
        grep -qF -- 'STEP4_NCCL_PREFLIGHT_DISTRIBUTED=PASS' \
            "${EVIDENCE_ROOT}/nccl_preflight.log"
    fi
    printf 'rank_pass_count=%s\nworld_size=%s\nrank_sum=%s\nparticipant_sum=%s\nhca_present=1\n' \
        "${rank_pass_count}" "${EP_SIZE}" "${expected_rank_sum}" "${EP_SIZE}" \
        >"${EVIDENCE_ROOT}/nccl_preflight.env"
    exit 0
fi

SOURCE_EVIDENCE_ROOT="${EVIDENCE_ROOT}/source_identity"
mkdir -p "${SOURCE_EVIDENCE_ROOT}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT}" \
EVIDENCE_ROOT="${SOURCE_EVIDENCE_ROOT}" \
RUNTIME_REPO="${RUNTIME_REPO}" \
PATCH_GZ_PATH="${PATCH_GZ_PATH}" \
BASE_MANIFEST_GZ_PATH="${BASE_MANIFEST_GZ_PATH}" \
PINNED_MANIFEST_GZ_PATH="${PINNED_MANIFEST_GZ_PATH}" \
IDENTITY_PACK_PATH="${IDENTITY_PACK_PATH}" \
IDENTITY_INDEX_PATH="${IDENTITY_INDEX_PATH}" \
PINNED_COMMIT="${PINNED_COMMIT}" \
    "${SOURCE_PROBE_SCRIPT}" \
    | tee "${EVIDENCE_ROOT}/source_probe.log"
grep -q '^pinned_manifest_files_verified=2103$' \
    "${EVIDENCE_ROOT}/source_probe.log"

tar xf "${AIC_PAYLOAD_PATH}" -C "${AIC_ROOT}"
tar xf "${AIC_METADATA_PATH}" -C "${AIC_METADATA_ROOT}"
export PYTHONPATH="${AIC_METADATA_ROOT}:${AIC_ROOT}/src:${AIC_ROOT}:${RUNTIME_REPO}"

python3 - "${RUNTIME_REPO}" "${AIC_ROOT}" <<'PY' \
    | tee "${EVIDENCE_ROOT}/runtime_import_identity.log"
from pathlib import Path
import importlib.metadata
import inspect
import sys

import aiconfigurator
import deep_ep
import torch
import vllm
import vllm._C
from collector.wideep.vllm import collect_step4_deepep_ht
from vllm.distributed.device_communicators.all2all import (
    DeepEPHTAll2AllManager,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
    DeepEPHTPrepareAndFinalize,
)

runtime_root = Path(sys.argv[1]).resolve()
aic_root = Path(sys.argv[2]).resolve()
paths = {
    "vllm": Path(vllm.__file__).resolve(),
    "manager": Path(inspect.getsourcefile(DeepEPHTAll2AllManager)).resolve(),
    "prepare_finalize": Path(
        inspect.getsourcefile(DeepEPHTPrepareAndFinalize)
    ).resolve(),
    "aiconfigurator": Path(aiconfigurator.__file__).resolve(),
    "collector": Path(
        inspect.getsourcefile(collect_step4_deepep_ht)
    ).resolve(),
}
for key, path in paths.items():
    print(key, path)
if any(runtime_root not in paths[key].parents for key in (
    "vllm",
    "manager",
    "prepare_finalize",
)):
    raise SystemExit(f"vLLM DeepEP source escaped pinned runtime root: {paths}")
if any(aic_root not in paths[key].parents for key in (
    "aiconfigurator",
    "collector",
)):
    raise SystemExit(f"AIC source escaped transferred root: {paths}")
if vllm.__version__ != "0.19.0.post20.dev26+gc820e5ae1":
    raise SystemExit(f"unexpected vLLM version: {vllm.__version__}")
print("torch", torch.__version__, torch.version.cuda)
print("capability", torch.cuda.get_device_capability())
print("vllm", vllm.__version__)
print("deep_ep", importlib.metadata.version("deep_ep"))
print("DeepEPHTAll2AllManager", DeepEPHTAll2AllManager)
print("provider", collect_step4_deepep_ht.PROVIDER)
if collect_step4_deepep_ht.PROVIDER != "vllm_deepep_high_throughput":
    raise SystemExit("unexpected Step4 DeepEP HT provider identity")
PY

torchrun --nproc-per-node="${PROC_PER_NODE}" \
    --nnodes="${NODE_COUNT}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    "${DRIVER_PATH}" \
    --ep-size "${EP_SIZE}" \
    --mode "${MODE}" \
    --output-dir "${OUTPUT_ROOT}" \
    2>&1 | tee "${EVIDENCE_ROOT}/torchrun.log"

rank_pass_count="$(
    count_marker_occurrences "STEP4_DEEPEP_HT_RANK=PASS" \
        "${EVIDENCE_ROOT}/torchrun.log"
)"
test "${rank_pass_count}" = "${PROC_PER_NODE}"
if [[ "${NODE_RANK}" == "0" ]]; then
    grep -qF -- 'STEP4_DEEPEP_HT_DISTRIBUTED=PASS' \
        "${EVIDENCE_ROOT}/torchrun.log"
    test -s "${OUTPUT_ROOT}/step4_deepep_ht_perf.txt"
    test -s "${OUTPUT_ROOT}/step4_deepep_ht_summary.json"
else
    test ! -e "${OUTPUT_ROOT}/step4_deepep_ht_perf.txt"
    test ! -e "${OUTPUT_ROOT}/step4_deepep_ht_summary.json"
fi

printf 'node_rank=%s\nrank_pass_count=%s\npinned_manifest_files_verified=2103\n' \
    "${NODE_RANK}" "${rank_pass_count}" \
    >"${EVIDENCE_ROOT}/node_metrics.env"
