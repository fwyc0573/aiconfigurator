#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO_ROOT
export MODE="${MODE:-smoke}"
export RJOB_NAME="${RJOB_NAME:-s4p-aic-core-$(date +%m%d-%H%M%S)}"
export CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
export POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
export REMOTE_COLLECTION_SCRIPT_HOST="${REPO_ROOT}/tests/performance/step4_pro_latest/remote_b300_attention_collection.sh"
export RESULT_MARKER="B300_STEP4_PROVIDER_CORE_COLLECTION"
export REMOTE_RESULT_MARKER="${RESULT_MARKER}"
export HOST_RESULT_MARKER="B300_STEP4_PROVIDER_CORE_HOST"
export COLLECTION_SUITE="provider_core"
export PROVIDER_CORE_SLICE="${PROVIDER_CORE_SLICE:?PROVIDER_CORE_SLICE is required: grouped_router, qkv_full, or qkv_swa}"
export WORKLOAD_LABEL="Step4 provider core"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/${MODE}_${RJOB_NAME}}"

exec bash "${REPO_ROOT}/tests/performance/step4_pro_latest/run_b300_attention_collection.sh"
