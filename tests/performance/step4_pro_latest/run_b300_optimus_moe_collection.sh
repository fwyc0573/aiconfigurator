#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO_ROOT
export MODE="${MODE:-smoke}"
export RJOB_NAME="${RJOB_NAME:-s4p-aic-moe-$(date +%m%d-%H%M%S)}"
export CHARGED_GROUP="${CHARGED_GROUP:-b300_train_infra}"
export POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"
export REMOTE_COLLECTION_SCRIPT_HOST="${REPO_ROOT}/tests/performance/step4_pro_latest/remote_b300_optimus_moe_collection.sh"
export RESULT_MARKER="B300_OPTIMUS_MOE_COLLECTION"
export HOST_RESULT_MARKER="B300_OPTIMUS_MOE_HOST"
export WORKLOAD_LABEL="Optimus MoE"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/ycfeng/tmp/step4_aic_optimus_moe_b300_20260815/${MODE}_${RJOB_NAME}}"

exec bash "${REPO_ROOT}/tests/performance/step4_pro_latest/run_b300_attention_collection.sh"
