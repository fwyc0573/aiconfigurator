#!/usr/bin/env bash
set -euo pipefail

: "${BOOTSTRAP_ROOT:?BOOTSTRAP_ROOT is required}"
: "${EVIDENCE_ROOT:?EVIDENCE_ROOT is required}"
: "${RUNTIME_REPO:?RUNTIME_REPO is required}"
: "${PATCH_GZ_PATH:?PATCH_GZ_PATH is required}"
: "${BASE_MANIFEST_GZ_PATH:?BASE_MANIFEST_GZ_PATH is required}"
: "${PINNED_MANIFEST_GZ_PATH:?PINNED_MANIFEST_GZ_PATH is required}"
: "${IDENTITY_PACK_PATH:?IDENTITY_PACK_PATH is required}"
: "${IDENTITY_INDEX_PATH:?IDENTITY_INDEX_PATH is required}"
: "${PINNED_COMMIT:?PINNED_COMMIT is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODELNAME:?MODELNAME is required}"
: "${PORT_AUTO0:?PORT_AUTO0 is required}"
: "${OPTIMUS_WHEEL_URL:?OPTIMUS_WHEEL_URL is required}"
: "${OPTIMUS_WHEEL_SHA256:?OPTIMUS_WHEEL_SHA256 is required}"
: "${RUNTIME_CONTRACT_LIB:?RUNTIME_CONTRACT_LIB is required}"
source "${RUNTIME_CONTRACT_LIB}"
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-300}"
HEADLESS_WAIT_TIMEOUT_SECONDS="${HEADLESS_WAIT_TIMEOUT_SECONDS:-1800}"
ENABLE_PROFILER="${ENABLE_PROFILER:-0}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-1}"
DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT:-5678}"
VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-allgather_reducescatter}"
if [[ "${VLLM_ALL2ALL_BACKEND}" != "allgather_reducescatter" ]]; then
    echo "VLLM_ALL2ALL_BACKEND must be allgather_reducescatter: ${VLLM_ALL2ALL_BACKEND}" >&2
    exit 1
fi
VLLM_ENABLE_SEQUENCE_PARALLEL="${VLLM_ENABLE_SEQUENCE_PARALLEL:-0}"
if [[ "${VLLM_ENABLE_SEQUENCE_PARALLEL}" != "0" ]]; then
    echo "VLLM_ENABLE_SEQUENCE_PARALLEL must be 0 with AgRs: ${VLLM_ENABLE_SEQUENCE_PARALLEL}" >&2
    exit 1
fi
if [[ "${ENABLE_PROFILER}" != "0" && "${ENABLE_PROFILER}" != "1" ]]; then
    echo "ENABLE_PROFILER must be 0 or 1: ${ENABLE_PROFILER}" >&2
    exit 1
fi

mkdir -p "${EVIDENCE_ROOT}" "${RUNTIME_REPO}"
exec > >(tee -a "${EVIDENCE_ROOT}/remote_stdout.log") 2>&1

METRICS_FILE="${EVIDENCE_ROOT}/metrics.env"
SERVER_LOG="${EVIDENCE_ROOT}/vllm_server.log"
PROFILER_DIR="${EVIDENCE_ROOT}/torch_profiler"
REMOTE_VALIDATION_READY_FILE="${EVIDENCE_ROOT}/remote_validation_ready"
ALLGATHER_REDUCESCATTER_CONFIG_MARKER="DeepEP runtime not available; using allgather_reducescatter all2all backend without sequence parallelism"
PINNED_GPU_MODEL_RUNNER_SHA256="298a43a69f3b5b43bdbb753b3cee642933a0dbd71368dfbf0271dba1fce32bcb"
SERVER_PID=""

record_metric() {
    printf '%s=%s\n' "$1" "$2" | tee -a "${METRICS_FILE}"
}

validate_distributed_all2all_runtime() {
    local require_local_manager_marker="${1:?require_local_manager_marker is required}"
    if [[ "${require_local_manager_marker}" != "0" &&
        "${require_local_manager_marker}" != "1" ]]; then
        echo "require_local_manager_marker must be 0 or 1" >&2
        return 1
    fi
    local agrs_manager_marker_count
    local backend_config_marker_count
    local deepep_manager_marker_count
    local auto_backend_selection_marker_count
    agrs_manager_marker_count="$(
        grep -c "Using AgRsAll2AllManager all2all manager" \
            "${SERVER_LOG}" || true
    )"
    backend_config_marker_count="$(
        grep -Fc "${ALLGATHER_REDUCESCATTER_CONFIG_MARKER}" \
            "${SERVER_LOG}" || true
    )"
    deepep_manager_marker_count="$(
        grep -Ec "Using DeepEP[A-Za-z0-9_]*All2AllManager" \
            "${SERVER_LOG}" || true
    )"
    auto_backend_selection_marker_count="$(
        grep -c 'Auto-configured .*VLLM_ALL2ALL_BACKEND=' \
            "${SERVER_LOG}" || true
    )"
    record_metric agrs_manager_marker_count "${agrs_manager_marker_count}"
    record_metric backend_config_marker_count "${backend_config_marker_count}"
    record_metric deepep_manager_marker_count "${deepep_manager_marker_count}"
    record_metric auto_backend_selection_marker_count \
        "${auto_backend_selection_marker_count}"
    if (( backend_config_marker_count < 1 )); then
        echo "allgather_reducescatter runtime configuration marker is missing" >&2
        return 1
    fi
    if (( require_local_manager_marker == 1 &&
        agrs_manager_marker_count < 1 )); then
        echo "AgRs all2all manager marker is missing" >&2
        return 1
    fi
    if (( deepep_manager_marker_count != 0 )); then
        echo "Unexpected DeepEP all2all manager selected" >&2
        return 1
    fi
    if (( auto_backend_selection_marker_count != 0 )); then
        echo "Unexpected Step MoE automatic backend selection" >&2
        return 1
    fi
    if (( agrs_manager_marker_count > 0 )); then
        record_metric runtime_all2all_manager AgRsAll2AllManager
    else
        record_metric runtime_all2all_manager_marker_scope global_peer
    fi
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

wait_for_server_log_marker() {
    local pattern="$1"
    local description="$2"
    local deadline=$(( $(date +%s) + HEADLESS_WAIT_TIMEOUT_SECONDS ))
    while (( $(date +%s) < deadline )); do
        if grep -qE "${pattern}" "${SERVER_LOG}"; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "vLLM stopped before ${description}" >&2
            cat "${SERVER_LOG}"
            return 1
        fi
        sleep 2
    done
    echo "Timed out waiting for ${description}" >&2
    cat "${SERVER_LOG}"
    return 1
}

hold_distributed_runtime_for_host_cleanup() {
    while true; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "vLLM stopped before host completed distributed cleanup" >&2
            cat "${SERVER_LOG}"
            return 1
        fi
        sleep 5
    done
}

finish() {
    local status=$?
    trap - EXIT
    stop_server
    if (( status == 0 )); then
        printf 'ONE_GPU_SMOKE=PASS\n' | tee "${EVIDENCE_ROOT}/result.env"
    else
        printf 'ONE_GPU_SMOKE=FAIL\nexit_code=%s\n' "${status}" |
            tee "${EVIDENCE_ROOT}/result.env"
    fi
    touch "${EVIDENCE_ROOT}/remote_result_ready"
    sleep "${EVIDENCE_HOLD_SECONDS}"
    exit "${status}"
}
trap finish EXIT

worker_started_epoch="$(date +%s)"
record_metric worker_started_epoch "${worker_started_epoch}"
record_metric worker_started_iso "$(date --iso-8601=seconds)"
record_metric runtime_all2all_backend "${VLLM_ALL2ALL_BACKEND}"
record_metric sequence_parallel false

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64
export OPTIMUS_MUST_LOAD_LIB=1
export VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_USE_OPTIMUS_MOE=1
export VLLM_USE_DEEP_GEMM_E8M0=1
export OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE=1
export VLLM_DUMMY_BATCH_DIAGNOSTICS=1
export VLLM_ALL2ALL_BACKEND
export VLLM_ENABLE_SEQUENCE_PARALLEL
export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap \
    --format=csv,noheader | tee "${EVIDENCE_ROOT}/gpu_before_load.csv"
gpu_memory_before_load_mib="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
        awk 'NR == 1 { print $1 }'
)"
record_metric gpu_memory_before_load_mib "${gpu_memory_before_load_mib}"

df -hT / /home /dev/shm | tee "${EVIDENCE_ROOT}/filesystem.txt"
printf 'write_probe=PASS\n' >"${EVIDENCE_ROOT}/write_probe.env"

command -v patch
native_vllm_package="$(
    python3 - <<'PY'
from pathlib import Path
import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"
native_vllm_version="$(
    python3 - <<'PY'
import vllm

print(vllm.__version__)
PY
)"
printf 'native_vllm_package=%s\nnative_vllm_version=%s\n' \
    "${native_vllm_package}" "${native_vllm_version}" |
    tee "${EVIDENCE_ROOT}/native_vllm_identity.env"
test "${native_vllm_version}" = "0.19.0.post20.dev26+gc820e5ae1"

cp -a "${native_vllm_package}" "${RUNTIME_REPO}/"

python3 - "${RUNTIME_REPO}" "${BASE_MANIFEST_GZ_PATH}" <<'PY'
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
checked = 0
with gzip.open(manifest, "rt") as stream:
    for raw in stream:
        kind, expected, relative = raw.rstrip("\n").split("\t", 2)
        path = root / relative
        if kind == "MISSING":
            if path.exists():
                raise SystemExit(f"expected missing base path: {path}")
        elif kind == "SHA256":
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise SystemExit(
                    f"base source mismatch: path={path} "
                    f"expected={expected} actual={actual}"
                )
        else:
            raise SystemExit(f"unknown base manifest kind: {kind}")
        checked += 1
print(f"base_changed_files_verified={checked}")
PY

gzip -dc "${PATCH_GZ_PATH}" |
    patch --batch --no-backup-if-mismatch -p1 -d "${RUNTIME_REPO}" |
    tee "${EVIDENCE_ROOT}/source_patch.log"

gzip -dc "${PINNED_MANIFEST_GZ_PATH}" |
    (
        cd "${RUNTIME_REPO}"
        sha256sum -c -
    ) >"${EVIDENCE_ROOT}/pinned_manifest_check.log"
manifest_verified="$(
    grep -c ': OK$' "${EVIDENCE_ROOT}/pinned_manifest_check.log"
)"
test "${manifest_verified}" = "2103"
record_metric pinned_manifest_files_verified "${manifest_verified}"

git -C "${RUNTIME_REPO}" init -q
mkdir -p "${RUNTIME_REPO}/.git/objects/pack"
cp "${IDENTITY_PACK_PATH}" "${RUNTIME_REPO}/.git/objects/pack/"
cp "${IDENTITY_INDEX_PATH}" "${RUNTIME_REPO}/.git/objects/pack/"
git -C "${RUNTIME_REPO}" update-ref refs/heads/pinned "${PINNED_COMMIT}"
git -C "${RUNTIME_REPO}" symbolic-ref HEAD refs/heads/pinned
test "$(git -C "${RUNTIME_REPO}" rev-parse HEAD)" = "${PINNED_COMMIT}"
test "$(git -C "${RUNTIME_REPO}" cat-file -t HEAD)" = "commit"
git -C "${RUNTIME_REPO}" rev-parse HEAD |
    tee "${EVIDENCE_ROOT}/runtime_git_head.txt"
git -C "${RUNTIME_REPO}" cat-file -p HEAD |
    tee "${EVIDENCE_ROOT}/runtime_git_commit.txt"
git -C "${RUNTIME_REPO}" ls-tree -r HEAD -- vllm |
    tee "${EVIDENCE_ROOT}/runtime_git_tree.txt" >/dev/null
test "$(wc -l <"${EVIDENCE_ROOT}/runtime_git_tree.txt")" = "2103"

python3 - \
    "${RUNTIME_REPO}" \
    "${EVIDENCE_ROOT}" \
    "${PINNED_GPU_MODEL_RUNNER_SHA256}" <<'PY'
from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
import shutil
import sys

runtime_root = Path(sys.argv[1])
evidence_root = Path(sys.argv[2])
expected_sha256 = sys.argv[3]
source_path = runtime_root / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
pinned_copy = evidence_root / "gpu_model_runner.pinned.py"
digest = hashlib.sha256()
with source_path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
actual_sha256 = digest.hexdigest()
if actual_sha256 != expected_sha256:
    raise RuntimeError(
        "unexpected pinned gpu_model_runner.py source: "
        f"expected={expected_sha256} actual={actual_sha256}"
    )
shutil.copyfile(source_path, pinned_copy)

source = source_path.read_text()
import_anchor = """from vllm.forward_context import (
    BatchDescriptor,
    is_local_deepep_decode_only,
"""
import_replacement = """from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_local_deepep_decode_only,
"""
if source.count(import_anchor) != 1:
    raise RuntimeError("unexpected pinned forward-context import block")
source = source.replace(import_anchor, import_replacement, 1)

forward_anchor = """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
"""
forward_replacement = forward_anchor + """
            if _dummy_diag_enabled():
                synchronize = current_platform.synchronize
                if synchronize is None:
                    raise RuntimeError(
                        "MODEL_FORWARD_COMPLETE requires platform synchronization"
                    )
                synchronize()
                diagnostic_context = get_forward_context()
                from vllm.v1.worker.deepep_diagnostics import distributed_rank

                logger.warning(
                    "MODEL_FORWARD_COMPLETE global_rank=%s dp_rank=%s "
                    "forward_seq=%s stage=%s batch=%s num_tokens=%s",
                    distributed_rank(),
                    diagnostic_context.diagnostic_dp_rank,
                    diagnostic_context.diagnostic_forward_sequence,
                    diagnostic_context.diagnostic_stage,
                    diagnostic_context.diagnostic_batch_type,
                    num_tokens_padded,
                )
"""
if source.count(forward_anchor) != 1:
    raise RuntimeError("unexpected pinned _model_forward call site")
source = source.replace(forward_anchor, forward_replacement, 1)
source_path.write_text(source)

patch = "".join(
    difflib.unified_diff(
        pinned_copy.read_text().splitlines(keepends=True),
        source.splitlines(keepends=True),
        fromfile="gpu_model_runner.py.pinned",
        tofile="gpu_model_runner.py.runtime",
    )
)
if not patch:
    raise RuntimeError("model forward completion overlay produced an empty diff")
(evidence_root / "model_forward_complete_overlay.patch").write_text(patch)
print(
    "model_forward_complete_overlay=APPLIED",
    f"pinned_sha256={expected_sha256}",
    f"path={source_path}",
)
PY
python3 -m py_compile \
    "${RUNTIME_REPO}/vllm/v1/worker/gpu_model_runner.py"
sha256sum \
    "${EVIDENCE_ROOT}/gpu_model_runner.pinned.py" \
    "${RUNTIME_REPO}/vllm/v1/worker/gpu_model_runner.py" \
    >"${EVIDENCE_ROOT}/model_forward_complete_overlay.sha256"

python3 - "${RUNTIME_REPO}" "${EVIDENCE_ROOT}" <<'PY'
from __future__ import annotations

import difflib
from pathlib import Path
import shutil
import sys

runtime_root = Path(sys.argv[1])
evidence_root = Path(sys.argv[2])
source_path = (
    runtime_root
    / "vllm"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "optimus_fp8_moe.py"
)
pinned_copy = evidence_root / "optimus_fp8_moe.pinned.py"
shutil.copyfile(source_path, pinned_copy)

source = source_path.read_text()
native_call = "torch.ops.Optimus.per_token_group_quant_fp8"
expected_call_count = 5
actual_call_count = source.count(native_call)
if actual_call_count != expected_call_count:
    raise RuntimeError(
        "unexpected pinned Optimus quant call count: "
        f"expected={expected_call_count} actual={actual_call_count}"
    )

import_anchor = "import torch\n\n"
if source.count(import_anchor) != 1:
    raise RuntimeError("unexpected pinned torch import anchor")
overlay = """import torch

from optimus_cutedsl.group_quant_fp8 import (
    per_token_group_quant_fp8 as _optimus_jit_group_quant_fp8,
)


def _optimus_jit_per_token_group_quant_fp8(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info_once(
        "Using owner-authorized Optimus JIT CuTe DSL activation quant overlay.",
        scope="local",
    )
    return _optimus_jit_group_quant_fp8(
        hidden_states,
        group_size=128,
        column_major_scales=False,
        use_ue8m0=True,
        scale_format="sm100_1d1d",
    )

"""
source = source.replace(import_anchor, overlay, 1)
source = source.replace(
    native_call,
    "_optimus_jit_per_token_group_quant_fp8",
)
if native_call in source:
    raise RuntimeError("native Optimus activation quant call remains after overlay")
if source.count("_optimus_jit_per_token_group_quant_fp8(") != (
    expected_call_count + 1
):
    raise RuntimeError("unexpected Optimus JIT overlay call count")
source_path.write_text(source)

patch = "".join(
    difflib.unified_diff(
        pinned_copy.read_text().splitlines(keepends=True),
        source.splitlines(keepends=True),
        fromfile="optimus_fp8_moe.py.pinned",
        tofile="optimus_fp8_moe.py.runtime",
    )
)
if not patch:
    raise RuntimeError("Optimus JIT quant overlay produced an empty diff")
(evidence_root / "optimus_jit_quant_overlay.patch").write_text(patch)
print(
    "optimus_jit_quant_overlay=APPLIED",
    f"replaced_calls={expected_call_count}",
    f"path={source_path}",
)
PY
python3 -m py_compile \
    "${RUNTIME_REPO}/vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py"
sha256sum \
    "${EVIDENCE_ROOT}/optimus_fp8_moe.pinned.py" \
    "${RUNTIME_REPO}/vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py" \
    >"${EVIDENCE_ROOT}/optimus_jit_quant_overlay.sha256"

python3 - "${RUNTIME_REPO}" "${EVIDENCE_ROOT}" <<'PY'
from __future__ import annotations

import difflib
from pathlib import Path
import shutil
import sys

runtime_root = Path(sys.argv[1])
evidence_root = Path(sys.argv[2])
source_path = (
    runtime_root
    / "vllm"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "deep_gemm_utils.py"
)
pinned_copy = evidence_root / "deep_gemm_utils.pinned.py"
shutil.copyfile(source_path, pinned_copy)

source = source_path.read_text()
old = "    BLOCK_D = min(hidden_size, 1024)\n"
new = (
    "    # Triton arange requires a power-of-two block that divides hidden_size.\n"
    "    BLOCK_D = min(1024, hidden_size & -hidden_size)\n"
)
if source.count(old) != 1:
    raise RuntimeError("unexpected pinned ep_gather BLOCK_D assignment")
source = source.replace(old, new, 1)
if "hidden_size & -hidden_size" not in source:
    raise RuntimeError("ep_gather block correction was not applied")
source_path.write_text(source)

patch = "".join(
    difflib.unified_diff(
        pinned_copy.read_text().splitlines(keepends=True),
        source.splitlines(keepends=True),
        fromfile="deep_gemm_utils.py.pinned",
        tofile="deep_gemm_utils.py.runtime",
    )
)
if not patch:
    raise RuntimeError("ep_gather block correction produced an empty diff")
(evidence_root / "ep_gather_block_overlay.patch").write_text(patch)
print(
    "ep_gather_block_overlay=APPLIED",
    "smoke_hidden=896:block=128",
    "target_hidden=3584:block=512",
    f"path={source_path}",
)
PY
python3 -m py_compile \
    "${RUNTIME_REPO}/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py"
sha256sum \
    "${EVIDENCE_ROOT}/deep_gemm_utils.pinned.py" \
    "${RUNTIME_REPO}/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py" \
    >"${EVIDENCE_ROOT}/ep_gather_block_overlay.sha256"

OPTIMUS_OVERLAY_ROOT="${BOOTSTRAP_ROOT}/optimus-fa4-3.23.24"
OPTIMUS_WHEEL_PATH="${OPTIMUS_OVERLAY_ROOT}/step_optimus-3.23.24.whl"
mkdir -p "${OPTIMUS_OVERLAY_ROOT}"
curl --fail --location --retry 3 \
    --output "${OPTIMUS_WHEEL_PATH}" "${OPTIMUS_WHEEL_URL}"
printf '%s  %s\n' "${OPTIMUS_WHEEL_SHA256}" "${OPTIMUS_WHEEL_PATH}" |
    sha256sum -c -

python3 - "${OPTIMUS_WHEEL_PATH}" "${OPTIMUS_OVERLAY_ROOT}" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import shutil
import sys

wheel_path = Path(sys.argv[1])
overlay_root = Path(sys.argv[2])
with ZipFile(wheel_path) as wheel:
    for name in wheel.namelist():
        if "/optimus/flash_attn_cute/" in name:
            relative = name.split("/optimus/", 1)[1]
            target = overlay_root / "optimus" / relative
        elif name.startswith("step_optimus-3.23.24.dist-info/"):
            target = overlay_root / name
        else:
            continue
        if name.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with wheel.open(name) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)

kernel_path = (
    overlay_root
    / "optimus"
    / "flash_attn_cute"
    / "sm100_hd512_fmha_forward.py"
)
source = kernel_path.read_text()
replacements = {
    "storage.tmem_holding_buf.ptr": "storage.tmem_holding_buf",
    "storage.tmem_dealloc_mbar.ptr": "storage.tmem_dealloc_mbar",
}
for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"unexpected Optimus 3.23.24 source for {old!r}: count={count}"
        )
    source = source.replace(old, new)
kernel_path.write_text(source)
print(f"patched_optimus_fa4={kernel_path}")

compat_path = overlay_root / "optimus" / "flash_attn_cute" / "__init__.py"
compat_source = compat_path.read_text()
import_anchor = "import sys\nimport types"
if compat_source.count(import_anchor) != 1:
    raise RuntimeError("unexpected Optimus flash_attn compatibility imports")
compat_source = compat_source.replace(
    import_anchor,
    "import importlib.machinery\nimport sys\nimport types",
)
spec_anchor = (
    '    elif not hasattr(flash_attn_pkg, "__path__"):\n'
    "        flash_attn_pkg.__path__ = []\n\n"
    '    sys.modules["flash_attn.cute"] = current_pkg'
)
if compat_source.count(spec_anchor) != 1:
    raise RuntimeError("unexpected Optimus flash_attn compatibility registration")
compat_source = compat_source.replace(
    spec_anchor,
    '    elif not hasattr(flash_attn_pkg, "__path__"):\n'
    "        flash_attn_pkg.__path__ = []\n\n"
    "    if flash_attn_pkg.__spec__ is None:\n"
    "        flash_attn_pkg.__spec__ = importlib.machinery.ModuleSpec(\n"
    '            "flash_attn", loader=None, is_package=True\n'
    "        )\n"
    "        flash_attn_pkg.__spec__.submodule_search_locations = (\n"
    "            flash_attn_pkg.__path__\n"
    "        )\n\n"
    '    sys.modules["flash_attn.cute"] = current_pkg',
)
compat_path.write_text(compat_source)
print(f"patched_optimus_flash_attn_spec={compat_path}")
PY

OPTIMUS_JIT_OVERLAY_ROOT="${BOOTSTRAP_ROOT}/optimus-jit-runtime"
mkdir -p "${OPTIMUS_JIT_OVERLAY_ROOT}"
OPTIMUS_TRITON_PACKAGE="$(
    python3 - <<'PY'
from pathlib import Path
import optimus_triton

print(Path(optimus_triton.__file__).resolve().parent)
PY
)"
cp -a "${OPTIMUS_TRITON_PACKAGE}" "${OPTIMUS_JIT_OVERLAY_ROOT}/"
python3 - "${OPTIMUS_JIT_OVERLAY_ROOT}" "${EVIDENCE_ROOT}" <<'PY'
from __future__ import annotations

import difflib
from pathlib import Path
import shutil
import sys

overlay_root = Path(sys.argv[1])
evidence_root = Path(sys.argv[2])
source_path = (
    overlay_root
    / "optimus_triton"
    / "deep_gemm_ep_gather_masked.py"
)
pinned_copy = evidence_root / "deep_gemm_ep_gather_masked.pinned.py"
shutil.copyfile(source_path, pinned_copy)

source = source_path.read_text()
old = """def _select_block_d(hidden_size: int) -> int:
    block_d = 1024 if hidden_size >= 1024 else hidden_size
    if hidden_size % block_d == 0:
        return block_d
    while block_d > 1 and hidden_size % block_d != 0:
        block_d //= 2
    if hidden_size % block_d != 0:
        raise ValueError(f"hidden_size={hidden_size} not divisible by BLOCK_D={block_d}")
    return block_d
"""
new = """def _select_block_d(hidden_size: int) -> int:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    return min(1024, hidden_size & -hidden_size)
"""
if source.count(old) != 1:
    raise RuntimeError("unexpected Optimus Triton block selector")
source = source.replace(old, new, 1)
source_path.write_text(source)

patch = "".join(
    difflib.unified_diff(
        pinned_copy.read_text().splitlines(keepends=True),
        source.splitlines(keepends=True),
        fromfile="deep_gemm_ep_gather_masked.py.pinned",
        tofile="deep_gemm_ep_gather_masked.py.runtime",
    )
)
if not patch:
    raise RuntimeError("Optimus Triton gather overlay produced an empty diff")
(evidence_root / "optimus_triton_gather_overlay.patch").write_text(patch)
print(
    "optimus_triton_gather_overlay=APPLIED",
    "smoke_hidden=896:block=128",
    "target_hidden=3584:block=512",
    f"path={source_path}",
)
PY
python3 -m py_compile \
    "${OPTIMUS_JIT_OVERLAY_ROOT}/optimus_triton/deep_gemm_ep_gather_masked.py"
sha256sum \
    "${EVIDENCE_ROOT}/deep_gemm_ep_gather_masked.pinned.py" \
    "${OPTIMUS_JIT_OVERLAY_ROOT}/optimus_triton/deep_gemm_ep_gather_masked.py" \
    >"${EVIDENCE_ROOT}/optimus_triton_gather_overlay.sha256"

cat >"${OPTIMUS_OVERLAY_ROOT}/sitecustomize.py" <<'PY'
from pathlib import Path
import importlib.metadata
import sys
import torch
import optimus
import optimus.lib as optimus_lib

overlay_root = Path(__file__).resolve().parent
overlay_optimus = overlay_root / "optimus"
if str(overlay_optimus) in optimus.__path__:
    optimus.__path__.remove(str(overlay_optimus))
optimus.__path__.insert(0, str(overlay_optimus))
optimus_lib._load_liboptimus()
from optimus.flash_attn_cute.interface import _flash_attn_fwd

for alias in [
    name
    for name in tuple(sys.modules)
    if name == "flash_attn" or name.startswith("flash_attn.cute")
]:
    sys.modules.pop(alias)
version = importlib.metadata.version("step-optimus")
if version != "3.23.24":
    raise ImportError(f"expected step-optimus==3.23.24, got {version}")
if not hasattr(torch.ops.Optimus, "RMSNorm_forward"):
    raise ImportError("image-native Optimus RMSNorm_forward is not registered")
print(
    "Optimus runtime ready:",
    f"version={version}",
    f"native={optimus_lib._LIB_OPTIMUS_FILE_PATH}",
    f"fa4_path={overlay_optimus / 'flash_attn_cute'}",
    f"fa4_forward={_flash_attn_fwd}",
    file=sys.stderr,
)
PY

export PYTHONPATH="${OPTIMUS_JIT_OVERLAY_ROOT}:${OPTIMUS_OVERLAY_ROOT}:${RUNTIME_REPO}"
python3 - "${RUNTIME_REPO}" > >(
    tee "${EVIDENCE_ROOT}/runtime_import_identity.log"
) <<'PY'
from pathlib import Path
import importlib.metadata
import inspect
import torch
import vllm
import vllm._C
from vllm.model_executor.models import step4pro
from vllm.model_executor.layers.fused_moe import optimus_fp8_moe
from vllm.v1.attention.backends import optimus_fa4

root = Path(__import__("sys").argv[1]).resolve()
paths = {
    "vllm_file": Path(vllm.__file__).resolve(),
    "vllm_C_file": Path(vllm._C.__file__).resolve(),
    "step4pro_file": Path(inspect.getsourcefile(step4pro)).resolve(),
    "optimus_fa4_file": Path(inspect.getsourcefile(optimus_fa4)).resolve(),
    "optimus_fp8_moe_file": Path(
        inspect.getsourcefile(optimus_fp8_moe)
    ).resolve(),
}
for key, path in paths.items():
    print(key, path)
for key in ("vllm_file", "step4pro_file", "optimus_fa4_file", "optimus_fp8_moe_file"):
    if root not in paths[key].parents:
        raise SystemExit(f"{key} is outside pinned runtime repo: {paths[key]}")
print("step4pro_class", step4pro.Step4ProForCausalLM)
print("step_optimus", importlib.metadata.version("step-optimus"))
print("rmsnorm_op", torch.ops.Optimus.RMSNorm_forward)
for package in ("deep_gemm", "torch"):
    print(package, importlib.metadata.version(package))
PY

sha256sum \
    "${RUNTIME_REPO}/vllm/model_executor/models/step4pro.py" \
    "${RUNTIME_REPO}/vllm/v1/attention/backends/optimus_fa4.py" \
    "${RUNTIME_REPO}/vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py" \
    "${RUNTIME_REPO}/vllm/v1/worker/gpu_model_runner.py" |
    tee "${EVIDENCE_ROOT}/runtime_key_files.sha256"

mkdir -p "${PROFILER_DIR}"
server_started_epoch="$(date +%s)"
record_metric server_started_epoch "${server_started_epoch}"
record_metric profiler_enabled "${ENABLE_PROFILER}"
DISTRIBUTED_ARGS=""
HEADLESS_MODE=0
HEADLESS_ARGS=""
if (( DATA_PARALLEL_SIZE > 1 )); then
    : "${MASTER_ADDR:?MASTER_ADDR is required for distributed smoke}"
    : "${NODE_RANK:?NODE_RANK is required for distributed smoke}"
    DATA_PARALLEL_START_RANK="$((NODE_RANK * DATA_PARALLEL_SIZE_LOCAL))"
    record_metric node_rank "${NODE_RANK}"
    record_metric data_parallel_start_rank "${DATA_PARALLEL_START_RANK}"
    record_metric data_parallel_size "${DATA_PARALLEL_SIZE}"
    record_metric data_parallel_size_local "${DATA_PARALLEL_SIZE_LOCAL}"
    if (( DATA_PARALLEL_START_RANK > 0 )); then
        HEADLESS_MODE=1
        HEADLESS_ARGS="--headless --api-server-count 0"
    fi
    record_metric headless_mode "${HEADLESS_MODE}"
    DISTRIBUTED_ARGS="--data-parallel-size ${DATA_PARALLEL_SIZE} \
--data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL} \
--data-parallel-start-rank ${DATA_PARALLEL_START_RANK} \
--data-parallel-address ${MASTER_ADDR} \
--data-parallel-rpc-port ${DATA_PARALLEL_RPC_PORT}"
fi
PROFILER_ARGS=""
if (( ENABLE_PROFILER == 1 )); then
    PROFILER_ARGS="--profiler-config.profiler=torch \
--profiler-config.torch_profiler_dir=${PROFILER_DIR} \
--profiler-config.torch_profiler_record_shapes=true \
--profiler-config.torch_profiler_with_stack=true \
--profiler-config.ignore_frontend=true \
--profiler-config.max_iterations=1"
fi
SERVER_COMMAND="vllm serve ${MODEL_PATH} \
--served-model-name ${MODELNAME} \
--port ${PORT_AUTO0} \
--load-format dummy \
--dtype bfloat16 \
--skip-tokenizer-init \
--tensor-parallel-size 1 \
${DISTRIBUTED_ARGS} \
--block-size 128 \
--enable-expert-parallel \
--all2all-backend ${VLLM_ALL2ALL_BACKEND} \
--max-model-len 8192 \
--max-num-seqs 8 \
--max-num-batched-tokens 8192 \
--gpu-memory-utilization 0.9 \
--enforce-eager \
${HEADLESS_ARGS} \
${PROFILER_ARGS}"
PORT_SERVING="${PORT_AUTO0}" python3 /usr/bin/signal_proxy.py \
    "${SERVER_COMMAND}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
record_metric server_pid "${SERVER_PID}"

if (( HEADLESS_MODE == 1 )); then
    wait_for_server_log_marker \
        "${ALLGATHER_REDUCESCATTER_CONFIG_MARKER}" \
        "allgather_reducescatter runtime configuration"
    wait_for_server_log_marker \
        "MODEL_FORWARD_COMPLETE.*batch=real" \
        "completed real distributed batch forward"
    validate_distributed_all2all_runtime 0
    grep -En \
        "Using .*All2AllManager|using allgather_reducescatter all2all backend|Auto-configured .*VLLM_ALL2ALL_BACKEND|FORWARD_CONTEXT.*batch=real|MODEL_FORWARD_COMPLETE.*batch=real|Model loading took|Traceback|ERROR|Broken pipe" \
        "${SERVER_LOG}" | tee "${EVIDENCE_ROOT}/provider_markers.log"
    assert_runtime_log_clean "${SERVER_LOG}"
    printf 'HEADLESS_RUNTIME=PASS\n' |
        tee "${EVIDENCE_ROOT}/remote_execution.env"
    printf 'DISTRIBUTED_RUNTIME_VALIDATION=PASS\n' |
        tee -a "${EVIDENCE_ROOT}/remote_execution.env"
    touch "${REMOTE_VALIDATION_READY_FILE}"
    hold_distributed_runtime_for_host_cleanup
fi

HEALTHY=0
for _ in $(seq 1 240); do
    if curl --fail --silent --show-error --max-time 3 \
        "http://127.0.0.1:${PORT_AUTO0}/health" >/dev/null; then
        HEALTHY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        cat "${SERVER_LOG}"
        exit 1
    fi
    sleep 2
done
test "${HEALTHY}" = "1"
health_ready_epoch="$(date +%s)"
health_ready_seconds="$((health_ready_epoch - server_started_epoch))"
record_metric health_ready_seconds "${health_ready_seconds}"

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap \
    --format=csv,noheader | tee "${EVIDENCE_ROOT}/gpu_after_load.csv"
gpu_memory_after_load_mib="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
        awk 'NR == 1 { print $1 }'
)"
record_metric gpu_memory_after_load_mib "${gpu_memory_after_load_mib}"

COMPLETIONS_REQUEST='{"model":"'"${MODELNAME}"'","prompt":[1,4,7,10,13,16,19,22],"max_tokens":4,"temperature":0,"top_p":1,"ignore_eos":true,"return_token_ids":true}'

validate_token_response() {
    local label="$1"
    local response="$2"
    python3 - "${label}" "${response}" <<'PY'
import json
import sys
from pathlib import Path

label, response_path = sys.argv[1], Path(sys.argv[2])
payload = json.loads(response_path.read_text())
choice = payload["choices"][0]
prompt_token_ids = choice.get("prompt_token_ids")
token_ids = choice.get("token_ids")
if prompt_token_ids != [1, 4, 7, 10, 13, 16, 19, 22]:
    raise SystemExit(
        f"{label} prompt token mismatch: {prompt_token_ids!r}"
    )
if not isinstance(token_ids, list) or len(token_ids) != 4:
    raise SystemExit(f"{label} expected 4 output token ids, got {token_ids!r}")
usage = payload.get("usage") or {}
prompt_tokens = int(usage.get("prompt_tokens", 0))
completion_tokens = int(usage.get("completion_tokens", 0))
if prompt_tokens != 8 or completion_tokens != 4:
    raise SystemExit(f"{label} unexpected token usage: {usage!r}")
print(f"{label}_TOKEN_IDS=PASS")
print(f"{label}_prompt_tokens={prompt_tokens}")
print(f"{label}_completion_tokens={completion_tokens}")
PY
}

if (( ENABLE_PROFILER == 1 )); then
    curl --fail --silent --show-error --max-time 30 \
        -X POST "http://127.0.0.1:${PORT_AUTO0}/start_profile"
fi
COMPLETIONS_RESPONSE="${EVIDENCE_ROOT}/completions_response.json"
request_latency_seconds="$(
    curl --fail --silent --show-error --max-time 600 \
        -o "${COMPLETIONS_RESPONSE}" -w '%{time_total}' \
        "http://127.0.0.1:${PORT_AUTO0}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "${COMPLETIONS_REQUEST}"
)"
record_metric request_latency_seconds "${request_latency_seconds}"
validate_token_response COMPLETIONS "${COMPLETIONS_RESPONSE}" |
    tee "${EVIDENCE_ROOT}/completions_validation.txt"
if (( ENABLE_PROFILER == 1 )); then
    curl --fail --silent --show-error --max-time 60 \
        -X POST "http://127.0.0.1:${PORT_AUTO0}/stop_profile"
fi

prompt_tokens="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["usage"]["prompt_tokens"])' \
        "${COMPLETIONS_RESPONSE}"
)"
completion_tokens="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["usage"]["completion_tokens"])' \
        "${COMPLETIONS_RESPONSE}"
)"
record_metric prompt_tokens "${prompt_tokens}"
record_metric completion_tokens "${completion_tokens}"

concurrent_started_ns="$(date +%s%N)"
REQUEST_PIDS=()
for index in 1 2 3 4; do
    (
        curl --fail --silent --show-error --max-time 600 \
            -o "${EVIDENCE_ROOT}/batch_${index}_response.json" \
            -w '%{time_total}\n' \
            "http://127.0.0.1:${PORT_AUTO0}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "${COMPLETIONS_REQUEST}" \
            >"${EVIDENCE_ROOT}/batch_${index}_latency_seconds.txt"
    ) &
    REQUEST_PIDS+=("$!")
done
for pid in "${REQUEST_PIDS[@]}"; do
    wait "${pid}"
done
concurrent_finished_ns="$(date +%s%N)"
concurrent_wall_seconds="$(
    python3 - "${concurrent_started_ns}" "${concurrent_finished_ns}" <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000_000)
PY
)"
record_metric concurrent_wall_seconds "${concurrent_wall_seconds}"
for index in 1 2 3 4; do
    validate_token_response "BATCH_${index}" \
        "${EVIDENCE_ROOT}/batch_${index}_response.json"
    record_metric "batch_${index}_request_latency_seconds" \
        "$(cat "${EVIDENCE_ROOT}/batch_${index}_latency_seconds.txt")"
done
echo "BATCH_TOKEN_IDS=PASS"

grep -En \
    "Resolved architecture: Step4ProForCausalLM|Loaded model info for class .*step4pro|Step4 Pro Optimus FA4 probe:|Optimus FA4 actual forward:|Using OPTIMUS_FP8 Fp8 MoE backend|OptimusFp8Experts uses legacy Optimus DeepGemm|Model loading took|Loading weights|VLLM_KV_CACHE_LAYOUT|Traceback|ERROR" \
    "${SERVER_LOG}" | tee "${EVIDENCE_ROOT}/provider_markers.log"

grep -q "Resolved architecture: Step4ProForCausalLM" "${SERVER_LOG}"
grep -q "Step4 Pro Optimus FA4 probe: available=True" "${SERVER_LOG}"
grep -q "Optimus FA4 actual forward:" "${SERVER_LOG}"
grep -q "Using OPTIMUS_FP8 Fp8 MoE backend" "${SERVER_LOG}"
grep -q "OptimusFp8Experts uses legacy Optimus DeepGemm" "${SERVER_LOG}"
if grep -q "Step4 Pro Optimus FA4 probe: available=False" "${SERVER_LOG}"; then
    echo "Full-MFA Optimus FA4 probe reported unavailable" >&2
    exit 1
fi
if (( DATA_PARALLEL_SIZE > 1 )); then
    validate_distributed_all2all_runtime 1
    grep -q "MODEL_FORWARD_COMPLETE.*batch=real" "${SERVER_LOG}"
fi
assert_runtime_log_clean "${SERVER_LOG}"

python3 - "${SERVER_LOG}" "${METRICS_FILE}" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors="replace")
metrics = Path(sys.argv[2])
patterns = {
    "model_loading_seconds": [
        r"Model loading took [0-9.]+ GiB(?: memory)? and ([0-9.]+) seconds",
        r"Loading model weights took ([0-9.]+) seconds",
    ],
}
with metrics.open("a") as stream:
    for key, candidates in patterns.items():
        for pattern in candidates:
            match = re.search(pattern, log)
            if match:
                stream.write(f"{key}={match.group(1)}\n")
                break
        else:
            raise SystemExit(f"missing numeric server metric: {key}")
PY

if (( ENABLE_PROFILER == 1 )); then
    for _ in $(seq 1 30); do
        if find "${PROFILER_DIR}" -type f -size +0c | grep -q .; then
            break
        fi
        sleep 1
    done
    find "${PROFILER_DIR}" -type f -size +0c -printf '%p %s\n' |
        tee "${EVIDENCE_ROOT}/profiler_files.txt"
    test -s "${EVIDENCE_ROOT}/profiler_files.txt"
fi

if (( DATA_PARALLEL_SIZE > 1 )); then
    printf 'DISTRIBUTED_RUNTIME_VALIDATION=PASS\n' |
        tee "${EVIDENCE_ROOT}/remote_execution.env"
    touch "${REMOTE_VALIDATION_READY_FILE}"
    hold_distributed_runtime_for_host_cleanup
else
    stop_server
    SERVER_PID=""
    printf 'ONE_GPU_REMOTE_EXECUTION=PASS\n' |
        tee "${EVIDENCE_ROOT}/remote_execution.env"
fi
