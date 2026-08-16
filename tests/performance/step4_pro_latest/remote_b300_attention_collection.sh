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
: "${OPTIMUS_WHEEL_URL:?OPTIMUS_WHEEL_URL is required}"
: "${OPTIMUS_WHEEL_SHA256:?OPTIMUS_WHEEL_SHA256 is required}"

MODE="${MODE:-smoke}"
COLLECTION_SUITE="${COLLECTION_SUITE:-attention}"
PROVIDER_CORE_SLICE="${PROVIDER_CORE_SLICE:-all}"
PROVIDER_CORE_SMOKE_EVIDENCE="${PROVIDER_CORE_SMOKE_EVIDENCE:-}"
PROVIDER_CORE_SMOKE_EVIDENCE_SHA256="${PROVIDER_CORE_SMOKE_EVIDENCE_SHA256:-}"
QKNORM_ROPE_SOURCE_SHA256="5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783"
FULL_CONTEXT_SMOKE_TOKENS="${FULL_CONTEXT_SMOKE_TOKENS:-512}"
FULL_CONTEXT_SMOKE_TOTAL_TOKENS="${FULL_CONTEXT_SMOKE_TOTAL_TOKENS:-${FULL_CONTEXT_SMOKE_TOKENS}}"
SWA_CONTEXT_SMOKE_TOKENS="${SWA_CONTEXT_SMOKE_TOKENS:-512}"
SWA_CONTEXT_SMOKE_TOTAL_TOKENS="${SWA_CONTEXT_SMOKE_TOTAL_TOKENS:-${SWA_CONTEXT_SMOKE_TOKENS}}"
REMOTE_RESULT_MARKER="${REMOTE_RESULT_MARKER:-B300_ATTENTION_COLLECTION}"
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-120}"
SOURCE_PROBE_SCRIPT="${BOOTSTRAP_ROOT}/remote_b300_source_probe.sh"
AIC_ROOT="${BOOTSTRAP_ROOT}/aic"
AIC_METADATA_ROOT="${AIC_ROOT}/python_metadata"
OUTPUT_ROOT="${EVIDENCE_ROOT}/dataset"
CHECKPOINT_ROOT="${EVIDENCE_ROOT}/checkpoint"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "MODE must be smoke or full: ${MODE}" >&2
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
if [[ "${COLLECTION_SUITE}" != "attention" && "${COLLECTION_SUITE}" != "provider_core" ]]; then
    echo "COLLECTION_SUITE must be attention or provider_core: ${COLLECTION_SUITE}" >&2
    exit 1
fi
if [[ "${COLLECTION_SUITE}" == "provider_core" \
    && "${PROVIDER_CORE_SLICE}" == "all" ]]; then
    echo "provider_core requires an explicit non-all PROVIDER_CORE_SLICE" >&2
    exit 1
fi
if [[ "${COLLECTION_SUITE}" == "provider_core" \
    && "${PROVIDER_CORE_SLICE}" != "grouped_router" \
    && "${PROVIDER_CORE_SLICE}" != "qkv_full" \
    && "${PROVIDER_CORE_SLICE}" != "qkv_swa" ]]; then
    echo "PROVIDER_CORE_SLICE must be grouped_router, qkv_full, or qkv_swa: ${PROVIDER_CORE_SLICE}" >&2
    exit 1
fi
if [[ "${COLLECTION_SUITE}" == "attention" && "${PROVIDER_CORE_SLICE}" != "all" ]]; then
    echo "PROVIDER_CORE_SLICE applies only to provider_core: ${PROVIDER_CORE_SLICE}" >&2
    exit 1
fi
if [[ "${REMOTE_RESULT_MARKER}" != "B300_ATTENTION_COLLECTION" \
    && "${REMOTE_RESULT_MARKER}" != "B300_STEP4_PROVIDER_CORE_COLLECTION" ]]; then
    echo "Unsupported remote result marker: ${REMOTE_RESULT_MARKER}" >&2
    exit 1
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
    python3 - "${evidence_path}" "${expected_slice}" <<'PY'
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

mkdir -p \
    "${EVIDENCE_ROOT}" \
    "${AIC_ROOT}" \
    "${AIC_METADATA_ROOT}" \
    "${OUTPUT_ROOT}" \
    "${CHECKPOINT_ROOT}"
exec > >(tee -a "${EVIDENCE_ROOT}/remote_stdout.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if (( status == 0 )); then
        printf '%s=PASS\nmode=%s\nsuite=%s\n' \
            "${REMOTE_RESULT_MARKER}" "${MODE}" "${COLLECTION_SUITE}" \
            | tee "${EVIDENCE_ROOT}/result.env"
    else
        printf '%s=FAIL\nmode=%s\nsuite=%s\nexit_code=%s\n' \
            "${REMOTE_RESULT_MARKER}" "${MODE}" "${COLLECTION_SUITE}" "${status}" \
            | tee "${EVIDENCE_ROOT}/result.env"
    fi
    touch "${EVIDENCE_ROOT}/remote_result_ready"
    sleep "${EVIDENCE_HOLD_SECONDS}"
    exit "${status}"
}
trap finish EXIT

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64
export OPTIMUS_MUST_LOAD_LIB=1
export VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1

nvidia-smi \
    --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap \
    --format=csv,noheader | tee "${EVIDENCE_ROOT}/gpu_identity.csv"

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
    "${SOURCE_PROBE_SCRIPT}"

tar xf "${AIC_PAYLOAD_PATH}" -C "${AIC_ROOT}"
tar xf "${AIC_METADATA_PATH}" -C "${AIC_METADATA_ROOT}"

OPTIMUS_OVERLAY_ROOT="${BOOTSTRAP_ROOT}/optimus-fa4-3.23.24"
OPTIMUS_WHEEL_PATH="${OPTIMUS_OVERLAY_ROOT}/step_optimus-3.23.24.whl"
mkdir -p "${OPTIMUS_OVERLAY_ROOT}"
curl --fail --location --retry 3 \
    --output "${OPTIMUS_WHEEL_PATH}" "${OPTIMUS_WHEEL_URL}"
printf '%s  %s\n' "${OPTIMUS_WHEEL_SHA256}" "${OPTIMUS_WHEEL_PATH}" \
    | sha256sum -c -

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
    if source.count(old) != 1:
        raise RuntimeError(f"unexpected Optimus 3.23.24 source: {old!r}")
    source = source.replace(old, new)
kernel_path.write_text(source)

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
print(f"OPTIMUS_FA4_OVERLAY={kernel_path}")
PY

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
)
PY

cat >"${OPTIMUS_OVERLAY_ROOT}/step4_swa_qkv_runtime_overlay.py" <<'PY'
from pathlib import Path
import hashlib
import inspect
import os

if os.environ.get("PROVIDER_CORE_SLICE") == "qkv_swa":
    import cutlass
    import optimus_cutedsl.qknorm_rope as qknorm_rope

    expected_source_sha256 = os.environ["QKNORM_ROPE_SOURCE_SHA256"]
    qknorm_rope_source = Path(inspect.getsourcefile(qknorm_rope)).resolve()
    actual_source_sha256 = hashlib.sha256(qknorm_rope_source.read_bytes()).hexdigest()
    if actual_source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "Unexpected image-native QKNorm source: "
            f"expected_sha256={expected_source_sha256}, "
            f"actual_sha256={actual_source_sha256}, path={qknorm_rope_source}"
        )

    annotations = qknorm_rope.FusedQKNormRope.kernel.__annotations__
    before = {}
    for annotation_name in ("reload_from", "delay_w_load"):
        before[annotation_name] = annotations.get(annotation_name)
        if before[annotation_name] != "cutlass.Constexpr":
            raise RuntimeError(
                "Unexpected postponed QKNorm annotation: "
                f"name={annotation_name}, value={before[annotation_name]!r}"
            )
        annotations[annotation_name] = cutlass.Constexpr
    if any(
        annotations[annotation_name] is not cutlass.Constexpr
        for annotation_name in before
    ):
        raise RuntimeError("SWA QKV annotation overlay did not install Constexpr")
    print(
        "SWA_QKV_ANNOTATION_OVERLAY=PASS",
        f"source={qknorm_rope_source}",
        f"source_sha256={actual_source_sha256}",
        f"before={before}",
        "after=cutlass.Constexpr",
    )
PY

export QKNORM_ROPE_SOURCE_SHA256
export PYTHONPATH="${OPTIMUS_OVERLAY_ROOT}:${AIC_METADATA_ROOT}:${AIC_ROOT}/src:${AIC_ROOT}:${RUNTIME_REPO}"

python3 - "${RUNTIME_REPO}" "${AIC_ROOT}" <<'PY' \
    | tee "${EVIDENCE_ROOT}/runtime_import_identity.log"
from pathlib import Path
import hashlib
import importlib.metadata
import inspect
import sys

import aiconfigurator
import cutlass.base_dsl.dsl as cutlass_dsl
import optimus_cutedsl.qknorm_rope as qknorm_rope
import torch
import vllm
import step4_swa_qkv_runtime_overlay
from collector.vllm import collect_step4_provider
from vllm.v1.attention.backends import flash_attn, optimus_fa4

runtime_root = Path(sys.argv[1]).resolve()
aic_root = Path(sys.argv[2]).resolve()
paths = {
    "vllm": Path(vllm.__file__).resolve(),
    "optimus_fa4": Path(inspect.getsourcefile(optimus_fa4)).resolve(),
    "flash_attn": Path(inspect.getsourcefile(flash_attn)).resolve(),
    "aiconfigurator": Path(aiconfigurator.__file__).resolve(),
    "collector": Path(inspect.getsourcefile(collect_step4_provider)).resolve(),
}
for key, path in paths.items():
    print(key, path)
if runtime_root not in paths["vllm"].parents:
    raise SystemExit(f"vLLM escaped pinned runtime root: {paths['vllm']}")
if runtime_root not in paths["optimus_fa4"].parents:
    raise SystemExit(f"Optimus adapter escaped pinned runtime root: {paths['optimus_fa4']}")
if aic_root not in paths["aiconfigurator"].parents:
    raise SystemExit(f"AIC escaped transferred source root: {paths['aiconfigurator']}")
if aic_root not in paths["collector"].parents:
    raise SystemExit(f"Collector escaped transferred source root: {paths['collector']}")
print("torch", torch.__version__, torch.version.cuda)
print("vllm", vllm.__version__)
print("step_optimus", importlib.metadata.version("step-optimus"))
print("optimus_jit", importlib.metadata.version("optimus-jit"))
print("nvidia_cutlass_dsl", importlib.metadata.version("nvidia-cutlass-dsl"))
qknorm_rope_source = Path(inspect.getsourcefile(qknorm_rope)).resolve()
cutlass_dsl_source = Path(inspect.getsourcefile(cutlass_dsl)).resolve()
print("qknorm_rope_source", qknorm_rope_source)
print(
    "qknorm_rope_source_sha256",
    hashlib.sha256(qknorm_rope_source.read_bytes()).hexdigest(),
)
print("cutlass_dsl_source", cutlass_dsl_source)
print(
    "cutlass_dsl_source_sha256",
    hashlib.sha256(cutlass_dsl_source.read_bytes()).hexdigest(),
)
print(
    "qknorm_kernel_annotations",
    inspect.get_annotations(qknorm_rope.FusedQKNormRope.kernel, eval_str=False),
)
print("capability", torch.cuda.get_device_capability())
PY

cd "${OUTPUT_ROOT}"
if [[ "${COLLECTION_SUITE}" == "attention" && "${MODE}" == "smoke" ]]; then
    python3 - <<'PY'
import os

from collector.vllm.collect_step4_provider import (
    run_step4_context_attention,
    run_step4_generation_attention,
)

common = {
    "kv_cache_dtype": "bfloat16",
    "attn_dtype": "bfloat16",
    "page_size": 128,
    "physical_page_bytes": 524288,
    "kv_cache_layout": "NHD",
}
full_context_tokens = int(os.environ["FULL_CONTEXT_SMOKE_TOKENS"])
full_total_context_tokens = int(os.environ["FULL_CONTEXT_SMOKE_TOTAL_TOKENS"])
swa_context_tokens = int(os.environ["SWA_CONTEXT_SMOKE_TOKENS"])
swa_total_context_tokens = int(os.environ["SWA_CONTEXT_SMOKE_TOTAL_TOKENS"])

run_step4_context_attention(
    provider="optimus_fa4",
    batch_size=1,
    query_tokens=full_context_tokens,
    total_context_tokens=full_total_context_tokens,
    num_heads=64,
    num_kv_heads=1,
    head_dim=512,
    window_size=0,
    kv_storage_alias=True,
    kv_block_stride_bytes=524288,
    perf_filename="step4_context_attention_perf.txt",
    **common,
)
run_step4_context_attention(
    provider="vllm_native_sliding_gqa",
    batch_size=1,
    query_tokens=swa_context_tokens,
    total_context_tokens=swa_total_context_tokens,
    num_heads=128,
    num_kv_heads=8,
    head_dim=128,
    window_size=512,
    kv_storage_alias=False,
    kv_block_stride_bytes=262144,
    perf_filename="step4_context_attention_perf.txt",
    **common,
)
run_step4_generation_attention(
    provider="optimus_fa4",
    batch_size=1,
    context_tokens=2048,
    num_heads=64,
    num_kv_heads=1,
    head_dim=512,
    window_size=0,
    kv_storage_alias=True,
    kv_block_stride_bytes=524288,
    perf_filename="step4_generation_attention_perf.txt",
    **common,
)
run_step4_generation_attention(
    provider="vllm_native_sliding_gqa",
    batch_size=1,
    context_tokens=2048,
    num_heads=128,
    num_kv_heads=8,
    head_dim=128,
    window_size=512,
    kv_storage_alias=False,
    kv_block_stride_bytes=262144,
    perf_filename="step4_generation_attention_perf.txt",
    **common,
)
PY
    printf '{"mode":"smoke","completed_cases":4}\n' \
        >"${CHECKPOINT_ROOT}/representative_attention.json"
elif [[ "${COLLECTION_SUITE}" == "attention" ]]; then
    python3 "${AIC_ROOT}/collector/collect.py" \
        --backend vllm \
        --ops step4_context_attention step4_generation_attention \
        --model-path stepfun-ai/Step4-Pro-Latest \
        --gpu b300_sxm \
        --checkpoint-dir "${CHECKPOINT_ROOT}" \
        --resume \
        --keep-csv \
        --profile
elif [[ "${MODE}" == "smoke" ]]; then
    PROVIDER_CORE_SLICE="${PROVIDER_CORE_SLICE}" python3 - <<'PY'
import os

import step4_swa_qkv_runtime_overlay
from collector.vllm.collect_step4_provider import (
    FULL_K_NORM_ROPE_PROVIDER,
    SWA_QKV_NORM_ROPE_PROVIDER,
    run_step4_fp32_output_gemm,
    run_step4_grouped_gemm,
    run_step4_qkv_norm_rope,
)

provider_core_slice = os.environ["PROVIDER_CORE_SLICE"]
if provider_core_slice == "grouped_router":
    run_step4_grouped_gemm(
        provider="vllm_step4pro_torch_einsum",
        groups=8,
        num_tokens=1,
        n=1024,
        k=4096,
        quant_mode="bfloat16",
        perf_filename="step4_grouped_gemm_perf.txt",
    )
    run_step4_fp32_output_gemm(
        provider="vllm.optimus_matmul_fp32",
        num_tokens=1,
        n=896,
        k=7168,
        weight_dtype="bfloat16",
        output_dtype="float32",
        perf_filename="step4_fp32_output_gemm_perf.txt",
    )
if provider_core_slice == "qkv_full":
    run_step4_qkv_norm_rope(
        provider=FULL_K_NORM_ROPE_PROVIDER,
        num_tokens=1,
        normalized_tensors="k",
        q_heads=64,
        kv_heads=1,
        head_dim=512,
        perf_filename="step4_qkv_norm_rope_perf.txt",
    )
if provider_core_slice == "qkv_swa":
    run_step4_qkv_norm_rope(
        provider=SWA_QKV_NORM_ROPE_PROVIDER,
        num_tokens=1,
        normalized_tensors="q+k+v",
        q_heads=128,
        kv_heads=8,
        head_dim=128,
        perf_filename="step4_qkv_norm_rope_perf.txt",
    )
PY
    completed_cases=1
    if [[ "${PROVIDER_CORE_SLICE}" == "grouped_router" ]]; then
        completed_cases=2
    fi
    printf '{"mode":"smoke","suite":"provider_core","slice":"%s","completed_cases":%s}\n' \
        "${PROVIDER_CORE_SLICE}" "${completed_cases}" \
        >"${CHECKPOINT_ROOT}/representative_provider_core.json"
elif [[ "${PROVIDER_CORE_SLICE}" == "grouped_router" ]]; then
    python3 "${AIC_ROOT}/collector/collect.py" \
        --backend vllm \
        --ops step4_grouped_gemm step4_fp32_output_gemm \
        --model-path stepfun-ai/Step4-Pro-Latest \
        --gpu b300_sxm \
        --checkpoint-dir "${CHECKPOINT_ROOT}" \
        --resume \
        --keep-csv \
        --profile
elif [[ "${PROVIDER_CORE_SLICE}" == "qkv_full" ]]; then
    python3 - <<'PY'
from collector.vllm.collect_step4_provider import (
    FULL_K_NORM_ROPE_PROVIDER,
    get_step4_qkv_norm_rope_test_cases,
    run_step4_qkv_norm_rope,
)

cases = [
    case
    for case in get_step4_qkv_norm_rope_test_cases()
    if case[0] == FULL_K_NORM_ROPE_PROVIDER
]
if len(cases) != 75 or len({tuple(case) for case in cases}) != 75:
    raise RuntimeError(
        "Expected 75 unique Full-MFA QKV cases, "
        f"got total={len(cases)}, unique={len({tuple(case) for case in cases})}"
    )
for case in cases:
    run_step4_qkv_norm_rope(
        *case,
        perf_filename="step4_qkv_norm_rope_perf.txt",
    )
PY
elif [[ "${PROVIDER_CORE_SLICE}" == "qkv_swa" ]]; then
    python3 - <<'PY'
import step4_swa_qkv_runtime_overlay
from collector.vllm.collect_step4_provider import (
    SWA_QKV_NORM_ROPE_PROVIDER,
    get_step4_qkv_norm_rope_test_cases,
    run_step4_qkv_norm_rope,
)

cases = [
    case
    for case in get_step4_qkv_norm_rope_test_cases()
    if case[0] == SWA_QKV_NORM_ROPE_PROVIDER
]
if len(cases) != 75 or len({tuple(case) for case in cases}) != 75:
    raise RuntimeError(
        "Expected 75 unique SWA QKV cases, "
        f"got total={len(cases)}, unique={len({tuple(case) for case in cases})}"
    )
for case in cases:
    run_step4_qkv_norm_rope(
        *case,
        perf_filename="step4_qkv_norm_rope_perf.txt",
    )
PY
fi

if [[ "${COLLECTION_SUITE}" == "attention" ]]; then
    test -s "${OUTPUT_ROOT}/step4_context_attention_perf.txt"
    test -s "${OUTPUT_ROOT}/step4_generation_attention_perf.txt"
    python3 - "${OUTPUT_ROOT}" <<'PY'
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1])
expected = {
    "step4_context_attention_perf.txt": {"optimus_fa4", "vllm_native_sliding_gqa"},
    "step4_generation_attention_perf.txt": {"optimus_fa4", "vllm_native_sliding_gqa"},
}
for filename, expected_providers in expected.items():
    with (root / filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    providers = {row["provider"] for row in rows}
    if not expected_providers.issubset(providers):
        raise RuntimeError(
            f"{filename} missing providers: expected={expected_providers}, actual={providers}"
        )
    print(filename, "rows", len(rows), "providers", sorted(providers))
PY

    sha256sum \
        "${OUTPUT_ROOT}/step4_context_attention_perf.txt" \
        "${OUTPUT_ROOT}/step4_generation_attention_perf.txt" \
        >"${EVIDENCE_ROOT}/dataset.sha256"
else
    selected_files=()
    if [[ "${PROVIDER_CORE_SLICE}" != "qkv_full" \
        && "${PROVIDER_CORE_SLICE}" != "qkv_swa" ]]; then
        selected_files+=(step4_grouped_gemm_perf.txt)
        selected_files+=(step4_fp32_output_gemm_perf.txt)
    fi
    if [[ "${PROVIDER_CORE_SLICE}" == "qkv_full" \
        || "${PROVIDER_CORE_SLICE}" == "qkv_swa" ]]; then
        selected_files+=(step4_qkv_norm_rope_perf.txt)
    fi
    for filename in "${selected_files[@]}"; do
        test -s "${OUTPUT_ROOT}/${filename}"
    done
    python3 - "${OUTPUT_ROOT}" "${MODE}" "${PROVIDER_CORE_SLICE}" <<'PY'
from pathlib import Path
import csv
import math
import sys

root = Path(sys.argv[1])
mode = sys.argv[2]
provider_core_slice = sys.argv[3]
expected = {}
if provider_core_slice not in {"qkv_full", "qkv_swa"}:
    expected.update(
        {
            "step4_grouped_gemm_perf.txt": (
                1 if mode == "smoke" else 75,
                {"vllm_step4pro_torch_einsum"},
                ("provider", "groups", "num_tokens", "n", "k", "quant_mode"),
            ),
            "step4_fp32_output_gemm_perf.txt": (
                1 if mode == "smoke" else 75,
                {"vllm.optimus_matmul_fp32"},
                (
                    "provider",
                    "num_tokens",
                    "n",
                    "k",
                    "weight_dtype",
                    "output_dtype",
                ),
            ),
        }
    )
if provider_core_slice == "qkv_full":
    expected["step4_qkv_norm_rope_perf.txt"] = (
        1 if mode == "smoke" else 75,
        {"vllm_step4pro_k_norm_rope"},
        (
            "provider",
            "num_tokens",
            "normalized_tensors",
            "q_heads",
            "kv_heads",
            "head_dim",
        ),
    )
elif provider_core_slice == "qkv_swa":
    expected["step4_qkv_norm_rope_perf.txt"] = (
        1 if mode == "smoke" else 75,
        {"vllm_step4pro_qkv_norm_rope"},
        (
            "provider",
            "num_tokens",
            "normalized_tensors",
            "q_heads",
            "kv_heads",
            "head_dim",
        ),
    )
for filename, (expected_rows, expected_providers, key_fields) in expected.items():
    with (root / filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{filename} row count mismatch: expected={expected_rows}, actual={len(rows)}"
        )
    providers = {row["provider"] for row in rows}
    if providers != expected_providers:
        raise RuntimeError(
            f"{filename} provider mismatch: expected={expected_providers}, actual={providers}"
        )
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"{filename} contains duplicate physical keys")
    if any(
        row["version"] != "0.19.0.post20.dev26+gc820e5ae1"
        or row["device"] != "NVIDIA B300 SXM6 AC"
        or not math.isfinite(float(row["latency"]))
        or float(row["latency"]) <= 0
        for row in rows
    ):
        raise RuntimeError(f"{filename} contains invalid runtime identity or latency")
    print(filename, "rows", len(rows), "providers", sorted(providers))
PY

    dataset_paths=()
    for filename in "${selected_files[@]}"; do
        dataset_paths+=("${OUTPUT_ROOT}/${filename}")
    done
    sha256sum "${dataset_paths[@]}" >"${EVIDENCE_ROOT}/dataset.sha256"
fi
