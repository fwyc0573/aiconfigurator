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
EVIDENCE_HOLD_SECONDS="${EVIDENCE_HOLD_SECONDS:-120}"
SOURCE_PROBE_SCRIPT="${BOOTSTRAP_ROOT}/remote_b300_source_probe.sh"
AIC_ROOT="${BOOTSTRAP_ROOT}/aic"
AIC_METADATA_ROOT="${AIC_ROOT}/python_metadata"
OUTPUT_ROOT="${EVIDENCE_ROOT}/dataset"
CHECKPOINT_ROOT="${EVIDENCE_ROOT}/checkpoint"
OPTIMUS_TRITON_OVERLAY_ROOT="${BOOTSTRAP_ROOT}/optimus-jit-runtime"
OPTIMUS_NATIVE_OVERLAY_ROOT="${BOOTSTRAP_ROOT}/optimus-native-runtime"
OPTIMUS_WHEEL_PATH="${OPTIMUS_NATIVE_OVERLAY_ROOT}/step_optimus-3.23.24.whl"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "MODE must be smoke or full: ${MODE}" >&2
    exit 1
fi

mkdir -p \
    "${EVIDENCE_ROOT}" \
    "${AIC_ROOT}" \
    "${AIC_METADATA_ROOT}" \
    "${OUTPUT_ROOT}" \
    "${CHECKPOINT_ROOT}" \
    "${OPTIMUS_TRITON_OVERLAY_ROOT}" \
    "${OPTIMUS_NATIVE_OVERLAY_ROOT}"
exec > >(tee -a "${EVIDENCE_ROOT}/remote_stdout.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if (( status == 0 )); then
        printf 'B300_OPTIMUS_MOE_COLLECTION=PASS\nmode=%s\n' "${MODE}" \
            | tee "${EVIDENCE_ROOT}/result.env"
    else
        printf 'B300_OPTIMUS_MOE_COLLECTION=FAIL\nmode=%s\nexit_code=%s\n' \
            "${MODE}" "${status}" | tee "${EVIDENCE_ROOT}/result.env"
    fi
    touch "${EVIDENCE_ROOT}/remote_result_ready"
    sleep "${EVIDENCE_HOLD_SECONDS}"
    exit "${status}"
}
trap finish EXIT

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64
export OPTIMUS_MUST_LOAD_LIB=1
export OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE=1
export VLLM_USE_DEEP_GEMM_E8M0=1
export VLLM_USE_OPTIMUS_MOE=1
export VLLM_OPTIMUS_MOE_MIN_CONTG_SIZE=6144
export VLLM_FUSED_MOE_CHUNK_SIZE=65536
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

curl --fail --location --retry 3 \
    --output "${OPTIMUS_WHEEL_PATH}" "${OPTIMUS_WHEEL_URL}"
printf '%s  %s\n' "${OPTIMUS_WHEEL_SHA256}" "${OPTIMUS_WHEEL_PATH}" \
    | sha256sum -c -
sha256sum "${OPTIMUS_WHEEL_PATH}" >"${EVIDENCE_ROOT}/optimus_wheel.sha256"

python3 - "${OPTIMUS_WHEEL_PATH}" "${OPTIMUS_NATIVE_OVERLAY_ROOT}" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import shutil
import sys

wheel_path = Path(sys.argv[1])
overlay_root = Path(sys.argv[2])
with ZipFile(wheel_path) as wheel:
    matched = 0
    for name in wheel.namelist():
        if not name.startswith("step_optimus-3.23.24.dist-info/"):
            continue
        matched += 1
        target = overlay_root / name
        if name.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with wheel.open(name) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
if matched == 0:
    raise RuntimeError("step-optimus 3.23.24 wheel has no dist-info payload")
print("optimus_metadata_overlay=APPLIED", f"files={matched}")
PY

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

OPTIMUS_TRITON_PACKAGE="$(
    python3 - <<'PY'
from pathlib import Path
import optimus_triton

print(Path(optimus_triton.__file__).resolve().parent)
PY
)"
cp -a "${OPTIMUS_TRITON_PACKAGE}" "${OPTIMUS_TRITON_OVERLAY_ROOT}/"
python3 - "${OPTIMUS_TRITON_OVERLAY_ROOT}" "${EVIDENCE_ROOT}" <<'PY'
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
    "target_hidden=3584:block=512",
    f"path={source_path}",
)
PY
python3 -m py_compile \
    "${OPTIMUS_TRITON_OVERLAY_ROOT}/optimus_triton/deep_gemm_ep_gather_masked.py"
sha256sum \
    "${EVIDENCE_ROOT}/deep_gemm_ep_gather_masked.pinned.py" \
    "${OPTIMUS_TRITON_OVERLAY_ROOT}/optimus_triton/deep_gemm_ep_gather_masked.py" \
    >"${EVIDENCE_ROOT}/optimus_triton_gather_overlay.sha256"

cat >"${OPTIMUS_NATIVE_OVERLAY_ROOT}/sitecustomize.py" <<'PY'
import importlib.metadata
import torch
import optimus.lib as optimus_lib

optimus_lib._load_liboptimus()
version = importlib.metadata.version("step-optimus")
if version != "3.23.24":
    raise ImportError(f"expected step-optimus==3.23.24, got {version}")
if not hasattr(torch.ops.Optimus, "RMSNorm_forward"):
    raise ImportError("image-native Optimus operators are not registered")
print(
    "Optimus native runtime ready:",
    f"version={version}",
    f"native={optimus_lib._LIB_OPTIMUS_FILE_PATH}",
)
PY

export PYTHONPATH="${OPTIMUS_NATIVE_OVERLAY_ROOT}:${OPTIMUS_TRITON_OVERLAY_ROOT}:${AIC_METADATA_ROOT}:${AIC_ROOT}/src:${AIC_ROOT}:${RUNTIME_REPO}"

python3 - "${RUNTIME_REPO}" "${AIC_ROOT}" <<'PY' \
    | tee "${EVIDENCE_ROOT}/runtime_import_identity.log"
from pathlib import Path
import importlib.metadata
import inspect
import sys

import aiconfigurator
import torch
import vllm
from collector.vllm import collect_step4_provider
from vllm.model_executor.layers.fused_moe.optimus_fp8_moe import (
    OptimusFp8Experts,
)

runtime_root = Path(sys.argv[1]).resolve()
aic_root = Path(sys.argv[2]).resolve()
paths = {
    "vllm": Path(vllm.__file__).resolve(),
    "optimus_fp8_moe": Path(inspect.getsourcefile(OptimusFp8Experts)).resolve(),
    "aiconfigurator": Path(aiconfigurator.__file__).resolve(),
    "collector": Path(inspect.getsourcefile(collect_step4_provider)).resolve(),
}
for key, path in paths.items():
    print(key, path)
if runtime_root not in paths["vllm"].parents:
    raise SystemExit(f"vLLM escaped pinned runtime root: {paths['vllm']}")
if runtime_root not in paths["optimus_fp8_moe"].parents:
    raise SystemExit(
        f"OptimusFp8Experts escaped pinned runtime root: {paths['optimus_fp8_moe']}"
    )
if aic_root not in paths["aiconfigurator"].parents:
    raise SystemExit(f"AIC escaped transferred source root: {paths['aiconfigurator']}")
if aic_root not in paths["collector"].parents:
    raise SystemExit(f"Collector escaped transferred source root: {paths['collector']}")
print("torch", torch.__version__, torch.version.cuda)
print("vllm", vllm.__version__)
step_optimus_version = importlib.metadata.version("step-optimus")
if step_optimus_version != "3.23.24":
    raise SystemExit(
        f"expected step-optimus==3.23.24, got {step_optimus_version}"
    )
if not hasattr(torch.ops.Optimus, "RMSNorm_forward"):
    raise SystemExit("image-native Optimus operators are not registered")
print("step_optimus", step_optimus_version)
print("capability", torch.cuda.get_device_capability())
print("provider_class", OptimusFp8Experts.__module__, OptimusFp8Experts.__name__)
required_ops = (
    "deepgemm_optimus_moe_masked_fp8",
    "deepgemm_optimus_moe_fp8",
)
missing_ops = [name for name in required_ops if not hasattr(torch.ops.vllm, name)]
if missing_ops:
    raise SystemExit(f"pinned vLLM Optimus ops are not registered: {missing_ops}")
print("provider_ops", ",".join(required_ops))
PY

cd "${OUTPUT_ROOT}"
if [[ "${MODE}" == "smoke" ]]; then
    python3 - <<'PY'
from collector.vllm.collect_step4_provider import run_step4_optimus_moe

common = {
    "provider": "optimus_fp8_moe",
    "hidden_size": 3584,
    "inter_size": 3584,
    "topk": 16,
    "num_experts": 896,
    "moe_tp_size": 1,
    "moe_dtype": "fp8_block",
    "activation": "situ_glu",
    "perf_filename": "step4_optimus_moe_perf.txt",
}

run_step4_optimus_moe(
    num_tokens=128,
    moe_ep_size=16,
    distribution="balanced",
    **common,
)
run_step4_optimus_moe(
    num_tokens=6144,
    moe_ep_size=16,
    distribution="power_law_1.2",
    **common,
)
run_step4_optimus_moe(
    num_tokens=12288,
    moe_ep_size=32,
    distribution="power_law_1.01",
    **common,
)
PY
    printf '{"mode":"smoke","completed_cases":3}\n' \
        >"${CHECKPOINT_ROOT}/representative_optimus_moe.json"
else
    python3 "${AIC_ROOT}/collector/collect.py" \
        --backend vllm \
        --ops step4_optimus_moe \
        --model-path stepfun-ai/Step4-Pro-Latest \
        --gpu b300_sxm \
        --checkpoint-dir "${CHECKPOINT_ROOT}" \
        --resume \
        --keep-csv \
        --profile
fi

test -s "${OUTPUT_ROOT}/step4_optimus_moe_perf.txt"
python3 - "${OUTPUT_ROOT}" "${MODE}" <<'PY'
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1])
mode = sys.argv[2]
with (root / "step4_optimus_moe_perf.txt").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

expected_count = 3 if mode == "smoke" else 174
if len(rows) != expected_count:
    raise RuntimeError(
        f"unexpected Step4 Optimus row count: expected={expected_count}, actual={len(rows)}"
    )
if {row["provider"] for row in rows} != {"optimus_fp8_moe"}:
    raise RuntimeError("Step4 Optimus rows contain an unexpected provider")
if {row["kernel_source"] for row in rows} != {"optimus_fp8_moe"}:
    raise RuntimeError("Step4 Optimus rows contain an unexpected kernel source")
if {row["moe_dtype"] for row in rows} != {"fp8_block"}:
    raise RuntimeError("Step4 Optimus rows contain an unexpected dtype")
if {row["activation"] for row in rows} != {"situ_glu"}:
    raise RuntimeError("Step4 Optimus rows contain an unexpected activation")
expected_modes = {
    "deepgemm_optimus_moe_masked_fp8": ("cuda_graph", True),
    "deepgemm_optimus_moe_fp8": ("eager", False),
}
observed_variants = set()
observed_modes = set()
for row in rows:
    expected_variant = (
        "deepgemm_optimus_moe_masked_fp8"
        if int(row["local_num_tokens"]) < 6144
        else "deepgemm_optimus_moe_fp8"
    )
    if row["kernel_variant"] != expected_variant:
        raise RuntimeError(
            "Step4 Optimus row selected the wrong pinned kernel: "
            f"local_num_tokens={row['local_num_tokens']} "
            f"expected={expected_variant} actual={row['kernel_variant']}"
        )
    expected_mode, expected_graph = expected_modes[expected_variant]
    if row["execution_mode"] != expected_mode:
        raise RuntimeError(
            "Step4 Optimus row selected the wrong execution mode: "
            f"kernel={expected_variant} expected={expected_mode} "
            f"actual={row['execution_mode']}"
        )
    raw_graph = row["used_cuda_graph"].strip().lower()
    if raw_graph not in {"true", "false"}:
        raise RuntimeError(
            f"Step4 Optimus row has invalid used_cuda_graph={row['used_cuda_graph']!r}"
        )
    used_cuda_graph = raw_graph == "true"
    if used_cuda_graph is not expected_graph:
        raise RuntimeError(
            "Step4 Optimus row used the wrong CUDA graph mode: "
            f"kernel={expected_variant} expected={expected_graph} "
            f"actual={used_cuda_graph}"
        )
    observed_variants.add(expected_variant)
    observed_modes.add(expected_mode)
if observed_variants != set(expected_modes):
    raise RuntimeError(
        "Step4 Optimus dataset does not cover both pinned kernel variants: "
        f"actual={sorted(observed_variants)}"
    )
if mode == "full":
    if {int(row["moe_ep_size"]) for row in rows} != {16, 32}:
        raise RuntimeError("Full Step4 Optimus rows do not cover EP16 and EP32")
    if {row["distribution"] for row in rows} != {
        "balanced",
        "power_law_1.01",
        "power_law_1.2",
    }:
        raise RuntimeError("Full Step4 Optimus rows do not cover routing distributions")
print(
    "step4_optimus_moe_perf.txt",
    "rows",
    len(rows),
    "ep_sizes",
    sorted({int(row["moe_ep_size"]) for row in rows}),
    "distributions",
    sorted({row["distribution"] for row in rows}),
    "kernel_variants",
    sorted(observed_variants),
    "execution_modes",
    sorted(observed_modes),
)
PY

python3 - "${OUTPUT_ROOT}" "${EVIDENCE_ROOT}" <<'PY'
from pathlib import Path
import csv
import json
import shutil
import sys

import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import MoE
from aiconfigurator.sdk.perf_database import PerfDatabase

dataset_root = Path(sys.argv[1])
evidence_root = Path(sys.argv[2])
systems_root = evidence_root / "consumer_systems"
data_dir = systems_root / "data" / "vllm" / "0.19.0"
data_dir.mkdir(parents=True, exist_ok=True)
system_spec = yaml.safe_load(
    Path("/home")
    .joinpath(
        Path(__import__("aiconfigurator").__file__).resolve().parent,
        "systems",
        "b300_sxm.yaml",
    )
    .read_text()
)
system_spec["data_dir"] = "data"
(systems_root / "b300_sxm.yaml").write_text(yaml.safe_dump(system_spec))
shutil.copyfile(
    dataset_root / "step4_optimus_moe_perf.txt",
    data_dir / "step4_optimus_moe_perf.txt",
)
database = PerfDatabase("b300_sxm", "vllm", "0.19.0", str(systems_root))

with (dataset_root / "step4_optimus_moe_perf.txt").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

queries = []
for row in rows:
    operation = MoE(
        "experts",
        1.0,
        int(row["hidden_size"]),
        int(row["inter_size"]),
        int(row["topk"]),
        int(row["num_experts"]),
        int(row["moe_tp_size"]),
        int(row["moe_ep_size"]),
        common.MoEQuantMode[row["moe_dtype"]],
        row["distribution"],
        1,
        provider=row["provider"],
        activation=row["activation"],
    )
    result = operation.query(database, x=int(row["num_tokens"]))
    measured = float(row["latency"])
    queried = float(result)
    if queried != measured or result.source != "silicon":
        raise RuntimeError(
            "Step4 Optimus consumer mismatch: "
            f"measured={measured}, queried={queried}, source={result.source}"
        )
    queries.append(
        {
            "num_tokens": int(row["num_tokens"]),
            "moe_ep_size": int(row["moe_ep_size"]),
            "distribution": row["distribution"],
            "measured_latency_ms": measured,
            "queried_latency_ms": queried,
            "source": result.source,
        }
    )

payload = {"status": "PASS", "rows": len(rows), "queries": queries}
(evidence_root / "consumer_validation.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print("OPTIMUS_MOE_CONSUMER=PASS", "rows", len(rows))
PY

sha256sum \
    "${OUTPUT_ROOT}/step4_optimus_moe_perf.txt" \
    >"${EVIDENCE_ROOT}/dataset.sha256"
