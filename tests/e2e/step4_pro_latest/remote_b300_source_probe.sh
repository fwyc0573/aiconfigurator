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

mkdir -p "${EVIDENCE_ROOT}" "${RUNTIME_REPO}"
exec > >(tee -a "${EVIDENCE_ROOT}/remote_stdout.log") 2>&1

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
    "${native_vllm_package}" "${native_vllm_version}"
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
with gzip.open(manifest, "rt", encoding="utf-8") as stream:
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
    patch --batch --no-backup-if-mismatch -p1 -d "${RUNTIME_REPO}"
gzip -dc "${PINNED_MANIFEST_GZ_PATH}" |
    (
        cd "${RUNTIME_REPO}"
        sha256sum -c -
    ) >"${EVIDENCE_ROOT}/pinned_manifest_check.log"
manifest_verified="$(grep -c ': OK$' "${EVIDENCE_ROOT}/pinned_manifest_check.log")"
test "${manifest_verified}" = "2103"
printf 'pinned_manifest_files_verified=%s\n' "${manifest_verified}"
sha256sum \
    "${RUNTIME_REPO}/vllm/model_executor/models/step4pro.py" \
    "${RUNTIME_REPO}/vllm/v1/attention/backends/optimus_fa4.py" \
    "${RUNTIME_REPO}/vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py"

git -C "${RUNTIME_REPO}" init -q
mkdir -p "${RUNTIME_REPO}/.git/objects/pack"
cp "${IDENTITY_PACK_PATH}" "${RUNTIME_REPO}/.git/objects/pack/"
cp "${IDENTITY_INDEX_PATH}" "${RUNTIME_REPO}/.git/objects/pack/"
git -C "${RUNTIME_REPO}" update-ref refs/heads/pinned "${PINNED_COMMIT}"
git -C "${RUNTIME_REPO}" symbolic-ref HEAD refs/heads/pinned
test "$(git -C "${RUNTIME_REPO}" rev-parse HEAD)" = "${PINNED_COMMIT}"
test "$(git -C "${RUNTIME_REPO}" cat-file -t HEAD)" = "commit"
git -C "${RUNTIME_REPO}" rev-parse HEAD
git -C "${RUNTIME_REPO}" ls-tree -r HEAD -- vllm |
    tee "${EVIDENCE_ROOT}/runtime_git_tree.log" >/dev/null
test "$(wc -l <"${EVIDENCE_ROOT}/runtime_git_tree.log")" = "2103"

PYTHONPATH="${RUNTIME_REPO}" python3 - "${RUNTIME_REPO}" <<'PY'
from pathlib import Path
import inspect
import sys

import vllm
import vllm._C
from vllm.model_executor.models import step4pro
from vllm.v1.attention.backends import optimus_fa4

root = Path(sys.argv[1]).resolve()
paths = {
    "vllm_file": Path(vllm.__file__).resolve(),
    "step4pro_file": Path(inspect.getsourcefile(step4pro)).resolve(),
    "optimus_fa4_file": Path(inspect.getsourcefile(optimus_fa4)).resolve(),
}
for key, path in paths.items():
    print(key, path)
    if root not in path.parents:
        raise SystemExit(f"{key} escaped runtime source root: {path}")
PY

printf 'SOURCE_PROBE=PASS\n' | tee "${EVIDENCE_ROOT}/result.txt"
