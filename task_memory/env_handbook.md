## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-04 | Added verified AIC maturin/Rust source-install recipe for the company Ubuntu host. |
| 2026-07-13 | Added verified environment for importing AIC performance runners with pandas and package metadata. |
| 2026-07-14 | Added current-worktree `PYTHONPATH=src:.` isolation rule for performance tests and runners. |
| 2026-07-15 | Added the verified headless Matplotlib test recipe for stale or unreachable SSH `DISPLAY` values. |
| 2026-07-15 | Added the verified pytest temp-directory recipe for `/tmp` inode exhaustion. |
| 2026-07-15 | Added the verified repository-local Git author identity recovery recipe. |
| 2026-07-16 | Added the verified short-`TMPDIR` recipe for Python multiprocessing AF_UNIX socket paths and documented pytest-native timing when `/usr/bin/time` is unavailable. |
| 2026-07-19 | Extended the AIC Rust recipe with a verified locked/offline current-worktree in-place build and source/binary isolation checks. |
| 2026-08-13 | Added the verified Git LFS installation and post-checkout hook recovery recipe. |
| 2026-08-15 | Added the verified unprivileged `systemd-run --user --scope` memory-limit recipe. |
| 2026-08-15 | Added the verified B300 full-RDMA NCCL recipe and recorded the unresolved shared-host-SHM requirement for DeepEP. |
| 2026-08-15 | Verified the supplied brainctl identity and legacy launcher; explicit NVSHMEM passed but DeepEP runtime PE attachment failed. |

# Environment Handbook

## AIC source editable install with maturin on company Ubuntu host

### Root Cause

- AIC source install uses `maturin` and builds the native Rust extension from `rust/aiconfigurator-core/Cargo.toml`.
- If `cargo` is unavailable, `maturin` may invoke `puccinialin` to bootstrap Rust; on this host that path timed out while downloading `rustup-init` from `static.rust-lang.org`.
- Ubuntu default `cargo/rustc 1.75` is too old for dependencies requiring `edition2024`.
- Cargo sparse registry access can hang behind the company network/proxy, while git registry protocol with CLI fetch completed successfully.
- A conda editable installation may point to a different AIC checkout. `PYTHONPATH=src:.` then selects the current Python source while the current checkout still lacks its own native extension; borrowing the other checkout's `.so` creates a Python/Rust source mismatch rather than fixing the environment.

### Verified Recipe

Use the conda env and Rust 1.85 system packages already available on this host:

```bash
# Conda env used for this task
CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design

# Rust/Cargo wrapper directory used for this task
mkdir -p /tmp/aic-rust-185-bin
cat >/tmp/aic-rust-185-bin/cargo <<'SH'
#!/usr/bin/env bash
export RUSTC=/usr/bin/rustc-1.85
exec /usr/bin/cargo-1.85 "$@"
SH
cat >/tmp/aic-rust-185-bin/rustc <<'SH'
#!/usr/bin/env bash
exec /usr/bin/rustc-1.85 "$@"
SH
chmod +x /tmp/aic-rust-185-bin/cargo /tmp/aic-rust-185-bin/rustc

# Validate versions
PATH=/tmp/aic-rust-185-bin:$CONDA_ENV/bin:$PATH cargo --version
PATH=/tmp/aic-rust-185-bin:$CONDA_ENV/bin:$PATH rustc --version

# Warm/verify Cargo metadata through the git registry protocol
PATH=/tmp/aic-rust-185-bin:$CONDA_ENV/bin:$PATH \
RUSTC=/usr/bin/rustc-1.85 \
CARGO_HOME=/tmp/aic-cargo-home-git \
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git \
CARGO_NET_GIT_FETCH_WITH_CLI=true \
CARGO_HTTP_TIMEOUT=120 \
cargo metadata \
  --format-version 1 \
  --manifest-path rust/aiconfigurator-core/Cargo.toml \
  --features pyo3/extension-module >/tmp/aic-cargo-metadata.json

# Install the checkout in editable mode
PATH=/tmp/aic-rust-185-bin:$CONDA_ENV/bin:$PATH \
RUSTC=/usr/bin/rustc-1.85 \
CARGO_HOME=/tmp/aic-cargo-home-git \
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git \
CARGO_NET_GIT_FETCH_WITH_CLI=true \
CARGO_HTTP_TIMEOUT=120 \
$CONDA_ENV/bin/python -m pip install -e . --no-deps --no-build-isolation
```

For a current worktree that must keep the conda environment's existing editable installation untouched, build the extension in place from the current checkout. First prove that the locked dependency cache is complete. In the 2026-07-19 incident, the offline probe named exactly one missing archive; only that exact archive was transferred after checksum validation:

```bash
EXPECTED_LINUX_RAW_SYS_SHA=32a66949e030da00e8c7d4434b251670a91556f4144941d37452769c25d58a53
SOURCE_CRATE=/home/i-fengyicheng/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/linux-raw-sys-0.12.1.crate
TARGET_CRATE=/tmp/aic-cargo-home-git/registry/cache/github.com-25cdd57fae9f0462/linux-raw-sys-0.12.1.crate

test "$(sha256sum "$SOURCE_CRATE" | awk '{print $1}')" = "$EXPECTED_LINUX_RAW_SYS_SHA"
mkdir -p "$(dirname "$TARGET_CRATE")"
cp "$SOURCE_CRATE" "$TARGET_CRATE"
test "$(sha256sum "$TARGET_CRATE" | awk '{print $1}')" = "$EXPECTED_LINUX_RAW_SYS_SHA"

PATH=/tmp/aic-rust-185-bin:$CONDA_ENV/bin:$PATH \
RUSTC=/usr/bin/rustc-1.85 \
CARGO_HOME=/tmp/aic-cargo-home-git \
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git \
CARGO_NET_GIT_FETCH_WITH_CLI=true \
CARGO_NET_OFFLINE=true \
cargo metadata \
  --locked \
  --offline \
  --format-version 1 \
  --manifest-path rust/aiconfigurator-core/Cargo.toml \
  --features pyo3/extension-module \
  >tests/.tmp/aic-cargo-metadata-current-worktree.json

PATH=/tmp/aic-rust-185-bin:$CONDA_ENV/bin:$PATH \
RUSTC=/usr/bin/rustc-1.85 \
CARGO_HOME=/tmp/aic-cargo-home-git \
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git \
CARGO_NET_GIT_FETCH_WITH_CLI=true \
CARGO_NET_OFFLINE=true \
$CONDA_ENV/bin/python -m maturin develop \
  --skip-install \
  --locked \
  --offline \
  --verbose \
  --target-dir /data/ycfeng/tmp/aic-step4-pro-target

PYTHONPATH=src:. "$CONDA_ENV/bin/python" - <<'PY'
from pathlib import Path

import aiconfigurator
import aiconfigurator_core
from aiconfigurator.sdk import engine

root = Path.cwd().resolve()
for module in (aiconfigurator, aiconfigurator_core, engine):
    assert str(Path(module.__file__).resolve()).startswith(str(root))
assert aiconfigurator_core._build_smoke() == 1
print(Path(aiconfigurator_core.__file__).parent / "_aiconfigurator_core.abi3.so")
PY
```

The archive transfer above is not a generic retry or fallback. Use it only when `cargo metadata --locked --offline` identifies that exact locked archive as missing and the source hash matches the recorded value. A different missing crate or hash is a new environment issue and must stop for root-cause analysis.

### Verification Evidence

Observed on 2026-07-04:

```text
$ /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -c "import aiconfigurator; print(aiconfigurator.__version__)"
0.10.0

$ /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/aiconfigurator --help
usage: aiconfigurator [-h] {cli,version} ...
...
```

Observed on 2026-07-19 for the locked/offline current-worktree build:

```text
cargo metadata: exit 0, 505353 bytes, 112 packages, 1 workspace member
maturin/cargo: exit 0, dev profile finished in 37.26 s
extension: src/aiconfigurator_core/_aiconfigurator_core.abi3.so
extension SHA-256: fbd7fc733fe7f183603377812b18f1a269f9e6ea65ae70c597b6f04541954061
Python: 3.11.15
_build_smoke(): 1
import smoke: exit 0
```

### Do Not Use

- Do not patch `src/aiconfigurator/__init__.py` or fake package metadata to bypass the native extension build.
- Do not rely on Ubuntu `cargo/rustc 1.75` for this checkout; it fails on `edition2024` dependencies.
- Do not prefer the `puccinialin` Rust bootstrap path on this host unless network behavior changes and is re-verified.
- Do not copy a compiled AIC extension from another checkout. Build from the current checkout and assert every imported module path is under the current root.
- Do not copy arbitrary crate archives into the task Cargo home. Require an exact locked-version diagnosis and checksum match first.

## Importing AIC performance runners for validation

### Root Cause

- The repository `.venv` may exist without synchronized runtime dependencies; on 2026-07-13 it used Python `3.13.13` but lacked `pandas`.
- System Python `3.12.3` had `pandas`, but the checkout was not installed there, so importing `aiconfigurator` failed with `importlib.metadata.PackageNotFoundError`.
- Performance helpers such as `tests/performance/aic_roofline_pareto/run_phase2_exp.py` require both third-party dependencies and installed AIC distribution metadata.

### Verified Recipe

Use the existing handbook conda environment directly:

```bash
CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design

PYTHONPATH=. "$CONDA_ENV/bin/python" -c \
  "from tests.performance.aic_roofline_pareto.run_phase2_exp import materialize_phase2_parallel_patterns; print(len(materialize_phase2_parallel_patterns()))"
```

### Verification Evidence

Observed on 2026-07-13:

```text
Python 3.11.15
pandas 3.0.3
aiconfigurator 0.10.0
materialized parallel configurations: 25
```

Do not compensate for missing distribution metadata by patching `src/aiconfigurator/__init__.py` or by inventing package-version fallbacks. Activate/use the correctly installed environment instead.

## Selecting the current AIC worktree over another editable checkout

### Root Cause

- The `aic-step-design` conda environment can contain an editable installation pointing to another AIC checkout.
- `PYTHONPATH=.` exposes the repository's `tests` package but does not put the current checkout's `src/` directory ahead of that editable installation.
- This can import stale SDK code while importing current-worktree performance scripts, producing symbol mismatches.

### Verified Recipe

Prefix both `src` and the repository root explicitly:

```bash
CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design
PYTHONPATH=src:. "$CONDA_ENV/bin/python" -m pytest tests/unit/performance -q
PYTHONPATH=src:. "$CONDA_ENV/bin/python" -m tests.performance.aic_roofline_pareto.run_dsv4pro_vs_step4_throughput --help
```

### Verification Evidence

Observed on 2026-07-14:

```text
Python 3.11.15
Focused throughput workflow tests: 9 passed
Parent performance regression tests: 162 passed
```

## Running Matplotlib tests from a headless SSH session

### Root Cause

- An SSH shell can retain a non-empty `DISPLAY` value even when the forwarded X11 endpoint is no longer reachable.
- Matplotlib probes that endpoint through `XOpenDisplay()` during backend selection, so tests can block before reporting their first result.
- This is an environment-entry problem, not a plotting-test failure or an AIC performance regression.

### Verified Recipe

Select Matplotlib's non-interactive backend explicitly for test and batch-plot commands:

```bash
CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design

MPLBACKEND=Agg \
PYTHONPATH=src:. \
"$CONDA_ENV/bin/python" -m pytest tests/unit/performance -q
```

Use the same `MPLBACKEND=Agg` setting for non-interactive figure-generation commands executed from a headless session. Do not unset or rewrite the host's SSH configuration as a substitute for selecting the correct batch backend.

### Verification Evidence

Observed on 2026-07-15:

```text
DISPLAY=localhost:17.0
Unqualified pytest: blocked in Matplotlib mpl_display_is_valid() -> XOpenDisplay()
MPLBACKEND=Agg pytest: 173 passed in 3.47 seconds
Exit code: 0
```

## Running pytest when `/tmp` has free bytes but no free inodes

### Root Cause

- A `tmpfs` can report substantial free capacity while its inode pool is exhausted.
- The verified symptom was `df -h /tmp` showing about `14G` free while `df -i /tmp` showed `IFree=0` and `IUse%=100%`.
- Under that condition, pytest can still execute existing code, but `tee` and fixtures that create new `/tmp` files fail with `No space left on device`.

### Verified Recipe

Use a task-local temporary directory on the data filesystem and keep its contents out of Git staging:

```bash
mkdir -p tests/.tmp
TMPDIR="$PWD/tests/.tmp" PYTHONPATH=src:. MPLBACKEND=Agg \
  /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest <paths>
```

### Verification Evidence

Observed on 2026-07-15:

```text
/tmp: 14G available, 0 free inodes
focused + integration: 178 passed in 8.27s
exit code: 0
```

Do not delete unrelated `/tmp` files as an implicit cleanup. Redirect temporary creation to the data filesystem, identify the owner of inode-heavy paths separately, and obtain permission before any deletion.

## Running multiprocessing tests from a long worktree path

### Root Cause

- Python `multiprocessing.Manager()` creates an AF_UNIX listener below `tempfile.gettempdir()` using a path shaped like `pymp-XXXXXXXX/listener-XXXXXXXX`.
- Linux AF_UNIX socket pathnames must fit in the fixed `sockaddr_un.sun_path` field. A long repository-local `TMPDIR` can therefore have sufficient bytes and inodes while still failing at `socket.bind()` with `OSError: AF_UNIX path too long`.
- In the Step4-Pro-V1 worktree, `TMPDIR="$PWD/tests/.tmp"` was `81` characters. A representative manager listener pathname was `113` characters, beyond the practical pathname limit.
- This is an environment-entry failure, not a `collector.parallel_run` logic failure. The parent process surfaces `EOFError` only because the SyncManager child exits before sending its listener address.

### Verified Recipe

Preflight capacity and inode availability, then select a short existing directory explicitly:

```bash
df -h /tmp /data
df -i /tmp /data

CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design

PYTHONPATH=src:. \
MPLBACKEND=Agg \
TMPDIR=/data/ycfeng/tmp \
  "$CONDA_ENV/bin/python" -m pytest tests/unit/collector/test_parallel_run.py -q
```

`TMPDIR=/tmp` is also verified when its inode preflight is healthy. Select one verified path before the run; do not implement an automatic retry or silently switch paths after a failure.

### Verification Evidence

Observed on 2026-07-16:

```text
Long TMPDIR base length: 81 characters
Representative manager socket length: 113 characters
Long-TMPDIR control: 1 failed in 0.36s, exit code 1, AF_UNIX path too long
/tmp single-test experiment: 1 passed in 0.24s, exit code 0
/tmp collector suite: 22 passed in 6.00s, exit code 0
/data/ycfeng/tmp collector suite: 22 passed in 7.18s, exit code 0
/tmp full unit suite: 2063 passed, 12 skipped, 1123 deselected in 770.74s, exit code 0
```

Do not patch `multiprocessing`, shorten application names, skip collector tests, or delete temporary files to mask this environment constraint. Correct the `TMPDIR` entry point.

## Timing pytest when `/usr/bin/time` is unavailable

### Root Cause

- This host does not provide `/usr/bin/time`; invoking it exits `127` before pytest starts.
- Pytest already emits suite elapsed time in its terminal summary, so a separate timing binary is not required for reproducible test evidence.

### Verified Recipe

Run pytest directly and record its summary duration plus the shell exit code:

```bash
CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design
"$CONDA_ENV/bin/python" -m pytest <paths> -q
printf 'exit=%s\n' "$?"
```

Observed on 2026-07-16: `/usr/bin/time -p ...` returned `/bin/bash: /usr/bin/time: No such file or directory` with exit code `127`; the direct pytest commands reported their own elapsed times and completed normally.

## Applying a host memory limit without privileged systemd authorization

### Root Cause

- `systemd-run --scope -p MemoryMax=2G ...` targets the system manager and
  requires interactive authorization on this host.
- Non-interactive agent sessions therefore fail before the command starts with
  `Failed to start transient scope unit: Interactive authentication required.`
- The user systemd manager is available and can create an equivalent
  user-owned transient scope without privileged authorization.

### Verified Recipe

Use the user manager and keep temporary files on the data filesystem:

```bash
timeout 900s systemd-run --user --scope -p MemoryMax=2G \
  env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. \
  /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  -m pytest -q <paths>
```

### Verification Evidence

Observed on 2026-08-15:

```text
system scope: Interactive authentication required
user scope: Running as unit run-<id>.scope
```

Do not remove the memory limit or retry through the privileged system manager.
Use `--user --scope` directly.

## Recovering a missing Git author identity without changing global configuration

### Root Cause

- A worktree can inherit no effective `user.name` or `user.email` when neither repository-local nor global Git configuration provides them.
- In that state, `git commit` fails before creating a commit with `Author identity unknown` or `empty ident name`.

### Verified Recipe

First inspect repository history and existing configuration rather than inventing an identity:

```bash
git config --show-origin --get-regexp '^user\.(name|email)$' || true
git log --all --format='%an <%ae>' | sort -u
```

When the intended identity is confirmed by existing commits, set it only for the current repository:

```bash
git config --local user.name 'fwyc0573'
git config --local user.email '935953068@qq.com'
git config --local --get-regexp '^user\.(name|email)$'
```

### Verification Evidence

Observed on 2026-07-15:

```text
Historical repository author: fwyc0573 <935953068@qq.com>
Effective repository-local user.name: fwyc0573
Effective repository-local user.email: 935953068@qq.com
```

Do not set or overwrite global Git identity as an implicit repair. If repository history does not establish the intended author, stop and request the identity instead of guessing.

## Git LFS missing from PATH during branch/worktree checkout

### Root Cause

- The repository installs a `post-checkout` hook that invokes `git-lfs`.
- A branch switch can complete while the hook returns exit code 2 when the
  `git-lfs` binary is absent.
- Deleting or bypassing the hook hides the missing dependency and can leave
  LFS-managed files unresolved.

### Verified Recipe

The company Ubuntu mirror provides the supported package:

```bash
apt-cache policy git-lfs
sudo -n apt-get install -y git-lfs
git lfs version
```

Re-run the existing hook with the current revision to verify the same path
that failed:

```bash
HOOK="$(git rev-parse --git-common-dir)/hooks/post-checkout"
"$HOOK" "$(git rev-parse HEAD)" "$(git rev-parse HEAD)" 1
git lfs env
```

### Verification Evidence

Observed on 2026-08-13:

```text
git-lfs/3.4.1
post-checkout hook exit code: 0
```

Do not remove the repository hook or set `GIT_LFS_SKIP_SMUDGE` as an implicit
repair. Diagnose registry/authentication separately if a later LFS fetch
fails.

## B300 two-node full-RDMA NCCL preflight

### Root Cause

- `rdma/mlnx_shared=8` alone left `NCCL_IB_HCA` empty.
- Adding host networking and `mellanox.com/mlnx_rdma=1` made worker-init
  expose eight bond HCAs, but it also prepended
  `/usr/local/cuda-12.8/compat`.
- That stale compat path makes torch `2.10.0+cu129` fail CUDA initialization
  with error `803`.

### Verified Recipe

Use both RDMA resources and host networking:

```bash
--host-network=true \
--custom-resources=rdma/mlnx_shared=8 \
--custom-resources=mellanox.com/mlnx_rdma=1 \
--topo-group=yes
```

After worker-init, keep the injected NCCL/NVSHMEM variables but restore the
pinned B300 CUDA path:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64
```

Do not hardcode `NCCL_IB_HCA`; require the platform value to be non-empty.

### Verification Evidence

Observed on 2026-08-15:

```text
NCCL_IB_HCA==mlx5_bond100,...,mlx5_bond107
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=bond0
world_size=16
all_reduce actual=136.0 expected=136.0
```

Evidence:

```text
/data/ycfeng/tmp/b300_step4_smoke_20260814/
  nccl_preflight_s4p-nccl2-0815-195936/
```

### DeepEP Limitation

This recipe validates NCCL only. `deep_ep.Buffer.runtime.sync` still fails
without the later B300 branch's shared-host-SHM/explicit NVSHMEM bootstrap.
The available `brainctl`/`rlaunch` client has no shared-host-SHM option. Do not
replace this requirement with privileged mode, an invented host volume,
socket fallback, or another all-to-all backend.

### Supplied brainctl and legacy launcher result

The user-supplied brainctl and installed `/kubebrain/brainctl` are
byte-identical:

```text
sha256:
06d5fffb00e67633e10e4a6d96752517eda7559230466a63ac86e6a424c839ad
```

The supplemental RJob guide documents the legacy launcher:

```bash
brainctl launch --i-know-i-am-using-legacy-rlaunch
```

This path does not expose `--share-host-shm`, but it was tested with two B300
nodes and full RDMA:

```text
NCCL:                  16/16 PASS
explicit nvshmem.init: 16/16 PASS
deep_ep.Buffer:         0/16 PASS
```

DeepEP failed:

```text
runtime.cu:136 'nvshmem_n_pes() == num_ranks'
```

The `/dev/shm` mounts were large tmpfs filesystems and memlock was unlimited.
Therefore shared-memory capacity is not the cause, and shared-host-SHM itself
remains unproven. The active blocker is the integration between the
externally initialized NVSHMEM runtime and the DeepEP-linked runtime/package.
