## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-04 | Added verified AIC maturin/Rust source-install recipe for the company Ubuntu host. |
| 2026-07-13 | Added verified environment for importing AIC performance runners with pandas and package metadata. |
| 2026-07-14 | Added current-worktree `PYTHONPATH=src:.` isolation rule for performance tests and runners. |
| 2026-07-15 | Added the verified headless Matplotlib test recipe for stale or unreachable SSH `DISPLAY` values. |
| 2026-07-15 | Added the verified pytest temp-directory recipe for `/tmp` inode exhaustion. |
| 2026-07-15 | Added the verified repository-local Git author identity recovery recipe. |

# Environment Handbook

## AIC source editable install with maturin on company Ubuntu host

### Root Cause

- AIC source install uses `maturin` and builds the native Rust extension from `rust/aiconfigurator-core/Cargo.toml`.
- If `cargo` is unavailable, `maturin` may invoke `puccinialin` to bootstrap Rust; on this host that path timed out while downloading `rustup-init` from `static.rust-lang.org`.
- Ubuntu default `cargo/rustc 1.75` is too old for dependencies requiring `edition2024`.
- Cargo sparse registry access can hang behind the company network/proxy, while git registry protocol with CLI fetch completed successfully.

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

### Verification Evidence

Observed on 2026-07-04:

```text
$ /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -c "import aiconfigurator; print(aiconfigurator.__version__)"
0.10.0

$ /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/aiconfigurator --help
usage: aiconfigurator [-h] {cli,version} ...
...
```

### Do Not Use

- Do not patch `src/aiconfigurator/__init__.py` or fake package metadata to bypass the native extension build.
- Do not rely on Ubuntu `cargo/rustc 1.75` for this checkout; it fails on `edition2024` dependencies.
- Do not prefer the `puccinialin` Rust bootstrap path on this host unless network behavior changes and is re-verified.

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
