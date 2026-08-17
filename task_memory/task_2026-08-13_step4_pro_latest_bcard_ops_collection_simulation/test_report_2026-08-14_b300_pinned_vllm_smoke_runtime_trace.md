## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-08-15 | Created the pinned-vLLM B300 smoke/runtime-trace report with the completed one-GPU result, runtime-deviation review, and current two-node stop gate. |
| 2026-08-15 | Recorded the owner-authorized fourth run: role/Gloo setup passed, NCCL failed because no usable HCA was injected. |
| 2026-08-15 | Recorded the passing 16-rank full-RDMA NCCL preflight and the remaining DeepEP Buffer shared-host-SHM/NVSHMEM blocker. |
| 2026-08-15 | Recorded the supplied brainctl/docs verification and legacy-launcher probe: explicit NVSHMEM passed, DeepEP Buffer PE-count assertion failed. |
| 2026-08-15 | Recorded exact checkout/static validation of `9bfd9a610e` and the blocking result that all installed launch paths reject its required shared-host-SHM flag. |
| 2026-08-17 | Marked this report as historical DeepEP/NVSHMEM evidence; the active runtime contract is now documented by `test_report_2026-08-17_non_deepep_runtime_backend.md`. |

# Test Report: B300 Pinned-vLLM Step4Pro Smoke and Runtime Trace

> **Archive scope:** This report preserves the historical DeepEP/NVSHMEM
> investigation and the earlier script identities. It does not describe the
> current active runtime backend. The active AgRs/NCCL-backed runtime,
> coordinated lifecycle, and quota-blocked final state are recorded in
> `test_report_2026-08-17_non_deepep_runtime_backend.md`.

**Execution date:** 2026-08-15
**Current overall status:** **PARTIAL PASS**

- Pinned source identity: PASS.
- One-GPU synthetic smoke: PASS.
- Optimus FA4 and Optimus FP8/DeepGEMM provider gates: PASS.
- Two-node DP16/EP16 DeepEP HT: NOT YET PASSED.
- Cleanup after every recorded attempt: PASS.

## 1. Test Script Information

### Scripts

| Script | SHA256 |
|---|---|
| `tests/e2e/step4_pro_latest/run_b300_single_smoke.sh` | `e1cd3e49e52e208ee589098cf2cfa182c99e2751e06bfa5b6c1df86f3016e4b8` |
| `tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh` | `ef718923d599aa01bbfe3ba58fd39e2bec2f8a76a9309e634b2142e06ca24353` |
| `tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh` | `8ea49ef0a85d4660cc17723301a3499a10f1a5d32661e4fe1cbd657005a203ec` |
| `tests/e2e/step4_pro_latest/run_b300_two_node_nccl_preflight.sh` | `7629d5b669698a8b7d2e027703a271ad1ffbd5cc8f5b6b1a9d2eb5a428a56ddc` |
| `tests/e2e/step4_pro_latest/run_b300_two_node_deepep_legacy_probe.sh` | `d77f7519accde297e2317f1e81e14e110ca695adba87fe48c732d42acd5b4308` |
| `tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py` | `1b094f1ed6ee9138af514bdc3961ea48defb99bfd07193add9263b0e6d5ccf4c` |
| `tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py` | `369a058a567c1924b00f126e63c7520e1dc26106fa2c2560794c618ab9481f0d` |
| `tests/e2e/step4_pro_latest/test_b300_two_node_nccl_preflight_contract.py` | `f23340a743452d8d29819c461ecedb0bb94b36b4cddea199f5ff2af80a32096b` |
| `tests/e2e/step4_pro_latest/test_b300_two_node_deepep_legacy_probe_contract.py` | `e4414ecdc0670f6181f082b588d605e58895dc10e1e8ce9f1c2f50044970315b` |

### Reproducible commands

Static contract:

```bash
CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design
PYTHONPATH=src:. "$CONDA_ENV/bin/python" -m pytest \
  tests/e2e/step4_pro_latest/test_generate_step4pro_dummy_configs.py \
  tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py \
  tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py \
  tests/e2e/step4_pro_latest/test_run_b300_source_probe_static.py -q

bash -n tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh
bash -n tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh
git diff --check -- tests/e2e/step4_pro_latest
```

One GPU:

```bash
RJOB_NAME=s4p-ok-0815-163232 \
ENABLE_PROFILER=0 \
CONTROL_MEMORY_MAX=3G \
bash tests/e2e/step4_pro_latest/run_b300_single_smoke.sh
```

Two nodes:

```bash
RJOB_NAME="s4p-nccl-$(date +%m%d-%H%M%S)" \
CONTROL_MEMORY_MAX=3G \
bash tests/e2e/step4_pro_latest/run_b300_two_node_nccl_preflight.sh

RJOB_NAME="s4p-2n-$(date +%m%d-%H%M%S)" \
ENABLE_PROFILER=0 \
CONTROL_MEMORY_MAX=3G \
NCCL_PREFLIGHT_EVIDENCE=<passing-preflight-launch.log> \
bash tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh
```

The full model command is gated by a passing 16-rank NCCL preflight.

### Environment

| Item | Value |
|---|---|
| Host test environment | `aic-step-design` |
| Host Python / pytest | `3.11.15` / `8.4.2` |
| Worker Python | `3.12` |
| vLLM | `0.19.0.post20.dev26+gc820e5ae1` |
| Pinned source commit | `607d1641ee3fec43653fca510d717725828890c2` |
| torch / CUDA | `2.10.0+cu129` / `12.9` |
| step-optimus | `3.23.24` |
| DeepGEMM / DeepEP | `2.4.2+torch210.py312.0ee2fab` / `1.2.1+c5284713.step.torch2.10.0.cu129.py312` |
| GPU | NVIDIA B300 SXM6 AC, SM `10.3` |
| GPU memory | `275040 MiB` |
| Driver | `580.159.03` |
| Image | `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled` |
| Image digest | `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b` |
| Controller limit | `MemoryMax=3G` |

## 2. Validation Criteria

### One GPU

- Verify all `2103` pinned source files.
- Load the 14-layer synthetic config with dummy weights.
- Reach `/health`.
- Complete one prefill/decode request and four concurrent requests.
- Prove Full-MFA uses Optimus FA4 at head dimension `512`, page size `128`.
- Prove MoE selects `OPTIMUS_FP8`, `OptimusFp8Experts`, and DeepGEMM.
- Preserve every runtime overlay diff/hash.
- Leave zero matching RJobs, Replicas, and local launcher processes.

### Two nodes

- Start two replicas with eight B300 GPUs each.
- Restore `NODE_RANK=0/1`, `NODE_COUNT=2`, one shared `MASTER_ADDR`, and
  `PROC_PER_NODE=8`.
- Use DP16/EP16 and `deepep_high_throughput`.
- Use one API-serving head; non-head nodes must run
  `--headless --api-server-count 0`.
- Prove `Using DeepEPHTAll2AllManager`, HT dispatch, HT combine, and
  `FORWARD_CONTEXT ... batch=real`.
- Complete at least one request.
- Leave zero matching resources.

## 3. Test Results and Evidence

### 3.1 Pinned source identity — PASS

- Runtime Git commit:
  `607d1641ee3fec43653fca510d717725828890c2`.
- Manifest verification: `2103/2103`.
- Source evidence:
  `/data/ycfeng/tmp/b300_step4_smoke_20260814/source_probe_s4p-src-0815-125214/`.

### 3.2 One-GPU synthetic smoke — PASS

Evidence root:

```text
/data/ycfeng/tmp/b300_step4_smoke_20260814/single_s4p-ok-0815-163232/
```

| Metric | Actual |
|---|---:|
| Scheduling | `11 s` |
| Model load | `9.083845 s` |
| Model-load memory | `3.16 GiB` |
| Health ready | `100 s` |
| GPU memory before load | `0 MiB` |
| GPU memory after load | `247628 MiB` |
| GPU memory free after load | `26486 MiB` |
| Single request | `13.810807 s` |
| Prompt/output tokens | `8 / 4` |
| Four-request wall time | `0.457171424 s` |
| Request 1 | `0.450382 s` |
| Request 2 | `0.450498 s` |
| Request 3 | `0.451179 s` |
| Request 4 | `0.450456 s` |

Provider evidence:

- Full MFA: Optimus FA4 actual forward, BF16, hd512, page size 128.
- MoE: `OPTIMUS_FP8`, `OptimusFp8Experts`, Optimus DeepGEMM, packed UE8M0.
- Real forward evidence includes prefill and decode.

Key hashes:

| Evidence | SHA256 |
|---|---|
| `metrics.env` | `c86c1308239dc1783bca604ac71a1335bf1de3f7a8e8fd14c0382300d4f4f783` |
| `vllm_server.log` | `57892faebe83d51cfac84067f80975d62604c9b32403e131ceba38fe3e8f0415` |
| `result.env` | `20a886db268f5e8c97eba79bbff20074465b2f8ebd795a18494fd788970943b4` |
| Optimus JIT quant diff | `f7348256cfd76d2db643cd6bbdfb206e425c804edf1aa15687da336b7eec27ce` |
| Pinned `ep_gather` diff | `09ddfaf70efb32ab948fa18b8ec1a663af65b12dff762a330d07c46418e47913` |
| Masked gather diff | `d439b12599a725576d9803588d1a54215a3b37c034e877add487ba51e54f66f7` |

### 3.3 Runtime-fix necessity and impact review — PASS WITH BOUNDARY

| Change | Required for this pinned run? | Impact |
|---|---|---|
| Optimus JIT activation quant | Yes | Preserves model/provider identity but changes FP8 quant implementation and scale representation; no bitwise or quality-equivalence claim |
| Contiguous `ep_gather` block fix | Yes | Inference tiling only; exact dimension coverage and top-k accumulation order preserved |
| Masked gather selector fix | Yes for hidden 896 | Same tiling-only boundary |
| Strict Triton driver signature | Yes for changing batch shapes | Cache-key behavior only |
| `flash_attn.__spec__` repair | Yes in this package composition | Import metadata only |
| Attention-type-scoped fallback assertion | Yes for correct validation | Test logic only |

The runtime copy writes no checkpoint, gradient, or optimizer state. It cannot
alter an existing training result. Dummy-weight evidence does not validate
generation quality, loss, or training convergence.

### 3.4 Two-node DP16/EP16 DeepEP HT — NOT PASSED

| Attempt | Scheduling | Result | Root cause | Cleanup |
|---|---:|---|---|---|
| `s4p-2d-0815-172744` | `18 s` | FAIL | Host probed env file before holder completed worker-init | RJob `0`, Replica `0` |
| `s4p-2e-0815-174618` | `695 s` | FAIL | Non-head remote engines lacked `--headless` | RJob `0`, Replica `0` |
| `s4p-2f-0815-181337` | `26 s` | FAIL | Headless node retained default `--api-server-count=8` | RJob `0`, Replica `0` |
| `s4p-2g-0815-185302` | `15 s` | FAIL | NCCL first all-reduce failed; worker environment had `NCCL_IB_HCA=''` | RJob `0`, Replica `0` |
| `s4p-2h-0815-202202` | `16 s` | FAIL | DeepEP Buffer `runtime.sync` failed at CUDA `runtime.cu:83` | RJob `0`, Replica `0` |

Live environment restoration and the head/worker role split passed in the
authorized fourth attempt:

```text
node rank: 0 / 1
data-parallel start rank: 0 / 8
node count: 2
processes per node: 8
shared MASTER_ADDR: PASS
RoCE / NCCL_IB_GID_INDEX=5: PASS
Gloo peer connectivity: 16/16 ranks
non-head CLI: --headless --api-server-count 0
```

The full-RDMA configuration resolved NCCL:

```text
NCCL_IB_HCA==mlx5_bond100,...,mlx5_bond107
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=bond0
NCCL rank passes=16/16
NCCL node passes=2/2
all-reduce actual=136.0 expected=136.0
```

The model then selected DeepEP HT and failed during Buffer synchronization:

```text
Using DeepEPHTAll2AllManager all2all manager
DeepEP HT get_handle start
RuntimeError: Failed: CUDA error
/workspace/DeepEP/csrc/kernels/runtime.cu:83 'unknown error'
```

The supplied brainctl binary was byte-identical to the installed binary. The
supplemental docs suggested the legacy Process/Worker launcher, so a minimal
two-node retry was run with explicit NVSHMEM initialization:

```text
brainctl supplied/installed SHA256: identical
legacy NCCL PASS:                 16/16
explicit NVSHMEM init PASS:       16/16
DeepEP Buffer PASS:                0/16
```

Every rank failed:

```text
Assertion error /workspace/DeepEP/csrc/kernels/runtime.cu:136
'nvshmem_n_pes() == num_ranks'
```

Legacy evidence:

```text
/data/ycfeng/tmp/b300_step4_smoke_20260814/
  legacy_deepep_s4p-legdep3-0815-220757/
```

| Evidence | SHA256 |
|---|---|
| supplied/installed brainctl | `06d5fffb00e67633e10e4a6d96752517eda7559230466a63ac86e6a424c839ad` |
| supplemental RJob docs | `627f8393cc2c0ce63b7e023a8e07930b376c50775a455c8d2afd8973cb8d0a0f` |
| legacy `legacy_probe.log` | `e83d82020afa5249fd5a52f957fe7f864a450e8780387b7d851e6d9e79bb99d6` |
| legacy `metrics.env` | `9f1c6b0cc6755019358b782d9b94c24d793cff19436330fedf4ecf7cb5b16abe` |

Static verification after the final fixes: `20 passed`, `0 failed`,
`0.34 seconds`.

Fourth-run evidence hashes:

| Evidence | SHA256 |
|---|---|
| `launch.log` | `7a342c7b3f31ef0cbd1f320e95bc6ce2856e7b7b1a7e827b8eb51ee823f383a2` |
| rank-0 `remote_exec` | `4bd0323d47039177b0e39209904f41f686c301436e3a3be94aa42573cb177320` |
| rank-1 `remote_exec` | `0e7004bb9dcdd3c0842a7d77a4b9cdd8815cc31fd5d54f667be4545a6a7d272a` |
| rank-0 `vllm_server.log` | `889defc2d4cd502a83dd54fec73f195568f388e7d33dde905769ac4c16cebfaf` |
| final RJob query | `bd5c08a53a555149fe4080aee6402ee1832ae97e47f7c399fb4d227c5eda3527` |
| final Replica query | `dda2863d9c907e337fd8dccd943d3006a5c47f38c517f3e3590c133f2a6f5255` |

Passing NCCL preflight:

```text
/data/ycfeng/tmp/b300_step4_smoke_20260814/
  nccl_preflight_s4p-nccl2-0815-195936/
```

| Evidence | SHA256 |
|---|---|
| preflight `launch.log` | `a4c183c3bf6a3764fbb44628fb6208f01fe8415e3955f079084db2a4c4089848` |
| preflight final RJob query | `64c6b509ccc0796a9f2166e3515e50a11a1b0479a1420a0771b7537f90c78ae4` |
| preflight final Replica query | `8167590c3c9700d2a79f2732fe9f5a860a1d3034593a187575d53d99037eae24` |
| full-model `vllm_server.log` | `7e3227ee828ba4271bac18b1ebf28b9ea00de476d01020c4dd86d99dc2bcf5a7` |

Required but still absent:

- successful DeepEP HT manager startup;
- HT dispatch and combine evidence;
- real two-node request latency;
- two-node GPU memory after load.

### 3.5 Cleanup — PASS

Every recorded attempt ended with:

- exact matching RJob count: `0`;
- exact matching Replica count: `0`;
- related local process count: `0` excluding the cleanup `awk` itself.

### 3.6 Exact `9bfd9a610e` launch fix — STATIC PASS, LAUNCHER CONTRACT FAIL

#### Test Script Information

- Scripts:
  - `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro/rjob-step4pro-2node.sh`
  - `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro/rjob-step4pro-deepep-probe.sh`
- Environment:
  - controller shell: Bash;
  - brainctl:
    `v2.12.0-alpha.4.1328-20260814040139-9da5d7fa9435`;
  - controller scopes: `MemoryMax=2G`;
  - Python: not invoked by this non-allocating launcher/static gate.
- Exact static command:

  ```bash
  git -C vllm-step4-pro checkout --detach \
    9bfd9a610ea4f2890010702ee7a207cf25edf8de
  git -C vllm-step4-pro rev-parse HEAD
  git -C vllm-step4-pro rev-parse HEAD^
  git -C vllm-step4-pro diff-tree --no-commit-id --name-status -r HEAD
  rg -n \
    'nvshmem\.init|NVSHMEM_ENABLE_NIC_PE_MAPPING|NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME|--share-host-shm=True|deep_ep\.Buffer' \
    vllm-step4-pro/rjob-step4pro-2node.sh \
    vllm-step4-pro/rjob-step4pro-deepep-probe.sh
  bash -n vllm-step4-pro/rjob-step4pro-2node.sh
  bash -n vllm-step4-pro/rjob-step4pro-deepep-probe.sh
  ```

- Exact launcher-parse commands:

  ```bash
  sudo -n systemd-run --quiet --scope -p MemoryMax=2G \
    timeout --signal=TERM --kill-after=5s 60s \
    /kubebrain/brainctl rjob launch --share-host-shm=True --help

  sudo -n systemd-run --quiet --scope -p MemoryMax=2G \
    timeout --signal=TERM --kill-after=5s 60s \
    /kubebrain/brainctl launch \
      --i-know-i-am-using-legacy-rlaunch \
      --share-host-shm=True --help
  ```

#### Validation Criteria

1. `HEAD` is exact `9bfd9a610e` and its parent is exact `607d1641ee`.
2. The commit changes only the two expected launch scripts.
3. Explicit NVSHMEM bootstrap, NIC-to-PE mapping, bootstrap socket interface,
   `deep_ep.Buffer`, and shared-host-SHM markers are present.
4. Both scripts pass `bash -n`.
5. An installed launcher must accept `--share-host-shm=True` before
   predict-only or live B300 allocation.

#### Results and Evidence

| Check | Result | Numeric result |
|---|---|---:|
| Exact commit | PASS | `1/1` |
| Exact parent | PASS | `1/1` |
| Expected changed files | PASS | `2/2` |
| Model Python/CUDA changed files | PASS | `0` |
| Shell syntax | PASS | `2/2` |
| RJob shared-host-SHM parse | FAIL | exit `1` |
| Legacy shared-host-SHM parse | FAIL | exit `1` |
| Standalone `rjob` client | FAIL | found `0` |
| B300 allocations created | PASS | `0` |

Both installed launch paths returned:

```text
unknown flag: --share-host-shm
```

Evidence directory:

```text
/data/ycfeng/tmp/step4_deepep_launcher_surface_20260815/
```

The commit itself is present and statically valid. Its required launch API is
not available on this controller, so matching predict-only and the minimal live
Buffer probe were not run.

## 4. Current Gate and Next Action

The one-GPU, 16-rank NCCL, and explicit 16-rank NVSHMEM bootstrap portions are
accepted. Exact direct-child commit `9bfd9a610e` is now applied and statically
verified, but no installed launcher accepts its required
`--share-host-shm=True` option. The standalone `rjob submit` client used by the
commit is absent.

Provide that client/version or a documented equivalent platform field. Then
run matching predict-only and the minimal two-node Buffer probe before any
complete-model retry.

## Handoff Matrix

| Gate | PASS/FAIL | Key numeric result | Evidence path |
|---|---|---:|---|
| Pinned source identity | PASS | `2103/2103` | `source_probe_s4p-src-0815-125214/` |
| One-GPU health | PASS | `100 s` | `single_s4p-ok-0815-163232/` |
| Prefill/decode request | PASS | `13.810807 s` | same |
| Four-request concurrency | PASS | `0.457171424 s` | same |
| Optimus FA4 provider | PASS | hd512/page128 | same |
| Grouped `wo_a` provider | PARTIAL | source/runtime path proven; no dedicated profiler result in this report | same |
| Optimus FP8 MoE | PASS | top-k `16`, Optimus DeepGEMM | same |
| Two-node NCCL | PASS | `16/16`, sum `136.0` | `nccl_preflight_s4p-nccl2-0815-195936/` |
| Explicit NVSHMEM bootstrap | PASS | `16/16` | `legacy_deepep_s4p-legdep3-0815-220757/` |
| `9bfd9a610e` static launch fix | PASS | `2/2` scripts | `step4_deepep_launcher_surface_20260815/` |
| Shared-host-SHM launcher API | FAIL | `0/2` installed modes accept flag | same |
| Two-node DP16/EP16 | FAIL | manager selected; Buffer sync failed | `two_node_s4p-2h-0815-202202/` |
| DeepEP HT dispatch | FAIL | no accepted marker | same |
| DeepEP HT combine | FAIL | no accepted marker | same |
| Resource cleanup | PASS | `0 / 0 / 0` | each attempt's cleanup logs |
