## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-14 | Created the standalone execution guide for the externally owned B300 pinned-vLLM smoke and runtime/provider trace task. |
| 2026-08-14 | Reduced controller scopes to 3 GiB and added exact-query, disk-streaming, and cleanup ordering rules after a host OOM. |
| 2026-08-15 | Reconciled the guide with the parent requirement: synthetic config and vLLM dummy weights replace real checkpoint mounts. |
| 2026-08-15 | Authorized and documented the single Optimus JIT activation-quant overlay required on B300 SM103. |
| 2026-08-15 | Re-evaluated and bounded every runtime deviation; separated exact tiling fixes from the non-bitwise quant implementation change. |
| 2026-08-15 | Added the verified holder-env readiness and coordinated-DP headless worker contract after three bounded two-node attempts. |
| 2026-08-15 | Added the verified full-RDMA NCCL preflight and documented the remaining shared-host-SHM/NVSHMEM DeepEP Buffer gate. |
| 2026-08-15 | Recorded the legacy launcher retry: explicit NVSHMEM initialized, but DeepEP Buffer observed a mismatched PE count. |
| 2026-08-15 | Authorized the exact `9bfd9a610e` DeepEP launch-script fix as a bounded exception and required its standalone Buffer probe before any full-model retry. |

# B300 Pinned-vLLM Smoke and Runtime/Provider Trace Execution Guide

## 1. Task Boundary

This document is the handoff contract for a separate session/agent.

The external session owns:

1. The one-GPU Step4Pro smoke using the pinned 14-layer recipe.
2. The two-node DP16/EP16 smoke needed to prove the DeepEP path.
3. Runtime/provider evidence for:
   - Step4Pro source identity;
   - Optimus FA4 for Full MFA;
   - Optimus FP8 MoE / DeepGEMM;
   - DeepEP high-throughput dispatch and combine.
4. Cleanup of every RJob/Replica created by that session.
5. A reproducible Markdown report and the raw logs needed by the AIC session.

The external session does **not** own:

- AIC model or operation implementation;
- Collector implementation or AIC operation microbenchmark collection;
- AIC correctness tests or simulator execution;
- the 78-layer formal performance matrix;
- MTP1 implementation, measurement, or simulation.

MTP1 remains deferred. `Step3p5MTP` must not be presented as a Step4Pro MTP1
implementation.

## 2. Authoritative Inputs

| Item | Required value |
|---|---|
| Repository | `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro` |
| Base model/source commit | `607d1641ee3fec43653fca510d717725828890c2` |
| Authorized DeepEP launch commit | `9bfd9a610ea4f2890010702ee7a207cf25edf8de` |
| Branch name for reference | `xwx/step4pro-fa3-optimus` |
| Runtime image | `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled` |
| Image digest already inspected | `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b` |
| GPU | `NVIDIA B300 SXM6 AC` |
| B300 quota group | `b300_train_infra` |
| Positive tag | `B300` |
| Namespace | `shai-core` |
| Full-MFA provider | Optimus CuTe FA4, head dimension 512, page/block size 128 |
| Distributed MoE provider | Optimus FP8 MoE / DeepGEMM with DeepEP high-throughput |

The base model implementation remains the pinned `607d1641ee` contract. For
the DeepEP launch lane only, the owner authorized direct-child commit
`9bfd9a610ea4f2890010702ee7a207cf25edf8de`. It modifies no model Python/CUDA
code and must be recorded as a two-script launch overlay.

Before launching anything, verify:

```bash
git -C \
  /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro \
  rev-parse HEAD
```

For the authorized DeepEP lane, the output must be exactly:

```text
9bfd9a610ea4f2890010702ee7a207cf25edf8de
```

Also verify that its parent is exact
`607d1641ee3fec43653fca510d717725828890c2` and that the commit changes only
`rjob-step4pro-2node.sh` plus `rjob-step4pro-deepep-probe.sh`. One-GPU evidence
continues to use the unchanged base model code.

Setting an environment variable named `VLLM_PULL_COMMIT` is not proof that the
runtime imported that source. The final evidence must include both the actual
runtime source path and its Git/file identity.

## 3. Required Reading

Read these files in order:

1. Project rules:
   `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/AGENTS.md`
2. GPU worker handbook:
   `/data/ycfeng/stepfun-env-handbook/guidence.md`
3. Docker handbook:
   `/data/ycfeng/stepfun-env-handbook/docker.md`
4. Main requirements:
   `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/task_memory/step4pro_v4_external_simulator_requirements.md`
   - especially sections 2.1, 2.3, 2.4, and 3.1;
5. Operation provenance:
   `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/op_provenance.md`
6. Pinned one-GPU recipe:
   `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro/rjob-step4pro-optimus-single.sh`
7. Pinned two-node recipe:
   `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro/rjob-step4pro-2node.sh`
8. Existing failure summary:
   `/data/ycfeng/tmp/b300_step4_smoke_20260814/entrypoint_failures_summary.txt`

Do not edit the two pinned recipe files. If adaptation is required, create a
wrapper under:

```text
tests/e2e/step4_pro_latest/
```

## 4. Reusable Existing Evidence

The following work has already been completed and should not be repeated
unless the environment has materially changed:

- B300 `predict-only`: PASS, exit code `0`, **10 candidate 8-GPU nodes**.
- Observed GPU:
  - name: `NVIDIA B300 SXM6 AC`;
  - memory: `275040 MiB`;
  - driver: `580.159.03`;
  - compute capability: `10.3`.
- At handoff, there were no active `step4pro-b300*` RJobs, Replicas, or local
  smoke processes.

Reusable evidence directory:

```text
/data/ycfeng/tmp/b300_step4_smoke_20260814/
```

Important files:

| File | Purpose |
|---|---|
| `predict_corrected.log` | Successful B300 capacity prediction |
| `pinned_inputs_final.sha256` | Hashes of the pinned scripts |
| `pinned_payload_sha256.txt` | Hashes of scripts and key vLLM source files |
| `single_remote_source.sh` | Remote body extracted from the pinned single-GPU script |
| `single_remote_adapted.sh` | Diagnostic draft only; do not run unchanged |
| `live_single_rlaunch.log` | Default image-entrypoint failure |
| `entrypoint_override_launch.log` | Double-Bash entrypoint failure |
| `entrypoint_failures_summary.txt` | Concise root-cause summary |

Pinned hashes at handoff:

```text
5f52554677bbdb96b38a851eb189b1ccc817c3b6fd688583ca824e85a6bda092  rjob-step4pro-optimus-single.sh
a0e64fd646293264d66727af455785d1e877d6c0e0a6c849f5e1077f900387bb  rjob-step4pro-2node.sh
bbfa147d0d2e08b4c7f602b9ff6609503b9f79ef4262dddb389047b9ad37dd0c  vllm/model_executor/models/step4pro.py
2d0c2dbb0b16d1a0ebd1a3009c65c042a94d4a52cc6ce38d90c45d1c1c7e2be8  vllm/v1/attention/backends/optimus_fa4.py
6fb061268a63235a8ab6ae408096beefed7f53c68f4442e81deccdeb70b8276b  vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py
```

The old two-node hash above is the base-commit copy. Under the authorized
DeepEP launch commit, the exact script hashes are:

```text
b9e7fb070aa654c6341518cf9962c7d46a5c40d96650c016f8f04afd5e604bf5  rjob-step4pro-2node.sh
2d823b17ff0d216ac291d92d9053cd86b2d40917aafc758bd7cfedad2b29888f  rjob-step4pro-deepep-probe.sh
```

Recompute and compare the applicable hashes before reusing the artifacts.

## 5. Environment and Safety Rules

### 5.1 B300 launch rules

- Use `--charged-group=b300_train_infra`.
- Use `--private-machine=group`.
- Use `--positive-tags=B300`.
- Use `--backoff-limit=1` or another positive value.
- Run `--predict-only` before a large allocation.
- The B300 zone differs from the workspace zone:
  - an explicit `--image` is mandatory;
  - workspace `--volume` mounting is forbidden;
  - do not assume `/data/ycfeng` is visible in the worker.
- Do not override platform-injected `NCCL_*` variables.
- Multi-node execution must preserve the distributed environment supplied by
  the platform.

The pinned scripts contain explicit JuiceFS model/tokenizer mounts. They may be
used only through the script/client path that supports them; they are not the
forbidden workspace NFS `--volume` mount.

### 5.2 Command and monitoring limits

- Wrap every `brainctl`/`rlaunch` inventory or diagnostic query with:

  ```bash
  sudo -n systemd-run --scope -p MemoryMax=3G \
    timeout --signal=TERM --kill-after=5s 60s \
    <command>
  ```

- Put `timeout` **inside** `systemd-run --scope`. Placing it outside can leave
  an orphaned process in the transient scope.
- Use exact-name RJob queries and
  `rjob.brainpp.cn/rjob-name=<exact-job-name>` Replica selectors. Do not run a
  namespace-wide Replica inventory.
- Keep large I/O disk-backed and streamed under `/data/ycfeng/tmp`; do not
  load Git packs, bundles, tar archives, or large logs into shell variables,
  command substitutions, or whole-file Python reads.
- For live launch, use direct `setsid sudo -n systemd-run` rather than an
  additional `bash -c` wrapper.
- During cleanup, explicitly delete and poll the RJob/Replica first, then
  terminate and wait for the local launcher process group.
- Give each live smoke an overall hard timeout.
- Report status at least every 10 minutes.
- Never wait indefinitely for an agent, RJob, health endpoint, or request.
- Store host artifacts under `/data/ycfeng/tmp`, never under host `/tmp`.
- First probe the worker for a writable, sufficiently large directory. The
  current image exposed `/jobutil` as read-only in the most recent live
  launches, so do not assume it is writable.
- Do not run `single_remote_adapted.sh` unchanged: it writes to `/jobutil`.

### 5.3 Source transport

Workspace mounting is unavailable cross-zone. The already verified generic
transport is tar over `brainctl exec`:

```bash
tar cf - -C <LOCAL_PARENT> <LOCAL_NAME> \
  | /kubebrain/brainctl -n shai-core exec -i replica/<REPLICA> -- \
      tar xf - -C <VERIFIED_WRITABLE_REMOTE_PARENT>
```

For worker-to-host output, first save the tar stream to a file and then
extract it:

```bash
/kubebrain/brainctl -n shai-core exec replica/<REPLICA> -- \
  tar cf - -C <REMOTE_PARENT> <REMOTE_OUTPUT_DIR> \
  > /data/ycfeng/tmp/<output>.tar

tar xf /data/ycfeng/tmp/<output>.tar -C <LOCAL_DEST>
```

Directly piping the worker output stream into `tar xf -` previously failed and
must not be repeated.

## 6. Known Failed Launches

Do not repeat these attempts:

| Failed attempt | Observed error | Root cause |
|---|---|---|
| Default image entrypoint with an ordinary shell command | `signal_proxy` received an empty command and raised `list index out of range` | The image entrypoint consumed/lost the platform command |
| `--entrypoint /bin/bash` followed by `bash -lc ...` | `/usr/bin/bash: cannot execute binary file`, exit `126` | Bash was supplied twice |
| Creating output under `/jobutil/...` | `Read-only file system`, exit `1` | `/jobutil` was not writable in that worker configuration |

The verified command-passing form is:

```bash
--entrypoint /bin/bash -- -lc '<COMMAND>'
```

Do not prepend another `bash` after `--`.

## 7. Execution Procedure

### Stage 0 — Preflight

1. Verify the pinned Git commit and script hashes.
2. Run:

   ```bash
   bash -n vllm-step4-pro/rjob-step4pro-optimus-single.sh
   bash -n vllm-step4-pro/rjob-step4pro-2node.sh
   ```

3. Confirm no prior `step4pro-b300*` jobs or replicas exist.
4. Reuse the existing successful prediction when it is still current. If a
   fresh allocation check is needed, run one bounded `predict-only`; do not
   repeat multiple equivalent probes.
5. Record the exact `rlaunch`, `brainctl`, Python, and image identities.

### Stage 1 — Writable-path and source-identity probe

Launch one bounded single-GPU worker with the correct entrypoint form. The
probe must:

1. print `nvidia-smi`;
2. print mounts and filesystem capacity;
3. test candidate directories for write access;
4. remain alive long enough for a bounded `brainctl exec`;
5. receive the pinned source through a verified mechanism;
6. print the actual source identity:

   ```bash
   git -C <REMOTE_VLLM_REPO> rev-parse HEAD
   sha256sum \
     <REMOTE_VLLM_REPO>/vllm/model_executor/models/step4pro.py \
     <REMOTE_VLLM_REPO>/vllm/v1/attention/backends/optimus_fa4.py \
     <REMOTE_VLLM_REPO>/vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py
   ```

7. prove Python imports the intended source:

   ```python
   import inspect
   import vllm
   from vllm.model_executor.models import step4pro
   from vllm.v1.attention.backends import optimus_fa4

   print("vllm_file", vllm.__file__)
   print("step4pro_file", inspect.getsourcefile(step4pro))
   print("optimus_fa4_file", inspect.getsourcefile(optimus_fa4))
   ```

If these paths or hashes do not match the pinned source, stop and report the
source-delivery root cause. Do not call the smoke valid based only on the image
tag or `VLLM_PULL_COMMIT` environment variable.

### Stage 2 — One-GPU 14-layer synthetic smoke

Base the provider and serving arguments on:

```text
vllm-step4-pro/rjob-step4pro-optimus-single.sh
```

The real checkpoint is not required. Create a reduced 14-layer Step4Pro
synthetic `config.json` that contains Full MFA head-dim 512, native SWA GQA,
Dense layers, Latent MoE layers, and FP8 block quantization. Start it with:

```bash
vllm serve <SYNTHETIC_CONFIG_DIR> \
  --load-format dummy \
  --skip-tokenizer-init
```

Submit token-ID requests. Do not require meaningful generated text from random
weights.

Required source/runtime overrides:

```bash
VLLM_PULL_COMMIT=607d1641ee3fec43653fca510d717725828890c2
IMAGE=hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled
CHARGED_GROUP=b300_train_infra
POSITIVE_TAGS=B300
```

Retain the recipe's:

- `step-optimus==3.23.24` overlay and checksum;
- `VLLM_KV_CACHE_LAYOUT=NHD`;
- `OPTIMUS_MUST_LOAD_LIB=1`;
- block size `128`;
- `--enable-expert-parallel`;
- `--enforce-eager` for smoke;

Do not mount `qy1-pt` or `step2-alignment-jfs` for the synthetic path.
- `/health` polling;
- completion, chat, and four-concurrent-request checks.

The smoke is PASS only if all of the following are true:

1. One B300 is visible with its actual memory value.
2. Runtime source files match the pinned commit/hashes.
3. `Step4ProForCausalLM` is registered and the model loads.
4. `step-optimus` reports version `3.23.24`.
5. The Optimus native RMSNorm operation is registered.
6. The FA4 Python provider imports from the expected path.
7. Server `/health` returns success before the bounded deadline.
8. The server log proves Optimus FA4 selection for the hd512 Full-MFA path.
9. No Triton/generic-attention fallback is used for that Full-MFA path.
10. At least one request performs prefill and decode successfully.
11. The four concurrent completion requests all succeed.
12. The service and RJob are stopped cleanly.

Record numeric values for:

- worker scheduling time;
- model loading time;
- health-ready time;
- GPU memory before and after model load;
- each request latency;
- four-request wall time;
- prompt and generated token counts where available.

### Stage 3 — Runtime/provider trace

For the successful one-GPU request, capture a provider matrix containing:

| Logical path | Required runtime evidence |
|---|---|
| Full MFA Attention | `optimus_fa4.py` import path, FA4 backend-selection log, hd512, page size 128, and successful request |
| Full MFA K/V | evidence that K/V share storage and use the NHD/paged layout expected by the pinned provider |
| Grouped `wo_a` | call/provider evidence for the pinned grouped/einsum implementation; do not relabel a dense GEMM |
| Routed MoE | `optimus_fp8_moe.py` import path and selected Optimus/DeepGEMM execution path |
| Router/projections | actual dtype/shape evidence when emitted by the runtime trace |

On B300 SM103, apply the owner-authorized runtime overlay only after the
`2103/2103` pinned-source check:

```text
torch.ops.Optimus.per_token_group_quant_fp8
    -> optimus_cutedsl.group_quant_fp8.per_token_group_quant_fp8
```

The overlay emits `sm100_1d1d` packed UE8M0 scales and sets
`VLLM_USE_DEEP_GEMM_E8M0=1`, because the fixed DeepGEMM 2.4.2 Blackwell
runtime rejects FP32 activation scales. It is required for this pinned
B300/image combination, not for the Step4Pro architecture in general.

The quant overlay preserves shapes, routing, experts, attention, and
DeepGEMM provider identity, but it changes the activation-quant implementation
and scale representation. It is therefore valid evidence for execution and
performance, not for bitwise output, generation quality, loss, or training
convergence. The normal long-term repair is a matching SM103 Optimus native
build for the image's torch/CUDA ABI.

After the pinned-source check, also apply the reviewed inference-only
`ep_gather` block correction:

```text
min(hidden_size, 1024)
    -> min(1024, hidden_size & -hidden_size)
```

This preserves exact dimension coverage and top-k accumulation order.
Previously legal dimensions keep their original block; 896 and 3584 become
legal. The function is guarded by `torch.no_grad()` and is not part of a
training graph, so it does not affect weights, routing training, gradients, or
optimizer behavior. Preserve its unified diff and before/after hashes as
separate evidence.

Apply the same reviewed selector correction to an isolated copy of
`optimus_triton/deep_gemm_ep_gather_masked.py`; do not edit the installed
system package. Preserve a separate package-overlay diff and hashes.

Set `OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE=1` for changing batch shapes. The
default weak signature omits tensor shape and can reuse an invalid driver-cache
entry between batch 1 and batch 4.

Also record these non-model changes:

- the two CuTe pointer-API edits inherited from the pinned smoke recipes;
- the `flash_attn.__spec__` import-metadata repair;
- the isolated masked-gather overlay rather than an installed-package edit;
- the Full-MFA-only provider assertion, which leaves legal SWA provider
  selection unchanged.

The report must preserve the exact overlay diff and must still prove that the
expert GEMMs use `OptimusFp8Experts` and the pinned DeepGEMM kernels. This is
not permission to change any other provider or operation.

Backend-selection logs plus runtime module paths are mandatory. If a profiler
is used, also preserve the raw trace and list the provider-relevant kernel
names. A generic kernel with the same tensor shape is not acceptable evidence.

### Stage 4 — Two-node DP16/EP16 DeepEP smoke

Base the execution on:

```text
vllm-step4-pro/rjob-step4pro-2node.sh
```

Retain:

```text
replicas = 2
GPUs per replica = 8
DATA_PARALLEL_SIZE = 16
DATA_PARALLEL_SIZE_LOCAL = 8
tensor parallel size = 1
VLLM_ALL2ALL_BACKEND = deepep_high_throughput
```

`brainctl exec` does not inherit the holder shell's distributed environment.
The holder must first write a quoted mode-0600 environment file under `/home`;
the host must poll that file with a bounded readiness deadline before starting
the remote smoke.

Use this role split:

```text
DATA_PARALLEL_START_RANK = 0:
    normal serving head

DATA_PARALLEL_START_RANK > 0:
    --headless --api-server-count 0
```

The headless node must not run health or HTTP request checks. It must remain
attached to the coordinated DP group and prove DeepEP HT manager,
dispatch/combine, and real-batch markers before being accepted.

Before the full model, require a 16-rank NCCL preflight with:

```text
host network
rdma/mlnx_shared=8
mellanox.com/mlnx_rdma=1
non-empty platform NCCL_IB_HCA
all-reduce sum 1..16 = 136
```

Reset `LD_LIBRARY_PATH` to the pinned CUDA 13 compatibility path after
worker-init; its CUDA 12.8 compat path produces CUDA error 803 with torch
cu129.

NCCL PASS does not replace the DeepEP gate. The available launcher currently
lacks the later branch's `--share-host-shm=True` option. A complete result
still requires the standalone DeepEP Buffer probe and then the full model to
pass; do not infer DeepEP dispatch/combine from NCCL alone.

The supplemental RJob guide also permits the legacy process launcher:

```text
brainctl launch --i-know-i-am-using-legacy-rlaunch
```

That path has now been tested. It completed NCCL and explicit NVSHMEM
initialization on all 16 ranks, but `deep_ep.Buffer` failed the assertion
`nvshmem_n_pes() == num_ranks`. Therefore neither the legacy launcher nor
explicit NVSHMEM initialization alone is sufficient, and shared-host-SHM must
remain a hypothesis rather than a claimed root cause.

The owner has now authorized testing the complete later-commit contract. Apply
`9bfd9a610e`, verify a launcher that genuinely accepts
`--share-host-shm=True`, run matching predict-only, and then run
`rjob-step4pro-deepep-probe.sh` before the complete model. The committed probe
writes worker artifacts under `/tmp`; the execution copy must use a verified
writable disk-backed path instead. This path-only safety change does not relax
the required NVSHMEM, RDMA, host-network, or DeepEP settings.

Current controller result:

```text
exact checkout/static scope: PASS
rjob-mode shared-host-SHM parse: FAIL, exit 1
legacy-mode shared-host-SHM parse: FAIL, exit 1
standalone rjob client: absent
```

Both installed modes report `unknown flag: --share-host-shm`. Stop before
predict-only or live allocation until the standalone client used by the commit
or a documented equivalent API is provided.

The two-node smoke is PASS only if:

1. Both replicas become ready.
2. Every rank reports the expected B300 count and distributed variables.
3. The actual runtime imports the pinned Step4Pro source.
4. The engine reports `deepep_high_throughput`.
5. Runtime evidence distinguishes DeepEP dispatch from combine.
6. At least one prefill/decode request succeeds.
7. No backend fallback, deadlock, rank loss, or unclassified CUDA error occurs.
8. Both replicas and the RJob are cleaned up.

Record:

- replica scheduling/readiness times;
- rank and DP/EP mapping;
- health-ready time;
- request latency;
- dispatch and combine provider evidence;
- GPU memory per rank when available.

### Stage 5 — Cleanup

After every attempt:

1. stop/delete the exact RJob created by the attempt;
2. query both RJobs and Replicas with bounded commands;
3. assert that no matching resource remains;
4. check local `rlaunch`, `brainctl`, `vllm serve`, and smoke processes;
5. save the cleanup output.

## 8. Failure Policy

- Fail fast on source mismatch, provider fallback, missing model/tokenizer
  mount, unwritable evidence path, health timeout, request failure, or leaked
  resources.
- Do not change the model shape, backend, precision, page size, or image to
  obtain a pass.
- Do not replace Optimus FA4 with generic FlashAttention.
- Do not replace grouped `wo_a` with dense GEMM.
- Do not replace DeepEP HT with another collective backend.
- Do not use H800 as a substitute.
- Do not introduce retry/fallback code that hides the first root cause.
- After three distinct failed approaches for the same blocker, stop and report
  the blocker instead of continuing to consume B300 resources.

## 9. Required Deliverables to the AIC Session

Create:

```text
task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-14_b300_pinned_vllm_smoke_runtime_trace.md
```

The report must include:

1. **Test Script Information**
   - full script paths;
   - exact reproducible commands;
   - image tag and digest;
   - Python, torch, CUDA, vLLM, step-optimus, driver, and GPU versions.
2. **Validation Criteria**
   - each PASS/FAIL condition from stages 2 through 4.
3. **Results and Evidence**
   - one-GPU smoke status;
   - two-node smoke status;
   - source-identity status;
   - FA4, grouped `wo_a`, MoE, and DeepEP provider status;
   - all required numeric timing and memory values;
   - exact raw log/trace paths and SHA256 hashes;
   - failure stack traces and root-cause analysis, if any.
4. **Cleanup Evidence**
   - final RJob count;
   - final Replica count;
   - final local related-process count.

Return a concise handoff matrix:

| Gate | PASS/FAIL | Key numeric result | Evidence path |
|---|---|---:|---|
| Pinned source identity |  |  |  |
| One-GPU health |  | health-ready seconds |  |
| Prefill/decode request |  | request latency |  |
| Four-request concurrency |  | wall time |  |
| Optimus FA4 provider |  | hd512/page128 |  |
| Grouped `wo_a` provider |  | groups/shape |  |
| Optimus FP8 MoE |  | experts/top-k |  |
| Two-node DP16/EP16 |  | ready seconds |  |
| DeepEP HT dispatch |  | measured/trace value |  |
| DeepEP HT combine |  | measured/trace value |  |
| Resource cleanup |  | remaining jobs/processes |  |

The AIC session will use this report as external provider-validation evidence.
It must not infer a PASS from missing or partial logs.
