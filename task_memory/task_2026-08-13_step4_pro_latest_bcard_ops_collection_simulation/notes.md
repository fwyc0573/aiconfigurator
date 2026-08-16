## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Recorded initial workspace, environment, and execution constraints. |
| 2026-08-13 | Added measured access facts: B300 quotagroup denial, the only grantable GPU path, Docker-via-sudo requirement, and shell cwd persistence. |
| 2026-08-13 | Revalidated B300 access and latest vLLM source/image identity after new platform evidence. |
| 2026-08-13 | Recorded user choice A: pinned local checkout and requirements shape are authoritative. |
| 2026-08-13 | Recorded the owner's no-action instruction for the temporary security-review file. |
| 2026-08-14 | Recorded the 3 GiB controller scope and low-memory I/O rules after a host OOM. |
| 2026-08-15 | Corrected the active branch note and recorded the provider-specific Collector data boundary. |
| 2026-08-15 | Recorded the approved runtime-deviation scope and the distinction between performance validation and model/training validation. |
| 2026-08-15 | Recorded the two-node `brainctl exec` environment boundary and the holder-file restoration rule. |
| 2026-08-15 | Recorded the passing full-RDMA NCCL contract and the unavailable shared-host-SHM requirement for DeepEP Buffer sync. |
| 2026-08-15 | Recorded the AIC DeepEP HT full-RDMA/NCCL preflight rule after two EP16 live attempts. |
| 2026-08-15 | Recorded the local DeepEP construction pass, explicit-destroy failure, and resulting no-retry gate. |
| 2026-08-15 | Recorded the byte-identical supplied brainctl, legacy launcher retry, and DeepEP/NVSHMEM PE-count mismatch. |
| 2026-08-15 | Recorded the bounded authorization to test the exact later DeepEP launch scripts and the worker disk-path adaptation requirement. |
| 2026-08-15 | Recorded restoration to pinned commit `607d1641ee` after the bounded DeepEP-only commit test and closure of grouped/router 65K coverage. |
| 2026-08-16 | Recorded complete SWA QKV measurement, the bounded annotation overlay, and the DeepEP-only no-retry state. |

# Operational Notes

- Repository: `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro`.
- Current branch: `task/step4-pro-latest-b300`.
- The worktree is already dirty with substantial changes from earlier tasks. No reset, bulk replacement, `rm`, or `mv` is authorized.
- This task has its own directory: `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation`.
- The local host is a CPU master unless `nvidia-smi` proves otherwise. GPU work must follow `/data/ycfeng/stepfun-env-handbook/guidence.md`; Docker work must follow `/data/ycfeng/stepfun-env-handbook/docker.md`.
- Large GPU allocations require a verified `rlaunch --predict-only` first and `--backoff-limit>0`.
- Do not write temporary files, logs, or caches to `/tmp`; use `/data/ycfeng/tmp`.
- Latest provider rows must not share generic physical keys with stock vLLM
  GEMM, attention, MoE, or communication rows. Missing exact provider data
  must fail rather than select a generic table.
- QKV is complete: Full-MFA `75/75` plus SWA `75/75`, canonical
  `150/150`, duplicate keys `0`, and exact silicon consumers `150/150`.
- The SWA-only process-local overlay must first verify QKNorm source SHA256
  `5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783`
  and may change only the `reload_from` and `delay_w_load` annotation objects.
  It must never rewrite installed source or alter provider, kernel, shapes,
  dtypes, QKV math, or persisted keys.
- DeepEP remains frozen at `0/116`. Do not retry it or fill its four EP16/EP32
  dispatch/combine contracts with H800, generic communication, analytic, or
  synthetic rows.
- Use `MemoryMax=3G` for every remaining controller-side `brainctl` and live
  launcher scope. Do not raise it back to 5 GiB without explicit owner
  approval.
- Query only the exact RJob name or the exact
  `rjob.brainpp.cn/rjob-name=<job>` Replica label. Namespace-wide Replica
  inventory is forbidden because it previously exhausted a bounded memory
  scope.
- Keep large I/O disk-backed and streamed: redirect `git diff`, manifests,
  logs, tar streams, and evidence directly to files under `/data/ycfeng/tmp`.
  Do not load Git packs, bundles, tar archives, or large logs into shell
  variables, command substitutions, or `Path.read_bytes()`.
- The one-GPU runtime is intentionally not byte-identical after source
  verification. Preserve every overlay diff/hash and describe it as a pinned
  source plus explicit runtime deviations.
- Dummy/random weights validate execution, provider selection, timing, and
  memory. They do not validate text quality, loss, MTP acceptance rate, expert
  routing distribution from a trained checkpoint, or training convergence.
- The `ep_gather` fixes are inference tiling corrections. The Optimus JIT
  quant overlay changes inference quantization implementation/scale encoding;
  do not collapse these two impact classes into one blanket “no numerical
  change” statement.
- `brainctl exec` starts a new process and does not inherit the holder shell's
  platform-distributed environment. The holder must write a mode-0600
  `/home/step4pro-distributed-<job>.env` file before sleeping; every remote
  execution must source that file and fail fast if required variables are
  absent.
- Preserve, rather than invent, platform `NCCL_*` and `NVSHMEM_*` values.
  Two-node allocation also requests `rdma/mlnx_shared=8`, topology grouping,
  and the later script's non-NCCL
  `NVSHMEM_ENABLE_NIC_PE_MAPPING=1` bootstrap setting.
- Coordinated multi-node DP uses one API-serving head only. Rank 0 serves
  HTTP; every node with `DATA_PARALLEL_START_RANK>0` must use
  `--headless --api-server-count 0`.
- Three distinct live two-node failures have consumed the execution allowance
  in the external guide. Do not launch a fourth B300 attempt without explicit
  owner approval, even though the latest role fix passes static validation.
- The owner-authorized fourth run proved the role split and Gloo connectivity,
  then failed NCCL initialization with `NCCL_IB_HCA=''`. Do not hardcode an
  HCA, disable IB, or use socket fallback. The next run requires a launcher
  that exposes the B300 branch's full RDMA/host-network contract and lets
  worker-init inject a non-empty HCA.
- `--host-network=true` plus both RDMA resources injects eight bond HCAs and
  makes 16-rank NCCL pass. The image's worker-init also prepends a stale CUDA
  12.8 compat path; reset `LD_LIBRARY_PATH` to the pinned
  `/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64` value before torch.
- NCCL success is not sufficient for DeepEP. The complete run reached
  `DeepEPHTAll2AllManager` and then failed `deep_ep.Buffer.runtime.sync`.
  The later branch's standalone probe requires `--share-host-shm=True` and
  explicit NVSHMEM bootstrap. Current brainctl/rlaunch exposes no shared-host
  SHM option; do not emulate it with an invented volume or privileged mode.
- The one-node `num_rdma_bytes=0` discriminator constructed the DeepEP Buffer
  on `8/8` ranks, then failed explicit `buffer.destroy()` on `8/8` ranks at
  `runtime.cu:29` with CUDA `999`. This narrows the EP16 construction failure
  to the cross-node path but does not make the local runtime an overall PASS.
  Do not retry EP16/EP32 until shared-host SHM or a documented vendor/runtime
  fix is available.
- The supplied brainctl and `/kubebrain/brainctl` are byte-identical
  (`sha256:06d5fffb00e67633e10e4a6d96752517eda7559230466a63ac86e6a424c839ad`);
  only their executable mode differs. The supplemental docs point to
  `brainctl launch --i-know-i-am-using-legacy-rlaunch`, not to a new
  `--share-host-shm` flag.
- Legacy launcher final probe: NCCL `16/16`, explicit NVSHMEM init `16/16`,
  DeepEP Buffer `0/16`. Every Buffer constructor asserted
  `nvshmem_n_pes() == num_ranks`. This is evidence of a DeepEP/NVSHMEM runtime
  integration mismatch; shared host SHM is not a proven root cause.
- Owner direction on 2026-08-15 reopens one exact test of direct-child commit
  `9bfd9a610e`. That commit changes only two launch scripts. Preserve its
  explicit NVSHMEM initialization, NIC-to-PE mapping, bootstrap socket
  interface, full RDMA/host-network resources, and shared-host-SHM request.
  Its `/tmp` worker paths must be adapted to a verified disk-backed path before
  execution; this safety adaptation is not a DeepEP runtime fix.
- Exact launcher check on 2026-08-15: current brainctl is
  `v2.12.0-alpha.4.1328-20260814040139-9da5d7fa9435`; both
  `brainctl rjob launch` and legacy `brainctl launch` reject
  `--share-host-shm` at argument parsing with exit code `1`. No standalone
  `rjob` binary exists. Stop before predict-only/live submission.
- After recording that bounded DeepEP launcher result, restore
  `vllm-step4-pro` to authoritative commit
  `607d1641ee3fec43653fca510d717725828890c2` before any provider measurement.
  This restoration is complete. Grouped/router now include measured
  `65536`-token rows; generic GEMM remains capped at `32768`.
- The same environment defect reproduced in AIC DeepEP HT run
  `s4p-aic-deepep-16-0815-191949`. Before another operation smoke, align both
  predict-only and live launches with the pinned branch by requesting
  `rdma/mlnx_shared=8`, `mellanox.com/mlnx_rdma=1`, and host networking. Run
  only the Q16-authorized NCCL preflight first. It must reject an empty
  platform HCA and record a real NCCL all-reduce; it must not set
  `VLLM_DISABLE_PYNCCL` or construct an EP-only substitute.
- Before Collector case changes, read `.agents/skills/aic-collector-op-development/SKILL.md`; before generator changes, read the applicable `.claude/rules/generator/**` rule.
- The referenced requirements document is `task_memory/step4pro_v4_external_simulator_requirements.md` and currently specifies B300 plus vLLM commit `607d1641ee3fec43653fca510d717725828890c2`.
- That requirements document links `step4pro_v4_shape_manifest.json` and `step4pro_v4_vllm_b300_8bit_task.md`, but those two files are not present at the expected `task_memory/` paths in the current checkout.
- The user supplied the source checkout path `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro`.
- The supplied checkout is detached at commit `607d1641ee3fec436fca510d717725828890c2`, matching the requirements document's pinned commit.
- User decision on 2026-08-13: the requirements-pinned local checkout and 78-layer shape are authoritative; the later branch head is reference-only.
- The pinned checkout's two B300 smoke scripts both select image `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`; inspected manifest digest: `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`.

## Measured access facts (2026-08-13)

- `nvidia-smi` confirms this host has no GPU. It is a CPU master.
- The earlier B300-denial conclusion is superseded. `b300_train_infra` is grantable and has passed single-node and multi-node predict-only checks; see resolved ISSUE-002.
- B300 is cross-zone. Launches must specify the pinned image, must not use `--volume`, and must move inputs/outputs through `tar | brainctl exec`.
- Docker requires `sudo`: the invoking user is not in the `docker` group, but `sudo -n docker` reaches the daemon. The `artifactory.stepfun-inc.com` mirror is already configured in `/etc/docker/daemon.json`, and `data-root` is `/data/var/lib/docker`.
- `hub.stepfun-inc.com` rejects unauthenticated catalog reads, so image tags cannot be enumerated from this host.
- Bash tool calls share a persistent working directory across invocations. Use absolute paths, or a stale `cd` from a previous call will make a relative path fail.
- Node listings from `brainctl get nodes` in plain-text form return a sample, not the full set. Use `-o json` with a label selector when an exact inventory is needed.
- Per explicit owner instruction, do not read, remove, alter, move, quote, or
  otherwise process
  `/data/ycfeng/tmp/aic_failure_domain1/codex_lane3_scope_baseline.md`.

## Revalidated runtime facts (2026-08-13)

- The vLLM repository is now reachable over SSH. The branch
  `xwx/step4pro-fa3-optimus` resolves to
  `9bfd9a610ea4f2890010702ee7a207cf25edf8de`.
- The requirements-document commit
  `607d1641ee3fec43653fca510d717725828890c2` is an ancestor of that branch
  head. The latest commit adds B300 DeepEP/NVSHMEM bootstrap changes.
- The branch scripts identify the B300 runtime image as
  `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`.
  Read-only registry inspection resolved manifest digest
  `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`
  and config digest
  `sha256:48253062026862b2921f354613d3e9a7255e9540e3a7bf9148c9c80ec2d72661`.
- The source checkout contains `step4pro.py`, `optimus_fa4.py`,
  `optimus_fp8_moe.py`, and the Step4Pro model registry entry. Choice A makes
  this checkout, its declared B300 image, and the requirements' 78-layer shape
  the authoritative task contract.
- B300 predict-only succeeds with `b300_train_infra`; a 2-replica, 8-GPU
  per-replica request returned candidate B300 nodes. This supersedes the
  earlier incomplete quota probe.
