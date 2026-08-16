## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Recorded evidence from the initial repository and requirements audit. |
| 2026-08-13 | Added measured cluster, credential, image, and AIC-shape evidence. Replaced the inconclusive local-cache note with a measured inspection result. |
| 2026-08-13 | Added verified latest branch head, image digests, source-file presence, and B300 predict-only evidence. |
| 2026-08-13 | Recorded choice A and the pinned Step4Pro source-to-logical-op inventory. |
| 2026-08-13 | Added the vLLM DeepEP consumer/Collector audit and the manifest reconstruction decision. |
| 2026-08-13 | Added the pinned-vLLM versus current-AIC fidelity-gap audit. |
| 2026-08-15 | Began the provider-attention contract audit after completing QKV preprocessing. |
| 2026-08-15 | Recorded the distinction between logical KV, resident physical KV, and monotonic peak physical KV for simulator capacity checks. |
| 2026-08-15 | Recorded the pinned Optimus MoE custom-op boundary and why its contiguous path must be measured eagerly. |
| 2026-08-15 | Added the pinned-image and pinned-repository DeepEP launch example/case inventory. |
| 2026-08-15 | Recorded corrected Attention coverage and the numeric MTP-off matrix outcome. |
| 2026-08-15 | Recorded Full-MFA QKV B300 completion and the reduced SWA-QKV/DeepEP coverage gap. |
| 2026-08-16 | Recorded complete SWA QKV measurement, canonical QKV validation, and the DeepEP-only formal gap. |
| 2026-08-16 | Recorded the B300 NCCL proxy data, dtype, topology, and message-volume contracts. |
| 2026-08-16 | Recorded the second scheduler-derived Attention workload gap revealed after DeepEP proxy unblocked later chunks. |
| 2026-08-16 | Recorded the completed MTP-off proxy matrix and its prefill/decode capacity findings. |

# Findings

## 2026-08-16 — Completed MTP-off proxy matrix

- The final explicit-proxy matrix completed all `72` prefill and `84` decode
  records with `0` missing or error records. Its top-level status is
  `PASS_WITH_PROXY`, and every result carries `result_fidelity=PROXY`.
- Prefill formal latency spans
  `129.22378441076575–782420.0031436655 ms`; per-replica input throughput
  spans `1340.1293369125042–69183.9264811136 token/s`; aggregate input
  throughput spans `1340.1293369125042–103512.27987036602 token/s`.
- Prefill peak HBM spans `142.00439453125–776.97998046875 GiB`.
  `48/72` records exceed the `241.734375 GiB` utilization limit and are
  explicitly classified as `OOM`; the remaining `24/72` are
  `PASS_WITH_PROXY`.
- Decode batch-1 steady step/TPOT spans
  `56.955545921130614–268.6990437660492 ms`; batch-1 HBM spans
  `138.58154296875–294.77685546875 GiB`.
- Every one of the `84` topology/context/budget combinations has
  `B_max=0`, `aggregate_B_max=0`, and `first_failed_batch=1`: even the fastest
  batch-1 proxy TPOT (`56.955545921130614 ms`) exceeds the loosest
  `33.33 ms` budget. This is a simulation result, not a missing-data blocker.
- The matrix consumed `134,976` proxy records with summed proxy latency
  `181471.55741956536 ms`. The proxy-only contribution must not be described
  as real DeepEP silicon performance.
- Evidence:
  `mtp_off_requirements_proxy_full_2026-08-16.json`, SHA256
  `370af7429c35036e30ce24bf5a0e6a57e1e91c834a02c1cb015e83229b9440a8`.

## 2026-08-16 — Proxy-unblocked Attention workload boundary

- `aic_silicon_coverage_proxy_2026-08-16.json` passed because its fixed probe
  set does not traverse every chunk emitted by the requirements scheduler.
- The full proxy matrix did traverse those chunks and returned `18/72`
  blocked prefill rows while decode completed `84/84`.
- Every blocker is one of four context Attention workloads, required for both
  `optimus_fa4` and `vllm_native_sliding_gqa`:
  `(batch=1, query=4096, total=8192)`,
  `(1,16384,32768)`, `(1,8160,1048544)`, and
  `(1,32736,1048544)`.
- The canonical context Parquet has `58` rows and contains `0/8` of those
  physical keys. The consumer correctly fails because the existing
  three-dimensional data does not bracket those workload coordinates.
- The correct repair is eight fresh pinned-provider B300 measurements. It is
  not an interpolation relaxation, proxy, or historical H800 substitution.
- Collector population after the local repair is `66` unique invocations:
  `58` kept, `8` added, `0` removed, and `0` deduplicated.

## 2026-08-16 — Complete QKV dataset and current formal gap

- Separate `qkv_full` and `qkv_swa` slices execute the actual pinned
  Full-MFA and SWA provider paths without substituting one for the other.
- Full-MFA collection passed `75/75` rows across `1–65,536` tokens with
  latency `0.029557332396507263–6.715685526529948 ms`.
- The accepted finite-value SWA smoke passed at
  `0.0052480002244313555 ms`. Full SWA collection passed `75/75` rows across
  `1–65,536` tokens with latency
  `0.006319999694824219–1.3165173530578613 ms`.
- The SWA runtime first verifies QKNorm source SHA256
  `5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783`,
  then resolves only `reload_from` and `delay_w_load` to
  `cutlass.Constexpr`. Pinned source, provider, kernel body, shapes, dtypes,
  argument values, QKV math, and persisted keys remain unchanged.
- Canonical Parquet contains `150` unique QKV physical keys with a `75+75`
  provider split. All `150/150` unchanged AIC consumer queries return exact
  silicon data with `0.0 ms` maximum absolute and relative error.
- Current scheduler-aware coverage is `36,420` records:
  `18,220` exact silicon, `15,160` analytic/non-silicon, `3,040` missing,
  and `0` errors.
- Only four physical contracts remain missing: DeepEP HT dispatch/combine at
  EP16 and EP32. DeepEP stays frozen at `0/116`.

## 2026-08-15 — Corrected Attention and formal matrix findings

- Exact row-consumer validation is necessary but not sufficient: the formal
  scheduler produced four extra Attention workload boundaries. After adding
  and measuring them, the canonical dataset is `58` context plus `167`
  generation rows.
- All `225` Attention rows have unique persisted keys and query exactly from
  canonical Parquet with `0.0 ms` maximum absolute error.
- After complete QKV collection, coverage is stable at four missing physical
  contracts: DeepEP HT dispatch/combine at EP16 and EP32.
- The full matrix executes even when formal latency is blocked. It computes
  every prefill schedule and every HBM curve, but stops timing queries after
  the first missing step to avoid presenting partial latency as a complete
  result.
- Prefill chunk counts range from `1` to `128`; because DeepEP is missing,
  only the first chunk is queried for timing and up to `127` later chunks are
  memory-only.
- The `0.9` memory-utilization limit is `241.734375 GiB`. `48/72` prefill
  workloads and `16/84` decode batch-1 records exceed it.
- Formal `B_max` cannot be derived while the first decode candidate is blocked.
  The reported decode latency range is known partial batch-1 latency, not TPOT.

## Confirmed from local files

- The supplied requirements document is `task_memory/step4pro_v4_external_simulator_requirements.md`.
- It names B300 as the target hardware and pins vLLM commit `607d1641ee3fec43653fca510d717725828890c2`.
- It requires synthetic 78-layer configuration, dummy weights, EP16/EP32/2×EP16, prefill/decode matrices, MTP-off/MTP1 comparisons, and numeric evidence.
- Historical v3/v4 work separates model support/figures from op profiling and emphasizes exact consumer-key and measurement provenance.
- The current checkout has extensive prior-task modifications; this task must not overwrite or reset them.

## Pinned Optimus MoE operation boundary

- `vllm/model_executor/layers/quantization/fp8.py:968-1040` selects
  `torch.ops.vllm.deepgemm_optimus_moe_masked_fp8` below
  `VLLM_OPTIMUS_MOE_MIN_CONTG_SIZE` and
  `torch.ops.vllm.deepgemm_optimus_moe_fp8` at or above it.
- The branch input is the rank-local token tensor shape, not the original
  global token count.
- `vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py:542-650` performs a
  CUDA-event host synchronization and reads `m_sum.item()` in the contiguous
  path. That path cannot be represented by a CUDA graph without changing the
  pinned implementation.
- The masked path is graph-compatible and must fail rather than silently use
  eager execution if capture does not succeed.
- The same provider module owns process-global workspaces through
  `_get_ws_size` and `_get_workspaces`; the Collector must initialize those
  buffers for the complete planned workload before timing.

## Measured cluster access (2026-08-13, from the CPU master)

- Local host has no GPU. `nvidia-smi` produces no device output; hostname is `ycfeng`.
- `/kubebrain/rlaunch` and `/kubebrain/brainctl` are present. `brainctl version` reports `v2.12.0-alpha.4.1260`.
- B300 hardware exists: `brainctl get nodes -l GPUType=B300 -o json` returns 20 nodes and 160 allocatable GPUs.
- All 20 B300 nodes belong to `b300_*` quotagroups: `b300_pretrain3` = 11, `b300_pretrain` = 5, `b300_pretrain2` = 2, `b300_sys_pro` = 1, `b300_train_infra` = 1. None is in `codesign`.
- All B300 quotagroups return HTTP 403 for this account on `quotagroups/predict` (user ID `e7163316ad76ba7266a33f9787afd70a`, project `shai/shai-core`).
- All B-card tag spellings return `no machine available`: `b300`, `b200`, `blackwell`, `b300_sxm`, `gb300`, `b300-sxm6`.
- Pinning `codesign` quota to a B300 node or to `feature/GPUType=B300` returns `no machine available`.
- The control case succeeds: `--charged-group=codesign --positive-tags=h800` returns multiple 8-GPU H800 nodes. The probe method is therefore sound, and the B300 failures are access failures rather than method failures.
- This matches the constraint recorded by the prior task at `task_memory/task_2026-07-28_step4_pro_v3_v4_op_profiling/notes.md:34`, which states that H100, H200, GB300, and Blackwell are not reachable.

## Measured credentials and runtime availability

- Docker is usable through `sudo`. The client and daemon are version 29.1.3, `data-root` is `/data/var/lib/docker`, and `/etc/docker/daemon.json` already configures the `artifactory.stepfun-inc.com` mirror. The invoking user is not in the `docker` group, so `sudo -n docker` is required.
- `gitlab.basemind.com` is reachable (port 22 open, HTTPS 302) but authentication fails. `git ls-remote git@gitlab.basemind.com:sys/stepcast/vllm.git HEAD` returns `Permission denied (publickey,password)`. `~/.ssh/` holds no private key.
- The internal registry `hub.stepfun-inc.com` responds (HTTP 200 at root) but rejects unauthenticated catalog reads: the tags endpoint returns `UNAUTHORIZED` and `/v2/` returns 401.
- The cached image `hub.stepfun-inc.com/stepcast/stepcast:2026-07-14-server-vllm-0.19.0.post15-8a8f1b3f` does not contain `step4pro.py` or `optimus_fa4.py`. It does contain `optimus_fp8_moe.py` and `optimus_moe.py`. Its `registry.py:218` maps `Step4ForCausalLM` to `("step3p5", "Step3p5ForCausalLM")`, and `Step4ProForCausalLM` is absent. This supersedes the earlier inconclusive local-cache note: the cached image is confirmed not to hold the target implementation.

## Measured AIC state relevant to this task

- AIC already defines a `b300_sxm` system at `src/aiconfigurator/systems/b300_sxm.yaml` with `sm_version: 103`, `mem_capacity` about 288 GB, and `mem_bw` 7.75 TB/s. Collected B300 data has an existing destination at `src/aiconfigurator/systems/data/b300_sxm/`, which already holds sglang, trtllm, and vllm subtrees. The vLLM subtree has `0.19.0` populated and `0.22.0` marked `SHARED_LAYER_REUSE.txt`.
- H800 reference data from the prior task lives at `src/aiconfigurator/systems/data/h800_sxm/vllm/0.19.0/`, including `measured_keys.json` with 180 identities and `step4_pro_v3_v4_coverage.json`. That coverage file reports, for `stepfun-ai/Step4-Pro-V4`, required and measured counts of attention 12, gemm 63, moe 7, communication 14, with zero missing and zero duplicates. Required op families are attention, gemm, moe, and communication.
- Existing Step4 models registered in `src/aiconfigurator/sdk/common.py:884-887` are `stepfun-ai/Step4`, `Step4-Pro-V1`, `Step4-Pro-V3`, and `Step4-Pro-V4`. The op graph is built in `src/aiconfigurator/sdk/models/step4.py`, whose Step4-Pro path emits per-layer `ElementWise`, `GEMM` (q/k/v/o projections), `ContextAttention` or `GenerationAttention` for full layers, `ContextMLA` for non-full layers, and `CustomAllReduce`.
- The prior task's collector entry point was `collector/collect.py --backend vllm --model-path ... --gpu h800_sxm --plan-only`, recorded at `task_memory/task_2026-07-28_step4_pro_v3_v4_op_profiling/progress.md:413`. That task pinned collection to `vllm/vllm-openai:v0.19.0` and treated the post15 StepCast image as diagnostic-only (`notes.md:42-43`).

## Missing or unresolved

- The requirements document links two files that are absent at the expected paths: `step4pro_v4_shape_manifest.json` and `step4pro_v4_vllm_b300_8bit_task.md`. The stated shape hash `53103019932b93b20a60b6f9dfe6154be6330befdd6a6fa2f6cb67278fc03fde` therefore cannot be verified.
- During the initial audit the two smoke scripts named in requirements section
  3.1 were absent from the then-available source. The subsequently supplied
  `vllm-step4-pro` checkout contains both scripts at pinned commit
  `607d1641ee3fec43653fca510d717725828890c2`; this supersedes the initial
  absence finding.
- No immutable identity for the "latest B-card vLLM image" appears in the current task request.
- It is not yet established whether "B card" means exactly the B300 contract in the requirements document or a newer B-card device/image.
- The requirements shape conflicts with the existing AIC `Step4-Pro-V4` on hidden size, layer count, expert count, topk, MTP layers, and full-attention type. See ISSUE-005 for the full comparison table.
- The prior task's exact collector command lines, including the rlaunch wrapper and image mount details, were not recovered in this pass. This is only needed once B-card access exists, so it is deferred rather than blocking.

## Revalidated latest-source evidence (2026-08-13)

- `git ls-remote` and a read-only clone succeeded for
  `git@gitlab.basemind.com:sys/stepcast/vllm.git`, branch
  `xwx/step4pro-fa3-optimus`. The current branch head is
  `9bfd9a610ea4f2890010702ee7a207cf25edf8de`, authored on August 13, 2026.
- The supplied commit
  `607d1641ee3fec43653fca510d717725828890c2` is an ancestor of the current
  head. The only commit after it changes
  `rjob-step4pro-2node.sh` and adds `rjob-step4pro-deepep-probe.sh` for
  B300/NVSHMEM bootstrap; it does not replace the Step4Pro model source.
- The latest checkout contains the required implementation files:
  `vllm/model_executor/models/step4pro.py`,
  `vllm/v1/attention/backends/optimus_fa4.py`, and
  `vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py`.
  `vllm/model_executor/models/registry.py:219` maps
  `Step4ProForCausalLM` to the Step4Pro implementation.
- The branch's B300 scripts use image tag
  `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`.
  Registry manifest inspection returned
  `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`;
  the image config digest is
  `sha256:48253062026862b2921f354613d3e9a7255e9540e3a7bf9148c9c80ec2d72661`.
- `/kubebrain/rlaunch --predict-only` with
  `--charged-group=b300_train_infra`, private-machine mode, two replicas,
  eight GPUs per replica, RDMA resources, and `DISTRIBUTED_JOB=true` returned
  B300 candidates. The earlier ISSUE-002 conclusion was caused by not probing
  every candidate quotagroup.
- The remaining unresolved decision is semantic rather than access-related:
  whether `step4-pro-latest` means the current branch/image implementation
  with the supplied requirements as the experiment envelope, or whether the
  older pinned commit and its 78-layer synthetic shape must be treated as the
  complete contract.

## Choice A and pinned-source operation findings

- The user selected the pinned checkout at commit
  `607d1641ee3fec43653fca510d717725828890c2` and the requirements' 78-layer
  shape. The later branch head is reference-only.
- The pinned smoke scripts themselves select the B300 image tag already
  inspected above, so the image tag/digest remains authoritative under choice
  A.
- The pinned model path is:
  `Step4ProForCausalLM -> Step4ProModel -> Step4ProDecoderLayer`.
  The decoder order is input RMSNorm, attention, residual add,
  post-attention RMSNorm, Dense/Latent-MoE FFN, residual add.
- Full layers use the low-rank shared-KV path:
  `wq_a -> q_lora_norm -> wq_b -> Q norm -> wkv -> K norm -> tail RoPE ->
  hd512 Attention -> optional inverse RoPE -> head gate -> grouped wo_a ->
  wo_b`.
- SWA layers use native GQA:
  packed QKV projection, Q/K/V norms, RoPE, window-512 attention, separate
  head-gate projection under pinned defaults, gate multiply, and output
  projection.
- Latent MoE uses the original hidden state for the FP32 router; routed tokens
  pass through BF16 hidden-to-latent projection, Optimus FP8 experts,
  post-projection norm and BF16 latent-to-hidden projection. The shared expert
  remains full-width BF16 and is added with the configured scale.
- Full/SWA attention custom kernels are implementation providers under logical
  Attention/GEMM/ElementWise boundaries. Optimus masked/contiguous DeepGEMM
  kernels are implementation providers under logical MoE. DeepEP dispatch and
  combine remain a communication-consumer gap requiring a dedicated audit.
- The pinned source does not contain a native Step4Pro MTP1 construction path.
  The requirements define its semantic composition, so this is an explicit
  implementation and validation gap rather than an unspecified requirement.
- Detailed read-only evidence:
  - `/data/ycfeng/tmp/step4_latest_vllm_ops.txt`
    (`sha256:7f3c43adf7c44d1cf8e6c54b44be76bb6506aa4a9dab2011194f693a0da2dc15`)
  - `/data/ycfeng/tmp/step4_latest_consumer_contract.txt`
    (`sha256:b595d0f1d73c0f58b4a66eb975d961457545ee804898d731f737e6f81caae855`)
  - `/data/ycfeng/tmp/step4_latest_scope_gate.txt`
    (`sha256:e3038632edb6ed1dfc69c645d10249e91b7734ffa4d3088657d958915b56ce60`)

## DeepEP consumer and Collector audit

- Current vLLM `MoEDispatch` queries CustomAllReduce/NCCL data and cannot
  consume DeepEP HT/LL performance rows.
- Existing WideEP loading is SGLang-specific. Its Python loader combines
  dispatch and combine, while Step4Pro requires separate pre/post
  communication timing, so direct reuse would double count.
- The pinned two-node runtime selects `deepep_high_throughput`; both prefill and
  decode must therefore use the HT identity unless runtime evidence shows a
  different selection.
- The minimal contract is:
  1. pass the vLLM `all2all_backend` explicitly;
  2. query dispatch and combine separately;
  3. add a vLLM-specific DeepEP Collector operation, registry entry, and
     framework-manifest route;
  4. reuse `OpEntry` rather than introducing a new registry dataclass.
- Proposed persisted identity:
  `(mode, operation, ep_size, ep_ranks_per_node, hidden_size, num_experts,
  topk, tokens_per_dp_rank, dispatch_format, num_sms,
  max_tokens_per_rank)`. Only `tokens_per_dp_rank` is interpolated.
- Evidence:
  `/data/ycfeng/tmp/step4_latest_vllm_deepep_contract.txt`
  (`sha256:41a2823c902d26353fc7d2a19b93237ff8815d985968d19844b21e7cc8065c2b`).

## Manifest decision

- The user selected reconstruction from the requirements and pinned source.
- The reconstructed artifact must carry a new SHA256 and must not claim the
  unavailable historical SHA256.

## AIC fidelity gaps

The read-only audit found four critical gaps:

1. The current Step4 schema cannot express Full MFA and SWA GQA in one model.
2. Full MFA is collapsed into ordinary Q/K/V/O Attention, losing low-rank Q,
   shared K/V storage, and grouped `wo_a`.
3. Latent MoE ordering/overlap differs from pinned execution, and current vLLM
   communication lookup does not model DeepEP dispatch/combine.
4. MTP1 is represented only by a uniform time scaling rather than an explicit
   SWA + Dense FFN + extra LM-head/verification graph.

High-impact missing details are packed SWA QKV plus Q/K/V norms and head gate,
Full/SWA head gates, shared-KV alias/page-size/allocated-byte accounting, and
FP32 router execution.

The minimum scoped design is a heterogeneous Step4-Pro-Latest config with
explicit grouped-GEMM, vLLM DeepEP HT dispatch/combine, KV-layout metadata,
FP32 router mode, and MTP1 graph. Existing operations remain the default for
all otherwise representable work.

Evidence:
`/data/ycfeng/tmp/step4_latest_aic_gap_audit.txt`
(`sha256:433c39c7dc1cf60b5dcef75a4053087a38a7e0b520071371d100bc3dd0860351`).

## DeepEP launch example/case inventory

### Pinned image

- Image:
  `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`.
- Manifest digest:
  `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`.
- The image has DeepEP `1.2.1`, NVSHMEM, `deep_ep/buffer.py`,
  `vllm/distributed/nvshmem_manager.py`, and the vLLM DeepEP HT/LL runtime
  modules. These are dependencies and implementation code, not launch cases.
- A direct scan of all `257` regular files under the image's `/app/examples`
  found `0` files containing exact DeepEP launch terms.
- `app/examples/online_serving/elastic_ep/serve_deepseek_v2.sh` is the only
  example containing an `--all2all-backend` option, but it selects
  `allgather_reducescatter`, not DeepEP.
- `app/examples/offline_inference/routed_experts_e2e.py` validates routed
  expert capture and forces `FLASH_ATTN`; it does not enable EP or DeepEP.
- `app/examples/offline_inference/data_parallel.py` is a generic EP/DP
  launcher. It accepts all engine arguments, so it can be adapted by supplying
  `--all2all-backend=deepep_high_throughput` or
  `--all2all-backend=deepep_low_latency`, but the image does not include such a
  ready-made command.

### Pinned local vLLM checkout

- `rjob-step4pro-2node.sh` is the authoritative Step4Pro-specific launch
  script. It defaults to `deepep_high_throughput`, uses DP16/EP16, and submits
  two eight-GPU replicas. It is a smoke launcher, not evidence of a completed
  successful run.
- Three executable scheduled-integration scripts directly loop over both
  DeepEP modes:
  - `.buildkite/scripts/scheduled_integration_test/deepseek_v2_lite_ep_eplb.sh`;
  - `.buildkite/scripts/scheduled_integration_test/qwen30b_a3b_fp8_block_ep_eplb.sh`;
  - `.buildkite/scripts/scheduled_integration_test/qwen3_next_mtp_async_eplb.sh`.
- `.buildkite/test_areas/distributed.yaml` contains a direct two-GPU H100
  invocation of `examples/offline_inference/data_parallel.py` with
  `deepep_high_throughput`.
- Six configs under
  `tests/evals/gsm8k/configs/moe-refactor-dp-ep/` select DeepEP HT or LL and
  are consumed by `tests/evals/gsm8k/test_gsm8k_correctness.py`.
- `tests/kernels/moe/test_deepep_moe.py` and
  `tests/kernels/moe/test_deepep_deepgemm_moe.py` construct actual DeepEP
  buffers through `tests/kernels/moe/parallel_utils.py`, but their configured
  world size is two local GPUs. They are kernel/integration cases, not a
  cross-node platform launcher.
- `docs/serving/expert_parallel_deployment.md` gives two-node
  `deepep_low_latency` commands. This is documentation, not an executable
  platform script.
- `rjob-step4pro-deepep-probe.sh` is not present at pinned commit
  `607d1641ee`. It was added only at later commit `9bfd9a610e`; the later probe
  includes explicit NVSHMEM initialization and `--share-host-shm=True`.

## Provider-attention audit in progress

- The Latest model plan already selects two provider-specific operations:
  `step4_context_attention` and `step4_generation_attention`. They have no
  vLLM registry entries yet, so the current plan cannot schedule them.
- The generic vLLM attention Collector is not an acceptable Full-MFA
  substitute. The pinned hd512 path requires paged BF16 K/V storage with page
  size 128 and calls `optimus_flash_attn_varlen_func`.
- The pinned Optimus entry rejects missing `seqused_k` or `block_table`, rejects
  page sizes other than 128, and uses `pack_gqa=False`.
- SWA must be audited separately. Its hd128/window-512 path is implemented by
  the pinned native vLLM FlashAttention backend rather than Optimus FA4, so
  implementing only the Full-MFA provider would leave a required Latest
  operation unmeasured.
- Both structures enter the same pinned
  `FlashAttentionImpl.forward -> fa_utils.flash_attn_varlen_func` call path.
  The adapter dispatches hd512/FA4 to Optimus and all other supported shapes to
  `vllm.vllm_flash_attn.flash_attn_varlen_func`. Therefore the Collector should
  invoke the pinned `FlashAttentionImpl.forward` path, not maintain two
  reimplemented attention formulas.
- Full MFA must use BF16, head dimension 512, block/page size 128, paged
  metadata, and one aliased K/V cache view. SWA must use BF16, QH128/KVH8,
  head dimension 128, page size 128, and sliding window 512.
- Collector population must keep each provider/structure as an exact physical
  identity. Batch and sequence/context lengths are workload axes; provider,
  heads, head dimension, window, cache dtype/layout, K/V aliasing, and page size
  cannot be silently crossed or deduplicated.

## Pinned hybrid KV physical-layout findings

- `unify_kv_cache_spec_page_size` pads the hd512 Full-MFA page to the SWA page
  size without changing the 128-token logical block size.
- Both Step4 attention families therefore allocate `524288` bytes per physical
  block on the pinned BF16 graph.
- `_reshape_attention_kv_cache` gives Full MFA a `524288`-byte block stride and
  a zero K/V-plane stride. Its logical K/V payload remains aliased even though
  the allocator reserves the full padded page.
- SWA uses a `262144`-byte per-plane block stride under the pinned `NHD`
  layout. K and V are separate, so their combined allocator page is `524288`
  bytes.
- SWA physical residency depends on the already-computed prefix and the current
  in-flight query chunk. For decode, 513 total tokens retain five blocks; at
  640 total tokens the first block is released and four remain.
- The old logical-byte cache cap admitted the wrong generation cases. The
  corrected physical-page cap is expected to produce 149 unique cases:
  65 Full-MFA and 84 SWA.
- SWA resident allocation is not monotonic: the whole Latest graph uses
  `204472320` physical bytes at 513 tokens and `174063616` bytes at 640 tokens
  after a stale SWA block is released.
- OOM/capacity inversion therefore cannot use resident bytes directly. The
  correct monotonic quantity is the highest physical allocation observed while
  growing the request to the target sequence length.
- For a `204472320`-byte budget, the safe MTP-off decode boundary is 640
  tokens. Token 641 starts the next block and raises peak allocation above the
  budget.

## B300 grouped/router 65K findings

- The Latest-only provider workload population now contains `75` grouped
  cases and `75` router cases. Generic GEMM remains unchanged at `35742`
  cases, `74` token buckets, and maximum token count `32768`.
- The only new provider physical key in each family is
  `num_tokens=65536`; no old key was removed.
- The B300 runtime profile executed `300` timed `torch.einsum` calls and `375`
  router custom-op calls. This matches four timed calls per grouped case and
  five calls per router case across `75` cases.
- Measured `65536`-token latency is `2.518085320790609 ms` for grouped
  `wo_a` and `0.539962649345398 ms` for the FP32 router.
- All `150` canonical queries return `source="silicon"` with exact numeric
  equality to the measured rows.
## 2026-08-15 — AIC continuation findings after DeepEP deferral

### Confirmed scheduler semantics

- The requirements define prefill `Batch size` as instance-level concurrent
  requests and decode `B` as active sequences in the stable engine step.
- `RuntimeConfig.batch_size` is consumed by each AIC operation query, while
  `run_static` reports `global_bs = batch_size * attention_dp_size`.
- Therefore EP16/EP32 simulation must make the instance/local mapping explicit;
  it cannot silently pass the requirement's global batch as the local
  per-attention-rank batch.
- `max_num_batched_tokens` is a global vLLM scheduler budget. Existing
  `run_static` correctly rejects batch>1 chunked prefill because a single
  uniform `s` cannot represent multiple requests progressing through a shared
  budget.

### Required simulation behavior

- For the requirements matrix, a task-local scheduler driver can model equal
  length requests deterministically:
  - allocate each scheduler step's global token budget across unfinished
    requests;
  - query one or more uniform-shape operation groups per step;
  - retain completed-request KV while later requests continue;
  - compute peak HBM from retained KV plus current in-flight chunk allocation.
- Formal SILICON simulation must run only when every queried operation resolves
  to exact measured coverage. Current known gaps are QKV norm/RoPE and DeepEP
  HT; the validator must discover and report them through the actual consumers.

### Exact consumer coverage result

- The validator queried all EP16/EP32 context and generation operation records
  independently, so missing QKV or DeepEP did not stop later consumers.
- Full result: `36,420` records, `16,048` exact-silicon, `15,160` analytic,
  `5,212` missing, and `0` unexpected errors.
- QKV has two missing physical contracts:
  - `vllm_step4pro_qkv_norm_rope`, `q+k+v`, `128/8/128`;
  - `vllm_step4pro_k_norm_rope`, `k`, `64/1/512`.
- DeepEP has four missing physical contracts:
  dispatch/combine for EP16 and dispatch/combine for EP32.
- Existing canonical data is not yet sufficient for a `65,536`-token formal
  prefill chunk: native SWA context attention does not bracket
  `batch=1, query_tokens=65,536, total_context_tokens=65,536`.
- The initial uniform-local coverage probe also requested Optimus MoE
  `num_tokens=131,072`, but this is not yet a confirmed measurement gap:
  `MoE.query()` multiplies rank-local `x` by `attention_dp_size`, while the
  requirements' `65,536` budget is global across the scheduler. Passing
  `65,536` as every rank's local token count overstates global MoE tokens.
- A scheduler-aware driver must therefore query:
  - attention/dense work from the most-loaded active attention-DP rank;
  - MoE compute from the actual global scheduled-token count;
  - DeepEP from the actual per-rank token distribution.
  The current uniform static API cannot express all three simultaneously.
- Therefore “Attention 199/199” and “Optimus MoE 174/174” remain valid
  row-level consumer checks but are not sufficient to claim full requirements
  matrix coverage.

## 2026-08-15 — Final 65K Attention and corrected scheduler coverage

- The missing native SWA context point was measured through the pinned vLLM
  provider at `(batch=1, query=65,536, total_context=65,536)`:
  `3.164181391398112 ms`.
- The matching Full FA4 point was measured at
  `196.56292724609375 ms`.
- Both B300 executions passed the exact source probe for `2103` manifest files
  on SM103 and cleaned RJob/Replica resources to `0/0`.
- Canonical context Attention now has `52` rows:
  `26` Optimus FA4 and `26` native SWA. Generation remains `149`, so total
  Attention coverage is `201` rows.
- The canonical context SHA256 is
  `4ba0473c26a9d418925ebce724798a848edf0c0afcdb88d092440f05a1e90ef2`;
  all `52/52` context queries are exact silicon with `0.0 ms` error.
- Scheduler-corrected EP16+EP32 coverage contains:
  `36,420` records, `16,660` exact silicon, `15,160`
  analytic/non-silicon, `4,600` missing, and `0` unexpected errors.
- Complete exact-silicon families in that matrix are:
  Attention `1,560/1,560`, Optimus MoE `1,520/1,520`, grouped GEMM
  `400/400`, and FP32 router `1,520/1,520`.
- Only six physical contracts remain missing:
  two QKV norm/RoPE provider identities and DeepEP dispatch/combine for EP16
  and EP32.
- DeepEP is now a frozen, documented gap at `0/116`; it must not be retried in
  the current phase.

## 2026-08-15 — Scheduler-derived Attention workload findings

- Exact row-consumer validation and requirements-matrix coverage are different
  gates. The former only proves that persisted rows can be queried; the latter
  must derive the shapes produced by chunking and attention-DP request
  placement.
- The requirements matrix produces three additional context shapes:
  `(batch=2, query=512, total=512)`, `(1, 4096, 4096)`, and
  `(1, 16384, 16384)`.
- The last decode row has context `1,048,544` and output `32`. The current
  steady-step model queries at `1,048,560`, so generation data needs an upper
  point. The requirements explicitly state the final length is `1,048,576`;
  measuring that endpoint supplies the required bracket without inventing a
  timing.
- The corrected Attention population is `58` context plus `167` generation
  rows. The prior `52 + 149 = 201` rows remain valid measurements but are not
  sufficient by themselves for the full requirements matrix.

## 2026-08-16 — DeepEP proxy source-data findings

- The selected proxy table is
  `src/aiconfigurator/systems/data/b300_sxm/nccl/2.27/nccl_perf.parquet`.
- It contains `504` rows and measured `alltoall` curves for `half` and
  `int8`.
- Measured rank counts are `2`, `4`, and `8`; requested EP16/EP32 queries use
  AIC's existing topology correction from the 8-GPU curve.
- The table has no `fp8` slice. Dispatch therefore uses `int8` only as the
  measured one-byte transport-equivalent lookup for the logical FP8 payload.
- Combine uses `CommQuantMode.half` for the logical BF16 payload.
- The retained Step4 MoE mapping is:
  `ceil(tokens_per_dp_rank * hidden_size * topk / ep_size)` message elements.
- A local inspection attempt with `.venv/bin/python` failed because that
  environment does not install `pandas`. This is an inspection-environment
  mismatch, not a dataset or AIC failure; system Python has both `pandas` and
  `pyarrow` for read-only Parquet inspection.
