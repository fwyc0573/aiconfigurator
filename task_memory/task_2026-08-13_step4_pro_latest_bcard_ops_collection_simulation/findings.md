## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Recorded evidence from the initial repository and requirements audit. |
| 2026-08-13 | Added measured cluster, credential, image, and AIC-shape evidence. Replaced the inconclusive local-cache note with a measured inspection result. |
| 2026-08-13 | Added verified latest branch head, image digests, source-file presence, and B300 predict-only evidence. |
| 2026-08-13 | Recorded choice A and the pinned Step4Pro source-to-logical-op inventory. |
| 2026-08-13 | Added the vLLM DeepEP consumer/Collector audit and the manifest reconstruction decision. |
| 2026-08-13 | Added the pinned-vLLM versus current-AIC fidelity-gap audit. |

# Findings

## Confirmed from local files

- The supplied requirements document is `task_memory/step4pro_v4_external_simulator_requirements.md`.
- It names B300 as the target hardware and pins vLLM commit `607d1641ee3fec43653fca510d717725828890c2`.
- It requires synthetic 78-layer configuration, dummy weights, EP16/EP32/2×EP16, prefill/decode matrices, MTP-off/MTP1 comparisons, and numeric evidence.
- Historical v3/v4 work separates model support/figures from op profiling and emphasizes exact consumer-key and measurement provenance.
- The current checkout has extensive prior-task modifications; this task must not overwrite or reset them.

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
- The two smoke scripts named in requirements section 3.1, `rjob-step4pro-optimus-single.sh` and `rjob-step4pro-2node.sh`, are absent locally. They reside in the unreachable vLLM branch.
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
