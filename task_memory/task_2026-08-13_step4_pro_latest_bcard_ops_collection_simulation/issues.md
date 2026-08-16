## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Opened the initial requirements identity blocker. |
| 2026-08-13 | Added measured blockers ISSUE-002 (B300 quota RBAC), ISSUE-003 (vLLM repository credentials), ISSUE-004 (cached image lacks target implementation), ISSUE-005 (AIC Step4-Pro-V4 shape conflict). Downgraded ISSUE-001 registry-absence caveat to a measured result. |
| 2026-08-13 | Revalidated the latest branch/image and B300 quota path; narrowed the remaining gate to explicit semantic scope decisions. |
| 2026-08-13 | Resolved ISSUE-001 source/runtime authority with user choice A. |
| 2026-08-13 | Resolved the new AIC identity as `stepfun-ai/Step4-Pro-Latest`; retained missing-manifest decision as the active requirements gate. |
| 2026-08-13 | Confirmed the manifest files are unavailable and opened ISSUE-007 for dirty linked-worktree isolation. |
| 2026-08-13 | Resolved missing-manifest handling through explicit reconstruction with a new SHA256. |
| 2026-08-13 | Confirmed the minimal fidelity-preserving operation strategy; ISSUE-007 is now the sole pre-implementation gate. |
| 2026-08-13 | Recorded and resolved ISSUE-008: missing Git LFS caused the branch-switch post-checkout hook to fail. |
| 2026-08-13 | Opened ISSUE-009 for 128 pre-existing baseline test failures discovered after the checkpoint commit. |
| 2026-08-13 | Closed the temporary-file security follow-up with the owner's explicit retain/no-action decision. |
| 2026-08-14 | Resolved ISSUE-011 as deferred: MTP1 structure, measurement, and simulation are postponed; MTP-off Latest work is active. |
| 2026-08-14 | Recorded ISSUE-012: whole-model pinned-vLLM B300 smoke/runtime trace is externally owned; current session is AIC-only. |
| 2026-08-14 | Opened and resolved ISSUE-013 for memory-heavy full Replica inventory and ISSUE-014 for an undersized live-launch timeout in the external B300 session. |
| 2026-08-14 | Recorded ISSUE-015 for the new host OOM and applied the owner-directed 3 GiB low-memory execution policy. |
| 2026-08-15 | Resolved ISSUE-016 for the source-probe lifecycle race and ISSUE-017 for redundant large Git-pack processing. |
| 2026-08-15 | Closed ISSUE-024 after the complete 199-row B300 Attention collection and consumer validation passed. |
| 2026-08-16 | Opened ISSUE-037 for the owner-approved, explicitly labeled DeepEP simulation proxy. |
| 2026-08-15 | Resolved ISSUE-025 host-wrapper executable-bit coupling and ISSUE-026 non-fatal Optimus metadata identity failure. |
| 2026-08-15 | Opened ISSUE-018: the current platform identity cannot mount the pinned qy1-pt model bucket. |
| 2026-08-15 | Opened ISSUE-019: image-native Optimus activation quantization is incompatible with B300 SM103. |
| 2026-08-15 | Opened and resolved ISSUE-016: Latest KV capacity incorrectly used the linear base-model inverse. |
| 2026-08-15 | Opened ISSUE-018 for the missing provider-specific dataset schema, loaders, queries, cases, and pinned-runtime routing. |
| 2026-08-15 | Partially resolved provider-data ISSUE-018 for grouped `wo_a`; opened ISSUE-019 for an unrelated stale CustomAllReduce regression test. |
| 2026-08-15 | Extended the partial ISSUE-018 resolution to the pinned Optimus FP32 router path. |
| 2026-08-15 | Extended ISSUE-018 through QKV norm/RoPE and resolved ISSUE-020 before measurement. |
| 2026-08-15 | Extended ISSUE-021 through the simulator OOM/capacity path using peak physical KV allocation. |
| 2026-08-15 | Opened ISSUE-022 for chunk-size-aware prefill peak KV accounting before formal simulation. |
| 2026-08-15 | Opened and resolved ISSUE-023 after the first B300 Attention smoke exposed an incomplete source-package transfer without AIC distribution metadata. |
| 2026-08-15 | Opened and resolved ISSUE-024 after the first full Attention run exposed an incorrect PEP 440 package version in the runtime profile. |
| 2026-08-15 | Opened ISSUE-027 for CUDA graph contamination of the default RNG during the full Optimus MoE case sequence; local root fix is verified and B300 rerun remains pending. |
| 2026-08-15 | Opened and locally resolved ISSUE-028 after discovering that the nominal 174-row Optimus run used the wrong call boundary and silently fell back to eager for 150 rows. |
| 2026-08-15 | Updated ISSUE-B300-006: full-RDMA injection and the real EP16 NCCL all-reduce passed; only line-based parsing of interleaved rank output remains to be repaired. |
| 2026-08-15 | Added ISSUE-030 for incomplete B300 provider dataset canonicalization. |
| 2026-08-15 | Resolved ISSUE-B300-006 after occurrence-based marker counting passed static checks and the live EP16 NCCL rerun. |
| 2026-08-15 | Opened ISSUE-B300-007 after EP16 DeepEP HT reached the pinned manager but failed NVSHMEM initialization because the direct Collector path skipped pinned worker environment harmonization. |
| 2026-08-15 | Opened ISSUE-B300-006 after the AIC DeepEP EP16 smoke fixed CUDA Error 803 but reproduced the empty-HCA PyNCCL failure under an incomplete RDMA launch contract. |
| 2026-08-15 | Re-reviewed ISSUE-B300-001 and resolved ISSUE-B300-002 by separating runtime necessity from model-quality and training claims. |
| 2026-08-15 | Live-verified ISSUE-B300-003 environment restoration and opened ISSUE-B300-004 for the required head/worker CLI role split. |
| 2026-08-15 | Resolved ISSUE-B300-005 with a 16-rank NCCL pass and opened ISSUE-B300-006 for the unavailable shared-host-SHM/NVSHMEM DeepEP bootstrap. |
| 2026-08-15 | Passed the ISSUE-028 replacement B300 smoke across both exact custom-op branches; retained full 174-case collection as the closure gate. |
| 2026-08-15 | Closed ISSUE-027 and ISSUE-028 after the replacement Optimus run completed 174/174 rows with zero errors and exact consumer matches. |
| 2026-08-15 | Narrowed ISSUE-B300-007 with an eight-rank local Buffer pass; blocked cross-node DeepEP collection on the unavailable shared-host-SHM/NVSHMEM launch contract and recorded a separate local destroy failure. |
| 2026-08-15 | Resolved ISSUE-029: a stale DeepEP operation test expected the pre-consumer NotImplemented path instead of exact-key fail-fast behavior. |
| 2026-08-15 | Opened ISSUE-B300-008 after the documented legacy launcher passed explicit NVSHMEM but DeepEP Buffer observed a mismatched NVSHMEM PE count. |
| 2026-08-15 | Opened ISSUE-031 after the replacement provider-core smoke reached pinned SWA QKV but the image-native CuTeDSL rejected `reload_from="smem"` as a dynamic JIT argument. |
| 2026-08-15 | Confirmed ISSUE-031 as an image-native postponed-annotation/Cutlass DSL contract defect; recorded exact package/source identities and stopped before an unapproved runtime overlay. |
| 2026-08-15 | Reopened DeepEP as ISSUE-B300-009 for exact validation of the owner-authorized `9bfd9a610e` shared-host-SHM/NVSHMEM launch contract. |
| 2026-08-15 | Partially closed ISSUE-030 by canonicalizing and revalidating all 199 accepted B300 Attention rows; grouped `wo_a`, router, and QKV remain open. |
| 2026-08-15 | Further closed ISSUE-030 with complete 74-row grouped `wo_a` and 74-row FP32 router B300 datasets plus 148/148 canonical exact-consumer matches; only QKV remains in provider-core. |
| 2026-08-15 | Applied the owner-confirmed DeepEP skip disposition: retain `0/116` accepted rows and continue the remaining provider families. |
| 2026-08-15 | Closed the grouped/router 65K workload gap with 75+75 canonical rows and 150/150 exact silicon consumer matches. |
| 2026-08-15 | Opened ISSUE-032 after the scheduler-aware requirements matrix exposed four Attention workload gaps absent from the earlier row-level coverage scan. |
| 2026-08-15 | Resolved ISSUE-033: MTP-off OOM classification now enforces the requirements' `gpu-memory-utilization=0.9` limit. |
| 2026-08-15 | Resolved the scheduler-derived Attention gap with 225/225 exact consumers and recorded ISSUE-034 for the formal QKV/DeepEP simulation block. |
| 2026-08-15 | Partially resolved ISSUE-030 with 75/75 Full-MFA QKV rows; narrowed ISSUE-031 and ISSUE-034 to SWA QKV plus the frozen DeepEP gap. |
| 2026-08-15 | Resolved ISSUE-035: a non-fail-fast aggregate validation command masked Ruff findings; all checks now run independently with fresh passing evidence. |
| 2026-08-15 | Added the ISSUE-031 SWA QKV RED contracts; implementation remains gated on explicit runtime-overlay approval. |
| 2026-08-16 | Resolved ISSUE-030 and ISSUE-031 with complete SWA QKV measurement and canonical `150/150` exact consumers; narrowed ISSUE-034 to DeepEP only. |
| 2026-08-16 | Resolved ISSUE-037: provider-core acceptance now requires an explicit slice, SHA256-bound matching smoke evidence, and the complete expected QKV key set. |
| 2026-08-16 | Opened ISSUE-038 for four scheduler-derived context Attention workloads hidden behind the former first-chunk DeepEP blocker. |
| 2026-08-16 | Opened ISSUE-039 after rejecting a successful Attention run that started with 247685 MiB of unrelated GPU memory already occupied. |
| 2026-08-16 | Reopened ISSUE-038 after the full matrix exposed a later blocker; the exhaustive 197-coordinate audit found 14 missing coordinates per provider and zero unexpected errors. |
| 2026-08-16 | Reopened ISSUE-039 after the minimal endpoint run started with 255606 MiB occupied; its rows were rejected and an isolated clean rerun was required. |
| 2026-08-16 | Resolved ISSUE-038 and ISSUE-039 with a clean two-provider endpoint, 68-row canonical context table, 235/235 exact row consumers, and 394/394 scheduler Attention queries. |
| 2026-08-16 | Resolved the DeepEP ISSUE-037 proxy stage with zero missing/error records and retained real DeepEP measurement as deferred work. |
| 2026-08-16 | Opened and resolved ISSUE-040: the Phase 10 manual-review row omitted required execution-status and decode engine-step fields. |
| 2026-08-16 | Revalidated ISSUE-037 and ISSUE-040 with the final 343-test focused suite, 401 Collector tests, a byte-identical full simulation rerun, and the normalized artifact audit. |
| 2026-08-16 | Opened and resolved ISSUE-041: the manual-review CSV used CRLF because `csv.DictWriter` retained its platform-independent Excel dialect default. |

# Issues and Resolutions

## ISSUE-039 — Targeted Attention run scheduled onto an occupied GPU

**Status:** Resolved again on 2026-08-16; the contaminated endpoint rows were
rejected and a clean, source-identical rerun was accepted.

**Observed facts:**

- Workload `(1,32736,1048544)` completed and emitted both provider rows.
- Before source reconstruction or Collector import, `nvidia-smi` reported
  `247685 MiB` used and only `26429 MiB` free on the assigned B300.
- The first three accepted targeted jobs began at `0–4 MiB` used.
- All four jobs requested one B300 through the same quota, tag, image, and
  runner contract.

**Root cause:**

The platform assigned this RJob to a B300 that was not idle. The exact
external owner of the pre-existing allocation is no longer inspectable after
the worker was cleaned. This is an environment-isolation failure, not a pinned
vLLM provider or Collector-kernel failure.

**Impact:**

The emitted latencies may include memory-pressure or compute-contention bias
and cannot be accepted as canonical evidence. Accepting them would violate the
fresh B300 measurement requirement.

**Resolution:**

1. Preserve the rejected raw artifact without modification.
2. Re-run only the same exact workload on a worker whose initial GPU usage is
   comparable to the accepted `0–4 MiB` baseline.
3. Merge only the clean rerun rows; compare the rejected and clean values as
   diagnostic evidence, not as repeated samples to average.

**Verification result:**

- clean rerun initial usage: `4/275040 MiB`;
- accepted clean latency: `2098.0398763020835 ms` Full FA4 and
  `1.5781973203023274 ms` native SWA;
- rejected-versus-clean difference: `-0.27667969164858475%` and
  `-0.059813726419322144%`;
- clean raw CSV SHA256:
  `55580b01d43b6a6018fc8bd7f818b4ccdb3f61049a04ffe662c15881cc304ff1`;
- final task RJob/Replica inventory: `0/0`.

**Recurrence during the minimal endpoint closure:**

- RJob `s4p-aic-attnep-0816-1515` passed the pinned source probe, SM103
  identity, exact two-provider workload, positive finite latency, evidence
  transfer, and final `0/0` resource cleanup.
- It nevertheless started with `255606/275040 MiB` occupied and only
  `18508 MiB` free before source reconstruction or provider execution.
- Rejected endpoint latencies are `256.19809977213544 ms` for Full FA4 and
  `0.7958026727040609 ms` for native SWA.
- Raw evidence is preserved under
  `/data/ycfeng/tmp/step4_attention_closure_20260816/endpoint_s4p-aic-attnep-0816-1515/`;
  no row from this run may enter the canonical Parquet.

**Clean endpoint resolution:**

- RJob `s4p-aic-attnep-clean-0816-1519` began at `0 MiB` used and
  `274114 MiB` free.
- Accepted latencies are `256.1752115885417 ms` for Full FA4 and
  `0.7958613236745199 ms` for native SWA.
- Source manifest `2103/2103`, source probe, SM103, exact workload, positive
  finite latency, evidence transfer, and final `0/0` cleanup all passed.
- Accepted raw CSV SHA256:
  `c6545f096320d456b24d55565b842167583fd62b033855b28c2e297cc3843d8f`.

## ISSUE-038 — Scheduler Attention coverage was discovered one blocker at a time

**Status:** Resolved on 2026-08-16; the clean minimum endpoint was measured,
canonicalized, and closed every scheduler Attention coordinate.

**Observed facts:**

- The first proxy matrix exposed four missing workloads for both providers.
  Those eight rows were measured, canonicalized, and passed `233/233` exact
  row-consumer queries.
- The next full matrix improved to `6/72` blocked prefill results and `0/84`
  blocked decode results, then stopped each affected workload at
  `(1,16384,49152)`.
- Exhaustive derivation found `197` unique scheduler Attention coordinates.
- Direct canonical consumer queries covered `394` provider-coordinate pairs:
  `366` succeeded, `28` were missing, and `0` raised unexpected errors.
- The `28` misses are the same `14` coordinates for both providers:
  batch `1`, query `16,384`, and total context `49,152` through `262,144`
  in `16,384` increments.

**Root cause:**

`simulate_prefill_case()` intentionally stops querying later chunks after the
first missing operation so it cannot manufacture a formal latency. That is
correct simulation behavior, but using repeated full-matrix runs as the
coverage-discovery mechanism reveals only the earliest missing coordinate on
each pass. The fixed-probe coverage validator likewise does not enumerate all
scheduler chunks.

**Impact:**

- Proxy simulation cannot yet report all `72` prefill results.
- Decode remains complete at `84/84`.
- Measuring only the currently visible blocker would repeat the same
  discovery loop.
- Relaxing interpolation or inserting synthetic latency would hide a real
  dataset boundary and is prohibited.

**Resolution in progress:**

1. Preserve the fail-fast simulator behavior.
2. Use the completed 197-coordinate diagnostic as the coverage source of
   truth before any further B300 launch.
3. Validated the minimum endpoint set under the unchanged consumer:
   one `(1,16384,262144)` row per provider changes missing queries
   `28 → 14 → 0`; two rows are both sufficient and necessary because provider
   structural keys are disjoint.
4. Added only that endpoint to the shared Collector population through
   RED→GREEN tests; the getter changed from `66` to `68` unique invocations.
5. Measure both pinned providers on a clean B300 and merge only accepted rows.
6. Re-run the 197-coordinate audit, canonical exact-row consumer validation,
   full coverage validator, and the complete `72 + 84` proxy matrix.

**Resolution result:**

- Context canonical rows: `66 → 68`, split `34+34` by provider.
- Merge accounting: kept `66`, added `2`, removed `0`, deduplicated `0`.
- Generation remains byte-identical at `167` rows.
- Canonical exact row consumers: `235/235`, all silicon, maximum absolute
  error `0.0 ms`.
- Exhaustive scheduler Attention queries: `394/394`, missing/error `0/0`.
- Canonical context SHA256:
  `7cf90e1f508e9f18dceb29c51dc90e88caa1f13d3d25d38f4a3435851c79522a`.
- Canonical generation SHA256 remains
  `bde836b884b410b21284645ab45f0a35f82d03cac5b91299b256823067cb14c0`.
- `step4_deepep_ht_perf.parquet` remains absent.

**Evidence:**

- `/data/ycfeng/tmp/step4_attention_closure_20260816/scheduler_attention_exhaustive_197_20260816.json`;
- SHA256
  `6b95bfb9af77902b1a7837a181fc3d66f5b6b6f4d4f0e55674cdf769e5cf83d0`.
- `/data/ycfeng/tmp/step4_attention_closure_20260816/scheduler_attention_minimal_endpoint_proof_20260816.json`;
- SHA256
  `8968f54b7f5aba21cf2c0681a4d5640916e1d85adc8f3763aaa2d76b41e8b797`.
- `/data/ycfeng/tmp/step4_attention_closure_20260816/scheduler_attention_exhaustive_197_post_endpoint_20260816.json`;
- SHA256
  `4d391ed29952c6ecac7f5fd90a094f27948cb270e287149a39dc2a41f0117937`.

## ISSUE-037 — Provider-core acceptance gates were weaker than the measured-data contract

**Status:** Resolved and independently approved on 2026-08-16.

**Observed facts:**

1. The shared provider-core runner still had an implicit `all` path that could
   couple independent provider families.
2. `MODE=full` did not require proof that the same slice had first passed its
   bounded smoke.
3. The data validator could accept the requested QKV row count without proving
   equality with the complete Collector-generated physical-key set.

**Root cause:** Slice selection, smoke admission, and persisted-row validation
were implemented at different layers without one fail-fast acceptance
contract joining them.

**Impact:** A full run could start with an unintended provider scope, a smoke
for another slice could be treated as sufficient, or a unique but incorrect
QKV token/key population could pass count-only validation. These risks affect
reproducibility and dataset identity, even though the already measured
canonical QKV rows were later shown to be correct.

**Resolution:**

1. Require the provider-core wrapper to receive exactly one of
   `grouped_router`, `qkv_full`, or `qkv_swa`; reject `all` on both host and
   worker.
2. Require `PROVIDER_CORE_SMOKE_EVIDENCE` and its SHA256 for every full run.
   Validate JSON mode `smoke`, suite `provider_core`, matching slice, and
   completed cases `2` for grouped/router or `1` for either QKV slice on both
   sides.
3. Derive the exact QKV expected-key set from
   `get_step4_qkv_norm_rope_test_cases()`, require provider split `75+75`, and
   reject missing, extra, wrong-shape, wrong-dtype, or duplicate keys.

**Verification result:**

- targeted RED: `5 failed`, `43 passed`;
- targeted GREEN: `48/48` passed in `6.25s`;
- canonical QKV revalidation: `150` rows, `150` unique keys, providers
  `75+75`, `150/150` exact silicon consumers, maximum error `0.0 ms`;
- independent reviewer verdict: **APPROVE**, with no remaining Blocking or
  Important finding;
- no B300 timing row, provider identity, or simulation output was changed.

## ISSUE-032 — Attention population omitted scheduler-derived requirements shapes

**Status:** Resolved on 2026-08-15.

**Observed facts:**

- The initial requirements matrix returned Attention missing-data errors for
  context workloads `(2, 512, 512)`, `(1, 4096, 4096)`, and
  `(1, 16384, 16384)` for both pinned providers.
- The 1M decode workload evaluates its steady step at context `1,048,560`,
  while the canonical generation data ended at `1,048,544`; the consumer
  therefore had no upper interpolation bracket.
- Existing canonical Attention data contained `52` context and `149`
  generation rows. Those rows were all individually valid, but their
  workload population mirrored the raw requirements batch table rather than
  all shapes produced after scheduler chunking and attention-DP mapping.

**Root cause:** The earlier coverage validator queried a fixed operation
workload grid and did not derive queries through the same scheduler mapping as
the formal MTP-off driver. As a result, row-level `201/201` exact-consumer
validation proved that every collected row was readable, but did not prove
that every requirements workload was bracketed.

**Impact:** Formal MTP-off simulation gained extra Attention blockers beyond
the known QKV and DeepEP gaps. Reporting Attention as complete would be
incorrect until the additional B300 rows are measured and consumed.

**Resolution:**

1. Added the three exact scheduler-derived context shapes to the existing
   Step4 Attention population.
2. Added generation endpoint `1,048,576`, which is the exact final
   context-plus-output length stated by the requirements and brackets the
   steady decode query.
3. Increased expected population from `52/149` to `58/167`.
4. Added focused tests and completed a fresh pinned-provider B300 collection.

**Verification result:**

- corrected B300 collection: context `58/58`, generation `167/167`;
- unique persisted keys: `225/225`;
- canonical exact consumers: `225/225`, all `source="silicon"`;
- maximum absolute consumer error: `0.0 ms`;
- post-fix requirements coverage has no Attention missing record.

No synthetic timing, H800 data, generic attention kernel, or interpolation
outside measured brackets is used.

## ISSUE-033 — MTP-off OOM threshold ignored the 0.9 memory limit

**Status:** RESOLVED 2026-08-15.

**Observed facts:** The driver emitted the 90% HBM limit but `_is_oom`
compared modeled memory against 100% physical capacity.

**Root cause:** The utilization constant was used only for reporting and was
not applied in the decision function.

**Impact:** Cases between 90% and 100% physical HBM could be labeled
non-OOM even though they violate the required vLLM
`--gpu-memory-utilization 0.9` runtime setting.

**Resolution:** `_is_oom` now compares total modeled HBM with
`physical_capacity × 0.9`. A focused boundary test verifies `89.99%` is
accepted and `90.0%` is OOM.

## ISSUE-018 — Latest provider operations have no accepted data contract

**Status:** Partially resolved on 2026-08-15. Grouped `wo_a`, FP32 router, QKV
norm/RoPE, pinned runtime routing, Latest model-plan selection, SM103 gating,
provider attention, and Optimus MoE are complete. DeepEP HT and remaining
provider measurement closure remain open.

**Observed facts:**

- Provider-tagged `ContextAttention`, `GenerationAttention`, `MoE`, and
  `MoEDispatch` still fail fast.
- `collector/framework_manifest.yaml` identifies stock vLLM `0.19.0`, not the
  pinned StepCast image and source commit.
- `collector/cases/models/Step4ForCausalLM_cases.yaml` contains only V3/V4
  H800 profiles.
- Existing generic collectors execute `RowParallelLinear`, generic attention,
  generic fused MoE, or generic communication paths. Those invocations do not
  prove the required Step4Pro providers ran.

**Root cause:** The model graph deliberately preserved provider identities
before a provider-specific persisted schema and exact pinned-vLLM collectors
were implemented.

**Impact:** MTP-off model construction is valid, but SILICON query, complete
B300 collection, and E2E simulation remain blocked.

**Required resolution:**

1. Define separate provider perf files and exact physical keys.
2. Add RED tests for loader/query behavior, collisions, model-case population,
   registry routing, and pinned runtime identity.
3. Implement single-GPU pinned-vLLM provider collectors.
4. Implement a separate multi-GPU DeepEP HT runner for EP16/EP32.
5. Accept rows only after exact-provider runtime evidence and consumer-query
   verification.

**Partial resolution evidence:**

- `step4_grouped_gemm_perf` has a distinct collector and consumer.
- Its case getter emits `74` unique invocations and `74` unique physical keys
  from one exact structural identity.
- `step4_fp32_output_gemm_perf` separately calls
  `torch.ops.vllm.optimus_matmul_fp32` and emits `74` unique invocations plus
  `74` unique physical keys.
- Focused tests passed `145/145`; Collector tests passed `349/349`.
- B300 runtime measurement remains pending and is not claimed.

## ISSUE-019 — Stale CustomAllReduce TP16 test contradicts the current fail-fast contract

**Status:** Open baseline anomaly; outside the Step4 provider-data scope.

**Observed failure:**

- Full SDK/database regression passed `607/608`.
- `test_query_custom_allreduce_large_tp_scaling` requests `tp_size=16` on an
  8-GPU node and expects synthetic cross-node scaling.
- Current production code raises `PerfDataNotAvailableError` and requires the
  runtime-selected multi-node collective to be collected.
- The failure reproduces alone.

**Root cause:** Commit `c3d2af05` already contains a contract mismatch:
`communication.py` enforces exact CustomAllReduce rank coverage, while the
older edge-case test still asserts the removed TP16 scaling behavior. The
current grouped-GEMM change did not modify either file.

**Impact:** This does not invalidate the grouped-GEMM slice, whose focused and
Collector suites pass. It prevents claiming that the entire SDK/database
directory is green.

**Resolution decision:** Do not restore synthetic scaling and do not rewrite an
unrelated baseline test inside this Step4 provider-data sub-step. Keep the
failure visible for a separately authorized communication-contract repair.

## ISSUE-016 — Latest nonlinear KV capacity used the linear inverse

**Status:** Resolved on 2026-08-15.

**Observed failure:** With a KV budget equal to the exact logical cost of 513
tokens, `get_kvcache_max_tokens()` returned `512`.

**Root cause:** `Step4ProLatestConfig` was absent from the Step4 model's
nonlinear-capacity type gate. The method therefore delegated to the base
linear floor-division path, which cannot represent the 512-token saturation
of the 58 SWA layers.

**Resolution:** Route Latest through the existing monotonic binary search used
for other hybrid Step4-Pro schemas. No scale factor, fallback, or altered KV
formula was added.

**Verification:** The RED test observed `512`; after the fix it observed the
expected `513`. The focused Latest suite passed `28/28`, and the historical
regression passed `899/899`.

## ISSUE-001 — Latest runtime identity conflicts with the fixed requirements input

**Status:** **Resolved for this task on 2026-08-13.**

**Observed facts:**
- `task_memory/step4pro_v4_external_simulator_requirements.md` specifies B300 and vLLM repository/branch/commit `607d1641ee3fec43653fca510d717725828890c2`.
- The user task asks for the latest B-card vLLM image and `step4-pro-latest`, but does not state whether the current branch head/image or the older pinned requirements commit is authoritative.
- The requirements file's linked `step4pro_v4_shape_manifest.json` and `step4pro_v4_vllm_b300_8bit_task.md` are absent at the referenced paths.
- A bounded local Docker image listing did not reveal a matching cached image; this is not proof that the internal registry lacks it.

**Root cause:** The runtime identity is now available, but the relationship between the current branch/image and the older fixed requirements contract is not explicitly decided.

**Impact:** The operation set, Collector keys, B-card measurements, and simulation shapes cannot be determined without risking an unrecorded assumption. Starting implementation or GPU collection now would violate the task's explicit no-assumption rule.

**Required decision:** Confirm the exact image/source identity, model shape, AIC model name, and treatment of custom kernels before implementation.

**Proposed alternatives:**
1. Recommended: use branch head `9bfd9a610ea4f2890010702ee7a207cf25edf8de` plus image manifest digest `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`; retain the requirements document's B300/78-layer experiment matrix unless the latest source explicitly contradicts it.
2. Use ancestor commit `607d1641ee3fec43653fca510d717725828890c2` and the requirements document as the complete runtime/source contract.

No workaround or fallback has been applied.

**Resolution:** The user selected the pinned local checkout and requirements contract. The later branch head is reference-only. The pinned checkout itself declares the B300 image, so that tag and its inspected digest are authoritative. The AIC canonical model path remains the next explicit user decision.

## ISSUE-002 — B300 quota is not grantable to this account (hard blocker)

**Status:** RESOLVED 2026-08-13. `b300_train_infra` is usable and B300 access is confirmed by predict-only and prior real-worker evidence. The earlier denial conclusion came from an incomplete candidate-quota probe.

**Resolution evidence:**
- `rlaunch --predict-only --charged-group=b300_train_infra --private-machine=group --gpu=8 --cpu=64 --memory=409600` returns 10 candidate 8-GPU B300 nodes.
- Multi-node predict succeeds at both required scales: `-P 2` (16 GPUs, `ep16_r1`) and `-P 4` (32 GPUs, `ep32_r1` / `ep16_r2`) each return 10 to 12 candidate nodes with `--custom-resources=rdma/mlnx_shared=8 --topo-group=yes --set-env=DISTRIBUTED_JOB=true`.
- A real 1-GPU worker launched and reported hardware matching requirements section 2.4 exactly:
  `NVIDIA B300 SXM6 AC`, `275040 MiB`, driver `580.159.03`, `compute_cap 10.3`.
  This agrees with `src/aiconfigurator/systems/b300_sxm.yaml`, which records `sm_version: 103` and `mem_capacity: 288400343040`.

**Follow-on constraint discovered during resolution:** see ISSUE-006. B300 lives in a different zone from this workspace, so the launch recipe is not the H800 recipe.

**Observed facts (all measured on 2026-08-13 from the CPU master):**
- The requirements document section 2.4 requires `NVIDIA B300 SXM6 AC`, with 16 GPUs (`ep16_r1`), 32 GPUs (`ep32_r1`), and 32 GPUs (`ep16_r2`).
- B300 nodes exist in the cluster. `brainctl get nodes -l GPUType=B300 -o json` returns exactly **20 nodes / 160 allocatable GPUs**, distributed by quotagroup as:
  `b300_pretrain3` = 11, `b300_pretrain` = 5, `b300_pretrain2` = 2, `b300_sys_pro` = 1, `b300_train_infra` = 1.
- Every B300 node carries `privatemachine.brainpp.cn/quotagroup` in the `b300_*` family. None carries `codesign`.
- Every B300-family quotagroup returns HTTP 403 on predict:
  `code: 403 message: subject {Type:user ID:e7163316ad76ba7266a33f9787afd70a Name:} cannot get resource quotagroups/predict in API group quota.brainpp.cn/v1alpha1 in the project shai/shai-core reason:Forbidden`
  Reproduced for `--charged-group=b300_sys_pro`, `b300_pretrain3`, and `b300_pretrain`.
- Tag-based selection fails for every B-card spelling attempted: `--positive-tags=` with `b300`, `b200`, `blackwell`, `b300_sxm`, `gb300`, `b300-sxm6` all return `no machine available`.
- Pinning the grantable quota to a B300 node fails:
  `--charged-group=codesign --positive-tags=node/gpu-b300-0339.qy.cnw.istep.fun` returns `no machine available`.
  `--charged-group=codesign --positive-tags=feature/GPUType=B300` returns `no machine available`.
- Control case confirms the probe method is sound: `--charged-group=codesign --positive-tags=h800` returns multiple 8-GPU H800 nodes with available capacity.

**Root cause:** B300 nodes are private machines owned by the `b300_*` quotagroups. This account holds only the `codesign` quotagroup, which contains H800 capacity exclusively. The task statement asserts B-card access has been granted, but platform RBAC does not reflect that grant.

**Correction (2026-08-13):** this root cause was wrong. The account does hold a B300 quotagroup, `b300_train_infra`. The error was probing an incomplete candidate set, chosen from a sampled node listing, and then concluding absence from three denials. Three denials do not establish that all B300 quotagroups are denied.

**Impact:** No B-card measurement is possible. There is no degraded substitute: requirements section 2.4 estimates about 230 GB/GPU of static weights for `ep16_r1`, while H800 provides 80 GB per GPU (`src/aiconfigurator/systems/h800_sxm.yaml` records `mem_capacity: 85029158912`). The target model cannot be resident on H800 at the specified parallelism, so H800 cannot stand in for B300 even as an approximation.

**Required decision:** Platform must add this account to a B300 quotagroup with at least 32 GPUs of headroom, or the B-card scope must be formally suspended.

**Proposed alternatives:**
1. Recommended: obtain B300 quotagroup membership, then resume the plan unchanged.
2. Suspend measurement and deliver simulator-side modeling only. This cannot satisfy the requirements document sections 4 and 5, which require measured-versus-simulated comparison tables, so it changes the deliverable contract and needs explicit approval.

No workaround or fallback has been applied. H800 has not been substituted.

## ISSUE-003 — The pinned vLLM repository is unreachable for lack of credentials

**Status:** RESOLVED 2026-08-13 for the source checkout. The current branch is readable; image/runtime scope remains part of ISSUE-001.

**Observed facts:**
- Requirements section 2.1 makes the pinned revision the authority for operator shape, dtype, KV cache, and communication: repository `git@gitlab.basemind.com:sys/stepcast/vllm.git`, branch `xwx/step4pro-fa3-optimus`, commit `607d1641ee3fec43653fca510d717725828890c2`.
- Network reachability is fine: `gitlab.basemind.com` port 22 accepts connections, and HTTPS returns 302.
- Authentication fails: `ssh -T git@gitlab.basemind.com` and `git ls-remote git@gitlab.basemind.com:sys/stepcast/vllm.git HEAD` both return `Permission denied (publickey,password)`.
- `~/.ssh/` contains only `authorized_keys` and `known_hosts`. No private key is present.
- No local clone exists. A filesystem search under `/data/ycfeng` found no `step4pro.py`, `optimus_fa4.py`, or `rjob-step4pro-*.sh`.

**Resolution:** SSH access became available during the current session. Read-only clone and `git ls-remote` succeeded.

**Impact:** The authoritative implementation cannot be read, so operation definitions cannot be derived or given provenance as the task requires.

**Remaining requirement:** Keep the exact branch head and source checkout hash in the final provenance record.

No workaround or fallback has been applied.

## ISSUE-004 — The cached StepCast image does not contain the target implementation

**Status:** Superseded for runtime selection. The branch scripts provide a candidate image and its manifest digest; the older cached image remains unsuitable.

**Observed facts:**
- Two StepCast images are cached locally: `hub.stepfun-inc.com/stepcast/stepcast:2026-07-14-server-vllm-0.19.0.post15-8a8f1b3f` and `hub.stepfun-inc.com/stepcast/stepcast:vllm-openai-v0.19.0`.
- Inspecting the post15 image at `/usr/local/lib/python3.12/dist-packages/vllm` shows:
  - `model_executor/models/` contains `step3p5.py`, `step3p5_mtp.py`, `step4_edge.py`, and other Step models. It does **not** contain `step4pro.py`.
  - `v1/attention/backends/` does **not** contain `optimus_fa4.py`.
  - `model_executor/layers/fused_moe/` **does** contain `optimus_fp8_moe.py` and `optimus_moe.py`.
  - `model_executor/models/registry.py:218` maps `"Step4ForCausalLM": ("step3p5", "Step3p5ForCausalLM")`, routing Step4 to the Step3p5 implementation. The architecture name the requirements document expects, `Step4ProForCausalLM` (section 3.2), is absent from the registry.
- The internal registry cannot be enumerated without credentials: `https://hub.stepfun-inc.com/v2/stepcast/stepcast/tags/list` returns `UNAUTHORIZED`, and `/v2/` returns HTTP 401.

**Root cause:** The cached images predate or exclude the Step4Pro branch. Two of the three implementation files named in requirements section 2.1 are absent, and the model registry does not expose the target architecture.

**Impact:** The older cached image cannot serve as the implementation source. A candidate latest image is now identified, but it must be explicitly accepted as the runtime contract.

**Required decision:** Confirm use of the branch-provided image tag and manifest digest.

No workaround or fallback has been applied. The Step3p5 implementation has not been treated as a proxy for Step4Pro.

## ISSUE-005 — The requirements shape conflicts with the existing AIC Step4-Pro-V4 model

**Status:** Open; blocks the model-definition decision.

**Observed facts:**
- The task statement names the target `step4-pro-latest`. The requirements document never uses the word "latest"; it consistently says v4, including the filename, the `step4pro_v4_shape_manifest.json` reference, and `--served-model-name step4pro-v4-perf` in section 3.2.
- AIC already registers `stepfun-ai/Step4-Pro-V4` in `src/aiconfigurator/sdk/common.py:887`, with configuration at `src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V4_config.json`.
- The existing AIC V4 shape differs from the requirements shape on every load-bearing dimension:

| Parameter | Requirements section 2.2 | AIC Step4-Pro-V4 | AIC Step4-Pro-V3 | AIC Step4-Pro-V1 |
| --- | ---: | ---: | ---: | ---: |
| hidden size | 7168 | 9216 | 12288 | 6144 |
| trunk layers | 78 | 80 | 80 | 80 |
| routed experts | 896 | 384 | 1024 | 512 |
| topk | 16 | 8 | 16 | 8 |
| MoE intermediate | 3584 | 3584 | 2048 | 2048 |
| MTP layers | 1 | absent | 3 | 3 |
| Full attention type | MFA, Q64 / KV1 / head dim 512 | MQA | MQA, Q96 / KV12 / head dim 128 | MFA, Q64 |
| Full / SWA layer split | 20 / 58 | not expressed | not expressed | not expressed |

- No shape manifest is available to arbitrate, because `step4pro_v4_shape_manifest.json` and its stated hash `53103019932b93b20a60b6f9dfe6154be6330befdd6a6fa2f6cb67278fc03fde` are absent (ISSUE-001).

**Root cause:** The requirements document describes a fourth, previously unmodelled shape that shares the name "V4" with an existing and materially different AIC model.

**Impact:** Whether to add a new model or modify the existing `Step4-Pro-V4` is undecidable, and choosing wrongly either corrupts committed V3/V4 roofline evidence or creates a duplicate identity. The AIC model name for this task is also unsettled.

**Required decision:** Confirm that `step4-pro-latest` denotes the 78-layer / hidden 7168 / 896-expert shape, confirm the AIC model identity string to register, and confirm that the existing `Step4-Pro-V4` entry must remain untouched.

No workaround or fallback has been applied. No model configuration has been edited.

**Resolution update (2026-08-13):** The user selected canonical identity `stepfun-ai/Step4-Pro-Latest`; the existing `Step4-Pro-V4` entry remains unchanged. The 78-layer shape is authoritative under choice A. The only remaining shape-contract question is whether to reconstruct the missing manifest with a new hash or wait for the original file.

**Final resolution update (2026-08-13):** The user selected reconstruction.
The new manifest must be explicitly labeled as reconstructed and must use its
own computed SHA256. The unavailable original hash must remain recorded only as
an unverified historical reference.

## ISSUE-006 — B300 is cross-zone, so the H800 launch recipe does not transfer

**Status:** Resolved with a verified recipe. Recorded because it changes every B-card launch and every data path in this task.

**Observed facts:**
- B300 nodes are in zone `shai-cn-qingyang-cm`. This workspace is in zone `shai-cn-shanghai-sj`. All H800 nodes are in the workspace zone, which is why the handbook recipe never had to address this.
- Launching without `--image` is refused: `cross-zone launch without --image is not allowed (it mounts the workspace NFS across zones and causes many D-state processes), please specify --image`.
- Launching with `--volume` is refused even when `--image` is supplied: `--volume is not allowed for cross-zone launch even with --image (--volume still mounts the workspace NFS across zones), please remove --volume`.
- Consequently `/data/ycfeng` is not visible inside a B300 worker. The only sizeable writable mount is `/jobutil` (1.0 TB). No `/gpfs`, `/jfs`, or `/shared` exists.
- `brainctl cp` hung with no output and no error for over 100 seconds and had to be killed. It is not usable here.
- `tar` over `brainctl exec` works in both directions and is the documented advanced path. `tar` is present in the container at `/usr/bin/tar`.

**Root cause:** The platform blocks workspace-NFS mounts across zones to avoid D-state processes. B300 and this workspace are in different zones, so no shared filesystem can be relied upon.

**Impact:** Collector code, model configs, and any input artifacts must be pushed into the worker, and all measured outputs must be pulled back out. Nothing can be read from or written to `/data/ycfeng` by a B300 worker directly. Any collection script that assumes a shared `/data` path will fail.

**Verified recipe:**

```bash
# 1. Launch. --image is mandatory; --volume is forbidden.
/kubebrain/rlaunch \
  --charged-group=b300_train_infra --private-machine=group \
  --gpu=8 --cpu=64 --memory=409600 --backoff-limit=1 --enable-sshd=false \
  --image <IMAGE> -- bash -lc '<COMMAND>'

# 2. Find the replica name.
/kubebrain/brainctl get replica -n shai-core | grep <job-fragment>

# 3. Push inputs (local -> worker).
tar cf - -C <LOCAL_DIR> <NAME> \
  | /kubebrain/brainctl -n shai-core exec -i replica/<REPLICA> -- tar xf - -C <REMOTE_DIR>

# 4. Pull outputs (worker -> local). Redirect to a file, then extract.
/kubebrain/brainctl -n shai-core exec replica/<REPLICA> -- tar cf - -C <REMOTE_PARENT> <REMOTE_DIR> \
  > out.tar
tar xf out.tar -C <LOCAL_DEST>
```

**Round-trip proof (2026-08-13):**
- Push: a local probe file arrived intact on the worker, verified by reading it back through `exec`.
- Pull: `nvidia-smi` output generated on the worker was retrieved as a 10240-byte tar and extracted locally, containing `NVIDIA B300 SXM6 AC, 275040 MiB, 10.3`.
- Note on step 4: piping the pull directly into `tar xf -` produced empty output and exit code 1. Redirecting the stream to a file first, then extracting, works reliably. Use the two-step form.

## ISSUE-007 — Production implementation cannot safely begin from the current dirty linked worktree

**Status:** Resolved on 2026-08-13.

**Observed facts:**
- The current checkout is already a linked worktree on branch `step4-pro`.
- It has 37 modified tracked files and many untracked historical artifacts.
- The untracked files include the existing Step4-Pro-V3/V4 configs and their
  focused tests, so a clean worktree created directly from HEAD would not
  contain the full current Step4 foundation.
- Project-local `.worktrees/` is not currently ignored.
- Repository rules prohibit resetting, stashing, committing, moving, or
  overwriting existing user changes without approval, and require existing
  modifications to be stashed or committed before large-scale changes.

**Root cause:** The earlier Step4 tasks were implemented but not consolidated
into a clean commit that can be used as the base of a new isolated branch.

**Impact:** Starting production edits in place risks mixing this task with
historical work. Creating a new worktree from HEAD risks implementing against
an incomplete baseline. Either path would weaken traceability and could lose
or duplicate existing work.

**Required decision:** After Q3 is resolved, obtain explicit approval for a
safe consolidation or isolation method before writing production code.

No reset, stash, commit, worktree creation, or production code edit has been
performed.

**Recommended resolution:** Keep the existing linked worktree, create
`task/step4-pro-latest-b300`, and checkpoint the existing relevant Step4
source/config/test/task-document state before new implementation. Exclude
generated outputs and the separate pinned vLLM checkout. This avoids both
destructive cleanup and a second incomplete worktree.

**Resolution in progress:** The user approved the recommended option. Branch
`task/step4-pro-latest-b300` exists and the audited 86-file baseline is staged.
ISSUE-007 closes only after the checkpoint commit and baseline verification
complete.

**Resolution:** Created branch `task/step4-pro-latest-b300` and baseline commit
`4f2b0c31` from an exact 86-file allowlist. All excluded caches, generated
outputs, H800 data, historical result directories, and the pinned vLLM clone
remained untracked. Tracked status was clean after the commit. Baseline test
failures are a separate consistency issue tracked as ISSUE-009.

## ISSUE-008 — Git LFS missing during approved branch creation

**Status:** Resolved on 2026-08-13.

**Observed facts:**
- `git switch -c task/step4-pro-latest-b300` created the branch, then returned
  nonzero because `.git/hooks/post-checkout` could not find `git-lfs`.
- The hook is valid and intentionally invokes `git lfs post-checkout`.
- `git lfs version` initially returned “git: 'lfs' is not a git command”.
- Ubuntu package metadata exposed `git-lfs 3.4.1-1ubuntu0.4` through the
  company Artifactory mirror.

**Root cause:** The host lacked the Git LFS package required by the existing
repository hook.

**Resolution:**
- Installed `git-lfs=3.4.1-1ubuntu0.4` with `sudo -n apt-get install -y
  git-lfs`.
- Re-ran the unchanged post-checkout hook with the current revision; it exited
  0.

No hook removal, skip flag, or fallback was used.

## ISSUE-009 — Checkpointed Step4 baseline had 128 failing tests

**Status:** Resolved on 2026-08-13.

**Observed facts:**
- Baseline commit: `4f2b0c31`.
- Focused test run collected 987 tests: 859 passed and 128 failed.
- Failures occur only in four files:
  - `test_factorized_attention_runtime_spec.py`: 79;
  - `test_step4_pro_v1.py`: 41;
  - `test_step4_pro_v1_roofline.py`: 4;
  - `test_deepseek_v4_runtime_spec.py`: 4.
- Representative missing contracts include
  `FactorizedAttentionRuntimeSpec`, `FullAttentionConfig`, updated
  `Step4MFAAttentionConfig` APIs, and expected KV-memory behavior.
- No Step4-Pro-Latest production code existed when the failures were observed.

**Initial root-cause hypothesis:** The previous dirty worktree combined
production changes from completed Step4 work with untracked tests from one or
more incomplete or separate work streams. This hypothesis is not yet accepted;
parallel read-only investigations are in progress.

**Impact:** A failing baseline makes later regressions ambiguous. Under the
task harness, implementation and GPU collection cannot proceed.

**Required resolution:** Establish provenance and ownership for each failure
cluster, then either repair the true baseline root cause or obtain explicit
approval to remove separately incomplete tests from this branch. No test
skipping or expectation weakening is allowed.

### Root-cause update

- 83 failures come from two DSV4 runtime-spec RED test files that the original
  task later marked obsolete.
- The remaining 45 failures are relevant baseline blockers:
  - 41 Step4-Pro-V1 model/config/parser/KV failures caused by incompatible V1
    attention contracts being mixed in one checkpoint;
  - 4 `SOL`/`SOL_FULL` formula-path loader failures.
- No existing branch, worktree, stash, or tracked commit contains the missing
  complete V1 implementation.
- The next required decision is whether to restore the previously approved
  Full/HCA V1 contract or redefine V1 around the newer shared-MFA contract.

### Scope clarification

- The pinned Step4-Pro vLLM implementation for Latest is not the historical
  V1 Full-Attention-plus-HCA graph. It uses shared-KV Full MFA/MQA together
  with sliding-window GQA/SWA.
- The V1 repair decision must not change the Step4-Pro-Latest implementation.
- Recommended resolution: restore the historical V1 contract only for the
  existing V1 model, then implement Latest separately from the pinned vLLM
  graph.
- The task owner confirmed this clarified combined decision on 2026-08-13.
- Remaining ISSUE-009 work is limited to restoring the historical V1 contract,
  resolving the four formula-only loader tests, and obtaining explicit
  ownership/removal decisions for the obsolete runtime-spec tests.

### Final test-ownership audit

- Remove, only with explicit permission:
  - `tests/unit/sdk/database/test_factorized_attention_runtime_spec.py`
  - `tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py`
- Retain:
  - all four formula-only `SOL`/`SOL_FULL` cases in
    `tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
- Production defect to repair:
  - `src/aiconfigurator/sdk/operations/dsv4.py` calls `load_data()` before
    checking formula-only modes.

**Resolution approval:** On 2026-08-13 the task owner explicitly approved
deleting the two obsolete files, retaining the four formula-only tests, and
repairing the production source-ordering defect. ISSUE-009 remains open only
until the resulting baseline test run is green.

### Resolution result

- Deleted the two approved obsolete DSV4 runtime-spec files.
- Restored the historical V1 Full/HCA schema, parser, graph, validation, and
  KV accounting.
- Removed the invalid test that required factorized-MFA runtime specs for V1;
  it contradicted the confirmed historical V1 contract and did not represent
  pinned Latest vLLM.
- Retained the four formula-only HCA tests and fixed the source ordering.
- Final adjusted baseline: `899 passed`, `0 failed`, `51.84s`.
- Ruff, format, JSON, and whitespace checks all pass.

Latest remains a separate implementation target and must not reuse the V1
Full/HCA graph.

## ISSUE-010 — Temporary audit file reported as potentially sensitive

**Status:** Closed by owner decision on 2026-08-13.

**Path:**
`/data/ycfeng/tmp/aic_failure_domain1/codex_lane3_scope_baseline.md`

**Resolution:** The task owner selected retain/no-action and explicitly stated
that the current environment is sufficiently secure. The file has not been
read, deleted, modified, moved, permission-changed, quoted, or otherwise
processed. No credential action will be taken.

## ISSUE-011 — Pinned vLLM has no Step4Pro MTP1 implementation

**Status:** **Resolved as deferred on 2026-08-14.**

**Observed facts:**

- `Step4ProForCausalLM` is registered as the Step4Pro trunk model and does not
  construct an MTP predictor:
  `vllm/model_executor/models/step4pro.py:626-649`.
- The only relevant MTP registry entry is `Step3p5MTP`, and it constructs
  `Step3p5DecoderLayer`:
  `vllm/model_executor/models/registry.py:616`;
  `vllm/model_executor/models/step3p5_mtp.py:171-190,286`.
- Step4 speculative configuration is rewritten to the Step3.5 MTP
  architecture:
  `vllm/config/speculative.py:357-362`.
- The requirements document says Step4Pro MTP1 construction still needs to be
  completed: `task_memory/step4pro_v4_external_simulator_requirements.md:113-115`.

**Root cause:** The requirements' MTP1 experiment scope is broader than the
pinned vLLM Step4Pro implementation. Reusing `Step3p5MTP` would violate the
owner's explicit requirement that Latest operations come from the actual
Step4Pro vLLM implementation.

**Impact:** MTP-off Latest operation definition, runtime trace, collection, and
simulation can proceed. MTP1 graph definition, measurement, and simulation are
explicitly deferred and are not part of the current acceptance claim.

**Resolution:** The task owner approved continuing with MTP-off
`stepfun-ai/Step4-Pro-Latest` correctness, Collector work, B300 measurement,
and prefill/decode simulation. MTP1 structure tests, MTP1 measurement, and
MTP1 simulation remain postponed until a native Step4Pro MTP implementation
source and runtime contract are supplied. `Step3p5MTP` is not used as a
substitute.

**Evidence:** `/data/ycfeng/tmp/step4_mtp1_boundary_audit_20260814.txt`

## ISSUE-012 — Pinned-vLLM smoke/runtime trace moved to an external session

**Status:** Scope resolved on 2026-08-14; external result pending.

**Observed facts:**

- Previous local launches proved B300 allocation and hardware visibility but
  did not complete the required service smoke.
- Known failures were command/entrypoint transfer and a read-only `/jobutil`,
  not a proven Step4Pro runtime failure.
- At handoff there were zero active matching RJobs, Replicas, or local smoke
  processes.

**Root cause of ownership change:** The task owner chose to execute the
whole-model pinned-vLLM smoke/runtime-provider trace in another session and
return its results later.

**Impact:**

- This session must not launch or monitor the whole-model smoke.
- AIC-side implementation, exact operation-provider collection, testing, and
  simulation remain active.
- Final runtime/provider sign-off remains pending until the external report is
  supplied.

**Resolution:** Created
`pinned_vllm_b300_smoke_runtime_trace_execution.md` with fixed identities,
handbooks, reusable artifacts, known failure roots, bounded launch rules,
acceptance criteria, cleanup requirements, and the report schema.

## ISSUE-013 — Full Replica inventory exceeds the diagnostic memory scope

**Status:** Resolved on 2026-08-14.

**Observed failure:** The first external Stage 1 attempt was killed with exit
`137` before creating an RJob. The failing command was the namespace-wide
`brainctl get replica` query inside `MemoryMax=2G`.

**Root cause:** The full namespace inventory can materialize enough Replica
state to exceed the mandatory 2 GiB diagnostic scope. This was an inventory
query problem, not a B300 scheduling or model-runtime failure.

**Resolution:** Query Replicas with the server-side label selector
`rjob.brainpp.cn/rjob-name=<exact-job-name>`. Keep exact-name RJob queries and
the 60-second diagnostic boundary.

**Verification:** A nonexistent selector query returned exit `0`,
`No resources found`, without exceeding the memory scope.

## ISSUE-014 — Live launch was incorrectly limited to 60 seconds

**Status:** Resolved on 2026-08-14; live verification pending.

**Observed failure:** The second Stage 1 attempt created the RJob and reached
`Queued`, but the wrapper sent SIGTERM at `60s`; the launcher returned exit
`124` and stopped the RJob.

**Root cause:** The handbook's 60-second bound applies to inventory and
diagnostic queries. It is too short for a live B300 allocation whose explicit
worker-ready deadline is longer.

**Resolution:** Bind the launch process timeout to
`READY_TIMEOUT_SECONDS`; retain 60 seconds for inventory/cleanup commands and
the outer hard timeout for the full Stage 1 attempt.

**Verification:** A RED static test reproduced the mismatch, then passed after
the timeout binding was corrected. The interrupted RJob and Replica were
explicitly deleted; final matching counts were both `0`.

## ISSUE-015 — New host OOM requires a smaller controller scope and bounded I/O

**Status:** Mitigation implemented on 2026-08-14; live verification pending.

**Observed failure:** The owner reported that the host OOMed while the external
B300 task was active and suspected the then-current 5 GiB controller setting.
No new kernel or transient-scope OOM stack has yet been preserved for that
specific event.

**Root-cause assessment:** The exact trigger is not proven. However, the
control path had avoidable peak-memory risks: an extra `bash -c` process around
the live launcher, prior namespace-wide inventory behavior, and potential
whole-artifact reads. These risks are independent of model GPU memory and can
increase host pressure.

**Resolution:**

- Enforce `MemoryMax=3G` for controller queries and live launch.
- Permit only exact-name RJob and exact-label Replica queries.
- Keep patches, manifests, tar streams, evidence, and logs disk-backed under
  `/data/ycfeng/tmp`; prohibit whole Git pack/bundle/tar/log reads into memory.
- Launch `systemd-run` directly under `setsid`.
- Delete and poll the remote resource before terminating the local launcher.

**Verification:** Focused static tests passed `10/10`; all three shell syntax
checks and the focused whitespace check exited `0`. A live attempt is still
required to prove that the 3 GiB path completes without host OOM.

## ISSUE-016 — Source probe waited for the worker holder before using exec

**Status:** Resolved on 2026-08-15.

**Observed failure:** The first 3 GiB source probe reached a Running B300
worker, but `brainctl exec` returned `websocket: bad handshake`.

**Root cause:** The wrapper synchronously waited for `brainctl rjob launch`.
That client remained attached until the worker's bounded sleep ended, so the
subsequent exec raced with container exit. Missing `-i` made streamed exec less
reliable but was not the primary lifecycle error.

**Resolution:** Run the live launcher in its own process group, poll the exact
Replica while it is alive, use `brainctl exec -i` for remote probes and
transport, then delete the RJob before terminating the launcher.

**Verification:** Live probe `s4p-src-0815-125214` completed all remote exec
steps and reported `SOURCE_PROBE=PASS`.

## ISSUE-017 — Rebuilding or transporting the local Git pack risks host OOM

**Status:** Resolved on 2026-08-15.

**Observed risk:** The local pinned checkout contains a 564 MiB pack and is
573 MiB under `.git`. The original source probe transferred the whole checkout,
and the one-GPU wrapper reran `git pack-objects`.

**Root cause:** Source identity and source transport were coupled to the full
checkout even though the runtime needs only the image-to-pinned patch, two
manifests, a small commit/tree identity pack, and the remote script.

**Resolution:** Reuse the already verified disk-backed payload. Its transferred
size is about 404 KiB. Large Git packs are neither read into memory nor
transferred, and remote hashing uses 1 MiB streaming reads.

**Verification:** Stage 1 matched `2103/2103` pinned source files and imported
the intended source union. The final exact RJob/Replica counts were both `0`.

## ISSUE-018 — Current identity cannot mount the pinned qy1-pt model bucket

**Status:** Resolved as non-blocking on 2026-08-15.

**Observed failure:** The exact one-GPU predict-only check passed with 10
candidate B300 nodes, but live RJob admission failed before resource creation:

```text
JuiceFS s3://oss-qy1.i-stepfun.com/qy1-pt:/mnt/qy1-pt validation failed:
no right to access the bucket: qy1-pt
```

**Root cause:** The current platform identity lacks permission to mount the
authoritative model bucket used by both pinned recipes. Predict-only checks
capacity but does not prove mount authorization.

**Corrected impact:** The unchanged checkpoint-backed recipe cannot run under
this identity. It does not block the required performance/provider tests,
because the parent requirements explicitly state that the target checkpoint
is unavailable and require synthetic `config.json` plus
`--load-format dummy`.

**Resolution:** Remove real model and tokenizer mounts from the synthetic test
path. Use vLLM's own `DummyModelLoader` for random weight initialization and
token-ID inputs with tokenizer initialization disabled. Real checkpoint access
remains optional future work for routing distribution, generated-text quality,
and measured MTP acceptance rate.

**Cleanup evidence:** No RJob was created. Final exact RJob and label-selected
Replica queries returned zero resources; no matching local process remains.

## ISSUE-019 — Image-native Optimus quant kernel is incompatible with B300 SM103

**Status:** RESOLVED BY OWNER AUTHORIZATION on 2026-08-15; live verification
pending.

**Observed failure:** The synthetic model loads and selects
`OPTIMUS_FP8`, but its first MoE profile forward fails after
`Optimus.per_token_group_quant_fp8` with:

```text
CUDA error: no kernel image is available for execution on the device
```

**Root cause evidence:**

- B300 reports compute capability `(10, 3)`.
- Image torch `2.10.0+cu129` supports ordinary CUDA allocation/fill on this
  GPU.
- Triton `count_expert_num_tokens` passes for `8192×16` routes and totals
  `131072` rows.
- Image-native Optimus quant returns tensors of shapes `[8192,896]` and
  `[8192,7]`, but poisons the subsequent CUDA launch.
- In separate processes, `count → zeros` passes while
  `Optimus quant → zeros` fails.
- The image-native Optimus library is `3.17.8.dev0`, built for
  torch2.10/cu129.
- The required 3.23.24 wheel contains only a torch2.8/cu128 native library;
  loading it in torch2.10 fails with an undefined C10 CUDA symbol.

**Impact:** Optimus FP8/DeepGEMM cannot complete its first routed-expert
forward. Therefore one-GPU service readiness and two-node DeepEP execution
cannot yet pass under the fixed runtime identity.

**Required resolution:** Supply either:

1. a B300 SM103-compatible Optimus native library built for
   torch2.10/cu129 and compatible with the fixed 3.23.24 Python FA4 overlay; or
2. a verified replacement image containing that library; or
3. explicitly authorize a small pinned-source overlay that replaces only
   `torch.ops.Optimus.per_token_group_quant_fp8` with the already installed
   Optimus JIT CuTe DSL `per_token_group_quant_fp8`.

Changing pinned vLLM to a different activation-quant implementation would
change the source/provider contract and requires explicit owner authorization;
the owner authorized this exact single-point overlay on 2026-08-15.

**Validated candidate evidence:** On B300 capability `(10,3)`,
`optimus-jit 0.1.10.post8+gitcfde41ba` produced FP8 output `[8192,896]` and
scale output `[8192,7]`; a subsequent CUDA fill passed. The pinned-vLLM Triton
quant implementation also passed the same test. The Optimus JIT candidate is
preferred because it keeps activation quantization within the Optimus family
while preserving the original Optimus DeepGEMM GEMMs.

**Authorized boundary:** Replace only calls to
`torch.ops.Optimus.per_token_group_quant_fp8` in the runtime copy of
`optimus_fp8_moe.py` with the installed Optimus JIT CuTe DSL
`per_token_group_quant_fp8`. Do not change FA4, expert GEMM, routing,
DeepGEMM, DeepEP, shape, precision, or the repository checkout.

**Cleanup evidence:** All diagnostic RJobs and label-selected Replicas were
queried by exact name and returned `0`.

## ISSUE-020 — Initial Full-MFA QKV benchmark omitted Q tail RoPE

**Status:** Resolved on 2026-08-15 before B300 measurement.

**Observed defect:** The first QKV GREEN implementation benchmarked only
Optimus K RMSNorm followed by K tail RoPE for the Full-MFA structure.

**Root cause:** `normalized_tensors=("k",)` was incorrectly read as meaning
that only K belongs to the operation. It describes only which tensors receive
normalization. The pinned forward path performs tail RoPE on both Q and K, and
Q head count is part of the structural key for that reason.

**Resolution:** The benchmark now applies
`Step4ProAttention._tail_rope` directly to
`Q[num_tokens,64,512]`, then applies pinned `OptimusRMSNorm` and the same
tail-RoPE method to `K[num_tokens,1,512]`. A runtime output-shape probe requires
both outputs before timing.

**Verification:** The new RED test failed once because the two-output contract
was absent, then passed. Provider-focused tests pass `31/31`, the final focused
suite passes `155/155`, and the full Collector suite passes `352/352`. No
measurement row existed before the correction.

## ISSUE-021 — Attention benchmark ignored hybrid KV physical-page padding

**Status:** Static implementation resolved and regression-verified on
2026-08-15; B300 runtime measurement remains pending under ISSUE-018.

**Observed failure:** The initial Attention implementation used contiguous
logical cache tensors and capped generation cases using logical K/V bytes. It
did not represent the pinned hybrid allocator's physical page padding, Full-MFA
block stride, or SWA retained-window release.

**Root cause:** The standalone Collector built one attention structure at a
time and therefore bypassed the whole-model
`unify_kv_cache_spec_page_size -> _reshape_attention_kv_cache` path. The AIC
config also stored only token page size and logical elements per token, so the
missing physical layout could not reach operation keys or allocation reports.

**Impact:** The current 139-case generation population is not valid. Full MFA
understates physical allocation by up to 4x relative to its aliased logical
payload, while long-context SWA over-allocates the full history instead of the
live window plus in-flight query chunk.

**Planned root-cause fix:** Carry physical page bytes, block stride, and `NHD`
layout from config to AIC operations, Collector cases, CSV rows, and consumers;
construct cache views through pinned `_reshape_attention_kv_cache`; populate
SWA block tables from the live materialized range; recompute the cache cap from
physical bytes.

**Verification:** The focused RED suite fails `11/61` exactly on the missing
contract. Log:
`/data/ycfeng/tmp/step4_attention_physical_layout_red_20260815.log`.

**Additional simulator root cause:** Point-in-time SWA residency falls when an
old block is released. At 513 tokens the whole graph occupies `204472320`
physical bytes, while at 640 tokens it occupies `174063616` bytes. The existing
capacity inverse assumed a monotonic byte curve, and its equal-value guard also
mistook temporary page plateaus for permanent saturation.

**Simulator resolution:** Preserve resident allocated bytes for reporting, add
a separate monotonic peak-allocated contract, use that peak in memory/OOM
checks and native KV-capacity estimation, and bound the inverse search by the
model context length.

**Focused verification:** The new contract first failed `6/260`, then passed
`260/260`. A `204472320`-byte budget safely reaches token 640 but not token
641. Logs:

- `/data/ycfeng/tmp/step4_physical_kv_peak_red_20260815.log`;
- `/data/ycfeng/tmp/step4_physical_kv_peak_green_20260815.log`.

**Regression verification:** Collector passed `358/358`; broad SDK passed
`1233/1234`, with only the pre-existing unrelated ISSUE-019 failure. Ruff
check, Ruff format, and `git diff --check` passed.

## ISSUE-022 — Formal prefill peak KV must include the in-flight chunk

**Status:** Resolved for exact batch=1 chunk sweeps on 2026-08-15. Batch>1
global-budget scheduling remains explicitly unsupported by the static context
path and requires a scheduler-aware driver.

**Observed gap:** The new monotonic peak contract exactly models MTP-off decode
growth with one in-flight token. vLLM chunked prefill allocates the live SWA
window plus the current query chunk before releasing skipped blocks. Therefore
an 8192/32768/65536-token prefill chunk can require materially more SWA pages
than the five-page decode peak.

**Root cause:** `RuntimeConfig` and `BaseBackend._get_memory_usage` currently do
not carry `max_num_batched_tokens` or an explicit per-request in-flight chunk
size into the model KV-allocation contract.

**Impact:** Decode capacity and current Attention Collector allocation are
correct. Formal prefill OOM/peak-HBM results are not yet acceptable because
they would not vary with the required chunk-size sweep.

**Required root fix:** Add an explicit chunk/in-flight token input, reproduce
the pinned scheduler order of release-before-allocation, and test the
8192/32768/65536 workload points. Do not use a scaling factor or infer peak
memory from logical KV bytes.

**Resolution:** The model peak-KV contract now accepts keyword-only
`in_flight_tokens`, while the one-token default remains unchanged for decode
and capacity inversion. `RuntimeConfig.max_num_batched_tokens` is mapped only
for `batch=1` `static_ctx`, where the global vLLM budget equals the
per-sequence chunk. Explicit batch>1 use raises `NotImplementedError` rather
than inventing a budget split. The bounded periodic formula matched an
independent brute-force oracle for `1,094,700` combinations. Focused tests
passed `14/14`; the focused SDK regression passed `351/351`.

## ISSUE-029 — DeepEP operation test expected the removed pre-consumer path

**Status:** Resolved on 2026-08-15.

**Observed failure:** The broad SDK regression failed
`test_provider_specific_deepep_rejects_generic_perf_database` before its
expected `NotImplementedError`, because the test constructed a DeepEP
operation without the required `fp8_block` quantization.

**Root cause:** The test still described the earlier state where provider-tagged
DeepEP had no measured consumer. The current operation has an exact
provider-specific data path and correctly validates its pinned quantization,
topology, dispatch format, and token contract before querying data.

**Resolution:** Keep the production `fp8_block` gate unchanged. Update the
test to construct the exact pinned DeepEP identity and assert
`PerfDataNotAvailableError` when the exact structural key is absent. This
preserves the no-fallback contract rather than testing an obsolete
`NotImplementedError`.

**Verification:** The focused operation suite passed `7/7`; Ruff check and
format check passed. Evidence:
`/data/ycfeng/tmp/step4_deepep_test_contract_20260815.log`.

## ISSUE-023 — Source-only AIC transfer omitted distribution metadata

**Status:** Resolved and verified on B300.

**Observed failure:** The first representative Attention worker completed the
pinned-vLLM reconstruction and Optimus FA4 overlay, then failed before kernel
execution:

```text
importlib.metadata.PackageNotFoundError:
No package metadata was found for aiconfigurator
```

**Root cause:** The cross-zone payload transferred `src/aiconfigurator` and
Collector source but not the `aiconfigurator-0.10.0.dist-info` installed-package
metadata. `src/aiconfigurator/__init__.py` deliberately obtains `__version__`
through `importlib.metadata.version("aiconfigurator")`, so a source tree alone
is not a complete executable package.

**Impact:** No Attention kernel executed and no B300 latency row was accepted.
The RJob and Replica were deleted; final exact-name inventories were empty.

**Root fix:** Transfer the exact `aiconfigurator-0.10.0.dist-info` directory
from the recorded `aic-step-design` environment alongside the source payload,
then include that metadata directory in remote `PYTHONPATH`. Product source is
unchanged; no version fallback or `__init__.py` patch is introduced.

**Verification:** The second worker imported AIC `0.10.0`, executed all four
representative provider/phase cases, and passed exact consumer queries for all
four measured rows. Final matching RJobs and Replicas were both `0`.

## ISSUE-024 — Runtime profile used the image-label spelling as package version

**Status:** Resolved and fully verified on B300.

**Observed failure:** The first full collection stopped before case execution:

```text
requires package version '0.19.0.post20.dev26.gc820e5ae1',
but the active runtime reports '0.19.0.post20.dev26+gc820e5ae1'
```

**Root cause:** The Collector profile copied the dot-separated version fragment
from the Docker image tag. The installed vLLM package exposes the valid PEP 440
local-version form with `+gc820e5ae1`. The exact runtime validator correctly
rejected this mismatch.

**Impact:** The model plan and pinned imports passed, but no full-grid case
executed and no row was accepted. The worker was deleted and exact-name RJob
and Replica inventories were empty.

**Root fix:** Set the profile `package_version` to the measured installed
package identity `0.19.0.post20.dev26+gc820e5ae1`. Keep the Docker image tag
unchanged because its registry name legitimately uses dots.

**Verification:** Runtime-profile RED was `2 failed, 1 passed`; after the
manifest correction the focused Collector file passed `19/19`. The subsequent
full B300 run produced `50` context and `149` generation rows with zero
Collector errors. All `199/199` AIC consumer queries exactly reproduced the
measured latency and returned `source="silicon"`.

## ISSUE-B300-001 — Pinned `ep_gather` uses a non-power-of-two Triton block

**Status:** RESOLVED AND LIVE-VERIFIED on 2026-08-15.

**Observed failure:** After the authorized quant overlay and both Optimus
DeepGEMM GEMMs succeeded, Triton compilation failed in
`deep_gemm_utils.py::_fwd_kernel_ep_gather`:

```text
off_d = tl.arange(0, BLOCK_D)
ValueError: arange's range must be a power of 2
```

**Root cause:** `ep_gather()` sets
`BLOCK_D=min(hidden_size,1024)` and checks only divisibility. For smoke latent
hidden `896`, it selects `896`, which Triton rejects. For target latent hidden
`3584`, it selects `1024`, which does not divide the dimension.

**Proposed root fix:** Select the largest power-of-two divisor not exceeding
1024:

```python
BLOCK_D = min(1024, hidden_size & -hidden_size)
```

This preserves the full tensor without padding: `896→128`, `3584→512`.

**Necessity evaluation:** The correction is mandatory for this fixed contract.
The old kernel cannot compile for 896 and cannot legally tile 3584. Changing
the hidden size, backend, or provider would avoid the symptom by changing the
requested workload. The only commit after the pinned revision changes
DeepEP/NVSHMEM launch scripts and does not repair this source.

**Safety evaluation:** The formula changes only kernel tiling. Every dimension
is covered exactly once and each element keeps the same top-k accumulation
order. Every dimension that was legal under the old rule retains the same
block. `ep_gather` is inference-only under `torch.no_grad()` and cannot alter
weights, gradients, optimizer state, router training, or convergence.

**Additional affected path:** After the contiguous path was corrected, warmup
entered `optimus_triton.deep_gemm_ep_gather_masked`, whose helper incorrectly
returns `896` because it checks divisibility but not the Triton power-of-two
constraint. The same reviewed formula is required in an isolated
`optimus_triton` package overlay.

**Live verification:** Both contiguous and masked paths completed model
warmup, a real single request, and four concurrent requests. Strict Optimus
Triton driver signatures were required when changing batch shapes.

## ISSUE-B300-002 — Runtime fixes were previously described with an overly broad safety claim

**Status:** RESOLVED on 2026-08-15.

**Problem:** Earlier wording grouped quantization, gather tiling, import
metadata, cache signatures, and validation assertions under a single “does not
affect training/design” statement. That was too broad: the JIT quant overlay
does change the inference activation-quant implementation and scale encoding,
even though it does not touch training artifacts.

**Root cause:** The review focused on whether model weights and provider
identity changed, but did not clearly separate mathematical tiling equivalence
from numerical quantization equivalence.

**Resolution:**

- classify contiguous and masked `ep_gather` changes as exact inference-tiling
  fixes;
- classify strict driver signatures, `ModuleSpec`, and fallback-gate changes
  as cache/import/test-control changes;
- classify Optimus JIT quant as a necessary pinned-runtime implementation
  change that is not bitwise or quality-equivalence evidence;
- preserve the inherited pinned FA4 pointer compatibility overlay as a
  separate, recipe-owned deviation;
- restrict all conclusions to dummy-weight provider/performance validation.

**Evidence:**

- native quant followed by CUDA work fails with
  `cudaErrorNoKernelImageForDevice`;
- the matching native `3.23.24` library cannot load into torch `2.10`/
  CUDA `12.9`;
- Optimus JIT quant and subsequent CUDA work pass;
- original `ep_gather` fails with Triton's power-of-two error;
- the corrected one-GPU run passes health, one request, four concurrent
  requests, FA4, Optimus FP8/DeepGEMM, and zero-resource cleanup.

**Boundary:** No model-quality or training-convergence conclusion is made.

## ISSUE-B300-003 — `brainctl exec` lost the platform-distributed environment

**Status:** RESOLVED AND LIVE-VERIFIED on 2026-08-15.

**Observed failure:** Both two-node attempts reached two running 8-GPU B300
replicas, but each separately executed smoke process stopped with:

```text
MASTER_ADDR: MASTER_ADDR is required for distributed smoke
```

The holder logs simultaneously proved that `NODE_RANK=0/1`, `NODE_COUNT=2`,
`MASTER_ADDR`, `PROC_PER_NODE=8`, RoCE, GID index 5, and eight GPUs per node
were present in the original worker shell.

**Root cause:** `brainctl exec` creates a new process. It does not inherit the
holder shell's runtime environment. Reading `/proc/1/environ` was also invalid
because PID 1 is not the worker-init/holder shell.

**Root fix:** The holder now writes the required variables and filtered
platform `NCCL_*`/`NVSHMEM_*` values to a quoted, mode-0600 file under
`/home`. Each `brainctl exec` verifies that file, sources it, and checks all
four required distributed variables before starting the pinned smoke.

The launch was also aligned with the already validated multi-node recipe by
requesting `rdma/mlnx_shared=8`, `--topo-group=yes`, and
`NVSHMEM_ENABLE_NIC_PE_MAPPING=1`, without replacing any platform-injected
`NCCL_*` value.

**Static verification:**

- RED: `1 failed, 2 passed` for the missing holder-file contract.
- GREEN: `3 passed`.
- Full E2E static contract: `16 passed`.
- `bash -n`: PASS.
- `git diff --check`: PASS.

**Live verification:** Attempts `s4p-2e-0815-174618` and
`s4p-2f-0815-181337` both restored:

```text
rank 0 -> DATA_PARALLEL_START_RANK=0
rank 1 -> DATA_PARALLEL_START_RANK=8
NODE_COUNT=2
PROC_PER_NODE=8
shared MASTER_ADDR
```

The environment loss is closed. Those attempts then exposed a separate vLLM
head/worker role error recorded as ISSUE-B300-004.

## ISSUE-B300-004 — Non-head DP node must be headless with zero API servers

**Status:** ROOT FIX STATICALLY VERIFIED; FOURTH LIVE LAUNCH AUTHORIZED on
2026-08-15.

**Observed failures:**

1. `s4p-2e-0815-174618` restored both ranks, then the rank-0 coordinator
   rejected the remote engines:

   ```text
   Remote engine 12 must use --headless unless in external or hybrid dp lb mode
   ```

2. `s4p-2f-0815-181337` added `--headless` on rank 1, then the pinned CLI
   rejected its default frontend count:

   ```text
   --api-server-count=8 cannot be used with --headless
   ```

**Root cause:** The multi-node wrapper treated both nodes as API-serving heads.
Pinned vLLM's coordinated internal-DP contract requires:

- `DATA_PARALLEL_START_RANK=0`: normal serving head;
- `DATA_PARALLEL_START_RANK>0`: `--headless --api-server-count 0`.

The pinned source and benchmark fixtures explicitly enforce this split.

**Root fix:** The remote script now derives the role from the already restored
start rank, keeps rank 0 as the only HTTP-serving node, and starts rank 1 with:

```text
--headless --api-server-count 0
```

The headless process uses a bounded wait and accepts only exit `0` or the
expected coordinated shutdown `143`; before success it must still prove
DeepEP HT manager, dispatch, combine, and real-batch markers.

**Static verification:**

- headless-role RED: `1 failed, 7 passed`;
- initial headless GREEN: `11 passed`;
- API-server-count RED: `1 failed, 7 passed`;
- final complete E2E static suite: `17 passed`, `0 failed`;
- shell syntax and whitespace checks: PASS.

**Stop gate:** The external guide prohibits a fourth B300 attempt after three
distinct live failures for the same two-node gate. Therefore no fourth launch
was consumed. The next live run requires an explicit owner decision to reopen
the gate.

**Owner decision:** The owner explicitly reopened the gate on 2026-08-15.
Proceed with one additional live attempt and preserve the result without
silently retrying again.

**Live result:** The fourth attempt accepted the final role split. Its failure
moved to the distinct RDMA/NCCL bootstrap issue recorded as ISSUE-B300-005.

## ISSUE-B300-005 — B300 launch did not inject a usable NCCL HCA

**Status:** RESOLVED BY FULL-RDMA PREFLIGHT on 2026-08-15.

**Observed result:** Authorized attempt `s4p-2g-0815-185302` completed:

- scheduling in `15 seconds`;
- two 8-GPU B300 nodes;
- ranks `0/1`, start ranks `0/8`;
- normal head plus `--headless --api-server-count 0` worker role;
- Gloo connection to all `15` peer ranks from every rank.

It then failed during the first NCCL all-reduce:

```text
RuntimeError: NCCL error: unhandled system error
```

The same log records:

```text
NCCL_IB_HCA=''
DeepEP runtime detected; skipping NVSHMEM_HCA_LIST inheritance because
NCCL_IB_HCA='' becomes empty after stripping NCCL exact-match prefix.
```

No `Using DeepEPHTAll2AllManager`, HT dispatch, HT combine, or real request was
reached.

**Root cause:** The current `brainctl rjob launch` job requested
`rdma/mlnx_shared=8`, but worker-init did not expose a concrete HCA. The pinned
branch's later B300 bootstrap recipe additionally uses host networking,
`mellanox.com/mlnx_rdma=1`, and shared host SHM. The current launcher help does
not expose the same `--share-host-shm` option, so the operational contracts are
not yet equivalent.

**Rejected workarounds:**

- do not invent or hardcode an HCA name;
- do not set `NCCL_IB_DISABLE=1`;
- do not fall back to socket communication;
- do not replace DeepEP HT with another all-to-all backend.

**Required root resolution:** Use a verified launcher/job specification that
reproduces the B300 branch's full RDMA/host-network/shared-SHM contract, or have
the platform inject a valid non-empty `NCCL_IB_HCA`. Re-run only after that
environment gate is proven.

**Cleanup:** Final exact RJob count `0`, Replica count `0`, and related local
process count `0`.

**Resolution:** Adding host networking and
`mellanox.com/mlnx_rdma=1` caused worker-init to inject:

```text
NCCL_IB_HCA==mlx5_bond100,...,mlx5_bond107
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=bond0
```

After removing the stale CUDA 12.8 compat path, all 16 ranks completed NCCL
all-reduce with actual/expected `136.0/136.0`.

## ISSUE-B300-006 — DeepEP Buffer sync requires unavailable shared-host-SHM bootstrap

**Status:** BLOCKING DEEPEP HT LIVE VALIDATION on 2026-08-15.

**Observed sequence:**

1. Full-RDMA predict-only: PASS.
2. 16-rank NCCL preflight:
   - scheduling `16 seconds`;
   - HCA/GID/socket interface injected correctly;
   - rank passes `16/16`;
   - node passes `2/2`;
   - all-reduce actual/expected `136.0/136.0`.
3. Full model run `s4p-2h-0815-202202`:
   - scheduling `16 seconds`;
   - pinned source `2103/2103`;
   - NCCL initialization passed;
   - `Using DeepEPHTAll2AllManager` emitted;
   - every local rank entered `DeepEP HT get_handle`;
   - `deep_ep.Buffer.runtime.sync` failed:

   ```text
   RuntimeError: Failed: CUDA error
   /workspace/DeepEP/csrc/kernels/runtime.cu:83 'unknown error'
   ```

**Root cause boundary:** The later authoritative B300 probe launches with
`--share-host-shm=True` and explicitly initializes NVSHMEM before creating the
DeepEP Buffer. The available `/kubebrain/brainctl` and `/kubebrain/rlaunch`
clients do not expose a shared-host-SHM option, and the expected `rjob` client
is absent on this controller.

**Rejected workarounds:**

- no privileged container or invented host-volume mount;
- no socket/NCCL fallback;
- no replacement all-to-all backend;
- no fabricated DeepEP PASS from the successful NCCL result.

**Required root resolution:** Provide the verified `rjob` client/job API that
supports `--share-host-shm=True`, or add an equivalent documented platform
option. Then run the later branch's standalone DeepEP Buffer probe before
another complete model run.

**Cleanup:** Both the NCCL preflight and full model attempt ended with exact
RJob `0`, Replica `0`, and related local process `0`.

## ISSUE-B300-008 — Explicit NVSHMEM init and DeepEP Buffer use incompatible runtime state

**Status:** BLOCKING DEEPEP BUFFER CONSTRUCTION on 2026-08-15.

**Supplied-tool verification:**

- `/data/ycfeng/stepfun-env-handbook/brainctl` and `/kubebrain/brainctl` have
  identical size and SHA256:
  `06d5fffb00e67633e10e4a6d96752517eda7559230466a63ac86e6a424c839ad`.
- The supplied copy is not executable, but no execution was needed because it
  is byte-identical.
- `brainctl-rjob.md` does not document `--share-host-shm`; it recommends the
  legacy process launcher when the new RJob backend is incompatible.

**Legacy launcher attempts:**

1. Removed unsupported legacy `--name`.
2. Disabled SSHD because the image has no `/usr/sbin/sshd`.
3. Final run used two B300 nodes, host network, both RDMA resources, explicit
   NVSHMEM initialization, and a 16-rank DeepEP Buffer probe.

**Observed final result:**

```text
NCCL PASS:       16/16
NVSHMEM init:    16/16
DeepEP Buffer:    0/16
```

Every rank failed:

```text
Assertion error /workspace/DeepEP/csrc/kernels/runtime.cu:136
'nvshmem_n_pes() == num_ranks'
```

The two nodes exposed tmpfs `/dev/shm` mounts of `245760000k`, unlimited
memlock, and distinct per-node IPC namespaces. Thus the result does not support
a simple shared-memory-capacity explanation.

**Root-cause interpretation:** Python `nvshmem.core` reports initialized for
the 16-rank world, but the DeepEP runtime does not observe a matching PE count.
The likely domain is duplicate/incompatible NVSHMEM runtime state, missing
DeepEP external-runtime attachment support, or package/ABI integration. The
later branch probe is therefore not validated against the installed
DeepEP `1.2.1` runtime.

**Required resolution:** Obtain the exact validated DeepEP/NVSHMEM package
combination or vendor guidance for attaching the external NVSHMEM runtime to
`deep_ep.Buffer`. Do not continue to full-model dispatch/combine until the
minimal Buffer constructor passes.

## ISSUE-B300-009 — Exact later DeepEP fix requires an available launcher contract

**Status:** Deferred by owner after the exact launcher-contract attempt on
2026-08-15. Do not retry in the current phase.

**New evidence:** Commit `9bfd9a610e` is the direct child of the pinned commit
and changes only:

- `rjob-step4pro-2node.sh`;
- new `rjob-step4pro-deepep-probe.sh`.

The fix explicitly combines `nvshmem.init(...)`, NIC-to-PE mapping,
`NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME`, full RDMA/host networking, and
`rjob submit --share-host-shm=True`. Therefore the earlier legacy-launcher
probe did not test the complete commit contract.

**Owner decision:** Apply the exact commit and retry without insisting on the
earlier pinned-script restriction.

**Required validation:** Confirm an installed client genuinely supports the
commit's shared-host-SHM option, run matching predict-only, then run only the
minimal two-node Buffer probe. A complete model retry remains prohibited until
that probe passes. Worker `/tmp` paths must first be adapted to a verified
disk-backed directory without changing the runtime hypothesis.

**Exact validation result:**

- checkout moved to exact `9bfd9a610ea4f2890010702ee7a207cf25edf8de`;
- parent is exact `607d1641ee3fec43653fca510d717725828890c2`;
- changed files are exactly the two expected shell scripts;
- required NVSHMEM/shared-host-SHM markers are present;
- `bash -n` passed for `2/2` scripts;
- model Python/CUDA changed-file count is `0`;
- `brainctl rjob launch --share-host-shm=True --help` exited `1`;
- legacy `brainctl launch ... --share-host-shm=True --help` exited `1`;
- both reported `unknown flag: --share-host-shm`;
- `command -v rjob` and the commit author's hard-coded client path are absent.

**Root cause:** The exact commit has been applied, but its launch API
dependency is not installed on this controller. The available clients cannot
represent the complete job contract, so neither matching predict-only nor the
minimal live probe can be submitted without dropping a required condition.

**Impact:** DeepEP dispatch/combine, the complete two-node model test, `116`
DeepEP rows, and formal fail-fast simulation remain blocked wherever those
exact communication rows are required.

**Required external input:** Provide the standalone `rjob` client/version used
for commit `9bfd9a610e`, or a documented current brainctl/API field that is
semantically equivalent to `--share-host-shm=True`.

**Owner disposition:** Preserve `0/116` accepted rows, do not launch another
DeepEP attempt in this phase, and continue the MTP-off AIC driver and remaining
validation work without H800, generic communication, or synthetic
substitutions.

## ISSUE-025 — Optimus host wrapper depended on an executable bit

**Status:** Resolved and regression-tested.

**Observed failure:** The first Optimus smoke stopped before creating an RJob:

```text
run_b300_optimus_moe_collection.sh: line 16:
run_b300_attention_collection.sh: Permission denied
```

**Root cause:** The shared host wrapper is intentionally a repository script
with mode `0664`, but the Optimus wrapper attempted to execute its path
directly.

**Root fix:** Invoke the script through its declared interpreter:
`exec bash "${REPO_ROOT}/tests/performance/step4_pro_latest/run_b300_attention_collection.sh"`.
No filesystem permission mutation or temporary `chmod` was used.

**Verification:** The new static assertion failed before the change, then all
four Optimus collection-contract tests passed. The subsequent B300 smoke
created the worker and completed normally.

## ISSUE-026 — Optimus metadata failure was non-fatal through `sitecustomize`

**Status:** Resolved and live-verified.

**Observed failure:** The first kernel-complete smoke repeatedly printed:

```text
Error in sitecustomize
ImportError: expected step-optimus==3.23.24, got 3.17.8.dev0
```

Python continued after the `sitecustomize` exception, so the collector produced
rows even though the required runtime identity gate had failed.

**Root cause:** The MoE runtime script did not reproduce the pinned vLLM
script's SHA256-verified `3.23.24` wheel metadata overlay. It also relied on a
`sitecustomize` exception, which Python reports but does not make process
startup fail.

**Root fix:** Download and verify the exact pinned wheel, extract its
`step_optimus-3.23.24.dist-info`, preserve the image-native torch-2.10 Optimus
shared library, and add an explicit runtime identity assertion outside
`sitecustomize`.

**Verification:** The final smoke reports Step Optimus `3.23.24`, the
image-native torch-2.10 shared library path, no `sitecustomize` error, `3/3`
exact `silicon` consumer matches, and zero remaining RJobs/Replicas.

## ISSUE-027 — Optimus CUDA graph capture contaminated the default CUDA RNG

**Status:** Resolved and B300 full-run verified.

**Observed failure:** The `174`-case collection produced `24` valid rows and
`150` failures. The first failing case was EP16 balanced at `8192` tokens:

```text
RuntimeError: Offset increment outside graph capture encountered unexpectedly.
```

The stack ended at `torch.randn` while constructing the next case's BF16
hidden states.

**Root cause:** The `6144`-token contiguous Optimus case completed CUDA graph
capture through `benchmark_with_power`. The following case reused PyTorch's
default CUDA generator for `torch.randn`; that generator retained an invalid
capture-offset state. The failure happened before the next Optimus kernel
invocation, so it is neither OOM nor an unsupported Optimus shape.

**Impact:** The partial `24`-row dataset is rejected. All cases after the first
contiguous capture were unable to construct inputs.

**Root fix:** Construct hidden states with a fresh device-local
`torch.Generator` seeded to `42`, pass it explicitly to `torch.randn`, and stop
mutating the default generator. CUDA graph measurement and all workload cases
remain unchanged.

**Verification:** Two tests first failed because the helper was absent, then
passed. Replacement run `s4p-aic-moe-0815-182707` completed all `174/174`
cases with zero errors and `174/174` exact `silicon` consumer matches.

## ISSUE-028 — Optimus Collector did not measure the pinned Fp8MoEMethod op path

**Status:** Resolved and replacement B300 dataset accepted.

**Observed discrepancy:** The second full run emitted `174` rows, but the
Collector called `OptimusFp8Experts.apply` and always requested CUDA graph.
Only `24` early masked rows used a graph. The contiguous path cannot be
captured because it contains `event.synchronize()` and `m_sum.item()`;
`allow_graph_fail=True` silently measured the remaining `150` rows eagerly.

**Root cause:** The Collector used the legacy expert class as a convenient
stand-alone entry point instead of following pinned `Fp8MoEMethod.apply`, whose
actual branch is:

```text
local tokens < 6144  -> torch.ops.vllm.deepgemm_optimus_moe_masked_fp8
local tokens >= 6144 -> torch.ops.vllm.deepgemm_optimus_moe_fp8
```

The timing harness also treated graph failure as an accepted alternate path,
which violated the task's no-fallback and exact-provider requirements.

**Root fix:**

- invoke the two exact pinned `torch.ops.vllm` operations;
- choose the branch from `local_num_tokens`, matching pinned vLLM;
- require CUDA graph for masked and explicit eager for contiguous;
- set `allow_graph_fail=False` and verify `used_cuda_graph`;
- reserve and validate the pinned provider's process-global workspaces using
  its own `_get_ws_size` and `_get_workspaces` logic;
- persist and remotely validate `kernel_variant`, `execution_mode`, and
  `used_cuda_graph`.

**Impact:** Dataset
`ed30f0642eb8d882161f5265f124e40e14d4cb8be9f4d39ab16f742b5b767f71`
is retained only as rejected evidence and must not enter the canonical B300
database.

**Closure gate:** Replacement smoke must cover both kernel variants and both
execution modes. The full run must emit `174` rows, zero errors, both variants,
strict mode agreement, and `174/174` exact `silicon` consumer hits.

**First replacement smoke:** Run `s4p-aic-moe-0815-181728` reached and timed
the first real masked custom op, then the next case failed the Collector's
workspace capacity assertion. Pinned `SharedResizableBuffer.get` reinterprets
the full `int8` backing allocation as BF16, halving `numel()` while preserving
the same byte capacity. The assertion incorrectly compared byte requirements
from `_get_ws_size` with post-view element counts.

**Follow-up root fix:** Capacity is now compared in bytes using
`numel * element_size`. A regression test first reproduced the false failure
with an 8-byte BF16 buffer and then passed. The failed worker was cleaned up;
no RJob or Replica remains.

**Replacement smoke verification:** Run `s4p-aic-moe-0815-182356` passed on
`NVIDIA B300 SXM6 AC`, SM103. Its three rows covered:

- masked custom op, mandatory CUDA graph, `used_cuda_graph=True`;
- contiguous custom op, explicit eager, `used_cuda_graph=False`;
- EP16 and EP32 plus all three required routing distributions.

All `3/3` unchanged-consumer queries matched the measured latency exactly with
`source="silicon"`. Dataset SHA256 is
`504d3d18e68a038f6facbd30293af212fb83a8d2ce56b11c9a3430a1041b2246`.
The RJob and Replica cleanup checks found no remaining resource.

**Full replacement verification:** Run `s4p-aic-moe-0815-182707` completed
`174/174` cases in `111.20 seconds` with zero collection errors. It produced
`145` masked/CUDA-graph rows and `29` contiguous/eager rows; all `174/174`
consumer queries matched exactly with `source="silicon"`. The accepted CSV
SHA256 is
`cbea7ec572729121df09784a0b06dcff5c92780c510b12fdda28f6982afca3fd`.
The canonical parquet has SHA256
`4bfb1ccdfa8007d3a23576b4e7d50e10dbb11fef6500db29668a59f530cad388`.
No matching RJob or Replica remained after cleanup.

## ISSUE-B300-006 — AIC DeepEP HT full-RDMA launch and NCCL evidence gate

**Status:** Resolved; full-RDMA, NCCL, evidence parsing, and cleanup passed.

**Observed failures:**

1. `s4p-aic-deepep-16-0815-191325` failed with CUDA Error `803` because the
   remote wrapper appended an inherited CUDA library path after the verified
   B300 CUDA 13 compatibility libraries.
2. After repairing that path,
   `s4p-aic-deepep-16-0815-191949` reached B300 capability `(10, 3)` and all
   `16` pinned-vLLM ranks, but failed in the DP `PyNcclCommunicator` warm-up
   all-reduce with `NCCL error: unhandled system error`.

**Root cause:**

- The CUDA Error `803` cause is resolved by using the exact pinned B300
  `LD_LIBRARY_PATH`.
- The remaining launch requests `rdma/mlnx_shared=8` but omits two parts of
  the pinned branch's two-node contract:
  `mellanox.com/mlnx_rdma=1` and host networking.
- Worker-init consequently reports `NCCL_IB_HCA=''` even though
  `NCCL_IB_GID_INDEX=5` is present.
- The pinned vLLM trace shows the first failure at the communicator warm-up,
  before `DeepEPHTAll2AllManager` can construct its Buffer. Bypassing that
  communicator would only hide the missing RDMA contract and would not prove
  DeepEP transport correctness.

**Impact:** EP16/EP32 DeepEP HT smoke, the `116`-row dataset, and exact
consumer validation remain blocked. Attention and Optimus MoE datasets are not
affected.

**Authorized root resolution:**

1. Add `mellanox.com/mlnx_rdma=1` and host networking to both predict-only and
   live AIC DeepEP launches.
2. Add a minimal two-node NCCL preflight that:
   - fails before collection when the platform HCA is empty;
   - performs a real rank-wide NCCL all-reduce;
   - records rank count, HCA presence, and numeric all-reduce result.
3. Run the preflight by itself before another EP16 DeepEP smoke.

**Rejected workarounds:** Do not hardcode HCA names, set
`NCCL_IB_DISABLE=1`, use socket fallback, set `VLLM_DISABLE_PYNCCL`, manually
construct only the EP group, or replace DeepEP HT.

**Evidence:**

- `/data/ycfeng/tmp/step4_aic_deepep_ht_b300_20260815/smoke_ep16_s4p-aic-deepep-16-0815-191325/`
- `/data/ycfeng/tmp/step4_aic_deepep_ht_b300_20260815/smoke_ep16_s4p-aic-deepep-16-0815-191949/`
- `/data/ycfeng/tmp/step4_aic_deepep_ht_b300_20260815/preflight_ep16_s4p-aic-deepep-16-0815-193938/`

All RJobs and Replicas were removed after each run.

**2026-08-15 full-RDMA update:**

- The platform supplied non-empty bonded HCAs, GID index `3`, `bond0`, and the
  StepFun NCCL net plugin.
- All `16` ranks completed a real NCCL all-reduce with observed/expected rank
  sum `136/136` and participant sum `16/16`.
- The remote wrapper still marked the run failed because eight intact
  `STEP4_NCCL_PREFLIGHT_RANK=PASS` markers were interleaved onto one stdout
  line. Its `grep -c` implementation counted lines instead of occurrences.
- Root resolution now required: count exact marker occurrences independently
  of line boundaries, retain the numeric assertions, rerun EP16 preflight, and
  only then advance to DeepEP HT smoke.

**Resolution:**

- Replaced line-based rank marker counting with exact fixed-string occurrence
  counting for both NCCL preflight and DeepEP HT runtime logs.
- RED reproduction: `1 failed, 5 passed`.
- Focused GREEN: `9 passed`.
- Live rerun:
  - global rank PASS count: `16`;
  - per-node rank PASS counts: `8/8`;
  - observed/expected rank sum: `136/136`;
  - observed/expected participant sum: `16/16`;
  - host result: `B300_DEEPEP_HT_HOST=PASS`;
  - final RJob count: `0`;
  - final Replica count: `0`.
- Accepted evidence:
  `/data/ycfeng/tmp/step4_aic_deepep_ht_b300_20260815/preflight_ep16_s4p-aic-deepep-16-0815-194622/`.

## ISSUE-B300-007 — DeepEP Buffer fails with CUDA 999 after NCCL

**Status:** Deferred by owner for the current phase. Accepted rows remain
`0/116`; local Buffer construction passes, while the required cross-node
NVSHMEM/shared-host-SHM launch contract is unavailable.

**Observed failure:**

- EP16 smoke verified pinned source and selected
  `DeepEPHTAll2AllManager`.
- `deep_ep.Buffer.runtime.sync` failed with CUDA error `999`.
- NVSHMEM reported `Unable to access device state` and bad device pointer
  callbacks before any dispatch/combine case ran.

**Resolved fidelity defect:**

- Formal vLLM workers call
  `_harmonize_nvshmem_env_for_deepep_optimus()` from
  `WorkerWrapperBase.init_worker` before device/distributed initialization.
- The Collector directly calls `init_distributed_environment` and
  `initialize_model_parallel`, bypassing that lifecycle.
- Consequently, the Collector did not execute the pinned implementation's
  NVSHMEM GID inheritance, traffic-class, NCCL-isolation, NIC-mapping, and
  team-budget setup.

**Required root fix:**

1. Import and invoke the actual pinned vLLM harmonization helper.
2. Call it before `init_distributed_environment`.
3. Do not duplicate its environment logic in AIC.
4. Verify by RED/GREEN tests and a replacement EP16 smoke.

**Local verification:**

- RED: `1 failed, 16 passed`, with the new contract rejecting the missing
  pinned helper call.
- GREEN: `34 passed`.
- Implementation imports and invokes the actual pinned vLLM helper before
  `init_distributed_environment`; no AIC-side copy of its logic was added.

**Replacement live result:**

- The pinned helper executed on all ranks and set the expected GID inheritance
  and team budget.
- The allocated GPUs were clean: `274114 MiB` free on all eight GPUs in the
  pulled node evidence.
- `deep_ep.Buffer.runtime.sync` still failed with the same CUDA `999` on every
  local rank.
- Therefore, the missing helper was not the final CUDA `999` root cause, and
  the earlier GPU occupancy is ruled out.

**Root-cause discriminator result:**

- Script:
  `tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py`.
- One clean B300 node exposed eight idle GPUs, each with `274114 MiB` free.
- The runtime was the pinned image, vLLM
  `0.19.0.post20.dev26+gc820e5ae1`, DeepEP
  `1.2.1+c5284713.step.torch2.10.0.cu129.py312`, and SM `10.3`.
- All eight ranks completed `deep_ep.Buffer` construction with
  `num_rdma_bytes=0`: construction PASS markers `8/8`; constructor
  `runtime.cu:83` failures `0`.
- All eight ranks then failed explicit `buffer.destroy()` at `runtime.cu:29`
  with CUDA `unknown error`/`999`; the overall torchrun exit code was `1`.
- RJob and Replica cleanup counts were both `0`.

This result separates two failures. The image supports local eight-rank Buffer
construction, so the EP16 constructor failure is narrowed to the cross-node
NVSHMEM/RDMA/shared-host-SHM path. The local destroy failure is a separate
vendor-runtime teardown defect and means the discriminator is not an overall
runtime PASS.

**Blocking root cause:**

- The pinned/later B300 DeepEP bootstrap requires shared host SHM for the
  cross-node NVSHMEM device-state exchange.
- The available `brainctl`/`rlaunch` interface does not expose the documented
  `--share-host-shm=True` contract.
- Basic full-RDMA NCCL is already proven healthy at `16/16` ranks, so another
  EP16 retry with the same launch contract cannot resolve the missing
  cross-node bootstrap.

**Impact:**

- EP16 and EP32 DeepEP dispatch/combine collection cannot produce accepted
  rows.
- The planned `116/116` DeepEP consumer validation cannot run.
- The complete B300 dataset and formal MTP-off prefill/decode simulation
  acceptance remain blocked because fail-fast provider queries cannot
  substitute generic communication data.

**Owner disposition:**

- Do not run another DeepEP measurement in the current phase.
- Preserve `0/116` as an explicit missing dataset, without H800, generic
  communication, or synthetic replacement rows.
- Continue grouped `wo_a`, FP32 router, QKV norm/RoPE, and subsequent AIC
  simulation work.

**Required resolution:**

Provide a documented B300 launch path that exposes shared host SHM with the
existing full-RDMA contract, or provide a platform/vendor fix that makes the
pinned DeepEP runtime initialize and destroy correctly under the available
launcher. Do not disable DeepEP, use socket fallback, synthesize rows, or
reuse H800 data.

**Separate observation:** One GPU on each allocated node was already heavily
occupied before torchrun. This is recorded as a possible platform resource
risk, not treated as the proven cause, and no guessed memory threshold or
fallback has been added.

**Evidence:**

- EP16:
  `/data/ycfeng/tmp/step4_aic_deepep_ht_b300_20260815/smoke_ep16_s4p-aic-deepep-16-0815-194903/`
- Replacement EP16:
  `/data/ycfeng/tmp/step4_aic_deepep_ht_b300_20260815/smoke_ep16_s4p-aic-deepep-16-0815-195827/`
- Local discriminator:
  `/data/ycfeng/tmp/step4_aic_deepep_local_b300_20260815/s4p-deepep-local-0815-2046/`

## ISSUE-030 — Step4 provider dataset is only partially canonicalized

**Status:** Resolved for every owner-approved measurable family on 2026-08-16;
DeepEP remains explicitly owner-deferred rather than canonicalized.

**Observed state:**

- Canonical B300 data now includes:
  - `step4_optimus_moe_perf.parquet`: `174` rows;
  - `step4_context_attention_perf.parquet`: `58` rows;
  - `step4_generation_attention_perf.parquet`: `167` rows;
  - `step4_grouped_gemm_perf.parquet`: `75` rows;
  - `step4_fp32_output_gemm_perf.parquet`: `75` rows;
  - `step4_qkv_norm_rope_perf.parquet`: `150` rows, split `75` Full MFA and
    `75` SWA.
- The QKV family has `150` planned physical keys:
  - Full MFA `vllm_step4pro_k_norm_rope`: `75/75` measured;
  - SWA `vllm_step4pro_qkv_norm_rope`: `75/75` measured.

**Root cause:** Work advanced from one provider vertical slice to the next
without a final family-wide canonicalization gate. Attention was validated
against its evidence-local files, while the three earlier provider slices
stopped after static RED/GREEN validation.

**Impact:** Every currently measurable provider family is canonicalized. The
database still cannot support formal end-to-end latency because DeepEP alone
is absent. Formal prefill/decode E2E remains `BLOCKED` when DeepEP is queried.

**Root resolution:**

1. Resolved ISSUE-031 and measured `75/75` SWA QKV cases through the pinned
   provider plus the authorized annotation compatibility overlay.
2. Merged SWA with the existing `75` Full-MFA rows and verified all `150`
   QKV physical keys through exact AIC consumer queries.
3. Kept missing DeepEP exact keys visible without generic communication data.

**Attention archival evidence:**

- Canonical rows: context `58`, generation `167`.
- Provider split:
  - context: Optimus FA4 `29`, native SWA `29`;
  - generation: Optimus FA4 `71`, native SWA `96`.
- Unique physical keys: context `58/58`, generation `167/167`.
- Canonical exact-consumer queries: `225/225`, all `source="silicon"`.
- Maximum absolute and relative query error: `0.0 ms`, `0.0%`.
- Added exact `65,536` context points:
  - Full FA4: `196.56292724609375 ms`;
  - native SWA: `3.164181391398112 ms`.
- Canonical files:
  - `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_context_attention_perf.parquet`,
    SHA256
    `9995b6a392076cfa481c943b4fceb9b7a0ade41c4c125d12be6b1d3930f46ce8`;
  - `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_generation_attention_perf.parquet`,
    SHA256
    `bde836b884b410b21284645ab45f0a35f82d03cac5b91299b256823067cb14c0`.

**Grouped/router archival evidence:**

- Full collection: grouped `75/75`, router `75/75`, Collector errors `0`.
- Runtime profile confirms `torch.einsum` for the exact pinned grouped
  expression and
  `torch.ops.vllm.optimus_matmul_fp32 -> step3p5_util.apply_optimus_matmul_fp32
  -> torch.ops.OptimusMoe.matmul_fp32` for the router.
- Unique physical keys: grouped `75/75`, router `75/75`.
- Canonical exact-consumer queries: `150/150`, all `source="silicon"`.
- Maximum absolute and relative query error: `0.0 ms`, `0.0%`.
- The only key added to each previous 74-row table is
  `num_tokens=65536`; removed keys: `0`.
- Canonical files:
  - `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_grouped_gemm_perf.parquet`,
    SHA256
    `78fdd68077e3fdaa4c4fa349ab0a72407e0421ae09ecfcfcd4a5ae22d103d760`;
  - `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_fp32_output_gemm_perf.parquet`,
    SHA256
    `3ec22771fa8577dec8b250ca1d6152552a8091b9502c1732461df65f5dc97af9`.

**Full-MFA QKV archival evidence:**

- Smoke: `1/1`, latency `0.028218666712443035 ms`.
- Full collection: `75/75`, token range `1–65,536`, latency
  `0.029557332396507263–6.715685526529948 ms`.
- Unique physical keys: `75/75`; duplicate keys: `0`.
- Canonical exact-consumer queries: `75/75`, all `source="silicon"`.
- Maximum absolute and relative query error: `0.0 ms`, `0.0%`.
- Canonical file:
  `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_qkv_norm_rope_perf.parquet`,
  SHA256
  `7091c0da8d445025df9e98547c2bc50c7f99276bee014212c1f4fd651b6789c1`.

**Complete QKV archival evidence:**

- SWA full collection: `75/75`, token range `1–65,536`, latency
  `0.006319999694824219–1.3165173530578613 ms`.
- Combined rows and unique physical keys: `150/150`; duplicate keys `0`.
- Canonical exact-consumer queries: `150/150`, all `source="silicon"`.
- Maximum absolute and relative query error: `0.0 ms`, `0.0`.
- Canonical file:
  `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_qkv_norm_rope_perf.parquet`,
  SHA256
  `cc14553d8e0d619ac5fc21b15a9dd78633e168114da270e55edec2c6e7afd579`.

## ISSUE-031 — Image-native SWA QKV CuTeDSL has an invalid `reload_from` JIT contract

**Status:** Resolved on 2026-08-16. The bounded overlay, finite-value smoke,
full collection, canonicalization, and exact consumer validation passed.

**Observed failure:**

- Initial replacement smoke: `s4p-aic-core-0815-221843`.
- Diagnostic replacement smoke: `s4p-aic-core-0815-225633`.
- Source identity passed for `2103/2103` pinned vLLM files.
- Runtime identity:
  - B300 capability `10.3`;
  - torch `2.10.0+cu129`;
  - vLLM `0.19.0.post20.dev26+gc820e5ae1`;
  - Step Optimus `3.23.24`;
  - `optimus-jit 0.1.10.post8+gitcfde41ba`;
  - `nvidia-cutlass-dsl 4.4.2`.
- The initial combined smoke could not accept Full-MFA output because all
  provider branches shared one gate.
- The first failing provider call was the exact pinned SWA path:
  `step3p5_util.fused_qknorm_rope_forward_impl` to
  `optimus_cutedsl.qknorm_rope.fused_qknorm_rope_forward_impl`.
- CuTeDSL compilation rejected kernel argument 15:

  ```text
  DSLRuntimeError: failed to generate argument #15 (reload_from)
  Call-site argument value: smem
  Call-site argument type: <class 'str'>
  Consider annotating the argument with reload_from : Constexpr
  ```

**Confirmed root cause:**

1. The pinned SWA shape has QKV width
   `(128 + 2 * 8) * 128 = 18432`.
2. Image-native `FusedQKNormRope` sets `reload_from="smem"` when
   `N > 16384`, so this production shape enters the affected branch.
3. `optimus_cutedsl/qknorm_rope.py` begins with
   `from __future__ import annotations`. Consequently,
   `FusedQKNormRope.kernel.__annotations__["reload_from"]` is the string
   `"cutlass.Constexpr"` rather than the `cutlass.Constexpr` object.
4. Cutlass DSL 4.4.2 calls `inspect.getfullargspec()` and consumes its
   annotations directly. It does not resolve postponed string annotations.
   It therefore treats `"smem"` as a dynamic argument and fails before kernel
   generation.

This is an image-native Optimus JIT/Cutlass annotation contract defect. It is
not a Collector shape, vLLM-config-context, operation-key, or mathematical
QKV implementation defect.

**Impact:**

- The new `qkv_full` slice independently passed the actual pinned Full-MFA
  path and produced `75/75` accepted rows.
- The combined provider-core smoke still cannot pass while SWA is included.
- The remaining `75` SWA rows cannot be measured or merged into the complete
  `150`-row QKV table without resolving this defect.

**Authorized root resolution:**

- Require the exact QKNorm source SHA256
  `5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783`.
- In the ephemeral B300 runtime only, resolve the intended
  `reload_from`/`delay_w_load` annotations to the installed
  `cutlass.Constexpr` object before any QKNorm compilation.
- Do not change the pinned vLLM source, QKV math, shape, dtype, provider,
  kernel body, or persisted key.
- Pass the RED/GREEN annotation contract, rerun an SWA-only representative smoke,
  then collect exactly the remaining `75` SWA cases.

**TDD continuation gate:**

- Added two focused contracts for an independent `qkv_swa` slice and the
  bounded annotation repair.
- Verified expected RED: `2 failed`, `10 deselected`, `0.19s`, pytest exit
  `1`.
- The contracts require the exact source SHA256, mutation of only
  `reload_from` and `delay_w_load` to `cutlass.Constexpr`, no QKNorm source
  rewrite, and exactly `1/75` smoke/full SWA rows.
- Runtime implementation is present after explicit approval and imports a
  fail-fast helper in every SWA collection process; it does not depend on
  non-fatal `sitecustomize` exception behavior.
- Fresh GREEN evidence on 2026-08-16: focused `2/2` and combined local
  regression `43/43`.
- RED evidence:
  `/data/ycfeng/tmp/step4_swa_qkv_overlay_contract_red_20260815.log`, SHA256
  `680272e184fe5a0e913ea31e272f0d01e8e4fc1a45f09da6b76d44d64f265665`.

**Safety/result:**

- Final exact RJob count: `0`.
- Final label-selected Replica count: `0`.
- No fallback kernel or synthetic timing was used.
- Full-MFA collection did not use the proposed SWA annotation overlay.
- The first approved SWA smoke compiled and ran the exact provider at
  `0.00590933362642924 ms`. Review then found a validation gap: the Collector
  checked output shape and BF16 dtype but did not reject NaN/Inf. This was a
  probe-validation defect, not a kernel failure.
- A focused RED/GREEN change now checks every QKV output for finite values.
- The updated smoke passed at `0.0052480002244313555 ms`; full collection then
  produced `75/75` accepted SWA rows.
- The combined canonical QKV table has `150/150` unique keys and `150/150`
  exact silicon consumer matches with `0.0 ms` maximum error.

**Evidence:**

- Initial failure:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-core-0815-221843/`.
- Diagnostic failure:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-core-0815-225633/`.
- Successful Full-MFA smoke:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-qkvfull-0815-050738/`.
- Successful Full-MFA full collection:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/full_s4p-aic-qkvfull-0815-051119/`.
- Accepted finite-check SWA smoke:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260816/smoke_s4p-aic-qkvswa-0816-115127/`.
- Successful SWA full collection:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260816/full_s4p-aic-qkvswa-full-0816-115343/`.
- Extracted QKNorm source:
  `/data/ycfeng/tmp/optimus_cutedsl__qknorm_rope.py`, SHA256
  `5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783`.
- Extracted Cutlass DSL source:
  `/data/ycfeng/tmp/cutlass_dsl_4_4_2_dsl.py`, SHA256
  `42889709db34cb25ec64712683a6725cd7f1c6b35af69236edf2d42ad86ce805`.

## ISSUE-032 — Formal 65K prefill exposes an unmeasured native SWA Attention point

**Status:** Resolved on 2026-08-15.

**Observed failure:**

- Full EP16+EP32 coverage validator queried `36,420` operation records.
- Exact silicon: `16,048`; analytic: `15,160`; missing: `5,212`;
  unexpected errors: `0`.
- In addition to the already known QKV and DeepEP files, the real native SWA
  consumer reported missing context attention at
  `batch=1, query_tokens=65,536, total_context_tokens=65,536`.
- The initial uniform-local probe also reported Optimus MoE at
  `num_tokens=131,072`; subsequent contract analysis shows that request is
  produced by applying a global `65,536` budget as if it were local on every
  EP rank. It is a scheduler-mapping issue, not yet an accepted data gap.

**Root cause:**

The prior dataset checks proved that every collected row can be queried
exactly. They did not prove that the native SWA active-rank workload at a
`65,536` chunk is bracketed. Separately, the current static API assumes a
uniform local batch on every attention-DP rank and therefore cannot directly
map the requirements' global scheduler budget to both attention and MoE query
shapes.

**Impact:**

- Formal `max_num_batched_tokens=65,536` prefill cannot be reported as complete
  SILICON simulation until the native SWA point is measured.
- This is separate from the deferred DeepEP gap and the blocked QKV provider.
- No fallback, extrapolated label, or H800 row may be used to close it.

**Root resolution:**

1. Measured native SWA context attention at
   `(batch=1, query=65,536, total_context=65,536)` through the pinned provider:
   `3.164181391398112 ms`.
2. Also measured the matching Full FA4 point:
   `196.56292724609375 ms`.
3. Canonicalized context Attention to `52` rows and passed `52/52` exact
   silicon queries.
4. Corrected the coverage validator so attention/dense use the busiest
   attention-DP rank while Optimus MoE receives actual global scheduled
   tokens.
5. Re-ran EP16+EP32 coverage. The false Optimus `131,072` gap disappeared.

**Verification result:**

- records: `36,420`;
- exact silicon: `16,660`;
- analytic/non-silicon: `15,160`;
- missing: `4,600`;
- unexpected errors: `0`;
- remaining physical contracts: `6`, comprising QKV `2` and DeepEP `4`.

**Evidence:**

- `/data/ycfeng/tmp/step4_aic_attention_b300_20260815/smoke_s4p-aic-swa65k-0815-024430/`;
- `/data/ycfeng/tmp/step4_aic_attention_b300_20260815/smoke_s4p-aic-fa465k-0815-030553/`;
- `/data/ycfeng/tmp/step4_attention_52_canonical_consumer_validation_20260815.json`;
- `/data/ycfeng/tmp/step4_aic_silicon_coverage_20260815/full_ep16_ep32_scheduler_corrected_post_attention.json`.

## ISSUE-034 — Formal MTP-off latency is blocked by exact QKV and DeepEP gaps

**Status:** Open only for owner-deferred DeepEP after complete QKV
canonicalization on 2026-08-16.

**Observed facts:**

- Corrected coverage queries `36,420` operation records:
  `18,220` exact silicon, `15,160` analytic/non-silicon, `3,040` missing,
  and `0` unexpected errors.
- The only missing physical contracts are:
  - DeepEP HT: dispatch/combine at EP16 and EP32, four identities.
- The full requirements driver executed all `72` prefill and `84` decode
  records. Every result is `BLOCKED`; formal latency and `B_max` remain null.
- DeepEP is frozen at `0/116` and must not be retried in this phase.

**Root cause:**

- The available launcher cannot satisfy the DeepEP/NVSHMEM launch contract;
  the owner directed the task to record and skip further DeepEP measurement.

**Impact:**

- The simulation driver, scheduling, HBM accounting, workload matrix, and
  partial known latency are validated.
- A formal all-silicon prefill latency, decode TPOT result, or `B_max` cannot
  be claimed.

**Required resolution:**

1. In a future phase, provide a working DeepEP launcher/runtime contract and
   collect EP16/EP32 dispatch/combine rows.
2. Re-run the same smoke and full driver without changing workload semantics.

**Evidence:**

- `aic_silicon_coverage_2026-08-16.json`;
- `mtp_off_requirements_smoke_2026-08-16.json`;
- `mtp_off_requirements_full_2026-08-16.json`.

## ISSUE-035 — Aggregate validation pipeline masked Ruff findings

**Status:** Resolved on 2026-08-15.

**Observed facts:**

- The original aggregate lint log contained four import-order findings and
  six files requiring formatting.
- Its wrapper nevertheless returned exit code `0`.
- After applying the generated formatting diff, a fail-fast Ruff rerun found
  one remaining import-block blank-line error.

**Root cause:**

The aggregate command did not use `set -euo pipefail`. Its brace group returned
the status of the last successful command, and the pipeline returned the
status of `tee`, so earlier Ruff failures were not propagated.

**Impact:**

The data, model graph, Collector behavior, and simulation outputs were not
affected. The checkpoint could not be called lint-clean until each command was
rerun independently.

**Resolution:**

1. Inspected Ruff's proposed import and format diffs.
2. Applied formatting-only changes to the six affected contract-test files.
3. Removed the one remaining extra blank line found by the first fail-fast
   rerun.
4. Re-ran tests, Ruff, shell syntax, and whitespace checks independently with
   `set -euo pipefail`.

**Verification result:**

- focused tests: `134/134` passed in `5.73s`;
- formatted contract tests: `31/31` passed in `1.97s`;
- Ruff check: `43/43` changed task Python files passed;
- Ruff format: `43/43` files already formatted;
- shell syntax: `14/14` passed;
- `git diff --check`: exit `0`.

No DeepEP, Docker, GPU, RJob, Replica, or SWA QKV process was started during
this repair.

## ISSUE-036 — Post-finite-check SWA smoke evidence stream was truncated

**Status:** Resolved by one bounded transport retry on 2026-08-16; kernel,
finite-value gate, evidence transfer, and cleanup passed.

**Observed failure:**

- RJob `s4p-aic-qkvswa-0816-114912` reconstructed and verified all
  `2103/2103` pinned-vLLM files.
- The source-hash-bounded annotation overlay passed in both the identity and
  measurement processes.
- The updated Collector, including the finite-value probe, reached
  `B300_STEP4_PROVIDER_CORE_COLLECTION=PASS` and validated one SWA row.
- The later host evidence transfer produced a `393,216`-byte tar stream that
  ended before the archive trailer. Local extraction failed with:

  ```text
  tar: Unexpected EOF in archive
  ```

**Root-cause classification:** The failure occurred after provider execution
and row validation, in the cross-zone `brainctl exec` evidence stream. There
was no CUDA, QKNorm, annotation, source-identity, or finite-value failure.
One earlier smoke using the same evidence path transferred a complete
`419,840`-byte archive, so this instance is classified as a transient
transport interruption rather than a Collector/kernel defect.

**Disposition:**

- Reject the incomplete archive and accept none of its dataset files.
- Confirm exact cleanup before retrying.
- Run one new isolated smoke. If the evidence stream truncates again, stop
  repeating the direct-stream method and repair the transfer protocol before
  any full collection.

**Cleanup evidence:** Exact RJob count `0`; exact label-selected Replica count
`0`.

**Resolution evidence:**

- Retry RJob: `s4p-aic-qkvswa-0816-115127`.
- Host and remote result markers: PASS.
- Accepted row latency: `0.0052480002244313555 ms`.
- Complete evidence tar: `419,840` bytes, SHA256
  `b5549c456831c59d24e674d10bfeb997b032ec72b0c4d389e2c42604361fe663`.
- Updated payload includes `_validate_qkv_norm_rope_probe`; the provider call
  passed shape, BF16 dtype, and finite-value checks.
- Exact RJob and Replica counts after cleanup: `0` and `0`.

## ISSUE-037 — Exact DeepEP data is unavailable for the required simulation

**Status:** Resolved for the owner-approved proxy simulation; exact
measurement remains deferred.

**Observed facts:**

- Exact B300 DeepEP collection is deferred at `0/116`.
- The real `vllm_deepep_high_throughput` consumer correctly reports four
  missing physical contracts: dispatch/combine for EP16 and EP32.
- All `72` prefill and `84` decode results therefore remain `BLOCKED` when
  the exact path is used.
- The existing B300 NCCL table has measured `alltoall` curves for `half` and
  `int8` at 2, 4, and 8 GPUs.

**Root cause:**

The available B300 launcher/runtime cannot establish the DeepEP/NVSHMEM
contract. The owner explicitly stopped further DeepEP retries and later
authorized a temporary NCCL `alltoall` simulation proxy.

**Impact:**

- Exact DeepEP latency remains unavailable and must not be claimed.
- MTP-off latency, TPOT, and `B_max` can be produced only as explicitly
  labeled proxy simulation results.
- EP16/EP32 proxy queries use AIC's existing rank-count correction from the
  measured 8-GPU NCCL curve and are not direct EP16/EP32 measurements.

**Approved resolution:**

1. Keep the exact DeepEP query as the default fail-fast path.
2. Add an explicit `b300_nccl_alltoall` task-level simulation option.
3. Map dispatch to the measured one-byte `int8` curve as the FP8 transport
   equivalent and combine to the measured `half` curve for BF16.
4. Use
   `ceil(tokens_per_dp_rank * hidden_size * topk / ep_size)` message
   elements and apply the original operation scale factor.
5. Mark every affected record and aggregate as `PROXY`.
6. Do not create or populate `step4_deepep_ht_perf.parquet`.
7. Replace all proxy results after real DeepEP EP16/EP32 measurement becomes
   available.

**Verification required:**

- TDD evidence for dtype, volume, topology, scaling, explicit opt-in, and
  summary separation.
- Proxy coverage with zero missing/error records.
- Full `72` prefill and `84` decode proxy matrix.
- Filesystem evidence that the DeepEP parquet remains absent.

**Resolution result:**

- Proxy contract tests and the final focused suite passed: `343/343` in
  `8.09s`.
- Full Collector regression passed: `401/401`.
- Proxy coverage completed `36,420` records:
  `18,220` exact silicon, `15,160` analytic/non-silicon, `3,040` proxy,
  `0` missing, and `0` errors.
- The proxy matrix completed `72/72` prefill and `84/84` decode records with
  `result_fidelity=PROXY`.
- Prefill latency is
  `129.22378441076575–782420.0031436655 ms`; decode batch-1 TPOT is
  `56.955545921130614–268.6990437660492 ms`.
- Every decode combination has `B_max=0`, `aggregate_B_max=0`, and
  `first_failed_batch=1`.
- `step4_deepep_ht_perf.parquet` remains absent. The proxy did not alter the
  exact DeepEP consumer or create substitute measured data.
- Real DeepEP EP16/EP32 collection remains future work and must replace these
  proxy results when the runtime environment is restored.

## ISSUE-040 — Manual-review output omitted execution-status fields

**Status:** Resolved on 2026-08-16.

**Observed failure:**

- The Phase 10 JSON contained operation-level missing/error counts, but the
  flat manual-review CSV did not expose `backend_fallback`, `retry_count`,
  `error_record_count`, `missing_record_count`, or `exception_log`.
- Decode candidates encoded batch indirectly but did not use the requirements'
  explicit `active_sequences_per_replica` and
  `batched_tokens_per_replica` names.
- Focused regression reproduced the gap as:
  `17 passed, 1 failed`, `KeyError: backend_fallback`.

**Root cause:**

Phase 10 prioritized exact/proxy query separation and matrix completion. The
flat review-row builder selected latency, memory, component, and fidelity
fields but did not thread the already available query-status information into
the normalized record. Decode batch semantics were likewise present only
through `global_batch_size_per_replica`.

**Impact:**

The latency, throughput, HBM, operation data, and DeepEP proxy mapping were
unchanged and correct. However, the output did not satisfy the requirements'
manual audit contract for fallback, retry, exception, and engine-step status.

**Root resolution:**

1. Aggregate missing/error counts and structured exception entries from every
   queried prefill chunk and decode candidate.
2. Set `backend_fallback=false` and `retry_count=0` explicitly. The
   owner-selected DeepEP proxy remains an explicit approximation, not a
   runtime fallback.
3. Record decode active sequences and batched tokens directly.
4. Add MTP1-deferred and runtime-only null reasons to the flat CSV.
5. Rerun the full matrix three times and regenerate the combined JSON and
   `156`-row review CSV.

**Verification result:**

- Focused tests: `18/18` passed.
- Repeated matrix: `468` executions; `156/156` identical; maximum spread
  `0.0`.
- Fallback, retry, error, missing, and exception counts:
  `0`, `0`, `0`, `0`, and `0`.
- Artifact schema audit: PASS.

## ISSUE-041 — Manual-review CSV used CRLF line endings

**Status:** Resolved on 2026-08-16.

**Observed failure:**

- `git diff --cached --check` reported trailing whitespace on all `157` CSV
  lines.
- Byte inspection found LF `157`, CRLF `157`, and bare CR `0`.
- A focused regression reproduced the behavior as `1` failed:
  `assert b"\r\n" not in raw`.

**Root cause:**

`csv.DictWriter` inherited the `excel` dialect's default
`lineterminator="\r\n"`. Opening the output with `newline=""` correctly
prevented Python newline translation but did not override the CSV dialect.

**Impact:**

Simulation values and the `156 × 87` review schema were unchanged, but the
artifact failed the repository whitespace gate and was harder to review in
line-oriented Git tooling.

**Root resolution:**

1. Added a focused test requiring no CRLF and exactly one LF per header/data
   row.
2. Set `lineterminator="\n"` on the existing `csv.DictWriter`.
3. Regenerated the CSV from the same three full-run JSON inputs.

**Verification result:**

- RED: `1` failed in `0.22s`.
- GREEN: `1/1` passed in `0.05s`.
- Regenerated CSV: `156` data rows, `87` columns, LF `157`, CRLF `0`.
- Repeat JSON SHA256 unchanged:
  `9d5c296c7c4e95859982cbb986cbda21ef65f9a56d286ebe37cc5a780f7208a1`.
- CSV SHA256:
  `4c1afd34a37877a6cb59cf79c37a326b537261ed9640ee4d657ce527ff49ac62`.
