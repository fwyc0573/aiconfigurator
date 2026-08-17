## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-17 | Recorded the completed Phase 14 publication: implementation commit `bd91ce3a` pushed with local/tracking/remote SHA equality and ahead/behind `0/0`; live acceptance remains quota-blocked. |
| 2026-08-17 | Added Phase 14 follow-up hardening: live evidence pull before one RJob delete, strict cleanup/query validation, same-argument predict-only, quota evidence, timeout margin, and completion-marker contracts. |
| 2026-08-17 | Completed the Phase 13 review and report normalization; final hash/staged-diff verification, commit, push, and remote SHA confirmation remain. |
| 2026-08-17 | Added the owner-requested final code/docs review and checkpoint commit/push phase without converting the quota-blocked two-node run to PASS. |
| 2026-08-17 | Corrected the final live admission gate after proving `predict-only` ignores total replica count and direct quota reads are RBAC-blocked. |
| 2026-08-17 | Completed the local two-node coordinator and global-marker-scope fixes; the final live rerun is blocked by B300 quota. |
| 2026-08-17 | Added Phase 12 to replace active DeepEP runtime settings with explicit `allgather_reducescatter` and preserve the AIC proxy boundary. |
| 2026-08-17 | Recorded passing B300 predict-only and single-B300 smoke evidence, rank-0 AgRs execution, and the blocking two-node lifecycle defect. |
| 2026-08-17 | Added the owner-authorized two-node lifecycle root-fix and bounded rerun gate. |
| 2026-08-13 | Initialized the execution plan and the first requirements-confirmation gate. |
| 2026-08-13 | Closed the source/runtime identity gate using user choice A. |
| 2026-08-13 | Confirmed the referenced manifest is unavailable and retained the explicit reconstruction decision gate. |
| 2026-08-13 | Closed the manifest reconstruction gate using user choice A. |
| 2026-08-13 | Added the operation-boundary architecture gate from the completed fidelity audit. |
| 2026-08-13 | Added a runtime-trace provenance gate for Full-MFA profiling cases. |
| 2026-08-13 | Closed the operation-boundary gate with option A and made branch/checkpoint safety the active gate. |
| 2026-08-13 | Closed the branch/checkpoint decision with option A; baseline checkpoint creation is in progress. |
| 2026-08-13 | Recorded the failed focused baseline run and paused implementation for root-cause analysis. |
| 2026-08-13 | Closed the baseline repair gate with 899/899 passing tests; Latest implementation planning resumed. |
| 2026-08-14 | Completed the pinned-vLLM MTP1 boundary audit and opened the owner decision gate. |
| 2026-08-14 | Deferred MTP1 structure tests/simulation; activated MTP-off Latest execution and parallel B300 smoke. |
| 2026-08-14 | Handed pinned-vLLM smoke/runtime trace to an external session; current execution now covers AIC-side work only. |
| 2026-08-15 | Completed the MTP-off Latest model/op graph gate and advanced the active work to Collector implementation. |
| 2026-08-15 | Split Collector work into provider-data vertical slices, beginning with runtime profile and grouped `wo_a`. |
| 2026-08-15 | Completed grouped `wo_a` and FP32 router slices; advanced the active slice to QKV norm/RoPE. |
| 2026-08-15 | Completed QKV norm/RoPE and advanced the active slice to provider context/generation attention. |
| 2026-08-15 | Completed provider Attention static/consumer support and advanced the active gate to B300 runtime validation. |
| 2026-08-15 | Resumed the external 6E lane, completed the two-node environment-persistence RED→GREEN fix, and advanced to live DP16/EP16 validation. |
| 2026-08-15 | Passed the 16-rank full-RDMA NCCL preflight and narrowed the external 6E blocker to shared-host-SHM/NVSHMEM DeepEP Buffer bootstrap. |
| 2026-08-15 | Passed the replacement Optimus MoE B300 smoke on both pinned custom-op branches; advanced to the 174-case full collection. |
| 2026-08-15 | Completed the replacement Optimus MoE 174-case B300 dataset and exact consumer validation; advanced to DeepEP HT. |
| 2026-08-15 | Completed the DeepEP HT runtime Buffer-call RED→GREEN slice; advanced to distributed runner validation. |
| 2026-08-15 | Completed the DeepEP HT distributed-driver and EP16/EP32 wrapper static gate; advanced to EP16 live smoke. |
| 2026-08-15 | Reached the AIC DeepEP full-RDMA gate after repairing CUDA Error 803; made the Q16 NCCL preflight the next required action. |
| 2026-08-15 | Completed the EP16 full-RDMA NCCL preflight with exact 16-rank numeric validation; advanced to EP16 DeepEP HT smoke. |
| 2026-08-15 | Completed the local DeepEP discriminator and blocked further collection on the unavailable cross-node shared-host-SHM/NVSHMEM contract. |
| 2026-08-15 | Resumed with a canonical-dataset and consumer-key audit before attempting MTP-off prefill/decode E2E. |
| 2026-08-15 | Used the supplied RJob guidance to test the legacy launcher; explicit NVSHMEM passed but DeepEP Buffer exposed a PE-count/runtime integration mismatch. |
| 2026-08-15 | Deferred further DeepEP measurement by owner direction and kept remaining provider-core collection active. |
| 2026-08-15 | Confirmed the next-op SWA QKV blocker as an image-native postponed-annotation/Cutlass contract defect; runtime-overlay approval is the active gate. |
| 2026-08-15 | Canonicalized the complete 199-row B300 Attention dataset and passed 199/199 exact canonical consumer queries. |
| 2026-08-15 | Completed and canonicalized the 74+74 grouped `wo_a`/FP32 router B300 slice with 148/148 exact canonical consumer matches. |
| 2026-08-15 | Reconciled the phase table after continuation: only QKV remains measurable in provider-core, while DeepEP is explicitly deferred. |
| 2026-08-15 | Reconfirmed DeepEP `0/116` as deferred and moved the active gate to missing 65K provider coverage. |
| 2026-08-15 | Reopened the exact DeepEP/NVSHMEM launch lane under owner authorization to apply and test direct-child commit `9bfd9a610e`. |
| 2026-08-15 | Closed the grouped/router 65K B300 gap with 75+75 measured rows and restored the checkout to the authoritative pinned commit. |
| 2026-08-15 | Closed the final DeepEP retry as deferred at 0/116 and advanced AIC execution to the exact SILICON coverage validator and scheduler-aware MTP-off simulation driver. |
| 2026-08-15 | Closed the 65K Attention gap, corrected scheduler-aware coverage to six real missing contracts, and made the MTP-off requirements driver the active phase. |
| 2026-08-15 | Canonicalized the corrected 225-row Attention dataset and completed the smoke/full MTP-off requirements matrix with explicit QKV/DeepEP blockers. |
| 2026-08-15 | Measured and canonicalized 75/75 Full-MFA QKV rows, reran coverage/simulation, and reduced the remaining gap to SWA QKV plus DeepEP. |
| 2026-08-15 | Completed Phase 8 for the current blocked checkpoint after fail-fast post-format verification; SWA QKV, DeepEP, and MTP1 remain explicitly open or deferred. |
| 2026-08-15 | Opened the SWA QKV continuation gate with two expected RED contracts; runtime implementation remains pending explicit overlay approval. |
| 2026-08-15 | Marked Phase 9 blocked after three consecutive continuation turns reached the same unapproved runtime-overlay gate. |
| 2026-08-16 | Received explicit SWA annotation-overlay approval and resumed Phase 9 at the GREEN implementation gate. |
| 2026-08-16 | Completed SWA QKV measurement, canonicalized all 150 QKV rows, and refreshed coverage plus the full MTP-off matrix. |
| 2026-08-16 | Completed Phase 8 after independent re-review, final fail-fast verification, report synchronization, and archive refresh. |
| 2026-08-16 | Opened Phase 10 after owner approval of an explicit B300 NCCL alltoall proxy for DeepEP simulation. |
| 2026-08-16 | Added the scheduler-derived eight-row Attention closure gate discovered by the full proxy matrix. |
| 2026-08-16 | Completed Phase 10 after Attention endpoint closure, full proxy simulation, artifact validation, and final local audit. |
| 2026-08-16 | Opened Phase 11 for a requirement-by-requirement simulation review, three-repeat audit, normalized test record, and task-scoped publication. |
| 2026-08-16 | Completed the Phase 11 requirement audit and final local publication gate; Git commit/push evidence is recorded by the branch and final handoff. |
| 2026-08-16 | Refreshed the final publication evidence with the user-requested requirement audit and pre-commit regression. |

# Plan: Step4-Pro-Latest B-Card Ops, Collection, and Simulation

## Objective

Deliver an auditable, latest-implementation-based `step4-pro-latest` operation definition, a fresh B-card performance dataset, correctness evidence, and the required prefill/decode simulation results without silently substituting historical H800 data or unspecified runtime inputs.

## Phase Status

| Phase | Status | Exit Criteria |
|---|---|---|
| 1. Requirements and historical review | Completed | Requirements, historical tasks, repository state, and environment rules are recorded. |
| 2. Latest vLLM source/image identity confirmation | Completed | Pinned checkout, commit, source files, and B300 path are recorded; later image is reference-only. |
| 3. Grill-me clarification gate | Completed | Source/runtime, model identity, manifest reconstruction, operation boundaries, profiling provenance, and branch/checkpoint handling are explicitly resolved. |
| 4. Latest op inventory and AIC design | Completed | MTP-off prefill/decode inventory and the pinned-vLLM implementation matrix are finalized; MTP1 structure work is explicitly deferred. |
| 5. AIC model/op and Collector implementation | Completed for the MTP-off graph | The MTP-off model graph, provider identities, case population, loaders, exact-key consumers, and isolated Full-MFA collection slice pass focused tests. |
| 6. AIC B-card operation collection | Completed for every owner-approved measurable family; DeepEP deferred | Attention `235/235`, Optimus MoE `174/174`, grouped `75/75`, FP32 router `75/75`, and QKV `150/150` are measured and canonicalized. DeepEP remains `0/116` and must not be retried in this phase. |
| 6E. External pinned-vLLM smoke/runtime trace | Blocked after applying exact later launch-script fix | Checkout and static validation of `9bfd9a610e` passed, but both installed brainctl launch paths reject `--share-host-shm`; the standalone `rjob` client required by the commit is absent. |
| 7. Correctness and simulation | Completed for the approved MTP-off scope | The exact path remains reproducibly blocked only by missing DeepEP data. The explicitly selected B300 NCCL proxy completed all `72` prefill and `84` decode rows with `result_fidelity=PROXY`; MTP1 remains deferred. |
| 8. Final review and archive | Completed for the approved MTP-off scope | Reports, hashes, issue records, independent review, and final fail-fast verification are synchronized; DeepEP and MTP1 remain explicit deferred items. |
| 9. SWA QKV completion and simulation refresh | Completed | The bounded overlay, correctness smoke, `75/75` SWA collection, `150/150` QKV canonicalization, exact consumer validation, and unchanged simulation refresh all passed. |
| 10. Temporary DeepEP proxy and simulation completion | Completed | The explicit opt-in maps DeepEP dispatch/combine to B300 NCCL `alltoall`, labels all affected results `PROXY`, preserves the exact path by default, and completed the full prefill/decode matrix without creating fake DeepEP data. |
| 11. Requirement-completeness audit and publication | Completed | The simulator exposes KV allocation, required component breakdown, normalized unavailable metrics, and three-repeat evidence; the manual-review packet is synchronized; fresh verification passes; only task-owned files enter the single publication commit. |
| 12. Non-DeepEP vLLM runtime configuration | In progress: locally hardened; final two-node live rerun blocked by B300 quota visibility and availability | Active wrappers pin `allgather_reducescatter`, disable sequence parallelism, reject DeepEP selection, and pass `26/26` focused contracts. Final live acceptance still requires direct confirmation of 16 available B300 GPUs, both nodes validated while the RJob remains live, and strict single-delete cleanup. |
| 13. Final code/docs review and checkpoint publication | Completed | Commit `0b8d651c` was pushed to `origin/task/step4-pro-latest-b300`; this publication did not convert the quota-blocked two-node runtime to PASS. |
| 14. Runtime lifecycle and evidence hardening | Completed and published; final live two-node acceptance remains `BLOCKED_BY_QUOTA` under Phase 12 | The obsolete shutdown-arm protocol is removed. Both replicas remain live after validation, the host pulls and validates both evidence sets, and one exact RJob delete performs teardown. Same-argument predict-only, explicit quota evidence, strict query-aware cleanup, timeout margin, and CUDA-complete forward evidence pass locally. Implementation commit `bd91ce3a` is on `origin/task/step4-pro-latest-b300`. |

The bounded SWA annotation compatibility overlay was explicitly authorized and
completed on 2026-08-16. It passed the source-hash, annotation-scope,
shape/dtype/finite-value, B300 smoke, and `75/75` full-collection gates. The
canonical QKV table now contains `150/150` unique keys and passes `150/150`
exact silicon queries with `0.0 ms` maximum error.

The exact `9bfd9a610e` DeepEP checkout and two-script static gate passed, but
the installed launcher cannot submit its shared-host-SHM contract. Per owner
direction, DeepEP measurement is deferred at `0/116` and must not be retried
in this phase. The checkout remains restored to authoritative commit
`607d1641ee`. Corrected scheduler-aware coverage now has only four missing
physical contracts: DeepEP dispatch/combine for EP16 and EP32. The smoke and
full exact-path MTP-off matrices completed all `72/72` prefill and `84/84`
decode rows with formal latency and `B_max` null only because DeepEP is
missing. The owner-approved proxy path subsequently completed the same matrix
with explicit `PROXY` fidelity and no missing/error records.

## Current Execution Gate

The base model/source and shape contract remains:

```text
source: /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro
commit: 607d1641ee3fec43653fca510d717725828890c2
shape: requirements document's 78-layer synthetic Step4-Pro shape
runtime/image: pinned contract; later branch/image is reference-only
```

For the DeepEP lane only, the owner explicitly authorized direct-child commit
`9bfd9a610ea4f2890010702ee7a207cf25edf8de`. It changes no Python/CUDA model
code; it adds explicit NVSHMEM bootstrap, NIC-to-PE mapping, and the
`--share-host-shm=True` launch contract in two shell scripts. Record this as a
bounded launch-script deviation rather than silently redefining the whole
model source.

The existing `Step4-Pro-V4` entry is materially different and must remain
unchanged. Continue with source-to-AIC mapping before production edits.

The user approved the ISSUE-007 resolution: branch
`task/step4-pro-latest-b300` and a relevant-file baseline checkpoint. No
generated output, H800 dataset, cache, or separate vLLM checkout may enter that
checkpoint.

The reconstructed manifest will receive a new SHA256 and must carry explicit
source fields for the requirements document, pinned vLLM commit, and
reconstruction status.

## Verification Strategy

- Keep latest runtime/source identity, AIC operation identity, Collector invocation identity, and persisted perf-key identity aligned.
- Before generating accepted Full-MFA rows, profile the actual pinned
  `Step4Pro` forward path and reconcile observed modules/kernels, shapes,
  dtypes, fusion, and backend selection with the logical AIC graph.
- Provider-sensitive cases must execute the pinned vLLM implementation:
  Optimus FA4 for hd512 Attention and the actual grouped/einsum `wo_a` path.
  A generic FlashAttention row, dense-GEMM substitution, or multiplied timing
  is not acceptable.
- Use fresh B-card measurements only; label historical H800/v3/v4 artifacts as reference evidence.
- Run focused static/consumer tests before any formal simulation.
- Record exact commands, environment, hardware facts, numeric metrics, output paths, and hashes.
- Stop on unresolved contradictions, missing runtime identity, failed required tests, or collection-key mismatches.
- The baseline repair gate is closed:

  ```text
  adjusted focused baseline: 899 passed, 0 failed, 51.84s
  log: /data/ycfeng/tmp/step4_latest_baseline_repaired_final_pytest.log
  sha256: c7a869263afd16b8694259ecafbe7df29e5a3a02320298a01e9e6009d5b68154
  ```

- The scheduler-aware MTP-off requirements driver is complete for the current
  blocked checkpoint.
  Provider context/generation Attention is complete:
  `68` context and `167` generation rows, with `235/235` exact `silicon`
  consumer queries. The context table contains `34` Optimus FA4 and `34`
  native SWA rows, including the exact `65,536`-token points at
  `196.56292724609375 ms` and `3.164181391398112 ms`. Runtime profile,
  Latest SM103 model plan, grouped `wo_a`,
  FP32 router, QKV norm/RoPE, and Attention static/consumer paths are complete.
  QKV measurement is complete: Full MFA `75/75`, SWA `75/75`, and combined
  canonical `150/150` exact `silicon` queries with `0.0 ms` maximum error.
  The SWA runtime used only the explicitly authorized source-hash-bounded
  annotation compatibility overlay.
  The earlier `174`-row Optimus artifact is rejected because `150` rows used a
  silent graph-to-eager fallback. Its replacement is complete: `174/174` B300
  rows, zero errors, strict masked/CUDA-graph and contiguous/eager agreement,
  and `174/174` exact `silicon` consumer queries. Distributed DeepEP HT now has
  a locally verified exact Buffer API runtime path. Its torchrun driver,
  pinned-source remote runner, and parameterized 2-node EP16 / 4-node EP32
  host wrapper pass static tests. Two EP16 attempts verified source identity
  and repaired CUDA Error 803, then exposed `NCCL_IB_HCA=''` under an
  incomplete launch contract. The standalone two-node full-RDMA NCCL preflight
  passes with `16/16` rank markers, rank sum `136`, participant sum `16`,
  non-empty platform HCA, and exact cleanup. The replacement EP16 smoke still
  fails during cross-node `deep_ep.Buffer.runtime.sync`. A one-node,
  eight-rank `num_rdma_bytes=0` discriminator completed Buffer construction on
  `8/8` ranks, then failed explicit destroy on `8/8` ranks at
  `runtime.cu:29`. The exact `9bfd9a610e` retry is complete and failed because
  the available launcher cannot submit its required shared-host-SHM contract.
  Per owner direction, DeepEP is frozen at `0/116`; do not retry it in this
  phase. Formal simulation remains blocked wherever its exact dispatch/combine
  rows are required.
- Corrected EP16+EP32 scheduler-aware coverage queried `36,420` records:
  `18,220` exact silicon, `15,160` analytic/non-silicon, `3,040` missing,
  and `0` unexpected errors. Attention, Optimus MoE, grouped GEMM, FP32
  router, and QKV have complete exact-silicon coverage. QKV contributes
  `1,560/1,560` exact-silicon records. The four missing physical contracts
  are DeepEP dispatch/combine at EP16 and EP32.
- The exact-path requirements matrix executed all `72` prefill and `84` decode
  records. All are explicitly `BLOCKED` only by DeepEP. Prefill peak HBM
  ranges from `142.00439453125` to `776.97998046875 GiB`, with `48/72`
  workloads over the `241.734375 GiB` utilization limit. Decode batch-1 HBM
  ranges from `138.58154296875` to `294.77685546875 GiB`, with `16/84`
  records over the same limit. Known prefill first-chunk partial latency is
  `99.18793976790863–8158.937195512976 ms`; known decode batch-1 partial
  latency is `38.4266577068447–250.7935116231927 ms`. These values are not
  formal latency or TPOT.
- The explicit `b300_nccl_alltoall` proxy coverage queried `36,420` records:
  `18,220` exact silicon, `15,160` analytic/non-silicon, `3,040` proxy,
  `0` missing, and `0` errors. The final proxy matrix completed all `72`
  prefill and `84` decode rows. Prefill latency is
  `129.22378441076575–782420.0031436655 ms`; per-replica input throughput is
  `1340.1293369125042–69183.9264811136 token/s`; aggregate input throughput is
  `1340.1293369125042–103512.27987036602 token/s`. Decode batch-1 TPOT is
  `56.955545921130614–268.6990437660492 ms`; all `84` decode combinations
  have `B_max=0` and `first_failed_batch=1`. Every such value carries
  `result_fidelity=PROXY`.
- MTP1 structure tests, measurement, and simulation are deferred by explicit
  user decision. They remain visible as deferred scope and must not be
  substituted with `Step3p5MTP`.
- The B300 pinned-vLLM whole-model smoke/runtime trace is externally owned.
  This session does not launch or monitor it. AIC-side implementation,
  Collector development, operation measurement, tests, and simulation
  continue; final provider/source sign-off will ingest the external report
  supplied by the task owner.

## Phase 10 Implementation Plan

### Task 10.1 — Proxy contract module

**Files:**

- Create `tests/performance/step4_pro_latest/deepep_proxy.py`.
- Create `tests/unit/performance/test_step4_pro_latest_deepep_proxy.py`.

**Interface:**

```python
query_deepep_proxy(
    operation,
    database,
    *,
    tokens_per_dp_rank: int,
    proxy_name: str,
) -> DeepEPProxyResult
```

The result carries scaled latency/energy plus complete proxy metadata. Tests
must first fail because the module is absent, then prove:

1. dispatch uses NCCL `alltoall` with `CommQuantMode.int8`;
2. combine uses NCCL `alltoall` with `CommQuantMode.half`;
3. message elements equal
   `ceil(tokens_per_dp_rank * hidden_size * topk / ep_size)`;
4. EP16 and EP32 are passed unchanged to the NCCL consumer and disclose the
   8-GPU measured-curve correction;
5. operation `_scale_factor` applies to latency and energy;
6. unknown proxy names and non-Step4 DeepEP operations fail fast.

### Task 10.2 — Coverage and simulation opt-in

**Files:**

- Modify
  `tests/performance/step4_pro_latest/validate_aic_silicon_coverage.py`.
- Modify
  `tests/performance/step4_pro_latest/run_mtp_off_requirements.py`.
- Modify
  `tests/unit/performance/test_step4_pro_latest_silicon_coverage.py`.
- Modify
  `tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py`.

Both CLIs add:

```text
--deepep-proxy b300_nccl_alltoall
```

The default is `None`. Tests must first prove that the default still reports
missing DeepEP data. With explicit opt-in, operation records use
`status=proxy`, summaries use `PASS_WITH_PROXY`, and exact-silicon,
non-silicon, missing, error, and proxy counts remain separate.

### Task 10.3 — Verification and full matrix

1. Run the focused proxy/unit tests.
2. Run the existing focused Step4-Pro-Latest tests and Collector tests.
3. Run coverage once without the proxy to preserve fail-fast evidence.
4. Run coverage with the proxy and verify zero missing/error records.
5. Run proxy smoke simulation.
6. Run the full `72` prefill and `84` decode matrix.
7. Record prefill latency/throughput/HBM and decode TPOT/`B_max` numeric
   ranges, always with `result_fidelity=PROXY`.
8. Verify that no `step4_deepep_ht_perf.parquet` was created.
9. Update the test report, issue resolution, review, summary, and deliverable
   inventory.

## Phase 11 Implementation Plan

1. Add failing unit contracts for KV requested/resolved dtype, logical and
   allocated/peak-allocated KV bytes, required component latency breakdown,
   throughput-per-GPU, explicit unavailable online-runtime metrics, and
   repeat auditing.
2. Extend only the task-local MTP-off requirements driver; do not alter
   measured operation data or the approved DeepEP proxy mapping.
3. Run the complete `72` prefill plus `84` decode proxy matrix three times
   and save a compact repeat-audit artifact with per-case equality and numeric
   spread.
4. Update the existing simulation report with a requirement-status matrix,
   commands, environment, per-topology/per-workload numeric tables, component
   and KV ranges, and explicit deferred/external items.
5. Run fresh focused tests, Collector tests, Ruff, shell syntax, artifact
   validation, DeepEP-Parquet absence, and `git diff --check`.
6. Stage only task-owned implementation, B300 data, test, and task-document
   files; exclude local caches, unrelated historical tasks, H800 data, and
   the `vllm-step4-pro` checkout. Commit once and push
   `task/step4-pro-latest-b300`.

**Completion evidence:** The final artifact audit passed with `52/52`
inventory hashes; a fresh complete proxy matrix was byte-identical to the
archived result; `343/343` focused and `401/401` Collector tests passed; Ruff
passed on `45/45` Python files; shell syntax passed on `14/14` scripts. Remote
publication is represented by the Git branch state rather than a
self-referential commit hash in this file.

## Phase 12 Implementation Plan

1. Add failing contracts requiring the active runtime wrappers to select
   `allgather_reducescatter`, disable sequence parallelism, pass the backend
   through both environment and CLI, and reject DeepEP manager selection.
2. Update the active single-node, two-node, and shared remote runtime scripts.
   Keep the DeepEP-specific legacy probe scripts unchanged as historical
   diagnostics.
3. Remove active runtime dependencies on explicit NVSHMEM settings and on
   importing the `deep_ep` package only for evidence collection.
4. Update the authoritative requirements and task records. Keep the AIC
   `b300_nccl_alltoall` result explicitly separate from the AgRs runtime.
5. Run focused pytest, shell syntax, backend-source audit, and whitespace
   checks under a `MemoryMax=2G` controller scope.
6. Run B300 predict-only, then single-node and two-node live smoke. The
   single-node run validates pinned model loading and real requests but does
   not instantiate an all2all manager at DP=1. Distributed acceptance requires
   `Using AgRsAll2AllManager all2all manager`, zero DeepEP and automatic
   backend-selection markers, successful real-batch forward markers, and
   exact resource cleanup.

**Configuration completion evidence:** `11/11` focused contracts passed in
`0.04s`; shell syntax passed for `3/3` active scripts; the source/backend audit
confirmed AgRs `all_gatherv`/`reduce_scatterv`, sequence parallel `0`, zero
active DeepEP/NVSHMEM settings, zero `deep_ep` evidence dependencies, generic
DeepEP-manager rejection, and zero allowed automatic backend selection. The
first final audit assertion used the obsolete exact DeepEP-HT shell pattern;
aligning it to the current generic rejection contract resolved the
validation-only failure without changing runtime code.

**Predict-only evidence:** The exact single-node resource shape returned `4`
candidate B300 nodes. The exact two-node shape, including two replicas,
host networking, both RDMA resources, topology grouping, and distributed
environment injection, returned `10` candidate B300 nodes. Exact-name cleanup
queries found `0` RJobs and `0` Replicas. The active gate is now the bounded
single-node live smoke.

**Single-node live evidence:** One B300 scheduled in `16s`; the pinned dummy
model loaded in `8.819692s`; one token-ID request and four concurrent requests
completed; `9` real-batch forward markers were observed; DeepEP and automatic
backend-selection markers were both `0`; cleanup left `0` RJobs and `0`
Replicas. DP=1 does not instantiate an all2all manager, so the active gate is
the two-node run that must prove `AgRsAll2AllManager`.

**Original two-node live evidence:** Two replicas reached Running in `16s`.
Rank 0 selected `AgRsAll2AllManager` once, emitted zero DeepEP and
automatic-selection markers, loaded in `12.957171s`, and completed real
requests. Rank 1 never wrote `remote_result_ready`: the reused one-node runner
stopped rank 0 and its TCPStore before the headless rank group completed.
Cleanup found `0` RJobs and `0` Replicas. The owner then authorized the
coordinated lifecycle root fix.

**Superseded lifecycle design:** The first root fix used validation-ready,
shutdown-arm, armed-acknowledgement, and concurrent release markers. It passed
local contracts but still required several remote control-plane writes during
distributed teardown. Phase 14 replaces that protocol.

**Current lifecycle design:**

1. The remote runner treats `DATA_PARALLEL_SIZE > 1` as one distributed
   execution and writes `remote_validation_ready` only after local validation.
2. Both remote runners keep vLLM and TCPStore alive after readiness; they do
   not wait for or create shutdown-arm/release marker files.
3. The host waits for both readiness markers, pulls both evidence trees while
   the RJob is still live, and validates
   `DISTRIBUTED_RUNTIME_VALIDATION=PASS`, a completed real-batch forward,
   backend markers, and a clean runtime log.
4. After all evidence is accepted, the host calls
   `brainctl delete rjob` exactly once. Cleanup passes only when both exact
   RJob and Replica queries succeed and contain no matching resource.

**Lifecycle implementation evidence:** The new two-node contract first failed
`4/6`, the runtime-contract transfer contract failed `1/1`, and the timeout
margin contract failed `1/1`. The final two-node contract passes `7/7`; all
related runtime contracts pass `26/26`; shell syntax passes `6/6`; Ruff
check/format and `git diff --check` pass. No live RJob was submitted because
current independent quota evidence for `16` B300 GPUs is unavailable.

The first post-coordinator live run
`s4p-agrs2-0817-174506` reached `2/2` Running in `78s`. Rank 0 completed real
requests and recorded AgRs `1`, DeepEP `0`, and automatic selection `0`.
Rank 1 did not emit its own manager line because the pinned source logs that
line with `scope="global"`. The validation contract was corrected to require
per-node backend configuration plus real-batch evidence and at least one
manager line across the whole job.

The final corrected payload was submitted as
`s4p-agrs2b-0817-175947`, but the platform created `0` replicas. Its event
reported a request for `16` B300 GPUs with only `6` remaining in
`b300-train-infra-default`. Exact cleanup and fresh post-stop queries both
found RJobs `0` and Replicas `0`.

A fresh diagnostic proved that `predict-only` is not a total-quota gate:
`--replica 2` and `--replica 8` both returned the same seven per-worker B300
candidates and created no resources. Direct reads of
`quotagroups.stepmind.com/b300_train_infra` are forbidden for the current
identity. No further live attempt is allowed until a platform-visible event,
quota owner, or other authorized direct source confirms at least `16`
currently available B300 GPUs; same-shape predict-only remains necessary for
node fit but is not sufficient for admission.

## Phase 13 Review and Publication Plan

1. Re-read the current task requirements and inspect the complete uncommitted
   code/doc diff.
2. Run the focused Step4-Pro-Latest regression, full Collector regression,
   Ruff check/format, shell syntax, artifact/hash audit, and
   `git diff --check`.
3. Normalize the final publication report, summary inventory, review record,
   and progress/issues records to the current quota-blocked state.
4. Stage only the current task code, tests, authoritative requirements update,
   and reviewed task reports. Exclude caches, unrelated historical tasks,
   H800 data, temporary outputs, and `vllm-step4-pro/`.
5. Commit once, push `task/step4-pro-latest-b300`, and verify local/remote
   ahead/behind is `0/0`.

**Publication boundary:** This phase may publish the reviewed checkpoint by
the owner's explicit request. It does not close the Phase 12 two-node live
acceptance gate or convert `BLOCKED_BY_QUOTA` to `PASS`.

**Phase 13 completion evidence:** The complete code/doc diff was reviewed with
blocking findings `0`; `347/347` focused tests, `401/401` Collector tests,
Ruff `45/45`, and shell syntax `14/14` passed. Commit `0b8d651c` is present
locally and on `origin/task/step4-pro-latest-b300`.

## Phase 14 Follow-up Hardening Plan

1. Replace the multi-marker shutdown protocol with live evidence pull followed
   by one exact RJob delete.
2. Share quota, cleanup, and runtime-log validation through
   `b300_runtime_contract.sh`; fail if inventory queries fail.
3. Run predict-only with the exact live launch argument array, require at least
   one candidate node, archive its SHA256, and prove it created no resources.
4. Require independent `>=16` B300 quota evidence before any live submission.
5. Add a source-hash-bounded `MODEL_FORWARD_COMPLETE` diagnostic after the
   model forward and a CUDA synchronize. Treat it as smoke evidence only
   because the synchronize perturbs request timing.
6. Keep remote execution alive for at least
   `2400 + 2 * 300 + 60 = 3060s`; the default remains `3600s`.
7. Update the existing task records and test report, rerun focused/static/hash
   gates, stage only task-owned files, then create and push one follow-up
   commit. The live result remains `BLOCKED_BY_QUOTA`.

**Phase 14 publication evidence:** Commit
`bd91ce3a41fabde65b2e6f5707907a72b3ffb9a0` published the 23-file
implementation/docs candidate. Local HEAD, the remote-tracking branch, and
the direct remote branch query all matched that SHA immediately after push;
ahead/behind was `0/0`. This closeout documentation records that completed
publication without changing the quota-blocked live status.
