## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-17 | Replaced the shutdown-arm gate with live evidence pull plus one RJob delete and added strict cleanup, quota-file, timeout, and completion-marker gates. |
| 2026-08-17 | Corrected the quota gate after proving `predict-only` is per-worker only, and reduced remaining controller scopes to `MemoryMax=2G`. |
| 2026-08-17 | Added the two-node ready/armed/shutdown barrier, global AgRs marker scope, and quota admission gate. |
| 2026-08-17 | Added the active AgRs runtime gate and separated it from both DeepEP diagnostics and the NCCL `alltoall` simulation proxy. |
| 2026-08-13 | Created task-specific gates and non-negotiable constraints. |
| 2026-08-13 | Added explicit missing-manifest and dirty-worktree gates. |
| 2026-08-13 | Added the pinned-vLLM runtime-trace gate for accepted Full-MFA profiling rows. |
| 2026-08-13 | Added the owner-directed no-action rule for the temporary security-review file. |
| 2026-08-13 | Made exact pinned-vLLM operation execution mandatory for every accepted Latest measurement. |
| 2026-08-14 | Added a hard gate for the missing pinned-vLLM Step4Pro MTP1 path. |
| 2026-08-14 | Deferred MTP1 structure tests/simulation and opened the parallel B300 smoke gate. |
| 2026-08-14 | Assigned whole-model pinned-vLLM smoke/runtime trace to an external session while retaining AIC-side provider identity checks. |
| 2026-08-14 | Added the owner-directed 3 GiB controller scope and low-memory I/O gate. |
| 2026-08-15 | Made synthetic config plus vLLM dummy weights authoritative for B300 runtime tests. |
| 2026-08-15 | Authorized one explicit Optimus JIT activation-quant overlay for SM103 compatibility. |
| 2026-08-15 | Approved the inference-only `ep_gather` power-of-two block correction after a design/training safety review. |
| 2026-08-15 | Limited all runtime-overlay conclusions to provider/performance validation and prohibited quality or training-equivalence claims. |
| 2026-08-15 | Deferred further DeepEP measurement after its recorded runtime blockers and required remaining op families to continue. |
| 2026-08-15 | Reconfirmed the DeepEP no-retry gate after the failed attempt produced zero accepted rows. |
| 2026-08-15 | Reopened one exact DeepEP probe using commit `9bfd9a610e`; retained fail-fast, predict-only, no-fallback, and minimal-probe-first gates. |
| 2026-08-15 | Closed the exact DeepEP retry and restored the unconditional no-retry gate for the remainder of this phase. |
| 2026-08-16 | Authorized a source-hash-bounded, process-local SWA QKV annotation compatibility overlay. |
| 2026-08-16 | Authorized an explicit simulation-only B300 NCCL alltoall proxy for DeepEP while preserving the exact path as the default. |

# Task Harness

## Gates

1. Exact latest vLLM image/source identity must be confirmed before implementation.
2. Requirements-document shape and runtime inputs must be reconciled with latest implementation.
3. Every op must have source/call-path provenance and an unchanged-consumer contract.
4. Collector recipe identity, benchmark invocation identity, and persisted physical key must reconcile.
5. B-card data must be freshly measured; H800 data is reference-only.
6. Correctness tests must pass before formal prefill/decode simulation.
7. Every failure, OOM, unsupported case, and missing point must remain explicit.
8. The missing manifest must be either reconstructed with explicit provenance
   and a new hash or supplied by the task owner before production edits.
9. The dirty linked worktree must have an explicitly approved isolation or
   consolidation path before production implementation.
10. External B300 controller commands must run under `MemoryMax=2G`, use only
    exact resource queries, and keep large I/O disk-backed and streamed.
11. B300 model tests must use synthetic `config.json` and vLLM
    `--load-format dummy`; missing real checkpoint mounts are not a blocker.
10. The source-derived Full-MFA graph must be reconciled against a runtime
    trace from the pinned image before profiling rows are accepted.
11. Every `Step4-Pro-Latest` definition, test, measurement, and validation
    must target the same graph extracted from pinned vLLM commit
    `607d1641ee3fec43653fca510d717725828890c2`.
12. Every accepted B300 measurement must execute the operation implementation
    provided by the pinned vLLM version and reconcile source path, runtime
    provider, Collector identity, persisted key, and AIC consumer.
13. MTP1 structure tests, measurement, and simulation are deferred by explicit
   owner decision. They remain unimplemented and unmeasured; do not use
   `Step3p5MTP` or an invented AIC-only graph as a substitute. This does not
   block MTP-off Latest AIC implementation, correctness, B300 collection, or
   prefill/decode simulation.
14. The pinned-vLLM whole-model B300 smoke/runtime trace is executed by an
   external session using
   `pinned_vllm_b300_smoke_runtime_trace_execution.md`. This session must not
   launch or monitor that work.
15. AIC-side Collector measurements must still execute the exact pinned-vLLM
    operation/provider implementation. The external whole-model trace is a
    later sign-off input, not permission to substitute generic kernels.
16. The B300 runtime copy may replace only the broken image-native
    `Optimus.per_token_group_quant_fp8` call with
    `optimus_cutedsl.group_quant_fp8.per_token_group_quant_fp8`, as explicitly
    authorized by the owner. All Optimus DeepGEMM expert GEMMs and other
    providers remain pinned and unchanged.
17. The B300 runtime copy may correct `ep_gather` block selection to the
    largest power-of-two divisor up to 1024. The change must preserve exact
    dimension coverage and remain inference-only.
18. Runtime deviations must be listed explicitly in the report. The Optimus
    JIT quant overlay is accepted for provider/performance validation only and
    must not be reported as bitwise, generation-quality, loss, or
    training-convergence equivalence.
19. The isolated masked-gather correction, strict Optimus Triton driver
    signature, `flash_attn.__spec__` import repair, and attention-type-scoped
    fallback assertion may not alter weights, routing, expert GEMMs, attention
    arithmetic, or backend identity.
20. The owner-authorized direct-child commit `9bfd9a610e` retry is complete
    and failed at the launcher-contract gate. DeepEP is frozen at `0/116`;
    do not launch another DeepEP attempt in this phase.
21. The B300 SWA QKV collection process may resolve only
    `FusedQKNormRope.kernel` annotations `reload_from` and `delay_w_load` to
    the installed `cutlass.Constexpr` object after verifying QKNorm source
    SHA256
    `5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783`.
    The overlay must be process-local, must not rewrite installed source, and
    must not change the pinned provider, kernel body, shape, dtype, argument
    values, mathematics, or persisted key.
22. SWA QKV full collection may start only after an isolated B300 smoke
    compiles and executes the exact pinned provider with valid output shape,
    dtype, and finite values. Any new provider/runtime failure must remain
    explicit; no fallback kernel or synthetic timing is allowed.
21. The exact DeepEP path remains the default and must still fail fast when
    its measured B300 data is unavailable. For the owner-authorized temporary
    simulation only, an explicit opt-in may replace DeepEP dispatch/combine
    latency with B300 NCCL `alltoall`: FP8 payload for dispatch and BF16
    payload for combine.
22. Every proxy-affected coverage and simulation result must be labeled
    `PROXY`, must identify B300 NCCL `alltoall` and the payload/topology
    mapping, and must not be reported as exact DeepEP silicon.
23. The proxy must not create or populate
    `step4_deepep_ht_perf.parquet`. Restoring the real DeepEP environment
    requires remeasurement and replacement of all proxy results.
24. Replace worker `/tmp` paths in the execution adaptation with a verified
    writable disk-backed path. Do not alter the model shape, backend,
    precision, RDMA resources, explicit NVSHMEM bootstrap, or DeepEP Buffer
    test to obtain a pass.
25. Active vLLM smoke/performance runs must use
    `VLLM_ALL2ALL_BACKEND=allgather_reducescatter` and
    `--all2all-backend allgather_reducescatter`. A literal backend value
    `nccl` is invalid for the pinned vLLM.
26. Active AgRs runs must set `VLLM_ENABLE_SEQUENCE_PARALLEL=0`. The pinned
    Step MoE runtime documents sequence parallelism as DeepEP-only; enabling
    it with AgRs risks a rank/size mismatch.
27. Active runtime acceptance requires
    `Using AgRsAll2AllManager all2all manager` and must fail if
    any `Using DeepEP*All2AllManager` marker appears.
28. Active runtime scripts must not require explicit NVSHMEM settings or the
    `deep_ep` package. DeepEP-specific legacy probes remain unchanged and
    inactive.
29. The AIC `b300_nccl_alltoall` proxy and the vLLM AgRs runtime are different
    algorithms. Both may be used only with their exact identity recorded;
    their communication difference must not be reported as same-backend
    simulator error.
30. Active distributed runtime evidence must record the backend,
    sequence-parallel state, AgRs manager marker count, DeepEP manager marker
    count, and automatic-backend-selection marker count. The latter two must
    be zero.
31. AgRs has no pinned per-dispatch/per-combine diagnostic marker equivalent
    to the DeepEP HT logs. Until profiler/provider evidence is added, runtime
    acceptance proves manager selection and real-batch forward only; it must
    not claim separately logged collective call counts.
32. The current AIC exact operation identity remains DeepEP HT and the
    completed simulation uses an explicit NCCL `alltoall` proxy. Neither is an
    AgRs-specific model, and all three identities must remain separate.
33. `Using AgRsAll2AllManager all2all manager` is emitted with global log
    scope in the pinned vLLM. Distributed acceptance requires at least one
    marker across the whole job, not one marker per replica. Each replica must
    still prove the explicit AgRs configuration, a real-batch forward, and
    zero DeepEP/automatic-selection markers.
34. After all replicas write validation readiness, every remote vLLM/TCPStore
    process must remain alive until the host has pulled and accepted both
    evidence trees. Remote shutdown-arm, acknowledgement, and release markers
    are prohibited in the current lifecycle.
35. The final two-node live rerun requires both a fresh same-shape
    predict-only check and independent total-quota evidence for
    `8 GPU/replica x 2 replicas = 16 GPU`. Predict-only is a per-worker node
    fit check and must not be treated as proof of total replica quota. Do not
    queue another live run while the platform reports less than `16` B300
    GPUs remaining or while current quota is unreadable and the last trusted
    event remains below `16`.
36. Predict-only and live launch must consume the same launch-argument array.
    Predict-only must archive its output and SHA256, report at least one
    candidate `Node:`, and prove through successful exact-name queries that it
    created no RJob or Replica.
37. Distributed teardown has one resource owner. After host-side evidence
    validation, call `brainctl delete rjob` exactly once. Cleanup PASS requires
    successful exact RJob and Replica queries plus empty matching inventories;
    query failure must remain FAIL.
38. Distributed real-batch acceptance requires a source-hash-bounded
    `MODEL_FORWARD_COMPLETE` marker emitted after
    `current_platform.synchronize()`. This proves runtime completion but
    instruments timing, so request latency from that run must not be reported
    as an unmodified performance benchmark.
39. Remote execution and worker-holder timeouts must be at least the validation
    budget plus two evidence-pull budgets and a 60-second margin. With current
    defaults the minimum is `3060s`, and both defaults remain `3600s`.

## Prohibited Actions

- No silent fallback logic. The owner-authorized DeepEP proxy must require an
  explicit simulation option and must never activate because measured data is
  merely missing.
- No automatic DeepEP selection in active vLLM runs, and no replacement with
  an unsupported literal `--all2all-backend nccl`.
- No claim that `allgather_reducescatter` is a direct NCCL `alltoall`; it is
  an all-gather dispatch plus reduce-scatter combine implementation.
- No H800 data reuse as B-card results.
- No unrecorded assumptions or invented shapes.
- No `rm`, `mv`, reset, or bulk replacement without explicit permission.
- No GPU/Docker launch before the identity gate is closed.
- No multi-replica live launch based only on a predict-only node list.
- No cleanup PASS from failed, missing, or unreadable exact resource queries.
- No restoration of the superseded shutdown-arm/release marker protocol.
- No temporary files under `/tmp`.
- No namespace-wide Replica inventory.
- No whole Git pack, bundle, tar archive, or large log in shell variables,
  command substitutions, or in-memory file reads.
- No generic FlashAttention substitution for Optimus FA4.
- No dense-GEMM or multiplied-time substitution for grouped `wo_a`.
- No stand-alone benchmark reimplementation when the pinned vLLM operation or
  provider path is available. The temporary DeepEP proxy is a labeled
  simulation approximation, not a replacement measurement.
- No MTP1 substitution from `Step3p5MTP` for Step4Pro, and no invented
  Step4Pro MTP1 provider under the pinned source identity.
- No claim that dummy/random-weight output validates model quality or training
  effect.
- Do not access or modify
  `/data/ycfeng/tmp/aic_failure_domain1/codex_lane3_scope_baseline.md`.
