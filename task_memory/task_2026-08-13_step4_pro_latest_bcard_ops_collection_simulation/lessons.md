## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-17 | Replaced the multi-marker shutdown lesson with live evidence collection, single-owner resource deletion, query-aware cleanup, and completion-marker timing boundaries. |
| 2026-08-17 | Added the verified vLLM backend-selection, communication-identity, distributed log-scope, coordinated-shutdown, and quota-admission lessons from replacing DeepEP with AgRs. |
| 2026-08-13 | Created the reusable-lessons scaffold; no lesson has been validated yet. |
| 2026-08-15 | Added vetted Collector-coverage and blocked-simulation lessons. |
| 2026-08-16 | Added vetted annotation, provider-output, and partial-slice validation lessons. |

# Vetted Lessons

1. **Validate generated scheduler queries, not only persisted rows.** Exact
   row-consumer equality does not prove that chunked prefill and decode search
   are bracketed. Derive workloads through the same scheduler used by formal
   simulation before declaring a dataset complete.
2. **Keep global and rank-local token meanings separate.** Attention latency
   follows the busiest attention-DP rank, while Optimus MoE consumes global
   scheduled tokens. Using one token count for both creates false gaps.
3. **Separate formal and partial latency.** When one required operation is
   missing, keep known partial latency visible but leave formal latency and
   `B_max` null.
4. **Use the runtime memory limit, not raw device capacity.** OOM classification
   must apply the same `gpu-memory-utilization` limit as the target vLLM
   launch.
5. **A failed provider must not block independent families.** Splitting
   grouped/router measurement from QKV allowed valid exact data to complete
   without fallback or placeholder rows.
6. **Treat postponed JIT annotations as part of the runtime ABI.** When a DSL
   consumes raw annotations, a stringified `Constexpr` can turn a compile-time
   argument into an invalid dynamic argument. Bound any compatibility repair
   to an exact source hash and the intended annotation fields.
7. **Validate provider outputs before timing acceptance.** Shape and dtype are
   insufficient; reject NaN and Inf before a row enters the canonical
   performance dataset.
8. **Scope result validation to the selected collection slice.** A valid
   provider-only run must not be rejected for files belonging to unselected
   families, while every file required by the selected slice remains
   mandatory.
9. **Pin Step MoE communication through both env and CLI.** In this vLLM
   branch, leaving `VLLM_ALL2ALL_BACKEND` unset allows runtime auto-selection
   to choose DeepEP when it is installed. The env variable's presence
   prevents that change. Repeating the value in `--all2all-backend` writes the
   same identity into `parallel_config`, including DP=1 paths, and makes the
   command auditable.
10. **Use backend implementation names precisely.** The supported replacement
    is `allgather_reducescatter`/`AgRsAll2AllManager`: NCCL-backed all-gather
    plus reduce-scatter. It is neither a literal backend named `nccl` nor the
    same algorithm as an NCCL `alltoall` simulation curve.
11. **Manager selection is not per-call evidence.** The pinned AgRs manager
    has no DeepEP-style dispatch/combine diagnostic logger. A manager marker
    plus real-batch forward proves the selected runtime path but does not
    independently count both collectives; use a profiler/provider trace for
    that stronger claim.
12. **Match validation scope to logger scope.** A line emitted through
    `logger.info_once(..., scope="global")` is a job-level singleton, not a
    per-replica contract. Require backend configuration and real-batch
    evidence on every replica, then require the global manager marker at
    least once across the complete job.
13. **Collect distributed evidence before resource teardown.** A successful
    API rank must not stop its TCPStore while headless ranks are still
    validating. Keep all ranks alive after validation, pull every evidence
    tree while the RJob exists, then let one host-owned RJob deletion terminate
    the distributed runtime.
14. **Treat predict-only as a per-worker fit check until replica sensitivity
    is proven.** Here, replica counts `2` and `8` returned the same seven-node
    list, while a real 2-replica submission later exposed only `6` GPUs of
    quota. Multi-replica admission therefore needs separate current quota
    evidence; a node list alone is insufficient.
15. **Cleanup proof includes query success.** An empty-looking output from a
    failed or OOM-killed control-plane query is not proof that resources are
    gone. Require successful exact-name queries and empty inventories before a
    launcher may report cleanup PASS.
16. **A synchronized completion marker is diagnostic instrumentation.** A log
    emitted before asynchronous CUDA work finishes proves only dispatch. A
    source-hash-bounded synchronize after model forward proves completion and
    catches deferred CUDA failures, but it changes timing and must not be used
    as an uninstrumented latency result.
