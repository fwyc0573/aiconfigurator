## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
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
