## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-17 | Updated the quota-gated live rerun to the current evidence-pull and single-delete lifecycle. |
| 2026-08-17 | Replaced the insufficient predict-only quota gate with separate per-worker fit and direct total-quota evidence requirements. |
| 2026-08-17 | Replaced the completed coordinator implementation item with the quota-gated final two-node live rerun. |
| 2026-08-17 | Added live AgRs runtime validation and direct AgRs simulation calibration as future execution work; DeepEP is no longer required by the active runtime. |
| 2026-08-17 | Removed the completed single-B300 smoke from future work and recorded the required coordinated two-node lifecycle fix. |
| 2026-08-13 | Created the out-of-scope future-work record. |
| 2026-08-16 | Added the explicitly deferred DeepEP and MTP1 completion work. |
| 2026-08-16 | Clarified that future DeepEP work replaces the completed proxy results with exact-silicon results. |

# Future Work

- Any real-checkpoint routing, generation-quality, or MTP acceptance-rate study not required by the current performance task.
- Additional B-card configurations not included in the confirmed requirements envelope.
- Broader refactors unrelated to the latest Step4-Pro operation and collection path.
- Obtain direct platform or quota-owner evidence that at least `16` B300 GPUs
  are currently available, then rerun the exact two-node predict-only command
  for per-worker node fit and one bounded live wrapper. Predict-only alone is
  insufficient because replica counts `2` and `8` returned the same candidate
  list. The coordinator and global-marker-scope fixes already pass local
  contracts. Final live acceptance still requires both nodes to finish, at
  least one global `AgRsAll2AllManager` marker, a real-batch forward on each
  node, no DeepEP or automatic backend-selection marker, no `Broken pipe`,
  both evidence trees pulled while the RJob is live, exactly one RJob delete,
  and successful empty exact-name cleanup queries.
- Add pinned AgRs execution-level evidence, such as profiler/provider markers
  for both `all_gatherv` dispatch and `reduce_scatterv` combine. The current
  manager plus real-batch gate does not expose separate collective call
  markers.
- Add or calibrate an AgRs-specific AIC communication model if a same-backend
  vLLM-versus-simulator communication error is required. Until then, keep the
  existing NCCL `alltoall` result labeled `PROXY` and keep the exact DeepEP HT
  operation identity separate from both.
- DeepEP EP16/EP32 dispatch/combine measurement may resume as an optional
  comparison after a launcher/runtime can satisfy the required NVSHMEM
  contract. It is no longer a prerequisite for the active task runtime.
- MTP1 structure, operation measurement, and simulation after an authoritative
  native Step4-Pro MTP1 implementation is provided. Do not substitute
  `Step3p5MTP`.
