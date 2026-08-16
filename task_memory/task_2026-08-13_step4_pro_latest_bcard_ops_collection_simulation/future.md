## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created the out-of-scope future-work record. |
| 2026-08-16 | Added the explicitly deferred DeepEP and MTP1 completion work. |
| 2026-08-16 | Clarified that future DeepEP work replaces the completed proxy results with exact-silicon results. |

# Future Work

- Any real-checkpoint routing, generation-quality, or MTP acceptance-rate study not required by the current performance task.
- Additional B-card configurations not included in the confirmed requirements envelope.
- Broader refactors unrelated to the latest Step4-Pro operation and collection path.
- DeepEP EP16/EP32 dispatch/combine measurement after a launcher/runtime can
  satisfy the required NVSHMEM contract; then rerun the unchanged MTP-off
  matrix without the proxy to obtain exact-silicon latency, TPOT, and decode
  `B_max`.
- MTP1 structure, operation measurement, and simulation after an authoritative
  native Step4-Pro MTP1 implementation is provided. Do not substitute
  `Step3p5MTP`.
