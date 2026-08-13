## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created task-specific gates and non-negotiable constraints. |
| 2026-08-13 | Added explicit missing-manifest and dirty-worktree gates. |
| 2026-08-13 | Added the pinned-vLLM runtime-trace gate for accepted Full-MFA profiling rows. |

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
10. The source-derived Full-MFA graph must be reconciled against a runtime
    trace from the pinned image before profiling rows are accepted.

## Prohibited Actions

- No silent fallback logic.
- No H800 data reuse as B-card results.
- No unrecorded assumptions or invented shapes.
- No `rm`, `mv`, reset, or bulk replacement without explicit permission.
- No GPU/Docker launch before the identity gate is closed.
- No temporary files under `/tmp`.
- No generic FlashAttention substitution for Optimus FA4.
- No dense-GEMM or multiplied-time substitution for grouped `wo_a`.
