## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created task-specific gates and non-negotiable constraints. |
| 2026-08-13 | Added explicit missing-manifest and dirty-worktree gates. |
| 2026-08-13 | Added the pinned-vLLM runtime-trace gate for accepted Full-MFA profiling rows. |
| 2026-08-13 | Added the owner-directed no-action rule for the temporary security-review file. |
| 2026-08-13 | Made exact pinned-vLLM operation execution mandatory for every accepted Latest measurement. |
| 2026-08-14 | Added a hard gate for the missing pinned-vLLM Step4Pro MTP1 path. |

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
11. Every `Step4-Pro-Latest` definition, test, measurement, and validation
    must target the same graph extracted from pinned vLLM commit
    `607d1641ee3fec43653fca510d717725828890c2`.
12. Every accepted B300 measurement must execute the operation implementation
    provided by the pinned vLLM version and reconcile source path, runtime
    provider, Collector identity, persisted key, and AIC consumer.
13. The MTP1 requirement cannot be accepted from `Step3p5MTP` or an invented
    AIC-only graph. Until Q8 is resolved, MTP1 remains explicitly
    unimplemented and unmeasured while MTP-off Latest work may proceed.

## Prohibited Actions

- No silent fallback logic.
- No H800 data reuse as B-card results.
- No unrecorded assumptions or invented shapes.
- No `rm`, `mv`, reset, or bulk replacement without explicit permission.
- No GPU/Docker launch before the identity gate is closed.
- No temporary files under `/tmp`.
- No generic FlashAttention substitution for Optimus FA4.
- No dense-GEMM or multiplied-time substitution for grouped `wo_a`.
- No stand-alone benchmark reimplementation when the pinned vLLM operation or
  provider path is available.
- No MTP1 substitution from `Step3p5MTP` for Step4Pro, and no invented
  Step4Pro MTP1 provider under the pinned source identity.
- Do not access or modify
  `/data/ycfeng/tmp/aic_failure_domain1/codex_lane3_scope_baseline.md`.
