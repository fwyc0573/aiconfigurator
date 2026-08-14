## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created task-specific gates and non-negotiable constraints. |
| 2026-08-13 | Added explicit missing-manifest and dirty-worktree gates. |
| 2026-08-13 | Added the pinned-vLLM runtime-trace gate for accepted Full-MFA profiling rows. |
| 2026-08-13 | Added the owner-directed no-action rule for the temporary security-review file. |
| 2026-08-13 | Made exact pinned-vLLM operation execution mandatory for every accepted Latest measurement. |
| 2026-08-14 | Added a hard gate for the missing pinned-vLLM Step4Pro MTP1 path. |
| 2026-08-14 | Deferred MTP1 structure tests/simulation and opened the parallel B300 smoke gate. |
| 2026-08-14 | Assigned whole-model pinned-vLLM smoke/runtime trace to an external session while retaining AIC-side provider identity checks. |

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
