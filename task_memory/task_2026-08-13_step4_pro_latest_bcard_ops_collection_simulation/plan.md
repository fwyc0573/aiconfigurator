## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Initialized the execution plan and the first requirements-confirmation gate. |
| 2026-08-13 | Closed the source/runtime identity gate using user choice A. |
| 2026-08-13 | Confirmed the referenced manifest is unavailable and retained the explicit reconstruction decision gate. |
| 2026-08-13 | Closed the manifest reconstruction gate using user choice A. |
| 2026-08-13 | Added the operation-boundary architecture gate from the completed fidelity audit. |
| 2026-08-13 | Added a runtime-trace provenance gate for Full-MFA profiling cases. |
| 2026-08-13 | Closed the operation-boundary gate with option A and made branch/checkpoint safety the active gate. |
| 2026-08-13 | Closed the branch/checkpoint decision with option A; baseline checkpoint creation is in progress. |

# Plan: Step4-Pro-Latest B-Card Ops, Collection, and Simulation

## Objective

Deliver an auditable, latest-implementation-based `step4-pro-latest` operation definition, a fresh B-card performance dataset, correctness evidence, and the required prefill/decode simulation results without silently substituting historical H800 data or unspecified runtime inputs.

## Phase Status

| Phase | Status | Exit Criteria |
|---|---|---|
| 1. Requirements and historical review | Completed | Requirements, historical tasks, repository state, and environment rules are recorded. |
| 2. Latest vLLM source/image identity confirmation | Completed | Pinned checkout, commit, source files, and B300 path are recorded; later image is reference-only. |
| 3. Grill-me clarification gate | Completed | Source/runtime, model identity, manifest reconstruction, operation boundaries, profiling provenance, and branch/checkpoint handling are explicitly resolved. |
| 4. Latest op inventory and AIC design | In progress | Prefill/decode inventory and the minimal-extension direction are fixed; exact implementation tasks await the branch/checkpoint gate. |
| 5. AIC model/op and Collector implementation | Pending | Definition and case population pass focused tests and identity/deduplication checks. |
| 6. B-card collection | Pending | Fresh collection runs complete or have explicit terminal outcomes and reproduce from recorded commands. |
| 7. Correctness and simulation | Pending | Correctness tests pass before prefill/decode experiments; required results are recorded. |
| 8. Final review and archive | Pending | Reports, hashes, differences, issues, and final inventory are complete. |

## Current Execution Gate

The user selected the pinned local checkout and requirements contract:

```text
source: /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro
commit: 607d1641ee3fec43653fca510d717725828890c2
shape: requirements document's 78-layer synthetic Step4-Pro shape
runtime/image: pinned contract; later branch/image is reference-only
```

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
