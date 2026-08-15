## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Recorded the initial design boundary; implementation design awaits runtime identity confirmation. |
| 2026-08-13 | Added the audited minimal-extension design candidate and its unresolved decision gate. |
| 2026-08-13 | Added the source-derived plus runtime-trace-validated profiling design. |
| 2026-08-13 | Marked the minimal-extension design as user-approved. |
| 2026-08-14 | Deferred MTP1 structure/tests/simulation and split execution into AIC and parallel B300 smoke tracks. |
| 2026-08-14 | Converted the pinned-vLLM smoke/runtime trace track into an external-session handoff and retained AIC-only execution locally. |
| 2026-08-15 | Defined the provider-specific perf-file boundary and exact-versus-interpolated lookup axes for Collector implementation. |

# Design

The task will use a source-first vertical path:

1. Pin the exact latest vLLM image/source identity.
2. Trace the actual Step4-Pro execution graph for prefill and decode.
3. Map each executed kernel-level operation to the existing AIC consumer contract.
4. Populate only the exact required Collector cases and preserve three identities: recipe, invocation, and persisted key.
5. Collect fresh B-card rows, then query them through the unchanged consumer.
6. Run MTP-off correctness tests before the requirements-defined prefill/decode
   experiments. MTP1 structure tests and simulation are deferred.
7. Archive provenance, numeric evidence, differences, and unresolved limitations.

The source/runtime identity gate is resolved: all implementation and collection
work must use the pinned local checkout and requirements shape. The existing
`Step4-Pro-V4` model remains untouched; the latest model receives a separate
identity derived from the unchanged consumer contract.

## Audited minimal-extension candidate

The current schema is not fidelity-complete for the pinned graph. The
recommended scoped design is:

1. Add `Step4ProLatestConfig` for the 20 Full-MFA / 58 SWA-GQA layer map.
2. Expand Full MFA into its low-rank Q, shared-KV, head gate, grouped `wo_a`,
   and `wo_b` sequence.
3. Expand SWA into packed QKV, Q/K/V norm, RoPE, head gate, and output
   projection.
4. Keep latent MoE serial in pinned execution order; represent the router as
   FP32 and charge DeepEP HT dispatch/combine separately.
5. Carry K/V alias, block/page size 128, requested/resolved dtype, logical
   bytes, and allocated bytes in the cache-layout contract.
6. Keep MTP1 outside the current production graph and reports until a native
   Step4Pro implementation source is approved; never substitute Step3p5MTP.
7. Reuse existing leaf operations for all work they already express. Add only
   the operation identities that are required to preserve grouped GEMM and
   vLLM DeepEP consumer keys.

This design was approved by the task owner on 2026-08-13. Implementation
planning remains gated only by safe branch/checkpoint handling.

## Profiling provenance

The Full-MFA graph has two evidence levels:

1. **Source graph:** extracted from the pinned
   `Step4ProDecoderLayer`/Full-MFA forward call path and used to define logical
   AIC boundaries.
2. **Runtime graph:** captured on B300 from the pinned image before formal
   collection. It validates actual operation order, shapes, dtypes, optional
   branches, fusion, backend selection, and CUDA kernels.

Accepted leaf measurements must run the exact implementation path when the
provider affects performance:

- hd512 Attention through Optimus FA4 with paged shared K/V;
- grouped `wo_a` through the actual vLLM grouped/einsum path;
- remaining projections and norms with the pinned vLLM modules and exact
  shape/dtype contract.

The runtime trace is an acceptance gate, not a replacement for AIC logical
operation boundaries. A mismatch fails fast and requires the logical graph or
case manifest to be corrected before collection continues.

## External runtime-trace interface

The current session no longer executes the whole-model pinned-vLLM B300 smoke
or runtime/provider trace. That task is specified in:

```text
task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/pinned_vllm_b300_smoke_runtime_trace_execution.md
```

The AIC track remains responsible for exact operation-level provider use in
Collector benchmarks. When supplied, the external report will be reconciled
against the AIC operation/provider matrix before final sign-off.

## Provider performance-data contract

Latest provider rows remain separate from generic vLLM tables. Structural and
provider fields require exact matches; only workload-size fields interpolate.

| Perf family | Exact physical key | Interpolated workload axes |
|---|---|---|
| Grouped `wo_a` | provider, groups, per-group N/K, quant mode | num tokens |
| FP32 router | provider, N/K, weight dtype, output dtype | num tokens |
| QKV norm/RoPE | provider, normalized tensors, Q/KV heads, head dim | num tokens |
| Context attention | provider, Q/KV heads, head dim, window, KV/FMHA dtype, K/V alias, page size | batch, query tokens, total context tokens |
| Generation attention | provider, Q/KV heads, head dim, window, KV dtype, K/V alias, page size | batch, context tokens |
| Optimus routed MoE | provider, phase, hidden/intermediate, top-k, experts, TP/EP, quant, routing distribution, gated flag | global input tokens |
| DeepEP HT | provider, operation, phase, hidden, top-k, experts, TP/EP/DP, quant | input tokens per rank |

The initial implementation proceeds as a vertical slice in this order:

1. pinned runtime profile and Latest model-plan selection;
2. grouped `wo_a` collector, persisted rows, loader, and consumer query;
3. FP32 router and QKV norm/RoPE token-curve families;
4. provider context/generation attention;
5. Optimus FP8 MoE;
6. distributed DeepEP HT EP16/EP32.

The runtime manifest keeps stock vLLM as the default and adds a named
Step4Pro profile. The profile records the exact StepCast image and pinned source
commit. A Latest plan selects that profile explicitly; unrelated model plans
remain unchanged.
