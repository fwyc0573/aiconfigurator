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
| 2026-08-15 | Rewrote the B300 runtime-deviation review with necessity, design, numerical, and training boundaries for every extra modification. |
| 2026-08-16 | Added the approved SWA QKV annotation compatibility design and completed-dataset boundary. |
| 2026-08-16 | Added the owner-approved, explicit simulation-only B300 NCCL alltoall proxy design for the deferred DeepEP family. |

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

This design was approved by the task owner on 2026-08-13 and is implemented
for the MTP-off graph. MTP1 remains deferred and DeepEP exact data remains the
only formal simulation blocker.

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

## B300 SM103 Optimus quant overlay

### Decision

This overlay is **required for this exact pinned-image B300 run**, but it is
not a general Step4Pro architecture requirement.

The fixed runtime contract combines B300 SM103, torch `2.10`/CUDA `12.9`, the
pinned vLLM source, and Optimus DeepGEMM. Under that contract:

- the image-native `torch.ops.Optimus.per_token_group_quant_fp8` leaves the
  next CUDA operation failing with `cudaErrorNoKernelImageForDevice`;
- the `step-optimus==3.23.24` wheel's native library targets
  torch `2.8`/CUDA `12.8` and cannot load because of an ABI mismatch;
- the installed Optimus JIT CuTe DSL quantizer runs on SM103 and supplies the
  packed UE8M0 scales required by the fixed DeepGEMM `2.4.2` Blackwell path.

Changing model shape, switching MoE backend, or silently using a generic
kernel would avoid the failure only by violating the pinned-provider contract.
The normal long-term repair is a torch `2.10`/CUDA `12.9` Optimus native build
with SM103 support; that build is not present in the pinned runtime.

- Replace only `torch.ops.Optimus.per_token_group_quant_fp8`.
- Use installed
  `optimus_cutedsl.group_quant_fp8.per_token_group_quant_fp8`.
- Emit `sm100_1d1d` packed UE8M0 scales and set
  `VLLM_USE_DEEP_GEMM_E8M0=1`, matching the fixed DeepGEMM 2.4.2 Blackwell
  contract.
- Preserve pinned vLLM source checkout identity before applying the overlay.
- Preserve Optimus FA4, Optimus DeepGEMM expert GEMMs, routing, DeepEP,
  precision, and all model shapes.
- Record the runtime file hash and overlay diff in evidence; never present the
  runtime copy as byte-identical to the pinned source.

### Design and training impact

- Model architecture, tensor shapes, weights, router logits, expert selection,
  top-k, DeepGEMM expert GEMMs, attention, and DeepEP are unchanged.
- The activation-quant implementation and scale representation do change.
  Therefore this overlay is not bitwise-equivalent evidence and may change
  inference rounding at the FP8 boundary.
- The current run uses dummy weights and inference-only vLLM execution. It
  writes no checkpoint, gradient, optimizer state, or training artifact, so it
  cannot damage an existing training result.
- This evidence is sufficient for provider reachability and performance
  smoke. It is not a generation-quality, accuracy, loss, or training-
  convergence validation. Reusing the overlay in a trainer would require a
  separate numerical and backward-pass review.

## B300 `ep_gather` block correction

The pinned Triton kernel requires a power-of-two `BLOCK_D` and performs
unmasked loads/stores, so the block must also divide the hidden dimension.
The existing `min(hidden_size,1024)` rule is invalid for latent sizes 896 and
3584.

This correction is **required for the requested hidden sizes**:

- smoke latent hidden `896`: the old rule selects `896`, which Triton rejects;
- target latent hidden `3584`: the old rule selects `1024`, which does not
  divide the dimension.

Changing the model dimensions or choosing another MoE backend would hide the
defect and violate the fixed test contract. The later branch commit after the
pinned revision changes only DeepEP/NVSHMEM launch scripts and does not repair
this kernel.

The approved runtime correction is:

```python
BLOCK_D = min(1024, hidden_size & -hidden_size)
```

Safety properties:

- `hidden_size & -hidden_size` is the largest power-of-two divisor.
- Every hidden element is covered exactly once; no padding or truncation is
  introduced.
- The per-element top-k accumulation order is unchanged.
- Existing valid sizes retain their previous block: 512→512, 1024→1024,
  2048→1024, and 3072→1024.
- Required sizes become valid: 896→128 and 3584→512.
- `ep_gather` is a `torch.no_grad()` vLLM inference kernel. It does not modify
  model weights, training graphs, optimizer state, routing, or training
  convergence.

For dimensions that the old implementation could execute legally, the selected
block is unchanged. For newly supported dimensions, only launch geometry and
inference performance change. Each output element still accumulates the same
top-k values in the same order, so the mathematical result is preserved.
Its exact runtime diff and hashes must be preserved with B300 evidence.

The same defect exists in image package
`optimus_triton.deep_gemm_ep_gather_masked._select_block_d`. Its current
implementation returns any divisor, including non-power-of-two 896. The
runtime package is copied to an isolated overlay and receives the same
largest-power-of-two-divisor rule. The system package is not edited.

Both gather functions are guarded by `torch.no_grad()` and are used only by
the vLLM inference MoE path. They do not participate in training forward/
backward graphs and cannot change model weights, optimizer state, router
training, or convergence.

Optimus Triton driver caching defaults to a weak runtime signature that omits
tensor shapes. Because smoke transitions from batch 1 to batch 4,
`OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE=1` is enabled so cached driver entries
are keyed by full runtime shapes/values. This changes cache identity only, not
kernel code or arithmetic. It may add compilation/cache overhead, which is a
runtime-performance effect rather than a model-semantic effect.

Provider fallback checks are scoped by attention type. Full-MFA must emit an
actual Optimus FA4 forward marker. SWA is a separate native sliding-GQA path
and may legitimately select Triton/TRTLLM; its marker must not invalidate
Full-MFA evidence.

## SWA QKV annotation compatibility overlay

The pinned SWA production width is `18,432`, which selects
`reload_from="smem"` in the image-native
`optimus_cutedsl.qknorm_rope.FusedQKNormRope` kernel. That module uses
postponed annotations, while Cutlass DSL `4.4.2` consumes raw annotations
without resolving strings. The intended `cutlass.Constexpr` annotation is
therefore seen as a string, and compilation rejects `reload_from` as a dynamic
argument before any kernel executes.

The owner-approved repair is deliberately narrower than a provider or kernel
change:

1. verify the exact image-native QKNorm source SHA256
   `5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783`;
2. in the isolated `qkv_swa` process only, resolve `reload_from` and
   `delay_w_load` to the installed `cutlass.Constexpr` object;
3. fail before collection if source identity or annotation scope differs;
4. keep source files, vLLM call path, provider, kernel body, shapes, dtypes,
   arguments, QKV arithmetic, and persisted keys unchanged;
5. require shape, BF16 dtype, and finite-value validation before accepting a
   timing row.

This is necessary for the fixed pinned image because the exact Step4-Pro SWA
shape always reaches the affected compile branch. Avoiding it by changing the
shape or provider would no longer measure the vLLM implementation requested
by the task. The completed dataset proves provider reachability and
performance for this bounded runtime contract; it is not a training or model-
quality claim.

## Temporary DeepEP simulation proxy

The exact `vllm_deepep_high_throughput` operation remains the production AIC
contract and remains fail-fast when its B300 table is absent. The temporary
proxy is isolated to the task-level coverage and requirements-matrix drivers:

```text
tests/performance/step4_pro_latest/deepep_proxy.py
tests/performance/step4_pro_latest/validate_aic_silicon_coverage.py
tests/performance/step4_pro_latest/run_mtp_off_requirements.py
```

The proxy is disabled by default and can be selected only with the explicit
name `b300_nccl_alltoall`. It never catches a DeepEP lookup failure and never
activates because data is missing. When selected, it directly replaces the
DeepEP query in simulation with this declared mapping:

```text
message_elements =
    ceil(tokens_per_dp_rank * hidden_size * topk / ep_size)

dispatch:
    logical payload = FP8
    NCCL table dtype = int8 (the measured one-byte transport-equivalent curve)

combine:
    logical payload = BF16
    NCCL table dtype = half
```

The query calls the existing B300 NCCL `alltoall` consumer with the actual
requested EP size and then applies the original operation `_scale_factor` to
latency and energy. The B300 table contains measured rank counts `2/4/8`;
EP16 and EP32 therefore use AIC's existing rank-count correction from the
8-GPU measured curve. This extrapolation is part of the proxy disclosure and
must not be described as measured EP16/EP32 DeepEP.

Each substituted operation record uses:

```text
status = proxy
source = proxy_b300_nccl_alltoall
result_fidelity = PROXY
```

Step summaries and matrix summaries use `PASS_WITH_PROXY`, retain exact
silicon, analytic/non-silicon, and proxy counts separately, and report proxy
latency separately. Formal latency, TPOT, and `B_max` may be calculated only
with `result_fidelity=PROXY` visible on the same result.

No proxy row is persisted into `step4_deepep_ht_perf.parquet`; the real
DeepEP dataset remains absent at `0/116`. Once the runtime environment is
restored, real EP16/EP32 dispatch/combine rows must replace this proxy and the
same drivers must be rerun with the proxy option omitted.

## Complete runtime-deviation ledger

| Change | Why it exists | Semantic boundary |
|---|---|---|
| `step-optimus 3.23.24` FA4 overlay and two CuTe pointer-API edits | Inherited directly from both pinned smoke recipes | Provider-preserving compatibility; not a new model change |
| Synthetic config, `--load-format dummy`, token-ID requests | Explicit parent requirement; avoids unavailable checkpoint/tokenizer mounts | Valid for execution/performance, not text quality |
| `flash_attn.__spec__` `ModuleSpec` repair | The compatibility package otherwise fails import with `flash_attn.__spec__ is None` | Import metadata only; no kernel or arithmetic change |
| Optimus JIT activation quant plus packed UE8M0 | Native SM103 quant path is unusable and fixed DeepGEMM rejects FP32 scales | Inference quant implementation changes; no training artifact change; no bitwise/quality claim |
| Pinned contiguous `ep_gather` block correction | Old block cannot compile for 896 and cannot divide 3584 | Inference tiling only; exact element coverage and accumulation order preserved |
| Isolated masked-gather selector correction | Old helper returns invalid non-power-of-two 896 | Same inference-only tiling boundary; installed package remains untouched |
| `OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE=1` | Prevents cross-shape driver-cache reuse between batch 1 and batch 4 | Cache-key behavior only |
| Attention-type-scoped fallback assertion | A global assertion incorrectly classified legal SWA Triton/TRTLLM execution as Full-MFA fallback | Test/validation logic only; runtime execution unchanged |
| Source-hash-bounded SWA QKV annotation resolution | Cutlass DSL 4.4.2 does not resolve the image-native postponed `Constexpr` annotations | Annotation metadata only; provider, kernel body, inputs, outputs, QKV arithmetic, and persisted identity unchanged |
| Explicit B300 NCCL `alltoall` proxy for DeepEP | Exact DeepEP measurement is owner-deferred at `0/116`, while MTP-off simulation must continue | Simulation-only approximation; always labeled `PROXY`; no fake DeepEP rows; must be replaced by real DeepEP measurement |

These deviations are approved only for the pinned B300 provider/performance
validation. They must remain visible in reports and must not be described as
an unmodified pinned binary run.
