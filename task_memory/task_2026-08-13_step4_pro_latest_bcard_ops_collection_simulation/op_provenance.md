## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-13 | Created the pinned-source Step4-Pro-Latest logical operation and provenance inventory. |
| 2026-08-13 | Replaced the unresolved DeepEP consumer note with the audited vLLM HT dispatch/combine contract. |
| 2026-08-15 | Added the implemented AIC/Collector status and B300 dataset coverage for each provider-sensitive family. |
| 2026-08-16 | Recorded complete Full-MFA/SWA QKV measurement and narrowed the remaining B300 gap to DeepEP. |

# Step4-Pro-Latest Operation Provenance

## Scope

This inventory uses:

```text
source: /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro
commit: 607d1641ee3fec43653fca510d717725828890c2
shape: task_memory/step4pro_v4_external_simulator_requirements.md
```

It describes the required AIC logical operations and their current
implementation/measurement status. MTP1 remains explicitly deferred because
the pinned Step4Pro source has no native MTP1 construction path.

## Current AIC and B300 Coverage

| Operation family | Pinned vLLM implementation | AIC/Collector status | B300 data status |
|---|---|---|---|
| Full-MFA Q/K normalization and RoPE | `step4pro.py` Full MFA path | Defined and exact-key consumed | `75/75` rows |
| SWA Q/K/V normalization and RoPE | `step3p5_util.fused_qknorm_rope_forward_impl` | Defined and exact-key consumed | `75/75` rows through the approved source-hash-bounded annotation overlay |
| Full-MFA context/generation Attention | Optimus FA4 hd512 | Defined and exact-key consumed | `29` context + `71` generation rows |
| SWA context/generation Attention | native vLLM sliding GQA hd128/window512 | Defined and exact-key consumed | `29` context + `96` generation rows |
| Full-MFA grouped `wo_a` | pinned `torch.einsum` expression | Defined and exact-key consumed | `75/75` rows |
| FP32 router | `torch.ops.vllm.optimus_matmul_fp32` | Defined and exact-key consumed | `75/75` rows |
| Optimus routed MoE | pinned masked/contiguous Optimus custom ops | Defined and exact-key consumed | `174/174` rows |
| DeepEP HT dispatch/combine | vLLM DeepEP manager/Buffer path | Defined and fail-fast exact-key consumed | `0/116`, owner-deferred |
| Generic BF16 GEMM/ElementWise/Embedding | existing AIC families mapped from pinned graph | Reused where identity is faithful | Existing B300 vLLM data/analytic behavior |

The combined canonical QKV table contains `150/150` unique physical keys,
passes `150/150` unchanged AIC silicon queries, and has `0.0 ms` maximum
absolute query error. The SWA compatibility overlay only resolves
`reload_from` and `delay_w_load` from postponed strings to the installed
`cutlass.Constexpr` object after checking the exact QKNorm source SHA256. It
does not replace the pinned vLLM call path, provider, kernel body, shapes,
dtypes, arguments, QKV arithmetic, or persisted operation identity.

## Top-Level Graph

| Logical area | Required AIC family | Source |
|---|---|---|
| Input embedding | `Embedding` | `vllm/model_executor/models/step3p5.py:1032-1088` |
| 78 decoder layers | ordered logical ops below | `step3p5.py:966-1002`, `step4pro.py:490-623` |
| Final norm | `ElementWise` | `step4pro.py:626-638`, `step3p5.py:1597-1602` |
| LM head | BF16 `GEMM` | `step3p5.py:1597-1602` |
| Pipeline transfer | `P2P` only when PP > 1 | inherited model/PP path |

## Full MFA Layers

Twenty layers use Q=64, KV=1, head dimension 512, Q LoRA 2048, O groups 8,
O LoRA 1024, and head-wise gating.

| Order | Logical operation | Shape/identity | Source |
|---:|---|---|---|
| 1 | attention RMSNorm | hidden 7168, FP32 norm path | `step3p5.py:973-975`, `step4pro.py:498-503` |
| 2 | `wq_a` BF16 GEMM | `7168 -> 2048` | `step4pro.py:161-181,273-274` |
| 3 | Q-latent RMSNorm | width 2048 | `step4pro.py:169-181,274` |
| 4 | `wq_b` BF16 GEMM | `2048 -> 64*512=32768` | `step4pro.py:182-189,275-277` |
| 5 | per-head Q RMSNorm | 64 heads × 512 | `step4pro.py:277` |
| 6 | shared `wkv` BF16 GEMM | `7168 -> 512` | `step4pro.py:190-197,278-279` |
| 7 | K RMSNorm and tail RoPE | head dim 512, RoPE dim 64 | `step4pro.py:223-236,250-281` |
| 8 | prefill/decode Attention | Q64/KV1/hd512/window0; K and V alias | `step4pro.py:239-248,282-285` |
| 9 | optional inverse RoPE | attention output | `step4pro.py:286-287` |
| 10 | head-gate BF16 GEMM | `7168 -> 64` | `step4pro.py:215-222,288` |
| 11 | sigmoid and multiply | per head | `step4pro.py:288-289` |
| 12 | grouped `wo_a` projection | 8 groups, each `4096 -> 1024` | `step4pro.py:290-295` |
| 13 | `wo_b` BF16 GEMM | `8192 -> 7168` | `step4pro.py:207-214,296-297` |

The hd512 Attention provider is Optimus FA4 through the FlashAttention adapter.
It requires `step-optimus==3.23.24`, page size 128, paged KV, and K/V storage
aliasing: `vllm/v1/attention/backends/optimus_fa4.py:16-18,71-209`.

## SWA GQA Layers

Fifty-eight layers use Q=128, KV=8, head dimension 128, window 512, V RMSNorm,
and head-wise gating.

| Order | Logical operation | Shape/identity | Source |
|---:|---|---|---|
| 1 | attention RMSNorm | hidden 7168 | `step3p5.py:973-975` |
| 2 | packed QKV BF16 GEMM | `7168 -> 16384+1024+1024=18432` | `step3p5.py:394-402,511-513` |
| 3 | Q/K/V norm and RoPE | Q128, KV8, hd128 | `step3p5.py:514-563`, `step4pro.py:59-90` |
| 4 | prefill/decode Attention | Q128/KV8/hd128/window512 | `step3p5.py:454-464,563-564` |
| 5 | head-gate BF16 GEMM | `7168 -> 128` under pinned defaults | `step3p5.py:440-448,565-573` |
| 6 | sigmoid and multiply | per head | `step3p5.py:565-573` |
| 7 | output BF16 GEMM | `16384 -> 7168` | `step3p5.py:403-409,574-580` |

`VLLM_STEP3P5_ENABLE_QKVG_PROJ` defaults to false and is not enabled by the
pinned scripts, so the baseline inventory keeps QKV and gate projections
separate.

## Dense FFN Layers

Layers 0 and 1 use:

1. post-attention RMSNorm;
2. BF16 merged gate/up GEMM `7168 -> 52224`;
3. SiTU-GLU activation;
4. BF16 down GEMM `26112 -> 7168`;
5. residual add.

Source: `vllm/model_executor/models/step3p5.py:229-286,860-877,987-1002`.

## Latent MoE Layers

Layers 2 through 77 use:

| Order | Logical operation | Shape/identity | Source |
|---:|---|---|---|
| 1 | post-attention RMSNorm | hidden 7168 | `step3p5.py:987-990` |
| 2 | FP32 router GEMM | `7168 -> 896` | `step4pro.py:359-376,464` |
| 3 | BF16 routed pre-projection | `7168 -> 3584` | `step4pro.py:378-387,424-439` |
| 4 | top-k routing and DeepEP dispatch | topk16, EP16/EP32 | `step4pro.py:424-447`; DeepEP prepare/finalize |
| 5 | Optimus FP8 routed experts | hidden3584, inter3584, E896, topk16 | `optimus_fp8_moe.py:286-650` |
| 6 | post-projection RMSNorm | latent width 3584 | `step4pro.py:396-405,478-480` |
| 7 | BF16 routed post-projection | `3584 -> 7168` | `step4pro.py:388-395,481` |
| 8 | BF16 shared expert | gate/up `7168 -> 7168`, down `3584 -> 7168` | `step4pro.py:407-418,469` |
| 9 | scale and add | shared scale from config | `step4pro.py:448,482` |

The routed path internally contains FP8 activation quantization, permute,
grouped DeepGEMM, SiTU-GLU, requantization, second grouped DeepGEMM, and
unpermute/reduce. These are kernel-provider details under the logical `MoE`
operation, not separate model-level layers.

The source calls routed experts before the explicit shared expert. The latest
AIC definition must not assume generation overlap unless a runtime trace proves
overlap for the pinned image.

## Communication

| Communication | Logical AIC family | Source/status |
|---|---|---|
| Expert dispatch | `MoEDispatch` pre-op with vLLM DeepEP HT identity | DeepEP prepare; requires a vLLM-specific DeepEP dataset |
| Expert combine | `MoEDispatch` post-op with vLLM DeepEP HT identity | DeepEP finalize; queried separately to avoid double counting |
| TP all-reduce | `CustomAllReduce` when TP > 1 | `step4pro.py:471-476`; target contract uses TP1 |
| Sequence-parallel all-gather | communication op when enabled | `step4pro.py:484-486,622`; Step4Pro disables SP in pinned source |
| Attention output reduction | none for target TP1 | Full MFA explicitly rejects TP > 1 |
| PP transfer | `P2P` when PP > 1 | target first version uses PP1 |

## Prefill and Decode Differences

- The model-level operation order is shared.
- Prefill uses `ContextAttention`; decode uses `GenerationAttention`.
- Optimus MoE chooses masked versus contiguous kernels by token count, not by a
  direct phase flag.
- DeepEP mixed mode can select HT for prefill and LL/two-stage for a DP-wide
  decode-only step. The pinned two-node script defaults to
  `deepep_high_throughput`, so the pinned task contract uses HT for prefill and
  decode. LL/two-stage is outside this pinned runtime identity unless observed
  during runtime validation.

## DeepEP Consumer Contract

The existing vLLM `MoEDispatch` path does not query WideEP/DeepEP rows. A
vLLM-specific Collector route is required, but no new registry dataclass is
needed.

Required persisted identity:

```text
(mode, operation, ep_size, ep_ranks_per_node, hidden_size,
 num_experts, topk, tokens_per_dp_rank, dispatch_format,
 num_sms, max_tokens_per_rank)
```

`operation` distinguishes `dispatch` from `combine`; only
`tokens_per_dp_rank` may be interpolated. This preserves the requirements'
separate dispatch/combine reporting and prevents the existing combined loader
from charging both phases twice.

Evidence:
`/data/ycfeng/tmp/step4_latest_vllm_deepep_contract.txt`.

## MTP1 Gap

The requirements define one MTP layer with SWA Attention, Dense FFN, and an
extra LM head. The pinned `step4pro.py` has no native Step4Pro MTP class or
registry path. Existing AIC `nextn` scaling is not evidence that the required
raw `T1(B)` operation graph is represented. MTP1 needs an explicit design,
tests, and runtime validation before it can be accepted.

## Evidence Artifacts

- `/data/ycfeng/tmp/step4_latest_vllm_ops.txt`
- `/data/ycfeng/tmp/step4_latest_consumer_contract.txt`
- `/data/ycfeng/tmp/step4_latest_scope_gate.txt`
