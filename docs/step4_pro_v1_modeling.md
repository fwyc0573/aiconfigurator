## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Added the Step4-Pro-V1 structure, formula-roofline boundary, provenance register, and human-update items. |
| 2026-07-16 | Added measured CLI output and the generic naive-weight-estimator mismatch boundary. |
| 2026-07-16 | Replaced the Pro temporary-MLA description with the reviewed per-layer full/HCA graph, parameter validation, differentiated KV formulas, and unresolved KV-target evidence. |
| 2026-07-19 | Replaced the deferred tuple failure with explicit Task-level rejection of diagnostic-only `SOL_FULL`, while preserving direct roofline tuple audits. |

# Step4-Pro-V1 Modeling in AIConfigurator

## 1. Scope and Safety Boundary

AIConfigurator recognizes the exact cached model ID:

```text
stepfun-ai/Step4-Pro-V1
```

The implemented support boundary is:

- package-local, offline model discovery;
- exact CSV-backed trunk composition with 80 ordered layer identities;
- independent nested full-attention and non-full HCA geometry;
- 20 explicit standard-attention subgraphs and 60 explicit HCA subgraphs per phase;
- fail-fast parameter validation against both CSV attention targets;
- sequence-aware full/SWA/HCA KV byte formulas and nonlinear capacity inversion;
- aggregate and disaggregate complete-graph execution in `DatabaseMode.SOL`;
- direct `PerfDatabase` component audit in `DatabaseMode.SOL_FULL`;
- formula-only operation queries with explicit `source="sol"` evidence;
- CLI `estimate` in `SOL` and CLI `generate` cached-identity smoke coverage.

This support does **not** claim:

- measured or silicon-calibrated Step4-Pro-V1 latency;
- that the reviewed standard-MHA full-attention candidate is the checkpoint's
  authoritative runtime cache representation;
- closure of the CSV `10.7 GB` KV target;
- end-to-end Task execution in `SOL_FULL`;
- AFD compatibility;
- an exact support-matrix silicon row;
- that the naive eight-GPU `generate` output fits memory or is production-ready.

No scaling factor, silent default, empirical fallback, hidden residual parameter,
or Step4 latent rank is used to close missing source data. The old Pro
`*_mla_approx*` graph is retained in this document only as explicitly
superseded historical evidence; the original `stepfun-ai/Step4` MLA graph
remains unchanged.

## 2. Authoritative Source and Precedence

The architecture source is:

```text
/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main - latest.csv
SHA256: f76fca03fd10eb145a04ff9ed906cdbd52beee37609103f9c99006e2bbf1920b
Size: 6,423 bytes
```

Precedence is strict:

1. Values explicitly present in the CSV are authoritative.
2. Exact arithmetic derived from those values is authoritative when the formula is stated.
3. The independently reviewed full/HCA candidate defines the current formula-only
   execution contract where the CSV omits operation-level geometry.
4. Generic top-level attention metadata is informational; execution consumes the
   nested `full_attention` and `nonfull_attention` sections only.
5. A candidate or borrowed value never overrides a conflicting CSV value.
6. A mismatch remains visible as a human-update item; it is not calibrated away.

The package-local representation is:

```text
src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json
```

## 3. CSV-Backed Structure

| Field | CSV value | AIC field or representation |
|---|---:|---|
| Hidden size | 6,144 | `hidden_size` |
| Main/trunk layers | 80 | `num_hidden_layers` |
| Full-attention layers | 20 | layer IDs `4-23`, `attention_type="full"` |
| Non-Full/HCA layers | 60 | layer IDs `0-3, 24-79`, `attention_type="nonfull"` |
| Dense FFN layers | 4 | layer IDs `0-3`, `ffn_type="dense"` |
| MoE layers | 76 | layer IDs `4-79`, `ffn_type="moe"` |
| Dense intermediate size | 16,384 | `intermediate_size` |
| Routed experts | 512 | `n_routed_experts` |
| Routed experts selected per token | 8 | `num_experts_per_tok` |
| Routed-expert intermediate size | 2,048 | `moe_intermediate_size` |
| Shared-expert intermediate size | 2,048 | `shared_expert_intermediate_size` |
| Vocabulary size | 128,896 | `vocab_size` |
| Full query heads | 64 | `full_attention.num_query_heads` |
| Non-full query heads | 96 | `nonfull_attention.num_query_heads` |

The executable layer sequence is exact and ordered:

```python
(
    Step4LayerSpec(layer_id=0..3, attention_type="nonfull", ffn_type="dense"),
    Step4LayerSpec(layer_id=4..23, attention_type="full", ffn_type="moe"),
    Step4LayerSpec(layer_id=24..79, attention_type="nonfull", ffn_type="moe"),
)
```

The three contiguous regions mean:

| Layer IDs | Attention group | FFN group | Count |
|---|---|---|---:|
| `0-3` | non-full HCA | dense SwiGLU FFN | 4 |
| `4-23` | standard full attention | routed MoE + shared expert | 20 |
| `24-79` | non-full HCA | routed MoE + shared expert | 56 |

Therefore:

```text
dense = 4
Full  = 20
HCA   = 4 + 56 = 60
MoE   = 20 + 56 = 76
total = 4 + 20 + 56 = 80
```

The earlier `dense_swa`/`moe_full`/`moe_swa` `block_types` sequence encoded
the same counts but had no explicit layer identity and routed both attention
labels through aggregate MLA. It is a superseded parser representation for
the Pro ID and is not accepted alongside the new schema.

## 4. Exact Derived Arithmetic

### 4.1 Dense and MoE quantities

| Quantity | Formula | Exact value |
|---|---|---:|
| Dense parameters per dense layer | `3 × 6144 × 16384` | 301,989,888 |
| All four dense layers | `4 × 301,989,888` | 1,207,959,552 |
| Router parameters per MoE layer | `6144 × 512` | 3,145,728 |
| Active routed-expert parameters per token/layer | `3 × 6144 × 2048 × 8` | 301,989,888 |
| Active shared-expert parameters per token/layer | `3 × 6144 × 2048` | 37,748,736 |
| Active MoE total per token/layer | router + active routed + shared | 342,884,352 |
| All MoE parameters per layer | `6144 × 512 + 3 × 6144 × (512 × 2048 + 2048)` | 19,368,247,296 |
| All 76 MoE layers | `76 × 19,368,247,296` | 1,471,986,794,496 |

The factor `3` is the gated FFN/SwiGLU weight structure: gate, up, and down projections.

### 4.2 Model totals

| Quantity | Formula or source | Exact value |
|---|---|---:|
| Attention total | `20 × 153,095,232 + 60 × 213,911,648` | 15,896,603,520 |
| RMS total | `6144 × 80 × 2` | 983,040 |
| Total without embeddings | CSV total | 1,489,092,340,608 |
| Input + output embeddings | `2 × 128896 × 6144` | 1,583,874,048 |
| Total with embeddings | sum | 1,490,676,214,656 |
| Total activation | CSV total | 43,164,756,864 |

These totals validate architecture accounting. They do not independently define the missing attention projection graph.

## 5. Reviewed Attention Inputs and Remaining Borrowing

The CSV does not provide an operation-level projection recipe or the runtime
full-attention cache representation. The current formula-only candidate was
therefore reviewed independently before implementation and is represented by
two mutually exclusive nested configs:

| Config | Executable fields | Current value |
|---|---|---|
| Full | `num_query_heads`, `num_kv_heads` | `64`, `64` |
| Full | `q_head_dim`, `k_head_dim`, `v_head_dim` | `96`, `96`, `96` |
| Full | Q/K/V/O projection modes | all `linear` |
| Full | `latent_rank` | `null`; latent full attention is rejected |
| Full | target parameter count | `153,095,232` |
| Non-full | mechanism | `hca` |
| Non-full | query heads, Q/O ranks, O groups | `96`, `1024`, `1024`, `16` |
| Non-full | compressed head / RoPE dimensions | `512`, `64` |
| Non-full | window / compression ratio | `512`, `128` |
| Non-full | indexer heads/dimension/top-k | `0`, `0`, `0` |
| Non-full | target parameter count | `213,911,648` |

All three named unknown parameter fields are explicit and equal to zero in
both configs:

```text
unknown_extra_projection_params = 0
unknown_router_params           = 0
unknown_compression_params      = 0
```

They are disclosure fields, not calibration knobs. The generic top-level
`num_attention_heads=64` and `num_key_value_heads=64` remain model metadata;
the Pro execution path validates and consumes only the nested fields.

The remaining family-level values are still provenance-visible rather than
CSV-derived:

| Field | Current value | Current use | Provenance status |
|---|---:|---|---|
| `architectures` / `model_type` | `Step4ForCausalLM` / `step4` | parser/model family reuse | not a measured property |
| `num_nextn_predict_layers` | 3 | checkpoint MTP default; caller may set `nextn` | requires checkpoint confirmation |
| `quant_algo` | `fp8` | GEMM and MoE coefficients | requires checkpoint confirmation |
| `kv_cache_quant_algo` | `fp8` | KV bytes/element | requires checkpoint confirmation |
| `torch_dtype` | `bfloat16` | non-quantized/FMHA dtype | requires checkpoint confirmation |
| `max_position_embeddings` | 1,048,576 | cached limit metadata | requires deployment confirmation |

## 6. Per-Layer Attention and Parameter Validation

### 6.1 Full standard-attention candidate

With `H=6144`, `64` Q/KV heads, and `96` elements per Q/K/V head,
each projection width is `64 × 96 = 6144`. The four trainable matrices are:

| Matrix | Formula | Elements |
|---|---|---:|
| Q | `6144 × 64 × 96` | 37,748,736 |
| K | `6144 × 64 × 96` | 37,748,736 |
| V | `6144 × 64 × 96` | 37,748,736 |
| O | `64 × 96 × 6144` | 37,748,736 |
| **Estimate** | sum | **150,994,944** |

Against the CSV target `153,095,232`, the absolute error is `2,100,288`
and the relative error is `1.3718833517%`: `PASS` under the explicit 5% gate.

### 6.2 Non-full HCA candidate

The six trainable matrix groups are:

| Matrix group | Formula | Elements |
|---|---|---:|
| Q down projection | `6144 × 1024` | 6,291,456 |
| Q up projection | `1024 × 96 × 512` | 50,331,648 |
| Main compressor | `6144 × 512` | 3,145,728 |
| Attention output to O rank | `96 × 512 × 1024` | 50,331,648 |
| Grouped O projection | `16 × 1024 × 6144` | 100,663,296 |
| Compressor K/V projections | `2 × 6144 × 512` | 6,291,456 |
| **Estimate** | sum | **217,055,232** |

Against the CSV target `213,911,648`, the absolute error is `3,143,584`
and the relative error is `1.4695712129%`: `PASS` under the same gate.
The DeepSeek-V4 analytic module also owns `96 + 128 × 512 = 65,632`
FP32 resident-state elements. They are reported separately and are not mixed
into the trainable parameter estimate.

Both parameter checks run before any operation construction. An error above
5% raises `ValueError`; no partial graph or fallback is created.

### 6.3 Explicit operation graph

Each phase contains exactly:

```text
20 full layers × 7 top-level operations
+ 60 HCA layers × 3 top-level operations
= 320 unique attention operations
```

A full layer such as layer 4 is:

```text
<phase>_layer_004_full_attn_norm
<phase>_layer_004_full_q_proj_gemm
<phase>_layer_004_full_k_proj_gemm
<phase>_layer_004_full_v_proj_gemm
<phase>_layer_004_full_attention
<phase>_layer_004_full_o_proj_gemm
<phase>_layer_004_full_attention_ar
```

An HCA layer such as layer 24 is:

```text
<phase>_layer_024_nonfull_attn_norm
<phase>_layer_024_nonfull_hca_attention
<phase>_layer_024_nonfull_attention_ar
```

`<phase>` is `context` or `generation`. Full uses standard
`ContextAttention`/`GenerationAttention`; HCA uses the analytic DeepSeek-V4
module. Both paths have an explicit `CustomAllReduce`. Generation applies the
existing MTP factor exactly once to every layer; context scale is exactly `1`.

### 6.4 Superseded temporary-MLA baseline

The earlier Pro graph aggregated one Step4 MLA subgraph at scale `20` and a
second at scale `60`. Its global widths `2,112 / 24,576 / 32,768 / 16,384`
and its `48.31838208 GB` uniform latent-cache estimate are historical evidence
only. No current Pro operation name contains `mla_approx`, and the Pro path no
longer reads Step4 `q_lora_rank=1536` or `kv_lora_rank=512`. The original
`stepfun-ai/Step4` class-level semantic names remain `14` context and `16`
generation entries and are covered by a separate regression.

## 7. Differentiated KV-Cache Formulas and Remaining Conflict

All values below are bytes, and `b` is the configured bytes per cache element.

### 7.1 Per-layer formulas

Full history, sharded by attention TP:

```text
KV_full(S, TP) = S × (num_kv_heads / TP)
                 × (k_head_dim + v_head_dim) × b
```

SWA-only (`compression_ratio=0`; supported by the config API but not used by
the current 60 HCA layer records):

```text
KV_swa(S) = min(S, window_size) × head_dim × b
```

HCA ratio 128:

```text
KV_hca(S) = [min(S, window_size) + floor(S / compression_ratio)]
            × head_dim × b
            + 2 × compression_ratio × head_dim × 4
```

The final term is persistent FP32 compressor state. For the current config it
is `524,288` bytes per HCA layer, including at `S=0`.

### 7.2 Exact FP8 examples

| Sequence length | Full TP1/layer | Full TP8/layer | SWA/layer | HCA/layer |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 524,288 |
| 1 | 12,288 | 1,536 | 512 | 524,800 |
| 511 | 6,279,168 | 784,896 | 261,632 | 787,456 |
| 512 | 6,291,456 | 786,432 | 262,144 | 788,480 |
| 513 | 6,303,744 | 787,968 | 262,144 | 788,480 |
| 1,048,576 | 12,884,901,888 | 1,610,612,736 | 262,144 | 4,980,736 |

Summing the explicit 20 full and 60 HCA records gives:

| Sequence length | Model TP1 bytes | Model TP8 bytes |
|---:|---:|---:|
| 0 | 31,457,280 | 31,457,280 |
| 1 | 31,733,760 | 31,518,720 |
| 511 | 172,830,720 | 62,945,280 |
| 512 | 173,137,920 | 63,037,440 |
| 513 | 173,383,680 | 63,068,160 |
| 1,048,576 | 257,996,881,920 | 32,511,098,880 |

At TP1 this is `257.99688192 GB`, versus the CSV target `10.7 GB`: an
absolute gap of `247.29688192 GB` and a ratio of `24.1118581234x`. The full
20-layer history alone is `257.69803776 GB`; the 60 HCA layers add
`0.29884416 GB`.

This conflict is structural. The parameter target is consistent with the
standard-MHA candidate, while the KV target implies an unconfirmed latent,
compressed, grouped, differently sharded, or per-rank representation. The
implementation reports the conflict at warning level and does not close it
with Step4 latent ranks or a scaling factor.

Because the total curve contains window caps, floor compression, and resident
state, `get_kvcache_elements_per_token()` is invalid for the Pro schema and
raises explicitly. `get_kvcache_max_tokens()` uses a monotonic binary-search
inverse. Original Step4 retains its linear `52,992` elements/token contract.

## 8. Operation and Roofline Contract

### 8.1 Complete graph composition

| Graph group | Operation families | Scale |
|---|---|---:|
| Input | Embedding, `CustomAllReduce` | 1 |
| Full attention | 20 independently named norm + Q/K/V GEMMs + standard Attention + O GEMM + all-reduce paths | 1 per layer |
| Non-full HCA | 60 independently named norm + DeepSeek-V4 HCA + all-reduce paths | 1 per layer |
| Dense FFN | norm, gate/up GEMM, SwiGLU, down GEMM, all-reduce | 4 |
| Routed/shared MoE | norm, router GEMM, dispatch, `MoE`, shared SwiGLU FFN, all-reduce, merge | 76 |
| Generation routed/shared path | `OverlapOp` over routed MoE and shared FFN branches | 76, then MTP scale |
| Output | logits GEMM, P2P | 1, then MTP scale for generation |

For `nextn=0`, context and generation use the exact structural scales above. For `nextn=3` with the tested acceptance rates, context remains unchanged and every generation leaf is multiplied recursively by the same MTP factor. MTP policy remains a human-update item.

### 8.2 Formula families

| Operation family | Formula-only basis | Roofline interpretation |
|---|---|---|
| GEMM, router, logits, dense/shared/projection GEMMs | matrix FLOPs and operand bytes with quant-specific system throughput/bandwidth | `max(math_time, memory_time)` |
| Embedding, RMS/ElementWise, SwiGLU, merge | bytes moved through HBM | memory roofline |
| `ContextAttention` | causal full-history Attention FLOPs plus current-token and KV bytes using local `8` heads at TP8 and head size `96` | `max(math_time, memory_time)` |
| `GenerationAttention` | single-step full-history Attention FLOPs plus KV read bytes using local `8` heads at TP8 and head size `96` | `max(math_time, memory_time)` |
| Context DeepSeek-V4 HCA | analytic Q/O/compressor, window, and ratio-128 compressed-history work | `max(math_time, memory_time)` |
| Generation DeepSeek-V4 HCA | analytic decode Q/O/compressor, window, and compressed-history work | `max(math_time, memory_time)` |
| `MoE` | gated three-projection routed-expert FLOPs and weight/activation bytes for `H=6144`, `I=2048`, `topk=8`, `experts=512` | `max(math_time, memory_time)` |
| `CustomAllReduce` | ring-transfer bytes divided by P2P bandwidth | communication-memory roofline |
| `NCCL` | collective transfer volume divided by collective bandwidth | communication-memory roofline |
| P2P | transfer bytes divided by link bandwidth; PP=1 is an explicit no-op | communication-memory roofline |
| `OverlapOp` | maximum of routed and shared branch latency; energy is accumulated | composite scheduling rule, not a new hardware roofline |

`CustomAllReduce` now reports its bandwidth-derived time as the memory component. Its direct `SOL_FULL` tuple is `(selected, 0, selected)`, matching the physical classification and the invariant below.

### 8.3 `SOL` versus `SOL_FULL`

Complete Task graphs execute in `SOL`. Every executed per-operation result is a `PerformanceResult` with:

```text
source = "sol"
```

Explicit unexecuted branches are recorded separately as zero latency with `source="not_executed"`; they are not silently treated as executed formulas. Formula-mode HCA queries return before `load_data()`, so neither `SOL` nor `SOL_FULL` reads collector files.

Direct database queries are audited in `SOL_FULL` as:

```text
(selected, math_roofline, memory_roofline)
selected == max(math_roofline, memory_roofline)
selected == scalar SOL latency
```

The direct audit covers GEMM, memory operations, context/generation standard Attention, context/generation HCA, MoE, custom all-reduce, P2P, and NCCL. On the simplified `1 TFLOPS / 1 TB/s` fixture, representative `(selected, math, memory)` times in ms are:

| Component | Selected | Math | Memory |
|---|---:|---:|---:|
| Context standard Attention | 51.539607552 | 51.539607552 | 0.037748736 |
| Generation standard Attention | 0.012585984 | 0.012579840 | 0.012585984 |
| Context HCA | 486.322733056 | 486.322733056 | 2.268364800 |
| Generation HCA | 0.121062400 | 0.121062400 | 0.047776768 |

Step4 and Step4-Pro Task execution now reject `SOL_FULL` explicitly at construction and at every public execution entry before database loading. The error directs callers to direct `PerfDatabase` diagnostic queries instead of allowing a later tuple-related `TypeError`. Direct `SOL_FULL` queries and their three-part roofline equality audits remain supported. Full graph-level detailed results would require a separate typed, cross-operation contract and are outside this support boundary.

## 9. Offline and CLI Behavior

The exact model ID is included in `DefaultHFModels`, so model discovery loads the package-local JSON and does not call HuggingFace.

Verified formula-only paths use:

```text
system=h200_sxm
backend=vllm
backend_version=0.22.0
database_mode=SOL
TP=8
PP=2
attention-DP=1
MoE-TP=8
EP=1
```

This parallel point is a representative, structurally valid formula test. It is not a memory-fit or optimal-deployment claim.

The CLI estimate path is:

```bash
aiconfigurator cli estimate \
  --model-path stepfun-ai/Step4-Pro-V1 \
  --estimate-mode agg \
  --system h200_sxm \
  --backend vllm \
  --backend-version 0.22.0 \
  --database-mode SOL \
  --isl 128 --osl 2 --batch-size 1 --ctx-tokens 128 \
  --tp 8 --pp 2 --dp 1 --etp 8 --ep 1 --nextn 0
```

The requested naive generation path is:

```bash
aiconfigurator cli generate \
  --model-path stepfun-ai/Step4-Pro-V1 \
  --total-gpus 8 \
  --system h200_sxm
```

The fresh command reported the following sizing values:

| Quantity | Observed / authoritative value |
|---|---:|
| Generic naive estimate | 1,559,313,383,424 parameters |
| Generic naive BF16 weight estimate | 2,904.447509765625 GiB |
| CSV trunk + both embeddings | 1,490,676,214,656 parameters |
| CSV-equivalent BF16 weight size | 2,776.600820302963 GiB |
| Naive-minus-CSV gap | 68,637,168,768 parameters |
| Relative gap | 4.604431739983% |
| Naive fit decision | required TP=32, available maximum TP=8, `fit=False` |

This difference is not MTP weight. The existing generic
`generator.naive._estimate_model_weight_bytes()` formula uses one embedding,
applies one `4 * H^2` attention estimate to every layer, and treats every one
of the 80 layers as the same 512-expert MoE layer. It does not consume
Step4-Pro-V1 `layers`, full/HCA geometry, shared-expert geometry, or the CSV
attention totals.
Consequently, the naive sizing result is not the authoritative model parameter
count. Correcting that generic cross-family estimator is a separate generator
refactor and is not hidden inside this model-support task.

`generate` deliberately warns that it performs no memory validation or
performance optimization. Successful artifact rendering at eight GPUs must not
be read as a model-fit or latency claim. The same run explicitly warned that the
model may not fit, selected an internal `TEP=8` strategy, and rendered a summary
whose dense `TP=1, PP=1` fields are not a complete deployment-feasibility
description.

## 10. Human-Update Register

| Item | Current treatment | Required authoritative update | Impact if unchanged |
|---|---|---|---|
| Full projection recipe | reviewed 64-head standard MHA, 96-dimensional Q/K/V, explicit Q/K/V/O matrices | Supply checkpoint Attention Detail | parameter closure passes, but checkpoint identity is not independently confirmed |
| Full runtime KV representation | full-history K/V, sharded by TP | Confirm KV heads/dimensions, cache sharding/replication, dtype, and any latent/compression stage | TP1 candidate remains 24.1118581234x the CSV model target after HCA is included |
| Non-full HCA recipe | 96 heads, ranks 1024/1024, 16 O groups, 512 window, ratio 128, no indexer | Supply checkpoint HCA detail | parameter closure passes; formula remains an independently reviewed candidate |
| HCA resident state | 65,632 FP32 elements reported separately | Confirm runtime ownership and replication | resident memory may be per-rank or differently shared |
| KV cache | explicit nonlinear full/SWA/HCA formulas; TP1 `257.99688192 GB` at 1,048,576 tokens | Supply a topology that explains the `10.7 GB` CSV target | capacity cannot be treated as target-closed |
| Superseded Pro MLA baseline | documented historical evidence only; no current Pro `mla_approx` ops | none; preserve only original Step4 MLA regression | restoring it would violate the reviewed schema and differentiated KV contract |
| Quantization | FP8 GEMM/MoE/KV and BF16 FMHA treatment | Confirm authoritative checkpoint quantization metadata | roofline throughput/byte coefficients may change |
| MTP | `num_nextn_predict_layers=3`; caller-controlled acceptance rates | Confirm Pro draft depth and acceptance policy | generation scale may change |
| Maximum context/window | 1,048,576 maximum positions and 512 HCA local window | Confirm deployment/checkpoint limits | capacity metadata may change |
| Backend support | vLLM `0.22.0` used for formula-only tests | Add measured support evidence separately if needed | no silicon-support claim exists |
| Naive generator sizing | generic 1,559,313,383,424-parameter approximation | Add a block-aware cross-family weight estimator in a separately approved generator task | generate fit/TP output is not authoritative for this model |
| AFD | explicitly excluded | Add a separate `dense_swiglu` classifier task and both phase regressions | AFD remains unsupported |
| Task-level `SOL_FULL` | Task admission rejects it; direct diagnostic queries remain supported | Design and approve a typed cross-operation detailed-result contract separately | full graph remains `SOL` only; no tuple element is selected silently |

When authoritative inputs arrive, update the existing cached config, model operations, tests, and this document together. Do not add compatibility fallbacks for obsolete assumptions.
