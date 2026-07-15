## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Added the Step4-Pro-V1 structure, formula-roofline boundary, provenance register, and human-update items. |
| 2026-07-16 | Added measured CLI output and the generic naive-weight-estimator mismatch boundary. |

# Step4-Pro-V1 Modeling in AIConfigurator

## 1. Scope and Safety Boundary

AIConfigurator recognizes the exact cached model ID:

```text
stepfun-ai/Step4-Pro-V1
```

The implemented support boundary is:

- package-local, offline model discovery;
- exact CSV-backed trunk composition;
- aggregate and disaggregate complete-graph execution in `DatabaseMode.SOL`;
- direct `PerfDatabase` component audit in `DatabaseMode.SOL_FULL`;
- formula-only operation queries with explicit `source="sol"` evidence;
- CLI `estimate` in `SOL` and CLI `generate` cached-identity smoke coverage.

This support does **not** claim:

- measured or silicon-calibrated Step4-Pro-V1 latency;
- a faithful Full-attention or SWA projection recipe;
- end-to-end Task execution in `SOL_FULL`;
- AFD compatibility;
- an exact support-matrix silicon row;
- that the naive eight-GPU `generate` output fits memory or is production-ready.

No scaling factor, silent default, empirical fallback, or fabricated operation is used to close missing source data.

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
3. Missing attention, KV-cache, MTP, and quantization details temporarily reuse the existing Step4 treatment.
4. A borrowed value never overrides a conflicting CSV value.
5. A mismatch remains visible as a human-update item; it is not calibrated away.

The package-local representation is:

```text
src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json
```

## 3. CSV-Backed Structure

| Field | CSV value | AIC field or representation |
|---|---:|---|
| Hidden size | 6,144 | `hidden_size` |
| Main/trunk layers | 80 | `num_hidden_layers` |
| Full-attention layers | 20 | `20 × "moe_full"` |
| Non-Full/SWA layers | 60 | `4 × "dense_swa" + 56 × "moe_swa"` |
| Dense FFN layers | 4 | `4 × "dense_swa"` |
| MoE layers | 76 | `20 × "moe_full" + 56 × "moe_swa"` |
| Dense intermediate size | 16,384 | `intermediate_size` |
| Routed experts | 512 | `n_routed_experts` |
| Routed experts selected per token | 8 | `num_experts_per_tok` |
| Routed-expert intermediate size | 2,048 | `moe_intermediate_size` |
| Shared-expert intermediate size | 2,048 | `shared_expert_intermediate_size` |
| Vocabulary size | 128,896 | `vocab_size` |
| Full query-head hint | 64 | audit metadata: `full_num_attention_heads` |
| SWA query-head hint | 96 | audit metadata: `sliding_num_attention_heads` |

The normalized block sequence is exact and ordered:

```python
("dense_swa",) * 4 + ("moe_full",) * 20 + ("moe_swa",) * 56
```

Each block label means:

| Block label | Attention group | FFN group | Count |
|---|---|---|---:|
| `dense_swa` | temporary SWA-labelled MLA | dense SwiGLU FFN | 4 |
| `moe_full` | temporary Full-labelled MLA | routed MoE + shared expert | 20 |
| `moe_swa` | temporary SWA-labelled MLA | routed MoE + shared expert | 56 |

Therefore:

```text
dense = 4
Full  = 20
SWA   = 4 + 56 = 60
MoE   = 20 + 56 = 76
total = 4 + 20 + 56 = 80
```

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

## 5. Temporary Step4-Borrowed Values

The CSV does not contain a complete attention-detail table, KV topology, MTP policy, or checkpoint quantization declaration. The cached config therefore carries the following temporary values from the existing Step4 treatment:

| Field | Temporary value | Current use | Provenance status |
|---|---:|---|---|
| `architectures` | `Step4ForCausalLM` | Select existing Step4 parser/model | family reuse, not a measured property |
| `model_type` | `step4` | Cached config identity | family reuse |
| `num_attention_heads` | 128 | temporary MLA projection/head geometry | Step4-borrowed |
| `num_key_value_heads` | 1 | generic cached config field | Step4-borrowed |
| `full_num_key_value_heads` | 8 | discrepancy audit only | Step4-borrowed |
| `sliding_num_key_value_heads` | 8 | discrepancy audit only | Step4-borrowed |
| `attention_head_dim` / `head_dim` | 128 | standard-GQA discrepancy audit | Step4-borrowed |
| `sliding_window_size` | 512 | provenance metadata; current temporary MLA graph is not window-faithful | Step4-borrowed |
| `q_lora_rank` | 1,536 | temporary MLA projection geometry | Step4-borrowed |
| `kv_lora_rank` | 512 | temporary MLA and KV arithmetic | Step4-borrowed |
| `qk_nope_head_dim` | 128 | temporary MLA projection geometry | Step4-borrowed |
| `qk_rope_head_dim` | 64 | temporary MLA projection and KV arithmetic | Step4-borrowed |
| `v_head_dim` | 128 | temporary MLA projection geometry | Step4-borrowed |
| `num_nextn_predict_layers` | 3 | checkpoint MTP default metadata | Step4-borrowed; caller may explicitly set `nextn` |
| `quant_algo` | `fp8` | GEMM and MoE formula coefficients | Step4-borrowed |
| `kv_cache_quant_algo` | `fp8` | KV-cache formula coefficients | Step4-borrowed |
| `torch_dtype` | `bfloat16` | non-quantized/FMHA dtype selection | Step4-borrowed |
| `max_position_embeddings` | 1,048,576 | cached limit metadata | Step4-borrowed |

These values are deliberately visible. They are not inferred from the CSV totals and must be replaced when authoritative Step4-Pro-V1 checkpoint or Attention Detail data is supplied.

## 6. Attention Mismatch and Temporary MLA Graph

### 6.1 Why standard GQA does not close

Using the CSV query-head hints with the borrowed `8` KV heads and `128` head dimension gives the standard Q/K/V/O projection count:

```text
standard_gqa = 2 × H × (query_heads × head_dim + kv_heads × head_dim)
```

| Group | Standard GQA | CSV target | Absolute gap | Relative gap |
|---|---:|---:|---:|---:|
| Full (`query_heads=64`) | 113,246,208 | 153,095,232 | 39,849,024 | 26.0289125137% |
| SWA (`query_heads=96`) | 163,577,856 | 213,911,648 | 50,333,792 | 23.5301782164% |

The CSV states that the target values were weighted from a separate Attention Detail source, but that source was not provided. The missing parameters cannot be reconstructed uniquely.

**Required interpretation:** do not add hidden projections and do not multiply the temporary graph by `CSV / standard_gqa`. Either action would fabricate structure.

### 6.2 Current temporary MLA geometry

Both Full and SWA labels currently reuse the existing Step4 MLA treatment. Projection dimensions are derived from the borrowed config instead of being hard-coded:

| Projection | Formula | Global width |
|---|---|---:|
| Downscale output | `q_lora_rank + kv_lora_rank + qk_rope_head_dim` | 2,112 |
| Q-B output | `num_attention_heads × (qk_nope_head_dim + qk_rope_head_dim)` | 24,576 |
| KV-B output | `num_attention_heads × (qk_nope_head_dim + v_head_dim)` | 32,768 |
| Attention output input | `num_attention_heads × v_head_dim` | 16,384 |

The Full-labelled graph uses scale factor `20`; the SWA-labelled graph uses scale factor `60`. The labels preserve the CSV composition, but the per-layer formula is still the borrowed 128-head MLA approximation.

Lower-level operation formulas also retain existing Step4 constants:

- `ContextMLA` uses fixed `192 + 128` terms in its compute/memory model.
- `GenerationMLA` uses fixed `1088` and latent-cache width `576`.
- `MLABmm` uses fixed `128 × 512` and `640 = 128 + 512` terms.

Changing those constants requires new operation parameters and a cross-operation API refactor. It is intentionally not hidden inside this model addition.

**Do not interpret current Full/SWA latency as measured or architecture-faithful Step4-Pro-V1 attention latency.** It is a formula-only placeholder with explicit provenance.

## 7. KV-Cache Mismatch

The current Step4 model API computes latent KV elements per token as:

```text
layers × (kv_lora_rank + qk_rope_head_dim)
= 80 × (512 + 64)
= 46,080 elements/token
```

At `T = 1,048,576` tokens and FP8 (`1 byte/element`):

```text
temporary bytes = 46,080 × 1,048,576
                = 48,318,382,080 bytes
temporary GB    = 48.31838208 decimal GB
CSV target      = 10.7 GB
absolute gap    = 37.61838208 GB
ratio           = 4.515736642991x
over target     = 351.5736642991%
```

The root cause is structural: the current formula applies full latent MLA storage to every layer and has no sequence/window-aware Full/SWA topology. A scaling factor would conceal that missing model and is prohibited.

## 8. Operation and Roofline Contract

### 8.1 Complete graph composition

| Graph group | Operation families | Scale |
|---|---|---:|
| Input | Embedding, `CustomAllReduce` | 1 |
| Full attention | ElementWise norm, projection GEMMs, `ContextMLA` or `MLABmm` + `GenerationMLA`, output GEMM, all-reduce | 20 |
| SWA attention | same temporary MLA operation family under SWA names | 60 |
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
| `ContextMLA` | borrowed Step4 attention FLOPs and bytes as a function of tokens/context | `max(math_time, memory_time)` within the temporary MLA model |
| `GenerationMLA` | borrowed Step4 decode FLOPs and latent-cache bytes | `max(math_time, memory_time)` within the temporary MLA model |
| `MLABmm` | borrowed Step4 BMM FLOPs and bytes | `max(math_time, memory_time)` |
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

Explicit unexecuted branches are recorded separately as zero latency with `source="not_executed"`; they are not silently treated as executed formulas.

Direct database queries are audited in `SOL_FULL` as:

```text
(selected, math_roofline, memory_roofline)
selected == max(math_roofline, memory_roofline)
selected == scalar SOL latency
```

The direct audit covers GEMM, memory operations, context MLA, generation MLA, MLA BMM, MoE, custom all-reduce, P2P, and NCCL.

Complete Task execution in `SOL_FULL` is intentionally not claimed. Operation wrappers currently call `float()` and access `.energy` on database results, while `SOL_FULL` returns tuples. The preserved regression expects the resulting tuple-related `TypeError`. Fixing that shared wrapper contract is a separate cross-cutting change.

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

This parallel point is a representative, memory-feasible formula test. It is not a claim of optimal deployment.

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
Step4-Pro-V1 `block_types`, shared-expert geometry, or the CSV attention totals.
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

| Item | Current temporary treatment | Required authoritative update | Impact if unchanged |
|---|---|---|---|
| Full projection recipe | Step4 128-head MLA formula, scale 20 | Supply complete Step4-Pro-V1 Full Attention Detail | Full latency is not architecture-faithful |
| SWA projection/window recipe | same Step4 MLA formula, scale 60 | Supply SWA projection, window, and cache semantics | SWA latency and cache are not window-faithful |
| KV heads and head dimensions | KV heads 8 and head dim 128 for discrepancy audit | Confirm checkpoint-specific Full/SWA K/V geometry | standard-GQA comparison may change |
| Lower-level MLA constants | existing `ContextMLA`, `GenerationMLA`, and `MLABmm` constants | Define operation-level Pro geometry and approve API refactor | formula remains a Step4 placeholder |
| KV cache | 46,080 elements/token, linear in total sequence length | Supply per-layer Full/SWA/window cache topology closing to 10.7 GB | current estimate is 4.515736642991x the CSV target |
| Quantization | FP8 GEMM/MoE/KV and BF16 FMHA treatment | Confirm authoritative checkpoint quantization metadata | roofline throughput/byte coefficients may change |
| MTP | `num_nextn_predict_layers=3`; caller-controlled acceptance rates | Confirm Pro draft depth and acceptance policy | generation scale may change |
| Maximum context/window | 1,048,576 maximum positions and 512 sliding window | Confirm deployment/checkpoint limits | capacity metadata may change |
| Backend support | vLLM `0.22.0` used for formula-only tests | Add measured support evidence separately if needed | no silicon-support claim exists |
| Naive generator sizing | generic 1,559,313,383,424-parameter approximation | Add a block-aware cross-family weight estimator in a separately approved generator task | generate fit/TP output is not authoritative for this model |
| AFD | explicitly excluded | Add a separate `dense_swiglu` classifier task and both phase regressions | AFD remains unsupported |
| Task-level `SOL_FULL` | direct queries only | Approve shared operation-wrapper result-contract refactor | full graph remains `SOL` only |

When authoritative inputs arrive, update the existing cached config, model operations, tests, and this document together. Do not add compatibility fallbacks for obsolete assumptions.
