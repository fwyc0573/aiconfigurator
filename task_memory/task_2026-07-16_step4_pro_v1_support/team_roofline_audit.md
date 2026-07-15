# Step4-Pro-V1 Roofline Audit

## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Added pre-implementation Step4-Pro-V1 geometry, formula applicability, provenance-risk, and numeric-assertion audit. |

## Summary

**Architectural status: WATCH.** Step4-Pro-V1 can safely reuse the current Step4 granular graph only as an explicitly borrowed DeepSeek-V3-style MLA approximation. The minimal supported change is to derive the projection GEMM dimensions now hard-coded as 2112, 24576, and 32768 from Step4Config while leaving the lower-level ContextMLA, GenerationMLA, and MLABmm formulas clearly marked as fixed borrowed geometry.

The graph is fully executable in DatabaseMode.SOL. DatabaseMode.SOL_FULL currently passes Task validation but cannot execute the operation graph because database queries return a (sol_time, sol_math, sol_mem) tuple while operation wrappers require PerformanceResult and immediately call float(result) and result.energy. Do not claim end-to-end SOL_FULL Task support without a separately approved cross-cutting operation-contract refactor.

## Evidence and Inference Boundary

### Evidence

- The CSV is authoritative for H=6144, layers=80, Full=20, SWA/non-Full=60, dense=4, MoE=76, dense intermediate=16384, routed experts=512, top-k=8, routed/shared intermediate=2048, and vocab=128896: task_memory/task_2026-07-16_step4_pro_v1_support/plan.md:19-25.
- Missing attention, MTP, quantization, KV-cache, and latent-MLA details must inherit the documented Step4 treatment and remain human-update items: requirements.md:11-24 and plan.md:20-25.
- Step4Config explicitly describes block_types as normalized audited classes rather than independently verified checkpoint layer order: src/aiconfigurator/sdk/common.py:252-274.
- The current temporary attention graph uses self._num_heads for both Full and SWA groups and does not consume full_num_attention_heads, sliding_num_attention_heads, their KV-head fields, or sliding_window_size: src/aiconfigurator/sdk/models/step4.py:129-206.
- The authoritative CSV has SHA256 f76fca03fd10eb145a04ff9ed906cdbd52beee37609103f9c99006e2bbf1920b, 6423 bytes, and 44 lines.

### Inference

- The normalized sequence 4 dense_swa + 20 moe_full + 56 moe_swa is aggregate-compatible with all audited counts, but is not evidence of checkpoint layer-by-layer order.
- Full=64 heads and SWA=96 heads are provenance/audit fields only under the temporary MLA path. The roofline kernel geometry remains the borrowed 128-head Step4 geometry.
- A CSV-faithful sequence-aware SWA KV-cache model cannot be inferred from the current inputs; implementing one is separate scope.

## Operation Graph and Formula Applicability

The graph construction is at src/aiconfigurator/sdk/models/step4.py:395-450. A recursive SOL test covers every top-level operation and every OverlapOp child across context and generation, verifies PerformanceResult.source == sol, and prevents profiling-data loads: tests/unit/sdk/database/test_step4_roofline.py:248-326.

| Phase / group | Operations | Geometry or scale for Pro-V1 | SOL | SOL_FULL Task execution |
|---|---|---|---|---|
| Context input | Embedding, CustomAllReduce | vocab/tp by H; scale 1 | Supported and formula-only | Blocked at Embedding float(tuple) |
| Context Full/SWA attention | ElementWise norm, downscale GEMM, q_b GEMM, kv_b GEMM, ContextMLA, output GEMM, CustomAllReduce | counts 20/60; TP1 GEMMs (2112,6144), (24576,1536), (32768,512), (6144,16384) | Supported and formula-only | Wrappers require PerformanceResult; tuple contract is incompatible |
| Context dense FFN | ElementWise norm, gate/up GEMM, SwiGLU, down GEMM, CustomAllReduce | count 4; TP1 GEMMs (32768,6144), (6144,16384) | Supported | Same tuple incompatibility for memory/GEMM/communication queries |
| Context MoE/shared | ElementWise norm, router GEMM, pre-dispatch, MoE, post-dispatch, shared gate/up, SwiGLU, shared down, CustomAllReduce, merge | count 76; router (512,6144); shared GEMMs (4096,6144), (6144,2048) | Supported, including direct SOL MoE | Direct database SOL_FULL MoE tuple is valid, but operation wrappers cannot aggregate it |
| Context output | logits GEMM, P2P | logits (128896/tp,6144); PP=1 P2P no-op | Supported | No-op P2P is safe, but full graph is already blocked |
| Generation input | Embedding, CustomAllReduce | scale mtp_scale | Supported | Blocked by tuple contract |
| Generation Full/SWA attention | norm, downscale, q_b, MLABmm pre, GenerationMLA, MLABmm post, output projection, CustomAllReduce | counts 20/60 multiplied by mtp_scale | Supported; 16 attention components observed in smoke run | Wrappers at mla.py:324-327, 498-502, and 675-679 require PerformanceResult |
| Generation dense FFN | norm, gate/up, SwiGLU, down, CustomAllReduce | count 4 multiplied by mtp_scale | Supported | Tuple incompatibility |
| Generation MoE/shared | norm, OverlapOp containing router/dispatch/MoE and shared FFN, merge | count 76 multiplied recursively by mtp_scale | Supported; overlap returns max latency and sums energy | Child wrappers fail before OverlapOp can aggregate SOL_FULL tuples |
| Generation output | logits GEMM, P2P | scale mtp_scale | Supported | Tuple incompatibility for non-noop queries |

Relevant wrapper evidence: Embedding at src/aiconfigurator/sdk/operations/embedding.py:48-62; ElementWise at elementwise.py:48-66; GEMM at gemm.py:779-803; ContextMLA/GenerationMLA/MLABmm at mla.py:302-328, 490-503, 668-680; MoE at moe.py:953-979; CustomAllReduce/P2P at communication.py:250-267 and 589-607; OverlapOp at overlap.py:146-169.

## Hard-Coded Geometry

### Required minimal refactor in Step4Model

At src/aiconfigurator/sdk/models/step4.py:129-206:

1. Replace 2112 with q_lora_rank + kv_lora_rank + qk_rope_head_dim.
2. Replace 24576 with num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim).
3. Replace 32768 with num_attention_heads * (qk_nope_head_dim + v_head_dim).
4. Preserve the already config-derived output projection width num_attention_heads * v_head_dim.
5. Add fail-fast divisibility checks before integer TP sharding; do not silently truncate.

For borrowed values 1536, 512, 128, 64, 128, and 128 heads, these formulas remain exactly 2112, 24576, 32768, and 16384, so original Step4 behavior is unchanged.

### Fixed lower-level MLA formulas that remain approximation

- ContextMLA uses fixed 192+128 in compute and memory formulas: src/aiconfigurator/sdk/operations/mla.py:217-227.
- GenerationMLA uses fixed 1088 and latent-cache width 576: mla.py:419-431.
- MLABmm uses fixed 128x512 and 640=128+512: mla.py:580-588.

Making config changes drive these formulas requires new operation parameters and test-contract changes across three operation classes. That is a larger refactor and is not part of the minimal safe change.

## Exact Numeric Assertions

### Architecture and composition

- 4 + 20 + 56 = 80 layers.
- Full attention 20 + SWA attention 60 = 80.
- MoE 20 + 56 = 76; dense + MoE = 4 + 76 = 80.
- Exact normalized block tuple: 4 dense_swa, 20 moe_full, 56 moe_swa.

### Parameter arithmetic

| Metric | Exact value |
|---|---:|
| Dense per layer: 6144 x 16384 x 3 | 301,989,888 |
| Four dense layers | 1,207,959,552 |
| Active MoE per layer | 342,884,352 |
| All MoE parameters per layer | 19,368,247,296 |
| All 76 MoE layers | 1,471,986,794,496 |
| Attention total | 15,896,603,520 |
| RMS total: 6144 x 80 x 2 | 983,040 |
| Total without input/output embeddings | 1,489,092,340,608 |
| Input + output embeddings | 1,583,874,048 |
| Total with embeddings | 1,490,676,214,656 |
| Total activation | 43,164,756,864 |

MoE formulas:
- Active = 6144 x 512 + 6144 x (8 x 2048 + 2048) x 3 = 342,884,352.
- All = 6144 x 512 + 6144 x (512 x 2048 + 2048) x 3 = 19,368,247,296.

Attention total:
20 x 153,095,232 + 60 x 213,911,648 = 15,896,603,520.

### Standard-GQA discrepancy that must remain visible

| Group | Standard borrowed GQA | CSV | Absolute gap | Relative gap |
|---|---:|---:|---:|---:|
| Full | 113,246,208 | 153,095,232 | 39,849,024 | 26.0289125137% |
| SWA | 163,577,856 | 213,911,648 | 50,333,792 | 23.5301782164% |

Do not add projection operations or a scale factor to close these gaps.

### Projection and sharding geometry

TP=1:
- downscale (n,k) = (2112,6144)
- q_b = (24576,1536)
- kv_b = (32768,512)
- output = (6144,16384)
- dense gate/up = (32768,6144), dense down = (6144,16384)
- shared gate/up = (4096,6144), shared down = (6144,2048)
- router = (512,6144), logits = (128896,6144)

TP=4:
- q_b local n=6144, kv_b local n=8192, local MLA heads=32, output local k=4096
- dense gate/up local n=8192, dense down local k=4096
- shared gate/up local n=1024, shared down local k=512

### Scale factors

For nextn=0: Full=20, SWA=60, dense=4, MoE/shared=76, embedding/logits=1.

For nextn=3:
- mtp_scale = 0.492874109263658
- Full = 9.857482185273161
- SWA = 29.572446555819482
- dense = 1.971496437054632
- MoE/shared = 37.458432304038013

Context leaves must remain unchanged. Every generation leaf, including both OverlapOp groups, must be multiplied by the same mtp_scale. Existing recursive precedent is tests/unit/sdk/models/test_step4.py:553-579.

## Source and Provenance Risks

### Important: local cache identity

DefaultHFModels currently contains only stepfun-ai/Step4 at src/aiconfigurator/sdk/common.py:520-521. Add stepfun-ai/Step4-Pro-V1; otherwise utils.py:1282-1289 performs a HuggingFace download. Test by monkeypatching the download helper to fail.

### Important: Full/SWA fields are audit-only under temporary MLA

The parser stores Full/SWA head counts and sliding_window_size at utils.py:837-852, but the graph uses borrowed self._num_heads for both groups at step4.py:151-162 and 190-202. Documentation must not call this a faithful Full-GQA/SWA roofline.

### Important: parser boundaries

The parser validates positive scalar fields, block length, labels, and topk <= experts at utils.py:807-852. It does not guarantee presence of each block class or TP divisibility. Add exact assertions for the Pro-V1 cached configuration and generic fail-fast divisibility invariants. Do not impose global 4/20/56 constraints on every future Step4-family model.

### Watch: KV-cache mismatch

Step4Model returns layers x (kv_lora_rank + qk_rope_head_dim) at step4.py:117-120 and BaseModel multiplies it linearly by sequence length at base.py:236-239.

For Pro-V1 borrowed geometry:
- elements/token = 80 x (512+64) = 46,080
- at T=1,048,576 and FP8: 48,318,382,080 bytes = 48.31838208 decimal GB
- CSV = 10.7 GB
- absolute gap = 37.61838208 GB
- ratio = 4.515736642991
- overestimate = 351.5736642991%

Do not introduce a calibration factor. A Full/SWA/window-aware KV-cache model is separate scope and requires authoritative semantics.

### Blocking if overstated: SOL_FULL

Task validation permits SOL_FULL at src/aiconfigurator/sdk/task_v2.py:1151-1158, and the existing test verifies only validation at tests/unit/sdk/database/test_step4_roofline.py:70-88. The direct GEMM database path returns a triple at gemm.py:499-502, but the operation wrapper calls float(result) at gemm.py:799-803. Fresh execution fails first at Embedding with TypeError: float() argument must be a string or a real number, not tuple.

Safe delivery boundary:
- execute complete graphs with SOL;
- audit SOL_FULL components through direct PerfDatabase calls and assert sol_time == max(sol_math, sol_mem);
- do not claim Task(database_mode=SOL_FULL).run* support.

## Minimal Recommended Change Set

1. Add the package-local Step4-Pro-V1 config and DefaultHFModels identity with exact CSV fields and explicit borrowed-value annotations.
2. Replace only Step4Model projection constants 2112, 24576, and 32768 with config-derived formulas; add fail-fast TP divisibility checks.
3. Preserve Full/SWA labels and counts while documenting that both use the borrowed 128-head MLA approximation.
4. Add exact architecture, parameter, geometry, TP-sharding, scale, offline-cache, and malformed-config tests.
5. Run end-to-end graphs in SOL. Use direct database SOL_FULL tuple assertions until a separately approved operation-contract refactor exists.
6. Record the 48.31838208 GB versus 10.7 GB KV-cache discrepancy as a human-update item; do not patch with a scale factor.
7. Do not change ContextMLA, GenerationMLA, or MLABmm APIs in this task.

## Verification Evidence

Environment:
- /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python
- Python 3.11.15
- PYTHONPATH=$PWD/src:$PWD
- MPLBACKEND=Agg

Results:
- PASS: SOL aggregate smoke at tp=8, pp=1, dp=1, moe_tp=8, moe_ep=1, batch=2, ctx_tokens=128. SinglePointEvaluation contained 49 mix_step operations and 16 generation attention components; all generation attention sources were sol; observed component latency range 0.000157013333333 to 0.1244438 ms.
- PASS: pytest -p no:cacheprovider tests/unit/sdk/models/test_step4.py tests/unit/sdk/database/test_step4_roofline.py -q — 89 passed in 7.59s.
- PASS: Ruff check --no-cache and Ruff format --check on seven audited source/test files — all checks passed; seven files already formatted.
- PASS: exact arithmetic assertions and CSV SHA/size/line checks.
- PASS: git diff --check and git status --porcelain — no output; read-only worker changed no repository files.
- EXPECTED CONTRACT FAILURE CONFIRMED: SOL_FULL complete-graph smoke raised TypeError for float(tuple).
- Initial SOL tp=1 smoke was memory-infeasible by design; rerunning with the tested feasible tp=8/moe_tp=8 configuration passed.

Subagent skip reason: this was one tightly coupled, read-only geometry and roofline contract audit; serial inspection preserved a single evidence chain and the architect runtime prohibited delegating implementation or artifact writes.
