# Step4-Pro-V1 Registration, Integration, CLI, and Test Audit

## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Added the pre-implementation registration, integration, CLI, and test audit for Step4-Pro-V1. |

## Scope

This report audits the pre-implementation support surface for model ID stepfun-ai/Step4-Pro-V1. It covers:

- package-local cached identity and configuration discovery;
- architecture/family/model-factory registration;
- Step4 graph construction and temporary MLA geometry;
- Step4-specific configuration validation;
- Task formula-only database-mode enforcement;
- recursive SOL and SOL_FULL operation behavior;
- SDK and CLI integration;
- existing Step4 regression coverage;
- AFD and support-matrix boundaries.

Production code, tests, shared task documentation, and Git state were treated as read-only. This report distinguishes verified evidence from implementation recommendations.

## Executive Outcome

Step4-Pro-V1 is not yet implementable as a safe cached model by registration alone. Four items block support:

1. The exact model ID is absent from DefaultHFModels.
2. The package-local cached configuration file does not exist.
3. Step4 temporary MLA projection shapes are encoded as literals derived from the original Step4 model rather than computed from validated configuration fields.
4. Step4-specific parsing silently accepts or substitutes malformed routed-MoE fields that the predefined graph requires.

The smallest sound implementation is to add the cached identity and config, derive the existing MLA shapes from config fields, strengthen Step4 validation, and add model-specific RED tests before production changes. No scaling factor, silent fallback, new family, or support-matrix silicon claim is justified.

## Evidence and Inference Convention

- **Evidence** means directly observed repository behavior or source code.
- **Derived** means exact arithmetic from authoritative values.
- **Recommendation** means a proposed change or test that has not yet been implemented.
- **Watch** means a limitation that must be documented or monitored but is not part of the minimum agg/disagg formula-only change set.

## Authoritative Model Facts

The authoritative CSV is:

    /data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main - latest.csv

Observed file identity:

- SHA256: f76fca03fd10eb145a04ff9ed906cdbd52beee37609103f9c99006e2bbf1920b
- Size: 6,423 bytes
- Step4-pro-v1 header: CSV line 2
- primary structure values: CSV lines 4-15
- derived MoE values: CSV lines 20-29
- attention summaries: CSV lines 31-34

Required exact structure:

| Field | Value |
|---|---:|
| Model ID | stepfun-ai/Step4-Pro-V1 |
| hidden_size | 6144 |
| num_hidden_layers | 80 |
| Full attention layers | 20 |
| SWA layers | 60 |
| dense layers | 4 |
| MoE layers | 76 |
| dense intermediate size | 16384 |
| routed experts | 512 |
| experts selected per token | 8 |
| routed expert intermediate size | 2048 |
| shared expert intermediate size | 2048 |
| vocab size | 128896 |

Required normalized block composition:

    ("dense_swa",) * 4 + ("moe_full",) * 20 + ("moe_swa",) * 56

Exact derived arithmetic:

| Quantity | Formula | Expected |
|---|---|---:|
| Dense FFN params per layer | 3 × 6144 × 16384 | 301,989,888 |
| Four dense layers | 4 × 301,989,888 | 1,207,959,552 |
| Selected routed-expert compute per token/layer | 3 × 6144 × 2048 × 8 | 301,989,888 |
| Shared-expert compute per token/layer | 3 × 6144 × 2048 | 37,748,736 |
| Router params per MoE layer | 6144 × 512 | 3,145,728 |

The task requirements explicitly prohibit inventing hidden attention operations or applying a correction factor. The current Step4 temporary MLA approximation must remain temporary until authoritative detail exists: task_memory/task_2026-07-16_step4_pro_v1_support/requirements.md:21-24, plan.md:17-26, notes.md:32-39, issues.md:11-18.

Observed attention gaps against standard GQA arithmetic:

| Attention type | CSV expected | Standard GQA | Absolute gap | Relative gap |
|---|---:|---:|---:|---:|
| Full | 153,095,232 | 113,246,208 | 39,849,024 | 26.029% |
| SWA | 213,911,648 | 163,577,856 | 50,333,792 | 23.530% |

These gaps are evidence that a calibration multiplier would hide unknown structure rather than correct a known formula.

## Registration and Cached-Configuration Audit

### Cached model identity

DefaultHFModels is the package-local cached identity allowlist at src/aiconfigurator/sdk/common.py:492-522. It currently contains stepfun-ai/Step4 but not stepfun-ai/Step4-Pro-V1.

Cached filenames are derived by replacing slash with double hyphen and appending _config.json at src/aiconfigurator/sdk/utils.py:974-979. Therefore the required path is:

    src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json

Cached identities load the local file rather than downloading from Hugging Face at src/aiconfigurator/sdk/utils.py:1283-1289. Config parsing and raw_config preservation occur at src/aiconfigurator/sdk/utils.py:1298-1325.

Package data already includes aiconfigurator/model_configs/*.json at pyproject.toml:114-128. No additional packaging rule is needed.

### Architecture and family reuse

STEP4 already exists and Step4ForCausalLM already maps to it at src/aiconfigurator/sdk/common.py:596-626. Model modules are imported for decorator side effects at src/aiconfigurator/sdk/models/__init__.py:43-49, and architecture-to-family-to-registry factory resolution is implemented at src/aiconfigurator/sdk/models/__init__.py:53-112.

STEP4 is already a MoE family at src/aiconfigurator/sdk/models/helpers.py:26-37. Step4Model is registered for STEP4 at src/aiconfigurator/sdk/models/step4.py:16-18. The factory validates backend, Step4Config type, attention/MoE width equality, and expert-parallel capacity at src/aiconfigurator/sdk/models/step4.py:55-96.

**Conclusion:** if the cached Pro config retains architectures=["Step4ForCausalLM"], no new family, architecture mapping, registry module, or package include is needed.

## Blocking Findings

### B1. Exact cached identity is missing

**Evidence:** src/aiconfigurator/sdk/common.py:492-522 lists the cached models and omits stepfun-ai/Step4-Pro-V1.

**Impact:** CLI validation will not take the package-local fast path, because cached IDs are accepted directly only when present in DefaultHFModels at src/aiconfigurator/cli/main.py:157-187. Offline execution may attempt Hugging Face access.

**Required RED test:** assert the exact ID is present in DefaultHFModels and that get_model_config_from_model_path resolves the package-local config without any download call.

### B2. Package-local Pro configuration is missing

**Evidence:** the required filename follows src/aiconfigurator/sdk/utils.py:974-979, but no Step4-Pro-V1 config currently exists.

**Impact:** model construction cannot reproduce the authoritative 80-layer topology offline.

**Required RED test:** load the exact cached path and assert every authoritative field, exact block tuple, 4/20/60/76 counts, raw_config preservation, and architecture mapping.

### B3. Temporary MLA projection geometry is hard-coded to original Step4 literals

**Evidence:** context graph construction uses 2112, 24576 // tp, and 32768 // tp at src/aiconfigurator/sdk/models/step4.py:129-167. Generation uses 2112 and 24576 // tp at src/aiconfigurator/sdk/models/step4.py:169-207.

The original Step4 config provides the source geometry at src/aiconfigurator/model_configs/stepfun-ai--Step4_config.json:118-128:

- 2112 = q_lora_rank + kv_lora_rank + qk_rope_head_dim = 1536 + 512 + 64
- 24576 = num_attention_heads × (qk_nope_head_dim + qk_rope_head_dim) = 128 × (128 + 64)
- 32768 = num_attention_heads × (qk_nope_head_dim + v_head_dim) = 128 × (128 + 128)

**Root cause:** the graph builder was written when only one Step4-family model existed, so derived geometry was frozen as model-specific literals. A second model exposes the missing config-derived contract.

**Required RED test:** a real Pro config alone is insufficient if its borrowed temporary fields equal the old literals. Construct a valid synthetic Step4Config with different geometry:

| Field | Synthetic value |
|---|---:|
| num_attention_heads | 64 |
| q_lora_rank | 1024 |
| kv_lora_rank | 256 |
| qk_nope_head_dim | 96 |
| qk_rope_head_dim | 32 |
| v_head_dim | 64 |

Expected derived shapes:

| Shape | Formula | Global | TP=4 local |
|---|---|---:|---:|
| downscale output | 1024 + 256 + 32 | 1312 | not sharded |
| q_b output | 64 × (96 + 32) | 8192 | 2048 |
| kv_b output | 64 × (96 + 64) | 10240 | 2560 |

The test must fail against the current literals and pass only when shapes are derived from config. Existing original-Step4 geometry and TP regression tests at tests/unit/sdk/models/test_step4.py:344-402 and :525-550 must remain unchanged.

### B4. Step4-specific config validation silently accepts malformed routed-MoE geometry

**Evidence:** generic parsing uses defaults and truthiness at src/aiconfigurator/sdk/utils.py:612-642:

- missing top-k becomes 0;
- missing routed-expert count becomes 0;
- missing or zero moe_intermediate_size falls back to dense intermediate_size.

Step4-specific required/positive validation at src/aiconfigurator/sdk/utils.py:787-835 does not cover num_experts_per_tok, n_routed_experts, moe_intermediate_size, or all core dimensions.

Observed read-only probes accepted:

| Mutation | Observed parse result |
|---|---|
| missing moe_intermediate_size | accepted as 13824 |
| missing num_experts_per_tok | accepted as 0 |
| zero num_experts_per_tok | accepted as 0 |
| bool num_experts_per_tok | accepted as True |
| float num_experts_per_tok | accepted as 1.5 |
| zero moe_intermediate_size | accepted as 13824 |
| bool hidden_size | accepted as True |

**Root cause:** Step4 inherits generic dense/MoE fallback behavior even though its predefined graph requires explicit routed-MoE geometry. Malformed input is silently transformed into a constructible but incorrect graph, violating fail-fast behavior.

**Required RED validation matrix:**

1. missing required fields:
   - num_experts_per_tok;
   - n_routed_experts;
   - moe_intermediate_size;
   - all existing Step4-required temporary MLA fields.
2. invalid scalar types/values:
   - zero, negative, bool, and float for top-k;
   - zero, negative, bool, and float for routed experts;
   - zero, negative, bool, and float for routed expert intermediate size;
   - zero, negative, bool, and float for hidden size, layer count, vocab size, and other core dimensions.
3. cross-field invalidity:
   - top-k 513 with 512 experts;
   - wrong block length;
   - empty block list;
   - unknown block label;
   - block_types that is neither list nor tuple.
4. factory and parallel guards:
   - unsupported backend;
   - attention/MoE width mismatch;
   - moe_ep_size greater than 512.

Every case must raise explicitly; none may substitute a default.

## Graph Composition and Arithmetic Tests

Step4 graph layer counting is implemented at src/aiconfigurator/sdk/models/step4.py:122-127:

- dense count from dense_swa;
- Full count from moe_full;
- SWA count from dense_swa plus moe_swa.

Step4-Pro-V1 expected counts:

| Count | Expected |
|---|---:|
| dense | 4 |
| Full | 20 |
| SWA | 60 |
| MoE | 76 |
| total layers | 80 |

Dense/shared FFN construction is at src/aiconfigurator/sdk/models/step4.py:209-270. Routed/shared MoE construction is at :272-393. Context/generation graph composition is at :395-450.

Exact RED assertions must cover:

- normalized block order and all four layer counts;
- context and generation graph component multiplicities;
- dense FFN arithmetic 301,989,888 per dense layer and 1,207,959,552 total;
- selected routed-expert arithmetic 301,989,888 per token per MoE layer;
- shared-expert arithmetic 37,748,736 per token per MoE layer;
- router parameters 3,145,728 per MoE layer;
- TP and EP sharding behavior;
- no use of profiled-database fallback in formula-only modes;
- nextn recursive scaling, preserving original Step4 behavior.

Existing templates are tests/unit/sdk/models/test_step4.py:78-128, :146-225, :307-320, :331-341, :344-402, :405-426, :429-490, :493-522, :525-550, and :553-579.

## Task Formula-Only Guard Audit

Task resolves model identity and applies the Step4 database-mode guard immediately at src/aiconfigurator/sdk/task_v2.py:585-595. Primary identity selection for agg/disagg and architecture/family/MoE/nextn extraction is at :708-737.

STEP4 accepts only SOL and SOL_FULL at src/aiconfigurator/sdk/task_v2.py:1151-1158. validate() reapplies the guard at :1185. Disaggregated mode requires identical prefill/decode models at :1211-1223. run(validate=False) performs the guard before performance-database loading at :1578-1586.

Because Pro reuses the STEP4 family, production logic can be reused. Tests must still use the exact Pro model ID rather than infer coverage from original Step4 tests.

Required Pro-ID matrix:

| Serving mode | Database mode | Expected |
|---|---|---|
| agg | SOL | accept |
| agg | SOL_FULL | accept |
| disagg | SOL | accept |
| disagg | SOL_FULL | accept |
| agg/disagg | None | reject in constructor |
| agg/disagg | SILICON | reject in constructor |
| agg/disagg | HYBRID | reject in constructor |
| agg/disagg | EMPIRICAL | reject in constructor |

Mutation tests:

- construct a valid SOL task, mutate database_mode, call validate(), and assert rejection occurs before any DB check;
- construct a valid SOL task, mutate database_mode, call run(validate=False), and assert rejection occurs before any perfdb load.

Existing patterns are at tests/unit/sdk/database/test_step4_roofline.py:70-156.

## SOL and SOL_FULL Contract

These two modes require different assertions.

### SOL

Low-level queries return PerformanceResult(source="sol"):

- GEMM: src/aiconfigurator/sdk/operations/gemm.py:499-503
- ContextMLA: src/aiconfigurator/sdk/operations/mla.py:264-270
- GenerationMLA: src/aiconfigurator/sdk/operations/mla.py:457-463
- MLABmm: src/aiconfigurator/sdk/operations/mla.py:616-622
- MoE: src/aiconfigurator/sdk/operations/moe.py:674-700

Recursive SOL tests must assert:

- every nested executed result is PerformanceResult;
- source equals sol;
- every non-no-op latency is finite and greater than zero;
- permitted explicit no-ops may have zero latency but still retain source=sol.

### SOL_FULL

Low-level queries return a three-value tuple:

    (selected, math_roofline, memory_roofline)

Required assertions:

- tuple length equals 3;
- selected equals max(math_roofline, memory_roofline);
- all three values are finite and non-negative;
- the actual selected, math, and memory values are recorded in test evidence;
- no .source assertion is made.

Existing SOL_FULL examples are tests/unit/sdk/database/test_moe_mla.py:61-90 and :600-611, and tests/unit/sdk/database/test_attention.py:127-143.

## Formula-Only Recursive Graph RED Tests

Recommended new file:

    tests/unit/sdk/database/test_step4_pro_v1_roofline.py

Representative SOL queries:

| Phase | batch_size | x | s | prefix |
|---|---:|---:|---:|---:|
| context | 2 | 8192 | 4096 | 0 |
| generation | 2 | 2 | 4096 | not applicable |

Monkeypatch every profiling/database loader to fail if called:

- GEMM
- ContextMLA
- GenerationMLA
- MLABmm
- MoE
- MoEDispatch
- CustomAllReduce
- NCCL

Then recursively traverse the graph and enforce the SOL contract above. The original pattern is tests/unit/sdk/database/test_step4_roofline.py:240-327.

Explicit zero-latency no-ops are permitted only where existing semantics require them:

- TP=1 CustomAllReduce;
- PP=1 P2P;
- communication-free MoEDispatch.

Existing no-op evidence is tests/unit/sdk/database/test_step4_roofline.py:191-237.

Additional required cases:

- OSL=1 context evidence using tests/unit/sdk/database/test_step4_roofline.py:329-368;
- decode generation components using :371-406;
- collective inner operation names using tests/unit/sdk/database/test_collective_query_capture.py:286-319;
- SOL_FULL recursive traversal with tuple-specific assertions and recorded numeric values.

## SDK Integration Test

Recommended new file:

    tests/integration/test_step4_pro_v1_support.py

Representative SDK call:

    Task(
        serving_mode="agg",
        model_path="stepfun-ai/Step4-Pro-V1",
        system_name="h200_sxm",
        backend_name="vllm",
        backend_version="0.22.0",
        database_mode="SOL",
        isl=128,
        osl=2,
        prefix=0,
        nextn=0,
    ).run_single_agg(
        tp=8,
        pp=1,
        dp=1,
        moe_tp=8,
        moe_ep=1,
        batch_size=2,
        ctx_tokens=128,
        include_per_ops=True,
    )

Required assertions:

- TTFT and TPOT are finite and non-negative;
- Full and SWA per-op entries are present;
- every executed per-op result reports source=sol;
- no profiling loader or LFS-backed database is accessed;
- the exact model ID is preserved in the result/report;
- original Step4 representative integration remains unchanged.

Disaggregated mode should receive its own representative SOL smoke because Task identity selection differs at src/aiconfigurator/sdk/task_v2.py:708-737 and model equality is enforced at :1211-1223.

## CLI Integration Test

CLI model validation accepts cached IDs without network access at src/aiconfigurator/cli/main.py:157-187. The estimate parser defines model/system/backend/batch/TP/PP/DP/MoE TP/EP options at :441-566.

The CLI parser intentionally excludes SOL_FULL at src/aiconfigurator/cli/main.py:799-809. Therefore:

- CLI smoke uses SOL;
- SDK/unit tests cover SOL_FULL;
- Step4-Pro support must not change the global parser merely to expose SOL_FULL.

Programmatic cli_estimate is implemented at src/aiconfigurator/cli/api.py:598-661. Non-SILICON execution without a database version uses the estimate database selector at :804-846. Agg/disagg both reach shared get_model construction at :882-979.

Real subprocess smoke:

    python -m aiconfigurator.main cli estimate \
      --model-path stepfun-ai/Step4-Pro-V1 \
      --estimate-mode agg \
      --system h200_sxm \
      --backend vllm \
      --backend-version 0.22.0 \
      --database-mode SOL \
      --isl 128 \
      --osl 2 \
      --batch-size 2 \
      --ctx-tokens 128 \
      --tp 8 \
      --pp 1 \
      --dp 1 \
      --etp 8 \
      --ep 1 \
      --nextn 0

The dispatch and output path is src/aiconfigurator/cli/main.py:1981-2087 and :2097-2105.

Required assertions:

- subprocess exit code is 0;
- stdout contains Performance Estimate (agg);
- stdout contains Model: stepfun-ai/Step4-Pro-V1;
- stdout contains the system/backend/ISL/OSL summary and TTFT/TPOT estimate;
- stderr contains no network/download failure;
- monkeypatched or isolated execution proves no Hugging Face or LFS access.

The established subprocess style is tests/e2e/cli/test_cli_api_equivalence.py:99-116.

## AFD Boundary

Direct read-only probes of the current original Step4 graph fail for both context and generation AFD partitioning:

    Cannot classify op 'context_dense_swiglu'
    Cannot classify op 'generation_dense_swiglu'

Evidence:

- Step4 creates dense SwiGLU at src/aiconfigurator/sdk/models/step4.py:223-229.
- AFD FFN markers at src/aiconfigurator/sdk/afd_partition.py:302-334 include act_gate, _act, and relu but not swiglu.
- AFD fails fast on unclassified operations at src/aiconfigurator/sdk/afd_partition.py:185-193.
- Inference-session AFD calls are at src/aiconfigurator/sdk/inference_session.py:1364-1372.
- generation_moe_overlap is already atomically assigned to the FFN side at src/aiconfigurator/sdk/afd_partition.py:138-170, with regression coverage at tests/unit/sdk/test_afd_partition.py:74-100.

The current task plan commits only to representative agg/disagg SOL coverage at task_memory/task_2026-07-16_step4_pro_v1_support/plan.md:88-93.

**Recommendation:** keep AFD outside the minimum Step4-Pro-V1 change set and state that boundary explicitly. If “CLI support” is redefined to mean every estimate-mode, open a separate scoped change containing:

1. a real Step4/Step4-Pro graph AFD RED test;
2. an explicit swiglu classifier fix;
3. context and generation AFD regressions.

Do not silently claim AFD support.

## Support-Matrix and Generator Boundary

Exact support-matrix rows take priority, otherwise architecture silicon PASS majority is used at src/aiconfigurator/sdk/common.py:380-463. There is no Step4 or Step4-Pro-V1 row and no Step4ForCausalLM majority basis.

Therefore the support command cannot prove formula-only graph support. Adding a silicon PASS row would be false evidence and is not recommended.

Generator config discovery uses get_model_config_from_model_path at src/aiconfigurator/generator/naive.py:342-373. Generator support checks use the support matrix at src/aiconfigurator/generator/enumerate.py:66-103. Cached identity/config registration enables discovery, but generator success is not a substitute for recursive roofline correctness.

## Finding Severity Summary

### Blocking

1. Missing stepfun-ai/Step4-Pro-V1 cached identity.
2. Missing package-local Step4-Pro-V1 config.
3. Step4 temporary MLA projection shapes remain original-model literals instead of config-derived formulas.
4. Step4-specific parser silently accepts or substitutes malformed routed-MoE geometry.

### Important

1. Pro model ID requires its own Task formula-only guard matrix.
2. Synthetic non-original geometry is required to expose the hard-coded MLA shapes.
3. SOL and SOL_FULL return types require separate test contracts.
4. CLI supports SOL but not SOL_FULL; SDK tests must cover SOL_FULL.
5. Integration must inspect actual per-op sources rather than only successful Task construction.
6. AFD currently cannot classify Step4 dense SwiGLU and must be explicitly excluded or separately fixed.
7. No malformed field may be converted through a dense/MoE fallback.

### Watch

1. Full/SWA head fields are currently provenance/audit hints while the temporary graph uses model-wide MLA geometry.
2. The support matrix has no Step4 silicon evidence.
3. Cached config discovery can help generator inspection but cannot establish performance correctness.
4. num_nextn_predict_layers, quantization, KV-cache, and temporary MLA fields must be documented as borrowed or human-update provenance, not authoritative CSV values.

## Exact Recommended Test Files

1. tests/unit/sdk/models/test_step4_pro_v1.py
   - cached identity and offline config resolution;
   - exact authoritative fields and block topology;
   - config validation error matrix;
   - synthetic geometry RED test;
   - exact graph counts and arithmetic;
   - TP/EP and factory guards;
   - original Step4 regression invariants.

2. tests/unit/sdk/database/test_step4_pro_v1_roofline.py
   - Pro-ID Task guard matrix for agg/disagg;
   - validate() and run(validate=False) post-mutation guards;
   - recursive SOL graph and source assertions;
   - recursive SOL_FULL tuple assertions;
   - database-loader fail-if-called checks;
   - explicit no-op semantics;
   - OSL=1 and decode component coverage;
   - collective inner operation names.

3. tests/integration/test_step4_pro_v1_support.py
   - representative agg SOL SDK run;
   - representative disagg SOL SDK run;
   - per-op Full/SWA presence and source=sol;
   - no network, LFS, or profiling database access;
   - real CLI subprocess smoke using SOL.

## Reproducible Test Commands

Environment:

- conda env: /home/i-fengyicheng/miniconda3/envs/aic-step-design
- Python: 3.11.15
- pytest: 8.4.2
- Ruff: 0.14.1

Recommended variables:

    export PYTHONPATH="$PWD/src:$PWD"
    export MPLBACKEND=Agg
    export TMPDIR=/tmp
    export PYTHONDONTWRITEBYTECODE=1

RED phase:

    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest \
      -p no:cacheprovider -q \
      tests/unit/sdk/models/test_step4_pro_v1.py \
      tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
      tests/integration/test_step4_pro_v1_support.py

Targeted GREEN plus original Step4 regression:

    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest \
      -p no:cacheprovider -q \
      tests/unit/sdk/models/test_step4.py \
      tests/unit/sdk/models/test_step4_pro_v1.py \
      tests/unit/sdk/database/test_step4_roofline.py \
      tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
      tests/unit/sdk/database/test_collective_query_capture.py \
      tests/integration/test_step4_prefill_ranking.py \
      tests/integration/test_step4_pro_v1_support.py

Static checks:

    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/ruff check \
      src/aiconfigurator/sdk/common.py \
      src/aiconfigurator/sdk/utils.py \
      src/aiconfigurator/sdk/models/step4.py \
      tests/unit/sdk/models/test_step4.py \
      tests/unit/sdk/models/test_step4_pro_v1.py \
      tests/unit/sdk/database/test_step4_roofline.py \
      tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
      tests/integration/test_step4_pro_v1_support.py

    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/ruff format --check \
      src/aiconfigurator/sdk/common.py \
      src/aiconfigurator/sdk/utils.py \
      src/aiconfigurator/sdk/models/step4.py \
      tests/unit/sdk/models/test_step4.py \
      tests/unit/sdk/models/test_step4_pro_v1.py \
      tests/unit/sdk/database/test_step4_roofline.py \
      tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
      tests/integration/test_step4_pro_v1_support.py

Final unit regression:

    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest \
      -p no:cacheprovider -q -m unit

## Baseline Validation Evidence

No Step4-Pro production files or tests existed during this audit, so baseline validation proves only that the original Step4 behavior was green before changes.

Command 1:

    PYTHONPATH="$PWD/src:$PWD" MPLBACKEND=Agg TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest \
      -p no:cacheprovider -q \
      tests/unit/sdk/models/test_step4.py \
      tests/unit/sdk/database/test_step4_roofline.py \
      tests/unit/sdk/database/test_collective_query_capture.py \
      -k step4

Observed:

| Metric | Value |
|---|---:|
| collected | 100 |
| selected | 90 |
| deselected | 10 |
| passed | 90 |
| failed | 0 |
| elapsed | 7.85 s |
| exit code | 0 |

Command 2:

    PYTHONPATH="$PWD/src:$PWD" MPLBACKEND=Agg TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest \
      -p no:cacheprovider -q \
      tests/integration/test_step4_prefill_ranking.py

Observed:

| Metric | Value |
|---|---:|
| passed | 1 |
| failed | 0 |
| elapsed | 0.17 s |
| exit code | 0 |

Repository state after baseline validation: git status --short was empty.

## Minimal Recommended Production Change Set

Only after the required RED tests fail for the expected reasons:

1. src/aiconfigurator/sdk/common.py
   - add stepfun-ai/Step4-Pro-V1 to DefaultHFModels.

2. src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json
   - add the exact CSV-backed structure;
   - preserve architectures=["Step4ForCausalLM"];
   - document borrowed temporary MLA, nextn, quantization, and KV-cache provenance.

3. src/aiconfigurator/sdk/models/step4.py
   - replace only 2112, 24576, and 32768 derived-shape literals with formulas from validated config;
   - preserve existing operation names, graph structure, and original Step4 numeric results.

4. src/aiconfigurator/sdk/utils.py
   - make Step4 routed-MoE and core dimensions explicitly required, integer, non-bool, and positive;
   - enforce top-k <= routed experts;
   - reject missing/zero moe_intermediate_size rather than substituting dense intermediate_size.

5. Add the three test files listed above.

Do not add a new family, new model registry module, scaling factor, silent fallback, dependency, global CLI SOL_FULL option, fabricated support-matrix row, or AFD claim in this minimum change.

## Acceptance Criteria

Implementation is ready only when all of the following are true:

- exact cached ID resolves offline from the packaged config;
- authoritative topology is 4 dense + 20 Full MoE + 56 SWA MoE = 80;
- counts are dense=4, Full=20, SWA=60, MoE=76;
- all exact dense/MoE/router arithmetic assertions pass;
- synthetic Step4 geometry proves all MLA projection shapes are config-derived;
- malformed Step4 configs fail fast without substitution;
- Pro-ID Task guard rejects every non-formula mode before DB loading;
- recursive SOL results have source=sol;
- recursive SOL_FULL results satisfy selected=max(math,memory);
- agg and disagg SDK integration pass without network/LFS/profile loaders;
- real CLI agg SOL smoke exits 0 and identifies the exact model;
- original Step4 targeted and unit regressions remain green;
- AFD and support-matrix limitations are stated accurately.

## Root Cause

The fundamental issue is not merely “a missing model entry.” Existing Step4 support combines package registration, a predefined formula graph, and generic config parsing that were sufficient while only one Step4 model existed. Step4-Pro-V1 introduces a second topology and exposes two hidden single-model assumptions: derived MLA geometry was frozen as literals, and required routed-MoE fields were allowed to inherit generic fallback behavior. Safe support therefore requires explicit configuration contracts and RED tests at those seams, not a registration-only patch.

## Recommended Handoff Order

1. Write RED tests for cached identity/config, malformed config rejection, synthetic MLA geometry, Pro Task guard, recursive SOL/SOL_FULL, SDK, and CLI.
2. Confirm failures match the four blockers rather than environment/network errors.
3. Apply the minimal production change set.
4. Run targeted GREEN tests and original Step4 regressions.
5. Run Ruff and full unit regression.
6. Record actual SOL/SOL_FULL numeric results and CLI output in the final test report.
7. Keep AFD and silicon support out of scope unless separately approved.
