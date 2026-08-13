## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Created the initial TDD implementation and verification plan for Step4-Pro-V1 support. |
| 2026-07-16 | Reconciled the completed Team audits, narrowed SOL_FULL and AFD claims, and added fail-fast validation and KV-provenance gates. |
| 2026-07-16 | Applied the independent StepCode Claude APPROVE verdict and added explicit original-Step4 formula-equivalence assertions. |
| 2026-07-16 | Marked configuration, graph, roofline, integration, CLI, and provenance-documentation tasks complete after focused verification. |
| 2026-07-16 | Recorded direct roofline, public SDK, per-operation, and CLI numeric evidence. |
| 2026-07-16 | Completed full regression, static validation, independent final review, requirement audit, and completion-archive gates. |
| 2026-07-16 | Added the follow-up standalone roofline-model review document phase with formula, memory, and independent-review gates. |
| 2026-07-16 | Completed the standalone roofline-model review, focused regression, source/formula validation, and independent Claude approval. |
| 2026-07-16 | Added Task 7 to replace only the Pro temporary-MLA boundary with reviewed per-layer full/HCA configs, differentiated KV formulas, and strict RED/GREEN verification. |
| 2026-07-16 | Applied the Task 7 independent Claude APPROVE verdict: matrix-only parameter definition, explicit HCA AllReduce, warning-level KV conflict, and MTP regression. |
| 2026-07-16 | Completed Task 7 schema/parameter RED and schema/parser/cached-config GREEN with full legacy Step4 regression. |
| 2026-07-16 | Completed Task 7 per-layer graph/report RED with ten expected Pro failures and one passing exact legacy-name regression. |
| 2026-07-16 | Completed Task 7 per-layer graph/report GREEN and differentiated KV RED with TP1/TP8 numeric boundaries. |
| 2026-07-16 | Completed Task 7 differentiated KV GREEN and advanced roofline/integration migration to Step 8 after reproducing four stale aggregate-MLA assertion failures. |
| 2026-07-16 | Completed Task 7 Step 8 roofline/integration migration, including HCA formula-loader RED/GREEN, direct component values, and static validation. |
| 2026-07-16 | Completed Task 7 Step 9 documentation migration with source-range, Markdown-structure, cross-document, graph-count, parameter, KV, pair-helper, and direct-roofline validation. |

# Step4-Pro-V1 AIC Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` or the OMX Team lifecycle. Every production behavior change follows RED → GREEN → REFACTOR.

**Goal:** Deliver cached, formula-only Step4-Pro-V1 support with 80 explicit layer identities, independently parameterized 20-layer full attention and 60-layer non-full HCA attention, differentiated KV-cache curves, and fail-loud parameter validation while preserving the original Step4 MLA graph unchanged.

**Architecture:** Keep `Step4Config` and the current MLA builders as the legacy `stepfun-ai/Step4` path. Introduce a mutually exclusive Step4-Pro schema containing `Step4LayerSpec`, `FullAttentionConfig`, and `NonFullAttentionConfig`; build unique per-layer full-attention operations from standard linear Q/K/V/O primitives and unique per-layer HCA modules from the existing analytic DeepSeek-V4 ratio-128 operation. Override only the Pro KV-capacity path with full-history plus SWA-window/compressed-history formulas; do not add perf-data fallback, calibration factors, hidden residual parameters, or unconfirmed latent ranks.

**Tech Stack:** Python 3.11, pytest, AIC SDK model registry, predefined operations, SOL/SOL_FULL roofline paths, Ruff.

## Global Constraints

- Authoritative CSV: `/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main - latest.csv`.
- Model support must remain formula-only: complete graph execution uses `DatabaseMode.SOL`; `DatabaseMode.SOL_FULL` is audited through direct `PerfDatabase` component queries because operation wrappers cannot consume its tuple result contract. No perfdb/profiling fallback.
- CSV-defined Step4-Pro-V1 values must be exact: H=6144, layers=80, Full=20, non-Full=60, dense=4, MoE=76, dense intermediate=16384, routed experts=512, top-k=8, routed expert intermediate=2048, shared expert intermediate=2048, vocab=128896.
- Tasks 1-6 record the completed initial delivery. Task 7 supersedes their temporary-MLA policy only for Step4-Pro-V1; the original `stepfun-ai/Step4` operation names, formulas, and KV values must remain byte-for-byte equivalent.
- Step4-Pro-V1 must use explicit layer records in the authoritative order `0-3 nonfull+dense`, `4-23 full+moe`, `24-79 nonfull+moe`; Attention cannot be represented only by aggregate scale factors.
- Full attention must not reuse Step4 `q_lora_rank=1536` or `kv_lora_rank=512`. Non-full attention must model HCA as local SWA plus main compressor and compressed-history attention, not as full attention with a mask.
- Parameter estimates must remain within 5% of `153,095,232` and `213,911,648`; all residual or unconfirmed components remain explicit named fields and default to zero rather than being tuned to the targets.
- The `10.7 GB` CSV KV target conflicts with the evidence-backed standard-MHA parameter geometry. Implement the requested differentiated formulas and report the conflict; do not reintroduce a latent width or scaling factor to force closure.
- Fail fast on malformed configuration or unsupported execution modes.
- Reject missing, zero, non-integral, or boolean routed-MoE/core geometry instead of inheriting dense dimensions or accepting Python numeric coercions.
- Keep AFD outside this minimum change because the existing Step4 graph cannot classify `*_dense_swiglu`; do not claim AFD support without a separately scoped RED fix.
- No scaling factors, silent defaults, compatibility fallbacks, or unrelated refactors.
- All production changes require observed RED and GREEN tests.

---

### Task 1: Ground the authoritative configuration contract

**Files:**
- Create: `tests/unit/sdk/models/test_step4_pro_v1.py`
- Create after RED: `src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json`
- Modify after RED if required: `src/aiconfigurator/sdk/common.py`

**Produces:** A cached local model identity `stepfun-ai/Step4-Pro-V1` that resolves through the existing Step4 family and exposes the exact CSV values.

- [x] Write tests asserting local cached resolution without network access.
- [x] Assert every CSV-provided architecture value and exact `4 dense + 20 full-MoE + 56 SWA-MoE = 80` block composition.
- [x] Assert derived counts (`76` MoE layers) and CSV FFN/MoE arithmetic.
- [x] Add RED assertions that required integer fields are present, non-boolean, integral, positive, and satisfy `top-k <= routed experts`.
- [x] Run the new test and observe expected failure because the config/model ID does not exist.
- [x] Add the minimal cached configuration and registry/default-model entry needed to pass.
- [x] Run focused tests and existing Step4 model tests.

### Task 2: Make validation fail fast and shared temporary-MLA geometry config-derived

**Files:**
- Modify: `tests/unit/sdk/models/test_step4_pro_v1.py`
- Modify: `src/aiconfigurator/sdk/models/step4.py`
- Modify only if validation requires it: `src/aiconfigurator/sdk/utils.py`, `src/aiconfigurator/sdk/common.py`

**Produces:** Both Step4 and Step4-Pro-V1 obtain temporary MLA projection shapes from validated config fields rather than Step4-specific numeric literals.

- [x] Write tests for exact context/generation projection tensor shapes on both models.
- [x] Assert the original Step4 config formulas explicitly remain `1536 + 512 + 64 = 2112`, `128 * (128 + 64) = 24576`, and `128 * (128 + 128) = 32768`.
- [x] Add an error-path test for inconsistent geometry.
- [x] Observe RED for missing/zero `moe_intermediate_size`, missing/zero/bool/float `num_experts_per_tok`, boolean core dimensions, invalid topology, and non-divisible parallel geometry.
- [x] Remove Step4-specific routed-MoE substitution/weak coercion at the parser boundary; do not add compatibility fallback.
- [x] Observe RED against current hard-coded dimensions where the test differentiates config-derived behavior.
- [x] Replace only the hard-coded derived dimensions with formulas based on existing validated fields.
- [x] Re-run both model test suites and verify original Step4 behavior is numerically unchanged.

### Task 3: Validate the complete Step4-Pro-V1 operation graph

**Files:**
- Modify: `tests/unit/sdk/models/test_step4_pro_v1.py`
- Create: `tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
- Modify production only if a RED test exposes a real shared-graph defect.

**Produces:** Structural and numeric evidence for every context/generation op category and all important branches.

- [x] Assert operation names, scale factors, counts, tensor dimensions, quant modes, collectives, MoE dispatch, routed/shared overlap, merge, logits, embedding, and P2P.
- [x] Assert exact Full/SWA audit counts (`20/60`), dense/MoE counts (`4/76`), expert values (`512/8/2048/2048`), and H/vocab dimensions (`6144/128896`).
- [x] Recursively query every op in SOL at representative prefill/decode points.
- [x] Query the underlying `PerfDatabase` SOL_FULL methods directly and assert `(selected, math, memory)` with `selected == max(math, memory)`; do not pass SOL_FULL tuples through operation wrappers.
- [x] Monkeypatch every perf-data loader to fail if called.
- [x] Require all non-zero results to report `source == "sol"`; permit only explicitly proven zero-latency no-ops.
- [x] Add boundary/error cases for TP/EP width mismatch, expert count, unsupported backend, invalid database mode, empty/invalid block composition, and `nextn=0/3` generation scaling.
- [x] Record SOL per-op latency/source plus direct SOL_FULL math roofline, memory roofline, and selected maximum for numeric audit.
- [x] Preserve a regression that documents the current Task-level SOL_FULL tuple/`PerformanceResult` incompatibility; do not refactor the shared operation-query contract without separate approval.

### Task 4: Exercise SDK/CLI integration and document provenance

**Files:**
- Create: `tests/integration/test_step4_pro_v1_support.py`
- Create: `docs/step4_pro_v1_modeling.md`
- Update: `task_memory/task_2026-07-16_step4_pro_v1_support/notes.md`
- Update: `task_memory/task_2026-07-16_step4_pro_v1_support/issues.md`

**Produces:** A complete user-facing support path and a visible human-update register.

- [x] Write integration tests that construct and run representative vLLM aggregate and disaggregate SOL tasks using the cached model identity.
- [x] Observe RED before the complete integration contract is available.
- [x] Implement only missing integration wiring exposed by RED; no production wiring change was needed, and the RED failures corrected test assumptions about `mix_step`/`genonly_step` evidence.
- [x] Document every CSV value, every Step4-borrowed value, every derived value, and every temporary approximation with source and impact.
- [x] Include the exact attention mismatch numbers and prohibit interpreting the approximation as measured performance.
- [x] Include the CSV KV-cache target `10.7 GB`, temporary MLA estimate `48.31838208 GB`, absolute gap `37.61838208 GB`, and ratio `4.515736642991x`; prohibit scaling-factor calibration.
- [x] State explicitly that aggregate/disaggregate SOL are covered while AFD and support-matrix silicon claims are not.
- [x] Run offline CLI `estimate` and requested CLI `generate` subprocess smoke tests that require no LFS data.

### Task 5: Regression, independent review, and completion archive

**Files:**
- Update: all task documentation in `task_memory/task_2026-07-16_step4_pro_v1_support/`
- Create: `task_memory/task_2026-07-16_step4_pro_v1_support/test_report_2026-07-16_step4_pro_v1_support.md`

- [x] Run focused Step4-Pro-V1 tests.
- [x] Run affected Step4 regression tests.
- [x] Run `pytest -m unit` with environment caveats recorded.
- [x] Run `ruff check .`, `ruff format --check .`, and `git diff --check`; preserve the generated-`tests/.tmp` format-scan anomaly and verify every Git-tracked Python file separately.
- [x] Run an independent code-review lane and remediate every Critical/Important issue; final verdict was `APPROVE` with no required remediation.
- [x] Audit requirements one-by-one against code, test output, docs, worktree/branch state, and numeric evidence.
- [x] Record exact paths, commands, Python environment, pass/fail counts, timings, expected/actual values, errors, resolutions, and SHA256 inventory.

### Task 6: Standalone human roofline-model review

**Files:**
- Create: `docs/step4_pro_v1_roofline_model_review.md`
- Update: `task_memory/task_2026-07-16_step4_pro_v1_support/{requirements,plan,progress,issues,review,summary}.md`
- Update or create test evidence: `task_memory/task_2026-07-16_step4_pro_v1_support/test_report_2026-07-16_step4_pro_v1_support.md`

**Produces:** A self-contained, two-level review of every current Step4-Pro-V1 graph operation, with forward order, exact formula provenance, memory accounting, and a human-readable correctness verdict.

- [x] Inventory every context and generation operation in actual forward order, including nested `OverlapOp` branches and delegated communication queries.
- [x] Define one consistent notation for tokens, batch, sequence length, hidden size, TP/EP/PP widths, quantization bytes, throughput, and bandwidth.
- [x] Record each operation's shape, FLOPs, memory bytes, selected roofline, scale factor, and source-code reference.
- [x] Analyze Full/SWA temporary MLA at projection, attention-core, BMM, KV-cache, and communication levels; distinguish config-derived widths from borrowed constants.
- [x] Analyze Dense FFN and routed/shared MoE at router, dispatch, expert GEMM, shared branch, overlap, merge, and communication levels.
- [x] Classify each model as `PASS`, `CONDITIONAL`, or `OPEN/INCORRECT FOR FAITHFUL PRO MODELING`, with the exact reason and required authoritative evidence.
- [x] Numerically recompute representative formulas and compare against the existing direct `SOL_FULL` evidence.
- [x] Run Markdown/reference/formula validation, obtain an independent StepCode Claude review, and record any remediation before completion.

### Task 7: Replace the Pro temporary MLA boundary with per-layer full/HCA attention

**Status:** In progress; Steps 1-9 complete, Step 10 regression and archival closure is in progress.

**Files:**
- Modify: `src/aiconfigurator/sdk/common.py`
- Modify: `src/aiconfigurator/sdk/utils.py`
- Modify: `src/aiconfigurator/sdk/models/step4.py`
- Modify: `src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json`
- Modify: `tests/unit/sdk/models/test_step4_pro_v1.py`
- Modify only for explicit legacy regression if needed: `tests/unit/sdk/models/test_step4.py`
- Modify: `tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
- Modify: `tests/integration/test_step4_pro_v1_support.py`
- Modify: `docs/step4_pro_v1_modeling.md`
- Modify: `docs/step4_pro_v1_roofline_model_review.md`
- Update in place: `task_memory/task_2026-07-16_step4_pro_v1_support/{requirements,plan,notes,progress,issues,review,summary,test_report_2026-07-16_step4_pro_v1_support}.md`

**Interfaces:**
- Produce `common.Step4LayerSpec(layer_id: int, attention_type: str, ffn_type: str)`.
- Produce `common.FullAttentionConfig.compute_parameter_count() -> int` and a sequence-aware full-history KV-byte method.
- Produce `common.NonFullAttentionConfig.compute_parameter_count() -> int`, an explicit `resident_state_elements` property/field, and a sequence-aware SWA/HCA KV-byte method supporting compression ratios `0` and `128` without CSA fallback.
- Produce `common.Step4ProConfig(layers, full_attention, nonfull_attention, dense_inter_size, shared_expert_inter_size)`.
- Preserve `common.Step4Config` and all original Step4 interfaces unchanged.
- Make the `Step4ForCausalLM` parser select exactly one schema: legacy `block_types` or Pro `layers + full_attention + nonfull_attention`; both, neither, partial Pro sections, malformed layer IDs/types, and invalid dimensions raise field-specific `ValueError`.
- Build per-layer names such as `context_layer_004_full_q_proj_gemm`, `context_layer_004_full_attention`, and `context_layer_024_nonfull_hca_attention`; context scale is `1`, generation scale is only the existing MTP factor, never an aggregate layer count.
- Set per-instance `MIXED_STEP_CONTEXT_ATTENTION_KEYS` and `MIXED_STEP_GENERATION_ATTENTION_KEYS` to the complete unique per-layer Attention subgraph so aggregate mixed-step accounting remains explicit.

- [x] **Step 1: Obtain the independent architecture verdict before code changes**
  - Send StepCode Claude the exact schema, parameter formulas, per-layer graph, reuse boundary, and KV contradiction.
  - Require one of `APPROVE`, `WATCH`, or `BLOCK`; record the raw artifact and reconciliation in `review.md`.
  - Verdict: `APPROVE`; no Critical and no BLOCK. Required clarifications: count only the six trainable HCA matrices in `compute_parameter_count()`, report the `65,632` FP32 resident-state elements separately, add an explicit HCA `CustomAllReduce`, surface the KV conflict at warning level, and test MTP scaling for no double application.

- [x] **Step 2: Write schema and parameter-count RED tests**
  - Add `test_step4_pro_v1_config_preserves_explicit_layer_identity` asserting 80 ordered `Step4LayerSpec` values and exact `20/60` Attention plus `4/76` FFN counts.
  - Add `test_step4_pro_v1_attention_configs_close_authoritative_targets` asserting full estimate `150,994,944` (`1.3718833517%`) and matrix-only HCA estimate `217,055,232` (`1.4695712129%`), with every named unknown equal to zero and `resident_state_elements == 65,632` reported separately.
  - Add parser error tests for mixed legacy/Pro schemas, partial Pro sections, non-contiguous/duplicate/out-of-range `layer_id`, unsupported attention/FFN labels, boolean/zero dimensions, incompatible projection recipes, unsupported compression ratios, and mismatched hidden size.
  - Run these tests before production edits. Expected RED: missing `Step4LayerSpec`, `FullAttentionConfig`, `NonFullAttentionConfig`, and `Step4ProConfig`, plus the cached config still parsing as `Step4Config`.

- [x] **Step 3: Implement the minimal schema and parser GREEN**
  - Add the four frozen dataclasses and zero-argument parameter-count methods in `common.py`; docstrings define parameter count as trainable matrix elements and identify DSV4 FP32 per-head/compressor elements as resident inference state. Reject unsupported projection modes or latent ranks instead of approximating them.
  - Convert the cached Pro JSON from `block_types`/legacy MLA fields to 80 explicit layer objects plus nested full/HCA configs. Keep top-level generic attention metadata informational only; the Pro model path must use nested per-type fields.
  - Split the Step4 parser into explicit legacy and Pro branches with shared fail-fast integer validation only where semantics match.
  - Run the exact RED selection until all schema/parameter/error tests pass, then rerun the complete original `tests/unit/sdk/models/test_step4.py` file to prove legacy parsing is unchanged.

- [x] **Step 4: Write per-layer graph and parameter-report RED tests**
  - Replace Pro assertions for `*_mla_approx*` aggregate ops with unique layer-ID assertions over all 80 attention blocks.
  - Assert each full layer has norm, independent Q/K/V GEMMs, standard `ContextAttention`/`GenerationAttention`, output GEMM, and explicit reduction with `64` query/KV heads and head dimension `96` before TP sharding.
  - Assert each non-full layer has norm, one analytic DeepSeek-V4 HCA module configured with `96` query heads, ranks `1024/1024`, `16` output groups, head dimension `512`, RoPE dimension `64`, window `512`, ratio `128`, zero CSA indexer fields, and one explicit `CustomAllReduce` because the reused SOL module has no network term.
  - Assert no Pro operation name contains `mla_approx`; assert the original Step4 graph still contains the exact existing 14 context and 16 generation MLA names.
  - Add `caplog` coverage for a warning-level initialization report containing parameter validation, `resident_state_elements`, and the unresolved KV target conflict; add a synthetic >5% target mismatch that raises `ValueError` before operation construction.
  - Run before model edits. Expected RED: Pro still exposes two aggregate temporary-MLA groups, has no per-layer names, and emits no parameter-validation report.

- [x] **Step 5: Implement the minimal per-layer graph GREEN**
  - Branch `Step4Model.create()`/`__init__()` on `Step4Config` versus `Step4ProConfig`; validate full and non-full TP divisibility independently and retain all existing MoE/topology checks.
  - Keep legacy `_context_attention_ops()` and `_generation_attention_ops()` unchanged. Add Pro-only builders that consume each `Step4LayerSpec` in order and create unique full or HCA operations with no aggregate Attention count; append one named `CustomAllReduce` after both full and HCA output paths.
  - Derive dense/MoE totals from the same layer records for the existing FFN builders; do not introduce a second layer-order source of truth.
  - Emit one deterministic warning-level initialization report containing targets, estimates, absolute deltas, relative errors, PASS/FAIL, HCA resident-state elements, and the unresolved KV target mismatch; raise `ValueError` for any parameter error greater than 5%.
  - Populate the mixed-step semantic-key tuples from the operations actually built and let the existing backend validate uniqueness/existence.
  - Run the graph/report RED selection to GREEN, then run original Step4 model tests again.
  - Add an explicit nextn=0/3 assertion that every generation layer uses only the existing `_mtp_scale_factor` once and that context layer scale remains exactly `1`.

- [x] **Step 6: Write differentiated KV-cache RED tests**
  - Cover `seq_len` values `0`, `1`, `511`, `512`, `513`, and `1,048,576` for full history, SWA-only ratio `0`, and HCA ratio `128` including compressor state.
  - Assert full attention grows linearly, SWA saturates at the window, HCA continues at `floor(seq_len/128)`, and the 80-layer Pro total equals the sum over explicit layer specs at TP1 and TP8.
  - Assert `Step4Pro` rejects `get_kvcache_elements_per_token()` because its cache curve is non-linear, while original Step4 remains `92 * (512 + 64) = 52,992` elements/token.
  - Assert `get_kvcache_max_tokens()` uses the non-linear binary-search inverse at exact-budget, one-byte-below, zero, and negative boundaries.
  - Run before cache edits. Expected RED: Pro returns the uniform `80 * (512 + 64) = 46,080` elements/token and a linear base-model byte curve.

- [x] **Step 7: Implement the minimal KV-cache GREEN**
  - Add config-level byte formulas using the model's `kvcache_quant_mode` byte width and explicit TP for full KV heads; retain DeepSeek-V4's non-sharded compressed entry and compressor-state semantics for HCA.
  - Override `Step4Model.get_kvcache_bytes_per_sequence()` and `get_kvcache_max_tokens()` only for `Step4ProConfig`; delegate legacy Step4 to existing behavior.
  - Make the Pro constant-elements API fail loudly with an explanatory exception instead of returning a misleading slope.
  - Run the KV RED selection to GREEN and record TP1/TP8 values plus the unresolved CSV `10.7 GB` comparison.

- [x] **Step 8: Update roofline and integration tests under RED/GREEN discipline**
  - Update the roofline suite to query every unique full and HCA operation recursively in `DatabaseMode.SOL`, forbid all perf-data loaders, and assert executed sources remain `sol`.
  - Replace aggregate attention-name counts with exact per-layer contract checks and direct SOL_FULL component checks for standard Attention and DeepSeek-V4 HCA formulas; retain the documented shared tuple-contract limitation.
  - Update aggregate/disaggregate and CLI integration assertions to require representative full/HCA layer names and to reject every Pro `mla_approx` source.
  - First run the updated tests against the old production behavior and retain the expected failures; after the implementation is present, run them to GREEN and inspect actual numeric output rather than only pass counts.

- [x] **Step 9: Update truth-bearing documentation and audit the architecture gap**
  - Rewrite the Pro Attention/KV sections of both existing docs in place; retain historical temporary-MLA numbers only as explicitly superseded baseline evidence.
  - Record the full/non-full parameter table, every formula term, per-layer ordering, operation names, TP semantics, full/SWA/HCA cache examples, and the `257.99688192 GB` TP1 candidate versus `10.7 GB` unresolved conflict.
  - State that the reused DeepSeek-V4 module has an upstream SWA-to-HCA latency approximation that this Pro path does not invoke.

- [ ] **Step 10: Complete regression, independent review, and archival evidence**
  - Run focused Pro model, roofline, and integration suites; original Step4 regression; all affected SDK/backend tests; then `pytest -m unit` with `TMPDIR=/tmp`.
  - Run `ruff check .`, `ruff format --check --exclude tests/.tmp .`, a tracked-Python format check, and `git diff --check`; preserve rather than delete `tests/.tmp/` and all existing stash/untracked evidence.
  - Obtain an independent final StepCode Claude code review. Stop on `BLOCK`; remediate and rerun affected tests for every accepted finding.
  - Update the Markdown test report with exact commands, environment, pass/fail counts, timings, parameter estimates/targets/errors, TP1/TP8 KV values, and failure-resolution evidence.
  - Update `review.md`, English `summary.md`, deliverable SHA256 inventory, and status only after fresh verification proves every non-blocked acceptance criterion.
