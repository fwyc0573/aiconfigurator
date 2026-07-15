## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Created the initial TDD implementation and verification plan for Step4-Pro-V1 support. |
| 2026-07-16 | Reconciled the completed Team audits, narrowed SOL_FULL and AFD claims, and added fail-fast validation and KV-provenance gates. |

# Step4-Pro-V1 AIC Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` or the OMX Team lifecycle. Every production behavior change follows RED → GREEN → REFACTOR.

**Goal:** Add cached, formula-only Step4-Pro-V1 support whose architecture matches the authoritative CSV, whose unresolved values inherit the documented Step4 treatment, whose complete SOL operation graph is validated, and whose SOL_FULL component formulas have direct numeric roofline evidence without claiming unsupported graph execution.

**Architecture:** Reuse the existing Step4 model family and granular predefined-op graph only where its semantics are intentionally shared. Add a distinct cached Step4-Pro-V1 configuration containing CSV-derived dimensions and layer composition. Refactor only hard-coded temporary-MLA geometry that must become config-derived for the second model; do not add fallback logic, empirical latency sources, calibration factors, or speculative attention operations.

**Tech Stack:** Python 3.11, pytest, AIC SDK model registry, predefined operations, SOL/SOL_FULL roofline paths, Ruff.

## Global Constraints

- Authoritative CSV: `/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main - latest.csv`.
- Model support must remain formula-only: complete graph execution uses `DatabaseMode.SOL`; `DatabaseMode.SOL_FULL` is audited through direct `PerfDatabase` component queries because operation wrappers cannot consume its tuple result contract. No perfdb/profiling fallback.
- CSV-defined Step4-Pro-V1 values must be exact: H=6144, layers=80, Full=20, non-Full=60, dense=4, MoE=76, dense intermediate=16384, routed experts=512, top-k=8, routed expert intermediate=2048, shared expert intermediate=2048, vocab=128896.
- Missing attention, MTP, quantization, KV-cache, and latent-MLA values use the existing Step4 treatment and must be identified as human-update items.
- Preserve Full/SWA operation labels and counts; do not claim the temporary MLA calculation is faithful GQA/SWA latency.
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

- [ ] Write tests asserting local cached resolution without network access.
- [ ] Assert every CSV-provided architecture value and exact `4 dense + 20 full-MoE + 56 SWA-MoE = 80` block composition.
- [ ] Assert derived counts (`76` MoE layers) and CSV FFN/MoE arithmetic.
- [ ] Add RED assertions that required integer fields are present, non-boolean, integral, positive, and satisfy `top-k <= routed experts`.
- [ ] Run the new test and observe expected failure because the config/model ID does not exist.
- [ ] Add the minimal cached configuration and registry/default-model entry needed to pass.
- [ ] Run focused tests and existing Step4 model tests.

### Task 2: Make validation fail fast and shared temporary-MLA geometry config-derived

**Files:**
- Modify: `tests/unit/sdk/models/test_step4_pro_v1.py`
- Modify: `src/aiconfigurator/sdk/models/step4.py`
- Modify only if validation requires it: `src/aiconfigurator/sdk/utils.py`, `src/aiconfigurator/sdk/common.py`

**Produces:** Both Step4 and Step4-Pro-V1 obtain temporary MLA projection shapes from validated config fields rather than Step4-specific numeric literals.

- [ ] Write tests for exact context/generation projection tensor shapes on both models.
- [ ] Add an error-path test for inconsistent geometry.
- [ ] Observe RED for missing/zero `moe_intermediate_size`, missing/zero/bool/float `num_experts_per_tok`, boolean core dimensions, invalid topology, and non-divisible parallel geometry.
- [ ] Remove Step4-specific routed-MoE substitution/weak coercion at the parser boundary; do not add compatibility fallback.
- [ ] Observe RED against current hard-coded dimensions where the test differentiates config-derived behavior.
- [ ] Replace only the hard-coded derived dimensions with formulas based on existing validated fields.
- [ ] Re-run both model test suites and verify original Step4 behavior is numerically unchanged.

### Task 3: Validate the complete Step4-Pro-V1 operation graph

**Files:**
- Modify: `tests/unit/sdk/models/test_step4_pro_v1.py`
- Create: `tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
- Modify production only if a RED test exposes a real shared-graph defect.

**Produces:** Structural and numeric evidence for every context/generation op category and all important branches.

- [ ] Assert operation names, scale factors, counts, tensor dimensions, quant modes, collectives, MoE dispatch, routed/shared overlap, merge, logits, embedding, and P2P.
- [ ] Assert exact Full/SWA audit counts (`20/60`), dense/MoE counts (`4/76`), expert values (`512/8/2048/2048`), and H/vocab dimensions (`6144/128896`).
- [ ] Recursively query every op in SOL at representative prefill/decode points.
- [ ] Query the underlying `PerfDatabase` SOL_FULL methods directly and assert `(selected, math, memory)` with `selected == max(math, memory)`; do not pass SOL_FULL tuples through operation wrappers.
- [ ] Monkeypatch every perf-data loader to fail if called.
- [ ] Require all non-zero results to report `source == "sol"`; permit only explicitly proven zero-latency no-ops.
- [ ] Add boundary/error cases for TP/EP width mismatch, expert count, unsupported backend, invalid database mode, empty/invalid block composition, and `nextn=0/3` generation scaling.
- [ ] Record SOL per-op latency/source plus direct SOL_FULL math roofline, memory roofline, and selected maximum for numeric audit.
- [ ] Preserve a regression that documents the current Task-level SOL_FULL tuple/`PerformanceResult` incompatibility; do not refactor the shared operation-query contract without separate approval.

### Task 4: Exercise SDK/CLI integration and document provenance

**Files:**
- Create: `tests/integration/test_step4_pro_v1_support.py`
- Create: `docs/step4_pro_v1_modeling.md`
- Update: `task_memory/task_2026-07-16_step4_pro_v1_support/notes.md`
- Update: `task_memory/task_2026-07-16_step4_pro_v1_support/issues.md`

**Produces:** A complete user-facing support path and a visible human-update register.

- [ ] Write an integration test that constructs and runs a representative vLLM aggregate or disaggregate SOL task using the cached model identity.
- [ ] Observe RED before the complete integration contract is available.
- [ ] Implement only missing integration wiring exposed by RED.
- [ ] Document every CSV value, every Step4-borrowed value, every derived value, and every temporary approximation with source and impact.
- [ ] Include the exact attention mismatch numbers and prohibit interpreting the approximation as measured performance.
- [ ] Include the CSV KV-cache target `10.7 GB`, temporary MLA estimate `48.31838208 GB`, absolute gap `37.61838208 GB`, and ratio `4.515736642991x`; prohibit scaling-factor calibration.
- [ ] State explicitly that aggregate/disaggregate SOL are covered while AFD and support-matrix silicon claims are not.
- [ ] Run a CLI smoke command that requires no LFS data.

### Task 5: Regression, independent review, and completion archive

**Files:**
- Update: all task documentation in `task_memory/task_2026-07-16_step4_pro_v1_support/`
- Create: `task_memory/task_2026-07-16_step4_pro_v1_support/test_report_2026-07-16_step4_pro_v1_support.md`

- [ ] Run focused Step4-Pro-V1 tests.
- [ ] Run affected Step4 regression tests.
- [ ] Run `pytest -m unit` with environment caveats recorded.
- [ ] Run `ruff check .`, `ruff format --check .`, and `git diff --check`.
- [ ] Run an independent code-review lane and remediate every Critical/Important issue.
- [ ] Audit requirements one-by-one against code, test output, docs, worktree/branch state, and numeric evidence.
- [ ] Record exact paths, commands, Python environment, pass/fail counts, timings, expected/actual values, errors, resolutions, and SHA256 inventory.
