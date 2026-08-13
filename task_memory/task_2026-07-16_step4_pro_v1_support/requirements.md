## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Captured the original Step4-Pro-V1 support request and its explicit unresolved-parameter policy. |
| 2026-07-16 | Captured the follow-up request for a standalone, two-level human roofline-model review document. |
| 2026-07-16 | Captured the follow-up request to replace the temporary uniform MLA attention model with per-layer hybrid attention informed by the latest DeepSeek-V4 vLLM implementation. |
| 2026-07-16 | Marked the original temporary uniform-MLA decision as superseded by the later hybrid-attention refactor request. |

# Requirements: Step4-Pro-V1 AIC Support

## Raw User Intent

1. [Original Request] Treat this as a new task whose core objective is to add Step4-Pro-V1 support to AIC.
2. [Original Request] Follow the implementation methodology and engineering practice recorded in `task_memory/task_2026-07-10_step4_predefined_ops_plan` for the earlier Step4 support.
3. [Original Request] Define the Step4-Pro-V1 architecture and operation composition accurately from `permormancebenchmark/architecture_calculator_v1 - Main - latest.csv`.
4. [Original Request] Ensure every Step4-Pro-V1 operation has a correct, reasonable, evidence-backed roofline calculation process.
5. [Original Request] For critical parameter values that cannot be determined from the authoritative CSV, follow the corresponding Step4 treatment and document each borrowed or approximate value clearly for later human update or modification.
6. [Original Request] Create a new worktree and branch named `step4-pro`, and make all code-module changes on that branch.
7. [Original Request] Use parallel Team mode and subagents fully to accelerate delivery.

## Follow-up Roofline Review Request

1. [Original Request] Create one standalone document that explains how AIC models the roofline of every current Step4-Pro-V1 operation.
2. [Original Request] Organize the document in two levels: an overall forward-ordered operation inventory grouped into Attention, MoE, and related sections; then detailed per-operation modeling, FLOPs, memory, and correctness review.
3. [Original Request] Treat Attention variants and MoE operations with extra depth so a human reviewer can understand how the algorithm's structure is represented.
4. [Original Request] Check whether every model and memory calculation is correct and reasonable, and clearly distinguish correct formulas, conditional approximations, and structural gaps requiring authoritative information.
5. [Original Request] Keep the document clear, tidy, and suitable for manual review without requiring the reader to reconstruct the operation graph from source code.

## Follow-up Hybrid Attention Refactor Request

1. [Original Request] Review the existing branch context in `task_memory/task_2026-07-16_step4_pro_v1_support/` before changing the Step4-Pro-V1 attention implementation.
2. [Original Request] In `/data/ycfeng/stepfun-performance-optimization/aiconfigurator`, fetch `origin` and bring the local `main` reference to the latest upstream state available on 2026-07-16.
3. [Original Request] Locate and confirm the concrete DeepSeek-V4 vLLM SWA/HCA operation implementation in that latest source, and use it as evidence for the Step4-Pro-V1 attention refactor.
4. [Original Request] Read `/home/i-fengyicheng/.codex/attachments/f944f531-ad21-4c09-9e6d-ac0f39e80bc5/pasted-text-1.txt` before continuing.
5. [Original Request] Replace the incorrect single Step4 MLA configuration reused by all 80 Step4-Pro-V1 layers with an explicit per-layer hybrid architecture containing 20 full-attention layers and 60 non-full-attention layers.
6. [Original Request] Represent each layer independently with its `layer_id`, attention type, and FFN type; do not reduce attention heterogeneity to aggregate counts.
7. [Original Request] Add an independently parameterized full-attention configuration with explicit Q/K/V/output projection choices and optional latent rank only when MLA-style structure is confirmed.
8. [Original Request] Make the full-attention parameter estimate close within 5% of the authoritative target `153,095,232` parameters per layer, without reusing the unconfirmed Step4 `q_lora_rank=1536` or `kv_lora_rank=512` assumptions.
9. [Original Request] Add an independently parameterized non-full-attention configuration that represents local SWA plus its additional sparse/global-memory and HCA/compression/indexing components rather than treating it as full attention with only a window mask.
10. [Original Request] Account for the authoritative non-full-attention target `213,911,648` parameters per layer and expose any unconfirmed residual terms as named configuration fields rather than hidden approximations.
11. [Original Request] Separate attention head counts by attention type, using the supplied hints of 64 query heads for full attention and 96 query heads for non-full attention unless stronger evidence establishes different values.
12. [Original Request] Separate KV-cache estimation by attention type: full history for full attention, bounded window storage for SWA, and window plus compressed-history storage for HCA/compressed attention.
13. [Original Request] Expose every unconfirmed component through explicit named fields such as `unknown_extra_projection_params`, `unknown_router_params`, and `unknown_compression_params`; do not silently approximate unknowns as MLA.
14. [Original Request] Require every attention configuration to expose `compute_parameter_count()` and print a Step4-Pro-V1 attention parameter validation report during simulator initialization.
15. [Original Request] Fail loudly with a warning or error when an attention parameter estimate differs from its target by more than 5%.
16. [Original Request] Add unit coverage for all changed logic, distinguish full/SWA/HCA KV-cache behavior, exercise edge and error cases, run the complete simulator workflow, and confirm existing Step4 configurations do not regress.
17. [Original Request] Report the final attention parameter validation table, the concrete tests and numeric evidence, and every remaining architecture gap.

## Captured Decisions Already Present in the Original Request

1. [Original Request] The CSV is authoritative for Step4-Pro-V1 structure and values that it explicitly provides.
2. [Original Request] Missing or non-closing critical values inherit the existing Step4 handling rather than being silently invented.
3. [Original Request] The inherited Step4 treatment must be explicitly marked in documentation so the user can update it manually after task completion.
4. [Original Request][Superseded by Follow-up Hybrid Attention Refactor Request items 5-13] Because the CSV attention totals did not close under the initially borrowed GQA/SWA geometry, the original delivery retained the CSV-defined `20 Full / 60 SWA` audit labels while temporarily using the existing Step4 MLA roofline treatment. The later request explicitly replaces that temporary decision with independently parameterized per-layer full and non-full attention and forbids silent MLA approximation.

## Pending Questions

None. The later hybrid-attention request resolves the prior temporary-MLA policy by requiring explicit per-layer attention configs, named unknown fields, fail-loud parameter validation, and honest reporting of remaining evidence gaps.
