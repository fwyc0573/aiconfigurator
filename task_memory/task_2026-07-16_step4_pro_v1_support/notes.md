## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Recorded authoritative inputs, branch constraints, environment notes, and safety boundaries. |
| 2026-07-16 | Corrected the test environment to the verified `aic-step-design` conda environment after `.venv` reproduction failed. |
| 2026-07-16 | Added Team shutdown state, report provenance, SOL_FULL/KV/parser/AFD execution reminders, and the preserved test-temp stash. |
| 2026-07-16 | Recorded the StepCode Claude APPROVE artifact and its explicit original-Step4 formula regression requirement. |
| 2026-07-16 | Added integration parallelism, per-op source semantics, and CLI generate interpretation reminders. |
| 2026-07-16 | Added numeric-evidence artifact details and the generic naive sizing limitation. |
| 2026-07-16 | Recorded the AF_UNIX-safe temporary-directory rule, full-unit result, static-scan caveat, and final StepCode Claude review artifact. |
| 2026-07-16 | Recorded the synchronized upstream reference and initial DeepSeek-V4 vLLM SWA/HCA implementation findings for the hybrid-attention follow-up. |
| 2026-07-16 | Recorded the mapped Step4-Pro-V1 attention boundary, candidate parameter formulas, exact per-layer order, and unresolved KV-target conflict before implementation review. |
| 2026-07-16 | Recorded the independent Task 7 APPROVE verdict and its required parameter-state, HCA AllReduce, KV-warning, and MTP decisions. |

# Operational Notes

## Authoritative Inputs

- CSV path: `/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main - latest.csv`
- CSV SHA256: `f76fca03fd10eb145a04ff9ed906cdbd52beee37609103f9c99006e2bbf1920b`
- CSV bytes: `6423`
- Step4 methodology: `task_memory/task_2026-07-10_step4_predefined_ops_plan/`
- Step4 implementation baseline: commit `fdd869b94bea58265ea2f72cbe142de570fdd1ad`

## Workspace

- Isolated worktree: `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro`
- Branch: `step4-pro`
- The original `step-design` worktree contains unrelated untracked files and must not be changed, stashed, moved, or deleted by this task.
- Team `step4-pro-v1-pre-impl-0ddff5cf` completed `6/6` tasks and was shut down; worker panes are gone and no worker diff remained.
- Architect reports are committed at `6667131`; their verified SHA256 values are `005796a87e7f3874132802461523f58d302d670f3013b42d7514599ce52478ef` and `40fa52a2708209c8def342767ee1a6ba55b5e5d142375bea2a21bb8691977e72`.
- Preserve `stash@{0}` (`e1b4de9d4ba0b73d7ae828f4bbf9b59fd9a0b269`), which contains only `tests/.tmp/tmpid1wo7aj/test_system.yaml`; do not drop it or restore it unless later evidence requires that exact temporary fixture.

## Runtime and Test Environment

- Verified environment: `/home/i-fengyicheng/miniconda3/envs/aic-step-design` with Python `3.11.15`, pytest `8.4.2`, and Ruff `0.14.1`.
- Bind this worktree explicitly with `PYTHONPATH="$PWD/src:$PWD"` and `MPLBACKEND=Agg`.
- Do not use the `81`-character task-local `tests/.tmp` path as `TMPDIR` for multiprocessing tests. Its representative SyncManager socket path is `113` characters and fails with `OSError: AF_UNIX path too long`. Use a preflighted short path such as `/tmp` or `/data/ycfeng/tmp`; both passed the 22-test collector suite.
- `/usr/bin/time` is absent on this host. Record pytest's built-in elapsed time and the shell exit code instead of wrapping test commands with that binary.
- The repository `.venv` is not a valid test environment on this host: it uses Python `3.13.13`, lacks pytest, and has no Ruff executable.
- Consult `task_memory/env_handbook.md` before diagnosing environment-specific failures.
- No GPU or Docker execution is currently planned. If later required, read the company handbooks before running commands.

## Modeling Boundary Requiring Human Follow-up

- The CSV supplies Full/SWA layer counts, query-head hints, and weighted attention parameter totals, but it does not provide enough projection detail to reproduce those totals.
- Standard GQA using borrowed Step4 values `num_key_value_heads=8` and `head_dim=128` gives:
  - Full: expected from CSV `153095232`, standard-GQA calculation `113246208`, absolute gap `39849024`, relative gap `26.029%`.
  - SWA: expected from CSV `213911648`, standard-GQA calculation `163577856`, absolute gap `50333792`, relative gap `23.530%`.
- Do not invent hidden projection operations or apply scaling factors to close these gaps.
- The original temporary-MLA policy above is superseded for Step4-Pro-V1 by the follow-up hybrid-attention request. Preserve the legacy behavior only for `stepfun-ai/Step4`; Step4-Pro-V1 must expose explicit full/non-full configurations and named unknown terms.

## Execution and Validation Reminders

- Complete operation-graph execution is supported in `DatabaseMode.SOL`. Current SOL_FULL database calls return `(selected, math, memory)` tuples that shared operation wrappers cannot consume; audit these methods directly and do not claim end-to-end SOL_FULL Task execution.
- Temporary Step4 MLA KV arithmetic is `80 * (512 + 64) = 46,080` elements/token and `48.31838208 GB` at `1,048,576` FP8 tokens. The CSV target is `10.7 GB`; record the `37.61838208 GB` gap and `4.515736642991x` ratio without calibration.
- Add fail-fast RED coverage before changing Step4 parsing: missing/zero `moe_intermediate_size`, missing/zero/bool/float `num_experts_per_tok`, boolean core dimensions, invalid block composition, `top-k > experts`, and non-divisible parallel geometry.
- Existing Step4 AFD partitioning fails on `context_dense_swiglu` and `generation_dense_swiglu`. The minimum delivery covers aggregate/disaggregate SOL only; do not claim AFD or silicon support-matrix coverage.
- Before modifying `src/aiconfigurator/generator/**`, read `.claude/rules/generator-development.md`. No generator edit is currently planned.
- Independent plan review artifact: `.omx/artifacts/claude-you-are-the-independent-stepcode-claude-reviewer-for-an-impo-2026-07-15T18-18-40-171Z.md`; verdict `APPROVE`, no BLOCK, implementation authorized.
- Preserve exact original-Step4 derived widths as explicit assertions: `2112`, `24576`, and `32768`; a passing broad regression alone is not sufficient evidence for this shared-path change.

## Integration and CLI Reminders

- Representative Step4-Pro-V1 formula execution uses `TP=8`, `PP=2`, attention-DP `1`, MoE-TP `8`, and EP `1`; the resulting 16-GPU worker shape is test evidence, not an optimized recommendation.
- Aggregate `per_ops_source` contains both `mix_step` and `genonly_step`. For OSL greater than one, `mix_step` may explicitly record `generation_attention (not executed)` with zero latency and source `not_executed`; all actually executed operations must remain `source="sol"`.
- Disaggregate source evidence is separated into `prefill` and `decode`, and every executed entry must be `sol`.
- CLI `generate --total-gpus 8` is a naive artifact-rendering smoke. The command itself warns that it performs no memory validation or performance optimization, so success must not be reported as eight-GPU feasibility for the 1.490676-trillion-parameter model.
- The fresh generate smoke reported a generic `1,559,313,383,424`-parameter estimate, required `TP=32`, maximum `TP=8`, and `fit=False`. The `68,637,168,768`-parameter (`4.604431739983%`) gap from the CSV total comes from the generic all-layers-MoE estimator, not MTP. Do not use this output as the authoritative parameter count.
- CLI `estimate` coverage uses `database_mode=SOL`; the global CLI intentionally excludes `SOL_FULL` from its choices.
- Numeric evidence is temporarily retained at `tests/.tmp/step4_pro_v1_numeric_evidence.json` (`32,593` bytes, SHA256 `ec8d9a1b37f343a56aa4ef52b57e1ebca2f33685eca9f0e4cc25a3ea39582555`) until it is transcribed into the final test report. Do not stage `tests/.tmp/`.
- Final independent code-review artifact: `.omx/artifacts/claude-act-as-the-independent-final-code-reviewer-for-the-step4-pro-2026-07-15T19-36-24-039Z.md` (`12,336` bytes, SHA256 `15669234b47250e4b9d1f07577f90b942139fbf4eeaa710b4283adad05aec9b2`). Verdict: `APPROVE`; no Critical, no BLOCK, and no code remediation required.
- `ruff format --check .` enumerates four copied Python fixtures below untracked `tests/.tmp/`. Preserve that failure as evidence and validate the delivery surface with the same command excluding `tests/.tmp` plus a `git ls-files '*.py'` check over all `432` tracked Python files.

## Hybrid Attention Follow-up Reference Evidence

- Reference repository: `/data/ycfeng/stepfun-performance-optimization/aiconfigurator`.
- Synchronized refs: local `main` and `origin/main` both resolve to `f4c58458ab1554c3e7678492e1bc9c7812c678e6`, committed at `2026-07-16T09:33:41+08:00` with subject `fix(sdk): skip MoE workspace for dense Gemma 4 (#1338)`.
- The vLLM 0.24.0 data/collector support entered through commit `d5bad971c95729437ff611e559ed2fc1db6d4318` (`feat(collector): upgrade the vLLM collector to 0.24.0 with eight-system data (#1344)`).
- The latest tree is split-package aware: root `src/aiconfigurator/sdk/models/deepseek_v4.py` and `operations/dsv4.py` are compatibility aliases; authoritative implementation is under `aic-core/src/aiconfigurator_core/sdk/`.
- `DeepSeekV4Model` keeps an explicit per-layer `compress_ratios` sequence and accepts only `{0, 4, 128}`: `0` means pure SWA, `4` means CSA with indexer/top-k, and `128` means HCA compressed attention.
- For latency grouping only, current AIC maps ratio-0 pure SWA layer counts into the ratio-128 HCA module because there is no dedicated SWA collector. This is an explicit approximation in `deepseek_v4.py:124-159`, not proof that SWA and HCA have identical weights or KV storage.
- KV capacity remains per-layer and does not use that latency grouping: every layer stores `min(seq_len, sliding_window) * head_dim`; ratio-4/128 layers additionally store `floor(seq_len / ratio) * head_dim` plus compressor state; ratio-4 adds indexer cache and a second compressor state.
- The vLLM collector maps `csa -> compress_ratio 4` and `hca -> compress_ratio 128`, instantiates one native `DeepseekV4Attention` layer, binds the main cache, `swa_cache_layer`, optional indexer cache, and compressor state caches, then benchmarks the full wrapper `attn_module(positions, hidden_states, None)`.
- The collector separately benchmarks `hca_attn` through `DeepseekV4Attention.forward_mqa(...)`; CSA's index selection is separately represented by `fp8_fp4_paged_mqa_logits(...)` and the module-level CSA measurement.
- The analytic operation explicitly decomposes projection, output absorption, compressor, window attention, compressed-history attention, optional CSA indexer, weights, activations, KV-cache traffic, indexer-cache traffic, and RoPE traffic. HCA has one compressor and no indexer/top-k; CSA has doubled main compression plus a second indexer compressor and top-k-limited compressed pairs.
- No local Python environment currently has vLLM installed. Any statement about the imported native class beyond the collector's pinned vLLM 0.24.0 contract must be checked against the exact upstream vLLM tag rather than inferred from the collector.
- Exact upstream vLLM `v0.24.0` source was retrieved from the official `vllm-project/vllm` repository for confirmation: `attention.py` SHA256 `7f7fd5eec15d5a6296760603590204f5378bd090471acf701831b180318351cf`, NVIDIA `model.py` SHA256 `a3bfb22178f116b9a7c5676dc88967023efaa7f06b42bec9b74b82318452309a`, `flashmla.py` SHA256 `42ccf698e6a1a27313d01758e05013138cea83d47a4ee7c76f54dac1641f5fec`, and `flashinfer_sparse.py` SHA256 `b709066a93d49e8e635424708b47ba52500555f9c986116882ff54bdcb75f413`.
- Official vLLM reads `compress_ratio` by layer index from `config.compress_ratios`, normalizing config value `0` to runtime value `1`; this preserves per-layer identity and distinguishes SWA-only (`<=1`), HCA (`128`), and CSA (`4`).
- Every DSv4 attention layer owns common projections `fused_wqa_wkv`, `wq_b`, `wo_a`, and `wo_b`, plus a separate `DeepseekV4SWACache`. A compressor is created only when `compress_ratio > 1`; an indexer and its second compressor are created only when `compress_ratio == 4`.
- The full forward order is: parallel input projections; fused Q/KV RMSNorm; Q up-projection and SWA KV insertion; optional main compressor; optional CSA indexer plus indexer compressor; sparse attention combining the SWA cache with optional compressed-history cache; inverse RoPE; and two-stage output projection.
- HCA is therefore not “SWA with a different mask”: it is SWA plus a main compressed-history cache and compressor, and its sparse attention combines window indices with deterministic ratio-128 compressed-history indices. SWA-only has no compressed cache, compressor, or indexer.
- CSA adds an independent query projection, per-head weight projection, indexer KV cache, indexer compressor, and top-k selection. Those CSA-only components must not be attributed to HCA/SWA unless Step4-Pro-V1 evidence confirms them.
- The official KV-cache contract matches the requested separation: the SWA cache is always allocated; the main compressed cache is absent for SWA-only and represented by an `MLAAttentionSpec` with the layer's `compress_ratio` for HCA/CSA; CSA additionally owns an indexer cache.
- Current `step4-pro` production changes are committed through `5b7f55a`; the only pre-existing uncommitted repository changes are task documentation, the untracked roofline review document, and preserved `tests/.tmp/` evidence. No production or tracked test file is dirty at follow-up intake.
- Current Step4-Pro-V1 integration tests explicitly expect operation names containing `swa_mla_approx`, confirming that the temporary uniform MLA boundary is observable in the public execution path and must be replaced rather than merely relabeled.
- The codebase already contains a generic per-layer hybrid-attention vocabulary in `HybridMoEConfig` and hybrid model tests, including explicit `attn_layer_pattern`, separate SWA/global dimensions, and window-capped KV-cache examples. These existing interfaces must be evaluated for semantic reuse before adding another overlapping abstraction.
- The base model still contains a DeepSeek-family MLA KV-cache default (`512 + 64`) when latent fields are absent. Step4-Pro-V1 must override this path explicitly; changing the shared fallback globally would risk unrelated DeepSeek/Step4 regressions.
- The required changes are expected to remain in SDK model/config/parser/operation/test surfaces. No generator or Collector edit is currently justified; if later evidence requires either, the corresponding mandatory rule/skill must be loaded before editing.

## Hybrid Attention Follow-up Code Mapping

- `src/aiconfigurator/sdk/models/step4.py` currently routes both Full and SWA labels through the same `_context_attention_ops()` and `_generation_attention_ops()` MLA builders. Both branches therefore reuse `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`, and `num_attention_heads=128`.
- The current Pro cache formula is the uniform latent-MLA expression `80 * (512 + 64) = 46,080` elements per token. This is precisely the boundary that the follow-up request replaces; the original Step4 formula must remain unchanged.
- The authoritative Pro layer order is already encoded by the cached config and must be preserved exactly as 80 explicit layer records: layers `0-3` are `nonfull+dense`, layers `4-23` are `full+moe`, and layers `24-79` are `nonfull+moe`.
- `HybridMoEConfig` provides useful vocabulary for per-layer attention/FFN patterns and window/global KV accounting, but it lacks the required full/non-full projection configs, HCA compressor/history/indexer fields, parameter targets, `compute_parameter_count()`, and named unknown terms. Reusing it directly would obscure the required contract; the minimum candidate is a Step4-Pro-specific schema while retaining the legacy `Step4Config` path.
- The parser boundary should be an explicit mutually exclusive schema union: legacy `block_types` selects `Step4Config`; explicit `layers`, `full_attention`, and `nonfull_attention` select the Step4-Pro schema; supplying both or neither fails immediately. This is schema discrimination, not fallback behavior.

## Candidate Parameter Geometry Pending Independent Review

- Candidate full attention is standard linear MHA with `hidden_size=6144`, `num_query_heads=num_kv_heads=64`, and `q/k/v_head_dim=96`; `64 * 96 = 6144`, so each Q/K/V/O matrix has `37,748,736` parameters. Estimated total is `150,994,944`, versus target `153,095,232`: absolute gap `2,100,288`, relative error `1.3718833517%`.
- Candidate non-full attention uses the latest DeepSeek-V4 HCA structure rather than a masked full-attention shortcut: `num_query_heads=96`, `q_lora_rank=1024`, `o_lora_rank=1024`, `o_groups=16`, `compressed_head_dim=512`, `rope_dimension=64`, `window_size=512`, and `compression_ratio=128`. HCA has one main compressor and no CSA indexer/top-k.
- Candidate non-full parameter terms are: `H*q_rank=6,291,456`, `q_rank*heads*head_dim=50,331,648`, `H*head_dim=3,145,728`, `heads*head_dim*o_rank=50,331,648`, `o_groups*o_rank*H=100,663,296`, and `2*H*head_dim=6,291,456`. Estimated total is `217,055,232`, versus target `213,911,648`: absolute gap `3,143,584`, relative error `1.4695712129%`.
- The reusable DeepSeek-V4 operation's `_estimate_weights()` additionally accounts for `96 + 128*512 = 65,632` FP32 weight/state elements. If those are classified as trainable parameters rather than runtime state, the non-full estimate becomes `217,120,864`, with a `3,209,216` gap and `1.5002530390%` relative error. Both interpretations pass the 5% gate, but the plan review must choose and document one definition rather than silently mixing parameter and resident-weight memory counts.
- Independent review selected the conventional matrix-only definition: `compute_parameter_count()` returns `217,055,232`; the `65,632` FP32 elements are exposed separately as `resident_state_elements` and included in resident-weight/memory reporting, not trainable parameter closure.
- Both candidate configs keep `unknown_extra_projection_params`, `unknown_router_params`, and `unknown_compression_params` explicit and initially zero. The remaining target gaps stay visible; they must not be absorbed into a scaling factor or an invented residual parameter.

## KV-Cache Formula Boundary

- Full attention candidate: `seq_len * (num_kv_heads / TP) * (k_head_dim + v_head_dim) * bytes_per_element`; it stores the complete history and grows linearly.
- SWA-only candidate: `min(seq_len, window_size) * compressed_head_dim * bytes_per_element`; it saturates after the window is full.
- HCA candidate: `[min(seq_len, window_size) + floor(seq_len / compression_ratio)] * compressed_head_dim * bytes_per_element + compressor_state_bytes`; it retains a bounded window plus compressed history and therefore continues to grow at the compression rate.
- At TP1, FP8, and `1,048,576` tokens, the reviewed candidate gives `257.69803776 GB` for 20 standard-MHA full layers and `0.29884416 GB` for 60 HCA layers, totaling `257.99688192 GB`. The CSV target is `10.7 GB`, a `24.1118581234x` ratio. This is a source-evidence conflict, not a formula bug: the 153M full-attention parameter target supports standard MHA, while the KV target implies an unconfirmed compressed/latent cache for the full layers.
- Do not reintroduce the unconfirmed Step4 `kv_lora_rank=512` or apply any calibration factor to close the `10.7 GB` target. The implementation can satisfy differentiated full/SWA/HCA formulas and must report the unresolved target conflict explicitly.

## Task 7 Independent Review Gate

- Approved artifact: `.omx/artifacts/claude-you-are-the-independent-stepcode-claude-reviewer-read-only-r-2026-07-16T08-53-12-302Z.md`, `12,729` bytes, SHA256 `c0fa487fbe3a643a1e0178036d857349395f398570d84969e8ab4666843735a6`.
- Verdict: `APPROVE`; `0` Critical findings and no BLOCK. Implementation is authorized after recording the review decisions.
- The DSV4 HCA SOL module has no network term, so each Pro non-full layer must append an explicit `CustomAllReduce`; this is not a duplicated cost in formula-only SOL.
- The parameter/KV initialization report must use at least warning level because the `24.1118581234x` KV mismatch remains open even though parameter closure passes.
- Per-layer Attention with FFN counts derived from the same layer records satisfies the request; the layer config remains the single source of truth.
- The generation regression must prove the existing MTP factor is applied once rather than being multiplied again during the per-layer conversion.
- Advisor execution note: a minimal channel diagnostic succeeded; the first oversized invocation produced no artifact, and a second full-repository prompt timed out at `300 s` with `RC=124`. A shorter self-contained third prompt completed with `RC=0` at `effort=max`. No lower-effort or alternate-provider fallback was used.
