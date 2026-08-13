## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Opened the attention-detail provenance issue and recorded its approved boundary. |
| 2026-07-16 | Recorded and resolved the invalid repository `.venv` baseline environment. |
| 2026-07-16 | Recorded the OMX Team clean-workspace gate and its commit-based resolution. |
| 2026-07-16 | Reconciled Team decomposition, SOL_FULL, KV-cache, parser-validation, and AFD findings with root-cause resolutions. |
| 2026-07-16 | Recorded StepCode Claude approval of the bounded SOL_FULL, parser, and AFD resolutions. |
| 2026-07-16 | Recorded the resolved cached-config block-count defect and focused parser resolution evidence. |
| 2026-07-16 | Recorded and resolved silent Step4 parallel-geometry truncation. |
| 2026-07-16 | Recorded and resolved the CustomAllReduce SOL_FULL memory-roofline contract defect. |
| 2026-07-16 | Documented the naive CLI generate feasibility boundary for the eight-GPU smoke command. |
| 2026-07-16 | Documented the generic naive-generator parameter-count mismatch exposed by numeric evidence. |
| 2026-07-16 | Resolved the full-unit AF_UNIX path failure and recorded the generated-temp Ruff format-scan anomaly. |
| 2026-07-16 | Added the small-token EP floor-to-zero and unverified vLLM routed/shared overlap findings from the detailed roofline audit. |
| 2026-07-16 | Recorded the hybrid-attention parameter/KV evidence conflict and prohibited reuse of the upstream SWA-to-HCA latency approximation. |
| 2026-07-16 | Opened the DeepSeek-V4 HCA formula-mode eager-loader defect discovered while extending Step 4-Pro roofline guards. |
| 2026-07-16 | Resolved the HCA eager-loader defect with formula-mode early returns and 36-test roofline/integration evidence. |
| 2026-07-16 | Resolved the migrated roofline review's stale quant-source range and ambiguous inventory-row ordering label. |

# Issues

## ISSUE-001: CSV attention totals do not close from available geometry

- **Status:** Open as a documented human-update item; not an implementation blocker under the original request.
- **Symptom:** Standard GQA/SWA projection counts do not match the CSV `param per full attn` and `param per nonfull attn avg` values.
- **Root cause:** The exported Main CSV contains summary counts and query-head hints but omits the detailed projection recipe that generated the weighted attention totals.
- **Impact:** A supposedly faithful GQA/SWA op graph would undercount attention weights by 23.530%–26.029% and would not be evidence-backed.
- **Resolution for this task:** Preserve `20 Full / 60 SWA` structure labels and use the same explicitly temporary MLA SOL/roofline treatment as Step4. Do not invent missing ops or apply a scaling factor. Record all borrowed values and approximation limits for human replacement.
- **Future resolution:** Replace the temporary attention graph only after a complete Step4-Pro-V1 Attention Detail source is supplied and independently reconciled with the CSV totals.

## ISSUE-002: Repository `.venv` cannot run the required test toolchain

- **Status:** Resolved using the existing verified environment; no dependency or source modification required.
- **Symptom:** `.venv/bin/python` reported Python 3.13.13 and could not import pytest; `.venv/bin/ruff` did not exist.
- **Root cause:** The repository `.venv` is present but unsynchronized, exactly matching the existing environment-handbook caveat.
- **Impact:** The first baseline command exited 1 before test collection; it provided no evidence about source correctness.
- **Resolution:** Use `/home/i-fengyicheng/miniconda3/envs/aic-step-design`, bind the current worktree with `PYTHONPATH="$PWD/src:$PWD"`, select `MPLBACKEND=Agg`, and redirect `TMPDIR` to `tests/.tmp`. The corrected baseline passed 90/90 tests.

## ISSUE-003: OMX Team requires a clean leader workspace

- **Status:** Resolved.
- **Symptom:** `omx team 3:architect` exited 1 before worker creation with `leader_workspace_dirty_for_worktrees`.
- **Root cause:** The newly created task documents were intentionally untracked, and Team mode refuses to launch worktree-capable workers from a dirty leader workspace.
- **Impact:** No worker pane or task was created; production and test source remain untouched.
- **Resolution:** Validated and committed the task documents, launched Team `step4-pro-v1-pre-impl-0ddff5cf`, completed all six tasks, saved both architect reports, and shut the Team down only after `pending=0`, `in_progress=0`, and `failed=0`.

## ISSUE-004: Legacy Team decomposition assigned malformed fragments

- **Status:** Resolved without changing audit ownership.
- **Symptom:** The initial long Team prompt produced task fragments whose owners and report responsibilities did not match the intended three-lane audit.
- **Root cause:** Legacy natural-language decomposition split the prompt at semantic boundaries that did not preserve the requested architecture/roofline/integration ownership mapping.
- **Impact:** Worker-3 initially had no claimable task, while worker-1 owned two fragments that duplicated worker-2/3 report responsibilities.
- **Resolution:** Completed worker-2's roofline scope as task 3, created owner-corrected integration task 6 for worker-3, and converted tasks 4/5 into lifecycle-only superseded fragments. Worker-1 verified both leader-saved reports and closed tasks 4/5 with hashes; the Team finished `6/6` tasks with no failures.

## ISSUE-005: SOL_FULL validation and operation-query contracts are incompatible

- **Status:** Open as an explicitly bounded platform limitation; not fixed in the minimum Step4-Pro-V1 change.
- **Symptom:** `Task` accepts `DatabaseMode.SOL_FULL`, but complete graph execution fails at the first wrapper that calls `float()` on the returned tuple.
- **Root cause:** Direct SOL_FULL database methods return `(sol_time, sol_math, sol_mem)`, while operation wrappers require a scalar-like `PerformanceResult` with `.energy`.
- **Impact:** End-to-end SOL_FULL Task execution cannot be claimed for Step4 or Step4-Pro-V1. A local model-specific cast would hide a shared contract defect.
- **Resolution for this task:** Execute the complete graph in SOL. Audit SOL_FULL through direct `PerfDatabase` calls and assert `selected == max(math, memory)`. Any shared operation-query contract refactor requires a separately approved cross-cutting change.

## ISSUE-006: Temporary MLA KV-cache estimate does not match the CSV

- **Status:** Open as a documented human-update item; scaling is prohibited.
- **Symptom:** Borrowed Step4 latent geometry yields `46,080` elements/token and `48.31838208 GB` at `1,048,576` FP8 tokens, while the CSV reports `10.7 GB`.
- **Root cause:** The current Step4 formula models every layer as full latent MLA and scales linearly with sequence length; it has no sequence-aware SWA/window topology for Step4-Pro-V1.
- **Impact:** The temporary estimate exceeds the CSV by `37.61838208 GB`, a ratio of `4.515736642991x` (`351.5736642991%` over target).
- **Resolution for this task:** Publish both values and the gap. Do not apply a scaling factor. A faithful fix requires authoritative KV topology plus sequence/window-aware cache modeling.

## ISSUE-007: Step4 parsing silently substitutes or accepts malformed geometry

- **Status:** Resolved; focused RED/GREEN and full unit regression passed.
- **Symptom:** Missing/zero `moe_intermediate_size` can inherit dense `intermediate_size`; missing/zero/bool/float `num_experts_per_tok` and boolean core dimensions can pass weak validation.
- **Root cause:** Step4-specific parsing reuses permissive generic numeric extraction and fallback behavior that was not exposed while only one cached Step4 configuration existed.
- **Impact:** Malformed Step4-Pro-V1 configuration can build a plausible but incorrect operation graph instead of failing at the configuration boundary.
- **Resolution:** Added `18` RED cases, required explicit non-boolean positive integers, and directly bound Step4 routed-MoE values after validation. The combined new/original model suites passed `75/75`; parallel-divisibility coverage continues in the graph phase.

## ISSUE-008: AFD cannot classify Step4 dense SwiGLU operations

- **Status:** Confirmed outside the minimum Step4-Pro-V1 scope by independent StepCode Claude review.
- **Symptom:** Existing Step4 context and generation AFD partitioning fail on `context_dense_swiglu` and `generation_dense_swiglu`.
- **Root cause:** The AFD FFN classifier recognizes several activation markers but not `swiglu`, while Step4 emits a dedicated dense SwiGLU operation.
- **Impact:** A generic claim of every CLI estimate mode would be false even before Step4-Pro-V1 is added.
- **Resolution for this task:** Cover aggregate/disaggregate SOL and state the AFD boundary explicitly. If AFD becomes required, open a separate RED fix for classifier behavior and both context/generation regressions.

## ISSUE-009: Initial cached config contained one extra SWA block

- **Status:** Resolved by correcting the authoritative block-count contract; no workaround added.
- **Symptom:** The first post-registration test run reported `Step4 block_types length 81 != num_hidden_layers 80` and failed `3/5` tests.
- **Root cause:** Manual JSON construction contained `57` identical `moe_swa` entries instead of the authoritative `56`.
- **Impact:** Offline resolution correctly failed fast rather than accepting an inconsistent operation graph.
- **Resolution:** Removed the single duplicate entry, independently counted `dense_swa=4`, `moe_full=20`, `moe_swa=56`, and reran the same test file to `5/5 passed`.

## ISSUE-010: Step4 operation construction silently truncates non-divisible parallel geometry

- **Status:** Resolved; focused RED/GREEN and full unit regression passed.
- **Symptom:** Non-divisible attention heads reach a shared `assert`, while vocabulary, dense/shared intermediate widths, routed expert count, and routed intermediate width are silently truncated by integer division.
- **Root cause:** `Step4Model` creates sharded operation dimensions without validating the exact divisibility assumptions of its TP/EP formulas; the shared `BaseModel` checks only attention heads and uses an optimization-removable assertion.
- **Impact:** Invalid topology can yield a plausible but numerically incomplete graph, and `python -O` can remove the one existing head check.
- **Resolution:** Added six Step4-specific construction checks in `Step4Model.create()` and field-specific `ValueError` messages. The RED run was `6/6` failed for the expected missing behavior; GREEN was `6/6` passed, and combined new/original Step4 model regression passed `82/82`.

## ISSUE-011: CustomAllReduce SOL_FULL omitted its communication-memory roofline

- **Status:** Resolved by correcting the shared formula contract and its existing unit test.
- **Symptom:** The direct Step4-Pro-V1 roofline audit returned `(0.44040192, 0, 0)` for `query_custom_allreduce(SOL_FULL)`, so `selected != max(math, memory)` even though the selected time was non-zero.
- **Root cause:** `CustomAllReduce.get_sol()` computes ring-transfer bytes divided by P2P bandwidth, but returned the result only as the selected value. Unlike the adjacent `NCCL` and `P2P` implementations, it incorrectly reported a zero communication-memory component.
- **Impact:** The shared `SOL_FULL` tuple violated the roofline invariant and prevented a uniform audit across Step4-Pro-V1 operation families; the old base-query test had frozen the incorrect tuple.
- **Resolution:** Changed the existing test first to require `sol_mem == expected_sol_time` and `sol_time == max(sol_math, sol_mem)`, observed `2/2` RED failures, then returned `(sol_time_ms, 0, sol_time_ms)` from the formula. Targeted GREEN passed `2/2`, and the final six-suite affected regression passed `192/192`.

## ISSUE-012: Naive CLI generate success is not eight-GPU feasibility evidence

- **Status:** Documented product boundary; not a defect in the requested support scope.
- **Symptom:** `aiconfigurator cli generate --model-path stepfun-ai/Step4-Pro-V1 --total-gpus 8 --system h200_sxm` can render artifacts for a 1,490,676,214,656-parameter model even though the generated command is not a measured or formula-validated deployment point.
- **Root cause:** `generate` intentionally creates a naive configuration without a parameter sweep and prints an explicit warning that it performs no memory validation or performance optimization.
- **Impact:** Treating subprocess success as fit/performance evidence would overstate support and could suggest an infeasible deployment.
- **Resolution for this task:** Assert only cached identity resolution, successful artifact rendering, and the exact CLI summary. Use aggregate/disaggregate `SOL` with the tested `TP=8, PP=2, MoE-TP=8` shape for performance-model integration evidence. Preserve the CLI warning in user documentation.

## ISSUE-013: Generic naive weight sizing does not consume Step4-Pro-V1 block composition

- **Status:** Root cause established and documented; cross-cutting generator correction is outside the approved model-support scope.
- **Symptom:** The requested generate smoke reports `1,559,313,383,424` parameters (`2,904.447509765625 GiB` in BF16), while the authoritative CSV trunk plus both embeddings is `1,490,676,214,656` parameters (`2,776.600820302963 GiB`). The absolute gap is `68,637,168,768` parameters and the relative gap is `4.604431739983%`.
- **Root cause:** `generator.naive._estimate_model_weight_bytes()` applies one generic layer formula to all 80 layers: one embedding, `4 * H^2` attention, and 512 routed experts in every FFN. It does not read `block_types`, the four dense layers, shared-expert dimensions, both embeddings, or the CSV attention totals. `num_nextn_predict_layers` is not included in this calculation, so MTP is not the cause.
- **Impact:** The naive fit check reports required `TP=32`, available maximum `TP=8`, and `fit=False`. Its sizing, TP, and memory messages are not authoritative Step4-Pro-V1 deployment evidence even though artifact rendering succeeds.
- **Resolution for this task:** Preserve the explicit CLI warning, assert the warning in integration coverage, and document exact observed-versus-authoritative values. Do not silently special-case Step4-Pro-V1 or change the generic generator under this task; a block-aware estimator would be a separately approved cross-family refactor and must first follow `.claude/rules/generator-development.md`.

## ISSUE-014: Long task-local TMPDIR exceeds the AF_UNIX manager-socket limit

- **Status:** Resolved as a test-environment entry issue; no source or test skip was required.
- **Symptom:** The first full unit run ended with `13 failed / 2050 passed / 12 skipped / 1123 deselected` after every multiprocessing test that called `mp.Manager()` raised parent-side `EOFError`. Each SyncManager child reported `OSError: AF_UNIX path too long` at `socket.bind()`.
- **Root cause:** `TMPDIR="$PWD/tests/.tmp"` was `81` characters. Python adds a `pymp-XXXXXXXX/listener-XXXXXXXX` suffix, producing a representative `113`-character AF_UNIX pathname that does not fit Linux `sockaddr_un.sun_path`.
- **Impact:** The failure was isolated to `tests/unit/collector/test_parallel_run.py`; Step4-Pro-V1 and all later suites continued to pass. It provided no evidence of a Collector or Step4 production defect.
- **Resolution:** A controlled single test failed with the long path (`1 failed`, `0.36 s`, exit `1`) and passed with `/tmp` (`1 passed`, `0.24 s`, exit `0`). The complete collector file then passed `22/22` with `/tmp` in `6.00 s` and with `/data/ycfeng/tmp` in `7.18 s`. The full unit suite passed with `/tmp`: `2063 passed / 12 skipped / 1123 deselected`, `770.74 s`, exit `0`.

## ISSUE-015: Full-tree Ruff format scan includes generated pytest fixture copies

- **Status:** Resolved as validation scoping; all delivery files are formatted.
- **Symptom:** `ruff format --check .` reported four files below untracked `tests/.tmp/pytest-of-i-fengyicheng/...` that would be reformatted.
- **Root cause:** Earlier tests intentionally copied tracked performance scripts into their pytest temp fixtures. Ruff recursively scanned those non-delivery copies because `tests/.tmp/` is untracked rather than ignored.
- **Impact:** No Git-tracked or changed file failed formatting. Deleting or moving `tests/.tmp/` was prohibited, so a clean whole-tree result could not be obtained by cleanup.
- **Resolution:** `ruff check .` passed, all `432` Git-tracked Python files passed `ruff format --check`, `ruff format --check --exclude tests/.tmp .` passed, and `git diff --check fdd869b..HEAD` passed. The original four-file diagnostic is retained in the test report rather than hidden.

## ISSUE-016: Routed-MoE SOL can floor small-token EP work to zero

- **Status:** Open formula defect; documented for human review and not changed in this documentation-only follow-up.
- **Symptom:** `MoE.get_sol()` computes route work with integer floor division by `moe_ep_size` and `moe_tp_size`. For `X=1`, `topk=8`, and `EP=16`, `total_tokens // EP == 0`, so per-rank FLOPs, route activations, and active-expert weights can all evaluate to zero.
- **Root cause:** The formula models an even average rank and uses `//` instead of a non-zero busiest-rank or explicit route-distribution model. It does not use `workload_distribution` in SOL.
- **Impact:** Small-batch decode with EP greater than the route count is materially underestimated, even though some ranks must execute expert work. This cannot be corrected with a latency scaling factor.
- **Proposed root-cause resolution:** Define the intended metric (average rank versus critical/busiest rank), model integer route ownership explicitly, add RED cases for `X×topk < EP`, skew and non-divisible TP/EP, and then replace floor-to-zero arithmetic with the approved distribution rule.

## ISSUE-017: Generation routed/shared full overlap is unverified for the vLLM-only Step4 graph

- **Status:** Open evidence gap; current document marks the result `CONDITIONAL`.
- **Symptom:** `generation_moe_overlap` returns `max(sum(routed), sum(shared))`, assuming complete branch overlap.
- **Root cause:** `OverlapOp` documents TRT-LLM `maybe_execute_in_parallel`/CUDA Graph behavior, while `Step4Model` rejects every backend except vLLM. No Step4-Pro-V1 vLLM trace in the repository proves complete overlap.
- **Impact:** If vLLM overlaps the branches only partially or runs them serially, generation MoE latency is underestimated.
- **Proposed root-cause resolution:** Capture a vLLM CUDA/Nsight timeline for the exact Step4-Pro-V1 routed/shared path, identify actual overlap intervals and synchronization, then choose serial, partial-overlap, or full-overlap modeling from that evidence.

## ISSUE-018: Full-attention parameter target and 1M-token KV target imply incompatible known structures

- **Status:** Open authoritative-evidence conflict; implementation must report it and must not calibrate around it.
- **Symptom:** A standard 64-head, 96-dimension MHA closes the authoritative `153,095,232` full-attention parameter target within `1.3718833517%`, but its 20 full-history layers consume `257.69803776 GB` at TP1/FP8/`1,048,576` tokens before adding non-full cache. With the 60-layer HCA candidate, total cache is `257.99688192 GB`, versus the CSV target `10.7 GB` (`24.1118581234x`).
- **Root cause:** The CSV provides aggregate parameter and KV targets without the full-attention cache representation. The parameter target is consistent with standard MHA weights, while the cache target implies an unconfirmed compressed, grouped, latent, or differently sharded full-layer KV representation.
- **Impact:** No evidence-backed configuration currently closes both targets. Reusing Step4's `kv_lora_rank=512`, adding a hidden residual, or scaling the result would violate the explicit no-MLA/no-scaling requirements.
- **Proposed root-cause resolution:** Obtain the Step4-Pro-V1 full-attention K/V projection and runtime cache specification, including KV-head count, head dimensions, cache dtype, cache sharding/replication, latent/compression stages, and whether `10.7 GB` is per-rank or model-wide. Until then, implement the requested differentiated formulas and expose the discrepancy without claiming KV-target closure.

## ISSUE-019: Latest AIC maps pure SWA latency to HCA collector data

- **Status:** Known upstream approximation; explicitly prohibited for Step4-Pro-V1.
- **Symptom:** The latest reference `DeepSeekV4Model` groups ratio-0 SWA layers with ratio-128 HCA for latency lookup because no dedicated SWA collector exists, even though its analytic and KV paths distinguish the structures.
- **Root cause:** This is a data-coverage shortcut in the reference model, not an architectural equivalence between SWA and HCA.
- **Impact:** Copying the grouping would silently give SWA HCA latency and violate this task's no-fallback requirement. It would also undermine the requested full/SWA/HCA behavioral distinction.
- **Resolution for this task:** Reuse only the explicit analytic HCA/SWA structure and formulas. Do not route ratio-0 SWA through HCA collector data, and fail loudly if a requested execution mode lacks an evidence-backed formula.

## ISSUE-020: DeepSeek-V4 HCA formula queries eagerly load profile data

- **Status:** Resolved by focused RED/GREEN and complete migrated roofline/integration verification.
- **Symptom:** `ContextDeepSeekV4AttentionModule._query_context_attn_table()` and `GenerationDeepSeekV4AttentionModule._query_generation_attn_table()` call `cls.load_data(database)` before resolving `DatabaseMode.SOL` or `DatabaseMode.SOL_FULL`.
- **Root cause:** These two query methods retain the older eager-loader ordering, while standard `ContextAttention` and `GenerationAttention` already return analytic formula results before calling their loaders.
- **Impact:** Step4-Pro-V1 currently produces `source=sol`, but its HCA operation can still read collector files during formula-only execution. Existing loader-prohibition tests did not include the new HCA classes, so they could not detect the side effect.
- **Proposed root-cause resolution:** First add loader-prohibition coverage for both HCA phase classes and observe the expected RED. Then move only their `load_data()` calls below the `SOL`/`SOL_FULL` early-return branches and before empirical/silicon/hybrid paths, preserving every non-formula behavior without adding a fallback.
- **RED evidence:** The migrated combined suite produced `28 passed / 4 failed` in `12.17s`; a cache-independent direct selection then failed all `4/4` combinations (`context/generation × SOL/SOL_FULL`) in `4.47s`, each at the eager `load_data()` call.
- **Resolution evidence:** The direct selection passed `4/4` in `3.85s` after moving only the two loader calls. The complete migrated roofline/integration suite passed `36/36` in `11.43s`, proving both formula modes avoid profile data while aggregate/disaggregate and CLI behavior remain intact.

## ISSUE-021: Migrated review retained one stale source range and an ambiguous inventory-order label

- **Status:** Resolved during Task 7 Step 9 validation; no production or test behavior was affected.
- **Symptom:** The automated source-range audit found two occurrences of `src/aiconfigurator/sdk/common.py:1232-1310`, although the file currently ends at line `1305`. The context/generation inventory tables also used the heading `Order` while grouping all non-full patterns before full patterns, which could be mistaken for the actual per-layer operation order.
- **Root cause:** The quant-enum reference retained an older end line after source movement. The compact inventory was organized by operation family for count auditing, but its column label did not distinguish that presentation grouping from the model's authoritative layer-order construction.
- **Impact:** Two references were out of range, and a reader could incorrectly infer `all nonfull -> all full` execution ordering even though the graph is built as `0-3 nonfull -> 4-23 full -> 24-79 nonfull`.
- **Resolution:** Narrowed both source references to `common.py:1232-1305`, renamed the column to `Inventory row`, and explicitly stated that the table groups patterns while the actual operation list follows layer `0-79`. The repeated validator then passed all `61` detailed-review references, `21` tables, `62` headings, and `104` code-fence markers; cross-document numeric/stale-name checks and `git diff --check` also passed.
