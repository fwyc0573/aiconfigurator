## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Initialized the execution ledger and recorded the evidence-grounding/worktree setup. |
| 2026-07-16 | Recorded the failed `.venv` baseline, root cause, corrected conda rerun, and exact passing counts. |
| 2026-07-16 | Recorded the initial OMX Team clean-workspace rejection and planned source-of-truth commit. |
| 2026-07-16 | Recorded the completed parallel Team audits, report handoff/verification, lifecycle recovery, and clean shutdown. |
| 2026-07-16 | Recorded the independent StepCode Claude APPROVE verdict and implementation authorization. |
| 2026-07-16 | Recorded cached-config, fail-fast parser, and config-derived geometry RED/GREEN cycles with numeric evidence. |
| 2026-07-16 | Recorded parallel-geometry truncation root cause, six-case RED/GREEN evidence, and focused regression results. |
| 2026-07-16 | Recorded formula-only graph coverage, the CustomAllReduce roofline correction, and the 192-test affected regression. |
| 2026-07-16 | Recorded offline aggregate/disaggregate integration, CLI subprocess coverage, modeling documentation, and focused regression. |
| 2026-07-16 | Recorded direct roofline, per-operation, SDK/CLI numeric evidence and the naive sizing mismatch root cause. |
| 2026-07-16 | Recorded full-unit AF_UNIX diagnosis, short-TMPDIR verification, repository-wide passing regression, static evidence, and final independent review. |
| 2026-07-16 | Completed the follow-up two-level roofline/memory review document, focused validation, and independent Claude review. |
| 2026-07-16 | Started the hybrid-attention refactor follow-up, captured the attachment requirements, and synchronized the reference repository to the latest 2026-07-16 main commit. |
| 2026-07-16 | Completed the hybrid-attention code-surface mapping, candidate parameter arithmetic, and KV-target conflict analysis before the independent plan gate. |
| 2026-07-16 | Completed the independent Task 7 StepCode Claude architecture gate with APPROVE and reconciled all required plan changes. |
| 2026-07-16 | Recorded the post-review plan self-check and fresh 117-test pre-Task-7 baseline. |
| 2026-07-16 | Recorded the observed 13-case Task 7 schema/parameter RED result before parser and cached-config implementation. |
| 2026-07-16 | Extended and observed the Task 7 parser-boundary RED matrix for every planned fail-fast schema branch. |
| 2026-07-16 | Completed Task 7 schema/parser/cached-config GREEN, legacy regression, and targeted static validation. |
| 2026-07-16 | Observed Task 7 per-layer graph/report RED: ten Pro failures at the legacy type guard and one passing exact original-Step4 name regression. |
| 2026-07-16 | Completed Task 7 per-layer full/HCA graph/report GREEN and observed all 17 differentiated KV tests fail against the missing nonlinear API. |
| 2026-07-16 | Completed Task 7 differentiated KV GREEN and reproduced the four stale aggregate-MLA roofline/integration assertions before test-contract migration. |
| 2026-07-16 | Mapped the actual standard Attention and DeepSeek-V4 HCA query/loader surfaces required by Task 7 Step 8. |
| 2026-07-16 | Observed Task 7 Step 8 HCA loader RED in both formula modes and both execution phases. |
| 2026-07-16 | Completed Task 7 Step 8 per-layer roofline/integration GREEN and resolved formula-mode HCA profile loading at its root cause. |
| 2026-07-16 | Restored the Task 7 Step 9 continuation point and re-audited the current source/document truth boundary before documentation edits. |
| 2026-07-16 | Completed Task 7 Step 9 documentation migration and independent source/formula/structure validation. |
| 2026-07-16 | Completed Task 7 fresh focused and affected regression before the full-unit gate. |

# Progress

## Status

- Completed: repository/history/CSV discovery; original-request decision resolution; isolated worktree creation; passing Step4 baseline; three-lane Team audit; report verification; clean Team shutdown.
- Completed: Team-finding reconciliation and independent StepCode Claude plan review (`APPROVE`, no BLOCK).
- Completed: cached identity/config, fail-fast Step4 parser validation, config-derived projection geometry, parallel-divisibility validation, and original-Step4 focused regression.
- Completed: Step4-Pro-V1 structural graph, direct SOL/SOL_FULL roofline audit, offline aggregate/disaggregate integration, CLI subprocess coverage, provenance documentation, numeric evidence capture, and full regression.
- Completed: standalone human-review document covering all context/generation operations, detailed Attention/MoE formulas, memory correctness verdicts, open-information register, and review checklist.
- Completed: independent final code review (`APPROVE`), test report preparation, requirement audit, and completion archive.
- Completed: current-tree focused regression, static checks, document/hash audit, and preserved stash verification immediately before the final archive commit.
- Completed: Task 7 hybrid-attention refactor planning and independent StepCode Claude architecture review (`APPROVE`, no BLOCK).
- Completed: Task 7 strict schema/parameter RED tests and extended parser-boundary RED coverage.
- Completed: Task 7 frozen dataclasses, exclusive schema parser, explicit 80-layer cached config, parameter arithmetic, and original-Step4 parser regression.
- Completed: Task 7 per-layer graph/report RED and GREEN, including independent full/HCA geometry, explicit HCA reductions, dynamic mixed-step keys, parameter validation, MTP, and exact legacy regression.
- Completed: Task 7 differentiated full/SWA/HCA config/model KV RED, including TP1/TP8 totals, invalid inputs, nonlinear API, and inverse boundaries.
- Completed: Task 7 differentiated full/SWA/HCA KV GREEN, including nonlinear model totals and exact inverse-capacity boundaries.
- Completed: Task 7 roofline/integration migration from aggregate temporary-MLA names to explicit per-layer full/HCA evidence, including HCA formula-mode no-loader correction.
- Completed: Task 7 truth-bearing documentation migration, source-range/Markdown audit, independent formula recomputation, and stale-claim scan.
- In progress: Task 7 focused/full regression and static validation.
- Pending: independent review, test-report, and hash closure.

## 2026-07-16 — Task 7 fresh focused and affected regression

- **Motivation:** Verify every modified Pro code path first, then broaden to all directly affected original-Step4, Attention, DeepSeek-V4 HCA, parser, backend, memory, and database contracts before spending the full repository-unit runtime.
- **Expectation:** The focused Pro suite passes all schema, parameter, graph, KV, roofline, no-loader, public SDK, and CLI cases; the affected suite preserves original Step4 and all shared/non-formula HCA branches with no regression.
- **Method:** Used Python 3.11.15 from `/home/i-fengyicheng/miniconda3/envs/aic-step-design` with `PYTHONPATH="$PWD/src:$PWD"`, `MPLBACKEND=Agg`, and `TMPDIR=/tmp`. Ran the three focused Pro files, then 15 additional original/shared SDK and integration files covering Step4, standard Attention, direct/base/collective queries, DeepSeek-V4 modules and sparse ops, encoder Attention/interpolation, base backend, context-parallel dense behavior, memory estimation, shared PerfDatabase registration, parser utils, and Step4 prefill ranking. The first affected command's verbose dot stream exceeded the tool display budget, so the identical command was repeated with output redirected to `/tmp/step4_pro_task7_affected_regression.log` and its final summary read back; no code or test input changed.
- **Result:** Focused regression passed `130/130` in `11.30s`, exit `0`. The evidence-captured affected regression passed `382` with `1` existing conditional skip in `39.49s`, exit `0`; there were zero failures or errors. Step 10 now advances to the complete `pytest -m unit` gate.

## 2026-07-16 — Task 7 Step 9 documentation validation and closure

- **Motivation:** Close the truth-bearing documentation phase only after proving that the migrated full/HCA architecture, formulas, source references, operation inventories, and numeric evidence match the current implementation rather than the superseded aggregate-MLA graph.
- **Expectation:** Both documents contain no current Pro MLA assertion; every source range exists; Markdown structure is valid; parameter, graph, KV, compressed-pair, and direct `SOL_FULL` values reproduce exactly from current APIs and independent arithmetic.
- **Method:** Read both documents and the referenced Attention/HCA/KV implementations end to end; ran a Python validator over heading hierarchy, Markdown table widths, code-fence balance, file existence, and all numeric source ranges; independently parsed the cached config, built TP1/TP8 models, counted top-level/recursive/leaf operations, evaluated all six KV boundaries, compared eight HCA compressed-context cases with a direct per-query sum, and called the standard/HCA formula tables against a `1 TFLOPS / 1 TB/s` database stub. Ran a cross-document marker and prohibited-stale-name audit plus `git diff --check`. Corrected the out-of-range `common.py:1232-1310` reference to `1232-1305` and clarified that the compact inventory rows group patterns rather than replace the runtime `0-3 nonfull -> 4-23 full -> 24-79 nonfull` ordering.
- **Result:** Documentation validation passed with `23/62` headings, `16/21` tables, `34/104` fence markers, and `1/61` parsed source references in the modeling/review documents respectively; all references are in range and all tables are column-consistent. Formula validation reproduced `339` context top-level/leaves, `332` generation top-level operations, `340` recursive generation nodes, `339` generation leaves, and `320 = 140 full + 180 HCA` Attention operations per phase. Full/HCA estimates were `150,994,944` versus `153,095,232` (`1.3718833517%`) and `217,055,232` versus `213,911,648` (`1.4695712129%`), resident state was `65,632`, TP1/TP8 KV at `1,048,576` tokens was `257,996,881,920 / 32,511,098,880` bytes, and the TP1 target ratio was `24.1118581234x`. All eight HCA pair-helper comparisons passed; the four documented direct tuples reproduced exactly; the stale-name scan found no prohibited current claim; `git diff --check` exited `0`. Step 9 is complete with no production or test source change in this phase.

## 2026-07-16 — Task 7 Step 9 continuation and documentation source audit

- **Motivation:** Resume the interrupted documentation phase without repeating completed implementation work or carrying the superseded aggregate-MLA review forward as current truth.
- **Expectation:** The worktree, Task 7 checklist, source geometry, graph counts, KV arithmetic, and Step 8 numeric evidence match the handoff; only the two existing truth-bearing docs require migration before fresh regression.
- **Method:** Re-read the active plan, progress, issues, requirements, and review records; ran `git status --short`, `git diff --stat`, and `git diff --check`; inspected the current Step4-Pro schema/builders, standard Attention and DeepSeek-V4 HCA formulas, affected tests, and both modeling documents.
- **Result:** The worktree matches the saved checkpoint and `git diff --check` exits `0`. `docs/step4_pro_v1_modeling.md` already describes the reviewed per-layer full/HCA graph. `docs/step4_pro_v1_roofline_model_review.md` still reports the superseded `33`/`28` aggregate-MLA inventories and must be updated to the current `339` context leaves, `332` generation top-level operations, `340` recursive generation nodes, `339` executable generation leaves, and `320 = 20×7 + 60×3` explicit Attention operations per phase. No production or test source was changed during this restoration audit.

## 2026-07-16 — Task 7 differentiated KV GREEN and roofline/integration migration diagnostic

- **Motivation:** Complete the reviewed sequence-aware KV implementation and identify the remaining integration failures before changing any roofline or end-to-end test contract.
- **Expectation:** All 17 differentiated KV cases pass with exact full/SWA/HCA boundaries, TP1/TP8 model totals, nonlinear inverse behavior, and unchanged original-Step4 linear semantics; the existing roofline/integration files should then fail only where they still require removed aggregate `*_mla_approx*` names.
- **Method:** Added config-level full-history, SWA-window, and HCA compressed-history byte formulas in `src/aiconfigurator/sdk/common.py`; added Pro-only nonlinear byte and binary-search capacity paths in `src/aiconfigurator/sdk/models/step4.py`; made the inapplicable constant-elements API fail explicitly while retaining legacy Step4 behavior. Ran the exact 17-test KV selection, targeted Ruff/format/diff checks, then ran `tests/unit/sdk/database/test_step4_pro_v1_roofline.py` with `tests/integration/test_step4_pro_v1_support.py` under Python 3.11.15, pytest 8.4.2, `PYTHONPATH="$PWD/src:$PWD"`, `MPLBACKEND=Agg`, and `TMPDIR=/tmp`.
- **Result:** KV GREEN passed `17/17` in `0.07s`; targeted Ruff and format checks passed after a single semantic `# noqa: TRY004` documented the nonlinear API's intentional `ValueError`, and `git diff --check` exited `0`. The combined roofline/integration diagnostic collected `31` tests and produced `27 passed / 4 failed` in `11.18s` (exit `1`). All four failures were stale name/count assertions filtering `_mla_approx_`, `full_mla_approx`, or `swa_mla_approx`; recursive SOL execution, loader prohibition, existing direct SOL_FULL checks, shared tuple-contract boundary, CLI estimate/generate, and database-mode guards all passed. No production change is indicated by this failure set.

## 2026-07-16 — Task 7 Step 8 query and loader interface mapping

- **Motivation:** Migrate roofline coverage to the operations that the new Pro graph actually executes without guessing query signatures or silently retaining MLA-only loader guards.
- **Expectation:** The repository exposes direct scalar `SOL` and tuple `SOL_FULL` APIs for standard context/generation Attention and analytic DeepSeek-V4 context/generation HCA, while existing tests reveal the exact stale imports and aggregate-name filters.
- **Method:** Searched the SDK for `query_context_attention`, `query_generation_attention`, and DeepSeek-V4 attention-module methods; read their signatures in `src/aiconfigurator/sdk/perf_database.py`, the corresponding operation classes in `operations/attention.py` and `operations/dsv4.py`, and the complete affected roofline/integration helpers. The first retrieval command referenced nonexistent `src/aiconfigurator/sdk/database.py` and exited before viewing the tests; the command was corrected to the actual `perf_database.py` path with no file modifications.
- **Result:** Verified direct query surfaces for all four new Attention/HCA phase variants. A real OSL=1 diagnostic observed exactly `320` scaled context names (`140` full, `180` nonfull), `0` Pro `mla_approx` names, and the explicit generation no-op; OSL=2 observed `320` context plus `320` generation names with the same `140/180` split. The roofline recursive loader tuple and integration loader guard currently include only old MLA classes, while both stale per-op assertions still filter `*_mla_approx*`. Source inspection also found that both HCA query methods eagerly call `load_data()` before their `SOL/SOL_FULL` branches, unlike standard Attention. Step 8 must therefore add the planned RED loader guards before deciding whether a minimal shared HCA query-order correction is required; it must not assume the remaining work is test-only.

## 2026-07-16 — Task 7 Step 8 HCA formula-loader RED

- **Motivation:** Prove independently that both HCA phase APIs violate the formula-only no-loader contract, without relying on graph traversal order or an already-populated `PerfDatabase` LRU cache.
- **Expectation:** New per-layer naming and direct standard/HCA roofline assertions pass against the implemented graph, while the four direct combinations `context/generation × SOL/SOL_FULL` fail exactly at `load_data()` until the query ordering is corrected.
- **Method:** Migrated the Pro roofline and integration imports/guards from MLA to standard Attention plus DeepSeek-V4 HCA; replaced stale aggregate counts with exact `320 = 140 full + 180 nonfull` per-phase checks; added four direct Attention/HCA SOL_FULL component cases; and added a parameterized direct loader guard that clears each PerfDatabase query cache before monkeypatching the relevant HCA loader to raise. Ran the combined two-file suite, then ran only the four direct HCA loader cases with the fixed Python 3.11.15 environment and `TMPDIR=/tmp`.
- **Result:** The combined migrated suite collected `32` tests and reported `28 passed / 4 failed` in `12.17s`; every failure was the new HCA loader guard, while all migrated name/count/direct-roofline assertions passed. The cache-independent direct RED selected four cases and failed `4/4` in `4.47s`: context and generation each failed in both `SOL` and `SOL_FULL`, at `dsv4.py:1085` and `dsv4.py:1492` respectively. This proves the eager call order is the root cause and authorizes the planned minimal query-order fix.

## 2026-07-16 — Task 7 Step 8 HCA loader and per-layer integration GREEN

- **Motivation:** Remove profile-data side effects from the two formula-only HCA paths while preserving empirical/silicon/hybrid loading and complete the migration from aggregate MLA evidence to explicit per-layer full/HCA evidence.
- **Expectation:** The four direct loader cases pass in `SOL` and `SOL_FULL`; the complete migrated roofline/integration suite passes with exact `320 = 140 full + 180 nonfull` phase counts, standard/HCA direct tuple closure, recursive `source=sol`, representative integration names, zero `mla_approx` sources, and offline CLI success.
- **Method:** Moved only the two HCA `cls.load_data(database)` calls below the `SOL` and `SOL_FULL` early returns and immediately before the empirical/silicon/hybrid branches in `src/aiconfigurator/sdk/operations/dsv4.py`. Reran the four direct RED cases first, then the entire roofline and integration files using the same Python 3.11.15 environment and `TMPDIR=/tmp`.
- **Result:** The cache-independent loader selection changed from `4 failed` to `4 passed / 28 deselected` in `3.85s`. The complete suite then collected `36` tests and passed `36/36` in `11.43s` with exit `0`. This includes both phase loaders, four direct standard/HCA Attention queries, three parallel graph shapes, OSL=1 explicit generation no-op, OSL=2 full decode accounting, aggregate/disaggregate execution, and both offline CLI subprocesses. On the simplified `1 TFLOPS / 1 TB/s` fixture, direct `(selected, math, memory)` values in ms were context standard Attention `(51.539607552, 51.539607552, 0.037748736)`, generation standard Attention `(0.012585984, 0.012579840, 0.012585984)`, context HCA `(486.322733056, 486.322733056, 2.268364800)`, and generation HCA `(0.121062400, 0.121062400, 0.047776768)`; every scalar SOL exactly equaled `selected` with `source=sol`. The first fixture lookup command exited early because strict `rg` treated a no-match path set as failure; a corrected search located `tests/unit/sdk/database/conftest.py` without file changes. Targeted `ruff check` passed, while the first format check identified one new test file; applying the project formatter resolved it. Final Ruff check/format and `git diff --check` passed, and the formatted `36/36` rerun passed in `10.91s`.

## 2026-07-16 — Task 7 per-layer graph/report GREEN and KV RED

- **Motivation:** Replace only the Pro temporary MLA boundary while preserving the original Step4 graph, then prove the remaining linear KV implementation is incorrect for the explicit hybrid schema.
- **Expectation:** The graph/report selection passes with 320 unique attention operations per phase, exact TP-sharded full/HCA geometry, explicit HCA AllReduce, deterministic parameter/KV evidence, and unchanged legacy tests; the subsequent KV selection fails only because sequence-aware config/model APIs do not yet exist.
- **Method:** Added a strict `Step4Config | Step4ProConfig` branch in `src/aiconfigurator/sdk/models/step4.py`; independently validated nested full/HCA TP geometry; built 20 per-layer standard-MHA paths and 60 per-layer DeepSeek-V4 HCA paths; derived FFN counts from the same layer tuple; emitted the reviewed numeric warning before graph construction; raised on both >5% parameter-error branches; and populated per-instance semantic-key tuples from constructed operations. Added the four nested divisibility RED cases plus a separate original-Step4 top-level-head regression. After graph GREEN, added config-level full/SWA/HCA KV boundaries at `0/1/511/512/513/1,048,576`, model TP1/TP8 totals, invalid inputs, constant-slope rejection, original-Step4 `52,992` elements/token, and nonlinear inverse boundaries.
- **Result:** The graph/report plus divisibility selection passed `16/16` in `0.09s`; the complete Pro model file passed `77/77` in `0.25s`; original Step4 passed `52/52` in `0.12s`. A first targeted Ruff run found two test-only `SIM300` assertions; swapping expected/actual sides resolved the root cause, after which Ruff check, format check, and diff check passed. The KV RED then selected 17 tests and failed `17/17` in `0.85s`: 13 failures were missing `FullAttentionConfig`/`NonFullAttentionConfig.compute_kv_cache_bytes`, and four exposed the old Pro call into `cfg.kv_lora_rank`. Captured initialization evidence was full `150,994,944 / 153,095,232 / 1.3718833517% PASS`, nonfull `217,055,232 / 213,911,648 / 1.4695712129% PASS`, resident state `65,632`, and KV `257.99688192 GB / 10.7 GB / 24.1118581234x unresolved`.

## 2026-07-16 — Task 7 per-layer graph and parameter-report RED

- **Motivation:** Prove the new full/HCA graph, parameter report, fail-fast mismatch path, dynamic semantic keys, and MTP behavior are absent before changing `Step4Model`, while locking the original Step4 MLA contract.
- **Expectation:** Every Pro graph/report case fails at the current legacy-only `Step4Config` type guard, and the original Step4 exact 14-context/16-generation semantic-name regression passes.
- **Method:** Replaced obsolete Pro `*_mla_approx*` assertions in `tests/unit/sdk/models/test_step4_pro_v1.py` with explicit `20 × 7` full plus `60 × 3` HCA names per phase, TP4 full/HCA geometry, explicit HCA AllReduce, 320-entry per-instance mixed-step keys, warning-report numeric evidence, both >5% parameter-error branches before operation construction, and nextn `0/3` per-layer scaling. Moved the synthetic MLA geometry fixture back to the real original-Step4 schema. Formatted the test file and ran the exact graph/report selection with Python 3.11.15, pytest 8.4.2, `PYTHONPATH="$PWD/src:$PWD"`, `MPLBACKEND=Agg`, and `TMPDIR=/tmp`.
- **Result:** Pytest collected 73 tests, selected 11, and reported `10 failed, 1 passed, 62 deselected in 0.68s` with exit code `1`. All ten Pro cases failed at `src/aiconfigurator/sdk/models/step4.py:59` with `TypeError: Step4Model requires Step4Config extra_params.`, exactly identifying the missing Pro branch. `test_original_step4_attention_graph_remains_legacy_mla` passed, proving the RED suite preserves the original class-level 14/16 semantic-name contract. Ruff format left the file unchanged and `git diff --check -- tests/unit/sdk/models/test_step4_pro_v1.py` passed.

## 2026-07-16 — Evidence and worktree setup

- **Motivation:** Avoid implementing Step4-Pro-V1 on an incorrect source base or contaminating unrelated local work.
- **Expectation:** Establish an isolated branch based on the completed Step4 support and identify a single authoritative CSV.
- **Method:** Inspected Git branches/worktrees/history, historical task artifacts, the complete CSV, Step4 model/config/tests, and created branch `step4-pro` at commit `fdd869b` in a sibling worktree.
- **Result:** Worktree `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro` is clean on `step4-pro`; unrelated untracked files in the original worktree remain untouched.

## 2026-07-16 — Attention closure analysis

- **Motivation:** Prevent invented attention projections or unexplained scaling from violating the CSV and roofline requirements.
- **Expectation:** Determine whether CSV data plus Step4 geometry uniquely defines Full/SWA projection work.
- **Method:** Recomputed standard GQA parameter totals using H=6144, Full Q heads=64, SWA Q heads=96, borrowed KV heads=8, and head_dim=128; compared with CSV weighted attention totals.
- **Result:** Full and SWA totals remain short by 39,849,024 (26.029%) and 50,333,792 (23.530%). The original request's explicit missing-parameter rule therefore selects Step4's temporary MLA treatment with visible human-update documentation.

## 2026-07-16 — Baseline environment failure and root-cause resolution

- **Motivation:** Prove that the completed Step4 baseline is executable before attributing any later failure to Step4-Pro-V1 changes.
- **Expectation:** The declared project environment provides Python, pytest, Ruff, and passes the focused Step4 suites.
- **Method:** First invoked the repository `.venv`; it reproducibly reported Python 3.13.13, `ModuleNotFoundError: pytest`, and a missing Ruff executable. Consulted `task_memory/env_handbook.md`, confirmed the historical verified environment, then reran the identical focused scope with `/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python`, absolute worktree `PYTHONPATH`, `MPLBACKEND=Agg`, and task-local `TMPDIR`.
- **Result:** Root cause was the known unsynchronized `.venv`, not source code. Corrected environment values were Python 3.11.15, pytest 8.4.2, Ruff 0.14.1. Baseline collected 90 tests and passed 90/90 in 7.85 seconds with exit code 0.

## 2026-07-16 — OMX Team clean-workspace gate

- **Motivation:** Start the user-requested parallel read-only audit without allowing workers to diverge from the task source of truth.
- **Expectation:** OMX Team starts from a clean leader worktree and all workers read identical committed requirements.
- **Method:** Ran `omx team 3:architect` after the baseline; the runtime inspected Git state before spawning panes and rejected the eight untracked task documents. Chose to commit the audited task documents rather than stash them, because stashing would remove the requirements from worker visibility.
- **Result:** No workers started and no code changed. The exact error was `leader_workspace_dirty_for_worktrees ... commit_or_stash_before_omx_team`. The task documents were subsequently validated and committed before a successful retry.

## 2026-07-16 — Parallel Team audit and report integration

- **Motivation:** Satisfy the user's parallel-Team requirement while keeping architecture, roofline, and integration evidence independently reviewable before production edits.
- **Expectation:** Three read-only lanes identify model/config assumptions, formula-contract risks, and exact RED/integration coverage without modifying shared production code.
- **Method:** Ran Team `step4-pro-v1-pre-impl-0ddff5cf`; reconciled a legacy decomposition mismatch through owner-corrected tasks 3/6 and superseded fragments 4/5; collected architect payloads; saved `team_roofline_audit.md` and `team_test_integration_audit.md`; committed them as `6667131`; had worker-1 verify required sections, references, hashes, and replacement-task states.
- **Result:** All `6/6` tasks completed with `0` failed. Roofline report SHA256 is `005796a87e7f3874132802461523f58d302d670f3013b42d7514599ce52478ef`; integration report SHA256 is `40fa52a2708209c8def342767ee1a6ba55b5e5d142375bea2a21bb8691977e72`. Team shutdown completed with no worker diffs and Git remained clean.

## 2026-07-16 — Newly grounded implementation boundaries

- **Motivation:** Prevent the implementation plan from promising behavior that the shared Step4 contracts cannot currently provide.
- **Expectation:** Convert every audit anomaly into either an approved minimal fix, an explicit scope boundary, or a human-update item.
- **Method:** Reproduced/inspected SOL_FULL tuple incompatibility, temporary MLA KV arithmetic, parser fallback acceptance, and AFD `dense_swiglu` classification failure; compared each against the original request and no-fallback/root-cause rules.
- **Result:** The revised plan uses SOL for complete graph execution, direct SOL_FULL database assertions for roofline components, fail-fast Step4 config validation, explicit `48.31838208 GB vs 10.7 GB` KV disclosure, and no AFD claim. These decisions now await independent StepCode Claude review before RED tests begin.

## 2026-07-16 — Independent StepCode Claude plan review

- **Motivation:** Obtain a separate cross-model decision before committing to shared parser and model-geometry changes.
- **Expectation:** The reviewer either approves the minimal scope, identifies WATCH mitigations, or returns a BLOCK for user adjudication.
- **Method:** Ran `omx ask claude` with the original objective, authoritative numeric evidence, Team reports, revised plan, and six explicit decisions covering SOL_FULL, parser compatibility, AFD, production files, TDD sufficiency, and blockers.
- **Result:** Verdict `APPROVE`; no BLOCK. Claude approved complete graph execution in SOL, direct SOL_FULL tuple assertions, fail-fast parser correction, and explicit AFD exclusion. It required one additional test contract: original Step4 formulas must explicitly equal `2112`, `24576`, and `32768`. Implementation is authorized to begin in strict plan order.

## 2026-07-16 — Cached identity and authoritative config TDD

- **Motivation:** Prove that the exact Step4-Pro-V1 ID resolves offline and that its cached topology/arithmetic comes from the authoritative CSV.
- **Expectation:** Tests fail only because registration/config are absent, then pass after the minimal cached identity is added.
- **Method:** Added five unit tests before production changes; the first RED run selected four new behavior tests and observed `4 failed / 1 deselected` in `0.27 s`. Added the model ID and config. The first GREEN attempt exposed `81` blocks (`4 dense + 20 Full + 57 SWA`) rather than `80`; removed the single duplicated SWA entry and reran.
- **Result:** Final focused result `5/5 passed` in `0.03 s`. Exact topology is `4 + 20 + 56 = 80`; all dense/MoE/attention/RMS/embedding/activation integer totals close.

## 2026-07-16 — Step4 fail-fast parser TDD

- **Motivation:** Remove silent routed-MoE substitution and Python bool/float coercion at the Step4 config boundary.
- **Expectation:** Every malformed routed-MoE/core integer case fails with a field-specific error while both valid Step4 configs remain accepted.
- **Method:** Added `18` missing/zero/bool/float RED cases before parser edits. All `18/18` failed in `0.39 s`, either through silent acceptance or the wrong downstream error. Added Step4-specific required/positive checks and direct reads of `num_experts_per_tok`, `n_routed_experts`, and `moe_intermediate_size`.
- **Result:** Combined Step4-Pro-V1 and original Step4 model suites passed `75/75` in `0.12 s`; no generic fallback remains on Step4 routed-MoE fields.

## 2026-07-16 — Config-derived projection geometry TDD

- **Motivation:** Remove three hidden single-model assumptions before constructing Step4-Pro-V1 through the shared Step4 graph.
- **Expectation:** A valid synthetic geometry must produce `1312`, `8192`, and `10240` global projection widths instead of original-Step4 literals.
- **Method:** Added a synthetic TP=4 registry/model test before editing `step4.py`. RED failed at actual downscale width `2112` versus expected `1312` (`1 failed / 23 deselected`, `0.18 s`). Replaced only the three derived literals with formulas from model/config fields.
- **Result:** Step4-Pro-V1 plus original Step4 suites passed `76/76` in `0.11 s`; the original formulas are explicitly asserted as `2112`, `24576`, and `32768`. The first `ruff format --check` correctly identified the new test's manual line wrapping; after applying the project formatter, the same `76/76` tests passed in `0.14 s`, Ruff check/format passed, and `git diff --check` was clean.

## 2026-07-16 — Parallel-geometry divisibility TDD

- **Motivation:** Prevent Step4 TP/EP geometry from silently losing heads, vocabulary rows, FFN width, experts, or routed-expert width through integer division.
- **Expectation:** Six non-divisible geometries must fail at the Step4 construction boundary with field-specific `ValueError` messages before `BaseModel` assertions or operation creation.
- **Method:** Added parameterized RED cases for `num_attention_heads % tp_size`, `vocab_size % tp_size`, dense/shared intermediate sizes `% tp_size`, `n_routed_experts % moe_ep_size`, and `moe_intermediate_size % moe_tp_size`. The fresh RED run selected six tests: the head case reached the shared `AssertionError`, and the other five did not raise. Added only Step4-specific checks in `Step4Model.create()` so the shared model contract remains unchanged.
- **Result:** RED was `6 failed / 24 deselected` in `0.22 s`; GREEN was `6 passed / 24 deselected` in `0.04 s`. After project formatting, the combined Step4-Pro-V1 and original Step4 suites passed `82/82` in `0.13 s`; targeted Ruff check/format and `git diff --check` passed.

## 2026-07-16 — Formula-only graph and direct SOL_FULL roofline TDD

- **Motivation:** Prove that every Step4-Pro-V1 operation can be evaluated from auditable formulas without loading profile/LFS data, while preserving the approved Task-level `SOL_FULL` boundary.
- **Expectation:** Complete aggregate/disaggregate graph requests accept only `SOL`/`SOL_FULL`; complete graphs run in `SOL`; each representative direct `SOL_FULL` tuple satisfies `selected == max(math, memory)` and equals the scalar `SOL` result.
- **Method:** Added a dedicated roofline suite covering nine direct database query families, three valid parallel layouts, recursive context/decode operation trees, loader-fail guards, OSL=1/decode attention accounting, and the known Task-level tuple `TypeError`. The first complete Pro roofline run produced `26 passed / 1 failed` in `7.53 s`: only `query_custom_allreduce(SOL_FULL)` returned `(0.44040192, 0, 0)`. Updated the existing base-query expectation first and observed `2 failed` in `4.27 s`, then changed `CustomAllReduce.get_sol()` to classify ring-transfer time as the communication-memory roofline.
- **Result:** The targeted correction passed `2/2` in `4.04 s`; the then-affected database regression passed `106/106` in `18.03 s`. `CustomAllReduce(SOL_FULL)` now returns `(selected, 0, selected)`, consistent with `NCCL`, `P2P`, and the universal roofline invariant. No fallback, scaling factor, or empirical loader was introduced.

## 2026-07-16 — Structural graph completion and affected regression

- **Motivation:** Verify exact CSV composition, Pro projection/FFN/MoE geometry, quant modes, decode overlap branches, temporary KV arithmetic, and MTP scaling before committing the graph/roofline slice.
- **Expectation:** New structural assertions pass for Step4-Pro-V1 without changing original Step4 behavior, and every modified path remains formatted and lint-clean.
- **Method:** Added recursive graph indexing and structural assertions. The first combined run collected `113` tests and reported `5 failed / 108 passed` in `7.71 s`; all failures were the same test-only `AttributeError` because `aiconfigurator.sdk.models` does not export `ops`. Imported `aiconfigurator.sdk.operations as ops`, reran to `113/113 passed` in `7.50 s`, then expanded to all six affected suites. The first Ruff pass identified four `E501` lines and one file requiring formatting; applied only the project formatter to that test file and reran tests/static checks.
- **Result:** Final affected regression passed `192/192` in `18.67 s`. Targeted `ruff check`, `ruff format --check`, and `git diff --check` all exited `0`. The test-only import and formatting corrections did not alter production behavior.

## 2026-07-16 — Offline SDK/CLI integration and provenance documentation

- **Motivation:** Prove that the cached identity works through complete aggregate/disaggregate public SDK paths and real CLI processes without network, LFS, or profile-loader dependence, while making every approximation visible to users.
- **Expectation:** Both SDK modes return finite positive TTFT/TPOT with executed per-op `source="sol"`; offline CLI `estimate` and the requested eight-GPU CLI `generate` resolve the exact model ID; one modeling document separates CSV-backed, derived, and Step4-borrowed values.
- **Method:** Added four integration tests. Network helpers and eight operation profile loaders are monkeypatched to fail for SDK runs; CLI subprocesses set HuggingFace/Transformers offline variables. The first run was `1 failed / 3 passed` in `3.43 s` because the aggregate assertion omitted the public `genonly_step` evidence. After adding it, the next run exposed the intentional zero-latency `generation_attention (not executed)` marker in `mix_step`; updated the assertion to allow only that explicit `not_executed` contract and require every executed source to equal `sol`. Added `docs/step4_pro_v1_modeling.md` with the complete human-update register.
- **Result:** Integration passed `4/4` in `3.23 s`; Pro model/roofline plus new and original Step4 integration passed `66/66` in `10.80 s`. Ruff check/format and `git diff --check` passed. Documentation validation found all `11/11` required provenance/boundary markers in a `16,889`-byte document. No integration RED required a production change.

## 2026-07-16 — Numeric roofline, SDK, and CLI evidence capture

- **Motivation:** Preserve actual values for every required direct roofline family, public aggregate/disaggregate metrics, per-operation source evidence, and both CLI smoke paths rather than relying on qualitative pass/fail claims.
- **Expectation:** Nine direct `SOL_FULL` tuples select `max(math, memory)` and equal scalar `SOL`; public SDK paths produce finite TTFT/TPOT with explicit sources; CLI output exposes its exact sizing and safety boundary.
- **Method:** Queried the same nine shapes used by `test_step4_pro_v1_direct_sol_full_components_close_the_roofline`, ran aggregate/disaggregate `TP=8, PP=2, MoE-TP=8`, and executed both offline CLI subprocesses. The first evidence script failed with `KeyError: 'scheduling'`: aggregate `per_ops_data` contains top-level scheduling metadata while `per_ops_source` intentionally contains only operation phases. A diagnostic run proved `data_only_phases=['scheduling']` with no per-operation key mismatch. The corrected script validates operation-phase sets explicitly and stores scheduling separately; no production code changed.
- **Result:** Direct selected times ranged from `0.002097152 ms` (MLA BMM) to `171.79869184 ms` (context MLA), all with `selected == max(math, memory) == scalar SOL` and `source=sol`. Aggregate observed `TTFT=105.16356682 ms`, `TPOT=1.3241611666666666 ms`; disaggregate observed `TTFT=45.28 ms`, `TPOT=1.43 ms`. The complete `32,593`-byte evidence JSON is task-local at `tests/.tmp/step4_pro_v1_numeric_evidence.json` with SHA256 `ec8d9a1b37f343a56aa4ef52b57e1ebca2f33685eca9f0e4cc25a3ea39582555` and will be transcribed into the final test report, not committed as a product artifact.
- **Result:** CLI estimate matched the SDK aggregate values after display rounding (`TTFT=105.164 ms`, `TPOT=1.324 ms`, memory `91.31 GB`). CLI generate succeeded as an artifact smoke but explicitly reported `fit=False`, required `TP=32` versus maximum `8`, and a generic `1,559,313,383,424`-parameter estimate. The authoritative CSV total is `1,490,676,214,656`; the gap is `68,637,168,768` parameters (`4.604431739983%`). Root cause is the pre-existing generic naive estimator treating all 80 layers as identical routed-MoE layers with one generic attention formula, not MTP or the Step4-Pro-V1 block sequence. The modeling document now records this cross-cutting generator boundary instead of changing it without approval.

## 2026-07-16 — Full unit AF_UNIX failure and environment correction

- **Motivation:** Complete the required repository-wide regression without misclassifying a test-runner environment failure as a product defect.
- **Expectation:** The full unit suite either identifies a reproducible source regression or passes after using the verified environment entry points.
- **Method:** The first run set `TMPDIR="$PWD/tests/.tmp"` and ended with 13 failures in `test_parallel_run.py`; every SyncManager traceback ended at `socket.bind()` with `OSError: AF_UNIX path too long`. Measured the temp base at `81` characters and a representative manager listener at `113`. Ran a controlled single-test comparison with identical code and only `TMPDIR` changed, then exercised the full collector file with both `/tmp` and `/data/ycfeng/tmp` before rerunning all unit tests.
- **Result:** Long-path control: `1 failed` in `0.36 s`, exit `1`. `/tmp` experiment: `1 passed` in `0.24 s`, exit `0`. Collector suite: `22/22 passed` in `6.00 s` with `/tmp` and `22/22 passed` in `7.18 s` with `/data/ycfeng/tmp`. Final full unit: `2063 passed / 12 skipped / 1123 deselected / 4 warnings` in `770.74 s`, exit `0`. Root cause is fully isolated to AF_UNIX pathname length; no code or test change was needed.

## 2026-07-16 — Full static checks and generated-temp scope

- **Motivation:** Prove all delivery files satisfy the repository's lint, formatting, and whitespace contracts while preserving prohibited-to-delete task-local evidence.
- **Expectation:** All tracked source files pass Ruff and Git diff checks; generated pytest fixture copies must not be mistaken for deliverables.
- **Method:** Ran `ruff check .`, `ruff format --check .`, a tracked-file format check over `git ls-files '*.py'`, an explicit `--exclude tests/.tmp` format check, and `git diff --check fdd869b..HEAD`. Inspected every path reported by the only non-zero command.
- **Result:** Ruff lint passed. The broad format command reported only four copied fixtures below untracked `tests/.tmp/`; all `432` tracked Python files passed formatting, the explicit temp-exclusion check passed, and Git diff checks passed. No tracked file required modification. `/usr/bin/time` was separately confirmed absent with exit `127`; pytest-native elapsed time is used in the test report.

## 2026-07-16 — Independent final code review

- **Motivation:** Satisfy the cross-model separation-of-duties gate before delivery and challenge the shared parser/communication changes and support claims.
- **Expectation:** Receive `APPROVE`, record a non-blocking `WATCH`, or stop for user adjudication on `BLOCK`.
- **Method:** Ran `omx ask claude` against StepCode Claude `claude-opus-4-6[1m]` at `effort=max` with the original objective, review range `fdd869b..e4a1083`, all task/modeling docs, changed production/tests, full numeric evidence, and explicit questions about parser scope, topology, formulas, SOL_FULL, CustomAllReduce, offline integration, generator sizing, and documentation truthfulness.
- **Result:** Verdict `APPROVE`; no Critical, no BLOCK, and explicit authorization to proceed to final archival/reporting. The reviewer recorded two low-severity Important observations and three Minor test-maintenance points, with no required code action. Artifact: `.omx/artifacts/claude-act-as-the-independent-final-code-reviewer-for-the-step4-pro-2026-07-15T19-36-24-039Z.md`, `12,336` bytes, SHA256 `15669234b47250e4b9d1f07577f90b942139fbf4eeaa710b4283adad05aec9b2`.

## 2026-07-16 — Follow-up human roofline-review intake

- **Motivation:** The completed modeling document states the support boundary but does not present every operation's FLOPs, memory bytes, algorithm mapping, and correctness verdict in one forward-ordered artifact suitable for manual review.
- **Expectation:** Produce one standalone two-level document covering the complete context/generation graph, with especially detailed Attention and MoE analysis and explicit `PASS`/`CONDITIONAL`/`OPEN` findings.
- **Method:** Added the raw follow-up request to `requirements.md`, appended Task 6 to the existing plan, preserved the completed production scope, and started a source-level audit of every operation implementation and delegated communication path.
- **Result:** Documentation-only analysis is in progress. No production logic, tests, Git history, untracked evidence, or Team state changed.

## 2026-07-16 — Standalone roofline review completion

- **Motivation:** Convert the existing implementation evidence into one human-auditable artifact that exposes every operation formula and distinguishes exact structure from temporary Step4 assumptions.
- **Expectation:** Cover all `33` context operations, all `28` generation top-level operations and `8` nested overlap children; provide exact FLOPs/memory equations, Attention/MoE deep dives, correctness verdicts, numeric closure, and independent approval without changing production logic.
- **Method:** Inspected the cached model, runtime shape propagation, GEMM/MLA/MoE/communication/memory formulas and quant coefficients; instantiated the real TP8/PP2/MoE-TP8 graph; created `docs/step4_pro_v1_roofline_model_review.md`; validated `44` source references and all key arithmetic; ran the three focused suites; requested StepCode Claude `claude-opus-4-6[1m]` at `effort=max`; applied two useful Minor documentation clarifications and independently rejected one stale line-range concern after checking the exact local source.
- **Result:** The document is `44,366` bytes before archival updates and covers `900+` lines, `241` Markdown table rows, `33` context top-level ops, `28` generation top-level ops, `36` recursive generation nodes, and `35` executable generation leaves. Formula/reference/diff checks passed; focused regression passed `65/65` in `13.35 s`; independent verdict was `APPROVE` with `0` Critical, `0` Important, and `3` Minor findings. No production or test code changed. Untracked `tests/.tmp/` remains preserved.

## 2026-07-16 — Hybrid-attention refactor intake and upstream synchronization

- **Motivation:** The new attachment supersedes the prior temporary assumption that all 80 Step4-Pro-V1 attention layers can reuse the Step4 MLA geometry and requests evidence from the latest DeepSeek-V4 vLLM SWA/HCA implementation.
- **Expectation:** Preserve the completed initial support while replacing only the unfaithful attention boundary with an explicit 20-full/60-non-full per-layer model, differentiated KV-cache accounting, parameter closure, fail-loud validation, and complete regression evidence.
- **Method:** Read the attached specification first; restored the existing task documents and worktree state; inspected both repository statuses; fetched `origin` in `/data/ycfeng/stepfun-performance-optimization/aiconfigurator`; fast-forwarded the local `main` reference only after verifying ancestry; and began source-level tracing of the DeepSeek-V4 model, operation, collector, config, and tests at `origin/main`.
- **Result:** The reference repository's local `main` and `origin/main` both resolve to `f4c58458ab1554c3e7678492e1bc9c7812c678e6` (`2026-07-16T09:33:41+08:00`). The working tree remains on `step-design`, so its unrelated untracked files were not touched. The attachment requirements are captured in `requirements.md`; production code has not yet changed.

## 2026-07-16 — Hybrid-attention code mapping and candidate arithmetic

- **Motivation:** Establish the exact root cause and a reviewable minimum design before modifying tests or production code.
- **Expectation:** Identify every uniform-MLA boundary, preserve the authoritative 80-layer order, find reusable HCA primitives without inheriting an upstream fallback, and produce parameter/KV formulas whose assumptions and gaps are explicit.
- **Method:** Traced `Step4Config`, the Step4 parser, both attention graph builders, base KV APIs, `HybridMoEConfig`, DeepSeek-V4 model/operation code, the cached Pro JSON, and affected unit/integration/roofline tests. Recomputed the candidate full-MHA and non-full-HCA parameter totals and evaluated full/SWA/HCA cache curves at the CSV 1M-token point. The first combined documentation patch failed atomically because one issue-history anchor was stale; verified that it made no partial edit, then applied exact per-file patches.
- **Result:** The candidate full config estimates `150,994,944` parameters (`1.3718833517%` below target); the candidate HCA config estimates `217,055,232` (`1.4695712129%` above target). The exact layer order is `0-3 nonfull+dense`, `4-23 full+moe`, `24-79 nonfull+moe`. The candidate full-history KV model totals `257.99688192 GB` with HCA, versus the `10.7 GB` CSV target, exposing a `24.1118581234x` evidence conflict that cannot be fixed without unconfirmed full-cache details. Requirements supersession, notes, issues, and progress now reflect the follow-up truth. No production or tracked test file changed; next gate is the detailed Task 7 plan plus independent StepCode Claude verdict.

## 2026-07-16 — Independent Task 7 architecture review

- **Motivation:** Obtain a separate-model decision on the hard-to-reverse full/HCA schema, parameter definition, per-layer operation graph, and honest KV-conflict boundary before any behavior change.
- **Expectation:** Receive `APPROVE`, record actionable `WATCH` risks, or stop for user adjudication on `BLOCK`.
- **Method:** Ran the required local StepCode Claude `claude-opus-4-6[1m]` advisor at `effort=max`. A small diagnostic proved the channel works. The first oversized review returned no artifact, and the second repository-heavy invocation timed out after `300 s` (`RC=124`); the third attempt used a shorter self-contained decision packet, retained the same model/effort/provider, and completed successfully.
- **Result:** Verdict `APPROVE`, `0` Critical findings, no BLOCK. Adopted the reviewer's matrix-only parameter definition (`217,055,232`) with `65,632` separately named resident-state elements; added explicit HCA `CustomAllReduce`, warning-level KV-conflict reporting, and a no-double-MTP regression to Task 7. Artifact: `.omx/artifacts/claude-you-are-the-independent-stepcode-claude-reviewer-read-only-r-2026-07-16T08-53-12-302Z.md`, `12,729` bytes, SHA256 `c0fa487fbe3a643a1e0178036d857349395f398570d84969e8ab4666843735a6`. Production and tracked tests remain unchanged; RED tests are next.

## 2026-07-16 — Task 7 plan self-review and fresh baseline

- **Motivation:** Prove the revised plan has no placeholder gaps and establish a current-tree passing baseline before changing tests under TDD.
- **Expectation:** No plan placeholder patterns or whitespace defects; all affected existing Pro/legacy model, roofline, and integration tests pass before RED edits.
- **Method:** Scanned `plan.md` for forbidden placeholder phrases, ran `git diff --check` on the task directory, then executed the four affected test files in the verified Python 3.11.15 conda environment with `PYTHONPATH="$PWD/src:$PWD"`, `MPLBACKEND=Agg`, and `TMPDIR=/tmp`.
- **Result:** Placeholder scan found `0` matches; documentation diff check exited `0`; focused baseline passed `117/117` tests in `13.14 s` with exit `0`. Production and tracked tests were still unchanged at the baseline boundary.

## 2026-07-16 — Task 7 schema and parameter-count RED

- **Motivation:** Prove that the new explicit Pro layer identity, independent attention configs, and exclusive parser contract are absent before implementing them.
- **Expectation:** The selected tests fail because the cached Pro config still uses the legacy `block_types` schema, the parser still accepts mixed schemas, and the new dataclass-backed fields are unavailable.
- **Method:** Added focused tests for the exact 80-layer order, full/HCA parameter arithmetic, mixed and partial schema rejection, malformed layer identity, boolean geometry, unsupported projection mode, unsupported compression ratio, and zero output groups. Ran the 13 selected cases in the verified Python 3.11.15 conda environment before changing `common.py`, `utils.py`, or the cached JSON.
- **Result:** The required RED run exited `1` with `13 failed in 0.69 s`. Failures matched the intended missing behaviors: `common.Step4ProConfig` was absent, the parsed config exposed no `full_attention`, mixed legacy/Pro schema input did not raise, and partial or malformed Pro inputs fell through to the legacy `missing block_types` error. No test failed from syntax, collection, or environment setup.

## 2026-07-16 — Task 7 extended parser-boundary RED

- **Motivation:** Cover every planned fail-fast parser branch before implementing the exclusive schema parser rather than leaving untested validation logic in production.
- **Expectation:** The additional cases fail against the legacy-only parser with the old `missing block_types` diagnostic.
- **Method:** Added separate tests for an absent schema, non-list/wrong-length/malformed layer records, non-mapping or incomplete nested configs, negative unknown fields, non-null full latent rank, incompatible full dimensions/head ratios, unsupported non-full mechanism/ratio pairings, enabled indexer fields, and oversized RoPE. Ran only these four parameterized test functions with the verified conda Python and `TMPDIR=/tmp`.
- **Result:** The RED run collected `20` cases and exited `1` with `20 failed in 1.18 s`. Every failure reached the existing parser and returned `Step4 config missing required field 'block_types'`, proving the new field-specific validation branches were not already implemented; collection and environment setup were clean.

## 2026-07-16 — Task 7 schema/parser/cached-config GREEN

- **Motivation:** Replace only the Pro legacy-MLA schema boundary while preserving the original Step4 parser and its public error behavior.
- **Expectation:** The cached Pro config parses into frozen explicit layer/full/HCA dataclasses; all original and extended schema cases pass; the complete legacy Step4 model test file remains unchanged and green.
- **Method:** Added `Step4LayerSpec`, `FullAttentionConfig`, `NonFullAttentionConfig`, and `Step4ProConfig`; implemented exclusive legacy/Pro schema selection plus field-specific nested validation; rewrote the cached Pro JSON in place to 80 explicit layer records and nested attention configs; ran the original 13-case selection, the separate 20-case extension, the complete legacy Step4 suite, targeted Ruff, format, and whitespace checks.
- **Result:** The original selection passed `13/13` in `0.10 s`. The first extended GREEN attempt reached `19/20` but one test fixture raised `KeyError: 'block_types'` after the intended JSON migration; root cause was stale test setup, so the case was corrected to remove all legacy and Pro schema entry fields and then passed `20/20` in `0.12 s`. The final combined schema matrix passed `33/33` in `0.09 s`, and legacy Step4 passed `52/52` in `0.08 s`. Cached topology measured `80` layers, `20` full, `60` non-full, `4` dense, and `76` MoE, with `0` legacy Pro attention keys. Parameter estimates were `150,994,944` versus `153,095,232` (`1.3718833517%`) and `217,055,232` versus `213,911,648` (`1.4695712129%`), with `65,632` resident-state elements. The first static run found four intentional `ValueError`/`TRY004` conflicts and one long test line; targeted `noqa` annotations preserved the required parser exception contract, Ruff formatting fixed layout, and the final Ruff check, format check, and `git diff --check` all exited `0`.
