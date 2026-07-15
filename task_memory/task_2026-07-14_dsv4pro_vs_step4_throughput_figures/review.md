## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Recorded independent Claude implementation review and WATCH remediation |
| 2026-07-14 | Reconciled the strict TTFT SLA wording across requirements and plan |
| 2026-07-14 | Added final four-system evidence review and APPROVE verdict |
| 2026-07-14 | Added the final completion-gate audit and corrected audit assumptions |
| 2026-07-15 | Added the corrected-v2 final artifact, scientific-data, and regression review |
| 2026-07-15 | Added the annotation-spacing, adjacent-pair test repair, and archive closure review |
| 2026-07-15 | Added the current-source v3 provenance and all-GPU completion review |

# Review: DSV4-Pro vs Step4 Throughput Figures

## 2026-07-14 Independent Implementation Review

- **Target Component/Phase:** Phase 1 runner, ranking semantics, shard combination, ratio computation, and figure generation.
- **Reviewer Agent Identity:** StepCode Claude Opus 4.6 via `omx ask claude`; artifact `.omx/artifacts/claude-review-the-approved-deepseek-v4-pro-vs-step4-throughput-figu-2026-07-14T07-26-43-387Z.md`.
- **Inspected Artifacts:** `requirements.md`, `plan.md`, `notes.md`, `run_dsv4pro_vs_step4_throughput.py`, `plot_throughput_ratio.py`, and `test_dsv4pro_vs_step4_throughput.py`.
- **Identified Issues/Anomalies:** Verdict `APPROVE` with three WATCH items: (1) strict `<5000 ms` differs from the original `<=5000 ms` wording; (2) scoped module-global override is intentionally single-process/thread-unsafe; (3) no direct test drove `workload_kind="throughput"` through the parent final ranking validator. Additional low-risk gaps covered non-disagg rejection, duplicate rank-one rows, and approximation metadata propagation.
- **Remediation/Verification Code Actions Taken:** The user explicitly approved strict `<5000 ms`; this is now documented in the ranking contract. The module-global override remains bounded to one single-threaded process, while concurrent system shards run in separate OS processes. Added tests for full parent ranking-pipeline acceptance, non-disagg rejection, duplicate rank-one rejection, and Step4 approximation metadata. Focused suite now passes 9/9; Ruff check and format check pass.

## 2026-07-14 Interim Menu Review

- **Target Component/Phase:** Completed-system figure menu and system-subset plotting.
- **Reviewer Agent Identity:** `/root` execution verification.
- **Inspected Artifacts:** `plot_throughput_ratio.py`, `test_dsv4pro_vs_step4_throughput.py`, H200/H100/H800 `results.json`, and `results/figures_completed/README.md`.
- **Identified Issues/Anomalies:** GB300 is intentionally absent from the interim menu because its fresh shard is still running. Nine paired ratio points are unavailable because the corresponding model/configuration searches ended in memory/SLA infeasibility.
- **Remediation/Verification Code Actions Taken:** Plotting now derives colored lines from supplied rank-one rows and records missing points instead of fabricating ratios. Focused tests pass 11/11, parent regression tests pass 162/162, and Ruff checks pass.

## 2026-07-14 Final Evidence Review

- **Target Component/Phase:** GB300 shard completion, four-system merge, final figures/menu, strict TTFT enforcement, numeric ratios, and completion documentation.
- **Reviewer Agent Identity:** `/root` final verifier using current-worktree artifacts and fresh commands.
- **Inspected Artifacts:** All four shard `results.json`/CSV/report/checkpoint files; `results/figures_final/combined_results.json`; both final PNGs; final README; runner, plotter, unit tests, requirements, plan, issues, and progress documents.
- **Identified Issues/Anomalies:** Nine long-ISL system/model pairings are unavailable because their searches ended in memory/SLA infeasibility; H800 remains simulated SOL rather than silicon measurement. No fabricated values, identity gaps, strict-TTFT violations, ratio-polarity errors, corrupt images, or regression failures were found.
- **Remediation/Verification Code Actions Taken:** Preserved gaps and the explicit missing-point inventory; retained H800 provenance in all final docs; validated all 64 mode-run identities, four manual ratio computations with `0.0` absolute delta, maximum TTFT margins, PNG integrity, 173 regression tests, and Ruff check/format. Final verdict: `APPROVE`.

## 2026-07-14 Final Completion Gate

- **Target Component/Phase:** Summary inventory, full execution contract, combined rank-one provenance, all stored ratios, figure/menu integrity, and final regression evidence.
- **Reviewer Agent Identity:** `/root` completion verifier using fresh current-worktree commands.
- **Inspected Artifacts:** `summary.md` inventory; runner and base 21-config materialization; all four shard `results.json` files and logs; final combined JSON, PNGs, and menu; focused and parent performance tests; Ruff output.
- **Identified Issues/Anomalies:** Two audit-harness assumptions were invalid: the inventory contains `35`, not `31`, entries; `canonical_config_id` is not globally unique. One RGBA preview rendered black blocks despite fully opaque PNG pixels. None was a product-data defect.
- **Remediation/Verification Code Actions Taken:** Corrected the inventory total; used `(model, system, isl, canonical_config_id)` for rank-one provenance; verified all alpha samples equal `255` and inspected the RGB rendering; reran the complete audit, pytest, and Ruff. Evidence: `35/35` hashes, `64/64` identities, `256/256` terminal outcomes, `50/50` source-provenance matches, maximum ratio delta `0.0`, `173/173` tests in `2.89 s`, Ruff `0` errors. Final verdict: `APPROVE`.

## 2026-07-15 Corrected-v2 Final Evidence Review

- **Target Component/Phase:** Fresh v2 GB300 completion, corrected four-system figures/menu, source provenance, strict TTFT, ratio recomputation, annotation layout, and affected regression.
- **Reviewer Agent Identity:** `/root` execution verifier using current-worktree v2 artifacts and fresh commands.
- **Inspected Artifacts:** Four `full-v2-*` shard `results.json`/CSV/report/SQLite files; `figures_completed_v2`; `figures_final_v2`; runner, plotter, focused tests, and fresh logs under `/tmp/step4_throughput_v2_final_*_20260715.log`.
- **Identified Issues/Anomalies:** The first affected-suite run found a test-only strict-zip `ValueError`; the first audit iterations assumed incorrect SQLite, ratio-container, identity, per-system-contract, and display/source-system schemas. Nine long-ISL pairings remain unavailable by explicit memory/SLA outcomes. H800 remains simulated SOL.
- **Remediation/Verification Code Actions Taken:** Replaced the invalid adjacent-pair assertion with `itertools.pairwise`; inspected authoritative persisted schemas and corrected only the audit harness; recomputed all 46 ratios; checked all 64 identities, 256 outcomes, and 50 rank-one provenance rows; validated both PNGs; reran Ruff and 177 tests. Evidence: maximum ratio delta `0.0`, `177/177` passed in `3.49 s`, Ruff `0` errors, format clean, and `git diff --check` exit `0`. Final figure-task verdict: `APPROVE`, pending only branch-level inventory/commit/integration gates.

## 2026-07-15 Corrected-v2 Layout and Archive Closure Review

- **Target Component/Phase:** Collision-aware annotation placement, X-axis edge margin, adjacent-pair unit assertion, corrected-v2 menu inventory, and final verification evidence.
- **Reviewer Agent Identity:** `/root` completion verifier using fresh current-worktree commands; prior independent corrected-v2 review verdict retained as `APPROVE`.
- **Inspected Artifacts:** `plot_throughput_ratio.py`; `test_dsv4pro_vs_step4_throughput.py`; both PNGs and `combined_results.json` in `figures_completed_v2` and `figures_final_v2`; four `full-v2-*` shards; `progress.md`; test report; and the 41-file `summary.md` inventory.
- **Identified Issues/Anomalies:** Fixed vertical annotation offsets could not prevent collisions across all ISL/system combinations. The first minimum-gap test used `zip(..., strict=True)` on sequences of lengths `N` and `N-1`, guaranteeing a test-only `ValueError`. Nine long-ISL ratios remain explicitly missing due to memory/SLA infeasibility, and H800 remains simulated SOL. No scientific-data, ratio-polarity, provenance, image-integrity, or background-worker defect was found.
- **Remediation/Verification Code Actions Taken:** Annotation positions are now sorted by actual ratio and spread in Matplotlib display-pixel coordinates with a minimum `18 px` gap while preserving their center; log-scale X limits add symmetric edge space. The adjacent-pair assertion now uses `itertools.pairwise`. Fresh evidence is `177/177` tests in `3.15 s`, Ruff check `0` errors, Ruff format `3` files already formatted, `git diff --check` exit `0`, `46/46` ratios with maximum absolute delta `0.0`, and two fully opaque `2160 × 1260` final PNGs. Final corrected-v2 deliverable verdict: `APPROVE`.

## 2026-07-15 Current-Source v3 Provenance and Final Figure Review

- **Target Component/Phase:** Four current-source throughput shards, non-GB300-first menu, all-GPU final menu, strict execution-contract provenance, v2/v3 scientific identity, TTFT, ratio, and image integrity.
- **Reviewer Agent Identity:** `/root` execution verifier using fresh current-worktree v3 artifacts; branch-level independent pre-commit review remains recorded in the parent task.
- **Inspected Artifacts:** All five files in each `full-v3-*` shard; `figures_completed_v3`; `figures_final_v3`; current runner and plotter sources; v2 comparison artifacts; checkpoint headers; combined rank-one rows, ratios, missing-point inventory, and both final PNGs.
- **Identified Issues/Anomalies:** The formatter changed shared-runner bytes and therefore invalidated strict v2 source provenance without changing scientific values. The first GB300 audit compared a rank-one TTFT maximum against all ranked rows; this was a validator population error. Nine long-ISL ratios remain explicitly unavailable because of memory/SLA infeasibility. H800 remains simulated SOL.
- **Remediation/Verification Code Actions Taken:** Re-ran all four shards into immutable v3 directories without rewriting provenance; computed and checked each current contract; normalized only the execution-contract hash for exact v2/v3 equality; corrected only the audit row selection; generated both v3 menus; and audited all `64/64` identities, `256/256` terminal outcomes, `50` rank-one rows, `23/9` paired/missing points, `46/46` ratios with maximum delta `0.0`, strict TTFT maxima, and two fully opaque `2160 × 1260` PNGs. Both PNGs are byte-identical to v2. Verdict: `APPROVE`.
