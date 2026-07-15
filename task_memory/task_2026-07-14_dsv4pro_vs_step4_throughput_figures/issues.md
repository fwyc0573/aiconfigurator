## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Recorded infeasible-point plotting issue and root-cause resolution |
| 2026-07-14 | Closed GB300 runtime and final four-system plotting issues with evidence |
| 2026-07-14 | Recorded and resolved final audit-harness assumption errors |
| 2026-07-15 | Recorded and resolved the headless Matplotlib verification block |
| 2026-07-15 | Resolved omission of the completed-system interim menu from the durable archive inventory |
| 2026-07-15 | Closed the final-v2 monitor lifecycle and annotation pairwise-test failure |
| 2026-07-15 | Recorded and resolved the non-GB300 menu validator display-text assumption |
| 2026-07-15 | Recorded the stale corrected-v2 execution contracts and current-source v3 resolution |
| 2026-07-15 | Resolved the GB300 v3 rank-one TTFT audit population mismatch |

# Issues and Resolutions

## Corrected-v2 checkpoints no longer match the current shared-runner contract

- **Status:** Resolved through fresh immutable v3 execution and full audit.
- **Observed evidence:** The throughput wrapper hashes both itself and `run_step4_comparison.py`. The current per-system throughput contract hashes differ from every stored corrected-v2 checkpoint header after the shared runner was reformatted.
- **Root cause:** Strict checkpoint provenance includes source bytes, so a style-only change to the shared runner correctly changes the execution contract even when the scientific behavior is unchanged.
- **Resolution:** Did not rewrite headers, copy ratios, or introduce a style-only exception. Re-ran H100/H800/H200 into `full-v3-*` directories and published the non-GB300 menu first. Used the second allowed compute lane for GB300 throughput while the independent primary job continued untouched, then published the four-system v3 menu.
- **Acceptance:** PASS. Every v3 header matches its current-source contract and Git HEAD. The final audit covers `64/64` identities and `256/256` terminal outcomes, recomputes `46/46` ratios with maximum absolute delta `0.0`, preserves all `9` missing points, and labels H800 as simulated SOL. All four v2/v3 scientific payloads are identical after normalizing only the contract hash.

## GB300 v3 TTFT audit compared a rank-one expectation against all ranked rows

- **Status:** Resolved in the one-off validator; no artifact change was needed.
- **Observed evidence:** The first GB300 v3 audit passed contract, HEAD, checkpoint, artifact existence, and v2/v3 scientific identity, then failed a late maximum-TTFT assertion.
- **Root cause:** The validator compared the known rank-one maximum `4176.999 ms` against all `58` ranked rows. One lower-ranked but still eligible candidate has TTFT `4200.186 ms`; this does not violate strict `TTFT < 5000 ms` and is not the selected rank-one evidence.
- **Resolution:** Inspected the authoritative rank distribution and `ttft_pass` flags, restricted the reported maximum assertion to `rank == 1`, retained a separate strict-TTFT assertion over all ranked rows, and reran the complete audit from contract construction.
- **Result:** Corrected audit PASS: mode runs=`16/16`, ranked/rank-one rows=`58/16`, all ranked rows strictly below `5000 ms`, rank-one maximum=`4176.999 ms`, v2/v3 scientific identity exact, and five current-source artifact hashes recorded. No source, checkpoint, JSON, CSV, Markdown report, or PNG was modified.

## Infeasible matrix points must not be converted into fabricated ratios

- **Status:** Resolved and verified in the final four-system artifact.
- **Observed evidence:** The completed H100 shard contains no rank-one DeepSeek-V4-Pro row for some long-ISL points because the search ended with memory/SLA infeasibility. The plotting function previously called `compute_throughput_ratios(..., require_complete=True)` and failed on the first unpaired point.
- **Root cause:** The experiment matrix is intentionally broader than the feasible hardware/configuration region, while the plotting layer assumed every matrix point had two successful model rows. This is a data-contract mismatch, not a numerical ratio error.
- **Fix motivation:** Preserve fail-fast validation for direct ratio computation, but allow figure generation to represent only paired feasible points and record missing model/system/ISL combinations explicitly.
- **Method:** Added an opt-in `allow_unpaired` path used only by figure generation, skipped missing points in plotted lines (matplotlib gaps), and wrote `missing_ratio_points`, `paired_ratio_point_count`, and `missing_ratio_point_count` to `combined_results.json`. Added a regression test reproducing the H100-style unpaired point.
- **Result:** The new regression test passes; direct `compute_throughput_ratios` still raises by default for missing model rows, so accidental incomplete data use remains fail-fast. Final coverage contains 23 paired points and 9 explicit missing points, totaling all 32 requested system/ISL points.

## Environment package resolution

- **Status:** Resolved before implementation validation.
- **Root cause:** The checkout `.venv` is unavailable and the default `PYTHONPATH=.` resolves an unrelated editable AIC package.
- **Resolution:** Use `/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python` with `PYTHONPATH=src:.`; the workaround is recorded in `task_memory/env_handbook.md`.

## GB300 SOL shard has unusually long data-processing time

- **Status:** Resolved by normal completion; no restart or workaround was required.
- **Evidence:** The GB300 worker has remained at mode run 1/16 for roughly 40 minutes with CPU usage near 100% and no error output. Read-only `gdb` samples showed the main thread in `pyarrow.ChunkedArray.Equals`, NumPy strided-copy code, and Python object conversion; OpenBLAS helper threads are alive and waiting. The checkpoint remains valid but has no completed records yet.
- **Root-cause assessment:** This is a data-loading/comparison cost in the GB300 SOL path, not a deadlock, failed checkpoint, or SLA rejection. H200 and H800 complete the same 16-run matrix normally under the same runner contract.
- **Action and result:** The process was left untouched and completed all 16 mode runs after approximately 2 hours 21 minutes. The final shard contains 58 successful normalized rows, 6 memory-infeasible terminal outcomes, 16 rank-one rows, and a maximum rank-one TTFT of `4176.999 ms`.

## Completed-system plotting while GB300 is pending

- **Status:** Resolved for both interim and final deliverables.
- **Root cause:** The plotting code originally iterated over all four configured systems, which would add an empty GB300 line when only H200/H100/H800 shards were available.
- **Fix motivation:** The interim menu must describe only systems backed by completed result shards.
- **Method:** Figure generation now derives the system list from rank-one rows and stores it in the combined-result summary. It still retains the fixed four-system validation behavior for direct complete-ratio checks.
- **Result:** `results/figures_completed/` preserves the H200/H100/H800 interim menu. `results/figures_final/` now contains the GB300/H200/H100/H800 figures, combined results, and final README menu.

## Final audit harness used incorrect inventory and identity assumptions

- **Status:** Resolved; the corrected full audit passes.
- **Observed evidence:** The first completion-audit run stopped because it expected `31` SHA256 inventory rows while `summary.md` correctly contains `35`. The second run stopped because it treated `canonical_config_id` as globally unique and found only `44` unique values among `50` rank-one rows.
- **Root cause:** The first assertion omitted four inventory entries when manually totaling the sections. The second assertion misunderstood `canonical_config_id`: it identifies a parallel configuration and can legitimately repeat across model/system/ISL groups.
- **Resolution:** Derived the inventory total explicitly as `3 workflow + 4 final-menu + 20 shard + 8 task-document = 35`. Compared rank-one provenance with the complete identity `(model, system, isl, canonical_config_id)`, which yields `50/50` unique rows.
- **Result:** The corrected audit passed from the beginning through all checks: `35/35` hashes, `64/64` identities, `256/256` terminal outcomes, `50/50` rank-one provenance rows, and maximum ratio absolute delta `0.0`. These were audit-script defects only; no experiment, source, figure, or result artifact required modification.

## RGBA preview showed black regions for the decode figure

- **Status:** Resolved as a preview-layer issue; the stored PNG is valid.
- **Observed evidence:** One direct RGBA preview displayed black blocks around parts of the decode plot.
- **Root cause:** The preview renderer mishandled that RGBA rendering path. Pixel inspection showed all `2,721,600` alpha values are exactly `255`, with white corner/background pixels; an RGB conversion displayed the complete figure normally.
- **Resolution:** Verified the original file with Pillow and `file`, then visually inspected an RGB-rendered copy under `/tmp` without modifying the deliverable.
- **Result:** The original decode PNG remains `2160 × 1260`, RGBA, fully opaque, `204025` bytes, and SHA256 `da6244b5221f991e1f3ef03f8c72000312132ee4c0a65ed86008efa95b9f8118`.

## Headless verification blocked on an unreachable SSH display

- **Status:** Resolved without product-code or result-artifact changes.
- **Observed evidence:** A fresh pytest invocation collected the expected 173 tests but did not report results. The process remained in `poll()` with no child process. A read-only `gdb` backtrace placed the main thread in Matplotlib `mpl_display_is_valid()` calling `XOpenDisplay()` through XCB.
- **Root cause:** The shell retained `DISPLAY=localhost:17.0`, but that forwarded X11 endpoint was not responsive. Matplotlib therefore blocked while probing an interactive display before the plotting tests ran.
- **Resolution:** Terminated only the fresh validation process started for this checkpoint, then reran the same suite with the explicit non-interactive backend `MPLBACKEND=Agg`. Added the verified recipe to `task_memory/env_handbook.md`.
- **Result:** The rerun passed `173/173` tests in `3.47 s` with exit code `0`. No GB300 worker, monitor, source, checkpoint, JSON, CSV, or figure artifact was modified.

## Fresh menu audit assumed the wrong source-count field name

- **Status:** Resolved in the audit command; no deliverable change was needed.
- **Observed evidence:** The first current-file contract audit raised `KeyError: 'source_result_count'` after Ruff passed.
- **Root cause:** The one-off audit script invented `source_result_count` instead of reading the persisted `summary` schema. The actual canonical field is `source_count`.
- **Resolution:** Printed both current `summary` objects, replaced the incorrect audit assertion with the exact stored schema, and reran the complete JSON/PNG/inventory audit.
- **Result:** The corrected audit passed: completed menu `3` sources / `15` paired / `9` missing, final menu `4` sources / `23` paired / `9` missing, four PNGs valid and fully opaque, and `35/35` inventory hashes matched. Product JSON and plotting code were not modified.

## Completed-system interim menu was not listed in the durable archive inventory

- **Status:** Resolved before branch archival.
- **Observed evidence:** `results/figures_completed/` contained the requested README, combined JSON, and two PNG figures, but `summary.md` listed only the four final-menu files. Because every `results/` directory is ignored by the repository-wide `.gitignore`, an unlisted interim artifact could be omitted from an otherwise precise staging operation.
- **Root cause:** The original completion inventory treated the final four-system menu as the sole durable figure product. The later temporary-priority requirement made the earlier three-system snapshot a separate user-facing deliverable, but the inventory contract was not expanded at that time.
- **Resolution:** Added the four exact completed-system artifacts, both required Python package markers, and their SHA256 values to `summary.md`; updated the test report and progress record; and explicitly excluded nested `.omc` runtime state plus partial/smoke outputs from archival staging. The package markers retain both the target worktree's SPDX headers and the feature worktree's descriptive docstrings.
- **Result:** A fresh inventory parser verified `41/41` hashes. The interim menu remains `3` sources / `15` paired / `9` missing, and the final menu remains `4` sources / `23` paired / `9` missing. No scientific result or plotting behavior changed.

## Final-v2 post-processing monitor used an unstable detached-shell form

- **Status:** Resolved; the persistent monitor completed and exited.
- **Observed evidence:** The first inline detached monitor failed because nested shell/Python quoting corrupted its generated expression. A corrected detached `nohup` invocation then failed to survive the command-runner process group.
- **Root cause:** The original monitor combined several quoting layers and assumed detached children would outlive the command-runner process group. Those were monitor-lifecycle assumptions, not experiment-runner or result-data defects.
- **Resolution:** Used a syntax-stable temporary script under `/tmp` and ran it in persistent exec session `8371`. The monitor required GB300 process termination, exactly `16` durable SQLite records, and an existing `results.json` before plotting; otherwise it failed explicitly.
- **Result:** The monitor generated `figures_final_v2/` only after GB300 completed successfully. Session `8371` has exited, and the final menu contains the expected four artifacts. No fallback result or partial GB300 output was accepted.

## Annotation-gap test used strict zip on intentionally unequal sequences

- **Status:** Resolved and fully regressed.
- **Observed evidence:** The fresh 177-test run reported `1 failed, 176 passed`; `test_annotation_positions_enforce_minimum_gap_without_center_drift` raised `ValueError: zip() argument 2 is shorter than argument 1`.
- **Root cause:** Adjacent-pair construction used `zip(adjusted, adjusted[1:], strict=True)`. A sequence and its one-element-shorter tail are intentionally unequal, so strict zip must fail before evaluating the minimum-gap condition.
- **Resolution:** Replaced the invalid construction with `itertools.pairwise(adjusted)`, which directly expresses adjacent-pair iteration and satisfies Ruff rule `RUF007`.
- **Result:** The single regression test passes `1/1`; the full affected suite passes `177/177` in `3.49 s`; Ruff check and format check pass. No production plotting logic, ratio, checkpoint, or figure data changed for this test-only correction.

## Non-GB300 menu validator assumed title capitalization

- **Status:** Resolved in the one-off validator; no source or result artifact changed.
- **Observed evidence:** Five focused plotting tests passed, but the following JSON assertion failed because it expected `Step4 / DeepSeek-V4-Pro`.
- **Root cause:** The persisted `combined_results.json` schema records the ratio direction as `step4 / DeepSeek-V4-Pro`. The validator copied the human-facing capitalization from the README instead of reading the exact serialized value.
- **Resolution:** Printed the current summary, retained all numeric and system assertions, changed only the one-off expected string to the stored value, and reran the complete five-test plus JSON/PNG validation sequence.
- **Result:** The rerun passed `5/5` tests and `NON_GB300_MENU_VALIDATION=PASS`; source count=`3`, rank-one rows=`34`, paired/missing points=`15/9`, ratio values=`30`, and both PNGs=`2160 × 1260`.
