## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Task initiated, grilling complete, docs landed |
| 2026-07-14 | Recorded final GB300 completion, four-system artifacts, and validation evidence |
| 2026-07-14 | Recorded the final inventory, contract, visual, and regression completion audit |
| 2026-07-15 | Revalidated the interim and final menus after the user's temporary-priority checkpoint |
| 2026-07-15 | Promoted both figure menus and package markers into the durable 41-file archive inventory |
| 2026-07-15 | Regenerated the corrected v2 completed-system menu, separated annotations, and armed final GB300 post-processing |
| 2026-07-15 | Completed the corrected GB300 v2 shard, generated the four-system v2 menu, and closed the fresh validation failure |
| 2026-07-15 | Recorded fresh closure evidence and confirmed the independent full-matrix GB300 worker remains active |
| 2026-07-15 | Revalidated the non-GB300 menu while the formatter-consistent full-matrix shards continued in the background |
| 2026-07-15 | Started the current-source v3 refresh while preserving the independent GB300 background run |
| 2026-07-15 | Completed all four current-source v3 shards, final figures, and strict scientific-identity audit |

# Progress: DSV4-Pro vs Step4 Throughput Figures

## Status: Complete — current-source v3 figures published; branch integration tracked by the parent task

### Completed
- [x] Grilling session with 10 decisions (D1-D10) all resolved
- [x] Requirements documented in requirements.md
- [x] Plan documented in plan.md
- [x] Notes and constraints documented in notes.md

### Pending
- [x] Phase 1: Develop experiment runner script
- [x] Phase 2: Execute experiments (GB300/H200/H100/H800 complete)
- [x] Phase 3: Best config selection and ratio computation for completed systems
- [x] Phase 4: Figure generation for completed systems (fig1 prefill, fig2 decode)
- [x] Phase 5: Validation and test report
- [x] Corrected v2 GB300 shard completion (`16/16` durable mode runs)
- [x] Corrected v2 four-system figures and final menu generation
- [x] Final v2 data, PNG, ratio, provenance, and focused-regression validation
- [x] Final v2 archive inventory refresh in `summary.md` (`41/41` hashes matched)
- [x] Current-source v3 H100/H800/H200 shards and three-system menu
- [x] Current-source v3 GB300 shard and four-system final menu
- [x] Current-source v3 provenance, ratio, figure, regression, and archive validation

### Current-source v3 refresh start — 2026-07-15 HKT

- **Motivation:** The user requested an immediately reviewable menu for completed non-GB300 systems while GB300 continues in the background, followed by final figures for all GPU types. The corrected-v2 checkpoints no longer match the current shared-runner source fingerprint.
- **Expectation:** New immutable v3 shards match the current execution contract, H100/H800/H200 appear first in one menu, the independent GB300 full-matrix process remains untouched, and the later final menu covers all four GPU types without fabricated ratios.
- **Method:** Verified the live primary H200 and GB300 processes and their durable checkpoint counts; confirmed H100/H800 primary completion; checked the throughput wrapper contract inputs; created a no-source-edit constraint; and started a fresh H100 throughput shard under `results/full-v3-h100_sxm/` with `PYTHONPATH=src:.` and `MPLBACKEND=Agg`.
- **Result:** The H100 v3 runner started successfully at `[1/16]`. At the checkpoint, the independent primary shards remained active with H200 at `116/120` and GB300 at `54/120`; no signal, restart, checkpoint mutation, or runner-source edit was performed.

## 2026-07-14 Implementation Progress

### Phase 1: Runner and plotting scripts — completed

- **Motivation:** The approved task requires a fresh disagg-only 64-point matrix with OSL=1024 and output-token-throughput ranking, which differs from the parent 480-run runner.
- **Expectation:** Add a dedicated runner without changing the parent runner's 240-point/480-mode contract; expose both cluster-normalized prefill and decode throughput from each selected rank-one row.
- **Method:** Added `tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py`, reusing the parent checkpoint, cap-expansion, SOL validation, and result-normalization infrastructure under a scoped ranking-contract override. Added `tests/performance/aic_roofline_pareto/plot_throughput_ratio.py` for strict `step4 / DeepSeek-V4-Pro` ratio computation and annotated log-scale figures. Added unit coverage in `tests/unit/performance/test_dsv4pro_vs_step4_throughput.py`.
- **Result:** Matrix construction, task contract, ranking metrics, ratio polarity, zero-baseline rejection, and figure rendering are covered. Targeted unit suite passed: 10/10. Parent performance regression suite passed: 162/162. Ruff check and format check passed for the new/modified files.

### Plotting infeasible points — completed

- **Motivation:** The completed H100 shard demonstrates that some long-ISL points can be memory/SLA infeasible and therefore have no paired rank-one rows. The original plotting path rejected the entire shard instead of preserving valid ratios.
- **Expectation:** Keep direct ratio computation fail-fast, while allowing figure generation to leave gaps for unpaired points and persist an explicit missing-point inventory.
- **Method:** Added `allow_unpaired` only to the figure-generation path, skipped unavailable points when plotting, and recorded `missing_ratio_points` plus paired/missing counts in combined output. Added a regression test that reproduces an H100-style missing DeepSeek row.
- **Result:** RED test reproduced the prior `ValueError`; the minimal implementation now passes the new test and the full focused suite (10/10). No runner source or execution-contract hash was changed.

### Completed-system figure menu — completed

- **Motivation:** The user requested an immediately reviewable figure set for all systems finished while GB300 continues in the background.
- **Expectation:** Generate figures containing only the systems represented by the supplied shards, with no empty GB300 line or fabricated points, and provide one menu containing figures and source data.
- **Method:** Extended figure generation to derive the plotted system list from rank-one rows. Generated `results/figures_completed/fig1_prefill_throughput_ratio.png`, `fig2_decode_throughput_ratio.png`, and `combined_results.json` from H200/H100/H800 shards. Added `results/figures_completed/README.md` as the menu.
- **Result:** Three-system figures generated successfully; 15 paired ratio points are available (5 per system), and 9 infeasible/unpaired points are recorded explicitly. GB300 process remains active under PID 2476384.

### Interim validation after menu update — completed

- **Motivation:** The system-subset plotting behavior changed to omit systems that are not present in the supplied shards.
- **Expectation:** Existing ratio validation and parent performance behavior remain unchanged, while the completed-system menu remains limited to H200/H100/H800.
- **Method:** Ran the focused throughput-ratio suite and the parent Step4 performance regression suite after the plotting update.
- **Result:** Focused suite passed `11/11`; parent regression suite passed `162/162`; Ruff check and format check passed. GB300 remains active at `4/16` with `3` durable checkpoint records.

### Environment note

- **Issue:** The repository `.venv` is unavailable in this checkout, and the default `PYTHONPATH=.` imports a different installed AIC checkout that lacks the current helper symbol.
- **Resolution:** Used the verified handbook environment `/home/i-fengyicheng/miniconda3/envs/aic-step-design` with `PYTHONPATH=src:.`; recorded exact commands in the forthcoming test report.

### Temporary background-run adjustment — 2026-07-14 16:45 HKT

- **Motivation:** Keep the fresh GB300 shard running while making the already completed systems immediately reviewable, then produce the four-system figures automatically after GB300 exits.
- **Expectation:** GB300 remains untouched; the interim menu contains only H200/H100/H800; the final merge includes GB300 only after all four source shards exist.
- **Method:** Left PID `2476384` running in the background and retained its checkpoint at `results/full-gb300/mode_runs.sqlite3`. The durable checkpoint currently contains 8 mode-run rows while the log has entered mode run `[9/16]`. Retained monitor PID `2713407`, which waits for process exit and invokes the four-shard plotting command.
- **Result:** GB300 is still active (`~99% CPU`, healthy `Rl` state); interim figures remain available under `results/figures_completed/`. No runner or checkpoint files were modified.

### Continuation audit — 2026-07-14 17:00 HKT

- **Motivation:** Resume from the persistent task objective and verify the current worktree rather than relying on the previous handoff summary.
- **Expectation:** Requirements, plan, runtime state, and artifacts agree; any documentation contradiction is reconciled before final validation.
- **Method:** Re-read all task documents, confirmed GB300 remains PID `2476384` with 9 durable rows and log position `[10/16]`, confirmed the final merge monitor remains PID `2713407`, and ran the focused plus parent performance regression suites.
- **Result:** The only documentation inconsistency was the old `<= 5000 ms` wording in `requirements.md`/`plan.md`; it was aligned to the user-approved strict `< 5000 ms` contract. Tests passed `173/173`; Ruff check and format check passed using the verified conda interpreter. GB300 has since advanced to `[11/16]` with 10 durable checkpoint rows; final four-system artifacts remain pending GB300 completion.

### GB300 runtime diagnostic — 2026-07-14 17:21 HKT

- **Motivation:** Mode `[11/16]` remained active for an extended interval, so verify that the background task is still making computation progress without altering it.
- **Expectation:** A healthy worker should retain near-100% main-thread CPU and show an active Python/pyarrow processing stack; no restart or fallback is permitted.
- **Method:** Inspected process CPU time and performed a short read-only `gdb` attach/detach. The process remained `Rl` at roughly `99.2%` CPU with CPU time nearly equal to elapsed time; the main thread was active in Python object/dict cleanup while worker threads waited on futexes.
- **Result:** Evidence supports continued CPU-bound SOL data processing rather than a deadlock. PID `2476384`, its checkpoint, and monitor PID `2713407` remain untouched.

### GB300 completion and final four-system figures — 2026-07-14 17:42 HKT

- **Motivation:** Complete the approved 64-point fresh experiment matrix and replace the interim three-system view with a verified four-system deliverable.
- **Expectation:** GB300 finishes all 16 mode-run identities, the monitor combines all four shards, and final ratios contain no fabricated values.
- **Method:** Allowed PID `2476384` to finish normally. The runner wrote all five GB300 artifacts and reported `completed 16 throughput mode runs`. Monitor PID `2713407` then generated `results/figures_final/combined_results.json` and both PNG figures from GB300/H200/H100/H800 shards. Added `results/figures_final/README.md` as the final menu.
- **Result:** All four shards contain 16 complete identities. The combined result contains 23 paired ratio points and 9 explicit missing points across 32 system/ISL requests. GB300 contributes all 8 paired ISL values; H200/H100/H800 contribute 5 each.

### Final validation and documentation — 2026-07-14 17:46 HKT

- **Motivation:** Prove the final task state against every acceptance criterion before completion.
- **Expectation:** Strict TTFT, ratio polarity, numeric ratios, artifact integrity, regression tests, and documentation all pass with concrete evidence.
- **Method:** Validated shard identities and terminal counts, checked every rank-one TTFT, manually recomputed four ratios, inspected both final figures, verified PNG dimensions, ran 173 tests and Ruff checks, and wrote the final test report, lessons, review, and summary artifacts.
- **Result:** Maximum rank-one TTFT values are GB300 `4176.999 ms`, H200 `3431.264 ms`, H100 `3718.155 ms`, and H800 `3832.023 ms`, all strictly below `5000 ms`. Manual ratio deltas are `0.0`; tests pass `173/173`; both figures are valid `2160 × 1260` RGBA PNGs.

### Final completion audit — 2026-07-14 18:03 HKT

- **Motivation:** Close the persistent goal only after independently rechecking the final inventory and every acceptance criterion from current files.
- **Expectation:** All inventory hashes match; 64 model/system/ISL identities and 256 AA/AB/BA/BB terminal outcomes are present; every selected row satisfies strict `TTFT < 5000 ms`; ratios retain exact rank-one provenance; figures and the menu remain readable; fresh tests pass.
- **Method:** Parsed all SHA256 rows from `summary.md`, recomputed each file hash, rebuilt the expected 64-point and 21-config contracts from the runner, audited all four shard JSON files, compared all 50 combined rank-one rows against their full source identities, recomputed all 46 stored metric ratios, checked all 9 missing points, verified PNG metadata and pixels, visually inspected RGB-rendered figures, checked historical background PIDs, and reran pytest plus Ruff.
- **Result:** All `35/35` inventory hashes matched. Contract audit passed with `64/64` mode-run identities, `256/256` terminal outcomes, a `21`-config search space (`17` Pattern A + `4` Pattern B), `23` paired plus `9` missing points, and maximum ratio absolute delta `0.0`. Historical GB300/monitor PIDs alive: `0`. Fresh validation passed `173/173` tests in `2.89 s`; Ruff reported `0` errors and `3` files already formatted.

### Temporary-priority checkpoint revalidation — 2026-07-15 HKT

- **Motivation:** The user reprioritized the session around keeping GB300 in the background, publishing completed-system figures first, and organizing the results under one menu. The persisted task state needed to be checked directly before reporting completion.
- **Expectation:** The H200/H100/H800 menu remains available independently, the later GB300 completion is reflected in the four-system menu, all historical worker/monitor processes have exited normally, and current tests still pass.
- **Method:** Re-read the task documents and both menu READMEs; checked historical PIDs `2476384` and `2713407`; inspected the final figures and an RGB rendering of the decode PNG; audited JSON coverage and SHA256 inventory; reran the 173 performance tests with `MPLBACKEND=Agg PYTHONPATH=src:.`; and reran targeted Ruff checks.
- **Result:** The completed-system menu contains 3 source shards, 15 paired points, and 9 explicit missing points. The final menu contains 4 source shards, 23 paired points, and 9 explicit missing points. Historical worker/monitor PIDs alive: `0`. Fresh pytest passed `173/173` in `3.47 s`; the figures remain valid `2160 × 1260` fully opaque RGBA PNGs. The initial unqualified test invocation was blocked by an unreachable SSH `DISPLAY`; the verified headless backend recipe is now recorded in `task_memory/env_handbook.md`. A first menu-audit assertion used the nonexistent field `source_result_count`; inspecting the stored schema identified the canonical field `source_count`, and the corrected audit passed both menus plus `35/35` inventory hashes without modifying product artifacts.

### Interim-menu archive reconciliation — 2026-07-15 HKT

- **Motivation:** The user explicitly requested a stable menu for the non-GB300 systems before GB300 completed, but the final archive inventory listed only the later four-system menu and therefore did not guarantee that the interim menu would be committed.
- **Expectation:** Preserve both menus and the two Python package markers as durable deliverables, exclude the nested `.omc` runtime state and disposable partial/smoke outputs, and make every documented inventory count and SHA256 assertion agree.
- **Method:** Added the four exact `results/figures_completed/` menu artifacts and both `tests/performance/**/__init__.py` package markers to `summary.md`, updated the inventory contract from `35` to `41`, refreshed the test-report evidence, and ran a parser that recomputed every listed SHA256 from current files.
- **Result:** The durable figure inventory now contains `41/41` matching files: workflow/package files=`5`, final menu=`4`, completed-system menu=`4`, four system shards=`20`, and task documentation=`8`. The package markers combine the target worktree's SPDX provenance with the feature worktree's package docstrings. No plot, result JSON, checkpoint, runner, or product-code value changed.

### Corrected-v2 temporary-priority execution — 2026-07-15 HKT

- **Motivation:** The user requested that GB300 continue uninterrupted in the background while figures for every already completed non-GB300 system are published first under one review menu.
- **Expectation:** H200/H100/H800 figures must use only the corrected v2 shards, omit GB300 entirely, expose missing points without fabrication, and remain directly reviewable while GB300 continues.
- **Method:** Re-ran `plot_throughput_ratio` with `full-v2-h200_sxm`, `full-v2-h100_sxm`, and `full-v2-h800_sxm`; validated the combined JSON and PNG contracts; visually inspected both figures; and retained the existing `figures_completed_v2/README.md` as the single completed-system menu.
- **Result:** The menu contains exactly `3` source shards, `34` rank-one rows, `15` paired ratio points, and `9` explicit missing points. Both figures are `2160 × 1260` RGBA PNGs. GB300 PID `455168` remains healthy at about `99.6%` CPU and subsequently advanced from `13/16` to `15/16` durable mode runs.

### Annotation-layout correction — 2026-07-15 HKT

- **Motivation:** Visual inspection found overlapping `gpu_type ratio` labels where multiple systems share the same ISL, especially at `ISL=1024`.
- **Expectation:** Labels for all four GPU types must occupy distinct deterministic positions without changing any ratio, source row, line, baseline, or experiment contract.
- **Method:** Added a RED unit test for distinct per-system offsets, observed the expected missing-helper failure, implemented one fixed offset per system in `plot_throughput_ratio.py`, regenerated the non-GB300 figures, and visually re-inspected them.
- **Result:** The RED test failed with `AttributeError` before implementation; the GREEN focused suite passed `14/14` in `2.40 s`. The broader related performance suite passed `176/176` in `2.78 s`. Ruff check passed with `0` errors, format check reported `2` files already formatted, and `git diff --check` passed. The labels are visually separated; ratio data and coverage counts are unchanged.

### Final-v2 background monitor correction — 2026-07-15 HKT

- **Motivation:** Automatically create the four-system v2 figures only after GB300 exits successfully, without interrupting either GB300 job.
- **Expectation:** The monitor must validate `16/16` durable GB300 mode runs and `results.json` before plotting; incomplete or failed output must stop explicitly.
- **Method:** The first inline `nohup bash -c` monitor attempt terminated because nested shell/Python quoting corrupted the generated Python expression. Replaced the malformed inline command with a syntax-stable temporary script at `/tmp/step4_throughput_final_v2_monitor_20260715.sh`, preserving the same fail-fast checks and output contract.
- **Result:** The detached `nohup` attempt did not survive the command-runner process group, so the corrected monitor was started as persistent exec session `8371` (process PID `861837`) and is polling GB300 PID `455168`. On validated completion it will generate `results/figures_final_v2/` containing two PNGs, `combined_results.json`, and a self-contained `README.md` menu. The GB300 checkpoint subsequently advanced to `15/16` durable mode runs.

### Corrected-v2 GB300 completion and all-system menu — 2026-07-15 HKT

- **Motivation:** Complete the temporary-priority requirement using only artifacts regenerated after the base-runner fingerprint and inner-collective provenance fixes.
- **Expectation:** GB300 must finish `16/16` durable mode runs before four-system plotting; the final menu must contain GB300, H200, H100, and H800 without fabricated values.
- **Method:** Allowed GB300 PID `455168` to exit naturally, verified its SQLite header and five output artifacts, then plotted the four fresh v2 shard `results.json` files into `results/figures_final_v2/`. Audited all 64 mode-run identities, 256 AA/AB/BA/BB terminal outcomes, 50 rank-one rows, 46 stored ratios, missing-point inventory, PNG metadata, and full-precision ratio recomputation.
- **Result:** GB300 completed `16/16` with exit code `0`. The final v2 menu contains 4 source shards, 50 rank-one rows, 23 paired points, and 9 explicit missing points. All 46 prefill/decode ratios recompute with maximum absolute delta `0.0`; both figures are valid `2160 × 1260` RGBA PNGs. H800 remains explicitly identified as simulated SOL.

### Fresh regression failure and root-cause resolution — 2026-07-15 HKT

- **Motivation:** Re-run the complete affected performance suite after final v2 generation rather than relying on earlier green evidence.
- **Expectation:** All annotation-layout, runner-contract, checkpoint, matrix, and CLI tests must pass; Ruff and `git diff --check` must remain clean.
- **Method:** The first 177-test run exposed a test-only `ValueError`: the annotation-gap assertion paired sequences of lengths 4 and 3 with `zip(..., strict=True)`. Replaced that invalid pairwise expression with `itertools.pairwise(adjusted)`, then reran the single test, Ruff, and the full affected suite.
- **Result:** RED evidence was `1 failed, 176 passed`; the failure occurred before the numeric assertion and did not indicate a plotting defect. GREEN evidence is `1/1` focused and `177/177` affected tests passed in `3.49 s`; Ruff check reports `0` errors, Ruff format reports `3` files already formatted, and `git diff --check` exits `0`.

### Corrected-v2 archive and background-worker closure — 2026-07-15 HKT

- **Motivation:** Complete the temporary-priority delivery with authoritative v2 paths and hashes while preserving the separate 120-run GB300 full-matrix job requested to continue in the background.
- **Expectation:** The non-GB300 and four-GPU menus remain directly reviewable; the 41-file archive contains only corrected-v2 deliverables; fresh tests and static checks pass; the independent full-matrix GB300 process remains alive and its checkpoint keeps advancing.
- **Method:** Re-ran the 177-test affected suite with `MPLBACKEND=Agg` and `PYTHONPATH=src:.`, ran Ruff check/format and `git diff --check`, refreshed the review and test report, and recomputed the 41-file summary inventory. The first read-only menu audit incorrectly treated nested `summary` metrics as top-level fields, then incorrectly treated scalar ratio values as objects and display system names as persisted source names. Inspected the persisted JSON schema, corrected the audit-only field access and explicit display-to-source mapping, and reran the audit without changing any artifact. Read-only process and SQLite checks were used for PID `585535`; no signal, restart, checkpoint edit, or process-control action was issued.
- **Result:** Fresh validation passed `177/177` in `3.15 s`; Ruff check reported `0` errors, Ruff format reported `3` files already formatted, and `git diff --check` exited `0`. The corrected final audit matched `41/41` hashes, both menu contracts, and `46/46` ratios with maximum absolute delta `0.0`. The separate full-matrix GB300 worker remains healthy: shell PID `585535`, Python PID `585536` at `97.0%` CPU, `63/120` durable rows, with the log entering `[64/120]`.

### Current non-GB300 menu revalidation — 2026-07-15 HKT

- **Motivation:** Honor the user's temporary priority by keeping the current formatter-consistent GB300 full-matrix shard running while making the already completed H200/H100/H800 figures immediately reviewable under one menu.
- **Expectation:** The stable menu must contain only the three non-GB300 systems, retain strict source-derived gaps, render both figures at stable dimensions, and pass the focused plotting tests without modifying scientific artifacts.
- **Method:** Read and visually inspected both `figures_completed_v2` PNGs; checked all four menu files and their SHA256 values; ran five focused plotting/annotation/ratio tests with `MPLBACKEND=Agg`; and validated the persisted JSON summary, ratio cardinality, missing-point inventory, PNG signature, and dimensions. The first one-off validator expected display text `Step4 / DeepSeek-V4-Pro`, while the stored schema uses `step4 / DeepSeek-V4-Pro`; inspected the actual summary, corrected only the validator expectation, and reran the complete focused sequence.
- **Result:** Focused tests passed `5/5` in `2.62 s`. The menu contains `3` source shards, `34` rank-one rows, `15` paired points, `9` explicit missing points, and `30` stored prefill/decode ratio values. Both PNGs are `2160 × 1260`; current sizes are `170465` and `173964` bytes. The full-matrix background shards remained active and untouched during this validation.

### Current-source non-GB300 v3 menu completion — 2026-07-15 HKT

- **Motivation:** Publish the requested early-review menu from artifacts whose strict source provenance matches the final formatted runner bytes, without waiting for GB300.
- **Expectation:** H200/H100/H800 current contracts match their checkpoint headers; v2/v3 scientific payloads remain identical; the menu contains only completed non-GB300 systems and no fabricated ratios.
- **Method:** Completed the H100, H200, and H800 v3 shards in new immutable directories; checked every header against its system-filtered current execution contract; normalized only `checkpoint_header.execution_contract_sha256` for v2/v3 equality; generated `figures_completed_v3`; and validated its JSON and PNGs.
- **Result:** All three shards completed `16/16` with exit code `0`. The v3 scientific payloads are exactly identical to v2. The menu contains `3` sources, `34` rank-one rows, `15` paired points, `9` explicit missing points, and `30` stored ratios. Both PNGs are `2160 × 1260` and byte-identical to v2. H800 remains simulated SOL.

### Current-source GB300 v3 completion and all-GPU menu — 2026-07-15 HKT

- **Motivation:** Close the user-approved strict-provenance rerun and publish one current-source menu for every GPU type while leaving the independent 120-run GB300 primary process untouched.
- **Expectation:** GB300 throughput completes `16/16`; its contract and HEAD match current source; the final menu has four sources, exact ratio recomputation, strict `TTFT < 5000 ms`, and no scientific drift from v2.
- **Method:** Allowed the GB300 throughput runner to exit naturally in the second compute lane; verified checkpoint contract `ab6d3a7f252f0e4e3c254f8fa03669ad0526f55e432a43d1c97f63abe6bd603e`, Git HEAD `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1`, and five artifact hashes; compared v2/v3 after normalizing only the contract hash; generated `figures_final_v3`; then audited all four v3 shards, combined ratios, missing points, PNGs, and source paths.
- **Result:** GB300 completed `16/16`, runner exit=`0`, and v2/v3 scientific identity is exact. Final evidence: identities=`64/64`; terminal outcomes=`256/256` (`156` success, `88` memory infeasible, `12` SLA infeasible); rank-one rows=`50`; paired/missing=`23/9`; ratios=`46/46` with maximum delta=`0.0`; PNG dimensions=`2160 × 1260`; v2/v3 PNG byte identity=`2/2`. Maximum rank-one TTFT values are GB300/H200/H100/H800=`4176.999/3431.264/3718.155/3832.023 ms`, all strictly below `5000 ms`.

### GB300 v3 audit assertion correction — 2026-07-15 HKT

- **Motivation:** Diagnose one late audit assertion without treating validator failure as a shard or figure defect.
- **Expectation:** Determine whether strict TTFT eligibility failed or the one-off validator selected the wrong row population; do not change runner code or artifacts.
- **Method:** Printed authoritative ranked-row schema, rank values, `ttft_pass`, all-row TTFT range, and rank-one TTFT range. The first validator asserted the known rank-one maximum against all `58` ranked rows; one lower-ranked eligible candidate had TTFT `4200.186 ms`, while the `16` rank-one rows retained maximum `4176.999 ms`. Corrected only the validator selection to `rank == 1` and reran the complete contract, checkpoint, identity, TTFT, and hash audit.
- **Result:** Root cause was an audit-only population mismatch, not scientific data. Corrected full audit passed; all `58` ranked rows satisfy strict TTFT, all `16` rank-one rows satisfy it, and the reported rank-one maximum remains `4176.999 ms`. No source, checkpoint, JSON, CSV, report, or PNG changed.
