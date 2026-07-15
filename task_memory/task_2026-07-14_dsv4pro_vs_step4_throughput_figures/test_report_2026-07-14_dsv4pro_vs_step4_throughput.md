## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Added final four-system experiment, ratio, artifact, and regression validation evidence |
| 2026-07-14 | Added the fresh completion audit, hash evidence, and audit-harness failure resolutions |
| 2026-07-15 | Added the headless-session rerun and current menu/inventory verification evidence |
| 2026-07-15 | Added archival validation for both menus and the package markers, expanding the inventory total to 41 files |
| 2026-07-15 | Added corrected-v2 completed-system figure and annotation-layout validation evidence |
| 2026-07-15 | Added corrected-v2 GB300 completion, final all-system figures, full numeric audit, and regression-failure resolution |
| 2026-07-15 | Added current-source v3 shard, all-GPU figure, scientific-identity, and audit-correction evidence |
| 2026-07-15 | Refreshed authoritative v2 menu hashes, current 177-test evidence, and archive status |
| 2026-07-15 | Removed three Markdown hard-break trailing spaces found by the final repository-wide documentation gate |
| 2026-07-15 | Added the current non-GB300 menu focused-test, JSON, and PNG revalidation evidence |

# Test Report: DeepSeek-V4-Pro vs Step4 Throughput Figures

**Date:** 2026-07-14
**Environment:** `aic-step-design` conda environment, Python `3.11.15`
**Interpreter:** `/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python`
**Import isolation:** `PYTHONPATH=src:.`

## 1. Test Script Information

### Scripts and artifacts

- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/tests/unit/performance/test_dsv4pro_vs_step4_throughput.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/tests/unit/performance/test_step4_roofline_matrix.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/tests/unit/performance/test_step4_comparison_checkpoint.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/tests/unit/performance/test_step4_comparison_cli.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/tests/performance/aic_roofline_pareto/plot_throughput_ratio.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_final_v2/combined_results.json`

### Exact test command

```bash
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python

MPLBACKEND=Agg PYTHONPATH=src:. "$PY" -m pytest -q \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py \
  tests/unit/performance/test_step4_roofline_matrix.py \
  tests/unit/performance/test_step4_comparison_checkpoint.py \
  tests/unit/performance/test_step4_comparison_cli.py
```

### Exact static-check commands

```bash
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python

"$PY" -m ruff check \
  tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py \
  tests/performance/aic_roofline_pareto/plot_throughput_ratio.py \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py

"$PY" -m ruff format --check \
  tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py \
  tests/performance/aic_roofline_pareto/plot_throughput_ratio.py \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py
```

### Exact figure-generation command

```bash
ROOT=task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python

MPLBACKEND=Agg PYTHONPATH=src:. "$PY" \
  -m tests.performance.aic_roofline_pareto.plot_throughput_ratio \
  "$ROOT/results/full-v2-gb300/results.json" \
  "$ROOT/results/full-v2-h200_sxm/results.json" \
  "$ROOT/results/full-v2-h100_sxm/results.json" \
  "$ROOT/results/full-v2-h800_sxm/results.json" \
  --output-dir "$ROOT/results/figures_final_v2" \
  --combined-output "$ROOT/results/figures_final_v2/combined_results.json"
```

## 2. Validation Criteria

1. Each system shard contains exactly 16 mode runs: 2 models × 8 ISL values.
2. Every shard contains the complete expected `(model, ISL)` identity set.
3. All selected rank-one rows satisfy strict disaggregated `TTFT < 5000 ms`.
4. The combined artifact contains four source shards and systems in order: `gb300`, `h200`, `h100`, `h800`.
5. Ratio direction is exactly `Step4 / DeepSeek-V4-Pro`.
6. Each stored ratio equals numerator divided by denominator with absolute delta no greater than `1e-12`.
7. Missing or infeasible model points are recorded explicitly and are not assigned fabricated ratios.
8. Both PNG figures are valid, non-empty `2160 × 1260` RGBA images.
9. All focused and parent regression tests pass; Ruff check and format check pass.

## 3. Test Results and Evidence

### Test-suite results

| Suite | Expected | Actual | Result |
|---|---:|---:|---|
| Affected performance tests | 177 passed | 177 passed in `3.15 s` | PASS |
| Ruff check | 0 errors | 0 errors | PASS |
| Ruff format check | 3 files formatted | 3 files already formatted | PASS |

### Per-shard execution and SLA evidence

| System shard | Expected mode runs | Actual mode runs | Identity count | Rank-one rows | Maximum rank-one TTFT (ms) | Strict threshold (ms) | Margin to threshold (ms) | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GB300 | 16 | 16 | 16 | 16 | 4176.999 | `< 5000` | 823.001 | PASS |
| H200 SXM | 16 | 16 | 16 | 12 | 3431.264 | `< 5000` | 1568.736 | PASS |
| H100 SXM | 16 | 16 | 16 | 11 | 3718.155 | `< 5000` | 1281.845 | PASS |
| H800 SXM (simulated SOL) | 16 | 16 | 16 | 11 | 3832.023 | `< 5000` | 1167.977 | PASS |

### Terminal-status evidence

| System shard | Success | Memory infeasible | SLA infeasible | Total terminal outcomes |
|---|---:|---:|---:|---:|
| GB300 | 58 | 6 | 0 | 64 |
| H200 SXM | 44 | 14 | 6 | 64 |
| H100 SXM | 27 | 34 | 3 | 64 |
| H800 SXM (simulated SOL) | 27 | 34 | 3 | 64 |

Each shard evaluates four disaggregated pairings for each of its 16 model/ISL mode runs, producing 64 explicit terminal outcomes per system.

### Combined-result coverage

| Metric | Expected | Actual | Delta | Result |
|---|---:|---:|---:|---|
| Source shards | 4 | 4 | 0 | PASS |
| Systems | 4 | 4 | 0 | PASS |
| Requested system/ISL points | 32 | 32 | 0 | PASS |
| Paired ratio points | Recorded, no fabrication | 23 | N/A | PASS |
| Missing ratio points | Explicitly recorded | 9 | N/A | PASS |
| Paired + missing points | 32 | 32 | 0 | PASS |
| Rank-one rows | Derived from shards | 50 | N/A | PASS |

GB300 provides all 8 paired ISL points. H200, H100, and H800 each provide 5 paired points (`1024` through `16384`). Their remaining long-ISL points are explicitly listed in `missing_ratio_points`.

### Manual numeric ratio checks

| System | ISL | Metric | Step4 numerator | DeepSeek-V4-Pro denominator | Expected computation | Stored ratio | Absolute delta | Result |
|---|---:|---|---:|---:|---|---:|---:|---|
| GB300 | 1024 | Prefill | 18316.823811870952 | 14380.608648026178 | numerator / denominator | 1.2737168683319284 | 0.0 | PASS |
| GB300 | 1024 | Decode | 14944.848000000002 | 12741.86752 | numerator / denominator | 1.172893061126412 | 0.0 | PASS |
| H200 | 4096 | Prefill | 9162.52384582622 | 5333.314814879116 | numerator / denominator | 1.7179791862772111 | 0.0 | PASS |
| H200 | 4096 | Decode | 1971.78816 | 1200.0096 | numerator / denominator | 1.6431436548507612 | 0.0 | PASS |

### Figure artifact evidence

| File | Expected | Actual | Result |
|---|---|---|---|
| `fig1_prefill_throughput_ratio.png` | Valid, non-empty PNG | `2160 × 1260`, RGBA, 203576 bytes | PASS |
| `fig2_decode_throughput_ratio.png` | Valid, non-empty PNG | `2160 × 1260`, RGBA, 204025 bytes | PASS |

Visual inspection confirmed four system lines, the Step4 `1.0` dashed baseline, log-scale ISL ticks, and per-point system/ratio annotations. Missing points appear as absent/gapped values rather than fabricated data.

### Environment issue and resolution

The first bare `ruff` invocation failed because `ruff` was not on the default shell `PATH`. Root cause: the validated tooling is installed in the `aic-step-design` conda environment. The check was rerun explicitly as:

```bash
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m ruff ...
```

The explicit-environment rerun passed. No source-code workaround or fallback was added.

### Final automated completion audit

The final audit was run with the same `aic-step-design` interpreter and `PYTHONPATH=src:.`. It parsed the inventory from `summary.md`, recomputed hashes and result identities, rebuilt the static runner contract, checked rank-one provenance and ratios, and verified the figures and menu.

| Validation | Expected | Actual | Delta | Result |
|---|---:|---:|---:|---|
| Inventory SHA256 rows | 41 | 41 matched | 0 | PASS |
| Matrix points | 64 | 64 | 0 | PASS |
| Completed mode-run identities | 64 | 64 | 0 | PASS |
| AA/AB/BA/BB terminal outcomes | 256 | 256 | 0 | PASS |
| Search-space configurations | 21 | 21 (`17` Pattern A + `4` Pattern B) | 0 | PASS |
| Combined rank-one provenance rows | 50 | 50 exact full-identity matches | 0 | PASS |
| Requested ratio points | 32 | 23 paired + 9 missing | 0 | PASS |
| Maximum ratio absolute delta | `<= 1e-12` | `0.0` | `-1e-12` margin | PASS |
| Historical GB300/monitor PIDs alive | 0 | 0 | 0 | PASS |

The first audit attempt expected `31` inventory entries and stopped before checking artifacts. Root cause: the manual total omitted four entries; the then-current section total was `3 + 4 + 20 + 8 = 35`. The user subsequently requested an interim menu for the three already-complete non-GB300 systems, so the four durable interim-menu artifacts were added to the archive contract. The final branch audit also included the two required Python package markers, bringing the exact inventory total to `41`. The second initial audit attempt used `canonical_config_id` as a global key and stopped because only `44` values were unique. Root cause: that field identifies a parallel configuration and is allowed to repeat across model/system/ISL groups. The corrected full key `(model, system, isl, canonical_config_id)` is unique for all `50/50` rank-one rows. The corrected full audit then passed end to end. No product artifact was modified to make the audit pass.

### Final visual-renderer diagnostic

One RGBA preview of the decode figure displayed black blocks. The original PNG was checked directly: all `2,721,600` alpha samples are `255`, sampled background pixels are white, Pillow reports `2160 × 1260` RGBA, and an RGB-rendered view displays the complete plot. This confirms a preview-layer issue rather than a deliverable defect.

### 2026-07-15 fresh checkpoint validation

The completed-system and final menus were revalidated from current files after the user's temporary-priority checkpoint.

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Completed-system source shards | 3 | 3 | PASS |
| Completed-system paired points | 15 | 15 | PASS |
| Completed-system missing points | 9 | 9 | PASS |
| Final source shards | 4 | 4 | PASS |
| Final paired points | 23 | 23 | PASS |
| Final missing points | 9 | 9 | PASS |
| Completed-system menu artifacts | 4 | 4, hashes matched | PASS |
| Final menu artifacts | 4 | 4, hashes matched | PASS |
| Historical GB300/monitor PIDs alive | 0 | 0 | PASS |
| Current focused + parent tests | 173 | 173 passed in `3.47 s` | PASS |
| PNG dimensions | `2160 × 1260` each | `2160 × 1260` each | PASS |
| PNG alpha range | Fully opaque | `(255, 255)` each | PASS |

The first unqualified pytest invocation blocked before test execution because `DISPLAY=localhost:17.0` pointed to an unreachable forwarded X11 endpoint. A `gdb` backtrace showed Matplotlib blocked in `mpl_display_is_valid() -> XOpenDisplay()`. The exact suite was rerun with `MPLBACKEND=Agg`, passed with exit code `0`, and the environment recipe was recorded in `task_memory/env_handbook.md`. This was an environment-entry correction; no source or experiment artifact was changed.

The first one-off menu contract audit also used the nonexistent field name `source_result_count` and raised `KeyError`. Inspection of the persisted JSON schema showed that the canonical field is `source_count`. The corrected audit passed both menus, all four PNG contracts, and all `41/41` inventory hashes. This was an audit-script assumption error; no deliverable was changed to satisfy the audit.

### 2026-07-15 corrected-v2 completed-system validation

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Included systems | H200, H100, H800 only | `h200`, `h100`, `h800` | PASS |
| Source shards | 3 | 3 | PASS |
| Rank-one rows | Derived from v2 shards | 34 | PASS |
| Paired ratio points | Recorded, no fabrication | 15 | PASS |
| Missing ratio points | Explicitly recorded | 9 | PASS |
| Prefill PNG dimensions | `2160 × 1260` | `2160 × 1260` RGBA | PASS |
| Decode PNG dimensions | `2160 × 1260` | `2160 × 1260` RGBA | PASS |
| Annotation-offset RED test | Must fail before implementation | Failed with missing `_annotation_offset` | PASS |
| Focused GREEN suite | All pass | `14/14` passed in `2.40 s` | PASS |
| Related performance regression | All pass | `176/176` passed in `2.78 s` | PASS |
| Ruff check | 0 errors | 0 errors | PASS |
| Ruff format check | 2 files formatted | 2 files already formatted | PASS |
| `git diff --check` | No whitespace errors | Exit code `0`, no output | PASS |

Current corrected-v2 figure hashes:

| File | SHA256 |
|---|---|
| `fig1_prefill_throughput_ratio.png` | `5c7f11767c144da2e6190118f547806ea982c3c86d120b7b10fedec915dfca88` |
| `fig2_decode_throughput_ratio.png` | `150df14423505e3022185fa4110d0dad23afa7ccb038e9946412de3a35fe7978` |

### 2026-07-15 corrected-v2 final all-system validation

#### Authoritative artifact paths

```text
task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v2-gb300/
task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v2-h200_sxm/
task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v2-h100_sxm/
task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v2-h800_sxm/
task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_completed_v2/
task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_final_v2/
```

#### Exact final figure command

```bash
ROOT=task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python

MPLBACKEND=Agg PYTHONPATH=src:. "$PY" \
  -m tests.performance.aic_roofline_pareto.plot_throughput_ratio \
  "$ROOT/results/full-v2-gb300/results.json" \
  "$ROOT/results/full-v2-h200_sxm/results.json" \
  "$ROOT/results/full-v2-h100_sxm/results.json" \
  "$ROOT/results/full-v2-h800_sxm/results.json" \
  --output-dir "$ROOT/results/figures_final_v2" \
  --combined-output "$ROOT/results/figures_final_v2/combined_results.json"
```

#### Exact affected regression command

```bash
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python

MPLBACKEND=Agg PYTHONPATH=src:. "$PY" -m pytest -q \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py \
  tests/unit/performance/test_step4_roofline_matrix.py \
  tests/unit/performance/test_step4_comparison_checkpoint.py \
  tests/unit/performance/test_step4_comparison_cli.py
```

#### Shard and scientific-result metrics

| System | Durable mode runs | Rank-one rows | Maximum rank-one TTFT | Terminal outcomes | Execution contract SHA256 | Result |
|---|---:|---:|---:|---|---|---|
| GB300 | 16/16 | 16 | `4176.999 ms` | 58 success, 6 memory infeasible | `dac6923cbce80a2b24e9c991da7e1dda52d605316ef64ccfe1f8aef61b0cd327` | PASS |
| H200 SXM | 16/16 | 12 | `3431.264 ms` | 44 success, 14 memory infeasible, 6 SLA infeasible | `6f8be8baf0998cc7eba36afbe891f91c4db3ef4124ea62ab9916aa1b1997ea2e` | PASS |
| H100 SXM | 16/16 | 11 | `3718.155 ms` | 27 success, 34 memory infeasible, 3 SLA infeasible | `307cb5d016ae7da5add01d15d774fe32c9522a5251b7afa990cf01dfaca31e56` | PASS |
| H800 SXM simulated SOL | 16/16 | 11 | `3832.023 ms` | 27 success, 34 memory infeasible, 3 SLA infeasible | `af9ac18b57ca789ca4a7540937aa85700d963bce941b0634aba7ef8a8f9fb0da` | PASS |

All four v2 shard headers pin Git HEAD `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1`. Their contract hashes differ because each throughput shard contains a different system-filtered 16-run matrix; this is expected and was verified individually.

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Full mode-run identities | 64 | 64 | PASS |
| AA/AB/BA/BB terminal outcomes | 256 | 256 | PASS |
| Success terminal outcomes | Recorded | 156 | PASS |
| Memory-infeasible outcomes | Explicit | 88 | PASS |
| SLA-infeasible outcomes | Explicit | 12 | PASS |
| Combined rank-one provenance rows | 50 | 50 exact source identities | PASS |
| Paired ratio points | Recorded | 23 | PASS |
| Missing ratio points | Explicit, no fabrication | 9 | PASS |
| Stored prefill/decode ratios | 46 | 46 | PASS |
| Maximum ratio recomputation delta | `<= 1e-12` | `0.0` | PASS |
| Prefill PNG | Valid `2160 × 1260` RGBA | `2160 × 1260` RGBA | PASS |
| Decode PNG | Valid `2160 × 1260` RGBA | `2160 × 1260` RGBA | PASS |

Final v2 figure hashes:

| File | SHA256 |
|---|---|
| `fig1_prefill_throughput_ratio.png` | `a1059fa972b275ceaf9b7c8a6eef244a7a5aa423b3f8d333b945b2bc56c61b1c` |
| `fig2_decode_throughput_ratio.png` | `eca2b011c4624de469512243f4e441d5c7b9f73369b80f0bff8fac65f6c81e05` |
| `combined_results.json` | `244e366ab0fc42dd8d1f898de9986a8bdc63e9eda9169b5a98271c5c0ac99b9c` |

Current final-render evidence:

| Artifact | Expected | Actual | Result |
|---|---|---|---|
| Prefill PNG | `2160 × 1260`, fully opaque RGBA | `2160 × 1260`, alpha `(255, 255)`, `205229` bytes | PASS |
| Decode PNG | `2160 × 1260`, fully opaque RGBA | `2160 × 1260`, alpha `(255, 255)`, `205541` bytes | PASS |

#### Fresh test failure and resolution

The first final affected-suite run collected 177 tests and ended with `1 failed, 176 passed`. The failure was test-only:

```text
ValueError: zip() argument 2 is shorter than argument 1
```

Root cause: the annotation minimum-gap assertion compared a sequence with its one-element-shorter tail using `zip(..., strict=True)`, so it was guaranteed to raise before checking the gap. A first minimal change to non-strict zip made the test pass but Ruff correctly reported `RUF007`, preferring an explicit adjacent-pair operation. The final fix uses `itertools.pairwise(adjusted)`.

Final evidence:

| Check | Actual result | Status |
|---|---|---|
| Focused annotation test | `1/1` passed in `0.06 s` | PASS |
| Affected performance suite | `177/177` passed in `3.49 s` | PASS |
| Ruff check | 0 errors | PASS |
| Ruff format check | 3 files already formatted | PASS |
| `git diff --check` | Exit code 0 | PASS |

The final fresh completion run on 2026-07-15 collected `177` tests and passed `177/177` in `3.15 s`. Ruff check returned `0` errors, Ruff format reported `3` files already formatted, and `git diff --check` exited `0`.

Several one-off read-only audit commands also initially encoded incorrect schema assumptions: SQLite checkpoint headers were treated as columns rather than a JSON `payload`; `ratios` was treated as a list rather than a phase/system mapping; mode-run identity dictionaries were converted to tuples of keys; a shared throughput contract was assumed across system-filtered shards; and display names such as `h200` were confused with source names such as `h200_sxm`. Each assumption was corrected by inspecting the persisted schema. The final audit passed all 64 identities, 256 terminal outcomes, 50 provenance rows, and 46 ratios without modifying any scientific result.

### Current non-GB300 menu validation

The early-review menu was revalidated while the separate formatter-consistent full-matrix shards continued in the background.

```bash
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python

MPLBACKEND=Agg PYTHONPATH=src:. "$PY" -m pytest \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py::test_ratio_direction_is_step4_over_deepseek \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py::test_annotation_positions_enforce_minimum_gap_without_center_drift \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py::test_plot_script_accepts_system_shards_and_writes_combined_results \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py::test_plot_script_records_and_skips_unpaired_infeasible_points \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py::test_plot_script_limits_lines_to_systems_present_in_shards \
  -q
```

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Focused plotting tests | 5 pass | `5/5` passed in `2.62 s` | PASS |
| Source systems | H200/H100/H800 only | `3/3` | PASS |
| Rank-one rows | 34 | `34` | PASS |
| Paired ratio points | 15 | `15` | PASS |
| Explicit missing points | 9 | `9` | PASS |
| Stored prefill/decode ratios | 30 | `30` | PASS |
| Prefill PNG | `2160 × 1260` | `2160 × 1260`, `170465` bytes | PASS |
| Decode PNG | `2160 × 1260` | `2160 × 1260`, `173964` bytes | PASS |

The first audit-only JSON assertion expected the README-style display string `Step4 / DeepSeek-V4-Pro`; the stored schema uses `step4 / DeepSeek-V4-Pro`. After inspecting the current summary and correcting only that validator expectation, the entire five-test plus JSON/PNG sequence was rerun and ended with `NON_GB300_MENU_VALIDATION=PASS`. No source, checkpoint, ratio, JSON, or figure was modified.

## Current-Source v3 Validation

**Environment:** `aic-step-design`, Python `3.11.15`, `PYTHONPATH=src:.`, `MPLBACKEND=Agg`.

### Test Script Information

- Runner: `tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py`
- Plotter: `tests/performance/aic_roofline_pareto/plot_throughput_ratio.py`
- Focused tests: `tests/unit/performance/test_dsv4pro_vs_step4_throughput.py`
- Current-source result directories:
  - `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-gb300/`
  - `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h200_sxm/`
  - `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h100_sxm/`
  - `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h800_sxm/`
- Current menus:
  - `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_completed_v3/README.md`
  - `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_final_v3/README.md`

Reproducible all-GPU plotting command:

```bash
ROOT=task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python

MPLBACKEND=Agg PYTHONPATH=src:. "$PY" \
  -m tests.performance.aic_roofline_pareto.plot_throughput_ratio \
  "$ROOT/results/full-v3-gb300/results.json" \
  "$ROOT/results/full-v3-h200_sxm/results.json" \
  "$ROOT/results/full-v3-h100_sxm/results.json" \
  "$ROOT/results/full-v3-h800_sxm/results.json" \
  --output-dir "$ROOT/results/figures_final_v3" \
  --combined-output "$ROOT/results/figures_final_v3/combined_results.json"
```

### Validation Criteria

- Each v3 header matches its current source-derived execution contract and Git HEAD.
- All `64` mode-run identities and `256` AA/AB/BA/BB terminal outcomes are present.
- Every rank-one row satisfies strict `TTFT < 5000 ms`.
- All `46` stored ratios recompute from the same rank-one rows with absolute delta at most `1e-12`.
- Nine infeasible points remain explicit; no ratio is synthesized.
- H800 rows remain labeled simulated SOL.
- v2/v3 scientific payloads differ only in strict provenance fields and combined source paths.
- Both PNGs are valid, fully opaque `2160 × 1260` RGBA images.

### Test Results and Evidence

| System | Expected contract | Mode runs | Rank-one rows | Max rank-one TTFT | Margin to 5000 ms | Result |
|---|---|---:|---:|---:|---:|---|
| GB300 | `ab6d3a7f252f0e4e3c254f8fa03669ad0526f55e432a43d1c97f63abe6bd603e` | 16/16 | 16 | `4176.999 ms` | `823.001 ms` | PASS |
| H200 SXM | `8d86bc5e230b0d6404e6de4930e57337442e2db8031939ca7358df6280e82b5e` | 16/16 | 12 | `3431.264 ms` | `1568.736 ms` | PASS |
| H100 SXM | `58638b227687f247ea8c208589a95c8a6d4076741ebfc824838204b4ea2d93d3` | 16/16 | 11 | `3718.155 ms` | `1281.845 ms` | PASS |
| H800 SXM simulated SOL | `41ac60b60c5a26c8acfd9b3983e9caef7c70a5506b8980e2eb169a9737293e62` | 16/16 | 11 | `3832.023 ms` | `1167.977 ms` | PASS |

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Mode-run identities | 64 | 64/64 | PASS |
| Terminal outcomes | 256 | 256/256 | PASS |
| Success / memory / SLA terminals | Recorded | `156 / 88 / 12` | PASS |
| Rank-one rows | 50 | 50 | PASS |
| Paired / missing points | 23 / 9 | 23 / 9 | PASS |
| Stored ratios | 46 | 46/46 | PASS |
| Maximum ratio delta | `<= 1e-12` | `0.0` | PASS |
| v2/v3 shard scientific identity | 4 | 4/4 exact after contract-hash normalization | PASS |
| Combined v2/v3 differences | Four source paths only | Four source paths only | PASS |
| PNG byte identity | 2 | 2/2 | PASS |
| PNG dimensions / alpha | `2160 × 1260`, `(255,255)` | `2160 × 1260`, `(255,255)` | PASS |

GB300 v3 artifact hashes:

| File | Bytes | SHA256 |
|---|---:|---|
| `mode_runs.sqlite3` | 4775936 | `339cb9eb68620fc1bc3f0fed01e8a2752c6aaad4fa9d7ba82aabe088a83f1bd0` |
| `results.json` | 10350532 | `821cececa76f4412002df38522765534462742f275324b43b4523471658ef1b2` |
| `ranked_rows.csv` | 2954757 | `e7c6a6414c6665d7d0021f9289c442808e94918108f15357c8e0ee35e952396a` |
| `model_comparisons.csv` | 9761 | `0d9da64532ec12cad8f5c4c9498fc13fc08717cf76748da392609df183867304` |
| `report.md` | 9092 | `48a163eb504aabb81ad8b2fd50c0a85eae87c4fcb021a1555a6f06536a8ae243` |

Final-v3 figure hashes:

| File | Bytes | SHA256 |
|---|---:|---|
| `combined_results.json` | 2535512 | `2ac69e74cb1054e83e6077b7b79faa0e62a23d20ea2a5ce8690f69f414c9c3e0` |
| `fig1_prefill_throughput_ratio.png` | 205229 | `a1059fa972b275ceaf9b7c8a6eef244a7a5aa423b3f8d333b945b2bc56c61b1c` |
| `fig2_decode_throughput_ratio.png` | 205541 | `eca2b011c4624de469512243f4e441d5c7b9f73369b80f0bff8fac65f6c81e05` |

#### Audit-only failure and resolution

The first GB300 v3 audit checked the known rank-one TTFT maximum against all `58` ranked rows and failed because a lower-ranked eligible row has TTFT `4200.186 ms`. The selected `16` rank-one rows have maximum `4176.999 ms`, and all `58` rows are still strictly below `5000 ms`. The validator was corrected to assert the reported maximum over `rank == 1` while retaining a strict-TTFT assertion over every ranked row. The complete contract/checkpoint/identity/hash audit was rerun and passed. No deliverable was changed to satisfy the validator.

## Final Result

**FINAL CURRENT-SOURCE V3 PASS.** The v3 interim menu satisfies the non-GB300 early-review requirement, and the v3 final menu covers GB300, H200, H100, and H800. Throughput GB300 completed `16/16`; no partial checkpoint, fabricated ratio, fallback result, or superseded v1/v2 artifact is part of the authoritative 41-file archive. Commit/push and `step-design` integration remain parent-task branch actions rather than figure-task correctness requirements.
