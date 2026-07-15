## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-15 | Confirmed primary GB300 corrected-v3 completion and closed the parent-task integration dependency |
| 2026-07-14 | Added the final task inventory, validation matrices, and open-item statement |
| 2026-07-14 | Added the final completion-audit evidence and refreshed documentation hashes |
| 2026-07-15 | Added the temporary-priority checkpoint revalidation and refreshed documentation hashes |
| 2026-07-15 | Added the completed-system interim menu and package markers to the durable 41-file archive inventory |
| 2026-07-15 | Replaced superseded result inventory with the corrected 41-file v2 archive and refreshed validation evidence |
| 2026-07-15 | Refreshed closure evidence, documentation hashes, and the independent full-matrix GB300 runtime note |
| 2026-07-15 | Recorded completion of the separate primary GB300 run and removed the stale active-runtime note |
| 2026-07-15 | Refreshed documentation hashes after the current non-GB300 menu revalidation |
| 2026-07-15 | Promoted the current-source v3 shards and menus into the authoritative 41-file inventory |

# Summary: DeepSeek-V4-Pro vs Step4 Throughput Figures

## Task Overview

This task executed and then regenerated a current-source v3 disaggregated throughput matrix for `stepfun-ai/Step4` and `deepseek-ai/DeepSeek-V4-Pro` on GB300, H200 SXM, H100 SXM, and H800 SXM. Each system shard evaluated eight ISL values (`1024` through `131072`) with OSL `1024`, a 64-GPU fixed cluster, the approved 21-configuration search space, SOL database mode, and strict disaggregated `TTFT < 5000 ms` eligibility.

The highest `output_token_throughput` rank-one configuration for each feasible model/system/ISL point supplied both the prefill and decode metrics. Final ratios use the approved direction `Step4 / DeepSeek-V4-Pro`. Infeasible points remain explicit gaps and are never synthesized. H800 results are simulated SOL results, not silicon measurements.

The interim deliverable contains the three already-complete non-GB300 systems, 15 paired ratio points, 9 explicit missing points, two PNG figures, a combined JSON payload, and a menu README. After GB300 completed, the final deliverable expanded to four system lines, 23 paired ratio points, 9 explicit missing points, two PNG figures, a combined JSON payload, and a final menu README. All 64 mode-run identities completed.

## Deliverables Inventory

The authoritative archive contains only current-source v3 experiment and figure artifacts. Earlier v1/v2 result trees remain local historical evidence and are intentionally excluded from staging. Every v3 scientific payload is exactly identical to its v2 predecessor after normalizing only the strict execution-contract hash.

### Workflow code and tests

| SHA256 | Exact path |
|---|---|
| `4089d5294de7775f30e1384262aeab895f90627a9893ad0df34be85f99579305` | `tests/performance/__init__.py` |
| `b065f9aac0d8a6496ca6eb265aee8b3540164cf3be9fab989f5c2aeb37b4ac16` | `tests/performance/aic_roofline_pareto/__init__.py` |
| `fb280e02a824cd700b759ddba7392505ffea3442500b816756389a70a9ba9489` | `tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py` |
| `40ef629fb079c6324c431442fda634d974e37cf4ca1420e6cac9a7ec6dc46c81` | `tests/performance/aic_roofline_pareto/plot_throughput_ratio.py` |
| `59693752549c953bbe3f326d1eb103a283b85b6e1ee3af2b6891b4490687e1f3` | `tests/unit/performance/test_dsv4pro_vs_step4_throughput.py` |

### Final v3 figure menu

| SHA256 | Exact path |
|---|---|
| `caeb0c9af9081938aba210c55954836658108d6597802ccde33ce2978e3dab72` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_final_v3/README.md` |
| `2ac69e74cb1054e83e6077b7b79faa0e62a23d20ea2a5ce8690f69f414c9c3e0` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_final_v3/combined_results.json` |
| `a1059fa972b275ceaf9b7c8a6eef244a7a5aa423b3f8d333b945b2bc56c61b1c` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_final_v3/fig1_prefill_throughput_ratio.png` |
| `eca2b011c4624de469512243f4e441d5c7b9f73369b80f0bff8fac65f6c81e05` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_final_v3/fig2_decode_throughput_ratio.png` |

### Completed-system interim v3 menu

| SHA256 | Exact path |
|---|---|
| `94851936f9e5fd8ee307af7c7338f1b2e29ba6bdd0cb92915a52ba6c8f29cbba` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_completed_v3/README.md` |
| `f05601d7b2e5e8435b097044c95c485086649f11b3f5166d85e70233d4963a6b` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_completed_v3/combined_results.json` |
| `5c7f11767c144da2e6190118f547806ea982c3c86d120b7b10fedec915dfca88` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_completed_v3/fig1_prefill_throughput_ratio.png` |
| `150df14423505e3022185fa4110d0dad23afa7ccb038e9946412de3a35fe7978` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/figures_completed_v3/fig2_decode_throughput_ratio.png` |

### GB300 v3 shard

| SHA256 | Exact path |
|---|---|
| `339cb9eb68620fc1bc3f0fed01e8a2752c6aaad4fa9d7ba82aabe088a83f1bd0` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-gb300/mode_runs.sqlite3` |
| `0d9da64532ec12cad8f5c4c9498fc13fc08717cf76748da392609df183867304` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-gb300/model_comparisons.csv` |
| `e7c6a6414c6665d7d0021f9289c442808e94918108f15357c8e0ee35e952396a` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-gb300/ranked_rows.csv` |
| `48a163eb504aabb81ad8b2fd50c0a85eae87c4fcb021a1555a6f06536a8ae243` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-gb300/report.md` |
| `821cececa76f4412002df38522765534462742f275324b43b4523471658ef1b2` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-gb300/results.json` |

### H200 SXM v3 shard

| SHA256 | Exact path |
|---|---|
| `839cb418995168b1c97557d813cc4dad7eb5e350e14377247ad0465d8073d7a4` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h200_sxm/mode_runs.sqlite3` |
| `3caec4c4e87c123cda77d85856ad8546a2e48199c8d58ac0be13f3c884cf695e` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h200_sxm/model_comparisons.csv` |
| `322402f33e09dfb899b8060b2bd1e3cf39ba62cd20721b474f1d26ca4b5bcdd8` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h200_sxm/ranked_rows.csv` |
| `8a62a6285c6b25f72ad5f77cfe164ce1a691fe351ce183f09eed8cd25b19d328` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h200_sxm/report.md` |
| `b1d57e7ac7a58a97f80131cd2fba1e00adfc6809d359422934a48dac2585a0f7` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h200_sxm/results.json` |

### H100 SXM v3 shard

| SHA256 | Exact path |
|---|---|
| `c749152fac440bc785f29df62ce5d63fa9e83da0c1517d756bb42957f8a44ac6` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h100_sxm/mode_runs.sqlite3` |
| `d435d8b3c4fe0c72202067bb38fe4825b95649c6a2ac0bc6f2077fe71a517a35` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h100_sxm/model_comparisons.csv` |
| `326f06a993590d3efc72fafd5adefef5b8aa08ab3361df753dab6e1cc7e08f1f` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h100_sxm/ranked_rows.csv` |
| `bfa2c56d854292535b5e0efef085c066b23ad6a6b42857db4ba12f2d35598acf` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h100_sxm/report.md` |
| `05662b31152b796786c39b81c06de480d6f40757157d2cf128864d2e13518910` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h100_sxm/results.json` |

### H800 SXM v3 simulated-SOL shard

| SHA256 | Exact path |
|---|---|
| `2f836955152da52c05807a92bef76c5b73c6300d5791e65f4a0c07bdad1b49e1` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h800_sxm/mode_runs.sqlite3` |
| `a259d4e5f159186210a8838fb0d63eaa2c3f60100e79a1219b982e8f5d9dcaec` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h800_sxm/model_comparisons.csv` |
| `2c07d6e08191fccb51497f86df2b26a24e0ac7127eb25e168dd3ac263b4ebc80` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h800_sxm/ranked_rows.csv` |
| `a70bde49e0040b9393e0a200c8b6d55b70f6abd17fd2f302b5f9f6957c4d697d` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h800_sxm/report.md` |
| `572b16f3bb48c9f96de44d91e3116ba4d8c4a6196abd7053e6f2f4e7c7fe9728` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/full-v3-h800_sxm/results.json` |

### Task documentation

| SHA256 | Exact path |
|---|---|
| `28d68e81725fa104cae8079ea5947eac79ed6f620b6a211af4ceec451a17b93e` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/requirements.md` |
| `9fd88669a4695d8e60cbb224ec009b497045afa5f0b0edcc1e2f92b03dee0029` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/plan.md` |
| `8849ef2614739e10fc8c8bd040abdb59e44a3ebbda26fc3d7b4799f7133682c6` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/notes.md` |
| `8e2214012cf81d729d157022fc688be09a9aa7728daab89d3cfd66e37f70ee36` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/progress.md` |
| `4c76f232aa45e3b0c5959e3da3763ed8b037dbe3d2920532e058801487bb40d0` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/issues.md` |
| `531b7790406d748d9d54dcc224f0cbc53e7bdf0f16d318fa9e78c48c878a705d` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/review.md` |
| `f4055735d22ddc906346af18acdef1c21a109a6308e7ca88ab611e97708f4399` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/lessons.md` |
| `5cc2aadc0f988d7f589173b77e9c625328573d5c7be8aebf142195b182bed2b5` | `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/test_report_2026-07-14_dsv4pro_vs_step4_throughput.md` |

## Validation Status

### Execution matrix

| System | Mode runs | Identity coverage | Rank-one rows | Max rank-one TTFT (ms) | Strict `< 5000 ms` | Terminal counts | Status |
|---|---:|---|---:|---:|---|---|---|
| GB300 | 16 | 16/16 | 16 | 4176.999 | PASS | 58 success, 6 memory infeasible | PASS |
| H200 SXM | 16 | 16/16 | 12 | 3431.264 | PASS | 44 success, 14 memory infeasible, 6 SLA infeasible | PASS |
| H100 SXM | 16 | 16/16 | 11 | 3718.155 | PASS | 27 success, 34 memory infeasible, 3 SLA infeasible | PASS |
| H800 SXM simulated SOL | 16 | 16/16 | 11 | 3832.023 | PASS | 27 success, 34 memory infeasible, 3 SLA infeasible | PASS |

### Combined ratios and figures

| Validation | Expected | Actual | Status |
|---|---|---|---|
| Source shards | 4 | 4 | PASS |
| Systems | GB300, H200, H100, H800 | `gb300`, `h200`, `h100`, `h800` | PASS |
| Ratio direction | Step4 / DeepSeek-V4-Pro | Step4 / DeepSeek-V4-Pro | PASS |
| Requested system/ISL points | 32 | 32 | PASS |
| Paired + missing points | 32 | 23 + 9 | PASS |
| Stored ratio recomputation | 46 ratios, absolute delta ≤ `1e-12` | 46/46, maximum delta `0.0` | PASS |
| Prefill PNG | Valid and non-empty | 2160 × 1260 RGBA, 205229 bytes | PASS |
| Decode PNG | Valid and non-empty | 2160 × 1260 RGBA, 205541 bytes | PASS |
| Inventory SHA256 | All listed files match | 41/41 matched | PASS |
| Full mode-run identities | 64 | 64/64 | PASS |
| AA/AB/BA/BB terminal outcomes | 256 | 256/256 | PASS |
| Full rank-one provenance | 50 | 50/50 exact full-identity matches | PASS |

### Automated checks

| Check | Actual result | Status |
|---|---|---|
| Focused annotation regression | 1/1 passed in 0.06 seconds | PASS |
| Current-source v3 artifact audit | 64/64 identities, 256/256 terminals, 46/46 ratios | PASS |
| Source-unchanged affected performance suite | 177/177 passed in 3.15 seconds | PASS |
| Ruff check | 0 errors | PASS |
| Ruff format | 3 files already formatted | PASS |
| `git diff --check` | Exit code 0 | PASS |

Detailed commands, numerators, denominators, ratios, TTFT margins, and image evidence are recorded in `test_report_2026-07-14_dsv4pro_vs_step4_throughput.md`.

## Open Items/Future Extensions

- No required current-source v3 figure/menu item remains open.
- The separate 120-run GB300 corrected-v3 full-matrix job under `task_2026-07-10_step4_predefined_ops_plan` completed and strict-merged successfully; the parent-task integration dependency is closed and the throughput figures remain unchanged.
- H800 should continue to be interpreted as simulated SOL until a separately approved silicon-data experiment is available.
- Step4 ISL values at or above `65536` remain approximation-dominated because of the approved temporary MLA substitute; this provenance must be retained in downstream analysis.
- A future performance-engineering task may profile and optimize the repeated GB300 pyarrow/Python object-processing path. That optimization is outside this figure-generation task and was not required for correctness.
