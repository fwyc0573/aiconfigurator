## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-15 | Recorded final independent corrected-v3 approval and staging release. |
| 2026-07-15 | Replaced corrected-v2 delivery provenance with formatter-consistent corrected-v3 artifacts and a unique 50-path inventory. |
| 2026-07-15 | Replaced the superseded v1 archive with the corrected-v2 four-system merge and updated review/staging boundaries. |
| 2026-07-10 | Created the initial Step4 predefined-ops planning summary. |
| 2026-07-14 | Replaced the planning-only snapshot with the implemented and fully merged SOL-matrix delivery summary. |
| 2026-07-14 | Recorded final independent approval, reconciled WATCH evidence, and refreshed the final deliverable inventory. |
| 2026-07-14 | Recorded the corrected final completion verifier and refreshed mutable documentation hashes. |

# Summary: Step4 Predefined Ops

## Task Overview

This task implements Step4 as a dedicated local AIC model family and validates it against DeepSeek-V4-Pro using formula-only SOL/roofline execution. The implementation runs in the isolated `task/step4-predefined-ops` worktree based on the user-selected `step-design` source state; the original checkout was not modified by this worktree workflow.

The Step4 model follows the architecture-calculator dimensions and the Step4Air algorithm/precision evidence: 92 transformer layers, 4 dense FFN layers, 88 routed-MoE layers, gated SwiGLU, FP8 shared-expert projections, and BF16 norm/merge behavior. The current attention implementation is intentionally not presented as faithful Step4 Full/SWA attention. Per user direction, all 92 attention layers use a declared temporary DeepSeek-V3-style MLA substitute while retaining 23 Full and 69 SWA audit labels. Every Step4 row at ISL >= 65536 is marked approximation dominated.

The final comparison fixes `vllm`, backend version `0.22.0`, Python engine step, 64 total GPUs, `pp=1`, `tp<=8`, common EP/MoE-TP rows through worker width 32, `prefix=0`, `nextn=0`, chunked prefill disabled, and TTFT-only eligibility. Primary workloads use ISL 4K/16K/64K/256K/1M with OSL=1; decode smoke uses 4K/1024. Both aggregate and disaggregate serving are covered on GB300, H200, H100, and test-side H800. H800 is simulated SOL validation only.

All four formatter-consistent corrected-v3 120-run system shards were executed under execution contract `a13a4fe6ef9b932d01772ee3f0b8844760c52ec991fdcc5af641186baf1b697c` and merged with the existing strict API. The authoritative merged result contains 480 mode runs, 545 normalized/ranked rows, 89 paired model comparisons, and 28 unpaired feasibility outcomes. Merged strict resume executed zero loader or executor calls and preserved all five hashes. Final semantic validation found 20,280 executed SOL op-source entries, six explicit zero-latency no-op entries, and zero profiling/unknown sources. Corrected nested collective evidence contains 2,935 pre-dispatch, 4,338 post-dispatch, and 1,403 shared-FFN all-reduce entries, with zero entries mislabeled as the outer `generation_moe_overlap` operation.

The final author-independent native code review returned `APPROVE` with CRITICAL/HIGH/MEDIUM=`0/0/0`. The final formatter-consistent corrected-v3 StepCode Claude review also returned `APPROVE`, explicitly confirmed the previous stale-contract blocker is resolved, reported CRITICAL/HIGH/MEDIUM=`0/0/0`, and required no action before exact staging. Its two WATCH observations are recorded without code changes: the existing per-config sweep exception boundary and local retained v1/v2 historical evidence. All `.omx` review artifacts are local-only and intentionally excluded from the committed archive.

The earlier v1 and corrected-v2 matrices remain local historical records. v1 was superseded because the throughput execution contract originally omitted the wrapper source and nested collectives inherited the outer overlap operation name. Corrected-v2 fixed those defects but was executed before final Ruff formatting changed the shared runner bytes. Formatter-consistent corrected-v3 artifacts were generated only in new immutable directories; no old result was overwritten, migrated, scaled, or relabeled.

## Deliverables Inventory

### Implementation Files

| Path | Bytes | SHA256 |
|---|---:|---|
| `src/aiconfigurator/model_configs/stepfun-ai--Step4_config.json` | 2346 | `4763a5a3934bbbdbb81effe76abbbf67cc8d83f0b5aff716541498f040eea6d3` |
| `src/aiconfigurator/sdk/backends/base_backend.py` | 76056 | `9020cf1c9c9aed135a393ba98aa2a8a3f082d6b3eb285730a8f16fd32ad390cf` |
| `src/aiconfigurator/sdk/common.py` | 38907 | `708f0bde4d13bacf3ab5639dc6547b11e31491e225aaed97ce75d0a85cc951f1` |
| `src/aiconfigurator/sdk/communication_evidence.py` | 3242 | `31887e3983e11f225b6e30bc5358f9b297d1f82c445a4e3c4c055b86fd38d649` |
| `src/aiconfigurator/sdk/models/base.py` | 12752 | `5fddfc38e97d3127aa70e2053f08bf365bdef0fe9c0116a80b7b0b769fba96f9` |
| `src/aiconfigurator/sdk/models/helpers.py` | 17353 | `5f204bb40139f4c37ee13ad5c7cae4bc24c0eb726bad2fe2d1ed6208555401a7` |
| `src/aiconfigurator/sdk/models/step4.py` | 16610 | `586dd3b23b6fcc3bd18a585f836ba395ea074fdeb391a09c511cb5fd67d49cc6` |
| `src/aiconfigurator/sdk/operations/communication.py` | 33194 | `6fd54f665b2fccd25e9c4f72311eaf2ad286fd4743221a86e47a66cccb6fa219` |
| `src/aiconfigurator/sdk/operations/moe.py` | 133368 | `07413e6349df6c360d3e597b995d0d36ab4b4e9e16ca2aedc0beba9b8fe8f7cb` |
| `src/aiconfigurator/sdk/operations/overlap.py` | 7251 | `9025902f9f7c3ccbf2ef184063de996504b096733de32d897aff98d646800647` |
| `src/aiconfigurator/sdk/perf_database.py` | 112927 | `c8b6cee9441644931c5d6f9faa2c7804893d037a2f1ebb1e93533817a05383ef` |
| `src/aiconfigurator/sdk/sweep.py` | 58101 | `5de14e74773f1279d0ae7ddf803a126db60f24e06978dacc27f0670f5f75aece` |
| `src/aiconfigurator/sdk/system_spec.py` | 2551 | `5fb5c5119fa5f233fc8c0220d645f072b57abcf9052358a61e1d67059adad23b` |
| `src/aiconfigurator/sdk/task_v2.py` | 87989 | `146d9c251de1e802bde4185aa74c07fadf32d1ed5e3dbebeb9b6531cd08f99c4` |
| `src/aiconfigurator/sdk/utils.py` | 57921 | `b14e0c4c13dfcf94a29f80fb63be10a5a9b648131da8821a2d790bca671a5ada` |

### Test and Matrix Runner Files

| Path | Bytes | SHA256 |
|---|---:|---|
| `tests/integration/test_step4_prefill_ranking.py` | 1644 | `fbf3073d082c1c636e25e543ac179b0f4cdc008297a30a115a44ea6571b0708a` |
| `tests/performance/aic_roofline_pareto/run_step4_comparison.py` | 135443 | `adf4691779bbb85a92de67ca19f94f4087a7ec1e9fe6c40e84f4085f04816f9e` |
| `tests/performance/aic_roofline_pareto/systems/h800_sxm.yaml` | 1313 | `74b109e2cc30e3000b3c47707f6147d072361218873e3b60e4e12c795699dd26` |
| `tests/unit/collector/test_parallel_run.py` | 16963 | `0be81a84bd82c40706f8339faf6529aa1b8f28f849d0b6c8ad1e3f8ee1ed0197` |
| `tests/unit/performance/test_step4_comparison_checkpoint.py` | 33818 | `7808797d6bb1181fdabcfb1c98b94e5dbc4b2521c2f4fb410f99adc6f862eb20` |
| `tests/unit/performance/test_step4_comparison_cli.py` | 19730 | `7a52102e2694098deb6496defa290a17a15799bcb51728e160dc00f726c8f7c5` |
| `tests/unit/performance/test_step4_roofline_matrix.py` | 83390 | `e4c66688b74c1209a7430e1bbac851f3b1e17f3904520455d81f018ecc3d2b05` |
| `tests/unit/sdk/backends/test_base_backend.py` | 17783 | `9b680ededd116e8c7556bb53e89970632adb112fdaf0d3f7e8fc76fe0ca1d845` |
| `tests/unit/sdk/database/test_collective_query_capture.py` | 11346 | `ff97ea44b80fae634506ea93c8c53efb63a20c0927a1c6823529c1923559ed6e` |
| `tests/unit/sdk/database/test_communication_capture.py` | 6128 | `b44ae783659f040eb70eb15cbe75d2563643f2f8c95e74e4aae728cbb00d2118` |
| `tests/unit/sdk/database/test_moe_dispatch.py` | 21602 | `af4ac3fb31e67e55d561d11a0d3a89f8eb362943785b104784a8584e9e14df16` |
| `tests/unit/sdk/database/test_step4_roofline.py` | 14764 | `4386bd55e82aa88b95c89f98b1db72639183cb122a90c7f4118ebcc73f28cb60` |
| `tests/unit/sdk/models/test_step4.py` | 23998 | `24843edfa592f110fe4a2d4e00ac9a10b1781f6a7ef085e1a9cb00870809dee2` |
| `tests/unit/sdk/sweep/test_cluster_ranking.py` | 23003 | `9c4efdeaf4d1290cdcb69a12b54851a730c0e8374fb82007939c1b56588acfe9` |
| `tests/unit/sdk/sweep/test_sweep.py` | 23648 | `e6d09d03fd2cf861e66e00022a57c799e943ba94447bc4f67ccca6a3b9a80e5c` |
| `tests/unit/sdk/task_v2/test_task_config.py` | 73902 | `ad27ce12b713d77a19394cc047a1fd717960c4dc87f612867329b347ab00dc07` |
| `tests/unit/sdk/test_communication_evidence.py` | 3733 | `c7936cbb153c54c9e2627f640a98d3e0d6617f73429d8c823191c2947ca7d1ae` |
| `tests/unit/sdk/test_performance_result.py` | 8114 | `d2dd12c8aa25ef691499bb7657a4e4c93109aa61c744005b3fb213ac2d44b350` |
| `tests/unit/sdk/test_system_spec.py` | 2773 | `d7a8d0b53fe4471c5a24438be96e7f22898e5a2fb822d5876a283238c365850a` |

### Final Merged Result Files

| Path | Bytes | SHA256 |
|---|---:|---|
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/mode_runs.sqlite3` | 21286912 | `f304f70d3742ed954272c49e157c139916a4fe364d0b87606ebacd70efdf3167` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/results.json` | 41636981 | `372b683274771b70f3a5a94caafe08ac0289268140c038194995a5fa72c24a3e` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/ranked_rows.csv` | 10392173 | `8a4756c1f11a6946c9a45dd0f2cc6a3cef39894f239bc8a18e64a3233db2e897` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/model_comparisons.csv` | 108819 | `ca7fd0a7c3d3630fcda8974d42643ac2ab1f71c4222675ebebdf73cac2e1eb65` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/report.md` | 95030 | `46c1681b58bd91eb374dc5358ece8026a3d06ab1b0542fdd924c9613bf321528` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/comparison_summary.md` | 25221 | `93e558d422ae4afd4a9a195b2eb3cfa7cdba71614ba0c8d281855430596887c5` |

### Task Documentation

| Path | Bytes | SHA256 |
|---|---:|---|
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/requirements.md` | 11968 | `e202d507180ac833c880137b77df0a4aabafd7b4a565565f5d2e05a004c04514` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/notes.md` | 44796 | `6b14ea6fed96876674f0ffd661e8b3f300af70a42c9ce3e6e81104c90c09db90` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/plan.md` | 56296 | `7a850a4c4e644c4c8408056d8278a84d9da7c6b9fa0d04386e9b701a2ee63c2d` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/progress.md` | 206112 | `145024d78826fd5195b1711379e2b5e0f037eebe7f41e3d3f10a4dec26b43152` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/issues.md` | 131227 | `cb271e02d23d278511c69a0a45dda9a5b54a06dbae125bddd0eae39f2e722ea2` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/review.md` | 124145 | `5877c54d13f8e533ca9945d4c26d79ff9863d9ae24eecff2dc7a258cbab76fac` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/lessons.md` | 4577 | `9224d5e1d49d8dfdfddeac0032a10f7f216c0354986d8d2077c33141891fa2f3` |
| `task_memory/task_2026-07-10_step4_predefined_ops_plan/test_report_2026-07-14_step4_predefined_ops.md` | 39960 | `aa87d9ecd1a90c70cd351bfb46044bd635872dcdd0e33999f91dd95de5ac30b4` |
| `task_memory/env_handbook.md` | 9313 | `acde85159bcac41d790a2afba58d8f4c2bc6792a5135c0b0bb5205ea610e3c5b` |

### Local-only Independent Review Evidence (Not Committed)

| Path | Bytes | SHA256 |
|---|---:|---|
| `.omx/artifacts/claude-you-are-the-independent-final-post-change-code-and-architect-2026-07-13T19-56-13-078Z.md` | 14430 | `9d7eb4468e1e76be3dfc051639c53fc58d7c97e770d55216d2e435e0df23d5d3` |
| `.omx/artifacts/claude-you-are-the-independent-read-only-reviewer-for-a-long-step4--2026-07-13T20-24-37-729Z.md` | 7018 | `cd9a9710d0f46a2aa2d15c5525ebdbb96175dbd9db3eeefd37bfbde2c334e223` |
| `.omx/artifacts/claude-you-are-the-final-independent-read-only-completion-reviewer--2026-07-14T02-29-46-655Z.md` | 8481 | `77bbe32d1d128c5457a8440915d1777b27176e0c1a2ec54129f2ed9436187d86` |
| `.omx/artifacts/claude-you-are-the-independent-cross-model-final-pre-commit-and-pre-2026-07-14T18-20-53-245Z.md` | 8730 | `f244559a7af91c31ed36db93e700ef4fe8ee0be45bbb4de7262e75f90db31e84` |
| `.omx/artifacts/ask-claude-step4-final-v3-precommit-review-20260715.md` | 9835 | `f1beae4f04a09331f8ca6692172bf7ea3be1f96ac2e17440c02072c71e82770c` |

`summary.md` is intentionally excluded from its own hash inventory to avoid a self-referential digest. Its final SHA256 is reported externally after the last write.

## Validation Status

### Test Outcome Matrix

| Validation | Expected | Actual | Status |
|---|---:|---:|---|
| Primary-prefill focused RED | New behavior absent | 15/15 failed | Expected RED |
| Primary-prefill focused GREEN | All pass | 15/15 passed | PASS |
| Primary-prefill broad regression | All pass | 401/401 passed | PASS |
| Real H200 integration | All pass | 1/1 passed | PASS |
| Checkpoint suite | All pass | 16/16 passed | PASS |
| Runner/checkpoint regression | All pass | 116/116 passed | PASS |
| CLI/artifact affected regression | All pass | 147/147 passed | PASS |
| Unknown-exception/SOL-admission focused | All pass | 28/28 passed | PASS |
| Gate 6/7 related regression | All pass | 109/109 passed | PASS |
| Communication/Python identity combined | All pass | 413/413 passed | PASS |
| TPOT zero-baseline valid RED | New contract absent | 6 failed, 119 passed | Expected RED |
| TPOT post-format combined GREEN | All pass | 419/419 passed | PASS |
| Full unit regression | Zero failures | 2002 passed, 12 skipped, 1119 deselected, 4 warnings | PASS |
| Full unit exit code / elapsed | 0 | 0 / 765.21s | PASS |
| Full matrix shards | 4 × 120 | 480/480 | PASS |
| Merged strict resume | 480 records, 0/0 calls, 5/5 hashes | 480, 0/0, 5/5 | PASS |
| Merged semantic validation | Exit 0 | Exit 0 | PASS |
| Ruff / format / diff / doc-whitespace checks | All pass | Exit 0 / exit 0 / exit 0 / 0 lines across 10 docs | PASS |
| Final independent completion review | Non-BLOCK verdict required | Corrected-v3 `APPROVE`; stale blocker resolved; 0/0/0; required actions=none | PASS |
| Final completion verifier | All artifact, identity, semantic, inventory, review, unit-log, static-log, and doc checks pass | Merged 5/5; primary 49+1; figures 41; tests/static PASS | PASS |

### Full Matrix Metrics

| Metric | Expected | Actual | Delta / Status |
|---|---:|---:|---|
| Common parallel rows | 21 | 21 | 0 |
| Pattern A / B | 17 / 4 | 17 / 4 | 0 / 0 |
| EP>8 / EP16 / EP32 | 8 / 4 / 4 | 8 / 4 / 4 | 0 / 0 / 0 |
| Max worker width | 32 | 32 | 0 |
| AA / AB / BA / BB | 289 / 68 / 68 / 16 | 289 / 68 / 68 / 16 | 0 |
| Mode runs | 480 | 480 | 0 |
| Per system | 120 | 120 each | 0 |
| Normalized / ranked rows | Equal | 545 / 545 | 0 |
| Step4 / DeepSeek rows | Informational | 348 / 197 | N/A |
| Paired / unpaired comparisons | Preserve both | 89 / 28 | PASS |
| Terminal success / memory / SLA | Typed only | 545 / 655 / 240 | PASS |
| Executed SOL / explicit no-op sources | No profiling | 20,280 / 6 | PASS |
| TPOT computed / zero-zero pairs | Explicit status | 59 / 30 | PASS |
| H800 simulation labels | simulated only | simulated only, 88 rows | PASS |
| ISL 256K / 1M mode runs | 80 per ISL | 80 / 80 | 0 / 0 |
| ISL 256K / 1M normalized rows | Typed terminal outcomes allowed | 0 / 0 | Documented feasibility boundary |

### Comparison Outcome Matrix

| System | Paired | Unpaired | Step4-only | DeepSeek-only | Boundary |
|---|---:|---:|---:|---:|---|
| GB300 | 33 | 2 | 0 | 2 | SOL model |
| H200 | 28 | 2 | 2 | 0 | SOL model |
| H100 | 14 | 14 | 14 | 0 | SOL model |
| H800 | 14 | 10 | 10 | 0 | simulated SOL |

All exact per-key model values and deltas are retained in `model_comparisons.csv`. Paired aggregates by system, workload, primary ISL, TTFT SLA, serving mode, and metric are in `comparison_summary.md`. Unpaired keys are not imputed.

## Open Items/Future Extensions

1. ISL 256K and 1M each executed all 80 required mode runs but produced zero normalized rows at the 64-GPU limit. A future 128/256-GPU study may test whether the typed memory/SLA terminal boundary moves; the current task must not fabricate long-context comparisons.
2. Replace the all-92-layer temporary MLA substitute only when authoritative Step4 Full/SWA attention geometry and a reviewed SOL implementation are available. Current ISL >= 65536 Step4 comparisons must remain labeled approximation dominated.
3. Treat H800 as simulated SOL until real H800 silicon/perfdb validation is separately authorized and executed. Do not reinterpret the current test-side system specification as measured evidence.
4. Revisit ISSUE-060 only if the governing contracts change: P2P tier selection, decode `cp>1`, generic mode identity, or embedding resume evidence into final JSON.
5. Preserve the current unpaired feasibility outcomes in downstream analysis. Any requirement for matched-feasibility-only conclusions should be a separately defined analysis rather than silent row deletion or imputation.
6. Broaden CI integration coverage beyond the current H200/4K point if the matrix runner becomes a regularly exercised product surface. This is not a delivery gap: the complete 480-run matrix provides the required four-system and workload evidence, and the exact 21-row search space already has a dedicated unit test.
