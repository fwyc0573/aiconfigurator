## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-15 | Added corrected final artifact/inventory validation and independent corrected-v3 pre-commit APPROVE |
| 2026-07-15 | Added formatter-consistent corrected-v3 merge, focused/integration/static evidence, and environment root-cause records |
| 2026-07-15 | Added primary GB300 completion and authoritative corrected-v2 four-system merge, strict-resume, artifact, and semantic evidence |
| 2026-07-15 | Added corrected-v2 blocker RED/GREEN evidence, throughput-menu validation, completed primary-shard audit, and final independent APPROVE |
| 2026-07-14 | Added failed-then-corrected final completion-verifier evidence and root cause. |
| 2026-07-14 | Added final independent completion-review approval and WATCH reconciliation evidence. |
| 2026-07-14 | Added the final Step4 predefined-ops implementation and full SOL-matrix validation report. |

# Test Report: Step4 Predefined Ops and SOL Matrix

**Date**: 2026-07-15
**Result**: PASS for implementation, formatter-consistent corrected-v3 four-system matrix execution, strict merge/resume, merged semantic validation, focused/integration regression, and static checks. Exact inventory, independent review, staging, commit/push, and target integration remain delivery gates.

## 1. Test Script Information

### 1.1 Environment

| Item | Value |
|---|---|
| Worktree | `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops` |
| Branch | `task/step4-predefined-ops` |
| Pinned base HEAD | `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1` |
| Conda environment | `/home/i-fengyicheng/miniconda3/envs/aic-step-design` |
| Python | `3.11.15` |
| pytest | `8.4.2` |
| pandas | `3.0.3` |
| aiconfigurator | `0.10.0` |
| Source binding | `PYTHONPATH="$PWD/src:$PWD"` |

Every pytest, runner, and validator command used the isolated worktree and explicit source binding. Commands using `tee` enabled `set -o pipefail` and propagated the Python/pytest exit code.

### 1.2 Test and Runner Paths

Core implementation tests:

- `tests/unit/sdk/models/test_step4.py`
- `tests/unit/sdk/database/test_step4_roofline.py`
- `tests/unit/sdk/database/test_collective_query_capture.py`
- `tests/unit/sdk/database/test_communication_capture.py`
- `tests/unit/sdk/test_communication_evidence.py`
- `tests/unit/sdk/test_system_spec.py`
- `tests/unit/sdk/sweep/test_cluster_ranking.py`
- `tests/unit/sdk/sweep/test_sweep.py`
- `tests/unit/sdk/task_v2/test_task_config.py`
- `tests/unit/sdk/backends/test_base_backend.py`
- `tests/unit/performance/test_step4_roofline_matrix.py`
- `tests/unit/performance/test_step4_comparison_checkpoint.py`
- `tests/unit/performance/test_step4_comparison_cli.py`
- `tests/integration/test_step4_prefill_ranking.py`

Matrix runner and system fixture:

- `tests/performance/aic_roofline_pareto/run_step4_comparison.py`
- `tests/performance/aic_roofline_pareto/systems/h800_sxm.yaml`

Final result and comparison paths:

- `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/results.json`
- `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/ranked_rows.csv`
- `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/model_comparisons.csv`
- `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/report.md`
- `task_memory/task_2026-07-10_step4_predefined_ops_plan/comparison_summary.md`

### 1.3 Reproducible Commands

Common setup:

```bash
cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops
export PYTHONPATH="$PWD/src:$PWD"
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python
```

Focused Step4 model and strict-roofline tests:

```bash
"$PY" -m pytest -p no:cacheprovider \
  tests/unit/sdk/models/test_step4.py \
  tests/unit/sdk/database/test_step4_roofline.py \
  tests/integration/test_step4_prefill_ranking.py -v
```

Performance runner, checkpoint, and CLI tests:

```bash
"$PY" -m pytest -p no:cacheprovider \
  tests/unit/performance/test_step4_roofline_matrix.py \
  tests/unit/performance/test_step4_comparison_checkpoint.py \
  tests/unit/performance/test_step4_comparison_cli.py -v
```

Communication, sweep, Task, and backend regression:

```bash
"$PY" -m pytest -p no:cacheprovider \
  tests/unit/sdk/database/test_collective_query_capture.py \
  tests/unit/sdk/database/test_communication_capture.py \
  tests/unit/sdk/test_communication_evidence.py \
  tests/unit/sdk/test_system_spec.py \
  tests/unit/sdk/sweep/test_cluster_ranking.py \
  tests/unit/sdk/sweep/test_sweep.py \
  tests/unit/sdk/task_v2/test_task_config.py \
  tests/unit/sdk/backends/test_base_backend.py -v
```

Full unit regression:

```bash
set -o pipefail
LOG=/tmp/step4_post_change_full_unit.log
"$PY" -m pytest -p no:cacheprovider -m unit 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "PYTEST_EXIT_CODE=$status" | tee -a "$LOG"
exit "$status"
```

Single-process reproduction of the same full 480-run matrix contract in a fresh directory:

```bash
set -o pipefail
OUT="task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-reproduction-$(date +%Y%m%d-%H%M%S)"
"$PY" tests/performance/aic_roofline_pareto/run_step4_comparison.py \
  --output-dir "$OUT" \
  --systems-paths default,tests/performance/aic_roofline_pareto/systems \
  --initial-agg-cap 1024 \
  --initial-prefill-cap 16 \
  --initial-decode-cap 1024 2>&1 | tee /tmp/step4_full_matrix_reproduction.log
status=${PIPESTATUS[0]}
exit "$status"
```

The actual long execution used four exact `120`-spec system partitions through the existing module API because filtered `--system` CLI runs intentionally fingerprint different execution contracts. All four partitions used the same full contract SHA256 and were merged only with `merge_completed_checkpoints()`. The exact partitioning, identity assertions, directories, and logs are recorded in `progress.md` and Checkpoints 65-67 of `review.md`.

Fresh static checks:

```bash
"$PY" -m ruff check .
"$PY" -m ruff format --check .
git diff --check
grep -nE '[[:blank:]]+$' \
  task_memory/task_2026-07-10_step4_predefined_ops_plan/{plan.md,requirements.md,notes.md,progress.md,issues.md,review.md,summary.md,lessons.md,comparison_summary.md,test_report_2026-07-14_step4_predefined_ops.md}
```

## 2. Validation Criteria

### 2.1 Structural and Modeling Criteria

1. Step4 resolves locally as `stepfun-ai/Step4`, family `STEP4`, without a network dependency.
2. Step4 implements the calculator-defined 92-layer composition with `4` dense FFN layers and `88` routed-MoE layers.
3. Dense and shared FFN use gated SwiGLU; shared-expert projections are FP8 while norm and routed/shared merge are BF16.
4. All Step4 latency estimation is formula-only `SOL` or `SOL_FULL`; no profiling-based perfdb source is admitted.
5. The current attention limitation is explicit: all 92 Step4 attention layers use the temporary MLA substitute, retain `23` Full and `69` SWA audit labels, and mark ISL >= 65536 rows approximation dominated.
6. All matrix Tasks bind `engine_step_backend="python"`, `backend="vllm"`, `backend_version="0.22.0"`, `total_gpus=64`, `prefix=0`, `nextn=0`, and chunked prefill disabled.
7. H800 is labeled simulated SOL and is never described as silicon validation.

### 2.2 Search-Space and Workload Criteria

| Criterion | Expected |
|---|---:|
| Common parallel rows | 21 |
| Pattern A / Pattern B | 17 / 4 |
| EP > 8 rows | 8 |
| EP=16 rows | 4 |
| EP=32 rows | 4 |
| Maximum worker width | 32 GPUs |
| Disaggregate AA / AB / BA / BB | 289 / 68 / 68 / 16 |
| Total disaggregate base pairs | 441 |
| Matrix points before serving-mode expansion | 240 |
| Mode runs after agg/disagg expansion | 480 |
| Mode runs per system | 120 |
| Mode runs per model | 240 |
| Mode runs per serving mode | 240 |

Primary workloads use binary ISL values `4096`, `16384`, `65536`, `262144`, and `1048576` with `OSL=1`. The independent decode smoke uses `ISL=4096`, `OSL=1024`. Both sweep TTFT SLA values `200`, `500`, `1000`, `2000`, and `5000` ms.

### 2.3 Ranking, SLA, and Delta Criteria

- TTFT is the only eligibility SLA; every published row must have `ttft_pass=true` and `ttft <= ttft_sla_ms`.
- TPOT must be published with `tpot_observed_only=true` and must not change eligibility.
- Primary `OSL=1` ranks by fixed-cluster Prefill input throughput.
- Decode smoke ranks by fixed-cluster output-token throughput.
- Absolute delta is `Step4 - DeepSeek-V4-Pro`; relative delta uses DeepSeek as baseline.
- Only exact TPOT `0/0` may emit `absolute_delta=0`, `relative_delta=null`, and `status=zero_baseline_both_zero`.
- Unknown exceptions and unknown terminal statuses must fail fast; only `success`, `memory_infeasible`, and `sla_infeasible` are publishable.

### 2.4 Resume and Artifact Criteria

- Each system shard must strict-resume with `120` records, loader/executor calls `0/0`, and five unchanged hashes.
- The merged checkpoint must strict-resume with `480` records, loader/executor calls `0/0`, and five unchanged hashes.
- Merged output must contain SQLite, JSON, ranked CSV, comparison CSV, and Markdown report.
- Full identity must equal:
  - execution contract `63aa70620a6fa8908bf0747390d3833756af6b88c0a11aecb21e63cd81da8297`;
  - matrix spec `78b4970381ca7d0fb7bdfb53619a3280b72f936703c3c4d813bd23b96398edd4`;
  - git HEAD `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1`.

## 3. Test Results and Evidence

### 3.1 TDD and Focused Regression

| Suite / Gate | Result | Actual Evidence |
|---|---|---|
| Step4 strict-roofline model tests | PASS | `tests/unit/sdk/database/test_step4_roofline.py`: `12/12` passed in `7.51s` before later expansion |
| Comparison runner baseline | PASS | `95/95` passed before primary-prefill ranking change |
| Primary-prefill ranking RED | Expected FAIL | `15/15` focused tests failed before missing APIs/behavior were implemented |
| Primary-prefill ranking GREEN | PASS | `15/15` focused tests passed; two-file suite `120/120`; broad regression `401/401` in `45.65s`; integration `1/1` |
| Compact checkpoint RED/GREEN | PASS | Initial RED `4 failed, 8 passed`; final checkpoint suite `16/16`; runner/checkpoint regression `116/116` |
| CLI/artifact/shard merge RED/GREEN | PASS | Initial CLI/artifact RED `6/6 failed`; merge RED `2/2 failed`; final affected regression `147/147` |
| Unknown-exception and SOL-admission gates | PASS | Focused `28/28`; related regression `109/109` |
| Communication provenance / Python identity | PASS | Performance `156/156`; SDK/capture `257/257`; combined post-format `413/413` |
| TPOT zero-baseline RED | Expected FAIL | Valid RED `6 failed, 119 passed` before the narrow TPOT contract was implemented |
| TPOT zero-baseline GREEN | PASS | Targeted `125/125`; performance `162/162`; post-format combined `419/419` in `52.69s` |

All implementation failures above were intentional RED evidence or were root-caused test/validator defects. No implementation change was accepted while an affected test remained failing.

### 3.2 Full Unit Regression

Command log: `/tmp/step4_post_change_full_unit.log`
Log SHA256: `068918e1c6d87b80c5f6187ba1725adee905e16e203b025cc666cc343b88b04e`

| Metric | Expected | Actual | Delta / Result |
|---|---:|---:|---|
| Pytest exit code | 0 | 0 | PASS |
| Failures | 0 | 0 | 0 |
| Collected tests | Informational | 3111 | N/A |
| Selected tests | Informational | 1992 | N/A |
| Passed tests | All non-skipped selected tests | 1986 | PASS |
| Skipped tests | Report explicitly | 12 | Reported |
| Deselected tests | Report explicitly | 1119 | Reported |
| Warnings | Report explicitly | 4 | Reported |
| Elapsed time | Informational | 829.31s | 13m49s |

### 3.3 Four-System Full Matrix Shards

| System | Validation Type | Mode Runs | Normalized / Ranked | Fresh Exit | Resume Records | Loader / Executor | Hashes Unchanged |
|---|---|---:|---:|---:|---:|---:|---:|
| GB300 | SOL model | 120/120 | 197 / 197 | 0 | 120 | 0 / 0 | 5/5 |
| H200 SXM | SOL model | 120/120 | 162 / 162 | 0 | 120 | 0 / 0 | 5/5 |
| H100 SXM | SOL model | 120/120 | 98 / 98 | 0 | 120 | 0 / 0 | 5/5 |
| H800 SXM | **simulated SOL** | 120/120 | 88 / 88 | 0 | 120 | 0 / 0 | 5/5 |

Shard artifact SHA256 values:

| System | SQLite | results.json | ranked_rows.csv | model_comparisons.csv | report.md |
|---|---|---|---|---|---|
| GB300 | `18ef3d97...cd96` | `363a0d5f...6617` | `6c3e6f4f...d0e9` | `44f97e20...1225` | `9907ce51...615` |
| H200 | `c43d9c14...0ab` | `4b1556b0...239b` | `f02f01c9...a71f` | `2e1772dd...379c` | `a0743042...4741` |
| H100 | `6bccb5ac...500` | `dfd69581...b275` | `f5166d35...679` | `5a3e1efc...f6f` | `1f8b4754...24f6` |
| H800 simulated | `38054404...8ee` | `a56357f3...2895` | `ab91b52a...70b9` | `85531e09...1447` | `a69fa68a...ef3` |

### 3.4 Strict Full Merge and Resume

Merge log: `/tmp/step4_full_matrix_merge_20260714_v1.log`
Merge log SHA256: `1c3642010becb49d94afaf9a0a8e04dfef2197f2ea0660688898595dc00dd8e9`
Resume log: `/tmp/step4_full_matrix_merged_20260714_v1_resume.log`
Resume log SHA256: `f36cf694c81bdaa8e1d31ca7711ec3d996919ddd41f1a181f5e505060ab65f74`

| Metric | Expected | Actual | Delta / Result |
|---|---:|---:|---|
| Merged mode runs | 480 | 480 | 0 |
| Mode runs per system | 120 | 120 each | 0 |
| Normalized rows | Informational | 545 | N/A |
| Ranked rows | Equal normalized rows | 545 | 0 |
| Paired comparisons | Informational | 89 | N/A |
| Unpaired comparisons | Preserve, do not fabricate | 28 | PASS |
| H800 simulated rows | Informational | 88 | N/A |
| Merge exit code | 0 | 0 | PASS |
| Resume records | 480 | 480 | 0 |
| Resume loader calls | 0 | 0 | 0 |
| Resume executor calls | 0 | 0 | 0 |
| Unchanged artifact hashes | 5/5 | 5/5 | PASS |
| Resume exit code | 0 | 0 | PASS |

Merged artifact hashes:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `mode_runs.sqlite3` | 21,164,032 | `d5fc2781445ec4d4d8f60ed85c38f1ae0d638e8edffe0a9f06c768eb562f39c4` |
| `results.json` | 41,332,418 | `40b3dab1f44da8c810973a446941ad3ea93c210f741f853d892e1997599edd4e` |
| `ranked_rows.csv` | 10,305,155 | `b4eede4cc04a31b96f7683b47910b213848c016ebf8735e11bd3b63fba76a3fd` |
| `model_comparisons.csv` | 108,819 | `ca7fd0a7c3d3630fcda8974d42643ac2ab1f71c4222675ebebdf73cac2e1eb65` |
| `report.md` | 95,030 | `46c1681b58bd91eb374dc5358ece8026a3d06ab1b0542fdd924c9613bf321528` |

### 3.5 Merged Semantic Validation

Validator log: `/tmp/step4_full_matrix_semantic_validation_20260714_v1.log`
Validator log SHA256: `5c6b9fc69743da3d994e367463897c6fda32e0fc20739b2af22bddc76bee84e9`

| Contract | Expected | Actual | Result |
|---|---:|---:|---|
| Header schema | 3 | 3 | PASS |
| Models | 2 | 2 | PASS |
| Systems | 4 | 4 | PASS |
| Serving modes | 2 | 2 | PASS |
| Model-system cells | 8 cells × 60 | 8 cells × 60 | PASS |
| Primary / decode mode runs | 400 / 80 | 400 / 80 | PASS |
| Common search rows | 21 | 21 | PASS |
| Pattern A / B | 17 / 4 | 17 / 4 | PASS |
| EP>8 / EP16 / EP32 | 8 / 4 / 4 | 8 / 4 / 4 | PASS |
| Max worker width | 32 | 32 | PASS |
| AA / AB / BA / BB | 289 / 68 / 68 / 16 | 289 / 68 / 68 / 16 | PASS |
| Executed SOL op-source entries | >0, no profiling | 20,280 | PASS |
| Explicit zero-latency no-op entries | Only named `not_executed` | 6 | PASS |
| Unknown/profiling op sources | 0 | 0 | PASS |
| Step4 / DeepSeek normalized rows | Informational | 348 / 197 | N/A |
| Step4 temporary-MLA rows | All Step4 rows | 348/348 | PASS |
| Approximation-dominated Step4 rows | ISL >= 65536 only | 18 | PASS |
| H800 simulation labels | `simulated` only | `simulated` only | PASS |
| Success terminals | Equal normalized rows | 545 | PASS |
| Memory terminals | Typed only | 655 × `InsufficientMemoryError` | PASS |
| SLA terminals | Typed only | 240 × `NoFeasibleConfigError` | PASS |
| Unknown terminal statuses | 0 | 0 | PASS |
| TPOT computed pairs | Informational | 59 | N/A |
| TPOT zero-zero pairs | Explicit null/status | 30 | PASS |
| Primary-disaggregate TPOT zero-zero | Explicit null/status | 29 | PASS |
| Semantic exit code | 0 | 0 | PASS |

### 3.6 Step4 vs DeepSeek-V4-Pro Numeric Comparison

The exact `117` aligned comparison keys are in `model_comparisons.csv`: `89` paired and `28` unpaired. Grouped tables with actual values by system, workload, primary ISL, TTFT SLA, serving mode, and metric are in `comparison_summary.md`.

Coverage:

| System | Paired | Unpaired | Step4-only | DeepSeek-only |
|---|---:|---:|---:|---:|
| GB300 | 33 | 2 | 0 | 2 |
| H200 | 28 | 2 | 2 | 0 |
| H100 | 14 | 14 | 14 | 0 |
| H800 simulated SOL | 14 | 10 | 10 | 0 |

Overall paired means:

| Metric | Step4 Mean | DeepSeek Mean | Mean Absolute Delta | Mean Relative Delta | Relative n | Step4 Wins | Ties | DeepSeek Wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ranking metric | 500,093.11677354155 | 1,116,074.3099285932 | -615,981.1931550517 | +170.05786349371137% | 89 | 67 | 0 | 22 |
| Request latency (ms) | 27,750.14061500003 | 32,823.90791374707 | -5,073.767298747037 | +84.31384412353693% | 89 | 41 | 0 | 48 |
| TPOT (ms/token) | 252.67879732872262 | 266.39717603322543 | -13.71837870450285 | +17.863379178697314% | 59 | 30 | 30 | 29 |
| TTFT (ms) | 1,172.6216089809693 | 1,033.8967406897225 | +138.7248682912468 | +69.60752261267367% | 89 | 49 | 0 | 40 |

Interpretation constraints:

- `ranking_metric_value` is Prefill input throughput for primary workloads and output-token throughput for decode smoke; the raw workload-specific tables must be used for engineering decisions.
- Relative means can be skewed by small DeepSeek baselines and are not equivalent to the ratio of the displayed means.
- Unpaired rows reflect one-model feasibility and are retained without fabricated deltas.
- H800 values are simulated SOL only.
- Long-context Step4 comparisons at ISL >= 65536 are dominated by the declared temporary MLA approximation and are not faithful Full/SWA attention predictions.

### 3.7 Failures, Root Causes, and Resolutions

| Failure | Root Cause | Resolution | Final Evidence |
|---|---|---|---|
| Direct filtered system shards had incompatible contract hashes | CLI filters systems before execution-contract construction | Build one full contract, execute exact partitions through existing API, strict-merge only after 120/120 completion | Checkpoints 65-66; merged 480/480 PASS |
| H200 primary v2 finalization rejected TPOT 0/0 | Generic relative-delta rule did not distinguish the physical OSL=1 TPOT boundary | Narrow TDD fix for TPOT both-zero only; no epsilon/scaling/fallback | Targeted 125/125; combined 419/419; fresh v3 PASS |
| Semantic validator used `ParallelRow.num_gpu` | Disposable validator guessed a nonexistent field | Read dataclass; use `worker_gpus` property | Final staged semantic PASS |
| Semantic validator expected DeepSeek groups `null` | Normalized schema represents no groups as `{}` | Align validator to declared schema | Final staged semantic PASS |
| Semantic validator rejected six `not_executed` entries | It conflated explicit zero-latency no-op provenance with executed-op source | Require exact no-op name and zero latency; require every executed op source=`sol` | SOL=`20,280`, no-op=`6`, profiling=`0` |
| Semantic validator required identical top-level data/source phases | Aggregate `per_ops_data` includes non-op scheduling metadata | Allow only extra `scheduling`; require exact keys within every actual op phase | Final staged semantic PASS |

No production fallback, checkpoint migration, epsilon, scaling factor, profiling-based Step4 data, or custom merge logic was added to resolve these failures.

## 4. Final Static and Review Gates

### 4.1 Corrected Static Gate

Log: `/tmp/step4_final_static_checks_20260714_v1.log`

Log SHA256: `8bbeca1fec3b073e60aab6c1f50ed7475f91124bcd87ea95e7025f50a4f13ffa`

| Check | First Pass | Corrected Full Rerun | Final Result |
|---|---:|---:|---|
| `ruff check .` | Exit 0 | Exit 0 | PASS |
| `ruff format --check .` | Exit 0; 426 files formatted | Exit 0; 426 files formatted | PASS |
| `git diff --check` | Exit 0 | Exit 0 | PASS |
| Task docs scanned | 10 | 10 | PASS |
| Trailing-whitespace lines | 6 | 0 | PASS after exact repair |

The six first-pass lines were intentional Markdown hard breaks in this report, not implementation or result defects. They were removed, and the entire static sequence was rerun rather than accepting partial earlier evidence.

A second pre-review rerun after the final report/summary updates also passed: Ruff lint exit=`0`; Ruff format exit=`0`; `git diff --check` exit=`0`; docs scanned=`10`; trailing-whitespace lines=`0`. Log: `/tmp/step4_pre_review_static_checks_20260714_v2.log`; SHA256=`f64927ae50cba523ebe0ac87bc688428c76a4632a0c64456349e9d309061d06c`.

### 4.2 Independent Completion Review

| Item | Result |
|---|---|
| Provider | StepCode Claude |
| Model / effort | `claude-opus-4-6[1m]` / `max` |
| Review mode | READ-ONLY, author-independent |
| Leading verdict | `APPROVE` |
| Required actions | None |
| Artifact | `.omx/artifacts/claude-you-are-the-final-independent-read-only-completion-reviewer--2026-07-14T02-29-46-655Z.md` |
| Artifact SHA256 | `77bbe32d1d128c5457a8440915d1777b27176e0c1a2ec54129f2ed9436187d86` |

The reviewer accepted every requested contract: Step4 dimensions and precision, all-92-layer temporary MLA labeling, SOL-only execution, matrix and search identity, typed terminals, actual communication provenance, TPOT zero-zero behavior, four shards, strict merge/resume, simulated-only H800 labels, and unit/matrix/semantic/static evidence.

WATCH reconciliation:

1. ISL 256K and 1M each have `80` mode runs and zero normalized rows (`160` runs combined). Their experiment terminals are memory/SLA=`160/80` and `225/15`, respectively. This is a documented 64-GPU feasibility boundary, not missing execution.
2. The review stated that the 21-row assertion lacks a dedicated unit test. Direct repository evidence closes this observation: `tests/unit/performance/test_step4_roofline_matrix.py:81-113` defines `test_common_vllm_parallel_rows_match_authoritative_21_row_space` and asserts `len(rows) == 21` together with exact row content.
3. `tests/integration/test_step4_prefill_ranking.py` has one H200/4K test function. Broader CI integration coverage is a valid future extension, but it is non-blocking because the finalized `480` mode runs provide complete required system/workload evidence.

No source, test, checkpoint, matrix result, or comparison artifact was changed during or after the independent review.

### 4.3 Final Completion Verification

Final static rerun before completion verification:

| Check | Actual | Result |
|---|---:|---|
| Ruff lint | Exit 0 | PASS |
| Ruff format | Exit 0; 426 files formatted | PASS |
| `git diff --check` | Exit 0 | PASS |
| Task docs / trailing-whitespace lines | 10 / 0 | PASS |

Static log: `/tmp/step4_final_completion_static_checks_20260714_v3.log`; bytes=`305`; SHA256=`0da22c7e5cc4eb6240669ee570b1bd724fb15a442bf2ad49ac2aa245afd32a3e`.

The first completion-verifier run failed after the merged, semantic, inventory, and independent-review checks had passed. Root cause: the full-unit log contains ANSI color escapes, while the disposable validator searched raw bytes for one contiguous plain-text pytest summary. The raw full-unit log SHA256 and `PYTEST_EXIT_CODE=0` were correct. No source, test, or result artifact failed.

The validator was corrected to strip ANSI escapes only for the semantic summary match while retaining exact raw-file hash and exit checks. The entire completion validator was rerun from the first merged hash assertion.

| Validation | Expected | Corrected Actual | Result |
|---|---:|---:|---|
| Merged artifact hashes | 5/5 | 5/5 | PASS |
| SQLite header / mode-run rows | 1 / 480 | 1 / 480 | PASS |
| Contract / matrix / HEAD | Exact frozen values | Exact match | PASS |
| Step4 / DeepSeek normalized rows | 348 / 197 | 348 / 197 | PASS |
| SOL / explicit no-op sources | 20,280 / 6 | 20,280 / 6 | PASS |
| Paired / unpaired comparisons | 89 / 28 | 89 / 28 | PASS |
| TPOT computed / zero-zero | 59 / 30 | 59 / 30 | PASS |
| H800 simulated rows | 88 | 88 | PASS |
| Summary inventory | All entries | 51/51 | PASS |
| Final review | `APPROVE`, no required actions | Exact match | PASS |
| Full unit | 1986 passed, 12 skipped, 0 failed | Exact match after ANSI normalization | PASS |
| Task-doc trailing whitespace | 0 | 0 across 10 docs | PASS |

Failure log: `/tmp/step4_final_completion_verification_20260714_v1.log`; bytes=`1122`; SHA256=`db33fc3229d8c3e773071364be815e03a88949a1363f5edc7ed8c8a6e360e7ea`; exit=`1`.

Corrected full-rerun log: `/tmp/step4_final_completion_verification_20260714_v2.log`; bytes=`1327`; SHA256=`582590688d065a8e88134f5c336509b51bfcca7005e43e643983d464036cc9d1`; exit=`0`.

## 5. Phase 6 Corrected-v2 Review and Regeneration

### 5.1 Execution-contract and nested-collective provenance TDD

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Strict RED | New contracts fail before remediation | 3 failed | Expected RED |
| Focused GREEN | All new contract tests pass | 3 passed in 0.10s | PASS |
| Affected regression | All affected tests pass | 299 passed in 10.61s | PASS |
| Corrected primary contract | Exact SHA256 | `0aec93612e630e584af243839f305df828df795f1e438f73a1210256974585d4` | PASS |
| Pinned Git HEAD | Exact SHA | `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1` | PASS |
| Existing ISSUE-051 initialization RED / GREEN | Failure before file publication / exact transactional init | `4 failed, 1 passed` / `5 passed in 0.08s` | PASS |

Evidence logs:

- `/tmp/step4_phase6_high_blockers_red_20260715.log`
- `/tmp/step4_phase6_high_blockers_green_20260715.log`
- `/tmp/step4_phase6_high_blockers_focused_20260715.log`

The initial review's single MEDIUM item was the checkpoint exact-schema/atomic-initialization boundary already resolved as ISSUE-051. Current code validates missing/extra fields before target creation, builds the schema/header transactionally in memory, and creates the target path only after complete initialization succeeds. No compatibility fallback or repair-on-load path exists.

### 5.2 Corrected-v2 throughput figure validation

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Final mode-run identities | 64 | 64/64 | PASS |
| Final rank-one rows | Preserve available evidence | 50 | PASS |
| Paired / missing points | Explicit, no imputation | 23 / 9 | PASS |
| Ratio recomputation | 46 exact stored ratios | 46/46; max absolute delta `0.0` | PASS |
| GB300 maximum rank-one TTFT | `< 5000 ms` | 4176.999 ms; margin 823.001 ms | PASS |
| H200 maximum rank-one TTFT | `< 5000 ms` | 3431.264 ms; margin 1568.736 ms | PASS |
| H100 maximum rank-one TTFT | `< 5000 ms` | 3718.155 ms; margin 1281.845 ms | PASS |
| H800 maximum rank-one TTFT | `< 5000 ms` | 3832.023 ms; margin 1167.977 ms | PASS |
| Annotation regression first run | All tests pass | 1 failed, 176 passed | FAIL, root-caused |
| Annotation corrected rerun | All tests pass | 177 passed in 3.49s | PASS |

The annotation failure was a test-only adjacent-pair expression: strict zip combined lengths `N` and `N-1`. `itertools.pairwise()` now represents the intended invariant. Scientific artifacts were not modified to satisfy the test.

### 5.3 Completed primary-v2 shard audit

| System | Durable mode runs | Normalized / ranked | Contract / HEAD | Terminal counts | Result |
|---|---:|---:|---|---|---|
| H200 SXM | 120/120 | 162 / 162 | exact / exact | 162 success, 135 memory, 63 SLA | PASS |
| H100 SXM | 120/120 | 98 / 98 | exact / exact | 98 success, 215 memory, 47 SLA | PASS |
| H800 SXM simulated SOL | 120/120 | 88 / 88 | exact / exact | 88 success, 215 memory, 57 SLA | PASS |

Completed-shard audit log: `/tmp/step4_primary_v2_completed_shards_audit_20260715.log`; SHA256=`725b9a6c8d23eb1f9b176e9babab97a7ed9e76f7b602c74162819f065a23e960`; exit=`0`.

### 5.4 Final independent native review

| Check | Actual | Result |
|---|---:|---|
| Verdict | `APPROVE` | PASS |
| CRITICAL / HIGH / MEDIUM | 0 / 0 / 0 | PASS |
| Reviewer-focused pytest | 114 passed in 14.22s | PASS |
| Focused Ruff | All checks passed | PASS |
| `git diff --check` | Exit 0 | PASS |

The primary GB300 `120`-run shard was still executing when this section was written. Four-system merge, strict resume, final full-unit regression, final static gates, staging audit, commit/push, and `step-design` integration remain separate required gates and are not claimed complete here.

### 5.5 GB300 completion and corrected four-system merged-v2 validation

Environment: `aic-step-design`, Python `3.11.15`, `PYTHONPATH=src:.`, `MPLBACKEND=Agg`.

Commands:

```bash
MPLBACKEND=Agg PYTHONPATH=src:. /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python /tmp/step4_merge_primary_v2_20260715.py
MPLBACKEND=Agg PYTHONPATH=src:. /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python /tmp/step4_audit_merged_primary_v2_20260715.py
PYTHONPATH=src:. /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python /tmp/step4_rebind_comparison_summary_v2_20260715.py
```

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| GB300 durable mode runs | 120 | 120 | PASS |
| GB300 contract / HEAD | Exact corrected contract / pinned HEAD | Exact / exact | PASS |
| Four shard rows | 4 x 120 | 480 | PASS |
| Merged normalized / ranked | Equal | 545 / 545 | PASS |
| Strict resume loader / executor calls | 0 / 0 | 0 / 0 | PASS |
| Strict resume unchanged hashes | 5/5 | 5/5 | PASS |
| Paired / unpaired comparisons | Preserve both | 89 / 28 | PASS |
| Success / memory / SLA terminals | Typed only | 545 / 655 / 240 | PASS |
| SOL / explicit no-op evidence | No profiling fallback | 20,280 / 6 | PASS |
| Inner pre / post / shared AR evidence | Present | 2,935 / 4,338 / 1,403 | PASS |
| Outer `generation_moe_overlap` evidence | 0 | 0 | PASS |
| Common rows | 21 | 21 | PASS |
| AA / AB / BA / BB | 289 / 68 / 68 / 16 | 289 / 68 / 68 / 16 | PASS |
| Comparison-table identity | No numeric drift | Unchanged | PASS |

Merged artifact inventory:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `mode_runs.sqlite3` | 21,286,912 | `2968fbe3253ee5bd8c1b17749207e61e2f61cc00210e9f5a3ad964851eee42ea` |
| `results.json` | 41,636,981 | `d18e05806968518ef5f46206f77a29eda76c72e140d8c7502b2c4305790f0ed8` |
| `ranked_rows.csv` | 10,392,173 | `8a4756c1f11a6946c9a45dd0f2cc6a3cef39894f239bc8a18e64a3233db2e897` |
| `model_comparisons.csv` | 108,819 | `ca7fd0a7c3d3630fcda8974d42643ac2ab1f71c4222675ebebdf73cac2e1eb65` |
| `report.md` | 95,030 | `46c1681b58bd91eb374dc5358ece8026a3d06ab1b0542fdd924c9613bf321528` |

Logs: `/tmp/step4_merge_primary_v2_20260715.log`, `/tmp/step4_audit_merged_primary_v2_20260715.log`, and `/tmp/step4_rebind_comparison_summary_v2_20260715.log`. All commands exited `0`. Fresh full unit and final static/staging gates remain pending and are reported separately after execution.

## 6. Formatter-Consistent Corrected-v3 and Minimum-Cost Delivery Validation

### 6.1 Authoritative primary-v3 execution and merge

Environment: `aic-step-design`, Python `3.11.15`, `PYTHONPATH=src:.`, `MPLBACKEND=Agg`.

Commands:

```bash
MPLBACKEND=Agg PYTHONPATH=src:. "$PY" /tmp/step4_merge_primary_v3_20260715.py
MPLBACKEND=Agg PYTHONPATH=src:. "$PY" /tmp/step4_audit_merged_primary_v3_20260715.py
PYTHONPATH=src:. "$PY" /tmp/step4_rebind_comparison_summary_v3_20260715.py
```

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| GB300 durable mode runs | 120 | 120 | PASS |
| GB300 normalized / ranked | Equal | 197 / 197 | PASS |
| GB300 paired / unpaired | Preserve both | 33 / 2 | PASS |
| GB300 success / memory / SLA | Typed only | 197 / 90 / 73 | PASS |
| GB300 v2/v3 scientific identity | Contract-only difference | Exact after execution-contract normalization | PASS |
| Four shard rows | 4 × 120 | 480 | PASS |
| Merged normalized / ranked | Equal | 545 / 545 | PASS |
| Strict resume loader / executor calls | 0 / 0 | 0 / 0 | PASS |
| Strict resume unchanged hashes | 5/5 | 5/5 | PASS |
| Paired / unpaired comparisons | Preserve both | 89 / 28 | PASS |
| Success / memory / SLA terminals | Typed only | 545 / 655 / 240 | PASS |
| SOL / explicit no-op evidence | No profiling fallback | 20,280 / 6 | PASS |
| Nested generation pre / post / shared AR | Exact | 2,935 / 4,338 / 1,403 | PASS |
| Outer `generation_moe_overlap` evidence | 0 | 0 | PASS |
| Common rows | 21 | 21 | PASS |
| AA / AB / BA / BB | 289 / 68 / 68 / 16 | 289 / 68 / 68 / 16 | PASS |
| v2/v3 comparison-table identity | No numeric drift | Unchanged | PASS |

Authoritative merged-v3 artifacts:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `mode_runs.sqlite3` | 21,286,912 | `f304f70d3742ed954272c49e157c139916a4fe364d0b87606ebacd70efdf3167` |
| `results.json` | 41,636,981 | `372b683274771b70f3a5a94caafe08ac0289268140c038194995a5fa72c24a3e` |
| `ranked_rows.csv` | 10,392,173 | `8a4756c1f11a6946c9a45dd0f2cc6a3cef39894f239bc8a18e64a3233db2e897` |
| `model_comparisons.csv` | 108,819 | `ca7fd0a7c3d3630fcda8974d42643ac2ab1f71c4222675ebebdf73cac2e1eb65` |
| `report.md` | 95,030 | `46c1681b58bd91eb374dc5358ece8026a3d06ab1b0542fdd924c9613bf321528` |
| `comparison_summary.md` | 25,221 | `93e558d422ae4afd4a9a195b2eb3cfa7cdba71614ba0c8d281855430596887c5` |

Evidence logs:

- `/tmp/step4_primary_v3_gb300_audit_20260715.log`: 1,347 bytes; SHA256 `4dd4763eedbad0b794cecd64f974cc8955bdc135cd10aecd76c4f24746c262df`.
- `/tmp/step4_merge_primary_v3_20260715.log`: 262,783 bytes; SHA256 `0d39bd36cb98feeea3ef145773431d32748b1ebd523c654dcc5cc6c97f582258`.
- `/tmp/step4_audit_merged_primary_v3_20260715.log`: 1,770 bytes; SHA256 `354dadfc2c4dee567f818cd48672a11cb8008a6d21874b9e1391020c5576c50c`.
- `/tmp/step4_rebind_comparison_summary_v3_20260715.log`: 280 bytes; SHA256 `f22e8296c00df26ebf4777bcdd69c97a67108f61230bd135114f68662acc795d`.

The initial GB300 disposable audit used the full-matrix hash and a guessed simulation label for a system shard. The initial merged disposable audit compared a complete 14-name evidence Counter with a three-name nested subset. Both failures were validator-population defects. The corrected full audits passed without source, test, checkpoint, result, or figure modification.

### 6.2 Fresh full-unit evidence reused under the user-approved minimum-cost policy

Command:

```bash
PYTHONPATH=src:. "$PY" -m pytest -m unit
```

| Metric | Expected | Actual | Result |
|---|---:|---:|---|
| Collected | Informational | 3,127 | PASS |
| Selected | Informational | 2,008 | PASS |
| Passed | All selected non-skipped | 2,002 | PASS |
| Skipped | Explicit | 12 | PASS |
| Deselected | Marker filter | 1,119 | PASS |
| Warnings | Informational | 4 | PASS |
| Failed | 0 | 0 | PASS |
| Exit code | 0 | 0 | PASS |
| Elapsed | Informational | 765.21 s | PASS |

Log: `/tmp/step4_final_v3_full_unit_20260715.log`; 289,786 bytes; SHA256 `251decb0aece1f0ee5f5c7eacae060c15d95a16a502db369716669e64849f41d`.

No production or test source changed after this run. The user explicitly requested minimum-cost branch completion, so the full unit suite was not repeated.

### 6.3 Focused unit, integration, and static validation

Commands:

```bash
mkdir -p tests/.tmp
TMPDIR="$PWD/tests/.tmp" PYTHONPATH=src:. MPLBACKEND=Agg "$PY" -m pytest -q \
  tests/unit/performance/test_dsv4pro_vs_step4_throughput.py \
  tests/unit/performance/test_step4_comparison_checkpoint.py \
  tests/unit/performance/test_step4_comparison_cli.py \
  tests/unit/performance/test_step4_roofline_matrix.py \
  tests/integration/test_step4_prefill_ranking.py

ruff check --exclude tests/.tmp .
ruff format --check --exclude tests/.tmp .
git diff --check
```

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Focused + integration tests | 178 | 178 passed in 8.27 s | PASS |
| Ruff check | 0 errors | All checks passed | PASS |
| Ruff format | All repository files formatted | 429 files already formatted | PASS |
| `git diff --check` | Exit 0 | Exit 0 | PASS |

The first attempt to write a focused-test log under `/tmp` failed because `/tmp` had zero free inodes despite about 14 GiB free capacity. Pytest itself still passed `178/178`. The reproducible rerun redirected `TMPDIR` and the log to `tests/.tmp`; its log is 4,604 bytes with SHA256 `283ff504d9c9181d7e7681f48a54e3399b14cad99702d16dd292a463ab89b8c9`.

The first format check then included pytest fixture copies inside `tests/.tmp` and reported four temporary files. The corrected repository-scope command excluded that temporary tree and reran the complete static sequence. First static log: 766 bytes, SHA256 `5e53bd820ae1e0480c31990ba7b6f3b32bdd93c030591c62395b1961f113a7e3`. Corrected log: 324 bytes, SHA256 `58bd797a24c73dbe443c9202420127779cf55ed80cffff7dada40edf6237337f`.

### 6.4 Exact inventories and independent pre-commit review

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Merged-v3 hashes | 5/5 | 5/5 | PASS |
| Primary hashed inventory | 49 unique | 49/49 unique | PASS |
| Primary committed deliverables | 49 inventory + summary self | 50 | PASS |
| Figure hashed inventory | 41 unique | 41/41 unique | PASS |
| Authoritative v1/v2 result paths | 0 | 0 | PASS |
| Documents / trailing whitespace | 19 / 0 | 19 / 0 | PASS |
| Previous stale-contract blocker resolved | YES | YES | PASS |
| Independent verdict | Non-BLOCK | `APPROVE` | PASS |
| Independent CRITICAL / HIGH / MEDIUM | 0 / 0 / 0 | 0 / 0 / 0 | PASS |
| Independent required actions | None | None | PASS |

The first combined validator reused the primary `Path / Bytes / SHA256` regex for the figure archive's declared `SHA256 / Exact path` table and therefore found zero figure rows. The corrected full rerun used the two declared schemas and passed every preceding and subsequent assertion. Corrected log: `tests/.tmp/step4_final_v3_inventory_artifact_validation_corrected_20260715.log`.

Independent review artifact: `.omx/artifacts/ask-claude-step4-final-v3-precommit-review-20260715.md`; 9,835 bytes; SHA256 `f1beae4f04a09331f8ca6692172bf7ea3be1f96ac2e17440c02072c71e82770c`. The reviewer retained two WATCH-only observations—an existing per-config sweep exception boundary and local historical v1/v2 directories—and explicitly permitted commit/push/`ff-only` integration after exact staging membership audit.

### 6.5 Exact staging and cached whitespace validation

Commands:

```bash
# Every path was supplied individually from the two summary inventories.
git add -- <exact-file>
git add -f -- <exact-ignored-result-file>

PYTHONPATH=src:. "$PY" tests/.tmp/step4_audit_staged_line_endings_20260715.py
git -c core.whitespace=cr-at-eol diff --cached --check
```

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Combined allowlist | 92 unique paths | 92 | PASS |
| Staged membership | Exact allowlist | 92/92 | PASS |
| Missing / unexpected / forbidden paths | 0 / 0 / 0 | 0 / 0 / 0 | PASS |
| Files over 100 MiB | 0 | 0 | PASS |
| Ignored results added by exact path | 28 | 28 | PASS |
| Staged text files | Informational | 83 | PASS |
| Consistent CRLF text files | Informational | 10 CSV files | PASS |
| Mixed text line endings | 0 | 0 | PASS |
| Real trailing space/tab lines | 0 | 0 | PASS |
| Cached diff check with `cr-at-eol` | Exit 0 | Exit 0 | PASS |

The first plain `git diff --cached --check` classified the carriage return in every generated CSV `CRLF` record as trailing whitespace. This was a Git policy mismatch, not corrupted CSV content. The first disposable byte audit then scanned SQLite/PNG binaries as text and produced false positives; the corrected audit selected declared text suffixes only. All ten CSV files use consistent `CRLF`, no text file mixes endings, and no line has an actual trailing space or tab. Authoritative artifacts were not normalized or rewritten. Corrected audit log: `tests/.tmp/step4_final_staged_line_endings_corrected_20260715.log`.
