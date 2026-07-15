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
- Until complete attention detail is supplied, use the same explicitly temporary MLA roofline treatment as Step4 while preserving `20 Full / 60 SWA` labels.

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
