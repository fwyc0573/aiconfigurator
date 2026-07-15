## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Recorded authoritative inputs, branch constraints, environment notes, and safety boundaries. |
| 2026-07-16 | Corrected the test environment to the verified `aic-step-design` conda environment after `.venv` reproduction failed. |
| 2026-07-16 | Added Team shutdown state, report provenance, SOL_FULL/KV/parser/AFD execution reminders, and the preserved test-temp stash. |

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
- Bind this worktree explicitly with `PYTHONPATH="$PWD/src:$PWD"`, `MPLBACKEND=Agg`, and task-local `TMPDIR="$PWD/tests/.tmp"`.
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
