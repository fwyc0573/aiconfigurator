## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Recorded authoritative inputs, branch constraints, environment notes, and safety boundaries. |
| 2026-07-16 | Corrected the test environment to the verified `aic-step-design` conda environment after `.venv` reproduction failed. |

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
