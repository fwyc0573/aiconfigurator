## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Captured the original Step4-Pro-V1 support request and its explicit unresolved-parameter policy. |

# Requirements: Step4-Pro-V1 AIC Support

## Raw User Intent

1. [Original Request] Treat this as a new task whose core objective is to add Step4-Pro-V1 support to AIC.
2. [Original Request] Follow the implementation methodology and engineering practice recorded in `task_memory/task_2026-07-10_step4_predefined_ops_plan` for the earlier Step4 support.
3. [Original Request] Define the Step4-Pro-V1 architecture and operation composition accurately from `permormancebenchmark/architecture_calculator_v1 - Main - latest.csv`.
4. [Original Request] Ensure every Step4-Pro-V1 operation has a correct, reasonable, evidence-backed roofline calculation process.
5. [Original Request] For critical parameter values that cannot be determined from the authoritative CSV, follow the corresponding Step4 treatment and document each borrowed or approximate value clearly for later human update or modification.
6. [Original Request] Create a new worktree and branch named `step4-pro`, and make all code-module changes on that branch.
7. [Original Request] Use parallel Team mode and subagents fully to accelerate delivery.

## Captured Decisions Already Present in the Original Request

1. [Original Request] The CSV is authoritative for Step4-Pro-V1 structure and values that it explicitly provides.
2. [Original Request] Missing or non-closing critical values inherit the existing Step4 handling rather than being silently invented.
3. [Original Request] The inherited Step4 treatment must be explicitly marked in documentation so the user can update it manually after task completion.
4. [Original Request] Because the CSV attention totals do not close under a standard borrowed GQA/SWA geometry, Step4-Pro-V1 will retain its CSV-defined `20 Full / 60 SWA` audit labels while using the existing Step4 temporary MLA roofline treatment until the user supplies complete attention details.

## Pending Questions

None. The original request explicitly resolves the missing-parameter policy by requiring Step4 treatment plus visible documentation.
