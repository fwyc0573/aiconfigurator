## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Initialized the reusable-lessons register. |
| 2026-07-16 | Added independently verified lessons for formula boundaries, provenance, CLI evidence, and multiprocessing test environments. |

# Lessons

## 1. Keep tuple-level roofline audit separate from scalar graph execution

When a database mode returns a diagnostic tuple but operation wrappers consume a scalar `PerformanceResult`, forcing the tuple through the wrapper is not additional coverage; it violates the interface. Execute the complete graph in the supported scalar `SOL` mode, audit `(selected, math, memory)` directly at the database boundary, and preserve a regression for the known incompatibility until a separately approved shared-contract refactor is available.

## 2. Architecture summaries require provenance partitions

A summary CSV can close layer counts and FFN/MoE totals while still omitting the projection recipe needed to close attention or KV-cache arithmetic. Partition values into source-provided, mathematically derived, and explicitly borrowed sets. If the borrowed geometry does not close, publish the expected value, actual value, gap, and impact; never hide the discrepancy behind a scaling factor.

## 3. Artifact generation is not deployment-feasibility evidence

A CLI can render a syntactically complete deployment bundle while simultaneously reporting `fit=False`. Treat artifact count and identity resolution as smoke evidence only. Parameter sizing, memory fit, and performance support require their own validated paths and must not be inferred from successful file generation.

## 4. Temporary-directory capacity and pathname length are independent constraints

Free bytes and inodes do not guarantee that Python multiprocessing can bind its AF_UNIX listener. In long worktrees, measure the chosen `TMPDIR` and account for the `pymp-*/listener-*` suffix. Use a short, preflighted directory and prove the diagnosis with a controlled one-variable test before considering source changes.

## 5. Static checks must target the delivery surface without erasing evidence

Repository-recursive tools can include untracked pytest fixture copies. Preserve the original diagnostic, inspect every reported path, and then verify all Git-tracked files plus an explicit temporary-path exclusion. Do not delete evidence merely to obtain a visually clean command result.
