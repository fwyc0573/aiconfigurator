## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Opened the attention-detail provenance issue and recorded its approved boundary. |
| 2026-07-16 | Recorded and resolved the invalid repository `.venv` baseline environment. |
| 2026-07-16 | Recorded the OMX Team clean-workspace gate and its commit-based resolution. |
| 2026-07-16 | Reconciled Team decomposition, SOL_FULL, KV-cache, parser-validation, and AFD findings with root-cause resolutions. |
| 2026-07-16 | Recorded StepCode Claude approval of the bounded SOL_FULL, parser, and AFD resolutions. |

# Issues

## ISSUE-001: CSV attention totals do not close from available geometry

- **Status:** Open as a documented human-update item; not an implementation blocker under the original request.
- **Symptom:** Standard GQA/SWA projection counts do not match the CSV `param per full attn` and `param per nonfull attn avg` values.
- **Root cause:** The exported Main CSV contains summary counts and query-head hints but omits the detailed projection recipe that generated the weighted attention totals.
- **Impact:** A supposedly faithful GQA/SWA op graph would undercount attention weights by 23.530%–26.029% and would not be evidence-backed.
- **Resolution for this task:** Preserve `20 Full / 60 SWA` structure labels and use the same explicitly temporary MLA SOL/roofline treatment as Step4. Do not invent missing ops or apply a scaling factor. Record all borrowed values and approximation limits for human replacement.
- **Future resolution:** Replace the temporary attention graph only after a complete Step4-Pro-V1 Attention Detail source is supplied and independently reconciled with the CSV totals.

## ISSUE-002: Repository `.venv` cannot run the required test toolchain

- **Status:** Resolved using the existing verified environment; no dependency or source modification required.
- **Symptom:** `.venv/bin/python` reported Python 3.13.13 and could not import pytest; `.venv/bin/ruff` did not exist.
- **Root cause:** The repository `.venv` is present but unsynchronized, exactly matching the existing environment-handbook caveat.
- **Impact:** The first baseline command exited 1 before test collection; it provided no evidence about source correctness.
- **Resolution:** Use `/home/i-fengyicheng/miniconda3/envs/aic-step-design`, bind the current worktree with `PYTHONPATH="$PWD/src:$PWD"`, select `MPLBACKEND=Agg`, and redirect `TMPDIR` to `tests/.tmp`. The corrected baseline passed 90/90 tests.

## ISSUE-003: OMX Team requires a clean leader workspace

- **Status:** Resolved.
- **Symptom:** `omx team 3:architect` exited 1 before worker creation with `leader_workspace_dirty_for_worktrees`.
- **Root cause:** The newly created task documents were intentionally untracked, and Team mode refuses to launch worktree-capable workers from a dirty leader workspace.
- **Impact:** No worker pane or task was created; production and test source remain untouched.
- **Resolution:** Validated and committed the task documents, launched Team `step4-pro-v1-pre-impl-0ddff5cf`, completed all six tasks, saved both architect reports, and shut the Team down only after `pending=0`, `in_progress=0`, and `failed=0`.

## ISSUE-004: Legacy Team decomposition assigned malformed fragments

- **Status:** Resolved without changing audit ownership.
- **Symptom:** The initial long Team prompt produced task fragments whose owners and report responsibilities did not match the intended three-lane audit.
- **Root cause:** Legacy natural-language decomposition split the prompt at semantic boundaries that did not preserve the requested architecture/roofline/integration ownership mapping.
- **Impact:** Worker-3 initially had no claimable task, while worker-1 owned two fragments that duplicated worker-2/3 report responsibilities.
- **Resolution:** Completed worker-2's roofline scope as task 3, created owner-corrected integration task 6 for worker-3, and converted tasks 4/5 into lifecycle-only superseded fragments. Worker-1 verified both leader-saved reports and closed tasks 4/5 with hashes; the Team finished `6/6` tasks with no failures.

## ISSUE-005: SOL_FULL validation and operation-query contracts are incompatible

- **Status:** Open as an explicitly bounded platform limitation; not fixed in the minimum Step4-Pro-V1 change.
- **Symptom:** `Task` accepts `DatabaseMode.SOL_FULL`, but complete graph execution fails at the first wrapper that calls `float()` on the returned tuple.
- **Root cause:** Direct SOL_FULL database methods return `(sol_time, sol_math, sol_mem)`, while operation wrappers require a scalar-like `PerformanceResult` with `.energy`.
- **Impact:** End-to-end SOL_FULL Task execution cannot be claimed for Step4 or Step4-Pro-V1. A local model-specific cast would hide a shared contract defect.
- **Resolution for this task:** Execute the complete graph in SOL. Audit SOL_FULL through direct `PerfDatabase` calls and assert `selected == max(math, memory)`. Any shared operation-query contract refactor requires a separately approved cross-cutting change.

## ISSUE-006: Temporary MLA KV-cache estimate does not match the CSV

- **Status:** Open as a documented human-update item; scaling is prohibited.
- **Symptom:** Borrowed Step4 latent geometry yields `46,080` elements/token and `48.31838208 GB` at `1,048,576` FP8 tokens, while the CSV reports `10.7 GB`.
- **Root cause:** The current Step4 formula models every layer as full latent MLA and scales linearly with sequence length; it has no sequence-aware SWA/window topology for Step4-Pro-V1.
- **Impact:** The temporary estimate exceeds the CSV by `37.61838208 GB`, a ratio of `4.515736642991x` (`351.5736642991%` over target).
- **Resolution for this task:** Publish both values and the gap. Do not apply a scaling factor. A faithful fix requires authoritative KV topology plus sequence/window-aware cache modeling.

## ISSUE-007: Step4 parsing silently substitutes or accepts malformed geometry

- **Status:** Open for RED/GREEN root-cause correction in this task.
- **Symptom:** Missing/zero `moe_intermediate_size` can inherit dense `intermediate_size`; missing/zero/bool/float `num_experts_per_tok` and boolean core dimensions can pass weak validation.
- **Root cause:** Step4-specific parsing reuses permissive generic numeric extraction and fallback behavior that was not exposed while only one cached Step4 configuration existed.
- **Impact:** Malformed Step4-Pro-V1 configuration can build a plausible but incorrect operation graph instead of failing at the configuration boundary.
- **Planned resolution:** Add a RED validation matrix, require explicit non-boolean positive integers, enforce topology/parallel invariants, and remove routed-MoE substitution. Preserve valid original Step4 behavior.

## ISSUE-008: AFD cannot classify Step4 dense SwiGLU operations

- **Status:** Confirmed outside the minimum Step4-Pro-V1 scope by independent StepCode Claude review.
- **Symptom:** Existing Step4 context and generation AFD partitioning fail on `context_dense_swiglu` and `generation_dense_swiglu`.
- **Root cause:** The AFD FFN classifier recognizes several activation markers but not `swiglu`, while Step4 emits a dedicated dense SwiGLU operation.
- **Impact:** A generic claim of every CLI estimate mode would be false even before Step4-Pro-V1 is added.
- **Resolution for this task:** Cover aggregate/disaggregate SOL and state the AFD boundary explicitly. If AFD becomes required, open a separate RED fix for classifier behavior and both context/generation regressions.
