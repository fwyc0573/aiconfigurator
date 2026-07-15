## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Opened the attention-detail provenance issue and recorded its approved boundary. |
| 2026-07-16 | Recorded and resolved the invalid repository `.venv` baseline environment. |
| 2026-07-16 | Recorded the OMX Team clean-workspace gate and its commit-based resolution. |

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

- **Status:** Resolution in progress; commit the task source of truth, then retry unchanged.
- **Symptom:** `omx team 3:architect` exited 1 before worker creation with `leader_workspace_dirty_for_worktrees`.
- **Root cause:** The newly created task documents were intentionally untracked, and Team mode refuses to launch worktree-capable workers from a dirty leader workspace.
- **Impact:** No worker pane or task was created; production and test source remain untouched.
- **Resolution:** Validate and commit the eight task documents on `step4-pro`. Do not stash them because workers require these files as their canonical context. Retry the identical team launch and verify panes, ACKs, and runtime state.
