## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created the raw requirements record for the Step4-Pro-latest B-card task. |
| 2026-08-13 | Recorded user choice A: the pinned local vLLM checkout and requirements shape are authoritative. |
| 2026-08-13 | Recorded user choice A for the canonical AIC model path. |
| 2026-08-13 | Recorded exhaustive filename-search evidence that the referenced manifest and companion task file are unavailable. |
| 2026-08-13 | Resolved the missing-manifest contract: reconstruct it from the authoritative requirements and pinned source with a new hash. |
| 2026-08-13 | Opened the operation-boundary decision after the pinned-source fidelity-gap audit. |
| 2026-08-13 | Recorded the user's clarification request about whether the Full-MFA profiling graph comes directly from Step4-Pro vLLM. |
| 2026-08-13 | Resolved the operation-boundary strategy with user confirmation of option A. |
| 2026-08-13 | Opened the safe branch/checkpoint decision for the existing dirty linked worktree. |
| 2026-08-13 | Resolved the branch/checkpoint strategy with user confirmation of option A. |

# Requirements: Step4-Pro-Latest B-Card Ops, Collection, and Simulation

## Raw User Intent

1. [Original Request] Act as a performance modeling and simulation task agent for `step4-pro-latest`.
2. [Original Request] Based on the actual implementation in the latest B-card vLLM image, define all required operations in AIConfigurator.
3. [Original Request] Collect performance datasets for those operations on B-card devices.
4. [Original Request] Validate `step4-pro-latest` correctness in AIConfigurator and execute the prefill and decode simulation tasks required by the requirements document.
5. [Original Request] Review the historical `v3/v4` support and profiling tasks for reusable methodology, but do not directly reuse old H800 data or old-version assumptions.
6. [Original Request] Treat the requirements document and the actual latest vLLM implementation as the highest-priority sources.
7. [Original Request] Record the source of every operation definition, including code files, functions/classes, and call paths.
8. [Original Request] Ensure the collected operation set exactly matches the operation set defined for `step4-pro-latest`, covering both prefill and decode.
9. [Original Request] Record B-card collection details, including hardware/interface/precision/performance-counter adaptations and any problems with their resolutions.
10. [Original Request] Run correctness tests before the formal prefill and decode simulation experiments.
11. [Original Request] Produce an operation definition and provenance list, the B-card performance dataset, a correctness and simulation results report, and a difference/problem report against H800 and `v3/v4`.
12. [Original Request] Do not use H800 historical data as the B-card result.
13. [Original Request] Do not make unrecorded assumptions; ambiguous, missing, or contradictory requirements must be confirmed through the `grill-me` mechanism before execution.
14. [Original Request] Follow the `stepfun-env-handbook` instructions for Docker and GPU use.
15. [Original Request] Create a new task directory under `task_memory/` so this work is distinguishable from similar work by other agents.
16. [Original Request] Use the user-provided vLLM checkout at `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro` as an available source checkout for the Step4-Pro implementation.

## Pending Q&A Follow-ups

### Q1 — Authoritative latest source/runtime contract

**Question:** Which source/runtime identity should define `step4-pro-latest` for this task?

**Recommended choice:** Use the user-provided checkout at `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro`, detached at commit `607d1641ee3fec43653fca510d717725828890c2`, together with the requirements document's 78-layer synthetic shape and experiment matrix. This keeps the implementation, requirements, and auditable local source aligned.

**Alternative:** Use the later branch head `9bfd9a610ea4f2890010702ee7a207cf25edf8de` and image manifest `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`; this may include runtime changes beyond the pinned checkout and requires re-validating the shape contract.

**Status:** **Resolved — user selected A on 2026-08-13.**

**Decision:** Use the user-provided checkout at `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro`, detached at commit `607d1641ee3fec43653fca510d717725828890c2`, together with the requirements document's 78-layer synthetic shape, experiment matrix, and MTP1 requirements. The later branch head is reference-only.

**Runtime identity derived from the selected checkout:** Both pinned smoke scripts select image `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`. Its inspected manifest digest is `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`. This image is part of choice A because it is declared by the pinned source, not because it appears in the later branch.

**Execution consequence:** Continue with source-to-AIC mapping and implementation against the pinned checkout and pinned-script image. Do not silently substitute the later branch head.

### Q2 — AIC canonical model identity

**Question:** What exact AIC model path should identify the new 78-layer model?

**Recommended choice:** `stepfun-ai/Step4-Pro-Latest`. It matches the requested `step4-pro-latest` name and keeps the existing, materially different `stepfun-ai/Step4-Pro-V4` unchanged.

**Alternative:** A different user-specified canonical path. It must be distinct from all existing Step4 entries.

**Status:** **Resolved — user selected A on 2026-08-13.**

**Decision:** Register the new model as `stepfun-ai/Step4-Pro-Latest`. Preserve the existing `stepfun-ai/Step4-Pro-V4` entry unchanged.

### Q3 — Missing shape manifest contract

**Question:** The requirements reference `step4pro_v4_shape_manifest.json` and its SHA256, but the file is absent. How should this task handle it?

**Recommended choice:** Generate a new canonical manifest from the requirements text and pinned source, store its newly computed SHA256, and explicitly state that it is a reconstruction rather than the unavailable original file.

**Alternative:** Require exact reproduction of the referenced original manifest and hash. This requires the task owner to provide the missing original file before model implementation and collection proceed.

**Verified availability evidence:** A recursive ordinary-file search under `/data/ycfeng/stepfun-performance-optimization` found neither referenced filename. The evidence record is `/data/ycfeng/tmp/step4_manifest_search.txt`. The original stated SHA256 therefore cannot be reproduced from available files.

**Status:** **Resolved — user selected A on 2026-08-13.**

**Decision:** Generate a new canonical manifest from the requirements text and
the pinned source. Store its newly computed SHA256 and label it explicitly as a
reconstruction. Do not claim or reuse the unavailable original SHA256.

### Q4 — AIC operation-boundary strategy

**Question:** How should AIC represent runtime behaviors that the current
Step4 schema and consumers cannot express faithfully?

**Recommended choice:** Make the smallest explicit extensions: add a
heterogeneous `Step4ProLatestConfig`; add a grouped-GEMM identity for Full MFA
`wo_a`; add vLLM DeepEP HT dispatch/combine identities and Collector support;
add shared-KV/page-layout metadata; represent the router as FP32; and construct
MTP1 explicitly. Reuse existing `GEMM`, `ElementWise`, Attention, Dense, and
MoE operations everywhere else.

**Alternative 1:** Force all behavior through current operation identities.
This is smaller code-wise but knowingly collapses grouped `wo_a`, DeepEP,
shared K/V layout, FP32 routing, and MTP1, so it cannot satisfy the
actual-implementation fidelity requirement.

**Alternative 2:** Generalize/refactor the operation framework for all models
before adding Step4-Pro-Latest. This can produce broader abstractions but is a
larger, higher-risk scope than this task requires.

**Status:** **Resolved — user confirmed option A on 2026-08-13.**

**Decision:** Use the minimum fidelity-preserving extensions:

- heterogeneous `Step4ProLatestConfig`;
- grouped-GEMM identity for Full-MFA `wo_a`;
- vLLM DeepEP HT dispatch/combine identities and Collector support;
- shared-KV/page-layout metadata and memory accounting;
- FP32 router representation;
- explicit MTP1 operation graph;
- existing AIC operations everywhere they already preserve pinned behavior.

### Q4 Clarification — Full-MFA profiling provenance

1. [Original Request] Confirm whether the proposed Full-MFA profiling
   operation graph is obtained directly from the Step4-Pro vLLM
   implementation.
2. [Original Request] Do not treat a manually invented or generic substitute
   graph as equivalent to the actual Step4-Pro vLLM execution path.

**Clarification:** The current inventory is a logical AIC graph extracted from
the pinned vLLM source call path; it is not yet a B300 profiler-exported graph.
The formal dataset must add a runtime-trace gate in the pinned image and execute
the exact vLLM/provider paths for provider-sensitive operations before any row
is accepted.

### Q5 — Safe branch/checkpoint strategy

**Question:** How should implementation proceed without mixing new work into
the 37 modified tracked files and existing untracked historical artifacts?

**Recommended choice:** Keep the already-isolated linked worktree, create
branch `task/step4-pro-latest-b300`, then make a baseline checkpoint commit
containing existing tracked Step4 changes plus relevant untracked source,
config, tests, and task documentation. Exclude generated profiler outputs,
caches, large result artifacts, and the separate pinned vLLM checkout. Do not
stash, reset, remove, or move anything.

**Alternative 1:** Continue directly on branch `step4-pro` without a baseline
checkpoint. This mixes old and new work and makes review and rollback unsafe.

**Alternative 2:** Pause until the task owner manually cleans or commits the
current worktree.

**Status:** **Resolved — user selected option A on 2026-08-13.**

**Decision:** Keep the current linked worktree, use branch
`task/step4-pro-latest-b300`, and create a baseline checkpoint containing
relevant Step4 source/config/tests/task documentation. Exclude generated
outputs, caches, large result artifacts, H800 data, and the separate pinned
vLLM checkout. Do not stash, reset, remove, or move existing files.
