## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-16 | Added the final simulation-completeness review, normalized test record, commit, and push request. |
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
| 2026-08-13 | Recorded the user's decision to retain and take no action on the temporary security-review file. |
| 2026-08-13 | Opened the Step4-Pro-V1 authoritative attention-contract decision. |
| 2026-08-13 | Clarified that Step4-Pro-Latest must follow the pinned vLLM operation graph; the V1 decision is legacy-baseline-only. |
| 2026-08-13 | Resolved Q6: preserve the historical V1 contract and implement Latest strictly from pinned vLLM. |
| 2026-08-13 | Opened the obsolete-test removal and V1 formula-test retention decision. |
| 2026-08-13 | Approved obsolete-test deletion and root-cause repair; reaffirmed strict pinned-vLLM fidelity for all Latest work. |
| 2026-08-14 | Confirmed pinned-vLLM MTP1 boundary: Step4Pro main graph has no native MTP1 path; opened an explicit scope gate. |
| 2026-08-14 | Deferred MTP1 structure tests and simulation; authorized MTP-off Latest execution and parallel B300 smoke. |
| 2026-08-14 | Moved the pinned-vLLM B300 smoke/runtime-provider trace to an external session and limited the current session to AIC-side implementation, measurement, tests, execution, and simulation. |
| 2026-08-14 | Recorded the new external-session request to execute the standalone pinned-vLLM B300 smoke/runtime trace handoff to completion. |
| 2026-08-14 | Reduced the controller memory scope to 3 GiB and required low-memory I/O after a host OOM. |
| 2026-08-15 | Confirmed that random/dummy weights, rather than a real checkpoint, are the required B300 test path. |
| 2026-08-15 | Authorized the single Optimus JIT activation-quant overlay for B300 SM103. |
| 2026-08-15 | Authorized continued execution after evaluating the required `ep_gather` block fix and documenting its safety. |
| 2026-08-15 | Clarified that the approved runtime fixes are limited to the pinned B300 performance/provider test and do not establish model-quality or training equivalence. |
| 2026-08-15 | Explicitly authorized reopening the fourth B300 two-node live run after the three-attempt stop gate. |
| 2026-08-15 | Authorized aligning the launch with the full B300 RDMA contract and running a minimal NCCL preflight before another model launch. |
| 2026-08-15 | Supplied a brainctl binary and supplemental RJob documentation, and requested a retry using the documented legacy launcher path. |
| 2026-08-15 | Authorized deferring further DeepEP measurement after the current attempt and continuing all remaining operation families. |
| 2026-08-15 | Requested an inventory of DeepEP launch examples and cases in the pinned image and local vLLM checkout. |
| 2026-08-15 | Authorized applying direct-child commit `9bfd9a610e` and reopening the exact DeepEP/NVSHMEM probe without retaining the earlier pinned-script restriction. |
| 2026-08-15 | Reconfirmed that a failed DeepEP attempt must be recorded and skipped so remaining operation families continue. |
| 2026-08-16 | Authorized the bounded SWA QKV runtime annotation compatibility overlay and continuation. |
| 2026-08-16 | Authorized an explicitly temporary B300 NCCL alltoall substitute for DeepEP so MTP-off simulation can complete before later DeepEP remeasurement. |

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
17. [Original Request] After the current DeepEP attempt finishes, if DeepEP still fails, stop measuring that operation, record the missing result, and continue with the next operation.
18. [Original Request] Temporarily allow another operation to substitute for
    DeepEP in simulation; after the DeepEP environment is restored, remeasure
    the real DeepEP operation and replace the temporary result.

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

### Q5 Follow-up — Temporary security-review file

1. [Original Request] Retain
   `/data/ycfeng/tmp/aic_failure_domain1/codex_lane3_scope_baseline.md`.
2. [Original Request] Do not perform any additional sensitive-file,
   credential, or security remediation because the current environment is
   considered sufficiently secure by the task owner.

**Status:** **Resolved — user selected option B on 2026-08-13.**

**Decision:** Do not read, delete, rewrite, move, permission-change, quote, or
otherwise process the file. Do not initiate credential rotation or any other
security action.

### Q6 — Authoritative Step4-Pro-V1 attention contract

**Question:** Which attention contract should be restored as the baseline for
the existing `stepfun-ai/Step4-Pro-V1` model?

**Recommended choice:** Restore the previously approved V1 contract with
separate `FullAttentionConfig` and `NonFullAttentionConfig`: standard Full
Attention plus HCA, TP-sharded KV, and formula-only HCA behavior for
`SOL`/`SOL_FULL`. Keep the new shared-KV Full MFA + SWA design exclusive to
`stepfun-ai/Step4-Pro-Latest`.

**Alternative:** Treat the current unified `Step4MFAAttentionConfig`,
replicated-KV, and SWA-retention implementation as a deliberate V1
replacement, then rewrite the historical V1 tests and documentation around
that new meaning.

**Status:** **Resolved — user confirmed the recommended combined decision on
2026-08-13.**

**Decision:** Restore and preserve the historically approved
`stepfun-ai/Step4-Pro-V1` Full-Attention-plus-HCA contract. Do not use that
legacy graph for `stepfun-ai/Step4-Pro-Latest`.

### Q6 Clarification — Legacy V1 versus Step4-Pro-Latest scope

1. [Original Request] The task owner does not intend the legacy
   `stepfun-ai/Step4-Pro-V1` choice to determine the implementation of
   `stepfun-ai/Step4-Pro-Latest`.
2. [Original Request] AIC support for `stepfun-ai/Step4-Pro-Latest` must use
   the actual operation implementation in the pinned Step4-Pro vLLM source.
3. [Original Request] Explain which Q6 option corresponds to the pinned vLLM
   implementation before asking for a decision.

**Source clarification:** The pinned Step4-Pro vLLM attention implementation
uses a heterogeneous graph: shared-KV Full MFA/MQA on the configured full
layers and native sliding-window GQA/SWA on the remaining layers. This is
closest to the shared-MFA wording in Q6 alternative B, not the historical V1
Full-Attention-plus-HCA contract.

**Scope correction:** Q6 controls only how the pre-existing
`stepfun-ai/Step4-Pro-V1` baseline is repaired. It does not authorize either
contract for `stepfun-ai/Step4-Pro-Latest`.

**Recommended combined decision:** Preserve the historically approved V1
contract (Q6 option A) to avoid changing the meaning of the existing V1 model,
while implementing Step4-Pro-Latest independently from the pinned vLLM graph:
shared-KV Full MFA plus sliding-window GQA/SWA.

**Status:** **Resolved — user confirmed on 2026-08-13.**

**Decision:** Keep V1 on its historical contract and implement
`stepfun-ai/Step4-Pro-Latest` independently and strictly from the pinned vLLM
operation graph, including shared-KV Full MFA and sliding-window GQA/SWA.

### Q7 — Obsolete DSV4 tests and V1 formula tests

**Question:** May baseline repair remove the two mistakenly checkpointed,
historically withdrawn DSV4 runtime-spec test files while retaining and fixing
the four V1 formula-only HCA tests?

**Recommended choice:**

1. Delete these two complete obsolete files:
   - `tests/unit/sdk/database/test_factorized_attention_runtime_spec.py`
   - `tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py`
2. Retain all four `SOL`/`SOL_FULL` tests in
   `tests/unit/sdk/database/test_step4_pro_v1_roofline.py`.
3. Fix the source-ordering defect in `src/aiconfigurator/sdk/operations/dsv4.py`
   so formula-only HCA queries return before `load_data()` is called.

**Evidence:** The two DSV4 files were added only by checkpoint `4f2b0c31`, do
not exist in its parent, and test a runtime-spec migration explicitly withdrawn
by the historical task owner. The four V1 roofline tests instead enforce the
confirmed historical V1 formula-only HCA contract and expose a production
source-ordering defect.

**Status:** **Resolved — user approved deletion and repair on 2026-08-13.**

**Decision:** Delete the two complete obsolete files, retain the four V1
formula-only HCA tests, and repair the `dsv4.py` source-ordering defect.

### Q7 Follow-up — End-to-end Latest vLLM fidelity

1. [Original Request] Treat the latest Step4-Pro model implemented in the
   pinned vLLM checkout as the standard for all `Step4-Pro-Latest` work.
2. [Original Request] Define the Latest operation graph from the actual vLLM
   implementation rather than from legacy V1/V3/V4 assumptions.
3. [Original Request] Make all Latest tests, measurements, and validation
   target that same vLLM-derived graph.
4. [Original Request] Every measured operation must execute the operation
   implementation supplied by the pinned vLLM version; a generic or
   reimplemented substitute is not acceptable.

**Status:** **Confirmed by the user on 2026-08-13.**

**Execution consequence:** Source call-path identity, runtime provider identity,
Collector case identity, persisted dataset key, and AIC consumer identity must
be reconciled for every accepted Latest operation row. Any mismatch fails the
gate and the row cannot enter the B300 dataset.

### Q8 — Step4-Pro MTP1 implementation boundary

**Question:** The requirements document requires one Step4-Pro MTP1 layer, but
the pinned vLLM commit does not contain a `Step4ProMTP` implementation or a
Step4Pro MTP registry path. How should the MTP1 requirement be handled while
preserving the rule that every Latest operation must use a pinned-vLLM
implementation?

**Source facts already verified:**

- `Step4ProForCausalLM` constructs the Step4Pro trunk and language-model head,
  but no MTP predictor: `vllm/model_executor/models/step4pro.py:626-649`.
- The registered MTP implementation is `Step3p5MTP`, which constructs
  `Step3p5DecoderLayer`, not `Step4ProDecoderLayer`:
  `vllm/model_executor/models/registry.py:616`,
  `vllm/model_executor/models/step3p5_mtp.py:171-190,286`.
- The Step4 speculative configuration is converted to `Step3p5MTP`:
  `vllm/config/speculative.py:357-362`.
- The requirements explicitly say that the Step4Pro MTP1 construction path
  still needs to be completed:
  `task_memory/step4pro_v4_external_simulator_requirements.md:113-115`.

**Recommended choice A:** Continue implementing and validating all MTP-off
Latest operations strictly from pinned vLLM; mark MTP1 as **not proven and not
accepted** for the pinned runtime; pause only the MTP1 graph, measurement, and
simulation until the owner supplies or approves a concrete Step4Pro MTP
implementation source.

**Alternative B:** Authorize a new Step4Pro MTP1 implementation in AIC/vLLM
outside the pinned commit. This changes the source-of-truth rule and requires
an explicit implementation specification, weight-loading contract, runtime
image identity, and new acceptance tests before measurement.

**Status:** **Resolved — user deferred MTP1 work on 2026-08-14.**

**Decision:**

- [Original Request] Defer MTP1 structure-related tests and MTP1 simulation.
- [Original Request] Allow `stepfun-ai/Step4-Pro-Latest` to run AIC tests and
  validation temporarily without MTP1.
- [Original Request] Continue with the MTP-off Latest AIC operation-set
  definition, Collector implementation, complete measurement tests, and
  end-to-end prefill/decode simulation.
- [Original Request] In parallel, run the pinned-vLLM Step4-Pro smoke on B300
  according to the requirements document, using its specified container. The
  vLLM checkout may be mounted when the platform permits it.

MTP1 must remain explicitly deferred and must not be replaced by `Step3p5MTP`
or an invented AIC-only implementation.

### Q9 — External ownership of pinned-vLLM smoke/runtime trace

1. [Original Request] Write a standalone execution document for the
   `B300 pinned-vLLM smoke and runtime/provider trace` subtask.
2. [Original Request] Include the concrete requirements, environment,
   authoritative file locations, reusable tests/evidence, execution gates, and
   result expectations so another session can execute it accurately without
   repeating completed probes.
3. [Original Request] The task owner will assign that document to another
   session and later return its results.
4. [Original Request] The current session must stop executing the pinned-vLLM
   whole-model smoke/runtime trace and focus only on AIC-side implementation,
   operation measurement, testing, execution, and simulation.

**Status:** **Resolved by explicit owner direction on 2026-08-14.**

**Execution consequence:**

- The external handoff document is
  `pinned_vllm_b300_smoke_runtime_trace_execution.md`.
- Current-session AIC development and AIC operation collection may continue.
- Final provider/source sign-off will ingest the external session's report
  when the task owner supplies it; missing external evidence must remain
  visible rather than being inferred.

### Q10 — Execute the external pinned-vLLM B300 sub-task

1. [Original Request] This is a new sub-task.
2. [Original Request] Fully understand
   `pinned_vllm_b300_smoke_runtime_trace_execution.md`.
3. [Original Request] Complete the task defined by that document.

**Status:** Active in the external execution session on 2026-08-14.

### Q11 — Reduce host memory scope and avoid I/O-driven OOM

1. [Original Request] A host OOM just occurred and may have been caused by the
   current 5 GiB memory setting.
2. [Original Request] Reduce the controller memory setting to 3 GiB.
3. [Original Request] Make I/O operations avoid possible OOM as much as
   practical.

**Status:** Active and mandatory for all remaining external B300 execution.

### Q12 — Use random weights instead of requiring the qy1-pt checkpoint

1. [Original Request] Re-read
   `task_memory/step4pro_v4_external_simulator_requirements.md` and determine
   whether it already provides a solution to the unavailable model mount.
2. [Original Request] Model weights may use random initialization.
3. [Original Request] Combine the parent requirements and the standalone
   pinned-vLLM execution guide to complete the Step4-Pro B300 test.

**Status:** Resolved on 2026-08-15. The authoritative parent requirement
explicitly requires synthetic `config.json` plus vLLM `--load-format dummy`;
real checkpoint availability must not block the MTP-off performance and
provider tests.

### Q13 — Authorize the Optimus JIT activation-quant overlay

1. [Original Request] Authorize the single-point Optimus JIT quant overlay.
2. [Original Request] Record the authorized overlay in the task documents.
3. [Original Request] Continue the B300 pinned-vLLM Step4-Pro test.

**Status:** Resolved and authorized on 2026-08-15.

### Q14 — Evaluate and document the `ep_gather` correction

1. [Original Request] Re-evaluate whether the proposed fix is required.
2. [Original Request] Evaluate whether it changes the original design or
   training behavior.
3. [Original Request] If the evaluation passes, record all extra changes in
   the task documents for later review.
4. [Original Request] Continue execution.

**Status:** Evaluation passed on 2026-08-15 for the pinned B300
performance/provider test. The `ep_gather` change is an inference-only tiling
correction that preserves tensor values. The Optimus JIT quant overlay remains
a runtime implementation change and is not evidence of bitwise output,
generation-quality, or training-convergence equivalence.

### Q15 — Reopen the fourth two-node B300 live run

1. [Original Request] Confirm the proposed next action.
2. [Original Request] Continue execution after the three-attempt stop gate.

**Status:** Explicitly authorized and executed on 2026-08-15. The
`--headless --api-server-count 0` role split passed its startup gate, but NCCL
initialization failed before DeepEP HT execution because the worker environment
contained an empty `NCCL_IB_HCA`.

### Q16 — Authorize full-RDMA launch alignment and NCCL preflight

1. [Original Request] Confirm the proposed launch configuration adjustment.
2. [Original Request] Continue with a minimal two-node NCCL preflight before
   another complete model run.

**Status:** Explicitly authorized and executed on 2026-08-15. The corrected
16-rank NCCL preflight passed with eight injected bond HCAs and all-reduce
result `136.0`. The subsequent model run selected
`DeepEPHTAll2AllManager`, then failed inside `deep_ep.Buffer.runtime.sync`.
The remaining gap is the unavailable shared-host-SHM/explicit NVSHMEM
bootstrap contract, not NCCL connectivity.

### Q17 — Inspect supplied brainctl/RJob docs and retry

1. [Original Request] Inspect
   `/data/ycfeng/stepfun-env-handbook/brainctl`.
2. [Original Request] Inspect
   `/data/ycfeng/stepfun-env-handbook/brainctl-rjob.md`.
3. [Original Request] Verify whether the supplied brainctl differs from the
   installed binary.
4. [Original Request] Retry based on the new launcher guidance.

**Status:** Executed on 2026-08-15. The supplied and installed brainctl files
are byte-identical. The documented legacy launcher passed predict-only and
started two nodes. NCCL and explicit NVSHMEM initialization passed on all
`16/16` ranks, but `deep_ep.Buffer` rejected the runtime with:

```text
nvshmem_n_pes() == num_ranks
```

No complete-model retry was run because the minimal Buffer gate failed.

### Q18 — Stop retrying DeepEP and continue remaining operations

1. [Original Request] If the current execution is still focused on DeepEP,
   finish the current attempt.
2. [Original Request] If DeepEP still fails, temporarily skip DeepEP operation
   measurement, record the failure, and continue with the next operation.

**Status:** Resolved by explicit owner direction on 2026-08-15.

**Execution consequence:**

- The current active failure is SWA QKV norm/RoPE, not DeepEP.
- Existing DeepEP evidence is sufficient to record the family as deferred and
  blocked; do not launch another DeepEP measurement in this task phase.
- Continue grouped `wo_a`, FP32 router, QKV norm/RoPE, canonical dataset
  archival, and MTP-off simulation work.
- Missing DeepEP exact keys must remain visible. Do not use H800 data,
  generic communication rows, synthetic latency, or another transport as a
  substitute.

### Q19 — Inspect DeepEP launch examples and cases

1. [Original Request] Inspect the image referenced by
   `task_memory/step4pro_v4_external_simulator_requirements.md` for example or
   case scripts that launch DeepEP.
2. [Original Request] Inspect
   `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro`
   for example or case scripts that launch DeepEP.

**Status:** Completed by read-only inspection on 2026-08-15.

### Q20 — Apply the later DeepEP/NVSHMEM launch fix and retry

1. [Original Request] Treat commit `9bfd9a610e` as an intended DeepEP/NVSHMEM
   fix based on its title and inspected content.
2. [Original Request] Relax the earlier insistence on remaining at pinned
   commit `607d1641ee`.
3. [Original Request] Apply commit `9bfd9a610e`.
4. [Original Request] Retry according to the launch modules and bootstrap
   behavior changed by that commit.

**Status:** Applied and statically verified on 2026-08-15; live validation is
blocked because every installed launcher rejects the required
`--share-host-shm` flag and no standalone `rjob` client is installed.

**Scope clarification:** Commit `9bfd9a610e` is the direct child of
`607d1641ee` and changes only `rjob-step4pro-2node.sh` plus the new
`rjob-step4pro-deepep-probe.sh`. This authorization reopens only the exact
DeepEP/NVSHMEM launch-contract probe. It does not authorize unrelated model,
backend, shape, precision, or simulator changes.

### Q21 — Authorize the bounded SWA QKV annotation compatibility overlay

1. [Original Request] Approve the proposed SWA QKV runtime annotation
   compatibility overlay.
2. [Original Request] Continue AIC-side implementation, B300 measurement,
   validation, and simulation refresh.

**Status:** Explicitly authorized on 2026-08-16.

**Scope clarification:** The approved overlay is process-local and limited to
resolving `FusedQKNormRope.kernel` annotations `reload_from` and
`delay_w_load` from postponed strings to the installed `cutlass.Constexpr`
object after verifying the exact image-native QKNorm source SHA256. It must
not rewrite installed source files, replace the pinned vLLM provider, change
kernel code, shape, dtype, argument values, QKV math, or persisted operation
identity. A representative B300 smoke remains mandatory before the `75`-case
SWA collection.

### Q22 — Select the temporary DeepEP simulation proxy

**Question:** Which measured operation may temporarily supply dispatch/combine
latency while the real B300 DeepEP environment is unavailable?

**Recommended choice A:** Use the existing B300 NCCL `alltoall` dataset as an
explicit simulation-only proxy. Preserve the Step4 DeepEP dispatch/combine
operation identities, derive their communication volume from the existing
Step4 MoE contract, label every affected result as proxy rather than exact
DeepEP silicon, and do not write a fake `step4_deepep_ht_perf.parquet`.

**Alternative B:** Use another operation or dataset selected by the task
owner. The exact operation, dtype, volume mapping, and EP16/EP32 topology must
be supplied before implementation.

**Status:** **Resolved — user selected A on 2026-08-16.**

**Decision:**

- [Original Request] Use B300 NCCL `alltoall` as the temporary,
  simulation-only DeepEP proxy.
- [Original Request] Model dispatch with an FP8 communication payload and
  combine with a BF16 communication payload.
- [Original Request] Mark affected simulation results explicitly as `PROXY`;
  do not represent them as measured DeepEP silicon.
- [Original Request] Do not create a substitute
  `step4_deepep_ht_perf.parquet`.
- [Original Request] Restore real DeepEP measurement and replace the proxy
  results after the DeepEP environment becomes available.

### Q23 — Final simulation review, test-record normalization, and publication

1. [Original Request] Re-review whether the simulation requirements in
   `task_memory/step4pro_v4_external_simulator_requirements.md` are complete
   for the currently approved scope, temporarily excluding real DeepEP.
2. [Original Request] Fix or supplement any missing simulation requirement.
3. [Original Request] Normalize and organize the test results and test records
   for manual review.
4. [Original Request] Create one task-scoped commit containing the relevant
   documentation and code modules, then push it to the remote repository.

**Scope carried forward from prior decisions:**

- The reviewed simulation variant is `mtp_off`; native Step4Pro MTP1 remains
  explicitly deferred.
- DeepEP dispatch/combine may use only the explicitly selected B300 NCCL
  `alltoall` simulation proxy and every affected result must remain labeled
  `PROXY`.
- Whole-model pinned-vLLM B300 execution remains externally owned and its
  missing comparison data must not be invented.
