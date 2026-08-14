## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created the task log and recorded the initial review and blocking scope discrepancy. |
| 2026-08-13 | Recorded the access-probe session: measured B300 quota RBAC denial, vLLM credential denial, cached-image implementation gap, and the AIC Step4-Pro-V4 shape conflict. Produced the platform ticket packet. |
| 2026-08-13 | Revalidated latest branch/image identity and B300 access; corrected stale blocker conclusions and stopped at the semantic scope gate. |
| 2026-08-13 | Confirmed the missing manifest by repository-wide filename search and recorded the dirty linked-worktree implementation gate. |
| 2026-08-13 | Recorded user choice A to reconstruct the missing manifest with explicit provenance and a new SHA256. |
| 2026-08-13 | Completed the pinned-vLLM versus AIC fidelity-gap audit and opened the operation-boundary decision. |
| 2026-08-13 | Clarified that the current Full-MFA inventory is source-derived and added a required B300 runtime-trace validation gate. |
| 2026-08-13 | Recorded user confirmation of operation-boundary option A and advanced to the dirty-worktree checkpoint gate. |
| 2026-08-13 | Created the approved task branch and resolved the missing Git LFS dependency that caused its post-checkout hook to fail. |
| 2026-08-13 | Recorded user confirmation of branch/checkpoint option A and staged the audited baseline file set. |
| 2026-08-13 | Created baseline commit `4f2b0c31`, then recorded the focused baseline result: 859 passed and 128 failed. |
| 2026-08-13 | Recorded the owner's decision to retain the temporary security-review file without any security action. |
| 2026-08-14 | Stopped the local pinned-vLLM smoke lane, created an external execution handoff, and restricted this session to AIC-side work. |
| 2026-08-14 | Strengthened the Latest MTP-off RED contract and verified the expected missing-implementation failure. |

# Progress Log

## 2026-08-13 — Initial audit

**Status:** blocked at requirements-confirmation gate.

### Motivation
Establish an isolated task record and verify the authoritative inputs before changing code or allocating B-card resources.

### Method
- Created this task directory and the required planning/archive documents.
- Read the supplied requirements document.
- Read the two historical task directories and their planning/profiling records.
- Read the GPU, Docker, and local environment handbooks.
- Inspected Git status, branches, local references, and Collector operation-development rules.

### Result
- Historical methodology and current repository constraints are recorded.
- The requirements document specifies B300 and fixed vLLM commit `607d1641ee3fec43653fca510d717725828890c2`.
- The user request specifies an unspecified latest B-card vLLM image and `step4-pro-latest`.
- The two linked machine-readable/reference files named by the requirements document are missing at their expected paths.
- A bounded local Docker image listing did not reveal a matching cached Step4/latest/B300 image. This does not establish registry absence.
- No code, GPU allocation, Docker launch, or collection has been performed.

### Next required action
Obtain the exact latest image/source identity and confirm whether the fixed requirements-document commit remains authoritative for this task.

## 2026-08-13 — Access probe and blocker quantification

**Status:** still blocked. Four blockers are now measured rather than suspected.

### Motivation
The initial audit left three questions inconclusive: whether B-card capacity is actually grantable, whether the pinned vLLM source is reachable, and whether the locally cached StepCast image could serve as the implementation source. Each had to be settled with evidence before either escalating to the user or starting work, because a wrong assumption on any of them would silently invalidate the whole operation set.

### Expectation
Either find at least one viable path to the target implementation and B-card hardware, so execution could begin, or produce citable evidence that no such path exists, so the escalation is actionable rather than a bare refusal.

### Method
- Probed cluster access with `rlaunch --predict-only` across B-card tag spellings and across every B300 quotagroup, with `codesign` plus H800 as the control case.
- Enumerated B300 hardware with `brainctl get nodes -l GPUType=B300 -o json` and tallied nodes, GPUs, and quotagroup ownership. Used the JSON form after observing that the plain-text listing returns a sample rather than the full set.
- Inspected a B300 node's labels with `brainctl describe node` to establish ownership.
- Tested vLLM repository authentication with `ssh -T` and `git ls-remote`, and checked `~/.ssh/` for usable keys.
- Inspected the cached StepCast post15 image's installed vLLM tree for the three implementation files named in requirements section 2.1, and read its model registry mapping.
- Probed the internal registry catalog endpoints unauthenticated.
- Compared the requirements shape against all three existing AIC Step4-Pro model configurations.
- Verified Docker daemon reachability through `sudo`.

### Result
- B300 exists but is not grantable. 20 nodes and 160 GPUs are present, all owned by `b300_*` quotagroups; every one returns HTTP 403 for this account. Every B-card tag spelling returns `no machine available`, as does pinning the grantable `codesign` quota to a B300 node. The H800 control case succeeds, which confirms the probes are valid. Recorded as ISSUE-002.
- The pinned vLLM repository is reachable over the network but rejects authentication, and no private key exists locally. Recorded as ISSUE-003.
- The cached post15 image lacks `step4pro.py` and `optimus_fa4.py`, and its registry routes `Step4ForCausalLM` to the Step3p5 implementation. It cannot serve as the implementation source. This converts the earlier inconclusive local-cache note into a measured result. Recorded as ISSUE-004.
- The requirements shape differs from the existing AIC `Step4-Pro-V4` on every load-bearing dimension, so the model-definition decision is undecidable without user confirmation. Recorded as ISSUE-005.
- Docker through `sudo` works and the mirror is configured, so containerization itself is not a blocker.
- Two delegated history-review agents returned only idle notifications and no reports. Their scope was covered directly instead, by reading the H800 coverage manifests, the prior task's notes and progress entries, and the AIC model and collector sources. The only item not recovered is the prior task's verbatim collector command lines, which are not needed until B-card access exists.
- No code was changed, no GPU was allocated, and no collection was run. One empty duplicate task directory created in error during this session was removed immediately; no pre-existing content was touched.

### Next required action
Escalate the four measured blockers to the user with the platform ticket packet at `b300_quota_ticket.md`. Execution stays halted pending B300 quotagroup membership, vLLM credentials or a correct image reference, and confirmation of the model identity and shape.

## 2026-08-13 — Latest identity and access revalidation

**Status:** paused for explicit scope confirmation; no production code or formal collection started.

### Motivation
The previous stop record contained conclusions that could have become stale after
the B300 quota and GitLab access changed. Continuing from those conclusions would
have violated the no-assumption rule.

### Expectation
Recheck the exact source/image identity and B300 availability, then either close
the access gate with evidence or stop with only the remaining semantic decisions.

### Method
- Cloned the latest `xwx/step4pro-fa3-optimus` branch read-only and recorded its
  current SHA.
- Checked ancestry against the requirements commit
  `607d1641ee3fec43653fca510d717725828890c2`.
- Read the branch's B300 scripts and resolved the referenced image tag,
  manifest digest, and config digest with read-only Docker registry inspection.
- Verified the target Step4Pro source files and model registry entry.
- Re-ran B300 predict-only for a two-replica, eight-GPU-per-replica request.

### Result
- Latest source SHA: `9bfd9a610ea4f2890010702ee7a207cf25edf8de`.
- Requirements SHA is an ancestor; the latest delta adds B300
  DeepEP/NVSHMEM bootstrap logic.
- Candidate image:
  `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`,
  manifest digest
  `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`,
  config digest
  `sha256:48253062026862b2921f354613d3e9a7255e9540e3a7bf9148c9c80ec2d72661`.
- B300 predict-only succeeds through `b300_train_infra`.
- The remaining blocker is not access: the user must confirm whether the
  requirements' 78-layer synthetic shape remains the experiment contract,
  what AIC model identity to register, and whether Optimus/DeepEP internal
  kernels remain provenance under existing logical op families or require new
  consumer-visible op types.

## 2026-08-14 — User supplied the pinned vLLM checkout

**Status:** paused at the semantic scope-confirmation gate.

### Motivation
The user added the local vLLM checkout path, removing the earlier source-location uncertainty.

### Expectation
Record the checkout as an auditable source input without silently resolving the remaining runtime, model-shape, or operation-boundary decisions.

### Method
- Verified the checkout exists at `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro`.
- Verified the checkout is detached at `607d1641ee3fec43653fca510d717725828890c2`, matching the requirements document's pinned commit.
- Compared the pinned checkout's model registry and Step4Pro implementation files with the requirements source list.
- Kept the later branch-head/image candidate as a separately recorded reference, not an implicit replacement.

### Result
- The source path and exact commit are now recorded in the task requirements and notes.
- The pinned checkout exposes `Step4ProForCausalLM` and the target implementation files.
- No AIC production code, Collector code, GPU job, dataset, or simulation was started.
- One decision remains before implementation: which source/runtime identity defines `step4-pro-latest` when the user wording says "latest" but the requirements document pins commit `607d1641ee3fec43653fca510d717725828890c2`.

## 2026-08-13 — Choice A recorded and operation inventory completed

**Status:** source/runtime gate completed; AIC identity decision pending.

### Motivation
Apply the user's choice without silently changing the pinned source or existing Step4-Pro-V4 identity.

### Expectation
Close the source/runtime gate, derive the actual prefill/decode logical graph, and identify consumer gaps before production edits.

### Method
- Recorded choice A in `requirements.md`, `plan.md`, `notes.md`, `issues.md`, and `design.md`.
- Audited pinned Step4Pro, Optimus FA4, Optimus FP8 MoE, inherited Step3p5 execution, and AIC Python consumers.
- Separated logical AIC operations from custom kernel/provider identities.
- Verified the pinned scripts declare the previously inspected B300 image tag.

### Result
- Authoritative source: pinned local commit `607d1641ee3fec43653fca510d717725828890c2`.
- Authoritative runtime image: pinned-script tag with manifest digest `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`.
- Prefill/decode logical graph and provenance are recorded in `findings.md` and `op_provenance.md`.
- Optimus kernels remain provider provenance under logical ops.
- Two material gaps are explicit: vLLM DeepEP consumer/data contract and native Step4Pro MTP1 construction.
- No production code or GPU collection was started.

## 2026-08-13 — Canonical AIC identity selected

**Status:** model identity resolved; missing-manifest decision pending.

### Motivation
Prevent collision with the existing, materially different Step4-Pro-V4 model.

### Expectation
Fix one stable model path before writing config, tests, Collector cases, or coverage manifests.

### Method
Presented the separate canonical identity `stepfun-ai/Step4-Pro-Latest` as the recommended choice.

### Result
The user selected `stepfun-ai/Step4-Pro-Latest`. Existing Step4 model identities remain unchanged. No production code was modified.

## 2026-08-13 — Missing-manifest search and worktree safety audit

**Status:** paused at Q3; no production code or B300 collection started.

### Motivation
Resolve whether the referenced machine-readable shape contract exists anywhere
in the available workspace, and determine whether implementation can safely
start in the current repository state.

### Expectation
Either recover the exact original manifest and verify its stated SHA256, or
prove that a user decision is required. Separately, establish whether a clean
isolated implementation branch can preserve the uncommitted Step4 foundation.

### Method
- Delegated a recursive ordinary-file filename search under
  `/data/ycfeng/stepfun-performance-optimization`.
- Inspected linked-worktree metadata, ignore rules, current HEAD coverage, and
  the tracked/untracked state without modifying any files outside this task
  record.

### Result
- Neither `step4pro_v4_shape_manifest.json` nor
  `step4pro_v4_vllm_b300_8bit_task.md` exists in the searched tree.
- Search evidence is stored at
  `/data/ycfeng/tmp/step4_manifest_search.txt`.
- The original manifest SHA256 cannot be verified from available files.
- This checkout is already a linked worktree, but it contains 37 modified
  tracked files plus many untracked historical artifacts.
- A new worktree from HEAD would omit the untracked V3/V4 configs and tests
  needed as implementation references; `.worktrees/` is also not ignored.
- No reset, stash, commit, worktree creation, code edit, GPU allocation, or
  collection was performed.

## 2026-08-13 — Manifest reconstruction decision

**Status:** Q3 resolved; production implementation remains paused at the
worktree-safety gate.

### Motivation
Close the missing machine-readable shape contract without pretending that the
unavailable original file or SHA256 has been recovered.

### Expectation
Obtain an explicit owner decision on whether to reconstruct or wait.

### Method
Presented reconstruction as choice A and exact-original recovery as choice B.

### Result
The user selected A. The task will reconstruct the canonical manifest from the
requirements document and pinned source, compute a new SHA256, and mark the
artifact as a reconstruction. No manifest file or production code has yet been
written.

## 2026-08-13 — AIC fidelity-gap audit

**Status:** read-only audit complete; Q4 is now the active design gate.

### Motivation
Determine whether the existing Step4 schema and leaf operations can represent
the pinned Full-MFA/SWA-GQA, latent-MoE, DeepEP, shared-KV, and MTP1 execution
without silent approximation.

### Expectation
Identify the smallest consumer-visible changes required before writing RED
tests or production code.

### Method
Delegated a bounded comparison of the pinned vLLM execution path against the
current AIC Step4 config, operation graph, Attention/MoE consumers, and MTP
handling.

### Result
- Current config cannot express Full MFA and SWA GQA in one model.
- Current Full MFA graph collapses low-rank/shared-KV/grouped-output behavior
  into ordinary Q/K/V/O projections.
- Current latent-MoE decode path assumes overlap not established by the pinned
  source and does not consume vLLM DeepEP data.
- Current MTP handling is a uniform scaling approximation rather than the
  required explicit MTP1 graph.
- The smallest audited extension is a heterogeneous latest config plus
  grouped-GEMM, DeepEP dispatch/combine, KV-layout, FP32-router, and explicit
  MTP1 contracts while reusing existing operations elsewhere.
- Evidence:
  `/data/ycfeng/tmp/step4_latest_aic_gap_audit.txt`
  (`sha256:433c39c7dc1cf60b5dcef75a4053087a38a7e0b520071371d100bc3dd0860351`).
- No production file, Collector case, GPU job, dataset, or simulation was
  modified or started.

## 2026-08-13 — Full-MFA profiling provenance clarification

**Status:** Q4 remains open; measurement provenance is now explicit.

### Motivation
Answer whether the proposed profiling graph is directly produced by the
pinned Step4-Pro vLLM implementation.

### Expectation
Distinguish the completed source-level inventory from runtime profiling
evidence that has not yet been collected.

### Method
Reconciled the proposed graph with the recorded pinned-source call paths and
defined the acceptance gate for formal B300 rows.

### Result
- The current table is a logical AIC graph extracted from
  `Step4ProDecoderLayer` and the Full-MFA forward path in pinned vLLM.
- It is not an automatically exported B300 profiler graph, and no B300
  Full-MFA profiling has yet run.
- Before accepting Collector cases or data, the actual pinned-image forward
  path must be runtime-profiled and reconciled for operation order, shapes,
  dtypes, fusion, optional branches, backend choice, and kernel provider.
- Optimus FA4 and grouped `wo_a` must be measured through their actual
  vLLM/provider paths. Generic substitution or multiplied estimates are
  prohibited.

## 2026-08-13 — Operation-boundary option A confirmed

**Status:** design direction confirmed; production implementation remains
paused at Q5.

### Motivation
Close the critical architecture question raised by the fidelity audit.

### Expectation
Choose between minimum fidelity-preserving extensions, known approximation, or
a broad framework refactor.

### Method
Explained the Full-MFA composite graph, direct pinned-vLLM provenance,
runtime-trace gate, shared-KV contract, and grouped `wo_a` measurement path.

### Result
The user confirmed option A. The implementation scope is now the minimum
explicit extension set recorded in `requirements.md` and `design.md`. No
production code, branch, commit, Collector case, GPU job, or dataset was
created.

## 2026-08-13 — Approved branch creation and Git LFS recovery

**Status:** branch created; baseline checkpoint selection is in progress.

### Motivation
Implement the approved isolation strategy without altering or losing the
existing Step4 working state.

### Expectation
Create `task/step4-pro-latest-b300`, then checkpoint the relevant baseline with
all repository hooks passing.

### Method
- Ran `git switch -c task/step4-pro-latest-b300`.
- Investigated the nonzero result from the repository `post-checkout` hook.
- Verified the hook itself was intact and the host lacked the `git-lfs`
  executable.
- Installed Ubuntu package `git-lfs=3.4.1-1ubuntu0.4` from the company
  Artifactory mirror.
- Re-ran the existing hook against the current revision.

### Result
- Branch `task/step4-pro-latest-b300` exists at
  `56c606a171b5ed5e132301490aba0fd33fbf458d`.
- The first branch-switch command returned exit code 1 only because the
  post-checkout hook could not find `git-lfs`; the branch switch itself had
  completed.
- `git lfs version` now reports `git-lfs/3.4.1`.
- The unchanged repository `post-checkout` hook now exits 0.
- No hook was deleted or bypassed, and no stash, reset, removal, move,
  baseline commit, production edit, or GPU action occurred.

## 2026-08-13 — Baseline checkpoint option A approved and staged

**Status:** exact baseline file set staged; commit verification is in progress.

### Motivation
Preserve the existing Step4 foundation on the new task branch before adding
Step4-Pro-Latest behavior.

### Expectation
Stage only relevant source/config/tests/task documentation and keep generated
outputs, caches, historical result data, H800 datasets, and the pinned vLLM
clone outside the checkpoint.

### Method
- The user selected branch/checkpoint option A.
- Delegated a read-only boundary audit of 37 tracked changes and 1,293
  untracked entries.
- Staged all 37 modified tracked files.
- Staged 4 untracked source/config files, 30 untracked test files, 14 current
  task documents, and the authoritative requirements document.
- Used exact path lists; did not use `git add .`, `git add -A`, or broad
  directory staging.

### Result
- Exactly 86 files are staged.
- No generated profiler output, cache, H800 performance dataset, historical
  task artifact directory, or `vllm-step4-pro` file is staged.
- No commit has yet been created; staged-diff checks remain pending.

## 2026-08-13 — Baseline checkpoint committed and focused tests failed

**Status:** implementation paused for systematic root-cause analysis.

### Motivation
Verify that the checkpointed Step4 foundation is internally consistent before
adding Step4-Pro-Latest behavior.

### Expectation
All selected Step4/Collector/runtime-spec baseline tests pass.

### Method
- Verified exactly 86 staged files, zero unstaged tracked files, no excluded
  paths, and no whitespace errors.
- Created commit `4f2b0c31` with message
  `chore: checkpoint existing Step4 support`.
- Ran 987 focused tests with Python 3.11.15 under a 4 GiB systemd memory scope.

### Result
- Checkpoint commit succeeded.
- Test result: 859 passed, 128 failed, duration 69.92 seconds.
- Failure distribution:
  - factorized-attention runtime spec: 79;
  - Step4-Pro-V1 model: 41;
  - Step4-Pro-V1 roofline: 4;
  - DeepSeek-V4 runtime spec: 4.
- Raw log:
  `/data/ycfeng/tmp/step4_latest_baseline_pytest.log`
  (`sha256:5a8d406abc9e382322ef9ecc973c9a579fad415c1c66a854682a50f5d2f25289`).
- No Step4-Pro-Latest production implementation has started.

## 2026-08-13 — Security-review file decision

**Status:** security follow-up closed by explicit owner decision; baseline
contract investigation remains active.

### Motivation
A read-only audit reported that one temporary evidence file might contain
environment variables, while repository rules prohibit deletion without owner
approval.

### Expectation
Obtain an explicit retain/delete decision without reading or exposing the file.

### Method
Presented deletion plus credential rotation as option A and retain/no-action
as option B.

### Result
The user selected B and stated that the current environment is sufficiently
secure. The file will not be read, deleted, modified, moved, permission-changed,
quoted, or otherwise processed. No credential action will be taken.

## 2026-08-13 — Q6 scope clarified against pinned vLLM

**Status:** awaiting confirmation of the clarified combined decision.

### Motivation
The original Q6 wording mixed a legacy `Step4-Pro-V1` baseline repair choice
with the separate requirement that `Step4-Pro-Latest` follow the actual pinned
vLLM operation graph.

### Expectation
Make it explicit that no legacy V1 choice can replace or simplify the
Step4-Pro-Latest graph extracted from vLLM.

### Method
- Rechecked the existing pinned-source operation inventory for commit
  `607d1641ee3fec43653fca510d717725828890c2`.
- Compared the two Q6 descriptions with the active attention construction and
  forward paths in `vllm/model_executor/models/step4pro.py`.
- Recorded the clarified user intent and scope boundary in `requirements.md`.

### Result
- The pinned vLLM uses heterogeneous attention: shared-KV Full MFA/MQA on full
  layers and sliding-window GQA/SWA on the other layers.
- This is closest to the shared-MFA wording of Q6 option B, but Q6 itself is
  only about repairing the pre-existing V1 baseline.
- The recommended combined decision is to preserve the historical V1 contract
  while implementing Latest independently and strictly from pinned vLLM.

## 2026-08-13 — Q6 combined contract confirmed

**Status:** resolved; baseline repair can proceed after obsolete-test ownership
is explicitly settled.

### Motivation
Remove ambiguity between preserving the existing V1 model and implementing the
new Latest model from the actual vLLM graph.

### Expectation
Keep the two model identities independent so legacy compatibility cannot alter
Latest fidelity.

### Method
Presented the pinned-source result and asked for one combined decision instead
of treating V1 and Latest as competing architectures.

### Result
The user explicitly confirmed:

- V1 keeps its historical contract.
- Latest strictly follows the pinned vLLM implementation.

No production code was changed as part of this clarification.

## 2026-08-13 — Obsolete-test and formula-loader audit completed

**Status:** deletion permission pending; no test or production file changed.

### Motivation
Separate mistakenly checkpointed withdrawn tests from valid V1 regression
tests before repairing the baseline.

### Expectation
Remove only tests whose entire contract was withdrawn, retain tests that
enforce the confirmed V1 behavior, and fix the production root cause rather
than weakening assertions.

### Method
- Audited failure names, historical task decisions, checkpoint file ancestry,
  and the relevant `dsv4.py` query ordering.
- Compared each failure cluster with the newly confirmed V1/Latest boundary.

### Result
- The 79 factorized-attention and 4 DeepSeek-V4 runtime-spec failures come from
  two files newly and mistakenly added in checkpoint `4f2b0c31`; the historical
  task explicitly withdrew that migration.
- The four V1 roofline failures are valid regression tests and must remain.
- Their root cause is an unconditional `load_data()` call before the
  `SOL`/`SOL_FULL` formula-only branch.
- Deleting the two obsolete files now requires explicit owner permission.

## 2026-08-13 — Baseline repair and strict Latest fidelity approved

**Status:** approved; execution started.

### Motivation
Close the final baseline ownership gate and make the vLLM implementation the
explicit end-to-end standard for Latest definitions, tests, measurements, and
validation.

### Expectation
Remove only the withdrawn tests, fix the valid V1 production defect, and reject
any Latest profiling row not produced through the pinned vLLM operation path.

### Method
The user explicitly approved deletion and repair and restated the Latest
operation-fidelity requirement.

### Result
- Permission exists to delete the two exact obsolete test files.
- The four V1 formula-only tests remain acceptance tests.
- Latest work is gated on exact pinned-vLLM operation/provider execution.

## 2026-08-13 — Obsolete tests removed and formula-loader root cause fixed

**Status:** this failure cluster is green; remaining V1 contract failures are
under analysis.

### Motivation
Restore the approved baseline boundary and ensure formula-only HCA queries do
not depend on collector data.

### Expectation
- The two withdrawn DSV4 runtime-spec files are absent.
- All four `SOL`/`SOL_FULL` no-load tests pass.
- The complete V1 roofline test file remains green.

### Method
- Deleted, with explicit owner approval:
  - `tests/unit/sdk/database/test_factorized_attention_runtime_spec.py`
  - `tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py`
- Reproduced the four valid RED cases:
  `4 failed in 4.33s`.
- Moved `ContextDeepSeekV4AttentionModule.load_data()` and
  `GenerationDeepSeekV4AttentionModule.load_data()` after the `SOL` and
  `SOL_FULL` early returns in
  `src/aiconfigurator/sdk/operations/dsv4.py`.
- Re-ran the focused cases and the complete roofline file.

### Result
- Focused formula-only cases: `4 passed in 3.72s`.
- Complete V1 roofline file: `34 passed in 7.65s`.
- `git diff --check`: PASS.
- The fix changes source ordering only; empirical/silicon modes still call
  `load_data()` before accessing measured data.

## 2026-08-13 — Historical V1 contract RED baseline reproduced

**Status:** RED confirmed; exact historical implementation recovery is in
progress.

### Motivation
Verify the remaining baseline failures after removing obsolete tests and
repairing the formula-loader defect.

### Expectation
The failures should be limited to the previously identified mixed V1 attention
contract rather than Latest code.

### Method
Ran:

```bash
PYTHONPATH=src:. \
  /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  -m pytest tests/unit/sdk/models/test_step4_pro_v1.py -q
```

### Result
- `268` tests collected.
- `227 passed`, `41 failed`, duration `3.26s`.
- Failure groups cover missing historical `FullAttentionConfig` and
  `NonFullAttentionConfig`, parser validation, explicit attention runtime
  specs, independent Full/HCA geometry, and TP-sharded KV accounting.
- Raw log:
  `/data/ycfeng/tmp/step4_v1_contract_red_20260813.log`
- SHA256:
  `d4e8c6fc170f0cfd9f2052321c53f51256de8c521c5a83f1b2b40c39390354e8`
- No Latest implementation or Collector file was changed.

## 2026-08-13 — Historical V1 baseline restored and verified

**Status:** baseline gate passed; Latest RED-test planning is now unblocked.

### Motivation
Restore the user-confirmed historical V1 boundary without allowing an
unfinished shared-MFA migration to define either V1 or Latest.

### Expectation
- V1 uses standard Full Attention plus HCA with TP-sharded full K/V.
- Generic MFA support can coexist but does not define the V1 cached model.
- All adjusted focused baseline tests and static checks pass.

### Method
- Corrected the mixed test boundary:
  - the cached V1 schema test now requires historical Full/HCA;
  - the invalid V1 factorized-runtime-spec test was removed;
  - generic MFA tests no longer claim to define the cached V1 model.
- Reconstructed frozen `FullAttentionConfig` and
  `NonFullAttentionConfig` from the approved historical requirements, tests,
  formulas, and remaining model call paths.
- Restored the cached V1 JSON Full/HCA sections.
- Added dual Pro-schema parsing so historical V1 and generic MFA are explicit
  alternatives rather than fallbacks.
- Restored V1 TP divisibility checks, parameter reporting, differentiated KV
  formulas, and formula-only loader ordering.
- Ran focused RED/GREEN cycles, the full adjusted baseline, Ruff, format,
  JSON, and whitespace checks.

### Result
- V1 model suite: `267 passed in 0.40s`.
- V1 roofline suite: `34 passed in 7.65s`.
- Final adjusted baseline: `899 passed in 51.84s`, `0 failed`.
- Static validation: Ruff PASS, format PASS, JSON PASS,
  `git diff --check` PASS.
- Final log:
  `/data/ycfeng/tmp/step4_latest_baseline_repaired_final_pytest.log`
- SHA256:
  `c7a869263afd16b8694259ecafbe7df29e5a3a02320298a01e9e6009d5b68154`
- No `Step4-Pro-Latest` production implementation or B300 collection has
  started.

## 2026-08-14 — Pinned vLLM MTP1 boundary confirmed

**Status:** MTP1 scope is blocked pending owner clarification; MTP-off Latest
work is unblocked.

### Motivation
Determine whether the requirements' MTP1 graph is part of the authoritative
pinned Step4-Pro implementation before defining or measuring it.

### Expectation
Accept MTP1 only if the pinned source contains a Step4Pro-specific construction
and runtime path. Do not silently use Step3p5MTP or an AIC-only substitute.

### Method
- Audited the pinned checkout at commit
  `607d1641ee3fec43653fca510d717725828890c2`.
- Traced Step4Pro model registration, main forward construction, MTP registry,
  speculative configuration conversion, and MTP layer construction.
- Compared the result with the requirements' explicit Step4Pro MTP1 note.

### Result
- Step4Pro main forward has no native MTP predictor.
- The available MTP path is `Step3p5MTP` with `Step3p5DecoderLayer`; it is not
  a Step4Pro implementation.
- MTP1 cannot be accepted, measured, or simulated under the pinned Latest
  contract without an owner-approved source extension.
- MTP-off Latest implementation, runtime trace, collection, and simulation may
  proceed independently.
- Evidence: `/data/ycfeng/tmp/step4_mtp1_boundary_audit_20260814.txt`.

## 2026-08-14 — Baseline checkpoint freshly reverified

**Status:** ready to commit after the MTP boundary record is included.

### Motivation
Create fresh evidence immediately before checkpointing the approved V1 repair.

### Expectation
The complete adjusted baseline must remain green with zero failures and no
whitespace errors.

### Method
Re-ran the same 899-test adjusted baseline under Python 3.11.15 and a 4 GiB
systemd memory scope, then ran `git diff --check`.

### Result
- `899 passed`, `0 failed`, `61.56s`.
- `git diff --check`: PASS.
- Log:
  `/data/ycfeng/tmp/step4_latest_baseline_repaired_final_rerun.log`
- SHA256:
  `30b93e06095b32c0bc81d5b3740c2b4f4d1875414075d13f30f064e6dfc27a63`.

## 2026-08-14 — MTP1 and B300 smoke status reconfirmed

**Status:** MTP1 is absent from the pinned Step4Pro implementation; the
requirements smoke has not been completed or evidenced.

### Motivation
Confirm the two execution prerequisites before starting Latest AIC
implementation and avoid treating resource probes as a successful vLLM smoke.

### Method
- Rechecked pinned checkout commit
  `607d1641ee3fec43653fca510d717725828890c2`.
- Compared the requirements' MTP1 contract and smoke commands with the pinned
  model registry, Step4Pro model, MTP implementation, speculative conversion,
  task progress, and available B300 evidence artifacts.

### Result
- The pinned checkout has no native `Step4Pro` MTP1 path. Its only Step-family
  MTP implementation is `Step3p5MTP`, whose predictor block is
  `Step3p5DecoderLayer`; this cannot be accepted as Latest Step4Pro ground
  truth.
- B300 quota prediction, hardware probing, and cross-zone tar transport were
  verified, but the requirements' Step4Pro service smoke was not run to a
  successful, auditable result. There is no evidence of `/health`, FA4
  backend, prefill/decode requests, four-way concurrency, or the 78-layer
  target-shape smoke.
- Evidence:
  `/data/ycfeng/tmp/step4_mtp1_boundary_audit_20260814.txt`,
  `task_memory/step4pro_v4_external_simulator_requirements.md:135-191`,
  and the B300 probe artifacts under `/data/ycfeng/tmp/b300probe/`.

## 2026-08-14 — Execution scope updated by owner

**Status:** MTP-off execution is active; MTP1 structure work is deferred.

### Motivation
Remove the confirmed MTP1 source gap from the immediate critical path without
silently changing the pinned-vLLM fidelity rule.

### Expectation
- AIC Latest can be implemented and tested without an MTP1 graph.
- B300 pinned-vLLM smoke runs in parallel.
- Formal profiling and simulation remain gated by runtime evidence and
  MTP-off correctness.

### Method
- Recorded the owner's scope decision in `requirements.md`, `plan.md`,
  `harness.md`, and `design.md`.
- Split the active work into an AIC/Collector track and a B300 smoke track.

### Result
- MTP1 structure tests, measurement, and simulation are explicitly deferred.
- MTP-off Latest AIC ops, Collector, B300 measurement, correctness, and
  prefill/decode E2E simulation are now active scope.
- The B300 smoke must use the requirements' pinned image and vLLM checkout;
  mount is allowed only when platform rules permit it.

## 2026-08-14 — Pinned-vLLM smoke/runtime trace handed off

**Status:** externalized; no active local B300 smoke resource remains.

### Motivation

Let a separate session own the whole-model pinned-vLLM smoke and provider trace
while this session stays focused on AIC implementation, operation collection,
testing, and simulation.

### Expectation

- The external agent receives enough fixed identities, environment rules,
  reusable evidence, failure history, acceptance criteria, and cleanup rules
  to execute without repeating earlier failed probes.
- This session launches no pinned-vLLM whole-model smoke.
- AIC work continues without silently treating missing external trace evidence
  as a PASS.

### Method

- Interrupted and closed the active B300 smoke agent.
- Verified zero matching RJobs, Replicas, and local smoke processes.
- Preserved all existing evidence under
  `/data/ycfeng/tmp/b300_step4_smoke_20260814/`.
- Created
  `pinned_vllm_b300_smoke_runtime_trace_execution.md`.
- Updated requirements, plan, harness, and design ownership boundaries.

### Result

- Existing B300 prediction remains PASS with 10 candidate nodes.
- The most recent hardware evidence remains
  `NVIDIA B300 SXM6 AC`, `275040 MiB`.
- Whole-model smoke status remains **not run to PASS**.
- No `/health`, prefill/decode, four-concurrency, FA4 runtime, or DeepEP
  runtime result is claimed in this session.
- Current execution scope is now AIC-only.

## 2026-08-14 — Complete Latest MTP-off RED contract established

**Status:** RED verified; production implementation is the next step.

### Motivation

The first RED file checked only one Full layer, one SWA layer, and a partial
MoE sequence. It could become green while still aggregating attention and FFN
families in an order that differs from the pinned vLLM decoder loop.

### Expectation

The test contract must fail until AIC provides:

- all 78 decoder layers in execution order;
- Full MFA with required inverse RoPE and grouped `wo_a`;
- SWA GQA with Q/K/V normalization;
- Dense SiTU-GLU and serial Latent MoE/shared-expert execution;
- exact FP32-router, Optimus FA4, grouped-einsum, Optimus MoE, and DeepEP HT
  identities;
- logical and page-allocated KV bytes;
- no MTP1 graph and no TP greater than one.

### Method

- Extended the reconstructed manifest with the missing provider, operation
  order, communication, and KV-layout fields.
- Strengthened `tests/unit/sdk/models/test_step4_pro_latest.py`.
- Ran the manifest test, the first missing-production identity test, Ruff, JSON
  validation, and whitespace validation.

### Result

- Manifest contract: `1 passed in 0.08s`.
- Latest identity RED: `1 failed in 0.26s`, at the expected missing
  `Step4ProForCausalLM` registration.
- Ruff: PASS.
- JSON parse: PASS.
- `git diff --check`: PASS.
- No production AIC code was changed during this RED step.
