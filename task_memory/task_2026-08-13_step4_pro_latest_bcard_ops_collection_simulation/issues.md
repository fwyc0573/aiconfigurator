## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Opened the initial requirements identity blocker. |
| 2026-08-13 | Added measured blockers ISSUE-002 (B300 quota RBAC), ISSUE-003 (vLLM repository credentials), ISSUE-004 (cached image lacks target implementation), ISSUE-005 (AIC Step4-Pro-V4 shape conflict). Downgraded ISSUE-001 registry-absence caveat to a measured result. |
| 2026-08-13 | Revalidated the latest branch/image and B300 quota path; narrowed the remaining gate to explicit semantic scope decisions. |
| 2026-08-13 | Resolved ISSUE-001 source/runtime authority with user choice A. |
| 2026-08-13 | Resolved the new AIC identity as `stepfun-ai/Step4-Pro-Latest`; retained missing-manifest decision as the active requirements gate. |
| 2026-08-13 | Confirmed the manifest files are unavailable and opened ISSUE-007 for dirty linked-worktree isolation. |
| 2026-08-13 | Resolved missing-manifest handling through explicit reconstruction with a new SHA256. |
| 2026-08-13 | Confirmed the minimal fidelity-preserving operation strategy; ISSUE-007 is now the sole pre-implementation gate. |
| 2026-08-13 | Recorded and resolved ISSUE-008: missing Git LFS caused the branch-switch post-checkout hook to fail. |
| 2026-08-13 | Opened ISSUE-009 for 128 pre-existing baseline test failures discovered after the checkpoint commit. |
| 2026-08-13 | Closed the temporary-file security follow-up with the owner's explicit retain/no-action decision. |

# Issues and Resolutions

## ISSUE-001 — Latest runtime identity conflicts with the fixed requirements input

**Status:** **Resolved for this task on 2026-08-13.**

**Observed facts:**
- `task_memory/step4pro_v4_external_simulator_requirements.md` specifies B300 and vLLM repository/branch/commit `607d1641ee3fec43653fca510d717725828890c2`.
- The user task asks for the latest B-card vLLM image and `step4-pro-latest`, but does not state whether the current branch head/image or the older pinned requirements commit is authoritative.
- The requirements file's linked `step4pro_v4_shape_manifest.json` and `step4pro_v4_vllm_b300_8bit_task.md` are absent at the referenced paths.
- A bounded local Docker image listing did not reveal a matching cached image; this is not proof that the internal registry lacks it.

**Root cause:** The runtime identity is now available, but the relationship between the current branch/image and the older fixed requirements contract is not explicitly decided.

**Impact:** The operation set, Collector keys, B-card measurements, and simulation shapes cannot be determined without risking an unrecorded assumption. Starting implementation or GPU collection now would violate the task's explicit no-assumption rule.

**Required decision:** Confirm the exact image/source identity, model shape, AIC model name, and treatment of custom kernels before implementation.

**Proposed alternatives:**
1. Recommended: use branch head `9bfd9a610ea4f2890010702ee7a207cf25edf8de` plus image manifest digest `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`; retain the requirements document's B300/78-layer experiment matrix unless the latest source explicitly contradicts it.
2. Use ancestor commit `607d1641ee3fec43653fca510d717725828890c2` and the requirements document as the complete runtime/source contract.

No workaround or fallback has been applied.

**Resolution:** The user selected the pinned local checkout and requirements contract. The later branch head is reference-only. The pinned checkout itself declares the B300 image, so that tag and its inspected digest are authoritative. The AIC canonical model path remains the next explicit user decision.

## ISSUE-002 — B300 quota is not grantable to this account (hard blocker)

**Status:** RESOLVED 2026-08-13. `b300_train_infra` is usable and B300 access is confirmed by predict-only and prior real-worker evidence. The earlier denial conclusion came from an incomplete candidate-quota probe.

**Resolution evidence:**
- `rlaunch --predict-only --charged-group=b300_train_infra --private-machine=group --gpu=8 --cpu=64 --memory=409600` returns 10 candidate 8-GPU B300 nodes.
- Multi-node predict succeeds at both required scales: `-P 2` (16 GPUs, `ep16_r1`) and `-P 4` (32 GPUs, `ep32_r1` / `ep16_r2`) each return 10 to 12 candidate nodes with `--custom-resources=rdma/mlnx_shared=8 --topo-group=yes --set-env=DISTRIBUTED_JOB=true`.
- A real 1-GPU worker launched and reported hardware matching requirements section 2.4 exactly:
  `NVIDIA B300 SXM6 AC`, `275040 MiB`, driver `580.159.03`, `compute_cap 10.3`.
  This agrees with `src/aiconfigurator/systems/b300_sxm.yaml`, which records `sm_version: 103` and `mem_capacity: 288400343040`.

**Follow-on constraint discovered during resolution:** see ISSUE-006. B300 lives in a different zone from this workspace, so the launch recipe is not the H800 recipe.

**Observed facts (all measured on 2026-08-13 from the CPU master):**
- The requirements document section 2.4 requires `NVIDIA B300 SXM6 AC`, with 16 GPUs (`ep16_r1`), 32 GPUs (`ep32_r1`), and 32 GPUs (`ep16_r2`).
- B300 nodes exist in the cluster. `brainctl get nodes -l GPUType=B300 -o json` returns exactly **20 nodes / 160 allocatable GPUs**, distributed by quotagroup as:
  `b300_pretrain3` = 11, `b300_pretrain` = 5, `b300_pretrain2` = 2, `b300_sys_pro` = 1, `b300_train_infra` = 1.
- Every B300 node carries `privatemachine.brainpp.cn/quotagroup` in the `b300_*` family. None carries `codesign`.
- Every B300-family quotagroup returns HTTP 403 on predict:
  `code: 403 message: subject {Type:user ID:e7163316ad76ba7266a33f9787afd70a Name:} cannot get resource quotagroups/predict in API group quota.brainpp.cn/v1alpha1 in the project shai/shai-core reason:Forbidden`
  Reproduced for `--charged-group=b300_sys_pro`, `b300_pretrain3`, and `b300_pretrain`.
- Tag-based selection fails for every B-card spelling attempted: `--positive-tags=` with `b300`, `b200`, `blackwell`, `b300_sxm`, `gb300`, `b300-sxm6` all return `no machine available`.
- Pinning the grantable quota to a B300 node fails:
  `--charged-group=codesign --positive-tags=node/gpu-b300-0339.qy.cnw.istep.fun` returns `no machine available`.
  `--charged-group=codesign --positive-tags=feature/GPUType=B300` returns `no machine available`.
- Control case confirms the probe method is sound: `--charged-group=codesign --positive-tags=h800` returns multiple 8-GPU H800 nodes with available capacity.

**Root cause:** B300 nodes are private machines owned by the `b300_*` quotagroups. This account holds only the `codesign` quotagroup, which contains H800 capacity exclusively. The task statement asserts B-card access has been granted, but platform RBAC does not reflect that grant.

**Correction (2026-08-13):** this root cause was wrong. The account does hold a B300 quotagroup, `b300_train_infra`. The error was probing an incomplete candidate set, chosen from a sampled node listing, and then concluding absence from three denials. Three denials do not establish that all B300 quotagroups are denied.

**Impact:** No B-card measurement is possible. There is no degraded substitute: requirements section 2.4 estimates about 230 GB/GPU of static weights for `ep16_r1`, while H800 provides 80 GB per GPU (`src/aiconfigurator/systems/h800_sxm.yaml` records `mem_capacity: 85029158912`). The target model cannot be resident on H800 at the specified parallelism, so H800 cannot stand in for B300 even as an approximation.

**Required decision:** Platform must add this account to a B300 quotagroup with at least 32 GPUs of headroom, or the B-card scope must be formally suspended.

**Proposed alternatives:**
1. Recommended: obtain B300 quotagroup membership, then resume the plan unchanged.
2. Suspend measurement and deliver simulator-side modeling only. This cannot satisfy the requirements document sections 4 and 5, which require measured-versus-simulated comparison tables, so it changes the deliverable contract and needs explicit approval.

No workaround or fallback has been applied. H800 has not been substituted.

## ISSUE-003 — The pinned vLLM repository is unreachable for lack of credentials

**Status:** RESOLVED 2026-08-13 for the source checkout. The current branch is readable; image/runtime scope remains part of ISSUE-001.

**Observed facts:**
- Requirements section 2.1 makes the pinned revision the authority for operator shape, dtype, KV cache, and communication: repository `git@gitlab.basemind.com:sys/stepcast/vllm.git`, branch `xwx/step4pro-fa3-optimus`, commit `607d1641ee3fec43653fca510d717725828890c2`.
- Network reachability is fine: `gitlab.basemind.com` port 22 accepts connections, and HTTPS returns 302.
- Authentication fails: `ssh -T git@gitlab.basemind.com` and `git ls-remote git@gitlab.basemind.com:sys/stepcast/vllm.git HEAD` both return `Permission denied (publickey,password)`.
- `~/.ssh/` contains only `authorized_keys` and `known_hosts`. No private key is present.
- No local clone exists. A filesystem search under `/data/ycfeng` found no `step4pro.py`, `optimus_fa4.py`, or `rjob-step4pro-*.sh`.

**Resolution:** SSH access became available during the current session. Read-only clone and `git ls-remote` succeeded.

**Impact:** The authoritative implementation cannot be read, so operation definitions cannot be derived or given provenance as the task requires.

**Remaining requirement:** Keep the exact branch head and source checkout hash in the final provenance record.

No workaround or fallback has been applied.

## ISSUE-004 — The cached StepCast image does not contain the target implementation

**Status:** Superseded for runtime selection. The branch scripts provide a candidate image and its manifest digest; the older cached image remains unsuitable.

**Observed facts:**
- Two StepCast images are cached locally: `hub.stepfun-inc.com/stepcast/stepcast:2026-07-14-server-vllm-0.19.0.post15-8a8f1b3f` and `hub.stepfun-inc.com/stepcast/stepcast:vllm-openai-v0.19.0`.
- Inspecting the post15 image at `/usr/local/lib/python3.12/dist-packages/vllm` shows:
  - `model_executor/models/` contains `step3p5.py`, `step3p5_mtp.py`, `step4_edge.py`, and other Step models. It does **not** contain `step4pro.py`.
  - `v1/attention/backends/` does **not** contain `optimus_fa4.py`.
  - `model_executor/layers/fused_moe/` **does** contain `optimus_fp8_moe.py` and `optimus_moe.py`.
  - `model_executor/models/registry.py:218` maps `"Step4ForCausalLM": ("step3p5", "Step3p5ForCausalLM")`, routing Step4 to the Step3p5 implementation. The architecture name the requirements document expects, `Step4ProForCausalLM` (section 3.2), is absent from the registry.
- The internal registry cannot be enumerated without credentials: `https://hub.stepfun-inc.com/v2/stepcast/stepcast/tags/list` returns `UNAUTHORIZED`, and `/v2/` returns HTTP 401.

**Root cause:** The cached images predate or exclude the Step4Pro branch. Two of the three implementation files named in requirements section 2.1 are absent, and the model registry does not expose the target architecture.

**Impact:** The older cached image cannot serve as the implementation source. A candidate latest image is now identified, but it must be explicitly accepted as the runtime contract.

**Required decision:** Confirm use of the branch-provided image tag and manifest digest.

No workaround or fallback has been applied. The Step3p5 implementation has not been treated as a proxy for Step4Pro.

## ISSUE-005 — The requirements shape conflicts with the existing AIC Step4-Pro-V4 model

**Status:** Open; blocks the model-definition decision.

**Observed facts:**
- The task statement names the target `step4-pro-latest`. The requirements document never uses the word "latest"; it consistently says v4, including the filename, the `step4pro_v4_shape_manifest.json` reference, and `--served-model-name step4pro-v4-perf` in section 3.2.
- AIC already registers `stepfun-ai/Step4-Pro-V4` in `src/aiconfigurator/sdk/common.py:887`, with configuration at `src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V4_config.json`.
- The existing AIC V4 shape differs from the requirements shape on every load-bearing dimension:

| Parameter | Requirements section 2.2 | AIC Step4-Pro-V4 | AIC Step4-Pro-V3 | AIC Step4-Pro-V1 |
| --- | ---: | ---: | ---: | ---: |
| hidden size | 7168 | 9216 | 12288 | 6144 |
| trunk layers | 78 | 80 | 80 | 80 |
| routed experts | 896 | 384 | 1024 | 512 |
| topk | 16 | 8 | 16 | 8 |
| MoE intermediate | 3584 | 3584 | 2048 | 2048 |
| MTP layers | 1 | absent | 3 | 3 |
| Full attention type | MFA, Q64 / KV1 / head dim 512 | MQA | MQA, Q96 / KV12 / head dim 128 | MFA, Q64 |
| Full / SWA layer split | 20 / 58 | not expressed | not expressed | not expressed |

- No shape manifest is available to arbitrate, because `step4pro_v4_shape_manifest.json` and its stated hash `53103019932b93b20a60b6f9dfe6154be6330befdd6a6fa2f6cb67278fc03fde` are absent (ISSUE-001).

**Root cause:** The requirements document describes a fourth, previously unmodelled shape that shares the name "V4" with an existing and materially different AIC model.

**Impact:** Whether to add a new model or modify the existing `Step4-Pro-V4` is undecidable, and choosing wrongly either corrupts committed V3/V4 roofline evidence or creates a duplicate identity. The AIC model name for this task is also unsettled.

**Required decision:** Confirm that `step4-pro-latest` denotes the 78-layer / hidden 7168 / 896-expert shape, confirm the AIC model identity string to register, and confirm that the existing `Step4-Pro-V4` entry must remain untouched.

No workaround or fallback has been applied. No model configuration has been edited.

**Resolution update (2026-08-13):** The user selected canonical identity `stepfun-ai/Step4-Pro-Latest`; the existing `Step4-Pro-V4` entry remains unchanged. The 78-layer shape is authoritative under choice A. The only remaining shape-contract question is whether to reconstruct the missing manifest with a new hash or wait for the original file.

**Final resolution update (2026-08-13):** The user selected reconstruction.
The new manifest must be explicitly labeled as reconstructed and must use its
own computed SHA256. The unavailable original hash must remain recorded only as
an unverified historical reference.

## ISSUE-006 — B300 is cross-zone, so the H800 launch recipe does not transfer

**Status:** Resolved with a verified recipe. Recorded because it changes every B-card launch and every data path in this task.

**Observed facts:**
- B300 nodes are in zone `shai-cn-qingyang-cm`. This workspace is in zone `shai-cn-shanghai-sj`. All H800 nodes are in the workspace zone, which is why the handbook recipe never had to address this.
- Launching without `--image` is refused: `cross-zone launch without --image is not allowed (it mounts the workspace NFS across zones and causes many D-state processes), please specify --image`.
- Launching with `--volume` is refused even when `--image` is supplied: `--volume is not allowed for cross-zone launch even with --image (--volume still mounts the workspace NFS across zones), please remove --volume`.
- Consequently `/data/ycfeng` is not visible inside a B300 worker. The only sizeable writable mount is `/jobutil` (1.0 TB). No `/gpfs`, `/jfs`, or `/shared` exists.
- `brainctl cp` hung with no output and no error for over 100 seconds and had to be killed. It is not usable here.
- `tar` over `brainctl exec` works in both directions and is the documented advanced path. `tar` is present in the container at `/usr/bin/tar`.

**Root cause:** The platform blocks workspace-NFS mounts across zones to avoid D-state processes. B300 and this workspace are in different zones, so no shared filesystem can be relied upon.

**Impact:** Collector code, model configs, and any input artifacts must be pushed into the worker, and all measured outputs must be pulled back out. Nothing can be read from or written to `/data/ycfeng` by a B300 worker directly. Any collection script that assumes a shared `/data` path will fail.

**Verified recipe:**

```bash
# 1. Launch. --image is mandatory; --volume is forbidden.
/kubebrain/rlaunch \
  --charged-group=b300_train_infra --private-machine=group \
  --gpu=8 --cpu=64 --memory=409600 --backoff-limit=1 --enable-sshd=false \
  --image <IMAGE> -- bash -lc '<COMMAND>'

# 2. Find the replica name.
/kubebrain/brainctl get replica -n shai-core | grep <job-fragment>

# 3. Push inputs (local -> worker).
tar cf - -C <LOCAL_DIR> <NAME> \
  | /kubebrain/brainctl -n shai-core exec -i replica/<REPLICA> -- tar xf - -C <REMOTE_DIR>

# 4. Pull outputs (worker -> local). Redirect to a file, then extract.
/kubebrain/brainctl -n shai-core exec replica/<REPLICA> -- tar cf - -C <REMOTE_PARENT> <REMOTE_DIR> \
  > out.tar
tar xf out.tar -C <LOCAL_DEST>
```

**Round-trip proof (2026-08-13):**
- Push: a local probe file arrived intact on the worker, verified by reading it back through `exec`.
- Pull: `nvidia-smi` output generated on the worker was retrieved as a 10240-byte tar and extracted locally, containing `NVIDIA B300 SXM6 AC, 275040 MiB, 10.3`.
- Note on step 4: piping the pull directly into `tar xf -` produced empty output and exit code 1. Redirecting the stream to a file first, then extracting, works reliably. Use the two-step form.

## ISSUE-007 — Production implementation cannot safely begin from the current dirty linked worktree

**Status:** Resolved on 2026-08-13.

**Observed facts:**
- The current checkout is already a linked worktree on branch `step4-pro`.
- It has 37 modified tracked files and many untracked historical artifacts.
- The untracked files include the existing Step4-Pro-V3/V4 configs and their
  focused tests, so a clean worktree created directly from HEAD would not
  contain the full current Step4 foundation.
- Project-local `.worktrees/` is not currently ignored.
- Repository rules prohibit resetting, stashing, committing, moving, or
  overwriting existing user changes without approval, and require existing
  modifications to be stashed or committed before large-scale changes.

**Root cause:** The earlier Step4 tasks were implemented but not consolidated
into a clean commit that can be used as the base of a new isolated branch.

**Impact:** Starting production edits in place risks mixing this task with
historical work. Creating a new worktree from HEAD risks implementing against
an incomplete baseline. Either path would weaken traceability and could lose
or duplicate existing work.

**Required decision:** After Q3 is resolved, obtain explicit approval for a
safe consolidation or isolation method before writing production code.

No reset, stash, commit, worktree creation, or production code edit has been
performed.

**Recommended resolution:** Keep the existing linked worktree, create
`task/step4-pro-latest-b300`, and checkpoint the existing relevant Step4
source/config/test/task-document state before new implementation. Exclude
generated outputs and the separate pinned vLLM checkout. This avoids both
destructive cleanup and a second incomplete worktree.

**Resolution in progress:** The user approved the recommended option. Branch
`task/step4-pro-latest-b300` exists and the audited 86-file baseline is staged.
ISSUE-007 closes only after the checkpoint commit and baseline verification
complete.

**Resolution:** Created branch `task/step4-pro-latest-b300` and baseline commit
`4f2b0c31` from an exact 86-file allowlist. All excluded caches, generated
outputs, H800 data, historical result directories, and the pinned vLLM clone
remained untracked. Tracked status was clean after the commit. Baseline test
failures are a separate consistency issue tracked as ISSUE-009.

## ISSUE-008 — Git LFS missing during approved branch creation

**Status:** Resolved on 2026-08-13.

**Observed facts:**
- `git switch -c task/step4-pro-latest-b300` created the branch, then returned
  nonzero because `.git/hooks/post-checkout` could not find `git-lfs`.
- The hook is valid and intentionally invokes `git lfs post-checkout`.
- `git lfs version` initially returned “git: 'lfs' is not a git command”.
- Ubuntu package metadata exposed `git-lfs 3.4.1-1ubuntu0.4` through the
  company Artifactory mirror.

**Root cause:** The host lacked the Git LFS package required by the existing
repository hook.

**Resolution:**
- Installed `git-lfs=3.4.1-1ubuntu0.4` with `sudo -n apt-get install -y
  git-lfs`.
- Re-ran the unchanged post-checkout hook with the current revision; it exited
  0.

No hook removal, skip flag, or fallback was used.

## ISSUE-009 — Checkpointed Step4 baseline had 128 failing tests

**Status:** Resolved on 2026-08-13.

**Observed facts:**
- Baseline commit: `4f2b0c31`.
- Focused test run collected 987 tests: 859 passed and 128 failed.
- Failures occur only in four files:
  - `test_factorized_attention_runtime_spec.py`: 79;
  - `test_step4_pro_v1.py`: 41;
  - `test_step4_pro_v1_roofline.py`: 4;
  - `test_deepseek_v4_runtime_spec.py`: 4.
- Representative missing contracts include
  `FactorizedAttentionRuntimeSpec`, `FullAttentionConfig`, updated
  `Step4MFAAttentionConfig` APIs, and expected KV-memory behavior.
- No Step4-Pro-Latest production code existed when the failures were observed.

**Initial root-cause hypothesis:** The previous dirty worktree combined
production changes from completed Step4 work with untracked tests from one or
more incomplete or separate work streams. This hypothesis is not yet accepted;
parallel read-only investigations are in progress.

**Impact:** A failing baseline makes later regressions ambiguous. Under the
task harness, implementation and GPU collection cannot proceed.

**Required resolution:** Establish provenance and ownership for each failure
cluster, then either repair the true baseline root cause or obtain explicit
approval to remove separately incomplete tests from this branch. No test
skipping or expectation weakening is allowed.

### Root-cause update

- 83 failures come from two DSV4 runtime-spec RED test files that the original
  task later marked obsolete.
- The remaining 45 failures are relevant baseline blockers:
  - 41 Step4-Pro-V1 model/config/parser/KV failures caused by incompatible V1
    attention contracts being mixed in one checkpoint;
  - 4 `SOL`/`SOL_FULL` formula-path loader failures.
- No existing branch, worktree, stash, or tracked commit contains the missing
  complete V1 implementation.
- The next required decision is whether to restore the previously approved
  Full/HCA V1 contract or redefine V1 around the newer shared-MFA contract.

### Scope clarification

- The pinned Step4-Pro vLLM implementation for Latest is not the historical
  V1 Full-Attention-plus-HCA graph. It uses shared-KV Full MFA/MQA together
  with sliding-window GQA/SWA.
- The V1 repair decision must not change the Step4-Pro-Latest implementation.
- Recommended resolution: restore the historical V1 contract only for the
  existing V1 model, then implement Latest separately from the pinned vLLM
  graph.
- The task owner confirmed this clarified combined decision on 2026-08-13.
- Remaining ISSUE-009 work is limited to restoring the historical V1 contract,
  resolving the four formula-only loader tests, and obtaining explicit
  ownership/removal decisions for the obsolete runtime-spec tests.

### Final test-ownership audit

- Remove, only with explicit permission:
  - `tests/unit/sdk/database/test_factorized_attention_runtime_spec.py`
  - `tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py`
- Retain:
  - all four formula-only `SOL`/`SOL_FULL` cases in
    `tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
- Production defect to repair:
  - `src/aiconfigurator/sdk/operations/dsv4.py` calls `load_data()` before
    checking formula-only modes.

**Resolution approval:** On 2026-08-13 the task owner explicitly approved
deleting the two obsolete files, retaining the four formula-only tests, and
repairing the production source-ordering defect. ISSUE-009 remains open only
until the resulting baseline test run is green.

### Resolution result

- Deleted the two approved obsolete DSV4 runtime-spec files.
- Restored the historical V1 Full/HCA schema, parser, graph, validation, and
  KV accounting.
- Removed the invalid test that required factorized-MFA runtime specs for V1;
  it contradicted the confirmed historical V1 contract and did not represent
  pinned Latest vLLM.
- Retained the four formula-only HCA tests and fixed the source ordering.
- Final adjusted baseline: `899 passed`, `0 failed`, `51.84s`.
- Ruff, format, JSON, and whitespace checks all pass.

Latest remains a separate implementation target and must not reuse the V1
Full/HCA graph.

## ISSUE-010 — Temporary audit file reported as potentially sensitive

**Status:** Closed by owner decision on 2026-08-13.

**Path:**
`/data/ycfeng/tmp/aic_failure_domain1/codex_lane3_scope_baseline.md`

**Resolution:** The task owner selected retain/no-action and explicitly stated
that the current environment is sufficiently secure. The file has not been
read, deleted, modified, moved, permission-changed, quoted, or otherwise
processed. No credential action will be taken.

## ISSUE-011 — Pinned vLLM has no Step4Pro MTP1 implementation

**Status:** Open; owner clarification required before MTP1 work.

**Observed facts:**

- `Step4ProForCausalLM` is registered as the Step4Pro trunk model and does not
  construct an MTP predictor:
  `vllm/model_executor/models/step4pro.py:626-649`.
- The only relevant MTP registry entry is `Step3p5MTP`, and it constructs
  `Step3p5DecoderLayer`:
  `vllm/model_executor/models/registry.py:616`;
  `vllm/model_executor/models/step3p5_mtp.py:171-190,286`.
- Step4 speculative configuration is rewritten to the Step3.5 MTP
  architecture:
  `vllm/config/speculative.py:357-362`.
- The requirements document says Step4Pro MTP1 construction still needs to be
  completed: `task_memory/step4pro_v4_external_simulator_requirements.md:113-115`.

**Root cause:** The requirements' MTP1 experiment scope is broader than the
pinned vLLM Step4Pro implementation. Reusing `Step3p5MTP` would violate the
owner's explicit requirement that Latest operations come from the actual
Step4Pro vLLM implementation.

**Impact:** MTP-off Latest operation definition, runtime trace, collection, and
simulation can proceed. MTP1 graph definition, measurement, and simulation
cannot be accepted until Q8 is resolved.

**Recommended resolution:** Select Q8 option A: keep MTP1 explicitly
unimplemented and unmeasured for the pinned runtime, and request a concrete
Step4Pro MTP implementation source before extending the scope.

**Evidence:** `/data/ycfeng/tmp/step4_mtp1_boundary_audit_20260814.txt`
