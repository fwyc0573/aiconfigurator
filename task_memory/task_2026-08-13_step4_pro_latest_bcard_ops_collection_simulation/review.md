## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-17 | Reviewed Phase 14 lifecycle/evidence hardening: one-delete teardown, strict cleanup, same-argument predict-only, quota evidence, timeout margin, and synchronized forward marker; local approve, live still quota-blocked. |
| 2026-08-17 | Reviewed the complete publication candidate, found no new blocking code issue, normalized stale reports, and approved a quota-blocked checkpoint commit/push. |
| 2026-08-17 | Reviewed the post-concurrency runtime files and approved the refreshed 16-file Phase 12 inventory. |
| 2026-08-17 | Reviewed the predict-only replica-sensitivity diagnostic and replaced it as a total-quota gate with direct platform confirmation. |
| 2026-08-17 | Reviewed the local two-node coordinator and global-marker-scope fixes; final live approval remains blocked by B300 quota. |
| 2026-08-17 | Reviewed and approved the configuration-only replacement of active DeepEP runtime settings with AgRs/NCCL communication. |
| 2026-08-17 | Reviewed the live two-node AgRs attempt; recorded the wrapper lifecycle blocker and rejected an ungrounded retry. |
| 2026-08-13 | Recorded the initial requirements and workspace checkpoint. |
| 2026-08-13 | Reviewed refreshed source/image/access evidence and recorded the remaining semantic gate. |
| 2026-08-13 | Reviewed pinned-source operation provenance, AIC fidelity gaps, and the approved minimal-extension boundary. |
| 2026-08-13 | Reviewed the branch checkpoint and failed focused baseline evidence. |
| 2026-08-13 | Reviewed and accepted the root-cause baseline repair with 899/899 passing tests. |
| 2026-08-14 | Reviewed the pinned-vLLM MTP1 boundary and fresh baseline checkpoint evidence. |
| 2026-08-14 | Recorded the owner-approved MTP-off execution scope and parallel B300 smoke gate. |
| 2026-08-14 | Reviewed the external B300 Stage 0 and Stage 1 probe wrapper after two platform-control failures and their root-cause fixes. |
| 2026-08-14 | Reviewed and approved the 3 GiB controller-scope and low-memory I/O wrapper repair for live validation. |
| 2026-08-15 | Approved the live Stage 1 pinned-source union and zero-resource cleanup evidence. |
| 2026-08-15 | Recorded the one-GPU admission blocker caused by missing qy1-pt mount authorization. |
| 2026-08-15 | Reviewed synthetic model load and isolated the Optimus SM103 native-kernel incompatibility. |
| 2026-08-15 | Reviewed and approved the complete Latest MTP-off model/op graph and nonlinear KV-capacity fix. |
| 2026-08-15 | Reviewed the grouped `wo_a` provider-data vertical slice and its regression evidence. |
| 2026-08-15 | Reviewed the exact pinned-vLLM FP32 router collector and consumer slice. |
| 2026-08-15 | Reviewed the exact Full-MFA/SWA QKV norm/RoPE slice and the pre-measurement Q-path correction. |
| 2026-08-15 | Reviewed provider Attention, hybrid physical pages, and simulator peak-KV capacity accounting. |
| 2026-08-15 | Re-reviewed all pinned B300 runtime deviations and narrowed the accepted claims to provider/performance validation. |
| 2026-08-15 | Reviewed the three bounded two-node attempts, live environment restoration, and final coordinated-DP headless/API role fix. |
| 2026-08-15 | Reviewed the owner-authorized fourth two-node attempt and isolated the blocking NCCL HCA launch gap. |
| 2026-08-15 | Reviewed the passing 16-rank NCCL preflight and the remaining DeepEP Buffer shared-host-SHM/NVSHMEM blocker. |
| 2026-08-15 | Reviewed the eight-rank local DeepEP discriminator and confirmed the cross-node no-retry blocker plus separate teardown defect. |
| 2026-08-15 | Reviewed the supplied brainctl/docs and legacy-launcher retry; narrowed the blocker to NVSHMEM runtime-state integration rather than proven shared-host-SHM absence. |
| 2026-08-15 | Reviewed and approved the complete B300 grouped `wo_a`/FP32 router collection, canonicalization, runtime trace, and 148/148 exact-consumer evidence. |
| 2026-08-15 | Reviewed exact application of `9bfd9a610e`; approved its static scope and confirmed a blocking mismatch with every installed launcher API. |
| 2026-08-15 | Reviewed corrected Attention 225/225 evidence and the full scheduler-aware MTP-off matrix checkpoint. |
| 2026-08-15 | Reviewed and approved Full-MFA QKV 75/75 collection, canonical exact-consumer evidence, and the updated SWA-QKV/DeepEP blocked matrix. |
| 2026-08-15 | Reviewed and approved the final formatting-only repair and fresh fail-fast verification checkpoint. |
| 2026-08-16 | Independent re-review approved all three provider-core acceptance remediations with no remaining Blocking or Important findings. |
| 2026-08-16 | Added the local Phase 10 proxy, Attention closure, artifact, and changed-symbol review. |
| 2026-08-16 | Removed an accidentally duplicated Phase 10 local-review entry before final archive verification. |
| 2026-08-16 | Added the local Phase 11 requirement-completeness, repeat, and manual-review artifact review. |
| 2026-08-16 | Added the final publication review with fresh requirement assertions, byte-identical simulation, regression, lint, shell, and structured-input evidence. |
| 2026-08-16 | Refreshed the final publication review with the user-requested pre-commit audit and regression evidence. |
| 2026-08-16 | Added the LF-only CSV serialization review and publication-format regression evidence. |

# Checkpoint Review Log

## Review 1 — Requirements and execution readiness

- **Target Component/Phase:** Phase 1 requirements and historical review; Phase 2 identity gate.
- **Reviewer Agent Identity:** Codex task execution agent, session dated 2026-08-13.
- **Inspected Artifacts:**
  - `task_memory/step4pro_v4_external_simulator_requirements.md`
  - `task_memory/task_2026-07-19_step4_pro_v3_v4_support_and_figures/`
  - `task_memory/task_2026-07-28_step4_pro_v3_v4_op_profiling/`
  - `/data/ycfeng/stepfun-env-handbook/guidence.md`
  - `/data/ycfeng/stepfun-env-handbook/docker.md`
  - `task_memory/env_handbook.md`
  - repository Git status/refs
  - `.agents/skills/aic-collector-op-development/SKILL.md`
- **Identified Issues/Anomalies:** The requested “latest B-card image” has no immutable identity and is not demonstrably the same as the fixed B300/vLLM commit in the requirements document; two linked requirement files are missing.
- **Remediation/Verification Code Actions Taken:** Created an isolated task directory and recorded the discrepancy in `requirements.md`, `plan.md`, `progress.md`, and `issues.md`. No production code or runtime action was taken.
- **Verdict:** BLOCK until user clarification through the grill-me gate.

### Review 2 — Revalidated latest source and B300 access

- **Target Component/Phase:** Phase 2 runtime identity and B300 access revalidation.
- **Reviewer Agent Identity:** Codex task execution agent, session dated 2026-08-13.
- **Inspected Artifacts:** Read-only vLLM checkout at
  `/data/ycfeng/tmp/step4pro-vllm-latest-readonly`; branch scripts
  `rjob-step4pro-optimus-single.sh`, `rjob-step4pro-2node.sh`,
  `vllm/model_executor/models/step4pro.py`,
  `vllm/v1/attention/backends/optimus_fa4.py`,
  `vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py`;
  requirements document; current task `notes.md`, `findings.md`, `issues.md`,
  and `progress.md`; read-only image manifest; B300 predict-only output.
- **Identified Issues/Anomalies:** The current branch head and candidate image
  are now identifiable and B300 access is available. The supplied requirements
  still pins an ancestor commit and a synthetic shape that is not the existing
  AIC `Step4-Pro-V4`; custom-kernel scope is also not explicit.
- **Remediation/Verification Code Actions Taken:** Updated task records to
  replace stale access/credential blockers with measured current evidence. No
  production code, GPU worker, or formal collection was started.
- **Verdict:** BLOCK pending explicit user decisions on runtime authority,
  model identity/shape, and logical-op versus custom-kernel scope.

### Review 3 — Pinned operation graph and AIC fidelity boundary

- **Target Component/Phase:** Phase 4 latest op inventory and AIC design.
- **Reviewer Agent Identity:** Codex lead agent with delegated read-only audits
  from agents `019ffc17-c458-7c01-bf15-a93ba141ddeb` and
  `019ffc18-8206-7190-a035-a13bbb6d4f6c`.
- **Inspected Artifacts:**
  - pinned `vllm-step4-pro` checkout at commit
    `607d1641ee3fec43653fca510d717725828890c2`;
  - `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/op_provenance.md`;
  - `/data/ycfeng/tmp/step4_latest_vllm_ops.txt`;
  - `/data/ycfeng/tmp/step4_latest_vllm_deepep_contract.txt`;
  - `/data/ycfeng/tmp/step4_latest_aic_gap_audit.txt`;
  - current AIC Step4 model, Attention, MoE, overlap, Collector registry, and
    framework-manifest consumers.
- **Identified Issues/Anomalies:** Existing schema cannot represent mixed Full
  MFA/SWA GQA; Full MFA loses low-rank/shared-KV/grouped-output structure;
  vLLM DeepEP rows are not consumed; latent-MoE overlap differs from pinned
  order; MTP1 is not explicit.
- **Remediation/Verification Code Actions Taken:** Defined the smallest
  fidelity-preserving extension boundary and added a pinned-image runtime-trace
  acceptance gate. The user approved option A. No production code was changed.
- **Verdict:** DESIGN APPROVED; implementation remains BLOCKED only by
  ISSUE-007 branch/checkpoint safety.

### Review 4 — Baseline checkpoint and test readiness

- **Target Component/Phase:** Branch/checkpoint gate and pre-implementation
  baseline validation.
- **Reviewer Agent Identity:** Codex lead agent; boundary audit by explorer
  agent `019ffc4a-69b3-7100-bcef-de2a685f20ee`.
- **Inspected Artifacts:**
  - Git branch `task/step4-pro-latest-b300`;
  - baseline allowlist and exclusion audit under
    `/data/ycfeng/tmp/aic_step4_baseline_boundary_audit_2026-08-13_020351/`;
  - commit `4f2b0c31`;
  - `/data/ycfeng/tmp/step4_latest_baseline_pytest.log`;
  - `test_report_2026-08-13_step4_pro_latest_baseline.md`.
- **Identified Issues/Anomalies:** The exact checkpoint completed safely, but
  128 of 987 focused baseline tests fail in four files. Missing contracts
  include factorized-attention runtime specifications and Step4-Pro-V1
  attention/KV APIs.
- **Remediation/Verification Code Actions Taken:** Installed the missing Git
  LFS dependency rather than bypassing hooks; committed an audited 86-file
  baseline; executed tests in the verified Python 3.11.15 environment under a
  4 GiB systemd scope; opened ISSUE-009 and dispatched independent read-only
  root-cause investigations.
- **Verdict:** CHECKPOINT APPROVED; IMPLEMENTATION BLOCKED by ISSUE-009.

### Review 5 — Baseline root-cause repair

- **Target Component/Phase:** Obsolete DSV4 tests, historical V1 contract,
  formula-only HCA loading, and pre-Latest baseline readiness.
- **Reviewer Agent Identity:** Codex lead execution agent with delegated
  read-only provenance audits from Bernoulli, Dewey, and Boyle.
- **Inspected Artifacts:**
  - baseline and repaired pytest logs;
  - historical V1 task requirements, plan, progress, and review records;
  - V1 cached config;
  - `src/aiconfigurator/sdk/common.py`;
  - `src/aiconfigurator/sdk/utils.py`;
  - `src/aiconfigurator/sdk/models/step4.py`;
  - `src/aiconfigurator/sdk/operations/dsv4.py`;
  - V1 model, roofline, integration, Collector, V3/V4, and performance tests.
- **Identified Issues/Anomalies:** Checkpoint `4f2b0c31` mixed a withdrawn
  DSV4 runtime-spec migration, an unfinished shared-MFA V1 migration, and the
  approved historical V1 Full/HCA contract. No Git commit or object contained
  a complete bit-exact copy of the approved historical implementation.
- **Remediation/Verification Code Actions Taken:** Reconstructed the approved
  V1 schema and formulas from authoritative historical records and RED tests;
  deleted only the two owner-approved obsolete files; removed the invalid V1
  factorized-runtime-spec assertion; preserved generic MFA as a separate
  explicit schema; fixed formula-loader ordering. Verification passed
  `899/899` tests in `51.84s`; Ruff, format, JSON parse, and
  `git diff --check` all passed.
- **Verdict:** BASELINE APPROVED; ISSUE-009 CLOSED. Latest RED-test planning
  may proceed, but Latest must use the pinned vLLM graph rather than V1.

### Review 6 — MTP1 boundary and checkpoint freshness

- **Target Component/Phase:** Pinned Step4Pro MTP1 source boundary and stable
  V1 baseline checkpoint.
- **Reviewer Agent Identity:** Codex lead execution agent with read-only
  explorer agent `019ffe60-555a-7692-9af4-7e9e86ed5927`.
- **Inspected Artifacts:**
  - pinned vLLM commit
    `607d1641ee3fec43653fca510d717725828890c2`;
  - Step4Pro model, Step3p5 MTP, model registry, and speculative config;
  - requirements MTP1 section;
  - fresh 899-test baseline log;
  - current production/test diff and `git diff --check`.
- **Identified Issues/Anomalies:** Pinned vLLM has no Step4Pro MTP class,
  registry entry, or construction path. `Step3p5MTP` is not a valid Step4Pro
  substitute under the confirmed fidelity rule.
- **Remediation/Verification Code Actions Taken:** Opened Q8/ISSUE-011 rather
  than inventing a graph; re-ran the complete adjusted baseline.
- **Verdict:** MTP-OFF LATEST WORK APPROVED TO PROCEED; MTP1 remains behind an
  explicit owner decision. Baseline evidence is fresh: `899/899` passed in
  `61.56s`.


### Review 7 — MTP-off scope activation

- **Target Component/Phase:** Phase 4–6, MTP-off Latest implementation and B300 smoke.
- **Reviewer Agent Identity:** Codex task execution agent, session dated 2026-08-14.
- **Inspected Artifacts:**
  - `task_memory/step4pro_v4_external_simulator_requirements.md`;
  - `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/requirements.md`;
  - `plan.md`, `harness.md`, `design.md`, `issues.md`;
  - pinned vLLM MTP boundary evidence `/data/ycfeng/tmp/step4_mtp1_boundary_audit_20260814.txt`.
- **Identified Issues/Anomalies:** The pinned Step4Pro source has no native MTP1 path. The requirements smoke and formal B300 collection remain uncompleted.
- **Remediation/Verification Code Actions Taken:** Marked ISSUE-011 resolved as deferred; limited current acceptance to MTP-off; activated two parallel lanes: AIC Latest ops/Collector/measurement/simulation and pinned-vLLM B300 smoke. Prohibited `Step3p5MTP` substitution.
- **Verdict:** MTP-off execution APPROVED; MTP1 structure, measurement, and simulation DEFERRED.

### Review 8 — External B300 Stage 0 and source-probe control path

- **Target Component/Phase:** External handoff Stage 0 and Stage 1 launch,
  source transport, identity validation, and cleanup.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-14.
- **Inspected Artifacts:**
  - `pinned_vllm_b300_smoke_runtime_trace_execution.md`;
  - `vllm-step4-pro/rjob-step4pro-optimus-single.sh`;
  - `vllm-step4-pro/rjob-step4pro-2node.sh`;
  - `tests/e2e/step4_pro_latest/run_b300_source_probe.sh`;
  - prior and current logs under
    `/data/ycfeng/tmp/b300_step4_smoke_20260814/`.
- **Identified Issues/Anomalies:** Namespace-wide Replica inventory exceeded
  `MemoryMax=2G`; a `60s` diagnostic timeout was incorrectly reused for the
  live launch and interrupted a queued RJob.
- **Remediation/Verification Code Actions Taken:** Replaced full Replica
  inventory with the exact RJob label selector; bound live launch to the
  worker-ready deadline; added four static contract tests; verified explicit
  zero-resource cleanup for the interrupted attempt.
- **Verdict:** Stage 0 APPROVED. Stage 1 wrapper control path APPROVED for the
  next bounded live attempt; runtime source identity remains unproven until
  that attempt completes.

### Review 9 — 3 GiB host-memory and cleanup control path

- **Target Component/Phase:** B300 one-GPU host wrapper, controller memory
  scope, I/O behavior, launcher process ownership, and cleanup ordering.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-14.
- **Inspected Artifacts:**
  - `tests/e2e/step4_pro_latest/run_b300_single_smoke.sh`;
  - `tests/e2e/step4_pro_latest/run_b300_source_probe.sh`;
  - `tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py`;
  - `tests/e2e/step4_pro_latest/test_run_b300_source_probe_static.py`;
  - owner-reported host OOM and current artifact sizes.
- **Identified Issues/Anomalies:** The prior launcher retained an unnecessary
  `bash -c` layer; cleanup killed the launcher before explicitly deleting the
  RJob; the handoff still documented `MemoryMax=2G`; large artifact handling
  needed an explicit disk-streaming rule.
- **Remediation/Verification Code Actions Taken:** Set controller scopes to
  3 GiB; kept exact resource queries; used direct
  `setsid sudo -n systemd-run`; reordered cleanup; retained disk-backed
  patch/manifest/tar/log handling; updated the stale static assertion.
  Verification passed `10/10` focused tests, three `bash -n` checks, and
  `git diff --check`.
- **Verdict:** STATIC CONTROL PATH APPROVED for a bounded live attempt.
  Runtime source, model, provider, and no-OOM evidence remain pending.

### Review 10 — Live Stage 1 pinned-source union

- **Target Component/Phase:** B300 Stage 1 writable path, source transport,
  Git/source identity, Python import identity, memory boundary, and cleanup.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - `tests/e2e/step4_pro_latest/run_b300_source_probe.sh`;
  - `tests/e2e/step4_pro_latest/remote_b300_source_probe.sh`;
  - `/data/ycfeng/tmp/b300_step4_smoke_20260814/source_probe_s4p-src-0815-125214/`;
  - exact post-run RJob and Replica queries.
- **Identified Issues/Anomalies:** The prior synchronous launcher delayed exec
  until worker exit; whole-repository transfer and repeated `pack-objects`
  work were unnecessary host-memory risks; platform RJob names must not exceed
  50 characters.
- **Remediation/Verification Code Actions Taken:** Backgrounded the launcher;
  used exact queries and streamed exec; reused the 404 KiB source payload;
  added a name-length gate; verified 2,103 source hashes, pinned Git identity,
  source import paths, B300 hardware metrics, and zero-resource cleanup.
- **Verdict:** STAGE 1 APPROVED. One-GPU model load and provider/runtime trace
  remain pending.

### Review 11 — One-GPU admission and mount authorization

- **Target Component/Phase:** Exact-resource prediction and Stage 2 one-GPU
  RJob admission.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - exact-resource predict log;
  - `tests/e2e/step4_pro_latest/run_b300_single_smoke.sh`;
  - pinned one-GPU recipe;
  - failed launch and cleanup logs for `s4p-one-0815-130422`;
  - exact post-attempt RJob, Replica, and process queries.
- **Identified Issues/Anomalies:** Current `brainctl` removed
  `--auto-port-num`, which was safely adapted to pod-local port `8000`.
  More importantly, admission rejected the authoritative `qy1-pt` mount
  because the current identity lacks bucket permission.
- **Remediation/Verification Code Actions Taken:** Kept the exact model and
  tokenizer paths, used the current CLI's valid launch form, preserved the
  streamed 404 KiB source payload, and verified that the rejected launch left
  zero RJobs, Replicas, and local processes. No model or mount fallback was
  introduced.
- **Verdict:** BLOCKED by ISSUE-018 before model runtime. Owner/platform access
  action is required before Stage 2 can continue.

### Review 12 — Synthetic model load and Optimus native compatibility

- **Target Component/Phase:** 14-layer dummy-weight model construction, source
  identity, FA4 availability, FP8 MoE backend selection, and first profile
  forward.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - generated 14-layer and 78-layer synthetic configs;
  - one-GPU wrappers and remote runtime script;
  - `s4p-dm2`, `s4p-dm3`, `s4p-dm4`, and launch-blocking evidence;
  - isolated torch, Triton count, Optimus quant, split-sequence, and wheel
    native-library probes.
- **Identified Issues/Anomalies:** The real checkpoint was incorrectly treated
  as mandatory and has been removed. Optimus 3.23.24's legacy namespace shim
  collided with vLLM RoPE and was corrected. The remaining failure is the
  image-native Optimus activation-quant kernel on SM103; the 3.23.24 wheel's
  torch2.8 native library is ABI-incompatible with image torch2.10.
- **Remediation/Verification Code Actions Taken:** Preserved pinned vLLM
  source, used vLLM dummy weights, removed data mounts, fixed only the external
  Optimus namespace overlay, ran launch-blocking and split-process kernel
  probes, and verified exact zero-resource cleanup.
- **Verdict:** CONFIG AND MODEL LOAD APPROVED. Runtime remains blocked by
  ISSUE-019 before the first Optimus routed-expert forward.

### Review 10 — Latest MTP-off model/op graph

- **Target Component/Phase:** Phase 4 and the model/op portion of Phase 5.
- **Reviewer Agent Identity:** Codex lead execution agent, 2026-08-15.
- **Inspected Artifacts:**
  - `src/aiconfigurator/sdk/common.py`;
  - `src/aiconfigurator/sdk/utils.py`;
  - `src/aiconfigurator/sdk/models/step4.py`;
  - `src/aiconfigurator/sdk/operations/identity.py`;
  - `src/aiconfigurator/sdk/operations/attention.py`;
  - `src/aiconfigurator/sdk/operations/moe.py`;
  - Latest config, reconstructed manifest, and focused unit tests;
  - focused 28-test log and historical 899-test log.
- **Identified Issues/Anomalies:** Latest initially bypassed the nonlinear KV
  inverse, returning 512 tokens for a budget that exactly fits 513. Provider
  operations intentionally still fail fast because their Collector datasets
  and consumer tables do not yet exist.
- **Remediation/Verification Code Actions Taken:** Added a RED capacity test,
  routed Latest through the existing binary-search inverse, and re-ran both
  the focused and historical suites.
- **Verdict:** MODEL/OP GRAPH APPROVED. Collector and provider-specific
  PerfDatabase support remain the next required gate.

### Review 12 — Grouped `wo_a` provider-data vertical slice

- **Target Component/Phase:** Phase 5–6 pinned runtime routing, Latest
  Collector plan, grouped-einsum benchmark, and consumer lookup.
- **Reviewer Agent Identity:** Codex lead execution agent, 2026-08-15.
  Independent merge-readiness review lanes were not available in this
  session; this is a sub-step checkpoint review, not final merge approval.
- **Inspected Artifacts:**
  - `collector/framework_manifest.py` and YAML;
  - `collector/model_cases.py`;
  - `collector/cases/models/Step4ProForCausalLM_cases.yaml`;
  - `collector/case_generator.py`;
  - `collector/vllm/collect_step4_provider.py`;
  - vLLM registry and collector version resolver;
  - `GroupedGEMM` loader/query and perf filename;
  - focused, Collector, and SDK/database test output.
- **Identified Issues/Anomalies:**
  - Pinned vLLM uses a non-PEP440 `.g<commit>` version suffix, which previously
    normalized to version zero and could reject compatible collectors.
  - Full SDK/database regression contains one baseline CustomAllReduce test
    that contradicts the production fail-fast contract.
- **Remediation/Verification Code Actions Taken:**
  - normalized the git-describe suffix only for compatibility comparison while
    retaining exact raw package-version validation;
  - kept grouped data separate from generic GEMM;
  - required exact structural keys and rejected conflicting rows;
  - recorded the unrelated communication mismatch as ISSUE-019 without
    changing its logic or test.
- **Verdict:** GROUPED SLICE ACCEPTED FOR CONTINUED DEVELOPMENT. B300 runtime
  measurement and the remaining provider operations are still required.

### Review 13 — FP32 router provider-data vertical slice

- **Target Component/Phase:** Phase 5–6 FP32 router case population, exact
  pinned-vLLM benchmark invocation, persisted rows, loader, and consumer query.
- **Reviewer Agent Identity:** Codex lead execution agent, 2026-08-15.
- **Inspected Artifacts:**
  - pinned `step4pro.py`, `step3p5.py`, and `step3p5_util.py` call path;
  - `collector/vllm/collect_step4_provider.py`;
  - vLLM registry and provider perf filename;
  - `FP32OutputGEMM` loader/query;
  - RED, focused GREEN, Collector, and SDK/database outputs.
- **Identified Issues/Anomalies:**
  - generic GEMM cannot represent the required FP32 output contract;
  - the full SDK/database suite retains the unrelated ISSUE-019 failure.
- **Remediation/Verification Code Actions Taken:**
  - called `torch.ops.vllm.optimus_matmul_fp32` after importing the pinned
    registration module;
  - asserted FP32 output shape and dtype before timing;
  - separated persisted data from generic GEMM;
  - required exact structural keys and rejected conflicting rows;
  - verified `74` invocations, `74` physical keys, and zero deduplication.
- **Verdict:** FP32 ROUTER SLICE ACCEPTED FOR CONTINUED DEVELOPMENT. B300
  measurement and the remaining provider operations are still required.

### Review 14 — QKV norm/RoPE provider-data vertical slice

- **Target Component/Phase:** Phase 5–6 Full-MFA and SWA QKV preprocessing
  population, exact pinned-vLLM invocation, persisted rows, loader, and query.
- **Reviewer Agent Identity:** Codex lead execution agent, 2026-08-15.
- **Inspected Artifacts:**
  - pinned `Step4ProAttention.forward` and `_tail_rope`;
  - pinned `Step3p5Attention.forward` and
    `fused_qknorm_rope_forward_impl`;
  - pinned `Step4ProSlidingAttention._prepare_value_for_attention`;
  - `collector/vllm/collect_step4_provider.py`;
  - QKV perf filename, loader, query, tests, and regression logs.
- **Identified Issues/Anomalies:**
  - initial Full-MFA implementation timed K only and omitted Q tail RoPE;
  - full SDK/database regression retains unrelated ISSUE-019.
- **Remediation/Verification Code Actions Taken:**
  - added a RED two-output runtime-probe contract;
  - executed pinned `_tail_rope` for 64 Q heads and for the normalized K head;
  - kept SWA on fused Q/K norm+RoPE plus exact Step4Pro V RMSNorm;
  - required exact structural keys and rejected conflicting rows;
  - verified `148` invocations, `148` physical keys, and zero deduplication;
  - passed `155` focused and `352` Collector tests.
- **Verdict:** QKV NORM/ROPE SLICE ACCEPTED FOR CONTINUED DEVELOPMENT. B300
  runtime measurement, provider attention, Optimus MoE, and DeepEP HT remain
  required.

### Review B300-1 — Authorized quant overlay and DeepGEMM execution

- **Target Component/Phase:** Optimus JIT activation quant, packed UE8M0
  scales, Optimus DeepGEMM grouped GEMMs, and unpermute.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - owner authorization and updated harness/design;
  - runtime overlay diff and source hashes;
  - `s4p-ovl-0815-145936` and `s4p-e8m-0815-151740` evidence;
  - pinned `deep_gemm_utils.py` and available branch history.
- **Identified Issues/Anomalies:** FP32 activation scales were corrected to
  packed UE8M0. Both expert GEMMs then ran. Pinned `ep_gather` chooses an
  invalid Triton block for latent sizes 896 and 3584.
- **Remediation/Verification Code Actions Taken:** Applied only the authorized
  quant overlay, enabled the matching E8M0 oracle, retained all provider
  identities, and verified zero-resource cleanup.
- **Verdict:** QUANT OVERLAY AND OPTIMUS DEEPGEMM APPROVED.
  ISSUE-B300-001 requires owner authorization.

### Review B300-2 — One-GPU final smoke

- **Target Component/Phase:** One-GPU health, request correctness, concurrency,
  FA4, Optimus MoE/DeepGEMM, and cleanup.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - `single_s4p-ok-0815-163232` host and remote evidence;
  - source identity and all runtime overlay diffs/hashes;
  - exact cleanup queries.
- **Identified Issues/Anomalies:** Weak Optimus Triton driver signatures caused
  batch-shape cache reuse and were corrected with the package's official
  strict-signature switch. A global Triton fallback gate incorrectly included
  SWA and was scoped to Full-MFA actual-forward evidence.
- **Remediation/Verification Code Actions Taken:** Ran a fresh no-profiler
  smoke, validated token IDs and usage for one and four concurrent requests,
  checked FA4 and MoE markers, parsed numeric metrics, and verified zero
  resources.
- **Verdict:** ONE-GPU SMOKE APPROVED. Proceed to two-node DP16/EP16 DeepEP.

### Review B300-3 — Runtime-deviation necessity and impact boundary

- **Target Component/Phase:** Optimus quant overlay, contiguous/masked gather
  corrections, strict driver signatures, FA4/import compatibility, and
  provider fallback validation.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - pinned commit `607d1641ee3fec43653fca510d717725828890c2`;
  - later branch delta `9bfd9a610e`;
  - native quant, ABI mismatch, Triton failure, illegal-memory, and final
    one-GPU evidence;
  - overlay patches and before/after SHA256 files;
  - pinned `deep_gemm_utils.py` and isolated
    `deep_gemm_ep_gather_masked.py`.
- **Identified Issues/Anomalies:**
  - the gather fixes are mathematically tiling-only, but the JIT quant overlay
    is not bitwise-equivalent to the broken native path;
  - prior wording could be read as a model-quality or training-convergence
    guarantee;
  - import/cache/test-control changes were not clearly separated from kernel
    changes.
- **Remediation/Verification Code Actions Taken:**
  - reclassified every deviation by necessity and semantic impact;
  - documented the proper long-term native Optimus rebuild alternative;
  - retained exact provider, model shape, routing, expert GEMM, attention, and
    DeepEP boundaries;
  - added a hard prohibition on quality/training-equivalence claims from
    dummy-weight evidence.
- **Verdict:** APPROVED FOR PINNED B300 PROVIDER/PERFORMANCE TESTING.
  `ep_gather` arithmetic is preserved. Optimus JIT quant remains an explicit
  inference-numerics deviation and requires separate quality validation before
  any production or training conclusion.

### Review 15 — Provider Attention and peak physical KV

- **Target Component/Phase:** Phase 5–6 Full-MFA/SWA Attention population,
  pinned provider invocation, consumer lookup, physical cache layout, and
  simulator memory/OOM capacity.
- **Reviewer Agent Identity:** Codex lead execution agent, 2026-08-15.
  Independent merge-readiness reviewer roles were unavailable in this
  environment; this is a verified sub-step checkpoint, not final merge
  approval.
- **Inspected Artifacts:**
  - pinned hybrid allocator, `_reshape_attention_kv_cache`,
    `get_swa_materialized_block_range`, and SWA release tests;
  - Collector case generator, provider benchmark, registry, and model YAML;
  - AIC Attention operations, CSV loader, model config, base memory backend,
    and native KV estimator;
  - RED/GREEN, Collector, broad SDK, Ruff, and whitespace logs.
- **Identified Issues/Anomalies:**
  - initial cache tensors ignored padded physical pages and K/V alias stride;
  - point-in-time SWA residency is non-monotonic and cannot be inverted
    directly for OOM capacity;
  - page-granularity plateaus were initially mistaken for permanent cache
    saturation;
  - broad SDK retains the unrelated ISSUE-019 failure.
- **Remediation/Verification Code Actions Taken:**
  - carried `NHD`, physical page bytes, and exact block strides through model,
    Collector, persisted key, and consumer;
  - built runtime cache views through pinned `_reshape_attention_kv_cache`;
  - separated logical, resident allocated, and peak allocated KV;
  - changed OOM/capacity consumers to the monotonic peak curve;
  - bounded nonlinear inversion by model context length;
  - verified 50 context and 149 generation cases with zero duplicates;
  - passed 260 focused tests and 358 Collector tests.
- **Verdict:** ATTENTION STATIC SLICE ACCEPTED FOR B300 RUNTIME VALIDATION.
  No B300 latency or power result is claimed yet. ISSUE-022 remains a WATCH
  item and must be closed before formal prefill simulation.

### Review B300-4 — Two-node environment and coordinated-DP role contract

- **Target Component/Phase:** Two-node holder readiness, distributed
  environment restoration, RDMA/topology allocation, head/API role split, and
  DeepEP HT launch gate.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - attempts `s4p-2d-0815-172744`, `s4p-2e-0815-174618`, and
    `s4p-2f-0815-181337`;
  - pinned `vllm/v1/engine/utils.py`, CLI `serve.py`, and multi-node benchmark
    fixtures;
  - wrapper RED/GREEN tests, shell syntax, and cleanup logs.
- **Identified Issues/Anomalies:**
  - a Running Replica can precede holder env-file readiness;
  - `brainctl exec` needs explicit holder-env restoration;
  - non-head coordinated-DP nodes must be headless;
  - pinned CLI default `api_server_count=8` is illegal in headless mode.
- **Remediation/Verification Code Actions Taken:**
  - added bounded env-file readiness polling;
  - live-verified ranks `0/1`, start ranks `0/8`, shared master address, and
    eight local processes per node;
  - added `--headless --api-server-count 0` for non-head nodes;
  - preserved exact NCCL/NVSHMEM values and zero-resource cleanup;
  - passed the final `17`-test static E2E suite.
- **Verdict:** ENVIRONMENT CONTRACT APPROVED; FINAL ROLE FIX STATICALLY
  APPROVED. DEEPEP HT LIVE SIGN-OFF IS NOT GRANTED. A fourth B300 launch is
  withheld until the owner explicitly reopens the three-attempt gate.

### Review B300-5 — Fourth run NCCL/RDMA blocker

- **Target Component/Phase:** Authorized fourth two-node DP16/EP16 startup,
  coordinated-DP role split, NCCL initialization, and DeepEP HT readiness.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - `two_node_s4p-2g-0815-185302`;
  - rank-0 full `vllm_server.log`;
  - holder environments, Gloo connectivity, cleanup logs;
  - later branch B300 NVSHMEM bootstrap delta.
- **Identified Issues/Anomalies:**
  - head/worker roles and all 16 Gloo ranks are correct;
  - platform supplied `NCCL_IB_GID_INDEX=5` but `NCCL_IB_HCA=''`;
  - pinned runtime consequently skipped NVSHMEM HCA inheritance;
  - PyNccl first all-reduce failed before DeepEP HT manager construction.
- **Remediation/Verification Code Actions Taken:**
  - preserved the failure without socket/backend fallback;
  - rejected hardcoded HCA, `NCCL_IB_DISABLE=1`, and non-DeepEP substitutes;
  - verified exact zero-resource cleanup;
  - documented the requirement for the full B300
    host-network/RDMA/shared-SHM launcher contract.
- **Verdict:** HEADLESS/API ROLE FIX LIVE-VERIFIED. DEEPEP HT REMAINS
  BLOCKED BY THE LAUNCH ENVIRONMENT. No further run is justified until a
  non-empty platform-injected HCA is proven.

### Review B300-6 — Full-RDMA NCCL pass and DeepEP Buffer blocker

- **Target Component/Phase:** Full-RDMA launch, CUDA compatibility path,
  16-rank NCCL all-reduce, DeepEP manager construction, and Buffer sync.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - `nccl_preflight_s4p-nccl-0815-195109`;
  - `nccl_preflight_s4p-nccl2-0815-195936`;
  - `two_node_s4p-2h-0815-202202`;
  - pinned and later-branch B300 DeepEP probe scripts;
  - current brainctl/rlaunch help.
- **Identified Issues/Anomalies:**
  - worker-init's CUDA 12.8 compat path caused error 803 with torch cu129;
  - corrected NCCL completed on all 16 ranks;
  - DeepEP HT manager selection succeeded;
  - Buffer sync failed before dispatch/combine;
  - current launcher lacks `--share-host-shm=True`.
- **Remediation/Verification Code Actions Taken:**
  - restored the pinned CUDA 13 compatibility path;
  - added exact numeric NCCL preflight and hard model-run gate;
  - added host network and both RDMA resources to the full runner;
  - rejected privileged/volume/backend fallbacks;
  - verified zero-resource cleanup after both runs.
- **Verdict:** NCCL/RDMA CONTRACT APPROVED. DEEPEP HT DISPATCH/COMBINE IS
  NOT APPROVED. A verified shared-host-SHM/explicit NVSHMEM bootstrap launcher
  is required before further full-model execution.

### Review B300-7 — Local DeepEP construction and teardown discriminator

- **Target Component/Phase:** One-node, eight-rank DeepEP Buffer construction,
  explicit destroy, and the decision to stop EP16/EP32 retries.
- **Reviewer Agent Identity:** Codex task execution agent, session dated
  2026-08-15.
- **Inspected Artifacts:**
  - `tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py`;
  - `/data/ycfeng/tmp/step4_aic_deepep_local_b300_20260815/s4p-deepep-local-0815-2046/`;
  - EP16 evidence at `smoke_ep16_s4p-aic-deepep-16-0815-194903` and
    `smoke_ep16_s4p-aic-deepep-16-0815-195827`;
  - pinned/later DeepEP launch contract and available brainctl/rlaunch
    interface.
- **Identified Issues/Anomalies:**
  - local `num_rdma_bytes=0` Buffer construction passed on all `8/8` ranks;
  - local explicit destroy failed on `8/8` ranks at `runtime.cu:29` with CUDA
    `999`;
  - EP16 fails earlier in constructor `runtime.sync`;
  - the available launcher still lacks shared host SHM.
- **Remediation/Verification Code Actions Taken:**
  - replaced the interrupted TTY attempt with one clean non-TTY run;
  - retained exact pinned vLLM helper and runtime identities;
  - separated construction markers from destroy errors;
  - verified py_compile, Ruff, formatting, and whitespace;
  - verified final RJob/Replica counts `0/0`;
  - prohibited further EP16/EP32 retries and all communication fallbacks.
- **Verdict:** LOCAL BUFFER CONSTRUCTION APPROVED AS A DISCRIMINATOR ONLY.
  OVERALL DEEPEP RUNTIME AND TEARDOWN ARE NOT APPROVED. PHASE 6 REMAINS
  BLOCKED UNTIL SHARED-HOST-SHM OR A DOCUMENTED VENDOR/RUNTIME FIX EXISTS.

### Review B300-8 — Supplied brainctl and legacy DeepEP retry

- **Target Component/Phase:** Supplied brainctl identity, supplemental RJob
  guidance, legacy Process/Worker launch, explicit NVSHMEM initialization, and
  DeepEP Buffer construction.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - supplied and installed brainctl binaries;
  - `brainctl-rjob.md`;
  - legacy predict-only evidence;
  - `legacy_deepep_s4p-legdep3-0815-220757`;
  - later-branch standalone probe source.
- **Identified Issues/Anomalies:**
  - supplied brainctl is byte-identical to the installed binary;
  - docs expose a legacy launcher but no shared-host-SHM flag;
  - NCCL and explicit NVSHMEM both pass on 16 ranks;
  - DeepEP observes a mismatched NVSHMEM PE count;
  - `/dev/shm` capacity and memlock are not limiting.
- **Remediation/Verification Code Actions Taken:**
  - used the documented legacy launcher;
  - preserved full RDMA and host networking;
  - ran the later probe's explicit NVSHMEM bootstrap;
  - stopped before full-model execution when the minimal Buffer gate failed.
- **Verdict:** LEGACY LAUNCHER AND EXPLICIT NVSHMEM BOOTSTRAP ARE
  INSUFFICIENT. SHARED-HOST-SHM IS NOT A PROVEN ROOT CAUSE. A VALIDATED
  DEEPEP/NVSHMEM RUNTIME-INTEGRATION CONTRACT IS REQUIRED.

### Review B300-9 — Grouped `wo_a` and FP32 router dataset closure

- **Target Component/Phase:** Phase 6 B300 provider-core measurement and
  canonical database archival for grouped `wo_a` and FP32 router.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, session
  dated 2026-08-15.
- **Inspected Artifacts:**
  - `collector/vllm/collect_step4_provider.py`;
  - `tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh`;
  - `tests/performance/step4_pro_latest/remote_b300_attention_collection.sh`;
  - `tests/performance/step4_pro_latest/validate_b300_provider_core_rows.py`;
  - smoke evidence `smoke_s4p-aic-core-gr-0815-2343`;
  - full evidence `full_s4p-aic-core-grf-0815-2350`;
  - canonical grouped/router Parquet files.
- **Identified Issues/Anomalies:**
  - the original all-provider smoke coupled these valid families to the
    unrelated SWA QKV image annotation defect;
  - the validator originally required all three provider files and could not
    express an explicit measured-family subset;
  - no data-quality, runtime-identity, duplicate-key, or consumer mismatch was
    found in the completed grouped/router datasets.
- **Remediation/Verification Code Actions Taken:**
  - added a fail-fast `grouped_router` collection slice under RED/GREEN TDD;
  - added explicit selected-family validation without generating a QKV
    placeholder or fallback;
  - passed B300 smoke `2/2`, full collection `148/148`, Collector errors `0`;
  - passed canonical exact-consumer validation `148/148`, all silicon, with
    maximum absolute/relative error `0.0 ms` / `0.0%`;
  - verified source/provider call traces and exact RJob/Replica cleanup `0/0`;
  - passed the shared runner regression `15/15`, Ruff, format, shell syntax,
    and whitespace checks.
- **Verdict:** APPROVED. GROUPED `wo_a` AND FP32 ROUTER ARE COMPLETE FOR
  B300/vLLM 0.19.0. QKV AND DEEPEP REMAIN SEPARATE, EXPLICIT GAPS.

### Review B300-10 — Exact later DeepEP fix and launcher contract

- **Target Component/Phase:** Direct-child commit `9bfd9a610e`, its two
  DeepEP/NVSHMEM launch scripts, and the controller launcher API needed for the
  minimal two-node probe.
- **Reviewer Agent Identity:** Codex external B300 execution session,
  2026-08-15.
- **Inspected Artifacts:**
  - exact commit and parent metadata;
  - `rjob-step4pro-2node.sh`;
  - `rjob-step4pro-deepep-probe.sh`;
  - installed brainctl version and RJob/legacy help surfaces;
  - `/data/ycfeng/tmp/step4_deepep_launcher_surface_20260815/`.
- **Identified Issues/Anomalies:**
  - the commit scope is exactly two shell scripts and no model Python/CUDA;
  - all required NVSHMEM/shared-host-SHM markers are present;
  - both scripts pass syntax validation;
  - both installed brainctl launch modes reject `--share-host-shm`;
  - the standalone `rjob` client and the author's hard-coded path are absent.
- **Remediation/Verification Code Actions Taken:**
  - moved the checkout to exact `9bfd9a610e`;
  - verified parent, changed-file scope, required markers, and `bash -n`;
  - ran non-allocating argument-parser checks under `MemoryMax=2G`;
  - created no predict-only request, RJob, Replica, or B300 allocation after
    the required flag was rejected;
  - retained the no-fallback and minimal-probe-first gates.
- **Verdict:** COMMIT SCOPE AND STATIC CONTENT APPROVED. LIVE DEEPEP
  VALIDATION BLOCKED. PROVIDE THE STANDALONE `rjob submit` CLIENT OR A
  DOCUMENTED EQUIVALENT SHARED-HOST-SHM API BEFORE RESUMING.

### Review B300-11 — Grouped/router 65K coverage extension

- **Target Component/Phase:** Phase 6 grouped `wo_a` and FP32 router
  `65536`-token B300 measurement and canonical database update.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, session
  dated 2026-08-15.
- **Inspected Artifacts:**
  - pinned vLLM checkout identity;
  - B300 predict-only output;
  - full evidence under
    `/data/ycfeng/tmp/step4_provider_65k_b300_20260815/`;
  - source CSVs and canonical Parquet files;
  - exact source and canonical consumer-validation JSON.
- **Identified Issues/Anomalies:**
  - the checkout was still at the bounded DeepEP-only child commit before the
    run and had to be restored to authoritative commit `607d1641ee`;
  - the old canonical files had 74 rows and lacked only the `65536` token key;
  - no provider, runtime, key, latency, or consumer mismatch was found.
- **Remediation/Verification Code Actions Taken:**
  - restored the pinned checkout;
  - passed predict-only with 10 candidate B300 nodes;
  - measured grouped `75/75` and router `75/75`, zero Collector errors;
  - proved key-set delta was exactly one `65536` key and zero removals per
    family;
  - passed `150/150` canonical exact silicon queries with zero error;
  - verified RJob and Replica cleanup `0/0`.
- **Verdict:** APPROVED. GROUPED `wo_a` AND FP32 ROUTER B300 COVERAGE IS
  COMPLETE THROUGH `65536` TOKENS. QKV AND DEEPEP REMAIN EXPLICIT GAPS.

### Review B300-12 — Corrected Attention and MTP-off requirements matrix

- **Target Component/Phase:** Corrected Attention canonicalization, exact
  SILICON coverage, and Phase 7 smoke/full MTP-off simulation.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, session
  dated 2026-08-15.
- **Inspected Artifacts:**
  - corrected B300 Attention evidence
    `full_s4p-aic-attnfix-0815-0347`;
  - canonical context/generation Attention Parquet files;
  - `attention_b300_consumer_validation_2026-08-15.json`;
  - `aic_silicon_coverage_2026-08-15.json`;
  - smoke and full MTP-off requirements JSON;
  - focused unit-test and execution logs under `/data/ycfeng/tmp`.
- **Identified Issues/Anomalies:**
  - initial row-level Attention validation had not covered scheduler-derived
    context/decode boundaries;
  - QKV and DeepEP exact data remain absent;
  - one report helper used stale memory-field names and was corrected without
    touching production outputs.
- **Remediation/Verification Code Actions Taken:**
  - canonicalized `58+167` Attention rows with unique keys and Zstd Parquet;
  - passed `225/225` exact silicon consumer queries with `0.0 ms` error;
  - passed `106` focused tests and Ruff/format on `36` changed Python files;
  - reran `36,420` coverage records with `0` unexpected errors;
  - executed smoke `2+1` and full `72+84` requirements records;
  - preserved expected exit code `2` and null formal latency/B_max rather than
    substituting missing data.
- **Verdict:** ATTENTION DATASET AND MTP-OFF DRIVER EXECUTION APPROVED.
  FORMAL ALL-SILICON LATENCY REMAINS BLOCKED BY ISSUE-031/ISSUE-034.

### Review B300-13 — Full-MFA QKV and post-collection matrix

- **Target Component/Phase:** Phase 6 Full-MFA QKV B300 collection,
  canonicalization, and Phase 7 coverage/simulation refresh.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, session
  dated 2026-08-15.
- **Inspected Artifacts:**
  - `remote_b300_attention_collection.sh` `qkv_full` slice;
  - `test_b300_provider_core_collection_contract.py`;
  - successful B300 smoke/full evidence under
    `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/`;
  - canonical `step4_qkv_norm_rope_perf.parquet`;
  - `qkv_full_b300_consumer_validation_2026-08-15.json`;
  - refreshed coverage and smoke/full MTP-off JSON;
  - focused regression log
    `/data/ycfeng/tmp/step4_final_focused_post_qkvfull_20260815.log`.
- **Identified Issues/Anomalies:**
  - the first `qkv_full` kernel smoke succeeded, but host validation still
    expected unrelated grouped/router files;
  - SWA QKV remains blocked by the independently confirmed CuTeDSL annotation
    defect;
  - DeepEP remains frozen at `0/116`.
- **Remediation/Verification Code Actions Taken:**
  - filtered expected output files by the selected provider-core slice;
  - passed the repeated Full-MFA smoke `1/1` and full collection `75/75`;
  - proved `75/75` canonical exact silicon queries with `0.0 ms` error;
  - reran coverage: `17,060` exact, `15,160` non-silicon, `4,200` missing,
    `0` errors, `5` missing contracts;
  - reran the complete `72+84` MTP-off matrix with expected exit `2`;
  - passed `134` focused tests in `5.95s`.
- **Verdict:** FULL-MFA QKV DATASET APPROVED. SWA QKV `0/75` AND DEEPEP
  `0/116` REMAIN EXPLICIT, NON-SUBSTITUTED BLOCKERS.

### Review B300-14 — Final formatting and fail-fast verification

- **Target Component/Phase:** Phase 8 current blocked-checkpoint archive and
  verification closure.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, session
  dated 2026-08-15.
- **Inspected Artifacts:**
  - Ruff import and format diffs for six contract-test files;
  - post-format focused and contract-test logs;
  - changed-task Python file list;
  - Ruff check/format, shell syntax, and `git diff --check` logs;
  - Phase 6/7 dataset and simulation status.
- **Identified Issues/Anomalies:**
  - the earlier aggregate validation pipeline masked Ruff failures because it
    lacked fail-fast and pipefail handling;
  - the first corrected Ruff rerun found one remaining extra blank line;
  - SWA QKV and DeepEP remain real data blockers and were not altered.
- **Remediation/Verification Code Actions Taken:**
  - applied formatting-only patches and removed the final import-block blank
    line;
  - passed `134/134` focused tests in `5.73s`;
  - passed `31/31` formatted contract tests in `1.97s`;
  - passed Ruff check and format on `43/43` changed task Python files;
  - passed shell syntax on `14/14` scripts and `git diff --check`;
  - launched no DeepEP, Docker, GPU, RJob, Replica, or SWA QKV process.
- **Verdict:** CURRENT BLOCKED CHECKPOINT APPROVED. FORMAL ALL-SILICON
  SIMULATION REMAINS BLOCKED ONLY BY THE RECORDED SWA QKV AND DEEPEP DATA
  GAPS; MTP1 REMAINS DEFERRED.

### Review B300-15 — Provider-core acceptance-contract re-review

- **Target Component/Phase:** Phase 8 provider-core slice admission,
  smoke-to-full evidence binding, exact QKV persisted-key validation, and
  final archive readiness.
- **Reviewer Agent Identity:** Independent Codex reviewer agent
  `01a008bc-413a-7672-a8b5-9768391193d4`, re-review completed 2026-08-16.
- **Inspected Artifacts:**
  - `tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh`;
  - `tests/performance/step4_pro_latest/run_b300_attention_collection.sh`;
  - `tests/performance/step4_pro_latest/remote_b300_attention_collection.sh`;
  - `tests/performance/step4_pro_latest/validate_b300_provider_core_rows.py`;
  - `tests/performance/step4_pro_latest/test_b300_provider_core_collection_contract.py`;
  - `tests/unit/collector/test_step4_pro_latest_provider_cases.py`;
  - canonical QKV validation output and the `48/48` focused re-review suite.
- **Identified Issues/Anomalies:** The reviewer rechecked the three prior
  Important findings: implicit provider-core `all`, missing full-mode
  smoke-JSON/SHA256 binding, and count-only QKV acceptance. All three are
  resolved. No remaining Blocking or Important finding was identified.
- **Remediation/Verification Code Actions Taken:** The provider wrapper now
  requires an explicit slice; host and worker reject `all`; both sides validate
  smoke JSON mode, suite, slice, completed-case count, and SHA256; the row
  validator requires the exact Collector-generated 150-key set and provider
  split `75+75`; shape and dtype negative tests pass. Fresh local evidence is
  `48/48` tests and canonical `150/150` exact consumers.
- **Verdict:** **APPROVE.** No remaining Blocking or Important issue; Phase 8
  may proceed to final archive completion.

### Review B300-16 — Phase 10 proxy simulation and local final audit

- **Target Component/Phase:** Phase 10 explicit DeepEP simulation proxy,
  scheduler-derived Attention closure, full MTP-off matrix, and final
  changed-file/changed-symbol audit.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, local
  review completed 2026-08-16. This is not an independent sub-agent review;
  Review B300-15 remains scoped to the earlier provider-core contract.
- **Inspected Artifacts:**
  - `tests/performance/step4_pro_latest/deepep_proxy.py`;
  - proxy integration in `validate_aic_silicon_coverage.py` and
    `run_mtp_off_requirements.py`;
  - the three proxy-focused unit-test files;
  - final `68+167` Attention Parquet files and exhaustive `394/394`
    scheduler-query evidence;
  - proxy coverage and full-matrix JSON artifacts;
  - AST-based changed-symbol inventory, producer/consumer reference audit,
    final focused/Collector logs, Ruff logs, shell syntax, whitespace, and
    DeepEP-Parquet absence evidence.
- **Identified Issues/Anomalies:**
  - the first artifact-validation command read the coverage JSON with the
    matrix schema; the corrected validator used each artifact's native fields
    and passed without changing code or data;
  - one Phase 10 Python file needed Ruff-only formatting; its focused
    regression passed `4/4` after formatting;
  - exact DeepEP measurement remains unavailable at `0/116`, so the completed
    simulation is necessarily `PROXY`, not DeepEP silicon;
  - no separate independent reviewer covered Phase 10.
- **Remediation/Verification Code Actions Taken:**
  - preserved exact DeepEP as the default fail-fast path and required explicit
    `--deepep-proxy b300_nccl_alltoall`;
  - kept proxy, silicon, analytic, missing, and error accounting separate;
  - measured and admitted only clean B300 Attention endpoint rows, producing
    `235/235` exact consumers and `394/394` scheduler queries;
  - passed `334/334` focused tests and `401/401` Collector tests;
  - passed Ruff check, Ruff format on `138` files, shell syntax on `14/14`
    scripts, `git diff --check`, artifact validation, and explicit
    `step4_deepep_ht_perf.parquet` absence;
  - audited every changed production Python symbol against its Collector
    recipe, runtime invocation, persisted key, or AIC consumer; found no
    orphan helper, unconsumed schema field, no-op deduplication, or unrelated
    orchestration change.
- **Verdict:** **LOCAL APPROVE FOR PHASE 10.** The MTP-off proxy deliverable is
  complete and correctly labeled. Exact DeepEP and MTP1 remain explicit future
  work.

### Review B300-17 — Phase 11 simulation completeness and manual-review records

- **Target Component/Phase:** Phase 11 requirement-by-requirement MTP-off
  simulation audit, execution-status schema, three-repeat run, and normalized
  manual-review artifacts.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, local
  review completed 2026-08-16. No independent reviewer was available or
  claimed for this phase.
- **Inspected Artifacts:**
  - `task_memory/step4pro_v4_external_simulator_requirements.md`;
  - `tests/performance/step4_pro_latest/run_mtp_off_requirements.py`;
  - `tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py`;
  - three independent full-matrix JSON files;
  - combined repeat-audit JSON and `156`-row CSV;
  - the updated MTP-off simulation test report and artifact assertion log.
- **Identified Issues/Anomalies:**
  - the normalized row omitted backend fallback, retry, error/missing counts,
    exception log, and explicit decode engine-step batch fields;
  - the first GREEN rerun used a stale test fixture lacking memory null
    reasons; production results already contained those reasons;
  - whole-model vLLM values, sampled runtime percentiles, actual CUDA
    allocation, live routing, real DeepEP, and MTP1 remain unavailable by the
    recorded scope boundaries.
- **Remediation/Verification Code Actions Taken:**
  - propagated query status into every prefill result and decode candidate;
  - recorded the explicit-proxy-versus-fallback distinction, zero retries,
    structured exceptions, active sequences, and batched tokens;
  - preserved runtime-only fields as `null + reason`;
  - passed the focused `18/18` contract;
  - executed `468` cases across three complete runs, with `156/156`
    identical results and `0.0` maximum spread;
  - passed the artifact requirements audit and reconfirmed that no DeepEP
    Parquet exists.
- **Verdict:** **LOCAL APPROVE FOR PHASE 11 ARTIFACTS.** Final scoped
  regression, staging audit, commit, and push remain required before
  publication.

### Review B300-18 — Final requirement and publication gate

- **Target Component/Phase:** Final Phase 11 review of the approved MTP-off
  simulation scope, normalized evidence packet, and pre-publication
  verification.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, local
  review completed 2026-08-16. No independent sub-agent review is claimed for
  this final documentation/publication pass.
- **Inspected Artifacts:**
  - `task_memory/step4pro_v4_external_simulator_requirements.md`;
  - the Phase 11 full and three-repeat proxy JSON files;
  - the `156 × 87` manual-review CSV;
  - all six canonical B300 Step4 Parquet files;
  - the MTP-off driver, DeepEP proxy, coverage validator, and focused tests;
  - fresh focused, Collector, Ruff, shell, structured-input, artifact-audit,
    and full-simulation comparison logs.
- **Identified Issues/Anomalies:**
  - no uncovered simulation condition remained inside the owner-approved
    MTP-off plus explicit DeepEP-proxy scope;
  - the summary and test report still carried Phase 10 hashes and the older
    `334`-test count;
  - one earlier Ruff command had passed YAML paths to a Python linter and
    produced false syntax findings; this was an invocation error rather than a
    code or YAML defect;
  - the first two publication-audit assertions used earlier CSV field names
    and the Phase-10 inventory count; inspecting the current artifacts showed
    byte-valued KV fields and `52` inventory entries. The corrected audit
    passed without changing production code or results;
  - real DeepEP, native Step4Pro MTP1, live vLLM request metrics/router
    observations, and the vLLM-versus-AIC error table remain intentionally
    deferred or externally owned.
- **Remediation/Verification Code Actions Taken:**
  - refreshed the requirement-status matrix and manual-review evidence without
    changing measured rows or proxy behavior;
  - asserted all `72` prefill and `84` decode identities, three topologies,
    four TPOT budgets, `468` repeated executions, `156/156` identical cases,
    zero fallback/retry/error/missing/exception entries, `709` canonical rows,
    and DeepEP-Parquet absence;
  - re-ran the full proxy matrix and obtained the same SHA256
    `b68008b8a86c0a67b90a51fc9b57c301c7561101a9e3613b405e75c6482dd61a`
    with byte-for-byte equality;
  - passed the requirement/artifact audit, including `52/52` inventory hashes,
    `156 × 87` review rows, and six Parquet files totaling `709` rows;
  - passed `343/343` focused tests in `8.16s` and `401/401` Collector tests
    in `31.29s`;
  - passed Ruff check/format on `45/45` Python files, shell syntax on `14/14`
    scripts, YAML parsing on `2/2` files, and JSON parsing on `3/3` files.
- **Verdict:** **LOCAL APPROVE FOR TASK-SCOPED PUBLICATION.** The approved AIC
  simulation requirement is complete and auditable with explicit `PROXY`
  fidelity. The deferred/external items must not be interpreted as completed
  whole-document vLLM or MTP1 validation.

### Review B300-19 — LF-only review artifact and final staging gate

- **Target Component/Phase:** Phase 11 manual-review CSV serialization,
  refreshed regression evidence, and final task-scoped staging readiness.
- **Reviewer Agent Identity:** Codex AIC-side task execution agent, local
  review completed 2026-08-16. No new independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - `task_memory/step4pro_v4_external_simulator_requirements.md`;
  - `run_mtp_off_requirements.py` and its focused unit contract;
  - the three full-run inputs, repeat JSON, and `156 × 87` review CSV;
  - the MTP-off test report, task progress/issues, and publication inventory;
  - fresh focused, Collector, full-simulation, Ruff, shell, structured-input,
    and whitespace outputs.
- **Identified Issues/Anomalies:**
  - `csv.DictWriter` emitted the Excel dialect's CRLF default, so
    `git diff --cached --check` marked all `157` lines as trailing whitespace;
  - the first RED command used unavailable bare `conda`; the task handbook
    already records the direct environment interpreter and no new environment
    workaround was needed;
  - real DeepEP, native Step4Pro MTP1, live whole-model vLLM metrics, and the
    vLLM-versus-AIC error table remain outside this approved AIC-only closure.
- **Remediation/Verification Code Actions Taken:**
  - added an LF-only regression, observed expected RED `1` failure, set the
    writer `lineterminator="\n"`, and observed GREEN `1/1`;
  - regenerated the CSV from the unchanged three full-run JSON inputs:
    `156 × 87`, LF `157`, CRLF `0`, bare CR `0`;
  - retained the repeat JSON SHA256
    `9d5c296c7c4e95859982cbb986cbda21ef65f9a56d286ebe37cc5a780f7208a1`;
  - passed `344/344` focused tests in `8.14s` and `401/401` Collector tests
    in `31.23s`;
  - reran all `72` prefill and `84` decode conditions and reproduced the
    archived full-result SHA256
    `b68008b8a86c0a67b90a51fc9b57c301c7561101a9e3613b405e75c6482dd61a`;
  - passed Ruff check/format on `45/45` Python files, shell syntax on `14/14`
    scripts, YAML parsing on `2/2`, JSON parsing on `3/3`, and the working-tree
    whitespace check.
  - passed the final requirement/inventory audit: `52/52` hashes, `709`
    canonical rows, `72+84` conditions, `468` repeated executions,
    `156 × 87` review cells, component error
    `6.51925802230835e-09 ms`, and KV error `0.0 GiB`.
- **Verdict:** **LOCAL APPROVE FOR FINAL STAGING.** The approved MTP-off plus
  explicit DeepEP-proxy simulation is complete and the review artifact is
  cleanly serialized. Commit, push, and remote ref verification remain.

### Review B300-20 — Active runtime communication replacement

- **Target Component/Phase:** Phase 12 active single-node/two-node vLLM
  runtime settings and their contract tests.
- **Reviewer Agent Identity:** Codex task execution agent, local review
  completed 2026-08-17. No independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - `tests/e2e/step4_pro_latest/run_b300_single_smoke.sh`;
  - `tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh`;
  - `tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh`;
  - the two focused runtime contract tests;
  - pinned vLLM `parallel.py`, `vllm.py`, `cuda_communicator.py`, and
    `all2all.py`;
  - authoritative requirements and Phase 12 task records;
  - focused pytest, shell syntax, backend audit, and whitespace logs.
- **Identified Issues/Anomalies:**
  - active scripts still hard-coded DeepEP after the runtime was declared
    unusable;
  - `nccl` is not a valid backend literal in the pinned vLLM;
  - relying only on the default/CLI value could allow Step MoE auto-selection
    to choose DeepEP when its package is installed;
  - AgRs uses NCCL-backed all-gather plus reduce-scatter, while the existing
    AIC proxy uses an NCCL `alltoall` curve, so they are not same-backend
    evidence.
- **Remediation/Verification Code Actions Taken:**
  - required `allgather_reducescatter` in both environment and CLI;
  - required sequence parallelism `0`;
  - required the `AgRsAll2AllManager` marker, rejected every
    `DeepEP*All2AllManager` variant, and rejected automatic Step MoE backend
    selection;
  - recorded backend, manager, sequence-parallel state, and marker counts in
    `metrics.env`;
  - removed active explicit NVSHMEM configuration and the `deep_ep`
    version-evidence dependency;
  - retained legacy DeepEP probe scripts unchanged and inactive;
  - observed the expected TDD failures before implementation;
  - passed `11/11` focused contracts, shell syntax on `3/3` scripts, static
    source/backend audit, and `git diff --check`;
  - corrected one final audit-only assertion that still expected the earlier
    exact DeepEP-HT shell line; the current generic rejection contract then
    passed without a runtime-code change.
- **Verdict:** **LOCAL APPROVE FOR CONFIGURATION.** The task's active runtime
  requirement no longer depends on DeepEP. A B300 predict-only plus live
  single/two-node smoke is still required before claiming runtime execution,
  performance, or memory success.

### Review B300-21 — Two-node AgRs live lifecycle gate

- **Target Component/Phase:** Phase 12 two-B300 distributed runtime smoke.
- **Reviewer Agent Identity:** Codex task execution agent, live review
  completed 2026-08-17. No independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - `run_b300_two_node_smoke.sh` and `remote_b300_single_smoke.sh`;
  - two-node evidence root
    `/data/ycfeng/tmp/step4_runtime_backend_live_20260817/two_node_s4p-agrs2-0817-163045`;
  - rank-0 `metrics.env`, `result.env`, provider markers, and server log;
  - rank-1 server tail containing ProcessGroupNCCL/TCPStore failures;
  - exact-name post-stop RJob and Replica queries.
- **Identified Issues/Anomalies:** Rank 0 successfully selected
  `AgRsAll2AllManager` and completed real requests, but its one-node-style
  EXIT handler stopped the TCPStore before the rank-1 headless process
  completed. Rank 1 then reported `Broken pipe` for ranks `8-15` and could
  not write `remote_result_ready`.
- **Remediation/Verification Code Actions Taken:** No runtime code change was
  made. The blocked wrapper was interrupted after evidence capture (exit
  `130`); cleanup verified RJobs `0` and Replicas `0`. The failure was
  recorded as ISSUE-043 with a dedicated coordinated-lifecycle root-fix
  requirement.
- **Verdict:** **BLOCK.** The configuration replacement remains supported by
  static checks and rank-0 live evidence, but the two-node smoke cannot be
  called PASS. Do not retry or patch without explicit owner approval of the
  two-node coordinator design.

### Review B300-22 — Coordinated lifecycle and global marker scope

- **Target Component/Phase:** Phase 12 two-node validation barrier, shutdown
  ordering, distributed AgRs evidence scope, and final live rerun admission.
- **Reviewer Agent Identity:** Codex task execution agent, local/live review
  completed 2026-08-17. No independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - `tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh`;
  - `tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh`;
  - `tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py`;
  - pinned `vllm/distributed/device_communicators/cuda_communicator.py`;
  - evidence roots for `s4p-agrs2-0817-174506` and
    `s4p-agrs2b-0817-175947`;
  - exact RJob event plus exact-name/label cleanup queries.
- **Identified Issues/Anomalies:**
  - the original per-node runner lifecycle stopped rank 0/TCPStore before
    rank 1 completed;
  - the first coordinator revision still treated the globally scoped AgRs
    manager line as a required local rank-1 marker;
  - the final corrected payload could not start because the `16`-GPU request
    exceeded the queue's `6` remaining B300 GPUs.
- **Remediation/Verification Code Actions Taken:**
  - added the validation-ready, shutdown-arm, armed-acknowledgement, and
    concurrent final-shutdown ordering;
  - changed rank-1 acceptance to local backend configuration plus real-batch
    evidence while retaining a job-level AgRs manager requirement;
  - passed final focused contracts `14/14`, active shell syntax `3/3`, Ruff
    check/format, and `git diff --check`;
  - observed `2/2` Running in `78s` on the first coordinator revision, rank-0
    AgRs/DeepEP/automatic markers `1/0/0`, model load `13.507140s`, first
    request `12.979565s`, and four-request wall `12.497070804s`;
  - preserved the exact quota event:
    request `16` B300 GPUs, queue remaining `6`;
  - verified final residual RJob/Replica counts `0/0`.
- **Verdict:** **LOCAL APPROVE; LIVE BLOCKED.** The code now represents the
  approved root design and the local contracts pass. Do not claim a completed
  two-node runtime result or commit/push final publication until the corrected
  payload runs with sufficient quota and passes both-node evidence plus
  coordinated cleanup.

### Review B300-23 — Phase 12 continuation and archive consistency

- **Target Component/Phase:** Phase 12 resumed local verification, task-record
  consistency, and admission gate for the corrected two-node live run.
- **Reviewer Agent Identity:** Codex task execution agent, local review
  completed 2026-08-17. No independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - the three active runtime shell scripts and two focused contract files;
  - Phase 12 plan, progress, issues, review, test report, lessons, future, and
    summary records;
  - evidence under
    `/data/ycfeng/tmp/step4_phase12_resume_20260817/`;
  - prior exact quota event and `0/0` cleanup evidence for
    `s4p-agrs2b-0817-175947`.
- **Identified Issues/Anomalies:**
  - `summary.md` still described the independent EXIT-handler lifecycle as
    unresolved even though the coordinator and global-marker-scope root fixes
    were already implemented and locally verified;
  - its active runtime test count still showed the earlier `11/11` result;
  - no new runtime-code defect was found;
  - complete two-node acceptance remains unavailable because the final
    corrected payload created `0/2` replicas when only `6` of the required
    `16` B300 GPUs remained in quota.
- **Remediation/Verification Code Actions Taken:**
  - changed no runtime code;
  - refreshed focused contracts to `14/14` in `0.04s`;
  - passed shell syntax `3/3`, Ruff check, Ruff format check, and
    `git diff --check`;
  - recorded the job-level meaning of the globally scoped AgRs manager line
    and the four-stage coordinated teardown as vetted lessons;
  - synchronized the summary and test report to distinguish local root-fix
    completion from the external quota blocker.
- **Verdict:** **LOCAL APPROVE; LIVE BLOCKED BY QUOTA.** The current code and
  local contracts are internally consistent. Final publication still requires
  direct evidence of at least `16` available B300 GPUs, one same-shape
  predict-only node-fit check, and one corrected two-node live PASS.

### Review B300-24 — Multi-replica quota admission evidence

- **Target Component/Phase:** Phase 12 quota visibility and the decision
  whether another corrected two-node live run may be queued.
- **Reviewer Agent Identity:** Codex task execution agent, local/platform
  review completed 2026-08-17. No independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - `/data/ycfeng/stepfun-env-handbook/guidence.md`;
  - `/data/ycfeng/stepfun-env-handbook/brainctl-rjob.md`;
  - current `brainctl rjob launch --help`;
  - the prior quota event for `s4p-agrs2b-0817-175947`;
  - fresh replica-2 and replica-8 predict-only outputs under
    `/data/ycfeng/tmp/step4_phase12_resume_20260817/`;
  - exact-name RJob and Replica cleanup queries.
- **Identified Issues/Anomalies:**
  - direct quota reads are RBAC-forbidden for the current identity;
  - predict-only returned the same seven per-worker candidates for replica
    counts `2` and `8`;
  - the prior plan treated same-shape predict-only as if it could prove total
    quota, which the sensitivity control disproved;
  - the latest trustworthy total-quota event remains `6`, below the required
    `16`.
- **Remediation/Verification Code Actions Taken:**
  - submitted no live RJob;
  - confirmed both predict-only probes created RJobs/Replicas `0/0`;
  - changed the task gate to require separate platform or quota-owner evidence
    of at least `16` currently available B300 GPUs;
  - retained predict-only only as the required per-worker node-fit check.
- **Verdict:** **BLOCK.** Another live run would violate the recorded quota
  gate. Resume only after direct quota evidence reaches `>=16`.

### Review B300-25 — Post-concurrency stability and inventory integrity

- **Target Component/Phase:** Phase 12 stable runtime contracts, archive
  hashes, and final local verification before the quota-gated live rerun.
- **Reviewer Agent Identity:** Codex task execution agent, local review
  completed 2026-08-17. No independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - `tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh`;
  - `tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py`;
  - both focused runtime contract files and all three active shell scripts;
  - the Phase 12 inventory in `summary.md`;
  - fresh evidence under
    `/data/ycfeng/tmp/step4_phase12_resume2_20260817/`.
- **Identified Issues/Anomalies:**
  - the remote runner and two-node contract were modified after the prior
    summary inventory was written;
  - the stale summary matched only `14/16` current entries;
  - the summary incorrectly described its 16-row inventory as `17/17`;
  - no new runtime-code defect or open file writer was found.
- **Remediation/Verification Code Actions Taken:**
  - changed no runtime code;
  - verified stable SHA256 values across a five-second window and open writers
    `0`;
  - confirmed the later edits only complete the approved declared validation
    marker path and matching contract;
  - passed focused contracts `14/14` in `0.04s`, shell syntax `3/3`, Ruff
    check/format, and `git diff --check`;
  - refreshed affected hashes and corrected the active-runtime inventory to
    `16/16`.
- **Verdict:** **LOCAL APPROVE; LIVE BLOCKED BY QUOTA.** The current files and
  archive are internally consistent. The next live action still requires
  direct evidence that at least `16` B300 GPUs are available.

### Review B300-26 — Final code/docs publication candidate

- **Target Component/Phase:** Phase 13 complete current-diff review,
  normalized test records, exact staging boundary, and commit/push readiness.
- **Reviewer Agent Identity:** Codex task execution agent, local review
  completed 2026-08-17. The earlier AIC core has recorded independent review;
  no new independent sub-agent review is claimed for the Phase 12 wrapper
  slice.
- **Inspected Artifacts:**
  - all five modified files under `tests/e2e/step4_pro_latest/`;
  - the authoritative external-simulator requirements update;
  - all modified task records;
  - the current AgRs runtime report, final publication report, and historical
    DeepEP/NVSHMEM report;
  - fresh focused, Collector, Ruff, shell-syntax, and whitespace evidence
    under `/data/ycfeng/tmp/step4_final_publication_20260817/`.
- **Identified Issues/Anomalies:**
  - the final publication report still treated the already-resolved lifecycle
    defect as the publication blocker;
  - its focused count was stale at `344/344`;
  - the historical DeepEP report did not clearly say that it had been
    superseded for active runtime use;
  - current code review found no new CRITICAL, HIGH, MEDIUM, or blocking
    lifecycle/backend defect;
  - final corrected two-node live acceptance is still externally blocked:
    request `16`, last trusted B300 remainder `6`, replicas `0/2`.
- **Remediation/Verification Code Actions Taken:**
  - synchronized all three reports to the current AgRs/quota state;
  - retained fail-fast DeepEP rejection and the distinction between active
    AgRs, exact DeepEP AIC identity, and NCCL `alltoall` `PROXY`;
  - passed focused regression `347/347` in `8.21s`;
  - passed Collector regression `401/401` in `31.60s`;
  - passed Ruff check/format on `45/45` Python files and shell syntax on
    `14/14` scripts;
  - corrected one audit-only repeat-JSON schema assumption from `summary` to
    `repeat_audit`, then passed all `70/70` hashes and simulation artifact
    checks without changing implementation or archived results;
  - defined an exact staging allowlist that excludes caches, unrelated tasks,
    H800 data, temporary outputs, `.worktrees/`, and `vllm-step4-pro/`.
- **Verdict:** **LOCAL APPROVE FOR TASK-SCOPED COMMIT/PUSH; LIVE
  BLOCKED_BY_QUOTA.** Publication is allowed by Q26 and must not be presented
  as a completed two-node runtime validation.

### Review B300-27 — Phase 14 lifecycle and evidence hardening

- **Target Component/Phase:** Active B300 single/two-node wrappers, shared
  runtime contract, distributed evidence lifecycle, and quota admission gate.
- **Reviewer Agent Identity:** Codex task execution agent, local review
  completed 2026-08-17. No independent sub-agent review is claimed.
- **Inspected Artifacts:**
  - `tests/e2e/step4_pro_latest/run_b300_single_smoke.sh`;
  - `tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh`;
  - `tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh`;
  - `tests/e2e/step4_pro_latest/b300_runtime_contract.sh`;
  - all five current runtime contract tests;
  - Phase 12/14 plan, notes, issues, design, harness, progress, lessons,
    summary, future, and runtime test report;
  - RED/GREEN and static evidence under
    `/data/ycfeng/tmp/step4_phase14_review_fix_20260817/`.
- **Identified Issues/Anomalies:**
  - the prior four-marker shutdown protocol was more complex than necessary
    and left evidence collection coupled to remote teardown;
  - cleanup could accept a failed control-plane query as an empty inventory;
  - predict-only could drift from live launch arguments;
  - the prior forward marker did not prove asynchronous CUDA completion;
  - the final corrected two-node live run remains blocked before Replica
    creation because current quota evidence is unavailable and the last
    trusted event reported `6` available B300 GPUs versus `16` requested.
- **Remediation/Verification Code Actions Taken:**
  - kept both replicas alive after `remote_validation_ready`, pulled and
    validated both evidence trees, and reduced teardown to one exact RJob
    delete;
  - added query-status-aware cleanup and exact zero-resource checks;
  - reused one launch argument array for predict-only/live, archived
    predict-only SHA256, and required a non-empty candidate list;
  - added disk-backed quota evidence validation for B300/`b300_train_infra`
    and `>=16` GPUs;
  - added source-hash-bounded `MODEL_FORWARD_COMPLETE` after
    `current_platform.synchronize()`;
  - added the `3060s` minimum timeout guard with `3600s` defaults;
  - observed lifecycle/transfer/timeout RED and final `26/26` focused
    contracts in `0.21s`, shell syntax `6/6`, Ruff clean, and
    `git diff --check` clean;
  - audited the complete publication inventory: `76/76` paths matched,
    duplicate/missing/mismatched counts `0/0/0`.
- **Verdict:** **LOCAL APPROVE; LIVE BLOCKED_BY_QUOTA.** The implementation
  addresses the lifecycle and evidence root causes without changing model
  weights, communication algorithm, or training graph. The synchronize
  overlay affects timing, so any future measurements from this smoke must be
  labeled instrumented runtime evidence.
