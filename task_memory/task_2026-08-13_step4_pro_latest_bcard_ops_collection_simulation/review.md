## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Recorded the initial requirements and workspace checkpoint. |
| 2026-08-13 | Reviewed refreshed source/image/access evidence and recorded the remaining semantic gate. |
| 2026-08-13 | Reviewed pinned-source operation provenance, AIC fidelity gaps, and the approved minimal-extension boundary. |
| 2026-08-13 | Reviewed the branch checkpoint and failed focused baseline evidence. |
| 2026-08-13 | Reviewed and accepted the root-cause baseline repair with 899/899 passing tests. |
| 2026-08-14 | Reviewed the pinned-vLLM MTP1 boundary and fresh baseline checkpoint evidence. |

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
