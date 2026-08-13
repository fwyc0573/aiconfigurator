## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Recorded the initial requirements and workspace checkpoint. |
| 2026-08-13 | Reviewed refreshed source/image/access evidence and recorded the remaining semantic gate. |
| 2026-08-13 | Reviewed pinned-source operation provenance, AIC fidelity gaps, and the approved minimal-extension boundary. |

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
