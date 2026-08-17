## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-08-17 | Created the configuration and static-validation report for replacing active DeepEP runtime settings with AgRs/NCCL communication. |
| 2026-08-17 | Added generic DeepEP-manager rejection, automatic-backend-selection rejection, runtime evidence fields, and final audit details. |
| 2026-08-17 | Added B300 live-validation evidence: one-node PASS and the two-node coordinated-lifecycle blocker with verified cleanup. |
| 2026-08-17 | Added the local coordinator/global-marker-scope fixes, first post-fix live metrics, final quota blocker, and fresh zero-resource cleanup evidence. |
| 2026-08-17 | Refreshed the local Phase 12 contracts, syntax, Ruff, and whitespace evidence after continuation. |
| 2026-08-17 | Added the replica-sensitivity proof that predict-only does not validate total quota and recorded the RBAC-blocked direct quota query. |
| 2026-08-17 | Revalidated the stable post-concurrency files and corrected the active-runtime hash audit from stale `14/16` evidence to `16/16`. |
| 2026-08-17 | Added the final publication regression: `347/347` focused tests, `401/401` Collector tests, Ruff `45/45`, shell syntax `14/14`, and zero blocking current-diff review findings. |

# Test Report: Non-DeepEP Step4-Pro Runtime Backend

**Date:** 2026-08-17
**Environment:** conda `aic-step-design`, Python `3.11.15`, pytest `8.4.2`
**Controller limit:** `systemd-run --user --scope -p MemoryMax=2G`
**GPU execution:** One-node B300 smoke PASS. The two-node lifecycle and
global-marker-scope fixes pass local contracts; the final corrected live
payload is blocked before Replica creation because the 16-GPU request exceeds
the current 6-GPU B300 queue remainder.

## 1. Test Script Information

### Scripts under test

- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/e2e/step4_pro_latest/run_b300_single_smoke.sh`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh`

### Contract tests

- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py`

### Reproducible commands

```bash
cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro

systemd-run --user --scope -p MemoryMax=2G \
  /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  -m pytest -q \
  tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py \
  tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py

systemd-run --user --scope -p MemoryMax=2G bash -lc '
  cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro
  for f in \
    tests/e2e/step4_pro_latest/run_b300_single_smoke.sh \
    tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh \
    tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh
  do
    bash -n "$f"
  done
'

git diff --check
```

Evidence logs:

- `/data/ycfeng/tmp/step4_runtime_backend_20260817/final_focused_pytest.log`
- `/data/ycfeng/tmp/step4_runtime_backend_20260817/final_shell_syntax.log`
- `/data/ycfeng/tmp/step4_runtime_backend_20260817/final_backend_audit.log`
- `/data/ycfeng/tmp/step4_runtime_backend_20260817/final_diff_check.log`
- `/data/ycfeng/tmp/step4_phase12_resume_20260817/focused_contracts.log`
- `/data/ycfeng/tmp/step4_phase12_resume_20260817/shell_syntax.log`
- `/data/ycfeng/tmp/step4_phase12_resume_20260817/ruff_check.log`
- `/data/ycfeng/tmp/step4_phase12_resume_20260817/ruff_format_check.log`
- `/data/ycfeng/tmp/step4_phase12_resume_20260817/git_diff_check.log`
- `/data/ycfeng/tmp/step4_phase12_resume2_20260817/focused_contracts_14.log`
- `/data/ycfeng/tmp/step4_phase12_resume2_20260817/ruff_check.log`
- `/data/ycfeng/tmp/step4_phase12_resume2_20260817/ruff_format.log`
- `/data/ycfeng/tmp/step4_phase12_resume2_20260817/git_diff_check.log`

## 2. Validation Criteria

1. Active runtime backend is exactly
   `allgather_reducescatter`; literal `nccl` is not used.
2. Both the environment and `vllm serve` CLI carry the backend value.
3. `VLLM_ENABLE_SEQUENCE_PARALLEL=0` is enforced.
4. Distributed runtime acceptance requires
   `Using AgRsAll2AllManager all2all manager`.
5. Runtime fails if any `Using DeepEP*All2AllManager` marker appears.
6. Runtime fails if Step MoE emits an automatic
   `VLLM_ALL2ALL_BACKEND` selection marker.
7. Runtime evidence records backend, manager, sequence-parallel state, and
   manager/automatic-selection marker counts.
8. Active scripts contain no `deepep_high_throughput` setting and no explicit
   `NVSHMEM_ENABLE_NIC_PE_MAPPING`.
9. The shared runner does not require the `deep_ep` package merely for
   package-version evidence.
10. All three active shell scripts parse successfully.
11. Repository whitespace validation passes.
12. Both replicas write validation readiness before shutdown is armed.
13. Both replicas acknowledge the shutdown arm before final shutdown requests
    are sent concurrently.
14. Each replica proves explicit AgRs configuration plus a real-batch forward;
    the globally scoped AgRs manager line appears at least once across the job.

## 3. Test Results and Evidence

### Summary

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| Focused pytest | `14/14` passed, `0` failed | `14/14` pass | PASS |
| Active shell syntax | `3/3` passed | `3/3` pass | PASS |
| Active runtime files audited | `3` | `3` | PASS |
| Runtime backend | `allgather_reducescatter` | exact match | PASS |
| Runtime manager | `AgRsAll2AllManager` | exact match | PASS |
| Dispatch/combine source mapping | `all_gatherv` / `reduce_scatterv` | exact match | PASS |
| Sequence parallel | `0` | `0` | PASS |
| Active `deepep_high_throughput` settings | `0` | `0` | PASS |
| Explicit active NVSHMEM settings | `0` | `0` | PASS |
| `deep_ep` package-evidence dependencies | `0` | `0` | PASS |
| DeepEP manager matching | generic `DeepEP*All2AllManager` | generic rejection | PASS |
| Automatic backend selection allowed | `0` | `0` | PASS |
| `git diff --check` findings | `0` | `0` | PASS |

### TDD evidence

- Initial backend contract RED: `3 failed, 8 passed`.
- Active dependency cleanup RED: `2 failed, 9 passed`.
- Controller fail-fast RED: `1 failed`.
- Lifecycle ordering RED: `3 passed, 1 failed`.
- Declared validation-path RED: `2 failed`.
- Intermediate lifecycle GREEN: `12 passed, 0 failed`.
- Final GREEN after global-marker-scope correction:
  `14 passed, 0 failed`.

### Source evidence

The pinned vLLM source maps:

```text
allgather_reducescatter -> AgRsAll2AllManager
dispatch -> all_gatherv
combine -> reduce_scatterv
```

The CUDA path uses PyNccl/NCCL collectives. This is not a direct backend named
`nccl`, and it is not the same algorithm as the AIC
`b300_nccl_alltoall` proxy.

### Resolved validation-command issue

The first final backend-audit assertion looked for the earlier exact
`DeepEPHTAll2AllManager` shell line. The runtime gate had already been
strengthened to reject every `DeepEP*All2AllManager` variant through one
generic regular expression, so that audit assertion failed before checking
the actual contract. The audit was corrected to the current generic contract
and then passed. No runtime code or test expectation changed for this retry.

### B300 live validation

The exact live two-node wrapper was recorded at:

```bash
RJOB_NAME=s4p-agrs2-0817-163045 \
ARTIFACT_ROOT=/data/ycfeng/tmp/step4_runtime_backend_live_20260817/two_node_s4p-agrs2-0817-163045 \
CONTROL_MEMORY_MAX=2G \
NCCL_PREFLIGHT_EVIDENCE=/data/ycfeng/tmp/b300_step4_smoke_20260814/nccl_preflight_s4p-nccl2-0815-195936/launch.log \
bash tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh
```

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| One-node host and remote smoke | `ONE_GPU_HOST_WRAPPER=PASS`, `ONE_GPU_SMOKE=PASS` | both PASS | PASS |
| One-node scheduling | `16s` | worker starts | PASS |
| One-node model loading | `8.819692s`, `3.16 GiB` reported model memory | model loads | PASS |
| One-node GPU memory | `644 MiB` before, `247628 MiB` after load | model resides on B300 | PASS |
| One-node first request | `8` prompt + `4` completion tokens in `13.737685s` | completion succeeds | PASS |
| One-node concurrent requests | wall `0.474799647s`, per-request `0.464150-0.467964s` | four responses | PASS |
| Two-node scheduling | `16s`, `2/2` replicas Running | both nodes start | PASS |
| Two-node rank 0 runtime | `AgRsAll2AllManager=1`, `DeepEP=0`, auto selection `0` | exact markers | PASS |
| Two-node rank 0 request | `8` prompt + `4` completion tokens in `10.762580s`; four-request wall `10.90261888s` | completion succeeds | PASS |
| Two-node rank 0 model loading | `12.957171s`, GPU after load `257103 MiB` | model loads | PASS |
| Two-node rank 1 completion | no `remote_result_ready` | all nodes complete | FAIL |

The two-node job is not a passing runtime validation. Rank 0 wrote
`remote_result_ready` at `2026-08-17 16:42:37+08:00`; rank 1 subsequently
reported `ProcessGroupNCCL`/`TCPStore` `Broken pipe` at
`2026-08-17 16:53:40+08:00` for ranks `8-15` and never wrote its result
marker. The shared runner's EXIT handler stops its local server before it
writes the result marker and before the evidence-hold sleep. Running that
runner independently on both distributed nodes therefore stops rank 0's
TCPStore while the rank 1 headless process still depends on it.

This is a two-node wrapper lifecycle defect, not a DeepEP, NVSHMEM, backend
selection, or AgRs manager-selection failure. The controller was interrupted
after the root cause was captured; exact-name queries verified residual RJobs
`0` and Replicas `0`.

### Coordinated lifecycle implementation

The root fix uses four explicit marker paths:

```text
remote_validation_ready
coordinated_shutdown_armed
remote_shutdown_armed
coordinated_shutdown
```

The host waits for validation readiness from both replicas, arms both,
waits for both acknowledgements, and then sends the final shutdown requests
concurrently. Neither remote runner may stop vLLM before the final release.

The first coordinator live run used:

```bash
RJOB_NAME=s4p-agrs2-0817-174506 \
ARTIFACT_ROOT=/data/ycfeng/tmp/step4_phase12_barrier_20260817/two_node_s4p-agrs2-0817-174506 \
CONTROL_MEMORY_MAX=2G \
NCCL_PREFLIGHT_EVIDENCE=/data/ycfeng/tmp/b300_step4_smoke_20260814/nccl_preflight_s4p-nccl2-0815-195936/launch.log \
bash tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh
```

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| Replica scheduling | `78s`, `2/2` Running | `2/2` Running | PASS |
| Rank-0 model loading | `13.507140s` | model loads | PASS |
| Rank-0 health readiness | `229s` | API becomes ready | PASS |
| Rank-0 GPU after load | `257103 MiB` | model resides on B300 | PASS |
| Rank-0 first request | `12.979565s`, prompt `8`, completion `4` | completion succeeds | PASS |
| Rank-0 concurrent requests | wall `12.497070804s` | four responses | PASS |
| Rank-0 manager markers | AgRs `1`, DeepEP `0`, automatic `0` | `>=1 / 0 / 0` | PASS |
| Rank-1 validation readiness | absent | present | FAIL |
| Post-coordinator `Broken pipe` | `0` matches | `0` | PASS |
| Cleanup | RJob `0`, Replica `0` | `0/0` | PASS |

The remaining failure was in the evidence contract, not the communication
runtime. Pinned `cuda_communicator.py` emits the manager line through
`logger.info_once(..., scope="global")`; rank 1 is not required to emit its
own copy. The corrected contract now requires local
`allgather_reducescatter` configuration and real-batch evidence on every
replica, while the host requires at least one AgRs manager line across the
whole job.

The remote commands in this run were terminated by exact cleanup after the
host could not complete the invalid rank-1 evidence gate. Their exit `137`
must not be classified as host or GPU OOM.

### Final corrected rerun and quota evidence

The final payload was submitted as:

```bash
RJOB_NAME=s4p-agrs2b-0817-175947 \
ARTIFACT_ROOT=/data/ycfeng/tmp/step4_phase12_barrier_20260817/two_node_s4p-agrs2b-0817-175947 \
CONTROL_MEMORY_MAX=2G \
NCCL_PREFLIGHT_EVIDENCE=/data/ycfeng/tmp/b300_step4_smoke_20260814/nccl_preflight_s4p-nccl2-0815-195936/launch.log \
bash tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh
```

The exact event query was:

```bash
sudo -n systemd-run --scope -p MemoryMax=2G --expand-environment=no \
  timeout --signal=TERM --kill-after=5s 60s \
  /kubebrain/brainctl get events -n shai-core \
  --field-selector involvedObject.name=s4p-agrs2b-0817-175947 \
  --limit=100 \
  -o custom-columns=TIME:.lastTimestamp,TYPE:.type,REASON:.reason,OBJECT:.involvedObject.name,MESSAGE:.message \
  --no-headers
```

Observed event:

```text
2026-08-17T10:03:57Z Normal QueueWaiting s4p-agrs2b-0817-175947
Waiting in queue "b300-train-infra-default". Insufficient GPU quota.
Request: task "worker" 8 GPU/replica x 2 replicas = 16 GPU (B300).
Queue remaining: B300=6.
```

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| RJob creation | created at `17:59:49+08:00` | created | PASS |
| Replica creation | `0/2` for about `308s` | `2/2` | BLOCKED |
| Requested quota | `16` B300 GPUs | `16` available | BLOCKED |
| Queue remainder | `6` B300 GPUs | `>=16` | BLOCKED |
| Runtime metrics | none | full two-node evidence | NOT RUN |
| Artifact cleanup | RJob `0`, Replica `0` | `0/0` | PASS |
| Fresh post-stop query | RJob `0`, Replica `0` | `0/0` | PASS |

Evidence:

- `/data/ycfeng/tmp/step4_phase12_barrier_20260817/two_node_s4p-agrs2-0817-174506/`
- `/data/ycfeng/tmp/step4_phase12_barrier_20260817/two_node_s4p-agrs2b-0817-175947/`

### Fresh continuation verification

No runtime code was changed during this continuation. The current working
tree was revalidated under the same `MemoryMax=2G` controller limit:

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| Focused runtime contracts | `14/14` in `0.04s` | `14/14` | PASS |
| Active shell syntax | `3/3` | `3/3` | PASS |
| Ruff check | `0` findings | `0` | PASS |
| Ruff format check | `2/2` already formatted | `2/2` | PASS |
| `git diff --check` | `0` findings | `0` | PASS |

The refreshed evidence is stored under:

```text
/data/ycfeng/tmp/step4_phase12_resume_20260817/
```

### Quota admission diagnostic

The current identity cannot directly read
`quotagroups.stepmind.com/b300_train_infra`; the API returns `Forbidden`.
Predict-only was therefore tested for replica sensitivity without creating a
live RJob:

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| Replica-2 predict-only | exit `0`; `7` nodes with `8` B300 each | per-worker fit | PASS |
| Replica-8 predict-only control | exit `0`; same `7` nodes | result changes or rejects if total demand is checked | FAIL |
| Direct quota read | RBAC `Forbidden` | current total quota visible | BLOCKED |
| Probe-created RJobs/Replicas | `0/0` for both names | `0/0` | PASS |
| Live reruns submitted | `0` | `0` while quota is unproven | PASS |

The control would require `64` GPUs if replica count were included, yet it
returned the same candidate list as replica `2`. Predict-only is therefore a
per-worker node-fit check, not total-quota evidence. The last trustworthy
quota value remains `6`, from the final corrected job event, so no new live
run was submitted.

### Post-concurrency stability and inventory audit

The previous summary inventory was written before two later edits completed.
The current files were therefore checked for stability and revalidated before
their hashes were accepted:

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| Stability window | two-node contract SHA256 unchanged for `5s` | unchanged | PASS |
| Open writers | `0` for remote runner and contract | `0` | PASS |
| Focused runtime contracts | `14/14` in `0.04s` | `14/14` | PASS |
| Two-node contract alone | `6/6` in `0.05s` | `6/6` | PASS |
| Active shell syntax | `3/3` | `3/3` | PASS |
| Ruff check | `0` findings | `0` | PASS |
| Ruff format | `2/2` already formatted | `2/2` | PASS |
| `git diff --check` | `0` findings | `0` | PASS |
| Stale summary inventory | `14/16` matched, `2/16` mismatched | `16/16` | FAIL |
| Refreshed summary inventory | `16/16` matched, missing/mismatched `0/0` | `16/16` | PASS |

Current changed-file identities:

```text
remote_b300_single_smoke.sh
7cc139aab85d3db29e53e2b750fd1d302d82d00a84a706368524e041d5d13e15

test_b300_two_node_smoke_contract.py
9b749a806fb5f60e78dd7b4a32967bfaaee14d25b39452ba776579f112f166c6
```

The later edits only complete the approved use of the declared
`REMOTE_VALIDATION_READY_FILE` path and its matching source-level contract.
No runtime code was changed during this continuation.

### Final publication regression

The complete Step4-Pro-Latest focused suite and full Collector suite were
rerun after the coordinated lifecycle implementation stabilized:

| Check | Actual result | Acceptance | Status |
| --- | ---: | ---: | --- |
| Focused Step4-Pro-Latest regression | `347/347` in `8.21s` | zero failures | PASS |
| Full Collector regression | `401/401` in `31.60s` | zero failures | PASS |
| Ruff check/format | `45/45` Python files | zero findings / all formatted | PASS |
| Shell syntax | `14/14` scripts | all parse | PASS |
| Current code/doc diff review | blocking findings `0` | `0` | PASS |
| Corrected two-node live acceptance | replicas `0/2`; requested `16`, remainder `6` | both replicas complete | BLOCKED_BY_QUOTA |

Evidence:

```text
/data/ycfeng/tmp/step4_final_publication_20260817_rerun/focused_pytest.log
/data/ycfeng/tmp/step4_final_publication_20260817_rerun/collector_pytest.log
/data/ycfeng/tmp/step4_final_publication_20260817_rerun/static_and_artifact_checks.log
```

**Final status:** configuration, source contracts, lifecycle coordination,
and global-marker-scope validation are locally PASS. Complete two-node runtime
acceptance remains BLOCKED until an authorized source confirms at least `16`
available B300 GPUs, same-shape predict-only passes for per-worker fit, and the
corrected payload completes both-node validation and coordinated cleanup.
