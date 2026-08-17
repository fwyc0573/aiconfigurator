## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-17 | Created the final independent-review and clean-commit verification report. |
| 2026-08-17 | Included Phase 12, fresh regression/artifact evidence, B300 live results, and the blocking two-node lifecycle finding. |
| 2026-08-17 | Replaced the obsolete lifecycle publication block with the completed local root-fix review, fresh `347/347` and `401/401` verification, and the remaining external B300 quota blocker. |

# Test Report: Final Review and Publication Gate

**Date:** 2026-08-17
**Base commit:** `c422e0ae6a44d382b2c163697c3b0dd70759cdf4`
**Candidate branch:** `task/step4-pro-latest-b300`
**Overall result:** **APPROVED FOR TASK-SCOPED COMMIT/PUSH; AIC MTP-off and
local Phase 12 contracts PASS; final two-node live acceptance remains
BLOCKED_BY_QUOTA**

## 1. Test Script Information

### Environment

- Conda environment: `aic-step-design`
- Python: `3.11.15`
- pytest: `8.4.2`
- Ruff: `0.14.1`
- Verification checkout:
  `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro`
- Controller limits: `timeout 60s`, `MemoryMax=2G`
- Temporary/log storage: `/data/ycfeng/tmp`

### Focused regression command

```bash
timeout 60s systemd-run --user --scope -p MemoryMax=2G \
  env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. MPLBACKEND=Agg \
  /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  -m pytest -q \
  tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py \
  tests/e2e/step4_pro_latest/test_b300_two_node_deepep_legacy_probe_contract.py \
  tests/e2e/step4_pro_latest/test_b300_two_node_nccl_preflight_contract.py \
  tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py \
  tests/e2e/step4_pro_latest/test_generate_step4pro_dummy_configs.py \
  tests/e2e/step4_pro_latest/test_run_b300_source_probe_static.py \
  tests/performance/step4_pro_latest/test_b300_attention_collection_contract.py \
  tests/performance/step4_pro_latest/test_b300_deepep_ht_collection_contract.py \
  tests/performance/step4_pro_latest/test_b300_optimus_moe_collection_contract.py \
  tests/performance/step4_pro_latest/test_b300_provider_core_collection_contract.py \
  tests/unit/performance/test_step4_pro_latest_deepep_proxy.py \
  tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py \
  tests/unit/performance/test_step4_pro_latest_silicon_coverage.py \
  tests/unit/collector/test_step4_pro_latest_provider_cases.py \
  tests/unit/collector/test_step4_deepep_ht_distributed_driver.py \
  tests/unit/collector/test_step4_deepep_ht_nccl_preflight.py \
  tests/unit/collector/test_step4_deepep_ht_runtime.py \
  tests/unit/collector/test_framework_manifest.py \
  tests/unit/collector/test_version_resolver.py \
  tests/unit/sdk/database/test_step4_pro_latest_provider_data.py \
  tests/unit/sdk/database/test_step4_pro_latest_attention_data.py \
  tests/unit/sdk/database/test_step4_pro_latest_deepep_data.py \
  tests/unit/sdk/database/test_step4_pro_latest_optimus_moe_data.py \
  tests/unit/sdk/models/test_step4_pro_latest.py \
  tests/unit/sdk/operations/test_step4_pro_latest_operations.py \
  tests/unit/sdk/backends/test_base_backend.py
```

### Collector regression command

```bash
timeout 60s systemd-run --user --scope -p MemoryMax=2G \
  env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. MPLBACKEND=Agg \
  /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  -m pytest -q tests/unit/collector
```

### Static and artifact checks

```bash
BASE=84021c6c8fa47bad9a72c335940c4be9fd0740db
RUFF=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/ruff
"$RUFF" check $(git diff --name-only "$BASE" -- '*.py')
"$RUFF" format --check $(git diff --name-only "$BASE" -- '*.py')

for f in $(git diff --name-only "$BASE" -- '*.sh' |
  grep '^tests/\(e2e\|performance\)/step4_pro_latest/'); do
  bash -n "$f"
done

git diff --check
```

The artifact audit recomputed the `summary.md` inventory hashes, read all six
canonical Parquet files, checked the complete proxy matrix and three-repeat
JSON, parsed the normalized CSV, parsed two YAML files and three JSON inputs,
and asserted that no DeepEP Parquet exists.

## 2. Validation Criteria

1. Both focused and full Collector tests must have zero failures.
2. All changed task Python files must pass Ruff check and format.
3. All changed Step4-Pro-Latest shell scripts must pass `bash -n`.
4. All recorded deliverable hashes must match.
5. Canonical B300 data must contain exactly `709` rows:
   `68 + 167 + 174 + 75 + 75 + 150`.
6. The simulation artifact must contain `72` prefill and `84` decode cases.
7. Prefill status must be `24 PASS_WITH_PROXY + 48 OOM`.
8. Decode must report `B_max=0` and `first_failed_batch=1` for `84/84`.
9. Three-repeat evidence must contain `468` executions, `156/156` identical
   cases, and maximum spread `0.0`.
10. `step4_deepep_ht_perf.parquet` must remain absent.
11. The complete current code/doc diff must contain no unresolved blocking
    defect.
12. The two-node B300 smoke must not be reported as PASS unless both replicas
    complete the corrected lifecycle.
13. The owner-authorized checkpoint may be committed and pushed while the
    live run is quota-blocked, provided the report keeps that limitation
    explicit.

## 3. Test Results and Evidence

### Final publication candidate

| Check | Result | Numeric evidence |
|---|---|---:|
| Focused regression | PASS | `347/347`, `8.21s` |
| Full Collector regression | PASS | `401/401`, `31.60s` |
| Ruff check/format | PASS | `45/45` files |
| Shell syntax | PASS | `14/14` scripts |
| Working-tree whitespace | PASS | exit `0` |
| Canonical B300 data | PASS | `709` rows |
| Prefill matrix | PASS_WITH_PROXY | `72`; `24 PASS_WITH_PROXY + 48 OOM` |
| Decode matrix | PASS_WITH_PROXY | `84`; `B_max=0` for `84/84` |
| Repeat audit | PASS | `468`; `156/156` identical; spread `0.0` |
| Manual-review CSV | PASS | `156 × 87` |
| Structured inputs | PASS | YAML `2/2`; JSON `3/3` |
| DeepEP persistence isolation | PASS | DeepEP Parquet absent |
| Local current-diff review | APPROVE | blocking findings `0` |
| Corrected two-node live run | BLOCKED_BY_QUOTA | requested `16`; last trusted remainder `6`; replicas `0/2` |

Evidence logs:

- `/data/ycfeng/tmp/step4_final_publication_20260817_rerun/focused_pytest.log`
  - SHA256:
    `0a0486bcb10f1296078d4ad9d24d4bdce83ef2be9109c8738653a4228f465293`
- `/data/ycfeng/tmp/step4_final_publication_20260817_rerun/collector_pytest.log`
  - SHA256:
    `7ef646b7d689e6a10152c26df4e36a90d33234a15d9a17ee24971142a474b4b6`
- `/data/ycfeng/tmp/step4_final_publication_20260817_rerun/static_and_artifact_checks.log`
  - SHA256:
    `5cfb52bb09b2bc088d52fed381b61a9f54abe4baba2328f1ab791f3a8b094b46`

### Review evidence

The previously committed AIC core and dataset received the following
independent review:

- Code/spec reviewer:
  `01a00e2e-e38e-78e3-9612-13c20d4edabc`
  - Verdict: **APPROVE**
  - CRITICAL/HIGH/MEDIUM: `0/0/0`
  - LOW: `2`
- Architecture reviewer:
  `01a00e2f-25b2-7b91-be89-203a9ece087b`
  - Status: **WATCH**
  - BLOCK: `0`

The architecture watch items are:

1. AIC data consumers select primarily by system/backend/version and do not
   yet enforce the pinned vLLM source/package identity.
2. Whole-model external runtime sign-off remains pending.
3. Some raw collection evidence remains outside the Git commit under
   `/data/ycfeng/tmp`.

The current uncommitted Phase 12 runtime-wrapper and documentation diff was
reviewed locally. No independent sub-agent review is claimed for this later
slice. The local review inspected the AgRs backend selection, both-node
validation barrier, shutdown-arm acknowledgement, concurrent final release,
manager-marker scope, fail-fast DeepEP rejection, tests, and task records.
Blocking findings: `0`.

### Audit-command failure and root cause

The first artifact-audit command failed with:

```text
KeyError: 'prefill'
```

Root cause: the one-off audit used obsolete top-level names `prefill` and
`decode`; the actual artifact uses `prefill_results` and `decode_results`.
The corrected audit read the real schema and passed. No production code,
dataset, or simulation artifact changed.

The first final rerun audit also used the obsolete path
`repeat["summary"]`; the archived repeat artifact stores these fields under
`repeat["repeat_audit"]`. Reading the actual top-level keys identified the
root cause. The corrected audit passed `70/70` hashes, `72+84` simulation
cases, `468/156/0.0` repeat evidence, `156 × 87` LF-only CSV, `709` canonical
rows, and DeepEP-Parquet absence. No repository file changed for this audit
retry.

### Publication boundary

The earlier Phase 12 lifecycle and script/test skew are resolved in the
current candidate. The final corrected two-node payload did not run because
the platform created `0/2` replicas for a request of `16` B300 GPUs while the
last trusted queue remainder was `6`. This checkpoint may be published by the
owner's explicit instruction, but publication is not evidence of a completed
two-node runtime PASS.
