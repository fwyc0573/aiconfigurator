## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Added grouped `wo_a` Collector/consumer test evidence and the known unrelated regression. |
| 2026-08-15 | Added complete B300 74-row collection, canonical dataset, runtime profile, and exact-consumer evidence. |
| 2026-08-15 | Extended B300 coverage to 75 rows with the required `65536`-token point and refreshed canonical evidence. |
| 2026-08-16 | Added the mandatory slice-matched smoke-evidence path and SHA256 to every reproducible full-collection command. |

# Test Report: Step4-Pro-Latest Grouped `wo_a`

**Date:** 2026-08-15
**Environment:** conda `aic-step-design`, Python `3.11.15`, pytest `8.4.2`

## 1. Test Script Information

- Test scripts:
  - `tests/unit/collector/test_step4_pro_latest_provider_cases.py`
  - `tests/unit/collector/test_framework_manifest.py`
  - `tests/unit/collector/test_version_resolver.py`
  - `tests/unit/sdk/database/test_step4_pro_latest_provider_data.py`
  - `tests/unit/sdk/models/test_step4_pro_latest.py`
  - `tests/unit/sdk/operations/test_step4_pro_latest_operations.py`
- Focused command:

  ```bash
  TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
    -m pytest -q \
    tests/unit/collector/test_step4_pro_latest_provider_cases.py \
    tests/unit/sdk/database/test_step4_pro_latest_provider_data.py \
    tests/unit/sdk/models/test_step4_pro_latest.py \
    tests/unit/sdk/operations/test_step4_pro_latest_operations.py \
    tests/unit/collector/test_framework_manifest.py \
    tests/unit/collector/test_version_resolver.py
  ```

- Full Collector command:

  ```bash
  timeout 900s systemd-run --user --scope -p MemoryMax=2G \
    env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
    -m pytest -q tests/unit/collector
  ```

- SDK/database command:

  ```bash
  timeout 900s systemd-run --user --scope -p MemoryMax=2G \
    env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
    -m pytest -q tests/unit/sdk/database \
    tests/unit/sdk/models/test_step4_pro_latest.py \
    tests/unit/sdk/operations/test_step4_pro_latest_operations.py
  ```

- B300 full collection:

  ```bash
  MODE=full \
  PROVIDER_CORE_SLICE=grouped_router \
  PROVIDER_CORE_SMOKE_EVIDENCE=/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-core-gr-0815-2343/s4p-aic-attn-evidence-s4p-aic-core-gr-0815-2343/checkpoint/representative_provider_core.json \
  PROVIDER_CORE_SMOKE_EVIDENCE_SHA256=9aa6cd63ab75256cb16c0116eaf64d0bc1844034c2e58e93e3b5e15cc075b924 \
  RJOB_NAME=s4p-aic-core-grf-0815-2350 \
  RESULT_TIMEOUT_SECONDS=3600 \
  bash tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh
  ```

- B300 65K extension collection:

  ```bash
  MODE=full \
  PROVIDER_CORE_SLICE=grouped_router \
  PROVIDER_CORE_SMOKE_EVIDENCE=/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-core-gr-0815-2343/s4p-aic-attn-evidence-s4p-aic-core-gr-0815-2343/checkpoint/representative_provider_core.json \
  PROVIDER_CORE_SMOKE_EVIDENCE_SHA256=9aa6cd63ab75256cb16c0116eaf64d0bc1844034c2e58e93e3b5e15cc075b924 \
  RJOB_NAME=s4p-aic-core-65k-0815-0136 \
  RESULT_TIMEOUT_SECONDS=3600 \
  ARTIFACT_ROOT=/data/ycfeng/tmp/step4_provider_65k_b300_20260815/full_s4p-aic-core-65k-0815-0136 \
  bash tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh
  ```

- Exact consumer validator:
  `tests/performance/step4_pro_latest/validate_b300_provider_core_rows.py`.

## 2. Validation Criteria

- Latest plan selects exactly 8 required ops, SM103, and runtime profile
  `step4_pro_latest`.
- Stock vLLM default remains `0.19.0`; Latest requires installed package
  `0.19.0.post20.dev26.gc820e5ae1`.
- Grouped cases contain only:
  - provider `vllm_step4pro_torch_einsum`;
  - groups `8`;
  - per-group N `1024`;
  - K `4096`;
  - dtype `bfloat16`.
- Invocation and persisted-key counts are equal; no unexplained duplicate is
  allowed.
- Consumer uses exact structural lookup and interpolates only `num_tokens`.
- Missing provider/shape, non-positive tokens, and conflicting rows must fail.

## 3. Test Results and Evidence

### Key outcomes

| Check | Result | Numeric evidence |
|---|---|---:|
| Focused suite | PASS | 139 passed, 0 failed, 1.92s |
| Full Collector suite | PASS | 347 passed, 0 failed, 29.91s |
| SDK/database suite | PARTIAL | 607 passed, 1 unrelated failure, 93.82s |
| Ruff check | PASS | 0 findings |
| Ruff format | PASS | 16/16 files formatted |
| Whitespace | PASS | `git diff --check` exit 0 |
| B300 grouped smoke | PASS | 1/1 row |
| B300 grouped full collection | PASS | 75/75 rows, 0 Collector errors |
| Unique physical keys | PASS | 75/75 |
| Canonical exact consumers | PASS | 75/75 silicon queries |
| Canonical query error | PASS | 0.0 ms absolute, 0.0% relative |

### Population evidence

| Metric | Value |
|---|---:|
| Latest plan ops | 8 |
| Getter tasks | 75 |
| Unique benchmark invocations | 75 |
| Unique persisted physical keys | 75 |
| Deduplicated tasks | 0 |
| Structural keys | 1 |
| Minimum token count | 1 |
| Maximum token count | 65536 |

The interpolation test used measured fixture points:

- `1 token -> 0.1 ms`;
- `4 tokens -> 0.4 ms`;
- query `2 tokens -> 0.2 ms`;
- operation scale factor `2.0`;
- returned latency `0.4 ms`;
- expected latency `0.4 ms`;
- absolute error `0.0 ms`;
- relative error `0.0%`.

### Known unrelated failure

`test_query_custom_allreduce_large_tp_scaling` expects TP16 scaling from an
8-GPU CustomAllReduce table. Current production code intentionally raises
`PerfDataNotAvailableError` and requires a measured multi-node collective.
The failure reproduces alone and predates this grouped-GEMM slice. It is
recorded as ISSUE-019; no fallback scaling was restored.

### B300 status

The B300 dataset is complete:

- runtime: vLLM `0.19.0.post20.dev26+gc820e5ae1`;
- device: `NVIDIA B300 SXM6 AC`;
- provider: `vllm_step4pro_torch_einsum`;
- rows: `75/75`;
- duplicate physical keys: `0`;
- combined grouped/router latency range:
  `0.009082666908701261–2.518085320790609 ms`;
- grouped runtime profile: `300` timed `torch.einsum` calls using the exact
  equation from pinned `Step4ProAttention.forward`;
- `65536`-token measured latency: `2.518085320790609 ms`;
- canonical consumer matches: `75/75`, all `source="silicon"`;
- maximum absolute/relative error: `0.0 ms` / `0.0%`.

Canonical file:

`src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_grouped_gemm_perf.parquet`

- size: `4761` bytes;
- SHA256:
  `78fdd68077e3fdaa4c4fa349ab0a72407e0421ae09ecfcfcd4a5ae22d103d760`.

Evidence:

`/data/ycfeng/tmp/step4_provider_65k_b300_20260815/`.
