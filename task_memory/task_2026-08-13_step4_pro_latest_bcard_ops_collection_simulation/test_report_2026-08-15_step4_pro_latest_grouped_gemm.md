## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Added grouped `wo_a` Collector/consumer test evidence and the known unrelated regression. |

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

### Population evidence

| Metric | Value |
|---|---:|
| Latest plan ops | 8 |
| Getter tasks | 74 |
| Unique benchmark invocations | 74 |
| Unique persisted physical keys | 74 |
| Deduplicated tasks | 0 |
| Structural keys | 1 |
| Minimum token count | 1 |
| Maximum token count | 32768 |

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

No B300 latency or power value was collected in this sub-step. The code and
data contract are ready, but formal B300 collection remains pending.
