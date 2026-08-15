## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Added FP32 router Collector/consumer RED-GREEN evidence and regression results. |

# Test Report: Step4-Pro-Latest FP32 Router

**Date:** 2026-08-15
**Environment:** conda `aic-step-design`, Python `3.11.15`, pytest `8.4.2`

## 1. Test Script Information

- Test scripts:
  - `tests/unit/collector/test_step4_pro_latest_provider_cases.py`
  - `tests/unit/sdk/database/test_step4_pro_latest_provider_data.py`
  - `tests/unit/sdk/models/test_step4_pro_latest.py`
  - `tests/unit/sdk/operations/test_step4_pro_latest_operations.py`
  - `tests/unit/collector/test_framework_manifest.py`
  - `tests/unit/collector/test_version_resolver.py`
- Focused command:

  ```bash
  timeout 900s systemd-run --user --scope -p MemoryMax=2G \
    env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. \
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

- Collector calls the pinned vLLM custom operation
  `torch.ops.vllm.optimus_matmul_fp32`, registered by
  `vllm/model_executor/models/step3p5_util.py`.
- Input and weights are BF16; output is FP32.
- Exact structural lookup fields are provider, N, K, weight dtype, and output
  dtype.
- Only `num_tokens` may interpolate.
- Missing provider/shape, non-positive tokens, and conflicting rows fail
  explicitly.
- Generic GEMM data is never used for this operation.

## 3. Test Results and Evidence

### Key outcomes

| Check | Result | Numeric evidence |
|---|---|---:|
| RED contract | PASS | 7 expected failures, 13 passes, 1.37s |
| Focused GREEN suite | PASS | 145 passed, 0 failed, 1.97s |
| Full Collector suite | PASS | 349 passed, 0 failed, 30.88s |
| SDK/database suite | PARTIAL | 611 passed, 1 unrelated failure, 93.93s |
| Router getter tasks | PASS | 74 tasks |
| Unique invocations | PASS | 74 |
| Unique physical keys | PASS | 74 |
| Deduplicated tasks | PASS | 0 |
| Token range | PASS | 1–32768 |

The one structural identity is:

```text
provider=vllm.optimus_matmul_fp32
n=896
k=7168
weight_dtype=bfloat16
output_dtype=float32
```

The interpolation fixture used:

- `1 token -> 0.2 ms`;
- `4 tokens -> 0.8 ms`;
- query `2 tokens -> 0.4 ms`;
- operation scale factor `2.0`;
- returned latency `0.8 ms`;
- expected latency `0.8 ms`;
- absolute error `0.0 ms`;
- relative error `0.0%`.

### Known unrelated failure

`test_query_custom_allreduce_large_tp_scaling` still expects TP16 scaling from
an 8-GPU CustomAllReduce table. Production code intentionally requires a
measured multi-node collective and raises `PerfDataNotAvailableError`. This is
the previously recorded ISSUE-019; neither its production code nor test was
changed.

### B300 status

No B300 router latency or power row was measured in this sub-step. The
Collector now invokes the exact pinned vLLM custom op; formal B300 collection
remains pending.
