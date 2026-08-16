## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Added FP32 router Collector/consumer RED-GREEN evidence and regression results. |
| 2026-08-15 | Added complete B300 74-row collection, canonical dataset, pinned custom-op trace, and exact-consumer evidence. |
| 2026-08-15 | Extended B300 coverage to 75 rows with the required `65536`-token point and refreshed canonical evidence. |
| 2026-08-16 | Added the mandatory slice-matched smoke-evidence path and SHA256 to every reproducible full-collection command. |

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
| Router getter tasks | PASS | 75 tasks |
| Unique invocations | PASS | 75 |
| Unique physical keys | PASS | 75 |
| Deduplicated tasks | PASS | 0 |
| Token range | PASS | 1–65536 |
| B300 router smoke | PASS | 1/1 row |
| B300 router full collection | PASS | 75/75 rows, 0 Collector errors |
| Canonical exact consumers | PASS | 75/75 silicon queries |
| Canonical query error | PASS | 0.0 ms absolute, 0.0% relative |

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

The B300 router dataset is complete:

- runtime: vLLM `0.19.0.post20.dev26+gc820e5ae1`;
- device: `NVIDIA B300 SXM6 AC`;
- provider: `vllm.optimus_matmul_fp32`;
- rows: `75/75`;
- duplicate physical keys: `0`;
- combined grouped/router latency range:
  `0.009082666908701261–2.518085320790609 ms`;
- runtime trace:
  `torch.ops.vllm.optimus_matmul_fp32` →
  `step3p5_util.apply_optimus_matmul_fp32` →
  `torch.ops.OptimusMoe.matmul_fp32`;
- `65536`-token measured latency: `0.539962649345398 ms`;
- canonical consumer matches: `75/75`, all `source="silicon"`;
- maximum absolute/relative error: `0.0 ms` / `0.0%`.

Canonical file:

`src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_fp32_output_gemm_perf.parquet`

- size: `4755` bytes;
- SHA256:
  `3ec22771fa8577dec8b250ca1d6152552a8091b9502c1732461df65f5dc97af9`.

Evidence:

`/data/ycfeng/tmp/step4_provider_65k_b300_20260815/`.
