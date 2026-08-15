## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Recorded the verified Step4-Pro-Latest MTP-off model/op graph and KV-capacity regression results. |

# Test Report: Step4-Pro-Latest Model and Operation Graph

**Date:** 2026-08-15

## 1. Test Script Information

### Focused Latest tests

- Scripts:
  - `tests/unit/sdk/models/test_step4_pro_latest.py`
  - `tests/unit/sdk/operations/test_step4_pro_latest_operations.py`
- Command:

```bash
PYTHONPATH=src:. MPLBACKEND=Agg TMPDIR=/data/ycfeng/tmp \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  -m pytest \
  tests/unit/sdk/models/test_step4_pro_latest.py \
  tests/unit/sdk/operations/test_step4_pro_latest_operations.py -q
```

### Historical Step4 and Collector regression

- Scripts: the 899-test scope recorded in
  `test_report_2026-08-13_step4_pro_latest_baseline.md`.
- Exact command: section 4 of that report, rerun unchanged on 2026-08-15.

### Environment

| Item | Actual Value |
|---|---|
| Conda environment | `/home/i-fengyicheng/miniconda3/envs/aic-step-design` |
| Python | `3.11.15` |
| pytest | `8.4.2` |
| Source binding | `PYTHONPATH=$PWD/src:$PWD` |
| Temporary storage | `/data/ycfeng/tmp` |
| Historical regression memory scope | `MemoryMax=4G` |

## 2. Validation Criteria

- Both prefill and decode graphs contain all 78 layers in execution order.
- Full MFA uses Optimus FA4, inverse RoPE, and grouped `wo_a`.
- SWA uses native sliding GQA with Q/K/V norm and RoPE.
- Router output is FP32; routed MoE is FP8 block; other required compute and
  KV paths are BF16.
- MTP is disabled and TP greater than one fails fast.
- The logical hybrid KV curve and page allocation return exact expected bytes.
- KV budget inversion returns the largest sequence length that actually fits.
- Existing Step4 V1/V3/V4, Collector, database, and performance tests do not
  regress.

## 3. Test Results and Evidence

### Summary

| Test Scope | Passed | Failed | Runtime | Result |
|---|---:|---:|---:|---|
| Latest focused suite | 28 | 0 | 0.20s | PASS |
| Historical regression | 899 | 0 | 67.69s | PASS |

### Key numeric evidence

| Metric | Expected | Actual | Result |
|---|---:|---:|---|
| Context operations | 1,821 | 1,821 | PASS |
| Generation operations | 1,821 | 1,821 | PASS |
| Logical KV, 513 tokens | 132,141,056 bytes | 132,141,056 bytes | PASS |
| Page-allocated KV, 513 tokens | 134,742,016 bytes | 134,742,016 bytes | PASS |
| KV max tokens before fix | 513 | 512 | RED reproduced |
| KV max tokens after fix | 513 | 513 | PASS |

### Evidence logs

```text
/data/ycfeng/tmp/step4_latest_model_ops_green_20260814.log
/data/ycfeng/tmp/step4_latest_baseline_repaired_final_rerun_20260815.log
```

The provider-specific database queries remain intentionally fail-fast until
Collector rows and matching consumer tables are implemented and measured on
B300. This report therefore approves the model/op graph only, not Collector
or end-to-end simulation completion.
