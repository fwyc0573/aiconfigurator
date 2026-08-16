## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Recorded the accepted exact-op Optimus MoE B300 dataset and consumer validation. |
| 2026-08-16 | Normalized Markdown line endings for the task-scoped publication commit. |

# Test Report: Step4-Pro-Latest Optimus MoE

**Date:** 2026-08-15
**Environment:** pinned StepCast vLLM container, torch `2.10.0+cu129`,
CUDA `12.9`, vLLM `0.19.0.post20.dev26+gc820e5ae1`, Step Optimus `3.23.24`;
host validation used `aic-step-design`, Python `3.11.15`.

## 1. Test Script Information

- Host script:
  `tests/performance/step4_pro_latest/run_b300_optimus_moe_collection.sh`
- Worker script:
  `tests/performance/step4_pro_latest/remote_b300_optimus_moe_collection.sh`
- Collector:
  `collector/vllm/collect_step4_provider.py`
- Unit tests:
  `tests/unit/collector/test_step4_pro_latest_provider_cases.py`

Commands:

```bash
MODE=smoke bash tests/performance/step4_pro_latest/run_b300_optimus_moe_collection.sh
MODE=full bash tests/performance/step4_pro_latest/run_b300_optimus_moe_collection.sh
```

## 2. Validation Criteria

- Exactly `174` full rows and zero collection errors.
- `<6144` local tokens use
  `deepgemm_optimus_moe_masked_fp8`, CUDA graph, and
  `used_cuda_graph=True`.
- `>=6144` local tokens use `deepgemm_optimus_moe_fp8`, eager execution,
  and `used_cuda_graph=False`.
- EP16/EP32 and all three routing distributions are present.
- Every accepted row queries back exactly with `source="silicon"`.
- Worker resources are deleted after evidence collection.

## 3. Test Results and Evidence

**Outcome: PASS**

| Metric | Actual |
|--------|-------:|
| Completed rows | 174/174 |
| Collection errors | 0 |
| EP16 / EP32 rows | 87 / 87 |
| Masked/CUDA-graph rows | 145 |
| Contiguous/eager rows | 29 |
| Consumer exact matches | 174/174 |
| Silicon source rows | 174/174 |
| Minimum latency | 0.27983999252319336 ms |
| Maximum latency | 16.00377019246419 ms |
| Collection elapsed time | 111.20 s |

- Accepted CSV SHA256:
  `cbea7ec572729121df09784a0b06dcff5c92780c510b12fdda28f6982afca3fd`
- Canonical parquet:
  `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_optimus_moe_perf.parquet`
- Canonical parquet SHA256:
  `4bfb1ccdfa8007d3a23576b4e7d50e10dbb11fef6500db29668a59f530cad388`
- Evidence:
  `/data/ycfeng/tmp/step4_aic_optimus_moe_b300_20260815/full_s4p-aic-moe-0815-182707/`
- Final matching RJobs: `0`
- Final matching Replicas: `0`
