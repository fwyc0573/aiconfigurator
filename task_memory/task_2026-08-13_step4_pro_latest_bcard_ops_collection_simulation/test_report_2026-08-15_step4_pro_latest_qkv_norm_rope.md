## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Added QKV norm/RoPE Collector, consumer, source-path, population, and regression evidence. |

# Test Report: Step4-Pro-Latest QKV Norm/RoPE

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

- Full MFA executes the pinned source path from
  `vllm/model_executor/models/step4pro.py`:
  - `Step4ProAttention._tail_rope` for 64 Q heads;
  - `OptimusRMSNorm` and `_tail_rope` for the single shared K head;
  - head dimension `512`, tail RoPE dimension `64`.
- SWA executes the pinned source path from `step3p5.py`,
  `step3p5_util.py`, and `step4pro.py`:
  - `fused_qknorm_rope_forward_impl` for Q/K;
  - `Step4ProSlidingAttention._prepare_value_for_attention`;
  - `OptimusRMSNorm` for V because `gqa_v_norm=True`;
  - Q/K/V geometry QH `128`, KVH `8`, HD `128`.
- Structural lookup fields are exact:
  `provider`, `normalized_tensors`, `q_heads`, `kv_heads`, and `head_dim`.
- Only `num_tokens` may interpolate.
- Missing structures, non-positive tokens, and conflicting rows fail
  explicitly.
- Generic RMSNorm, generic RoPE, or
  `optimus_fused_qknorm_rope_cache` substitution is not accepted.

## 3. Test Results and Evidence

### Key outcomes

| Check | Result | Numeric evidence |
|---|---|---:|
| Initial RED contract | PASS | 10 expected failures, 20 passes, 1.50s |
| Full-MFA Q-output correction RED | PASS | 1 expected failure, 0 passes, 0.31s |
| Focused GREEN suite | PASS | 155 passed, 0 failed, 2.06s |
| Full Collector suite | PASS | 352 passed, 0 failed, 29.56s |
| SDK/database suite | PARTIAL | 618 passed, 1 unrelated failure, 92.27s |
| Ruff check | PASS | 0 findings |
| Ruff format | PASS | 9/9 files formatted |
| Whitespace | PASS | `git diff --check` exit 0 |

### Population evidence

| Metric | Value |
|---|---:|
| Getter tasks | 148 |
| Unique benchmark invocations | 148 |
| Unique persisted physical keys | 148 |
| Deduplicated tasks | 0 |
| Structural keys | 2 |
| Token points per structure | 74 |
| Minimum token count | 1 |
| Maximum token count | 32768 |

The exact structures are:

```text
vllm_step4pro_k_norm_rope / normalized=k / QH=64 / KVH=1 / HD=512
vllm_step4pro_qkv_norm_rope / normalized=q+k+v / QH=128 / KVH=8 / HD=128
```

The Full-MFA runtime probe requires both outputs:

```text
Q: [num_tokens, 64, 512]
K: [num_tokens, 1, 512]
```

This prevents the K-only timing defect found during review. No latency scaling
factor was introduced; the benchmark now executes the missing pinned Q tail
RoPE operation directly.

The interpolation fixture used:

- `1 token -> 0.2 ms`;
- `4 tokens -> 0.8 ms`;
- query `2 tokens -> 0.4 ms`;
- operation scale factor `2.0`;
- returned latency `0.8 ms`;
- expected latency `0.8 ms`;
- absolute error `0.0 ms`;
- relative error `0.0%`.

### Evidence logs

| Log | SHA256 |
|---|---|
| `/data/ycfeng/tmp/step4_qkv_focused_green.log` | `cfc4f8119b5125f511002092f896ff5edb84365e0747f7da838bbfbfb857d771` |
| `/data/ycfeng/tmp/step4_qkv_collector_green.log` | `8a866e54d33656e76a88e6bf9d657241439e0a0b9b3def1deab3df7c76687b33` |
| `/data/ycfeng/tmp/step4_qkv_sdk_database_green.log` | `ca9788c6ddc94b5663b4c9b74eab7968610847097360aa9adb1a7abe1f932e25` |

### Known unrelated failure

`test_query_custom_allreduce_large_tp_scaling` expects TP16 scaling from an
8-GPU CustomAllReduce table. Production code intentionally requires a measured
multi-node collective and raises `PerfDataNotAvailableError`. This is the
previously recorded ISSUE-019; no fallback scaling was restored.

### B300 status

No B300 QKV latency or power row was measured in this sub-step. The Collector
now invokes the exact pinned vLLM paths; formal B300 collection remains
pending.
