## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Recorded provider Attention, hybrid physical-layout, peak-KV capacity, and regression evidence. |
| 2026-08-15 | Added complete B300 measurement, canonical Parquet archival, and 199/199 canonical exact-consumer evidence. |
| 2026-08-15 | Added the 65K Full FA4/native SWA measurements and updated canonical evidence to 201/201 exact consumers. |
| 2026-08-15 | Added scheduler-derived workload rows and updated canonical evidence to 225/225 exact consumers. |
| 2026-08-16 | Added ten clean scheduler-closure rows and updated canonical evidence to 235/235 exact consumers. |
| 2026-08-16 | Normalized Markdown line endings for the task-scoped publication commit. |

# Test Report: Step4-Pro-Latest Provider Attention

**Date:** 2026-08-15–2026-08-16
**Environment:** `aic-step-design`, Python `3.11.15`, pytest `8.4.2`

## 1. Test Script Information

### Scripts

- `tests/unit/collector/test_step4_pro_latest_provider_cases.py`
- `tests/unit/sdk/database/test_step4_pro_latest_attention_data.py`
- `tests/unit/sdk/models/test_step4_pro_latest.py`
- `tests/unit/sdk/operations/test_step4_pro_latest_operations.py`
- `tests/unit/sdk/models/test_model_config.py`
- `tests/unit/sdk/backends/test_base_backend.py`
- `tests/unit/sdk/test_memory_estimation.py`
- `tests/performance/step4_pro_latest/validate_b300_attention_rows.py`

### Commands

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest -q \
  tests/unit/sdk/models/test_step4_pro_latest.py \
  tests/unit/sdk/models/test_model_config.py \
  tests/unit/sdk/backends/test_base_backend.py \
  tests/unit/sdk/test_memory_estimation.py
```

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest -q \
  tests/unit/collector
```

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest -q \
  tests/unit/sdk/database tests/unit/sdk/models tests/unit/sdk/operations \
  tests/unit/sdk/backends/test_base_backend.py \
  tests/unit/sdk/test_memory_estimation.py
```

```bash
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/ruff check <changed-python-files>
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/ruff format --check <changed-python-files>
git diff --check
```

```bash
MODE=full \
RJOB_NAME=s4p-aic-attn-0815-161251 \
bash tests/performance/step4_pro_latest/run_b300_attention_collection.sh
```

```bash
MODE=smoke \
SWA_CONTEXT_SMOKE_TOKENS=65536 \
RJOB_NAME=s4p-aic-swa65k-0815-024430 \
bash tests/performance/step4_pro_latest/run_b300_attention_collection.sh
```

```bash
MODE=smoke \
FULL_CONTEXT_SMOKE_TOKENS=65536 \
RJOB_NAME=s4p-aic-fa465k-0815-030553 \
bash tests/performance/step4_pro_latest/run_b300_attention_collection.sh
```

Final interpolation-endpoint measurement used:

```bash
MODE=smoke \
FULL_CONTEXT_SMOKE_TOKENS=16384 \
FULL_CONTEXT_SMOKE_TOTAL_TOKENS=262144 \
SWA_CONTEXT_SMOKE_TOKENS=16384 \
SWA_CONTEXT_SMOKE_TOTAL_TOKENS=262144 \
RJOB_NAME=s4p-aic-attnep-clean-0816-1519 \
bash tests/performance/step4_pro_latest/run_b300_attention_collection.sh
```

Canonical consumer validation used:

- source implementation:
  `src/aiconfigurator/sdk/operations/attention.py`;
- canonical data directory:
  `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0`;
- evidence command output:
  `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/attention_b300_consumer_validation_2026-08-15.json`.

## 2. Validation Criteria

- Full MFA and SWA use separate provider identities and persisted keys.
- Full MFA represents Optimus FA4 hd512 with aliased K/V storage.
- SWA represents native vLLM hd128/window-512 GQA.
- Both structures carry `NHD` layout and `524288` physical page bytes.
- Generation population contains exactly 71 Full-MFA and 96 SWA cases, with
  no duplicate invocations or persisted keys.
- Logical KV, resident physical KV, and peak physical KV remain distinct.
- OOM and capacity inversion use peak physical allocation.
- All Step4/Attention-focused and Collector tests pass.
- The only accepted broad-regression failure is the pre-existing ISSUE-019
  stale `CustomAllReduce TP16` test.

## 3. Test Results and Evidence

### Key Outcome

| Check | Result | Numeric Evidence |
|-------|--------|------------------|
| Physical-layout RED | PASS | 11 failed, 50 passed |
| Physical-layout GREEN | PASS | 61 passed |
| Peak-KV RED | PASS | 6 failed, 254 passed |
| Peak-KV final GREEN | PASS | 260 passed in 0.79s |
| Collector regression | PASS | 358 passed in 31.24s |
| SDK/database/models/operations regression | PASS with known baseline anomaly | 1233 passed, 1 unrelated ISSUE-019 failure in 94.90s |
| Ruff check | PASS | 0 findings |
| Ruff format | PASS | 16 files already formatted |
| Whitespace check | PASS | `git diff --check` exit 0 |
| B300 collection plus scheduler extension | PASS | 68 context + 167 generation rows, 0 accepted-row errors |
| Source data physical keys | PASS | 68/68 context, 167/167 generation unique |
| Canonical exact consumers | PASS | 235/235 silicon queries |
| Canonical query error | PASS | 0.0 ms absolute, 0.0% relative |
| Exhaustive scheduler consumers | PASS | 394/394 queries; missing/error 0/0 |
| Native SWA 65K | PASS | 3.164181391398112 ms |
| Full FA4 65K | PASS | 196.56292724609375 ms |
| Pinned source identity | PASS | 2103/2103 files verified in both 65K runs |
| Resource cleanup | PASS | final RJob 0, Replica 0 |

### Case Population

| Phase | Full MFA | SWA | Total | Duplicate Invocations | Duplicate Keys |
|-------|---------:|----:|------:|----------------------:|---------------:|
| Context | 34 | 34 | 68 | 0 | 0 |
| Generation | 71 | 96 | 167 | 0 | 0 |

### KV Numeric Evidence

| Sequence Length | Logical KV Bytes | Resident Physical KV Bytes | Peak Physical KV Bytes |
|----------------:|-----------------:|---------------------------:|-----------------------:|
| 513 | 132,141,056 | 204,472,320 | 204,472,320 |
| 640 | 134,742,016 | 174,063,616 | 204,472,320 |

For a `204,472,320-byte` physical KV budget:

- predicted maximum safe MTP-off decode length: `640` tokens;
- peak at 640: `204,472,320 bytes`;
- peak at 641: `214,958,080 bytes`;
- absolute excess at 641: `10,485,760 bytes`;
- relative excess at 641: `5.128205%`.

### Logs

- `/data/ycfeng/tmp/step4_attention_physical_layout_red_20260815.log`
- `/data/ycfeng/tmp/step4_attention_physical_layout_green_20260815.log`
- `/data/ycfeng/tmp/step4_physical_kv_peak_red_20260815.log`
- `/data/ycfeng/tmp/step4_physical_kv_peak_final_20260815.log`
- `/data/ycfeng/tmp/step4_attention_collector_regression_post_peak_20260815.log`
- `/data/ycfeng/tmp/step4_attention_sdk_regression_post_peak_20260815.log`
- `/data/ycfeng/tmp/step4_attention_canonicalization_preflight_20260815.json`
- `/data/ycfeng/tmp/step4_attention_canonical_consumer_validation_20260815.json`
- `/data/ycfeng/tmp/step4_aic_attention_b300_20260815/smoke_s4p-aic-swa65k-0815-024430/`
- `/data/ycfeng/tmp/step4_aic_attention_b300_20260815/smoke_s4p-aic-fa465k-0815-030553/`
- `/data/ycfeng/tmp/step4_attention_52_canonical_consumer_validation_20260815.json`
- `/data/ycfeng/tmp/step4_attention_225_canonical_consumer_validation_20260815.log`
- `/data/ycfeng/tmp/step4_aic_attention_b300_20260815/full_s4p-aic-attnfix-0815-0347/`
- `/data/ycfeng/tmp/step4_attention_closure_20260816/`
- `/data/ycfeng/tmp/step4_attention_closure_20260816/scheduler_attention_exhaustive_197_post_endpoint_20260816.json`

### B300 Dataset and Canonical Evidence

| Phase | Provider split | Rows | Canonical SHA256 |
|---|---|---:|---|
| Context | Optimus FA4 `34`, native SWA `34` | 68 | `7cf90e1f508e9f18dceb29c51dc90e88caa1f13d3d25d38f4a3435851c79522a` |
| Generation | Optimus FA4 `71`, native SWA `96` | 167 | `bde836b884b410b21284645ab45f0a35f82d03cac5b91299b256823067cb14c0` |

Canonical paths:

- `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_context_attention_perf.parquet`;
- `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_generation_attention_perf.parquet`.

Measured latency values span
`0.013167999684810638–49858.390625 ms`. Every canonical consumer query returned
the exact measured value with `source="silicon"`; maximum absolute error was
`0.0 ms` and maximum relative error was `0.0%`.

The added `65,536`-token context values are:

| Provider | Batch | Query tokens | Total context tokens | Latency |
|---|---:|---:|---:|---:|
| `vllm_native_sliding_gqa` | 1 | 65,536 | 65,536 | 3.164181391398112 ms |
| `optimus_fa4` | 1 | 65,536 | 65,536 | 196.56292724609375 ms |

Scheduler-derived context measurements added by the corrected full run:

| Provider | Batch | Query | Total context | Latency |
|---|---:|---:|---:|---:|
| `optimus_fa4` | 2 | 512 | 512 | 0.08673066894213359 ms |
| `vllm_native_sliding_gqa` | 2 | 512 | 512 | 0.04465599854787191 ms |
| `optimus_fa4` | 1 | 4,096 | 4,096 | 0.7718506654103597 ms |
| `vllm_native_sliding_gqa` | 1 | 4,096 | 4,096 | 0.20472000042597452 ms |
| `optimus_fa4` | 1 | 16,384 | 16,384 | 11.520037333170572 ms |
| `vllm_native_sliding_gqa` | 1 | 16,384 | 16,384 | 0.7907306353251139 ms |

Additional scheduler-closure measurements accepted on 2026-08-16:

| Provider | Batch | Query | Total context | Latency |
|---|---:|---:|---:|---:|
| `optimus_fa4` | 1 | 4,096 | 8,192 | 1.701456069946289 ms |
| `vllm_native_sliding_gqa` | 1 | 4,096 | 8,192 | 0.21289066473642984 ms |
| `optimus_fa4` | 1 | 16,384 | 32,768 | 27.7849858601888 ms |
| `vllm_native_sliding_gqa` | 1 | 16,384 | 32,768 | 0.7991733551025391 ms |
| `optimus_fa4` | 1 | 8,160 | 1,048,544 | 534.923095703125 ms |
| `vllm_native_sliding_gqa` | 1 | 8,160 | 1,048,544 | 0.4092746575673421 ms |
| `optimus_fa4` | 1 | 32,736 | 1,048,544 | 2098.0398763020835 ms |
| `vllm_native_sliding_gqa` | 1 | 32,736 | 1,048,544 | 1.5781973203023274 ms |
| `optimus_fa4` | 1 | 16,384 | 262,144 | 256.1752115885417 ms |
| `vllm_native_sliding_gqa` | 1 | 16,384 | 262,144 | 0.7958613236745199 ms |

The final endpoint began with `0 MiB` GPU memory used and `274114 MiB` free.
The canonical merge kept `66` rows, added `2`, removed `0`, and deduplicated
`0`. All `197` scheduler coordinates were queried through both providers:
`394/394` succeeded with `0` missing, `0` errors, and maximum canonical
latency error `0.0 ms`.

### Known Unrelated Failure

`tests/unit/sdk/database/test_edge_cases.py::TestAllreduceEdgeCases::test_query_custom_allreduce_large_tp_scaling`
expects removed TP16 synthetic scaling. Current production correctly fails
fast because an exact 16-rank multi-node collective row is unavailable. This
is tracked as ISSUE-019 and was not changed by the Attention slice.
