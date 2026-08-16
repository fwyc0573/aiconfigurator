## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Added QKV norm/RoPE Collector, consumer, source-path, population, and regression evidence. |
| 2026-08-15 | Added the diagnostic B300 smoke failure, exact runtime identities, numeric boundary, root cause, and cleanup evidence. |
| 2026-08-15 | Added the isolated Full-MFA B300 measurement, canonical dataset, exact-consumer validation, and remaining SWA-only gap. |
| 2026-08-15 | Added final formatting-only repair and fresh fail-fast regression, Ruff, shell, and whitespace evidence. |
| 2026-08-16 | Added the approved SWA overlay, finite-value smoke, `75/75` SWA collection, and final `150/150` canonical validation. |
| 2026-08-16 | Added mandatory slice-matched smoke evidence to full commands and recorded the final reviewer-remediation validation. |

# Test Report: Step4-Pro-Latest QKV Norm/RoPE

**Date:** 2026-08-15–2026-08-16
**Environment:** conda `aic-step-design`, Python `3.11.15`, pytest `8.4.2`

## 1. Test Script Information

- Test scripts:
  - `tests/unit/collector/test_step4_pro_latest_provider_cases.py`
  - `tests/unit/sdk/database/test_step4_pro_latest_provider_data.py`
  - `tests/unit/sdk/models/test_step4_pro_latest.py`
  - `tests/unit/sdk/operations/test_step4_pro_latest_operations.py`
  - `tests/unit/collector/test_framework_manifest.py`
  - `tests/unit/collector/test_version_resolver.py`
  - `tests/performance/step4_pro_latest/test_b300_provider_core_collection_contract.py`
  - `tests/performance/step4_pro_latest/remote_b300_attention_collection.sh`
  - `tests/performance/step4_pro_latest/validate_b300_provider_core_rows.py`
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

- Final post-format focused command:

  ```bash
  set -euo pipefail
  timeout 900s systemd-run --user --scope -p MemoryMax=2G \
    env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. MPLBACKEND=Agg \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
    -m pytest -q \
    tests/performance/step4_pro_latest/test_b300_provider_core_collection_contract.py \
    tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py \
    tests/unit/performance/test_step4_pro_latest_silicon_coverage.py \
    tests/unit/collector/test_step4_pro_latest_provider_cases.py \
    tests/unit/sdk/database/test_step4_pro_latest_provider_data.py \
    tests/unit/sdk/database/test_step4_pro_latest_attention_data.py \
    tests/unit/sdk/database/test_step4_pro_latest_deepep_data.py \
    tests/unit/sdk/database/test_step4_pro_latest_optimus_moe_data.py \
    tests/unit/sdk/models/test_step4_pro_latest.py \
    tests/unit/sdk/operations/test_step4_pro_latest_operations.py
  ```

- Final formatted-contract command:

  ```bash
  set -euo pipefail
  timeout 900s systemd-run --user --scope -p MemoryMax=2G \
    env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. MPLBACKEND=Agg \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
    -m pytest -q \
    tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py \
    tests/e2e/step4_pro_latest/test_b300_two_node_deepep_legacy_probe_contract.py \
    tests/e2e/step4_pro_latest/test_b300_two_node_nccl_preflight_contract.py \
    tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py \
    tests/e2e/step4_pro_latest/test_run_b300_source_probe_static.py \
    tests/performance/step4_pro_latest/test_b300_provider_core_collection_contract.py
  ```

- Final static commands:

  ```bash
  set -euo pipefail
  xargs -a /data/ycfeng/tmp/step4_changed_task_python_files_20260815.txt \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/ruff check
  xargs -a /data/ycfeng/tmp/step4_changed_task_python_files_20260815.txt \
    /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/ruff format --check
  find tests/e2e/step4_pro_latest tests/performance/step4_pro_latest \
    -maxdepth 1 -type f -name '*.sh' -print0 | sort -z | xargs -0 -n1 bash -n
  git diff --check
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
| Post-format focused regression | PASS | 134 passed, 0 failed, 5.73s |
| Formatted contract regression | PASS | 31 passed, 0 failed, 1.97s |
| SWA overlay contracts | PASS | 2 passed, 10 deselected |
| SWA finite-value gate | PASS | 1 expected RED followed by 1/1 GREEN |
| Post-SWA focused regression | PASS | 255 passed, 0 failed, 13.12s |
| Post-SWA full Collector suite | PASS | 399 passed, 0 failed, 32.21s |
| Reviewer-remediation RED/GREEN | PASS | 5 expected failures then 48/48 passes |
| Complete expected-key revalidation | PASS | 150 rows, provider split 75+75, 150/150 exact consumers |
| Final focused regression | PASS | 259 passed, 0 failed, 7.88s |
| Final full Collector regression | PASS | 401 passed, 0 failed, 31.21s |
| Final reviewer contracts after format | PASS | 48 passed, 0 failed, 6.40s |
| Ruff check | PASS | 43/43 changed task Python files |
| Ruff format | PASS | 43/43 files already formatted |
| Shell syntax | PASS | 14/14 scripts |
| Whitespace | PASS | `git diff --check` exit 0 |

### Population evidence

| Metric | Value |
|---|---:|
| Getter tasks | 150 |
| Unique benchmark invocations | 150 |
| Unique persisted physical keys | 150 |
| Deduplicated tasks | 0 |
| Structural keys | 2 |
| Token points per structure | 75 |
| Minimum token count | 1 |
| Maximum token count | 65536 |

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
| `/data/ycfeng/tmp/step4_final_focused_post_format_20260815.log` | `347b251905042f5d2e75f99fc7d7bf6a80b18d50f1cfa6306df3cee27398b357` |
| `/data/ycfeng/tmp/step4_final_formatted_contract_tests_20260815.log` | `49ddabed431e511c9d109835fa38adf7f80b0fcbeb6bf3e7451f8fb560834eb4` |
| `/data/ycfeng/tmp/step4_final_ruff_check_post_format_20260815.log` | `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18` |
| `/data/ycfeng/tmp/step4_final_ruff_format_post_format_20260815.log` | `d6675b63a4767317d34641fb3901c01a3e6f8536bb9289fd7d0bdf7ae2bfe59b` |
| `/data/ycfeng/tmp/step4_final_shell_syntax_20260815.log` | `2c317ea048f882d1d56062bca99c048b57d99602f684110c8a36e56d8cd9e3f9` |
| `/data/ycfeng/tmp/step4_swa_qkv_finite_probe_red_20260816.log` | `ce28a268c61612180d0557b0057e4eee5e97c4ea4d83220fcc5be20dbfe402ce` |
| `/data/ycfeng/tmp/step4_swa_qkv_finite_probe_green_20260816.log` | `b64e945aef28eb434ed16da5831936487c79cf5f3a35e54337c4b92aed154c58` |
| `/data/ycfeng/tmp/step4_final_focused_post_qkvswa_20260816.log` | `61441e30c7485fbbef5689471fc01ed79f3b56b805ac734bf6a14f36f4f47744` |
| `/data/ycfeng/tmp/step4_full_collector_post_qkvswa_20260816.log` | `de1b6b39e4b2a6a9d84b186a38bd2855751d784e23262dff3425a730b8bb8a04` |
| `/data/ycfeng/tmp/step4_final_focused_after_review_20260816.log` | `4b15ac39d012ef7d9f032c7f173cc3933b09ef8fa0a785f97e904f497b5cfb4b` |
| `/data/ycfeng/tmp/step4_final_full_collector_after_review_20260816.log` | `b0fd882e68b57189896fbb3aa554864c2ba913441f67da073b80dc566d4a6eb3` |
| `/data/ycfeng/tmp/step4_final_reviewer_contracts_post_format_20260816.log` | `1c9707446211e3c8edb62638712581f4323befb552c6c9a193992cb94252b801` |
| `/data/ycfeng/tmp/step4_final_qkv_expected_key_validation_post_format_20260816.log` | `b8a630a613deee73ef01c0002cf7c9254b04d55a00080ddf6ba52968dff617dc` |

### Known unrelated failure

`test_query_custom_allreduce_large_tp_scaling` expects TP16 scaling from an
8-GPU CustomAllReduce table. Production code intentionally requires a measured
multi-node collective and raises `PerfDataNotAvailableError`. This is the
previously recorded ISSUE-019; no fallback scaling was restored.

### B300 status

Full MFA and SWA are different pinned providers and persisted physical
identities. Dedicated `qkv_full` and `qkv_swa` slices execute each actual
provider independently. The SWA slice uses only the owner-approved,
source-hash-bounded annotation compatibility overlay described below; no
provider or mathematical fallback is present.

The successful smoke artifacts and full-run identities are preserved below.
Under the current acceptance contract, replaying each full run also requires
the corresponding accepted smoke JSON and SHA256:

```bash
MODE=smoke \
PROVIDER_CORE_SLICE=qkv_full \
RJOB_NAME=s4p-aic-qkvfull-0815-050738 \
bash tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh
```

```bash
MODE=full \
PROVIDER_CORE_SLICE=qkv_full \
PROVIDER_CORE_SMOKE_EVIDENCE=/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-qkvfull-0815-050738/s4p-aic-attn-evidence-s4p-aic-qkvfull-0815-050738/checkpoint/representative_provider_core.json \
PROVIDER_CORE_SMOKE_EVIDENCE_SHA256=4dc60624752f300a343e4488be2037a57e0bd8972cc1acfe57f3a22059002a7b \
RJOB_NAME=s4p-aic-qkvfull-0815-051119 \
bash tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh
```

```bash
MODE=smoke \
PROVIDER_CORE_SLICE=qkv_swa \
RJOB_NAME=s4p-aic-qkvswa-0816-115127 \
bash tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh
```

```bash
MODE=full \
PROVIDER_CORE_SLICE=qkv_swa \
PROVIDER_CORE_SMOKE_EVIDENCE=/data/ycfeng/tmp/step4_aic_provider_core_b300_20260816/smoke_s4p-aic-qkvswa-0816-115127/s4p-aic-attn-evidence-s4p-aic-qkvswa-0816-115127/checkpoint/representative_provider_core.json \
PROVIDER_CORE_SMOKE_EVIDENCE_SHA256=c9015e23661b6975e8ee16c840828727bbd655d830ff6139c7117accdfb6ee70 \
RJOB_NAME=s4p-aic-qkvswa-full-0816-115343 \
bash tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh
```

The measured rows can be reproduced through the unchanged exact AIC consumer
with:

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  tests/performance/step4_pro_latest/validate_b300_provider_core_rows.py \
  --dataset-dir \
  /data/ycfeng/tmp/step4_qkv_150_canonicalization_20260816/dataset \
  --work-dir /data/ycfeng/tmp/step4_qkv_150_consumer_validation_20260816 \
  --output \
  task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/qkv_b300_consumer_validation_2026-08-16.json \
  --families qkv \
  --expected-qkv-rows 150
```

Runtime environment:

- image:
  `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`;
- B300 capability: `10.3`;
- torch: `2.10.0+cu129`;
- vLLM: `0.19.0.post20.dev26+gc820e5ae1`;
- `optimus-jit`: `0.1.10.post8+gitcfde41ba`;
- `nvidia-cutlass-dsl`: `4.4.2`.

| B300 check | Result | Numeric evidence |
|---|---|---:|
| Pinned source identity | PASS | 2103/2103 files |
| Isolated Full-MFA smoke | PASS | 1/1 row; 0.028218666712443035 ms |
| Full-MFA full collection | PASS | 75/75 rows |
| Full-MFA token range | PASS | 1–65536 |
| Full-MFA latency range | PASS | 0.029557332396507263–6.715685526529948 ms |
| Full-MFA duplicate physical keys | PASS | 0 |
| Full-MFA canonical exact consumers | PASS | 75/75; max error 0.0 ms |
| SWA QKV width | PASS | 18432; affected compile branch exercised |
| Accepted finite-value SWA smoke | PASS | 1/1 row; 0.0052480002244313555 ms |
| SWA QKV rows measured | PASS | 75/75 |
| SWA QKV latency range | PASS | 0.006319999694824219–1.3165173530578613 ms |
| SWA duplicate physical keys | PASS | 0 |
| Total QKV family coverage | PASS | 150/150 |
| Canonical exact consumers | PASS | 150/150; max absolute/relative error 0.0 |
| Final exact RJob cleanup | PASS | 0 resources |
| Final exact Replica cleanup | PASS | 0 resources |

The first `qkv_full` smoke executed and measured the kernel, but the host-side
result validator still required grouped/router files that are intentionally
absent from this slice. The root cause was an unfiltered expected-file map.
The validator now applies the same slice identity as the collector. The
contract test changed from RED to GREEN, and the repeated smoke plus full
collection passed. No row from the failed host validation was accepted.

Canonical combined data:

- Full-MFA source CSV rows: `75`;
- Full-MFA source CSV SHA256:
  `55a3a135ba7513b78b54f206baa80087d8d07ab3ad5350724e56f83a8ad0bb88`;
- SWA source CSV rows: `75`;
- SWA source CSV SHA256:
  `46807ec85c4cfd4bdecc546fa24a4964f289b320c3cdcb782ecd983967b9c8b3`;
- canonical Parquet:
  `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_qkv_norm_rope_perf.parquet`;
- canonical Parquet rows: `150`;
- canonical Parquet bytes: `5,329`;
- canonical Parquet SHA256:
  `cc14553d8e0d619ac5fc21b15a9dd78633e168114da270e55edec2c6e7afd579`;
- canonical validation:
  `qkv_b300_canonical_consumer_validation_2026-08-16.json`.

The original SWA failure was resolved at its runtime ABI boundary. The pinned
call reached the image-native
`optimus_cutedsl.qknorm_rope.FusedQKNormRope` implementation. Its source uses
postponed annotations, so the intended
`reload_from: cutlass.Constexpr` is stored as the string
`"cutlass.Constexpr"`. Cutlass DSL 4.4.2 reads
`inspect.getfullargspec()` without resolving that string and rejects
`reload_from="smem"` as a dynamic argument.

The approved process-local helper verifies source SHA256
`5c052658c210f5a24598d31fb6cf8f753df429bdd026da32fd715ad9696bc783`
and resolves only `reload_from` and `delay_w_load` to the installed
`cutlass.Constexpr` object. It does not write installed source or change the
pinned vLLM call path, provider, kernel body, shapes, dtypes, arguments, QKV
math, or persisted operation identity. The accepted smoke also checks output
shape, BF16 dtype, and finite values before timing acceptance.

Diagnostic evidence:

- initial SWA failure:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-core-0815-225633/`;
- successful Full-MFA smoke:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/smoke_s4p-aic-qkvfull-0815-050738/`;
- successful Full-MFA full collection:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260815/full_s4p-aic-qkvfull-0815-051119/`;
- accepted SWA smoke:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260816/smoke_s4p-aic-qkvswa-0816-115127/`;
- successful SWA full collection:
  `/data/ycfeng/tmp/step4_aic_provider_core_b300_20260816/full_s4p-aic-qkvswa-full-0816-115343/`;
- exact consumer evidence:
  `qkv_b300_consumer_validation_2026-08-16.json`;
- canonical consumer evidence:
  `qkv_b300_canonical_consumer_validation_2026-08-16.json`.
