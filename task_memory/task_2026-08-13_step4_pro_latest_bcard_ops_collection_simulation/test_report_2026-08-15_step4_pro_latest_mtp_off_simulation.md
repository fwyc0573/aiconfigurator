## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Recorded corrected SILICON coverage and smoke/full MTP-off requirements-matrix results. |
| 2026-08-15 | Updated the matrix after canonicalizing 75 Full-MFA QKV rows; only SWA QKV and DeepEP remain missing. |
| 2026-08-16 | Updated coverage and the complete matrix after canonicalizing all 150 QKV rows; only owner-deferred DeepEP remains missing. |
| 2026-08-16 | Added the explicit B300 NCCL DeepEP proxy, final 235-row Attention closure, full proxy matrix, and final verification evidence. |
| 2026-08-16 | Recorded the fresh post-archive focused, Collector, Ruff, artifact, shell, and whitespace verification. |
| 2026-08-16 | Added the requirement-by-requirement simulation audit, three-repeat evidence, execution-status fields, topology comparisons, and normalized manual-review table. |
| 2026-08-16 | Finalized the manual-review packet with a fresh byte-identical full run, artifact assertions, 343 focused tests, 401 Collector tests, and scoped static checks. |
| 2026-08-16 | Refreshed the final pre-commit requirement audit, regression timings, inventory validation, and evidence hashes. |
| 2026-08-16 | Normalized manual-review CSV output to LF and added RED-to-GREEN serialization evidence. |

# Test Report: Step4-Pro-Latest MTP-Off Simulation

**Date:** 2026-08-15–2026-08-16
**Environment:** `aic-step-design`, Python `3.11.15`, pytest `8.4.2`

## 1. Test Script Information

### Scripts

All paths below are relative to the full repository root
`/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro`.

- `tests/performance/step4_pro_latest/validate_aic_silicon_coverage.py`
- `tests/performance/step4_pro_latest/run_mtp_off_requirements.py`
- `tests/performance/step4_pro_latest/deepep_proxy.py`
- `tests/unit/performance/test_step4_pro_latest_deepep_proxy.py`
- `tests/unit/performance/test_step4_pro_latest_silicon_coverage.py`
- `tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py`
- `tests/unit/collector/test_step4_pro_latest_provider_cases.py`
- `tests/unit/sdk/database/test_step4_pro_latest_attention_data.py`
- `tests/unit/sdk/database/test_step4_pro_latest_deepep_data.py`
- `tests/unit/sdk/database/test_step4_pro_latest_optimus_moe_data.py`
- `tests/unit/sdk/models/test_step4_pro_latest.py`
- `tests/unit/sdk/operations/test_step4_pro_latest_operations.py`

### Commands

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest -q \
  tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py \
  tests/unit/performance/test_step4_pro_latest_silicon_coverage.py \
  tests/unit/collector/test_step4_pro_latest_provider_cases.py \
  tests/unit/sdk/database/test_step4_pro_latest_attention_data.py \
  tests/unit/sdk/database/test_step4_pro_latest_deepep_data.py \
  tests/unit/sdk/database/test_step4_pro_latest_optimus_moe_data.py \
  tests/unit/sdk/models/test_step4_pro_latest.py \
  tests/unit/sdk/operations/test_step4_pro_latest_operations.py
```

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  tests/performance/step4_pro_latest/validate_aic_silicon_coverage.py \
  --output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/aic_silicon_coverage_2026-08-16.json
```

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  tests/performance/step4_pro_latest/run_mtp_off_requirements.py \
  --smoke \
  --output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_exact_smoke_2026-08-16.json
```

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python \
  tests/performance/step4_pro_latest/run_mtp_off_requirements.py \
  --output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_full_2026-08-16.json
```

The three runtime commands were wrapped with:

```bash
systemd-run --user --scope -p MemoryMax=2G bash -lc '<command>'
```

Final proxy coverage and simulation commands:

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
  tests.performance.step4_pro_latest.validate_aic_silicon_coverage \
  --deepep-proxy b300_nccl_alltoall \
  --output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/aic_silicon_coverage_proxy_2026-08-16.json
```

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
  tests.performance.step4_pro_latest.run_mtp_off_requirements \
  --smoke \
  --deepep-proxy b300_nccl_alltoall \
  --output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_smoke_2026-08-16.json
```

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
  tests.performance.step4_pro_latest.run_mtp_off_requirements \
  --deepep-proxy b300_nccl_alltoall \
  --output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_full_2026-08-16.json
```

Phase 11 repeated the complete matrix three times as separately bounded
commands:

```bash
for repeat in 1 2 3; do
  timeout 60s systemd-run --user --scope -p MemoryMax=2G bash -lc \
    "cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro && \
     env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. \
     /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
       tests.performance.step4_pro_latest.run_mtp_off_requirements \
       --deepep-proxy b300_nccl_alltoall \
       --output /data/ycfeng/tmp/step4_phase11_repeat${repeat}_20260816.json"
done
```

The three independently completed artifacts were combined without rerunning
or averaging any case:

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
  tests.performance.step4_pro_latest.run_mtp_off_requirements \
  --repeat-input /data/ycfeng/tmp/step4_phase11_repeat1_20260816.json \
  --repeat-input /data/ycfeng/tmp/step4_phase11_repeat2_20260816.json \
  --repeat-input /data/ycfeng/tmp/step4_phase11_repeat3_20260816.json \
  --output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_repeat3_2026-08-16.json \
  --review-output task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_review_2026-08-16.csv
```

Final focused and Collector commands:

```bash
PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest -q \
  tests/e2e/step4_pro_latest/test_*.py \
  tests/performance/step4_pro_latest/test_*.py \
  tests/unit/performance/test_step4_pro_latest_*.py \
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

PYTHONPATH=src:. \
/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest -q \
  tests/unit/collector
```

Fresh publication recheck of the complete proxy matrix:

```bash
timeout 900s systemd-run --user --scope -p MemoryMax=2G bash -lc \
  'cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro && \
   env TMPDIR=/data/ycfeng/tmp PYTHONPATH=src:. \
   /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
     tests.performance.step4_pro_latest.run_mtp_off_requirements \
     --deepep-proxy b300_nccl_alltoall \
     --output /data/ycfeng/tmp/step4_phase11_fresh_full_20260816.json'

sha256sum \
  /data/ycfeng/tmp/step4_phase11_fresh_full_20260816.json \
  task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_full_2026-08-16.json

cmp \
  /data/ycfeng/tmp/step4_phase11_fresh_full_20260816.json \
  task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_full_2026-08-16.json
```

Static checks were run only on Python inputs; YAML was parsed separately:

```bash
mapfile -t task_python_files <<'EOF'
collector/case_generator.py
collector/registry_types.py
collector/vllm/collect_step4_provider.py
collector/vllm/registry.py
collector/wideep/vllm/__init__.py
collector/wideep/vllm/collect_step4_deepep_ht.py
collector/wideep/vllm/registry.py
src/aiconfigurator/sdk/backends/base_backend.py
src/aiconfigurator/sdk/common.py
src/aiconfigurator/sdk/config.py
src/aiconfigurator/sdk/models/base.py
src/aiconfigurator/sdk/models/step4.py
src/aiconfigurator/sdk/operations/moe.py
src/aiconfigurator/sdk/perf_database.py
tests/e2e/step4_pro_latest/generate_step4pro_dummy_configs.py
tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py
tests/e2e/step4_pro_latest/test_b300_two_node_deepep_legacy_probe_contract.py
tests/e2e/step4_pro_latest/test_b300_two_node_nccl_preflight_contract.py
tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py
tests/e2e/step4_pro_latest/test_generate_step4pro_dummy_configs.py
tests/e2e/step4_pro_latest/test_run_b300_source_probe_static.py
tests/performance/step4_pro_latest/deepep_proxy.py
tests/performance/step4_pro_latest/run_mtp_off_requirements.py
tests/performance/step4_pro_latest/run_step4_deepep_ht_distributed.py
tests/performance/step4_pro_latest/run_step4_deepep_ht_nccl_preflight.py
tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py
tests/performance/step4_pro_latest/test_b300_attention_collection_contract.py
tests/performance/step4_pro_latest/test_b300_deepep_ht_collection_contract.py
tests/performance/step4_pro_latest/test_b300_optimus_moe_collection_contract.py
tests/performance/step4_pro_latest/test_b300_provider_core_collection_contract.py
tests/performance/step4_pro_latest/validate_aic_silicon_coverage.py
tests/performance/step4_pro_latest/validate_b300_attention_rows.py
tests/performance/step4_pro_latest/validate_b300_provider_core_rows.py
tests/unit/collector/test_step4_deepep_ht_distributed_driver.py
tests/unit/collector/test_step4_deepep_ht_nccl_preflight.py
tests/unit/collector/test_step4_deepep_ht_runtime.py
tests/unit/collector/test_step4_pro_latest_provider_cases.py
tests/unit/performance/test_step4_pro_latest_deepep_proxy.py
tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py
tests/unit/performance/test_step4_pro_latest_silicon_coverage.py
tests/unit/sdk/backends/test_base_backend.py
tests/unit/sdk/database/test_step4_pro_latest_deepep_data.py
tests/unit/sdk/database/test_step4_pro_latest_optimus_moe_data.py
tests/unit/sdk/models/test_step4_pro_latest.py
tests/unit/sdk/operations/test_step4_pro_latest_operations.py
EOF

/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
  ruff check "${task_python_files[@]}"

/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m \
  ruff format --check "${task_python_files[@]}"

find tests/e2e/step4_pro_latest tests/performance/step4_pro_latest \
  -type f -name '*.sh' -print0 |
  xargs -0 -n1 bash -n
```

## 2. Validation Criteria

- Use model `stepfun-ai/Step4-Pro-Latest`, MTP-off, B300, vLLM `0.19.0`,
  and database mode `SILICON`.
- Preserve one global `max_num_batched_tokens` prefill budget.
- Map Attention/Dense work to the busiest attention-DP rank and Optimus MoE
  work to global scheduled tokens.
- Enforce `gpu-memory-utilization=0.9`.
- Query every available operation through the unchanged AIC consumer.
- With no proxy option, missing DeepEP must set formal latency and decode
  `B_max` to null and exit with code `2`.
- The proxy must require explicit
  `--deepep-proxy b300_nccl_alltoall`, map dispatch FP8 to the measured NCCL
  `int8` transport curve, and map combine BF16 to `half`.
- Proxy results must use `status=proxy`, `result_fidelity=PROXY`, and
  `PASS_WITH_PROXY`; proxy counts and latency must remain separate from exact
  silicon and analytic/non-silicon records.
- No H800, generic communication, synthetic timing, or analytic fallback may
  be labeled as exact silicon.
- No proxy row may be persisted as
  `step4_deepep_ht_perf.parquet`.
- Every required condition must expose requested/resolved KV dtype, logical
  bytes, modeled resident/peak allocated bytes, component latency, normalized
  per-GPU throughput, OOM, backend fallback, retry, error/missing counts, and
  exception-log fields.
- Runtime-only metrics that AIC cannot produce must remain `null` with an
  explicit reason; they must not be inferred from deterministic operation
  sums.
- Three complete matrix runs must preserve all `156` case identities and show
  the numeric spread for every audited metric.

## 3. Test Results and Evidence

### Key Outcome

| Check | Result | Numeric Evidence |
|---|---|---|
| Final focused tests | PASS | `344 passed` in `8.14s` |
| Full Collector tests | PASS | `401 passed` in `31.23s` |
| Fresh full proxy rerun | PASS_WITH_PROXY | `72` prefill + `84` decode; byte-identical to the archived JSON |
| Requirement artifact audit | PASS | `156` JSON cases, `156 × 87` CSV, `52/52` inventory hashes, component error `6.51925802230835e-09 ms`, KV error `0.0 GiB` |
| Review CSV serialization | PASS | RED `1` failed, GREEN `1` passed; `157` LF endings, `0` CRLF endings |
| Scoped Ruff | PASS | check and format passed on `45/45` Python files |
| Shell and structured inputs | PASS | `14/14` shell scripts; `2/2` YAML and `3/3` JSON files parsed |
| Exact-path SILICON coverage | EXPECTED BLOCKED | `36,420` records; `18,220` exact silicon; `15,160` non-silicon; `3,040` missing; `0` errors |
| Exact-path smoke | EXPECTED BLOCKED | prefill `2/2`, decode `1/1`, proxy `0`, exit `2` |
| Proxy coverage | PASS_WITH_PROXY | `36,420` records; `18,220` exact silicon; `15,160` non-silicon; `3,040` proxy; missing/error `0/0` |
| Proxy smoke | PASS_WITH_PROXY | prefill `2/2`, decode `1/1`, blocked `0` |
| Full proxy prefill | PASS_WITH_PROXY | `72/72`; fidelity `PROXY`; PASS `24`, OOM `48` |
| Full proxy decode | PASS_WITH_PROXY | `84/84`; fidelity `PROXY`; `B_max=0` for `84/84` |
| Attention in coverage | PASS | `1,560/1,560` exact silicon |
| Optimus MoE in coverage | PASS | `1,520/1,520` exact silicon |
| Grouped GEMM in coverage | PASS | `400/400` exact silicon |
| FP32 router in coverage | PASS | `1,520/1,520` exact silicon |
| Complete QKV in coverage | PASS | `1,560/1,560` exact silicon |
| Canonical B300 rows | PASS | Attention `235`, Optimus `174`, grouped `75`, router `75`, QKV `150`; total `709` |
| Proxy persistence isolation | PASS | `step4_deepep_ht_perf.parquet` absent |

### Requirement-Completeness Verdict

The currently approved **AIC MTP-off scope is complete with the explicit
DeepEP proxy**. This does not mean the whole source requirements document is
closed: native Step4Pro MTP1, real DeepEP measurement, whole-model vLLM
results, live CUDA allocation counters, live latency distributions, router
histograms, and the vLLM-versus-AIC error table remain deferred or externally
owned by prior owner decisions.

| Requirements area | Status | Evidence or boundary |
|---|---|---|
| Three topologies | PASS | `ep16_r1`, `ep32_r1`, `ep16_r2` |
| Full prefill matrix and chunk scans | PASS_WITH_PROXY | `72/72` conditions per repeat; chunk count `1–128` |
| Decode workloads and four TPOT budgets | PASS_WITH_PROXY | `84/84` conditions per repeat; search starts at batch `1` |
| Three repetitions | PASS | `468` executions; `156/156` full results identical; all audited spreads `0.0` |
| Latency fields | PARTIAL_SIMULATOR_ONLY | Prefill and deterministic steady-step p50/TPOT are modeled; TTFT, first step, sampled p90/p99, ITL, generation, and E2E are explicit `null + reason` |
| Capacity and throughput | PASS_WITH_PROXY | `B_max`, first failed batch, input/output/total tok/s, and tok/s/GPU are present |
| KV and HBM | PASS | `auto -> bfloat16`, logical/resident/peak allocated KV, weights, activations, and peak HBM are present |
| Separate scale/workspace and live CUDA allocation | PENDING_EXTERNAL_VLLM | Kept `null + reason`; current AIC accounts scales with weights and workspace with activations |
| Component breakdown | PASS for MTP-off | Full MFA, SWA, Dense, Latent MoE compute/total, dispatch, combine, and other are present |
| MTP1 iteration | DEFERRED_BY_OWNER | Explicit `null`; no Step3p5 or invented substitute |
| MoE observability | PARTIAL_SIMULATOR_ONLY | Performance workload is `power_law_1.2`; live histogram, max/mean load, and padding are `null + reason` |
| OOM/fallback/retry/exceptions | PASS | Fallback `0`, retries `0`, error/missing records `0/0`, exception entries `0` in the proxy run |
| Real DeepEP | DEFERRED_PROXY_ACTIVE | All affected results say `PROXY`; exact dataset remains absent |
| Whole-model vLLM and error table | PENDING_EXTERNAL_VLLM | No actual values or errors were fabricated |

### Three-Repeat Audit

| Metric | Value |
|---|---:|
| Repeats | 3 |
| Cases per repeat | 156 |
| Total case executions | 468 |
| Fully identical cases | 156 |
| Non-identical cases | 0 |
| Maximum spread across every audited numeric metric | 0.0 |

The three independent single-run JSON files have the same SHA256:
`b68008b8a86c0a67b90a51fc9b57c301c7561101a9e3613b405e75c6482dd61a`.

### Prefill Numeric Results

| Metric | Minimum | Maximum |
|---|---:|---:|
| Formal proxy latency | 129.22378441076575 ms | 782420.0031436655 ms |
| Input throughput per replica | 1340.1293369125042 token/s | 69183.9264811136 token/s |
| Aggregate input throughput | 1340.1293369125042 token/s | 103512.27987036602 token/s |
| Chunk count | 1 | 128 |
| Peak HBM | 142.00439453125 GiB | 776.97998046875 GiB |

- Utilization limit: `241.734375 GiB`.
- OOM classification: `48/72`.
- Status split: `24` `PASS_WITH_PROXY`, `48` `OOM`; all `72` records retain
  `result_fidelity=PROXY`.

### Decode Numeric Results

| Metric | Minimum | Maximum |
|---|---:|---:|
| Batch-1 TPOT | 56.955545921130614 ms | 268.6990437660492 ms |
| Batch-1 HBM | 138.58154296875 GiB | 294.77685546875 GiB |
| Batch-1 output throughput per replica | 3.72163587180712 token/s | 17.557552716372054 token/s |

- Batch-1 OOM classification: `16/84`.
- `B_max=0`, `aggregate_B_max=0`, and `first_failed_batch=1` for all `84/84`
  topology/workload/budget combinations.
- Every decode record is `PASS_WITH_PROXY` with
  `result_fidelity=PROXY`; `B_max=0` means batch 1 already misses the requested
  TPOT budget or exceeds the memory limit, not that the matrix failed to run.

### Per-Topology Summary

#### Prefill

| Topology | Latency range (ms) | OOM | Input tok/s/GPU | Peak HBM (GiB) |
|---|---:|---:|---:|---:|
| `ep16_r1` | 213.0058–782420.0031 | 18/24 | 83.7581–3234.7587 | 218.3755–776.9800 |
| `ep32_r1` | 129.2238–768126.6709 | 12/24 | 42.6583–2161.9977 | 142.0044–700.6089 |
| `ep16_r2` | 213.0058–782420.0031 | 18/24 | 83.7581–3234.7587 | 218.3755–776.9800 |

#### Decode batch 1

| Topology | TPOT range (ms) | OOM | Output tok/s/GPU | Peak HBM (GiB) |
|---|---:|---:|---:|---:|
| `ep16_r1` | 73.9618–268.6990 | 8/28 | 0.2326–0.8450 | 214.9526–294.7769 |
| `ep32_r1` | 56.9555–251.6928 | 0/28 | 0.1242–0.5487 | 138.5815–218.4058 |
| `ep16_r2` | 73.9618–268.6990 | 8/28 | 0.2326–0.8450 | 214.9526–294.7769 |

At matched conditions, `ep32_r1` is `1.013–2.298×` faster than `ep16_r1`
for prefill and `1.068–1.299×` faster for the modeled decode step. On the same
32-GPU count, `ep16_r2 / ep32_r1` aggregate throughput is
`0.870–1.974×` for prefill and `1.540–1.873×` for decode batch 1. These are
raw proxy-model throughput comparisons; all four requested decode budgets
still have `B_max=0`, so there is no budget-qualified capacity gain to claim.

### Long-Context EP32 Boundary

`ep32_r1` is the requirements-preferred long-context topology:

| Context | Prefill latency (ms) | Prefill peak HBM / KV (GiB) | Decode TPOT (ms) | Decode peak HBM / KV (GiB) | Decode OOM |
|---:|---:|---:|---:|---:|---|
| 128K | 20834.5357 | 389.5591 / 17.3633 | 83.4479 | 148.4253 / 10.1611 | No |
| 512K | 209689.1177 | 419.5591 / 47.3633 | 152.7476 | 178.4155 / 40.1514 | No |
| 1,048,544 | 757956.7884 | 459.3589 / 87.3916 | 251.6928 | 218.4058 / 80.1416 | No |

All three prefill points exceed the configured `241.734375 GiB` utilization
limit because modeled activation/workspace dominates the peak. EP32 can hold
the three decode batch-1 states, but every TPOT is above the loosest
`33.33 ms` budget. EP16 additionally OOMs at 512K and approximately 1M decode.

### Memory and Component Accounting

| Metric | Prefill range | Decode batch-1 range |
|---|---:|---:|
| Logical KV (GiB/GPU) | 0.1230–20.1127 | 0.1572–20.1133 |
| Resident allocated KV (GiB/GPU) | 0.1523–80.1416 | 0.2891–80.1133 |
| Peak allocated KV (GiB/GPU) | 0.1523–94.6416 | 0.3174–80.1416 |
| Throughput (tok/s/GPU) | input 42.6583–3234.7587 | output 0.1242–0.8450 |

- Maximum component-to-total accounting error:
  `6.51925802230835e-09 ms`.
- Maximum peak-KV-to-memory-accounting error: `0.0 GiB`.
- Full MFA contributes `0.83%–91.83%` of prefill and `4.20%–78.63%` of
  decode latency.
- Latent MoE including dispatch/combine contributes `4.36%–97.51%` of
  prefill and `18.38%–85.54%` of decode latency.
- Therefore the modeled bottleneck changes with context: Latent MoE dominates
  many short/token-heavy points, while Full MFA dominates the longest
  attention-heavy points. A single global correction factor would be invalid.

### Manual-Review Records

- JSON:
  `mtp_off_requirements_proxy_repeat3_2026-08-16.json`
- Flat CSV:
  `mtp_off_requirements_proxy_review_2026-08-16.csv`
- CSV rows: `156` plus one header.
- CSV line endings: `157` LF and `0` CRLF.
- Every row contains the workload/topology, latency, throughput, KV/HBM,
  component, MoE, proxy, OOM, fallback/retry/error, MTP1-deferred, and
  vLLM-pending fields.
- Decode rows explicitly record `active_sequences_per_replica=1` and
  `batched_tokens_per_replica=1`; the complete candidate path remains in the
  JSON.

### Proxy Accounting and Fidelity

- Coverage proxy records: `3,040`; summed proxy latency:
  `2430.509110714305 ms`.
- Full-matrix proxy operation records: `134,976`; summed proxy latency:
  `181471.55741956536 ms`.
- The exact path still exposes four unavailable physical contracts: DeepEP
  dispatch/combine at EP16 and EP32.
- EP16/EP32 use AIC's existing rank-count correction from the measured 8-GPU
  NCCL curve; they are not direct EP16/EP32 DeepEP measurements.
- No DeepEP Parquet exists, so none of these proxy values can be confused with
  measured DeepEP silicon.

### Evidence

- `aic_silicon_coverage_2026-08-16.json`, SHA256
  `00d4c46fc068867c7d480dca03e19c0c0ed867c217a62e07187061ecc461bc25`
- `mtp_off_requirements_exact_smoke_2026-08-16.json`, SHA256
  `142ec21e10e0adb5132e011f058166196539918e4d85d0d6710b074b99e3e82d`
- `mtp_off_requirements_full_2026-08-16.json`, SHA256
  `f2b3f8f04c4e966c48bb55277f33a8e8b3d243062efd1674c1eb3687ca3cb7de`
- `aic_silicon_coverage_proxy_2026-08-16.json`, SHA256
  `0b5f19fa558970a6d8940052521b0e68fb62c301ded833881b9a0fbc27038285`
- `mtp_off_requirements_proxy_smoke_2026-08-16.json`, SHA256
  `d270ae17b0176eda4baf88266d3f803d2e51ae88564a96613a9d4fc5209dac48`
- `mtp_off_requirements_proxy_full_2026-08-16.json`, SHA256
  `b68008b8a86c0a67b90a51fc9b57c301c7561101a9e3613b405e75c6482dd61a`
- `mtp_off_requirements_proxy_repeat3_2026-08-16.json`, SHA256
  `9d5c296c7c4e95859982cbb986cbda21ef65f9a56d286ebe37cc5a780f7208a1`
- `mtp_off_requirements_proxy_review_2026-08-16.csv`, SHA256
  `4c1afd34a37877a6cb59cf79c37a326b537261ed9640ee4d657ce527ff49ac62`
- `/data/ycfeng/tmp/step4_phase11_csv_lf_red_20260816.log`, SHA256
  `772a5c23863ac909a57486eddbd58f2df0436c40661af0f838b78a850229ce5b`
- `/data/ycfeng/tmp/step4_phase11_csv_lf_green_20260816.log`, SHA256
  `ac071f881d8ba1a207f89dea2f8dd12dc165cbe1ef8c9bd73c8690fb818379bb`
- `/data/ycfeng/tmp/step4_phase11_csv_lf_regeneration_20260816.log`,
  SHA256
  `8bedf9a4fbdfcf5698365479665cd24e0e0880b9f96bb7002b5e03079b2fdf52`
- `/data/ycfeng/tmp/step4_phase11_final_focused_20260816.log`,
  SHA256
  `a4889334fa760d2c457aa032b7bb1ce4671da9953bc8182ff270ee9f5dafa128`
- `/data/ycfeng/tmp/step4_phase11_final_collector_20260816.log`,
  SHA256
  `d94287553341a0b48777de9a19a07db96c3470bf3a85a6571499c7f201472494`
- `/data/ycfeng/tmp/step4_phase11_fresh_requirement_audit_20260816.log`,
  SHA256
  `1d249a59a03670c923fe6025047d0dfa7b565542867bdbbd805e3fbf6473db63`
- `/data/ycfeng/tmp/step4_phase11_fresh_full_sha_20260816.log`,
  SHA256
  `85569c0fdaa92aeea35fd8ba9f5c8aa8531a04e3a4335e1f1b083c1ca22c5e19`
- `/data/ycfeng/tmp/step4_phase11_ruff_python_only_20260816.log`, SHA256
  `ef0248af871deee68a83e15c814b0255bf03aa4f1651f4f7591a8430f8bea6b5`
- `/data/ycfeng/tmp/step4_phase11_shell_syntax_20260816.log`, SHA256
  `2c317ea048f882d1d56062bca99c048b57d99602f684110c8a36e56d8cd9e3f9`
- `/data/ycfeng/tmp/step4_phase11_structured_files_20260816.log`, SHA256
  `9d256a9586bf430f3a9775922daf795c1655b17affe6af738324efe2bbafe751`
- `/data/ycfeng/tmp/step4_phase11_publication_audit_20260816.log`, SHA256
  `1296ffcc75d5d4ef85f8567c258b247b66dbe2911db88cb2e509e5d33d484f66`
- `/data/ycfeng/tmp/step4_phase11_publication_focused_20260816.log`, SHA256
  `5c17a8b74801ae773299b2de1dd0370bcf21101153df5e7772f0f98880f45e7d`
- `/data/ycfeng/tmp/step4_phase11_publication_collector_20260816.log`, SHA256
  `fe26195befcfea4bce15a3e333b0dae67e4ef02d00cb5560cd5590e81665d6b5`
- `/data/ycfeng/tmp/step4_phase11_publication_full_20260816.log`, SHA256
  `c14abde3f149a1d302c158415901df622e35f6aa17c8dd9c1d929f51a7ebf8ca`
- `/data/ycfeng/tmp/step4_phase11_publication_ruff_20260816.log`, SHA256
  `1a30e664c74efbf8a58db5bd15bd30a865b4394bdef506ea76986b7b9494101b`
- `/data/ycfeng/tmp/step4_phase11_publication_shell_20260816.log`, SHA256
  `349d89e73c90ab156bc044f9b287293bd3977c2aadcc279dcb82cb9925347c11`
- `/data/ycfeng/tmp/step4_phase11_publication_structured_20260816.log`, SHA256
  `5eecbbd5a701f468a49c0888f064fb9c9e5851ae080e42321d18f35f6c758cf2`
- `/data/ycfeng/tmp/step4_phase11_final_focused_lf_20260816.log`,
  SHA256
  `d71510087cee192179c3c9cf0f40d639742730fb95b8bee0e72bcc9d5c9c9c15`
- `/data/ycfeng/tmp/step4_phase11_final_collector_lf_20260816.log`,
  SHA256
  `f91b864863c007677e48e6f0572c33af6f391988e897c3c9b0dd637c91a449f9`
- `/data/ycfeng/tmp/step4_phase11_final_full_lf_20260816.log`, SHA256
  `171fdf88db2867f6b705058b0f74a0fc8f1df8ba9b768b81f738c59eadaf7bdc`
- `/data/ycfeng/tmp/step4_phase11_final_requirement_audit_lf_20260816.log`,
  SHA256
  `aa90fc864f93b128f8d120d9388ad8daf961767b34e17c34cc5951d5d8b00b4c`
