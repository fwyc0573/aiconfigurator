## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-13 | Recorded the first focused baseline run and its pre-existing failure clusters. |
| 2026-08-13 | Recorded the root-cause repairs and final green baseline verification. |

# Test Report: Step4-Pro-Latest Baseline

**Date:** 2026-08-13
**Overall Result:** PASS after root-cause repair
**Branch:** `task/step4-pro-latest-b300`
**Baseline Commit:** `4f2b0c31`

## 1. Test Script Information

### Environment

| Item | Actual Value |
|---|---|
| Conda environment | `/home/i-fengyicheng/miniconda3/envs/aic-step-design` |
| Python | `3.11.15` |
| pytest | `8.4.2` |
| Source binding | `PYTHONPATH=$PWD/src:$PWD` |
| Matplotlib backend | `Agg` |
| Temporary directory | `/data/ycfeng/tmp/aic-step4-latest-baseline` |
| Memory limit | systemd scope `MemoryMax=4G` |

The repository-local `.venv` does not exist in this linked worktree. The
historically verified absolute conda interpreter was used instead.

### Test Scripts

- `tests/integration/test_step4_pro_v1_support.py`
- `tests/unit/collector/test_getter_deduplication.py`
- `tests/unit/collector/test_model_cases.py`
- `tests/unit/collector/test_step4_profiling_cases.py`
- `tests/unit/collector/test_vllm_gemm_policy.py`
- `tests/unit/sdk/database/test_attention.py`
- `tests/unit/sdk/database/test_collective_query_capture.py`
- `tests/unit/sdk/database/test_factorized_attention_runtime_spec.py`
- `tests/unit/sdk/database/test_interpolation.py`
- `tests/unit/sdk/database/test_moe_dispatch.py`
- `tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
- `tests/unit/sdk/database/test_step4_roofline.py`
- `tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py`
- `tests/unit/sdk/models/test_step4_pro_v1.py`
- `tests/unit/sdk/models/test_step4_pro_v3_v4.py`
- `tests/unit/performance/test_step4_*.py`
- `tests/unit/performance/test_dsv4pro_vs_step4pro_throughput.py`

### Reproducible Command

```bash
cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro

ROOT="$PWD"
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python
TMPBASE=/data/ycfeng/tmp/aic-step4-latest-baseline

mkdir -p "$TMPBASE"

timeout 1800s sudo -n systemd-run --scope -p MemoryMax=4G \
  --uid="$(id -un)" --gid="$(id -gn)" \
  env PYTHONPATH="$ROOT/src:$ROOT" MPLBACKEND=Agg TMPDIR="$TMPBASE" \
  "$PY" -m pytest -p no:cacheprovider -q \
    tests/integration/test_step4_pro_v1_support.py \
    tests/unit/collector/test_getter_deduplication.py \
    tests/unit/collector/test_model_cases.py \
    tests/unit/collector/test_step4_profiling_cases.py \
    tests/unit/collector/test_vllm_gemm_policy.py \
    tests/unit/sdk/database/test_attention.py \
    tests/unit/sdk/database/test_collective_query_capture.py \
    tests/unit/sdk/database/test_factorized_attention_runtime_spec.py \
    tests/unit/sdk/database/test_interpolation.py \
    tests/unit/sdk/database/test_moe_dispatch.py \
    tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
    tests/unit/sdk/database/test_step4_roofline.py \
    tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py \
    tests/unit/sdk/models/test_step4_pro_v1.py \
    tests/unit/sdk/models/test_step4_pro_v3_v4.py \
    tests/unit/performance/test_step4_*.py \
    tests/unit/performance/test_dsv4pro_vs_step4pro_throughput.py
```

Raw log:

```text
/data/ycfeng/tmp/step4_latest_baseline_pytest.log
SHA256: 5a8d406abc9e382322ef9ecc973c9a579fad415c1c66a854682a50f5d2f25289
```

## 2. Validation Criteria

- All selected baseline tests must pass before Step4-Pro-Latest production
  implementation begins.
- Expected failed tests: `0`.
- Expected unexpected exceptions or collection errors: `0`.
- The test interpreter must import `aiconfigurator` from this worktree's
  `src/` directory.
- No test may write temporary files under `/tmp`.

## 3. Test Results and Evidence

### Summary

| Metric | Expected | Actual | Result |
|---|---:|---:|---|
| Collected tests | greater than 0 | 987 | PASS |
| Passed tests | 987 | 859 | FAIL |
| Failed tests | 0 | 128 | FAIL |
| Pass rate | 100% | 87.03% | FAIL |
| Runtime | recorded | 69.92 seconds | INFO |

### Failure Distribution

| Test file | Failed |
|---|---:|
| `tests/unit/sdk/database/test_factorized_attention_runtime_spec.py` | 79 |
| `tests/unit/sdk/models/test_step4_pro_v1.py` | 41 |
| `tests/unit/sdk/database/test_step4_pro_v1_roofline.py` | 4 |
| `tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py` | 4 |
| **Total** | **128** |

### Representative Evidence

```text
AssertionError: dsv4.FactorizedAttentionRuntimeSpec must exist
AttributeError: module 'aiconfigurator.sdk.common' has no attribute
'FullAttentionConfig'
TypeError: Step4MFAAttentionConfig.__init__() got an unexpected keyword argument
assert 0.0 == 31457280
```

The failures predate Step4-Pro-Latest implementation: no new model or Collector
production code had been written when this run was executed. Root-cause
investigation is active and implementation remains paused.

## 4. Repaired Baseline Verification

### Test Script Information

The final run used the same environment and test scope, with the two
owner-approved obsolete files removed from the command:

- `tests/unit/sdk/database/test_factorized_attention_runtime_spec.py`
- `tests/unit/sdk/models/test_deepseek_v4_runtime_spec.py`

Command:

```bash
cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro

ROOT="$PWD"
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python
TMPBASE=/data/ycfeng/tmp/aic-step4-latest-baseline-repaired-final

mkdir -p "$TMPBASE"

timeout 1800s sudo -n systemd-run --scope -p MemoryMax=4G \
  --uid="$(id -un)" --gid="$(id -gn)" \
  env PYTHONPATH="$ROOT/src:$ROOT" MPLBACKEND=Agg TMPDIR="$TMPBASE" \
  "$PY" -m pytest -p no:cacheprovider -q \
    tests/integration/test_step4_pro_v1_support.py \
    tests/unit/collector/test_getter_deduplication.py \
    tests/unit/collector/test_model_cases.py \
    tests/unit/collector/test_step4_profiling_cases.py \
    tests/unit/collector/test_vllm_gemm_policy.py \
    tests/unit/sdk/database/test_attention.py \
    tests/unit/sdk/database/test_collective_query_capture.py \
    tests/unit/sdk/database/test_interpolation.py \
    tests/unit/sdk/database/test_moe_dispatch.py \
    tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
    tests/unit/sdk/database/test_step4_roofline.py \
    tests/unit/sdk/models/test_step4_pro_v1.py \
    tests/unit/sdk/models/test_step4_pro_v3_v4.py \
    tests/unit/performance/test_step4_*.py \
    tests/unit/performance/test_dsv4pro_vs_step4pro_throughput.py
```

Environment:

| Item | Actual Value |
|---|---|
| Conda environment | `/home/i-fengyicheng/miniconda3/envs/aic-step-design` |
| Python | `3.11.15` |
| pytest | `8.4.2` |
| Source binding | `PYTHONPATH=$PWD/src:$PWD` |
| Temporary directory | `/data/ycfeng/tmp/aic-step4-latest-baseline-repaired-final` |
| Memory limit | systemd scope `MemoryMax=4G` |

### Validation Criteria

- Expected failed tests: `0`.
- V1 must retain standard Full Attention plus HCA and TP-sharded full K/V.
- `SOL` and `SOL_FULL` HCA queries must not call `load_data()`.
- Existing V3/V4, Collector, integration, and performance tests must remain
  green.
- Ruff, format, JSON parse, and whitespace checks must pass.

### Test Results and Evidence

| Metric | Expected | Actual | Result |
|---|---:|---:|---|
| Collected tests | 899 | 899 | PASS |
| Passed tests | 899 | 899 | PASS |
| Failed tests | 0 | 0 | PASS |
| Pass rate | 100% | 100% | PASS |
| Runtime | recorded | 51.84 seconds | PASS |
| V1 model tests | all pass | 267 passed | PASS |
| V1 roofline tests | all pass | 34 passed | PASS |

Key numeric V1 checks:

| Metric | Target | Actual | Result |
|---|---:|---:|---|
| Full Attention parameters | 153,095,232 | 150,994,944 | PASS, 1.3718833517% error |
| HCA parameters | 213,911,648 | 217,055,232 | PASS, 1.4695712129% error |
| HCA resident FP32 state | 65,632 elements | 65,632 elements | PASS |
| TP1 FP8 KV at 1,048,576 tokens | documented conflict: 10.7 GB | 257.99688192 GB | OPEN historical V1 evidence conflict |

Static validation:

```text
ruff check: PASS
ruff format --check: PASS
git diff --check: PASS
JSON parse: PASS
```

Raw final log:

```text
/data/ycfeng/tmp/step4_latest_baseline_repaired_final_pytest.log
SHA256: c7a869263afd16b8694259ecafbe7df29e5a3a02320298a01e9e6009d5b68154
```
